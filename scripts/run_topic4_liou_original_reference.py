#!/usr/bin/env python3
"""Equation/update-order port of Liou et al. LAS-Model Exp2/4 (MIT).

Author archive: results/topic4_sef_hfo/liou_original_design_20260915/literature/LAS-Model
Original copyright (c) 2019 Jyun-you Liou; full MIT notice is in that archive.
This is a Python execution port, not a claim of MATLAB random-stream identity.
No extra global state, rectification threshold, recovery rule, or dual core.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from numba import njit
from scipy import fft
from scipy.signal import convolve

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/topic4_sef_hfo/liou_original_design_20260915'
STATE_NAMES = ['V_mV', 'phi_mV', 'Cl_mM', 'gK_stored', 'input_E', 'input_I', 'output', 'elapsed_since_spike_ms']

@dataclass(frozen=True)
class Protocol:
    experiment: str
    n: int
    spiking: bool
    e_l: float
    phi_0: float
    beta: float
    gk_max: float
    w_local_i: float
    w_global_i: float
    stimulus_pA: float
    stimulus_left: float
    stimulus_right: float
    duration_ms: int
    dt_ms: float = 1.0
    stimulus_start_ms: float = 2000.
    stimulus_stop_ms: float = 5000.
    tau_syn_ms: float = 15.
    tau_phi_ms: float = 100.
    tau_cl_ms: float = 5000.
    tau_k_ms: float = 5000.
    seed: int = 20260915
    record_ms: int = 10

def protocol(name, seed=20260915):
    if name == 'exp2a':
        return Protocol(name, 500, False, -57.5, -45., 2.5, 40., 250., 50., 200., .1, .15, 99999, seed=seed)
    if name in ('exp4a', 'exp4b'):
        weights = (250., 50.) if name == 'exp4a' else (150., 150.)
        return Protocol(name, 2000, True, -60., -55., 1.5, 40., *weights, 80., .05, .1, 50000, seed=seed)
    if name == 'exp5_first':
        return Protocol(name, 2000, True, -57., -55., 1.5, 50., 250., 50., 200., .475, .525, 49999, seed=seed)
    raise ValueError(name)

def gaussian_kernel(n, sigma_fraction):
    # Kernelize samples -KerSize+1 : KerSize-1, NOT -KerSize : KerSize.
    half = int(np.ceil(2.5*n*sigma_fraction)) - 1
    x = np.arange(-half, half+1, dtype=float)
    w = np.exp(-.5*(x/(n*sigma_fraction))**2)
    return w/w.sum()

class Projection:
    def __init__(self, p):
        self.p = p
        self.ke, self.ki = gaussian_kernel(p.n, .02), gaussian_kernel(p.n, .03)
        self.length = fft.next_fast_len(p.n + len(self.ki)-1)
        self.fe = fft.rfft(self.ke, self.length)
        self.fi = fft.rfft(self.ki, self.length)
    def __call__(self, output):
        f = fft.rfft(output, self.length)
        ce = fft.irfft(f*self.fe, self.length)[len(self.ke)//2:len(self.ke)//2+self.p.n]
        ci = fft.irfft(f*self.fi, self.length)[len(self.ki)//2:len(self.ki)//2+self.p.n]
        glob = self.p.w_global_i*output.mean()
        return ce*100., ci*self.p.w_local_i + glob, glob

def initial_state(p):
    s = np.zeros((8, p.n), dtype=np.float64)
    s[0] = p.e_l
    s[1] = p.phi_0
    s[2] = 6.
    return s

@njit(cache=True)
def individual_update(s, pe, pi, stim, random_uniform, spiking, e_l, phi0, beta, gkmax, dt, tau_syn, tau_phi, tau_cl, tau_k):
    # Preserves IndividualModelUpdate's sequential use of NEW V and OLD output.
    vd = .25*np.sqrt(2.)/12.*20.**3/1000.
    ds = np.exp(-dt/tau_syn)
    dp = np.exp(-dt/tau_phi)
    dc = np.exp(-dt/tau_cl)
    dk = np.exp(-dt/tau_k)
    for j in range(s.shape[1]):
        v, phi, cl, gk, ie, ii, old, age = s[:, j]
        ie += pe[j]/tau_syn
        ii += pi[j]/tau_syn
        ecl = 26.7*np.log(cl/110.)
        gsum = 4. + (ie+ii+gk)/.2
        vinf = (4.*e_l + ii/.2*ecl + gk/.2*(-90.) + stim[j])/gsum
        v = vinf+(v-vinf)*np.exp(-dt*gsum/100.)
        if spiking:
            phi = phi0 + (phi-phi0)*dp + 2.5*old
            tap = max(.5-age, 0.)
            veff = tap*40./dt + (dt-tap)*v/dt
            cinf = tau_cl/vd/96500.*ii*(veff-ecl)+6.
            cl = cinf+(cl-cinf)*dc
            gk = gk*dk + gkmax*old/tau_k
            value = 1. if (.002*np.exp((v-phi)/beta)*dt > random_uniform[j] and age >= 5.) else 0.
            if value:
                v -= 20.
                age = 0.
            else:
                age += dt
        else:
            pinf = phi0+60.*old/.2
            phi += (pinf-phi)*(1.-dp)
            cinf = tau_cl/vd/96500.*ii*(v-ecl)+6.
            cl = cinf+(cl-cinf)*dc
            gk = gk*dk + gkmax*old*dt/tau_k
            value = .2/(1.+np.exp(-(v-phi)/beta))
        s[0,j], s[1,j], s[2,j], s[3,j] = v, phi, cl, gk
        s[4,j], s[5,j], s[6,j], s[7,j] = ie*ds, ii*ds, value, age

def numpy_reference_update(s, pe, pi, stim, u, p):
    # Independent vectorized transcription of the MATLAB source, for QA.
    v, phi, cl, gk, ie, ii, old, age = s.copy()
    ie = ie+pe/p.tau_syn_ms
    ii = ii+pi/p.tau_syn_ms
    ecl = 26.7*np.log(cl/110.)
    gsum = 4.+ie/.2+ii/.2+gk/.2
    vinf = (4.*p.e_l+ii/.2*ecl+gk/.2*(-90.)+stim)/gsum
    v = vinf+(v-vinf)*np.exp(-p.dt_ms/(100./gsum))
    if p.spiking:
        phi = p.phi_0+(phi-p.phi_0)*np.exp(-p.dt_ms/p.tau_phi_ms)+2.5*old
        tap = np.maximum(.5-age, 0.)
        veff = tap*40./p.dt_ms+(p.dt_ms-tap)*v/p.dt_ms
    else:
        phi += (p.phi_0+60.*old/.2-phi)*(1.-np.exp(-p.dt_ms/p.tau_phi_ms))
        veff = v
    cinf = p.tau_cl_ms/(.25*np.sqrt(2.)/12.*20.**3/1000.)/96500.*ii*(veff-ecl)+6.
    cl = cinf+(cl-cinf)*np.exp(-p.dt_ms/p.tau_cl_ms)
    gk = gk*np.exp(-p.dt_ms/p.tau_k_ms)+p.gk_max*old*(1. if p.spiking else p.dt_ms)/p.tau_k_ms
    if p.spiking:
        out = (.002*np.exp((v-phi)/p.beta)*p.dt_ms > u) & (age >= 5.)
        v[out] -= 20.
        age += p.dt_ms
        age[out] = 0.
    else:
        out = .2/(1.+np.exp(-(v-phi)/p.beta))
    return np.array([v,phi,cl,gk,ie*np.exp(-p.dt_ms/p.tau_syn_ms),ii*np.exp(-p.dt_ms/p.tau_syn_ms),out,age])

def update(s, pe, pi, stim, u, p):
    individual_update(s, pe, pi, stim, u, p.spiking, p.e_l, p.phi_0, p.beta, p.gk_max, p.dt_ms, p.tau_syn_ms, p.tau_phi_ms, p.tau_cl_ms, p.tau_k_ms)

def write_json(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')
    tmp.replace(path)

def qa():
    rng = np.random.RandomState(193)
    evidence = {}
    for name in ('exp2a', 'exp4a', 'exp4b'):
        p = protocol(name)
        proj = Projection(p)
        for label, x in [('random', rng.rand(p.n)), ('left_impulse', np.eye(1,p.n)[0]), ('constant', np.ones(p.n))]:
            pe, pi, glob = proj(x)
            re = convolve(x, proj.ke, mode='same', method='direct')*100.
            ri = convolve(x, proj.ki, mode='same', method='direct')*p.w_local_i + p.w_global_i*x.mean()
            err = max(np.max(np.abs(pe-re)), np.max(np.abs(pi-ri)))
            assert err < 5e-12, (name,label,err)
            evidence[name+'_projection_'+label] = float(err)
        a = initial_state(p)
        a[0] += rng.normal(0,4,p.n)
        a[1] += rng.rand(p.n)*10
        a[2] += rng.rand(p.n)*15
        a[3:6] = rng.rand(3,p.n)*5
        a[6] = (rng.rand(p.n) < .1) if p.spiking else rng.rand(p.n)*.2
        a[7] = rng.randint(0,15,p.n)
        b = a.copy()
        largest = 0.
        for _ in range(100):
            # Both paths independently project their own previous output.
            xa = a[6]*(1. if p.spiking else p.dt_ms)
            xb = b[6]*(1. if p.spiking else p.dt_ms)
            pe, pi, _ = proj(xa)
            re = convolve(xb,proj.ke,mode='same',method='direct')*100.
            ri = convolve(xb,proj.ki,mode='same',method='direct')*p.w_local_i + p.w_global_i*xb.mean()
            u = rng.rand(p.n)
            stimulus = np.full(p.n,80.)
            update(a,pe,pi,stimulus,u,p)
            b = numpy_reference_update(b,re,ri,stimulus,u,p)
            largest = max(largest,float(np.max(np.abs(a-b))))
            assert np.allclose(a,b,rtol=1e-9,atol=1e-9), (name,largest)
            if p.spiking:
                assert np.array_equal(a[6],b[6])
        evidence[name+'_100_steps_max_abs_error'] = largest
    evidence['status'] = 'PASS'
    evidence['boundary'] = 'Python/Numba versus independent NumPy plus direct convolution; not MATLAB-runtime validation.'
    write_json(OUT/'reference_qa.json', evidence)
    print(json.dumps(evidence,indent=2),flush=True)

def run(p, label):
    dest = OUT/'reference_runs'/label
    dest.mkdir(parents=True, exist_ok=True)
    if (dest/'result.json').exists():
        raise RuntimeError('Completed result already exists: '+str(dest))
    write_json(dest/'protocol.json', asdict(p))
    begin = time.time()
    proj = Projection(p)
    s = initial_state(p)
    pe, pi = np.zeros(p.n), np.zeros(p.n)
    rng = np.random.RandomState(p.seed)
    indices = np.arange(1,p.n+1)
    stim = ((indices > p.stimulus_left*p.n)&(indices < p.stimulus_right*p.n))*p.stimulus_pA
    zeros = np.zeros(p.n)
    nstep = int(round(p.duration_ms/p.dt_ms))
    record_every = int(round(p.record_ms/p.dt_ms))
    save_times = np.r_[np.arange(0,nstep,record_every)*p.dt_ms,p.duration_ms]
    nsave = len(save_times)
    # State is saved before update, as in the author's recorder.
    saved = np.lib.format.open_memmap(dest/'state.npy',mode='w+',dtype='float32',shape=(nsave,6,p.n))
    rate = np.lib.format.open_memmap(dest/'output.npy',mode='w+',dtype='float32',shape=(nstep,p.n)) if not p.spiking else None
    spikes = np.lib.format.open_memmap(dest/'spikes.npy',mode='w+',dtype='bool',shape=(nstep,p.n)) if p.spiking else None
    trace = np.zeros((nstep,9),np.float32)
    global_filter = 0.
    active_masks = [stim>0, (indices >= .45*p.n)&(indices < .55*p.n), indices>.8*p.n]
    for k in range(nstep):
        t = k*p.dt_ms
        if k%record_every == 0:
            saved[k//record_every] = s[:6]
        # ExternalInput.Evaluate consumes Gaussian draws even when sigma=0.
        rng.standard_normal(p.n)
        u = rng.rand(p.n) if p.spiking else zeros
        ext = stim if p.stimulus_start_ms < t < p.stimulus_stop_ms else zeros
        update(s,pe,pi,ext,u,p)
        out = s[6]
        hz = out*1000./(p.dt_ms if p.spiking else 1.)
        if p.spiking:
            spikes[k] = out
        else:
            rate[k] = hz
        trace[k,:4] = [hz.mean()]+[hz[mask].mean() for mask in active_masks]
        trace[k,4:] = [s[2].mean(),s[3].mean()/.2,s[1].mean(),s[5].mean()/.2,global_filter/.2]
        pe, pi, pg = proj(out*(1. if p.spiking else p.dt_ms))
        global_filter = (global_filter+pg/p.tau_syn_ms)*np.exp(-p.dt_ms/p.tau_syn_ms)
        if (k+1)%int(1000/p.dt_ms) == 0:
            if not np.all(np.isfinite(s)) or np.any(s[2]<=0):
                raise RuntimeError('Nonfinite state or nonpositive chloride')
            write_json(dest/'progress.json', {'status':'RUNNING','simulated_s':(k+1)*p.dt_ms/1000,'target_s':p.duration_ms/1000,'elapsed_wall_s':time.time()-begin,'last_second_E_Hz':float(trace[max(0,k-int(1000/p.dt_ms)+1):k+1,0].mean()),'Cl_mean_mM':float(s[2].mean()),'gK_mean_nS':float(s[3].mean()/.2)})
    saved[-1] = s[:6]
    saved.flush()
    if spikes is not None: spikes.flush()
    if rate is not None: rate.flush()
    np.savez_compressed(dest/'traces.npz',trace=trace,state_time_ms=save_times,trace_time_ms=(np.arange(nstep)+1)*p.dt_ms,trace_names=np.array(['E_mean_Hz','stimulus_region_Hz','middle_Hz','far_Hz','Cl_mean_mM','gK_mean_nS','phi_mean_mV','gI_mean_nS','gI_global_mean_nS']),kernel_E=proj.ke,kernel_I=proj.ki)
    np.savez_compressed(dest/'final_state.npz',state=s,projection_E=pe,projection_I=pi,global_filter=global_filter)
    result = {'status':'COMPLETE','protocol':asdict(p),'elapsed_wall_s':time.time()-begin,'tail_5s_E_Hz':float(trace[-int(5000/p.dt_ms):,0].mean()),'final_Cl_mM':float(s[2].mean()),'final_gK_nS':float(s[3].mean()/.2),'source_commit':'95ca7bdf71edbc46b3b26f6a5b73c43c5aa90ca7','port_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'MATLAB_runtime_identity':False,'no_added_global_slow_variable':True,'no_core':True}
    write_json(dest/'result.json',result)
    write_json(dest/'progress.json',result)
    print(json.dumps(result,indent=2),flush=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--qa',action='store_true')
    ap.add_argument('--experiment',choices=['exp2a','exp4a','exp4b','exp5_first'])
    ap.add_argument('--seed',type=int,default=20260915)
    ap.add_argument('--label')
    args = ap.parse_args()
    if args.qa: qa()
    else: run(protocol(args.experiment,args.seed),args.label or f'{args.experiment}_s{args.seed}')
