"""Fixed-resource response of the selected g40 mean model, not certified branches.

The particle update is the same as run_topic4_spatial_kinetic_candidate.py.
Full microscopic state, delayed arrivals and input RNG are resumable. The D path
scales log depletion of the actual Fig.5 9.42-s field, with physical endpoints.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
import time
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from scipy.sparse import load_npz
from run_topic4_spatial_kinetic_candidate import (
    ROOT, SOURCE, OUT as MODEL, NE, NI, DT, init_drive, EventDelay, write,
)

OUT = ROOT/'results/topic4_sef_hfo/kinetic_fixed_D_response_20260916'
GRID = MODEL/'coarse_40'
SEED = 9108401
DS = (0., .15, .20, .225, .25, .30, .50, 1.)


def field_at(D):
    with np.load(SOURCE/f'replay/runs/eta0.0005_s{SEED}/checkpoints/t9420ms.npz') as a:
        ref = a['slow__z'][:NE]
    assert np.all((ref > 0) & (ref < 1)) and 0 <= D <= 1
    if D in (0., 1.):
        z = np.full(NE, 1.-D)
        alpha = 0. if D == 0 else None
    else:
        h = -np.log(ref)
        hi = 1.
        while np.mean(np.exp(-hi*h)) > 1-D:
            hi *= 2
        alpha = brentq(lambda a: np.mean(np.exp(-a*h))-(1-D), 0., hi, xtol=1e-13)
        z = np.exp(-alpha*h)
    assert abs((1-z.mean())-D) < 1e-12
    return np.r_[z, np.ones(NI)], alpha


class Stepper:
    def __init__(self, history_ms=8000, anchor_ms=10370, D=None, fast=False):
        self.prep = json.loads((GRID/'prepared.json').read_text())
        self.geo = dict(np.load(GRID/'geometry.npz'))
        self.md = dict(np.load(GRID/'model.npz'))
        self.obs = dict(np.load(SOURCE/'approx/coarse_20/geometry.npz'))
        self.protocol = json.loads((ROOT/'results/topic4_sef_hfo/fig5_preentry_event_audit_20260914/protocol.json').read_text())
        assert self.prep['graph_identity'] == self.protocol['identity']
        self.p = p = self.prep['params']; self.n = 1600
        self.cell = np.r_[self.geo['cell_e'], self.geo['cell_i']+self.n]
        self.counts = np.r_[self.geo['count_e'], self.geo['count_i']]
        self.safe = np.maximum(self.counts, 1.)
        self.rc = np.bincount(self.geo['g175'], minlength=3)
        self.theta = np.r_[self.geo['vtheta_e'], self.geo['vtheta_i']]
        assert (self.theta[:NE]<18).sum() == 781 and not (self.theta[:NE]>18).any()
        tau = np.r_[np.full(NE,p['tau_m_E']),np.full(NI,p['tau_m_I'])]
        self.dv = np.exp(-DT/tau)
        self.scale_a = tau/p['tau_r_AMPA']; self.scale_g = tau/p['tau_r_GABA']
        self.ext_incr = self.scale_a*np.r_[np.full(NE,p['J_ext_E']),np.full(NI,p['J_ext_I'])]
        self.rsa = np.exp(-DT/p['tau_r_AMPA']); self.ria = np.exp(-DT/p['tau_d_AMPA'])
        self.rsg = np.exp(-DT/p['tau_r_GABA']); self.rig = np.exp(-DT/p['tau_d_GABA'])
        self.refs = np.r_[np.full(NE,round(p['tau_ref_E']/DT)),np.full(NI,round(p['tau_ref_I']/DT))]
        self.oua = np.exp(-DT/p['tau_n'])
        self.oub = p['sigma_n']*1e-3*np.sqrt(p['tau_n']/2.)*np.sqrt(1-self.oua**2)
        weights = {k:load_npz(GRID/f'delay_{k}.npz') for k in ('ee','ei','ie','ii')}
        if fast:
            from topic4_kinetic_delay_flat import FlatMeanDelay
            self.delay = FlatMeanDelay(weights, self.n, self.prep['max_delay_steps'])
        else:
            self.delay = EventDelay(weights, None, self.n, self.prep['max_delay_steps'])
        cpdir = SOURCE/f'replay/runs/eta0.0005_s{SEED}/checkpoints'
        with np.load(cpdir/f't{history_ms}ms.npz') as f:
            cp = {k:f[k] for k in ('V','ref','s_E','I_E','s_I','I_I','slow__z','slow__m','ring_sE','ring_sI','__meta__')}
        old = json.loads(str(cp['__meta__']))
        with np.load(cpdir/f't{anchor_ms}ms.npz') as f:
            meta = json.loads(str(f['__meta__']))
            drive_state = {k:f[k] for k in ('external_drive__field_state','external_drive__cached')}
        for key in ('V','ref','s_E','I_E','s_I','I_I'):
            setattr(self, key, cp[key].copy())
        self.Z = cp['slow__z'].copy(); self.M = cp['slow__m'].copy()
        self.initial_M = self.M.copy()
        shift = (meta['step']-old['step']) % len(cp['ring_sE'])
        self.pending_a = np.roll(cp['ring_sE'], shift, axis=0)
        self.pending_g = np.roll(cp['ring_sI'], shift, axis=0)
        self.rng = np.random.default_rng(); self.rng.bit_generator.state = meta['rng_state']
        self.drive = init_drive(self.geo, self.prep, drive_state, meta, SEED)
        self.xi = meta['xi']; self.anchor_ms = anchor_ms; self.origin_step = meta['step']; self.t = 0
        self.D = D; self.history_ms = history_ms; self.alpha = None
        if D is not None:
            self.Z, self.alpha = field_at(D)
        self.initial_Z = self.Z.copy()

    def advance(self, duration_ms, progress=None):
        steps = round(duration_ms/DT)
        assert steps % 100 == 0 and self.t % 100 == 0
        field = np.zeros((steps//10,400),np.int32)
        regions = np.zeros((steps//10,3),np.int32)
        slow = np.zeros((steps//100,7))
        external = np.empty(steps,np.float32)
        block = np.zeros(400,np.int32); rb = np.zeros(3,np.int32)
        last = time.time()
        for k in range(steps):
            t = self.t
            (ma,mg),_ = self.delay.take(t)
            self.s_E *= self.rsa; self.s_I *= self.rsg
            self.s_E += ma[self.cell]*self.scale_a; self.s_I += mg[self.cell]*self.scale_g
            if t < len(self.pending_a):
                slot = (self.origin_step+t)%len(self.pending_a)
                self.s_E += self.pending_a[slot]; self.s_I += self.pending_g[slot]
            self.xi = self.oua*self.xi+self.oub*self.rng.standard_normal()
            nu_now = max(float(self.md['nu_ext_per_ms'])+self.xi,0.)
            nu = np.full(NE+NI,nu_now)
            nu[:NE] = np.maximum(nu_now+self.drive.step(self.anchor_ms+t*DT),0.)
            self.s_E += self.rng.poisson(nu*DT,size=NE+NI)*self.ext_incr
            external[k] = nu_now
            self.I_E = self.s_E+(self.I_E-self.s_E)*self.ria
            self.I_I = self.s_I+(self.I_I-self.s_I)*self.rig
            current = self.I_E-self.Z*self.I_I-.0005*self.M
            self.ref -= 1; np.maximum(self.ref,0,out=self.ref)
            free = self.ref == 0
            self.V = np.where(free,current+(self.V-current)*self.dv,self.p['V_reset'])
            spk = free & (self.V>=self.theta)
            self.V[spk] = self.p['V_reset']; self.ref[spk] = self.refs[spk]
            if self.D is None:
                self.Z[:NE] += DT/5000.*((self.I_I[:NE]<self.protocol['I_th']).astype(float)-self.Z[:NE])
                np.clip(self.Z[:NE],0.,1.,out=self.Z[:NE])
            self.M[:NE] -= DT/1000.*self.M[:NE]; self.M[:NE] += spk[:NE]
            s = np.bincount(self.cell[spk],minlength=2*self.n).astype(np.int32)
            self.delay.push(t,s[:self.n]/self.safe[:self.n],s[self.n:]/self.safe[self.n:])
            block += np.bincount(self.obs['cell_e'][spk[:NE]],minlength=400).astype(np.int32)
            rb += np.bincount(self.geo['g175'][spk[:NE]],minlength=3).astype(np.int32)
            if (k+1)%10 == 0:
                field[k//10] = block; regions[k//10] = rb
                block.fill(0); rb.fill(0)
            if (k+1)%100 == 0:
                slow[k//100] = [(t+1)*DT,self.Z[:NE].mean(),self.M[:NE].mean(),
                    *[self.M[:NE][self.geo['g175']==j].mean() for j in range(3)],
                    np.mean(self.I_I[:NE]>=self.protocol['I_th'])]
            self.t += 1
            if progress and time.time()-last>25:
                progress(self.t*DT); last=time.time()
        assert np.array_equal(field.sum(1),regions.sum(1))
        assert all(np.isfinite(getattr(self,k)).all() for k in ('V','I_E','I_I','Z','M'))
        assert np.all((self.Z>=0)&(self.Z<=1)) and np.all(self.M>=0)
        if self.D is not None:
            assert np.array_equal(self.Z,self.initial_Z)
        return dict(field_1ms=field,regions_1ms=regions,slow_10ms=slow,global_external_rate=external)

    def save(self,path):
        meta = dict(t=self.t,xi=self.xi,anchor_ms=self.anchor_ms,origin_step=self.origin_step,
            D=self.D,alpha=self.alpha,history_ms=self.history_ms,rng_state=self.rng.bit_generator.state,
            drive_rng=self.drive._rng.bit_generator.state,drive_next=self.drive._next_step,drive_last=self.drive._last_step)
        arrays = {k:getattr(self,k) for k in ('V','ref','s_E','I_E','s_I','I_I','Z','M','initial_M','initial_Z')}
        empty = np.empty((0,NE+NI))
        np.savez_compressed(path,**arrays,delay_mean=self.delay.mean,
            pending_a=self.pending_a if self.t<len(self.pending_a) else empty,
            pending_g=self.pending_g if self.t<len(self.pending_g) else empty,
            drive_state=self.drive._state,drive_cached=self.drive._cached,__meta__=json.dumps(meta))

    def restore(self,path):
        with np.load(path) as f:
            meta=json.loads(str(f['__meta__']))
            for k in ('V','ref','s_E','I_E','s_I','I_I','Z','M','initial_M','initial_Z','pending_a','pending_g'):
                setattr(self,k,f[k].copy())
            self.delay.mean[:]=f['delay_mean']; self.delay.var.fill(0)
            self.drive._state=f['drive_state'].copy(); self.drive._cached=f['drive_cached'].copy()
        for k in ('t','xi','anchor_ms','origin_step','D','alpha','history_ms'):setattr(self,k,meta[k])
        self.rng.bit_generator.state=meta['rng_state']; self.drive._rng.bit_generator.state=meta['drive_rng']
        self.drive._next_step=meta['drive_next']; self.drive._last_step=meta['drive_last']


def canary(fast=False):
    dest=OUT/('canary_flat' if fast else 'canary');dest.mkdir(parents=True,exist_ok=False)
    s=Stepper(8000,8000,fast=fast)
    out=s.advance(100)
    prior=ROOT/'results/topic4_sef_hfo/fig5_spatial_kinetic_full_trajectory_20260916/canary'
    with np.load(prior/'trajectory.npz') as f:
        checks={k:bool(np.array_equal(out[k],f[k])) for k in ('field_1ms','regions_1ms','global_external_rate')}
    with np.load(prior/'end_state.npz') as f:
        checks.update({k:bool(np.array_equal(getattr(s,k),f[k])) for k in ('V','ref','s_E','I_E','s_I','I_I','Z','M')})
    assert all(checks.values()), checks
    s.save(dest/'restart.npz'); one=s.advance(50)
    expected={k:getattr(s,k).copy() for k in ('V','ref','s_E','I_E','s_I','I_I','Z','M')}
    ring=s.delay.mean.copy();rng=s.rng.bit_generator.state;xi=s.xi
    s.restore(dest/'restart.npz');two=s.advance(50)
    checks['restart_observers']=all(np.array_equal(one[k],two[k]) for k in one)
    checks['restart_state']=all(np.array_equal(expected[k],getattr(s,k)) for k in expected)
    checks['restart_delay']=np.array_equal(ring,s.delay.mean)
    checks['restart_input']=rng==s.rng.bit_generator.state and xi==s.xi
    assert all(checks.values()), checks
    refD=1-np.load(SOURCE/f'replay/runs/eta0.0005_s{SEED}/checkpoints/t9420ms.npz')['slow__z'][:NE].mean()
    z,a=field_at(refD);checks['field_anchor_alpha1']=abs(a-1)<1e-12
    for d in DS:
        z,a=field_at(d);assert np.all((z>=0)&(z<=1)) and abs(1-z[:NE].mean()-d)<1e-12
    write(dest/'qa.json',{k:bool(v) for k,v in checks.items()})
    print(json.dumps(checks,default=bool),flush=True)


def run(D,history,duration,extend=False,fast=False):
    folder=OUT/'runs'/f'D{D:.6f}_h{history}'
    if not extend:folder.mkdir(parents=True,exist_ok=False)
    start=time.time();s=Stepper(history,10370,D,fast=fast)
    if extend:s.restore(folder/'checkpoint.npz')
    offset=s.t*DT
    cfg=dict(D=D,history_ms=history,external_anchor_ms=10370,seed=SEED,Z='fixed field',M='dynamic',model='g40_mean',delay_engine='flat' if fast else 'original')
    def status(ms):
        write(folder/'status.json',dict(status='RUNNING',pid=os.getpid(),config=cfg,completed_ms=ms,wall_s=time.time()-start))
        print(folder.name,'ms',ms,flush=True)
    status(offset);out=s.advance(duration,status)
    np.savez_compressed(folder/f'observations_{int(offset):05d}_{int(offset+duration):05d}.npz',**out)
    s.save(folder/'checkpoint.npz')
    write(folder/'qa.json',dict(frozen_Z_bitwise_constant=bool(np.array_equal(s.Z,s.initial_Z)),
        M_dynamic=bool(np.any(s.M!=s.initial_M)),finite_physical_state=True,count_conservation=True,
        D_realized=float(1-s.Z[:NE].mean()),alpha=s.alpha,complete_delay_and_input_checkpoint=True))
    write(folder/'status.json',dict(status='COMPLETE',config=cfg,completed_ms=s.t*DT,wall_s=time.time()-start))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('task',choices=('canary','run'))
    ap.add_argument('--D',type=float);ap.add_argument('--history',type=int,choices=(8000,10370))
    ap.add_argument('--duration',type=int,default=4000);ap.add_argument('--extend',action='store_true')
    ap.add_argument('--fast',action='store_true')
    a=ap.parse_args()
    if a.task=='canary':canary(a.fast)
    else:run(a.D,a.history,a.duration,a.extend,a.fast)
