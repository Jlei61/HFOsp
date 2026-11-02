#!/usr/bin/env python3
"""Independent Z/G/M candidate: one current-type spike-triggered adaptation.

The historical carrier and graph are frozen read-only dependencies. New code
and outputs live separately; no K conductance/state is retained.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import time

MAIN = Path('/home/honglab/leijiaxin/HFOsp')
sys.path.insert(0, str(MAIN / 'scripts'))
import numpy as np
import run_topic4_fixed_zm_termination as fixed
import checkpoint

carrier, base = fixed.carrier, fixed.carrier.base
SOURCE = Path('/data/hfosp/topic4_sef_hfo/fig5_global_feedback_response_20260923')
OUT = Path('/data/hfosp/topic4_sef_hfo/unified_current_adaptation_20261030')
MODEL = 'ZGM_UNIFIED_CURRENT_ADAPTATION_V1'


class UnifiedCurrentSlow(fixed.TerminationSlow):
    """M is the sole intrinsic adaptation state; G remains a global synapse."""
    high_jump_mV = 7.68  # 0.16 * (18 - (-30)): frozen reference-voltage conversion.
    global_tau_ms = 500.
    tau_active_ms = 500.
    tau_quiet_ms = 5000.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_state = 0.
        self.gate = 0.
        self.dense_times = []
        self.dense_values = []
        self.unified_records = []

    def apply_currents(self, ie, ii, labels=None, rec=None):
        self.gate = float(np.clip((self.r_global - 200.) / 300., 0., 1.))
        raw = self.global_gain * self.global_state
        self.g_global[:] = raw * self.z[:self.NE]
        total = ii.copy()
        total[:self.NE] += raw * (18. - self.global_reversal)
        self.delivered = self._I_I_last = total
        self.raw_mean = float(ii[:self.NE].mean())
        value = ie - self.z * ii - self.cfg.eta_m * self.m
        v = self.voltage[:self.NE] if self.voltage is not None else np.zeros(self.NE)
        gi = self.g_global * (v - self.global_reversal)
        if self._step_index % 10 == 0:
            self.extra_records.append([self._step_index*.1, self.r_global, raw,
                self.g_global.mean(), 0., 0., gi.mean()])
        if self._step_index % 200 == 0:
            self.unified_records.append([self._step_index*.1, self.gate, self.global_state,
                self.current_tau_ms()/1000., self.cfg.eta_m*self.m[:self.NE].mean(),
                self.cfg.eta_m*self.m[:self.NE].max()])
        if self.current_recorder is not None:
            # Native 10 kHz pre-Z proxy, matching the current Fig6 waveform.
            self.dense_times.append(self._step_index*.1)
            self.dense_values.append(self.current_recorder.sample(ie, ii))
            if self._step_index % 20 == 0:
                actual = abs(ie[:self.NE]) + abs(self.z[:self.NE]*ii[:self.NE]) + abs(gi)
                self.contact_records.append(np.array([np.dot(w, actual[ix])
                    for ix, w in zip(self.current_recorder._idx, self.current_recorder._w)]))
                self.field_records.append(np.bincount(self.current_cells, weights=actual,
                    minlength=400)/self.current_cell_counts)
                self.current_record_times.append(self._step_index*.1)
        return value

    def current_tau_ms(self):
        return self.tau_quiet_ms if self.r_global <= 5. else self.tau_active_ms

    def step(self, spk, labels, dt):
        # All gates use the pre-step causal global E rate, as in the parent.
        tau = self.current_tau_ms()
        z = self.z[:self.NE]
        z += (dt/self.cfg.tau_z) * ((self._I_I_last[:self.NE] < self.cfg.I_th_EI) - z)
        np.clip(z, 0., 1., out=z)
        m = self.m[:self.NE]
        m *= np.exp(-dt/tau)
        # Baseline integrated current eta_old * tau_old = 0.0005 mV s.
        jump = .0005 * 1000./tau + self.high_jump_mV*self.gate
        m[np.asarray(spk[:self.NE], bool)] += jump/self.cfg.eta_m
        decay = np.exp(-dt/self.global_tau_ms)
        self.global_state = self.global_state*decay + self.gate*(1.-decay)
        self.r_global = self.r_global*np.exp(-dt/15.) + np.count_nonzero(spk[:self.NE])/self.NE*1000./15.
        self._step_index += 1


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    p = copy.deepcopy(base.read(SOURCE/'protocol.json'))
    for path, digest in p['source_hashes'].items():
        assert base.sha(path) == digest, path
    p.update(model=MODEL, producer_sha256=base.sha(carrier.__file__),
        wrapper_sha256=base.sha(fixed.__file__), new_runner_sha256=base.sha(__file__),
        deadline_epoch=time.time()+24*3600, protected_global_authorized=False,
        authorization='2026-10-30 user approved unified eta_M M adaptation and Fig6 termination candidate',
        equations=dict(membrane='tau_m Vdot=-V+IE-Z II-Z G(V-EG)-eta_M M',
            adaptation='M*=exp(-dt/tauA); spike adds [0.0005*1s/tauA+7.68*q]/eta_M',
            tauA='5s if causal global E R<=5Hz, otherwise0.5s',
            global_feedback='G=30*s; 0.5s sdot=q-s; q=clip((R-200)/300,0,1); R filter15ms',
            resource='5s Zdot=1{II+(18-EG)G<95.19851312666987}-Z',
            K_state=False, K_conductance=False),
        interpretation='Reference-voltage mean-current matched phenomenological reduction; not exact conductance equivalence.',
        observation='From zero adaptation and Z=1 at t=0, no reset or scheduled intervention.',
        independent_unit='One network and one noise realization; development candidate, not cohort validation.',
        human_review='PENDING', original_figure_replaced=False)
    base.write(OUT/'protocol.json', p)
    if not (OUT/'geometry.npz').exists():
        shutil.copy2(SOURCE/'geometry.npz', OUT/'geometry.npz')
    return p


def qa():
    p = prepare()
    cfg = base.old.MZSlowVarsConfig(use_z=True,use_m=True,tau_z=5000.,
        I_th_EI=95.19851312666987,tau_adp=1000.,eta_m=.0005)
    o = UnifiedCurrentSlow(8,18.,cfg,NE=6,global_gain=30.,mode='native',gamma=0.)
    o.global_reversal = -17.662847938268442
    assert not hasattr(o, 'g_k')
    maxerr = 0.
    for rate in [0.,5.,5.1,100.,350.,500.]:
        o.r_global=rate; o.global_state=.2; o.m[:6]=np.arange(6); o.z[:6]=.7
        ie=np.arange(8)*10.; ii=np.arange(8)*3.
        cur=o.apply_currents(ie,ii)
        assert np.array_equal(cur,ie-o.z*ii-cfg.eta_m*o.m)
        assert np.allclose(o.shunt_g_at_E(),.7*6.)
        tau=o.current_tau_ms(); q=o.gate; z0=o.z.copy(); m0=o.m.copy()
        sp=np.array([True,False,True,False,True,False,True,True])
        expected=m0[:6]*np.exp(-.1/tau)
        expected[sp[:6]]+=(.0005*1000./tau+7.68*q)/cfg.eta_m
        zi=o._I_I_last[:6]<cfg.I_th_EI
        o.step(sp,None,.1)
        maxerr=max(maxerr,float(abs(o.m[:6]-expected).max()))
        assert np.allclose(o.z[:6],z0[:6]+.1/5000*(zi-z0[:6]))
        assert np.all(o.z[6:]==1.) and np.all(o.m[6:]==0.)
    # Changing M changes current, not the membrane conductance.
    old_g=o.shunt_g_at_E().copy();o.m[:6]+=10000
    assert np.array_equal(old_g,o.shunt_g_at_E())
    result=dict(status='PASS',no_K_state=True,no_K_conductance=True,
        first_order_spike_update_max_abs_error=maxerr,I_cells_unchanged=True,
        native_resource_update=True,causal_global_feedback=True,
        runner_sha256=p['new_runner_sha256'])
    base.write(OUT/'equation_qa.json',result)
    print(json.dumps(result),flush=True)


def worker(args):
    p=prepare()
    assert base.read(OUT/'equation_qa.json')['status']=='PASS'
    name=args.name
    job=copy.deepcopy(base.read(SOURCE/'runs/G30_response0.5_s9108405/result.json')['job'])
    job.update(name=name,horizon_s=args.seconds,checkpoint_s=2.,device=args.device,
        stage='unified_current_candidate',qa=False,stop_after_second_entry=False,
        sahp_gain=0.,high_jump_mV=args.high_jump,backend=args.backend,
        model=MODEL,K_state=False,seed=args.seed)
    jobpath=OUT/'jobs'/f'{name}.json'
    if jobpath.exists():assert base.read(jobpath)==job
    else:base.write(jobpath,job)
    folder=OUT/'runs'/name;folder.mkdir(parents=True,exist_ok=True)
    UnifiedCurrentSlow.high_jump_mV=args.high_jump
    capture0,restore0=checkpoint.capture,checkpoint.restore_slow
    def capture(**kw):
        state=capture0(**kw);o=kw['slow'];k=int(kw['step'])
        state['unified_current_adaptation']=dict(global_state=o.global_state,model=MODEL)
        if o.dense_times:
            d=folder/'dense_readout_chunks';d.mkdir(exist_ok=True)
            np.savez_compressed(d/f'{round(o.dense_times[0]*10):010d}_{k:010d}.npz',
                time_ms=o.dense_times,contact_current=o.dense_values)
            o.dense_times.clear();o.dense_values.clear()
        if o.unified_records:
            d=folder/'adaptation_chunks';d.mkdir(exist_ok=True);r=np.asarray(o.unified_records)
            np.savez_compressed(d/f'{round(r[0,0]*10):010d}_{k:010d}.npz',
                time_ms=r[:,0],q=r[:,1],global_s=r[:,2],tau_A_s=r[:,3],
                adaptation_mean_mV=r[:,4],adaptation_max_mV=r[:,5])
            o.unified_records.clear()
        return state
    def restore(state,obj):
        restore0(state,obj)
        obj.global_state=state['unified_current_adaptation']['global_state']
    checkpoint.capture,checkpoint.restore_slow=capture,restore
    fixed.OUT,fixed.prepare,fixed.TerminationSlow=OUT,lambda:p,UnifiedCurrentSlow
    if args.backend=='cpu':carrier.wrap_simulator=lambda fn,device_index:fn
    fixed.worker(name)
    r=base.read(folder/'result.json')
    r.update(model=MODEL,display_stop_s=r['end_s'],single_intrinsic_adaptation_state=True,
        independent_K_absent=True,whole_trajectory_from_t0=True,original_figure_replaced=False)
    r['tracker']['stop_reason']='SIMULATION_HORIZON'
    base.write(folder/'result.json',r);base.write(folder/'progress.json',r)
    print(json.dumps(dict(name=name,end_s=r['end_s'],tracker=r['tracker'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['qa','worker'])
    parser.add_argument('--name',default='unified_ref48_s9108405')
    parser.add_argument('--seconds',type=float,default=60.)
    parser.add_argument('--seed',type=int,default=9108405)
    parser.add_argument('--device',type=int,default=0)
    parser.add_argument('--backend',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--high-jump',type=float,default=7.68)
    args=parser.parse_args()
    qa() if args.command=='qa' else worker(args)
