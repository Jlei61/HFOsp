#!/usr/bin/env python3
"""Paired native Figure 6 check: one logistic replaces both G/M clip gates.

The frozen spatial network, noise seed, Z, M retention rule and current routing
are preserved. This experiment tests a complete recovery/re-entry sequence;
equation QA alone never authorizes replacing the illustrated trajectory.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from scipy.special import expit

MAIN = Path('/home/honglab/leijiaxin/HFOsp')
SOURCE = Path('/data/hfosp/topic4_sef_hfo/adaptation_history_20261031')
OUT = Path('/data/hfosp/topic4_sef_hfo/logistic_feedback_20261102')
MODEL = 'ZGM_SHARED_LOGISTIC_G_M_V1'
sys.path.insert(0, str(MAIN/'scripts'))
spec = importlib.util.spec_from_file_location('frozen_clip_unified',
    SOURCE/'source/run_topic4_unified_current_adaptation.py')
u = importlib.util.module_from_spec(spec)
spec.loader.exec_module(u)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sigmoid(rate, midpoint_Hz, width_Hz):
    if not (np.isfinite(midpoint_Hz) and np.isfinite(width_Hz) and width_Hz > 0):
        raise ValueError('Logistic midpoint must be finite and width strictly positive')
    return expit((np.asarray(rate)-midpoint_Hz)/width_Hz)


class LogisticCurrentSlow(u.UnifiedCurrentSlow):
    midpoint_Hz = 350.
    width_Hz = 75.

    def apply_currents(self, ie, ii, labels=None, rec=None):
        # The frozen parent's current calculation does not depend on its gate;
        # its only gate consumers here are the recorded q and the next step().
        # Override both before that step, keeping all current/readout code intact.
        value = super().apply_currents(ie, ii, labels, rec)
        self.gate = float(sigmoid(self.r_global, self.midpoint_Hz, self.width_Hz))
        if self._step_index % 200 == 0:
            self.unified_records[-1][1] = self.gate
        return value


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'protocol.json').exists():
        p = read(OUT/'protocol.json')
        assert p['runner_sha256'] == sha(__file__), 'Prepared runner changed'
        return p
    p = copy.deepcopy(read(SOURCE/'protocol.json'))
    assert p['new_runner_sha256'] == sha(u.__file__)
    for path, digest in p['source_hashes'].items():
        assert sha(path) == digest, path
    # Remove unrelated prior seed-search jobs/budget from this new protocol.
    for key in ('initial_jobs', 'candidate_seeds', 'max_workers', 'source_jobs'):
        p.pop(key, None)
    p.update(model=MODEL,runner_sha256=sha(__file__),deadline_epoch=time.time()+24*3600,
        authorization='2026-11-02 author: logistic, replacing both G and M clip gates',
        paired_reference=str(SOURCE/'runs/seed9108404'),
        parameters=dict(midpoint_Hz=350.,width_Hz=75.,global_gain=30.,global_tau_s=.5,
                        eta_M=.0005,high_jump_mV=7.68,seed=9108404,horizon_s=120.),
        parameter_rationale='Match clip midpoint q=1/2 at R=350 Hz and local slope 1/300 Hz^-1; width=300/4. No trajectory-dependent fitting.',
        equations=dict(sigmoid='S(R;R0,k)=1/(1+exp(-(R-R0)/k)), k>0',
          global_feedback='tau_G*dG/dt=G_max*S(R)-G',
          adaptation='dM_i/dt=-M_i/tau_M(R)+[a0/tau_M(R)+a1*S(R)]/eta_M*sum_k delta(t-t_i^k)',
          membrane='tau_m*dV_i/dt=-V_i+IE_i-Z_i*II_i-Z_i*G*(V_i-EG)-eta_M*M_i',
          tau_M='5s if causal all-E R<=5Hz, otherwise 0.5s (unchanged)',
          resource='5s*dZ_i/dt=1{II_i+(18-EG)*G<95.19851312666987}-Z_i (unchanged)'),
        changed_mechanisms=['shared logistic rate gate for G and spike-triggered M increment'],
        unchanged=['network and input statistics','seed9108404','Z load and recovery','M retention time rule','G kinetics','readout definitions'],
        qualification='Complete native 120s; initial brief activity, high entry, autonomous exit, both-core Z recovery, >=10 complete short events spanning >=5s, then a new sustained high entry.',
        statistical_unit='One fixed network and one paired noise realization; qualitative trajectory check, not population equivalence.',
        figure_replaced=False,human_review='PENDING',formal_bifurcation=False)
    write(OUT/'protocol.json',p)
    shutil.copy2(SOURCE/'geometry.npz',OUT/'geometry.npz')
    (OUT/'source').mkdir(exist_ok=True)
    for path in [Path(__file__),Path(u.__file__),
                 MAIN/'.worktrees/topic4-continuous-core-state-r1/scripts/analyze_topic4_unified_current_adaptation.py']:
        shutil.copy2(path,OUT/'source'/path.name)
    return p


def qa():
    p = prepare()
    cfg = u.base.old.MZSlowVarsConfig(use_z=True,use_m=True,tau_z=5000.,
        I_th_EI=95.19851312666987,tau_adp=1000.,eta_m=.0005)
    old = u.UnifiedCurrentSlow(8,18.,cfg,NE=6,global_gain=30.,mode='native',gamma=0.)
    new = LogisticCurrentSlow(8,18.,cfg,NE=6,global_gain=30.,mode='native',gamma=0.)
    grid = np.array([-1e9,0,100,200,350,500,1e9])
    s = sigmoid(grid,350,75)
    assert np.all(np.isfinite(s)) and np.all(np.diff(s)>0) and s[0]==0 and s[-1]==1
    assert s[4] == .5
    slope = (sigmoid(350.001,350,75)-sigmoid(349.999,350,75))/.002
    assert abs(slope-1/300)<1e-10
    errors=[]
    for rate in [0.,5.,5.1,100.,200.,350.,500.]:
        for o in (old,new):
            o.r_global=rate; o.global_state=.2; o.m[:6]=np.arange(6); o.z[:6]=.7
            o.global_reversal=-17.662847938268442
        ie=np.arange(8)*10.; ii=np.arange(8)*3.
        np.testing.assert_array_equal(new.apply_currents(ie,ii),old.apply_currents(ie,ii))
        np.testing.assert_array_equal(new._I_I_last,old._I_I_last)
        np.testing.assert_array_equal(new.shunt_g_at_E(),old.shunt_g_at_E())
        q=float(sigmoid(rate,350,75)); tau=new.current_tau_ms()
        assert new.gate==q==new.unified_records[-1][1]
        spikes=np.array([1,0,1,0,1,0,1,1],bool)
        expected=new.m[:6]*np.exp(-.1/tau)
        expected[spikes[:6]]+=(.0005*1000/tau+7.68*q)/cfg.eta_m
        expected_g=.2*np.exp(-.1/500)+q*(1-np.exp(-.1/500))
        new.step(spikes,None,.1);old.step(spikes,None,.1)
        errors.append(float(abs(new.m[:6]-expected).max()))
        assert new.global_state==expected_g
        np.testing.assert_array_equal(new.z,old.z)
        assert np.all(new.m[6:]==0) and np.all(new.z[6:]==1)
        new._step_index=old._step_index=0
    assert max(errors)==0
    result=dict(status='PASS',shared_logistic_G_M=True,midpoint_and_slope_match=True,
        currents_at_fixed_state_unchanged=True,Z_routing_unchanged=True,
        maximum_spike_increment_error=max(errors),zero_rate_tail=float(s[1]),
        native_outcome_not_established_by_equation_QA=True,runner_sha256=p['runner_sha256'])
    write(OUT/'equation_qa.json',result)
    print(json.dumps(result),flush=True)


def worker(args):
    p=prepare();assert read(OUT/'equation_qa.json')['status']=='PASS'
    job=copy.deepcopy(read(SOURCE/'runs/seed9108404/result.json')['job'])
    job.update(name=args.name,model=MODEL,seed=9108404,horizon_s=120.,checkpoint_s=2.,
        device=args.device,stop_after_second_entry=False,qa=False,
        stage='paired_logistic_G_M_validation',feedback_kind='logistic_shared',
        logistic_midpoint_Hz=args.midpoint,logistic_width_Hz=args.width,
        no_intervention=True,mechanism='One parameterized logistic drives both G and M increment; no K state.')
    path=OUT/'jobs'/f'{args.name}.json'
    if path.exists():assert read(path)==job
    else:write(path,job)
    folder=OUT/'runs'/args.name;folder.mkdir(parents=True,exist_ok=True)
    LogisticCurrentSlow.midpoint_Hz=args.midpoint
    LogisticCurrentSlow.width_Hz=args.width
    capture0,restore0=u.checkpoint.capture,u.checkpoint.restore_slow
    def capture(**kw):
        state=capture0(**kw);o=kw['slow'];step=int(kw['step'])
        state['unified_current_adaptation']=dict(global_state=o.global_state,model=MODEL)
        for stream,rows,keys in [
            ('adaptation_chunks',o.unified_records,['time_ms','q','global_s','tau_A_s','adaptation_mean_mV','adaptation_max_mV'])]:
            if rows:
                dest=folder/stream;dest.mkdir(exist_ok=True);r=np.asarray(rows)
                np.savez_compressed(dest/f'{round(r[0,0]*10):010d}_{step:010d}.npz',
                    **{key:r[:,i] for i,key in enumerate(keys)})
                rows.clear()
        if o.dense_times:
            dest=folder/'dense_readout_chunks';dest.mkdir(exist_ok=True)
            np.savez_compressed(dest/f'{round(o.dense_times[0]*10):010d}_{step:010d}.npz',
                time_ms=o.dense_times,contact_current=o.dense_values)
            o.dense_times.clear();o.dense_values.clear()
        return state
    def restore(state,obj):
        restore0(state,obj)
        assert state['unified_current_adaptation']['model']==MODEL
        obj.global_state=state['unified_current_adaptation']['global_state']
    u.checkpoint.capture,u.checkpoint.restore_slow=capture,restore
    u.fixed.OUT,u.fixed.prepare,u.fixed.TerminationSlow=OUT,lambda:p,LogisticCurrentSlow
    u.fixed.worker(args.name)
    result=read(folder/'result.json')
    result.update(model=MODEL,display_stop_s=result['end_s'],whole_trajectory_from_t0=True,
                  independent_K_absent=True,shared_logistic_G_M=True)
    result['tracker']['stop_reason']='SIMULATION_HORIZON' if result['status']=='COMPLETE' else result['tracker']['stop_reason']
    write(folder/'result.json',result);write(folder/'progress.json',result)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['qa','worker'])
    ap.add_argument('--name',default='logistic_r350_k75_s9108404')
    ap.add_argument('--midpoint',type=float,default=350.)
    ap.add_argument('--width',type=float,default=75.)
    ap.add_argument('--device',type=int,default=0)
    args=ap.parse_args()
    sigmoid(0,args.midpoint,args.width)
    if args.action=='qa':qa()
    else:worker(args)
