#!/usr/bin/env python3
"""Bounded eta_M step at the same high state; never a native-cycle claim."""
import argparse
from datetime import datetime
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
import numpy as np
import run_topic4_weak_fast_recurrence as base
from run_topic4_reset_state_diagnosis import read, write, save_pickle, load_pickle, sha

ROOT = base.ROOT
WINDOW = ROOT / 'results/topic4_sef_hfo/fig5_m_overnight_exploration_20260913'
OUT = WINDOW / 'high_state_M_gain_probe'
PARENT = ROOT / 'results/topic4_sef_hfo/reset_state_diagnosis_20260911/parents/high_parent.pkl'
SOURCE = ROOT / 'results/topic4_sef_hfo/fig5_manual_core_release_v1/m_runaway_return_v1/runs/weak_fast.npz'


class Stop(Exception): pass


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'plan.json'
    if path.exists(): return read(path)
    p = read(ROOT / 'results/topic4_sef_hfo/m_parameter_modes_fig5_20260913/protocol.json')
    plan = dict(source_hashes=p['source_hashes'],identity=p['identity'],parent=str(PARENT),
        parent_time_s=75.5,parent_sha256=sha(PARENT),source_native_control=str(SOURCE),
        interventions=[dict(name='eta_2',eta_m=2.,duration_s=5.),
                       dict(name='eta_0p2',eta_m=.2,duration_s=5.)],
        unchanged_control=dict(eta_m=.02,duration_s=5.,reuse_existing=True),
        tau_M_s=2.,tau_Z_s=5.,I_th=base.THRESHOLD,
        question='Can a stronger current from the existing per-neuron M state drive the actual high state back under native Z, or does the sampled range fail?',
        intervention_scope='One parameter step at75.5s, eta_M only. Per-neuron M and Z, fast state and OU/RNG history unchanged. No Z or M reset.',
        not_a_constant_parameter_cycle=True,not_part_of_forty_job_grid=True,
        statistical_unit='one fixed high-state/noise realization; two explicitly changed parameter continuations',
        qa='Before interventions, eta_M=.02 continuation must reproduce existing native trace for200ms.',
        dispatch_deadline=read(WINDOW/'window.json')['deadline'],max_workers=1)
    write(path,plan);return plan


def worker(job):
    import fcntl
    plan=read(OUT/'plan.json');folder=OUT/'runs'/job['name'];folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (folder/'result.json').exists():return read(folder/'result.json')
    for path,digest in plan['source_hashes'].items():assert sha(path)==digest,path
    assert sha(PARENT)==plan['parent_sha256']
    started=time.time();write(folder/'progress.json',dict(status='BUILDING',pid=os.getpid(),job=job))
    s,tr,frozen,identity=base.setup(9108401);assert identity==plan['identity']
    cfg=base.MZSlowVarsConfig(use_z=True,use_m=True,tau_z=5000.,I_th_EI=base.THRESHOLD,
                            tau_adp=2000.,eta_m=job['eta_m'])
    slow=base.ReleaseZ(40000,s.params.V_th,cfg,NE=32000);slow.restore_ms=None
    cp=folder/'checkpoint.pkl';data=dict(time_ms=[],spikes=[],raster=[],slow_time_ms=[],Z=[],M=[],currents=[],lfp=[])
    if cp.exists():
        saved=load_pickle(cp);assert saved['job']==job
        state=saved['engine'];data=saved['data'];initial=saved['initial']
    else:
        source=load_pickle(PARENT);state=source['engine'];assert source['identity']==identity
        initial=dict(time_s=75.5,Z_sha256=hashlib.sha256(state['slow']['z'].tobytes()).hexdigest(),
            M_sha256=hashlib.sha256(state['slow']['m'].tobytes()).hexdigest(),
            mean_Z=float(state['slow']['z'][:32000].mean()),mean_M=float(state['slow']['m'][:32000].mean()),
            old_eta_M=.02,new_eta_M=job['eta_m'],Z_reset=False,M_reset=False,OU_RNG_preserved=True)
    first=int(state['step']);last=755000+round(job['duration_s']*10000)
    g=np.load(ROOT/'results/topic4_sef_hfo/m_parameter_modes_fig5_20260913/geometry.npz')
    samples=g['sample_ids'];recorder=base.LFPRecorder(s.params,s.net['pos'],s.net['labels'],sites=s.contact_xy)
    z0=state['slow']['z'].copy();m0=state['slow']['m'].copy()
    apply0=slow.apply_currents
    def apply(ie,ii,labels=None,rec=None):
        if slow._step_index==first:
            assert np.array_equal(slow.z,z0) and np.array_equal(slow.m,m0)
        result=apply0(ie,ii,labels,rec);k=slow._step_index
        assert cfg.eta_m==job['eta_m'] and slow.restore_ms is None
        if k==first:assert np.array_equal(result,ie-slow.z*ii-cfg.eta_m*slow.m)
        if k%50==0:
            data['slow_time_ms'].append(k*.1)
            data['Z'].append(float(slow.z[:32000].mean()));data['M'].append(float(slow.m[:32000].mean()))
            data['currents'].append([float(ie[:32000].mean()),float(ii[:32000].mean()),float(np.mean(slow.z[:32000]*ii[:32000]))])
            data['lfp'].append(recorder.sample(ie,ii))
        return result
    slow.apply_currents=apply
    def observe(tm,sp):
        data['time_ms'].append(tm)
        data['spikes'].append([int(sp[:32000].sum()),int(sp[32000:].sum())])
        data['raster'].append(sp[samples])
    def checkpoint(k,engine):
        arrays={key:np.asarray(value) for key,value in data.items()}
        save_pickle(cp,dict(job=job,engine=engine,data=data,initial=initial))
        np.savez_compressed(folder/'observations.tmp.npz',**arrays,sample_ids=samples,eta_m=cfg.eta_m)
        (folder/'observations.tmp.npz').replace(folder/'observations.npz')
        write(folder/'progress.json',dict(status='RUNNING',pid=os.getpid(),job=job,time_s=k*.0001,
            elapsed_s=time.time()-started,mean_Z=float(slow.z[:32000].mean()),
            mean_M_current=float(cfg.eta_m*slow.m[:32000].mean())))
        if k>=last:raise Stop()
    s.net['rng']=np.random.default_rng(9108401)
    drive=base.make_external_drive(s,tr['spatial_ou'],9108401)
    s.params.T=(last-first+1)*.1
    try:
        base.simulate_kick(s.params,s.net,KICK_BOOST=0.,slow=slow,V_th_per_neuron=s.vtheta,
            external_e_rate_drive=drive,early_stop_runaway=False,spike_observer=observe,
            record_dense_spikes=False,fast_scatter=True,resume_state=state,time_offset_ms=first*.1,
            checkpoint_steps=set(range((first//10000+1)*10000,last+1,10000))|{last},
            checkpoint_sink=checkpoint,verbose=False)
    except Stop:pass
    a=np.asarray(data['spikes']);rates=a.reshape(-1,100,2).sum(1)/[32000,8000]/.01
    recovery=None
    for hi in range(200,len(rates)+1):
        rr=rates[hi-200:hi,0].reshape(2,100)
        if np.all(rr.mean(1)<50) and np.all((rr<5).mean(1)>=.2):recovery=75.5+hi*.01;break
    r=dict(status='COMPLETE',job=job,initial_state=initial,end_s=last*.0001,
        observed_recovery_confirmation_s=recovery,late_E_mean_Hz=float(rates[-min(100,len(rates)):,0].mean()),
        last_Z=float(slow.z[:32000].mean()),last_M_current=float(cfg.eta_m*slow.m[:32000].mean()),
        external_parameter_step=True,native_constant_parameter_termination=False,
        current_application_checked=True,state_and_noise_not_reset=True)
    write(folder/'result.json',r);write(folder/'progress.json',r);return r


def supervise():
    import psutil,fcntl
    plan=prepare();lock=(OUT/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    deadline=datetime.fromisoformat(plan['dispatch_deadline']).timestamp()
    qa=worker(dict(name='qa_unchanged_200ms',eta_m=.02,duration_s=.2))
    with np.load(OUT/'runs/qa_unchanged_200ms/observations.npz') as a,np.load(SOURCE) as ref:
        lo,hi=755000,757000;geo=np.load(ROOT/'results/topic4_sef_hfo/m_parameter_modes_fig5_20260913/geometry.npz')
        assert np.array_equal(a['spikes'][:,0],np.rint(ref['rate_e_hz'][lo:hi]*32000*.0001).astype(int))
        assert np.array_equal(a['spikes'][:,1],np.rint(ref['rate_i_hz'][lo:hi]*8000*.0001).astype(int))
        assert np.array_equal(a['raster'],ref['sample_spikes'][lo:hi,geo['sample_source_indices']])
        assert np.array_equal(a['Z'],ref['z_stats'][lo//50:hi//50,0])
        assert np.array_equal(a['M'],ref['m_stats'][lo//50:hi//50,0])
    write(OUT/'qa.json',dict(status='PASS',native_high_state_resume_s=.2,counts_raster_Z_M_bitwise=True,
        comparison=str(SOURCE),parameters_unchanged_for_QA=True))
    for job in plan['interventions']:
        while time.time()<deadline:
            if psutil.virtual_memory().available/2**30>=68 and psutil.cpu_percent(interval=1)<85:break
            time.sleep(20)
        if time.time()>=deadline:break
        write(OUT/'status.json',dict(status='RUNNING',pid=os.getpid(),job=job))
        worker(job)
    rows=[read(p) for p in sorted((OUT/'runs').glob('*/result.json')) if not p.parent.name.startswith('qa_')]
    write(OUT/'status.json',dict(status='COMPLETE_PENDING_ANALYSIS' if len(rows)==2 else 'WINDOW_ENDED',
        completed=len(rows),total=2,rows=rows))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','supervise']);args=parser.parse_args()
    if args.mode=='prepare':prepare()
    else:
        try:supervise()
        except Exception as exc:
            OUT.mkdir(parents=True,exist_ok=True)
            write(OUT/'status.json',dict(status='FAILED',error=repr(exc)));raise
