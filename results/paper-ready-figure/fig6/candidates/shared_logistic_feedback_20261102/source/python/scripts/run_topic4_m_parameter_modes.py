#!/usr/bin/env python3
"""Full-observation M response assay, with native return and explicit Z rescue."""
import argparse
import hashlib
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
for _key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[_key]='1'
os.environ['TOPIC4_MANUAL_ARM']='manual_hard'
import numpy as np
import run_topic4_weak_fast_recurrence as old
from run_topic4_reset_state_diagnosis import read, write, save_pickle, load_pickle, sha

ROOT = old.ROOT
OUT = ROOT/'results/topic4_sef_hfo/m_parameter_modes_fig5_20260913'
SEEDS = [9108401, 9108402]
ETA = [.005, .01, .02, .04]
TAU = [1., 2., 4., 8., 20.]


class Stop(Exception):
    pass


def tracker_step(tr, rate, sec, rescue=True):
    rr=tr['rates']; rr.append(float(rate))
    if len(rr)>200: del rr[0]
    tr['high_bins']=tr['high_bins']+1 if rate>=200 else 0
    if tr['high_bins']>=20 and tr['phase'] in ['PRE_ENTRY','RECOVERED']:
        tr['entries'].append(dict(onset_s=sec-.2, confirmation_s=sec))
        tr['phase']='HIGH'; tr['last_entry_s']=sec
    recovered=False
    if tr['phase']=='HIGH' and len(rr)==200 and sec>=tr['last_entry_s']+2:
        x=np.asarray(rr).reshape(2,100)
        recovered=bool(np.all(x.mean(1)<50) and np.all((x<5).mean(1)>=.2))
    if recovered:
        mode='EXTERNAL_Z' if tr['restore_s'] is not None and sec>=tr['restore_s'] else 'NATIVE'
        tr['recoveries'].append(dict(start_s=sec-2, confirmation_s=sec, mechanism=mode))
        tr['phase']='RECOVERED'
    # Observe native dynamics for at least 60 s after first entry, covering
    # three times the largest tau_M. Z rescue is conditional and happens once.
    if (rescue and tr['entries'] and not tr['recoveries'] and tr['restore_s'] is None
            and sec>=tr['entries'][0]['confirmation_s']+60):
        tr['restore_s']=round(sec+.01,2); tr['release_s']=round(sec+1.01,2)
    if len(tr['entries'])>=2:
        tr['stop_s']=min(tr['stop_s'],tr['entries'][1]['confirmation_s']+10)
    elif tr['recoveries']:
        # A complete 180-s recurrence window after the first established return.
        origin=max(tr['recoveries'][0]['confirmation_s'],tr['release_s'] or 0.)
        tr['stop_s']=min(430.,origin+180)
    elif tr['entries']:
        tr['stop_s']=430.


def fresh_tracker():
    return dict(phase='PRE_ENTRY',rates=[],high_bins=0,entries=[],recoveries=[],
        last_entry_s=None,restore_s=None,release_s=None,stop_s=180.,wall_s=0.)


def prepare():
    if (OUT/'protocol.json').exists(): return read(OUT/'protocol.json')
    jobs=[dict(name=f'e{e}_t{t}_s{s}',eta_m=eta,tau_M_s=tau,seed=s,
               eta_index=e,tau_index=t,tau_z_ms=5000.,threshold=old.THRESHOLD,horizon_s=430.)
          for e,eta in enumerate(ETA) for t,tau in enumerate(TAU) for s in SEEDS]
    source=[Path(__file__),Path(old.__file__),ROOT/'scripts/run_topic4_reset_state_diagnosis.py',
        ROOT/'scripts/topic4_historical_manual_z_common.py',ROOT/'scripts/validate_topic4_fixed_rate_base.py',
        ROOT/'src/topic4_raster_protocol_engine.py',ROOT/'src/snn_engine/mz_slow_vars.py',
        ROOT/'config/topic4_rate_model_dynamics_validation_v1.json',
        ROOT/'results/topic4_sef_hfo/data_driven_core_field/config/stage_config.json']
    p=dict(status='DEFINED_BEFORE_RUNS',jobs=jobs,total=40,eta_M=ETA,tau_M_s=TAU,seeds=SEEDS,
        tau_Z_s=5.,I_th=old.THRESHOLD,topology_seed=6101,identity=read(old.PREVIOUS/'substrate.json')['identity'],
        source_hashes={str(x):sha(x) for x in source},max_workers=32,combined_scan_workers=40,
        minimum_available_memory_GiB=60,worker_memory_budget_GiB=4,
        first_entry_horizon_s=180,native_return_observation_s=60,recurrence_observation_s=180,
        maximum_trajectory_s=430,post_second_entry_s=10,
        intervention='If no established return in 60 s after first confirmed entry, refill E Z to 1 over 1 s once, then release native Z. M never cleared. All fast and random states continue.',
        M='E-only spike increment 1; current eta_M*M; tau_M decay. eta_M*tau_M_seconds is steady adaptation current per Hz.',
        endpoint='All E 10-ms rate >=200Hz continuously for 200ms. Recovery: two 1-s means<50Hz and each quiet(<5Hz) fraction>=.2. High-rate entry is not proof of sustained oscillation.',
        display='All completed conditions get a full Fig5 layout; observed mode gallery chosen by distance to original M parameters, then fixed seed order, never patient match.',
        F='This batch M-strength by M-timescale first-entry surface, horizon180s, paired noise n=2; old Z scan is only applicable to original M setting.',
        E2='Reuse canonical Fig3C E1146/SZ3. Model first-onset readout and native field are separately measured; no substitute seizure or assumption of agreement.',
        stop='Forty prescribed full trajectories, automatic figures and comparison; no adaptive grid expansion or model freeze.')
    write(OUT/'protocol.json',p)
    for j in jobs:write(OUT/'jobs'/(j['name']+'.json'),j)
    return p


def worker(job):
    import fcntl
    pdef=read(OUT/'protocol.json')
    for path,digest in pdef['source_hashes'].items():assert sha(path)==digest,path
    folder=OUT/'runs'/job['name'];folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (folder/'result.json').exists():return read(folder/'result.json')
    started=time.time();write(folder/'progress.json',dict(status='BUILDING',pid=os.getpid(),job=job))
    s,tr,frozen,identity=old.setup(job['seed']);assert identity==pdef['identity']
    ne,ni=s.n_e,s.n_i;p=s.params
    assert (ne,ni,p.dt)==(32000,8000,.1)
    assert (s.vtheta[:ne]<18).sum()==781 and not (s.vtheta[:ne]>18).any()
    cp=folder/'checkpoint.pkl';state=None;tracker=fresh_tracker();restore_from=None
    if cp.exists():
        saved=load_pickle(cp);assert saved['job']==job and saved['identity']==identity
        state=saved['engine'];tracker=saved['tracker'];restore_from=saved['restore_from'];del saved
    first=0 if state is None else int(state['step']);block_start=first;prior_wall=tracker['wall_s']
    cfg=old.MZSlowVarsConfig(use_z=True,use_m=True,tau_z=job['tau_z_ms'],I_th_EI=job['threshold'],
                           tau_adp=job['tau_M_s']*1000,eta_m=job['eta_m'])
    slow=old.ReleaseZ(ne+ni,p.V_th,cfg,NE=ne)
    slow.restore_ms=None if tracker['restore_s'] is None else round(tracker['restore_s']*1000)
    slow.restore_from=restore_from
    centers=np.asarray(frozen['candidate']['node_field']['centers_mm'])
    def group(pos):
        d=np.linalg.norm(pos[:,None]-centers[None],axis=2);g=np.full(len(pos),2)
        g[d[:,0]<1.75]=0;g[(d[:,1]<1.75)&(d[:,1]<d[:,0])]=1;return g
    ge,gi=group(s.positions_e),group(s.positions_i)
    nr=np.r_[np.bincount(ge,minlength=3),np.bincount(gi,minlength=3)]
    ix=[np.flatnonzero(ge==g) for g in range(3)]
    cells=old.spatial_cell_index(s.positions_e,n_grid=20,sheet_l_mm=p.L);nc=np.bincount(cells,minlength=400)
    source=np.load(old.PREVIOUS/'reference_samples/runs/z_current_e_seed9108401.npz')
    sel=np.r_[np.arange(0,60,3),np.arange(60,120,3),np.arange(120,240,6),np.arange(240,300,3)]
    samples=source['sample_ids'][sel];recorder=old.LFPRecorder(p,s.net['pos'],s.net['labels'],sites=s.contact_xy)
    geometry=OUT/'geometry.npz'
    if not geometry.exists():
        tmp=folder/'geometry.tmp.npz'
        np.savez_compressed(tmp,region_counts=nr,cell_e_counts=nc,centers_mm=centers,
            positions_e=s.positions_e,contact_xy=s.contact_xy,contact_names=s.contact_names,sample_ids=samples,
            sample_source_indices=sel,core_radius_mm=frozen['candidate']['node_field']['core_radius_mm'])
        tmp.replace(geometry)
    data={};count=np.zeros(2,np.uint32);regions=np.zeros(6,np.uint32);field=np.zeros(400,np.uint32);recent=0
    def clear():
        data.clear();data.update(time_ms=[],spikes_1ms=[],regions_1ms=[],field_1ms=[],raster=[],
            slow_time_ms=[],Z=[],M=[],currents=[],lfp_time_ms=[],lfp_raw=[],inputs=[])
    clear();original=slow.apply_currents
    def apply(ie,ii,labels=None,rec=None):
        value=original(ie,ii,labels,rec);k=slow._step_index;tm=k*.1
        if k%5==0:data['lfp_time_ms'].append(tm);data['lfp_raw'].append(recorder.sample(ie,ii))
        if k%50==0:
            z,m=slow.z[:ne],slow.m[:ne]
            data['slow_time_ms'].append(tm)
            data['Z'].append([z.mean(),z.std(),*np.quantile(z,[.1,.5,.9]),*[z[v].mean() for v in ix],np.mean(ii[:ne]>=cfg.I_th_EI)])
            data['M'].append([m.mean(),*[m[v].mean() for v in ix]])
            data['currents'].append([ie[:ne].mean(),ii[:ne].mean(),np.mean(z*ii[:ne])])
        return value
    slow.apply_currents=apply
    def inputs(tm,nu,xi):
        if round(tm/.1)%1000==0:data['inputs'].append([tm,xi,nu[:ne].mean(),nu[ne:].mean()])
    def observe(tm,spk):
        nonlocal recent
        k=round(tm/.1);ec=int(spk[:ne].sum());ic=int(spk[ne:].sum())
        count[0]+=ec;count[1]+=ic;recent+=ec
        regions[:3]+=np.bincount(ge[spk[:ne]],minlength=3).astype(np.uint32)
        regions[3:]+=np.bincount(gi[spk[ne:]],minlength=3).astype(np.uint32)
        field[:]+=np.bincount(cells[spk[:ne]],minlength=400).astype(np.uint32)
        data['raster'].append(spk[samples])
        if (k+1)%10==0:
            data['time_ms'].append((k+1)*.1-.5);data['spikes_1ms'].append(count.copy());count.fill(0)
            data['regions_1ms'].append(regions.copy());regions.fill(0)
            data['field_1ms'].append(field.copy());field.fill(0)
        if (k+1)%100==0:
            sec=(k+1)*.0001;rate=recent/ne/.01;recent=0
            tracker_step(tracker,rate,sec,rescue=not job.get('qa'))
            if tracker['restore_s'] is not None:slow.restore_ms=round(tracker['restore_s']*1000)
        if (k+1)%10000==0:
            write(folder/'progress.json',dict(status='RUNNING',pid=os.getpid(),job=job,time_s=(k+1)*.0001,
                phase=tracker['phase'],entries=tracker['entries'],recoveries=tracker['recoveries'],
                restore_s=tracker['restore_s'],stop_s=min(job['horizon_s'],tracker['stop_s']),
                Z=float(slow.z[:ne].mean()),adaptation_current=float(cfg.eta_m*slow.m[:ne].mean()),
                wall_s=prior_wall+time.time()-started))
    def checkpoint(k,engine):
        nonlocal block_start
        stop=k*.0001>=min(job['horizon_s'],tracker['stop_s'])-1e-9
        flush=k-block_start>=100000 or stop or (job.get('qa') and k-block_start>=5000)
        if flush:
            a={key:np.asarray(val) for key,val in data.items()}
            assert a['spikes_1ms'][:,0].sum()==a['regions_1ms'][:,:3].sum()==a['field_1ms'].sum()
            assert a['spikes_1ms'][:,1].sum()==a['regions_1ms'][:,3:].sum()
            for key in ['spikes_1ms','regions_1ms','field_1ms']:a[key]=a[key].astype(np.uint16)
            a.update(start_step=block_start,end_step=k)
            chunks=folder/'chunks';chunks.mkdir(exist_ok=True)
            dest=chunks/f'{block_start:010d}_{k:010d}.npz';tmp=dest.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,**a);tmp.replace(dest)
            tracker['wall_s']=prior_wall+time.time()-started
            save_pickle(cp,dict(job=job,identity=identity,engine=engine,tracker=tracker,restore_from=slow.restore_from))
            block_start=k;clear()
        if stop or (job.get('qa_pause') and first==0):raise Stop()
    s.net['rng']=np.random.default_rng(job['seed']);drive=old.make_external_drive(s,tr['spatial_ou'],job['seed'])
    last=round(job['horizon_s']*10000);p.T=(last-first+1)*.1
    try:
        old.simulate_kick(p,s.net,KICK_BOOST=0.,slow=slow,V_th_per_neuron=s.vtheta,external_e_rate_drive=drive,
            early_stop_runaway=False,spike_observer=observe,input_observer=inputs,record_dense_spikes=False,
            fast_scatter=True,resume_state=state,time_offset_ms=first*.1,
            checkpoint_steps=set(range((first//5000+1)*5000,last+1,5000))|{last},checkpoint_sink=checkpoint,verbose=False)
    except Stop:pass
    if job.get('qa_pause') and first==0:return
    assert np.all(slow.m[ne:]==0) and np.all(slow.z[ne:]==1)
    r=dict(status='COMPLETE',job=job,end_s=block_start*.0001,identity=identity,tracker=tracker,
        M_enabled=True,M_reset=False,eta_m=cfg.eta_m,tau_M_s=cfg.tau_adp/1000,
        Vth_E_counts=dict(lowered=781,equal=31219,raised=0),spatial_count_conservation=True)
    write(folder/'result.json',r);write(folder/'progress.json',r);return r


def qa():
    # Previous large-network parity remains a separate prerequisite, not rerun.
    assert read(ROOT/'results/topic4_sef_hfo/m_on_z_kinetics_20260912/qa.json')['status']=='PASS'
    j=dict(name='qa_full_readout',eta_m=.02,tau_M_s=2.,seed=SEEDS[0],tau_z_ms=5000.,
           threshold=old.THRESHOLD,horizon_s=1.,qa=True,qa_pause=True)
    worker(j);worker(j)
    paths=sorted((OUT/'runs'/j['name']/'chunks').glob('*.npz'))
    with np.load(old.OUT/'runs/weak_fast_z_refill_recurrence.npz') as ref, np.load(OUT/'geometry.npz') as g:
        for path in paths:
            with np.load(path) as a:
                lo,hi=int(a['start_step']),int(a['end_step'])
                assert np.array_equal(a['raster'],ref['sample_spikes'][lo:hi,g['sample_source_indices']])
                assert np.array_equal(a['field_1ms'],ref['field_e_count_1ms'][lo//10:hi//10])
                assert np.array_equal(a['lfp_raw'],ref['lfp_raw'][lo//5:hi//5])
                assert np.array_equal(a['Z'],ref['z_stats'][lo//50:hi//50,:9])
                assert np.array_equal(a['M'],ref['m_stats'][lo//50:hi//50][:,[0,5,6,7]])
    # Feed controlled observations to the protocol; these are decision tests,
    # never presented as network activity or biological results.
    a=fresh_tracker()
    for i in range(20):tracker_step(a,220,(i+1)*.01)
    for i in range(20,6210):tracker_step(a,220,(i+1)*.01)
    assert a['restore_s'] is not None and len(a['entries'])==1
    for i in range(6210,6510):tracker_step(a,0,(i+1)*.01)
    assert a['recoveries'][0]['mechanism']=='EXTERNAL_Z'
    for i in range(6510,6530):tracker_step(a,220,(i+1)*.01)
    assert len(a['entries'])==2
    b=fresh_tracker()
    for i in range(20):tracker_step(b,220,(i+1)*.01)
    for i in range(20,320):tracker_step(b,0,(i+1)*.01)
    assert b['recoveries'][0]['mechanism']=='NATIVE' and b['restore_s'] is None
    for eta in ETA:
        for tau in TAU:
            cfg=old.MZSlowVarsConfig(use_z=True,use_m=True,tau_z=5000.,I_th_EI=old.THRESHOLD,
                                   tau_adp=tau*1000,eta_m=eta)
            s=old.ReleaseZ(4,18.,cfg,NE=3);s.m[:3]=[2,3,4]
            expected=s.m.copy();s.apply_currents(np.ones(4)*200,np.ones(4)*100)
            spike=np.array([1,0,1,0],bool);s.step(spike,None,.1)
            expected[:3]-=.1/(tau*1000)*expected[:3];expected[:3]+=spike[:3]
            assert np.array_equal(s.m,expected)
            assert np.array_equal(s.apply_currents(np.ones(4)*200,np.ones(4)*100),200-s.z*100-eta*expected)
    write(OUT/'qa.json',dict(status='PASS',full_observation_prefix_s=1.,checkpoint_resume_s=.5,
        raster_field_readout_Z_M_bitwise=True,native_vs_external_recovery_logic=True,
        M_parameter_application_checks=20,M_never_reset=True))


def controller():
    import fcntl,psutil,shutil
    OUT.mkdir(parents=True,exist_ok=True);lock=(OUT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=read(OUT/'protocol.json');assert read(OUT/'qa.json')['status']=='PASS'
    # Start around the known M working point and then cover short/long memory.
    rank=[(2,1),(1,2),(0,3),(3,0),(1,1),(2,2),(0,1),(3,1)]
    rank += [(e,t) for e in range(4) for t in range(5) if (e,t) not in rank]
    jobs=sorted(p['jobs'],key=lambda j:(rank.index((j['eta_index'],j['tau_index'])),SEEDS.index(j['seed'])))
    pending=[j for j in jobs if not (OUT/'runs'/j['name']/'result.json').exists()];running={};failed=[];done=set();last=0
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
        TOPIC4_MANUAL_ARM='manual_hard',LD_LIBRARY_PATH='/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib')
    while pending or running:
        for name,(child,log) in list(running.items()):
            rc=child.poll()
            if rc is None:continue
            log.close();del running[name]
            if rc or not (OUT/'runs'/name/'result.json').exists():failed.append(dict(name=name,returncode=rc))
        active_old=sum(1 for proc in psutil.process_iter(['cmdline']) if
            'run_topic4_m_on_z_kinetics.py' in ' '.join(proc.info['cmdline'] or []) and
            'worker' in (proc.info['cmdline'] or []))
        cap=max(0,min(32,40-active_old))
        mem=psutil.virtual_memory().available/1024**3;reserve=0
        for child,_ in running.values():
            try:reserve+=max(0,4-psutil.Process(child.pid).memory_info().rss/1024**3)
            except psutil.NoSuchProcess:pass
        while pending and not failed and len(running)<cap and mem-reserve>64 and shutil.disk_usage(OUT).free/1024**3>60:
            changed=[source for source,digest in p['source_hashes'].items() if sha(source)!=digest]
            if changed:
                failed.append(dict(error='Source changed before dispatch',sources=changed));break
            j=pending.pop(0);log=OUT/'logs'/(j['name']+'.log');log.parent.mkdir(exist_ok=True);f=log.open('a')
            child=subprocess.Popen([sys.executable,'-u',__file__,'worker','--job',str(OUT/'jobs'/(j['name']+'.json'))],
                cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
            running[j['name']]=(child,f);reserve+=4
        completed={j['name'] for j in jobs if (OUT/'runs'/j['name']/'result.json').exists()}
        write(OUT/'status.json',dict(status='DRAINING_AFTER_FAILURE' if failed else 'RUNNING',pid=os.getpid(),
            completed=len(completed),total=40,running={n:c.pid for n,(c,_) in running.items()},
            pending=len(pending),failed=failed,current_worker_cap=cap,available_memory_GiB=mem))
        if completed!=done and time.time()-last>60:
            with (OUT/'analysis.log').open('a') as f:
                rc=subprocess.call([sys.executable,str(ROOT/'scripts/plot_topic4_m_parameter_modes.py'),'--update'],
                                   cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
            if rc:failed.append(dict(error='Figure/analysis failed; see analysis.log'))
            done=completed;last=time.time()
        if failed and not running:break
        time.sleep(20)
    with (OUT/'analysis.log').open('a') as f:
        rc=subprocess.call([sys.executable,str(ROOT/'scripts/plot_topic4_m_parameter_modes.py'),'--update','--final'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
    write(OUT/'status.json',dict(status='FAILED' if failed or rc else 'COMPLETE_PENDING_FIGURE_REVIEW',
        completed=len(done),total=40,running={},pending=len(pending),failed=failed,analysis_returncode=rc))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','qa','worker','controller']);p.add_argument('--job');a=p.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='qa':qa()
    elif a.mode=='worker':worker(read(a.job))
    else:controller()
