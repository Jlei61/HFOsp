#!/usr/bin/env python3
"""Log-spaced M first-passage screen; native physics, reduced observation overhead."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import argparse,copy,fcntl,json,time,subprocess,sys,shutil
from pathlib import Path
import numpy as np
import psutil
import run_topic4_m_parameter_modes as base
from src.topic4_cuda_ordered_scatter import wrap_simulator
ROOT=base.ROOT;OUT=ROOT/'results/topic4_sef_hfo/fig5_log_m_kinetics_20260914'
ETAS=[.0001,.0005,.001,.01,.1,1.];TAUS=[1.,10.,100.,1000.];SEEDS=base.SEEDS

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():return base.read(OUT/'protocol.json')
    original=base.read(base.OUT/'protocol.json')
    for path,h in original['source_hashes'].items():assert base.sha(path)==h,path
    refs=[];jobs=[]
    for i,eta in enumerate(ETAS):
        for j,tau in enumerate(TAUS):
            for k,seed in enumerate(SEEDS):
                name=f'eta{eta:g}_tau{tau:g}_s{seed}'
                job=dict(name=name,eta_m=eta,tau_M_s=tau,seed=seed,eta_index=i,tau_index=j,
                    tau_z_ms=5000.,threshold=base.old.THRESHOLD,horizon_s=300.,device=(i+j+k)%2)
                source=None
                if tau==1 and eta==.0005:source=ROOT/'results/topic4_sef_hfo/fig5_preentry_event_audit_20260914/runs'/f'eta0.0005_s{seed}'
                if tau==1 and eta==.001:source=ROOT/'results/topic4_sef_hfo/weaker_M_onset_paired_20260914/runs'/f'eta0.001_s{seed}'
                if tau==1 and eta==.01:source=base.OUT/'runs'/f'e1_t0_s{seed}'
                if source and (source/'result.json').exists():
                    r=base.read(source/'result.json');assert r['identity']==original['identity']
                    assert r['job']['eta_m']==eta and r['job']['tau_M_s']==tau
                    e=r['tracker']['entries'][0];assert e['confirmation_s']<300
                    refs.append(dict(job=job,source=str(source),first_entry=e,result_sha256=base.sha(source/'result.json')))
                else:jobs.append(job)
    p=dict(status='DEFINED_BEFORE_NEW_RUNS',question='How do M strength and timescale alter finite-time first high-state entry?',
        approval='User requested log-scale tau M expansion including10,100,1000 seconds and wider eta M; 2026-09-14.',
        eta_M=ETAS,tau_M_s=TAUS,seeds=SEEDS,total_cells=len(ETAS)*len(TAUS),total_realizations=48,
        new_jobs=len(jobs),reused=len(refs),jobs=jobs,references=refs,horizon_s=300.,topology_seed=6101,
        identity=original['identity'],source_hashes=original['source_hashes'],producer_sha256=base.sha(__file__),
        max_workers=16,minimum_available_memory_GiB=80,worker_budget_GiB=4,
        endpoint='All-E10ms rate>=200Hz for20 consecutive bins; report first confirmation, preserve onset separately.',
        intervention='None. New trajectories begin from native initial state with their own paired noise seed. Stop at first confirmation or300s.',
        sampling='10ms E/I and6regional counts;100ms actual mean Z/M/applied inhibition;20s checkpoints. No virtual-contact readout in timing-only runs.',
        physics='Same40k manual two-core substrate, dt0.1ms, native OU, Z tau5s, E-only Z/M; only eta_M and tau_M change.',
        interpretation='Operational first-passage diagram at300s, not an asymptotic attractor or proven bifurcation diagram. tauM1000s is incompletely relaxed by300s.',
        scale='Both parameter axes logarithmic. Eta*tau controls steady feedback per Hz; short-time M is approximately accumulated spikes for t<<tau.',
        statistics='Two paired noise realizations on one fixed topology. Restricted mean at300s plus n/2 entries, not an uncensored mean when events absent.',
        stop='42 new runs plus6 exact observed endpoints reused. No automatic parameter expansion or model freeze.',
        qa='Before dispatch, compare population, Z/M and inputs against both existing half-M prefixes, including checkpoint continuation.')
    base.write(OUT/'protocol.json',p)
    for job in jobs:base.write(OUT/'jobs'/(job['name']+'.json'),job)
    for k,seed in enumerate(SEEDS):
        job=dict(name=f'qa_s{seed}',eta_m=.0005,tau_M_s=1.,seed=seed,tau_z_ms=5000.,threshold=base.old.THRESHOLD,horizon_s=1.,device=k,qa=True)
        base.write(OUT/'jobs'/(job['name']+'.json'),job)
    return p

class Stop(Exception):pass

def worker(name):
    pdef=prepare();assert base.sha(__file__)==pdef['producer_sha256']
    for p,h in pdef['source_hashes'].items():assert base.sha(p)==h,p
    job=base.read(OUT/'jobs'/(name+'.json'));folder=OUT/'runs'/name;folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (folder/'result.json').exists():return
    started=time.time();base.write(folder/'progress.json',dict(status='BUILDING',pid=os.getpid(),job=job))
    s,tr,frozen,identity=base.old.setup(job['seed']);assert identity==pdef['identity']
    p=s.params;ne,ni=s.n_e,s.n_i;assert (ne,ni,p.dt)==(32000,8000,.1)
    assert (s.vtheta[:ne]<18).sum()==781 and not (s.vtheta[:ne]>18).any()
    centers=np.array(frozen['candidate']['node_field']['centers_mm'])
    def groups(pos):
        d=np.linalg.norm(pos[:,None]-centers[None],axis=2);g=np.full(len(pos),2)
        g[d[:,0]<1.75]=0;g[(d[:,1]<1.75)&(d[:,1]<d[:,0])]=1;return g
    ge,gi=groups(s.positions_e),groups(s.positions_i);nr=np.r_[np.bincount(ge,minlength=3),np.bincount(gi,minlength=3)]
    cfg=base.old.MZSlowVarsConfig(use_z=True,use_m=True,tau_z=5000.,I_th_EI=job['threshold'],tau_adp=job['tau_M_s']*1000,eta_m=job['eta_m'])
    slow=base.old.ReleaseZ(ne+ni,p.V_th,cfg,NE=ne)
    state=None;tracker=dict(high_bins=0,first_entry=None,wall_s=0.);first=0
    cp=folder/'checkpoint.pkl'
    if cp.exists():
        saved=base.load_pickle(cp);assert saved['job']==job and saved['identity']==identity
        state=saved['engine'];tracker=saved['tracker'];first=int(state['step'])
    prior_wall=tracker['wall_s'];block_start=first;pop=np.zeros(2,np.uint32);regs=np.zeros(6,np.uint32)
    data={k:[] for k in ['time_s','spikes_10ms','regions_10ms','slow_time_ms','Z','M','H','inputs']}
    original=slow.apply_currents
    def apply(ie,ii,labels=None,rec=None):
        value=original(ie,ii,labels,rec)
        if slow._step_index%1000==0:
            data['slow_time_ms'].append(slow._step_index*.1)
            data['Z'].append(slow.z[:ne].mean());data['M'].append(slow.m[:ne].mean());data['H'].append(np.mean(slow.z[:ne]*ii[:ne]))
        return value
    slow.apply_currents=apply
    def inputs(tm,nu,xi):
        if round(tm/.1)%1000==0:data['inputs'].append([tm,xi,nu[:ne].mean(),nu[ne:].mean()])
    def observe(tm,spk):
        k=round(tm/.1);pop[0]+=np.count_nonzero(spk[:ne]);pop[1]+=np.count_nonzero(spk[ne:])
        regs[:3]+=np.bincount(ge[spk[:ne]],minlength=3).astype(np.uint32)
        regs[3:]+=np.bincount(gi[spk[ne:]],minlength=3).astype(np.uint32)
        if (k+1)%100==0:
            sec=(k+1)*.0001;rate=pop[0]/320
            data['time_s'].append(sec);data['spikes_10ms'].append(pop.copy());data['regions_10ms'].append(regs.copy());pop.fill(0);regs.fill(0)
            tracker['high_bins']=tracker['high_bins']+1 if rate>=200 else 0
            if tracker['high_bins']>=20 and tracker['first_entry'] is None:tracker['first_entry']=dict(onset_s=sec-.2,confirmation_s=sec)
        if (k+1)%10000==0:
            base.write(folder/'progress.json',dict(status='RUNNING',pid=os.getpid(),job=job,time_s=(k+1)*.0001,
                wall_s=prior_wall+time.time()-started,first_entry=tracker['first_entry'],Z=float(slow.z[:ne].mean()),M_feedback=float(job['eta_m']*slow.m[:ne].mean())))
        if (k+1)%100==0 and tracker['first_entry'] is not None:raise Stop()
    def flush(k,engine=None):
        nonlocal block_start
        if not data['time_s']:return
        records={k:np.asarray(v) for k,v in data.items()};records.update(start_step=block_start,end_step=k)
        assert np.array_equal(records['spikes_10ms'][:,0],records['regions_10ms'][:,:3].sum(1))
        chunk=folder/'chunks'/f'{block_start:010d}_{k:010d}.npz';chunk.parent.mkdir(exist_ok=True)
        tmp=chunk.with_suffix('.tmp.npz');np.savez_compressed(tmp,**records);tmp.replace(chunk)
        tracker['wall_s']=prior_wall+time.time()-started
        if engine is not None:base.save_pickle(cp,dict(job=job,identity=identity,engine=engine,tracker=tracker))
        block_start=k
        for v in data.values():v.clear()
    def checkpoint(k,engine):
        flush(k,engine)
        if job.get('qa') and k==5000 and first==0:raise Stop()
    s.net['rng']=np.random.default_rng(job['seed']);drive=base.old.make_external_drive(s,tr['spatial_ou'],job['seed'])
    last=round(job['horizon_s']*10000);p.T=(last-first+1)*.1
    points=sorted(set(range((first//200000+1)*200000,last+1,200000))|{last})
    if job.get('qa'):points=sorted(set(points)|({5000} if first==0 else set()))
    simulate=wrap_simulator(base.old.simulate_kick,device_index=job['device'])
    stopped=False
    try:
        simulate(p,s.net,KICK_BOOST=0.,slow=slow,V_th_per_neuron=s.vtheta,external_e_rate_drive=drive,
            early_stop_runaway=False,spike_observer=observe,input_observer=inputs,record_dense_spikes=False,
            fast_scatter=True,resume_state=state,time_offset_ms=first*.1,checkpoint_steps=set(points),checkpoint_sink=checkpoint,verbose=False)
    except Stop:stopped=True
    if job.get('qa') and first==0:return
    # Native observer is called after spike/M updates; first-entry stop has a whole10ms bin.
    if data['time_s']:flush(round(data['time_s'][-1]*10000))
    elapsed=block_start*.0001;entry=tracker['first_entry']
    assert entry or elapsed>=job['horizon_s']-1e-8
    result=dict(status='COMPLETE',job=job,identity=identity,first_entry=entry,elapsed_s=elapsed,
        event_observed=entry is not None,censored_at_s=None if entry else job['horizon_s'],wall_s=prior_wall+time.time()-started,
        no_reset=True,parameter_units='tau_M seconds; executor tau_adp milliseconds',source_hashes=pdef['source_hashes'])
    base.write(folder/'result.json',result);base.write(folder/'progress.json',result)

def qa_check():
    reports=[]
    for seed in SEEDS:
        folder=OUT/'runs'/f'qa_s{seed}';parts={}
        for path in sorted((folder/'chunks').glob('*.npz')):
            with np.load(path) as a:
                for k in a.files:
                    if k not in ['start_step','end_step']:parts.setdefault(k,[]).append(a[k])
        parts={k:np.concatenate(v) for k,v in parts.items()}
        source=ROOT/'results/topic4_sef_hfo/fig5_preentry_event_audit_20260914/runs'/f'eta0.0005_s{seed}'/'chunks'
        ref={}
        for path in sorted(source.glob('*.npz'))[:2]:
            with np.load(path) as a:
                for k in ['spikes_1ms','regions_1ms','slow_time_ms','Z','M','currents','inputs']:ref.setdefault(k,[]).append(a[k])
        ref={k:np.concatenate(v) for k,v in ref.items()}
        assert np.array_equal(parts['spikes_10ms'],ref['spikes_1ms'].reshape(-1,10,2).sum(1))
        assert np.array_equal(parts['regions_10ms'],ref['regions_1ms'].reshape(-1,10,6).sum(1))
        assert np.array_equal(parts['Z'],ref['Z'][::20,0]) and np.array_equal(parts['M'],ref['M'][::20,0])
        assert np.array_equal(parts['H'],ref['currents'][::20,2])
        assert np.array_equal(parts['inputs'],ref['inputs'])
        reports.append(dict(seed=seed,exact_population_regions_Z_M_H_inputs=True,checkpoint_resume=True))
    base.write(OUT/'qa.json',dict(status='PASS',reports=reports))

def analyze():
    cmd=[sys.executable,str(ROOT/'scripts/analyze_topic4_fig5_log_m_scan.py')]
    with (OUT/'analysis.log').open('a') as log:
        rc=subprocess.call(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if rc:raise RuntimeError(f'Automatic analysis failed: {rc}')

def supervise():
    p=prepare();lock=(OUT/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # QA uses the same worker with explicit stop+resume to verify the lean observer.
    if not (OUT/'qa.json').exists():
        for repeat in range(2):
            children=[]
            for seed in SEEDS:
                folder=OUT/'runs'/f'qa_s{seed}';folder.mkdir(parents=True,exist_ok=True)
                with (folder/'worker.log').open('a') as log:
                    children.append(subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'worker','--name',f'qa_s{seed}'],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT))
            for c in children:
                if c.wait():raise RuntimeError('Native observer QA worker failed')
        qa_check()
    pending=[j['name'] for j in p['jobs'] if not (OUT/'runs'/j['name']/'result.json').exists()]
    children={};failed=[];last=-1
    while pending or children:
        for name,child in list(children.items()):
            rc=child.poll()
            if rc is None:continue
            del children[name]
            if rc or not (OUT/'runs'/name/'result.json').exists():failed.append(name)
        mem=psutil.virtual_memory().available/2**30
        while pending and not failed and len(children)<p['max_workers'] and mem>80 and shutil.disk_usage(OUT).free/2**30>40:
            name=pending.pop(0);folder=OUT/'runs'/name;folder.mkdir(parents=True,exist_ok=True)
            with (folder/'worker.log').open('a') as log:
                children[name]=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'worker','--name',name],
                    stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,start_new_session=True)
            mem-=4
        completed=sum((OUT/'runs'/j['name']/'result.json').exists() for j in p['jobs'])
        base.write(OUT/'status.json',dict(status='DRAINING_FAILURE' if failed else 'RUNNING',pid=os.getpid(),updated_at=time.time(),
            completed_new=completed,total_new=p['new_jobs'],reused=p['reused'],running={n:c.pid for n,c in children.items()},pending=len(pending),failed=failed))
        if completed!=last:analyze();last=completed
        if failed and not children:raise RuntimeError(failed)
        time.sleep(10)
    analyze();base.write(OUT/'status.json',dict(status='COMPLETE_PENDING_HUMAN_REVIEW',completed_new=p['new_jobs'],total_new=p['new_jobs'],reused=p['reused'],running={},pending=0,failed=[]))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','worker','supervise']);ap.add_argument('--name');args=ap.parse_args()
    if args.mode=='prepare':prepare()
    elif args.mode=='worker':
        try:worker(args.name)
        except Exception as e:
            base.write(OUT/'runs'/args.name/'failure.json',dict(error=repr(e),time=time.time()));raise
    else:
        try:supervise()
        except Exception as e:base.write(OUT/'supervisor_failure.json',dict(error=repr(e),time=time.time()));raise
