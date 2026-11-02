#!/usr/bin/env python3
"""Bounded, no-intervention continuation of the manual two-core Fig5 SNN."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
import argparse,copy,fcntl,json,subprocess,sys,time,shutil
from pathlib import Path
import numpy as np
import psutil
import run_topic4_m_parameter_modes as base
from src.topic4_cuda_ordered_scatter import wrap_simulator

ROOT=base.ROOT
OUT=ROOT/'results/topic4_sef_hfo/autonomous_recovery_exploration_20260914'
START=1789401030.; DEADLINE=START+9*3600

class Stop(Exception): pass

class RecoverySlow(base.old.MZSlowVars):
    """Native equations with an explicit optional redistribution of GABA to E.

    mix: J_i=(1-gamma) I_i + gamma mean_E(I), conserving raw E input sum.
    add: J_i=I_i + gamma mean_E(I), increasing the raw input sum.
    Both the delivered current Z_i J_i and Z depletion use J_i. I cells are
    untouched. This is a current-based local/global motif, not LAS biophysics.
    """
    def __init__(self,*args,mode='native',gamma=0.,**kwargs):
        super().__init__(*args,**kwargs)
        assert mode in ['native','mix','add'] and 0<=gamma<=1
        self.mode=mode;self.gamma=float(gamma);self.raw_mean=0.
        self.delivered=None

    def _record_trace(self,*args): pass

    def apply_currents(self,ie,ii,labels=None,rec=None):
        self.raw_mean=float(ii[:self.NE].mean())
        if self.mode=='native' or self.gamma==0:
            self.delivered=ii
            return super().apply_currents(ie,ii,labels,rec)
        jj=ii.copy()
        if self.mode=='mix':jj[:self.NE]=(1-self.gamma)*ii[:self.NE]+self.gamma*self.raw_mean
        else:jj[:self.NE]+=self.gamma*self.raw_mean
        self.delivered=jj
        return super().apply_currents(ie,jj,labels,rec)

def fresh_tracker():
    return dict(phase='PRE_ENTRY',high_bins=0,entries=[],recoveries=[],recent_rates=[],
                last_entry_s=None,last_recovery_s=None,wall_s=0.,stop_reason=None)

def track(tr,rates,sec):
    # rates = [all E, core A E, core B E, surround E]. Neither the detector
    # nor the horizon changes any model state or input.
    tr['recent_rates'].append([float(v) for v in rates])
    if len(tr['recent_rates'])>200:del tr['recent_rates'][0]
    tr['high_bins']=tr['high_bins']+1 if rates[0]>=200 else 0
    if tr['high_bins']>=20 and tr['phase'] in ['PRE_ENTRY','RECOVERED']:
        tr['entries'].append(dict(onset_s=round(sec-.2,5),confirmation_s=round(sec,5)))
        tr['phase']='HIGH';tr['last_entry_s']=sec
    if tr['phase']=='HIGH' and sec>=tr['last_entry_s']+2 and len(tr['recent_rates'])==200:
        x=np.array(tr['recent_rates']).reshape(2,100,4)
        mean=x.mean(1);quiet=(x<5).mean(1)
        if np.all(mean[:,0]<50) and np.all(quiet[:,0]>=.2) and np.all(mean[:,1:3]<50) and np.all(quiet[:,1:3]>=.2):
            tr['recoveries'].append(dict(start_s=round(sec-2,5),confirmation_s=round(sec,5),
                mechanism='AUTONOMOUS',all_and_core_means_Hz=mean.tolist(),quiet_fraction=quiet.tolist()))
            tr['last_recovery_s']=sec;tr['phase']='RECOVERED'

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():return base.read(OUT/'protocol.json')
    previous=base.read(base.OUT/'protocol.json')
    jobs=[]
    def add(label,mode,eta,tau,gamma=0,tz=5,horizon=60):
        jobs.append(dict(name=label+'_s9108401',round=1,mode=mode,eta_m=eta,tau_M_s=tau,
            gamma=gamma,tau_Z_s=tz,threshold=base.old.THRESHOLD,seed=9108401,
            horizon_s=horizon,device=len(jobs)%2,checkpoint_s=10.))
    add('native_weak','native',.0005,1)
    add('native_longM','native',.001,100,horizon=120)
    add('native_midM','native',.003,100,horizon=120)
    add('native_fastZ','native',.01,100,tz=1,horizon=120)
    for eta in [.005,.02]:
        for gamma in [.25,.5,.75]:add(f'mix_g{gamma:g}_eta{eta:g}','mix',eta,2,gamma)
    for gamma in [.25,.5]:add(f'add_g{gamma:g}','add',.005,2,gamma)
    protocol=dict(status='DEFINED_BEFORE_EXPERIMENTS',created_at=time.time(),start_epoch=START,
        deadline_epoch=DEADLINE,window_hours=9,approval='User requested 8-10h autonomous exploration, native recovery and recurrence, Liou2020/global inhibition.',
        question='Can continuous noisy manual two-core SNN enter a high state, recover autonomously, and enter again?',
        identity=previous['identity'],source_hashes=previous['source_hashes'],producer_sha256=base.sha(__file__),
        initial_jobs=jobs,max_workers=12,max_combined_with_log_scan=24,min_available_memory_GiB=80,
        disk_reserve_GiB=40,max_total_realizations=48,
        stage_policy='12 initial single-noise diagnostic runs, then inspect mechanisms and select paired-noise confirmation / bounded revisions. No blind automatic expansion.',
        substrate='Historical manually placed, lowering-only hard two cores on fixed C fast carrier; 40k cells; dt0.1ms; original OU/Poisson; no stimulation.',
        native_M='Spike increment 1; decay tau_M; current eta_M*M. eta_M*tau_M controls steady current per Hz.',
        inhibition='mix conserves instantaneous raw GABA sum on E; add changes raw total. Both branches apply and deplete through local native Z; I cells retain native currents.',
        no_intervention=True,endpoint='All-E10ms rate>=200Hz for200ms. Recovery requires two full1s means<50Hz and >=20%quiet bins(<5Hz), both all-E and each core E.',
        recurrent_stop='2s after a second confirmed high entry, only after a verified autonomous recovery; otherwise full horizon.',
        scientific_gate='Detector is a screen. Must inspect raster, local/global rates, native fields, finite events, Z recovery, and noise. A lower high plateau or pre-entry suppression is not recovery. High-rate entry does not establish a sustained oscillatory seizure.',
        deadline='Stop dispatch90min before9h deadline; each worker checkpoints and stops cleanly at wall deadline. No model freeze.',
        observations='1ms global/regional counts;5ms400-cell E fields;0.1ms fixed80-neuron raster;20ms Z/M/current regional means;2ms native contact current readout;100ms native input summaries.',
        literature='https://elifesciences.org/articles/50927 ; archived LAS-Model source stored under literature. Current-based motif is not a full Liou conductance model reproduction.')
    base.write(OUT/'protocol.json',protocol)
    for job in jobs:base.write(OUT/'jobs'/(job['name']+'.json'),job)
    for seed in base.SEEDS:
        job=copy.deepcopy(jobs[0]);job.update(name=f'qa_s{seed}',seed=seed,horizon_s=1.,checkpoint_s=.5,qa=True,device=seed%2)
        base.write(OUT/'jobs'/(job['name']+'.json'),job)
    return protocol

def worker(name):
    protocol=prepare();assert base.sha(__file__)==protocol['producer_sha256']
    for path,h in protocol['source_hashes'].items():assert base.sha(path)==h,path
    job=base.read(OUT/'jobs'/(name+'.json'));folder=OUT/'runs'/name;folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (folder/'result.json').exists():return
    started=time.time();base.write(folder/'progress.json',dict(status='BUILDING',pid=os.getpid(),job=job))
    s,tr,frozen,identity=base.old.setup(job['seed']);assert identity==protocol['identity']
    p=s.params;ne,ni=s.n_e,s.n_i;assert (ne,ni,p.dt)==(32000,8000,.1)
    assert (s.vtheta[:ne]<18).sum()==781 and not (s.vtheta[:ne]>18).any()
    with np.load(base.OUT/'geometry.npz') as a:geo={k:a[k] for k in a.files}
    assert np.array_equal(geo['positions_e'],s.positions_e)
    centers=geo['centers_mm'];samples=geo['sample_ids'];nr=geo['region_counts']
    def groups(pos):
        d=np.linalg.norm(pos[:,None]-centers[None],axis=2);g=np.full(len(pos),2)
        g[d[:,0]<1.75]=0;g[(d[:,1]<1.75)&(d[:,1]<d[:,0])]=1;return g
    ge,gi=groups(s.positions_e),groups(s.positions_i);ix=[np.flatnonzero(ge==g) for g in range(3)]
    assert np.array_equal(nr,np.r_[np.bincount(ge,minlength=3),np.bincount(gi,minlength=3)])
    cells=base.old.spatial_cell_index(s.positions_e,n_grid=20,sheet_l_mm=p.L)
    if not (OUT/'geometry.npz').exists():
        dest=folder/'geometry.tmp.npz';np.savez_compressed(dest,**geo);dest.replace(OUT/'geometry.npz')
    cfg=base.old.MZSlowVarsConfig(use_z=True,use_m=True,tau_z=job['tau_Z_s']*1000,
        I_th_EI=job['threshold'],tau_adp=job['tau_M_s']*1000,eta_m=job['eta_m'])
    slow=RecoverySlow(ne+ni,p.V_th,cfg,NE=ne,mode=job['mode'],gamma=job['gamma'])
    cp=folder/'checkpoint.pkl';state=None;tracker=fresh_tracker();first=0
    if cp.exists():
        saved=base.load_pickle(cp);assert saved['job']==job and saved['identity']==identity
        state=saved['engine'];tracker=saved['tracker'];first=int(state['step'])
    prior_wall=tracker['wall_s'];block_start=first;stop_s=job['horizon_s']
    pop=np.zeros(2,np.uint32);regions=np.zeros(6,np.uint32);field=np.zeros(400,np.uint32);recent=np.zeros(4,np.uint32)
    names=['time_ms','spikes_1ms','regions_1ms','field_time_ms','field_5ms','raster','slow_time_ms','Z','M','currents','regional_currents','inputs','lfp_time_ms','lfp_raw']
    data={k:[] for k in names}
    recorder=base.old.LFPRecorder(p,s.net['pos'],s.net['labels'],sites=s.contact_xy)
    original=slow.apply_currents
    def apply(ie,ii,labels=None,rec=None):
        value=original(ie,ii,labels,rec);k=slow._step_index;z=slow.z[:ne];m=slow.m[:ne]
        if k%20==0:
            data['lfp_time_ms'].append(k*.1);data['lfp_raw'].append(recorder.sample(ie,slow.delivered))
        if k%200==0:
            jj=slow.delivered[:ne]
            data['slow_time_ms'].append(k*.1)
            data['Z'].append([z.mean(),z.std(),*np.quantile(z,[.1,.5,.9]),*[z[v].mean() for v in ix],np.mean(jj>=cfg.I_th_EI)])
            data['M'].append([m.mean(),*[m[v].mean() for v in ix]])
            data['currents'].append([ie[:ne].mean(),ii[:ne].mean(),np.mean(z*jj),jj.mean(),job['gamma']*slow.raw_mean])
            data['regional_currents'].append([[ie[v].mean(),ii[v].mean(),jj[v].mean(),np.mean(z[v]*jj[v]),job['eta_m']*m[v].mean()] for v in ix])
        return value
    slow.apply_currents=apply
    def inputs(tm,nu,xi):
        if round(tm/.1)%1000==0:data['inputs'].append([tm,xi,nu[:ne].mean(),nu[ne:].mean()])
    def observe(tm,spk):
        nonlocal stop_s
        k=round(tm/.1);ec=np.count_nonzero(spk[:ne]);ic=np.count_nonzero(spk[ne:])
        rr=np.bincount(ge[spk[:ne]],minlength=3).astype(np.uint32)
        pop[0]+=ec;pop[1]+=ic;regions[:3]+=rr;regions[3:]+=np.bincount(gi[spk[ne:]],minlength=3).astype(np.uint32)
        recent[0]+=ec;recent[1:]+=rr;field[:]+=np.bincount(cells[spk[:ne]],minlength=400).astype(np.uint32)
        data['raster'].append(spk[samples])
        if (k+1)%10==0:
            data['time_ms'].append((k+1)*.1-.5);data['spikes_1ms'].append(pop.copy());pop.fill(0)
            data['regions_1ms'].append(regions.copy());regions.fill(0)
        if (k+1)%50==0:
            data['field_time_ms'].append((k+1)*.1-2.5);data['field_5ms'].append(field.copy());field.fill(0)
        if (k+1)%100==0:
            sec=(k+1)*.0001;track(tracker,recent/np.r_[ne,nr[:3]]/.01,sec);recent.fill(0)
            if len(tracker['entries'])>=2:stop_s=min(stop_s,tracker['entries'][1]['confirmation_s']+2)
        if (k+1)%10000==0:
            base.write(folder/'progress.json',dict(status='RUNNING',pid=os.getpid(),job=job,time_s=(k+1)*.0001,
                phase=tracker['phase'],entries=tracker['entries'],recoveries=tracker['recoveries'],stop_s=stop_s,
                Z=float(slow.z[:ne].mean()),M_feedback=float(cfg.eta_m*slow.m[:ne].mean()),
                wall_s=prior_wall+time.time()-started))
    def checkpoint(k,engine):
        nonlocal block_start
        a={key:np.asarray(v) for key,v in data.items()}
        assert a['spikes_1ms'][:,0].sum()==a['regions_1ms'][:,:3].sum()==a['field_5ms'].sum()
        assert a['spikes_1ms'][:,1].sum()==a['regions_1ms'][:,3:].sum()
        a.update(start_step=block_start,end_step=k)
        chunks=folder/'chunks';chunks.mkdir(exist_ok=True);dest=chunks/f'{block_start:010d}_{k:010d}.npz'
        tmp=dest.with_suffix('.tmp.npz');np.savez_compressed(tmp,**a);tmp.replace(dest)
        tracker['wall_s']=prior_wall+time.time()-started
        base.save_pickle(cp,dict(job=job,identity=identity,engine=engine,tracker=tracker))
        block_start=k
        for v in data.values():v.clear()
        if job.get('qa') and first==0:raise Stop()
        if time.time()>=DEADLINE:
            tracker['stop_reason']='WALL_DEADLINE';raise Stop()
        if k*.0001>=stop_s:
            tracker['stop_reason']='RECURRENCE_OBSERVED' if len(tracker['entries'])>=2 else 'SIMULATION_HORIZON';raise Stop()
    s.net['rng']=np.random.default_rng(job['seed']);drive=base.old.make_external_drive(s,tr['spatial_ou'],job['seed'])
    last=round(job['horizon_s']*10000);p.T=(last-first+1)*.1
    stride=round(job['checkpoint_s']*10000)
    points=set(range((first//stride+1)*stride,last+1,stride))|{last}
    # Once an entry/return is found the checkpoint cadence remains fixed,
    # so stored traces may extend up to one checkpoint beyond the display stop.
    try:
        wrap_simulator(base.old.simulate_kick,device_index=job['device'])(p,s.net,KICK_BOOST=0.,slow=slow,
            V_th_per_neuron=s.vtheta,external_e_rate_drive=drive,early_stop_runaway=False,
            spike_observer=observe,input_observer=inputs,record_dense_spikes=False,fast_scatter=True,
            resume_state=state,time_offset_ms=first*.1,checkpoint_steps=points,checkpoint_sink=checkpoint,verbose=False)
    except Stop:pass
    if job.get('qa') and first==0:return
    assert np.all(slow.m[ne:]==0) and np.all(slow.z[ne:]==1)
    result=dict(status='CENSORED_WALL_DEADLINE' if tracker['stop_reason']=='WALL_DEADLINE' else 'COMPLETE',
        job=job,identity=identity,tracker=tracker,end_s=block_start*.0001,display_stop_s=min(stop_s,block_start*.0001),
        no_external_intervention=True,M_reset=False,Z_reset=False,source_hashes=protocol['source_hashes'])
    base.write(folder/'result.json',result);base.write(folder/'progress.json',result)

def qa():
    prepare();reports=[]
    rng=np.random.default_rng(93)
    for mode in ['native','mix','add']:
        cfg=base.old.MZSlowVarsConfig(use_z=True,use_m=True,tau_z=5000,I_th_EI=95.2,tau_adp=2000,eta_m=.02)
        a=base.old.ReleaseZ(10,18,cfg,NE=8);b=RecoverySlow(10,18,cfg,NE=8,mode=mode,gamma=0.)
        for k in range(100):
            ie,ii=rng.uniform(0,200,(2,10));sp=rng.random(10)<.1
            assert np.array_equal(a.apply_currents(ie,ii),b.apply_currents(ie,ii))
            a.step(sp,None,.1);b.step(sp,None,.1)
            assert np.array_equal(a.z,b.z) and np.array_equal(a.m,b.m)
    for mode in ['mix','add']:
        b=RecoverySlow(10,18,cfg,NE=8,mode=mode,gamma=.5);ie,ii=rng.uniform(0,200,(2,10))
        b.z[:8]=rng.uniform(.3,1,8);b.m[:8]=rng.uniform(0,3,8);out=b.apply_currents(ie,ii)
        jj=ii.copy();jj[:8]=(.5*ii[:8]+.5*ii[:8].mean()) if mode=='mix' else ii[:8]+.5*ii[:8].mean()
        assert np.array_equal(b.delivered,jj)
        assert np.array_equal(out,ie-b.z*jj-cfg.eta_m*b.m)
        assert np.isclose(jj[:8].sum(),ii[:8].sum()*(1 if mode=='mix' else 1.5))
        z=b.z.copy();b.step(np.zeros(10,bool),None,.1)
        assert np.array_equal(b.z[:8],z[:8]+.1/5000*((jj[:8]<cfg.I_th_EI)-z[:8]))
    for seed in base.SEEDS:
        worker(f'qa_s{seed}');worker(f'qa_s{seed}')
        def load(folder,keys):
            out={k:[] for k in keys}
            for path in sorted((folder/'chunks').glob('*.npz')):
                with np.load(path) as a:
                    for k in keys:out[k].append(a[k])
            return {k:np.concatenate(v) for k,v in out.items()}
        got=load(OUT/'runs'/f'qa_s{seed}',['raster','spikes_1ms','regions_1ms','field_5ms','Z','M','currents','inputs','lfp_raw'])
        ref=load(ROOT/'results/topic4_sef_hfo/fig5_preentry_event_audit_20260914/runs'/f'eta0.0005_s{seed}',
            ['raster','spikes_1ms','regions_1ms','field_1ms','Z','M','currents','inputs','lfp_raw'])
        for key,n in [('raster',10000),('spikes_1ms',1000),('regions_1ms',1000),('inputs',10)]:assert np.array_equal(got[key],ref[key][:n]),key
        assert np.array_equal(got['field_5ms'],ref['field_1ms'][:1000].reshape(200,5,400).sum(1))
        for key in ['Z','M']:assert np.array_equal(got[key],ref[key][:200:4]),key
        assert np.array_equal(got['currents'][:,:3],ref['currents'][:200:4])
        assert np.array_equal(got['lfp_raw'],ref['lfp_raw'][:2000:4])
        reports.append(dict(seed=seed,native_bitwise=True,checkpoint_resume=True))
    # No short dip, suppressed initiation, or a surviving tonic core may pass.
    tr=fresh_tracker()
    for k in range(20):track(tr,[220,300,300,210],(k+1)*.01)
    for k in range(20,250):track(tr,[0,300,0,0],(k+1)*.01)
    assert not tr['recoveries']
    for k in range(250,500):track(tr,[0,0,0,0],(k+1)*.01)
    assert len(tr['recoveries'])==1
    for k in range(500,520):track(tr,[220,300,300,210],(k+1)*.01)
    assert len(tr['entries'])==2
    base.write(OUT/'qa.json',dict(status='PASS',native_reports=reports,redistribution_and_Z_equations=True,core_recovery_required=True))

def supervise():
    p=prepare();assert base.read(OUT/'qa.json')['status']=='PASS'
    lock=(OUT/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pending=[j['name'] for j in p['initial_jobs'] if not (OUT/'runs'/j['name']/'result.json').exists()]
    children={};failed=[]
    while pending or children:
        for name,child in list(children.items()):
            if child.poll() is None:continue
            del children[name]
            if child.returncode or not (OUT/'runs'/name/'result.json').exists():failed.append(name)
        old_count=sum('run_topic4_fig5_log_m_scan.py' in ' '.join(x.info['cmdline'] or []) and 'worker' in (x.info['cmdline'] or []) for x in psutil.process_iter(['cmdline']))
        cap=min(p['max_workers'],max(0,p['max_combined_with_log_scan']-old_count))
        mem=psutil.virtual_memory().available/2**30
        while pending and not failed and len(children)<cap and mem>80 and shutil.disk_usage(OUT).free/2**30>40 and time.time()<DEADLINE-5400:
            name=pending.pop(0);folder=OUT/'runs'/name;folder.mkdir(parents=True,exist_ok=True)
            with (folder/'worker.log').open('a') as log:
                children[name]=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'worker','--name',name],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            mem-=4
        base.write(OUT/'status.json',dict(status='RUNNING' if not failed else 'DRAINING_FAILURE',pid=os.getpid(),updated_at=time.time(),deadline_epoch=DEADLINE,
            completed=sum((OUT/'runs'/j['name']/'result.json').exists() for j in p['initial_jobs']),total_initial=len(p['initial_jobs']),
            running={n:c.pid for n,c in children.items()},pending=pending,failed=failed,log_scan_workers=old_count))
        if not children and (failed or time.time()>=DEADLINE-5400):break
        time.sleep(10)
    base.write(OUT/'round1_dispatch_finished.json',dict(time=time.time(),failed=failed,not_dispatched=pending,needs_scientific_review=True))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','qa','worker','supervise']);ap.add_argument('--name');a=ap.parse_args()
    try:
        if a.mode=='prepare':prepare()
        elif a.mode=='qa':qa()
        elif a.mode=='worker':worker(a.name)
        else:supervise()
    except Exception as e:
        path=OUT/'runs'/a.name/'failure.json' if a.name else OUT/(a.mode+'_failure.json')
        base.write(path,dict(error=repr(e),time=time.time()));raise
