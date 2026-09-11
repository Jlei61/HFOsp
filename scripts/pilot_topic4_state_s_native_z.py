#!/usr/bin/env python3
"""Independent historical manual-core s x native-Z factorial pilot; M stays off."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, sys, time, subprocess, traceback
from pathlib import Path
import numpy as np
W = Path(__file__).resolve().parents[1]
MAIN = Path('/home/honglab/leijiaxin/HFOsp')
R1 = MAIN/'.worktrees/topic4-continuous-core-state-r1'
OUT = Path('/data/hfosp/topic4_sef_hfo/state_s_native_z_pilot_20260909')
PARENT = MAIN/'results/topic4_sef_hfo/historical_manual_hard_native_z_v1'
DESIGN = W/'config/topic4_state_s_native_z_pilot.json'
sys.path.insert(0, str(MAIN/'scripts'))
from topic4_historical_manual_z_common import setup as historical_setup, read, write
from validate_topic4_fixed_rate_base import make_external_drive, spatial_cell_index
import src
src.__path__.append(str(R1/'src'))
from src import topic4_initial_state_runtime as rt
from src.topic4_observation_repaired import observe
from src.sef_hfo_observation import sample_envelopes
from src.sef_hfo_snn_adapter import snn_event_envelope


def load(name):
    spec=importlib.util.spec_from_file_location('szpilot_'+name,W/'scripts/state_s_z_pilot'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module
engine=load('engine'); parent_engine=load('engine_parent'); state=load('state_projection'); native=load('native_zm')
class EndRun(Exception): pass
class CompactZ(native.MZSlowVars):
    def _record_trace(self, spikes, dt): pass


def setup(seed):
    # The unchanged historical placement reader resolves its data relative to cwd.
    previous=Path.cwd()
    try:
        os.chdir(MAIN)
        return historical_setup(seed)
    finally:
        os.chdir(previous)


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def source_hashes():
    files={Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m,'__file__',None) and str(m.__file__).endswith('.py') and str(MAIN) in str(m.__file__)}
    files.add(Path(__file__).resolve())
    return {str(p):sha(p) for p in sorted(files)}
def sources_match(record):
    for p,h in record.items():
        if sha(p)!=h: raise RuntimeError('Qualified source changed: '+p)


def projection(sub, frozen, value, seed, duration, warmup=1000.):
    centers=np.asarray(frozen['candidate']['node_field']['centers_mm'])
    distances=np.linalg.norm(np.asarray(sub.net['pos'])[:,None]-centers[None],axis=2)
    nearest=distances.argmin(1);ne=sub.net['NE'];ni=sub.net['NI'];radius=1.5
    ids=[ne+np.flatnonzero((distances[ne:,j]<=radius)&(nearest[ne:]==j)) for j in (0,1)]
    n=min(map(len,ids));loading=np.r_[np.full(len(ids[0]),n/len(ids[0])),np.full(len(ids[1]),-n/len(ids[1]))]
    ctl=state.ContinuousIState(np.r_[*ids],loading,np.full(round(duration/.1),value),amplitude=.2,dt_ms=.1,seed=seed+100000,warmup_ms=warmup,ramp_ms=200.)
    groups=np.full(ne+ni,2); groups[ne:]=5
    for j in (0,1):
        groups[:ne][(distances[:ne,j]<=radius)&(nearest[:ne]==j)]=j
        groups[ids[j]]=j+3
    return ctl,groups


def slow_for(sub, use_z):
    return CompactZ(sub.net['NE']+sub.net['NI'],sub.params.V_th,native.MZSlowVarsConfig(use_z=use_z,use_m=False,tau_z=5000.,I_th_EI=95.19851312666987),NE=sub.net['NE'])


def short_run(sub,tr,frozen,use_z,value,which,control=True,warmup=0.):
    sub.params.T=250.;seed=9108401;sub.net['rng']=np.random.default_rng(seed)
    slow=slow_for(sub,use_z);ctl,_=projection(sub,frozen,value,seed,250.,warmup)
    digest=hashlib.sha256(); samples=[];rates=[];last=[]
    ids=np.load(PARENT/'reference_samples/runs/z_current_e_seed9108401.npz')['sample_ids']
    def cb(tm,sp):
        digest.update(sp.tobytes());samples.append(sp[ids].copy());rates.append([sp[:sub.n_e].sum()/sub.n_e*10000,sp[sub.n_e:].sum()/sub.n_i*10000])
    kw={'external_i_state':ctl if control else None} if which is engine else {}
    which.simulate_kick(sub.params,sub.net,KICK_BOOST=0.,V_th_per_neuron=sub.vtheta,slow=slow,external_e_rate_drive=make_external_drive(sub,tr['spatial_ou'],seed),spike_observer=cb,record_dense_spikes=False,fast_scatter=True,verbose=False,**kw)
    return dict(spike_hash=digest.hexdigest(),sample=np.asarray(samples),rates=np.asarray(rates),Z=slow.z.copy(),audit=ctl.audit() if control and which is engine else None)


def qualify():
    t=time.time();OUT.mkdir(parents=True,exist_ok=True)
    write(OUT/'status.json',{'status':'QUALIFYING','started_unix':t})
    sub,tr,frozen,identity=setup(9108401)
    reference=read(PARENT/'substrate.json');tests={'parent_substrate_identity':identity==reference['identity']}
    for uz in (False,True):
        old=short_run(sub,tr,frozen,uz,0.,parent_engine,False)
        new=short_run(sub,tr,frozen,uz,0.,engine)
        tests[f'zero_exact_all_spikes_Z{int(uz)}']=old['spike_hash']==new['spike_hash']
        tests[f'zero_exact_resource_Z{int(uz)}']=np.array_equal(old['Z'],new['Z'])
        if uz:
            with np.load(PARENT/'trajectory.npz') as ref:
                tests['historical_first250ms_sample_spikes']=np.array_equal(new['sample'],ref['sample_spikes'][:2500])
            zero=new
    active=short_run(sub,tr,frozen,True,.5,engine)
    tests['active_common_background']=active['audit']['legacy_external_counts_sha256']==zero['audit']['legacy_external_counts_sha256']
    tests['active_changes_I_input']=active['audit']['changed_cell_steps']>0
    tests['expected_I_budget_conserved']=active['audit']['maximum_relative_total_rate_error']<1e-12
    # Binning/smoothing the streaming counts is the same observer as dense spikes.
    synthetic=np.random.default_rng(8).random((500,sub.n_e))<.01
    env,_,_=snn_event_envelope(synthetic,sub.positions_e,sub.montage,.1)
    b=synthetic.reshape(25,20,sub.n_e).sum(1)
    tests['streaming_envelope_matches_adapter']=np.array_equal(envelope(b,sub),env)
    design1=read(R1/'config/topic4_continuous_core_state_r1.json')
    evaluator=rt.load_evaluator(design1);contract=rt.load_observation_contract(design1)
    tests['observer_contact_order']=list(sub.contact_names)==list(contract['contact_names'])
    tests={k:bool(v) for k,v in tests.items()}
    print(json.dumps(tests),flush=True)
    q=dict(status='PASS' if all(tests.values()) else 'FAIL',tests=tests,identity=identity,source_hashes=source_hashes(),wall_seconds=time.time()-t)
    write(OUT/'qualification.json',q)
    if not all(tests.values()): raise RuntimeError(str(tests))
    print(json.dumps({'qualification':q['status'],'seconds':q['wall_seconds']}),flush=True)


def envelope(binned,sub):
    sig=2.5;half=int(np.ceil(3*sig));x=np.arange(-half,half+1);k=np.exp(-x*x/(2*sig*sig));k/=k.sum()
    sm=np.apply_along_axis(lambda col:np.convolve(col,k,mode='same'),0,np.asarray(binned,float))
    return sample_envelopes(sm,np.asarray(sub.positions_e,float),sub.montage,.25)


def worker(job):
    start=time.time();q=read(OUT/'qualification.json');assert q['status']=='PASS';sources_match(q['source_hashes'])
    folder=OUT/'workers';folder.mkdir(exist_ok=True);stem=folder/job['id'];progress=stem.with_suffix('.progress.json')
    if stem.with_suffix('.json').exists(): raise RuntimeError('Refuse overwriting completed job')
    write(progress,dict(status='BUILDING',job=job,started_unix=start))
    sub,tr,frozen,identity=setup(job['seed']);assert identity==q['identity']
    duration=20000.;dt=.1;steps=200000;ne=sub.n_e;ni=sub.n_i;sub.params.T=duration
    slow=slow_for(sub,job['Z_dynamic']);ctl,groups=projection(sub,frozen,job['s'],job['seed'],duration)
    with np.load(PARENT/'reference_samples/runs/z_current_e_seed9108401.npz') as z:sample=z['sample_ids'];sample_groups=z['sample_groups']
    group_n=np.bincount(groups,minlength=6);assert np.all(group_n>0)
    cells=spatial_cell_index(sub.positions_e,n_grid=20,sheet_l_mm=sub.params.L);cell_n=np.bincount(cells,minlength=400)
    spikes=np.zeros((steps,len(sample)),bool);rates=np.zeros((steps,2),np.float32)
    binned=np.zeros((steps//20,ne),np.uint8);region_counts=np.zeros((steps//20,6),np.uint32)
    trace=[];zfield=[];seen=0;count10=0;high_ms=0;detected=None;prefix=hashlib.sha256();fullhash=hashlib.sha256()
    def current(tm,ie,ii,v):
        if round(tm/dt)%100:return
        row=[tm]
        for j in range(6):
            ids=np.flatnonzero(groups==j)
            row.extend([float(np.mean(ie[ids])),float(np.mean(ii[ids])),float(np.mean(slow.z[ids]*ii[ids])),float(np.mean(slow.z[ids])),float(np.mean(ii[ids]>=slow.cfg.I_th_EI))])
        row.extend([float(slow.z[:ne].std()),float(np.mean(slow.z[:ne])),float(np.min(slow.z[:ne]))]);trace.append(row)
        zfield.append(np.divide(np.bincount(cells,weights=slow.z[:ne],minlength=400),cell_n,out=np.full(400,np.nan),where=cell_n>0))
        if not np.isfinite(v).all():raise RuntimeError('Nonfinite membrane state')
    def spike(tm,sp):
        nonlocal seen,count10,high_ms,detected
        i=round(tm/dt);seen=i+1;ec=int(sp[:ne].sum());ic=int(sp[ne:].sum())
        spikes[i]=sp[sample];rates[i]=[ec/ne*10000,ic/ni*10000];binned[i//20]+=sp[:ne]
        region_counts[i//20]+=np.bincount(groups[sp],minlength=6).astype(np.uint32)
        fullhash.update(sp.tobytes())
        if seen<=10000:prefix.update(sp.tobytes())
        count10+=ec
        if seen%100==0:
            rate=count10/ne/.01;count10=0;high_ms=high_ms+10 if rate>=200 else 0
            if detected is None and high_ms>=200:detected=seen*dt
        if seen%10000==0:
            write(progress,dict(status='SIMULATING',job=job,simulated_ms=seen*dt,detected_ms=detected,Z_mean=float(slow.z[:ne].mean()),wall_seconds=time.time()-start))
        if detected is not None and seen*dt>=detected+2000:raise EndRun()
    sub.net['rng']=np.random.default_rng(job['seed'])
    try:
        engine.simulate_kick(sub.params,sub.net,KICK_BOOST=0.,V_th_per_neuron=sub.vtheta,slow=slow,external_e_rate_drive=make_external_drive(sub,tr['spatial_ou'],job['seed']),external_i_state=ctl,early_stop_runaway=False,current_observer=current,spike_observer=spike,record_dense_spikes=False,fast_scatter=True,verbose=False)
    except EndRun:pass
    write(progress,dict(status='OBSERVING',job=job,simulated_ms=seen*dt,detected_ms=detected))
    nf=seen//20;binned=binned[:nf];env=envelope(binned,sub)
    design1=read(R1/'config/topic4_continuous_core_state_r1.json');contract=rt.load_observation_contract(design1);evaluator=rt.load_evaluator(design1)
    observation=observe(env,2.,contract);mu=np.asarray(observation['centroid_ms'],float).reshape(-1,len(sub.contact_names));labels,support,dist=rt.classify_with_both_modes(evaluator,mu)
    events=[];end_interictal=detected-200 if detected is not None else seen*dt
    for k,e in enumerate(observation['events']):
        window=e['window_ms'];eligible=bool(e['primary_eligible'] and window[0]>=1500 and window[1]<=end_interictal)
        a=max(0,int(window[0]/2));b=min(nf,int(window[1]/2)+1);onsets=[]
        for j in (0,1):
            r=region_counts[a:b,j]/group_n[j]*500.;ids=np.flatnonzero(r>=.2*r.max(initial=0)) if r.max(initial=0)>0 else []
            onsets.append(float((a+ids[0])*2) if len(ids) else None)
        events.append(dict(index=k,window_ms=window,mode=int(labels[k]),support=int(support[k]),observer_primary=bool(e['primary_eligible']),pilot_interictal_eligible=eligible,exclusions=e['primary_exclusion_reasons'],n_contacts=int(np.isfinite(mu[k]).sum()),core_A_B_rise20_ms=onsets))
    field=np.zeros((nf,400),np.uint32)
    for k in range(nf):field[k]=np.bincount(cells,weights=binned[k],minlength=400).astype(np.uint32)
    checks={'spatial_spike_count_conserved':int(field.sum())==int(binned.sum()),'group_spike_count_conserved':int(region_counts[:nf,:3].sum())==int(binned.sum()),'I_Z_exactly_one':bool(np.all(slow.z[ne:]==1)),'Z_bounded':bool(np.all((slow.z>=0)&(slow.z<=1))),'frozen_Z_exactly_one':bool(job['Z_dynamic'] or np.all(slow.z==1)),'source_snapshot_unchanged':True}
    sources_match(q['source_hashes'])
    if not all(checks.values()):raise RuntimeError(str(checks))
    arrays=dict(sample_spikes=spikes[:seen],sample_ids=sample,sample_groups=sample_groups,rate_E=rates[:seen,0],rate_I=rates[:seen,1],dt_ms=np.array(dt),trace=np.asarray(trace),Z_field=np.asarray(zfield,np.float32),field_E_counts=field,region_counts=region_counts[:nf],group_n=group_n,frame_ms=np.array(2.),contact_envelope=env.astype(np.float32),centroid_ms=mu,event_mode=labels,event_support=support,contact_names=np.asarray(sub.contact_names),contact_xy=sub.contact_xy,centers=np.asarray(frozen['candidate']['node_field']['centers_mm']),Z_final=slow.z.astype(np.float32),groups=groups,positions_E=sub.positions_e,I_target_indices=ctl.indices,I_loading=ctl.loading)
    rt.atomic_npz(stem.with_suffix('.npz'),**arrays)
    record=dict(status='COMPLETE',job=job,design_sha256=sha(DESIGN),actual_duration_ms=seen*dt,detected_ms=detected,high_interval_start_ms=detected-200 if detected is not None else None,latency_right_censored=detected is None,observation_stop='2s_after_detection' if detected is not None and seen*dt<duration else '20s_budget',manual_restore=False,M=False,identity=identity,input_audit=ctl.audit(),first1s_spike_sha256=prefix.hexdigest(),all_spike_sha256=fullhash.hexdigest(),events=events,checks=checks,wall_seconds=time.time()-start,arrays_sha256=sha(stem.with_suffix('.npz')))
    write(stem.with_suffix('.json'),record);write(progress,dict(status='COMPLETE',job=job,simulated_ms=seen*dt,detected_ms=detected));print(json.dumps({'job':job,'detected_ms':detected,'seconds':time.time()-start}),flush=True)


def controller():
    import fcntl
    lock=open(OUT/'controller.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    design=read(DESIGN);assert read(OUT/'qualification.json')['status']=='PASS'
    todo=list(design['jobs']);active={};failed=[];done=[];(OUT/'logs').mkdir(exist_ok=True)
    while todo or active:
        while todo and len(active)<3:
            j=todo.pop(0);result=OUT/'workers'/f"{j['id']}.json"
            if result.exists():
                r=read(result)
                if r['status']=='COMPLETE' and r['design_sha256']==sha(DESIGN) and r['arrays_sha256']==sha(result.with_suffix('.npz')):done.append(j['id']);continue
                raise RuntimeError('Unverified existing result')
            log=open(OUT/'logs'/f"{j['id']}.log",'w')
            proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'worker','--job',j['id']],stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy());active[j['id']]=(proc,log)
        for k,(p,log) in list(active.items()):
            code=p.poll()
            if code is not None:
                log.close();del active[k];(failed if code else done).append(k)
        write(OUT/'status.json',dict(status='RUNNING',total=len(design['jobs']),complete=len(done),failed=failed,active={k:p.pid for k,(p,l) in active.items()},pending=len(todo),updated_unix=time.time()))
        if todo or active:time.sleep(5)
    if failed:
        write(OUT/'status.json',dict(status='INCOMPLETE_WORKER_FAILURE',complete=len(done),failed=failed));return
    write(OUT/'status.json',dict(status='ANALYZING',complete=len(done),total=len(done)))
    rc=subprocess.call([sys.executable,str(W/'scripts/analyze_topic4_state_s_native_z_pilot.py')])
    write(OUT/'status.json',dict(status='ROUND_COMPLETE_PENDING_SCIENTIFIC_REVIEW' if rc==0 else 'ANALYSIS_FAILED',complete=len(done),total=len(done),updated_unix=time.time()))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['qualify','worker','controller']);ap.add_argument('--job');a=ap.parse_args()
    if a.action=='qualify':qualify()
    elif a.action=='controller':controller()
    else:worker(next(j for j in read(DESIGN)['jobs'] if j['id']==a.job))
if __name__=='__main__':main()
