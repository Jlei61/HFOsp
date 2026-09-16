#!/usr/bin/env python3
"""Bounded physical pilot for the frozen five-block observable objective.

Reuse nine initial conditions; 12 new conditions x two TRAIN units, then
at most five same-pool nominees x two new-noise units. Native diagnostics
never enter proposals or nomination. Does not modify the active v1 run.
"""
from pathlib import Path
import argparse,copy,csv,fcntl,json,os,pickle,shutil,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts import run_topic4_xy_research as base
from scripts import run_topic4_multievent_distribution_v2_1 as engine
from scripts import run_topic4_contact_timing_shape_pilot as prior
from scripts.audit_topic4_multievent_execution_parameters_v2_1 import audit as parameter_audit
from src.topic4_contact_event_objective_v2 import worker_events
from src.topic4_envelope_joint_pilot import model_events,combined_score
from src.topic4_multievent_condition_identity_v2_1 import condition_key

OUT=ROOT/'results/topic4_sef_hfo/observable_loss_physical_pilot'
REV=ROOT/'results/topic4_sef_hfo/contact_event_objective_revision_v2'
TRAIN=[(6101,7101),(6102,7101)]
CONFIRM=[(6101,842901),(6102,842901)]
KEYS=['participation','centroid_structure','local_shape','recruitment','joint_envelope']


def write(path,data):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)


def frozen():
    d=base.read(OUT/'design.json')
    for p,h in d['frozen_files'].items():
        if base.sha(Path(p))!=h:raise RuntimeError(f'frozen pilot source changed: {p}')
    return d,pickle.load((OUT/'objective.pkl').open('rb'))


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'design.json').exists():return frozen()
    old_design,v1,old=prior.get_frozen()
    shutil.copyfile(REV/'objective.pkl',OUT/'objective.pkl')
    files=[OUT/'objective.pkl',OUT/'protocol.md',REV/'manifest.json',prior.OUT/'design.json',prior.OUT/'objective.pkl',
           prior.OUT/'baseline_train_scores.json',prior.OUT/'A_scores.json',prior.OLD/'training_objective_v2_1.pkl',
           ROOT/'src/topic4_contact_event_objective_v2.py',Path(__file__),
           ROOT/'scripts/analyze_topic4_observable_loss_physical_pilot.py',
           ROOT/'scripts/run_topic4_contact_timing_shape_pilot.py',
           ROOT/'scripts/run_topic4_multievent_distribution_v2_1.py',
           ROOT/'scripts/paper_figures/audit_topic4_native_activity_shortcuts.py']
    d=dict(version='observable_loss_physical_pilot_v1',status='FROZEN_BEFORE_NEW_PHYSICS',
        objective_source=str(REV/'manifest.json'),anchors=old_design['anchors'],
        weights={k:.2 for k in KEYS},train_pairs=TRAIN,confirmation_pairs=CONFIRM,
        parameter_names=prior.PARAMETERS,global_lower=prior.LOW.tolist(),global_upper=prior.HIGH.tolist(),
        anchor_half_width=prior.STEP.tolist(),physical_readout='unchanged: 24s; 2ms bins; 5ms smoothing; 250ms window; no Z/M, no directed stimulation; same OU law',
        fixed='core centers, core membership rule, topology units, EE ellipse angle/aspect and observation model',
        initialization='Each anchor receives a distinct new randomized direction per batch; proposals are persisted before dispatch.',
        budget=dict(reused_conditions=9,new_training_conditions=12,new_training_runs=24,maximum_confirmation_conditions=5,
                    maximum_confirmation_runs=10,maximum_total_new_runs=34,duration_ms=24000,maximum_global_SNN_workers=8),
        adaptation='C parents selected by revised loss from initial nine conditions within each anchor; D parents updated after all C outcomes; no native diagnostics in selection.',
        comparison='Top two under revised and prior-v1 losses from exactly the same 21-condition pool, plus original starting point 1; physical-identity deduplication before confirmation.',
        attribution_limit='Same-pool selection comparison, not equal-budget proof of optimizer superiority. Proposal pool is adaptively generated under revised loss.',
        native_validation=dict(selection_used=False,primary_diagnostics=['simultaneous secondary family mass','threshold-sensitive disconnected native mass',
            'core peak lag and lead identity by label','full-contact width and overlap'],
            interpretation='No automatic causal pass. Family segmentation and 1mm/2ms observation limit conclusions; outside activity alone is not a violation.'),
        final_delivery='paired fresh-noise diagnostics, parameter x five-loss plots, complete event index, label-organized multi-event full-contact/native GIFs; review stop',
        patient_waveforms='No held-back waveforms used for proposals. TRAIN reference only in automatic movie comparison; Fig2C path judgement remains human validation.',
        stop='After this bounded round, automatic analysis and review; no expanded search, model freeze, or Fig5.',
        frozen_files={str(p):base.sha(p) for p in files})
    write(OUT/'design.json',d);write(OUT/'status.json',dict(status='PREPARED',updated_unix=time.time()))
    return frozen()


def score(candidates,paths,phase,obj):
    _,v1,old=prior.get_frozen();records=[]
    for c in candidates:
        units={}
        for uid,path in paths[c['candidate_id']].items():
            engine.repaired_observation(path)
            tab,details,names,w=worker_events(path)
            if names!=obj.names:raise RuntimeError('contact identity mismatch')
            new=obj.score(tab,w.get('physical_status'))
            model=model_events(path,engine.repaired_observation);previous=combined_score(old,v1,model)
            units[uid]=dict(worker_path=str(path),new=new,old=previous)
        good=bool(units) and all(u['new']['loss'] is not None for u in units.values())
        old_good=bool(units) and all(u['old']['loss'] is not None for u in units.values())
        records.append(dict(candidate_id=c['candidate_id'],candidate=c,units=units,ranking_eligible=good,
            loss=float(np.mean([u['new']['loss'] for u in units.values()])) if good else None,
            old_loss=float(np.mean([u['old']['loss'] for u in units.values()])) if old_good else None))
    report=dict(phase=phase,candidates=records);write(OUT/f'{phase}_scores.json',report);return report


def initial(obj):
    if (OUT/'initial_scores.json').exists():return base.read(OUT/'initial_scores.json')
    rr=[r for f in ['baseline_train_scores.json','A_scores.json'] for r in base.read(prior.OUT/f)['candidates']]
    paths={r['candidate_id']:{u:v['worker_path'] for u,v in r['units'].items()} for r in rr}
    return score([r['candidate'] for r in rr],paths,'initial',obj)


def anchor_index(c,anchors):
    centers=c['node_field']['centers_mm']
    for i,a in enumerate(anchors):
        if centers==a['node_field']['centers_mm']:return i
    raise ValueError('new core placement is outside this pilot')


def parents(pool,anchors):
    return [min([r for r in pool if r['ranking_eligible'] and anchor_index(r['candidate'],anchors)==i],
                key=lambda r:(r['loss'],r['candidate_id']))['candidate'] for i in range(len(anchors))]


def proposals(phase,d,pp):
    path=OUT/f'{phase}_proposals.json'
    if path.exists():return base.read(path)['candidates']
    cc=[];draws=[]
    for i,(anchor,parent) in enumerate(zip(d['anchors'],pp),1):
        rng=np.random.default_rng((842100 if phase=='C' else 842200)+i)
        direction=rng.choice([-1.,1.],5)*rng.uniform(.35,1.,5)
        lo=np.maximum(prior.LOW,prior.vector(anchor)-prior.STEP)
        hi=np.minimum(prior.HIGH,prior.vector(anchor)+prior.STEP);width=hi-lo
        for sign,tag in [(1.,'plus'),(-1.,'minus')]:
            raw=prior.vector(parent)+sign*(1. if phase=='C' else .65)*prior.STEP*direction
            v=lo+width-np.abs((raw-lo)%(2*width)-width)
            cid=f'obs5_anchor{i}_{phase}_{tag}'
            c=prior.with_vector(parent,v,cid,parent['candidate_id'],phase);cc.append(c)
            draws.append(dict(candidate_id=cid,parent=parent['candidate_id'],seed=(842100 if phase=='C' else 842200)+i,
                direction=direction.tolist(),sign=sign,raw=raw.tolist(),applied=v.tolist(),bounds=[lo.tolist(),hi.tolist()]))
    if len({condition_key(c) for c in cc})!=len(cc):raise RuntimeError('duplicate physical proposal')
    write(path,dict(candidates=cc,draws=draws,phase=phase,delayed_update=True));return cc


def external_workers(exclude=()):
    out=[]
    for p in Path('/proc').glob('[0-9]*'):
        try:
            if int(p.name) in exclude:continue
            args=(p/'cmdline').read_bytes().split(b'\0')
            if any(Path(os.fsdecode(a)).name in {'run_topic4_multidimensional_worker.py','run_topic4_rev12_node_worker.py'} for a in args if a):out.append(int(p.name))
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    return out


def run_jobs(phase,jobs,execution,workers):
    folder,cp,mp,snapshot=execution
    for name in ['workers','run_logs','resource_logs']:(folder/name).mkdir(exist_ok=True)
    pending=[];complete=[];active={};failures=[]
    for job in jobs:
        path=folder/'workers'/f'{engine._stem(*job)}.json'
        (complete if engine._complete(path,snapshot) else pending).append(job)
    # A killed controller must not rerun jobs still alive under another parent.
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            args=(proc/'cmdline').read_bytes().split(b'\0')
            if str(cp).encode() in args and any(a.endswith(b'run_topic4_multidimensional_worker.py') for a in args):
                raise RuntimeError(f'orphan worker {proc.name} still owns this execution; wait for it before restart')
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    while pending or active:
        for job,(proc,stream,peaks) in list(active.items()):
            sample=engine._memory_sample(proc.pid)
            for k in peaks:peaks[k]=max(peaks[k],sample[k])
            stem=engine._stem(*job)
            engine._append_jsonl(folder/'resource_logs'/f'{stem}.jsonl',dict(unix=time.time(),**sample))
            if proc.poll() is None:continue
            stream.close();del active[job]
            write(folder/'resource_logs'/f'{stem}_summary.json',dict(candidate_id=job[0],topology_seed=job[1],dynamics_seed=job[2],
                exit_code=proc.returncode,peak_process_tree=peaks))
            if proc.returncode or not engine._complete(folder/'workers'/f'{stem}.json',snapshot):failures.append(dict(job=job,exit_code=proc.returncode))
            else:complete.append(job)
        pids=set()
        for proc,_,_ in active.values():pids.update(engine._process_tree(proc.pid))
        external=external_workers(pids)
        reserve=engine._other_snn_reserve_gib(pids)
        allowance=max(0,int((base.available_gib()-40-reserve-18*len(active))/18))
        slots=max(0,min(workers-len(active),8-len(external)-len(active),allowance,len(pending)))
        # Foreign controllers do not share an atomic global slot allocator.
        # Start a new batch only after their current workers have drained,
        # avoiding a race with their immediate replacement dispatch.
        waiting_for_external=bool(external and not active)
        if waiting_for_external:slots=0
        if shutil.disk_usage(OUT).free<30*1024**3:failures.append(dict(error='disk free below 30 GiB'))
        if failures:slots=0
        for _ in range(slots):
            job=pending.pop(0);stem=engine._stem(*job);out=folder/'workers'/f'{stem}.json'
            stream=(folder/'run_logs'/f'{stem}.log').open('w')
            cmd=['/usr/bin/prlimit',f'--as={18*1024**3}','--',engine.PYTHON,str(engine.WORKER),
                '--config',str(cp),'--candidate-id',job[0],'--seed',str(job[1]),'--topology-seed',str(job[1]),
                '--dynamics-seed',str(job[2]),'--expected-commit',commit,'--runtime-manifest',str(snapshot),
                '--artifact-root',str(engine.ARTIFACT_ROOT),'--out-json',str(out),'--out-npz',str(out.with_suffix('.npz'))]
            proc=subprocess.Popen(cmd,cwd=ROOT,env=engine.ENV,stdout=stream,stderr=subprocess.STDOUT)
            active[job]=(proc,stream,dict(rss_bytes=0,pss_bytes=0,vms_bytes=0))
        write(OUT/'status.json',dict(status='FAILURE_DRAINING' if failures else (f'QUEUED_{phase}_WAITING_FOR_EXTERNAL_BATCH' if waiting_for_external else f'RUNNING_{phase}'),stage=phase,total=len(jobs),
            complete=len(complete),running=len(active),pending=len(pending),external_SNN_workers=len(external),
            maximum_global_SNN_workers=8,memory_available_gib=base.available_gib(),failures=failures,updated_unix=time.time(),
            active=[dict(candidate_id=j[0],topology_seed=j[1],dynamics_seed=j[2],pid=v[0].pid) for j,v in active.items()]))
        if failures and not active:raise RuntimeError(f'worker failure: {failures}')
        if pending or active:time.sleep(10)


def run_phase(phase,candidates,pairs,obj,workers):
    execution=execution_files(phase,candidates,pairs)
    jobs=[(c['candidate_id'],t,n) for c in candidates for t,n in pairs]
    run_jobs(phase,jobs,execution,workers);frozen()
    audit=parameter_audit(require_complete=True,execution=execution[0],seed_pairs=pairs,output_path=OUT/f'{phase}_parameter_audit.json')
    if audit['status']!='PARAMETER_APPLICATION_AUDIT_PASS':raise RuntimeError('actual physical parameter audit failed')
    paths={c['candidate_id']:{f'topo_{t}_dyn_{n}':str(execution[0]/'workers'/f'{engine._stem(c["candidate_id"],t,n)}.json') for t,n in pairs} for c in candidates}
    return score(candidates,paths,phase,obj)


def execution_files(phase,candidates,pairs):
    folder=OUT/'execution'/phase;folder.mkdir(parents=True,exist_ok=True)
    cp=folder/'execution_config.json';mp=folder/'candidate_manifest.json';sp=folder/'runtime_snapshot.json'
    if not cp.exists():
        source=prior.OLD/'execution/confirmation_24s'
        cfg=copy.deepcopy(base.read(source/'execution_config.json'))
        cfg.update(output_root=str(folder),candidate_manifest=str(mp))
        cfg['search']['dynamics_seeds']=sorted({n for _,n in pairs})
        cfg['search']['fit_network_seeds']=sorted({t for t,_ in pairs})
        write(cp,cfg);write(mp,dict(config_sha256=base.sha(cp),candidates=candidates,phase=phase,frozen_before_simulation=True))
        hashes=base.read(source/'runtime_snapshot.json')['source_hashes']
        write(sp,dict(source_hashes=hashes,input_hashes={str(cp):base.sha(cp),str(mp):base.sha(mp)},
            identity_kind='dependency_scoped_source_hash_snapshot',not_final_substrate_freeze=True))
    snapshot=base.read(sp)
    for p,h in snapshot['source_hashes'].items():
        if base.sha(ROOT/p)!=h:raise RuntimeError(f'physical dependency changed: {p}')
    for p,h in snapshot['input_hashes'].items():
        if base.sha(Path(p))!=h:raise RuntimeError(f'execution inputs changed: {p}')
    if base.read(mp)['candidates']!=candidates:raise RuntimeError('persisted proposal changed')
    return folder,cp,mp,sp


def nominate(pool,original):
    selected={};seen={};roles={}
    for key,role in [('loss','revised'),('old_loss','prior')]:
        rr=sorted([r for r in pool if r['ranking_eligible'] and r[key] is not None],key=lambda r:(r[key],r['candidate_id']))
        chosen=[]
        for r in rr:
            ident=condition_key(r['candidate'])
            if ident in chosen:continue
            chosen.append(ident);cid=seen.setdefault(ident,r['candidate_id']);selected[cid]=next(x for x in pool if x['candidate_id']==cid)
            roles.setdefault(cid,[]).append(role)
            if len(chosen)==2:break
    orig=next(r for r in pool if r['candidate_id']==original);ident=condition_key(orig['candidate'])
    cid=seen.setdefault(ident,original);selected[cid]=next(x for x in pool if x['candidate_id']==cid);roles.setdefault(cid,[]).append('starting_reference')
    assert len(selected)<=5
    return dict(nominees=list(selected.values()),roles=roles,selection_data='same TRAIN candidate pool',
                native_diagnostics_used=False,confirmation_data_used=False)


def run(workers):
    d,obj=prepare();pool=initial(obj)['candidates']
    for phase in ['C','D']:
        cc=proposals(phase,d,parents(pool,d['anchors']))
        pool+=run_phase(phase,cc,TRAIN,obj,workers)['candidates']
        subprocess.run([engine.PYTHON,str(ROOT/'scripts/analyze_topic4_observable_loss_physical_pilot.py'),'--training-only'],env=engine.ENV,cwd=ROOT,check=True)
    proposal=nominate(pool,d['anchors'][0]['candidate_id'])
    if (OUT/'nomination.json').exists():
        if base.read(OUT/'nomination.json')!=proposal:raise RuntimeError('nomination changed on restart')
    else:write(OUT/'nomination.json',proposal)
    run_phase('confirmation',[r['candidate'] for r in proposal['nominees']],CONFIRM,obj,workers)
    write(OUT/'status.json',dict(status='PHYSICAL_COMPLETE_ANALYZING',updated_unix=time.time()))
    subprocess.run([engine.PYTHON,str(ROOT/'scripts/analyze_topic4_observable_loss_physical_pilot.py'),'--final'],env=engine.ENV,cwd=ROOT,check=True)
    write(OUT/'status.json',dict(status='PILOT_COMPLETE_PENDING_SCIENTIFIC_REVIEW',updated_unix=time.time(),
        automatic_next_search=False,model_frozen=False,native_mechanism_accepted=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    if not 1<=a.workers<=8:p.error('workers must be 1 to 8')
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if a.prepare_only:d,o=prepare();initial(o);proposals('C',d,parents(initial(o)['candidates'],d['anchors']))
            else:run(a.workers)
        except Exception as exc:
            write(OUT/'status.json',dict(status='ERROR_REVIEW_REQUIRED',error=repr(exc),updated_unix=time.time()));raise
