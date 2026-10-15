#!/usr/bin/env python3
"""Bounded nightly batches. Scientific decisions occur between persisted batches."""
from pathlib import Path
import argparse,copy,fcntl,json,os,pickle,shutil,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts import run_topic4_multievent_distribution_v2_1 as engine
from scripts import run_topic4_observable_loss_physical_pilot as dispatch
from scripts import run_topic4_xy_research as base
from src.topic4_xy_search import field_descriptor,audit_geometry,geometry_allowed
OUT=ROOT/'results/topic4_sef_hfo/nightly_central_workpoint'
SOURCE=ROOT/'results/topic4_sef_hfo/contact_native_integrated_pilot'
PAIRS=[(6101,842901),(6102,842901)]

def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp')
    q.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n');q.replace(p)

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'design.json').exists():return base.read(OUT/'design.json')
    refs=base.read(SOURCE/'confirmation_scores.json')['candidates']
    ids=['tshape_anchor3_B_minus','v2_1_pop1_de_b_002']
    anchors=[next(r for r in refs if r['candidate_id']==cid) for cid in ids]
    positions={}
    for topo,dyn in PAIRS:
        p=Path(anchors[0]['units'][f'topo_{topo}_dyn_{dyn}']['worker_path']).with_suffix('.npz')
        with np.load(p) as z:positions[topo]=np.array(z['positions_E']);xy=np.array(z['contact_xy_mm']);names=z['contact_names'].astype(str)
    means=[xy[np.char.startswith(names,s)].mean(0) for s in ['SCL','ICL']]
    center=np.mean(means,axis=0)
    rng=np.random.default_rng(20260908301);geometry=[];attempts=[]
    for g,angle in enumerate([-45.,45.,90.],1):
        for attempt in range(1000):
            mid=center+rng.normal(0,[.8,.8]);theta=float(angle+rng.uniform(-12,12));sep=float(rng.uniform(5.5,7.5))
            d=np.array([np.cos(np.deg2rad(theta)),np.sin(np.deg2rad(theta))]);centers=mid+np.array([-.5,.5])[:,None]*sep*d
            valid=all(geometry_allowed(centers,pos,domain='whole_sheet') for pos in positions.values())
            attempts.append(dict(geometry=g,attempt=attempt,centers=centers.tolist(),valid=valid))
            if valid:
                geometry.append(dict(index=g,midpoint_mm=mid.tolist(),separation_mm=sep,angle_deg=theta,field=field_descriptor(centers)));break
        else:raise RuntimeError('central proposal pool exhausted')
    cc=[]
    for ai,a in enumerate(anchors,1):
        for g in geometry:
            c=copy.deepcopy(a['candidate']);c.update(candidate_id=f'central_A_anchor{ai}_geom{g["index"]}',domain='whole_sheet',
                node_field=g['field'],parent_candidate_id=a['candidate_id'],proposal='shaft_balanced_central_priority_random_geometry',
                geometry=audit_geometry(positions[6101],g['field']['centers_mm'],1499))
            for k in ['de_plan_record','parent_target_candidate_id','sobol_draw_index']:c.pop(k,None)
            cc.append(c)
    now=time.time()
    design=dict(version='nightly_central_workpoint_v1',started_unix=now,dispatch_deadline_unix=now+8*3600,
         maximum_new_units=60,maximum_workers=6,duration_ms=24000,pairs=PAIRS,anchors=ids,
         center_definition='Equal-weight centroid of SCL and ICL contact centroids; descriptive proposal center, not a truth label or hard gate.',
         shaft_balanced_center_mm=center.tolist(),contact_names=names.tolist(),contact_xy_mm=xy.tolist(),
         geometry_prior='Central priority in first six candidates; historical off-center controls retained. Later batches may use whole-sheet proposals.',
         initial_proposals=cc,geometry_draws=attempts,
         objective=str(SOURCE/'objective.pkl'),objective_sha256=base.sha(SOURCE/'objective.pkl'),
         physical_worker='Original repaired multidimensional worker. No new AMPA filter, smoothing, axis target, or prescribed TA/TB route in batch A.',
         review='Fixed five-term training loss for search only; compare per-observable distributions, all-contact event-scale envelopes, TA/TB-organized events and full native movies before any provisional acceptance.',
         stages='A:12 new geometry units. Review plus at most B:12 additional residual-guided units. Reserve up to36 units for early noise replay, selected one-parameter paired responses and additional topology/noise contrasts; do not advance on a score alone.',
         acceptance='Agent scientific review required between stages; machine labels do not establish propagation recovery. Human final visual review remains pending. Negative results may exhaust the bounded task without an accepted working point.',
         stop='No new dispatch after 8h or 60 new units; drain running jobs and analyze. No substrate freeze or Fig5.')
    write(OUT/'design.json',design);write(OUT/'reference_controls.json',dict(candidates=refs))
    write(OUT/'dispatch_ledger.json',dict(units=[]))
    build_batch('A',cc,PAIRS)
    return design

def build_batch(name,candidates,pairs):
    folder=OUT/'batches'/name;folder.mkdir(parents=True,exist_ok=True)
    cp=folder/'execution_config.json';mp=folder/'candidate_manifest.json';sp=folder/'runtime_snapshot.json'
    if cp.exists():
        if base.read(mp)['candidates']!=candidates:raise RuntimeError('persisted candidates differ')
        return folder,cp,mp,sp
    cfg=base.read(SOURCE/'execution/confirmation/execution_config.json')
    cfg.update(output_root=str(folder),candidate_manifest=str(mp))
    cfg['search']['fit_network_seeds']=sorted({t for t,n in pairs});cfg['search']['dynamics_seeds']=sorted({n for t,n in pairs})
    for topo,_ in pairs:
        if str(topo) not in cfg['corrected_networks']:cfg['corrected_networks'][str(topo)]=base.network_record(topo)
    write(cp,cfg);write(mp,dict(candidates=candidates,pairs=pairs,config_sha256=base.sha(cp),frozen_before_dispatch=True))
    from scripts import run_topic4_multidimensional_worker as worker
    hashes=base.read(SOURCE/'execution/confirmation/runtime_snapshot.json')['source_hashes']
    # A new batch records actual current dependencies; historical outputs remain unchanged.
    hashes={p:base.sha(ROOT/p) for p in hashes};hashes.update(worker._runtime_provenance(None)['runtime_module_sha256'])
    write(sp,dict(source_hashes=hashes,input_hashes={str(cp):base.sha(cp),str(mp):base.sha(mp)},identity_kind='nightly_bounded_batch'))
    return folder,cp,mp,sp

def run_batch(name,maximum_workers):
    d=base.read(OUT/'design.json');folder=OUT/'batches'/name
    cp=folder/'execution_config.json';mp=folder/'candidate_manifest.json';sp=folder/'runtime_snapshot.json'
    manifest=base.read(mp);jobs=[(c['candidate_id'],t,n) for c in manifest['candidates'] for t,n in manifest['pairs']]
    for k in ['workers','run_logs','resource_logs']:(folder/k).mkdir(exist_ok=True)
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            args=(proc/'cmdline').read_bytes().split(b'\0')
            if str(cp).encode() in args and any(a.endswith(b'run_topic4_multidimensional_worker.py') for a in args):raise RuntimeError(f'orphan worker {proc.name} remains active')
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    complete=[j for j in jobs if engine._complete(folder/'workers'/f'{engine._stem(*j)}.json',sp)]
    pending=[j for j in jobs if j not in complete];active={};failures=[];budget_stop=False
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    while pending or active:
        for job,(proc,stream,peaks) in list(active.items()):
            sample=engine._memory_sample(proc.pid)
            for k in peaks:peaks[k]=max(peaks[k],sample[k])
            engine._append_jsonl(folder/'resource_logs'/f'{engine._stem(*job)}.jsonl',dict(unix=time.time(),**sample))
            if proc.poll() is None:continue
            stream.close();del active[job]
            write(folder/'resource_logs'/f'{engine._stem(*job)}_summary.json',dict(exit_code=proc.returncode,peak_process_tree=peaks))
            if proc.returncode or not engine._complete(folder/'workers'/f'{engine._stem(*job)}.json',sp):failures.append(dict(job=job,exit_code=proc.returncode))
            else:complete.append(job)
        ledger=base.read(OUT/'dispatch_ledger.json');own=set()
        for proc,_,_ in active.values():own.update(engine._process_tree(proc.pid))
        external=dispatch.external_workers(own);reserve=engine._other_snn_reserve_gib(own)
        allowance=max(0,int((base.available_gib()-40-reserve-18*len(active))/18))
        slots=max(0,min(maximum_workers-len(active),8-len(external)-len(active),allowance,len(pending)))
        if external and not active:slots=0
        budget_stop=time.time()>=d['dispatch_deadline_unix'] or len(ledger['units'])>=d['maximum_new_units']
        if budget_stop or failures:slots=0
        if shutil.disk_usage(OUT).free<35*1024**3:failures.append(dict(reason='free disk below 35 GiB'));slots=0
        for _ in range(slots):
            job=pending.pop(0);stem=engine._stem(*job);out=folder/'workers'/f'{stem}.json'
            if len(ledger['units'])>=d['maximum_new_units']:pending.insert(0,job);break
            stream=(folder/'run_logs'/f'{stem}.log').open('a')
            cmd=['/usr/bin/prlimit',f'--as={18*1024**3}','--',engine.PYTHON,str(engine.WORKER),'--config',str(cp),'--candidate-id',job[0],
                '--seed',str(job[1]),'--topology-seed',str(job[1]),'--dynamics-seed',str(job[2]),'--expected-commit',commit,
                '--runtime-manifest',str(sp),'--artifact-root',str(engine.ARTIFACT_ROOT),'--out-json',str(out),'--out-npz',str(out.with_suffix('.npz'))]
            proc=subprocess.Popen(cmd,cwd=ROOT,env=engine.ENV,stdout=stream,stderr=subprocess.STDOUT)
            active[job]=(proc,stream,dict(rss_bytes=0,pss_bytes=0,vms_bytes=0))
            ledger['units'].append(dict(batch=name,job=job,pid=proc.pid,dispatch_unix=time.time()));write(OUT/'dispatch_ledger.json',ledger)
        status='FAILURE_DRAINING' if failures else ('BUDGET_DRAINING' if budget_stop else ('WAITING_FOR_EXTERNAL_BATCH' if external and not active else 'RUNNING'))
        write(OUT/'status.json',dict(status=status,batch=name,total=len(jobs),complete=len(complete),running=len(active),pending=len(pending),
             new_units_dispatched=len(ledger['units']),deadline_unix=d['dispatch_deadline_unix'],external_workers=external,
             active=[dict(candidate_id=j[0],topology_seed=j[1],dynamics_seed=j[2],pid=v[0].pid) for j,v in active.items()],failures=failures,updated_unix=time.time()))
        if not active and (failures or budget_stop):break
        if pending or active:time.sleep(10)
    write(folder/'completion.json',dict(complete=[list(j) for j in complete],pending=[list(j) for j in pending],failures=failures,budget_stop=budget_stop))
    subprocess.run([engine.PYTHON,str(ROOT/'scripts/analyze_topic4_nightly_workpoint.py'),'--batch',name],cwd=ROOT,env=engine.ENV,check=True)
    write(OUT/'status.json',dict(status='BATCH_READY_FOR_SCIENTIFIC_REVIEW',batch=name,complete=len(complete),total=len(jobs),pending=len(pending),failures=failures,updated_unix=time.time(),new_units_dispatched=len(base.read(OUT/'dispatch_ledger.json')['units'])))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--batch',default='A');ap.add_argument('--workers',type=int,default=6);a=ap.parse_args()
    if not 1<=a.workers<=6:raise ValueError('workers must be 1..6')
    prepare()
    if a.prepare_only:print(OUT);return
    with open(OUT/'controller.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run_batch(a.batch,a.workers)
        except Exception as exc:
            write(OUT/'controller_error.json',dict(batch=a.batch,error=repr(exc),unix=time.time()));raise
if __name__=='__main__':main()
