"""Bounded dependency queue. An OS lock covers the entire live task, not just dispatch."""
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import time
import numpy as np
from .train import RunConfig,run_cell,source_digest,atomic_json,ROOT,tag


def acquire_lock(path):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);f=open(p,'a+')
    try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:f.close();return None
    f.seek(0);f.truncate();f.write(json.dumps(dict(pid=os.getpid(),host=socket.gethostname(),start=time.time())));f.flush()
    return f


def make_recipe(cards,out_path,device='cpu'):
    rows=[json.loads(Path(p).read_text()) for p in cards]
    if {r['config']['stage'] for r in rows}!={'inner0','inner1'}:raise ValueError('both temporal INNER origins required')
    if any(r['status']!='COMPLETE' or r['source_digest']!=source_digest()[0] for r in rows):raise ValueError('unfinished or changed INNER')
    keys=('subject','protocol','inputs','family','arm','history_hours','crossview','old_targets','lr','dynamics_lr','seed','split_seed','train_paths','batch_size')
    for k in keys:
        if len({str(r['config'][k]) for r in rows})!=1:raise ValueError(f'INNER recipes differ: {k}')
    updates=int(np.median([r['selected_updates'] for r in rows]));drops=[]
    for r in rows:
        curve=r['curve'];d=[]
        for prev,cur in zip(curve,curve[1:]):
            if cur['lrs'][0]<prev['lrs'][0]:d.append(int(prev['update']))
        drops.append(d)
    agreed=[int(np.median([d[i] for d in drops])) for i in range(min(map(len,drops)))]
    both_learned=all(int(r['selected_updates'])>0 for r in rows)
    both_plateau=all(r['stop_reason']=='plateau' and int((r.get('plateau') or {}).get('drops',0))>=2 for r in rows)
    recipe=dict(status='FROZEN',source_digest=source_digest()[0],config=rows[0]['config'],updates=updates,
        lr_drop_updates=[u for u in agreed if 0<u<updates],rule='median selected updates of two temporal INNER origins; median common LR milestones; no OUTER scores',
        inputs=[dict(path=str(p),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest(),selected_updates=r['selected_updates'],stop_reason=r['stop_reason']) for p,r in zip(cards,rows)],
        learned_state_eligible=both_learned,
        training_adequacy=('both temporal INNER origins reached a two-stage learning-rate patience plateau; local optimization evidence only'
                           if both_plateau else 'unresolved; both temporal INNER origins did not satisfy the registered plateau record'))
    if rows[0]['config']['arm']=='state':
        from .frozen import load_selected,fit_reference
        from .train import evaluate
        scores=[];grid=(.5,2.,8.,24.)
        for path in cards:
            model,prep,cfg,_=load_selected(Path(path).with_name('selected.pt'),device)
            prep.frozen_query_cache={};ref=fit_reference(model,prep,cfg);vv=[]
            for tau in grid:
                ref['tau_hours']=tau;vv.append(evaluate(model,prep,cfg,'inner',rule='RELAX',reference=ref)['selection'])
            scores.append(vv)
        ranks=np.argsort(np.argsort(np.array(scores),axis=1),axis=1).mean(0)
        recipe['relax_tau_hours']=grid[int(np.argmin(ranks))]
        recipe['relax_inner_scores']=scores;recipe['relax_selection_rule']='mean rank over two temporal INNER origins, fixed grid 0.5/2/8/24 hours'
    atomic_json(recipe,out_path);return recipe


def build_plan(path,root=ROOT,quick=False):
    root=Path(root);tasks=[];priority=0
    def add(kind,**kw):
        nonlocal priority
        priority+=1;task=dict(id=f't{priority:03d}',kind=kind,priority=priority,depends=[],**kw)
        tasks.append(task);return task['id']
    # First seed forms a complete chain. More patients/capacity are not silently added.
    arms=[('P_stats','state'),('P_marks','state'),('P_marks','marked_history'),
          ('P_marks','fixed_marked_history'),('P_stats','constant_state')]
    for inputs,arm in arms:
        dep=[];cards=[]
        for stage in ('inner0','inner1'):
            cfg=RunConfig(inputs=inputs,arm=arm,stage=stage,out_dir=str(root/'runs'))
            if quick:
                cfg.batch_size=4;cfg.microbatch=4;cfg.train_paths=2;cfg.eval_paths=4;cfg.max_updates=2;cfg.extended_updates=2;cfg.eval_every=2;cfg.eval_stride=180
            c=asdict(cfg);tid=add('train',config=c);dep.append(tid);cards.append(str(root/'runs'/tag(cfg)/'card.json'))
        recipe=str(root/'recipes'/f'{inputs}_{arm}.json');rid=add('recipe',cards=cards,output=recipe);tasks[-1]['depends']=dep
        for seed in ((20260906,) if quick else (20260906,20260907,20260908)):
            cfg.stage='outer';cfg.seed=seed;cfg.recipe_path=recipe
            # Keep seed 1 complete chain ahead of repeat fits.
            tid=add('train',config=asdict(cfg));tasks[-1]['depends']=[rid]
            selected=str(root/'runs'/tag(cfg)/'selected.pt')
            bid=add('bundle',selected=selected,output=str(root/'frozen'/tag(cfg)),quick=quick);tasks[-1]['depends']=[tid]
    # One deterministic independent baseline bank suffices on this shared split.
    cfg.seed=20260906
    bid=add('baselines',config=asdict(cfg),output=str(root/'baselines'/'independent.json'))
    tasks[-1]['priority']=0
    if not quick:
        # One independent realization per world in this window; replication and
        # power remain registered extensions, not a license to interpret a null.
        for scenario,realization in (('morphology',901),('identity',902),('zero',903)):
            packets=str(root/'instruments'/'packets')
            gid=add('synthetic_data',output=packets,scenario=scenario,realization=realization);tasks[-1]['priority']=5
            for inputs in ('P_stats','P_marks'):
                cfg0=RunConfig(subject=f'synthetic_{scenario}_{realization}',inputs=inputs,stage='inner0',packets_root=packets,out_dir=str(root/'instruments'/'runs'))
                tid=add('train',config=asdict(cfg0));tasks[-1]['depends']=[gid];tasks[-1]['priority']=20 if scenario=='morphology' else 500
                sid=add('synthetic_score',selected=str(Path(cfg0.out_dir)/tag(cfg0)/'selected.pt'),output=str(root/'instruments'/f'{scenario}_{inputs}_test.pt'))
                tasks[-1]['depends']=[tid];tasks[-1]['priority']=21 if scenario=='morphology' else 501
    if not quick:
        for inputs,arm in (('P_stats','state'),('P_marks','state'),('P_marks','marked_history')):
            cfg0=RunConfig(inputs=inputs,arm=arm,stage='sid',protocol='S-ID',out_dir=str(root/'runs'))
            tid=add('train',config=asdict(cfg0));tasks[-1]['priority']=200
            bid=add('bundle',selected=str(Path(cfg0.out_dir)/tag(cfg0)/'selected.pt'),output=str(root/'frozen'/tag(cfg0)),quick=False)
            tasks[-1]['depends']=[tid];tasks[-1]['priority']=201
    # Priority ensures initial input pair and reference finish before optimization repeats.
    for t in tasks:
        seed=t.get('config',{}).get('seed',20260906)
        if seed!=20260906:t['priority']+=1000
        if t['kind']=='bundle' and 'seed20260906' not in t['selected']:t['priority']+=1000
    plan=dict(version='v0312',scope='D-delayed development; one-patient chain before extension',source_digest=source_digest()[0],
              root=str(root),tasks=tasks,created_epoch=time.time(),workers_per_gpu=1,cpu_workers=2,consumer_backend='cpu',reserved_headroom_gib=6,
              gpu_concurrency_reason='one training process per GPU; concurrent same-GPU FP64 Lyapunov solves are not admitted',
              deadline_policy='PAUSED preserves optimizer and queue; no budget/clock relabel as convergence',
              non_admitted=['D-local realtime','nonlinearity mechanism','seizure risk','S-B without raw outcome manifest'])
    atomic_json(plan,path);return plan


def run_worker(plan_path,device,deadline,allow_unready=False,slot=0):
    import torch
    plan=json.loads(Path(plan_path).read_text());root=Path(plan['root']);state=root/'queue_state'
    if plan['source_digest']!=source_digest()[0]:raise ValueError('plan/source changed; make a new plan explicitly')
    if not allow_unready:
        ready=json.loads((root/'readiness'/'admission.json').read_text())
        if ready.get('status')!='READY' or ready.get('source_digest')!=source_digest()[0]:raise ValueError('readiness admission missing or stale')
    if not 0<=slot<(plan.get('cpu_workers',2) if device=='cpu' else plan['workers_per_gpu']):raise ValueError('worker slot exceeds admitted concurrency')
    worker=acquire_lock(Path(ROOT)/'resource_locks'/f'gpu_{device.replace(":","_")}_slot{slot}.lock')
    if worker is None:raise RuntimeError('another worker owns this GPU in this queue')
    completed=[]
    try:
        while time.time()<deadline:
            if source_digest()[0]!=plan['source_digest']:raise RuntimeError('source changed during queue; freeze a new plan before continuing')
            chosen=None;live=False
            task_status={t['id']:(json.loads((state/(t['id']+'.json')).read_text()).get('status') if (state/(t['id']+'.json')).exists() else 'PENDING') for t in plan['tasks']}
            index={t['id']:t for t in plan['tasks']}
            def runnable_eventually(t):
                return task_status[t['id']]!='FAILED' and all(runnable_eventually(index[d]) for d in t['depends'])
            eligible=[t for t in plan['tasks'] if (t['kind']!='train' if device=='cpu' else t['kind']=='train')]
            for task in sorted(eligible,key=lambda t:t['priority']):
                path=state/(task['id']+'.json')
                existing=json.loads(path.read_text()) if path.exists() else {}
                if existing.get('status') in ('COMPLETE','FAILED'):continue
                if any(not (state/(d+'.json')).exists() or json.loads((state/(d+'.json')).read_text()).get('status')!='COMPLETE' for d in task['depends']):continue
                lock=acquire_lock(state/(task['id']+'.lock'))
                if lock is None:live=True;continue
                # Re-check after acquiring, because another worker may have just completed it.
                if path.exists() and json.loads(path.read_text()).get('status') in ('COMPLETE','FAILED'):
                    lock.close();continue
                chosen=(task,path,lock);break
            if chosen is None:
                pending=any(task_status[t['id']] not in ('COMPLETE','FAILED') and runnable_eventually(t) for t in eligible)
                if live or pending:time.sleep(min(5,max(0,deadline-time.time())));continue
                break
            task,path,lock=chosen
            if task['kind'] in ('train','bundle') and device.startswith('cuda') and torch.cuda.mem_get_info(device)[0]<8*2**30:
                lock.close();time.sleep(min(5,max(0,deadline-time.time())));continue
            atomic_json(dict(status='RUNNING',task=task['id'],device=device,pid=os.getpid(),started=time.time()),path)
            try:
                if task['kind']=='train':
                    cfg=RunConfig(**(task['config']|{'device':device}))
                    result=run_cell(cfg,deadline=deadline,progress=lambda row,action:atomic_json(dict(status='RUNNING',task=task['id'],device=device,pid=os.getpid(),heartbeat=time.time(),update=row['update']),path))
                elif task['kind']=='recipe':result=make_recipe(task['cards'],task['output'],device)|{'status':'COMPLETE'}
                elif task['kind']=='bundle':
                    from .frozen import run_bundle
                    result=run_bundle(task['selected'],task['output'],device,quick=task['quick'])
                elif task['kind']=='synthetic_data':
                    from .instruments import generate
                    result=generate(task['output'],task['realization'],task['scenario'])
                elif task['kind']=='synthetic_score':
                    from .instruments import score_untouched_future
                    result=score_untouched_future(task['selected'],task['output'],device)
                elif task['kind']=='baselines':
                    from .baselines import run_baselines
                    result=run_baselines(RunConfig(**(task['config']|{'device':'cpu'})),task['output'])
                else:raise ValueError(task['kind'])
                atomic_json(dict(status=result['status'],task=task['id'],device=device,finished=time.time(),source_digest=source_digest()[0]),path)
                completed.append(task['id'])
                if result['status']=='PAUSED':break
            except Exception as exc:
                import traceback
                atomic_json(dict(status='FAILED',task=task['id'],error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc(),device=device),path)
                if 'device-side assert' in str(exc):raise
                # Failed prerequisite blocks only its descendants; independent tasks continue.
            finally:
                lock.close()
                import gc;gc.collect()
                if device.startswith('cuda'):
                    try:torch.cuda.empty_cache()
                    except RuntimeError:pass  # Preserve the original poisoned-context exception.
        return dict(status='WINDOW_STOPPED',completed=completed,remaining=[t['id'] for t in plan['tasks'] if not (state/(t['id']+'.json')).exists() or json.loads((state/(t['id']+'.json')).read_text()).get('status')!='COMPLETE'])
    finally:worker.close()
