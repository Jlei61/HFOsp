"""Single-network/single-noise dense responses and label-free batched BO.

Physical simulation/observation remain the frozen September 14 executor.
All optimizer feedback comes from the separately frozen label-free scorer.
"""
from pathlib import Path
import argparse, copy, fcntl, json, os, pickle, shutil, subprocess, sys, time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
import numpy as np
from scripts import run_topic4_three_observable_bo as old
from src.topic4_three_observable_objective import GROUPS
rt=old.rt
SOURCE=old.OUT
AUDIT=SOURCE.parent/'label_free_objective_audit_20260915'
OUT=SOURCE.parent/'label_free_dense_parameter_search_20260915'
SCRIPT=Path(__file__).resolve()
TOPOLOGY=2511
NOISE=847401
STAGE='search'
AXES=['左核X (mm)','左核Y (mm)','右核X (mm)','右核Y (mm)','核向外EE权重倍数','全局EE轴偏移 (°)']
GPU_ENV={**rt.ENV,'CUDA_VISIBLE_DEVICES':'0,1'}

def candidates():
    return {p.stem:rt.read(p) for p in (OUT/'candidates').glob('*.json')}

def validate(c,anchor):
    assert old.background(c)==old.background(anchor),'fixed physics changed'
    x=old.vector(c);lo,hi=old.bounds()
    assert np.all(x>=lo-1e-10) and np.all(x<=hi+1e-10)
    xy=np.array(c['centers_mm']);r=np.array(c['radii_mm'])
    assert np.all(xy>=r[:,None]) and np.all(xy<=20-r[:,None])
    assert np.linalg.norm(xy[0]-xy[1])>r.sum()

def prepare():
    if (OUT/'plan.json').exists():return rt.read(OUT/'plan.json')
    for d in ['candidates','scores','logs','analysis','proposals','global_graph_cache','confirmation']:(OUT/d).mkdir(parents=True,exist_ok=True)
    p=rt.read(SOURCE/'plan.json')
    for path,h in p['source_snapshot'].items():assert rt.sha(path)==h,path
    anchor=rt.read(SOURCE/'candidates/g1_axis3_plus.json');rows=[];warm=[]
    for sp in sorted((AUDIT/'per_unit').glob('*.json')):
        row=rt.read(sp)
        if row['stage'] not in ['initial','adaptive'] or row['topology']!=TOPOLOGY or row['noise']!=NOISE:continue
        c=rt.read(SOURCE/'candidates'/f"{row['candidate']}.json");validate(c,anchor)
        c.update(search_role='historical_single_noise',search_source=str(SOURCE));rows.append(c);warm.append(row)
    assert len(rows)==32
    # A response line changes exactly one scalar, against the SAME anchor.
    lo,hi=old.bounds();x0=old.vector(anchor);lines=[];new=[]
    for j in range(6):
        ids=[];values=np.linspace(lo[j],hi[j],13)
        for k,value in enumerate(values):
            x=x0.copy();x[j]=value
            match=next((c for c in rows if np.allclose(old.vector(c),x,rtol=0,atol=1e-10)),None)
            if match is None:
                cid=f'dense_axis{j}_{k:02d}'
                c=old.candidate_at(anchor,x,cid,'dense_single_axis',j)
                c.update(search_role='dense_single_axis',display_name=f'{AXES[j]}={value:.4g}')
                validate(c,anchor);rows.append(c);new.append(cid);match=c
            ids.append(match['id'])
        lines.append(dict(axis=j,label=AXES[j],values=values.tolist(),ids=ids,anchor_id=anchor['id']))
    # Interleave axes so partial curves grow together. BO proposals get priority.
    dense_order=[]
    for k in [6,3,9,0,12,1,11,2,10,4,8,5,7]:
        for line in lines:
            cid=line['ids'][k]
            if cid in new and cid not in dense_order:dense_order.append(cid)
    for c in rows:rt.write(OUT/'candidates'/f"{c['id']}.json",c)
    for row in warm:
        src=Path(row['source']);r=rt.read(src);c=next(c for c in rows if c['id']==row['candidate'])
        applied=rt.read(src.parents[1]/'applied_physics.json')
        assert old.physics(c)==old.physics(applied['candidate'])
        assert r['actual_duration_ms']==60000 and r['job']['topology_seed']==TOPOLOGY and r['job']['dynamics_seed']==NOISE
        assert rt.sha(src.with_suffix('.npz'))==r['arrays_sha256']
        dest=OUT/STAGE/'units'/c['id']/f'{TOPOLOGY}_{NOISE}'
        dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():dest.symlink_to(src.parents[1].resolve())
        groups=row['groups']
        rt.write(OUT/'scores'/f"{c['id']}.json",dict(candidate=c['id'],J=row['label_free_J'],N=row['N'],groups=groups,components=[groups[g]['scaled'] for g in GROUPS],status='SCORABLE',source=str(src),source_sha256=rt.sha(src),reused=True,vector=old.vector(c).tolist(),topology=TOPOLOGY,noise=NOISE,objective_sha256=rt.sha(AUDIT/'label_free_objective.pkl')))
    for source in (SOURCE/'global_graph_cache').glob('*.pkl'):
        target=OUT/'global_graph_cache'/source.name
        if not target.exists():target.symlink_to(source.resolve())
    shutil.copy2(AUDIT/'label_free_objective.pkl',OUT/'analysis/training_objective.pkl')
    shutil.copy2(AUDIT/'label_free_scale_calibration.json',OUT/'analysis/scale_calibration.json')
    with (OUT/'analysis/training_objective.pkl').open('rb') as f:obj=pickle.load(f)
    assert not any(hasattr(obj,k) for k in ['km','proportions','mode_means'])
    p.update(schema='topic4.label_free_dense_single_seed.v1',authorization='2026-09-15 user: 无标签下一版并行运行；单网络单噪声；加密参数响应',
        reference_id=anchor['id'],historical_reference_id='bridge_circle_out125_xminus075',topology_seed=TOPOLOGY,topology_seeds=[TOPOLOGY],seeds=[NOISE],
        confirmation_seeds=None,candidates=rows,initial_ids=[c['id'] for c in rows],warm_ids=[r['candidate'] for r in warm],dense_ids=dense_order,response_lines=lines,
        optimizer=dict(method='Matérn-5/2 GP; CUDA autograd multistart qLogEI; six local plus two global uncertainty proposals',batches=4,batch_size=8,numerical_nugget_relative=1e-5,noise_interpretation='one fixed noise realization; no empirical between-seed variance estimate',batch_update='freeze observed training table; wait all eight outcomes before next batch; dense responses run concurrently'),
        budget=dict(reused_runs=32,dense_new_runs=len(new),adaptive_new_runs=32,max_new_runs=len(new)+32,additional_seed_runs=0),
        resources=dict(max_workers=48,per_tree_limit_GiB=18,projected_per_tree_GiB=6,min_available_GiB=40,launch_interval_seconds=2,cpu_dispatch_ceiling_percent=94),
        objective=dict(labels_used=False,artifact=str(OUT/'analysis/training_objective.pkl'),sha256=rt.sha(OUT/'analysis/training_objective.pkl'),weights=[1/3]*3,negative_values_clipped=False,minimum_events=16),
        contract='Only single-noise label-free scores rank candidates. Historical candidate pool was developed with labels; new selection is label-free. Modes, raw envelopes, field and rotation remain diagnostic. Fixed seed is not proof of identical graph across EE-angle changes or identical noise innovations across changed masks. No confirmation, model freeze or Fig5 automatic launch.',created_unix=time.time())
    rt.write(OUT/'plan.json',p);rt.write(OUT/'confirmation/frozen_networks.json',dict(replication_networks={}))
    rt.write(OUT/'analysis/objective_frozen.json',p['objective'])
    rt.write(OUT/'status.json',dict(status='PREPARED',warm_complete=32,planned_new=len(new)+32,active=[],time=time.time()))
    return p

def result(cid):return OUT/STAGE/'units'/cid/f'{TOPOLOGY}_{NOISE}'/'workers/trajectory.json'

def worker(cid):
    p=rt.read(OUT/'plan.json');c=rt.read(OUT/'candidates'/f'{cid}.json');validate(c,rt.read(OUT/'candidates'/f"{p['reference_id']}.json"))
    old.physical.OUT=OUT;old.physical.configure()
    old.physical.run.worker(STAGE,cid,TOPOLOGY,NOISE,60000.)

def load_times(source):
    r=rt.read(source)
    with np.load(source.with_suffix('.npz')) as z:
        times=z['centroid_ms'];names=z['contact_names'].tolist()
        ids=np.array([int(i) for i in z['primary_event_indices'] if r['events'][i]['window_ms'][0]>=1500 and r['events'][i]['window_ms'][1]<=r['actual_duration_ms']],int)
    return r,times[ids],names

def gpu_device():
    import torch
    if not torch.cuda.is_available():return 'cpu'
    return 'cuda:'+str(max(range(torch.cuda.device_count()),key=lambda i:torch.cuda.mem_get_info(i)[0]))

def score_ready():
    from scripts.analyze_topic4_three_observable_bo import audit_applied
    p=rt.read(OUT/'plan.json');path=Path(p['objective']['artifact'])
    assert rt.sha(path)==p['objective']['sha256']
    with path.open('rb') as f:obj=pickle.load(f)
    device=gpu_device();updated=0
    for cid,c in candidates().items():
        dest=OUT/'scores'/f'{cid}.json';src=result(cid)
        if dest.exists() or not old.valid_complete(src):continue
        r,t,names=load_times(src);assert names==obj.names
        applied=rt.read(src.parents[1]/'applied_physics.json')
        assert old.physics(applied['candidate'])==old.physics(c)
        audit_applied(applied,c)
        assert rt.sha(src.with_suffix('.npz'))==r['arrays_sha256']
        assert r['job']['topology_seed']==TOPOLOGY and r['job']['dynamics_seed']==NOISE
        sc=obj.score(t,device=device)
        if r['physical_status']=='RUNAWAY':sc.update(status='PHYSICAL_RUNAWAY',J=None)
        sc.update(candidate=cid,source=str(src),source_sha256=rt.sha(src),vector=old.vector(c).tolist(),topology=TOPOLOGY,noise=NOISE,reused=False,
            components=[sc['groups'][g]['scaled'] for g in GROUPS] if sc['J'] is not None else None,objective_sha256=p['objective']['sha256'],time=time.time())
        rt.write(dest,sc);updated+=1
    return updated

def score_rows():return [rt.read(p) for p in sorted((OUT/'scores').glob('*.json'))]

def propose(batch):
    import torch
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.fit import fit_gpytorch_mll
    from botorch.acquisition.logei import qLogExpectedImprovement
    from botorch.sampling.normal import SobolQMCNormalSampler
    from botorch.optim import optimize_acqf
    from gpytorch.kernels import ScaleKernel,MaternKernel
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from scipy.stats import qmc
    dest=OUT/'proposals'/f'batch_{batch:02d}.json'
    if dest.exists():return rt.read(dest)
    seed=2026091500+batch;torch.manual_seed(seed);torch.set_num_threads(2)
    input_path=OUT/'proposals'/f'input_{batch:02d}.json'
    if input_path.exists():frozen=rt.read(input_path)
    else:
        frozen=dict(rows=score_rows(),known_vectors=[old.vector(c).tolist() for c in candidates().values()],time=time.time())
        rt.write(input_path,frozen)
    rows=frozen['rows'];good=[r for r in rows if r['J'] is not None];assert len(good)>=8
    lo,hi=old.bounds();device=gpu_device()
    X=torch.tensor((np.array([r['vector'] for r in good])-lo)/(hi-lo),dtype=torch.float64,device=device)
    Y=torch.tensor([[-r['J']] for r in good],dtype=torch.float64,device=device)
    # Numerical regularization of a deterministic single-realization surrogate;
    # this is NOT an estimate of biological or noise-seed variation.
    Yvar=torch.full_like(Y,max(float(Y.var().item()),1e-12)*1e-5)
    kernel=ScaleKernel(MaternKernel(nu=2.5,ard_num_dims=6)).to(device=device,dtype=torch.float64)
    model=SingleTaskGP(X,Y,train_Yvar=Yvar,covar_module=kernel,outcome_transform=Standardize(m=1)).to(device)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood,model),optimizer_kwargs={'options':{'maxiter':160}})
    best=int(Y[:,0].argmax());center=X[best];halfwidth=.25
    if batch>1:
        prev=rt.read(OUT/'proposals'/f'batch_{batch-1:02d}.json')
        feedback=[r for r in rows if r['candidate'] in prev['ids']]
        assert len(feedback)==8
        improvement=any(r['J'] is not None and r['J']<prev['best_observed_J'] for r in feedback)
        halfwidth=min(.5,prev['halfwidth']*1.25) if improvement else max(.125,prev['halfwidth']*.75)
    bounds=torch.stack([(center-halfwidth).clamp_min(0),(center+halfwidth).clamp_max(1)])
    # Reserve every registered dense point, avoiding redundant launches. Using
    # all pending points in qEI would be costly; uniqueness is enforced below.
    sampler=SobolQMCNormalSampler(sample_shape=torch.Size([64]),seed=seed)
    acq=qLogExpectedImprovement(model,best_f=Y.max(),sampler=sampler)
    proposed,acqval=optimize_acqf(acq,bounds=bounds,q=6,num_restarts=8,raw_samples=128,sequential=True,options={'maxiter':80,'batch_limit':2})
    selected=proposed.detach().cpu().numpy().tolist()
    pool=qmc.Sobol(6,scramble=True,seed=seed+100).random_base2(12)
    with torch.no_grad():std=model.posterior(torch.tensor(pool,device=device,dtype=torch.float64)).variance[:,0].sqrt().cpu().numpy()
    used=(np.array(frozen['known_vectors'])-lo)/(hi-lo)
    actual=[];kinds=[]
    def novel(z):return np.min(np.linalg.norm(np.vstack([used,np.asarray(actual).reshape(-1,6)])-z,axis=1))>1e-5
    for z in selected:
        if novel(z):actual.append(z);kinds.append('CUDA_autograd_qLogEI')
    for i in np.argsort(-std):
        if len(actual)>=8:break
        if novel(pool[i]):actual.append(pool[i].tolist());kinds.append('global_posterior_uncertainty')
    assert len(actual)==8
    p=rt.read(OUT/'plan.json');anchor=rt.read(OUT/'candidates'/f"{p['reference_id']}.json")
    ids=[];vectors=lo+np.array(actual)*(hi-lo)
    for k,x in enumerate(vectors):
        cid=f'lf_bo{batch:02d}_{k+1:02d}';c=old.candidate_at(anchor,x,cid,'label_free_adaptive')
        c.update(search_role='label_free_adaptive',display_name=f'无标签优化{batch}-{k+1}')
        validate(c,anchor);rt.write(OUT/'candidates'/f'{cid}.json',c);ids.append(cid)
    checkpoint=OUT/'proposals'/f'gp_{batch:02d}.pt'
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},X=X.cpu(),Y=Y.cpu(),Yvar=Yvar.cpu()),checkpoint)
    # Only numeric label-free scores enter this durable proposal snapshot.
    snapshot=[{k:r.get(k) for k in ['candidate','J','N','status','vector','objective_sha256']} for r in rows]
    payload=dict(batch=batch,ids=ids,vectors=vectors.tolist(),kinds=kinds,training_snapshot=snapshot,best_observed_J=min(r['J'] for r in good),halfwidth=halfwidth,seed=seed,device=device,checkpoint=str(checkpoint),objective_sha256=p['objective']['sha256'],time=time.time())
    rt.write(dest,payload);return payload

def controller():
    import psutil
    p=prepare();active={};helper=None;failed=[];next_report=0;analysis_proc=None
    # Recover this controller's existing workers after an interrupted session.
    for process in psutil.process_iter(['pid','cmdline','create_time']):
        cmd=process.info['cmdline'] or []
        if str(SCRIPT) in cmd and 'worker' in cmd and '--candidate' in cmd:
            cid=cmd[cmd.index('--candidate')+1];active[process.pid]=(process,cid,None)
        if str(SCRIPT) in cmd and 'propose' in cmd and '--batch' in cmd:
            helper=(process,int(cmd[cmd.index('--batch')+1]),None)
    psutil.cpu_percent()
    def alive(proc):
        if isinstance(proc,subprocess.Popen):return proc.poll() is None
        try:return proc.is_running() and proc.status()!=psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:return False
    def launch(action,extra,logname,physical=False):
        log=(OUT/'logs'/logname).open('a')
        proc=subprocess.Popen([rt.PYTHON if physical else old.BO_PY,'-u',str(SCRIPT),action,*extra],cwd=ROOT,env=rt.ENV if physical else GPU_ENV,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        return proc,log
    while True:
        detail=[]
        for pid,(proc,cid,log) in list(active.items()):
            if not alive(proc):
                if log:log.close()
                active.pop(pid)
                if not old.valid_complete(result(cid)):failed.append(dict(candidate=cid,pid=pid,reason='WORKER_NO_VALID_RESULT'))
                continue
            try:
                q=psutil.Process(pid)
                rss=sum(v.memory_info().rss for v in [q]+q.children(recursive=True))/2**30
            except psutil.NoSuchProcess:continue
            detail.append(dict(pid=pid,candidate=cid,rss_GiB=rss))
            if rss>18:
                for child in reversed([q]+q.children(recursive=True)):child.terminate()
                failed.append(dict(candidate=cid,reason='PROCESS_TREE_MEMORY_LIMIT',rss_GiB=rss))
        if helper and not alive(helper[0]):
            if helper[2]:helper[2].close()
            if not (OUT/'proposals'/f'batch_{helper[1]:02d}.json').exists():failed.append(dict(reason='OPTIMIZER_FAILURE',batch=helper[1]))
            helper=None
        if analysis_proc and not alive(analysis_proc[0]):
            analysis_proc[1].close()
            if analysis_proc[0].returncode:failed.append(dict(reason='ANALYSIS_FAILURE',code=analysis_proc[0].returncode))
            analysis_proc=None
        if time.time()>=next_report and analysis_proc is None and not failed:
            analysis_proc=launch('analyze',[],f'analysis.log');next_report=time.time()+180
        props=[rt.read(f) for f in sorted((OUT/'proposals').glob('batch_*.json'))]
        scores={r['candidate']:r for r in score_rows()}
        if not failed and helper is None and len(props)<4 and (not props or all(cid in scores for cid in props[-1]['ids'])):
            b=len(props)+1;proc,log=launch('propose',['--batch',str(b)],f'optimizer_{b:02d}.log');helper=(proc,b,log)
        adaptive=[cid for prop in props for cid in prop['ids']]
        # Registered but not-yet-committed proposals are never dispatched.
        queue=adaptive+p['dense_ids']
        active_ids={v[1] for v in active.values()}
        pending=[cid for cid in queue if cid not in active_ids and not old.valid_complete(result(cid))]
        available=psutil.virtual_memory().available/2**30;cpu=psutil.cpu_percent()
        projected_growth=sum(max(0,6-r['rss_GiB']) for r in detail)
        if not failed and pending and len(active)<48 and available-projected_growth>46 and cpu<94:
            cid=pending.pop(0);proc,log=launch('worker',['--candidate',cid],cid+'.log',physical=True)
            active[proc.pid]=(proc,cid,log)
            with (OUT/'dispatch.jsonl').open('a') as f:f.write(json.dumps(dict(candidate=cid,pid=proc.pid,time=time.time(),available_GiB=available,cpu_percent=cpu))+'\n')
            detail.append(dict(pid=proc.pid,candidate=cid,rss_GiB=0))
        if available<24 and active:
            # Protect other work: stop new dispatch and only terminate our largest
            # growing worker in a real emergency; scientific/execution states differ.
            largest=max(detail,key=lambda r:r['rss_GiB']);q=psutil.Process(largest['pid'])
            for child in reversed([q]+q.children(recursive=True)):child.terminate()
            failed.append(dict(reason='SYSTEM_MEMORY_EMERGENCY',candidate=largest['candidate']))
        rt.write(OUT/'status.json',dict(status='DRAINING_FAILURE' if failed else 'RUNNING',active=detail,queued_registered=len(pending),complete=len(scores),new_complete=sum(not r.get('reused') for r in scores.values()),reused=32,adaptive_batches_proposed=len(props),optimizer_running=helper[1] if helper else None,analysis_running=analysis_proc is not None,available_GiB=available,cpu_percent=cpu,failures=failed,time=time.time()))
        if failed and not active and not helper and not analysis_proc:raise RuntimeError(json.dumps(failed))
        if len(props)==4 and not pending and not active and helper is None:
            if analysis_proc:
                analysis_proc[0].wait();analysis_proc[1].close();analysis_proc=None
            score_ready()
            from scripts.report_topic4_label_free_dense import report
            report()
            rt.write(OUT/'completion.json',dict(status='COMPLETE_PENDING_SCIENTIFIC_REVIEW',runs=len(score_rows()),new_runs=len(score_rows())-32,additional_seeds=0,time=time.time()))
            rt.write(OUT/'status.json',dict(status='COMPLETE_PENDING_SCIENTIFIC_REVIEW',active=[],complete=len(score_rows()),failures=[],time=time.time()))
            return
        time.sleep(2)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','worker','controller','propose','analyze']);ap.add_argument('--candidate');ap.add_argument('--batch',type=int);a=ap.parse_args()
    if a.action=='prepare':print(json.dumps(prepare()['budget']));return
    if a.action=='worker':worker(a.candidate);return
    if a.action=='propose':propose(a.batch);return
    if a.action=='analyze':
        with (OUT/'analysis/analyzer.lock').open('w') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:return
            score_ready()
            from scripts.report_topic4_label_free_dense import report
            report()
        return
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'controller.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:controller()
        except Exception as exc:rt.write(OUT/'failure.json',dict(error=repr(exc),time=time.time()));raise

if __name__=='__main__':main()
