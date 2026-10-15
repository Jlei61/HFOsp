"""Bounded wider conditional XY grids, followed by two feedback-driven 4D BO batches."""
from pathlib import Path
import argparse,copy,fcntl,json,os,pickle,shutil,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
import numpy as np
from scripts import run_topic4_xy_wide_response as prior
dense,old,rt=prior.dense,prior.old,prior.rt
BASE=prior.OUT.parent
OUT=BASE/'core_position_overnight_expansion_20260918'
SCRIPT=Path(__file__).resolve()
REPORT=ROOT/'scripts/report_topic4_xy_overnight_expansion.py'
REF=prior.REF
STAGE='xy_expansion'
TOPOLOGY,NOISE=2511,847401


def configure():
    dense.OUT=OUT;dense.STAGE=STAGE;dense.gpu_device=lambda:'cpu'


def result(cid):return OUT/STAGE/'units'/cid/f'{TOPOLOGY}_{NOISE}'/'workers/trajectory.json'
def read_scores():return [rt.read(p) for p in sorted((OUT/'scores').glob('*.json'))]
def candidates():return {p.stem:rt.read(p) for p in (OUT/'candidates').glob('*.json')}


def validate(c,ref,lo,hi):
    assert old.background(c)==old.background(ref)
    x=old.vector(c);np.testing.assert_array_equal(x[4:],old.vector(ref)[4:])
    assert np.all(x[:4]>=np.array(lo)-1e-10) and np.all(x[:4]<=np.array(hi)+1e-10)
    xy=np.array(c['centers_mm']);r=np.array(c['radii_mm'])
    assert np.all(xy>=r[:,None]) and np.all(xy<=20-r[:,None])
    assert np.linalg.norm(xy[0]-xy[1])>r.sum()


def prepare():
    if (OUT/'plan.json').exists():return rt.read(OUT/'plan.json')
    for d in ['candidates','scores','analysis','logs','proposals','global_graph_cache','confirmation','key_results']:(OUT/d).mkdir(parents=True,exist_ok=True)
    parent=rt.read(prior.OUT/'plan.json');ref=rt.read(prior.OUT/'candidates'/f'{REF}.json');x0=old.vector(ref)
    for path,h in parent['source_snapshot'].items():assert rt.sha(path)==h,path
    # Expand upward where the score minimum meets the boundary. Outward x extension
    # is asymmetric so both full core disks remain inside the actual 20 mm sheet.
    xoffset=[[-1.5,-.75,0,.75,1.5,2.25],[-2.25,-1.5,-.75,0,.75,1.5]]
    yoffset=[-1.5,-.75,0,.75,1.5,2.25,3.]
    lo=x0[:4]+np.array([-1.5,-1.5,-2.25,-1.5]);hi=x0[:4]+np.array([2.25,3.,1.5,3.])
    roots=[prior.OUT,BASE/'xy_block_joint_observation_pilot_20260916',BASE/'geometry_axis_joint_distribution_20260917',prior.SOURCE]
    registry={};reused=[];seen=[];src_records={}
    for folder in roots:
        for sp in sorted((folder/'scores').glob('*.json')):
            s=rt.read(sp);cid=s['candidate'];cp=folder/'candidates'/f'{cid}.json'
            if not cp.exists() or s.get('objective_sha256')!=parent['objective']['sha256']:continue
            c=rt.read(cp)
            try:validate(c,ref,lo,hi)
            except AssertionError:continue
            x=old.vector(c)
            if any(np.allclose(x,v,rtol=0,atol=1e-10) for v in seen):continue
            src=Path(s['source']);r=rt.read(src)
            if r['actual_duration_ms']!=60000 or r['job']['topology_seed']!=TOPOLOGY or r['job']['dynamics_seed']!=NOISE:continue
            assert old.physics(c)==old.physics(rt.read(src.parents[1]/'applied_physics.json')['candidate'])
            assert rt.sha(src)==s['source_sha256'] and rt.sha(src.with_suffix('.npz'))==r['arrays_sha256']
            dest=result(cid).parents[1];dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():dest.symlink_to(src.parents[1].resolve())
            s.update(reused=True,reuse_source=str(src));rt.write(OUT/'scores'/f'{cid}.json',s)
            registry[cid]=c;reused.append(cid);seen.append(x);src_records[cid]=str(folder)
    assert REF in registry and 'xywide_right_41' in registry and 'xy_left_20' in registry
    grids=[];new=[]
    for core in [0,1]:
        ids=[]
        for iy,dy in enumerate(yoffset):
            row=[]
            for ix,dx in enumerate(xoffset[core]):
                x=x0.copy();x[2*core:2*core+2]+=[dx,dy]
                match=next((c for c in registry.values() if np.allclose(old.vector(c),x,rtol=0,atol=1e-10)),None)
                if match is None:
                    cid=f'xyext_{"left" if core==0 else "right"}_{iy}{ix}'
                    match=old.candidate_at(ref,x,cid,'expanded_fixed_background_XY')
                    match.update(search_role='conditional_grid',xy_block_reference=REF)
                    validate(match,ref,lo,hi);registry[cid]=match;new.append(cid)
                row.append(match['id'])
            ids.append(row)
        grids.append(dict(core=core,x_offsets_mm=xoffset[core],y_offsets_mm=yoffset,ids=ids))
    assert len(set(c for g in grids for row in g['ids'] for c in row))==83
    assert len(new)<=34
    for cid,c in registry.items():rt.write(OUT/'candidates'/f'{cid}.json',c)
    # Prioritize newly extended right-core upper edge; dispatch is outcome-independent.
    new.sort(key=lambda cid:(0 if cid.startswith('xyext_right') else 1,
        np.linalg.norm(old.vector(registry[cid])[:4]-old.vector(registry['xywide_right_41'])[:4]),cid))
    for src in (prior.OUT/'global_graph_cache').glob('*.pkl'):
        dest=OUT/'global_graph_cache'/src.name
        if not dest.exists():dest.symlink_to(src.resolve())
    shutil.copy2(parent['objective']['artifact'],OUT/'analysis/training_objective.pkl')
    objective=copy.deepcopy(parent['objective']);objective['artifact']=str(OUT/'analysis/training_objective.pkl')
    assert rt.sha(objective['artifact'])==objective['sha256']
    p={k:copy.deepcopy(parent[k]) for k in ['physics','source_snapshot','analysis','runaway']}
    p.update(schema='topic4.xy_overnight_expansion.v1',authorization='2026-09-18 user: 扩大范围的优化结果怎么样了，今晚继续扩大一下',
        reference_id=REF,source=str(prior.OUT),topology_seed=TOPOLOGY,seeds=[NOISE],duration_ms=60000.,
        objective=objective,candidates=list(registry.values()),reused_ids=reused,initial_ids=new,grids=grids,
        bounds_mm=dict(lower=lo.tolist(),upper=hi.tolist()),previous_bounds_offsets_mm=[-1.5,1.5],
        optimizer=dict(method='4D Matern-5/2 GP, six sequential qLogEI proposals plus two global posterior-uncertainty proposals',
            batches=2,batch_size=8,feedback='Grid finished before batch 1; batch 2 uses all 8 outcomes from batch 1; durable input snapshots.',
            labels_used=False,training='Only compatible fixed-background historical rows and new numeric label-free J scores.',
            numerical_nugget_relative=1e-5,noise_interpretation='Single realization numerical regularization; not measured noise variance.'),
        budget=dict(grid_unique_conditions=83,grid_new_runs=len(new),historical_reused=len(reused),adaptive_new_runs=16,max_new_runs=len(new)+16,additional_seeds=0),
        resources=dict(max_workers=40,projected_per_tree_GiB=4,per_tree_limit_GiB=18,reserve_GiB=60,
            historical_peak_GiB=2.94,dispatch_window_hours=10,after_window='No new dispatch; drain active full trajectories, report incomplete budget explicitly.',
            GPU='Existing GPUs busy: do not reserve or interrupt them; small GP and scoring use CPU.'),
        contract='Only core A/B XY changes. EE outward1.25, EE core0.85, axis0deg, radii and all noise laws fixed. Core E thresholds only lowered; external fluctuations limited to core. Z/M, spatial OU, slow I and kicks off. Labels/visuals excluded from optimization. Fixed seeds do not ensure identical timewise innovations after masks change. No freeze or Fig5 promotion.',
        source_records=src_records,created_unix=time.time(),execution_sources={str(f):rt.sha(f) for f in [SCRIPT,REPORT]})
    rt.write(OUT/'plan.json',p);rt.write(OUT/'confirmation/frozen_networks.json',dict(replication_networks={}))
    rt.write(OUT/'status.json',dict(status='PREPARED',active=[],complete=len(reused),new_complete=0,queued=len(new),budget=p['budget'],time=time.time()))
    return p


def worker(cid):
    p=rt.read(OUT/'plan.json');c=rt.read(OUT/'candidates'/f'{cid}.json');ref=rt.read(OUT/'candidates'/f'{REF}.json')
    validate(c,ref,p['bounds_mm']['lower'],p['bounds_mm']['upper'])
    old.physical.OUT=OUT;old.physical.configure();old.physical.run.worker(STAGE,cid,TOPOLOGY,NOISE,60000.)


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
    p=rt.read(OUT/'plan.json');seed=2026091800+batch;torch.manual_seed(seed);torch.set_num_threads(2)
    ip=OUT/'proposals'/f'input_{batch:02d}.json'
    if not ip.exists():
        rows=[{k:r.get(k) for k in ['candidate','vector','J','N','status','objective_sha256']} for r in read_scores()]
        rt.write(ip,dict(rows=rows,known_vectors=[old.vector(c).tolist() for c in candidates().values()],seed=seed,time=time.time()))
    frozen=rt.read(ip);rows=frozen['rows'];good=[r for r in rows if r['J'] is not None]
    if batch>1:
        previous=rt.read(OUT/'proposals'/f'batch_{batch-1:02d}.json')
        assert len([r for r in rows if r['candidate'] in previous['ids']])==8
    assert len(good)>=8 and all(r['objective_sha256']==p['objective']['sha256'] for r in rows)
    lo=np.array(p['bounds_mm']['lower']);hi=np.array(p['bounds_mm']['upper'])
    X=torch.tensor((np.array([r['vector'][:4] for r in good])-lo)/(hi-lo),dtype=torch.float64)
    Y=torch.tensor([[-r['J']] for r in good],dtype=torch.float64)
    Yvar=torch.full_like(Y,max(float(Y.var()),1e-12)*1e-5)
    model=SingleTaskGP(X,Y,train_Yvar=Yvar,covar_module=ScaleKernel(MaternKernel(nu=2.5,ard_num_dims=4)),outcome_transform=Standardize(m=1))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood,model),optimizer_kwargs={'options':{'maxiter':160}})
    acq=qLogExpectedImprovement(model,best_f=Y.max(),sampler=SobolQMCNormalSampler(sample_shape=torch.Size([64]),seed=seed))
    proposed,_=optimize_acqf(acq,bounds=torch.tensor([[0.]*4,[1.]*4],dtype=torch.float64),q=6,
        num_restarts=8,raw_samples=128,sequential=True,options={'maxiter':80,'batch_limit':2})
    pool=qmc.Sobol(4,scramble=True,seed=seed+100).random_base2(12)
    with torch.no_grad():std=model.posterior(torch.tensor(pool,dtype=torch.float64)).variance[:,0].sqrt().numpy()
    used=(np.array(frozen['known_vectors'])[:,:4]-lo)/(hi-lo);actual=[];kinds=[]
    def novel(z):return np.min(np.linalg.norm(np.vstack([used,np.array(actual).reshape(-1,4)])-z,axis=1))>1e-5
    for z in proposed.detach().numpy():
        if novel(z):actual.append(z.tolist());kinds.append('qLogEI')
    for i in np.argsort(-std,kind='stable'):
        if len(actual)==8:break
        if novel(pool[i]):actual.append(pool[i].tolist());kinds.append('global_uncertainty')
    assert len(actual)==8
    ref=rt.read(OUT/'candidates'/f'{REF}.json');ids=[];vectors=[]
    for i,z in enumerate(actual):
        x=np.r_[lo+np.array(z)*(hi-lo),old.vector(ref)[4:]];cid=f'xyext_bo{batch:02d}_{i+1:02d}'
        c=old.candidate_at(ref,x,cid,'label_free_joint_XY_adaptive');c.update(search_role='adaptive_joint_XY',batch=batch)
        validate(c,ref,lo,hi);rt.write(OUT/'candidates'/f'{cid}.json',c);ids.append(cid);vectors.append(x.tolist())
    torch.save(dict(state_dict=model.state_dict(),X=X,Y=Y,Yvar=Yvar),OUT/'proposals'/f'gp_{batch:02d}.pt')
    value=dict(batch=batch,ids=ids,vectors=vectors,kinds=kinds,seed=seed,device='cpu',training_snapshot=str(ip),
        training_snapshot_sha256=rt.sha(ip),best_observed_J=min(r['J'] for r in good),objective_sha256=p['objective']['sha256'],time=time.time())
    rt.write(dest,value);return value


def controller():
    import psutil
    from scripts.analyze_topic4_three_observable_bo import audit_applied
    p=prepare();configure();active={};helper=None;reporter=None;fail=[];peaks={};audited=set();next_report=0.
    window=OUT/'window.json'
    if not window.exists():rt.write(window,dict(start_unix=time.time(),last_dispatch_unix=time.time()+10*3600))
    deadline=rt.read(window)['last_dispatch_unix']
    for path,h in {**p['source_snapshot'],**p['execution_sources']}.items():assert rt.sha(path)==h,path
    def alive(proc):
        if isinstance(proc,subprocess.Popen):return proc.poll() is None
        try:return proc.is_running() and proc.status()!=psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:return False
    def launch(action,extra=(),tag=None):
        log=(OUT/'logs'/f'{tag or action}.log').open('a')
        command=[rt.PYTHON if action=='worker' else old.BO_PY,'-u',str(SCRIPT),action,*extra]
        proc=subprocess.Popen(command,cwd=ROOT,env={**rt.ENV,'CUDA_VISIBLE_DEVICES':''},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        log.close();return proc
    for q in psutil.process_iter(['pid','cmdline']):
        cmd=q.info['cmdline'] or []
        if str(SCRIPT) not in cmd:continue
        if 'worker' in cmd:active[q.pid]=(q,cmd[cmd.index('--candidate')+1])
        elif 'score' in cmd:helper=(q,'score')
        elif 'propose' in cmd:helper=(q,'propose')
        elif 'report' in cmd:reporter=q
    psutil.cpu_percent();last_scored=-1;report_n=-1
    while True:
        detail=[]
        for pid,(proc,cid) in list(active.items()):
            if not alive(proc):
                active.pop(pid)
                if not old.valid_complete(result(cid)):fail.append(dict(candidate=cid,reason='NO_VALID_RESULT'))
                continue
            try:
                tree=[psutil.Process(pid)]+psutil.Process(pid).children(recursive=True)
                rss=sum(q.memory_info().rss for q in tree)/2**30
            except psutil.NoSuchProcess:continue
            peaks[cid]=max(peaks.get(cid,0),rss);detail.append(dict(pid=pid,candidate=cid,rss_GiB=rss))
            ap=result(cid).parents[1]/'applied_physics.json'
            if ap.exists() and cid not in audited:
                actual=rt.read(ap);c=rt.read(OUT/'candidates'/f'{cid}.json');audit_applied(actual,c)
                assert old.physics(actual['candidate'])==old.physics(c)
                assert actual['threshold']['n_raised']==0
                assert all(actual['input'][k]=='off' for k in ['ZM','slow_I_state','spatial_ou','kick'])
                audited.add(cid)
            if rss>18:
                for q in reversed(tree):
                    try:q.terminate()
                    except psutil.NoSuchProcess:pass
                fail.append(dict(candidate=cid,reason='PROCESS_TREE_MEMORY_LIMIT'))
        if helper and not alive(helper[0]):
            if isinstance(helper[0],subprocess.Popen) and helper[0].returncode:fail.append(dict(reason='HELPER_FAILURE',action=helper[1]))
            helper=None
        if reporter and not alive(reporter):
            if isinstance(reporter,subprocess.Popen) and reporter.returncode:
                rt.write(OUT/'report_failure.json',dict(returncode=reporter.returncode,time=time.time()))
            reporter=None
        scores={s['candidate']:s for s in read_scores()};cs=candidates()
        raw_done=sum(old.valid_complete(result(cid)) for cid in cs)
        if not helper and not fail and raw_done>len(scores):helper=(launch('score'),'score')
        grid_done=all(cid in scores for cid in p['initial_ids']);queue=list(p['initial_ids']);batches_done=0
        for batch in [1,2]:
            dest=OUT/'proposals'/f'batch_{batch:02d}.json'
            if dest.exists():
                proposal=rt.read(dest);queue+=proposal['ids']
                if all(cid in scores for cid in proposal['ids']):batches_done=batch
                else:break
            elif grid_done and batches_done==batch-1 and not helper and not fail and time.time()<deadline:
                helper=(launch('propose',['--batch',str(batch)],f'propose_{batch:02d}'),'propose');break
            else:break
        pending=[c for c in queue if c not in scores and c not in {v[1] for v in active.values()} and not old.valid_complete(result(c))]
        available=psutil.virtual_memory().available/2**30
        if available<28 and detail:
            worst=max(detail,key=lambda d:d['rss_GiB']);q=psutil.Process(worst['pid'])
            for child in reversed([q]+q.children(recursive=True)):
                try:child.terminate()
                except psutil.NoSuchProcess:pass
            fail.append(dict(candidate=worst['candidate'],reason='SYSTEM_MEMORY_EMERGENCY'))
        growth=sum(max(0,4-d['rss_GiB']) for d in detail);cpu=psutil.cpu_percent()
        cap=40 if len(audited)>=2 or len([s for s in scores.values() if not s.get('reused')])>=2 else 2
        if pending and not fail and time.time()<deadline and len(active)<cap and available-growth>64 and cpu<94:
            cid=pending[0];proc=launch('worker',['--candidate',cid],cid);active[proc.pid]=(proc,cid)
            detail.append(dict(pid=proc.pid,candidate=cid,rss_GiB=0))
            with (OUT/'dispatch.jsonl').open('a') as f:f.write(json.dumps(dict(candidate=cid,pid=proc.pid,time=time.time()))+'\n')
        done=grid_done and batches_done==2 and not active and not helper
        cutoff=time.time()>=deadline and not active and not helper
        state='DRAINING_ENGINEERING_FAILURE' if fail else 'DRAINING_TIME_WINDOW' if time.time()>=deadline else 'RUNNING'
        status=dict(status=state,complete=len(scores),new_complete=sum(not s.get('reused') for s in scores.values()),reused=len(p['reused_ids']),
            active=detail,queued=len(pending),grid_done=grid_done,adaptive_batches_complete=batches_done,helper=helper[1] if helper else None,
            application_audited=sorted(audited),peak_process_tree_GiB=peaks,available_GiB=available,cpu_percent=cpu,failures=fail,time=time.time(),last_dispatch_unix=deadline)
        rt.write(OUT/'status.json',status)
        if (done or cutoff or (fail and not active)):
            if reporter:
                while alive(reporter):time.sleep(5)
            with (OUT/'logs/final_report.log').open('a') as log:
                report=subprocess.run([old.BO_PY,'-u',str(SCRIPT),'report','--final-media'],cwd=ROOT,env={**rt.ENV,'CUDA_VISIBLE_DEVICES':''},stdout=log,stderr=subprocess.STDOUT)
            status.update(status='FAILED_PENDING_REVIEW' if fail else 'COMPLETE_PENDING_SCIENTIFIC_REVIEW' if done else 'WINDOW_ENDED_PARTIAL_PENDING_REVIEW',
                media_report_exit=report.returncode,active=[],time=time.time())
            rt.write(OUT/'status.json',status);rt.write(OUT/'completion.json',status);return
        if not reporter and time.time()>=next_report and len(scores)!=report_n:
            reporter=launch('report');report_n=len(scores);next_report=time.time()+1800
        time.sleep(2)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','controller','worker','score','propose','report'])
    parser.add_argument('--candidate');parser.add_argument('--batch',type=int);parser.add_argument('--final-media',action='store_true');args=parser.parse_args()
    if args.action=='prepare':print(json.dumps(prepare()['budget']))
    elif args.action=='worker':worker(args.candidate)
    elif args.action=='score':configure();print(dense.score_ready())
    elif args.action=='propose':print(json.dumps(propose(args.batch)))
    elif args.action=='report':
        from scripts.report_topic4_xy_overnight_expansion import main
        main(OUT,final_media=args.final_media)
    else:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'controller.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);controller()
