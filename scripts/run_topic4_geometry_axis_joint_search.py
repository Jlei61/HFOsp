"""Authorized 24-condition interaction design followed by three 8-point MOBO batches.

Two frozen, label-free distribution objectives; native physics is unchanged.
TA/TB, movies and diagnostic observables never enter propose().
"""
from pathlib import Path
import argparse, copy, csv, fcntl, json, os, pickle, shutil, subprocess, sys, time
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src/snn_engine')]
import numpy as np
from scripts import run_topic4_label_free_dense_search as dense
from scripts.audit_topic4_joint_observation import JointMap, statistic
old, rt = dense.old, dense.rt
SOURCE = dense.OUT
XY = SOURCE.parent/'xy_block_joint_observation_pilot_20260916'
OUT = SOURCE.parent/'geometry_axis_joint_distribution_20260917'
SCRIPT = Path(__file__).resolve()
REPORT = ROOT/'scripts/report_topic4_geometry_axis_joint_search.py'
STAGE = 'search'
TOPOLOGY, NOISE = 2511, 847401
GPU_ENV = {**rt.ENV, 'CUDA_VISIBLE_DEVICES': '1', 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
CPU_ENV = {**rt.ENV, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}


def candidates():
    return {f.stem: rt.read(f) for f in sorted((OUT/'candidates').glob('*.json'))}


def scores():
    return [rt.read(f) for f in sorted((OUT/'scores').glob('*.json'))]


def result(cid):
    return OUT/STAGE/'units'/cid/f'{TOPOLOGY}_{NOISE}'/'workers/trajectory.json'


def feasible_vector(x, lo, hi, radius):
    x = np.asarray(x, float)
    xy = x[:4].reshape(2, 2)
    return bool(np.isfinite(x).all() and np.all(x >= lo-1e-10) and np.all(x <= hi+1e-10)
                and np.all(xy >= radius[:, None]) and np.all(xy <= 20-radius[:, None])
                and np.linalg.norm(xy[0]-xy[1]) > radius.sum())


def validate(c, plan):
    ref = rt.read(OUT/'candidates'/f"{plan['reference_id']}.json")
    assert old.background(c) == old.background(ref), 'frozen background differs'
    assert feasible_vector(old.vector(c), np.array(plan['bounds']['lower']),
                           np.array(plan['bounds']['upper']), np.array(c['radii_mm']))


def pareto_indices(values):
    a = np.asarray(values, float)
    return [i for i in range(len(a)) if not np.any(np.all(a <= a[i], axis=1) & np.any(a < a[i], axis=1))]


def nominees(rows, norm):
    good = sorted([r for r in rows if r.get('J') is not None and r.get('J_joint') is not None], key=lambda r: r['candidate'])
    if not good:
        return []
    y = np.array([[r['J'], r['J_joint']] for r in good])
    front = pareto_indices(y)
    z = (y-np.asarray(norm['center']))/np.asarray(norm['scale'])
    ideal = z[front].min(0)
    knee = min(front, key=lambda i: (float(np.max(z[i]-ideal)), good[i]['candidate']))
    chosen = [(int(np.argmin(y[:, 0])), 'original_objective'),
              (int(np.argmin(y[:, 1])), 'same_event_joint_objective'), (knee, 'balanced_pareto')]
    out = {}
    for i, role in chosen:
        cid = good[i]['candidate']
        if cid not in out: out[cid] = dict(candidate=cid, roles=[], J=good[i]['J'], J_joint=good[i]['J_joint'])
        out[cid]['roles'].append(role)
    return list(out.values())


def prepare():
    if (OUT/'plan.json').exists():
        return rt.read(OUT/'plan.json')
    for name in ['candidates', 'scores', 'logs', 'analysis', 'proposals', 'global_graph_cache', 'confirmation', 'key_results']:
        (OUT/name).mkdir(parents=True, exist_ok=True)
    parent = rt.read(XY/'plan.json')
    for path, sha in parent['source_snapshot'].items():
        assert rt.sha(path) == sha, path
    pool, old_scores = {}, {}
    for directory in [SOURCE, XY]:
        for f in sorted((directory/'scores').glob('*.json')):
            sc = rt.read(f); cid = sc['candidate']; c = rt.read(directory/'candidates'/f'{cid}.json')
            if cid in pool:
                assert old.physics(pool[cid]) == old.physics(c)
                assert old_scores[cid]['source_sha256'] == sc['source_sha256']
                continue
            pool[cid], old_scores[cid] = c, sc
    assert len(pool) == 184
    ref = pool['xy_right_33']; other = pool['lf_bo02_01']
    for c in pool.values():
        assert old.background(c) == old.background(ref)
        rt.write(OUT/'candidates'/f"{c['id']}.json", c)
    # Same event-level map and patient target as the already-frozen offline audit.
    joint_dir = OUT/'analysis/joint_objective'; joint_dir.mkdir(exist_ok=True)
    for name in ['frozen_contract.json', 'feature_map.npz', 'target.npz', 'calibration.json']:
        shutil.copy2(XY/'joint_feature_audit'/name, joint_dir/name)
    shutil.copy2(SOURCE/'analysis/training_objective.pkl', OUT/'analysis/training_objective.pkl')
    joint_values = {}
    with (XY/'joint_feature_audit/candidate_scores.csv').open() as f:
        for r in csv.DictReader(f):
            joint_values[r['candidate']] = dict(N=int(r['N']), A=float(r['joint_A']), B=float(r['joint_B']),
                D_off=float(r['joint_Doff']), scaled=float(r['joint_scaled']))
    joint_values.update(rt.read(XY/'analysis/offline_joint_scores.json'))
    imported = []
    for cid, sc in old_scores.items():
        c = pool[cid]; src = Path(sc['source']); payload = rt.read(src)
        assert rt.sha(src) == sc['source_sha256']
        assert rt.sha(src.with_suffix('.npz')) == payload['arrays_sha256']
        assert payload['job']['topology_seed'] == TOPOLOGY and payload['job']['dynamics_seed'] == NOISE
        assert payload['actual_duration_ms'] == 60000
        assert old.physics(rt.read(src.parents[1]/'applied_physics.json')['candidate']) == old.physics(c)
        assert sc['objective_sha256'] == parent['objective']['sha256']
        joint = copy.deepcopy(joint_values[cid]); joint.pop('diagnostic_only', None)
        assert joint['N'] == sc['N']
        sc = copy.deepcopy(sc)
        sc.update(J_joint=joint['scaled'], joint=joint, reused=True, search_role='historical',
                  vector=old.vector(c).tolist(), joint_source=str(XY/'joint_feature_audit'),
                  physical_status=payload['physical_status'])
        rt.write(OUT/'scores'/f'{cid}.json', sc); imported.append(sc)
        dest = result(cid).parents[1]; dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists(): dest.symlink_to(src.parents[1].resolve())
    # Exact factorial conditions: all other coordinates are the old common anchor.
    anchor = pool['g1_axis3_plus']; x0 = old.vector(anchor)
    yy = [float(x0[3]+v) for v in [0., .5, 1., 1.5]]
    angles = [0., 20/3, 40/3]; gains = [1.25, 1.5]
    factorial, new_ids = [], []
    for iy, y in enumerate(yy):
        for ia, angle in enumerate(angles):
            for ig, gain in enumerate(gains):
                x = x0.copy(); x[3:6] = [y, gain, angle]
                match = next((c for c in pool.values() if np.allclose(old.vector(c), x, rtol=0, atol=1e-10)), None)
                if match is None:
                    cid = f'interaction_y{iy}_a{ia}_g{ig}'
                    match = old.candidate_at(anchor, x, cid, 'position_angle_EE_interaction')
                    pool[cid] = match; new_ids.append(cid); rt.write(OUT/'candidates'/f'{cid}.json', match)
                factorial.append(dict(candidate=match['id'], right_y_mm=y, angle_offset_deg=angle,
                                      EE_core_to_out_scale=gain, reused=match['id'] not in new_ids))
    coordinates = np.array([old.vector(ref)[:4], old.vector(other)[:4]])
    lo = np.r_[coordinates.min(0)-.5, 1., -20.]
    hi = np.r_[coordinates.max(0)+.5, 1.5, 20.]
    X = np.array([r['vector'] for r in imported]); vals = np.array([[r['J'], r['J_joint']] for r in imported])
    center = np.median(vals, axis=0); scale = np.maximum(np.quantile(vals,.75,axis=0)-np.quantile(vals,.25,axis=0), .05)
    norm = dict(center=center.tolist(), scale=scale.tolist(),
                reference_loss=(np.quantile(vals,.95,axis=0)+.1*scale).tolist(),
                source='184 historical label-free scores; fixed before new outcomes')
    for directory in [SOURCE, XY]:
        for f in (directory/'global_graph_cache').glob('*.pkl'):
            dest = OUT/'global_graph_cache'/f.name
            if not dest.exists(): dest.symlink_to(f.resolve())
    p = {k: copy.deepcopy(parent[k]) for k in ['physics', 'source_snapshot', 'analysis', 'runaway']}
    p.update(schema='topic4.geometry_axis_joint_distribution.v1',
        authorization='2026-09-17 user: 开始下一轮。不要让CPU和GPU空置。 Accepted 24 interaction cells then 3x8 adaptive proposals.',
        reference_id='xy_right_33', joint_reference_id='lf_bo02_01',
        historical_visual_control='g2_b03_p01', topology_seed=TOPOLOGY, seeds=[NOISE], duration_ms=60000.,
        historical_ids=sorted(old_scores), interaction_cells=factorial, interaction_new_ids=new_ids,
        bounds=dict(lower=lo.tolist(), upper=hi.tolist(), order=['left_x','left_y','right_x','right_y','EE_out','EE_angle']),
        input_normalization=dict(lower=np.minimum(X.min(0),lo).tolist(), upper=np.maximum(X.max(0),hi).tolist()),
        objective=dict(artifact=str(OUT/'analysis/training_objective.pkl'), sha256=rt.sha(OUT/'analysis/training_objective.pkl'),
            joint_directory=str(joint_dir), joint_hashes={n:rt.sha(joint_dir/n) for n in ['feature_map.npz','target.npz','frozen_contract.json']},
            method='original three-group Doff and frozen same-event joint Doff; minimize both', labels_used=False,
            negative_values_clipped=False, minimum_events=16, normalization=norm),
        optimizer=dict(method='two-output Matern GP, qLogNEHVI on GPU; fixed batches, delayed updates', batches=3,
            batch_size=8, simulation_seeds='single fixed topology/noise identity',
            nuisance='GP likelihood models surrogate residual only; not empirical between-seed variance',
            nomination='original minimum, joint minimum, normalized minimax distance to Pareto ideal; at most three',
            physical_evaluations='never gradient through SNN; no polish or unbudgeted evaluations'),
        budget=dict(interaction_conditions=24, interaction_new_runs=len(new_ids), interaction_reused=24-len(new_ids),
            adaptive_new_runs=24, max_new_runs=len(new_ids)+24, historical_reused=184, additional_seeds=0),
        resources=dict(max_workers=24, per_tree_limit_GiB=18, projected_tree_GiB=6, reserve_GiB=50,
            gpu_physical_index=1, cpu_dispatch_ceiling_percent=94, threads_per_simulation=1),
        acceptance=dict(figures=['same-network multi-event GIF with full native field',
            'patient mean, model mean/spread, same model event envelope and native snapshots'],
            no_label_objective=True, endpoint_geometry='development prior',
            review='labels and native visual residuals informed experimental design; not blind validation',
            phenotype='patient-compatible two-mode propagation not guaranteed by scores or label counts',
            notes='fixed seed does not freeze EE edges across angle changes, or per-neuron noise across core moves'),
        created_unix=time.time())
    for cell in factorial:
        c=pool[cell['candidate']]
        assert old.background(c) == old.background(ref)
        assert feasible_vector(old.vector(c),lo,hi,np.array(c['radii_mm']))
    rt.write(OUT/'plan.json', p)
    rt.write(OUT/'confirmation/frozen_networks.json', dict(replication_networks={}))
    rt.write(OUT/'status.json', dict(status='PREPARED',new_complete=0,historical_reused=184,active=[],queued=len(new_ids),budget=p['budget'],time=time.time()))
    return p


def worker(cid):
    p=rt.read(OUT/'plan.json');c=rt.read(OUT/'candidates'/f'{cid}.json');validate(c,p)
    old.physical.OUT=OUT;old.physical.configure()
    old.physical.run.worker(STAGE,cid,TOPOLOGY,NOISE,p['duration_ms'])


def objective_maps():
    p=rt.read(OUT/'plan.json'); spec=p['objective'];path=Path(spec['artifact'])
    assert rt.sha(path)==spec['sha256']
    with path.open('rb') as f:obj=pickle.load(f)
    assert not any(hasattr(obj,k) for k in ['km','proportions','mode_means'])
    directory=Path(spec['joint_directory'])
    for name,h in spec['joint_hashes'].items():assert rt.sha(directory/name)==h
    mapping=JointMap(obj)
    with np.load(directory/'feature_map.npz') as a:mapping.W=a['weights'];mapping.phase=a['phases']
    with np.load(directory/'target.npz') as a:target=a['target'];scale=float(a['scale'])
    return obj,mapping,target,scale


def score_ready(check_reference=False):
    from scripts.analyze_topic4_three_observable_bo import audit_applied
    obj,mapping,target,scale=objective_maps();device=dense.gpu_device();count=0
    for cid,c in candidates().items():
        dest=OUT/'scores'/f'{cid}.json';src=result(cid)
        if (dest.exists() and not (check_reference and cid=='xy_right_33')) or not old.valid_complete(src):continue
        r,t,names=dense.load_times(src);assert names==obj.names
        applied=rt.read(src.parents[1]/'applied_physics.json');audit_applied(applied,c)
        assert old.physics(applied['candidate'])==old.physics(c)
        assert rt.sha(src.with_suffix('.npz'))==r['arrays_sha256']
        assert r['job']['topology_seed']==TOPOLOGY and r['job']['dynamics_seed']==NOISE
        assert r['actual_duration_ms']==60000 or r['physical_status']=='RUNAWAY'
        sc=obj.score(t,device=device)
        joint=statistic(mapping.features(obj.raw_features(t),device),target) if len(t)>=16 else dict(N=len(t),A=None,B=None,D_off=None)
        value=joint['D_off']/scale if joint['D_off'] is not None else None
        if r['physical_status']=='RUNAWAY':sc.update(status='PHYSICAL_RUNAWAY',J=None);value=None
        joint['scaled']=value
        if check_reference and cid=='xy_right_33':
            before=rt.read(dest);np.testing.assert_allclose([sc['J'],value],[before['J'],before['J_joint']],rtol=1e-10,atol=1e-10)
            rt.write(OUT/'analysis/reference_score_parity.json',dict(candidate=cid,device=device,passed=True,values=[sc['J'],value],time=time.time()));continue
        sc.update(candidate=cid,source=str(src),source_sha256=rt.sha(src),vector=old.vector(c).tolist(),
            topology=TOPOLOGY,noise=NOISE,reused=False,J_joint=value,joint=joint,physical_status=r['physical_status'],
            objective_sha256=rt.read(OUT/'plan.json')['objective']['sha256'],device=device,time=time.time())
        rt.write(dest,sc);count+=1
    print('scored',count,'on',device,flush=True)
    return count


def propose(batch, dry=False):
    import torch
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.fit import fit_gpytorch_mll
    from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
    from botorch.optim import optimize_acqf
    from botorch.sampling.normal import SobolQMCNormalSampler
    from gpytorch.kernels import MaternKernel,ScaleKernel
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from scipy.stats import qmc
    p=rt.read(OUT/'plan.json');dest=OUT/'proposals'/('smoke.json' if dry else f'batch_{batch:02d}.json')
    if dest.exists():return rt.read(dest)
    snapshot=OUT/'proposals'/('smoke_input.json' if dry else f'input_{batch:02d}.json')
    if snapshot.exists():frozen=rt.read(snapshot)
    else:
        if not dry:
            required=[x['candidate'] for x in p['interaction_cells']]
            for k in range(1,batch):required+=rt.read(OUT/'proposals'/f'batch_{k:02d}.json')['ids']
            assert all((OUT/'scores'/f'{cid}.json').exists() for cid in required), 'batch feedback incomplete'
        # Explicit allowlist, excluding diagnostic labels and paths from model feedback.
        frozen=dict(rows=[{k:r.get(k) for k in ['candidate','J','J_joint','N','status','vector']} for r in scores()],
            known_vectors=[old.vector(c).tolist() for c in candidates().values()],time=time.time())
        rt.write(snapshot,frozen)
    good=[r for r in frozen['rows'] if r['J'] is not None and r['J_joint'] is not None]
    device=dense.gpu_device();seed=2026091700+batch;torch.manual_seed(seed);torch.set_num_threads(2)
    xl=np.array(p['input_normalization']['lower']);xh=np.array(p['input_normalization']['upper'])
    norm=p['objective']['normalization'];yc=np.array(norm['center']);ys=np.array(norm['scale'])
    X=torch.tensor((np.array([r['vector'] for r in good])-xl)/(xh-xl),device=device,dtype=torch.float64)
    Y=torch.tensor(-(np.array([[r['J'],r['J_joint']] for r in good])-yc)/ys,device=device,dtype=torch.float64)
    cov=ScaleKernel(MaternKernel(nu=2.5,ard_num_dims=6,batch_shape=torch.Size([2])),batch_shape=torch.Size([2])).to(X)
    model=SingleTaskGP(X,Y,covar_module=cov,outcome_transform=Standardize(m=2)).to(X)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood,model),optimizer_kwargs={'options':{'maxiter':45 if dry else 160}})
    ref=(-(np.array(norm['reference_loss'])-yc)/ys).tolist()
    acq=qLogNoisyExpectedHypervolumeImprovement(model=model,ref_point=ref,X_baseline=X,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([16 if dry else 64]),seed=seed),prune_baseline=True)
    lo=np.array(p['bounds']['lower']);hi=np.array(p['bounds']['upper'])
    bounds=torch.tensor(np.stack([(lo-xl)/(xh-xl),(hi-xl)/(xh-xl)]),device=device,dtype=torch.float64)
    proposed,_=optimize_acqf(acq,bounds=bounds,q=2 if dry else 8,num_restarts=2 if dry else 8,
        raw_samples=32 if dry else 128,sequential=True,options={'maxiter':20 if dry else 80,'batch_limit':2})
    used=(np.array(frozen['known_vectors'])-xl)/(xh-xl);actual=[];kinds=[]
    def novel(z):return np.min(np.linalg.norm(np.vstack([used,np.asarray(actual).reshape(-1,6)])-z,axis=1))>1e-5
    for z in proposed.detach().cpu().numpy():
        if novel(z):actual.append(z.tolist());kinds.append('qLogNEHVI')
    need=2 if dry else 8
    # A recorded uncertainty proposal fills duplicates only, never extra simulations.
    if len(actual)<need:
        pool=(lo-xl)/(xh-xl)+qmc.Sobol(6,scramble=True,seed=seed+100).random_base2(11)*(hi-lo)/(xh-xl)
        with torch.no_grad():unc=model.posterior(torch.tensor(pool,device=device,dtype=torch.float64)).variance.sum(-1).cpu().numpy()
        for i in np.argsort(-unc):
            if len(actual)>=need:break
            if novel(pool[i]):actual.append(pool[i].tolist());kinds.append('posterior_uncertainty_duplicate_fallback')
    assert len(actual)==need
    vec=xl+np.array(actual)*(xh-xl);anchor=rt.read(OUT/'candidates'/f"{p['reference_id']}.json")
    ids=[]
    for k,x in enumerate(vec):
        cid=f'joint_bo{batch:02d}_{k+1:02d}';c=old.candidate_at(anchor,x,cid,'two_objective_joint_proposal');validate(c,p)
        if not dry:rt.write(OUT/'candidates'/f'{cid}.json',c)
        ids.append(cid)
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},X=X.cpu(),Y=Y.cpu()),dest.with_suffix('.pt'))
    payload=dict(batch=batch,ids=ids,vectors=vec.tolist(),kinds=kinds,device=device,seed=seed,dry_run=dry,
        training_snapshot=str(snapshot),training_snapshot_sha256=rt.sha(snapshot),n_training=len(good),
        learned_surrogate_residual=model.likelihood.noise.detach().cpu().numpy().tolist(),
        scores_use_labels=False,reference_point=ref,time=time.time())
    rt.write(dest,payload);print(json.dumps(payload),flush=True);return payload


def controller():
    import psutil
    from scripts.analyze_topic4_three_observable_bo import audit_applied
    p=prepare();active={};audited=set();peaks={};failure=[];helper=None;last_score=0.;next_report=0.
    if (OUT/'status.json').exists():
        last=rt.read(OUT/'status.json');audited=set(last.get('application_audited',[]));peaks=last.get('peak_process_tree_GiB',{})
    for q in psutil.process_iter(['pid','cmdline']):
        cmd=q.info['cmdline'] or []
        if str(SCRIPT) in cmd and 'worker' in cmd and '--candidate' in cmd:
            active[q.pid]=(q,cmd[cmd.index('--candidate')+1],None)
    def alive(q):
        if isinstance(q,subprocess.Popen):return q.poll() is None
        try:return q.is_running() and q.status()!=psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:return False
    def spawn_helper(action,*args):
        log=(OUT/'logs'/f'{action}.log').open('a')
        cmd=[old.BO_PY,'-u',str(REPORT)] if action=='report' else [old.BO_PY,'-u',str(SCRIPT),action]
        q=subprocess.Popen(cmd+list(args),cwd=ROOT,env=GPU_ENV,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        log.close();return (q,action)
    psutil.cpu_percent()
    while True:
        details=[]
        for pid,(proc,cid,log) in list(active.items()):
            if not alive(proc):
                if log:log.close()
                active.pop(pid)
                if not old.valid_complete(result(cid)):failure.append(dict(candidate=cid,reason='WORKER_WITHOUT_RESULT'))
                continue
            try:
                q=psutil.Process(pid);rss=sum(z.memory_info().rss for z in [q]+q.children(recursive=True))/2**30
            except psutil.NoSuchProcess:continue
            peaks[cid]=max(peaks.get(cid,0),rss);details.append(dict(candidate=cid,pid=pid,rss_GiB=rss))
            applied=result(cid).parents[1]/'applied_physics.json'
            if applied.exists() and cid not in audited:
                c=rt.read(OUT/'candidates'/f'{cid}.json');a=rt.read(applied);audit_applied(a,c)
                assert old.physics(a['candidate'])==old.physics(c);audited.add(cid)
            if rss>p['resources']['per_tree_limit_GiB']:
                for z in reversed([q]+q.children(recursive=True)):
                    try:z.terminate()
                    except psutil.NoSuchProcess:pass
                failure.append(dict(candidate=cid,reason='PROCESS_TREE_MEMORY_LIMIT'))
        if helper and helper[0].poll() is not None:
            if helper[0].returncode:failure.append(dict(reason='HELPER_FAILED',action=helper[1],returncode=helper[0].returncode))
            helper=None
        table={r['candidate']:r for r in scores()};queue=list(p['interaction_new_ids']);ready=True;next_batch=None
        for k in range(1,4):
            if not all(cell['candidate'] in table for cell in p['interaction_cells']):ready=False;break
            f=OUT/'proposals'/f'batch_{k:02d}.json'
            if not f.exists():next_batch=k;break
            ids=rt.read(f)['ids'];queue+=ids
            if not all(cid in table for cid in ids):ready=False;break
        queued=[cid for cid in queue if cid not in table and cid not in {v[1] for v in active.values()} and not old.valid_complete(result(cid))]
        unscored=[cid for cid in queue if cid not in table and old.valid_complete(result(cid))]
        available=psutil.virtual_memory().available/2**30;cpu=psutil.cpu_percent()
        growth=sum(max(0,p['resources']['projected_tree_GiB']-r['rss_GiB']) for r in details)
        cap=p['resources']['max_workers'] if len(audited)>=2 else 2
        if queued and not failure and len(active)<cap and available-growth>56 and cpu<94:
            cid=queued[0];log=(OUT/'logs'/f'{cid}.log').open('a')
            q=subprocess.Popen([rt.PYTHON,'-u',str(SCRIPT),'worker','--candidate',cid],cwd=ROOT,env=CPU_ENV,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active[q.pid]=(q,cid,log)
            with (OUT/'dispatch.jsonl').open('a') as f:f.write(json.dumps(dict(candidate=cid,pid=q.pid,time=time.time(),available_GiB=available))+'\n')
        if available<28 and details:
            worst=max(details,key=lambda r:r['rss_GiB']);q=psutil.Process(worst['pid'])
            for z in reversed([q]+q.children(recursive=True)):
                try:z.terminate()
                except psutil.NoSuchProcess:pass
            failure.append(dict(reason='SYSTEM_MEMORY_EMERGENCY',candidate=worst['candidate']))
        if not failure and helper is None:
            if unscored:
                helper=spawn_helper('score');last_score=time.time()
            elif ready and next_batch:
                helper=spawn_helper('propose','--batch',str(next_batch))
            elif time.time()>=next_report and REPORT.exists():
                helper=spawn_helper('report');next_report=time.time()+900
        new_complete=sum(not r.get('reused') for r in table.values())
        done=ready and next_batch is None and not active and len(queue)==p['budget']['max_new_runs'] and all(cid in table for cid in queue)
        state=dict(status='DRAINING_FAILURE' if failure else 'RUNNING',active=details,helper=dict(pid=helper[0].pid,action=helper[1]) if helper else None,
            complete=len(table),new_complete=new_complete,historical_reused=184,queued=len(queued),
            interaction_complete=sum(x['candidate'] in table for x in p['interaction_cells']),
            adaptive_proposed=sum((OUT/'proposals'/f'batch_{k:02d}.json').exists() for k in range(1,4)),
            budget=p['budget'],application_audited=sorted(audited),peak_process_tree_GiB=peaks,
            available_GiB=available,failures=failure,time=time.time())
        rt.write(OUT/'status.json',state)
        if done and not failure:
            if helper:
                helper[0].wait()
                if helper[0].returncode:raise RuntimeError('analysis helper failed')
            state.update(status='GENERATING_FINAL_ACCEPTANCE_FIGURES',active=[],helper=None);rt.write(OUT/'status.json',state)
            with (OUT/'logs/final_report.log').open('a') as log:
                subprocess.run([old.BO_PY,'-u',str(REPORT),'--final-media'],cwd=ROOT,env=GPU_ENV,stdout=log,stderr=subprocess.STDOUT,check=True)
            state.update(status='COMPLETE_PENDING_SCIENTIFIC_AND_VISUAL_REVIEW',time=time.time())
            rt.write(OUT/'status.json',state);rt.write(OUT/'completion.json',state);return
        if failure and not active:raise RuntimeError(json.dumps(failure))
        time.sleep(2)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','controller','worker','score','propose','smoke'])
    ap.add_argument('--candidate');ap.add_argument('--batch',type=int,default=1);args=ap.parse_args()
    if args.action=='prepare':print(json.dumps(prepare()['budget']))
    elif args.action=='worker':worker(args.candidate)
    elif args.action in ['score','propose','smoke']:
        # A resumed controller may encounter a helper from its predecessor.
        # Serialize helpers so frozen proposals and atomic score files have one writer.
        with (OUT/'analysis/helper.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if args.action=='score':score_ready()
            elif args.action=='propose':propose(args.batch)
            else:score_ready(check_reference=True);propose(1,dry=True)
    else:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'controller.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);controller()
