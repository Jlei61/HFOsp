"""Expanded fixed-background XY response grid; no adaptive proposals or extra seeds."""
from pathlib import Path
import argparse, copy, fcntl, json, os, shutil, subprocess, sys, time
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src/snn_engine')]
import numpy as np
from scripts import run_topic4_label_free_dense_search as dense
old, rt = dense.old, dense.rt
SOURCE = dense.OUT
OUT = SOURCE.parent/'core_position_wide_response_20260918'
SCRIPT = Path(__file__).resolve()
REF = 'dense_axis3_12'
STAGE = 'xy_wide'
TOPOLOGY, NOISE = 2511, 847401


def configure_score():
    dense.OUT, dense.STAGE = OUT, STAGE


def result(cid):
    return OUT/STAGE/'units'/cid/f'{TOPOLOGY}_{NOISE}'/'workers/trajectory.json'


def validate(c, ref):
    assert old.background(c) == old.background(ref)
    x, x0 = old.vector(c), old.vector(ref)
    np.testing.assert_allclose(x[4:], x0[4:], rtol=0, atol=0)
    assert np.all(np.abs(x[:4]-x0[:4]) <= 1.5000000001)
    xy, radius = np.array(c['centers_mm']), np.array(c['radii_mm'])
    assert np.all(xy >= radius[:, None]) and np.all(xy <= 20-radius[:, None])
    assert np.linalg.norm(xy[0]-xy[1]) > radius.sum()


def prepare():
    if (OUT/'plan.json').exists():
        return rt.read(OUT/'plan.json')
    for d in ['candidates', 'scores', 'logs', 'analysis', 'proposals', 'global_graph_cache', 'confirmation']:
        (OUT/d).mkdir(parents=True, exist_ok=True)
    parent = rt.read(SOURCE/'plan.json')
    for path, sha in parent['source_snapshot'].items():
        assert rt.sha(path) == sha, path
    ref = rt.read(SOURCE/'candidates'/f'{REF}.json')
    existing = [(rt.read(p), rt.read(SOURCE/'scores'/p.name)) for p in sorted((SOURCE/'candidates').glob('*.json'))]
    registered, grids, reused = {}, [], []
    offsets = [-1.5, -.75, 0., .75, 1.5]
    for core in range(2):
        ids = []
        for iy, dy in enumerate(offsets):
            row = []
            for ix, dx in enumerate(offsets):
                x = old.vector(ref).copy(); x[2*core:2*core+2] += [dx, dy]
                match = next(((c, s) for c, s in existing if np.allclose(old.vector(c), x, rtol=0, atol=1e-10)
                              and old.background(c) == old.background(ref)), None)
                if match:
                    c, score = copy.deepcopy(match[0]), copy.deepcopy(match[1])
                    cid = c['id']; src = Path(score['source'])
                    payload = rt.read(src)
                    assert payload['actual_duration_ms'] == 60000
                    assert payload['job']['topology_seed'] == TOPOLOGY and payload['job']['dynamics_seed'] == NOISE
                    assert score['objective_sha256'] == parent['objective']['sha256']
                    assert old.physics(c) == old.physics(rt.read(src.parents[1]/'applied_physics.json')['candidate'])
                    assert rt.sha(src.with_suffix('.npz')) == payload['arrays_sha256']
                    dest = result(cid).parents[1]; dest.parent.mkdir(parents=True, exist_ok=True)
                    if not dest.exists(): dest.symlink_to(src.parents[1].resolve())
                    score.update(reused=True, reuse_source=str(src)); rt.write(OUT/'scores'/f'{cid}.json', score)
                    if cid not in reused: reused.append(cid)
                else:
                    cid = f'xywide_{"left" if core == 0 else "right"}_{iy}{ix}'
                    c = old.candidate_at(ref, x, cid, 'two_dimensional_core_response')
                validate(c, ref)
                c.update(xy_block_reference=REF)
                registered[cid] = c; row.append(cid)
            ids.append(row)
        grids.append(dict(core=core, x_offsets_mm=offsets, y_offsets_mm=offsets, ids=ids))
    assert len(registered) == 49
    for cid, c in registered.items(): rt.write(OUT/'candidates'/f'{cid}.json', c)
    for path in (SOURCE/'global_graph_cache').glob('*.pkl'):
        dest = OUT/'global_graph_cache'/path.name
        if not dest.exists(): dest.symlink_to(path.resolve())
    shutil.copy2(parent['objective']['artifact'], OUT/'analysis/training_objective.pkl')
    objective = copy.deepcopy(parent['objective']); objective['artifact'] = str(OUT/'analysis/training_objective.pkl')
    assert rt.sha(objective['artifact']) == objective['sha256']
    # Near-reference cardinal/diagonal conditions first, then expand; no outcome-dependent order.
    initial = [c for c in registered if c not in reused]
    initial.sort(key=lambda cid: (float(np.linalg.norm(old.vector(registered[cid])[:4]-old.vector(ref)[:4])), cid))
    plan = {k: copy.deepcopy(parent[k]) for k in ['physics', 'source_snapshot', 'analysis', 'runaway']}
    plan.update(schema='topic4.xy_wide_response.v1', authorization='2026-09-18 user requests a larger core XY range and step, launch simulations and update A',
        reference_id=REF, source=str(SOURCE), source_plan_sha256=rt.sha(SOURCE/'plan.json'),
        topology_seed=TOPOLOGY, seeds=[NOISE], duration_ms=60000., objective=objective,
        candidates=list(registered.values()), grids=grids, initial_ids=initial, reused_ids=reused,
        boundary_extension=dict(previous_offsets_mm=[-.5,-.25,0,.25,.5], offsets_mm=offsets, step_mm=.75, reason='Core A minimum on previous grid boundary; larger conditional response, not an optimum claim'),
        budget=dict(grid_conditions=49, grid_new_runs=len(initial), reused_runs=len(reused), max_recombination_runs=0,
            max_new_runs=len(initial), additional_seeds=0),
        resources=dict(max_workers=24, projected_per_tree_GiB=6, per_tree_limit_GiB=18, reserve_GiB=60),
        recombination='None: this is a conditional response grid, no adaptive optimization claimed.',
        contract='Fixed background, EE weight and angle; only core XY varies. Existing label-free objective retained for reporting; no label-guided selection. TA/TB and new joint feature only diagnose. Fixed seed is not identical per-time innovation when masks change. No model freeze or Fig5.',
        created_unix=time.time())
    rt.write(OUT/'plan.json', plan)
    rt.write(OUT/'confirmation/frozen_networks.json', dict(replication_networks={}))
    rt.write(OUT/'status.json', dict(status='PREPARED', complete=len(reused), active=[], budget=plan['budget'], time=time.time()))
    return plan


def worker(cid):
    p = rt.read(OUT/'plan.json'); c = rt.read(OUT/'candidates'/f'{cid}.json')
    validate(c, rt.read(OUT/'candidates'/f'{REF}.json'))
    old.physical.OUT = OUT; old.physical.configure()
    old.physical.run.worker(STAGE, cid, TOPOLOGY, NOISE, p['duration_ms'])


def combine(plan, scores):
    return dict(ids=[])


def controller():
    import psutil
    from scripts.analyze_topic4_three_observable_bo import audit_applied
    p = prepare(); configure_score(); dense.gpu_device = lambda: 'cpu'; active = {}; failures = []; next_score = 0.; next_plot = 0.
    audited = set(); max_rss = {}; analysis_proc = None
    # Adopt only this pilot's workers on controller restart.
    for q in psutil.process_iter(['pid', 'cmdline']):
        cmd = q.info['cmdline'] or []
        if str(SCRIPT) in cmd and 'worker' in cmd and '--candidate' in cmd:
            active[q.pid] = (q, cmd[cmd.index('--candidate')+1], None)
    def alive(q):
        if isinstance(q, subprocess.Popen): return q.poll() is None
        try: return q.is_running() and q.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess: return False
    def stop_tree(q):
        try:
            for child in reversed([q]+q.children(recursive=True)): child.terminate()
        except psutil.NoSuchProcess: pass
    psutil.cpu_percent()
    while True:
        detail=[]
        for pid, (proc, cid, log) in list(active.items()):
            if not alive(proc):
                if log: log.close()
                active.pop(pid)
                if not old.valid_complete(result(cid)): failures.append(dict(candidate=cid, reason='NO_VALID_RESULT'))
                continue
            try:
                q=psutil.Process(pid); rss=sum(c.memory_info().rss for c in [q]+q.children(recursive=True))/2**30
            except psutil.NoSuchProcess: continue
            max_rss[cid]=max(max_rss.get(cid,0),rss)
            detail.append(dict(candidate=cid, pid=pid, rss_GiB=rss))
            applied=result(cid).parents[1]/'applied_physics.json'
            if applied.exists() and cid not in audited:
                actual=rt.read(applied); candidate=rt.read(OUT/'candidates'/f'{cid}.json')
                audit_applied(actual, candidate); assert old.physics(actual['candidate']) == old.physics(candidate)
                audited.add(cid)
            if rss > 18:
                stop_tree(q); failures.append(dict(candidate=cid, reason='PROCESS_TREE_MEMORY_LIMIT'))
        if time.time() >= next_score and not failures:
            dense.score_ready(); next_score=time.time()+120
        scores={r['candidate']:r for r in dense.score_rows()}
        grid_done=all(c in scores for c in p['initial_ids'])
        proposal=combine(p,scores) if grid_done and not failures else None
        queue=p['initial_ids']+(proposal['ids'] if proposal else [])
        pending=[c for c in queue if c not in scores and c not in {v[1] for v in active.values()} and not old.valid_complete(result(c))]
        available=psutil.virtual_memory().available/2**30; cpu=psutil.cpu_percent()
        projected=sum(max(0,6-r['rss_GiB']) for r in detail)
        cap=24 if len(audited)>=2 or sum(not r.get('reused') for r in scores.values())>=2 else 2
        if pending and not failures and len(active)<cap and available-projected>66 and cpu<94:
            cid=pending[0]; log=(OUT/'logs'/f'{cid}.log').open('a')
            proc=subprocess.Popen([rt.PYTHON,'-u',str(SCRIPT),'worker','--candidate',cid],cwd=ROOT,env=rt.ENV,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active[proc.pid]=(proc,cid,log)
            with (OUT/'dispatch.jsonl').open('a') as f: f.write(json.dumps(dict(candidate=cid,pid=proc.pid,time=time.time(),available_GiB=available))+'\n')
        if available<28 and detail:
            largest=max(detail,key=lambda r:r['rss_GiB']); stop_tree(psutil.Process(largest['pid']))
            failures.append(dict(reason='SYSTEM_MEMORY_EMERGENCY',candidate=largest['candidate']))
        done=proposal is not None and all(c in scores for c in queue) and not active
        if analysis_proc and analysis_proc.poll() is not None:
            if analysis_proc.returncode: failures.append(dict(reason='ANALYSIS_FAILURE',returncode=analysis_proc.returncode))
            analysis_proc=None
        if not failures and analysis_proc is None and (time.time()>=next_plot or done):
            log=(OUT/'logs/analysis.log').open('a')
            analysis_proc=subprocess.Popen([old.BO_PY,'-u',str(ROOT/'scripts/analyze_topic4_xy_wide_response.py')],cwd=ROOT,env={**rt.ENV, 'CUDA_VISIBLE_DEVICES': ''},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            log.close();next_plot=time.time()+600
        status=dict(status='DRAINING_FAILURE' if failures else 'RUNNING',active=detail,complete=len(scores),
            new_complete=sum(not r.get('reused') for r in scores.values()),reused=len(p['reused_ids']),
            queued=len(pending),grid_done=grid_done,recombination_proposed=proposal is not None,
            application_audited=sorted(audited),peak_process_tree_GiB=max_rss,available_GiB=available,failures=failures,time=time.time())
        rt.write(OUT/'status.json',status)
        if done and not failures:
            if analysis_proc: analysis_proc.wait()
            if analysis_proc and analysis_proc.returncode: raise RuntimeError('Final analysis failed')
            status.update(status='GENERATING_FINAL_KEY_RESULTS',active=[])
            rt.write(OUT/'status.json',status)
            with (OUT/'logs/analysis.log').open('a') as log:
                subprocess.run([old.BO_PY,'-u',str(ROOT/'scripts/analyze_topic4_xy_wide_response.py'),'--final-media'],
                    cwd=ROOT,env={**rt.ENV, 'CUDA_VISIBLE_DEVICES': ''},stdout=log,stderr=subprocess.STDOUT,check=True)
            status.update(status='COMPLETE_PENDING_SCIENTIFIC_REVIEW',active=[])
            rt.write(OUT/'status.json',status); rt.write(OUT/'completion.json',status); return
        if failures and not active: raise RuntimeError(json.dumps(failures))
        time.sleep(2)


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('action', choices=['prepare','controller','worker','score']); ap.add_argument('--candidate')
    a=ap.parse_args()
    if a.action=='prepare': print(json.dumps(prepare()['budget']))
    elif a.action=='worker': worker(a.candidate)
    elif a.action=='score': configure_score(); print(dense.score_ready())
    else:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'controller.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); controller()
