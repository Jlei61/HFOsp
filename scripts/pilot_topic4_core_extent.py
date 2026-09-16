"""Bounded radius-first pilot; frozen R1 physics/readout and deferred local proposals."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src/snn_engine')]
import numpy as np
from src import topic4_initial_state_runtime as rt
from src.topic4_core_field_rev9 import reconstruct_node_from_h, array_sha256
from src.topic4_continuous_core_state import spatial_groups, ContinuousIState, ou_path
from scripts import run_topic4_continuous_core_state_r1 as r1

OUT = Path('/data/hfosp/topic4_sef_hfo/core_extent_state_pilot_20260909')
PARENT = Path('/data/hfosp/topic4_sef_hfo/continuous_core_state_r1_20260909')
SCRIPT = Path(__file__).resolve()


def geometry_field(positions, centers, radii):
    d = np.linalg.norm(np.asarray(positions)[:, None]-np.asarray(centers)[None], axis=2)
    within = d <= np.asarray(radii)[None]
    return within.any(1).astype(float), np.argmin(d, axis=1)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'plan.json').exists():
        return
    design = rt.read(PARENT/'design.json')
    centers = rt.candidate_record(design)['node_field']['centers_mm']
    radius = rt.read(PARENT/'workers/ou_dyn847101.json')['radius_mm']
    baseline = dict(id='baseline', centers_mm=centers, radii_mm=[radius, radius],
                    preserve_exact_baseline=True, dose_matched=False, stage='reference')
    candidates = [baseline]
    for radius_new in [2.5, 4.0]:
        for sides in ['A', 'B', 'AB']:
            c = copy.deepcopy(baseline)
            c.update(id=f'expand_{sides}_{radius_new:g}', stage='A', preserve_exact_baseline=False)
            c['radii_mm'] = [radius_new if name in sides else radius for name in ['A','B']]
            candidates.append(c)
    plan = dict(parent_design=design, candidates=candidates, max_parallel=8,
                training_seeds=[847101,847102], confirmation_seed=847103,
                duration_ms=30000., proposal_seed=9092601, local_proposals=4,
                maximum_new_runs=23,
                score='frozen v2.1 L_off; equal noise-run weights; N>=16 per complete run; no partial pooling',
                stage_B='4 proposals from the two lowest scored A/reference conditions, frozen after A completes; radius +/-0.35 mm and centers +/-0.5 mm; no validation-based selection',
                confirmation='one selected condition with third noise; baseline third replay reused',
                dose_control='selected non-baseline condition, separately match positive/negative threshold totals to baseline, two training noises',
                scientific_scope='single topology; development comparison, no mechanism acceptance from score',
                physics='only E threshold support changes; I state indices/loading, graph, noise law, static threshold quantiles, GABA and readout fixed',
                stop='finish bounded stages and plots; no further search or Fig5',
                source_sha256=rt.sha(SCRIPT))
    rt.write(OUT/'plan.json', plan)
    (OUT/'execution_plan.md').write_text('''# Core 范围与中心微调 pilot

固定 R1 手放中心、拓扑 2511、30 秒时长与外加 OU 状态规律。第一批分别扩大 A、B、两核至 2.5/4 mm，原约 1.75 mm 基线复用已完成轨迹。扩大阈值场不扩大 I 状态作用范围。

两条配对噪声用于开发：847101、847102。使用原冻结 L_off 选择后续出发点；完整运行少于 16 个合格事件不取得正式分数，不能以跨运行拼接补足。损失允许负值。参与、时差、逐模式顺序和原生场用于独立解释，不用于暗中改变提名。

第一批结束后，按损失选两个出发点，固定四个随机局部提案：中心各坐标 ±0.5 mm、半径 ±0.35 mm。半径限于 [原基线半径,4.5] mm，中心偏移相对原中心不超过 0.75 mm；保留实际边界裁切，记录 core 重叠。第二批全部结束后提名。

提名后用第三条噪声确认，并对非基线提名补两个正/负阈值总量分别匹配的对照。最多 23 次新仿真。若无可评分条件，跳过依赖排名的阶段，仍交付全部失败/低事件条件，不伪造损失。

最多 8 worker，保留 60 GiB 可用内存，并按每个活动 worker 24 GiB 未占用余量保守派发；单 worker 进程树超过 28 GiB 或机器可用内存低于 35 GiB 时停止新派发并终止超限/最新进程树，记录执行失败。物理 runaway 单独保留。运行完自动汇总表、参数图、原生多事件 GIF 和科学说明，停在审阅点。
''')


def build_modified(design, seed, candidate):
    sub, cand, transition, execution, audit, network = rt.build_frozen_substrate(design, design['topology_seed'], seed)
    frozen_identity = rt.static_identity(sub, audit)
    groups0, indices, loading, radius0 = spatial_groups(sub, cand['node_field']['centers_mm'])
    old_delta = sub.delta_vtheta.copy()
    if not candidate.get('preserve_exact_baseline'):
        h, nearest = geometry_field(sub.positions_e, candidate['centers_mm'], candidate['radii_mm'])
        node = reconstruct_node_from_h(h, n_total=sub.n_e+sub.n_i,
            quantile_seed=sub.stage['quantile_seed'], core_mean=sub.engine['core_mean'],
            core_std=sub.engine['core_std'], v_base=sub.engine['v_base'])
        if candidate.get('dose_matched'):
            d = node['delta_vtheta'].copy()
            for sign in [-1,1]:
                selected = d*sign > 0
                target = abs(old_delta[old_delta*sign > 0].sum())
                d[selected] *= target/abs(d[selected].sum())
            node['delta_vtheta'] = d
            node['vtheta'][:sub.n_e] = sub.engine['v_base']+d
        sub.h_e = node['h']; sub.vtheta = node['vtheta']; sub.delta_vtheta = node['delta_vtheta']
    else:
        _, nearest = geometry_field(sub.positions_e, candidate['centers_mm'], candidate['radii_mm'])
    # State I support is calculated BEFORE E-field mutation and is invariant.
    groups = dict(groups0)
    for k, name in enumerate(['coreA','coreB']):
        groups[name+'E'] = np.flatnonzero((sub.h_e > 0) & (nearest == k))
    groups['surroundE'] = np.flatnonzero(sub.h_e == 0)
    identity = rt.static_identity(sub, audit)
    fixed_keys = [k for k in identity if k.startswith(('positions_', 'ampa_', 'gaba_'))]
    assert all(identity[k] == frozen_identity[k] for k in fixed_keys)
    delta = sub.delta_vtheta
    details = dict(candidate=candidate, n_modulated=int((sub.h_e>0).sum()),
        threshold_lowering_total_mV=float(-delta[delta<0].sum()),
        threshold_raising_total_mV=float(delta[delta>0].sum()),
        baseline_lowering_total_mV=float(-old_delta[old_delta<0].sum()),
        baseline_raising_total_mV=float(old_delta[old_delta>0].sum()),
        frozen_I_indices_sha256=array_sha256(indices), frozen_I_loading_sha256=array_sha256(loading),
        baseline_identity=frozen_identity, applied_identity=identity,
        boundary_clipped=[bool(min(*xy,20-xy[0],20-xy[1])<r) for xy,r in zip(candidate['centers_mm'],candidate['radii_mm'])],
        disk_overlap=bool(np.linalg.norm(np.diff(candidate['centers_mm'],axis=0))<sum(candidate['radii_mm'])))
    cand = copy.deepcopy(cand); cand['candidate_id'] = candidate['id']
    cand['node_field']['centers_mm'] = candidate['centers_mm']
    return (sub,cand,transition,execution,audit,network), (groups,indices,loading,radius0), details


def run_worker(candidate_path, seed, duration=None):
    plan = rt.read(OUT/'plan.json')
    if rt.sha(SCRIPT) != plan['source_sha256']:
        raise RuntimeError('pilot implementation differs from frozen plan')
    candidate = rt.read(candidate_path)
    unit = OUT/'units'/candidate['id']/str(seed); unit.mkdir(parents=True,exist_ok=True)
    design = copy.deepcopy(plan['parent_design']); design['output_root']=str(unit)
    sourcejob = next(j for j in design['jobs'] if j['id']==f'ou_dyn{seed}')
    job = copy.deepcopy(sourcejob); job['id']='trajectory'; job['duration_ms']=duration or plan['duration_ms']
    design['jobs']=[job]; design['pilot_candidate']=candidate
    path=unit/'design.json'
    if path.exists() and rt.read(path)!=design:
        raise RuntimeError('resume changes physical unit')
    rt.write(path,design)
    if (unit/'workers/trajectory.json').exists():
        old=rt.read(unit/'workers/trajectory.json')
        if old['status']=='COMPLETE' and old['design_sha256']==rt.sha(path) and rt.sha(unit/'workers/trajectory.npz')==old['arrays_sha256']:
            return
        raise RuntimeError('invalid existing result')
    built, frozen_control, details = build_modified(design,seed,candidate)
    rt.write(unit/'applied_geometry.json',details)
    r1.DESIGN=path
    r1.build=lambda d,s: built
    def control_for(sub,cand,d,j,t):
        groups,indices,loading,radius=frozen_control
        dt=sub.params.dt
        z=ou_path(round(t/dt),dt,d['state']['tau_ms'],j['state_seed'])
        co=ContinuousIState(indices,loading,z,amplitude=d['state']['amplitude'],dt_ms=dt,
            seed=j['coupling_seed'],warmup_ms=d['state']['warmup_ms'],ramp_ms=d['state']['ramp_ms'])
        return co,groups,radius
    r1.control_for=control_for
    rt.write(unit/'qualification.json',dict(status='PASS', qualification_scope='static graph and frozen I projection; shared R1 dynamic qualification; pilot canary checked separately',
        static_array_identity=details['applied_identity'],source_hashes=rt.loaded_source_hashes()))
    r1.worker(design,job)


def path_for(candidate, seed):
    if candidate['id']=='baseline':
        return PARENT/'workers'/f'ou_dyn{seed}.json'
    return OUT/'units'/candidate['id']/str(seed)/'workers/trajectory.json'


def score_candidate(candidate, seeds):
    plan=rt.read(OUT/'plan.json'); objective=rt.load_objective(plan['parent_design'])
    units=[]
    for seed in seeds:
        path=path_for(candidate,seed)
        if not path.exists():
            units.append(dict(seed=seed,status='MISSING',loss_off=None));continue
        r=rt.read(path)
        with np.load(path.with_suffix('.npz')) as z:
            ids=[i for i in z['primary_event_indices'] if r['events'][i]['window_ms'][0]>=1500 and r['events'][i]['window_ms'][1]<=30000]
            table=z['centroid_ms'][ids]
        score=objective.score_network(table)
        if r['actual_duration_ms']<30000 or r['physical_status']=='RUNAWAY':
            score={**score,'status':'PHYSICAL_RUNAWAY_OR_INCOMPLETE','loss_off':None}
        units.append(dict(seed=seed,path=str(path),**score))
    loss=float(np.mean([u['loss_off'] for u in units])) if all(u['loss_off'] is not None for u in units) else None
    return dict(candidate=candidate,units=units,loss_off=loss)


def run_batch(candidates, seeds, stage):
    import psutil
    plan=rt.read(OUT/'plan.json'); active={}; pending=[]; failed=[]
    (OUT/'candidates').mkdir(exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
    for c in candidates:
        if c['id']=='baseline':continue
        cp=OUT/'candidates'/(c['id']+'.json')
        if cp.exists() and rt.read(cp)!=c:raise RuntimeError('proposal changed on restart')
        rt.write(cp,c)
        for seed in seeds:pending.append((c,seed,cp))
    while pending or active:
        tree_rss={}
        for pid,(p,c,seed,stream) in list(active.items()):
            if p.poll() is not None:
                stream.close();del active[pid]
                if p.returncode:failed.append(dict(candidate=c['id'],seed=seed,exit_code=p.returncode))
                continue
            try:
                proc=psutil.Process(pid);family=[proc]+proc.children(recursive=True)
                rss=sum(x.memory_info().rss for x in family if x.is_running())/2**30
                tree_rss[pid]=rss
                if rss>28 or rt.available_gib()<35:
                    for x in reversed(family):x.terminate()
                    failed.append(dict(candidate=c['id'],seed=seed,reason='RESOURCE_GUARD',tree_rss_gib=rss))
            except psutil.NoSuchProcess:pass
        if failed:pending=[]
        headroom=sum(max(0,24-tree_rss.get(pid,0)) for pid in active)
        while pending and len(active)<plan['max_parallel'] and rt.available_gib()>60+headroom+24:
            c,seed,cp=pending.pop(0);log=(OUT/'logs'/f'{c["id"]}_{seed}.log').open('a')
            p=subprocess.Popen([rt.PYTHON,'-u',str(SCRIPT),'worker','--candidate',str(cp),'--seed',str(seed)],cwd=ROOT,env=rt.ENV,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active[p.pid]=(p,c,seed,log);headroom+=24
        rt.write(OUT/'status.json',dict(status='RUNNING' if not failed else 'STOPPING_AFTER_ENGINEERING_FAILURE',stage=stage,
            active=[dict(pid=pid,candidate=c['id'],seed=s,tree_rss_gib=tree_rss.get(pid)) for pid,(p,c,s,f) in active.items()],
            queued=len(pending),failed=failed,available_gib=rt.available_gib(),updated_unix=time.time()))
        if active or pending:time.sleep(10)
    if failed:raise RuntimeError(f'failed batch: {failed}')


def controller():
    prepare();plan=rt.read(OUT/'plan.json');seeds=plan['training_seeds'];candidates=plan['candidates']
    if rt.read(OUT/'canary_audit.json')['status']!='PASS':
        raise RuntimeError('physical canary audit not passed')
    run_batch(candidates,seeds,'A_RADIUS')
    scores=[score_candidate(c,seeds) for c in candidates]
    rt.write(OUT/'stage_A_scores.json',scores)
    ranked=sorted([s for s in scores if s['loss_off'] is not None],key=lambda s:s['loss_off'])
    if ranked:
        pp=OUT/'stage_B_proposals.json'
        if pp.exists():proposals=rt.read(pp)
        else:
            rng=np.random.default_rng(plan['proposal_seed']);proposals=[]
            for i in range(plan['local_proposals']):
                parent=ranked[i%min(2,len(ranked))]
                c=copy.deepcopy(parent['candidate'])
                c.update(id=f'local_{i:02d}',stage='B',preserve_exact_baseline=False,
                         parent_id=c['id'],parent_loss_off=parent['loss_off'])
                centers=np.asarray(c['centers_mm'])+rng.uniform(-.5,.5,(2,2))
                anchor=np.asarray(candidates[0]['centers_mm'])
                c['centers_mm']=np.clip(centers,anchor-.75,anchor+.75).tolist()
                c['radii_mm']=np.clip(np.asarray(c['radii_mm'])+rng.uniform(-.35,.35,2),candidates[0]['radii_mm'][0],4.5).tolist()
                proposals.append(c)
            rt.write(pp,proposals)
        run_batch(proposals,seeds,'B_LOCAL')
        candidates=candidates+proposals;scores=[score_candidate(c,seeds) for c in candidates]
        ranked=sorted([s for s in scores if s['loss_off'] is not None],key=lambda s:s['loss_off'])
        selected=ranked[0]['candidate'];rt.write(OUT/'nomination.json',dict(candidate=selected,selection='training L_off only',scores=scores))
        run_batch([selected],[plan['confirmation_seed']],'C_NEW_NOISE')
        if selected['id']!='baseline':
            matched=copy.deepcopy(selected);matched.update(id='selected_dose_matched',stage='dose_control',dose_matched=True,preserve_exact_baseline=False)
            run_batch([matched],seeds,'D_DOSE_CONTROL');candidates.append(matched)
    rt.write(OUT/'all_candidates.json',candidates)
    rt.write(OUT/'scores.json',[score_candidate(c,seeds) for c in candidates])
    status=dict(status='ANALYZING',updated_unix=time.time());rt.write(OUT/'status.json',status)
    p=subprocess.run([rt.PYTHON,str(ROOT/'scripts/paper_figures/plot_topic4_core_extent_pilot.py')],cwd=ROOT,env=rt.ENV)
    status.update(status='ROUND_COMPLETE_PENDING_SCIENTIFIC_REVIEW' if p.returncode==0 else 'ANALYSIS_FAILED',analysis_exit_code=p.returncode,updated_unix=time.time())
    rt.write(OUT/'status.json',status)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','worker','controller'])
    ap.add_argument('--candidate',type=Path);ap.add_argument('--seed',type=int);ap.add_argument('--duration',type=float)
    args=ap.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='worker':run_worker(args.candidate,args.seed,args.duration)
    else:
        try:controller()
        except Exception as exc:
            rt.write(OUT/'status.json',dict(status='FAILED',error=repr(exc),updated_unix=time.time()))
            raise
