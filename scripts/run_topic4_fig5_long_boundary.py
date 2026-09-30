#!/usr/bin/env python3
"""One bounded fixed-seed campaign: eight extensions and eleven boundary points."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import copy
import csv
import fcntl
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import psutil
from scipy.interpolate import PchipInterpolator
import run_topic4_fig5_log_m_scan as engine
import run_topic4_fig5_entry_extension as continuation
from analyze_topic4_fig5_entry_progress import audit_counts

base=engine.base
ROOT=engine.ROOT
OUT=ROOT/'results/topic4_sef_hfo/fig5_long_boundary_20260923'
STORAGE=Path('/data/hfosp/topic4_sef_hfo/fig5_long_boundary_20260923')
PARENT=ROOT/'results/topic4_sef_hfo/fig5_boundary_refinement_20260917'
PAPER=ROOT/'results/paper-ready-figure/fig5'
SEED=9108401
HORIZON=3000.


def prepare():
    if not OUT.exists():
        STORAGE.mkdir(parents=True,exist_ok=True)
        OUT.symlink_to(STORAGE,target_is_directory=True)
    if (OUT/'protocol.json').exists():
        return base.read(OUT/'protocol.json')
    parent=base.read(PARENT/'protocol.json')
    snapshot=base.read(PAPER/'source_snapshot.json')
    grid=snapshot['grid']
    assert grid['all_complete'] and grid['progress_summary']['new_complete']==24
    surface=grid['continuous_surface']
    assert surface['n_total']==59 and surface['n_entered']==27
    for path,digest in parent['source_hashes'].items():
        assert base.sha(path)==digest,path
    assert base.sha(engine.__file__)==parent['producer_sha256']
    assert parent['inherited_checkpoint_qa']['entire_executor_state_exact']
    records=grid['base_grid']['records']+grid['refinement_records']
    brackets=surface['boundary_brackets']
    sources={};extended=[];fresh=[];design=[]
    for bracket in brackets:
        record=next(r for r in records if r['job']['tau_M_s']==bracket['tau_M_s']
                    and r['job']['eta_m']==bracket['smallest_censored_eta_M'])
        source=Path(record['source']);result=base.read(source/'result.json')
        assert not result['event_observed'] and result['elapsed_s']==1000
        assert result['identity']==parent['identity']
        job=dict(result['job'],horizon_s=HORIZON)
        extended.append(job)
        sources[job['name']]=dict(source=str(source),original_job=result['job'],
            result_sha256=base.sha(source/'result.json'),checkpoint_sha256=base.sha(source/'checkpoint.pkl'))
        design.append(dict(name=job['name'],kind='continue_nearest_censored',
            tau_M_s=job['tau_M_s'],eta_M=job['eta_m'],start_s=1000.,horizon_s=HORIZON))
    specifications=[(b['tau_M_s'],round(b['log_midpoint_eta_M'],12),'eta_bracket_midpoint') for b in brackets]
    curve=PchipInterpolator(np.log10([b['tau_M_s'] for b in brackets]),
                            np.log10([b['log_midpoint_eta_M'] for b in brackets]),extrapolate=False)
    for logtau in (1.5,2.5,3.5):
        specifications.append((round(10**logtau,12),round(float(10**curve(logtau)),12),'tau_gap_midpoint'))
    old_pairs={(r['job']['tau_M_s'],r['job']['eta_m']) for r in records}
    for index,(tau,eta,kind) in enumerate(specifications):
        assert (tau,eta) not in old_pairs
        name=f'eta{eta:.10g}_tau{tau:.10g}_s{SEED}'
        job=dict(name=name,eta_m=eta,tau_M_s=tau,seed=SEED,tau_z_ms=5000.,
                 threshold=base.old.THRESHOLD,horizon_s=HORIZON,device=index%2)
        fresh.append(job)
        design.append(dict(name=name,kind=kind,tau_M_s=tau,eta_M=eta,start_s=0.,horizon_s=HORIZON))
    # Interleave old continuations and new points so both questions progress.
    jobs=[]
    for i in range(max(len(extended),len(fresh))):
        if i<len(extended):jobs.append(extended[i])
        if i<len(fresh):jobs.append(fresh[i])
    assert len(extended)==8 and len(fresh)==11 and len({j['name'] for j in jobs})==19
    shutil.copy2(PAPER/'source_snapshot.json',OUT/'previous_1000s_snapshot.json')
    protocol=dict(status='DEFINED_BEFORE_LONG_FOLLOWUP',jobs=jobs,sources=sources,
        seed=SEED,seeds=[SEED],topology_seed=6101,new_jobs=19,continuations=8,fresh_points=11,
        prior_unique_points=59,total_unique_points_after_batch=70,horizon_s=HORIZON,
        identity=parent['identity'],source_hashes=parent['source_hashes'],
        producer_sha256=parent['producer_sha256'],long_producer_sha256=base.sha(__file__),
        continuation_helper_sha256=base.sha(continuation.__file__),
        inherited_checkpoint_qa=parent['inherited_checkpoint_qa'],
        previous_snapshot_sha256=base.sha(OUT/'previous_1000s_snapshot.json'),
        endpoint=parent['endpoint'],colorbar=parent['colorbar'],design=design,
        max_workers=12,minimum_available_memory_GiB=84,worker_budget_GiB=4,
        approval='2026-09-23 user requested longer simulations at nodes near the boundary to clarify it; same single-seed preference persists.',
        question='Does delayed entry after1000s move the observed boundary, and do new bracket/gap points narrow its finite-time location?',
        intervention='Only observation horizon3000s and11 newly specified eta/tau combinations. Existing8 continue exact1000s checkpoints. All neural/synaptic/delay/OU/RNG/Z/M states unchanged; no reset or new seed.',
        display='Retain1-1000s log colorbar. Update the common1000s boundary from eligible records. Draw a separate3000s boundary only where both entered and3000s non-entered supports exist; do not relabel untouched1000s censorings. Entries above1000s get explicit markers and actual times inCSV.',
        stop='Exactly19 jobs; stop each at first confirmation or3000s. No automatic new seeds, further points, or horizon extension. One checkpoint retry per failed job, then drain and report.',
        created_at=time.time())
    base.write(OUT/'protocol.json',protocol)
    for job in jobs:base.write(OUT/'jobs'/(job['name']+'.json'),job)
    with (OUT/'parameter_points.csv').open('w') as file:
        w=csv.DictWriter(file,fieldnames=list(design[0]));w.writeheader();w.writerows(design)
    (OUT/'execution_plan.md').write_text(
        '# Fig5边界附近长时单种子实验\n\n'
        '科学问题：1000秒未进入的边界邻点是否只是进入较晚；沿ηM和τM插入实测点后，进入边界能定位到多窄的区间。\n\n'
        '固定seed9108401、拓扑6101及全部背景动力学，观察上限从1000秒提高到3000秒。8个原τM截面各选最靠近边界的未进入点，从原1000秒检查点接续；8个夹区几何中点从原生初态运行；τM约31.6、316、3162秒各补1个原连续边界附近点，共19条、11个新参数组合。参数清单见parameter_points.csv。\n\n'
        '进入仍按全体E的10ms分箱发放率≥200Hz持续200ms判定，并记录首次确认时间。沿用既有完整执行器续跑一致性验证；逐一核对8个1000秒检查点和原始计数连续性，复制后核对全部执行器/追踪器状态。只修改停止时限，原数据保留。\n\n'
        '最多12条并发，至少保留84GiB系统可用内存。每条到首次确认或3000秒停止；失败最多按检查点重试1次，不自动增加参数、种子或观察窗。原1000秒图保留，结果写入本目录并每10分钟/新增终点刷新候选图。\n\n'
        '解释：若出现1000秒后进入，报告真实进入时间及3000秒边界的移动；若未进入，报告3000秒下界。未延长的旧点仍只有1000秒观察证据，不混入3000秒未进入集合。colorbar保持viridis、log1–1000秒及原刻度，晚进入另用标记和表格给出数值。连续线是采样间估计；该批不检验多种子稳定性或严格分岔。只在两侧均有合格实测支持的截面连接长时边界，缺少上/下夹点处留断，不外推。\n')
    return protocol


def seed_all(protocol):
    continuation.OUT=OUT
    checks=[]
    for name,source_ref in protocol['sources'].items():
        source=Path(source_ref['source']);folder=OUT/'runs'/name
        if (folder/'continuation.json').exists():
            checks.append(base.read(folder/'continuation_qa.json'))
            continue
        values=audit_counts(source)
        assert values['first_entry'] is None and values['followup_s']==1000
        assert base.sha(source/'result.json')==source_ref['result_sha256']
        job=base.read(OUT/'jobs'/(name+'.json'))
        continuation.seed_checkpoint(name,source,job,source_ref['original_job'],source_ref['checkpoint_sha256'])
        original=base.load_pickle(source/'checkpoint.pkl')
        copied=base.load_pickle(folder/'checkpoint.pkl')
        continuation.same(original['engine'],copied['engine'])
        continuation.same(original['tracker'],copied['tracker'])
        assert original['engine']['step']==10000000
        after=audit_counts(folder)
        assert after==values
        check=dict(name=name,status='PASS',resume_s=1000.,whole_engine_exact=True,
                   whole_tracker_exact=True,counts_exact=True,only_job_change='horizon_s1000->3000')
        base.write(folder/'continuation_qa.json',check);checks.append(check)
        del original,copied
    assert len(checks)==8
    base.write(OUT/'continuation_qa.json',dict(status='PASS',checks=checks,
        inherited_full_executor_replay=protocol['inherited_checkpoint_qa']))


def worker(name):
    protocol=prepare()
    assert base.sha(__file__)==protocol['long_producer_sha256']
    assert base.sha(continuation.__file__)==protocol['continuation_helper_sha256']
    if name in protocol['sources']:
        assert base.read(OUT/'runs'/name/'continuation_qa.json')['status']=='PASS'
    assert base.read(OUT/'jobs'/(name+'.json'))['seed']==SEED
    engine.OUT=OUT;engine.prepare=lambda:protocol
    engine.worker(name)


def launch(name):
    folder=OUT/'runs'/name;folder.mkdir(parents=True,exist_ok=True)
    with (folder/'worker.log').open('a') as log:
        return subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'worker','--name',name],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)


def checkpoint_for_retry(name):
    folder=OUT/'runs'/name
    saved=base.load_pickle(folder/'checkpoint.pkl')
    step=int(saved['engine']['step']);quarantine=folder/'retry1_preserved'
    quarantine.mkdir(exist_ok=False)
    marker=folder/'failure.json'
    if marker.exists():marker.rename(quarantine/'failure.json')
    for path in sorted((folder/'chunks').glob('*.npz')):
        if '.tmp.' in path.name:
            path.rename(quarantine/path.name);continue
        with np.load(path) as data:
            end=int(data['end_step']);start=int(data['start_step'])
        if end>step:
            assert start>=step
            path.rename(quarantine/path.name)
    values=audit_counts(folder)
    assert values['followup_s']==step*.0001 and values['first_entry'] is None
    base.write(quarantine/'retry.json',dict(resume_s=step*.0001,exact_checkpoint=True))


def report():
    import analyze_topic4_fig5_long_boundary as analysis
    analysis.report()


def supervise():
    protocol=prepare()
    lock=(OUT/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    seed_all(protocol)
    pending=[j['name'] for j in protocol['jobs'] if not (OUT/'runs'/j['name']/'result.json').exists()]
    children={};failed=[];retried=[];last_complete=-1;last_render=0
    while True:
        for name,child in list(children.items()):
            rc=child.poll()
            if rc is None:continue
            del children[name]
            if rc or not (OUT/'runs'/name/'result.json').exists():
                if name not in retried and (OUT/'runs'/name/'checkpoint.pkl').exists():
                    try:
                        checkpoint_for_retry(name);retried.append(name);pending.append(name)
                    except Exception as exc:
                        failed.append(name);base.write(OUT/'runs'/name/'retry_failure.json',dict(error=repr(exc)))
                else:failed.append(name)
        available=psutil.virtual_memory().available/2**30
        while pending and not failed and len(children)<12 and available>84 and shutil.disk_usage(OUT).free/2**30>40:
            name=pending.pop(0);children[name]=launch(name);available-=4
        completed=sum((OUT/'runs'/j['name']/'result.json').exists() for j in protocol['jobs'])
        running={}
        for name,child in children.items():
            path=OUT/'runs'/name/'progress.json'
            progress=base.read(path) if path.exists() else {}
            running[name]=dict(pid=child.pid,time_s=progress.get('time_s'),status=progress.get('status','STARTING'))
        finished=not pending and not children
        base.write(OUT/'status.json',dict(status='DRAINING_FAILURE' if failed else 'COMPLETE_PENDING_HUMAN_REVIEW' if finished else 'RUNNING_LONG_BOUNDARY',
            pid=os.getpid(),seed=SEED,horizon_s=HORIZON,completed=completed,total=19,continuations=8,fresh_points=11,
            running=running,pending=len(pending),failed=failed,retried=retried,updated_at=time.time()))
        if completed!=last_complete or time.time()-last_render>=600 or finished:
            try:
                report()
                base.write(OUT/'figure_refresh_status.json',dict(status='UPDATED_PENDING_HUMAN_REVIEW',updated_at=time.time()))
            except Exception as exc:
                base.write(OUT/'figure_refresh_status.json',dict(status='REFRESH_FAILED',error=repr(exc),updated_at=time.time()))
            last_complete=completed;last_render=time.time()
        if failed and not children:raise RuntimeError(failed)
        if finished:return
        time.sleep(10)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','seed','worker','report','supervise']);ap.add_argument('--name');args=ap.parse_args()
    try:
        if args.mode=='prepare':prepare()
        elif args.mode=='seed':seed_all(prepare())
        elif args.mode=='worker':worker(args.name)
        elif args.mode=='report':report()
        else:supervise()
    except Exception as exc:
        path=OUT/'runs'/args.name/'failure.json' if args.mode=='worker' else OUT/'supervisor_failure.json'
        base.write(path,dict(error=repr(exc),updated_at=time.time()))
        raise
