#!/usr/bin/env python3
"""One retained noise seed; one additional decade on each upper parameter edge."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import fcntl
from pathlib import Path
import shutil
import subprocess
import sys
import time
import psutil
import run_topic4_fig5_log_m_scan as engine

base=engine.base
ROOT=engine.ROOT
OUT=ROOT/'results/topic4_sef_hfo/fig5_single_seed_wide_m_20260916'
STORAGE=Path('/data/hfosp/topic4_sef_hfo/fig5_single_seed_wide_m_20260916')
EXT=ROOT/'results/topic4_sef_hfo/fig5_log_m_entry_extension_20260915'
SEED=9108401


def prepare():
    if not OUT.exists():
        STORAGE.mkdir(parents=True,exist_ok=True);OUT.symlink_to(STORAGE,target_is_directory=True)
    if (OUT/'protocol.json').exists():
        return base.read(OUT/'protocol.json')
    old=base.read(EXT/'protocol.json')
    amendment=base.read(EXT/'single_seed_amendment_20260916.json')
    references=[];adopted=[]
    for ref in old['references']:
        if ref['job']['seed']!=SEED:continue
        references.append(dict(ref,event_observed=True,followup_s=ref['first_entry']['confirmation_s']))
    for job in old['jobs']:
        if job['seed']!=SEED:continue
        source=EXT/'runs'/job['name']
        if (source/'result.json').exists():
            result=base.read(source/'result.json')
            assert result['identity']==old['identity'] and result['job']==job
            references.append(dict(job=job,source=str(source),result_sha256=base.sha(source/'result.json'),
                event_observed=result['event_observed'],first_entry=result['first_entry'],followup_s=result['elapsed_s']))
        else:
            pid=amendment['adopt_running'][job['name']]['pid']
            process=psutil.Process(pid)
            assert process.is_running() and job['name'] in process.cmdline()
            adopted.append(dict(job=job,source=str(source),pid=pid,create_time=process.create_time()))
    assert len(references)+len(adopted)==24
    etas=old['eta_M']+[10.];taus=old['tau_M_s']+[10000.]
    jobs=[]
    for i,eta in enumerate(etas):
        for j,tau in enumerate(taus):
            if eta in old['eta_M'] and tau in old['tau_M_s']:continue
            jobs.append(dict(name=f'eta{eta:g}_tau{tau:g}_s{SEED}',eta_m=eta,tau_M_s=tau,seed=SEED,
                eta_index=i,tau_index=j,tau_z_ms=5000.,threshold=base.old.THRESHOLD,horizon_s=1000.,device=(i+j)%2))
    assert len(jobs)==11 and len({(j['eta_m'],j['tau_M_s']) for j in jobs})==11
    for path,digest in old['source_hashes'].items():assert base.sha(path)==digest,path
    assert base.sha(engine.__file__)==old['producer_sha256']
    qa=base.read(EXT/'qa.json')
    assert qa['status']=='PASS'
    check=next(c for c in qa['checks'] if c['seed']==SEED)
    assert check['entire_executor_state_exact'] and check['tracker_exact']
    protocol=dict(status='DEFINED_BEFORE_NEW_SINGLE_SEED_RUNS',seed=SEED,seeds=[SEED],eta_M=etas,tau_M_s=taus,
        jobs=jobs,references=references,adopted=adopted,new_jobs=11,total_cells=35,total_realizations=35,horizon_s=1000.,
        identity=old['identity'],source_hashes=old['source_hashes'],producer_sha256=old['producer_sha256'],
        single_seed_producer_sha256=base.sha(__file__),endpoint=old['endpoint'],max_workers=12,
        minimum_available_memory_GiB=84,worker_budget_GiB=4,topology_seed=6101,
        approval='2026-09-16 user requested stopping duplicate noise seeds and widening E parameter limits by another log decade.',
        intervention='Retain seed9108401, same displayed trajectory and fixed topology. Raise tau_M maximum1000->10000s and eta_M maximum1->10. Only11 new parameter cells; retain1000s stopping horizon.',
        statistics='One fixed noise realization per parameter cell. Display its first confirmation or audited censoring lower bound, never a multi-seed mean.',
        stop='Complete the5 already-running retained-seed continuations and11 new single-seed cells to first confirmation or1000s. No additional seeds or automatic grid expansion.',
        interpretation='Finite-time operational entry map; tau_M10000s exceeds the1000s horizon. No permanent-stability or exact-bifurcation claim.',
        inherited_checkpoint_qa=check,created_at=time.time())
    base.write(OUT/'protocol.json',protocol)
    for job in jobs:base.write(OUT/'jobs'/(job['name']+'.json'),job)
    (OUT/'execution_plan.md').write_text('# Fig5 单种子扩大参数范围\n\n'
        '固定噪声9108401及原拓扑，保留主图同一轨迹。停止第二种子9108402的5条续跑，并停止旧双种子调度与绘图监视器；历史结果保留。\n\n'
        '参数上限各提高一个数量级：τM=1–10000秒、ηM=0.0001–10（保留0.0005工作点）。35格，每格仅1条轨迹；复用原24格，新增外侧11格。保留第一种子的5条在跑续跑原样完成，不重启或改变状态。\n\n'
        '每条到首次全E≥200Hz持续200ms的确认时间或1000秒停止。E不再显示均值或n/2；实测时间直接标数值，未进入用完整落盘计数核对出的≥时间下界，未有数据用灰格。参数和色条保持log，绘图区正方形。\n\n'
        '复用同一种子的完整执行器续跑一致性验证，不新增其他种子。最多12条同时运行（计入已接管的5条），终点完成或每10分钟刷新完整Fig5。τM=10000秒尚未充分弛豫，未进入不能解释为永久稳定；本批结束后停止扩展。\n')
    return protocol


def worker(name):
    protocol=prepare()
    assert base.sha(__file__)==protocol['single_seed_producer_sha256']
    job=base.read(OUT/'jobs'/(name+'.json'))
    assert job['seed']==SEED
    engine.OUT=OUT;engine.prepare=lambda:protocol
    engine.worker(name)


def launch(name):
    folder=OUT/'runs'/name;folder.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    env['LD_LIBRARY_PATH']=str(Path(sys.executable).parent.parent/'lib')+os.pathsep+env.get('LD_LIBRARY_PATH','')
    with (folder/'worker.log').open('a') as log:
        return subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'worker','--name',name],
            cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)


def report():
    import analyze_topic4_fig5_single_seed_scan as analysis
    from plot_topic4_m_parameter_modes import safe
    grid=analysis.collect();base.write(OUT/'current_grid.json',safe(grid))
    dest=OUT/'figures';dest.mkdir(exist_ok=True)
    fig=analysis.plt.figure(figsize=(10,9));ax=analysis.draw(fig,fig.add_gridspec(1,1)[0],grid)
    fig.canvas.draw()
    assert abs(ax.bbox.width-ax.bbox.height)<1e-6 and ax.child_axes[0].get_yscale()=='log'
    for ext in ('png','pdf'):fig.savefig(dest/f'first_entry.{ext}',dpi=160,bbox_inches='tight')
    analysis.plt.close(fig)
    (dest/'README.md').write_text('### first_entry.png / .pdf\n同一噪声9108401下的35格首次进入图，τM上限10000秒，ηM上限10。直接显示首次确认时间；斜线及≥表示未进入的已核对时间下界，灰格尚无完整记录。\n**关注点**：每格只有一条轨迹，没有跨种子平均；共同目标观察窗1000秒，参数和色条为log，不能据此认定永久稳定。\n')
    with (OUT/'figure_refresh.log').open('a') as log:
        subprocess.run([sys.executable,str(ROOT/'scripts/plot_topic4_fig5_clean_panels.py'),'--eta','.0005','--seed',str(SEED)],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    p=grid['progress_summary']
    link=ROOT/'results/topic4_sef_hfo/fig5_preentry_event_audit_20260914/clean_panels_v4_single_seed/README.md'
    (OUT/'README.md').write_text('# Fig5 单种子扩大范围\n\n'
        f'仅使用seed9108401；35格，已到终点{p["complete"]}格，已进入{p["observed"]}格，1000秒未进入{p["censored"]}格。新增外侧参数点完成{p["new_complete"]}/11。\n\n'
        'τM=1–10000秒、ηM=0.0001–10；直接显示单次首次确认时间或≥随访下界，不再使用双种子均值和n/2。\n\n'
        f'[完整Fig5]({link}) · [参数图](figures/first_entry.png) · [执行方案](execution_plan.md) · [状态](status.json)\n\n'
        '原先5条第一种子续跑保留原进程；第二种子任务已按用户要求停止。每10分钟或新增终点刷新，本批结束后不自动增加种子或参数。候选待人工检查。\n')
    base.write(OUT/'figure_refresh_status.json',dict(status='UPDATED_PENDING_HUMAN_REVIEW',updated_at=time.time(),summary=p))


def supervise():
    protocol=prepare();lock=(OUT/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    adopted={r['job']['name']:r for r in protocol['adopted'] if not (Path(r['source'])/'result.json').exists()}
    pending=[j['name'] for j in protocol['jobs'] if not (OUT/'runs'/j['name']/'result.json').exists()]
    children={};failed=[];last_completed=-1;last_render=0
    while True:
        for name,ref in list(adopted.items()):
            if (Path(ref['source'])/'result.json').exists():del adopted[name];continue
            try:
                proc=psutil.Process(ref['pid'])
                alive=proc.is_running() and proc.status()!=psutil.STATUS_ZOMBIE and proc.create_time()==ref['create_time']
            except psutil.NoSuchProcess:alive=False
            if not alive:failed.append(name);del adopted[name]
        for name,child in list(children.items()):
            code=child.poll()
            if code is None:continue
            if code or not (OUT/'runs'/name/'result.json').exists():failed.append(name)
            del children[name]
        available=psutil.virtual_memory().available/2**30
        while pending and not failed and len(children)+len(adopted)<protocol['max_workers'] and available>84 and shutil.disk_usage(OUT).free/2**30>40:
            name=pending.pop(0);children[name]=launch(name);available-=4
        running={name:dict(pid=r['pid'],source=r['source'],adopted=True) for name,r in adopted.items()}
        running.update({name:dict(pid=child.pid,source=str(OUT/'runs'/name),adopted=False) for name,child in children.items()})
        for name,info in running.items():
            path=Path(info['source'])/'progress.json'
            data=base.read(path) if path.exists() else {}
            info.update(time_s=data.get('time_s'),status=data.get('status','STARTING'),seed=SEED)
        completed=len(protocol['references'])+sum((Path(r['source'])/'result.json').exists() for r in protocol['adopted'])+sum((OUT/'runs'/j['name']/'result.json').exists() for j in protocol['jobs'])
        finished=not pending and not running
        base.write(OUT/'status.json',dict(status='DRAINING_FAILURE' if failed else 'COMPLETE_PENDING_HUMAN_REVIEW' if finished else 'RUNNING_SINGLE_SEED',
            pid=os.getpid(),seed=SEED,total_cells=35,completed_cells=completed,new_total=11,
            new_completed=sum((OUT/'runs'/j['name']/'result.json').exists() for j in protocol['jobs']),
            running=running,pending=len(pending),failed=failed,updated_at=time.time()))
        if completed!=last_completed or time.time()-last_render>=600 or finished:
            try:
                report();last_completed=completed;last_render=time.time()
            except Exception as exc:
                base.write(OUT/'figure_refresh_status.json',dict(status='REFRESH_FAILED_RETRY_NEXT_CYCLE',error=repr(exc),updated_at=time.time()))
                last_completed=completed;last_render=time.time()
        if failed and not running:raise RuntimeError(failed)
        if finished:return
        time.sleep(10)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','worker','report','supervise']);ap.add_argument('--name');args=ap.parse_args()
    try:
        if args.mode=='prepare':prepare()
        elif args.mode=='worker':worker(args.name)
        elif args.mode=='report':report()
        else:supervise()
    except Exception as exc:
        target=OUT/'runs'/args.name/'failure.json' if args.mode=='worker' else OUT/'supervisor_failure.json'
        base.write(target,dict(error=repr(exc),updated_at=time.time()));raise
