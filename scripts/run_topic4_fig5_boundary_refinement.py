#!/usr/bin/env python3
"""Bounded24-point refinement of the single-seed Fig5 first-entry boundary."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse
import csv
import fcntl
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import psutil
import run_topic4_fig5_log_m_scan as engine

base=engine.base;ROOT=engine.ROOT
PARENT=ROOT/'results/topic4_sef_hfo/fig5_single_seed_wide_m_20260916'
OUT=ROOT/'results/topic4_sef_hfo/fig5_boundary_refinement_20260917'
STORAGE=Path('/data/hfosp/topic4_sef_hfo/fig5_boundary_refinement_20260917')
FIGURE=ROOT/'results/topic4_sef_hfo/fig5_preentry_event_audit_20260914/clean_panels_v6_boundary_20260917'
SEED=9108401


def prepare():
    if not OUT.exists():
        STORAGE.mkdir(parents=True,exist_ok=True);OUT.symlink_to(STORAGE,target_is_directory=True)
    if (OUT/'protocol.json').exists():return base.read(OUT/'protocol.json')
    prior=base.read(PARENT/'protocol.json');audit=base.read(PARENT/'endpoint_audit_complete_20260917.json')
    assert audit['status']=='PASS_ALL35_NATIVE_COUNT_ENDPOINTS'
    grid=base.read(PARENT/'current_grid.json');assert grid['all_complete']
    shutil.copy2(PARENT/'current_grid.json',OUT/'coarse_grid.json')
    references=[]
    for r in grid['records']:
        references.append(dict(job=r['job'],source=r['source'],result_sha256=base.sha(Path(r['source'])/'result.json')))
    brackets=[]
    for j,tau in enumerate(grid['tau_M_s']):
        bits=np.asarray(grid['entered'])[:,j]
        transitions=np.flatnonzero((bits[:-1]==1)&(bits[1:]==0))
        assert len(transitions)==1
        i=int(transitions[0]);brackets.append(dict(tau_M_s=tau,eta_enter=grid['eta_M'][i],eta_censored=grid['eta_M'][i+1]))
    specs=[]
    def vertical(fraction):
        for b in brackets:
            eta=10**(np.log10(b['eta_enter'])+fraction*np.log10(b['eta_censored']/b['eta_enter']))
            specs.append((round(float(eta),12),b['tau_M_s'],'eta_bracket',fraction))
    elbow_taus=[round(10**f,12) for f in (.25,.5,.75)]
    vertical(.5)
    specs.extend((.01,tau,'tau_1_to_10_elbow',None) for tau in elbow_taus)
    vertical(.25);vertical(.75)
    for eta in (round(10**-2.5,12),round(10**-1.5,12)):
        specs.extend((eta,tau,'tau_1_to_10_elbow',None) for tau in elbow_taus)
    assert len(specs)==24
    jobs=[];design=[]
    oldpairs={(r['job']['eta_m'],r['job']['tau_M_s']) for r in grid['records']}
    for index,(eta,tau,kind,fraction) in enumerate(specs):
        assert (eta,tau) not in oldpairs
        name=f'eta{eta:.8g}_tau{tau:.8g}_s{SEED}'
        job=dict(name=name,eta_m=eta,tau_M_s=tau,seed=SEED,tau_z_ms=5000.,
            threshold=base.old.THRESHOLD,horizon_s=1000.,device=index%2)
        jobs.append(job);design.append(dict(name=name,eta_M=eta,tau_M_s=tau,kind=kind,log_fraction=fraction))
    assert len({j['name'] for j in jobs})==24
    for path,digest in prior['source_hashes'].items():assert base.sha(path)==digest,path
    assert base.sha(engine.__file__)==prior['producer_sha256']
    p=dict(status='DEFINED_BEFORE_BOUNDARY_REFINEMENT',seed=SEED,seeds=[SEED],jobs=jobs,references=references,
        design=design,brackets=brackets,new_jobs=24,reused=35,total_realizations=59,horizon_s=1000.,max_workers=12,
        identity=prior['identity'],source_hashes=prior['source_hashes'],producer_sha256=prior['producer_sha256'],
        refinement_producer_sha256=base.sha(__file__),coarse_grid_sha256=base.sha(OUT/'coarse_grid.json'),
        inherited_checkpoint_qa=prior['inherited_checkpoint_qa'],endpoint=prior['endpoint'],
        colorbar=dict(cmap='viridis',scale='log',vmin=1.,vmax=1000.,ticks=[1.,10.,100.,1000.],label='Entry time / lower bound (s)'),
        approval='2026-09-17 user requested finer parameters inside the interface with unchanged colorbar; previous single-seed requirement persists.',
        design_rule='Three log-spaced interior eta points in each of5 measured entry/censored brackets (15);3 interior tau values in1-10s times3 eta values0.003162/0.01/0.031623 around the elbow (9).',
        intervention='Only eta_M and tau_M change. Same fixed topology,noise9108401,native initial state,dt0.1ms,Z5s,and1000s stopping horizon.',
        display='Retain the35 measured coarse cells as background; overlay24 actual parameter samples. Do not interpolate missing combinations; use the exact existing1-1000s log colorbar.',
        stop='Exactly24 new single-seed trajectories to first confirmation or1000s; no automatic extra waves,seeds,or horizon extension.',
        interpretation='Refines the finite1000s first-entry/non-entry boundary, not an asymptotic bifurcation.',created_at=time.time())
    base.write(OUT/'protocol.json',p)
    for j in jobs:base.write(OUT/'jobs'/(j['name']+'.json'),j)
    with (OUT/'parameter_points.csv').open('w') as file:
        w=csv.DictWriter(file,fieldnames=list(design[0]));w.writeheader();w.writerows(design)
    (OUT/'execution_plan.md').write_text('# Fig5 分界区域加密\n\n'
        '仅用噪声9108401，复用已完成35点，新增24点。每条到首次全E在10ms分箱下≥200Hz持续200ms的确认时刻或1000秒停止。\n\n'
        'τM=1秒时在ηM=0.01–0.1间插入3个对数点；τM=10/100/1000/10000秒时分别在ηM=0.001–0.01间插入3点。另在τM=1.778279/3.162278/5.623413秒处各测ηM=0.003162278/0.01/0.031622777，刻画1–10秒的转折。\n\n'
        'colorbar完全沿用viridis、log范围1–1000秒、1/10/100/1000刻度及原标签，坐标范围不扩展。E以原35点为底图，小圆点表示真实加密采样位置，白心为尚无完整记录，彩色圆点为已进入，彩色三角为未进入的随访下界；不填造未测组合。\n\n'
        '最多12条并发，复用既有同种子执行器与续跑验证。每个终点完成或每10分钟自动重绘整图；固定24点结束后停止，不自动追加种子或参数。该批定位1000秒内进入与未进入的边界。\n')
    return p


def worker(name):
    p=prepare();assert base.sha(__file__)==p['refinement_producer_sha256']
    assert base.read(OUT/'jobs'/(name+'.json'))['seed']==SEED
    engine.OUT=OUT;engine.prepare=lambda:p;engine.worker(name)


def report():
    import analyze_topic4_fig5_boundary_refinement as analysis
    from plot_topic4_m_parameter_modes import safe
    g=analysis.collect();base.write(OUT/'current_grid.json',safe(g));s=g['progress_summary']
    rows=[]
    for r in g['refinement_records']:
        rows.append(dict(name=r['job']['name'],eta_M=r['job']['eta_m'],tau_M_s=r['job']['tau_M_s'],seed=SEED,
            complete=r['complete'],event_observed=r['event_observed'],
            confirmation_s=r['first_entry']['confirmation_s'] if r['event_observed'] else None,
            followup_s=r['followup_s'],stage=r['stage']))
    with (OUT/'first_entry_points.csv').open('w') as file:
        w=csv.DictWriter(file,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    dest=OUT/'figures';dest.mkdir(exist_ok=True)
    fig=analysis.coarse.plt.figure(figsize=(10,9));ax=analysis.draw(fig,fig.add_gridspec(1,1)[0],g)
    fig.canvas.draw();assert abs(ax.bbox.width-ax.bbox.height)<1e-6
    for ext in ('png','pdf'):fig.savefig(dest/f'boundary_refinement.{ext}',dpi=160,bbox_inches='tight')
    analysis.coarse.plt.close(fig)
    (dest/'README.md').write_text('### boundary_refinement.png / .pdf\n原35点参数图上叠加24个真实加密采样位置，colorbar与上一版完全一致。白心圆尚无完整记录，彩色圆表示已进入，彩色三角表示未进入的已核对时间下界；底图数值仍来自原35点。\n**关注点**：新点没有插值成未测网格，仍为单种子、1000秒观察窗的首次进入图。\n')
    with (OUT/'figure_refresh.log').open('a') as log:
        subprocess.run([sys.executable,str(ROOT/'scripts/plot_topic4_fig5_clean_panels.py'),'--eta','.0005','--seed',str(SEED),'--boundary-refinement'],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    state='本批已完成，后台退出。' if g['all_complete'] else '后台每10分钟或新增终点更新图。'
    (OUT/'README.md').write_text('# Fig5 分界区域单种子加密\n\n'
        f'新增24点已完成{s["new_complete"]}/24，已进入{s["new_observed"]}点，1000秒未进入{s["new_censored"]}点。另复用已完成35点，固定seed9108401。\n\n'
        'colorbar保持原viridis、log1–1000秒与1/10/100/1000刻度。加密ηM方向的5个分界区间，并补τM=1–10秒的转折区。\n\n'
        f'[完整Fig5]({FIGURE/"README.md"}) · [参数图](figures/boundary_refinement.png) · [24点设计](parameter_points.csv) · [逐点进度](first_entry_points.csv) · [方案](execution_plan.md)\n\n'+state+' 固定24点后不再自动增加种子或参数，候选待人工检查。\n')
    base.write(OUT/'figure_refresh_status.json',dict(status='UPDATED_PENDING_HUMAN_REVIEW',updated_at=time.time(),summary=s))


def launch(name):
    folder=OUT/'runs'/name;folder.mkdir(parents=True,exist_ok=True)
    with (folder/'worker.log').open('a') as log:
        return subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'worker','--name',name],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy(),start_new_session=True)


def supervise():
    p=prepare();lock=(OUT/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pending=[j['name'] for j in p['jobs'] if not (OUT/'runs'/j['name']/'result.json').exists()]
    children={};failed=[];last=-1;last_render=0
    while True:
        for name,child in list(children.items()):
            rc=child.poll()
            if rc is None:continue
            if rc or not (OUT/'runs'/name/'result.json').exists():failed.append(name)
            del children[name]
        available=psutil.virtual_memory().available/2**30
        while pending and not failed and len(children)<12 and available>84 and shutil.disk_usage(OUT).free/2**30>40:
            name=pending.pop(0);children[name]=launch(name);available-=4
        completed=sum((OUT/'runs'/j['name']/'result.json').exists() for j in p['jobs'])
        running={}
        for name,child in children.items():
            path=OUT/'runs'/name/'progress.json';d=base.read(path) if path.exists() else {}
            running[name]=dict(pid=child.pid,seed=SEED,time_s=d.get('time_s'),status=d.get('status','STARTING'))
        finished=not pending and not children
        base.write(OUT/'status.json',dict(status='DRAINING_FAILURE' if failed else 'COMPLETE_PENDING_HUMAN_REVIEW' if finished else 'RUNNING_BOUNDARY_REFINEMENT',
            pid=os.getpid(),seed=SEED,completed_new=completed,total_new=24,reused=35,running=running,pending=len(pending),failed=failed,updated_at=time.time()))
        if completed!=last or time.time()-last_render>=600 or finished:
            try:report()
            except Exception as exc:base.write(OUT/'figure_refresh_status.json',dict(status='REFRESH_FAILED',error=repr(exc),updated_at=time.time()))
            last=completed;last_render=time.time()
        if failed and not children:raise RuntimeError(failed)
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
        path=OUT/'runs'/args.name/'failure.json' if args.mode=='worker' else OUT/'supervisor_failure.json'
        base.write(path,dict(error=repr(exc),updated_at=time.time()));raise
