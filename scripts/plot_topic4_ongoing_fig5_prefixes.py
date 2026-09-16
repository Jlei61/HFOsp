#!/usr/bin/env python3
"""Full-layout figures from committed ongoing prefixes, never fake run results."""
import argparse
from datetime import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import pickle
import sys
import time
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
import plot_topic4_m_parameter_modes as f

ROOT=f.ROOT
WINDOW=ROOT/'results/topic4_sef_hfo/fig5_m_overnight_exploration_20260913'
OUT=WINDOW/'ongoing_fig5_prefixes'
SOURCES=[f.OUT,WINDOW/'early_z_refill_branches']
CHECKED={}
PLOTTER_SHA=hashlib.sha256(Path(f.__file__).read_bytes()).hexdigest()


def write(path,value):
    assert path.resolve().is_relative_to(OUT.resolve()),path
    f.write(path,value)


def update():
    global PLOTTER_SHA
    current_sha=hashlib.sha256(Path(f.__file__).read_bytes()).hexdigest()
    if current_sha!=PLOTTER_SHA:
        importlib.reload(f)
        PLOTTER_SHA=current_sha
        CHECKED.clear()
    OUT.mkdir(parents=True,exist_ok=True)
    index=f.read(OUT/'index.json') if (OUT/'index.json').exists() else {}
    refreshed=False
    additional=[Path(p) for p in f.read(WINDOW/'window.json').get('additional_preview_sources',[])]
    for source in list(dict.fromkeys(SOURCES+additional)):
        for folder in sorted((source/'runs').glob('*')):
            if folder.name.startswith('qa') or (folder/'result.json').exists():continue
            if not (folder/'checkpoint.pkl').exists() or not (folder/'progress.json').exists():continue
            progress=f.read(folder/'progress.json')
            if not progress.get('entries'):continue
            key=source.name+'/'+folder.name
            previous=index.get(key,{})
            stamp=(folder/'checkpoint.pkl').stat().st_mtime_ns
            if (previous.get('checkpoint_mtime_ns')==stamp and previous.get('plotter_sha256')==PLOTTER_SHA) or CHECKED.get(key)==stamp:continue
            blob=(folder/'checkpoint.pkl').read_bytes();saved=pickle.loads(blob)
            CHECKED[key]=stamp
            tr=saved['tracker'];end=int(saved['engine']['step']);sec=end*.0001
            if not tr['entries'] or sec<tr['entries'][0]['onset_s']+1:continue
            assert saved['identity']==f.read(f.OUT/'protocol.json')['identity']
            stage=[len(tr['entries']),len(tr['recoveries']),
                   bool(tr['release_s'] is not None and sec>=tr['release_s'])]
            if previous.get('stage')==stage and previous.get('plotter_sha256')==PLOTTER_SHA:continue
            job=saved['job'];identity=saved['identity'];del saved
            # Stop at the exact checkpoint boundary even if another immutable
            # observation block becomes visible while plotting.
            a=f.load(folder,end_step=end)
            r=dict(status='COMMITTED_PREFIX_FOLLOWUP_RUNNING',job=job,tracker=tr,end_s=sec,identity=identity)
            metrics=f.analyze(a,r)
            destination=OUT/source.name/folder.name/f'through_{sec:.1f}s'
            f.render(a,r,metrics,destination/'figures',f.grid_summary())
            qualification=dict(source_run=str(folder),source_checkpoint_step=end,
                source_checkpoint_sha256=hashlib.sha256(blob).hexdigest(),
                checkpoint_mtime_ns=stamp,stage=stage,source_duration_s=sec,
                plotter_sha256=PLOTTER_SHA,
                physical_trajectory_complete=False,planned_followup_still_running=True,
                original_run_result_written=False,independent_sample_added=False,
                complete_layout_not_complete_followup=True,metrics=f.safe(metrics),
                figure=str(destination/'figures/fig5.png'),
                agent_visual_review='PENDING',human_review='PENDING')
            if (folder/'branch_audit.json').exists():qualification['branch_audit']=f.read(folder/'branch_audit.json')
            write(destination/'prefix_qualification.json',qualification)
            readme=destination/'figures/README.md'
            readme.write_text(readme.read_text()+'\n此版本仅到已完整保存的'+f'{sec:.1f}'+'秒，后续原生返回/再进入随访仍在运行，不能据此宣布整条仿真完成。阶段1–5只按该前缀实际观察绘制；高活动不等于持续振荡。\n')
            index[key]=qualification;refreshed=True
            del a,blob
    if refreshed or not (OUT/'index.json').exists():write(OUT/'index.json',index)
    write(OUT/'status.json',dict(status='WATCHING',pid=os.getpid(),figure_prefixes=len(index),
        time=datetime.now().astimezone().isoformat(),new_simulations_dispatched=0,
        full_trajectory_completion_claim=False))
    return len(index)


def watch():
    import fcntl
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'watcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    deadline=datetime.fromisoformat(f.read(WINDOW/'window.json')['deadline']).timestamp()
    while time.time()<deadline:
        update();time.sleep(50)
    n=update()
    write(OUT/'status.json',dict(status='WINDOW_ENDED_PENDING_VISUAL_REVIEW',figure_prefixes=n,
        new_simulations_dispatched=0,full_trajectory_completion_claim=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    try:
        if args.watch:watch()
        else:update()
    except Exception as exc:
        OUT.mkdir(parents=True,exist_ok=True)
        write(OUT/'status.json',dict(status='FAILED',pid=os.getpid(),error=repr(exc),physical_workers_untouched=True))
        raise
