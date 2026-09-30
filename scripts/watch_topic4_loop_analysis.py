#!/usr/bin/env python3
"""Read-only harvest of the frozen Figure5 campaign; never dispatch simulations."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1]
ROOT=Path('/data/hfosp/topic4_sef_hfo/fig5_autonomous_loop_zk_20260924')
OUT=ROOT/'postprocessing'


def read(path):return json.loads(path.read_text())


def write(path,value):
    tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def signature(paths):
    h=hashlib.sha256()
    for path in paths:
        h.update(str(path).encode())
        h.update(path.read_bytes() if path.exists() else b'MISSING')
    return h.hexdigest()


def harvest(state):
    actions=[]
    def run(key,script,args,inputs):
        sig=signature([REPO/script]+inputs)
        previous=state.get(key,{})
        if previous.get('signature')==sig and previous.get('status')=='PASS':return True
        # Retry a failed read once per minute, never conceal it or touch simulations.
        result=subprocess.run([sys.executable,str(REPO/script)]+list(map(str,args)),cwd=REPO,
            capture_output=True,text=True)
        logfile=OUT/'logs'/f'{key}.log';logfile.write_text(result.stdout+result.stderr)
        state[key]=dict(status='PASS' if result.returncode==0 else 'FAILED',signature=sig,
            updated_epoch=time.time(),exit_code=result.returncode,log=str(logfile))
        actions.append(key)
        write(OUT/'tasks.json',state)
        return result.returncode==0

    groups=[('primary',ROOT,read(ROOT/'queue.json')['names'])]
    for condition in ['rotated','isotropic']:
        groups.append((f'native_{condition}',ROOT/'axis_controls/native_runs'/condition,[f'{condition}_s9108405']))
        root=ROOT/'axis_controls/conditional_runs'/condition
        groups.append((f'conditional_{condition}',root,read(root/'queue.json')['names']))
    all_results=[];done={};native_complete=[]
    for label,root,names in groups:
        results=[root/'runs'/name/'result.json' for name in names]
        present=[p for p in results if p.exists()]
        done[label]=dict(completed=len(present),total=len(results))
        all_results.extend(results)
        # Only completed trajectories are spatially reviewed. Prefix plots have
        # their own explicit incomplete status and cannot satisfy this check.
        for path in present:
            name=path.parent.name
            run(f'spatial_{label}_{name}','scripts/review_topic4_loop_native_states.py',
                ['--root',root,'--name',name],[path])
        if label.startswith('native_'):native_complete.extend(present)

    primary_results=[ROOT/'runs'/n/'result.json' for n in groups[0][2]]
    if run('primary_summary','scripts/analyze_topic4_loop_zk_conditional.py',[],primary_results):
        run('primary_map','scripts/paper_figures/build_fig5_conditional_zk.py',[],[ROOT/'conditional_summary.json'])
    conditional_results=[p for p in all_results if '/conditional_runs/' in str(p)]+primary_results
    if run('structure_conditional_summary','scripts/analyze_topic4_loop_axis_conditional.py',[],conditional_results):
        run('structure_conditional_map','scripts/plot_topic4_loop_axis_responses.py',[],
            [ROOT/'axis_controls/conditional_runs/comparison.json'])
    # Native prefix source signatures change only at8s and full120s: avoid
    # rereading entire in-progress trajectories on every polling cycle.
    prefix_markers=[]
    for condition in ['rotated','isotropic']:
        folder=ROOT/'axis_controls/native_runs'/condition/'runs'/f'{condition}_s9108405'
        chunks=sorted(p for p in (folder/'chunks').glob('*.npz') if '.tmp.' not in p.name)
        enough=any(int(p.stem.split('_')[-1])>=80000 for p in chunks)
        marker=OUT/f'{condition}_prefix_ready.json'
        if enough and not marker.exists():write(marker,dict(first8s_saved=True))
        prefix_markers.append(marker)
    if any(p.exists() for p in prefix_markers):
        run('structure_prefix','scripts/plot_topic4_loop_axis_prefix.py',[],prefix_markers)
        run('structure_native_summary','scripts/analyze_topic4_loop_axis_native.py',[],prefix_markers+native_complete)
    failed=[k for k,v in state.items() if v['status']!='PASS']
    complete=all(v['completed']==v['total'] for v in done.values()) and not failed
    status=dict(stage='COMPLETE_ANALYSIS_CANDIDATES' if complete else 'ANALYSIS_ERROR' if failed else 'WAITING_SIMULATIONS',
        watcher_pid=os.getpid(),updated_epoch=time.time(),simulation_results=done,
        failed_analysis=failed,actions_this_pass=actions,dispatches_simulations=False,
        formal_bifurcation='NOT_ESTABLISHED_STATIC_RATE_GATE_FAILED',human_review='PENDING',
        scope='Native count/field/contact review and conditional/drift plots. Scientific interpretation and final figure review remain agent tasks.')
    write(OUT/'status.json',status)
    return status


def main():
    p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');a=p.parse_args()
    OUT.mkdir(exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
    lock=(OUT/'watcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=read(OUT/'tasks.json') if (OUT/'tasks.json').exists() else {}
    while True:
        try:
            status=harvest(state)
        except Exception as exc:
            status=dict(stage='ANALYSIS_ERROR',watcher_pid=os.getpid(),updated_epoch=time.time(),
                error=repr(exc),dispatches_simulations=False)
            write(OUT/'status.json',status)
        if a.once or status['stage']=='COMPLETE_ANALYSIS_CANDIDATES':
            print(json.dumps(status),flush=True);break
        time.sleep(60)


if __name__=='__main__':main()
