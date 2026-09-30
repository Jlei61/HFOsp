#!/usr/bin/env python3
"""Wait for the prespecified gates, then dispatch at most sixteen graph branches."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
import psutil
import run_topic4_loop_axis_conditional as run


def prerequisites():
    missing=[]
    path=run.OUT/'reference/route_qa.json'
    if not path.exists() or run.native.base.read(path)['status']!='PASS':missing.append('reference_route_qa')
    q=run.native.base.read(run.PRIMARY/'queue.json')['names']
    if not all((run.PRIMARY/'runs'/n/'result.json').exists() for n in q):missing.append('original18_finished')
    else:
        status=run.native.base.read(run.PRIMARY/'status.json')
        if status['stage']!='COMPLETE' or status['active']:missing.append('original18_supervisor_clean_completion')
        else:
            summary=run.native.base.read(run.PRIMARY/'conditional_summary.json')
            if summary['completed']!=18 or not all(r['full_horizon'] for r in summary['rows']):missing.append('original18_full30s_and_analysis')
    for condition in ['rotated','isotropic']:
        root=run.axis.OUT/condition/'runs'/f'{condition}_s9108405'
        qa=root/'graph_loader_qa.json'
        if not qa.exists() or run.native.base.read(qa)['status']!='PASS':missing.append(condition+'_native_graph_qa')
        files=list((root/'chunks').glob('*.npz'))
        end=max([int(p.stem.split('_')[-1]) for p in files if '.tmp.' not in p.name],default=0)
        if end<80000:missing.append(condition+'_native8s_prefix')
    return missing


def main():
    run.OUT.mkdir(exist_ok=True)
    lock=(run.OUT/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for condition in ['rotated','isotropic']:run.prepare(condition)
    names=run.native.base.read(run.OUT/'rotated/queue.json')['names']
    central=[n for n in names if n.startswith('z0.75_k2_')]
    names=central+[n for n in names if n not in central]
    queue=[(c,n) for n in names for c in ['rotated','isotropic']]
    assert len(queue)==16 and len(set(queue))==16
    complete=[(c,n) for c,n in queue if (run.OUT/c/'runs'/n/'result.json').exists()]
    pending=[x for x in queue if x not in complete];active={};failed=[]
    logs=run.OUT/'logs';logs.mkdir(exist_ok=True)
    opened=False
    while pending or active:
        for key,(proc,handle) in list(active.items()):
            if proc.poll() is None:continue
            handle.close();del active[key];c,n=key
            if proc.returncode or not (run.OUT/c/'runs'/n/'result.json').exists():
                failed.append(dict(condition=c,name=n,exit_code=proc.returncode));pending=[]
            else:
                complete.append(key)
                result=subprocess.run([sys.executable,str(Path(__file__).with_name('analyze_topic4_loop_axis_conditional.py'))],
                    cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True)
                (logs/'latest_analysis.log').write_text(result.stdout+result.stderr)
                if result.returncode:
                    failed.append(dict(stage='analysis',reason=result.stderr[-3000:]));pending=[]
        missing=prerequisites() if not opened else []
        if not missing and not opened:
            opened=True
            run.native.write(run.OUT/'dispatch_gate.json',dict(status='PASS',epoch=time.time(),
                original18_finished=True,native_graph_prefixes_saved=True,reference_route_exact=True,maximum_scientific_jobs=16))
        available=psutil.virtual_memory().available/2**30;cpu=psutil.cpu_percent(interval=.2)
        while opened and pending and len(active)<4 and available>=88 and cpu<=65 and not failed:
            c,n=pending.pop(0);handle=(logs/f'{c}_{n}.log').open('a')
            proc=subprocess.Popen([sys.executable,str(Path(run.__file__)),'worker',c,n],
                cwd=Path(__file__).resolve().parents[1],stdout=handle,stderr=subprocess.STDOUT)
            active[(c,n)]=(proc,handle);available-=8;cpu+=10
        detail=[]
        for (c,n),(proc,_) in active.items():
            path=run.OUT/c/'runs'/n/'progress.json';r=run.native.base.read(path) if path.exists() else {}
            detail.append(dict(condition=c,name=n,pid=proc.pid,time_s=r.get('time_s'),state=r.get('status')))
        stage='FAILED' if failed else 'COMPLETE' if not pending and not active else 'WAITING_PREREQUISITES' if missing else 'RUNNING' if active else 'WAITING_RESOURCE'
        run.native.write(run.OUT/'status.json',dict(stage=stage,supervisor_pid=os.getpid(),updated_epoch=time.time(),
            missing_prerequisites=missing,completed=complete,pending=pending,active=detail,failed=failed,
            new_scientific_total=16,available_memory_GiB=available,max_workers=4,threads_each=8))
        if pending or active:time.sleep(30)


if __name__=='__main__':main()
