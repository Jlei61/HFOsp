#!/usr/bin/env python3
"""One additional GPU slot, two fixed structure controls, no expansion."""
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
import run_topic4_loop_axis_native as run


def main():
    out=run.OUT;out.mkdir(exist_ok=True)
    lock=(out/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    qa=json.loads((out/'reference/qa/reference_loader_qa.json').read_text())
    assert qa['status']=='PASS'
    conditions=['rotated','isotropic']
    protocols={c:run.prepare(c) for c in conditions}
    jobs={c:protocols[c]['initial_jobs'][0]['name'] for c in conditions}
    pending=[c for c in conditions if not (out/c/'runs'/jobs[c]/'result.json').exists()]
    finished=[c for c in conditions if c not in pending];active=None;failures=[]
    logs=out/'logs';logs.mkdir(exist_ok=True)
    while pending or active:
        if active and active[1].poll() is not None:
            c,proc,handle=active;handle.close();active=None
            if proc.returncode or not (out/c/'runs'/jobs[c]/'result.json').exists():
                failures.append(dict(condition=c,exit_code=proc.returncode));pending=[]
            else:
                finished.append(c)
        available=psutil.virtual_memory().available/2**30
        query=subprocess.run(['nvidia-smi','--id=1','--query-gpu=memory.free','--format=csv,noheader,nounits'],
                             capture_output=True,text=True)
        gpu_free=float(query.stdout.strip())/1024 if query.returncode==0 else 0.
        if pending and active is None and available>=78 and gpu_free>=5:
            c=pending.pop(0);handle=(logs/f'{c}.log').open('a')
            proc=subprocess.Popen([sys.executable,str(Path(run.__file__)),'worker',c],
                cwd=Path(__file__).resolve().parents[1],stdout=handle,stderr=subprocess.STDOUT)
            active=(c,proc,handle)
        detail=None
        if active:
            c,proc,_=active;path=out/c/'runs'/jobs[c]/'progress.json'
            value=json.loads(path.read_text()) if path.exists() else {}
            detail=dict(condition=c,pid=proc.pid,time_s=value.get('time_s'),state=value.get('status'))
        stage='FAILED' if failures else 'COMPLETE' if not pending and active is None else 'RUNNING' if active else 'WAITING_RESOURCE'
        run.native.write(out/'status.json',dict(stage=stage,supervisor_pid=os.getpid(),updated_epoch=time.time(),
            completed=finished,pending=pending,active=detail,failures=failures,
            available_memory_GiB=available,GPU1_free_GiB=gpu_free,
            capacity='One additional GPU1 worker; at least78GiB host memory and5GiB device memory before dispatch, allowing8GiB setup over70GiB reserve. Other jobs untouched.',
            total_new_runs=2,horizon_s=120,single_seed=9108405))
        if pending or active:time.sleep(15)


if __name__=='__main__':main()
