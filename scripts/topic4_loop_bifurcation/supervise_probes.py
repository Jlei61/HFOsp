#!/usr/bin/env python3
"""Bounded explicit-field probes; one worker with a global per-GPU cap."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import fcntl
import subprocess
import time
from pathlib import Path
import psutil
from campaign import REPO,PYTHON,read,write,sha

RUNNER=REPO/'scripts/topic4_loop_bifurcation/spatial_probes.py'
NATIVE_MARKERS=('native_campaign.py','followup_campaign.py','spatial_probes.py',
                'run_topic4_loop_cuda_override.py')


def live_native():
    rows=[]
    for p in psutil.process_iter(['pid','cmdline','status','create_time']):
        try:
            c=p.info['cmdline'] or []
            if p.info['status']==psutil.STATUS_ZOMBIE or 'worker' not in c:continue
            if not any(any(x.endswith(m) for x in c) for m in NATIVE_MARKERS):continue
            if '--device' not in c:continue
            rows.append(dict(pid=p.pid,device=int(c[c.index('--device')+1]),created_epoch=p.info['create_time']))
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    return rows


def main(root,device):
    lock=(root/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    q=read(root/'queue.json');names=q['names'];assert 1<=len(names)<=12
    (root/'logs').mkdir(exist_ok=True)
    for name in names:
        folder=root/'runs'/name
        if (folder/'result.json').exists() and read(folder/'result.json')['status']=='COMPLETE':continue
        # Restarting the controller must not dispatch a duplicate live worker.
        matches=[]
        for p in psutil.process_iter(['cmdline']):
            try:
                c=p.info['cmdline'] or []
                if str(RUNNER) in c and name in c and str(root) in c and 'worker' in c:matches.append(p)
            except psutil.NoSuchProcess:pass
        assert len(matches)<=1
        proc=matches[0] if matches else None
        while proc is None:
            active=live_native();available=psutil.virtual_memory().available/2**30
            report=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
            free={int(s.split(',')[0]):float(s.split(',')[1])/1024 for s in report.strip().splitlines()}
            ok=len(active)<8 and sum(p['device']==device for p in active)<4 and available>=70 and free[device]>=3.
            write(root/'status.json',dict(stage='DISPATCH' if ok else 'WAITING_RESOURCES',supervisor_pid=os.getpid(),current=name,global_native=active,updated_epoch=time.time()))
            if not ok:time.sleep(20);continue
            assert sha(root/'jobs'/f'{name}.json')==q['job_sha256'][name]
            handle=(root/'logs'/f'{name}.log').open('a')
            child=subprocess.Popen([PYTHON,str(RUNNER),'worker','--root',str(root),'--name',name,'--device',str(device)],cwd=REPO,stdout=handle,stderr=subprocess.STDOUT)
            handle.close();proc=psutil.Process(child.pid)
        while proc.is_running() and proc.status()!=psutil.STATUS_ZOMBIE:
            progress=read(folder/'progress.json') if (folder/'progress.json').exists() else {}
            write(root/'status.json',dict(stage='RUNNING',supervisor_pid=os.getpid(),current=name,worker_pid=proc.pid,created_epoch=proc.create_time(),progress=progress,updated_epoch=time.time()))
            time.sleep(20)
        if not (folder/'result.json').exists() or read(folder/'result.json')['status']!='COMPLETE':
            write(root/'status.json',dict(stage='FAILED',name=name,updated_epoch=time.time()));return
    write(root/'status.json',dict(stage='COMPLETE',total=len(names),names=names,updated_epoch=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--device',type=int,choices=[0,1],default=1)
    a=p.parse_args();main(a.root,a.device)
