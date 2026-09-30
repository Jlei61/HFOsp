#!/usr/bin/env python3
"""Resource-only multiworker controller for already frozen probe jobs."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import fcntl
import subprocess
import time
from pathlib import Path
import psutil
from campaign import ROOT,REPO,PYTHON,read,write,sha
from supervise_probes import live_native,RUNNER


def live(p):
    try:return p.is_running() and p.status()!=psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:return False


def main(root,maximum):
    assert read(ROOT/'execution_resources_v3.json')['global_native_cap']==12
    lock=(root/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    global_lock=(ROOT/'native_dispatch.lock').open('a')
    queue=read(root/'queue.json');names=queue['names'];active={};complete=[];failed=[]
    for p in psutil.process_iter(['cmdline','create_time']):
        try:
            c=p.info['cmdline'] or []
            if str(RUNNER) not in c or str(root) not in c or 'worker' not in c or not live(p):continue
            name=c[c.index('--name')+1];device=int(c[c.index('--device')+1]);assert name in names and name not in active
            active[name]=(p,device,p.create_time())
        except psutil.NoSuchProcess:pass
    for n in names:
        path=root/'runs'/n/'result.json'
        if path.exists() and n not in active:
            if read(path)['status']=='COMPLETE':complete.append(n)
            else:failed.append(dict(name=n,status=read(path)['status']))
    pending=[n for n in names if n not in complete and n not in active]
    (root/'logs').mkdir(exist_ok=True)
    children=[]
    while active or pending:
        for n,(proc,device,created) in list(active.items()):
            if live(proc):continue
            del active[n];path=root/'runs'/n/'result.json'
            if path.exists() and read(path)['status']=='COMPLETE':complete.append(n)
            else:failed.append(dict(name=n,reason='Worker ended without complete result'))
        # Reap only children launched by this controller; adopted workers are
        # checked by PID creation-time identity and never restarted while alive.
        for proc in children:proc.poll()
        fcntl.flock(global_lock,fcntl.LOCK_EX)
        try:
            workers=live_native();memory=psutil.virtual_memory().available/2**30
            gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
            free={int(x.split(',')[0]):float(x.split(',')[1])/1024 for x in gpu.strip().splitlines()}
            resident=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
            used={int(x.split(',')[0]):float(x.split(',')[1])/1024 for x in resident.strip().splitlines() if x.strip()}
            for p in workers:free[p['device']]-=max(0.,3.-used.get(p['pid'],0.))
            while pending and not failed and len(active)<maximum and len(workers)<12 and memory>=70:
                devices=[d for d in [0,1] if sum(p['device']==d for p in workers)<6 and free[d]>=3.]
                if not devices:break
                device=max(devices,key=lambda d:free[d]);name=pending.pop(0)
                assert sha(root/'jobs'/f'{name}.json')==queue['job_sha256'][name]
                with (root/'logs'/f'{name}.log').open('a') as handle:
                    proc=subprocess.Popen([PYTHON,str(RUNNER),'worker','--root',str(root),'--name',name,'--device',str(device)],cwd=REPO,stdout=handle,stderr=subprocess.STDOUT)
                children.append(proc);pp=psutil.Process(proc.pid);active[name]=(pp,device,pp.create_time())
                workers.append(dict(pid=proc.pid,device=device,created_epoch=pp.create_time()));free[device]-=3.;memory-=10.
        finally:fcntl.flock(global_lock,fcntl.LOCK_UN)
        detail=[]
        for name,(p,d,c) in active.items():
            path=root/'runs'/name/'progress.json';v=read(path) if path.exists() else {}
            detail.append(dict(name=name,pid=p.pid,device=d,created_epoch=c,time_s=v.get('time_s'),status=v.get('status')))
        stage='FAILED' if failed else 'COMPLETE' if not pending and not active else 'RUNNING' if active else 'WAITING_RESOURCES'
        write(root/'status.json',dict(stage=stage,supervisor_pid=os.getpid(),completed=complete,pending=pending,active=detail,failed=failed,total=len(names),updated_epoch=time.time(),global_native_cap=12,per_GPU_cap=6,max_workers=maximum))
        if failed and not active:break
        if active or pending:time.sleep(20)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--max-workers',type=int,default=6);a=p.parse_args();main(a.root,a.max_workers)
