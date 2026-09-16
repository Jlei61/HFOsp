#!/usr/bin/env python3
"""Adopt the live sentinel and finish the reviewed global-feedback controls."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
import time,sys,subprocess,fcntl
import psutil
from pathlib import Path
import supervise_topic4_activity_global_pool as common
ROOT,PARENT,OUT=common.ROOT,common.PARENT,common.OUT
read,write=common.read,common.write
def resources():
    total=gpu=0
    scripts={'run_topic4_fig5_log_m_scan.py','run_topic4_autonomous_recovery.py','run_topic4_fast_threshold_recovery.py','run_topic4_activity_global_pool.py'}
    for p in psutil.process_iter(['cmdline','status']):
        try:
            cmd=p.info['cmdline'] or []
            if 'worker' in cmd and any(Path(x).name in scripts for x in cmd) and not any(x.startswith('qa_') for x in cmd) and p.info['status']!=psutil.STATUS_ZOMBIE:
                total+=1
                if not any(Path(x).name=='run_topic4_quiet_cpu_resume.py' for x in cmd):gpu+=1
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    return total,gpu
def alive(pid):
    try:return psutil.Process(pid).status()!=psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:return False
def main():
    previous=read(OUT/'status.json');prior_pid=previous['pid']
    if alive(prior_pid):
        pp=psutil.Process(prior_pid)
        assert any(Path(x).name in ['supervise_topic4_activity_global_pool.py',
                                   'supervise_topic4_reviewed_global_batch.py'] for x in pp.cmdline())
        pp.terminate();pp.wait(timeout=10)
    lock=(OUT/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert read(OUT/'qa.json')['status']=='PASS'
    p=read(OUT/'protocol.json');deadline=p['deadline_epoch'];all_names=[j['name'] for j in p['initial_jobs']]
    pids=previous.get('running',{}).copy();children={};failed=[];last=0;seen=set()
    for name,pid in pids.items():
        assert name in psutil.Process(pid).cmdline() and 'worker' in psutil.Process(pid).cmdline()
    if (OUT/'supervisor_adoption.json').exists():
        write(OUT/f'supervisor_adoption_before_{os.getpid()}.json',read(OUT/'supervisor_adoption.json'))
    write(OUT/'supervisor_adoption.json',dict(previous_pid=prior_pid,current_pid=os.getpid(),time=time.time(),adopted=pids,no_worker_restarted=True))
    while True:
        auth=read(OUT/'dispatch_authorization.json');approved=auth['approved_names'];assert set(approved)<=set(all_names)
        for name,pid in list(pids.items()):
            if name in children:children[name].poll()
            if not alive(pid):
                pids.pop(name)
                if not (OUT/'runs'/name/'result.json').exists():failed.append(dict(name=name,pid=pid))
        complete={n for n in approved if (OUT/'runs'/n/'result.json').exists()}
        pending=[n for n in approved if n not in complete and n not in pids and n not in {f['name'] for f in failed}]
        if pending and not failed and time.time()<deadline-5400:
            total,gpu=resources();available=psutil.virtual_memory().available/2**30
            r2=read(PARENT/'fast_threshold_round2/status.json')
            allow_reviewed_extra=len(approved)<=4 or not r2.get('pending')
            if allow_reviewed_extra and total<auth['maximum_combined_workers'] and gpu<auth['maximum_gpu_workers'] and available>=auth['minimum_available_memory_GiB']:
                n=pending.pop(0);folder=OUT/'runs'/n;folder.mkdir(parents=True,exist_ok=True)
                with (folder/'worker.log').open('a') as log:
                    child=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/run_topic4_activity_global_pool.py'),'worker','--name',n],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                children[n]=child;pids[n]=child.pid
        total,gpu=resources()
        write(OUT/'status.json',dict(status='RUNNING' if not failed else 'DRAINING_FAILURE',pid=os.getpid(),updated_at=time.time(),deadline_epoch=deadline,completed=len(complete),approved=len(approved),prepared_total=len(all_names),running=pids,pending=pending,not_approved=[n for n in all_names if n not in approved],failed=failed,combined_workers=total,gpu_workers=gpu))
        if time.time()-last>=600 or complete-seen:common.analyze(complete-seen);last=time.time();seen=complete.copy()
        if not pids and (not pending or failed or time.time()>=deadline-5400):break
        time.sleep(15)
    common.analyze(complete-seen)
    write(OUT/'status.json',dict(status='REVIEW_MILESTONE' if not failed else 'FAILED_REVIEW',pid=os.getpid(),updated_at=time.time(),completed=len(complete),approved=len(approved),prepared_total=len(all_names),running={},pending=pending,not_approved=[n for n in all_names if n not in approved],failed=failed))
if __name__=='__main__':main()
