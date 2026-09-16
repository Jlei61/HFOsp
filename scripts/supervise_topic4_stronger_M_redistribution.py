#!/usr/bin/env python3
"""Two reviewed parameter-only jobs; yield resource priority to the joint pool batch."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
import fcntl,subprocess,sys,time
import psutil
import run_topic4_stronger_M_redistribution as run
from supervise_topic4_reviewed_global_batch import resources
OUT=run.OUT;ROOT=run.ROOT;read=run.carrier.base.read;write=run.carrier.base.write

def analyze(names):
    if not (OUT/'geometry.npz').exists():return
    with (OUT/'analysis.log').open('a') as log:
        for script in ['analyze_topic4_autonomous_recovery.py','analyze_topic4_autonomous_events.py','analyze_topic4_recruitment_origin.py']:
            r=subprocess.run([sys.executable,str(ROOT/'scripts'/script),'--root',str(OUT)],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
            if r.returncode:write(OUT/'analysis_failure.json',dict(script=script,returncode=r.returncode,time=time.time()))
        for name in names:
            r=subprocess.run([sys.executable,str(ROOT/'scripts/analyze_topic4_autonomous_recovery.py'),'--root',str(OUT),'--name',name],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
            if r.returncode:write(OUT/'figure_failure.json',dict(name=name,returncode=r.returncode,time=time.time()))

def main():
    p=run.prepare();lock=(OUT/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs=[j['name'] for j in p['initial_jobs']];children={};failed=[];seen=set();last=0
    while True:
        for name,child in list(children.items()):
            rc=child.poll()
            if rc is not None:
                children.pop(name)
                if rc or not (OUT/'runs'/name/'result.json').exists():failed.append(dict(name=name,returncode=rc))
        complete={n for n in jobs if (OUT/'runs'/n/'result.json').exists()}
        pending=[n for n in jobs if n not in complete and n not in children and n not in {v['name'] for v in failed}]
        parent=read(run.PARENT/'activity_global_pool_round3/status.json')
        if pending and not failed and not parent.get('pending') and time.time()<p['deadline_epoch']-5400:
            total,gpu=resources()
            if total<28 and gpu<24 and psutil.virtual_memory().available/2**30>=120:
                name=pending.pop(0);folder=OUT/'runs'/name;folder.mkdir(parents=True,exist_ok=True)
                with (folder/'worker.log').open('a') as log:
                    children[name]=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/run_topic4_stronger_M_redistribution.py'),'worker','--name',name,'--producer-script',str(ROOT/'scripts/run_topic4_autonomous_recovery.py')],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,start_new_session=True)
        total,gpu=resources()
        write(OUT/'status.json',dict(status='RUNNING' if not failed else 'DRAINING_FAILURE',pid=os.getpid(),updated_at=time.time(),completed=len(complete),total=2,running={n:c.pid for n,c in children.items()},pending=pending,failed=failed,combined_workers=total,gpu_workers=gpu,waiting_for_round3_pending=bool(parent.get('pending'))))
        if time.time()-last>=600 or complete-seen:analyze(sorted(complete-seen));seen=complete.copy();last=time.time()
        if not children and (not pending or failed or time.time()>=p['deadline_epoch']-5400):break
        time.sleep(15)
    analyze(sorted(complete-seen))
    write(OUT/'status.json',dict(status='REVIEW_MILESTONE' if not failed else 'FAILED_REVIEW',pid=os.getpid(),updated_at=time.time(),completed=len(complete),total=2,running={},pending=pending,failed=failed))

if __name__=='__main__':main()
