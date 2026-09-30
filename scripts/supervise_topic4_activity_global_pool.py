#!/usr/bin/env python3
"""Only execute locally reviewed global-pool jobs, with a bounded sentinel."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
import json,time,sys,subprocess,fcntl
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'results/topic4_sef_hfo/autonomous_recovery_exploration_20260914'
OUT=PARENT/'activity_global_pool_round3'
def read(path):return json.loads(path.read_text())
def write(path,data):
    tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)
def count():
    n=0
    scripts={'run_topic4_fig5_log_m_scan.py','run_topic4_autonomous_recovery.py','run_topic4_fast_threshold_recovery.py','run_topic4_activity_global_pool.py'}
    for p in psutil.process_iter(['cmdline','status']):
        try:
            cmd=p.info['cmdline'] or []
            if 'worker' in cmd and any(Path(x).name in scripts for x in cmd) and not any(x.startswith('qa_') for x in cmd) and p.info['status']!=psutil.STATUS_ZOMBIE:n+=1
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    return n
def analyze(new):
    with (OUT/'analysis.log').open('a') as log:
        for script in ['analyze_topic4_autonomous_recovery.py','analyze_topic4_autonomous_events.py','analyze_topic4_recruitment_origin.py','analyze_topic4_global_depletion_coupling.py']:
            # Before the first completed checkpoint there may be no geometry.
            if not (OUT/'geometry.npz').exists():continue
            p=subprocess.run([sys.executable,str(ROOT/'scripts'/script),'--root',str(OUT)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if p.returncode:write(OUT/'analysis_failure.json',dict(script=script,returncode=p.returncode,time=time.time()))
        for name in sorted(new):
            p=subprocess.run([sys.executable,str(ROOT/'scripts/analyze_topic4_autonomous_recovery.py'),'--root',str(OUT),'--name',name],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if p.returncode:write(OUT/'figure_failure.json',dict(name=name,returncode=p.returncode,time=time.time()))
def main():
    lock=(OUT/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert read(OUT/'qa.json')['status']=='PASS'
    p=read(OUT/'protocol.json');deadline=p['deadline_epoch'];all_names=[j['name'] for j in p['initial_jobs']]
    children={};failed=[];last=0;seen=set()
    while True:
        auth=read(OUT/'dispatch_authorization.json');approved=auth['approved_names'];assert set(approved)<=set(all_names)
        for name,c in list(children.items()):
            code=c.poll()
            if code is not None:
                children.pop(name)
                if code or not (OUT/'runs'/name/'result.json').exists():failed.append(dict(name=name,code=code))
        complete={n for n in approved if (OUT/'runs'/n/'result.json').exists()}
        pending=[n for n in approved if n not in complete and n not in children and n not in {f['name'] for f in failed}]
        if pending and not failed and time.time()<deadline-5400:
            n=pending[0];total=count();available=psutil.virtual_memory().available/2**30
            sentinel=n==auth.get('sentinel_name') and len(approved)==1 and total<auth.get('single_extra_sentinel_cap',24) and available>=auth.get('sentinel_min_available_memory_GiB',120)
            r2=read(PARENT/'fast_threshold_round2/status.json')
            regular=not r2.get('pending') and total<auth['maximum_combined_workers'] and available>=auth['minimum_available_memory_GiB']
            if sentinel or regular:
                folder=OUT/'runs'/n;folder.mkdir(parents=True,exist_ok=True)
                with (folder/'worker.log').open('a') as log:
                    children[n]=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/run_topic4_activity_global_pool.py'),'worker','--name',n],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                pending.remove(n)
        write(OUT/'status.json',dict(status='RUNNING' if not failed else 'DRAINING_FAILURE',pid=os.getpid(),updated_at=time.time(),deadline_epoch=deadline,
            completed=len(complete),approved=len(approved),prepared_total=len(all_names),running={n:c.pid for n,c in children.items()},pending=pending,
            not_approved=[n for n in all_names if n not in approved],failed=failed,combined_workers=count()))
        if time.time()-last>=600 or complete-seen:analyze(complete-seen);last=time.time();seen=complete.copy()
        if not children and (not pending or failed or time.time()>=deadline-5400):break
        time.sleep(15)
    analyze(complete-seen)
    write(OUT/'status.json',dict(status='REVIEW_MILESTONE' if not failed else 'FAILED_REVIEW',pid=os.getpid(),updated_at=time.time(),completed=len(complete),approved=len(approved),prepared_total=len(all_names),running={},pending=pending,not_approved=[n for n in all_names if n not in approved],failed=failed))
if __name__=='__main__':main()
