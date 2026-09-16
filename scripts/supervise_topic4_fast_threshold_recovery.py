#!/usr/bin/env python3
"""Bounded reviewed second round; never dispatches an unlisted condition."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
import json,time,sys,subprocess
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/topic4_sef_hfo/autonomous_recovery_exploration_20260914/fast_threshold_round2'
def read(path):return json.loads(path.read_text())
def write(path,data):
    tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)
def production_count():
    count=0
    for p in psutil.process_iter(['cmdline','status']):
        try:
            cmd=p.info['cmdline'] or []
            if 'worker' in cmd and any(Path(x).name in ['run_topic4_fig5_log_m_scan.py','run_topic4_autonomous_recovery.py','run_topic4_fast_threshold_recovery.py'] for x in cmd):
                if not any(x.startswith('qa_') for x in cmd) and p.info['status']!=psutil.STATUS_ZOMBIE:count+=1
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    return count
def main():
    assert read(OUT/'qa.json')['status']=='PASS'
    assert read(OUT/'dispatch_authorization.json')['status']=='GO_BOUNDED_ROUND2'
    p=read(OUT/'protocol.json');deadline=p['deadline_epoch'];names=[x['name'] for x in p['initial_jobs']]
    children={};failed=[];last_analysis=0;seen=set()
    while True:
        for n,c in list(children.items()):
            code=c.poll()
            if code is not None:
                del children[n]
                if code or not (OUT/'runs'/n/'result.json').exists():failed.append(dict(name=n,code=code))
        completed={n for n in names if (OUT/'runs'/n/'result.json').exists()}
        pending=[n for n in names if n not in completed and n not in children and n not in {f['name'] for f in failed}]
        while pending and not failed and time.time()<deadline-5400 and production_count()<24 and psutil.virtual_memory().available/2**30>=80:
            n=pending.pop(0);folder=OUT/'runs'/n;folder.mkdir(parents=True,exist_ok=True)
            with (folder/'worker.log').open('a') as log:
                children[n]=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/run_topic4_fast_threshold_recovery.py'),'worker','--name',n],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            time.sleep(3)
        write(OUT/'status.json',dict(status='RUNNING' if not failed else 'DRAINING_FAILURE',pid=os.getpid(),updated_at=time.time(),deadline_epoch=deadline,
            completed=len(completed),total=len(names),running={n:c.pid for n,c in children.items()},pending=pending,failed=failed,combined_workers=production_count()))
        if time.time()-last_analysis>=600 or completed-seen:
            with (OUT/'analysis.log').open('a') as log:
                proc=subprocess.run([sys.executable,str(ROOT/'scripts/analyze_topic4_autonomous_recovery.py'),'--root',str(OUT)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                if proc.returncode:write(OUT/'analysis_failure.json',dict(returncode=proc.returncode,time=time.time()))
                for n in sorted(completed-seen):
                    proc=subprocess.run([sys.executable,str(ROOT/'scripts/analyze_topic4_autonomous_recovery.py'),'--root',str(OUT),'--name',n],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                    if proc.returncode:write(OUT/'figure_failure.json',dict(name=n,returncode=proc.returncode,time=time.time()))
            last_analysis=time.time();seen=completed.copy()
        if not children and (not pending or failed or time.time()>=deadline-5400):break
        time.sleep(15)
    write(OUT/'status.json',dict(status='COMPLETE' if len(completed)==len(names) and not failed else 'STOPPED_AT_REVIEW',pid=os.getpid(),updated_at=time.time(),completed=len(completed),total=len(names),failed=failed,pending=pending))
if __name__=='__main__':main()
