#!/usr/bin/env python3
"""Adopt current workers; use validated locality CPU backend for future jobs."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import psutil

REPO=Path(__file__).resolve().parents[1]
OUT=Path('/data/hfosp/topic4_sef_hfo/fig5_autonomous_loop_zk_20260924')
RUNNER=REPO/'scripts/run_topic4_loop_zk_locality_cpu.py'
ANALYZE=REPO/'scripts/analyze_topic4_loop_zk_conditional.py'


def read(path):return json.loads(path.read_text())


def write(path,value):
    tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def alive(proc):
    if isinstance(proc,subprocess.Popen):return proc.poll() is None
    try:return proc.is_running() and proc.status()!=psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:return False


def main():
    lock=(OUT/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gate=read(OUT/'qa/locality_native_gate.json');assert gate['status']=='PASS'
    for path,key in [(RUNNER,'wrapper_sha256'),(REPO/'scripts/topic4_loop_locality_cpu.py','source_sha256')]:
        assert hashlib.sha256(path.read_bytes()).hexdigest()==gate[key]
    names=read(OUT/'queue.json')['names'];assert len(names)==18
    central=[n for n in names if n.startswith('z0.75_k2_')]
    names=central+[n for n in names if n not in central]
    old=read(OUT/'status.json');active={};failed=old.get('failed',{})
    for name,row in old.get('active',{}).items():
        try:proc=psutil.Process(row['pid'])
        except psutil.NoSuchProcess:continue
        if not alive(proc):continue
        cmd=proc.cmdline()
        assert cmd[-1]==name and any('run_topic4_loop_zk_' in v for v in cmd), (name,cmd)
        active[name]=(proc,None,'serial_cpu_adopted' if 'conditional.py' in ' '.join(cmd) else 'locality_cpu')
    complete=[n for n in names if (OUT/'runs'/n/'result.json').exists() and n not in active]
    pending=[n for n in names if n not in complete and n not in active and n not in failed]
    logs=OUT/'logs'
    while pending or active:
        for name,(proc,handle,backend) in list(active.items()):
            if alive(proc):continue
            if handle:handle.close()
            del active[name]
            result=OUT/'runs'/name/'result.json'
            if result.exists():
                complete.append(name)
                analysis=subprocess.run([sys.executable,str(ANALYZE)],cwd=REPO,capture_output=True,text=True)
                (logs/'latest_analysis.log').write_text(analysis.stdout+analysis.stderr)
                if analysis.returncode:
                    failed['analysis_after_'+name]=analysis.stderr[-3000:];pending=[]
            else:
                failed[name]=dict(reason='Worker ended without durable result',log=str(logs/f'{name}.log'))
                pending=[]
        available=psutil.virtual_memory().available/2**30
        cpu_busy=psutil.cpu_percent(interval=.2)
        parallel=sum(backend=='locality_cpu' for _,_,backend in active.values())
        while pending and len(active)<8 and parallel<4 and available>=88 and cpu_busy<=65 and not failed:
            name=pending.pop(0);handle=(logs/f'{name}.log').open('a')
            proc=subprocess.Popen([sys.executable,str(RUNNER),'worker',name],cwd=REPO,stdout=handle,stderr=subprocess.STDOUT)
            active[name]=(proc,handle,'locality_cpu');available-=8;cpu_busy+=10;parallel+=1
        detail={}
        for name,(proc,_,backend) in active.items():
            path=OUT/'runs'/name/'progress.json';row=read(path) if path.exists() else {}
            detail[name]=dict(pid=proc.pid,time_s=row.get('time_s'),state=row.get('status'),backend=backend)
        status=dict(stage='RUNNING' if pending or active else 'FINISHED_WITH_FAILURES' if failed else 'COMPLETE',
            supervisor_pid=os.getpid(),updated_epoch=time.time(),completed=complete,pending=pending,active=detail,failed=failed,
            total=18,available_memory_GiB=available,backend='Existing serial workers retained; new workers use validated ordered locality CPU,8threads',
            max_total_workers=8,max_locality_workers=4,CPU_busy_percent_at_dispatch=cpu_busy,
            memory_policy='Reserve80GiB plus8GiB per not-yet-resident worker; at most4locality workers even after serial workers finish.',
            new_gpu_processes=0,diagnostic_only=True)
        write(OUT/'status.json',status)
        if pending or active:time.sleep(15)


if __name__=='__main__':main()
