#!/usr/bin/env python3
"""Bounded queue, four workers, live status, no extra scientific proposals."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src import topic4_initial_state_runtime as rt


def main():
    design=rt.read(ROOT/'config/topic4_continuous_core_state_r1.json')
    out=Path(design['output_root']);(out/'logs').mkdir(exist_ok=True)
    if rt.read(out/'qualification.json')['status']!='PASS':raise RuntimeError('qualification failed')
    # Long zero-state replays also provide the exact short control prefixes.
    jobs=sorted(design['jobs'],key=lambda j:(-j['duration_ms'],j['id']))
    active={};failed=[];complete=[];started=time.time()
    while jobs or active:
        for pid,(p,j,stream) in list(active.items()):
            if p.poll() is not None:
                stream.close();del active[pid]
                if p.returncode==0:complete.append(j['id'])
                else:failed.append(dict(job=j['id'],exit_code=p.returncode))
        while jobs and len(active)<design['budget']['max_parallel'] and rt.available_gib()>60:
            j=jobs.pop(0);stream=(out/'logs'/(j['id']+'.log')).open('a')
            cmd=[rt.PYTHON,'-u',str(ROOT/'scripts/run_topic4_continuous_core_state_r1.py'),'--job',j['id']]
            p=subprocess.Popen(cmd,cwd=ROOT,env=rt.ENV,stdout=stream,stderr=subprocess.STDOUT)
            active[p.pid]=(p,j,stream)
        snapshot=dict(status='RUNNING' if jobs or active else ('ENGINEERING_FAILURES' if failed else 'SIMULATIONS_COMPLETE'),
                      controller_pid=os.getpid(),started_unix=started,updated_unix=time.time(),
                      n_total=len(design['jobs']),n_complete=len(complete),completed=complete,
                      active=[dict(pid=pid,job=j['id']) for pid,(p,j,f) in active.items()],
                      remaining=[j['id'] for j in jobs],failed=failed,available_gib=rt.available_gib())
        rt.write(out/'status.json',snapshot)
        if jobs or active:time.sleep(10)
    if failed:sys.exit(1)
    snapshot.update(status='ANALYZING',updated_unix=time.time())
    rt.write(out/'status.json',snapshot)
    with (out/'analysis.log').open('a') as stream:
        result=subprocess.run([rt.PYTHON,'-u',str(ROOT/'scripts/analyze_topic4_continuous_core_state_r1.py')],
                              cwd=ROOT,env=rt.ENV,stdout=stream,stderr=subprocess.STDOUT)
    snapshot.update(status='ROUND_COMPLETE_PENDING_SCIENTIFIC_REVIEW' if result.returncode==0 else 'ANALYSIS_FAILED',
                    analysis_exit_code=result.returncode,updated_unix=time.time())
    rt.write(out/'status.json',snapshot)
    sys.exit(result.returncode)

if __name__=='__main__':main()
