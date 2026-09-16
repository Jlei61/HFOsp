#!/usr/bin/env python3
"""Finish only the original two interrupted trajectories with resumable workers."""
from pathlib import Path
import json,os,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/topic4_sef_hfo/fig5_manual_core_release_v1/m_runaway_return_v1'
NAMES=['m0_native','weak_fast']
ALL=['m0_native','weak_fast','weak_20s','matched_gain_80s','slow_80s','slow_200s']

def read(p):
    try:return json.loads(p.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return {}

def write(p,r):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(r,indent=2)+'\n');tmp.replace(p)

def launch(name):
    completed=OUT/'runs'/(name+'.json')
    if read(completed).get('status')=='COMPLETE':return read(completed)
    for attempt in (1,2):
        log=OUT/'logs'/f'{name}_recovery_attempt{attempt}.log'
        with log.open('a') as stream:
            child=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/resume_topic4_fig5_m_return.py'),
                '--job',str(OUT/'jobs'/(name+'.json'))],stdout=stream,stderr=subprocess.STDOUT)
            write(OUT/'recovery_pids'/f'{name}.json',dict(pid=child.pid,attempt=attempt,started_at=time.time()))
            code=child.wait()
        if code==0 and read(completed).get('status')=='COMPLETE':return read(completed)
        if code>=0 or attempt==2:
            raise RuntimeError(f'{name} exited {code}; see {log}')
    raise AssertionError('unreachable')

def main():
    assert read(OUT/'checkpoint_recovery_qa.json').get('status')=='PASS'
    (OUT/'recovery_pids').mkdir(exist_ok=True)
    write(OUT/'recovery_pids/supervisor.json',dict(pid=os.getpid(),started_at=time.time(),detached=True))
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending={n:pool.submit(launch,n) for n in NAMES};reported=set()
        while pending:
            complete=[n for n in ALL if read(OUT/'runs'/(n+'.json')).get('status')=='COMPLETE']
            write(OUT/'status.json',dict(status='RUNNING_RECOVERY',completed=len(complete),total=6,
                completed_names=complete,replaying=list(pending),supervisor_pid=os.getpid()))
            finished=[n for n,f in pending.items() if f.done()]
            for n in finished:
                pending.pop(n).result()
                with (OUT/'incremental_analysis.log').open('a') as stream:
                    subprocess.run([sys.executable,str(ROOT/'scripts/plot_topic4_fig5_m_return.py'),'--available'],
                        stdout=stream,stderr=subprocess.STDOUT,check=True)
            if pending:time.sleep(30)
    rows=[read(OUT/'runs'/(n+'.json')) for n in ALL]
    write(OUT/'batch.json',rows)
    write(OUT/'status.json',dict(status='SIMULATIONS_COMPLETE',completed=6,total=6))
    with (OUT/'final_recovery_analysis.log').open('a') as stream:
        subprocess.run([sys.executable,str(ROOT/'scripts/plot_topic4_fig5_m_return.py')],
            stdout=stream,stderr=subprocess.STDOUT,check=True)
    write(OUT/'incremental_analysis_status.json',dict(status='ALL_RESULTS_ANALYZED_PENDING_VISUAL_REVIEW',completed=ALL,total=6))
    write(OUT/'delivery_status.json',dict(status='COMPUTED_AND_PLOTTED_PENDING_REVIEW',completed=6,
        recovered_interrupted_runs=NAMES,supervisor_pid=os.getpid()))

if __name__=='__main__':
    try:main()
    except Exception as e:
        write(OUT/'status.json',dict(status='RECOVERY_FAILED',error=repr(e),supervisor_pid=os.getpid()))
        raise
