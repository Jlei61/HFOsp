#!/usr/bin/env python3
"""Adopt the fixed input-pilot workers and finish its remaining units in parallel.

Scheduling-only amendment. The original worker, frozen plan, physics, units,
durations and raw output contract stay unchanged. Never restart a failed unit.
"""
from pathlib import Path
import json,os,subprocess,time,sys,fcntl,hashlib
import psutil
OUT=Path('/data/hfosp/topic4_sef_hfo/core_driven_input_pilot_20260910')
WT=Path('/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-continuous-core-state-r1')
PY='/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python'
WORKER=WT/'scripts/run_topic4_core_driven_input_pilot.py'
MAX_WORKERS=12
def read(p):return json.loads(p.read_text())
def write(p,d):
    tmp=p.with_suffix('.tmp.json');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2));os.replace(tmp,p)
def result(cid,s):return OUT/'formal/units'/cid/str(s)/'workers/trajectory.json'
def valid(cid,s):
    p=result(cid,s)
    if not p.exists():return False
    d=read(p)
    if d['status']!='COMPLETE' or hashlib.sha256(p.with_suffix('.npz').read_bytes()).hexdigest()!=d['arrays_sha256']:raise RuntimeError(f'invalid completed unit {cid} {s}')
    return True
def identity(process):
    try:
        cmd=process.cmdline()
        if str(WORKER) in cmd and 'worker' in cmd and '--canary' not in cmd:
            return cmd[cmd.index('--candidate')+1],int(cmd[cmd.index('--seed')+1])
    except (psutil.NoSuchProcess,psutil.AccessDenied,ValueError):pass
    return None
def main():
    with (OUT/'finish_queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=read(OUT/'plan.json');jobs=[(c['id'],s) for c in plan['candidates'] for s in plan['seeds']]
        active={};completed=set();failures=[];logs=[]
        for p in psutil.process_iter():
            key=identity(p)
            if key is not None:
                if key not in jobs or key in active:raise RuntimeError('unrecognized or duplicate worker')
                active[key]=p
        for key in jobs:
            if key not in active and valid(*key):completed.add(key)
        pending=[key for key in jobs if key not in completed and key not in active]
        write(OUT/'scheduling_amendment.json',dict(version='finish_fixed_pilot_parallel_v1',old_max_parallel=6,new_max_parallel=MAX_WORKERS,
            reason='User requested fast bounded completion; available memory permits all remaining fixed units. Adopt live processes without restarting trajectories.',
            adopted=[dict(candidate=k[0],seed=k[1],pid=p.pid) for k,p in active.items()],fixed_formal_runs=12,source_sha256=hashlib.sha256(WORKER.read_bytes()).hexdigest(),time_unix=time.time()))
        env=dict(os.environ,LD_LIBRARY_PATH='/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
        while pending or active:
            rss={}
            for key,p in list(active.items()):
                try:alive=p.is_running() and p.status()!=psutil.STATUS_ZOMBIE
                except psutil.NoSuchProcess:alive=False
                if not alive:
                    del active[key]
                    if valid(*key):completed.add(key)
                    else:failures.append(dict(candidate=key[0],seed=key[1],error='worker ended without complete artifact'))
                    continue
                try:
                    tree=[p]+p.children(recursive=True);rss[key]=sum(t.memory_info().rss for t in tree)/2**30
                    if rss[key]>18 or psutil.virtual_memory().available/2**30<40:
                        for t in reversed(tree):t.terminate()
                        failures.append(dict(candidate=key[0],seed=key[1],error='RESOURCE_GUARD'))
                except psutil.NoSuchProcess:pass
            if failures:pending=[]
            reserve=sum(max(0,8-rss.get(k,0)) for k in active)
            while pending and len(active)<MAX_WORKERS and psutil.virtual_memory().available/2**30>60+reserve+8:
                key=pending.pop(0);cid,seed=key;log=(OUT/'logs'/f'formal_{cid}_{seed}.log').open('a');logs.append(log)
                pr=subprocess.Popen([PY,'-u',str(WORKER),'worker','--candidate',cid,'--seed',str(seed)],cwd=WT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active[key]=psutil.Process(pr.pid);reserve+=8
            write(OUT/'status.json',dict(status='PILOT_RUNNING',controller='finish_fixed_pilot_parallel_v1',complete=len(completed),total=12,queued=len(pending),
                active=[dict(pid=p.pid,candidate=k[0],seed=k[1],rss_gib=rss.get(k)) for k,p in active.items()],failures=failures,updated_unix=time.time()))
            if pending or active:time.sleep(5)
        for f in logs:f.close()
        if failures:raise RuntimeError(str(failures))
        write(OUT/'status.json',dict(status='ANALYZING',formal_runs=12,updated_unix=time.time()))
        subprocess.run([PY,str(WT/'scripts/analyze_topic4_core_driven_input_pilot.py')],cwd=WT,env=env,check=True)
        write(OUT/'status.json',dict(status='COMPLETE_PENDING_SCIENTIFIC_REVIEW',formal_runs=12,updated_unix=time.time()))
if __name__=='__main__':
    try:main()
    except Exception as exc:write(OUT/'status.json',dict(status='FAILED',error=repr(exc),updated_unix=time.time()));raise
