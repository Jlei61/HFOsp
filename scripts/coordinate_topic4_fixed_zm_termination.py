#!/usr/bin/env python3
"""Adopt already running mechanism workers; finish the declared batches together."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import json,signal,subprocess,sys,time
from pathlib import Path
import psutil

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO/'results/topic4_sef_hfo/fig5_fixed_zm_termination_20260915'
FOLDERS=[ROOT/'matched_spatial_round2',ROOT/'hyperpolar_spatial_round3',ROOT/'source_sahp_round4']
if (ROOT/'sahp_bracket_round5/protocol.json').exists():FOLDERS.append(ROOT/'sahp_bracket_round5')
WORKER=REPO/'scripts/run_topic4_zm_matched_spatial_termination.py'

def read(p):return json.loads(p.read_text())
def write(p,x):
    p.parent.mkdir(exist_ok=True,parents=True);temp=p.with_suffix('.new.json')
    temp.write_text(json.dumps(x,indent=2)+'\n');temp.replace(p)
def alive(pid,name):
    try:
        proc=psutil.Process(pid);args=proc.cmdline()
        return proc.status()!=psutil.STATUS_ZOMBIE and 'worker' in args and name in args and any(Path(x).name==WORKER.name for x in args)
    except (psutil.NoSuchProcess,psutil.AccessDenied):return False

def main():
    protocols={f:read(f/'protocol.json') for f in FOLDERS}
    assert read(ROOT/'source_sahp_round4/zero_mechanism_replay_qa.json')['status']=='PASS'
    for folder in FOLDERS:assert read(folder/'equation_qa.json')['status']=='PASS'
    deadline=min(p['deadline_epoch'] for p in protocols.values())
    # Stop only these parent dispatch loops. Their independent worker processes
    # keep their own logs and continue with their in-memory network state.
    stopped=[]
    for proc in psutil.process_iter(['pid','cmdline']):
        args=proc.info['cmdline'] or []
        if 'supervise' not in args or not any(Path(x).name==WORKER.name for x in args):continue
        folder=Path(proc.environ().get('TOPIC4_TERMINATION_OUT',str(FOLDERS[0])))
        if folder not in FOLDERS:continue
        assert Path(proc.cwd())==REPO
        stopped.append(dict(pid=proc.pid,folder=str(folder)));os.kill(proc.pid,signal.SIGTERM)
    contract=dict(time=time.time(),parent_dispatchers_replaced=stopped,no_worker_stopped=True,
        max_combined_mechanism_workers=12,min_available_host_GiB=80,min_free_GPU_MiB=4096,
        deadline_epoch=deadline,scope='Only the declared mechanism jobs in listed protocols, unchanged simulation parameters, seeds and horizons.',folders=[str(f) for f in FOLDERS])
    write(ROOT/'dispatch_takeover.json',contract)
    all_jobs=[(f,j) for f,p in protocols.items() for j in p['initial_jobs']]
    adopted={};pending=[];children={};handles={};failures=[]
    # Interleave batches so the two old controls cannot be starved by later jobs.
    maxn=max(len(p['initial_jobs']) for p in protocols.values())
    for i in range(maxn):
        for folder,p in protocols.items():
            if i>=len(p['initial_jobs']):continue
            j=p['initial_jobs'][i];run=folder/'runs'/j['name']
            if (run/'result.json').exists():continue
            progress=read(run/'progress.json') if (run/'progress.json').exists() else {}
            if progress.get('pid') and alive(progress['pid'],j['name']):adopted[(folder,j['name'])]=progress['pid']
            else:pending.append((folder,j))
    while pending or adopted:
        for key,pid in list(adopted.items()):
            folder,name=key;run=folder/'runs'/name
            if (run/'result.json').exists() or not alive(pid,name):
                if not (run/'result.json').exists():failures.append(dict(folder=str(folder),name=name,pid=pid,reason='Worker exited without result'))
                if key in children:children[key].wait();handles[key].close()
                del adopted[key]
        if failures or time.time()>deadline-300:pending=[]
        while pending and len(adopted)<12 and psutil.virtual_memory().available/2**30>80:
            free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
            gpu=[float(v) for v in free.split()]
            folder,j=pending[0]
            if gpu[j['device']]<4096:break
            pending.pop(0);key=(folder,j['name']);logs=folder/'logs';logs.mkdir(exist_ok=True)
            handle=(logs/(j['name']+'.log')).open('a');env=dict(os.environ,TOPIC4_TERMINATION_OUT=str(folder))
            proc=subprocess.Popen([sys.executable,'-u',str(WORKER),'worker','--name',j['name']],stdout=handle,stderr=subprocess.STDOUT,env=env,cwd=REPO)
            adopted[key]=proc.pid;children[key]=proc;handles[key]=handle
            time.sleep(.5)
        write(ROOT/'combined_dispatch_status.json',dict(time=time.time(),running=[dict(folder=str(f),name=n,pid=pid) for (f,n),pid in adopted.items()],queued=[dict(folder=str(f),name=j['name']) for f,j in pending],failures=failures))
        for folder,p in protocols.items():
            done=sum((folder/'runs'/j['name']/'result.json').exists() for j in p['initial_jobs'])
            write(folder/'supervisor_status.json',dict(time=time.time(),coordinator_pid=os.getpid(),running={n:pid for (f,n),pid in adopted.items() if f==folder},queued=[j['name'] for f,j in pending if f==folder],failures=[v for v in failures if v['folder']==str(folder)]))
            if done==len(p['initial_jobs']):write(folder/'batch_complete.json',dict(time=time.time(),status='FINISHED',failures=[]))
        if pending or adopted:time.sleep(10)
    write(ROOT/'combined_dispatch_complete.json',dict(time=time.time(),failures=failures,complete=all((f/'runs'/j['name']/'result.json').exists() for f,j in all_jobs)))

if __name__=='__main__':main()
