#!/usr/bin/env python3
"""Same18branches, at most8workers including up to4GPU0workers; full-state handoffs."""
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
import pickle
import shutil
import psutil

REPO=Path(__file__).resolve().parents[1]
OUT=Path('/data/hfosp/topic4_sef_hfo/fig5_autonomous_loop_zk_20260924')
RUNNER=REPO/'scripts/run_topic4_loop_zk_locality_cpu.py'
CUDA_RUNNER=REPO/'scripts/run_topic4_loop_cuda_override.py'
ANALYZE=REPO/'scripts/analyze_topic4_loop_zk_conditional.py'


def read(path):return json.loads(path.read_text())


def write(path,value):
    tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def alive(proc):
    if isinstance(proc,subprocess.Popen):return proc.poll() is None
    try:return proc.is_running() and proc.status()!=psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:return False


def gpu_resources(active):
    row=subprocess.run(['nvidia-smi','--id=0','--query-gpu=memory.free,uuid','--format=csv,noheader,nounits'],
        capture_output=True,text=True)
    if row.returncode:return 0.,0.,0.
    amount,uuid=[v.strip() for v in row.stdout.strip().split(',')];raw=float(amount)/1024
    query=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits'],
        capture_output=True,text=True)
    memory={}
    if query.returncode==0:
        for line in query.stdout.strip().splitlines():
            pid,device,used=[v.strip() for v in line.split(',')]
            if device==uuid and used.replace('.','').isdigit():memory[int(pid)]=float(used)/1024
    # Native complete allocations are~1.073GiB. Until the allocation is visible,
    # carry the startup reservation across polling cycles, not just one loop.
    reserved=sum(max(0.,1.25-memory.get(proc.pid,0.)) for proc,_,backend in active.values()
        if backend=='cuda_ordered:0' and memory.get(proc.pid,0.)<1.)
    return raw,reserved,max(0.,raw-reserved)


def launch(name,backend,logs):
    handle=(logs/f'{name}.log').open('a')
    command=[sys.executable,str(CUDA_RUNNER),'worker','--device','0','--name',name] if backend=='cuda_ordered:0' else [sys.executable,str(RUNNER),'worker',name]
    proc=subprocess.Popen(command,cwd=REPO,stdout=handle,stderr=subprocess.STDOUT)
    return proc,handle,backend


def handoff(name,proc,logs):
    """Resume a fully committed checkpoint; recompute at most one uncommitted block."""
    folder=OUT/'runs'/name;cp=folder/'checkpoint.pkl'
    if not cp.exists():return None
    progress=read(folder/'progress.json')
    if progress.get('status')!='RUNNING':return None
    protocol=read(OUT/'protocol.json')
    for path,digest in protocol['source_hashes'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    for path,key in [('scripts/run_topic4_loop_zk_conditional.py','runner_sha256'),
                     ('scripts/run_topic4_autonomous_recovery.py','producer_sha256'),
                     ('scripts/run_topic4_fixed_zm_termination.py','wrapper_sha256')]:
        assert hashlib.sha256((REPO/path).read_bytes()).hexdigest()==protocol[key],path
    proc=psutil.Process(proc.pid)
    assert proc.cmdline()[-1]==name and any('run_topic4_loop_zk_' in v for v in proc.cmdline())
    proc.suspend();terminated=False
    try:
        with cp.open('rb') as f:saved=pickle.load(f)
        step=int(saved['engine']['step'])
        if round(progress['time_s']*10000)<step:return None
        expected=read(OUT/'jobs'/f'{name}.json')
        assert saved['job']==expected and saved['identity']==protocol['identity']
        required=['chunks','actual_current_chunks','conditional_drift_chunks','feedback_chunks',
                  'global_response_chunks','intrinsic_adaptation_chunks','mechanism_chunks','regional_chunks']
        committed={}
        for subdir in required:
            paths=list((folder/subdir).glob('*.npz'))
            if not paths or any('.tmp.' in p.name for p in paths):return None
            endpoints=[int(p.stem.split('_')[-1]) for p in paths]
            if max(endpoints)!=step:return None
            committed[subdir]=step
        # Saved arrays/RNG/rings are copied byte-for-byte before execution changes.
        archive=OUT/'backend_handoffs/gpu0'/name;archive.mkdir(parents=True,exist_ok=True)
        backup=archive/f'checkpoint_step{step}.pkl'
        shutil.copy2(cp,backup)
        digest=hashlib.sha256(backup.read_bytes()).hexdigest()
        assert hashlib.sha256(cp.read_bytes()).hexdigest()==digest
        record=dict(status='CHECKPOINT_VERIFIED',old_pid=proc.pid,step=step,time_s=step*.0001,
            checkpoint_sha256=digest,backup=str(backup),committed_observation_endpoints=committed,
            identical_job=True,identical_identity=True,complete_engine_unmodified=True,
            source_files_verified=len(protocol['source_hashes']),
            latest_reported_time_s=progress['time_s'],
            maximum_uncommitted_time_to_recompute_s=expected['checkpoint_s'],
            persisted_observations_discarded=False,
            new_backend='cuda_ordered:0',actual_device=0,planned_job_unchanged=True,epoch=time.time())
        write(archive/'handoff.json',record)
        proc.terminate();proc.resume();proc.wait(timeout=10);terminated=True
        child,handle,backend=launch(name,'cuda_ordered:0',logs)
        record.update(status='RESUMED_FROM_VERIFIED_CHECKPOINT',new_pid=child.pid,epoch=time.time())
        write(archive/'handoff.json',record)
        return child,handle,backend
    finally:
        if not terminated and alive(proc):proc.resume()


def main():
    lock=(OUT/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gate=read(OUT/'qa/locality_native_gate.json');assert gate['status']=='PASS'
    for path,key in [(RUNNER,'wrapper_sha256'),(REPO/'scripts/topic4_loop_locality_cpu.py','source_sha256')]:
        assert hashlib.sha256(path.read_bytes()).hexdigest()==gate[key]
    gpu_gate=read(OUT/'qa/cuda_device0_route/gate.json');assert gpu_gate['status']=='PASS'
    for path,key in [(CUDA_RUNNER,'wrapper_sha256'),(REPO/'src/topic4_cuda_ordered_scatter.py','backend_sha256')]:
        assert hashlib.sha256(path.read_bytes()).hexdigest()==gpu_gate[key]
    names=read(OUT/'queue.json')['names'];assert len(names)==18
    central=[n for n in names if n.startswith('z0.75_k2_')]
    names=central+[n for n in names if n not in central]
    old=read(OUT/'status.json');active={};failed=old.get('failed',{})
    for name,row in old.get('active',{}).items():
        try:proc=psutil.Process(row['pid'])
        except psutil.NoSuchProcess:continue
        if not alive(proc):continue
        cmd=proc.cmdline()
        assert cmd[-1]==name and any('run_topic4_loop_zk_' in v or 'run_topic4_loop_cuda_override.py' in v for v in cmd),(name,cmd)
        backend='cuda_ordered:0' if any('run_topic4_loop_cuda_override.py' in v for v in cmd) else 'serial_cpu_adopted' if 'conditional.py' in ' '.join(cmd) else 'locality_cpu'
        active[name]=(proc,None,backend)
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
        raw_free,startup_reserved,free=gpu_resources(active)
        ngpu=sum(b=='cuda_ordered:0' for _,_,b in active.values())
        heavy=[n for n in active if n.startswith(('z0.25_k0.02_','z0.25_k2_'))]
        candidates=heavy+[n for n in active if n not in heavy]
        for name in candidates:
            proc,handle,backend=active[name]
            if backend=='cuda_ordered:0' or ngpu>=4 or available<88 or free<9.3 or cpu_busy>65:continue
            progress=read(OUT/'runs'/name/'progress.json')
            job=read(OUT/'jobs'/f'{name}.json')
            if progress.get('phase')!='HIGH' or job['horizon_s']-progress.get('time_s',job['horizon_s'])<=4:continue
            migrated=handoff(name,proc,logs)
            if migrated:
                if handle:handle.close()
                active[name]=migrated;available-=8;free-=1.25;startup_reserved+=1.25;ngpu+=1
        ncpu=sum(backend!='cuda_ordered:0' for _,_,backend in active.values())
        while pending and len(active)<8 and available>=88 and cpu_busy<=65 and not failed:
            if ncpu<4:
                backend='locality_cpu';ncpu+=1;cpu_busy+=10
            elif ngpu<4 and free>=9.3:
                backend='cuda_ordered:0';ngpu+=1;free-=1.25;startup_reserved+=1.25;cpu_busy+=2
            elif ncpu<8:
                backend='locality_cpu';ncpu+=1;cpu_busy+=10
            else:break
            name=pending.pop(0);active[name]=launch(name,backend,logs);available-=8
        detail={}
        for name,(proc,_,backend) in active.items():
            path=OUT/'runs'/name/'progress.json';row=read(path) if path.exists() else {}
            detail[name]=dict(pid=proc.pid,time_s=row.get('time_s'),state=row.get('status'),backend=backend)
        status=dict(stage='RUNNING' if pending or active else 'FINISHED_WITH_FAILURES' if failed else 'COMPLETE',
            supervisor_pid=os.getpid(),updated_epoch=time.time(),completed=complete,pending=pending,active=detail,failed=failed,
            total=18,available_memory_GiB=available,backend='Validated locality CPU and original CUDA; immutable physical jobs, explicit runtime device overrides.',
            max_total_workers=8,max_locality_workers=8,max_GPU0_workers=4,CPU_busy_percent_at_dispatch=cpu_busy,
            GPU0_free_GiB=raw_free,GPU0_pending_startup_reserved_GiB=startup_reserved,
            GPU0_effective_free_for_dispatch_GiB=free,GPU0_below_nominal8GiB_reserve=raw_free<8,
            memory_policy='Reserve80GiB host+8GiB startup; GPU0>=9.3GiB before dispatch reserves8GiB plus1.25GiB estimated startup. At most8total including up to4GPU0; CPU fallback retains existing workers, CPU<=65percent before dispatch/handoff. OtherGPUjobs remain running.',
            new_gpu_processes=ngpu,diagnostic_only=True)
        write(OUT/'status.json',status)
        if pending or active:time.sleep(15)


if __name__=='__main__':main()
