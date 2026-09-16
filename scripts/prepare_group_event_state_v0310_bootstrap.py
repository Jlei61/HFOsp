#!/usr/bin/env python3
"""Freeze and optionally run the finite, existing-runner LR audit (18 cells)."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
FILES = ['scripts/run_group_event_state_v039_instrument.py', 'src/__init__.py',
         'src/topic5_group_event_state/__init__.py', 'src/topic5_group_event_state/contract.py',
         'src/topic5_group_event_state/v039/__init__.py',
         'src/topic5_group_event_state/v039/transition.py', 'src/topic5_group_event_state/v039/synthetic.py',
         'src/topic5_group_event_state/v035/__init__.py', 'src/topic5_group_event_state/v035/contracts.py']


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    manifest = root/'bootstrap_manifest.json'
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        for path, sha in saved['source_hashes'].items():
            if digest(root/'source_snapshot'/path) != sha: raise ValueError('Bootstrap snapshot changed')
        return saved
    for name in FILES:
        destination=root/'source_snapshot'/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,destination)
    jobs=[]
    for seed in (20260905,20260906,20260907):
        for lr in (.001,.0003):
            # Complete a matched family triplet before moving to another recipe.
            for family in ('N','L','F'):
                identifier=f'nonlinear_{family}_lr{lr}_seed{seed}'
                folder=root/'bootstrap_lr_audit'/identifier
                argv=[sys.executable,str(root/'source_snapshot/scripts/run_group_event_state_v039_instrument.py'),
                      '--case','nonlinear_transition','--family',family,'--lr',str(lr),
                      '--seed',str(seed),'--max-steps','3200','--patience','12',
                      '--output',str(folder/'card.json')]
                jobs.append(dict(id=identifier,argv=argv,output=str(folder/'card.json')))
    result=dict(status='PREPARED_NOT_STARTED',source_root=str(ROOT),source_hashes={f:digest(ROOT/f) for f in FILES},
                scope='synthetic data seed 39001; optimization/LR diagnostic; no human power claim',
                n_jobs=len(jobs),jobs=jobs,workers_per_gpu=1,devices=[0,1],default_wall_hours=2.,
                no_new_dispatch_after_hours=1.5,automatic_runtime_retries=0)
    atomic(manifest,result)
    return result


def execute(root, manifest):
    lock=(root/'bootstrap.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status_path=root/'bootstrap_status.json'
    if status_path.exists(): raise FileExistsError('An attempt exists; inspect it and preserve failed attempts before any retry')
    start=time.monotonic();pending=list(manifest['jobs']);running={};done=[];quarantined=set()
    def save(status):
        atomic(status_path,dict(status=status,elapsed_seconds=time.monotonic()-start,
            running=[dict(id=v['job']['id'],pid=v['process'].pid,gpu=g) for g,v in running.items()],quarantined_gpus=sorted(quarantined),
            completed=done,pending=[j['id'] for j in pending],manifest_sha256=digest(root/'bootstrap_manifest.json')))
    try:
        while pending or running:
            elapsed=time.monotonic()-start
            for gpu in list(running):
                info=running[gpu];code=info['process'].poll()
                if code is None:continue
                info['log'].close();card_path=Path(info['job']['output'])
                card=json.loads(card_path.read_text()) if code==0 and card_path.exists() else {}
                ok=code==0 and card.get('status')=='COMPLETE'
                if ok:
                    ok=all(digest(card[k])==card[k+'_sha256'] for k in ('checkpoint','scores'))
                    ok=ok and all(manifest['source_hashes'].get(k)==v for k,v in card['source_hashes'].items())
                done.append(dict(id=info['job']['id'],returncode=code,status='COMPLETE' if ok else 'FAILED',source=str(card_path)))
                print(json.dumps(done[-1]),flush=True)
                if not ok:quarantined.add(gpu)
                del running[gpu]
            if elapsed>=7200 or (not running and len(quarantined)==len(manifest['devices'])):break
            if elapsed<5400:
                for gpu in manifest['devices']:
                    if gpu in running or gpu in quarantined or not pending:continue
                    query=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
                    if int(query.strip())<6144:continue
                    job=pending.pop(0);folder=Path(job['output']).parent
                    folder.mkdir(parents=True,exist_ok=True)
                    if Path(job['output']).exists(): raise FileExistsError(job['output'])
                    log=(folder/'run.log').open('w');env=os.environ.copy()
                    env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
                    process=subprocess.Popen(job['argv']+['--device','cuda:0'],cwd=root/'source_snapshot',
                        env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    running[gpu]=dict(job=job,process=process,log=log)
                    print(json.dumps(dict(status='STARTED',id=job['id'],gpu=gpu,pid=process.pid)),flush=True)
            elif not running:break
            save('RUNNING');time.sleep(10)
    finally:
        for gpu,info in list(running.items()):
            try:os.killpg(info['process'].pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:info['process'].wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(info['process'].pid,signal.SIGKILL);info['process'].wait()
            info['log'].close()
            done.append(dict(id=info['job']['id'],status='WALL_TIME_LIMITED',gpu=gpu))
        running.clear()
        complete=len(done)==manifest['n_jobs'] and all(d['status']=='COMPLETE' for d in done)
        save('COMPLETE' if complete else 'INCOMPLETE_REQUIRES_REVIEW')
        lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--execute',action='store_true')
    args=p.parse_args();manifest=prepare(args.root.resolve())
    if args.execute:execute(args.root.resolve(),manifest)
    else:print(json.dumps(dict(status=manifest['status'],n_jobs=manifest['n_jobs'],root=str(args.root))))
