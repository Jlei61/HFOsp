#!/usr/bin/env python3
"""Bounded 7x7 refinement of the existing Fig5 Z-kinetics experiment."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/topic4_sef_hfo/fig5_manual_core_release_v1'
OUT = BASE / 'latency_dense_v1'
THRESHOLD = 95.19851312666987


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def worker(job):
    import run_topic4_fig5_manual_release as original
    original.OUT = OUT
    return original.worker(job)


def controller(workers):
    import numpy as np
    import psutil
    import fcntl
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    tau = np.r_[np.geomspace(2500, 5000, 4), np.geomspace(5000, 10000, 4)[1:]]
    threshold = np.r_[np.linspace(75, THRESHOLD, 4), np.linspace(THRESHOLD, 120, 4)[1:]]
    old = [json.loads(p.read_text()) | {'path': str(p)} for p in (BASE / 'runs').glob('*.json')]
    jobs, reuse = [], []
    for t in tau:
        for h in threshold:
            for seed in (9108401, 9108402, 9108403):
                found = [r for r in old if r.get('status') == 'COMPLETE' and
                         r['job']['seed'] == seed and abs(r['job']['tau_z_ms']-t) < 1e-8 and
                         abs(r['job']['threshold']-h) < 1e-8]
                if found:
                    assert len(found) == 1
                    reuse.append(dict(tau_z_ms=float(t), threshold=float(h), seed=seed, source=found[0]['path']))
                else:
                    jobs.append(dict(name=f'tau{t:.6f}_th{h:.6f}_seed{seed}', seed=seed,
                                     tau_z_ms=float(t), threshold=float(h), duration_ms=24000., detail=False))
    assert len(jobs) == 120 and len(reuse) == 27
    producer_paths = ['scripts/run_topic4_fig5_manual_release.py', 'scripts/topic4_historical_manual_z_common.py',
                      'src/snn_engine/mz_slow_vars.py', 'src/topic4_raster_protocol_engine.py']
    hashes = {s: hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in producer_paths}
    protocol = dict(tau_z_ms=tau.tolist(), threshold=threshold.tolist(), seeds=[9108401,9108402,9108403],
        grid_shape=[7,7], total_runs=147, new_runs=120, reused_runs=reuse, jobs=jobs,
        max_workers=workers, minimum_available_memory_GiB=60,
        scope='Only refine the original tau_Z and depletion-current threshold domain; same historical manual substrate and SNN equations. No new gain parameter.',
        transition='All-E 10-ms binned rate >=200 Hz for 200 consecutive ms; detection time, horizon 24 s.',
        figure='Restricted mean min(T,24s), censoring fraction retained. No invented values between simulated nodes.',
        stopping='Complete these 120 jobs and stop. No adaptive expansion or model selection.',
        producer_hashes=hashes)
    write(OUT/'protocol.json', protocol)
    stop = threading.Event(); completed=[]; failed=[]
    started = time.time()
    # Start long expected jobs first, using only tau, never observed response.
    jobs.sort(key=lambda j: (-j['tau_z_ms'], j['threshold'], j['seed']))
    def launch(job):
        target = OUT/'runs'/(job['name']+'.json')
        if target.exists() and json.loads(target.read_text()).get('status') == 'COMPLETE':
            return dict(name=job['name'], status='COMPLETE')
        while not stop.is_set():
            if psutil.virtual_memory().available > 60*1024**3: break
            time.sleep(5)
        if stop.is_set(): return dict(name=job['name'],status='NOT_DISPATCHED')
        for name, digest in hashes.items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
                stop.set()
                return dict(name=job['name'],status='FAILED',error='Producer changed: '+name)
        spec=OUT/'jobs'/(job['name']+'.json');write(spec,job)
        log=OUT/'logs'/(job['name']+'.log');log.parent.mkdir(exist_ok=True)
        env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                 NUMEXPR_NUM_THREADS='1', TOPIC4_MANUAL_ARM='manual_hard',
                 LD_LIBRARY_PATH='/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib')
        with log.open('w') as f:
            result=subprocess.run([sys.executable,__file__,'worker','--job',str(spec)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:
            stop.set()
            return dict(name=job['name'],status='FAILED',returncode=result.returncode,log=str(log))
        return dict(name=job['name'],status='COMPLETE')
    def status(state):
        write(OUT/'status.json',dict(status=state,new_completed=len(completed),new_total=120,reused=27,
              failed=failed,elapsed_s=time.time()-started,controller_pid=os.getpid(),max_workers=workers))
    status('RUNNING')
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(launch,j) for j in jobs]
        for future in as_completed(futures):
            result=future.result()
            if result['status']=='COMPLETE': completed.append(result['name'])
            elif result['status']=='FAILED': failed.append(result)
            status('DRAINING_AFTER_FAILURE' if stop.is_set() else 'RUNNING')
    status('FAILED' if failed else 'SIMULATIONS_COMPLETE')


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['controller','worker'])
    parser.add_argument('--workers',type=int,default=28);parser.add_argument('--job');args=parser.parse_args()
    if args.mode=='controller':controller(args.workers)
    else:
        job=json.loads(Path(args.job).read_text())
        try:worker(job)
        except Exception as exc:
            write(OUT/'progress'/(job['name']+'.json'),dict(status='FAILED',error=repr(exc),job=job))
            raise
