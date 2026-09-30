#!/usr/bin/env python3
"""Compare ordered scatter paths under the current load; no simulation dispatch."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import psutil
import run_topic4_loop_zk_conditional as native
from topic4_loop_locality_cpu import Manager
import src.topic4_serial_spike_scatter as serial
from src.topic4_raster_protocol_engine import _flatten_by_source


def main():
    available=psutil.virtual_memory().available/2**30;cpu=psutil.cpu_percent(interval=1)
    assert available>=88 and cpu<=65,(available,cpu)
    s,_,_,identity=native.carrier.base.old.setup(9108405)
    bins=[i for i,m in enumerate(s.net['ampa_by_delay']) if m.nnz]
    ptr,dst,delay,weight=_flatten_by_source(s.net['ampa_by_delay'],bins,s.n_e)
    shape=(s.net['max_delay_steps']+1,s.n_e+s.n_i)
    reference=np.zeros(shape);target=np.zeros(shape);manager=Manager()
    rng=np.random.default_rng(920925);rows=[]
    for fraction in [.02,.05,.1]:
        frames=[np.sort(rng.choice(s.n_e,round(s.n_e*fraction),replace=False)).astype(np.int64) for _ in range(12)]
        # Warm up all three signatures before comparing; all paths retain the
        # same per-target source-addition order, including pending delay bins.
        reference.fill(0)
        for i,fired in enumerate(frames):
            reference[i]=0.;serial.scatter(reference,fired,ptr,dst,delay,weight,i,.7)
        timings={}
        for mode in ['local_source','local_target','legacy_source']:
            manager.threshold=10**18 if mode=='local_source' else 0
            elapsed=[]
            for repeat in range(4):
                target.fill(0)
                for item in manager.items.values():item.local.fill(0)
                start=time.perf_counter()
                for i,fired in enumerate(frames):
                    if mode=='legacy_source':
                        target[i]=0.;serial.scatter(target,fired,ptr,dst,delay,weight,i,.7)
                    else:
                        manager.before_step(i);target[i]=0.
                        manager.scatter(target,fired,ptr,dst,delay,weight,i,.7)
                if mode!='legacy_source':manager.flush()
                duration=time.perf_counter()-start
                assert np.array_equal(reference,target),(fraction,mode,repeat)
                if repeat:elapsed.append(duration)
            timings[mode]=dict(wall_seconds=elapsed,median_wall_seconds=float(np.median(elapsed)))
        rows.append(dict(spike_fraction=fraction,frames=12,all_outputs_bitwise_equal=True,timings=timings))
        print(json.dumps(rows[-1]),flush=True)
    native.write(native.OUT/'qa/scatter_path_load_benchmark.json',dict(status='PASS',rows=rows,identity=identity,
        producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path('scripts/topic4_loop_locality_cpu.py').read_bytes()).hexdigest(),
        initial_available_GiB=available,initial_CPU_busy_percent=cpu,
        scope='Synthetic sorted spike sets on the original full graph under the current host load. This is a kernel timing check, not full-trajectory throughput or a scientific network experiment.'))


if __name__=='__main__':main()
