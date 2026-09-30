#!/usr/bin/env python3
"""Benchmark and validate the locality-only backend before scientific adoption."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import json
import time
import numpy as np
import run_topic4_loop_zk_conditional as native
from topic4_loop_locality_cpu import Manager,wrap_simulator
import src.topic4_serial_spike_scatter as serial


def worker(name):
    fixed0=native.fixed.worker
    def cpu_worker(target):
        native.carrier.wrap_simulator=wrap_simulator
        return fixed0(target)
    native.fixed.worker=cpu_worker
    native.worker(name)


def bench():
    s,_,_,identity=native.carrier.base.old.setup(9108405)
    from src.topic4_raster_protocol_engine import _flatten_by_source
    bins=[i for i,m in enumerate(s.net['ampa_by_delay']) if m.nnz]
    ptr,dst,delay,weight=_flatten_by_source(s.net['ampa_by_delay'],bins,s.n_e)
    shape=(s.net['max_delay_steps']+1,s.n_e+s.n_i)
    a=np.zeros(shape);b=np.zeros(shape);manager=Manager()
    rng=np.random.default_rng(920924)
    for fraction in [.005,.05]:
        fired=np.sort(rng.choice(s.n_e,round(s.n_e*fraction),replace=False)).astype(np.int64)
        serial.scatter(a,fired,ptr,dst,delay,weight,0,.7)
        manager.scatter(b,fired,ptr,dst,delay,weight,0,.7)
    manager.flush();assert np.array_equal(a,b)
    rows=[]
    for fraction in [.005,.02,.05,.1]:
        frames=[np.sort(rng.choice(s.n_e,round(s.n_e*fraction),replace=False)).astype(np.int64) for _ in range(8)]
        a.fill(0);b.fill(0)
        for item in manager.items.values():item.local.fill(0)
        start=time.time()
        for k,spk in enumerate(frames):
            a[k]=0.
            serial.scatter(a,spk,ptr,dst,delay,weight,k,.7)
        ts=time.time()-start;start=time.time()
        for k,spk in enumerate(frames):
            manager.before_step(k);b[k]=0.
            manager.scatter(b,spk,ptr,dst,delay,weight,k,.7)
        tp=time.time()-start;manager.flush()
        exact=np.array_equal(a,b);assert exact
        rows.append(dict(spike_fraction=fraction,serial_seconds=ts,locality_seconds=tp,speedup=ts/tp,bitwise=exact))
    native.write(native.OUT/'qa/locality_scatter_benchmark.json',dict(status='PASS',rows=rows,threads=8,identity=identity))
    print(json.dumps(rows),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['bench','worker']);p.add_argument('name',nargs='?');a=p.parse_args()
    bench() if a.command=='bench' else worker(a.name)
