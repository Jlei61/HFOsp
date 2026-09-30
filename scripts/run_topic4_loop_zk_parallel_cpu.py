#!/usr/bin/env python3
"""Only changes the accumulation backend of the frozen conditional runner."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import hashlib
import json
import time
import numpy as np
import run_topic4_loop_zk_conditional as original
from topic4_loop_ordered_cpu import TargetScatter
import src.topic4_serial_spike_scatter as serial


def worker(name):
    serial.scatter=TargetScatter(serial.scatter,threads=8)
    original.worker(name)


def bench():
    s,tr,frozen,identity=original.carrier.base.old.setup(9108405)
    from src.topic4_raster_protocol_engine import _flatten_by_source
    bins=[i for i,m in enumerate(s.net['ampa_by_delay']) if m.nnz]
    ptr,dst,delay,weight=_flatten_by_source(s.net['ampa_by_delay'],bins,s.n_e)
    shape=(s.net['max_delay_steps']+1,s.n_e+s.n_i)
    rng=np.random.default_rng(920924);manager=TargetScatter(serial.scatter,threads=8,threshold=0)
    a=np.zeros(shape);b=np.zeros(shape);rows=[]
    # Compile both signatures before timing, including a nonunit multiplier.
    small=np.array([0,20,50],dtype=np.int64)
    serial.scatter(a,small,ptr,dst,delay,weight,0,.7)
    manager(b,small,ptr,dst,delay,weight,0,.7)
    assert np.array_equal(a,b)
    for fraction in [.005,.02,.05,.1]:
        frames=[np.sort(rng.choice(s.n_e,round(s.n_e*fraction),replace=False)).astype(np.int64) for _ in range(8)]
        a.fill(0);b.fill(0);start=time.time()
        for k,spk in enumerate(frames):serial.scatter(a,spk,ptr,dst,delay,weight,k,.7)
        ts=time.time()-start;start=time.time()
        for k,spk in enumerate(frames):manager(b,spk,ptr,dst,delay,weight,k,.7)
        tp=time.time()-start;exact=np.array_equal(a,b);assert exact
        rows.append(dict(spike_fraction=fraction,serial_seconds=ts,parallel_seconds=tp,speedup=ts/tp,bitwise=exact))
    original.write(original.OUT/'qa/parallel_scatter_benchmark.json',dict(status='PASS',rows=rows,threads=8,
        identity=identity,preserves='Native ascending-source/within-row weight addition order independently for every postsynaptic target.'))
    print(json.dumps(rows),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['bench','worker']);p.add_argument('name',nargs='?')
    a=p.parse_args()
    if a.command=='bench':bench()
    else:worker(a.name)


if __name__=='__main__':main()
