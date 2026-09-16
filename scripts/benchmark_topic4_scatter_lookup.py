#!/usr/bin/env python3
"""Synthetic numerical/performance checks; no scientific simulation output."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
import sys,json,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.topic4_serial_spike_scatter import scatter as original
from src.topic4_serial_spike_scatter_lookup import scatter as lookup

OUT=ROOT/'results/topic4_sef_hfo/fig5_m_overnight_exploration_20260913/scatter_lookup_qa'


def main():
    OUT.mkdir(exist_ok=True)
    rng=np.random.default_rng(2026091303)
    nsrc,ntarget,nlag,degree=3200,40000,201,400
    ptr=np.arange(nsrc+1,dtype=np.int64)*degree
    dst=rng.integers(0,ntarget,size=nsrc*degree,dtype=np.int64)
    delay=rng.integers(0,nlag,size=nsrc*degree,dtype=np.int32)
    weight=rng.uniform(0,5,size=nsrc*degree)
    samples=[np.sort(rng.choice(nsrc,1000,replace=False)) for _ in range(12)]
    ring=rng.uniform(0,100,size=(nlag,ntarget));a=ring.copy();b=ring.copy()
    # Compile both before timing; cases include ring wraps, repeated targets,
    # non-unit gain and an empty firing set, preserving addition order.
    original(a,samples[0],ptr,dst,delay,weight,755000,1.)
    lookup(b,samples[0],ptr,dst,delay,weight,755000,1.)
    assert np.array_equal(a,b)
    cases=[(0,1.),(200,.92),(201,2.),(755003,1.)]
    for (step,gain),src in zip(cases,samples[1:5]):
        original(a,src,ptr,dst,delay,weight,step,gain)
        lookup(b,src,ptr,dst,delay,weight,step,gain)
        assert np.array_equal(a,b)
    original(a,np.zeros(0,np.int64),ptr,dst,delay,weight,99,1.)
    lookup(b,np.zeros(0,np.int64),ptr,dst,delay,weight,99,1.)
    assert np.array_equal(a,b)
    timings={'original':[],'lookup':[]}
    for repetition in range(3):
        order=[('original',original),('lookup',lookup)]
        if repetition%2:order.reverse()
        for name,fn in order:
            work=ring.copy();start=time.perf_counter()
            for i,src in enumerate(samples):fn(work,src,ptr,dst,delay,weight,755000+i,1.)
            timings[name].append(time.perf_counter()-start)
    result=dict(status='PASS',synthetic_data_not_research_data=True,whole_ring_bitwise_identical=True,
        shape=dict(targets=ntarget,delay_slots=nlag,edges=len(weight),active_sources=1000),timings_s=timings,
        median_speedup=float(np.median(timings['original'])/np.median(timings['lookup'])),
        full_SNN_parity_pending=True,physical_workers_changed=False)
    (OUT/'synthetic_benchmark.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
