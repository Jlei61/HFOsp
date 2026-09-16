#!/usr/bin/env python3
"""Bounded CUDA parity/performance trial, no biological simulations."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
import sys,json,time,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from numba import cuda
from src.topic4_cuda_ordered_scatter import Manager
from src.topic4_serial_spike_scatter_lookup import scatter

OUT=ROOT/'results/topic4_sef_hfo/fig5_m_overnight_exploration_20260913/cuda_ordered_scatter_qa'


def main():
    cuda.select_device(1)
    rng=np.random.default_rng(902613)
    ns=6400;nt=8000;degree=200;steps=80;slots=37
    ptr=np.arange(ns+1,dtype=np.int64)*degree
    dst=rng.integers(nt,size=ns*degree,dtype=np.int32)
    delay=rng.integers(0,slots,size=len(dst),dtype=np.int32)
    weight=rng.normal(0,1,size=len(dst)).astype(np.float64)
    # Include duplicate target/slot edges and signed/cancellation-sensitive sums.
    dst[1:20]=dst[0];delay[1:20]=delay[0]
    weight[1:4]=[1e10,1e-7,-1e10]
    sources=[]
    for k in range(steps):
        fraction=0 if k%11 in [4,5,6] else .05 if k%2 else .2
        sources.append(np.flatnonzero(rng.random(ns)<fraction))
    gains=np.where(np.arange(steps)%3==0,1.1542317,1.)
    original=rng.normal(size=(slots,nt));actual=original.copy();manager=Manager()
    # Warm only the original CPU dispatcher; first GPU launch includes compilation.
    scatter(original.copy(),sources[0],ptr,dst,delay,weight,0,1.)
    cpu_seconds=gpu_seconds=0.;checkpoint_count=0
    for k,(spk,gain) in enumerate(zip(sources,gains)):
        start=time.perf_counter();row=k%slots
        expected_arrival=original[row].copy();original[row]=0
        if len(spk):scatter(original,spk,ptr,dst,delay,weight,k,gain)
        cpu_seconds+=time.perf_counter()-start
        start=time.perf_counter();manager.before_step(k)
        assert np.array_equal(actual[row],expected_arrival),('arrival',k)
        actual[row]=0
        if len(spk):manager.scatter(actual,spk,ptr,dst,delay,weight,k,gain)
        cuda.synchronize();gpu_seconds+=time.perf_counter()-start
        if k%9==0 or k==steps-1:
            manager.flush();assert np.array_equal(original,actual),('whole_ring',k)
            checkpoint_count+=1
    # Measure warm dispatch plus actual row transfer separately from compile/prep.
    benchmarks=[]
    for fraction in [.02,.05,.2]:
        a=np.zeros((slots,nt));b=a.copy();m=Manager()
        spk=np.sort(rng.choice(ns,int(ns*fraction),replace=False))
        m.scatter(b,spk,ptr,dst,delay,weight,0,1.);m.flush();b.fill(0)
        # Reset resident state to the corresponding host state for this benchmark.
        item=m.items[id(b)];item.d_ring.copy_to_device(b)
        t0=time.perf_counter()
        for k in range(40):
            a[k%slots]=0;scatter(a,spk,ptr,dst,delay,weight,k,1.)
        cpu=time.perf_counter()-t0
        t0=time.perf_counter()
        for k in range(40):
            m.before_step(k);b[k%slots]=0;m.scatter(b,spk,ptr,dst,delay,weight,k,1.)
        m.flush();gpu=time.perf_counter()-t0
        assert np.array_equal(a,b)
        benchmarks.append(dict(source_fraction=fraction,cpu_s=cpu,gpu_s=gpu,speedup=cpu/gpu))
    source=ROOT/'src/topic4_cuda_ordered_scatter.py'
    result=dict(status='PASS',device_index=1,device=str(cuda.get_current_device().name),
        rows_and_full_ring_bitwise_identical=True,steps=steps,checkpoint_comparisons=checkpoint_count,
        silent_frames_tested=True,ring_wrap_tested=True,duplicate_edges_tested=True,
        nonunit_gain_and_signed_weights_tested=True,benchmarks=benchmarks,
        preparation_inclusive_gpu_s=gpu_seconds,cpu_s=cpu_seconds,
        source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        physical_workers_modified=False,full_SNN_verification='PENDING',biological_sample_increment=0)
    OUT.mkdir(exist_ok=True);(OUT/'synthetic.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        OUT.mkdir(exist_ok=True);(OUT/'synthetic.json').write_text(json.dumps(dict(status='FAILED',error=repr(exc),physical_workers_modified=False),indent=2)+'\n')
        raise
