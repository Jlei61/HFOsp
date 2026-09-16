#!/usr/bin/env python3
"""Synthetic ring-update benchmark only; never dispatches biological conditions."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,time,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.topic4_serial_spike_scatter_lookup import scatter
from src.topic4_serial_spike_gather_ordered import prepare,gather
OUT=ROOT/'results/topic4_sef_hfo/fig5_m_overnight_exploration_20260913/scatter_lookup_qa'


def main():
    rng=np.random.default_rng(905310);ns=32000;nt=40000;degree=160;delay_slots=201
    indptr=np.arange(ns+1,dtype=np.int64)*degree
    dst=rng.integers(0,nt,size=ns*degree,dtype=np.int64)
    delays=rng.integers(0,delay_slots,size=len(dst),dtype=np.int32)
    weights=rng.normal(size=len(dst));prepared=prepare(indptr,dst,delays,weights,nt)
    original=rng.normal(scale=.01,size=(delay_slots,nt));a=original.copy();b=original.copy()
    probe=np.array([0,1,299,31999],dtype=np.int64)
    scatter(a,probe,indptr,dst,delays,weights,199,.81)
    gather(b,probe,*prepared,ns,199,.81);assert np.array_equal(a,b)
    rows=[]
    for fraction in [0.,.01,.05,.1,.25]:
        selections=[np.sort(rng.choice(ns,int(ns*fraction),replace=False)).astype(np.int64) for _ in range(8)]
        timings={'lookup':[],'gather':[]}
        for repeat in range(2):
            results={}
            for variant in (['lookup','gather'] if repeat==0 else ['gather','lookup']):
                ring=original.copy();start=time.perf_counter()
                for k,sources in enumerate(selections):
                    if variant=='lookup':scatter(ring,sources,indptr,dst,delays,weights,197+k,.83)
                    else:gather(ring,sources,*prepared,ns,197+k,.83)
                timings[variant].append(time.perf_counter()-start);results[variant]=ring
            assert np.array_equal(results['lookup'],results['gather'])
        rows.append(dict(active_fraction=fraction,active_sources=int(ns*fraction),timings_s=timings,
            speedup=float(np.median(timings['lookup'])/np.median(timings['gather'])),bitwise_equal=True))
    result=dict(status='PASS',synthetic_data_not_research_data=True,physical_workers_changed=False,
        preparation_extra_bytes=sum(a.nbytes for a in prepared),n_edges=len(dst),n_sources=ns,n_targets=nt,delay_slots=delay_slots,
        rows=rows,full_SNN_validation=False,adopted=False)
    (OUT/'ordered_gather_benchmark.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
