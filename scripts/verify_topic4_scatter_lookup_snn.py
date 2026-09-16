#!/usr/bin/env python3
"""Full-state200ms native high-state parity for an integer-index optimization."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
import sys,time,json,cProfile,pstats,io,hashlib
import numpy as np
import run_topic4_high_state_M_gain_probe as probe

ROOT=probe.ROOT
OUT=ROOT/'results/topic4_sef_hfo/fig5_m_overnight_exploration_20260913/scatter_lookup_qa'
ORIGINAL=probe.OUT/'runs/qa_unchanged_200ms'


def same(a,b,path='engine'):
    if isinstance(a,np.ndarray):
        assert isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b),path
    elif isinstance(a,dict):
        assert isinstance(b,dict) and a.keys()==b.keys(),path
        for key in a:same(a[key],b[key],path+'.'+str(key))
    elif isinstance(a,(tuple,list)):
        assert type(a)==type(b) and len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):same(x,y,path+'.'+str(i))
    else:assert a==b,path


def main():
    import src.topic4_serial_spike_scatter as original_scatter
    from src.topic4_serial_spike_scatter_lookup import scatter
    assert probe.read(OUT/'synthetic_benchmark.json')['whole_ring_bitwise_identical']
    destination=OUT/'full_network';destination.mkdir(exist_ok=True)
    plan=probe.read(probe.OUT/'plan.json')
    plan['computational_variant']=dict(
        original=str(Path(original_scatter.__file__).resolve()),
        replacement=str(ROOT/'src/topic4_serial_spike_scatter_lookup.py'),
        scope='Same edge/addition order and floating operations; integer delay slot calculation moved outside edge loop.')
    probe.write(destination/'plan.json',plan)
    probe.OUT=destination
    original_scatter.scatter=scatter
    job=dict(name='lookup_unchanged_200ms',eta_m=.02,duration_s=.2)
    started=time.time();profiler=cProfile.Profile();profiler.enable()
    r=probe.worker(job)
    profiler.disable();profiler.dump_stats(str(OUT/'full_network_profile.prof'))
    report=io.StringIO();pstats.Stats(profiler,stream=report).sort_stats('cumulative').print_stats(35)
    (OUT/'full_network_profile.txt').write_text(report.getvalue())
    folder=destination/'runs'/job['name']
    with np.load(folder/'observations.npz') as a,np.load(ORIGINAL/'observations.npz') as b:
        assert set(a.files)==set(b.files)
        for key in a.files:same(a[key],b[key],'observation.'+key)
    expected=probe.load_pickle(ORIGINAL/'checkpoint.pkl')['engine']
    actual=probe.load_pickle(folder/'checkpoint.pkl')['engine']
    same(actual,expected)
    result=dict(status='PASS',native_high_state_duration_s=.2,
        full_engine_state_bitwise_identical=True,all_observations_bitwise_identical=True,
        including=['V','refractory','AMPA/GABA states','delay rings','Z','M','global OU','spatial OU','RNG states','counts','raster','current readout'],
        reference=str(ORIGINAL),replacement=str(ROOT/'src/topic4_serial_spike_scatter_lookup.py'),
        replacement_sha256=hashlib.sha256((ROOT/'src/topic4_serial_spike_scatter_lookup.py').read_bytes()).hexdigest(),
        wall_s=time.time()-started,physical_workers_changed=False,
        performance_not_yet_adopted=True,synthetic_benchmark=probe.read(OUT/'synthetic_benchmark.json'))
    probe.write(OUT/'full_network_qa.json',result)
    print(json.dumps(result))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        OUT.mkdir(exist_ok=True)
        probe.write(OUT/'full_network_qa.json',dict(status='FAILED',error=repr(exc),physical_workers_changed=False))
        raise
