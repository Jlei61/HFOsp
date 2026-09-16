#!/usr/bin/env python3
"""Actual200ms full-state SNN parity for the isolated CUDA ring backend."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
import sys,time,json,hashlib
import numpy as np
import run_topic4_high_state_M_gain_probe as probe
from verify_topic4_scatter_lookup_snn import same

ROOT=probe.ROOT
OUT=ROOT/'results/topic4_sef_hfo/fig5_m_overnight_exploration_20260913/cuda_ordered_scatter_qa'
REFERENCE=probe.OUT/'runs/qa_unchanged_200ms'


def main():
    sys.path.insert(0,str(ROOT))
    from src.topic4_cuda_ordered_scatter import wrap_simulator
    assert probe.read(OUT/'synthetic.json')['status']=='PASS'
    destination=OUT/'full_network';destination.mkdir(exist_ok=True)
    plan=probe.read(probe.OUT/'plan.json')
    plan['computational_variant']='CUDA per-target sequential incoming edges, native CPU membrane/slow/input equations unchanged.'
    probe.write(destination/'plan.json',plan)
    probe.OUT=destination
    probe.base.simulate_kick=wrap_simulator(probe.base.simulate_kick,device_index=1)
    job=dict(name='cuda_unchanged_200ms',eta_m=.02,duration_s=.2)
    start=time.time()
    probe.worker(job)
    folder=destination/'runs'/job['name']
    with np.load(folder/'observations.npz') as actual,np.load(REFERENCE/'observations.npz') as expected:
        assert set(actual.files)==set(expected.files)
        for key in actual.files:same(actual[key],expected[key],'observation.'+key)
    expected=probe.load_pickle(REFERENCE/'checkpoint.pkl')['engine']
    actual=probe.load_pickle(folder/'checkpoint.pkl')['engine']
    same(actual,expected)
    source=ROOT/'src/topic4_cuda_ordered_scatter.py'
    result=dict(status='PASS',native_high_state_duration_s=.2,device_index=1,
        full_engine_state_bitwise_identical=True,all_observations_bitwise_identical=True,
        including=['V','refractory','AMPA/GABA states','delay rings','Z','M','global OU','spatial OU','RNG states','counts','raster','current readout'],
        reference=str(REFERENCE),replacement=str(source),replacement_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        wall_s=time.time()-start,physical_workers_changed=False,biological_sample_increment=0,
        performance_not_yet_adopted=True,synthetic_benchmark=str(OUT/'synthetic.json'))
    probe.write(OUT/'full_network_qa.json',result);print(json.dumps(result))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        OUT.mkdir(exist_ok=True)
        probe.write(OUT/'full_network_qa.json',dict(status='FAILED',error=repr(exc),physical_workers_changed=False))
        raise
