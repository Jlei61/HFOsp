#!/usr/bin/env python3
"""Continue the same bounded M-gain diagnostic on validated CUDA device0."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
import json,hashlib
import run_topic4_high_state_M_gain_probe as probe


def main():
    qa_path=probe.WINDOW/'cuda_ordered_scatter_qa/full_network_qa.json'
    qa=probe.read(qa_path)
    assert qa['status']=='PASS' and qa['full_engine_state_bitwise_identical'] and qa['all_observations_bitwise_identical']
    assert hashlib.sha256(Path(qa['replacement']).read_bytes()).hexdigest()==qa['replacement_sha256']
    from src.topic4_cuda_ordered_scatter import wrap_simulator
    probe.base.simulate_kick=wrap_simulator(probe.base.simulate_kick,device_index=0)
    probe.write(probe.OUT/'computational_backend.json',dict(backend='CUDA_ordered_incoming',device=0,
        validation=str(qa_path),replacement_sha256=qa['replacement_sha256'],
        same_existing_conditions_and_endpoints=True,new_independent_samples=0,pid=os.getpid(),
        executor=str(Path(__file__).resolve()),executor_sha256=probe.sha(__file__)))
    probe.supervise()
    import analyze_topic4_high_state_M_gain_probe as analyze
    analyze.main()


if __name__=='__main__':main()
