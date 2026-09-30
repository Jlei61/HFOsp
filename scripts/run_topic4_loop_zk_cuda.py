#!/usr/bin/env python3
"""Use the already validated original CUDA backend for bounded high-rate branches."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import sys
import run_topic4_loop_zk_conditional as original
from src.topic4_cuda_ordered_scatter import wrap_simulator


def main(name):
    fixed0=original.fixed.worker
    def gpu_worker(target):
        original.carrier.wrap_simulator=wrap_simulator
        return fixed0(target)
    original.fixed.worker=gpu_worker
    job=original.base.read(original.OUT/'jobs'/f'{name}.json')
    assert job['backend']=='cuda_ordered' and job['device']==1
    original.write(original.OUT/'runs'/name/'runtime_backend.json',
                   dict(backend='original_cuda_ordered_scatter',device=1,physics_changed=False))
    original.worker(name)


if __name__=='__main__':main(sys.argv[1])
