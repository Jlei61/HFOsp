#!/usr/bin/env python3
"""Resume the frozen revised-Z physics with the original ordered CPU scatter."""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
import argparse, time
import run_topic4_continuous_resource_recovery as original


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['worker'])
    parser.add_argument('--name', required=True)
    parser.add_argument('--producer-script', required=True)
    args = parser.parse_args()
    assert args.producer_script.endswith('run_topic4_autonomous_recovery.py')
    qa = original.carrier.base.read(original.OUT / 'backend_benchmarks' / (args.name + '_1000ms.json'))
    job = original.carrier.base.read(original.OUT / 'jobs' / (args.name + '.json'))
    assert qa['status'] == 'PASS' and qa['entire_checkpoint_recursive_bitwise']
    assert qa['Z_M_pool_resource_flux_and_RNG_bitwise'] and qa['job'] == job
    folder = original.OUT / 'runs' / args.name
    assert (folder / 'checkpoint.pkl').exists()
    original.carrier.wrap_simulator = lambda fn, device_index: fn
    try:
        original.worker(args.name)
    except Exception as exc:
        original.carrier.base.write(folder / 'failure.json', dict(
            error=repr(exc), time=time.time(), backend='original_ordered_CPU_scatter'))
        raise
