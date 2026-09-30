#!/usr/bin/env python3
"""Execution-only CUDA route for the frozen structural conditional branches."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
from pathlib import Path
import run_topic4_loop_axis_conditional as run
import src.topic4_cuda_ordered_scatter as cuda_backend

SCIENCE = run.OUT
QA = run.PRIMARY / 'qa/axis_cuda_device0_route'


def prepare_qa():
    QA.mkdir(parents=True, exist_ok=True)
    run.OUT = QA
    run.prepare('reference')
    print(str(QA), flush=True)


def worker(root, condition, name, device):
    run.OUT = root
    folder = root / condition / 'runs' / name
    job = run.native.base.read(root / condition / 'jobs' / f'{name}.json')
    runtime = dict(actual_backend='original_cuda_ordered_scatter', actual_device=device,
        planned_job_backend=job['backend'], planned_job_device=job['device'],
        execution_override_only=True, physics_and_job_unchanged=True,
        wrapper_sha256=run.native.base.sha(__file__),
        backend_sha256=run.native.base.sha(cuda_backend.__file__))
    run.native.write(folder / 'runtime_backend.json', runtime)
    run.wrap_simulator = lambda fn, device_index: cuda_backend.wrap_simulator(fn, device_index=device)
    run.worker(condition, name)
    result = run.native.base.read(folder / 'result.json')
    result['runtime_backend'] = runtime
    run.native.write(folder / 'result.json', result)
    run.native.write(folder / 'progress.json', result)


def verify_qa():
    run.OUT = QA
    run.reference_qa()
    gate = run.native.base.read(QA / 'reference/route_qa.json')
    gate.update(device=0, wrapper_sha256=run.native.base.sha(__file__),
        backend_sha256=run.native.base.sha(cuda_backend.__file__),
        scope='Original graph, both full clamped histories, 0.2s each. Complete engine and eight observation streams must match the original serial reference.')
    run.native.write(QA / 'gate.json', gate)
    print(json.dumps(gate), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare-qa', 'worker', 'verify-qa'])
    parser.add_argument('--root', type=Path, default=SCIENCE)
    parser.add_argument('--condition', choices=['reference', 'rotated', 'isotropic'])
    parser.add_argument('--name')
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    if args.command == 'prepare-qa':
        prepare_qa()
    elif args.command == 'verify-qa':
        verify_qa()
    else:
        worker(args.root, args.condition, args.name, args.device)
