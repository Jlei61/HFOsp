#!/usr/bin/env python3
"""Run a bounded, frozen repair queue while independent code repairs proceed."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def atomic(path, payload):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
    os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def gpu_snapshot():
    call = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total',
                           '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
    if call.returncode:
        return {'error': call.stderr.strip()}
    return {'devices': [dict(zip(('index', 'utilization_percent', 'memory_used_mib', 'memory_total_mib'),
                                (int(x.strip()) for x in line.split(','))))
                        for line in call.stdout.strip().splitlines()]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--poll-seconds', type=float, default=10)
    args = parser.parse_args()
    if not 1 <= args.poll_seconds <= 60:
        parser.error('poll interval must be 1..60 seconds')
    plan = json.loads(args.manifest.read_text())
    cpu_workers = int(plan.get('cpu_workers', 0))
    if cpu_workers < 0 or cpu_workers > 8:
        raise ValueError('CPU worker count must be 0..8')
    workers_per_gpu = int(plan.get('workers_per_gpu', 1))
    if not 1 <= workers_per_gpu <= 4:
        raise ValueError('workers_per_gpu must be 1..4')
    slots = list(range(cpu_workers)) if cpu_workers else [gpu for gpu in plan['gpus'] for _ in range(workers_per_gpu)]
    expected_flags = plan.get('expected_output_flags', {
        'development_targets_read': False, 'sealed_partition_opened': False,
        'seizure_targets_read': False})
    out = args.manifest.parent
    lock = (out / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if len({j['id'] for j in plan['jobs']}) != len(plan['jobs']):
        raise ValueError('duplicate job ids')
    for file, digest in plan['source_hashes'].items():
        if sha(Path(plan['source_root']) / file) != digest:
            raise ValueError(f'frozen source mismatch: {file}')
    status_path = out / 'queue_status.json'
    if status_path.exists():
        raise FileExistsError('existing queue state; inspect interrupted jobs before a new registered run')
    states = {job['id']: {'status': 'PENDING'} for job in plan['jobs']}
    active = {}
    logs = out / 'logs'; logs.mkdir(exist_ok=True)
    started = time.time()
    while True:
        for key, (process, gpu, handle, job) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            state = states[key]
            state.update(returncode=code, finished_at=time.time())
            result_path = Path(job['output'])
            if code == 0 and result_path.exists():
                try:
                    result = json.loads(result_path.read_text())
                    if result['status'] not in ('COMPLETE', 'NOT_ESTIMABLE'):
                        raise ValueError('unrecognized result status')
                    if any(result.get(k) is not value for k, value in expected_flags.items()):
                        raise ValueError('partition contract missing or violated')
                    state.update(status=result['status'], output_sha256=sha(result_path))
                except Exception as error:
                    state.update(status='FAILED', error=repr(error))
            else:
                state.update(status='FAILED', error='worker failed or output missing; see log')
            del active[key]
        resource = gpu_snapshot()
        waiting_for = []
        for dependency in plan.get('wait_for_queues_to_finish', []):
            path = Path(dependency)
            state = json.loads(path.read_text()) if path.exists() else {}
            if state.get('status') not in ('COMPLETE', 'FAILED'):
                waiting_for.append(dependency)
        for gpu in slots:
            occupied = Counter(value[1] for value in active.values())
            if occupied[gpu] >= (1 if cpu_workers else workers_per_gpu) or waiting_for:
                continue
            # Do not compete with jobs launched outside this repair queue.
            device = next((d for d in resource.get('devices', []) if d['index'] == gpu), None)
            if not cpu_workers and not occupied[gpu] and (device is None or device['memory_used_mib'] > 512 or device['utilization_percent'] > 10):
                continue
            job = next((j for j in plan['jobs'] if states[j['id']]['status'] == 'PENDING'), None)
            if job is None:
                continue
            key = job['id']
            try:
                for file, digest in job['input_hashes'].items():
                    if sha(file) != digest:
                        raise ValueError(f'input mismatch: {file}')
                if Path(job['output']).exists():
                    raise FileExistsError('refusing existing output')
                handle = (logs / f'{key}.log').open('w')
                env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '' if cpu_workers else str(gpu), 'PYTHONUNBUFFERED': '1',
                       'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                       'NUMEXPR_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
                process = subprocess.Popen(job['argv'], cwd=plan['source_root'], env=env,
                                           stdout=handle, stderr=subprocess.STDOUT)
                active[key] = (process, gpu, handle, job)
                states[key].update(status='RUNNING', pid=process.pid, gpu=gpu, started_at=time.time(),
                                    log=str(logs / f'{key}.log'))
            except Exception as error:
                states[key].update(status='FAILED', error=repr(error))
        pending = sum(s['status'] == 'PENDING' for s in states.values())
        failed = sum(s['status'] == 'FAILED' for s in states.values())
        done = not active and not pending
        payload = {'status': ('FAILED' if failed else 'COMPLETE') if done else 'RUNNING',
                   'scope': plan['scope'], 'manifest_sha256': sha(args.manifest),
                   'updated_at': time.time(), 'started_at': started, 'jobs': states,
                   'pending': pending, 'running': len(active), 'failed': failed,
                   'complete': sum(s['status'] == 'COMPLETE' for s in states.values()),
                   'not_estimable': sum(s['status'] == 'NOT_ESTIMABLE' for s in states.values()),
                   'gpu': resource,
                   'cpu_workers': cpu_workers,
                   'workers_per_gpu': workers_per_gpu,
                   'waiting_for_queues': waiting_for,
                   'idle_reason': ('no remaining jobs in this registered batch' if done else
                                   'waiting for earlier resource queue to finish' if waiting_for else
                                   'waiting for available GPU' if pending and not active else None),
                   'entire_repair_goal_complete': False}
        atomic(status_path, payload)
        with (out / 'monitor.jsonl').open('a') as handle:
            handle.write(json.dumps({k: v for k, v in payload.items() if k != 'jobs'}) + '\n')
        if done:
            return 1 if failed else 0
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    raise SystemExit(main())
