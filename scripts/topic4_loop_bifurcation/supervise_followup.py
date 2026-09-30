#!/usr/bin/env python3
"""Execution-only concurrency amendment; identical frozen30-job queue."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import fcntl
import argparse
import subprocess
import time
import psutil
import numpy as np
from campaign import REPO, ROOT, NATIVE, PYTHON, read, write, sha
parser=argparse.ArgumentParser()
parser.add_argument('--root',type=type(ROOT),required=True)
args=parser.parse_args()
NATIVE=args.root
MAX_WORKERS=6
PER_GPU=3
def prepare(): return read(NATIVE/'queue.json')
import analyze_topic4_loop_zk_conditional as analysis

RUNNER = REPO / 'scripts/topic4_loop_bifurcation/followup_campaign.py'


def process_alive(proc):
    try:
        if isinstance(proc, subprocess.Popen):
            return proc.poll() is None
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def gpu_free():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    return {int(line.split(',')[0]): float(line.split(',')[1])/1024
            for line in result.stdout.strip().splitlines()}


def summarize(names):
    analysis.OUT = NATIVE
    rows = []
    reference = None
    for name in names:
        if not (NATIVE / 'runs' / name / 'result.json').exists():
            continue
        result = read(NATIVE / 'runs' / name / 'result.json')
        if result['status'] != 'COMPLETE':
            continue
        row, inputs = analysis.analyze(name)
        row['full_horizon'] = abs(row['duration_s']-120.)<1e-9
        row['interpretation'] = 'Finite120s continuation; persistence does not certify an attractor or bistability.'
        if reference is not None:
            assert np.array_equal(inputs, reference), name
        reference = inputs
        rows.append(row)
    write(NATIVE / 'analysis_summary.json', dict(completed=len(rows), total=len(names), rows=rows,
          common_future_input_records_exact=True, formal_bifurcation='NOT_ESTABLISHED',
          human_review='PENDING', status='COMPLETE' if len(rows)==len(names) else 'PARTIAL'))


def main():
    NATIVE.mkdir(parents=True, exist_ok=True)
    lock = (NATIVE / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    queue = prepare()
    names = queue['names']
    assert len(names) == 6
    active = {}
    if (NATIVE / 'status.json').exists():
        for row in read(NATIVE / 'status.json').get('active', []):
            try:
                proc = psutil.Process(row['pid'])
                if not process_alive(proc):
                    continue
                assert proc.create_time() == row['created_epoch']
                assert str(RUNNER) in proc.cmdline() and row['name'] in proc.cmdline()
                active[row['name']] = (proc, None, row['device'], row['created_epoch'])
            except psutil.NoSuchProcess:
                pass
    complete = []
    failed = []
    for name in names:
        result = NATIVE / 'runs' / name / 'result.json'
        if result.exists() and name not in active:
            if read(result)['status'] == 'COMPLETE':
                complete.append(name)
            else:
                failed.append(dict(name=name, result=read(result)['status']))
    pending = [n for n in names if n not in complete and n not in active]
    logs = NATIVE / 'logs'
    logs.mkdir(exist_ok=True)
    while pending or active:
        changed = False
        for name, (proc, handle, device, started) in list(active.items()):
            if process_alive(proc):
                continue
            if handle:
                handle.close()
            del active[name]
            path = NATIVE / 'runs' / name / 'result.json'
            if not path.exists() or read(path)['status'] != 'COMPLETE':
                failed.append(dict(name=name, reason='Worker ended without complete result'))
            else:
                complete.append(name)
                changed = True
        if changed:
            try:
                summarize(names)
            except Exception as exc:
                failed.append(dict(stage='analysis', reason=repr(exc)))
        available = psutil.virtual_memory().available / 2**30
        free = gpu_free()
        # Per-device reservations persist until nvidia-smi lists the worker.
        report = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
        memory = {int(line.split(',')[0]): float(line.split(',')[1])/1024
                  for line in report.stdout.strip().splitlines() if line.strip()}
        for proc, _, device, _ in active.values():
            free[device] -= max(0., 3.0-memory.get(proc.pid, 0.))
        while pending and len(active)<MAX_WORKERS and available>=70 and not failed:
            candidates = [d for d in [0, 1] if free[d]>=3.0 and
                          sum(v[2]==d for v in active.values())<PER_GPU]
            if not candidates:
                break
            device = max(candidates, key=lambda d: free[d])
            name = pending.pop(0)
            assert sha(NATIVE / 'jobs' / f'{name}.json') == queue['job_sha256'][name]
            handle = (logs / f'{name}.log').open('a')
            proc = subprocess.Popen([PYTHON, str(RUNNER), 'worker', '--root', str(NATIVE), '--name', name,
                                     '--device', str(device)], cwd=REPO, stdout=handle,
                                    stderr=subprocess.STDOUT)
            active[name] = proc, handle, device, psutil.Process(proc.pid).create_time()
            free[device] -= 3.0
            available -= 10
        detail = []
        for name, (proc, _, device, started) in active.items():
            path = NATIVE / 'runs' / name / 'progress.json'
            progress = read(path) if path.exists() else {}
            detail.append(dict(name=name, pid=proc.pid, created_epoch=started, device=device,
                               time_s=progress.get('time_s'), state=progress.get('status')))
        stage = ('FAILED' if failed else 'COMPLETE' if not active and not pending else
                 'RUNNING' if active else 'WAITING_RESOURCES')
        write(NATIVE / 'status.json', dict(stage=stage, supervisor_pid=os.getpid(),
              updated_epoch=time.time(), completed=complete, pending=pending, active=detail,
              failed=failed, total=len(names), available_memory_GiB=available,
              GPU_effective_free_GiB=free, max_workers=MAX_WORKERS, max_workers_per_GPU=PER_GPU,
              execution_override="Six120s extensions;3GiB reservation perworker; frozen physics unchanged"))
        if failed and not active:
            break
        if pending or active:
            time.sleep(20)
    if not failed:
        summarize(names)


if __name__ == '__main__':
    main()
