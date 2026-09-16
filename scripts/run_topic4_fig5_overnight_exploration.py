#!/usr/bin/env python3
"""Finite Fig5 exploration window; retain inherited batch and physics identities."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/topic4_sef_hfo/fig5_m_overnight_exploration_20260913'
PILOT = OUT / 'fast_state_pilot'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def snapshot():
    import psutil
    batches = {}
    for name in read(OUT / 'window.json')['inherited_batches']:
        folder = ROOT / 'results/topic4_sef_hfo' / name
        status = read(folder / 'status.json')
        rows = []
        for path in sorted((folder / 'runs').glob('*/progress.json')):
            p = read(path)
            if path.parent.name.startswith('qa_'):
                continue
            rows.append(dict(name=path.parent.name, **{k: p[k] for k in
                ['status', 'time_s', 'end_s', 'phase', 'entries', 'recoveries',
                 'restore_s', 'recurrence_onset_s', 'recovered', 'wall_s'] if k in p}))
        batches[name] = dict(status=status, progress=rows)
    pilots = {}
    for path in sorted((PILOT / 'runs').glob('*/progress.json')):
        p = read(path)
        pilots[path.parent.name] = {k: p[k] for k in ['status', 'pid', 'time_s',
            'end_s', 'recurrence_observed', 'recurrence_onset_s', 'recovered',
            'interpretation', 'E_rate', 'mean_Z', 'adaptation_current'] if k in p}
    now = datetime.now().astimezone()
    return dict(time=now.isoformat(), inherited=batches, fast_state_pilot=pilots,
                memory_available_GiB=psutil.virtual_memory().available / 2**30,
                cpu_busy_percent=psutil.cpu_percent(interval=.2))


def worker(name):
    import run_topic4_reset_state_diagnosis as base
    # Only output location and the explicitly saved intervention job differ.
    # Existing source hashes, parent identity, exact M reset and recorder apply.
    base.OUT = PILOT
    return base.worker(read(PILOT / 'jobs' / f'{name}.json'))


def supervise():
    import psutil
    lock = (OUT / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    deadline = datetime.fromisoformat(read(OUT / 'window.json')['deadline']).timestamp()
    children = {}
    failed = []
    attempted = set()
    last_snapshot = 0
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
               LD_LIBRARY_PATH='/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib')
    while time.time() < deadline:
        for name, (child, stream) in list(children.items()):
            rc = child.poll()
            if rc is None:
                continue
            stream.close()
            del children[name]
            if rc or not (PILOT / 'runs' / name / 'result.json').exists():
                failed.append(dict(name=name, returncode=rc))
        all_result = PILOT / 'runs/all_fast_90s/result.json'
        eligible = ['all_fast_90s']
        if all_result.exists() and read(all_result)['recurrence_observed']:
            eligible += ['voltage_ref_90s', 'synapse_delay_90s']
        for name in eligible:
            if failed or len(children) >= 2:
                break
            folder = PILOT / 'runs' / name
            if name in attempted or (folder / 'result.json').exists():
                continue
            # Respect a surviving worker after this supervisor was restarted.
            progress = folder / 'progress.json'
            if progress.exists():
                pid = read(progress).get('pid')
                if pid and psutil.pid_exists(pid):
                    try:
                        cmd = psutil.Process(pid).cmdline()
                    except psutil.Error:
                        cmd = []
                    if str(Path(__file__).resolve()) in cmd and name in cmd:
                        attempted.add(name)
                        continue
            if psutil.virtual_memory().available / 2**30 < 68:
                break
            if psutil.cpu_percent(interval=1) >= 85:
                break
            log = PILOT / 'logs' / f'{name}.log'
            log.parent.mkdir(exist_ok=True)
            stream = log.open('a')
            child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                'worker', '--name', name], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            children[name] = (child, stream)
            attempted.add(name)
        state = dict(status='ACTIVE' if not failed else 'NEW_PILOT_DISPATCH_STOPPED',
            supervisor_pid=os.getpid(), deadline=read(OUT / 'window.json')['deadline'],
            pilot_running={k: v[0].pid for k, v in children.items()},
            pilot_attempted=sorted(attempted), failed=failed,
            inherited_workers_untouched=True)
        write(OUT / 'status.json', state)
        if time.time() - last_snapshot >= 300:
            snap = snapshot()
            write(OUT / 'latest_snapshot.json', snap)
            with (OUT / 'snapshot_history.jsonl').open('a') as stream:
                stream.write(json.dumps(snap, ensure_ascii=False) + '\n')
            last_snapshot = time.time()
        time.sleep(20)
    snap = snapshot()
    write(OUT / 'deadline_snapshot.json', snap)
    write(OUT / 'status.json', dict(status='WINDOW_ENDED_PENDING_AGENT_REVIEW',
        deadline=read(OUT / 'window.json')['deadline'], failed=failed,
        pilot_running={k: v[0].pid for k, v in children.items() if v[0].poll() is None},
        new_dispatch_stopped=True, inherited_workers_untouched=True,
        scientific_acceptance=False))
    for _, stream in children.values():
        stream.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['supervise', 'worker', 'snapshot'])
    parser.add_argument('--name', choices=['all_fast_90s', 'voltage_ref_90s', 'synapse_delay_90s'])
    args = parser.parse_args()
    if args.mode == 'worker':
        if args.name is None:
            parser.error('--name is required for worker')
        try:
            worker(args.name)
        except Exception as exc:
            write(PILOT / 'runs' / args.name / 'failure.json',
                  dict(type=type(exc).__name__, error=str(exc)))
            raise
    elif args.mode == 'supervise':
        supervise()
    else:
        print(json.dumps(snapshot(), ensure_ascii=False))
