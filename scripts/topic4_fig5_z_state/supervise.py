#!/usr/bin/env python3
"""Resource-aware dispatcher for the native continuation batches (M1 main, extension, fixed-M)."""
import argparse
import subprocess
import psutil
import shutil
import numpy as np
from common import *  # noqa: F401,F403
import native_continue as nc

MAX_HEAVY = 4
MIN_AVAILABLE_GIB = 64
MIN_DISK_GIB = 40


def my_heavy_processes():
    tags = ('topic4_fig5_z_state/native_continue.py', 'topic4_fig5_z_state/replay.py')   # full-network GPU processes only; reduced-model CPU runs are budgeted separately (<=4)
    found = {}
    for proc in psutil.process_iter(['pid', 'cmdline']):
        cmd = ' '.join(proc.info['cmdline'] or [])
        if 'worker' in cmd and any(t in cmd for t in tags):
            found[proc.info['pid']] = cmd
    return found


def gpu_choice(running):
    counts = {0: 0, 1: 0}
    for dev in running.values():
        counts[dev] += 1
    return min(counts, key=lambda d: (counts[d], d))


def run_batch(names, label):
    nc.prepare()
    pending = [n for n in names if not (nc.NATIVE / 'runs' / n / 'result.json').exists()]
    children = {}; devices = {}; failed = []; started = time.time()
    status_path = OUT / 'status.json'
    while pending or children:
        for name, child in list(children.items()):
            if child.poll() is None:
                continue
            del children[name]; devices.pop(name, None)
            if not (nc.NATIVE / 'runs' / name / 'result.json').exists():
                failed.append(name)
        heavy = my_heavy_processes()
        available = psutil.virtual_memory().available / 2**30
        disk = shutil.disk_usage('/data/hfosp').free / 2**30
        while pending and not failed and len(heavy) + 0 < MAX_HEAVY and len(children) < MAX_HEAVY and available > MIN_AVAILABLE_GIB and disk > MIN_DISK_GIB:
            name = pending.pop(0); dev = gpu_choice(devices)
            children[name] = nc.launch(name, dev); devices[name] = dev; available -= 4
            heavy[children[name].pid] = name
            time.sleep(3)
        done = [n for n in names if (nc.NATIVE / 'runs' / n / 'result.json').exists()]
        progress = {}
        for name, child in children.items():
            pth = nc.NATIVE / 'runs' / name / 'progress.json'
            d = read(pth) if pth.exists() else {}
            progress[name] = dict(pid=child.pid, device=devices[name], time_s=d.get('time_s'), status=d.get('status', 'STARTING'))
        prev = read(status_path) if status_path.exists() else {}
        prev.update(status='DRAINING_AFTER_FAILURE' if failed else 'RUNNING', batch=label, updated_at=time.time(),
                    batch_completed=len(done), batch_total=len(names), running=progress, pending=pending, failed=failed,
                    available_memory_GiB=psutil.virtual_memory().available / 2**30, elapsed_s=time.time() - started)
        write(status_path, prev)
        if failed and not children:
            raise RuntimeError(failed)
        if pending or children:
            time.sleep(15)
    prev = read(status_path); prev.update(status=f'{label}_COMPLETE', running={}, pending=[], failed=failed, updated_at=time.time())
    write(status_path, prev)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('batch', choices=['main', 'extension', 'fixed_m'])
    ap.add_argument('--names', nargs='*'); a = ap.parse_args()
    if a.batch == 'main':
        names = [j['name'] for j in nc.main_jobs()]
        # Order: earliest Z pairs first for the low history so section 5.3 selection can start early.
        names = sorted(names, key=lambda n: (nc.read(nc.NATIVE / 'jobs' / (n + '.json'))['history_ms'],
                                             nc.read(nc.NATIVE / 'jobs' / (n + '.json'))['z_source_ms'], n))
        run_batch(names, 'M1_MAIN')
    else:
        assert a.names, 'pass --names for extension/fixed_m batches'
        run_batch(a.names, 'M1_' + a.batch.upper())
