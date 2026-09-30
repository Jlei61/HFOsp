#!/usr/bin/env python3
"""Define and dispatch reduced-model batches (version = frozen variant + job table)."""
import argparse
import subprocess
import psutil
import numpy as np
from common import *  # noqa: F401,F403
from approx_system import APPROX, DEFAULT_VARIANT
import approx_run as AR
import native_continue as nc

MAX_APPROX = 6   # CPU-only single-thread reduced-model runs; GPU full-network processes are capped separately at 4
MIN_AVAILABLE_GIB = 64


def define(version, variant_overrides=None, note=''):
    vd = APPROX / version; vd.mkdir(parents=True, exist_ok=True); (vd / 'jobs').mkdir(exist_ok=True)
    if (vd / 'variant.json').exists():
        return read(vd / 'variant.json')
    variant = dict(DEFAULT_VARIANT); variant.update(variant_overrides or {}); variant['version'] = version; variant['note'] = note
    variant['frozen_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    write(vd / 'variant.json', variant)
    for j in nc.main_jobs():
        write(vd / 'jobs' / (j['name'] + '.json'), dict(name=j['name'], mode='continuation', native=j['name'], duration_ms=CONTINUATION_MS,
                                                        future=j['future'], z_source_ms=j['z_source_ms'], history_ms=j['history_ms']))
    for seed in SEEDS:
        zp = APPROX / 'input' / f'z_path_s{seed}.npz'
        write(vd / 'jobs' / f'path_s{seed}.json', dict(name=f'path_s{seed}', mode='path', seed=seed, start_ms=8000, duration_ms=REPLAY_END_MS - 8000,
                                                        input=f'replay_s{seed}', z_path=str(zp)))
    return variant


def add_job(version, job):
    vd = APPROX / version; write(vd / 'jobs' / (job['name'] + '.json'), job)


def consolidate_z_path(seed):
    run = OUT / 'replay' / 'runs' / f'eta0.0005_s{seed}'; steps = []; zs = []
    for path in sorted((run / 'fields').glob('*.npz')):
        with np.load(path) as a:
            sel = a['zm_step'] >= ms_to_step(8000)
            if sel.any():
                steps.append(a['zm_step'][sel]); zs.append(a['z'][sel])
    steps = np.concatenate(steps); z = np.concatenate(zs)
    (APPROX / 'input').mkdir(parents=True, exist_ok=True)
    np.savez(APPROX / 'input' / f'z_path_s{seed}.npz', zm_step=steps, z=z)
    return dict(seed=seed, frames=int(len(steps)), first_ms=float(steps[0] * DT_MS), last_ms=float(steps[-1] * DT_MS))


def add_extension_jobs(version, native_names):
    for name in native_names:
        parent = name[:-4] if name.endswith('_ext') else name
        njob = read(nc.NATIVE / 'jobs' / (parent + '.json'))
        add_job(version, dict(name=parent + '_ext', mode='extension', parent=parent, input=njob['future'], duration_ms=EXTENSION_MS))


def add_fixed_zm_jobs(version, native_names):
    for name in native_names:
        njob = read(nc.NATIVE / 'jobs' / (name + '.json')); assert njob['freeze_m']
        add_job(version, dict(name=name, mode='fixed_zm', native=name, duration_ms=njob['duration_ms'], future=njob['future'],
                              z_source_ms=njob['z_source_ms'], history_ms=njob['history_ms'], m_source_ms=njob['m_source_ms']))


def add_deterministic_jobs(version, z_pair, histories=HISTORY_MS, duration_ms=CONTINUATION_MS):
    names = []
    for z in z_pair:
        for h in histories:
            name = f'det_z{z}_h{h}'; names.append(name)
            add_job(version, dict(name=name, mode='deterministic', seed=MAIN_SEED, start_ms=int(h), z_source_ms=int(z), duration_ms=duration_ms,
                                  input=None, note='OU innovations off: nominal external mean nu_sig, Poisson diffusion variance retained'))
    return names


def my_approx_processes():
    found = {}
    for proc in psutil.process_iter(['pid', 'cmdline']):
        cmd = ' '.join(proc.info['cmdline'] or [])
        if 'topic4_fig5_z_state/approx_run.py worker' in cmd:
            found[proc.info['pid']] = cmd
    return found


def run_batch(version, names, label):
    vd = APPROX / version; pending = [n for n in names if not (vd / 'runs' / n / 'result.json').exists()]
    children = {}; failed = []; started = time.time(); status_path = vd / f'status_{label}.json'
    adopted = {}
    for pid, cmd in my_approx_processes().items():                     # adopt live workers of this version instead of relaunching
        for n in list(pending):
            if f'--version {version} --name {n}' in cmd or cmd.endswith(f'--name {n}'):
                adopted[n] = psutil.Process(pid); pending.remove(n)
    while pending or children or adopted:
        for name, proc in list(adopted.items()):
            if proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE:
                continue
            del adopted[name]
            if not (vd / 'runs' / name / 'result.json').exists():
                failed.append(name)
        for name, child in list(children.items()):
            if child.poll() is None:
                continue
            del children[name]
            if not (vd / 'runs' / name / 'result.json').exists():
                failed.append(name)
        available = psutil.virtual_memory().available / 2**30
        while pending and not failed and len(my_approx_processes()) < MAX_APPROX and available > MIN_AVAILABLE_GIB:
            name = pending.pop(0); children[name] = AR.launch(version, name); available -= 3; time.sleep(2)
        done = [n for n in names if (vd / 'runs' / n / 'result.json').exists()]
        prog = {}
        for name, child in list(children.items()) + list(adopted.items()):
            p = vd / 'runs' / name / 'progress.json'; d = read(p) if p.exists() else {}
            prog[name] = dict(pid=child.pid, step=d.get('step'), steps=d.get('steps'), mean_E_hz=d.get('mean_E_hz'))
        write(status_path, dict(status='DRAINING_AFTER_FAILURE' if failed else 'RUNNING', completed=len(done), total=len(names), running=prog,
                                pending=pending, failed=failed, elapsed_s=time.time() - started, updated_at=time.time()))
        if failed and not children and not adopted:
            raise RuntimeError(failed)
        if pending or children or adopted:
            time.sleep(15)
    write(status_path, dict(status='COMPLETE', completed=len(names), total=len(names), running={}, pending=[], failed=failed, elapsed_s=time.time() - started))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['define', 'zpath', 'run']); ap.add_argument('--version', default='v1')
    ap.add_argument('--futures', nargs='*', default=['W1']); ap.add_argument('--names', nargs='*'); ap.add_argument('--label', default='batch')
    ap.add_argument('--override', default=None, help='JSON dict of variant overrides'); ap.add_argument('--note', default='')
    a = ap.parse_args()
    if a.mode == 'define':
        print(define(a.version, json.loads(a.override) if a.override else None, a.note))
    elif a.mode == 'zpath':
        for seed in SEEDS:
            print(consolidate_z_path(seed))
    else:
        vd = APPROX / a.version
        if a.names:
            names = a.names
        else:
            names = [j['name'] for j in nc.main_jobs() if j['future'] in a.futures]
        run_batch(a.version, names, a.label)
