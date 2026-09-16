#!/usr/bin/env python3
"""Adopt live M runs without restarting state; prioritize weak-M recurrence tests.

Only dispatch policy changes. The original frozen simulation worker, forty jobs,
physics hashes, endpoints, seeds, and automatic figures remain unchanged.
"""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import psutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/topic4_sef_hfo/m_parameter_modes_fig5_20260913'
RUNNER = ROOT / 'scripts/run_topic4_m_parameter_modes.py'


def read(p): return json.loads(p.read_text())


def write(p, value):
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(p)


def alive(record):
    try:
        p = psutil.Process(record['pid'])
        return p.create_time() == record['create_time'] and p.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def discover():
    result = {}
    for p in psutil.process_iter(['pid', 'cmdline', 'create_time']):
        cmd = p.info['cmdline'] or []
        if str(RUNNER) not in cmd or 'worker' not in cmd or '--job' not in cmd:
            continue
        path = Path(cmd[cmd.index('--job') + 1])
        if path.parent.resolve() != (OUT / 'jobs').resolve():
            continue
        j = read(path)
        result[j['name']] = dict(pid=p.pid, create_time=p.info['create_time'])
    return result


def main():
    import hashlib
    lock = (OUT / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = read(OUT / 'protocol.json')
    amendment = read(OUT / 'scheduling_amendment_20260913.json')
    rank = amendment['priority_eta_tau_indices']
    jobs = sorted(protocol['jobs'], key=lambda j:
        (rank.index([j['eta_index'], j['tau_index']]), j['seed']))
    active = discover(); owned = {}; streams = {}; failed = []
    completed = {j['name'] for j in jobs if (OUT / 'runs' / j['name'] / 'result.json').exists()}
    pending = [j for j in jobs if j['name'] not in active and j['name'] not in completed]
    write(OUT / 'adopted_workers_20260913.json', dict(time=time.time(), active=active,
        no_worker_restart=True, unchanged_jobs=40, new_controller_pid=os.getpid()))
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
        LD_LIBRARY_PATH='/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib')
    analyzed = set(completed); last_analysis = 0
    while pending or active:
        for name, rec in list(active.items()):
            if alive(rec): continue
            if name in owned: owned.pop(name).poll()
            if name in streams: streams.pop(name).close()
            del active[name]
            if not (OUT / 'runs' / name / 'result.json').exists():
                failed.append(dict(name=name, error='Worker exited without complete result'))
        old_active = 0
        for p in psutil.process_iter(['cmdline']):
            cmd = p.info['cmdline'] or []
            if 'worker' in cmd and any(Path(a).name == 'run_topic4_m_on_z_kinetics.py' for a in cmd):
                old_active += 1
        cap = max(0, min(32, amendment['combined_scan_dispatch_target'] - old_active))
        mem = psutil.virtual_memory().available / 2**30
        cpu = psutil.cpu_percent(interval=1)
        reserve = 0
        for rec in active.values():
            try: reserve += max(0, 4 - psutil.Process(rec['pid']).memory_info().rss / 2**30)
            except psutil.Error: pass
        while pending and not failed and len(active) < cap and mem-reserve > 64 and cpu < 85:
            if shutil.disk_usage(OUT).free / 2**30 <= 60: break
            changed = [p for p, sha in protocol['source_hashes'].items()
                       if hashlib.sha256(Path(p).read_bytes()).hexdigest() != sha]
            if changed:
                failed.append(dict(error='Frozen source changed', paths=changed)); break
            j = pending.pop(0)
            stream = (OUT / 'logs' / (j['name'] + '.log')).open('a')
            child = subprocess.Popen([sys.executable, '-u', str(RUNNER), 'worker', '--job',
                str(OUT / 'jobs' / (j['name'] + '.json'))], cwd=ROOT, env=env,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True)
            owned[j['name']] = child; streams[j['name']] = stream
            active[j['name']] = dict(pid=child.pid, create_time=psutil.Process(child.pid).create_time())
            reserve += 4
        completed = {j['name'] for j in jobs if (OUT / 'runs' / j['name'] / 'result.json').exists()}
        write(OUT / 'status.json', dict(status='DRAINING_AFTER_FAILURE' if failed else 'RUNNING',
            pid=os.getpid(), completed=len(completed), total=40,
            running={n:r['pid'] for n,r in active.items()}, pending=len(pending), failed=failed,
            current_worker_cap=cap, available_memory_GiB=mem, cpu_busy_percent=cpu,
            source_worker_unchanged=True, adopted=True,
            scheduling_amendment=str(OUT / 'scheduling_amendment_20260913.json')))
        if completed != analyzed and time.time()-last_analysis >= 60:
            with (OUT / 'analysis.log').open('a') as f:
                rc = subprocess.call([sys.executable, str(ROOT / 'scripts/plot_topic4_m_parameter_modes.py'),
                    '--update'], cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
            if rc: failed.append(dict(error='Figure/analysis failed', returncode=rc))
            analyzed=set(completed);last_analysis=time.time()
        if failed and not active: break
        time.sleep(20)
    with (OUT / 'analysis.log').open('a') as f:
        rc = subprocess.call([sys.executable, str(ROOT / 'scripts/plot_topic4_m_parameter_modes.py'),
            '--update', '--final'], cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    write(OUT / 'status.json', dict(status='FAILED' if failed or rc else 'COMPLETE_PENDING_FIGURE_REVIEW',
        completed=len(completed), total=40, running={}, pending=len(pending), failed=failed,
        analysis_returncode=rc, scheduling_amendment=str(OUT / 'scheduling_amendment_20260913.json')))


if __name__ == '__main__': main()
