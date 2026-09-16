#!/usr/bin/env python3
"""Adopt only the approved R5 jobs; move tested jobs at their next checkpoint."""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
import fcntl, subprocess, sys, time
from pathlib import Path
import numpy as np
import psutil
import run_topic4_continuous_resource_recovery as run
from supervise_topic4_continuous_resource_recovery import analyze
from supervise_topic4_reviewed_global_batch import alive

OUT, ROOT = run.OUT, run.ROOT
read, write = run.carrier.base.read, run.carrier.base.write


def main():
    review = read(OUT / 'backend_migration_authorization.json')
    prior = read(OUT / 'status.json')
    assert not prior['pending'] and not prior['failed']
    pids = prior['running'].copy()
    approved = read(OUT / 'dispatch_authorization.json')['approved_names']
    targets = review['target_minimum_steps'].copy()
    assert set(targets) <= set(pids) <= set(approved)
    for name in targets:
        qa = read(OUT / 'backend_benchmarks' / (name + '_1000ms.json'))
        assert qa['status'] == 'PASS' and qa['entire_checkpoint_recursive_bitwise']
        assert qa['speed_ratio_gpu_over_cpu'] >= 1.3
    old = psutil.Process(prior['pid'])
    assert any(Path(arg).name == 'supervise_topic4_continuous_resource_recovery.py' for arg in old.cmdline())
    # Avoid abandoning an analysis child that would write the same figure files.
    while any(p.pid not in pids.values() and p.status() != psutil.STATUS_ZOMBIE for p in old.children()):
        time.sleep(5)
    for name, pid in pids.items():
        assert name in psutil.Process(pid).cmdline() and 'worker' in psutil.Process(pid).cmdline()
    old.terminate()
    old.wait(timeout=10)
    lock = (OUT / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write(OUT / 'backend_supervisor_adoption.json', dict(previous_pid=prior['pid'],
        current_pid=os.getpid(), adopted=pids.copy(), time=time.time(),
        no_new_scientific_conditions=True, source_physics_unchanged=True))
    children, failed, migrations = {}, [], []
    seen, last = set(), 0
    while pids:
        for name, pid in list(pids.items()):
            folder = OUT / 'runs' / name
            if name in children:
                children[name].poll()
            if not alive(pid):
                pids.pop(name)
                if not (folder / 'result.json').exists():
                    failed.append(dict(name=name, pid=pid))
                continue
            if name not in targets or (folder / 'result.json').exists():
                continue
            cp = folder / 'checkpoint.pkl'
            state = run.carrier.base.load_pickle(cp)
            if int(state['engine']['step']) < targets[name]:
                continue
            proc = psutil.Process(pid)
            assert name in proc.cmdline() and 'worker' in proc.cmdline()
            proc.suspend()
            try:
                state = run.carrier.base.load_pickle(cp)
                step = int(state['engine']['step'])
                job = read(OUT / 'jobs' / (name + '.json'))
                assert state['job'] == job and state['identity'] == read(OUT / 'protocol.json')['identity']
                ends = []
                for path in sorted((folder / 'chunks').glob('*.npz')):
                    if '.tmp.' not in path.name:
                        with np.load(path) as q:
                            ends.append(int(q['end_step']))
                extra_ends = [max(int(p.stem.split('_')[-1]) for p in (folder / kind).glob('*.npz')
                                  if '.tmp.' not in p.name) for kind in ['pool_chunks', 'resource_chunks']]
                if max(ends) != step or any(end != step for end in extra_ends):
                    continue
                sha = run.carrier.base.sha(cp)
                progress = read(folder / 'progress.json')
                proc.kill()
                proc.wait(timeout=10)
                with (folder / 'worker.log').open('a') as log:
                    log.write(f'\nORDERED_CPU_RESUME from full checkpoint step {step}; previous pid {pid}\n')
                    log.flush()
                    child = subprocess.Popen([sys.executable, '-u', str(ROOT / 'scripts/run_topic4_resource_cpu_resume.py'),
                        'worker', '--name', name, '--producer-script', str(ROOT / 'scripts/run_topic4_autonomous_recovery.py')],
                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                children[name], pids[name] = child, child.pid
                targets.pop(name)
                item = dict(name=name, old_pid=pid, new_pid=child.pid, checkpoint_step=step,
                    checkpoint_sha256=sha, previously_reported_s=progress.get('time_s'), time=time.time(),
                    no_reset=True, restored='All fast states, Z/M, global-pool state and all OU/Poisson/delay history.')
                migrations.append(item)
                write(folder / 'backend_resume.json', item)
                write(OUT / 'backend_migrations.json', dict(migrations=migrations, pending=targets))
            finally:
                if alive(pid) and psutil.Process(pid).status() == psutil.STATUS_STOPPED:
                    psutil.Process(pid).resume()
        complete = {n for n in approved if (OUT / 'runs' / n / 'result.json').exists()}
        write(OUT / 'status.json', dict(status='RUNNING' if not failed else 'DRAINING_FAILURE',
            pid=os.getpid(), updated_at=time.time(), completed=len(complete), total=len(approved),
            running=pids.copy(), pending=[], failed=failed, backend='mixed_GPU_and_verified_ordered_CPU',
            migration_pending=targets))
        if time.time() - last >= 600 or complete - seen:
            analyze(sorted(complete - seen))
            seen, last = complete.copy(), time.time()
        if pids:
            time.sleep(5)
    analyze(sorted(complete - seen))
    write(OUT / 'status.json', dict(status='REVIEW_MILESTONE' if not failed else 'FAILED_REVIEW',
        pid=os.getpid(), updated_at=time.time(), completed=len(complete), total=len(approved),
        running={}, pending=[], failed=failed, backend='mixed_GPU_and_verified_ordered_CPU'))


if __name__ == '__main__':
    main()
