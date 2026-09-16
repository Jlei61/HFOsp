#!/usr/bin/env python3
"""Bounded capacity comparison, with no automatic extension after three jobs."""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
import fcntl, subprocess, sys, time
import psutil
import run_topic4_adaptation_capacity as run
from supervise_topic4_reviewed_global_batch import resources

OUT, ROOT = run.OUT, run.ROOT
read, write = run.carrier.base.read, run.carrier.base.write


def analyze(names):
    if not (OUT / 'geometry.npz').exists():
        return
    with (OUT / 'analysis.log').open('a') as log:
        commands = [[sys.executable, str(ROOT / 'scripts' / script), '--root', str(OUT)]
            for script in ['analyze_topic4_autonomous_recovery.py', 'analyze_topic4_autonomous_events.py',
                           'analyze_topic4_recruitment_origin.py', 'analyze_topic4_event_extent.py']]
        commands += [[sys.executable, str(ROOT / 'scripts/analyze_topic4_autonomous_recovery.py'),
                      '--root', str(OUT), '--name', name] for name in names]
        for cmd in commands:
            result = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                write(OUT / 'analysis_failure.json', dict(command=cmd, returncode=result.returncode, time=time.time()))


def main():
    p = run.prepare()
    lock = (OUT / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    auth = read(OUT / 'dispatch_authorization.json')
    jobs = auth['approved_names']
    assert set(jobs) == {job['name'] for job in p['initial_jobs']}
    children, failed, seen, last = {}, [], set(), 0
    while True:
        for name, child in list(children.items()):
            rc = child.poll()
            if rc is not None:
                children.pop(name)
                if rc or not (OUT / 'runs' / name / 'result.json').exists():
                    failed.append(dict(name=name, returncode=rc))
        complete = {n for n in jobs if (OUT / 'runs' / n / 'result.json').exists()}
        pending = [n for n in jobs if n not in complete and n not in children
                   and n not in {v['name'] for v in failed}]
        earlier_pending = any(read(run.PARENT / d / 'status.json').get('pending') for d in
            ['activity_global_pool_round3', 'stronger_M_redistribution_round4', 'continuous_resource_recovery_round5'])
        if pending and not failed and not earlier_pending and time.time() < p['deadline_epoch'] - 5400:
            total, gpu = resources()  # New CPU wrapper is conservatively counted as GPU by this inherited counter.
            if total < 28 and gpu < 24 and psutil.virtual_memory().available / 2**30 >= 120:
                name = pending.pop(0)
                folder = OUT / 'runs' / name
                folder.mkdir(parents=True, exist_ok=True)
                with (folder / 'worker.log').open('a') as log:
                    children[name] = subprocess.Popen([sys.executable, '-u', str(ROOT / 'scripts/run_topic4_adaptation_capacity.py'),
                        'worker', '--name', name, '--producer-script', str(ROOT / 'scripts/run_topic4_autonomous_recovery.py')],
                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write(OUT / 'status.json', dict(status='RUNNING' if not failed else 'DRAINING_FAILURE',
            pid=os.getpid(), updated_at=time.time(), completed=len(complete), total=len(jobs),
            running={n: child.pid for n, child in children.items()}, pending=pending, failed=failed))
        if time.time() - last >= 600 or complete - seen:
            analyze(sorted(complete - seen))
            seen, last = complete.copy(), time.time()
        if not children and (not pending or failed or time.time() >= p['deadline_epoch'] - 5400):
            break
        time.sleep(15)
    analyze(sorted(complete - seen))
    write(OUT / 'status.json', dict(status='REVIEW_MILESTONE' if not failed else 'FAILED_REVIEW',
        pid=os.getpid(), updated_at=time.time(), completed=len(complete), total=len(jobs),
        running={}, pending=pending, failed=failed))


if __name__ == '__main__':
    main()
