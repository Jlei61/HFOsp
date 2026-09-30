#!/usr/bin/env python3
"""Adopt the existing fixed 24 jobs and resume failed jobs from saved states."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import fcntl
import shutil
import signal
import time
import psutil
import run_topic4_fig5_boundary_refinement as run


def main():
    protocol = run.prepare()
    assert run.base.sha(run.__file__) == protocol['refinement_producer_sha256']
    old = run.base.read(run.OUT / 'status.json')
    supervisor = psutil.Process(old['pid'])
    assert str(run.Path(run.__file__).resolve()) in supervisor.cmdline()
    assert 'supervise' in supervisor.cmdline()
    adopted = {}
    for name, record in old['running'].items():
        process = psutil.Process(record['pid'])
        assert name in process.cmdline() and 'worker' in process.cmdline()
        adopted[name] = process
    history = run.OUT / 'recovery_20260918'
    history.mkdir(exist_ok=False)
    run.base.write(history / 'previous_status.json', old)
    resumes = {}
    from analyze_topic4_fig5_entry_progress import audit_counts
    for name in old['failed']:
        folder = run.OUT / 'runs' / name
        saved = run.base.load_pickle(folder / 'checkpoint.pkl')
        counts = audit_counts(folder)
        assert saved['job'] == run.base.read(run.OUT / 'jobs' / (name + '.json'))
        assert saved['identity'] == protocol['identity']
        assert abs(saved['engine']['step'] * .0001 - counts['followup_s']) < 1e-9
        assert saved['tracker']['first_entry'] is None and counts['first_entry'] is None
        shutil.copy2(folder / 'failure.json', history / (name + '-failure.json'))
        resumes[name] = dict(resume_s=counts['followup_s'], checkpoint_sha256=run.base.sha(folder / 'checkpoint.pkl'))
    run.base.write(history / 'recovery_manifest.json', dict(
        seed=run.SEED, total_new=24, adopted={n: dict(pid=p.pid, create_time=p.create_time()) for n, p in adopted.items()},
        resume=resumes, rule='One retry of each failed job; exact saved engine and RNG state. No new parameter points or seeds.',
        recovery_producer=__file__, producer_sha256=run.base.sha(__file__)))
    supervisor.send_signal(signal.SIGTERM)
    supervisor.wait(timeout=30)
    lock = (run.OUT / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pending = [j['name'] for j in protocol['jobs'] if j['name'] not in adopted
               and not (run.OUT / 'runs' / j['name'] / 'result.json').exists()]
    children = {}
    failed = []
    last_complete = -1
    last_render = 0
    while True:
        for name, process in list(adopted.items()):
            if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                continue
            if not (run.OUT / 'runs' / name / 'result.json').exists():
                failed.append(name)
            del adopted[name]
        for name, child in list(children.items()):
            code = child.poll()
            if code is None:
                continue
            if code or not (run.OUT / 'runs' / name / 'result.json').exists():
                failed.append(name)
            del children[name]
        available = psutil.virtual_memory().available / 2**30
        while pending and not failed and len(adopted) + len(children) < 12 and available > 84 and shutil.disk_usage(run.OUT).free / 2**30 > 40:
            name = pending.pop(0)
            marker = run.OUT / 'runs' / name / 'failure.json'
            if marker.exists():
                marker.rename(history / (name + '-resolved-failure.json'))
            children[name] = run.launch(name)
            available -= 4
        completed = sum((run.OUT / 'runs' / j['name'] / 'result.json').exists() for j in protocol['jobs'])
        running = {}
        for name, process in {**adopted, **children}.items():
            path = run.OUT / 'runs' / name / 'progress.json'
            progress = run.base.read(path) if path.exists() else {}
            running[name] = dict(pid=process.pid, seed=run.SEED, time_s=progress.get('time_s'), status=progress.get('status', 'STARTING'))
        finished = not pending and not running
        run.base.write(run.OUT / 'status.json', dict(
            status='DRAINING_FAILURE' if failed else 'COMPLETE_PENDING_HUMAN_REVIEW' if finished else 'RUNNING_BOUNDARY_REFINEMENT',
            pid=os.getpid(), seed=run.SEED, completed_new=completed, total_new=24, reused=35,
            running=running, pending=len(pending), failed=failed, recovered_failures=list(resumes),
            recovery_manifest=str(history / 'recovery_manifest.json'), updated_at=time.time()))
        if completed != last_complete or time.time() - last_render >= 600 or finished:
            try:
                run.report()
            except Exception as exc:
                run.base.write(run.OUT / 'figure_refresh_status.json', dict(status='REFRESH_FAILED', error=repr(exc), updated_at=time.time()))
            last_complete, last_render = completed, time.time()
        if failed and not running:
            raise RuntimeError(failed)
        if finished:
            return
        time.sleep(10)


if __name__ == '__main__':
    main()
