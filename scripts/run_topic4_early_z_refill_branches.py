#!/usr/bin/env python3
"""Two bounded early-Z rescue siblings; preserve the original native cohort."""
import argparse
import copy
from datetime import datetime
import hashlib
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
import numpy as np
import run_topic4_m_parameter_modes as core

ROOT = core.ROOT
SOURCE = core.OUT
WINDOW = ROOT / 'results/topic4_sef_hfo/fig5_m_overnight_exploration_20260913'
OUT = WINDOW / 'early_z_refill_branches'
read = core.read
ORIGINAL_TRACKER_STEP = core.tracker_step


def write(destination, value):
    # This supervisor may only write its own branch outputs.
    assert Path(destination).resolve().is_relative_to(OUT.resolve()), destination
    core.write(destination, value)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    plan_path = OUT / 'plan.json'
    if plan_path.exists(): return read(plan_path)
    source_protocol = read(SOURCE / 'protocol.json')
    for source_path, digest in source_protocol['source_hashes'].items():
        assert core.sha(source_path) == digest, source_path
    plan = dict(source=str(SOURCE), source_protocol=source_protocol,
        sources=['e0_t0_s9108401', 'e0_t0_s9108402'], max_branches=2,
        eligibility='Exact saved10s prefix from each specified source, before any high entry or recovery; paired continuations use an event-aligned early rescue rule.',
        intervention='If no established return2s after first high confirmation, begin a1s E-Z refill on the next10ms boundary, then release native Z. Keep M, parameters, fast state and OU/RNG unchanged.',
        native_return_observation_s=2., maximum_trajectory_s=243.5,
        maximum_post_release_s=60, after_second_entry_s=10,
        independent_replicates=False, included_in_original_F=False,
        deadline=read(WINDOW / 'window.json')['deadline'],
        early_refill_is_external=True, native_parent_continues=True)
    write(plan_path, plan)
    p = copy.deepcopy(plan['source_protocol'])
    p.update(total=2, jobs=[], branch_plan=str(plan_path), recurrence_observation_s=60,
             intervention=plan['intervention'], F='Use original M grid; shared-prefix branches never add observations.')
    write(OUT / 'protocol.json', p)
    geometry = OUT / 'geometry.npz'
    if not geometry.exists(): geometry.symlink_to((SOURCE / 'geometry.npz').resolve())
    return plan


def eligibility(saved):
    tr = saved['tracker']; sec = saved['engine']['step'] * .0001
    if tr['recoveries'] or tr['restore_s'] is not None:
        return 'SKIP_ALREADY_RECOVERED_OR_RESTORED'
    if sec==10. and not tr['entries'] and tr['phase']=='PRE_ENTRY': return 'ELIGIBLE'
    return 'WAIT'


def engine_digest(engine):
    return hashlib.sha256(pickle.dumps(engine, protocol=5)).hexdigest()


def create_branch(source_name, saved, parent_bytes):
    """Link closed prefix chunks and copy the exact full engine state once."""
    source_folder = SOURCE / 'runs' / source_name
    sec = saved['engine']['step'] * .0001; end = int(saved['engine']['step'])
    assert eligibility(saved) == 'ELIGIBLE' and end % 5000 == 0
    assert saved['restore_from'] is None
    assert saved['job']['eta_m'] == .005 and saved['job']['tau_M_s'] == 1.
    name = 'early_z_refill_s' + str(saved['job']['seed'])
    folder = OUT / 'runs' / name
    folder.mkdir(parents=True, exist_ok=True)
    assert not (folder / 'checkpoint.pkl').exists()
    chunks = folder / 'chunks'; chunks.mkdir(exist_ok=True)
    expected = 0; prefix = []
    for path in sorted((source_folder / 'chunks').glob('*.npz')):
        if '.tmp.' in path.name: continue
        lo, hi = [int(x) for x in path.stem.split('_')]
        if hi > end: continue
        assert lo == expected and hi > lo
        with np.load(path) as a:
            assert int(a['start_step']) == lo and int(a['end_step']) == hi
            assert len(a['raster']) == hi-lo and len(a['spikes_1ms'])*10 == hi-lo
            assert np.array_equal(a['spikes_1ms'][:, 0], a['field_1ms'].sum(1))
        target = chunks / path.name
        if not target.exists(): target.symlink_to(path.resolve())
        assert target.resolve() == path.resolve()
        prefix.append(dict(path=str(path), sha256=core.sha(path), start=lo, end=hi))
        expected = hi
    assert expected == end
    job = copy.deepcopy(saved['job'])
    job.update(name=name, horizon_s=243.5, early_refill_branch=True, source_run=source_name,
        first_entry_horizon_s=180., native_return_observation_s=2., recurrence_observation_s=60.)
    before = engine_digest(saved['engine'])
    tr = copy.deepcopy(saved['tracker'])
    assert tr['restore_s'] is None and tr['release_s'] is None
    result = dict(job=job, identity=saved['identity'], engine=saved['engine'],
                  tracker=tr, restore_from=None)
    core.save_pickle(folder / 'checkpoint.pkl', result)
    reloaded = core.load_pickle(folder / 'checkpoint.pkl')
    assert engine_digest(reloaded['engine']) == before
    assert reloaded['tracker']['entries'] == saved['tracker']['entries']
    audit = dict(source_run=source_name, fork_s=sec, source_checkpoint_step=end,
        source_checkpoint_sha256=hashlib.sha256(parent_bytes).hexdigest(),
        engine_sha256=before, copied_engine_verified_exact=True,
        parent_tracker=saved['tracker'], branch_tracker=tr, closed_prefix=prefix,
        intervention='Z only; conditional on no established return2s after first high confirmation,1s refill then release',
        M_cleared=False, fast_state_cleared=False, random_state_cleared=False,
        original_native_run_continues=True, independent_replicate=False,
        maximum_post_release_s=60, parent_job=saved['job'], branch_job=job)
    write(folder / 'branch_audit.json', audit)
    write(OUT / 'jobs' / (name+'.json'), job)
    return job


def early_tracker_step(tr,rate,sec,rescue=True):
    """Only the explicitly authorized intervention/observation schedule changes."""
    ORIGINAL_TRACKER_STEP(tr,rate,sec,rescue=False)
    if (rescue and tr['entries'] and not tr['recoveries'] and tr['restore_s'] is None
            and sec>=tr['entries'][0]['confirmation_s']+2):
        tr['restore_s']=round(sec+.01,2);tr['release_s']=round(sec+1.01,2)
    if len(tr['entries'])>=2:
        tr['stop_s']=min(243.5,tr['entries'][1]['confirmation_s']+10)
    elif tr['recoveries']:
        origin=max(tr['recoveries'][0]['confirmation_s'],tr['release_s'] or 0.)
        tr['stop_s']=min(243.5,origin+60)
    elif tr['release_s'] is not None:
        tr['stop_s']=min(243.5,tr['release_s']+60)
    else:tr['stop_s']=min(tr['stop_s'],243.5)


def render_result(name):
    # Import the plotter before assigning core.OUT in any worker process.
    # Its F and geometry must keep pointing at the original 40-job cohort.
    import plot_topic4_m_parameter_modes as f
    assert f.OUT == SOURCE
    folder = OUT / 'runs' / name; r = read(folder / 'result.json')
    a = f.load(folder); metrics = f.analyze(a, r)
    dest = OUT / 'candidates' / name
    f.render(a, r, metrics, dest / 'figures', f.grid_summary())
    write(dest / 'branch_qualification.json', dict(source=str(folder),
        branch_audit=read(folder / 'branch_audit.json'), metrics=f.safe(metrics),
        original_F_extra_samples=0, human_review='PENDING', agent_visual_review='PENDING'))


def supervise():
    import fcntl, psutil
    plan = prepare(); lock = (OUT / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    deadline = datetime.fromisoformat(plan['deadline']).timestamp()
    launched = {}; skipped = {}; rendered = set(); processes = {}; checked = {}
    for source_name in plan['sources']:
        seed = source_name.rsplit('_s', 1)[1]; name = 'early_z_refill_s'+seed
        if (OUT / 'jobs' / (name+'.json')).exists(): launched[source_name] = name
    while True:
        failures = []
        for source_name, name in list(launched.items()):
            folder = OUT / 'runs' / name
            if (folder / 'result.json').exists():
                if name not in rendered: render_result(name); rendered.add(name)
            elif name in processes and processes[name].poll() is not None:
                failures.append(dict(name=name, returncode=processes[name].returncode))
            elif name not in processes:
                # Reattach to a still-live owned worker after supervisor restart.
                pg = read(folder / 'progress.json') if (folder / 'progress.json').exists() else {}
                pid = pg.get('pid')
                if pid and psutil.pid_exists(pid):
                    command = ' '.join(psutil.Process(pid).cmdline())
                    assert str(Path(__file__).resolve()) in command and name in command
                else:
                    log = (folder / 'worker.log').open('a')
                    processes[name] = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                        'worker', '--name', name], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    log.close()
        if failures:
            write(OUT / 'status.json', dict(status='FAILED', failed=failures, pid=os.getpid()))
            raise RuntimeError(failures)
        if time.time() < deadline and psutil.virtual_memory().available / 2**30 >= 68 and psutil.cpu_percent(interval=1) < 85:
            for source_name in plan['sources']:
                if source_name in launched or source_name in skipped: continue
                seed=source_name.rsplit('_s',1)[1]
                cp = OUT / 'parents' / ('s'+seed+'_10s.pkl')
                if not cp.exists(): continue
                stamp = cp.stat().st_mtime_ns
                if checked.get(source_name) == stamp: continue
                blob = cp.read_bytes(); saved = pickle.loads(blob)
                checked[source_name] = stamp
                check = eligibility(saved)
                if check.startswith('SKIP'): skipped[source_name] = check
                elif check == 'ELIGIBLE':
                    job = create_branch(source_name, saved, blob)
                    launched[source_name] = job['name']
                    del saved, blob
                    break  # Launch next loop; do not retain both large parent states.
                del saved, blob
        pending = [s for s in plan['sources'] if s not in launched and s not in skipped]
        all_done = all((OUT / 'runs' / n / 'result.json').exists() for n in launched.values())
        window_ended = time.time() >= deadline
        status = 'COMPLETE_PENDING_REVIEW' if all_done and not pending else 'RUNNING'
        if window_ended and all_done: status = 'WINDOW_ENDED'
        write(OUT / 'status.json', dict(status=status, pid=os.getpid(), launched=launched,
            skipped=skipped, pending=pending, rendered=sorted(rendered),
            no_new_dispatch_after=plan['deadline'], original_cohort_untouched=True))
        if all_done and (not pending or window_ended): break
        time.sleep(45)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'supervise', 'worker', 'render'])
    parser.add_argument('--name'); args = parser.parse_args()
    if args.mode == 'prepare': prepare()
    elif args.mode == 'supervise':
        try: supervise()
        except Exception as exc:
            write(OUT / 'status.json', dict(status='FAILED', error=repr(exc), pid=os.getpid()))
            raise
    elif args.mode == 'render': render_result(args.name)
    else:
        core.OUT = OUT
        core.tracker_step = early_tracker_step
        core.worker(read(OUT / 'jobs' / (args.name+'.json')))
