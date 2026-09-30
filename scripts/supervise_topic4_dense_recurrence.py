#!/usr/bin/env python3
"""One reviewed candidate's dense replay, source equality and native energy."""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
import fcntl
import subprocess
import sys
import time
import psutil
import run_topic4_recurrence_pair_controls as control
from supervise_topic4_reviewed_global_batch import resources

ROOT, BASE = control.ROOT, control.PARENT
NAME = 'resource_rho0.25_k50_tau10_s9108401'
SOURCE = BASE / 'preserved_global_gain_round7'
OUT = BASE / 'native_field_candidates_recurrence' / NAME
read, write = control.carrier.base.read, control.carrier.base.write


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    deadline = read(SOURCE / 'protocol.json')['deadline_epoch']
    source_result = SOURCE / 'runs' / NAME / 'result.json'
    script = ROOT / 'scripts/record_topic4_recurrence_candidate_native_fields.py'
    common = ['--name', NAME, '--source-root', str(SOURCE)]
    while not source_result.exists():
        write(OUT / 'recorder_status.json', dict(status='WAITING_SOURCE_COMPLETION', pid=os.getpid(),
            source=str(source_result), updated_at=time.time()))
        if time.time() >= deadline - 5400:
            write(OUT / 'recorder_status.json', dict(status='NOT_STARTED_TIME_BUDGET', updated_at=time.time()))
            return
        time.sleep(15)
    result = read(source_result)
    assert len(result['tracker']['entries']) >= 2 and result['tracker']['recoveries']
    while True:
        total, gpu = resources()
        pending = read(BASE / 'paired_recurrence_confirmation_round8/status.json').get('pending', [])
        if not pending and total < 28 and gpu < 24 and psutil.virtual_memory().available / 2**30 >= 120:
            break
        if time.time() >= deadline - 5400:
            write(OUT / 'recorder_status.json', dict(status='NOT_STARTED_RESOURCE_TIME_BUDGET', updated_at=time.time()))
            return
        time.sleep(15)
    with (OUT / 'supervisor.log').open('ab') as log:
        prepare = subprocess.run([sys.executable, str(script), 'prepare'] + common,
                                 cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if prepare.returncode:
            raise RuntimeError('Dense source preparation failed')
        worker = subprocess.Popen([sys.executable, '-u', str(script), 'worker'] + common +
            ['--producer-script', str(ROOT / 'scripts/run_topic4_autonomous_recovery.py')],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while worker.poll() is None:
            write(OUT / 'recorder_status.json', dict(status='REPLAY_RUNNING', pid=os.getpid(),
                worker_pid=worker.pid, source_end_s=result['end_s'], updated_at=time.time()))
            time.sleep(15)
        if worker.returncode:
            raise RuntimeError(f'Dense replay failed: {worker.returncode}')
        check = subprocess.run([sys.executable, str(script), 'verify'] + common,
                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if check.returncode:
            raise RuntimeError('Dense replay did not match its actual source')
        onset = result['tracker']['entries'][0]['onset_s']
        energy = subprocess.run([sys.executable, str(ROOT / 'scripts/analyze_topic4_native_band_energy.py'),
            '--root', str(OUT), '--name', NAME, '--baseline', '.5',
            '--early', str(round(onset - .5, 2)), '--duration', '1'],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if energy.returncode:
            raise RuntimeError('Native band-energy analysis failed after replay equality passed')
    write(OUT / 'recorder_status.json', dict(status='VERIFIED_READY_FOR_FULL_FIGURE',
        source_end_s=result['end_s'], updated_at=time.time(),
        next='Render full measured Fig5 from1ms state and actual source spikes; inspect native-band energy and optional native movies. No automatic model freeze.'))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        write(OUT / 'recorder_status.json', dict(status='FAILED_REVIEW', error=repr(exc), updated_at=time.time()))
        raise
