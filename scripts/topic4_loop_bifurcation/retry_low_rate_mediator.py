#!/usr/bin/env python3
"""Repair only the declared diagnostic subclass name after a pre-step failure."""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
import copy
import shutil
import subprocess
import time
from campaign import read, write, sha, PYTHON
from probe_natural_exit_mediators import OUT, RECON, CONTROLS
import run_topic4_loop_zk_conditional as native
from run_topic4_recovery_window import assert_same_state


def main():
    name = CONTROLS[1];folder = OUT/'runs'/name
    dest = OUT/'failed_low_rate_before_first_step';dest.mkdir(exist_ok=False)
    def status(value, **kw):
        write(OUT/'repair_supervisor.json', dict(status=value, pid=os.getpid(), updated_epoch=time.time(), **kw))
    try:
        assert not (folder/'result.json').exists()
        assert not list((folder/'chunks').glob('*.npz'))
        error = (OUT/f'{name}.log').read_text()
        assert 'checkpoint slow protocol differs from the live object' in error
        state = native.read_pickle(folder/'checkpoint.pkl')
        reference = native.read_pickle(OUT/'runs'/RECON/'checkpoint.pkl')
        assert state['engine']['step'] == 168000
        assert_same_state(state['engine'], reference['engine'])
        for path in [folder/'checkpoint.pkl', folder/'progress.json', folder/'runtime_backend.json', OUT/f'{name}.log']:
            shutil.copy2(path, dest/path.name)
        expected = copy.deepcopy(state['engine'])
        expected['slow']['kind'] = 'NoRetention'
        state['engine']['slow']['kind'] = 'NoRetention'
        assert_same_state(state['engine'], expected)
        native.base.save_pickle(folder/'checkpoint.pkl', state)
        write(dest/'repair.json', dict(status='METADATA_ONLY_REPAIR', no_simulation_steps_executed=True,
            original_protocol_kind='ConditionalSlow', declared_subclass_kind='NoRetention',
            physical_state_bitwise_unchanged=True, expected_low_rate_tau_s=.5,
            original_worker_unchanged=True, repair_producer_sha256=sha(__file__)))
        with (OUT/'remove_low_rate_K_retention_retry.log').open('w') as log:
            process = subprocess.Popen([PYTHON, str(__import__('probe_natural_exit_mediators').__file__),
                'worker', '--name', name, '--device', '1'], stdout=log, stderr=subprocess.STDOUT)
            status('RUNNING_METADATA_REPAIRED_LOW_RATE_CONTROL', worker_pid=process.pid)
            code = process.wait()
        assert code == 0, code
        while True:
            done = []
            for n in CONTROLS:
                p = OUT/'runs'/n/'result.json'
                d = read(p) if p.exists() else {}
                done.append(d.get('status') == 'COMPLETE' and d.get('diagnostic_intervention') == n)
            if all(done):break
            old = read(OUT/'supervisor.json')
            if old['status'] == 'FAILED':
                assert done[0], 'Independent G control also failed'
            status('WAITING_OTHER_ORIGINAL_CONTROL', completed=done)
            time.sleep(20)
        status('COMPLETE_REPAIRED_TWO_NATIVE_MEDIATOR_CONTROLS_ANALYSIS_PENDING',
            controls=CONTROLS, metadata_only_repair=True,
            original_supervisor_expected_failed_old_child='The first low-rate process exited before simulation; original supervisor retains that exit code.')
    except Exception as exc:
        status('FAILED', error=repr(exc));raise


if __name__ == '__main__':main()
