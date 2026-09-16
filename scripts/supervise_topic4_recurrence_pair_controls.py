#!/usr/bin/env python3
"""The same bounded scheduler, dispatched only after saved-return review."""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
import subprocess
import run_topic4_recurrence_pair_controls as physical
import supervise_topic4_adaptation_capacity as controller


if __name__ == '__main__':
    assert (physical.OUT / 'dispatch_authorization.json').exists()
    original_popen = subprocess.Popen
    def popen(cmd, *args, **kwargs):
        old = physical.ROOT / 'scripts/run_topic4_adaptation_capacity.py'
        new = physical.ROOT / 'scripts/run_topic4_recurrence_pair_controls.py'
        cmd = [str(new) if str(arg) == str(old) else arg for arg in cmd]
        return original_popen(cmd, *args, **kwargs)
    controller.run = physical
    controller.OUT, controller.ROOT = physical.OUT, physical.ROOT
    controller.read, controller.write = physical.carrier.base.read, physical.carrier.base.write
    controller.subprocess.Popen = popen
    try:
        controller.main()
    finally:
        controller.subprocess.Popen = original_popen
