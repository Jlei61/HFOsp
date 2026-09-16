#!/usr/bin/env python3
"""Run the reviewed three-parameter extension with the same bounded scheduler."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
import subprocess
import run_topic4_preserved_global_gain as physical
import supervise_topic4_adaptation_capacity as controller


if __name__=='__main__':
    # Reuse the scheduler; only its producer argv is remapped. This adds no
    # simulation equation, event detector, parameter, or capacity beyond protocol.
    popen0=subprocess.Popen
    def popen(cmd,*args,**kwargs):
        old=physical.ROOT/'scripts/run_topic4_adaptation_capacity.py'
        new=physical.ROOT/'scripts/run_topic4_preserved_global_gain.py'
        cmd=[str(new) if str(arg)==str(old) else arg for arg in cmd]
        return popen0(cmd,*args,**kwargs)
    controller.run=physical
    controller.OUT,controller.ROOT=physical.OUT,physical.ROOT
    controller.read,controller.write=physical.carrier.base.read,physical.carrier.base.write
    controller.subprocess.Popen=popen
    try:controller.main()
    finally:controller.subprocess.Popen=popen0
