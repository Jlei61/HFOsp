#!/usr/bin/env python3
"""Finish the current diagnostic report after its existing batch exits; dispatch no experiments."""
from pathlib import Path
import json
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/topic4_sef_hfo/rate_model_dynamics_validation_v1'
PYTHON='/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python'


def main():
    deadline=time.monotonic()+3600
    while time.monotonic()<deadline:
        status=json.loads((OUT/'v1_batch_status.json').read_text())
        if status['status']!='RUNNING_V1_DIAGNOSTICS':break
        active=subprocess.run(['systemctl','--user','is-active','--quiet','hfosp-rate-validation-v1-20260908.service']).returncode==0
        if not active:
            status['status']='BATCH_INTERRUPTED_BEFORE_COMPLETE';status['error']='Service exited without final batch status'
            (OUT/'v1_batch_status.json').write_text(json.dumps(status,indent=2)+'\n');break
        time.sleep(5)
    else:raise TimeoutError('Current V1 batch did not finish within one hour; no extra dispatch')
    subprocess.run([PYTHON,str(ROOT/'scripts/analyze_topic4_rate_validation.py')],cwd=ROOT,check=True)
    print('Final diagnostic report refreshed; no V2/V3/V4 or Hopf dispatch.',flush=True)


if __name__=='__main__':main()
