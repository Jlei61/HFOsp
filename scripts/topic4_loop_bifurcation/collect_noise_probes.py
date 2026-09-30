#!/usr/bin/env python3
"""Finish the independent-input readout after the bounded queue completes."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import time
import traceback
import fcntl
from campaign import ROOT,read,write
from analyze_native import main as analyze


def main():
    root=ROOT/'independent_noise_probes';lock=(root/'collector.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        status=read(root/'status.json') if (root/'status.json').exists() else {}
        if status.get('stage') in ['COMPLETE','FAILED']:break
        write(root/'collector_status.json',dict(status='WAITING_NATIVE',pid=os.getpid(),updated_epoch=time.time(),completed=len(status.get('completed',[])),total=10))
        time.sleep(30)
    try:
        analyze(root)
        write(root/'collector_status.json',dict(status='ANALYSIS_COMPLETE' if status['stage']=='COMPLETE' else 'PARTIAL_NATIVE_FAILED',updated_epoch=time.time()))
    except Exception:
        write(root/'collector_status.json',dict(status='ANALYSIS_ERROR',error=traceback.format_exc(),updated_epoch=time.time()));raise


if __name__=='__main__':main()
