#!/usr/bin/env python3
"""Finite readout collector; never dispatches scientific jobs or promotes plots."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import time
import fcntl
import traceback
import numpy as np
from campaign import ROOT,NATIVE,read,write
from analyze_native import analyze
import analyze_topic4_loop_zk_conditional as original
from plot_native import main as plot


def main():
    lock=(ROOT/'collector.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    names=read(NATIVE/'queue.json')['names'];cache={};input_reference=None
    while True:
        try:
            changed=False
            for name in names:
                if name in cache:continue
                path=NATIVE/'runs'/name/'result.json'
                if not path.exists() or read(path)['status']!='COMPLETE':continue
                row,inputs=analyze(NATIVE,name)
                if input_reference is None:input_reference=inputs
                else:assert np.array_equal(input_reference,inputs),name
                cache[name]=row;changed=True
            if changed:
                write(NATIVE/'extended_analysis_summary.json',dict(
                    status='COMPLETE' if len(cache)==len(names) else 'PARTIAL',
                    completed=len(cache),total=len(names),rows=[cache[n] for n in names if n in cache],
                    common_future_inputs_exact=True,formal_bifurcation='NOT_ESTABLISHED',human_review='PENDING'))
                plot()
            status=read(NATIVE/'status.json')
            write(ROOT/'collector_status.json',dict(status='COMPLETE' if len(cache)==len(names) else 'WAITING_NATIVE',
                 completed=len(cache),total=len(names),pid=os.getpid(),updated_epoch=time.time(),
                 native_status=status['stage'],human_review='PENDING'))
            if len(cache)==len(names) or status['stage']=='FAILED':break
        except Exception:
            write(ROOT/'collector_status.json',dict(status='ANALYSIS_ERROR',pid=os.getpid(),
                 completed=len(cache),updated_epoch=time.time(),error=traceback.format_exc(),
                 native_jobs_unaffected=True))
            raise
        time.sleep(30)


if __name__=='__main__':main()
