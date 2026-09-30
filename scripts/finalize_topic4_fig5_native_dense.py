#!/usr/bin/env python3
"""Consume this bounded batch and render its figures when the data are complete."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/topic4_sef_hfo/fig5_manual_core_release_v1'
OUT=BASE/'layout_v11'


def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return {}


def main():
    OUT.mkdir(exist_ok=True)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def run(script,*args):
        log=OUT/(Path(script).stem+'.log')
        with log.open('w') as stream:
            subprocess.run([sys.executable,str(ROOT/'scripts'/script),*args],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    native_done=False
    while True:
        native=read(BASE/'native_transition_v2/status.json')
        dense=read(BASE/'latency_dense_v1/status.json')
        if 'FAIL' in native.get('status','') or dense.get('status') in ['FAILED','DRAINING_AFTER_FAILURE']:
            raise RuntimeError(dict(native=native,dense=dense))
        if native.get('status')=='COMPLETE' and not native_done:
            run('analyze_topic4_fig5_native_transition.py')
            run('plot_topic4_fig5_native_dense_v11.py','--native-only')
            native_done=True
            (OUT/'progress.json').write_text(json.dumps(dict(status='NATIVE_FIGURES_READY_WAITING_FOR_GRID'),indent=2)+'\n')
            print('Native figures ready.',flush=True)
        if native_done and dense.get('status')=='SIMULATIONS_COMPLETE':
            run('analyze_topic4_fig5_latency_dense.py')
            run('plot_topic4_fig5_native_dense_v11.py')
            (OUT/'progress.json').write_text(json.dumps(dict(status='ALL_RENDERED_PENDING_AGENT_REVIEW'),indent=2)+'\n')
            print('All analyses and figures rendered; agent visual review pending.',flush=True)
            return
        time.sleep(10)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        OUT.mkdir(exist_ok=True)
        (OUT/'progress.json').write_text(json.dumps(dict(status='FINALIZER_FAILED',error=repr(exc)),indent=2)+'\n')
        raise
