#!/usr/bin/env python3
"""Analyze completed native M/Z runs as they arrive, without starting new runs."""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/topic4_sef_hfo/fig5_manual_core_release_v1/m_runaway_return_v1'
NAMES={'m0_native','weak_fast','weak_20s','matched_gain_80s','slow_80s','slow_200s'}

def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return {}

def write(data):
    (OUT/'incremental_analysis_status.json').write_text(json.dumps(data,indent=2)+'\n')

seen=set()
while True:
    done={n for n in NAMES if read(OUT/'runs'/(n+'.json')).get('status')=='COMPLETE'}
    failed={n:read(OUT/'progress'/(n+'.json')) for n in NAMES if read(OUT/'progress'/(n+'.json')).get('status')=='FAILED'}
    if done-seen:
        log=OUT/'incremental_analysis.log'
        with log.open('a') as stream:
            result=subprocess.run([sys.executable,str(ROOT/'scripts/plot_topic4_fig5_m_return.py'),'--available'],stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            write(dict(status='ANALYSIS_FAILED',completed=sorted(done),log=str(log)));raise SystemExit(result.returncode)
        seen=done
    write(dict(status='ALL_RESULTS_ANALYZED_PENDING_VISUAL_REVIEW' if done==NAMES else 'WAITING_FOR_REMAINING_RUNS',
               completed=sorted(done),analyzed=sorted(seen),failed=list(failed),total=6))
    if done==NAMES:break
    if done|set(failed)==NAMES:
        write(dict(status='RUN_FAILURE_REVIEW_REQUIRED',completed=sorted(done),failed=failed));break
    time.sleep(30)
