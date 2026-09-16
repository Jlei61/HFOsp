#!/usr/bin/env python3
"""Render both reviewed full figures after their exact dense replay is verified."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'results/topic4_sef_hfo/autonomous_recovery_exploration_20260914'
NAMES = ['resource_rho0.25_k50_tau10_s9108401', 'resource_rho0.25_k200_tau10_s9108401']
OUT = BASE/'native_field_candidates_recurrence'


def save(rows):
    path=OUT/'full_figure_status.json'
    temp=path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(dict(updated_at=time.time(),pid=os.getpid(),rows=rows),indent=2)+'\n')
    temp.replace(path)


def main():
    rows={name:dict(status='WAITING_VERIFIED_DENSE_REPLAY') for name in NAMES}
    deadline=json.loads((BASE/'protocol.json').read_text())['deadline_epoch']
    while True:
        for name in NAMES:
            if rows[name]['status'] in ['RENDERED_PENDING_VISUAL_REVIEW','FAILED_REVIEW']:
                continue
            status=OUT/name/'recorder_status.json'
            if not status.exists():
                continue
            state=json.loads(status.read_text())
            if state['status']=='FAILED_REVIEW':
                rows[name]=dict(status='FAILED_REVIEW',source=state)
                continue
            if state['status']!='VERIFIED_READY_FOR_FULL_FIGURE':
                continue
            rows[name]=dict(status='RENDERING');save(rows)
            with (OUT/name/'full_figure_render.log').open('ab') as log:
                result=subprocess.run([sys.executable,str(ROOT/'scripts/plot_topic4_autonomous_recurrence_fig5.py'),
                    '--name',name],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            rows[name]=dict(status='RENDERED_PENDING_VISUAL_REVIEW' if result.returncode==0 else 'FAILED_REVIEW',
                            returncode=result.returncode)
        save(rows)
        if all(row['status'] in ['RENDERED_PENDING_VISUAL_REVIEW','FAILED_REVIEW'] for row in rows.values()):
            return
        if time.time()>=deadline:
            return
        time.sleep(20)


if __name__=='__main__':
    main()
