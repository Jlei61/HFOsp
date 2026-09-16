#!/usr/bin/env python3
"""Refresh Fig5 from audited progress every10min and on each new endpoint."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/topic4_sef_hfo/fig5_log_m_entry_extension_20260915'
OUT=ROOT/'results/topic4_sef_hfo/fig5_preentry_event_audit_20260914/clean_panels_v3_broadband_z'


def write(value):
    path=OUT/'extension_refresh_status.json';tmp=path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def signature():
    protocol=json.loads((SOURCE/'protocol.json').read_text())
    chunks=[];completed=0
    for job in protocol['jobs']:
        folder=SOURCE/'runs'/job['name']
        complete=(folder/'result.json').exists();completed+=complete
        paths=sorted(p.name for p in (folder/'chunks').glob('*.npz') if '.tmp.' not in p.name)
        chunks.append((job['name'],paths[-1] if paths else None,complete))
    return chunks,completed


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'extension_refresh.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    current=OUT/'current_grid.json'
    last_signature,last_completed=signature() if current.exists() else (None,-1)
    last_render=current.stat().st_mtime if current.exists() else 0.
    while True:
        status=json.loads((SOURCE/'status.json').read_text())
        grid_path=SOURCE/'measured_grid.json'
        complete=grid_path.exists() and json.loads(grid_path.read_text())['all_complete']
        now_signature,now_completed=signature()
        refresh=(complete or now_completed!=last_completed or
            (time.time()-last_render>=600 and now_signature!=last_signature))
        if refresh:
            env=os.environ.copy();env['MPLBACKEND']='Agg'
            env['LD_LIBRARY_PATH']=str(Path(sys.executable).parent.parent/'lib')+os.pathsep+env.get('LD_LIBRARY_PATH','')
            with (OUT/'extension_refresh.log').open('a') as log:
                result=subprocess.run([sys.executable,str(ROOT/'scripts/plot_topic4_fig5_clean_panels.py'),'--eta','.0005'],
                    cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                write(dict(status='RENDER_FAILED',returncode=result.returncode,updated_at=time.time()))
                return result.returncode
            last_render=time.time();last_signature=now_signature;last_completed=now_completed
            if complete:
                write(dict(status='UPDATED_1000S_PENDING_HUMAN_REVIEW',completed_extensions=22,
                    total_extensions=22,last_render_at=last_render,updated_at=time.time()))
                return 0
        if (SOURCE/'supervisor_failure.json').exists():
            write(dict(status='EXTENSION_NEEDS_ATTENTION',source=str(SOURCE/'supervisor_failure.json'),updated_at=time.time()))
            return 1
        grid=json.loads(current.read_text()) if current.exists() else {}
        write(dict(status='LIVE_PROGRESS_AUTO_REFRESH',pid=os.getpid(),
            completed_extensions=status['completed_extensions'],total_extensions=22,
            target_horizon_s=1000,refresh_interval_s=600,last_render_at=last_render,
            current_figure_summary=grid.get('progress_summary'),
            reason='Use fully saved count chunks for interim lower bounds; refresh every10min on progress or immediately after a completed endpoint. Final common1000s grid replaces interim estimates.',updated_at=time.time()))
        time.sleep(30)


if __name__=='__main__':sys.exit(main())
