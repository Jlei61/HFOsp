#!/usr/bin/env python3
"""Exact native-Z source replay followed by its own full Fig5."""
import subprocess
import sys
import time
import supervise_topic4_dense_recurrence as pipeline

if __name__=='__main__':
    pipeline.NAME='resource_rho0_k200_tau10_s9108401'
    pipeline.SOURCE=pipeline.BASE/'paired_recurrence_confirmation_round8'
    pipeline.OUT=pipeline.BASE/'native_field_candidates_recurrence'/pipeline.NAME
    try:
        pipeline.main()
        state=pipeline.read(pipeline.OUT/'recorder_status.json')
        if state['status']=='VERIFIED_READY_FOR_FULL_FIGURE':
            with (pipeline.OUT/'full_figure_render.log').open('ab') as log:
                result=subprocess.run([sys.executable,str(pipeline.ROOT/'scripts/plot_topic4_autonomous_recurrence_fig5.py'),
                    '--name',pipeline.NAME,'--source-root',str(pipeline.SOURCE),'--grid','resource'],
                    cwd=pipeline.ROOT,stdout=log,stderr=subprocess.STDOUT)
            pipeline.write(pipeline.OUT/'full_figure_status.json',dict(
                status='RENDERED_PENDING_VISUAL_REVIEW' if result.returncode==0 else 'FAILED_REVIEW',
                returncode=result.returncode,updated_at=time.time()))
    except Exception as exc:
        pipeline.write(pipeline.OUT/'recorder_status.json',dict(status='FAILED_REVIEW',error=repr(exc),updated_at=time.time()))
        raise
