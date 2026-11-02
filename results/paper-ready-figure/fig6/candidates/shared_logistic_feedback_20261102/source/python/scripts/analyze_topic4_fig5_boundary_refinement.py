#!/usr/bin/env python3
"""Sparse measured points inside the completed Fig5 entry/non-entry brackets."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import analyze_topic4_fig5_single_seed_scan as coarse
from analyze_topic4_fig5_entry_progress import audit_counts

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/topic4_sef_hfo/fig5_boundary_refinement_20260917'


def read(path):
    return json.loads(path.read_text())


def collect():
    p=read(OUT/'protocol.json')
    base_path=OUT/'coarse_grid.json'
    assert hashlib.sha256(base_path.read_bytes()).hexdigest()==p['coarse_grid_sha256']
    background=read(base_path)
    for key in ('value','entered','followup','stages'):background[key]=np.asarray(background[key])
    for r in p['references']:
        assert hashlib.sha256((Path(r['source'])/'result.json').read_bytes()).hexdigest()==r['result_sha256']
    records=[]
    for job in p['jobs']:
        folder=OUT/'runs'/job['name']
        chunks=[x for x in (folder/'chunks').glob('*.npz') if '.tmp.' not in x.name]
        a=audit_counts(folder) if chunks else dict(first_entry=None,followup_s=0.)
        done=(folder/'result.json').exists()
        if done:
            result=read(folder/'result.json')
            assert result['job']==job and result['identity']==p['identity']
            assert np.isclose(a['followup_s'],result['elapsed_s'])
            assert (a['first_entry'] is not None)==result['event_observed']
            if result['event_observed']:
                for k in ('onset_s','confirmation_s'):assert np.isclose(a['first_entry'][k],result['first_entry'][k])
            else:assert np.isclose(a['followup_s'],1000.)
        observed=a['first_entry'] is not None
        stage='OBSERVED' if observed else 'CENSORED' if done else 'RUNNING' if (folder/'progress.json').exists() else 'QUEUED'
        records.append(dict(job=job,source=str(folder),complete=done,event_observed=observed,stage=stage,**a))
    assert len(records)==24 and {r['job']['seed'] for r in records}=={9108401}
    summary=dict(new_total=24,new_complete=sum(r['complete'] for r in records),
        new_observed=sum(r['event_observed'] for r in records),new_censored=sum(r['stage']=='CENSORED' for r in records),
        running=sum(r['stage']=='RUNNING' for r in records),queued=sum(r['stage']=='QUEUED' for r in records),reused=35,total=59)
    return dict(quantity_kind='single_seed_boundary_refinement',seed=9108401,horizon_s=1000.,
        all_complete=all(r['complete'] for r in records),base_grid=background,refinement_records=records,
        colorbar=p['colorbar'],progress_summary=summary,snapshot_unix_s=time.time(),
        interpretation='The background is the measured35-point coarse map. Small circles are24 additional parameter samples; triangles indicate censored lower bounds. No interpolation or unmeasured full cross-product cells.')


def draw(fig,spec,grid,job=None):
    # Calling the exact existing painter preserves the colorbar, all color
    # limits/ticks, axis extents and the35 coarse results.
    ax=coarse.draw(fig,spec,grid['base_grid'],job)
    mapped=ax.collections[0]
    assert mapped.norm.vmin==1 and mapped.norm.vmax==1000 and mapped.cmap.name=='viridis'
    points=[(r['job']['tau_M_s'],r['job']['eta_m']) for r in grid['refinement_records']]
    for label in ax.texts:
        x,y=label.get_position()
        if any(abs(np.log10(x/tau))<.32 and abs(np.log10(y/eta))<.32 for tau,eta in points):
            label.set_visible(False)
    for r in grid['refinement_records']:
        x,y=r['job']['tau_M_s'],r['job']['eta_m']
        if r['event_observed']:
            color=mapped.to_rgba(r['first_entry']['confirmation_s']);marker='o'
        elif r['followup_s']>0:
            color=mapped.to_rgba(r['followup_s']);marker='^'
        else:
            color='white';marker='o'
        ax.scatter([x],[y],s=46,marker=marker,c=[color],edgecolors='#333333',linewidths=.7,zorder=9)
    assert ax.child_axes[0].get_yscale()=='log'
    assert np.array_equal(ax.child_axes[0].get_yticks(),[1,10,100,1000])
    return ax
