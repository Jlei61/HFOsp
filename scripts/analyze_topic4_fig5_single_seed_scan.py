#!/usr/bin/env python3
"""Single-noise first-entry values and audited censoring bounds for Fig5 E."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import LogFormatterSciNotation
from analyze_topic4_fig5_entry_progress import audit_counts

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/topic4_sef_hfo/fig5_single_seed_wide_m_20260916'


def read(path):
    return json.loads(path.read_text())


def collect():
    protocol = read(OUT / 'protocol.json')
    records = []
    for ref in protocol['references']:
        source = Path(ref['source'])
        assert hashlib.sha256((source / 'result.json').read_bytes()).hexdigest() == ref['result_sha256']
        records.append(dict(ref, complete=True, stage='OBSERVED' if ref['event_observed'] else 'CENSORED', reused=True))
    for ref in protocol['adopted'] + [dict(job=j, source=str(OUT / 'runs' / j['name'])) for j in protocol['jobs']]:
        source = Path(ref['source'])
        chunks = list((source / 'chunks').glob('*.npz'))
        values = audit_counts(source) if any('.tmp.' not in p.name for p in chunks) else dict(first_entry=None, followup_s=0.)
        path = source / 'result.json'
        done = path.exists()
        if done:
            result = read(path)
            assert result['identity'] == protocol['identity'] and result['job'] == ref['job']
            assert np.isclose(values['followup_s'], result['elapsed_s'])
            assert (values['first_entry'] is not None) == result['event_observed']
            if result['event_observed']:
                for key in ('onset_s', 'confirmation_s'):
                    assert np.isclose(values['first_entry'][key], result['first_entry'][key])
            else:
                assert values['followup_s'] >= protocol['horizon_s']
        observed = values['first_entry'] is not None
        stage = 'OBSERVED' if observed else 'CENSORED' if done else 'RUNNING' if (source / 'progress.json').exists() else 'QUEUED'
        records.append(dict(job=ref['job'], source=str(source), complete=done, event_observed=observed,
                            stage=stage, reused=False, **values))
    assert len(records) == protocol['total_cells'] == 35
    assert {r['job']['seed'] for r in records} == {protocol['seed']}
    shape = (len(protocol['eta_M']), len(protocol['tau_M_s']))
    value = np.full(shape, np.nan)
    observed = np.zeros(shape, int)
    followup = np.zeros(shape)
    stages = np.full(shape, '', dtype=object)
    for i, eta in enumerate(protocol['eta_M']):
        for j, tau in enumerate(protocol['tau_M_s']):
            cell = [r for r in records if r['job']['eta_m'] == eta and r['job']['tau_M_s'] == tau]
            assert len(cell) == 1
            r = cell[0]
            observed[i, j] = r['event_observed']
            followup[i, j] = r['followup_s']
            stages[i, j] = r['stage']
            if r['event_observed']:
                value[i, j] = r['first_entry']['confirmation_s']
            elif r['followup_s'] > 0:
                value[i, j] = r['followup_s']
    return dict(eta_M=protocol['eta_M'], tau_M_s=protocol['tau_M_s'], seed=protocol['seed'],
                value=value, entered=observed, followup=followup, stages=stages, records=records,
                horizon_s=protocol['horizon_s'], all_complete=all(r['complete'] for r in records),
                quantity_kind='single_seed_confirmation_time', snapshot_unix_s=time.time(),
                endpoint=protocol['endpoint'],
                interpretation='One fixed noise realization per cell. Observed confirmation time or an audited right-censoring lower bound; no averaging across seeds.',
                progress_summary=dict(total_cells=35, observed=sum(r['event_observed'] for r in records),
                    complete=sum(r['complete'] for r in records), censored=sum(r['stage']=='CENSORED' for r in records),
                    running=sum(r['stage']=='RUNNING' for r in records), queued=sum(r['stage']=='QUEUED' for r in records),
                    new_complete=sum((OUT/'runs'/j['name']/'result.json').exists() for j in protocol['jobs']), new_total=11))


def edges(centers):
    x = np.log10(centers)
    return 10**np.r_[x[0]-(x[1]-x[0])/2, (x[:-1]+x[1:])/2, x[-1]+(x[-1]-x[-2])/2]


def draw(fig, spec, grid, job=None):
    ax = fig.add_subplot(spec)
    ax.set_box_aspect(1)
    x, y = edges(grid['tau_M_s']), edges(grid['eta_M'])
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('#ededed')
    im = ax.pcolormesh(x, y, np.ma.masked_invalid(grid['value']), cmap=cmap,
        norm=LogNorm(1, grid['horizon_s']), edgecolors='#ffffff99', lw=.6)
    ax.set(xscale='log', yscale='log', xticks=grid['tau_M_s'], yticks=grid['eta_M'],
        xlabel=r'$\tau_M$ (s)', ylabel=r'$\eta_M$')
    ax.set_xticklabels([r'$10^{%d}$' % round(np.log10(v)) for v in grid['tau_M_s']])
    ax.set_yticklabels([f'{v:g}' for v in grid['eta_M']])
    ax.minorticks_off()
    for i in range(len(grid['eta_M'])):
        for j in range(len(grid['tau_M_s'])):
            v = grid['value'][i, j]
            observed = bool(grid['entered'][i, j])
            label = '…' if not np.isfinite(v) else f'{v:.1f}' if observed else f'≥{np.floor(v):.0f}'
            if np.isfinite(v) and not observed:
                ax.add_patch(Rectangle((x[j],y[i]),x[j+1]-x[j],y[i+1]-y[i],
                    fc='none',ec='#55555588',lw=0,hatch='///'))
            ax.text(np.sqrt(x[j]*x[j+1]),np.sqrt(y[i]*y[i+1]),label,
                    ha='center',va='center',fontsize=12,color='white' if np.isfinite(v) and v<80 else '#111')
    if job is not None:
        i=grid['eta_M'].index(job['eta_m']); j=grid['tau_M_s'].index(job['tau_M_s'])
        ax.add_patch(Rectangle((x[j],y[i]),x[j+1]-x[j],y[i+1]-y[i],fc='none',ec='black',lw=2))
    slot=spec.get_position(fig)
    fig.text(slot.x0,slot.y1+.012,'E',weight='bold',fontsize=24,ha='left')
    cb=fig.colorbar(im,cax=ax.inset_axes([1.06,0,.055,1]))
    cb.set_label('Entry time / lower bound (s)')
    cb.set_ticks([1,10,100,1000])
    cb.formatter=LogFormatterSciNotation(base=10,labelOnlyBase=False,minor_thresholds=(np.inf,np.inf))
    cb.update_ticks();cb.ax.minorticks_off()
    return ax
