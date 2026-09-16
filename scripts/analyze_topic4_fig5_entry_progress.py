#!/usr/bin/env python3
"""Audit saved counts for an interim Fig5 first-entry lower-bound map."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/topic4_sef_hfo/fig5_log_m_entry_extension_20260915'


def read(path):
    return json.loads(path.read_text())


def audit_counts(folder):
    """Only complete, atomically published chunks enter the current estimate."""
    end = 0
    counts = []
    for path in sorted((folder / 'chunks').glob('*.npz')):
        if '.tmp.' in path.name:
            continue
        with np.load(path) as data:
            assert int(data['start_step']) == end, path
            end = int(data['end_step'])
            spikes = data['spikes_10ms']
            assert np.array_equal(spikes[:, 0], data['regions_10ms'][:, :3].sum(1))
            counts.append(spikes[:, 0])
    assert counts, folder
    rate = np.concatenate(counts) / 320  # 32,000 E cells, 10 ms bins.
    assert len(rate) * 100 == end
    changes = np.diff(np.r_[False, rate >= 200, False].astype(int))
    episodes = [(a, b) for a, b in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)) if b - a >= 20]
    first = None if not episodes else dict(onset_s=episodes[0][0] * .01, confirmation_s=(episodes[0][0] + 20) * .01)
    return dict(first_entry=first,followup_s=end * .0001,complete_count_coverage=True,
                audited_chunks=len(counts),late_mean_E_Hz=float(rate[-100:].mean()))


def collect():
    protocol = read(SOURCE / 'protocol.json')
    records = []
    for ref in protocol['references']:
        source = Path(ref['source'])
        assert hashlib.sha256((source / 'result.json').read_bytes()).hexdigest() == ref['result_sha256']
        records.append(dict(job=ref['job'],source=str(source),event_observed=True,
            first_entry=ref['first_entry'],followup_s=ref['first_entry']['confirmation_s'],reused=True,stage='OBSERVED'))
    completed = 0
    for job in protocol['jobs']:
        folder = SOURCE / 'runs' / job['name']
        parent = Path(protocol['sources'][job['name']]['source'])
        # A new worker may still be copying the original 300-second chunks.
        seeded = (folder / 'continuation.json').exists()
        source = folder if seeded else parent
        values = audit_counts(source)
        assert values['followup_s'] >= 300
        result_path = folder / 'result.json'
        done = result_path.exists()
        if done:
            result = read(result_path)
            assert result['job'] == job and result['identity'] == protocol['identity']
            assert np.isclose(values['followup_s'], result['elapsed_s'])
            assert (values['first_entry'] is not None) == result['event_observed']
            if result['event_observed']:
                for key in ('onset_s', 'confirmation_s'):
                    assert np.isclose(values['first_entry'][key], result['first_entry'][key])
            completed += 1
        entry = values['first_entry']
        records.append(dict(job=job,source=str(source),event_observed=entry is not None,
            reused=False,extension_complete=done,stage='OBSERVED' if entry else ('CENSORED_1000S' if done else 'RUNNING' if seeded else 'QUEUED'),**values))
    etas = protocol['eta_M']; taus = protocol['tau_M_s']
    shape = (len(etas), len(taus))
    mean = np.zeros(shape); entered = np.zeros(shape, int); count = np.zeros(shape, int)
    for i, eta in enumerate(etas):
        for j, tau in enumerate(taus):
            cell = [r for r in records if r['job']['eta_m'] == eta and r['job']['tau_M_s'] == tau]
            assert len(cell) == 2
            count[i, j] = 2
            entered[i, j] = sum(r['event_observed'] for r in cell)
            mean[i, j] = np.mean([r['first_entry']['confirmation_s'] if r['event_observed'] else r['followup_s'] for r in cell])
    assert len(records) == 48
    extensions = [r for r in records if not r['reused']]
    return dict(eta_M=etas,tau_M_s=taus,mean=mean,count=count,entered=entered,horizon_s=protocol['horizon_s'],
        completed_new=completed,total_new=22,reused=26,records=records,all_complete=completed == 22,
        pending=[r['job']['name'] for r in extensions if not r['extension_complete']],
        endpoint=protocol['endpoint'],snapshot_unix_s=time.time(),
        quantity_kind='mean_confirmation_time_lower_bound',
        scale='Log parameter axes and fixed1-1000s log color scale. Unhatched values are observed means; hatched values are lower bounds.',
        interpretation='Each cell averages two confirmation times, replacing each unobserved time by its own last fully audited follow-up. This is a lower bound when any endpoint is censored, not a common-horizon restricted mean. Queued continuations retain audited300s follow-up.',
        progress_summary=dict(observed_total=sum(r['event_observed'] for r in records),
            newly_observed=sum(r['event_observed'] for r in extensions),completed_extensions=completed,
            running=sum(r['stage'] == 'RUNNING' for r in extensions),queued=sum(r['stage'] == 'QUEUED' for r in extensions),
            audited_running_followup_s=[r['followup_s'] for r in extensions if r['stage'] == 'RUNNING']))


def draw(fig, spec, grid, job, renderer):
    ax = renderer(fig, spec, grid, letter='E', working_point=(job['tau_M_s'], job['eta_m']))
    assert len(ax.texts) == grid['mean'].size
    for label, val, observed in zip(ax.texts, grid['mean'].flat, grid['entered'].flat):
        # Round lower bounds down so displayed numbers never overstate follow-up.
        label.set_text((f'{val:.1f}' if observed == 2 else f'≥{np.floor(val):.0f}') + f'\n{observed}/2')
    ax.child_axes[0].set_ylabel('Mean entry time / lower bound (s)')
    return ax
