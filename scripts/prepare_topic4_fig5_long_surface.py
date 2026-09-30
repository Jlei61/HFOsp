#!/usr/bin/env python3
"""Audit the completed 3000-s supplement and freeze the common1000-s surface."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import matplotlib.tri as mtri
import analyze_topic4_fig5_long_boundary as analysis
import plot_topic4_fig5_continuous_boundary as painter


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def collect():
    run = analysis.run
    status = run.base.read(run.OUT / 'status.json')
    assert status['completed'] == status['total'] == 19
    assert not status['running'] and not status['failed'] and not status['pending']
    # Re-read all saved 10-ms counts; check contiguous coverage, regional E
    # conservation, first qualifying episode, endpoint and network identity.
    protocol, original, records, jobs = analysis.collect()
    assert len(records) == 70 and len(jobs) == 19 and all(r['complete'] for r in jobs)
    assert {r['seed'] for r in records} == {9108401}
    assert len({(r['tau_M_s'], r['eta_M']) for r in records}) == 70
    for reference in original['continuous_surface']['source_records']:
        assert sha(Path(reference['source']) / 'result.json') == reference['result_sha256']
    continuation_qa = run.base.read(run.OUT / 'continuation_qa.json')
    assert continuation_qa['status'] == 'PASS' and len(continuation_qa['checks']) == 8
    assert all(c['whole_engine_exact'] and c['whole_tracker_exact'] and c['counts_exact']
               for c in continuation_qa['checks'])
    for r in jobs:
        result = run.base.read(Path(r['source']) / 'result.json')
        assert result['no_reset'] and result['source_hashes'] == protocol['source_hashes']
        r['onset_s'] = result['first_entry']['onset_s'] if r['event_observed'] else None
        if r['kind'] == 'continue_nearest_censored':
            assert not r['event_observed'], 'Inspect any delayed entry in an original censored point.'
    common = analysis.at_horizon(records, 1000.)
    assert len(common) == 70
    xy = np.array([[r['tau_M_s'], r['eta_M']] for r in common])
    values = np.array([r['plot_time_s'] for r in common])
    entered = np.array([r['entered_by_horizon'] for r in common])
    # Clip each measured node to the common horizon BEFORE interpolation.
    # No unextended1000-s censor is ever imputed as a3000-s observation.
    assert np.all(values[~entered] == 1000)
    logxy = np.log10(xy)
    tri = mtri.Triangulation(*logxy.T)
    interpolation = mtri.LinearTriInterpolator(tri, np.log10(values))
    error = float(np.max(abs(interpolation(*logxy.T) - np.log10(values))))
    assert error < 1e-10
    late = [r for r in jobs if r['event_observed'] and r['confirmation_s'] > 1000]
    summary = dict(
        total_unique_points=70, original_points=59, new_parameter_points=11,
        completed_long_jobs=19, continued_jobs=8, fresh_jobs=11,
        long_jobs_entered=sum(r['event_observed'] for r in jobs),
        long_jobs_censored3000=sum(not r['event_observed'] for r in jobs),
        entered_by1000=int(entered.sum()), no_entry_by1000=int((~entered).sum()),
        observed_entries=sum(r['event_observed'] for r in records), late_entries=len(late),
        censored1000=sum(not r['event_observed'] and r['followup_s'] == 1000 for r in records),
        censored3000=sum(not r['event_observed'] and r['followup_s'] == 3000 for r in records),
        eligible3000_points=len(analysis.at_horizon(records, 3000)),
        actual_entry_range_s=[min(r['confirmation_s'] for r in records if r['event_observed']),
                              max(r['confirmation_s'] for r in records if r['event_observed'])])
    assert summary['entered_by1000'] == 33 and summary['observed_entries'] == 34
    assert summary['censored1000'] == 24 and summary['censored3000'] == 12
    assert len(late) == 1 and np.isclose(late[0]['confirmation_s'], 1535.03)
    taus = sorted({b['tau_M_s'] for b in original['continuous_surface']['boundary_brackets']})
    brackets = {str(h): analysis.bracket_curve(records, h, taus)['brackets'] for h in (1000, 3000)}
    grid = copy.deepcopy(original)
    grid['continuous_surface'] = dict(
        parameters=xy.tolist(), log10_parameters=logxy.tolist(), triangles=tri.triangles.tolist(),
        entry_observed=entered.tolist(), entry_observed_definition='Confirmed by the common1000-s horizon.',
        time_or_lower_bound_s=values.tolist(), n_total=70, n_entered=33, n_censored=37,
        common_horizon_s=1000, display_cap_s=1000,
        entry_range_s=[float(values[entered].min()), float(values[entered].max())],
        interpolation_at_samples_max_error_log10_s=error,
        source_records=[dict(source=r['source'], result_sha256=sha(Path(r['source']) / 'result.json'))
                        for r in common],
        interpolation='Piecewise-linear barycentric interpolation of log10(min(T_confirmation,1000 s)) in log10(tau_M),log10(eta_M); no extrapolation outside the measured convex hull.',
        censoring='Color capped at1000s for any later entry or non-entry. Actual follow-up is preserved separately:24 untouched1000s censorings,12 audited3000s censorings,1 observed1535.03s entry.',
        producer=str(Path(__file__).resolve()), producer_sha256=sha(__file__))
    grid['long_followup'] = dict(
        summary=summary, records=records, jobs=jobs, late_entries=late, measured_brackets=brackets,
        protocol_file=str(run.OUT / 'protocol.json'), protocol_sha256=sha(run.OUT / 'protocol.json'),
        count_audit='PASS_ALL19_RAW_COUNT_ENDPOINTS',
        continuation_qa=continuation_qa,
        display_interpretation='Common1000-s restricted entry-time surface; longer follow-up reported in caption and tables. This is not a complete3000-s map.')
    grid['all_complete'] = True
    grid['snapshot_unix_s'] = time.time()
    grid['interpretation'] = grid['continuous_surface']['interpolation'] + ' ' + grid['continuous_surface']['censoring']
    painter.surface_only(grid)
    grid['display_style']['author_request'] = '2026-09-28: incorporate completed long runs; retain the surface-only display and fixed1-1000s colorbar.'
    snapshot = dict(grid=grid, simulation_status=status, live_processes={}, frozen_at_unix_s=time.time(),
                    seed=9108401, source_figure=str(run.PAPER / 'figures/fig5-complete-layout.png'),
                    previous_snapshot=str(run.OUT / 'previous_1000s_snapshot.json'),
                    previous_snapshot_sha256=protocol['previous_snapshot_sha256'])
    return snapshot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot = collect()
    write(args.output / 'source_snapshot.json', snapshot)
    long = snapshot['grid']['long_followup']
    write(args.output / 'long_followup_audit.json', long)
    fields = ['tau_M_s', 'eta_M', 'seed', 'event_observed', 'confirmation_s', 'followup_s',
              'entered_by1000', 'display_time_s', 'long_job', 'source']
    with (args.output / 'entry_points.csv').open('w') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for r in long['records']:
            writer.writerow({**{k:r[k] for k in fields if k in r},
                'entered_by1000':r['event_observed'] and r['confirmation_s'] <= 1000,
                'display_time_s':min(r['confirmation_s'], 1000) if r['event_observed'] else 1000})
    with (args.output / 'boundary_brackets.csv').open('w') as file:
        writer = csv.DictWriter(file, fieldnames=['horizon_s', 'tau_M_s', 'eta_enter', 'eta_censored', 'eta_mid'])
        writer.writeheader()
        for horizon, rows in long['measured_brackets'].items():
            writer.writerows(dict(horizon_s=int(horizon), **r) for r in rows)
    print(json.dumps(long['summary'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
