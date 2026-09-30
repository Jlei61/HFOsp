#!/usr/bin/env python3
"""Audit high -> recurrent brief events -> high in continuous native SNN data.

This observer never changes network state. A temporal screen is not a clinical
seizure classification or acceptance of patient propagation geometry.
"""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import analyze_topic4_fixed_zm_termination as old

PARENT = old.OUT
OUT = Path('/data/hfosp/topic4_sef_hfo/fig5_interictal_recurrence_20260915')
RULE = dict(bin_ms=10, quiet_Hz=5., quiet_duration_ms=20., peak_Hz=20.,
            short_duration_ms=[20., 200.], required_brief_events=5,
            minimum_event_span_s=2., minimum_brief_fraction=.8)


def interval_events(events, lo, hi, max_duration=.2):
    complete = [e for e in events if e['start_s'] >= lo-1e-9 and e['end_s'] <= hi+1e-9]
    brief = [e for e in complete if .02-1e-9 <= e['duration_s'] <= max_duration+1e-9]
    span = brief[-1]['end_s']-brief[0]['start_s'] if brief else 0.
    return dict(window_s=[lo, hi], finite_events=complete, brief_events=brief,
                finite_count=len(complete), brief_count=len(brief), event_span_s=span,
                brief_fraction=len(brief)/len(complete) if complete else 0.,
                event_rate_Hz=len(brief)/(hi-lo) if hi > lo else None)


def qualifies(part, minimum_n=5, minimum_span=2., minimum_fraction=.8):
    return (part['brief_count'] >= minimum_n and part['event_span_s'] >= minimum_span-1e-9
            and part['brief_fraction'] >= minimum_fraction-1e-9)


def temporal_audit(rates, dt=.01, quiet=5., max_duration=.2, minimum_n=5):
    end = len(rates)*dt
    # Keep the historical high/exit gate at 10ms for all event sensitivities.
    assert np.isclose(dt, .01) or np.isclose(dt, .005)
    high_rates = rates if np.isclose(dt, .01) else rates[:len(rates)//2*2].reshape(-1, 2, 4).mean(1)
    tracker = old.run.carrier.fresh_tracker()
    for i, rate in enumerate(high_rates):
        old.run.carrier.track(tracker, rate, (i+1)*.01)
    events = old.event_audit.events(rates[:, 0], dt=dt, quiet=quiet, end=end)
    first = tracker['entries'][0]['onset_s'] if tracker['entries'] else end
    pre = interval_events(events, .5, first, max_duration)
    pairs = []
    for left, right in zip(tracker['entries'][:-1], tracker['entries'][1:]):
        recovery = next((r for r in tracker['recoveries']
                         if left['confirmation_s'] < r['confirmation_s'] < right['onset_s']), None)
        if recovery is None:
            continue
        part = interval_events(events, recovery['confirmation_s'], right['onset_s'], max_duration)
        part.update(first_entry_s=left['onset_s'], second_entry_s=right['onset_s'],
                    low_activity_confirmation_s=recovery['confirmation_s'],
                    temporal_pass=qualifies(part, minimum_n))
        pairs.append(part)
    post = None
    if tracker['recoveries']:
        last = tracker['recoveries'][-1]['confirmation_s']
        next_high = next((e['onset_s'] for e in tracker['entries'] if e['onset_s'] > last), end)
        post = interval_events(events, last, next_high, max_duration)
    passed = any(p['temporal_pass'] for p in pairs)
    if passed:
        label = 'TEMPORAL_LOOP_PASS_PENDING_SPATIAL'
    elif pairs:
        label = 'HIGH_LOW_HIGH_WITHOUT_INTERICTAL_RETURN'
    elif post is not None and qualifies(post, minimum_n):
        label = 'BRIEF_EVENTS_RETURNED_NO_SECOND_HIGH_YET'
    elif tracker['recoveries']:
        label = 'HIGH_ENDED_NO_INTERICTAL_RETURN'
    elif tracker['entries']:
        label = 'HIGH_WITHOUT_EXIT'
    else:
        label = 'NO_HIGH_OBSERVED'
    return dict(observed_s=end, entries=tracker['entries'], low_activity_exits=tracker['recoveries'],
                preentry=pre, interhigh_intervals=pairs, latest_postexit=post, events=events,
                temporal_loop_pass=passed, preentry_brief_screen=qualifies(pre, minimum_n),
                classification=label, high_gate='All E >=200Hz for200ms; unchanged',
                event_settings=dict(dt_s=dt, quiet_Hz=quiet, max_duration_s=max_duration,
                                    minimum_events=minimum_n))


def analyze_folder(folder, geometry, sensitivities=True):
    d = old.load(folder, keys=['time_ms', 'spikes_1ms', 'regions_1ms'])
    if not d:
        return None
    with np.load(geometry) as g:
        counts = g['region_counts'][:3].copy()
    nr = np.r_[32000, counts]
    def rates_at(ms):
        n = len(d['spikes_1ms'])//ms
        all_e = d['spikes_1ms'][:n*ms, 0].reshape(n, ms).sum(1)
        local = d['regions_1ms'][:n*ms, :3].reshape(n, ms, 3).sum(1)
        return np.column_stack([all_e, local])/nr/(ms*.001)
    primary = temporal_audit(rates_at(10))
    variants = []
    if sensitivities:
        for ms, quiet, duration, minimum in [(5,5.,.2,5), (10,1.,.2,5),
                                              (10,5.,.3,5), (10,5.,.2,3)]:
            v = temporal_audit(rates_at(ms), ms*.001, quiet, duration, minimum)
            variants.append({k:v[k] for k in ['event_settings','classification','temporal_loop_pass']})
    zeros = old.spans(d['spikes_1ms'][:, 0] == 0)
    longest = max(zeros, key=lambda p:p[1]-p[0], default=(0,0))
    result = json.loads((folder/'result.json').read_text()) if (folder/'result.json').exists() else {}
    applied_path = folder/'applied_configuration.json'
    applied = json.loads(applied_path.read_text()) if applied_path.exists() else {}
    return dict(source=str(folder), run_status=result.get('status','RUNNING'),
                job=result.get('job',applied.get('job',{})), rule=RULE, primary=primary,
                sensitivity=variants, exact_all_E_zero_interval_s=np.array(longest)*.001,
                morphology_and_spatial_acceptance='PENDING_NATIVE_REVIEW',
                full_Fig5_acceptance='NOT_ESTABLISHED', human_review='PENDING')


def audit_previous():
    rows=[]
    batches=['','matched_spatial_round2','hyperpolar_spatial_round3','source_sahp_round4',
             'sahp_bracket_round5','positive_candidate_confirmation','low_fraction_round6',
             'autonomous_recurrence_continuation']
    for batch in batches:
        root=PARENT/batch
        if not (root/'protocol.json').exists():
            continue
        for job in json.loads((root/'protocol.json').read_text())['initial_jobs']:
            folder=root/'runs'/job['name']
            if not (folder/'result.json').exists():
                continue
            row=analyze_folder(folder,root/'geometry.npz')
            row['same_trajectory_continuation']=batch=='autonomous_recurrence_continuation'
            old.write(OUT/'previous_audit'/f"{job['name']}.json", row)
            rows.append(dict(name=job['name'],batch=batch or 'initial',
                             observed_s=row['primary']['observed_s'],
                             classification=row['primary']['classification'],
                             preentry_brief_count=row['primary']['preentry']['brief_count'],
                             n_high=len(row['primary']['entries']),
                             n_low_exit=len(row['primary']['low_activity_exits']),
                             n_brief_between_high=max([p['brief_count'] for p in row['primary']['interhigh_intervals']],default=0),
                             temporal_loop_pass=row['primary']['temporal_loop_pass'],
                             continuation=row['same_trajectory_continuation']))
    old.write(OUT/'previous_audit_summary.json',dict(rule=RULE,rows=rows,
              n_unique_runs=sum(not r['continuation'] for r in rows),
              note='New criterion is prospective for new runs; old results are re-audited, not altered.'))
    with (OUT/'previous_audit_summary.csv').open('w') as h:
        writer=csv.DictWriter(h,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(rows,ensure_ascii=False),flush=True)


def qa():
    def synth(brief_count=8, long=False, second=True):
        r=np.zeros((1600,4));r[100:150]=250
        for i in range(brief_count):
            start=400+i*50;r[start:start+(30 if long else 10)]=40
        if second:r[1300:1400]=250
        return r
    assert not temporal_audit(synth(0))['temporal_loop_pass'], 'Silence counted as interictal'
    assert not temporal_audit(synth(1))['temporal_loop_pass'], 'Single event counted as state return'
    assert not temporal_audit(synth(8,True))['temporal_loop_pass'], 'Long waves counted as brief IEDs'
    assert temporal_audit(synth())['temporal_loop_pass']
    r=temporal_audit(synth(second=False))
    assert not r['temporal_loop_pass'] and r['classification']=='BRIEF_EVENTS_RETURNED_NO_SECOND_HIGH_YET'
    assert temporal_audit(np.repeat(synth(),2,axis=0),dt=.005)['temporal_loop_pass']
    old.write(OUT/'temporal_observer_qa.json',dict(status='PASS',
              checks=['silence rejected','one event rejected','long waves rejected',
                      'recurrent short events between two highs accepted','recurrence required','5ms sensitivity']))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['qa','previous','one'])
    parser.add_argument('--folder',type=Path);parser.add_argument('--geometry',type=Path)
    args=parser.parse_args()
    if args.action=='qa':qa()
    elif args.action=='previous':audit_previous()
    else:print(json.dumps(old.safe(analyze_folder(args.folder,args.geometry)),indent=2))
