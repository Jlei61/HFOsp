#!/usr/bin/env python3
"""Conditional native evidence, including censored activity and propagation.

No observer is changed. Temporal quantiles are not seed confidence intervals.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from pathlib import Path
import numpy as np
from campaign import ROOT, NATIVE, read, write
import analyze_topic4_loop_zk_conditional as original
from zoom_topic4_return_core_propagation import event_metrics, summarize


def activity_censor(rates, dt=.01):
    quiet = [(a, b) for a, b in original.event_audit.f.spans(rates[:, :3].max(1) < 5.)
             if (b-a)*dt >= .02-1e-9]
    tail_start = quiet[-1][1] if quiet else 0
    right = None
    if len(rates)-tail_start >= 2 and rates[tail_start:, 0].max() >= 20.:
        right = dict(start_s=tail_start*dt, observed_end_s=len(rates)*dt,
            observed_duration_lower_bound_s=(len(rates)-tail_start)*dt,
            left_boundary_confirmed=bool(quiet), right_censored=True,
            mean_Hz_allE_A_B_other=rates[tail_start:].mean(0).tolist(),
            peak_Hz_allE_A_B_other=rates[tail_start:].max(0).tolist())
    left = None
    first_quiet = quiet[0][0] if quiet else len(rates)
    if first_quiet >= 2 and rates[:first_quiet, 0].max() >= 20.:
        left = dict(observed_start_s=0., observed_end_s=first_quiet*dt,
                    observed_duration_lower_bound_s=first_quiet*dt,
                    left_censored=True, right_boundary_confirmed=bool(quiet))
    return dict(initial_left_censored=left, final_right_censored=right,
                both_describe_same_unbounded_episode=not quiet and left is not None and right is not None)


def analyze(root, name):
    original.OUT = root
    row, inputs = original.analyze(name)
    folder = root/'runs'/name
    with np.load(root/'geometry.npz') as g:
        counts = np.r_[32000, g['region_counts'][:3]]
        cell_count = g['cell_e_counts'].copy()
    d = original.load(folder/'chunks', ['time_ms','spikes_1ms','regions_1ms',
                                      'field_time_ms','field_5ms','slow_time_ms','Z','M'])
    raw = np.c_[d['spikes_1ms'][:, 0], d['regions_1ms'][:, :3]]
    r10 = raw.reshape(-1, 10, 4).sum(1)/counts/.01
    r5 = raw.reshape(-1, 5, 4).sum(1)/counts/.005
    duration = len(raw)/1000.
    assert abs(duration-row['duration_s']) < 1e-8
    expected = row['job']['horizon_s']-row['job']['branch_start_s']
    row['full_horizon'] = abs(duration-expected)<1e-8
    row['observation_horizon_s'] = expected
    row['interpretation'] = 'Finite-time conditional native response; persistence alone does not certify an attractor or bifurcation.'
    tail_events = [e for e in row['events'] if e['start_s'] >= duration-10]
    brief = [e for e in tail_events if .02-1e-9 <= e['duration_s'] <= .2+1e-9]
    spatial = [event_metrics(dict(rates=r5), e) for e in brief]
    onset_intervals = np.diff([e['start_s'] for e in brief])
    intervals = dict(n=len(onset_intervals),
        median_s=float(np.median(onset_intervals)) if len(onset_intervals) else None,
        coefficient_of_variation=float(np.std(onset_intervals,ddof=1)/np.mean(onset_intervals))
        if len(onset_intervals) >= 2 else None,
        interpretation='Intervals between complete brief events within this one trajectory; missing nonbrief events can lengthen them. Neither a stationarity test nor a limit-cycle certificate.')
    feedback = original.load(folder/'feedback_chunks', ['time_ms','R_global_Hz','gate','G_raw','G_applied_mean','K_mean'])
    tail_start = (row['job']['horizon_s']-10)*1000.
    # Feedback is sampled at left endpoints, unlike the preceding-bin drift.
    mask = (feedback['time_ms'] >= tail_start-1e-8) & (feedback['time_ms'] < row['job']['horizon_s']*1000.-1e-8)
    assert mask.sum() == 500
    fb = {k: dict(mean=float(v[mask].mean()), minimum=float(v[mask].min()),
                  maximum=float(v[mask].max())) for k,v in feedback.items() if k != 'time_ms'}
    windows=[]
    for lo in np.arange(0.,duration,5.):
        hi=min(lo+5.,duration);rates=r10[round(lo*100):round(hi*100)]
        ev=[e for e in row['events'] if lo <= e['start_s'] and e['end_s'] <= hi]
        windows.append(dict(relative_s=[float(lo),float(hi)],
            mean_Hz_allE_A_B_other=rates.mean(0).tolist(),
            high_fraction=float((rates[:,0]>=200.).mean()),
            joint_quiet_fraction=float((rates[:,:3]<5.).all(1).mean()),
            complete_brief_events=int(sum(.02-1e-9<=e['duration_s']<=.2+1e-9 for e in ev))))
    half_second_means=r10[-1000:].reshape(20,50,4).mean(1)
    row.update(censoring=activity_censor(r10), tail_event_metrics=spatial,
        tail_core_recruitment=summarize(spatial) if spatial else None,
        tail_onset_intervals=intervals, tail_feedback=fb, five_second_windows=windows,
        tail_500ms_mean_quantiles_allE_A_B_other_Hz=np.quantile(half_second_means,[.1,.5,.9],axis=0).tolist(),
        temporal_quantile_definition='10/50/90percentiles across20 nonoverlapping500ms means in the final10s; within-trajectory variability, not seed uncertainty.',
        right_censor_definition='Incomplete activity after the last jointquiet interval>=20ms, wholeE peak>=20Hz and>=20ms observed activity; excluded from complete-event counts. Initial ongoing activity is separately left-censored.',
        field_template_limit='Held Z/K spatial fields are defined by the job templates; matching their means does not match a natural stage-specific full state.',
        held_Z_template=row['job'].get('Z_template',row['job'].get('field_template','source t20s')),
        held_K_template=row['job'].get('K_template',row['job'].get('field_template','source t20s')),
        human_spatial_review='PENDING')
    dest=root/'extended_analysis';dest.mkdir(exist_ok=True)
    write(dest/f'{name}.json',row)
    np.savez_compressed(dest/f'{name}_readouts.npz',
        relative_time_5ms_s=(np.arange(len(r5))+.5)*.005,
        rate_5ms_Hz=r5,field_rate_5ms_Hz=(d['field_5ms']/cell_count/.005).astype('f4'),
        absolute_start_s=row['job']['horizon_s']-duration)
    return row,inputs


def main(root):
    rows=[];anchors={};names=read(root/'queue.json')['names']
    for name in names:
        path=root/'runs'/name/'result.json'
        if not path.exists() or read(path)['status']!='COMPLETE':continue
        row,inputs=analyze(root,name)
        noise=row['job'].get('external_noise_source','original t50s paired input')
        key=(noise,row['observation_horizon_s'])
        if key not in anchors:anchors[key]=inputs
        else:assert np.array_equal(anchors[key],inputs),name
        rows.append(row)
    write(root/'extended_analysis_summary.json',dict(
        status='COMPLETE' if len(rows)==len(names) else 'PARTIAL',completed=len(rows),total=len(names),
        rows=rows,common_future_inputs_exact_within_noise_and_horizon=True,
        distinct_future_input_groups=len(anchors),
        formal_bifurcation='NOT_ESTABLISHED',human_review='PENDING'))
    print([(r['name'],r['finite_window_state'],r['tail_brief_events'],
            r['tail_core_recruitment']['core_over100'] if r['tail_core_recruitment'] else 0)
           for r in rows],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=NATIVE)
    main(p.parse_args().root)
