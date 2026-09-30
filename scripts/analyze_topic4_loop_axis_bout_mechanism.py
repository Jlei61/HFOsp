#!/usr/bin/env python3
"""Resource/feedback budgets for observed activity bouts, without new detectors.

Completed native graph controls only. Bouts use the existing jointquiet event
observer, including long bouts that do not meet operational high-entry criteria.
No simulation, altered threshold, or autonomous-return reclassification occurs.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import hashlib
import json
from pathlib import Path
import numpy as np
from analyze_topic4_loop_axis_native import load_chunks

ROOT = Path('/data/hfosp/topic4_sef_hfo/fig5_autonomous_loop_zk_20260924')
OUT = ROOT / 'axis_controls/native_runs'
SOURCE = Path('/data/hfosp/topic4_sef_hfo/fig5_global_feedback_response_20260923')


def read(path):
    return json.loads(path.read_text())


def one(row):
    source = row['condition'] == 'current'
    folder = (SOURCE if source else OUT / row['condition']) / 'runs' / row['name']
    result = read(folder / 'result.json')
    assert result['status'] == 'COMPLETE' and row['full120s']
    assert result['no_external_intervention'] and not result['Z_reset'] and not result['M_reset']
    slow = load_chunks(folder / 'chunks', ['slow_time_ms', 'Z', 'spikes_1ms', 'regions_1ms'])
    fb = load_chunks(folder / 'feedback_chunks', ['time_ms', 'K_mean', 'G_raw', 'G_applied_mean'])
    budget = load_chunks(folder / 'z_budget_chunks', ['time_ms', 'values'])
    causal = load_chunks(folder / 'mechanism_chunks', ['time_ms', 'global_E_rate_Hz'])
    response = load_chunks(folder / 'global_response_chunks', ['time_ms', 'q'])
    assert np.array_equal(slow['slow_time_ms'], fb['time_ms'])
    assert np.array_equal(causal['time_ms'], response['time_ms'])
    t = slow['slow_time_ms'] / 1000
    tb, tc = budget['time_ms'] / 1000, causal['time_ms'] / 1000
    with np.load(folder.parent.parent / 'geometry.npz') as geometry:
        counts = np.r_[32000, geometry['region_counts'][:3]]
    raw = np.c_[slow['spikes_1ms'][:, 0], slow['regions_1ms'][:, :3]]
    rates = raw.reshape(-1, 10, 4).sum(1) / counts / .01
    rt = np.arange(len(rates)) * .01
    b = budget['values']
    assert np.max(np.abs(b[:, :, 5])) < 1e-10
    assert np.max(np.abs(b[1:, :, 0] - b[:-1, :, 1])) < 1e-10

    def state(moment):
        k = max(0, np.searchsorted(t, moment, side='right') - 1)
        return dict(recorded_s=float(t[k]), requested_s=moment,
                    Z_all_A_B=slow['Z'][k, [0, 5, 6]].tolist(),
                    K_mean=float(fb['K_mean'][k]), G_raw=float(fb['G_raw'][k]),
                    G_applied_mean=float(fb['G_applied_mean'][k]))

    def window(lo, hi):
        # Budget endpoint b describes the preceding20ms; retain complete bins.
        keep = (tb - .02 >= lo - 1e-9) & (tb <= hi + 1e-9)
        native = (tc >= lo) & (tc < hi)
        selected = b[keep]
        if not len(selected):
            return None
        gain = .02 * selected[:, :, 2].sum(0)
        loss = .02 * selected[:, :, 3].sum(0)
        delta = selected[-1, :, 1] - selected[0, :, 0]
        error = delta - gain + loss
        assert np.max(np.abs(error)) < 1e-9
        r = causal['global_E_rate_Hz'][native]
        q = response['q'][native]
        observed = rates[(rt >= lo - 1e-9) & (rt + .01 <= hi + 1e-9)]
        return dict(requested_window_s=[lo, hi],
                    exact_budget_window_s=[float(tb[keep][0] - .02), float(tb[keep][-1])],
                    Z_recovery=gain.tolist(), Z_consumption=loss.tolist(),
                    Z_change=delta.tolist(), balance_error=error.tolist(),
                    causal_R_mean_Hz=float(r.mean()), causal_R_peak_Hz=float(r.max()),
                    fraction_R_at_or_below_5Hz=float(np.mean(r <= 5)),
                    fraction_jointquiet_10ms=float(np.mean(observed[:, :3].max(1) < 5)),
                    rates_mean_Hz_all_A_B_other=observed.mean(0).tolist(),
                    rates_peak_Hz_all_A_B_other=observed.max(0).tolist(),
                    fraction_q_positive=float(np.mean(q > 0)),
                    fraction_q_one=float(np.mean(q == 1)),
                    state_at_start=state(lo), state_at_end=state(hi))

    events = [dict(e, right_censored=False) for e in row['complete_activity_episodes']]
    unfinished = row['right_censored_activity_episode']
    if unfinished:
        events.append(dict(start_s=unfinished['start_s'], end_s=120.,
                           duration_s=unfinished['observed_duration_lower_bound_s'], right_censored=True))
    bouts = []
    for i, event in enumerate(events):
        lo, hi = event['start_s'], event['end_s']
        next_start = events[i + 1]['start_s'] if i + 1 < len(events) else 120.
        bouts.append(dict(start_s=lo, end_s=hi, duration_s=hi - lo,
                          right_censored=event['right_censored'],
                          long_relative_to_brief_definition=hi - lo > .2 + 1e-9,
                          operational_entries_inside=[e for e in row['entries'] if lo <= e['onset_s'] < hi],
                          activity=window(lo, hi),
                          following_interepisode_interval=window(hi, next_start) if next_start > hi else None,
                          following_window_right_censored=(i == len(events) - 1)))
    return dict(condition=row['condition'], name=row['name'], observed_s=120., bouts=bouts,
                result_sha256=hashlib.sha256((folder / 'result.json').read_bytes()).hexdigest(),
                region_order=['all_E', 'core_A', 'core_B', 'other_E'])


def main():
    analysis = read(OUT / 'analysis.json')
    rows = [one(row) for row in analysis['rows'] if row['full120s'] and row['run_status'] == 'COMPLETE']
    payload = dict(status='COMPLETE' if len(rows) == 3 else 'COMPLETED_GRAPHS_ONLY', rows=rows,
                   analysis_role='Post hoc descriptive mechanism alignment using the unchanged event observer; not a new primary endpoint.',
                   producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   native_analysis_sha256=hashlib.sha256((OUT / 'analysis.json').read_bytes()).hexdigest(),
                   selection='All existing jointquiet-bounded activity episodes within matched0-120s, plus right-censored final episode. No threshold or event selection changed.',
                   readout='Actual0.1ms Z gain/loss accumulated in full20ms bins. Causal global rate and feedback gate sampled at1ms. State landmarks use last recorded20ms sample at or before requested time.',
                   limits='Descriptive within-trajectory mechanism alignment; bouts are not independent replicates. Interepisode gaps may include activity below the event qualification threshold; actual jointquiet fraction is reported separately. Long activity is not automatically an operational high-state entry or recovered interictal activity. Neither delay, bifurcation, nor component necessity is established by temporal alignment.')
    (OUT / 'bout_mechanism.json').write_text(json.dumps(payload, indent=2) + '\n')
    lines = ['# 空间结构对照：长活动段与资源收支', '',
             '沿用原有共同低活动分隔，不改变进入阈值，也不增加自主返回次数。只读取已完成的原生轨迹，共同观察窗0–120秒；下表列所有超过200ms的活动段，完整JSON保留短事件与尾部截尾。资源收支用完整20ms原生积分块核对，核A/核B变化与全网平均分别保留。', '',
             '|结构|活动段(s)|进入阈值次数|K起→止|核A Z起→止|核B Z起→止|后续间隔(s)|后续核A/B ΔZ|',
             '|---|---|---|---|---|---|---|---|']
    for row in rows:
        for bout in row['bouts']:
            if not bout['long_relative_to_brief_definition']:
                continue
            w = bout['activity']; a, b = w['state_at_start'], w['state_at_end']; quiet = bout['following_interepisode_interval']
            tail = '（右截尾）' if bout['right_censored'] else ''
            gap = f"{quiet['requested_window_s'][1] - quiet['requested_window_s'][0]:.2f}" if quiet else '—'
            dz = '/'.join(f'{x:+.3f}' for x in quiet['Z_change'][1:3]) if quiet else '—'
            lines.append(f"|{row['condition']}|{bout['start_s']:.2f}–{bout['end_s']:.2f}{tail}|{len(bout['operational_entries_inside'])}|{a['K_mean']:.3f}→{b['K_mean']:.3f}|{a['Z_all_A_B'][1]:.3f}→{b['Z_all_A_B'][1]:.3f}|{a['Z_all_A_B'][2]:.3f}→{b['Z_all_A_B'][2]:.3f}|{gap}|{dz}|")
    lines += ['', '事件间隔可能包含未满足事件强度门槛的活动，JSON另列实际共同低活动占比，不能把整个间隔自动当作静默。表中时间对齐说明反馈与恢复的先后关系；不能凭时间相关性证明单个反馈项的必要性，也不能把未达到进入阈值的长活动直接算作已认证发作。原图与重配图的输出度及部分源权重不同，解释仍是复合空间结构效应。']
    (OUT / 'bout_mechanism.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(status=payload['status'], completed_graphs=[r['condition'] for r in rows],
                          long_bouts={r['condition']:sum(b['long_relative_to_brief_definition'] for b in r['bouts']) for r in rows})))


if __name__ == '__main__':
    main()
