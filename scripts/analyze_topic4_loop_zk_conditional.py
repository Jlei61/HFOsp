#!/usr/bin/env python3
"""Finite-window native state/conditional drift summaries; no bifurcation naming."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import analyze_topic4_fig5_preentry_events as event_audit

OUT = Path('/data/hfosp/topic4_sef_hfo/fig5_autonomous_loop_zk_20260924')


def load(folder, keys):
    records = {key: [] for key in keys}
    for path in sorted(folder.glob('*.npz')):
        with np.load(path) as z:
            for key in keys:
                records[key].append(z[key])
    return {key: np.concatenate(value) for key, value in records.items()}


def compare_readouts(reference, comparison):
    """Descriptive differences, with no stationarity or equivalence test."""
    left, right = reference['tail_native_readouts'], comparison['tail_native_readouts']
    assert left['cell_e_counts'] == right['cell_e_counts']
    assert left['contact_names'] == right['contact_names']
    counts = np.asarray(left['cell_e_counts'])
    field_delta = np.asarray(right['mean_field_Hz'])-left['mean_field_Hz']
    contact_delta = np.asarray(right['mean_contact_current_mV_equiv'])-left['mean_contact_current_mV_equiv']
    return dict(reference=reference['name'], comparison=comparison['name'],
        same_coarse_state_label=reference['finite_window_state']==comparison['finite_window_state'],
        mean_rate_delta_allE_A_B_other_Hz=(np.asarray(comparison['tail_mean_Hz'])-reference['tail_mean_Hz']).tolist(),
        cell_weighted_absolute_mean_field_difference_Hz=float(np.average(abs(field_delta), weights=counts)),
        contact_names=left['contact_names'], mean_contact_current_delta_mV_equiv=contact_delta.tolist(),
        interpretation='Comparison minus reference, over each last10s. Mean spatial occupancy and mean contact current do not measure propagation direction. One paired trajectory per condition; equal labels or small differences do not establish convergence, equivalence or a shared attractor.')


def analyze(name):
    folder = OUT/'runs'/name
    result = json.loads((folder/'result.json').read_text())
    job = result['job']
    data = load(folder/'chunks', ['time_ms', 'spikes_1ms', 'regions_1ms', 'field_time_ms', 'field_5ms', 'inputs', 'Z', 'M'])
    n = len(data['spikes_1ms'])//10
    with np.load(OUT/'geometry.npz') as g:
        counts = np.r_[32000, g['region_counts'][:3]]
        cell_counts = g['cell_e_counts'].copy()
        contact_names = g['contact_names'].astype(str).tolist()
    raw = np.c_[data['spikes_1ms'][:, 0], data['regions_1ms'][:, :3]]
    rates = raw[:n*10].reshape(n, 10, 4).sum(1)/counts/.01
    assert np.array_equal(data['spikes_1ms'][:, 0], data['regions_1ms'][:, :3].sum(1))
    assert np.array_equal(data['field_5ms'].sum(1), data['spikes_1ms'][:, 0].reshape(-1, 5).sum(1))
    duration = n*.01
    last = rates[-1000:]
    events = event_audit.events(rates[:, :3].max(1), end=duration)
    events = [e for e in events if rates[round(e['start_s']*100):round(e['end_s']*100), 0].max() >= 20]
    tail_events = [e for e in events if e['start_s'] >= duration-10]
    brief = [e for e in tail_events if .02-1e-9 <= e['duration_s'] <= .2+1e-9]
    high_fraction = float(np.mean(last[:, 0] >= 200))
    joint_quiet = float(np.mean(np.all(last[:, :3] < 5, axis=1)))
    span = brief[-1]['start_s']-brief[0]['start_s'] if len(brief)>1 else 0.
    if high_fraction >= .8:
        state = 'sustained_high'
    elif len(brief) >= 10 and span >= 5 and len(brief)/max(1, len(tail_events)) >= .8:
        state = 'recurrent_brief'
    elif joint_quiet >= .95:
        state = 'quiet'
    else:
        state = 'mixed_or_transient'
    tail_start_ms = (job['horizon_s']-10)*1000.
    field_mask = data['field_time_ms'] >= tail_start_ms
    assert field_mask.sum() == 2000
    mean_field = data['field_5ms'][field_mask].sum(0)/cell_counts/10.
    assert np.isclose(np.average(mean_field, weights=cell_counts), last[:, 0].mean(), rtol=1e-13, atol=1e-13)
    contacts = load(folder/'actual_current_chunks', ['time_ms', 'contact_current'])
    contact_mask = contacts['time_ms'] >= tail_start_ms
    assert contact_mask.sum() == 5000
    assert np.allclose(np.diff(contacts['time_ms']), 2., rtol=0., atol=1e-8)
    readouts = dict(relative_window_s=[duration-10, duration],
        absolute_window_s=[job['horizon_s']-10, job['horizon_s']],
        mean_field_Hz=mean_field.tolist(), cell_e_counts=cell_counts.tolist(),
        contact_names=contact_names,
        mean_contact_current_mV_equiv=contacts['contact_current'][contact_mask].mean(0).tolist(),
        definition='Native400bin mean E rate and original15contact mean current over last10s. Spatial occupancy, not propagation direction. Contact proxy is weighted absIE+absZII+absGcurrent, not HFO energy; K/M excluded.')
    drift = load(folder/'conditional_drift_chunks', ['time_ms', 'values'])
    # Drift records are means over the preceding20ms, timestamped at their right
    # endpoints. Exclude the left endpoint to avoid including the preceding bin.
    mask = (drift['time_ms'] > tail_start_ms+1e-8) & (drift['time_ms'] <= job['horizon_s']*1000.+1e-8)
    assert mask.sum() == 500
    assert np.max(abs(data['Z'][:, 0]-job['target_Z'])) < 1e-12
    drift_mean = drift['values'][mask].mean(0)
    held_z = data['Z'][0, [0, 5, 6, 7]]
    assert np.max(abs(data['Z'][:, [0, 5, 6, 7]]-held_z)) < 1e-12
    # At held Z, averaging dZ=(eligible-Z)/tau_Z gives the exact mean
    # eligibility in each observer region, without treating spike rate as load.
    eligible_fraction = held_z+job['tau_Z_s']*drift_mean[:, 0]
    assert np.all((eligible_fraction >= -1e-12) & (eligible_fraction <= 1+1e-12))
    row = dict(name=name, job=job, duration_s=duration, finite_window_state=state,
        full_horizon=abs(duration-30.)<1e-9, tail_mean_Hz=last.mean(0).tolist(),
        tail_high_fraction=high_fraction, tail_joint_quiet_fraction=joint_quiet,
        tail_brief_events=len(brief), tail_brief_span_s=span, events=events,
        tail_native_readouts=readouts,
        counterfactual_drift_mean_allE_A_B_other=drift_mean.tolist(),
        held_Z_mean_allE_A_B_other=held_z.tolist(),
        tail_Z_recovery_eligible_fraction_allE_A_B_other=eligible_fraction.tolist(),
        eligibility_definition='Reconstructed as held regional Z + tau_Z times native mean counterfactual dZ/dt; each is a neuron-time fraction over last10s. Recovery criterion is J<I_th, not low firing rate. Regional means use the same observer masks as the rates and drift.',
        drift_definition='Native dZ/dt and discrete native dK/dt at the held spatial fields; G and M dynamic. Last500 preceding20ms bins, with right endpoints in (horizon-10,horizon].',
        independent_unit='one conditional trajectory; paired initial histories, one shared future noise stream',
        certified_bifurcation=False, counts_as_autonomous_loop=False,
        interpretation='Finite30s conditional response, not demonstrated attractor convergence or original full-model bistability.')
    (OUT/'analysis').mkdir(exist_ok=True)
    (OUT/'analysis'/f'{name}.json').write_text(json.dumps(row, indent=2)+'\n')
    return row, data['inputs'][:, 1:]


def main():
    queue = json.loads((OUT/'queue.json').read_text())['names']
    rows, inputs = [], {}
    for name in queue:
        if (OUT/'runs'/name/'result.json').exists():
            row, array = analyze(name)
            rows.append(row); inputs[name] = array
    pairs = []; paired_readouts = []
    by_name = {row['name']: row for row in rows}
    for row in rows:
        name = row['name']
        if not name.endswith('_high'):
            continue
        partner = name[:-5]+'_interictal'
        if partner in inputs:
            equal = bool(np.array_equal(inputs[name], inputs[partner]))
            pairs.append(dict(high=name, interictal=partner, identical_future_input_records=equal))
            assert equal, (name, partner)
            paired_readouts.append(compare_readouts(row, by_name[partner]))
    common_inputs = []
    if inputs:
        anchor = next(iter(inputs))
        for name, value in inputs.items():
            equal = bool(np.array_equal(inputs[anchor], value))
            common_inputs.append(dict(reference=anchor, name=name,
                                      identical_future_input_records=equal))
            assert equal, (anchor, name)
    summary = dict(completed=len(rows), total=len(queue), rows=rows, paired_input_checks=pairs,
                   paired_history_readout_comparisons=paired_readouts,
                   common_future_input_checks=common_inputs,
                   stage='COMPLETE_FINITE_WINDOW' if len(rows)==len(queue) else 'RUNNING_PARTIAL',
                   bifurcation_status='NOT_ESTABLISHED', human_review='PENDING')
    (OUT/'conditional_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    lines = ['# 原生 Z–K 条件切片', '', '每格两种历史，共用完整空间Z/K场及未来外源输入。这里只报告30秒有限窗响应，不命名分岔；短事件还需空间传播和接触读出验收。', '',
             '|Z|K/gL|初始历史|末10秒状态|高态占比|短事件数|Z自然漂移/s|K自然漂移/s|',
             '|---|---|---|---|---|---|---|---|']
    for row in rows:
        j=row['job'];d=row['counterfactual_drift_mean_allE_A_B_other'][0]
        lines.append(f"|{j['target_Z']}|{j['target_K']}|{row['name'].rsplit('_',1)[-1]}|{row['finite_window_state']}|{row['tail_high_fraction']:.3f}|{row['tail_brief_events']}|{d[0]:.5f}|{d[1]:.5f}|")
    lines += ['', '## 两核活动与资源恢复方向', '',
              '全网平均Z向上不保证活跃核也在恢复。这里并列同一末10秒的两核平均率和反事实Z漂移；完整区域恢复条件比例保存在JSON，由原方程和已固定的Z场反解，不能用低放电率代替输入负荷判定。', '',
              '|条件与历史|Core A均率Hz|Core B均率Hz|全E dZ/s|Core A dZ/s|Core B dZ/s|',
              '|---|---|---|---|---|---|']
    for row in rows:
        d = np.asarray(row['counterfactual_drift_mean_allE_A_B_other'])[:, 0]
        lines.append(f"|{row['name']}|{row['tail_mean_Hz'][1]:.2f}|{row['tail_mean_Hz'][2]:.2f}|{d[0]:.5f}|{d[1]:.5f}|{d[2]:.5f}|")
    lines += ['', '## 两种历史的末段读出', '',
              '差值为间期历史减高态历史，各用末10秒；空间差为400原生格点平均率的绝对差，按E细胞数加权。它描述活动位置和强度，不测传播方向；相同分类不能证明收敛到相同吸引子。逐触点电流差保存在JSON，仍是电流代理，不是HFO能量。', '',
              '|条件|粗分类相同|全E率差Hz|Core A率差Hz|Core B率差Hz|核外率差Hz|空间平均率绝对差Hz|',
              '|---|---|---|---|---|---|---|']
    for pair in paired_readouts:
        delta = pair['mean_rate_delta_allE_A_B_other_Hz']
        lines.append('|'+pair['reference'].rsplit('_',1)[0]+'|'+str(pair['same_coarse_state_label'])+'|'+
                     '|'.join(f'{value:.3f}' for value in delta)+'|'+
                     f"{pair['cell_weighted_absolute_mean_field_difference_Hz']:.3f}|")
    (OUT/'conditional_table.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(completed=len(rows), total=len(queue), paired_inputs_checked=len(pairs))))


if __name__ == '__main__':
    main()
