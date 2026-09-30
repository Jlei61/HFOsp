#!/usr/bin/env python3
"""Existing native evidence: Z balance, the waiting interval, and paired G kinetics.

Unit: a trajectory or a paired seed, never a recorded time sample. This reads
completed simulations; it neither launches trials nor certifies bifurcations.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import hashlib
import json
from pathlib import Path
import numpy as np

SOURCE = Path('/data/hfosp/topic4_sef_hfo/fig5_global_feedback_response_20260923')
OUT = Path('/data/hfosp/topic4_sef_hfo/fig5_autonomous_loop_zk_20260924/recovery_mechanism')
NAME = 'G30_response0.5_s9108405'


def read(path):
    return json.loads(path.read_text())


def load(name, directory, keys):
    parts = {key: [] for key in keys}
    for path in sorted((SOURCE / 'runs' / name / directory).glob('*.npz')):
        if '.tmp.' in path.name:
            continue
        with np.load(path) as data:
            for key in keys:
                parts[key].append(data[key])
    return {key: np.concatenate(value) for key, value in parts.items()}


def at(times, value):
    return max(0, int(np.searchsorted(times, value, side='right') - 1))


def first_sustained(times, mask, minimum_s, dt):
    padded = np.r_[False, mask, False]
    starts = np.flatnonzero(np.diff(padded.astype(int)) == 1)
    ends = np.flatnonzero(np.diff(padded.astype(int)) == -1)
    good = starts[(ends - starts) * dt >= minimum_s - 1e-9]
    return float(times[good[0]]) if len(good) else None


def source_budget():
    analysis = read(SOURCE / 'analysis' / f'{NAME}.json')
    spatial = read(SOURCE / 'spatial_review' / NAME / 'review.json')
    reviewed = {group['label']: group['events']['brief_events'] for group in spatial['groups']}
    assert np.array_equal(analysis['absolute_Z_recovery']['reference_core_Z'],
                          spatial['common_current_graph_core_Z_reference'])
    job = analysis['job']
    data = load(NAME, 'chunks', ['slow_time_ms', 'Z'])
    budget = load(NAME, 'z_budget_chunks', ['time_ms', 'values'])
    feedback = load(NAME, 'feedback_chunks', ['time_ms', 'K_mean', 'G_raw', 'R_global_Hz'])
    regional = load(NAME, 'regional_chunks', ['time_ms', 'values'])
    response = load(NAME, 'global_response_chunks', ['time_ms', 'q', 'G_raw'])
    rate = load(NAME, 'mechanism_chunks', ['time_ms', 'global_E_rate_Hz'])
    with np.load(SOURCE / 'geometry.npz') as geometry:
        weights = geometry['region_counts'][:3] / 32000
    tz, tb = data['slow_time_ms'] / 1000, budget['time_ms'] / 1000
    tf, tr = feedback['time_ms'] / 1000, regional['time_ms'] / 1000
    assert np.array_equal(tz, tf) and np.array_equal(tf, tr)
    assert np.allclose(np.diff(tb), .02)
    values = budget['values']
    assert np.max(np.abs(values[:, :, 5])) < 1e-11
    # Consecutive budgets must describe the same carried native Z state.
    assert np.max(np.abs(values[1:, :, 0] - values[:-1, :, 1])) < 1e-11
    limit = job['threshold'] / (18 - job['global_reversal_mV'])

    def state(time):
        i = at(tf, time)
        currents = regional['values'][i].T @ weights
        return dict(recorded_time_s=float(tf[i]), mean_Z=float(data['Z'][i, 0]),
            core_Z=data['Z'][i, [5, 6]].tolist(), mean_K=float(feedback['K_mean'][i]),
            raw_G=float(feedback['G_raw'][i]), K_current_mV=float(currents[3]),
            G_current_mV=float(currents[4]), M_current_mV=float(currents[7]))

    rows = []
    for number, episode in enumerate(analysis['absolute_Z_recovery']['episodes'], 1):
        start, end = episode['exit_start_s'], episode['window_end_s']
        events = (episode.get('post_absolute_recovery_events') or {}).get('brief_events', [])
        first_event = events[0]['start_s'] if events else None
        if f'post_exit{number}' in reviewed:
            current_events = reviewed[f'post_exit{number}']
            assert first_event == (current_events[0]['start_s'] if current_events else None)
        recovered = episode['first_both_at_reference_s']
        stop = first_event if first_event is not None else end
        mask = (tb - .02 >= start - 1e-9) & (tb <= stop + 1e-9)
        selected = values[mask]
        assert len(selected)
        gain = selected[:, :, 2].sum(0) * .02
        loss = selected[:, :, 3].sum(0) * .02
        delta = selected[-1, :, 1] - selected[0, :, 0]
        assert np.max(np.abs(delta - gain + loss)) < 1e-10
        net_mask = (tb >= start) & (tb < end) & np.all(values[:, 1:3, 4] > 0, axis=1)
        positive = first_sustained(tb, net_mask, 1., .02)
        gtime = response['time_ms'] / 1000
        below = np.flatnonzero((gtime >= start) & (gtime < end) & (response['G_raw'] < limit))
        rows.append(dict(exit_number=number, exit_observer_s=start, window_end_s=end,
            first_G_below_global_recovery_block_s=float(gtime[below[0]]) if len(below) else None,
            first_core_net_positive_1s_budget_endpoint_s=positive,
            first_both_Z_at_reference_s=recovered, first_post_recovery_brief_s=first_event,
            delay_Z_reference_to_brief_s=first_event - recovered if first_event is not None and recovered is not None else None,
            sustained_return_screen=episode['sustained_return_after_absolute_recovery'],
            state_at_Z_reference=state(recovered) if recovered is not None else None,
            state_before_first_brief=state(first_event) if first_event is not None else None,
            exact_budget_window_s=[float(tb[mask][0] - .02), float(tb[mask][-1])],
            accumulated_recovery=gain.tolist(), accumulated_consumption=loss.tolist(),
            observed_Z_change=delta.tolist(), balance_error=(delta - gain + loss).tolist(),
            region_order=['all_E', 'core_A', 'core_B', 'other_E']))

    # Diagnostic window lies inside the first observed low interval. Predict
    # from its first native state using the fixed equations; do not fit tau.
    mask = (tf >= 20) & (tf <= 40)
    elapsed = tf[mask] - tf[mask][0]
    kp = feedback['K_mean'][mask][0] * np.exp(-elapsed / job['off_tau_s'])
    zp = 1 - (1 - data['Z'][mask][0, 0]) * (1 - .0001 / job['tau_Z_s']) ** np.rint(elapsed / .0001)
    qmask = (response['time_ms'] >= 20000) & (response['time_ms'] <= 40000)
    rmask = (rate['time_ms'] >= 20000) & (rate['time_ms'] <= 40000)
    quiet_law = dict(window_s=[20, 40], exploratory_window=True,
        largest_recorded_q=float(response['q'][qmask].max()),
        largest_recorded_causal_R_Hz=float(rate['global_E_rate_Hz'][rmask].max()),
        fixed_K_decay_tau_s=job['off_tau_s'], K_exponential_max_abs_error=float(np.max(np.abs(kp - feedback['K_mean'][mask]))),
        Z_ideal_all_cells_eligible_upper_bound_max_deficit=float(np.max(zp - data['Z'][mask, 0])),
        Z_above_ideal_upper_bound_max=float(np.max(data['Z'][mask, 0] - zp)),
        interpretation='K exponential is a consequence of the fixed low-rate law, not a fitted recovery timer. The Z formula is an all-eligible upper bound; recurrent inhibition can make actual Z smaller. Neither predicts the first stochastic returned event independently.')
    assert quiet_law['largest_recorded_q'] == 0
    assert quiet_law['K_exponential_max_abs_error'] < 1e-8
    assert quiet_law['Z_above_ideal_upper_bound_max'] < 1e-8
    return dict(source=NAME, episodes=rows, quiet_law=quiet_law,
        first_brief_identity_matches_current_spatial_review=True,
        global_raw_G_block_threshold=limit,
        threshold_meaning='When rawG*(18-EG)>=Ith, nonnegative local II makes all E recovery targets zero. Falling below this bound permits recovery but does not guarantee it; native cell-specific II still matters.',
        native_budget_max_abs_error=float(np.max(np.abs(values[:, :, 5]))),
        reference_core_Z=analysis['absolute_Z_recovery']['reference_core_Z'],
        timing='Budget records integrate20ms; net-positive time is a budget endpoint, G crossing uses1ms records, Z and K state use last20ms sample at or before landmark.')


def paired_kinetics():
    rows = []
    for seed in [9108402, 9108403]:
        names = [f'G30_response{tau}_s{seed}' for tau in ['0', '0.5']]
        results = [read(SOURCE / 'runs' / name / 'result.json') for name in names]
        assert results[0]['identity'] == results[1]['identity']
        jobs = [result['job'] for result in results]
        differences = {key for key in jobs[0].keys() | jobs[1].keys() if jobs[0].get(key) != jobs[1].get(key)}
        assert differences == {'name', 'global_tau_s'}, differences
        inputs = [load(name, 'chunks', ['inputs'])['inputs'] for name in names]
        assert np.array_equal(*inputs)
        pair = []
        for name, result in zip(names, results):
            assert result['status'] == 'COMPLETE' and result['end_s'] == 120
            preservation = read(SOURCE / 'runs' / name / 'rhythm_preservation.json')
            assert preservation['status'] == 'PASS'
            analysis = read(SOURCE / 'analysis' / f'{name}.json')
            data = load(name, 'feedback_chunks', ['time_ms', 'R_global_Hz', 'K_mean', 'G_raw'])
            z = load(name, 'chunks', ['slow_time_ms', 'Z'])
            causal = load(name, 'mechanism_chunks', ['time_ms', 'global_E_rate_Hz'])
            onset = analysis['primary']['entries'][0]['onset_s']
            after = causal['time_ms'] >= onset * 1000
            retained = after & (causal['global_E_rate_Hz'] <= result['job']['retention_below_Hz'])
            edges = np.diff(np.r_[False, retained, False].astype(int))
            lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
            tail = data['time_ms'] >= 110000
            pair.append(dict(name=name, global_tau_s=result['job']['global_tau_s'],
                entry_count=len(analysis['primary']['entries']), exit_count=len(analysis['primary']['low_activity_exits']),
                full_return_count=sum(e['sustained_return_after_absolute_recovery'] for e in analysis['absolute_Z_recovery']['episodes']),
                minimum_causal_R_after_entry_Hz=float(causal['global_E_rate_Hz'][after].min()),
                sampled_K_retention_time_after_entry_s=float(retained.sum() * .001),
                longest_sampled_K_retention_span_after_entry_s=float(lengths.max() * .001) if len(lengths) else 0.,
                retention_observation='1ms samples of the preset causal R<=5Hz rule; approximate duration, not a manual quiet window.',
                tail110_120s_R_range_Hz=[float(data['R_global_Hz'][tail].min()), float(data['R_global_Hz'][tail].max())],
                tail110_120s_mean_K=float(data['K_mean'][tail].mean()),
                tail110_120s_mean_Z=float(z['Z'][z['slow_time_ms'] >= 110000, 0].mean())))
        rows.append(dict(seed=seed, job_differences=sorted(differences), same_network_identity=True,
            exact_full_future_input_records=True, conditions=pair))
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    source, paired = source_budget(), paired_kinetics()
    result = dict(status='COMPLETE_EXISTING_DATA_ANALYSIS', source_budget=source, paired_G_kinetics=paired,
        statistical_unit='One selected240s trajectory for the recovery sequence; two paired120s seeds for the sole tauG change. Episodes/samples nested within trajectory.',
        interpretation='Z recovery and event return are separate observations. Matched tauG controls support a role for feedback kinetics in this fixed model; the quiet-interval K decay is equation-consistent. These do not identify an SN/Hopf transition or prove K alone sets return timing.',
        producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), new_simulations=0)
    (OUT / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# 已有轨迹中的恢复机制', '',
        '统计单位是轨迹；同一轨迹里的多次退出不能当成独立种子。下表把两核Z恢复到原间期参考，与之后首个短事件分开。', '',
        '|退出序号|退出观察点/s|两核Z回到参考/s|首个短事件/s|二者间隔/s|持续返回筛查|',
        '|---|---:|---:|---:|---:|---|']
    def fmt(value):
        return '未观察到' if value is None else f'{value:.2f}'
    for row in source['episodes']:
        lines.append(f'|{row["exit_number"]}|{fmt(row["exit_observer_s"])}|{fmt(row["first_both_Z_at_reference_s"])}|{fmt(row["first_post_recovery_brief_s"])}|{fmt(row["delay_Z_reference_to_brief_s"])}|{row["sustained_return_screen"]}|')
    first = source['episodes'][0]
    recovered = first['state_at_Z_reference']
    returned = first['state_before_first_brief']
    lines += ['',
        f'第一轮约17.6秒开始持续净恢复，23.90秒两核已回到参考，但首个满足20–200ms标准的完整短事件在49.31秒才出现；时刻已与当前原生空间审阅逐一核对。恢复参考时K/gL仍约{recovered["mean_K"]:.2f}，全E平均K电流约{recovered["K_current_mV"]:.1f}mV-equiv，而G和M的电流已很小；首个短事件前K/gL约{returned["mean_K"]:.3f}。因此这段等待不是Z始终未恢复，记录显示K尾部继续存在；未据此把K指定为唯一的事件触发量。',
        '四个完整返回中，Z回到参考后至首个完整短事件仍间隔约23–25秒。第三次退出后的首个短事件较弱且非强核事件，强核事件稍后出现；参考Z不是一个已认证的分岔阈值。']
    lines += ['', 'Z逐步收支完整闭合；恢复项与消耗项均来自原生每个积分步累计，未用平滑后的曲线反推。全局分流仍受Z门控，其资源负荷也会暂时压住Z恢复；在本模型中rawG大于约2.67时，仅新增负荷已足以令E细胞恢复目标为零。低于此值仍需看各细胞的局部抑制。', '',
        '首轮20–40秒窗口中，记录的高率门q为零，K与固定5秒指数衰减一致；Z的理想全可恢复公式只是上界。这解释长尾如何延续，不能独立预测第一个随机短事件的时刻，也不能把恢复到参考等同于事件返回。', '',
        '|配对种子|全局响应/s|120秒进入数|退出数|充分Z恢复后返回数|', '|---|---:|---:|---:|---:|']
    for pair in paired:
        for row in pair['conditions']:
            lines.append(f'|{pair["seed"]}|{row["global_tau_s"]}|{row["entry_count"]}|{row["exit_count"]}|{row["full_return_count"]}|')
    lines += ['', '配对job仅name和global_tau_s不同，网络身份及120秒外源输入记录相同，原间期保留门均通过。该对照支持全局反馈响应时间参与转换；不等于已经认证Hopf、鞍结或极限环。没有新增仿真或改变当前队列。', '']
    (OUT / 'scientific_review.md').write_text('\n'.join(lines))
    print(json.dumps(dict(episodes=[{key: row[key] for key in ['exit_number', 'first_both_Z_at_reference_s', 'first_post_recovery_brief_s', 'delay_Z_reference_to_brief_s']} for row in source['episodes']], quiet_law=source['quiet_law'], paired=paired)), flush=True)


if __name__ == '__main__':
    main()
