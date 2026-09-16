#!/usr/bin/env python3
"""Compare actual shared-noise reset arms over a fixed saved50s interval.

This observational prefix leaves all original finite followups unchanged.
"""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/topic4_sef_hfo'
WINDOW = BASE / 'fig5_m_overnight_exploration_20260913'
OUT = WINDOW / 'reset_matched_50s'
PREFIX = BASE / 'fig5_manual_core_release_v1/weak_fast_z_refill_recurrence_v2/runs/weak_fast_z_refill_recurrence.npz'
START, END = 76.5, 126.5


def read(path):
    return json.loads(path.read_text())


def continuation(folder):
    keys = ['time_ms', 'spikes_1ms', 'raster', 'slow_time_ms', 'Z', 'M', 'inputs']
    pieces = {key: [] for key in keys}
    expected = round(START * 10000)
    paths, digests = [], []
    for path in sorted((folder / 'chunks').glob('*.npz')):
        lo, hi = map(int, path.stem.split('_'))
        if lo >= round(END * 10000):
            break
        assert lo == expected and hi <= round(END * 10000)
        with np.load(path) as data:
            assert int(data['start_step']) == lo and int(data['end_step']) == hi
            for key in keys:
                pieces[key].append(data[key])
            assert np.array_equal(data['spikes_1ms'][:, 0], data['regions_1ms'][:, :3].sum(1))
            assert np.array_equal(data['spikes_1ms'][:, 1], data['regions_1ms'][:, 3:].sum(1))
            digests.append(str(data['input_digest']))
        paths.append(str(path))
        expected = hi
    assert expected == round(END * 10000), 'Required fixed50s prefix not yet committed'
    data = {key: np.concatenate(values) for key, values in pieces.items()}
    data.update(source=paths, input_digests=digests)
    return data


def control(sample_indices):
    low, high = round(START * 10000), round(END * 10000)
    with np.load(PREFIX) as source:
        counts = np.empty(((high-low)//10, 2), np.uint16)
        for col, key, n in [(0, 'rate_e_hz', 32000), (1, 'rate_i_hz', 8000)]:
            counts[:, col] = np.rint(source[key][low:high].reshape(-1, 10).mean(1) * n * .001).astype(np.uint16)
        raw_inputs = source['input_summary']
        data = dict(time_ms=np.arange(low//10, high//10)+.5,
            spikes_1ms=counts, raster=source['sample_spikes'][low:high, sample_indices],
            slow_time_ms=source['z_time_ms'][low//50:high//50],
            Z=source['z_stats'][low//50:high//50, :9],
            M=source['m_stats'][low//50:high//50][:, [0, 5, 6, 7]],
            inputs=raw_inputs[(raw_inputs[:, 0] >= START*1000) & (raw_inputs[:, 0] < END*1000)],
            source=[str(PREFIX)])
        assert np.array_equal(counts[:, 0], source['region_spikes_1ms'][low//10:high//10, :3].sum(1))
    return data


def summary(data):
    rate10 = data['spikes_1ms'].reshape(-1, 10, 2).sum(1) / np.array([32000, 8000]) / .01
    time10 = data['time_ms'].reshape(-1, 10).mean(1)/1000-START
    slowtime = data['slow_time_ms']/1000-START
    high = rate10[:, 0] >= 200
    edges = np.diff(np.r_[False, high, False].astype(int))
    spans = (np.flatnonzero(edges == -1)-np.flatnonzero(edges == 1))*.01
    windows = []
    for lo, hi in [(0., 2.), (2., 10.), (10., 50.)]:
        r = (time10 >= lo) & (time10 < hi)
        s = (slowtime >= lo) & (slowtime < hi)
        windows.append(dict(after_release_s=[lo, hi], mean_E_Hz=float(rate10[r, 0].mean()),
            mean_I_Hz=float(rate10[r, 1].mean()), quiet_E_10ms_fraction=float((rate10[r, 0]<5).mean()),
            mean_Z=float(data['Z'][s, 0].mean()),
            mean_applied_M=float((.02*data['M'][s, 0]).mean()),
            mean_GABA_above_depletion_fraction=float(data['Z'][s, 8].mean()),
            mean_Z_drift_per_s=float(((1-data['Z'][s, 8]-data['Z'][s, 0])/5).mean())))
    return dict(windows=windows, max_contiguous_global_high_s=float(max(spans, default=0)),
        global_high200Hz_for200ms_observed=bool(np.any(spans >= .2)),
        initial_applied_M=float(.02*data['M'][0, 0]), last_applied_M=float(.02*data['M'][-1, 0]),
        last_Z=float(data['Z'][-1, 0]), observation_after_release_s=END-START)


def main():
    with np.load(BASE / 'reset_state_diagnosis_20260911/geometry.npz') as geo:
        indices=geo['sample_source_indices']; assert len(indices) == 80
    arms = [control(indices),
            continuation(BASE / 'reset_state_diagnosis_20260911/runs/z_m_reset_long'),
            continuation(WINDOW / 'fast_state_pilot/runs/all_fast_90s')]
    names = ['Z only', 'Z + M clear', 'Z + M + fast-state clear']
    identifiers = ['Z_only', 'Z_M_clear', 'Z_M_fast_clear']
    for data in arms:
        assert data['raster'].shape == (500000, 80)
        assert np.array_equal(data['time_ms'], arms[0]['time_ms'])
        assert np.array_equal(data['slow_time_ms'], arms[0]['slow_time_ms'])
        assert np.array_equal(data['inputs'], arms[0]['inputs'])
    assert arms[1]['input_digests'] == arms[2]['input_digests']
    rows = {name: summary(data) for name, data in zip(identifiers, arms)}
    # Actual M update is forward Euler, then +1 for a spike. This is the
    # homogeneous contribution of M already present at release along this
    # observed trajectory, not a new counterfactual network simulation.
    slow_steps = np.rint((arms[0]['slow_time_ms']-START*1000)/.1).astype(np.int64)
    carried = .02*arms[0]['M'][0,0]*np.exp(slow_steps*np.log1p(-.1/2000))
    measured = .02*arms[0]['M'][:,0]
    assert np.all(carried <= measured+1e-10)
    direct_memory = [dict(after_release_s=float(slow_steps[k]*.0001),
        carried_initial_M_current=float(carried[k]), observed_M_current=float(measured[k]),
        fraction_of_observed_M=float(carried[k]/measured[k])) for k in [0, 2000, 4000, len(slow_steps)-1]]
    OUT.mkdir(exist_ok=True)
    result = dict(status='MATCHED_SAVED50S_PREFIX', absolute_window_s=[START, END],
        fixed_parameter=dict(eta_M=.02, tau_M_s=2., tau_Z_s=5., seed=9108401),
        intervention_time_s=76.5, Z_refill_s=[75.5, 76.5],
        M_clear='Once at76.5; all arms then evolve native M and Z.',
        fast_clear='V/refractory, AMPA/GABA and delay rings only; OU state and RNG retained.',
        sources={name: data['source'] for name, data in zip(identifiers, arms)},
        external_input_summary_bitwise_equal_all_arms=True,
        full_input_vector100ms_digests_equal_between_M_and_fast_clear=True,
        statistical_unit='One shared release-state and future-noise realization, under three interventions.',
        independent_sample_increment=0, original_followup_truncated=False,
        full_followup_complete=False, permanent_nonentry_claim=False, metrics=rows,
        original_M_homogeneous_contribution=direct_memory,
        M_update='m[k+1]=(1-0.1/2000)*m[k]+spike[k]; per E cell, M on throughout.',
        agent_visual_review='PENDING', human_review='PENDING')
    (OUT/'analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    plt.rcParams.update({'font.size':13, 'axes.labelsize':15, 'axes.titlesize':16,
                         'pdf.fonttype':42, 'axes.spines.top':False})
    fig, axes = plt.subplots(4, 3, figsize=(18, 12), sharex=True, sharey='row',
        gridspec_kw={'height_ratios':[1, 1.8, 1, 1], 'hspace':.13, 'wspace':.14})
    rate_upper = max(float((data['spikes_1ms'].reshape(-1,5,2).sum(1)
                           / np.array([32000,8000]) / .005).max()) for data in arms)
    rate_upper = np.ceil(rate_upper / 100) * 100
    m_upper = np.ceil(max(float((.02*data['M'][:,0]).max()) for data in arms))
    for col, (name, data) in enumerate(zip(names, arms)):
        times = data['time_ms'].reshape(-1, 5).mean(1)/1000-START
        rates = data['spikes_1ms'].reshape(-1, 5, 2).sum(1)/np.array([32000, 8000])/.005
        for i, color in enumerate(['#287aaf', '#d88431']):
            axes[0, col].plot(times, rates[:, i], color=color, lw=.55, label=['E','I'][i])
        axes[0, col].set_title(name)
        axes[0, col].set_ylim(0, rate_upper)
        t, neuron = np.nonzero(data['raster'])
        for mask, color in [(neuron < 60, '#287aaf'), (neuron >= 60, '#d88431')]:
            axes[1, col].scatter(t[mask]*.0001, neuron[mask], s=.25, c=color, marker='.', rasterized=True)
        axes[1, col].set(ylim=(-1, 80), yticks=[9.5,29.5,49.5,69.5])
        if col == 0:
            axes[1, col].set_yticklabels(['Core A E','Core B E','Other E','I'])
        for y in [19.5,39.5,59.5]: axes[1, col].axhline(y, c='#cccccc', lw=.5)
        slowtime = data['slow_time_ms']/1000-START
        axes[2, col].plot(slowtime, data['Z'][:,0], color='#78468d', lw=1.5)
        axes[2, col].fill_between(slowtime, data['Z'][:,2], data['Z'][:,4], color='#78468d', alpha=.15)
        axes[2, col].set_ylim(.65, 1.02)
        axes[3, col].plot(slowtime, .02*data['M'][:,0], color='#aa6323', lw=1.5)
        axes[3, col].set(xlabel='Time after Z release (s)', xlim=(0,50), xticks=[0,10,20,30,40,50], ylim=(0,m_upper))
    axes[0,0].set_ylabel('Population rate (Hz)'); axes[0,0].legend(frameon=False, ncol=2, loc='upper right')
    axes[2,0].set_ylabel('Mean Z / 10–90%')
    axes[3,0].set_ylabel('Applied M current\n(mV equiv.)')
    fig.subplots_adjust(left=.095, right=.985, bottom=.08, top=.95)
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/'matched_reset_prefix.png', dpi=170, bbox_inches='tight')
    fig.savefig(folder/'matched_reset_prefix.pdf', bbox_inches='tight');plt.close(fig)
    (folder/'README.md').write_text('### matched_reset_prefix.png / .pdf\n'
        '三列是同一76.5秒释放状态、同一后续噪声下的Z-only、额外M清零和额外快状态清零对照。统一展示实际保存的释放后50秒：E/I率、固定80神经元raster、Z分布与M电流，所有参数保持弱快M工作点。\n'
        '**关注点**：这是配对的短程状态干预读出，原1000秒/90秒随访继续；短窗没有高态不证明永久不能再进入，三列也不是三份独立噪声样本。\n')
    lines=['# 同噪声的reset状态干预：固定50秒前缀', '',
        '问题是清除M或快状态是否只改变释放后的短暂行为，还是在当前观察窗内改变了平均活动和Z收支；并非提前判定长程再进入。三臂的输入摘要逐位相同，M清零与全快状态清零两臂另外核对了每100ms完整输入向量摘要。', '',
        '| 条件 | 释放后区间(s) | E均率(Hz) | M电流 | 平均Z | 静默比例 |', '|---|---|---:|---:|---:|---:|']
    for name, row in rows.items():
        for v in row['windows']:
            lines.append(f'| {name} | {v["after_release_s"]} | {v["mean_E_Hz"]:.3f} | {v["mean_applied_M"]:.4f} | {v["mean_Z"]:.4f} | {v["quiet_E_10ms_fraction"]:.1%} |')
    a, b, c = [rows[key]['windows'] for key in identifiers]
    lines += ['',
        f'在此配对噪声实现，保留M时释放后前2秒E均率{a[0]["mean_E_Hz"]:.2f}Hz，'
        f'M清零后为{b[0]["mean_E_Hz"]:.2f}Hz；全快状态额外清零后为{c[0]["mean_E_Hz"]:.2f}Hz。'
        '所以既有M确实延缓了初期有限活动的恢复。',
        f'释放后10–50秒三臂E均率分别为{a[2]["mean_E_Hz"]:.2f}、{b[2]["mean_E_Hz"]:.2f}、{c[2]["mean_E_Hz"]:.2f}Hz，'
        f'平均Z分别为{a[2]["mean_Z"]:.4f}、{b[2]["mean_Z"]:.4f}、{c[2]["mean_Z"]:.4f}。'
        '此时平均M电流也接近，仍有反复有限活动；短暂遗留M的作用已不能直接等同于解释长期不再runaway。'
        '目前三臂均未在这50秒满足持续200毫秒的全局高态判据。']
    lines += ['', '时间窗固定为释放后0–2、2–10和10–50秒；每个时间点、神经元和事件不作为独立实验样本。相近的均值不证明完整状态或再次进入概率相同；后续长程阴性/阳性仍须各自随访。',
        'Z-only的1000秒结果另外已经完成；此图仅为保证三个干预具有相同观察长度而截取前缀，不改变任何原仿真终点。']
    v=direct_memory[2]
    lines += ['',
        '代码中M每0.1毫秒按m←(1−0.1/2000)m衰减，再为发放细胞加1。'
        f'按这条实际离散方程，释放时已有M在20秒后的直接电流贡献为{v["carried_initial_M_current"]:.6g}，'
        f'只占该时刻实测M电流的{v["fraction_of_observed_M"]:.3%}；其余由释放后的新放电补充。'
        '这是给定实际轨迹的M组成分解，并非把整个网络重新仿真成没有历史M。'
        '旧M通过早期放电影响Z、突触和后续状态的间接效应仍可能存在，不能由此排除。']
    (OUT/'scientific_review.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(rows, ensure_ascii=False))


if __name__ == '__main__':
    main()
