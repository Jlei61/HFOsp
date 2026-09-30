"""Recompute conditional count diagnostics directly and export a diagnostic plot."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/topic4_sef_hfo/fig5_zm_runaway_mechanism_20260918'
BASE = OUT / 'native_current_memory'


def main():
    native = np.load(BASE / 'millisecond_assay/native_counts.npz')
    source = np.load(OUT / 'native_input_bridge/selected_input_history.npz')
    keep = (source['time_ms'] >= 9000) & (source['time_ms'] < 10350)
    assert np.array_equal(source['spikes'][keep].reshape(1350, 10, 6).sum(1), native['counts'])
    original = np.load(OUT / 'native_input_bridge/membership.npz')
    checks = []; comparison = []; curves = {}
    for case in ['millisecond_assay', 'threshold_phase']:
        folder = BASE / case
        result = json.loads((folder / 'result.json').read_text())
        computed = {}
        for dt in [.1, .05]:
            data = np.load(folder / f'dt{dt:g}.npz')
            for j, g in enumerate(data['groups']):
                N = int(data['group_sizes'][j]); nr = int(data['replicates'][j])
                assert nr % N == 0 and np.max(data['counts'][j, :nr]) <= 1
                if case == 'threshold_phase':
                    th = original['actual_threshold_mv'][original['cell_group'] == g]
                    assert np.array_equal(data['thresholds'][j, :nr], np.tile(th, nr // N))
                paths = data['counts'][j, :nr].astype(float)
                trials = paths.reshape(nr // N, N, 1350).sum(1)
                cut = trials.shape[0] // 2
                if dt == .1:
                    curves[case, j] = trials.mean(0)
                selected = [r for r in result['rows'] if r['group'] == g and r['dt_ms'] == dt]
                for row in selected:
                    b = row['bin_ms']; lo, hi = row['window_ms']; starts = np.arange(1350 // b) * b + 9000
                    take = (starts >= lo) & (starts + b <= hi)
                    batch = trials.reshape(nr // N, 1350 // b, b).sum(2)[:, take]
                    obs = native['counts'][:, j].reshape(1350 // b, b).sum(1)[take].astype(float)
                    mu = batch[:cut].mean(0); variance = batch[:cut].var(0, ddof=1)
                    denominator = max(variance.sum(), 1e-12)
                    x = np.concatenate((obs[None], batch[cut:]), axis=0); d = x - mu
                    metric = row['metric']
                    if metric == 'total_count': v = x.sum(1)
                    elif metric == 'coincidence_sum':
                        # Independent equivalent expression avoids integer overflow.
                        v = (x*x).sum(1) - x.sum(1)
                    elif metric == 'residual_energy': v = np.einsum('ij,ij->i', d, d) / denominator
                    elif metric == 'residual_lag1_crossmoment': v = np.einsum('ij,ij->i', d[:, :-1], d[:, 1:]) / denominator
                    else: raise AssertionError(metric)
                    pred = np.quantile(v[1:], [.025, .5, .975])
                    recorded = [row['observed'], row['predictive_low'], row['predictive_median'], row['predictive_high']]
                    actual = np.r_[v[0], pred]
                    assert np.allclose(actual, recorded, rtol=2e-12, atol=2e-10), (case, row, actual)
                    key = (int(g), b, tuple(row['window_ms']), metric, dt)
                    computed[key] = (float(v[0]), pred)
                if case == 'millisecond_assay':
                    old = np.load(OUT / f'native_input_bridge/local_lif/measured_mean_contrast/dt{dt:g}.npz')
                    assert np.array_equal(paths.reshape(nr, 27, 50).sum(2), old['counts'][j, :nr])
        for row in result['paired']:
            values = [computed[(row['group'], row['bin_ms'], tuple(row['window_ms']), row['metric'], dt)] for dt in [.1, .05]]
            low = all(obs < interval[0] for obs, interval in values)
            high = all(obs > interval[2] for obs, interval in values)
            assert row['side'] == ('low' if low else 'high' if high else None)
            if row['bin_ms'] == 1 and row['window_ms'][0] == 9000:
                comparison.append(dict(case=case, **row))
        checks.append(dict(case=case, rows_reproduced=len(result['rows']), paired_reproduced=len(result['paired'])))
    out = dict(status='INDEPENDENT_COUNT_AUDIT_PASS', checks=checks, original_threshold_weights_exact=True,
               native_1ms_and_reference_50ms_bitwise=True, preentry_1ms=comparison,
               scope='Checks calculations and physical threshold control only. These are conditional Gaussian reference predictions, not a validated autonomous rate field or native confidence intervals.',
               model_promoted=False)
    (BASE / 'phase_independent_audit.json').write_text(json.dumps(out, indent=2)+'\n')
    plt.rcParams.update({'font.size':11, 'pdf.fonttype':42, 'svg.fonttype':'none',
                         'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.8))
    fig.subplots_adjust(left=.075, right=.98, top=.91, bottom=.16, wspace=.24, hspace=.37)
    names = ['Core A E', 'Core B E', 'Surround E', 'I', 'Core A E, low threshold', 'Core B E, low threshold']
    t = (np.arange(84)*5+9002.5)/1000
    for j, ax in enumerate(axes.ravel()):
        N = native['group_sizes'][j]
        scale = 1000/(5*N)
        obs = native['counts'][:420, j].reshape(84, 5).sum(1)*scale
        mean = curves['millisecond_assay', j][:420].reshape(84, 5).sum(1)*scale
        mixture = curves['threshold_phase', j][:420].reshape(84, 5).sum(1)*scale
        ax.plot(t, obs, color='#222222', lw=1.1, label='Native SNN')
        ax.plot(t, mean, color='#c37e24', lw=1.1, label='LIF, mean threshold')
        ax.plot(t, mixture, color='#835ba6', lw=1.1, ls='--', label='LIF, original thresholds')
        ax.text(.5, 1.04, names[j], ha='center', transform=ax.transAxes)
        ax.text(-.18, 1.04, 'ABCDEF'[j], fontweight='bold', fontsize=16, transform=ax.transAxes)
        ax.set_xlim(9, 9.42); ax.set_ylim(bottom=0); ax.set_xticks([9, 9.2, 9.4])
        if j % 3 == 0: ax.set_ylabel('Rate (Hz / neuron)')
        if j >= 3: ax.set_xlabel('Time (s)')
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(.5, .01))
    folder = OUT / 'figures'; stem = 'fig_native_input_threshold_phase'
    for ext in ['png', 'pdf', 'svg']: fig.savefig(folder/f'{stem}.{ext}', dpi=180)
    plt.close(fig)
    meta = dict(source=str(BASE/'phase_independent_audit.json'), groups=source['groups'].tolist(),
                displayed_bins_ms=5, primary_statistics_bins_ms=[1,5,50], interval_ms=[9000,9420],
                reference='Conditional mean response from independent Gaussian local LIF paths; supplied actual group mean, approximate private fluctuations and observed Z. Same-clock, no alignment or fitted shifts.',
                agent_visual_check='PENDING', human_visual_acceptance='PENDING', bifurcation_figure=False)
    (folder/f'{stem}.json').write_text(json.dumps(meta, indent=2)+'\n')
    p = folder/'README.md'; text = p.read_text(); heading = f'### {stem}.png'
    if heading not in text:
        p.write_text(text+'\n'+heading+'\n六格为原先固定的六个局部群体，比较9–9.42秒的原生放电与相同真实群体均值输入下的独立高斯LIF参考；橙色使用平均阈值，紫色保留原始逐细胞阈值。为可读性显示固定5ms计数窗，统计另保留1、5和50ms；没有时间对齐或调整物理参数。它是局部输入近似诊断，不是自主率模型轨迹或分岔图。**关注点**：平均放电量相近也可能掩盖毫秒级同步误差，恢复阈值异质性是否足以解释差异。\n')
    print(json.dumps(checks), flush=True)


if __name__ == '__main__': main()
