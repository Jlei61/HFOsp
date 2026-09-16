#!/usr/bin/env python3
"""Observe the actual revised-Z drift; no closure or nullcline is inferred."""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import analyze_topic4_autonomous_recovery as common


def draw(root, name):
    folder = root / 'runs' / name
    job = common.read(root / 'jobs' / (name + '.json'))
    a = common.load(folder, ['time_ms', 'spikes_1ms', 'regions_1ms',
                            'slow_time_ms', 'Z', 'M'])
    parts = {}
    for path in sorted((folder / 'resource_chunks').glob('*.npz')):
        if '.tmp.' in path.name:
            continue
        with np.load(path) as q:
            for key in q.files:
                parts.setdefault(key, []).append(q[key])
    drift = {key: np.concatenate(values) for key, values in parts.items()}
    assert np.array_equal(a['slow_time_ms'], drift['time_ms'])
    native = (1 - a['Z'][:, 8] - a['Z'][:, 0]) / job['tau_Z_s']
    error = float(np.max(abs(native - drift['native_derivative_per_s'])))
    assert error < 1e-12, error
    assert np.allclose(drift['native_derivative_per_s'] + drift['extra_recovery_per_s'],
                       drift['net_derivative_per_s'], atol=1e-13, rtol=0)
    assert np.all(drift['extra_recovery_per_s'] >= 0)
    geo = np.load(root / 'geometry.npz')
    t = a['time_ms'] / 1000
    ts = a['slow_time_ms'] / 1000
    rates = np.column_stack([a['spikes_1ms'][:, 0] / 32,
                            a['regions_1ms'][:, :2] * 1000 / geo['region_counts'][:2]])
    plt.rcParams.update({'font.size': 16, 'axes.labelsize': 19,
                         'xtick.labelsize': 16, 'ytick.labelsize': 16})
    fig, axes = plt.subplots(4, 1, figsize=(16, 13), sharex=True,
                             gridspec_kw={'hspace': .22})
    for i, (color, label) in enumerate([('#333333', 'All E'),
                                       ('#bd6fa2', 'Core A E'), ('#438ec0', 'Core B E')]):
        axes[0].plot(t, gaussian_filter1d(rates[:, i], 3), c=color, lw=.8, label=label)
    axes[0].set_ylabel('Rate (Hz)')
    axes[0].legend(loc='upper left', fontsize=12, ncol=3, framealpha=.95)
    for i, color, label in [(0, '#743895', 'All E'), (5, '#bd6fa2', 'Core A'),
                             (6, '#438ec0', 'Core B')]:
        axes[1].plot(ts, a['Z'][:, i], c=color, lw=1.4, label=label)
    axes[1].set_ylabel('Resource Z')
    axes[1].set_ylim(0, 1.04)
    for key, color, label in [('native_derivative_per_s', '#b85a4d', 'Original net drift'),
                              ('extra_recovery_per_s', '#357f7b', 'Added recovery'),
                              ('net_derivative_per_s', '#333333', 'Total drift')]:
        y = drift[key]
        axes[2].plot(ts, y, c=color, lw=.5, alpha=.15)
        axes[2].plot(ts, gaussian_filter1d(y, 2.5), c=color, lw=1.3, label=label)
    axes[2].axhline(0, c='.5', ls=':', lw=.8)
    axes[2].set_ylabel('Mean dZ/dt (s⁻¹)')
    axes[2].legend(loc='upper left', fontsize=12, ncol=3, framealpha=.95)
    axes[3].plot(ts, a['Z'][:, 8], c='#ad743b', lw=.7)
    axes[3].set_ylabel('E fraction\nJ ≥ I threshold')
    axes[3].set_ylim(-.03, 1.03)
    axes[3].set_xlabel('Time (s)')
    for i, ax in enumerate(axes):
        ax.set_xlim(0, t[-1])
        ax.spines[['top', 'right']].set_visible(False)
        ax.text(-.1, 1.02, 'ABCD'[i], transform=ax.transAxes, fontsize=23, weight='bold')
    fig.subplots_adjust(left=.14, right=.97, top=.97, bottom=.08)
    out = folder / 'figures'
    out.mkdir(exist_ok=True)
    for ext in ['png', 'pdf']:
        fig.savefig(out / f'resource_balance.{ext}', dpi=170)
    plt.close(fig)
    common.write(folder / 'resource_balance_plot.json', dict(
        job=job, observed_s=float(t[-1]), original_drift_identity_error=error,
        observable='Actual neuronwise drift, averaged over E; original and added terms evaluated at the same pre-step Zi and delivered Ji.',
        samples_s=.02, drift_display_gaussian_sigma_s=.05, rate_display_gaussian_sigma_s=.003,
        limitation='Faint lines are saved instantaneous derivatives; bold drift is a display smoothing. Sparse samples are not an exact time integral. The original mean-drift identity is exact at sampled states; the added term uses actual joint Zi/Ji, not a product of means. This is not a reduced autonomous vector field or a nullcline.',
        human_review='PENDING'))
    with (out / 'README.md').open('a') as f:
        f.write('\n### resource_balance.png / .pdf\n连续核内与全局放电、Z、实际原生净变化项和新增恢复项，以及超过耗竭阈值的E细胞比例在同一时间轴对齐。变化项直接来自逐细胞方程，浅线为20ms采样，实线仅作50ms宽度的显示平滑。**关注点**：增加资源补充可能预防进入，也可能帮助退出；这里不把平均轨迹当成nullcline或独立闭合的动力系统。\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=common.OUT / 'continuous_resource_recovery_round5')
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    draw(args.root, args.name)
