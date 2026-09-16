#!/usr/bin/env python3
"""Measured gain-by-timescale comparison within the same revised Z equation."""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import analyze_topic4_autonomous_recovery as a


def main():
    root = a.OUT
    destination = root / 'preserved_global_gain_round7'
    sources = [(root / 'continuous_resource_recovery_round5',
                'resource_rho0.25_k50_s9108401')]
    sources += [(destination, name) for name in [
        'resource_rho0.25_k200_tau2_s9108401',
        'resource_rho0.25_k50_tau10_s9108401',
        'resource_rho0.25_k200_tau10_s9108401']]
    records = []
    keys = ['time_ms', 'spikes_1ms', 'regions_1ms', 'slow_time_ms',
            'Z', 'field_time_ms', 'field_5ms']
    for source, name in sources:
        folder = source / 'runs' / name
        data = a.load(folder, keys)
        if data is None or data['time_ms'][-1] < 19999:
            print('WAITING_FOR_COMMON_20S', name)
            return
        job = a.read(source / 'jobs' / (name + '.json'))
        records.append((source, folder, job, data))
    reference = records[0][2]
    invariant = ['mode', 'gamma', 'eta_m', 'tau_M_s', 'tau_Z_s', 'threshold',
                 'seed', 'pool_threshold_Hz', 'recovery_ratio', 'horizon_s']
    for _, _, job, _ in records:
        assert all(job[key] == reference[key] for key in invariant)
    end = min(len(data['time_ms']) for _, _, _, data in records) * .001
    geo = np.load(sources[0][0] / 'geometry.npz')
    nr = geo['region_counts'][:3]
    plt.rcParams.update({'font.size':15, 'axes.labelsize':18,
                         'xtick.labelsize':14, 'ytick.labelsize':14})
    fig, axes = plt.subplots(5, 4, figsize=(23, 18),
        gridspec_kw={'height_ratios':[1.1, 1, 1, 1, 1.15], 'hspace':.24, 'wspace':.22})
    metadata = []
    for col, (source, folder, job, data) in enumerate(records):
        other = np.load(source / 'geometry.npz')
        for key in ['region_counts', 'cell_e_counts', 'centers_mm']:
            assert np.array_equal(geo[key], other[key])
        count = round(end * 1000)
        t = data['time_ms'][:count] / 1000
        rates = np.column_stack([data['spikes_1ms'][:count, 0] / 32,
                                 data['regions_1ms'][:count, :2] * 1000 / nr[:2]])
        for i, (label, color) in enumerate([('All E', '#333333'),
                                           ('Core A E', '#ae5d9c'), ('Core B E', '#347ea0')]):
            axes[0, col].plot(t, gaussian_filter1d(rates[:, i], 3),
                              color=color, lw=.75, label=label)
        axes[0, col].set(ylim=(-5, 505), title=f'κ = {job["pool_gain"]:g}; τG = {job["pool_tau_s"]:g} s')
        ts = data['slow_time_ms'] / 1000
        axes[1, col].plot(ts, data['Z'][:, 0], color='#7e399d', lw=1.4)
        axes[1, col].fill_between(ts, data['Z'][:, 2], data['Z'][:, 4],
                                  color='#7e399d', alpha=.14)
        axes[1, col].set_ylim(0, 1.03)
        pool_keys = ['time_ms', 'rate_Hz', 'raw_global_current', 'effective_global_current']
        chunks = {key: [] for key in pool_keys}
        for path in sorted((folder / 'pool_chunks').glob('*.npz')):
            if '.tmp.' in path.name:
                continue
            with np.load(path) as block:
                for key in pool_keys:
                    chunks[key].append(block[key])
        pool = {key: np.concatenate(values) for key, values in chunks.items()}
        assert np.allclose(np.diff(pool['time_ms']), 5.)
        pt = pool['time_ms'] / 1000
        sel = pt < end
        axes[2, col].plot(pt[sel], pool['rate_Hz'][sel], color='#41654d', lw=1.2)
        axes[2, col].axhline(job['pool_threshold_Hz'], color='.6', ls=':', lw=.9)
        for key, label, color in [('raw_global_current', 'Before Z', '#bc9870'),
                                  ('effective_global_current', 'After Z', '#417f70')]:
            axes[3, col].plot(pt[sel], pool[key][sel], color=color, lw=1.1, label=label)
        for row in range(4):
            axes[row, col].set_xlim(0, end)
            axes[row, col].spines[['top', 'right']].set_visible(False)
            if row < 3:
                axes[row, col].tick_params(labelbottom=False)
            else:
                axes[row, col].set_xlabel('Time (s)')
        ft = data['field_time_ms'] / 1000
        selected = (ft >= end - .05) & (ft < end)
        assert selected.sum() == 10
        field = data['field_5ms'][selected].sum(0) / geo['cell_e_counts'] / .05
        im = axes[4, col].imshow(field.reshape(20, 20), origin='lower',
            extent=[0, 20, 0, 20], cmap='magma', vmin=0, vmax=500)
        for center in geo['centers_mm']:
            axes[4, col].add_patch(plt.Circle(center, 1.5, fill=False, ec='#5fd8dc', lw=1.3))
        axes[4, col].set_xlabel('x (mm)')
        metadata.append(dict(source=str(folder), job=job, matched_window_s=[0, end],
            field_window_s=[end - .05, end], last5s_mean_all_A_B_Hz=rates[-5000:].mean(0).tolist(),
            last5s_quiet10ms_fraction=(rates[-5000:].reshape(500, 10, 3).mean(1) < 5).mean(0).tolist(),
            full_run_complete=(folder / 'result.json').exists()))
    for row in [2, 3]:
        upper = max(ax.get_ylim()[1] for ax in axes[row])
        for ax in axes[row]:
            ax.set_ylim(0, upper)
    for row, label in enumerate(['E rate (Hz)', 'Resource Z', 'Global pool rate (Hz)',
                                  'Global inhibition\n(mV equiv.)', 'y (mm)']):
        axes[row, 0].set_ylabel(label)
        axes[row, 0].text(-.25, 1.04, 'ABCDE'[row], transform=axes[row, 0].transAxes,
                          fontsize=23, weight='bold')
    for row in [0, 3]:
        axes[row, 0].legend(loc='upper right', fontsize=11, framealpha=.9)
    fig.subplots_adjust(left=.08, right=.935, bottom=.065, top=.95)
    cax = fig.add_axes([.953, .07, .012, .13])
    fig.colorbar(im, cax=cax, label='E rate (Hz)')
    figures = destination / 'figures'
    figures.mkdir(exist_ok=True)
    for ext in ['png', 'pdf']:
        fig.savefig(figures / f'preserved_global_comparison.{ext}', dpi=170)
    plt.close(fig)
    a.write(destination / 'preserved_global_comparison.json', dict(updated_at=time.time(),
        records=metadata, comparison='Fixed topology/noise/etaM/tauM/tauZ/Ith/rho.25/r0=50; 2x2 gain50/200 by pool timescale2/10s. Both local and global inhibition are multiplied by Zi and enter the same revisedZ law.',
        note='Matched observed duration, native50ms fields,3ms smoothing for rate display only. These are measured trajectories, not nullclines, fitted vector fields, or independent network replicates.',
        human_review='PENDING'))
    with (figures / 'README.md').open('a') as file:
        file.write('\n### preserved_global_comparison.png / .pdf\n在相同rho=.25方程、手放双核和同一噪声上，比较全局增益50/200与建立时间2/10秒的四种组合。各列使用共同已保存时长，展示全E和双核率、Z、全局池状态、乘Z前后全局输入及末50ms原生场。**关注点**：降低全网均率、维持局部持续态和真正终止双核活动是不同结果；新rho方程的效果不能归为原生ZM已实现自主恢复。\n')
    print('DRAWN_COMMON_WINDOW', end)


if __name__ == '__main__':
    main()
