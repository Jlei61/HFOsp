#!/usr/bin/env python3
"""Reanalyse only the pre-intervention prefixes of two existing native controls."""
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
from analyze_topic4_autonomous_events import finite_events, temporal_audit


def main():
    root = a.OUT
    old = a.run.ROOT / 'results/topic4_sef_hfo/m_parameter_modes_fig5_20260913/runs'
    identity = a.read(root / 'runs/native_weak_s9108401/result.json')['identity']
    nr = np.load(root / 'geometry.npz')['region_counts']
    plt.rcParams.update({'font.size':15, 'axes.labelsize':18,
                         'xtick.labelsize':14, 'ytick.labelsize':14})
    fig, axes = plt.subplots(4, 2, figsize=(18, 17), gridspec_kw={'hspace':.33, 'wspace':.30})
    records, prefixes = [], []
    for col, seed in enumerate([9108401, 9108402]):
        folder = old / f'e0_t1_s{seed}'
        result = a.read(folder / 'result.json')
        assert result['identity'] == identity
        job = result['job']
        assert job['eta_m'] == .005 and job['tau_M_s'] == 2 and job['tau_z_ms'] == 5000
        restore = result['tracker']['restore_s']
        first = result['tracker']['entries'][0]
        end = round(min(restore - .01, first['confirmation_s'] + 60), 2)
        keys = ['time_ms', 'spikes_1ms', 'regions_1ms', 'slow_time_ms', 'Z', 'M', 'currents', 'raster']
        pieces = {key: [] for key in keys}
        expected = 0
        for path in sorted((folder / 'chunks').glob('*.npz')):
            if '.tmp.' in path.name:
                continue
            with np.load(path) as chunk:
                assert int(chunk['start_step']) == expected
                expected = int(chunk['end_step'])
                for key in keys:
                    pieces[key].append(chunk[key])
            if expected >= round(end * 10000):
                break
        data = {key: np.concatenate(values) for key, values in pieces.items()}
        assert expected >= round(end * 10000) and end < restore
        count = round(end * 1000)
        spikes, regions = data['spikes_1ms'][:count], data['regions_1ms'][:count]
        assert np.array_equal(spikes[:, 0], regions[:, :3].sum(1))
        assert np.array_equal(spikes[:, 1], regions[:, 3:].sum(1))
        n = count // 10
        r10 = np.column_stack([spikes[:, 0].reshape(n, 10).sum(1) / 320,
                              regions[:, :3].reshape(n, 10, 3).sum(1) / nr[:3] / .01])
        tracker = a.run.fresh_tracker()
        for k, rates in enumerate(r10):
            a.run.track(tracker, rates, (k + 1) * .01)
        events = finite_events(r10[:, 0], regions.reshape(n, 10, 6).sum(1), nr)
        pre = [event for event in events if event['start_s'] >= .2 and event['end_s'] < first['onset_s']]
        prefixes.append(spikes[:10000].copy())
        t = data['time_ms'][:count] / 1000
        # Historical chunks store uint16 counts; cast before multiplication.
        continuous = np.column_stack([spikes[:, 0] / 32, regions[:, :2].astype(float) * 1000 / nr[:2]])
        assert np.allclose(continuous[-5000:, 1:].mean(0), r10[-500:, 1:3].mean(0))
        for i, (label, color) in enumerate([('All E', '#333333'), ('Core A E', '#ae5d9c'), ('Core B E', '#347ea0')]):
            axes[0, col].plot(t, gaussian_filter1d(continuous[:, i], 3), color=color, lw=.75, label=label)
        axes[0, col].set(ylim=(-5, 505), title=f'Noise {seed}')
        use = data['slow_time_ms'] / 1000 < end
        ts = data['slow_time_ms'][use] / 1000
        z = data['Z'][use]
        axes[1, col].plot(ts, z[:, 0], color='#7e399d', lw=1.4)
        axes[1, col].fill_between(ts, z[:, 2], z[:, 4], color='#7e399d', alpha=.14)
        axes[1, col].set_ylim(0, 1.04)
        for key, label, color in [(0, 'Excitation', '#c56350'), (2, 'Applied inhibition', '#447ca4')]:
            axes[2, col].plot(ts, data['currents'][use, key], color=color, lw=1., label=label)
        for row in range(3):
            axes[row, col].set_xlim(0, 75)
            axes[row, col].spines[['top', 'right']].set_visible(False)
            axes[row, col].axvline(first['onset_s'], color='#bd3846', ls=':', lw=.9)
            if row == 2:
                axes[row, col].set_xlabel('Time (s)')
            else:
                axes[row, col].tick_params(labelbottom=False)
        lo = round((end - .3) * 10000)
        hi = round(end * 10000)
        raster = data['raster'][lo:hi]
        it, ix = np.nonzero(raster)
        for lower, upper, color in [(0, 20, '#ae5d9c'), (20, 40, '#347ea0'),
                                    (40, 60, '#28536c'), (60, 80, '#c17730')]:
            selected = (ix >= lower) & (ix < upper)
            axes[3, col].scatter(it[selected] * .0001, ix[selected],
                s=5, marker='|', c=color, lw=.6, rasterized=True)
        axes[3, col].set(xlim=(0, .3), ylim=(-1, 80), yticks=[9.5, 29.5, 49.5, 69.5],
            yticklabels=['Core A E', 'Core B E', 'Other E', 'I'], xlabel='Time within late zoom (s)')
        records.append(dict(source=str(folder), seed=seed, job=job,
            native_observation_end_s=end, external_restore_starts_s=restore,
            post_entry_confirmation_observation_s=end - first['confirmation_s'],
            entries=tracker['entries'], recoveries=tracker['recoveries'],
            preentry_finite_event_count=len(pre),
            temporal_audit=temporal_audit(r10[:, 0], r10[:, 1:], tracker['entries'], tracker['recoveries']),
            tail5s_all_A_B_surround_mean_Hz=r10[-500:].mean(0).tolist(),
            tail5s_all_A_B_surround_quiet_fraction=(r10[-500:] < 5).mean(0).tolist(),
            final_Z_mean_A_B=z[-1, [0, 5, 6]].tolist(),
            final_M_feedback_mean_A_B=(job['eta_m'] * data['M'][use][-1, :3]).tolist(),
            late_raster_window_s=[end - .3, end]))
    assert not np.array_equal(prefixes[0], prefixes[1])
    for row, label in enumerate(['E rate (Hz)', 'Resource Z', 'Current (mV equiv.)', 'Sampled cells']):
        axes[row, 0].set_ylabel(label)
        axes[row, 0].text(-.22, 1.04, 'ABCD'[row], transform=axes[row, 0].transAxes, fontsize=23, weight='bold')
    axes[0, 0].legend(loc='upper right', fontsize=11, framealpha=.9)
    axes[2, 0].legend(loc='upper left', fontsize=11, framealpha=.9)
    for ax in axes[2]:
        ax.set_ylim(0, 1700)
    fig.subplots_adjust(left=.12, right=.97, top=.95, bottom=.065)
    out = root / 'paired_native_controls'
    figures = out / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    for ext in ['png', 'pdf']:
        fig.savefig(figures / f'paired_native_prefixes.{ext}', dpi=170)
    plt.close(fig)
    a.write(out / 'observed_controls.json', dict(updated_at=time.time(), status='PASS', records=records,
        actual_first10s_spike_counts_differ_between_seeds=True,
        definition='Original nativeZ/M current model, etaM.005/tauM2s/tauZ5s on the same fixed manual two-core substrate. Only pre-intervention observations are used. Original high/return detector is reapplied to all-cell and regional counts.',
        replication='Two existing noise realizations, identical topology. Reused controls, not newly run conditions or independent network replicates. The later externally restored trajectory is excluded.',
        human_review='PENDING'))
    (figures / 'README.md').write_text('### paired_native_prefixes.png / .pdf\n复用两条原生中等M控制的人工恢复之前完整前缀，各包含首次高态确认后的60秒。展示全E/双核率、Z、实际兴奋/抑制输入与末300ms固定神经元raster；依据原生计数重新核查恢复判据。**关注点**：两个噪声是否都维持高率，而非把后续外部Z恢复当成原生终止；同一拓扑的噪声重复不是独立网络重复。\n')
    print([(r['seed'], r['native_observation_end_s'], r['recoveries'], r['tail5s_all_A_B_surround_mean_Hz']) for r in records])


if __name__ == '__main__':
    main()
