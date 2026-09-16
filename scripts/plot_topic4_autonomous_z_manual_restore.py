#!/usr/bin/env python3
"""Raster, applied neuron-wise Z and population rates from the saved trajectory."""
from validate_topic4_fixed_rate_base import ROOT, read, write
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = ROOT / 'results/topic4_sef_hfo/autonomous_z_manual_restore_v1'


def render():
    a = np.load(OUT / 'trajectory.npz'); meta = read(OUT / 'run.json')
    assert all(np.isfinite(a[k]).all() for k in a.files)
    duration = meta['duration_ms'] / 1000.; restore = meta['restore_start_ms']
    t = a['z_time_ms'] / 1000.; z = a['z_stats']; rates = np.c_[a['rate_e_hz'], a['rate_i_hz']].reshape(-1, 50, 2).mean(1)
    tr = (np.arange(len(rates)) + .5) * .005
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(4, 1, figsize=(14, 11), sharex=True, layout='constrained',
                           gridspec_kw={'height_ratios': [1, 1.6, 1.1, .9]})
    ax[0].fill_between(t, z[:, 2], z[:, 4], color='#7b3294', alpha=.15, label='E cells: 10th–90th percentile')
    ax[0].plot(t, z[:, 0], c='#7b3294', label='E mean Z')
    for k, label, c in [(5, 'Core A', '#ac4086'), (6, 'Core B', '#3b83b4'), (7, 'Surround', '.45')]:
        ax[0].plot(t, z[:, k], c=c, lw=.8, label=label)
    ax[0].set(ylabel='Applied inhibition\ncoefficient Z', ylim=(-.03, 1.08))
    ax[0].legend(ncol=3, fontsize=9, loc='lower left')
    for lo, hi, color in [(0, 240, '#2876ad'), (240, 300, '#e17c24')]:
        times, ids = np.nonzero(a['sample_spikes'][:, lo:hi])
        ax[1].scatter(times * .0001, ids + lo, s=.3, c=color, linewidths=0, rasterized=True)
    for y in [60, 120, 240]: ax[1].axhline(y-.5, c='.75', lw=.6)
    ax[1].set(ylim=(-1, 300), yticks=[30, 90, 180, 270], yticklabels=['Core A E', 'Core B E', 'Other E', 'I'])
    for k, label, c in [(0, 'All E', '#2876ad'), (1, 'All I', '#e17c24')]:
        ax[2].plot(tr, rates[:, k], c=c, lw=.8, label=label)
    ax[2].set(ylabel='Population rate\n(Hz; 5-ms bins)'); ax[2].legend(loc='upper right')
    ax[3].plot(t, z[:, 8], color='#a65f20', label='E cells with GABA above depletion threshold')
    ax[3].plot(t, z[:, 1], color='#7b3294', label='Across-E standard deviation of Z')
    ax[3].set(ylabel='Fraction / Z SD', xlabel='Time (s)', xlim=(0, duration)); ax[3].legend(fontsize=9, loc='upper right')
    if restore is not None:
        r = restore / 1000.
        for axis in ax:
            axis.axvspan(r, r + 1, color='#26845d', alpha=.10)
            axis.axvline(r, c='#26845d', ls='--', lw=.8)
            axis.axvline(r + 1, c='#26845d', ls=':', lw=.8)
        ax[0].set_title(f'Native Z dynamics until {r:.2f} s; external refill to 1 over 1 s, then clamp at 1', fontsize=12)
    else:
        ax[0].set_title('No sustained-high trigger within 20 s: Z remains autonomous; no manual refill was applied', fontsize=12)
    fig.suptitle('Same dual-core SNN with native OU: neuron-wise Z evolution and manual inhibition restoration', fontsize=14)
    fig.supxlabel(r'Native E-only Z: $5000\,\mathrm{ms}\,\dot z_i=\mathbf{1}[I_{\mathrm{GABA},i}<95.20]-z_i$; I-cell Z = 1; M off.'
                  '\nManual refill changes only Z; membrane, synapses, delays and random state are continuous. Raster uses fixed sampled neurons; rates use all 40,000.', fontsize=10)
    folder = OUT / 'figures'; folder.mkdir(exist_ok=True)
    fig.savefig(folder / 'autonomous_z_manual_restore.png', dpi=180)
    fig.savefig(folder / 'autonomous_z_manual_restore.pdf'); plt.close(fig)
    def stats(lo, hi):
        sel = (tr >= lo) & (tr < hi); y = rates[sel, 0]
        return {'time_s': [lo, hi], 'mean_E_hz': float(y.mean()), 'peak_E_5ms_hz': float(y.max()),
                'CV_E_5ms': float(y.std() / max(y.mean(), 1e-12)), 'fraction_E_below1Hz': float(np.mean(y < 1))}
    summary = {'status': 'COMPLETE_PENDING_REVIEW', 'baseline': stats(.5, 1.),
               'final_1s': stats(duration-1., duration), 'minimum_mean_Z': float(z[:, 0].min()),
               'max_spatial_Z_std': float(z[:, 1].max()),
               'sustained_high_detected_ms': meta['sustained_high_detected_ms'],
               'restore_start_ms': restore,
               'no_new_biological_variable': True,
               'interpretation_limit': 'One historical parameter transfer; manually restored Z is not autonomous event termination or patient IED validation.'}
    if restore is not None:
        r = restore / 1000.; summary['before_manual_restore'] = stats(r-.5, r)
    write(OUT / 'analysis.json', summary)
    (folder / 'README.md').write_text('### autonomous_z_manual_restore.png\n同一双核 SNN 在原 OU 背景下使用逐神经元 Z 原方程，M 关闭；上排显示 Z 分布及区域均值，中排为真实 raster 和全群放电率，下排区分耗竭驱动与空间异质性。若满足持续高活动触发条件，绿色段只将 Z 手动恢复到 1，随后夹持；若未触发，则全程保持自主演化。PDF 为同名版本。\n**关注点**：旧 Z 能否自行把网络带入高活动，以及补回 Z 后是否退出；不能把手动补回解释为自主终止。\n')


if __name__ == '__main__':
    render()
