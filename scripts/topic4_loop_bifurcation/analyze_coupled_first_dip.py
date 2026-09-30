#!/usr/bin/env python3
"""Localize free-coupling disagreement at the first activity dip; no new runs."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import time
import numpy as np
from campaign import ROOT, read, write, sha
from analyze_coupled_density_exit import native_data, candidate, ADAPTED

OUT = ROOT / 'coupled_first_dip'


def main(wait):
    OUT.mkdir(exist_ok=True)
    while not (ROOT / 'coupled_density_exit/result.json').exists():
        if not wait:
            return
        write(OUT / 'progress.json', dict(status='WAITING_FIXED_COUPLED_RUNS', pid=os.getpid(), updated_epoch=time.time()))
        time.sleep(20)
    assert read(ROOT / 'coupled_density_exit/result.json')['status'] == 'COMPLETE'
    geo = dict(np.load(ADAPTED / 'geometry.npz'))
    data = [native_data(geo)] + [candidate(seed, geo) for seed in [927671, 927672]]
    rows = []
    for d in data:
        windows = []
        for lo, hi in [(13.4, 13.5), (13.6, 13.7), (13.7, 13.8), (13.9, 14.0)]:
            r = (d['rate_time_ms'] >= lo*1000) & (d['rate_time_ms'] < hi*1000)
            g = (d['global_time_ms'] >= lo*1000) & (d['global_time_ms'] < hi*1000)
            s = (d['slow_time_ms'] >= lo*1000) & (d['slow_time_ms'] < hi*1000)
            windows.append(dict(interval_s=[lo, hi], rate_allE_A_B_other=d['rates'][r].mean(0).tolist(),
                causal_R_min_max=[float(d['R'][g].min()), float(d['R'][g].max())],
                Graw_mean=float(d['Graw'][g].mean()), Z_mean=d['Z'][s].mean(0).tolist(),
                K_mean=d['K'][s].mean(0).tolist()))
        # Restrict to a q=0 interval and measure the actual exponential law.
        glob = (d['global_time_ms'] >= 13800) & (d['global_time_ms'] <= 13980)
        assert d['R'][glob].max() < 200
        slow = np.flatnonzero((d['slow_time_ms'] >= 13820) & (d['slow_time_ms'] <= 13980))
        i, j = slow[0], slow[-1]
        dt = (d['slow_time_ms'][j]-d['slow_time_ms'][i])/1000
        tau = -dt/np.log(d['K'][j, 0]/d['K'][i, 0])
        rows.append(dict(name=d['name'], windows=windows,
            zero_q_decay=dict(interval_s=[float(d['slow_time_ms'][i]/1000), float(d['slow_time_ms'][j]/1000)],
                             observed_tauK_s=float(tau), R_range=[float(d['R'][glob].min()), float(d['R'][glob].max())])))
    report = dict(status='COMPLETE', rows=rows, source_comparison=str(ROOT/'coupled_density_exit/comparison.json'),
        question='Does loss of residual surrounding activity switch the physical R<=5 K-retention rule at the early dip?',
        selection='Posthoc localization of the first coupled discrepancy; fixed13.4-14.3 diagnostic, not a prospective independent validation.',
        limits='The observed sequence explains divergence amplification but doesnotidentify which approximation orfinite-count fluctuation caused initial residual loss. Separatepaired-nativecount experiment addresses this.',
        producer_sha256=sha(__file__), formal_bifurcation_allowed=False, human_review='PENDING')
    write(OUT / 'analysis.json', report)
    plot(data, geo)
    write(OUT / 'progress.json', dict(status='COMPLETE', updated_epoch=time.time()))


def plot(data, geo):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'svg.fonttype':'none'})
    fig, axes = plt.subplots(3, 3, figsize=(12.6, 10), layout='constrained', gridspec_kw={'height_ratios':[.8, 1, 1]})
    colors = ['black', '#267f8e', '#cf7b2b']
    for d, color in zip(data, colors):
        g = (d['global_time_ms'] >= 13400) & (d['global_time_ms'] <= 14300)
        s = (d['slow_time_ms'] >= 13400) & (d['slow_time_ms'] <= 14300)
        axes[0, 0].plot(d['global_time_ms'][g]/1000, d['R'][g], color=color, lw=1.1, label=d['name'])
        axes[0, 1].plot(d['global_time_ms'][g]/1000, d['Graw'][g], color=color, lw=1.1)
        axes[0, 2].plot(d['slow_time_ms'][s]/1000, d['K'][s, 0], color=color, lw=1.1)
    for ax, ylabel in zip(axes[0], ['Causal E rate (Hz)', 'Raw global G / gL', 'Mean K / gL']):
        ax.set_xlim(13.4, 14.3)
        ax.set_xlabel('Native time (s)')
        ax.set_ylabel(ylabel)
    axes[0, 0].set_ylim(-10, 500)
    axes[0, 0].axhline(5, color='.5', ls=':', lw=.8)
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].axhline(95.19851312666987/(18+17.662847938268442), color='.5', ls=':', lw=.8)
    for row, (lo, hi) in enumerate([(13.7, 13.8), (13.9, 14.0)], start=1):
        for col, d in enumerate(data):
            ax = axes[row, col]
            keep = (d['field_time_ms'] >= lo*1000) & (d['field_time_ms'] < hi*1000)
            field = d['fields'][keep].mean(0).reshape(20, 20)
            im = ax.imshow(field, origin='lower', extent=[0, 20, 0, 20], cmap='magma', vmin=0, vmax=500)
            for center in geo['centers_mm']:
                ax.add_patch(Circle(center, 1.5, fill=False, color='#00bec7', lw=.9))
            ax.set_title(f'{d["name"]}: {lo:g}–{hi:g} s', fontsize=10)
            ax.set_xticks([0, 10, 20]); ax.set_yticks([0, 10, 20])
            if col == 0:
                ax.set_ylabel('y (mm)')
            if row == 2:
                ax.set_xlabel('x (mm)')
    fig.colorbar(im, ax=axes[1:].ravel().tolist(), shrink=.8, label='E population rate (Hz)', pad=.02)
    fig.suptitle('Residual activity determines whether the K decay rule switches', weight='bold')
    fig.text(.5, -.012, 'Original native and both fixed numerical streams. Absolute windows; posthoc discrepancy localization, not bifurcation certification.', ha='center', fontsize=9)
    for ext in ['png', 'svg']:
        fig.savefig(ROOT/'figures'/f'coupled_first_dip.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    write(OUT/'figure_metadata.json', dict(agent_visual='PENDING', human_review='PENDING', producer_sha256=sha(__file__)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--wait', action='store_true')
    main(parser.parse_args().wait)
