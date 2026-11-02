"""Candidate Fig4 I: old D values, explicit display aggregate and uncertainty.

Public API: render_i(fig, rect) -> matplotlib.axes.Axes.
rect is (left, bottom, width, height) in normalized figure coordinates and
describes the DATA AXIS (labels need external margin). Intended size 65 x 53 mm.
No global rcParams, parent figure geometry or other panels are modified.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from matplotlib.patches import Patch

SOURCE = Path(__file__).resolve().parent
METRICS = ('mean_rank_error', 'within_shaft_order_error', 'participation_error')
COLORS = ('#3D5C6F', '#E47159', '#6B7546')


def render_i(fig, rect):
    """Draw exact 13-point curves; black mean +/- time-block bootstrap SD."""
    rows = list(csv.DictReader((SOURCE / 'i_display_angle_response.csv').open()))
    x = np.array([float(r['parameter_value']) for r in rows])
    raw = np.array([[float(r[k]) for k in METRICS] for r in rows])
    mean = np.array([float(r['display_loss']) for r in rows])
    sd = np.array([float(r['display_loss_bootstrap_sd']) for r in rows])
    np.testing.assert_array_equal(raw.mean(1), mean)
    assert len(x) == 13 and np.all(np.diff(x) > 0)
    ax = fig.add_axes(rect)
    ax.set_gid('fig4_i_angle_response')
    band = ax.fill_between(x, mean-sd, mean+sd, color='#202020', alpha=.15, linewidth=0, zorder=1)
    band.set_gid('i_conditional_block_bootstrap_sd')
    handles = []
    for j, (label, color, marker) in enumerate(zip(('Mean rank', 'Order', 'Participation'), COLORS, ('o','s','^'))):
        line, = ax.plot(x, raw[:, j], color=color, lw=.8, marker=marker, ms=2.3,
                        mew=.4, label=label, zorder=2)
        line.set_gid('i_' + METRICS[j])
        np.testing.assert_array_equal(line.get_ydata(), raw[:, j])
        handles.append(line)
    total, = ax.plot(x, mean, color='#171717', lw=1.35, marker='o', ms=2.5,
                     label='Mean display error', zorder=3)
    total.set_gid('i_display_loss')
    handles.extend([total, Patch(facecolor='#202020', alpha=.15, edgecolor='none', label='±1 block-bootstrap SD')])
    best = int(np.argmin(mean))
    ax.plot(x[best], mean[best], marker='*', ms=7, color='black', mec='white', mew=.35, zorder=5)
    ax.annotate(f'Empirical min: +{x[best]:.2f}°', xy=(x[best],mean[best]), xytext=(.31,.10),
        textcoords='axes fraction', fontsize=7, ha='center', va='center',
        arrowprops=dict(arrowstyle='-', lw=.65, color='#353535'), zorder=6)
    ax.set(xlim=(-21.5,21.5), ylim=(0,.42), xticks=[-20,-10,0,10,20], yticks=[0,.1,.2,.3,.4])
    ax.set_xlabel('E→E angle offset (°)', fontsize=9, labelpad=3)
    ax.set_ylabel('Display error', fontsize=9, labelpad=3)
    ax.tick_params(axis='both', labelsize=8, width=.65, length=2.7, pad=2)
    ax.spines[['top','right']].set_visible(False)
    for spine in ('left','bottom'):
        ax.spines[spine].set_linewidth(.65)
    ax.legend(handles=handles, loc='upper right', fontsize=7, frameon=False,
              handlelength=1.5, handletextpad=.5, labelspacing=.2, borderaxespad=.3)
    return ax


def preview():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = SOURCE / 'figures'
    folder.mkdir(exist_ok=True)
    with plt.rc_context({'font.family':'DejaVu Sans', 'pdf.fonttype':42, 'svg.fonttype':'none'}):
        fig = plt.figure(figsize=(86/25.4,70/25.4), dpi=240)
        render_i(fig, (15/86,12/70,65/86,53/70))
        fig.text(1.5/86,68/70,'I',fontsize=12,weight='bold',va='top')
        for suffix in ('png','pdf','svg'):
            fig.savefig(folder / ('fig4-paneli-display-diagnostic.' + suffix), dpi=300, facecolor='white')
        plt.close(fig)


if __name__ == '__main__':
    preview()
