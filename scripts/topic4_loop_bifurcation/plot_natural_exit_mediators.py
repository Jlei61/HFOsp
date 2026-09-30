#!/usr/bin/env python3
"""Display measured exit and recovery mechanisms for the paired interventions."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import shutil
import time
import numpy as np
from campaign import ROOT, read, write, sha
from analyze_natural_exit_mediators import DEST, START, END, CONTROLS

OUT = ROOT/'natural_exit_mediator_figure_v2'
STEM = 'natural_exit_mediators_v2'


def main(wait):
    OUT.mkdir(exist_ok=True)
    while not (DEST/'result.json').exists():
        if (DEST/'progress.json').exists() and read(DEST/'progress.json')['status'] == 'STOPPED_ON_NATIVE_FAILURE':
            write(OUT/'progress.json', dict(status='STOPPED_ON_NATIVE_FAILURE'));return
        write(OUT/'progress.json', dict(status='WAITING_MEDIATOR_ANALYSIS', pid=os.getpid(), updated_epoch=time.time()))
        if not wait:return
        time.sleep(20)
    result = read(DEST/'result.json')
    assert result['status'] == 'COMPLETE_TWO_NATIVE_MEDIATOR_CONTROLS'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(5, 3, figsize=(13.5, 12), sharey='row', layout='constrained')
    names = ['original_unmodified', *CONTROLS]
    titles = ['Original autonomous trajectory', 'Remove existing G once', 'Shorten low-rate K decay']
    colors = ['#8a63b4', '#d63378', '#008eb3']
    for column, (name, title, row) in enumerate(zip(names, titles, result['rows'])):
        with np.load(DEST/f'{name}.npz') as z:
            t, tz, tb = z['time_s']-START, z['slow_time_s']-START, z['budget_time_s']-START
            tr, rate, Z, budget = z['rate_time_s']-START, z['rates_allE_A_B_other_Hz'], z['Z_allE_A_B_other'], z['Z_budget']
            zoom, a, g, resource, balance = axes[:, column]
            for j, color in enumerate(colors):
                a.plot(tr, rate[:, j], color=color, lw=.8, alpha=.85)
            a.plot(t, z['causal_R_Hz'], color='.2', lw=.65, alpha=.75)
            a.set(ylim=(-8, 510))
            g.plot(t, z['Graw'], color='#bd681e', lw=1.2, label='Raw G')
            g.plot(t, z['K_mean'], color='#33557b', lw=1.2, label='Mean K')
            g.axhline(row['G_resource_block_threshold'], color='#bd681e', lw=.75, ls=':')
            for j, color in enumerate(colors[1:], 1):
                resource.plot(tz, Z[:, j], color=color, lw=1.25)
                resource.axhline(row['core_reference'][j-1], color=color, lw=.85, ls='--')
                balance.plot(tb, budget[:, j, 4], color=color, lw=1.)
            resource.set(ylim=(0, 1.03))
            balance.axhline(0, color='.45', lw=.7, ls=':')
            low = row['first_R_at_or_below5_for100ms_s']
            recovered = row['first_both_core_Z_reference_s']
            for ax in axes[1:, column]:
                ax.set_xlim(0, END-START)
                if low is not None:
                    ax.axvline(low-START, color='.65', ls=':', lw=.65)
            if recovered is not None:
                resource.axvline(recovered-START, color='.3', ls=':', lw=.8)
                resource.text(recovered-START+.15, .12, 'Both cores\nreach reference', fontsize=8, va='bottom')
            else:
                resource.text(.98, .94, 'Reference not reached\nin this 10 s window', transform=resource.transAxes,
                              ha='right', va='top', fontsize=8)
            # A separate axis keeps the exit zoom from hiding later bursts.
            early = t <= .3
            zoom.plot(t[early], z['causal_R_Hz'][early], color='.2', lw=1.1)
            zoom.set(title=title, xlim=(0, .3), yscale='log', ylim=(1e-5, 300),
                     xticks=[0, .15, .3], yticks=[.001, 5, 200], xlabel='First 0.3 s after shared state')
            zoom.axhline(5, color='.5', ls='--', lw=.8)
            zoom.set_yticklabels(['0.001', '5', '200'])
            for ax in axes[1:4, column]:ax.tick_params(labelbottom=False)
            balance.set_xlabel('Time since shared state at 16.8 s (s)')
    for a, label in zip(axes[:, 0], ['Causal R (Hz; log)', 'E rate (Hz)', 'Conductance / leak', 'Core resource Z', r'Net core dZ/dt (s$^{-1}$)']):
        a.set_ylabel(label)
    axes[1, 0].legend([Line2D([], [], color=c, lw=1.4) for c in colors]+[Line2D([], [], color='.2', lw=1)],
                      ['All E', 'Core A', 'Core B', 'Causal R'], loc='upper left', frameon=False, fontsize=8)
    axes[2, 0].legend(frameon=False, fontsize=9, loc='upper right')
    axes[3, 0].text(.03, .96, 'Dashed: pre-onset core Z reference', transform=axes[3, 0].transAxes,
                   fontsize=8, va='top')
    fig.suptitle('Actual exit-state interventions: suppression, feedback decay and core recovery', fontsize=13)
    for ext in ['png', 'svg']:
        fig.savefig(ROOT/f'figures/{STEM}.{ext}', dpi=180)
    plt.close(fig)
    import xml.etree.ElementTree as ET
    ET.parse(ROOT/f'figures/{STEM}.svg')
    write(OUT/'result.json', dict(status='COMPLETE_CANDIDATE', data_source=str(DEST/'result.json'),
        producer_sha256=sha(__file__), agent_visual_review='PENDING', human_visual_review='PENDING',
        scope='Paired native causal interventions from one complete state; not new autonomous loops or a formal bifurcation diagram.', SVG_XML='PASS'))
    shutil.copy2(__file__, OUT/'producer.py')
    title = f'### {STEM}.png / {STEM}.svg'
    path = ROOT/'figures/README.md'
    if title not in path.read_text():
        with path.open('a') as f:
            f.write('\n\n'+title+'\n从同一已验证的实际下降状态出发，对照原轨迹、一次移除已有 G 尾迹和缩短低率 K 保留；三列未来输入完全配对。依次显示群体活动及退出放大、G/K、两核 Z 与真实净恢复预算，虚线参考值沿用原间期标准。\n**关注点**：低活动、两核恢复和短事件返回是不同结果；干预不计入自主闭环，图中没有用静默时长预设恢复，人工待审。\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--wait', action='store_true');main(p.parse_args().wait)
