#!/usr/bin/env python3
"""Display repair only: shorter title and nonnegative rate axis."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from campaign import ROOT, read, write, sha


def main():
    src = ROOT/'mean_boundary_resolution/analysis';assert read(src/'result.json')['retained']
    a = dict(np.load(src/'readouts.npz'))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.2), layout='constrained')
    for key, color in [('R64', '#8266ad'), ('R256', '#c07935')]:
        ax[0].plot(a[key+'_time'], a[key+'_R'], lw=1, color=color, label=key)
        for core, ls in [(1, '-'), (2, '--')]:ax[1].plot(a[key+'_time'], a[key+'_drift'][:, core], color=color, ls=ls, lw=1)
    ax[0].set_yscale('symlog', linthresh=1);ax[0].set(xlim=(0, 4), ylim=(0, 250), xlabel='Time (s)', ylabel='Causal E rate (Hz)')
    ax[0].axhline(5, color='.6', ls=':', lw=.8);ax[0].legend(frameon=False)
    ax[1].set(xlim=(0, 4), xlabel='Time (s)', ylabel='Core dZ/dt if released (1/s)', title='Core A solid; Core B dashed')
    ax[1].axhline(0, color='.6', lw=.8)
    ax[2].plot((np.arange(40)+.5)*.1, a['transient_RMS'], color='black')
    ax[2].set(xlim=(0, 4), xlabel='Time (s)', ylabel='Spatial field RMS difference (Hz)', title='Common 100 ms bins\nNo time shift')
    fig.suptitle('Exit at held K = 9.5: numerical replica resolution', weight='bold')
    for ext in ['png', 'svg']:fig.savefig(ROOT/f'figures/mean_boundary_resolution_v2.{ext}', dpi=180)
    plt.close(fig)
    dest = ROOT/'mean_boundary_resolution/figure_v2';dest.mkdir(exist_ok=True)
    write(dest/'result.json', dict(status='COMPLETE_DISPLAY_ONLY_REVIEW_PENDING', agent_visual='PENDING', human_visual='PENDING',
        changes='Shorten/wrap long righttitle to avoid clipping, restrict firingrateaxis to nonnegative values. Same arrays, metrics and thresholds; v1 retained.', producer_sha256=sha(__file__)))
    shutil.copy2(__file__, dest/'producer.py')
    p = ROOT/'figures/README.md';title = '### mean_boundary_resolution_v2.png / mean_boundary_resolution_v2.svg'
    if title not in p.read_text():
        with p.open('a') as f:f.write('\n\n'+title+'\n复制分辨率检验的显示修订：右标题换行避免截断，率轴限定非负。原始数值与全部判断不变，v1保留。\n**关注点**：该工作点退出时刻未随64至256复制改变；不认证大复制极限或分支稳定性，人工待审。\n')


if __name__ == '__main__':main()
