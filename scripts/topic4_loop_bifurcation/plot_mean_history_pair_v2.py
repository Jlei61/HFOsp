#!/usr/bin/env python3
"""Common time bins and a meaningful G scale for the completed history pair."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from campaign import ROOT, read, write, sha


def main():
    source = ROOT/'dynamic_mean_history_pair/analysis'
    assert read(source/'result.json')['both_histories_retained']
    dest = ROOT/'dynamic_mean_history_pair/figure_v2';dest.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(3, 4, figsize=(14, 10), layout='constrained')
    centers = np.load(ROOT/'native_slices/geometry.npz')['centers_mm'];arrays = {}
    for j, case in enumerate(['high', 'asymmetric']):
        a = dict(np.load(source/f'{case}_readouts.npz'));arrays[case] = a
        for key, label, color in [('native', 'Native', 'black'), ('model', 'Leading recurrent mean', '#8266ad')]:
            t = a[key+'_time'];ax[0, j*2].plot(t, a[key+'_R'], lw=.6, color=color, label=label)
            ax[0, j*2+1].plot(t, a[key+'_G'], lw=1, color=color)
            values = a[key+'_rate'];width = 4 if key == 'native' else 20
            rates = values.reshape(500, width, 4).mean(1)
            tt = (np.arange(500)+.5)*.02
            for core, ls in [(1, '-'), (2, '--')]:
                ax[1, j*2].plot(tt, rates[:, core], lw=.7, color=color, ls=ls)
                dt = a.get(key+'_drift_time', t)
                ax[1, j*2+1].plot(dt, a[key+'_drift'][:, core], lw=.8, color=color, ls=ls)
        ax[0, j*2].set(title=case.capitalize()+' history', ylabel='Causal E rate (Hz)')
        ax[0, j*2+1].set(title='Global feedback stays off', ylabel='Global G / gL', ylim=(-.005, .105), yticks=[0, .05, .1])
        ax[1, j*2].set(title='Core A solid; Core B dashed', ylabel='Core rate (Hz; 20 ms bins)')
        ax[1, j*2+1].set(title='Z held: drift if released', ylabel='Core dZ/dt (1/s)')
        for axis in ax[:2, j*2:j*2+2].ravel():axis.set(xlim=(0, 10), xlabel='Time from native state (s)')
        for k, (key, label) in enumerate([('native', 'Native'), ('model', 'Leading recurrent mean')]):
            f = a[key+'_field'];f = f[len(f)//2:].mean(0);axis = ax[2, j*2+k]
            im = axis.imshow(f.reshape(20, 20), origin='lower', extent=(0, 20, 0, 20), vmin=0, vmax=500, cmap='magma', interpolation='nearest')
            for xy in centers:axis.add_patch(Circle(xy, 1.5, fill=False, edgecolor='#00c3c5', lw=.9))
            axis.set(title=label+' (5–10 s)', xlabel='x (mm)', ylabel='y (mm)', xticks=[0, 10, 20], yticks=[0, 10, 20])
    fig.colorbar(im, ax=ax[2].tolist(), shrink=.8, label='E rate (Hz)')
    handles, labels = ax[0, 0].get_legend_handles_labels();fig.legend(handles, labels, loc='outside lower center', ncol=2, frameon=False)
    fig.suptitle('Same Z/K, different histories: conditional correspondence over 10 s', weight='bold')
    for ext in ['png', 'svg']:fig.savefig(ROOT/f'figures/dynamic_mean_history_pair_v2.{ext}', dpi=180)
    plt.close(fig)
    write(dest/'result.json', dict(status='COMPLETE_DISPLAY_ONLY_REVIEW_PENDING',
        changes='Both core-rate traces shown in common20ms nonoverlapping bins, conserving counts and fullwindow mean. G axis0-.1 shows feedback is off instead of magnifying1e-26 initialresidual. Original1ms/model and5ms/native readouts and all scientific guards unchanged.',
        agent_visual='PENDING', human_visual='PENDING', producer_sha256=sha(__file__)))
    shutil.copy2(__file__, dest/'producer.py')
    p = ROOT/'figures/README.md';title = '### dynamic_mean_history_pair_v2.png / dynamic_mean_history_pair_v2.svg'
    if title not in p.read_text():
        with p.open('a') as f:f.write('\n\n'+title+'\n同一Z/K场两种完整历史的十秒原生对应；双核率统一20毫秒计数窗口，G使用0至0.1的可解释尺度，避免放大接近数值零的初始尾迹。空间场、原始读出及科学判断不变，v1保留。\n**关注点**：同参数下有限时间历史依赖已保留；仍不是稳定性或分岔认证，人工待审。\n')


if __name__ == '__main__':main()
