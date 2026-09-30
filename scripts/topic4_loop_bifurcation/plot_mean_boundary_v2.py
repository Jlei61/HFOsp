#!/usr/bin/env python3
"""Show measured endpoints and the actual unaligned spatial collapse."""
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
    source = ROOT/'mean_boundary_correspondence/analysis';result = read(source/'result.json')
    assert result['status'] == 'COMPLETE_MATCHED_UPPER_EXIT_COMPARISON'
    dest = ROOT/'mean_boundary_correspondence/figure_v2';dest.mkdir(exist_ok=True)
    d = dict(np.load(source/'readouts.npz'));ref = dict(np.load(ROOT/'dynamic_mean_history_pair/analysis/high_readouts.npz'))
    centers = np.load(ROOT/'native_slices/geometry.npz')['centers_mm']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(2, 3, figsize=(12, 8), layout='constrained')
    for key, label, color, marker in [('native', 'Native', 'black', 'o'), ('model', 'Leading recurrent mean', '#8266ad', 'x')]:
        ax[0, 0].plot(d[key+'_time'], d[key+'_R'], color=color, lw=1, label=label)
        for core, ls in [(1, '-'), (2, '--')]:
            ax[0, 1].plot(d.get(key+'_drift_time', d[key+'_time']), d[key+'_drift'][:, core], ls=ls, color=color, lw=.9)
        low = ref[key+'_rate'];upper = d[key+'_rate']
        ax[0, 2].plot([9.35, 9.5], [low[len(low)//2:, 0].mean(), upper[len(upper)//2:, 0].mean()],
            ls='none', marker=marker, markersize=7, markerfacecolor='none', color=color, label=label)
    ax[0, 0].set_yscale('symlog', linthresh=1);ax[0, 0].axhline(5, color='.6', ls=':', lw=.8)
    ax[0, 0].set(ylim=(0, 250), xlim=(0, 10), xlabel='Time from high history (s)', ylabel='Causal E rate (Hz)', title='Held K = 9.5; Z unchanged')
    ax[0, 0].legend(frameon=False, fontsize=8)
    ax[0, 1].axhline(0, color='.6', lw=.8);ax[0, 1].set(xlim=(0, 10), xlabel='Time from high history (s)', ylabel='Core dZ/dt if released (1/s)', title='Core A solid; Core B dashed')
    ax[0, 2].axvspan(9.35, 9.5, color='.92', zorder=-2)
    ax[0, 2].text(.5, .5, 'Interior not yet resolved\nNo certified branch', ha='center', va='center', transform=ax[0, 2].transAxes, fontsize=10)
    ax[0, 2].set(xlabel='Held mean K / gL', ylabel='All-E rate, 5–10 s (Hz)', title='Measured conditional endpoints', xticks=[9.35, 9.5])
    for axis, label, f in zip(ax[1], ['Native K = 9.35', 'Native K = 9.5', 'Leading mean K = 9.5'],
        [ref['native_field'][1000:].mean(0), d['native_field'][1000:].mean(0), d['model_field'][5000:].mean(0)]):
        im = axis.imshow(f.reshape(20, 20), origin='lower', extent=(0, 20, 0, 20), vmin=0, vmax=500, cmap='magma', interpolation='nearest')
        for xy in centers:axis.add_patch(Circle(xy, 1.5, fill=False, edgecolor='#00c3c5', lw=.9))
        axis.set(title=label, xlabel='x (mm)', ylabel='y (mm)', xticks=[0, 10, 20], yticks=[0, 10, 20])
    fig.colorbar(im, ax=ax[1].tolist(), shrink=.8, label='E rate, 5–10 s (Hz)')
    fig.suptitle('High-state persistence and the sign of core resource balance', weight='bold')
    for ext in ['png', 'svg']:fig.savefig(ROOT/f'figures/mean_boundary_correspondence_v2.{ext}', dpi=180)
    plt.close(fig)
    windows = [(0, .1), (1, 1.1), (1.5, 1.6), (1.8, 1.9), (1.95, 2.05), (2.1, 2.2)]
    fig, ax = plt.subplots(2, 6, figsize=(14, 5.2), layout='constrained');fields = []
    for row, (key, label, factor) in enumerate([('native', 'Native', 200), ('model', 'Leading mean', 1000)]):
        fields.append([])
        for col, (lo, hi) in enumerate(windows):
            f = d[key+'_field'][round(lo*factor):round(hi*factor)].mean(0);fields[-1].append(f)
            im = ax[row, col].imshow(f.reshape(20, 20), origin='lower', extent=(0, 20, 0, 20), vmin=0, vmax=500, cmap='magma', interpolation='nearest')
            for core, xy in zip(['A', 'B'], centers):
                ax[row, col].add_patch(Circle(xy, 1.5, fill=False, edgecolor='#00c3c5', lw=.9));ax[row, col].text(xy[0], xy[1]+1.8, core, color='#008e92', ha='center', fontsize=8)
            ax[row, col].set(xticks=[0, 10, 20], yticks=[0, 10, 20])
            if row == 0:ax[row, col].set_title(f'{lo:g}–{hi:g} s', fontsize=10)
            if row == 1:ax[row, col].set_xlabel('x (mm)')
            if col == 0:ax[row, col].set_ylabel(label+'\ny (mm)')
    fig.colorbar(im, ax=ax.ravel().tolist(), shrink=.7, label='E rate, 100 ms bins (Hz)')
    fig.suptitle('Spatial collapse at held K = 9.5: common physical windows, no time alignment', weight='bold', fontsize=12)
    for ext in ['png', 'svg']:fig.savefig(ROOT/f'figures/mean_boundary_spatial_collapse.{ext}', dpi=180)
    plt.close(fig)
    np.savez_compressed(dest/'spatial_display.npz', windows_s=windows, native=fields[0], model=fields[1])
    write(dest/'result.json', dict(status='COMPLETE_DISPLAY_ONLY_REVIEW_PENDING', agent_visual='PENDING', human_visual='PENDING',
        scientific_result_retained=result['retained'], native_firstlow_s=result['first100ms_causal_R_le5_s']['native'],
        model_firstlow_s=result['first100ms_causal_R_le5_s']['model'], producer_sha256=sha(__file__),
        changes='Nonnegative rateaxis; separate measured endpoints without interpolation; explicit unresolvedinterior. Second figure shows native/model spatialfields in identical100ms physicalwindows, no eventalignment or newmetric.'))
    shutil.copy2(__file__, dest/'producer.py')
    p = ROOT/'figures/README.md'
    for name, description in [('mean_boundary_correspondence_v2', '退出与核心Z收支的同条件核对；右图只画已测端点，灰区标记尚未解析的内部区间，取消两点插值线。'),
        ('mean_boundary_spatial_collapse', '退出期间六个相同物理时间窗口的原生与候选模型空间场；每窗100毫秒，同一色标，不作退出时间对齐。')]:
        title = f'### {name}.png / {name}.svg'
        if title not in p.read_text():
            with p.open('a') as f:f.write('\n\n'+title+'\n'+description+'原始读出及所有科学门保持不变。\n**关注点**：这是固定Z/K条件响应，Z正收支不等于本次已经释放并恢复；未认证稳定/不稳定支或分岔类型，人工待审。\n')


if __name__ == '__main__':main()
