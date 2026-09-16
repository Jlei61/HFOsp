#!/usr/bin/env python3
"""Streamline rendering of existing conditional fields; no new reduction."""
from plot_topic4_prescribed_z_phase import OUT, REFERENCE, read, save
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    status = read(OUT / 'cycle_plane_plot_status.json')
    assert status['status'] == 'COMPLETE'
    native = np.load(REFERENCE / 'trajectory.npz')
    fig = plt.figure(figsize=(12, 10), layout='constrained')
    grid = fig.add_gridspec(3, 2, height_ratios=[.4, 1, 1])
    top = fig.add_subplot(grid[0, :])
    top.plot(native['z_time_ms'] / 1000, native['z_stats'][:, 0], color='#70388d')
    top.fill_between(native['z_time_ms'] / 1000, native['z_stats'][:, 2],
                     native['z_stats'][:, 4], color='#70388d', alpha=.15)
    top.axvspan(10.68, 11.68, color='#238662', alpha=.13)
    top.set(xlim=(0, 13.68), ylim=(.5, 1.03), xlabel='Native reference time (s)', ylabel='E-target Z')
    phases = ['Before sustained recruitment', 'High activity', 'During external refill', 'After refill']
    for idx, row in enumerate(status['rows']):
        tm = row['time_ms']
        a = np.load(OUT / f'cycle_plane_{tm}.npz')
        X, Y, U, V = (a[k] for k in ('X', 'Y', 'U', 'V'))
        assert all(np.isfinite(v).all() for v in (X, Y, U, V))
        ax = fig.add_subplot(grid[1 + idx // 2, idx % 2])
        # Integrate the saved raw Hz/s vector field. No smoothing or rotations.
        ax.streamplot(X[0], Y[:, 0], U, V, color='#769b82', density=1.15,
                      linewidth=.85, arrowsize=.9, zorder=1)
        for drift, color in ((U, '#b13c76'), (V, '#24818b')):
            ax.contour(X, Y, drift, levels=[0], colors=[color], linewidths=1.5)
        tr = a['trajectory_hz']
        ax.plot(tr[:, 0], tr[:, 1], color='black', lw=1.8, zorder=3)
        for j in (10, 35, 65, 85):
            ax.annotate('', xy=tr[j + 3], xytext=tr[j],
                        arrowprops={'arrowstyle': '->', 'color': 'black', 'lw': 1.3})
        ax.scatter(*row['intersection_hz'], marker='D', s=48, c='white', edgecolor='#526c60', zorder=5)
        ax.scatter(*a['center_hz'], s=38, c='#e3a731', edgecolor='black', zorder=6)
        ax.set(xlim=(0, X.max()), ylim=(0, Y.max()), xlabel='Mean E rate (Hz)',
               ylabel='Mean I rate (Hz)', title=f'{idx + 1}. {tm / 1000:g} s — {phases[idx]}')
        top.axvline(tm / 1000, color='#b69245', lw=.8, ls='--')
        top.text(tm / 1000, .51, str(idx + 1), ha='center')
    handles = [Line2D([], [], color='#769b82', label='Conditional flow'),
               Line2D([], [], color='black', lw=1.8, label='Evolving rate replay: ±50 ms'),
               Line2D([], [], color='#b13c76', label='Projected dE/dt = 0'),
               Line2D([], [], color='#24818b', label='Projected dI/dt = 0'),
               Line2D([], [], marker='D', ls='', mfc='white', mec='#526c60', label='Conditional zero-drift intersection'),
               Line2D([], [], marker='o', ls='', mfc='#e3a731', mec='black', label='Exact replay snapshot')]
    fig.legend(handles=handles, loc='outside lower center', ncol=2, fontsize=10, frameon=False)
    fig.suptitle('Streamline view along the prescribed native Z path\nConditional rate slices: synaptic states and input held fixed', fontsize=15)
    save(fig, 'native_z_cycle_conditional_streamlines',
         '沿原空间Z轨迹的四个时刻，把已计算条件方向场改为连续流线，叠加rate回放轨迹、零漂移线和交点；未修改方程、原向量或轨迹。流线数值积分使用现有41×41网格的Hz/s漂移，四图坐标范围各自适配。',
         '这是流线展示候选，不是已验证的二维动力学约化。流线固定隐藏状态，黑色轨迹的隐藏状态继续演化，因此二者不能作为full/reduced模型一致性验证；菱形不表示全网络稳定性，未添加人为极限环。')


if __name__ == '__main__':
    main()
