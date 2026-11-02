#!/usr/bin/env python3
"""One continuous time colormap for the complete native SNN trajectory."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import plot_topic4_fig5_manual_release_layout_v6 as previous

prior = previous.prior
ROOT = previous.ROOT
OUT = previous.BASE / 'layout_v7'
FIG = OUT / 'figures'
CMAP = plt.get_cmap('viridis')
NEUTRAL = '#414750'


def write(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f'{name}.png', dpi=180, bbox_inches='tight', pad_inches=.16)
    fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight', pad_inches=.16)
    plt.close(fig)


def time_arrows(ax, p, norm, count):
    normalized = p['coords'] / [.4, 500, 400]
    distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(normalized, axis=0), axis=1))]
    if distance[-1] <= 1e-10:
        return
    for fraction in np.linspace(.16, .85, count):
        k = int(np.searchsorted(distance, fraction * distance[-1]))
        j = min(int(np.searchsorted(distance, distance[k] + .065)), len(distance) - 1)
        if j > k:
            color = CMAP(norm((p['time'][k] + p['time'][j]) / 2))
            prior.arrow(ax, p['coords'][k], p['coords'][j], color,
                        size=15 if count == 4 else 12)


def plot_summary(ax, t, xyz, paths, run, windows, panel_letter=False):
    norm = Normalize(vmin=paths[0]['time'][0], vmax=paths[-1]['time'][-1])
    # Fixed artist order preserves the readable intervention overlay at crossings.
    ax.computed_zorder = False
    for k, p in enumerate(paths):
        width = 1.05 if k == 0 else 3.2 if k == 3 else 1.8
        order = 3 + 2 * k
        if k == 3:
            # A neutral outline keeps late yellow colors visible on the pale pane.
            ax.plot(*p['coords'].T, c='#707780', lw=width + 1.2,
                    alpha=.75, zorder=order)
        segments = np.stack([p['coords'][:-1], p['coords'][1:]], axis=1)
        midpoint_times = (p['time'][:-1] + p['time'][1:]) / 2
        line = Line3DCollection(segments, colors=CMAP(norm(midpoint_times)),
                                linewidths=width, zorder=order + 1,
                                capstyle='round', joinstyle='round')
        ax.add_collection3d(line)
        if k == 0:
            for time in [2., 4., 6., 8., 9.9]:
                ix = np.flatnonzero(np.abs(p['time'] - time) < .12)
                if len(ix) > 1:
                    j = min(int(ix[np.argmax(p['coords'][ix, 2])]), len(p['time']) - 2)
                    color = CMAP(norm((p['time'][j] + p['time'][j + 1]) / 2))
                    prior.arrow(ax, p['coords'][j], p['coords'][j + 1], color, size=10, lw=.8)
        else:
            time_arrows(ax, p, norm, count=4 if k == 3 else 2)

    # Numbered samples and intervention annotations are neutral, not stage colors.
    offsets = {1: (-35, 30), 2: (-28, -22), 3: (24, 16), 4: (18, -26)}
    for w in windows[:4]:
        point = prior.point_at(t, xyz, w['time'])
        ax.scatter(*point, s=34, facecolor='white', edgecolor=NEUTRAL,
                   lw=1.4, depthshade=False, zorder=30)
        prior.label(ax, str(w['number']), point, offsets[w['number']], NEUTRAL)
    for time, text, offset in [
        (run['restore_start_ms'] / 1000, 'Refill starts\n11.18 s', (155, 10)),
        (run['release_ms'] / 1000, 'Z released\n12.18 s', (-10, 100)),
    ]:
        point = prior.point_at(t, xyz, time)
        ax.scatter(*point, marker='s', s=28, facecolor=CMAP(norm(time)),
                   edgecolor=NEUTRAL, lw=.8, depthshade=False, zorder=30)
        prior.label(ax, text, point, offset, NEUTRAL, size=11, circle=False)

    ax.set(xlim=(.63, 1.03), ylim=(-15, 500), zlim=(-10, 410),
           xticks=[.65, .8, 1.], yticks=[0, 250, 500], zticks=[0, 200, 400])
    prior.style_3d(ax)
    title = 'Recurrent events, high-state entry and forced return'
    ax.set_title(('E  ' if panel_letter else '') + title, fontsize=16,
                 loc='left', weight='bold', pad=17)
    ax.set_anchor('N')
    cax = ax.inset_axes([.025, .965, .64, .026])
    cb = ax.figure.colorbar(ScalarMappable(norm=norm, cmap=CMAP), cax=cax,
                           orientation='horizontal', ticks=[2, 4, 6, 8, 10, 12])
    cb.set_label('Time, t (s)', fontsize=14, labelpad=2)
    cb.ax.tick_params(labelsize=12, length=3, pad=2)
    cb.outline.set_linewidth(.6)
    return dict(quantity='Simulation time (s)', normalization='linear', cmap=CMAP.name,
                vmin_s=float(norm.vmin), vmax_s=float(norm.vmax),
                segment_colors='Color at each segment midpoint time; one normalization for all paths',
                stage_color_legend=False, landmarks='Neutral numbered markers',
                forced_refill='Same time colormap; thicker line with neutral outline for readability')


def main():
    metadata = json.loads((previous.OUT / 'figure_metadata.json').read_text())
    windows = metadata['windows']
    a, run, *_ = prior.old.previous.source.load_main()
    t, xyz = prior.native_coordinates(a)
    paths = previous.complete_paths(t, xyz, run, windows)
    saved = np.load(previous.OUT / 'trajectory_arrays.npz')
    checks = dict(all_trajectory_samples_bitwise_equal_to_v6=all(
        np.array_equal(p['time'], saved[f'path{k}_time_s']) and
        np.array_equal(p['coords'], saved[f'path{k}_Z_I_E'])
        for k, p in enumerate(paths)))
    norm = Normalize(paths[0]['time'][0], paths[-1]['time'][-1])
    checks['shared_boundaries_have_identical_colors'] = all(
        np.array_equal(CMAP(norm(left['time'][-1])), CMAP(norm(right['time'][0])))
        for left, right in zip(paths[:-1], paths[1:]))
    checks['single_linear_time_normalization'] = bool(np.isclose(norm(7),
        (7 - norm.vmin) / (norm.vmax - norm.vmin)))
    assert all(checks.values()), checks

    fig = plt.figure(figsize=(11.5, 10.5))
    ax = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(left=.065, right=.91, top=.93, bottom=.055)
    color = plot_summary(ax, t, xyz, paths, run, windows)
    assert len(fig.axes) == 1 and len(ax.child_axes) == 1 and ax.get_legend() is None
    save(fig, 'single_native_state_trajectory')

    fig = plt.figure(figsize=(23, 14))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], left=.06, right=.952,
                         top=.925, bottom=.065, wspace=.2)
    left = prior.old.left_panels(fig, gs[0], a, run, windows)
    ax = fig.add_subplot(gs[1], projection='3d')
    plot_summary(ax, t, xyz, paths, run, windows, panel_letter=True)
    assert len([axis for axis in fig.axes if axis.name == '3d']) == 1
    fig.suptitle('Slow inhibitory depletion and spatial recruitment in a dual-core SNN',
                 fontsize=19, y=.99)
    save(fig, 'fig5_manual_core_release_layout_v7')

    write('figure_metadata.json', dict(source=str(previous.OUT), simulation_rerun=False,
          windows=windows, left=left, local_event_panels=False, phase_axis_count=1,
          color_encoding=color, rate_bin_ms=5, smoothing_sigma_ms=5, view=list(prior.VIEW),
          trajectory_arrays=str(previous.OUT / 'trajectory_arrays.npz'),
          coordinates=metadata['coordinates'], landmark_values=metadata['landmark_values'],
          intermediate_repeated_events_omitted_for_clarity=False,
          human_acceptance='PENDING_USER_REVIEW'))
    scripts = [Path(__file__), Path(previous.__file__), Path(prior.__file__),
               Path(prior.old.__file__), Path(prior.old.previous.__file__),
               Path(prior.old.previous.source.__file__)]
    write('producer_manifest.json', {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in scripts})
    (FIG / 'README.md').write_text('''### single_native_state_trajectory.png / .pdf
整条三维轨迹改为按实际仿真时间线性映射的viridis连续渐变，共用上方Time色条，不再用分阶段颜色或阶段图例。①–④改为中性编号，手动补回和释放时刻保留标注；补回段只用较粗线和灰色描边增强可读性，颜色仍服从同一时间映射。
**关注点**：全部坐标与v6逐点一致，①至②及高活动后返回的完整路径均保留；颜色编码时间，不表示活动强弱或预设状态类别。

### fig5_manual_core_release_layout_v7.png / .pdf
Figure 5右侧使用上述单一时间渐变相轨迹；左侧连续SEEG读出、raster、Z曲线和空间取样窗口沿用上一版。只调整相轨迹配色和相应标注，未重跑仿真。
**关注点**：右侧Time色条与左侧仿真秒数直接对应；候选图待用户目视审阅。
''', encoding='utf-8')
    write('artifact_qa.json', dict(numeric_checks=checks, numeric_status='PASS',
          agent_visual_review='PENDING', human_acceptance='PENDING_USER_REVIEW'))
    write('delivery_status.json', dict(status='RENDERED_PENDING_AGENT_VISUAL_REVIEW',
          simulation_rerun=False, human_acceptance='PENDING_USER_REVIEW'))
    print(json.dumps(dict(out=str(OUT), color_encoding=color, checks=checks)))


if __name__ == '__main__':
    main()
