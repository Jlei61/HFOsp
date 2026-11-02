#!/usr/bin/env python3
"""Assemble Fig. 5 with the continuous trajectory's time colorbar on its right."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import plot_topic4_fig5_manual_release_layout_v8 as previous

ROOT = previous.ROOT
OUT = previous.OUT.parent / 'layout_v9'
FIG = OUT / 'figures'


def write(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f'{name}.png', dpi=180, bbox_inches='tight', pad_inches=.16)
    fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight', pad_inches=.16)
    plt.close(fig)


def plot_summary(ax, t, xyz, paths, run, windows, panel_letter=False):
    color = previous.plot_summary(ax, t, xyz, paths, run, windows,
                                   panel_letter=panel_letter)
    assert len(ax.child_axes) == 1
    ax.child_axes[0].remove()
    cax = ax.inset_axes([1.025, .20, .027, .65])
    norm = Normalize(color['vmin_s'], color['vmax_s'])
    cb = ax.figure.colorbar(ScalarMappable(norm=norm, cmap=color['cmap']), cax=cax,
                           orientation='vertical', ticks=[2, 4, 6, 8, 10, 12])
    cb.set_label('Time, t (s)', fontsize=14, labelpad=15, rotation=270)
    cb.ax.tick_params(labelsize=12, length=3, pad=3)
    cb.outline.set_linewidth(.6)
    assert len(ax.child_axes) == 1
    return dict(**color, orientation='vertical', location='right of the 3D trajectory',
                inset_bounds=[1.025, .20, .027, .65])


def main():
    metadata = json.loads((previous.OUT / 'figure_metadata.json').read_text())
    windows = metadata['windows']
    a, run, *_ = previous.prior.old.previous.source.load_main()
    t, xyz, _ = previous.current_coordinates(a)
    paths = previous.previous.previous.complete_paths(t, xyz, run, windows)
    saved = np.load(previous.OUT / 'trajectory_arrays.npz')
    checks = dict(all_trajectory_coordinates_and_times_unchanged=all(
        np.array_equal(p['time'], saved[f'path{k}_time_s']) and
        np.array_equal(p['coords'], saved[f'path{k}_Z_H_E']) for k, p in enumerate(paths)))
    assert all(checks.values()), checks

    fig = plt.figure(figsize=(12, 10.5))
    ax = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(left=.065, right=.85, top=.93, bottom=.055)
    color = plot_summary(ax, t, xyz, paths, run, windows)
    save(fig, 'single_native_inhibition_state_trajectory')

    fig = plt.figure(figsize=(24, 14))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], left=.06, right=.925,
                         top=.925, bottom=.065, wspace=.2)
    left = previous.prior.old.left_panels(fig, gs[0], a, run, windows)
    ax = fig.add_subplot(gs[1], projection='3d')
    plot_summary(ax, t, xyz, paths, run, windows, panel_letter=True)
    fig.suptitle('Slow inhibitory depletion and spatial recruitment in a dual-core SNN',
                 fontsize=19, y=.99)
    assert len([axis for axis in fig.axes if axis.name == '3d']) == 1
    save(fig, 'fig5_manual_core_release_layout_v9')

    metadata.update(source=str(previous.OUT), color_encoding=color, left=left,
                    trajectory_arrays=str(previous.OUT / 'trajectory_arrays.npz'),
                    human_acceptance='PENDING_USER_REVIEW',
                    layout_revision='Vertical time colorbar to the right of E; full Fig. 5 assembled.')
    write('figure_metadata.json', metadata)
    manifest = json.loads((previous.OUT / 'producer_manifest.json').read_text())
    manifest[str(Path(__file__).relative_to(ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write('producer_manifest.json', manifest)
    (FIG / 'README.md').write_text('''### fig5_manual_core_release_layout_v9.png / .pdf
完整Figure 5拼版：左侧保留连续SEEG读出、raster、Z及空间快照，右侧使用E放电率—实际抑制电流—Z的单张三维轨迹。时间渐变色条移至三维图右侧竖排，完整轨迹、①–④及手动补回标注沿用v8。
**关注点**：核对整图阅读顺序、时间色条与三维轴标签的间距；所有轨迹坐标及时间与v8逐点一致，候选待用户目视审阅。

### single_native_inhibition_state_trajectory.png / .pdf
右侧时间色条版本的独立三维图，与完整拼版共用绘制函数、坐标和颜色归一化。颜色始终表示实际仿真时间。
**关注点**：保留放电—抑制回环和手动补回的可读性，无下方额外相图。
''', encoding='utf-8')
    write('artifact_qa.json', dict(numeric_checks=checks, numeric_status='PASS',
          agent_visual_review='PENDING', human_acceptance='PENDING_USER_REVIEW'))
    write('delivery_status.json', dict(status='RENDERED_PENDING_AGENT_VISUAL_REVIEW',
          simulation_rerun=False, human_acceptance='PENDING_USER_REVIEW'))
    print(json.dumps(dict(out=str(OUT), checks=checks)))


if __name__ == '__main__':
    main()
