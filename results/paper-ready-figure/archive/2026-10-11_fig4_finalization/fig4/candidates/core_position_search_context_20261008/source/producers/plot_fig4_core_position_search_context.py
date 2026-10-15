#!/usr/bin/env python3
"""Show the low-J_joint position density in the context of evaluated positions."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import shutil
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.paper_figures import plot_fig4_best_joint_positions as selection
from scripts.paper_figures import build_fig4_joint_core_density as kde
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d import proj3d
import numpy as np

read, write, sha = selection.read, selection.write, selection.sha
PURPLE = '#65509A'
STEM = 'core-position-search-context'


def mass_levels(density, cell_area):
    values = np.sort(density.ravel())[::-1]
    cumulative = np.cumsum(values) * cell_area
    return {str(mass): float(values[np.searchsorted(cumulative, mass * cumulative[-1])])
            for mass in (.5, .9)}


def draw(out, snapshot):
    records = snapshot['records']
    selected = snapshot['selections']['0.2']
    ids = set(selected['candidate_ids'])
    mask = np.array([r['candidate'] in ids for r in records])
    all_xy = np.array([r['centers_mm'] for r in records])
    best_xy = all_xy[mask]
    coords = np.linspace(0, 20, 321)
    xx, yy = np.meshgrid(coords, coords)
    cell_area = float((coords[1] - coords[0]) ** 2)
    per_core = np.array([kde.density(xx, yy, best_xy[:, core], sigma=snapshot['sigma_mm'])
                         for core in range(2)])
    z = per_core.mean(axis=0)
    integral = float(z.sum() * cell_area)
    assert abs(integral - 1) < 2e-6
    assert np.all(z >= 0)
    np.savez_compressed(out / f'source/{STEM}.npz', x_mm=coords, y_mm=coords,
                        density_per_mm2=z, per_core_density_per_mm2=per_core,
                        all_positions_mm=all_xy, selected_positions_mm=best_xy,
                        selected_mask=mask, candidate_ids=np.array([r['candidate'] for r in records]))
    geometry = read(out / 'source/electrode_geometry.json')
    names = np.array(geometry['contact_names'])
    contacts = np.array(geometry['contact_xy_mm'])
    limit_points = np.vstack([all_xy.reshape(-1, 2), contacts])
    lower = np.maximum(0, np.floor(limit_points.min(axis=0) - .7))
    upper = np.minimum(20, np.ceil(limit_points.max(axis=0) + .7))
    assert np.all(limit_points >= lower) and np.all(limit_points <= upper)
    # mplot3d does not clip a surface at x/y limits. Slice its display mesh so
    # the zero-density margin cannot cover tick labels outside the plane.
    view = np.ix_((coords >= lower[1]) & (coords <= upper[1]),
                  (coords >= lower[0]) & (coords <= upper[0]))

    current = read(ROOT / 'results/paper-ready-figure/fig4/current_version.json')
    registry_path = ROOT / current['registry']
    reference = read(registry_path)
    fonts = reference['font_pt']
    xy_label_pt = reference['original_panel_checks']['C_axis_label_font_pt']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': fonts['tick'],
                         'axes.labelsize': xy_label_pt, 'xtick.labelsize': fonts['tick'],
                         'ytick.labelsize': fonts['tick'], 'axes.linewidth': .65,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'savefig.facecolor': 'white', 'figure.facecolor': 'white'})
    fig = plt.figure(figsize=(7.1, 6.25), dpi=140)
    ax = fig.add_axes([.018, .12, .90, .88], projection='3d', computed_zorder=False)
    deep = np.array(to_rgb(PURPLE))
    cmap = LinearSegmentedColormap.from_list('single_purple',
        [np.ones(3) * (1-t) + deep * t for t in np.linspace(.015, 1, 256)])
    norm = Normalize(0, float(z.max()))
    # The density remains a nonnegative, equal-weight Gaussian mixture.
    surface = ax.plot_surface(xx[view], yy[view], z[view], rcount=len(coords), ccount=len(coords),
        cmap=cmap, norm=norm, linewidth=0, edgecolor='none',
        antialiased=True, shade=False, rasterized=True, zorder=2)
    ground_z = .00005
    ax.scatter(*all_xy.reshape(-1, 2).T, zs=ground_z, s=7, c='#A6A3AE',
               alpha=.5, linewidths=0, depthshade=False, zorder=3)

    for shaft in ('SCL', 'ICL'):
        ii = sorted(np.flatnonzero(np.char.startswith(names, shaft)),
                    key=lambda i: int(names[i][len(shaft):]), reverse=True)
        ax.plot(*contacts[ii].T, zs=ground_z, color='#85828B', lw=.85,
                marker='o', ms=2.8, mfc='white', mec='#85828B', mew=.65, zorder=4)

    contours = {}
    for core in range(2):
        levels = mass_levels(per_core[core], cell_area)
        contours['AB'[core]] = {}
        for mass, linestyle, width in ((.9, (0, (3, 2)), .85), (.5, '-', 1.1)):
            # Extract true 2D KDE contours, then project the paths onto z=0.
            auxiliary, planar = plt.subplots()
            curve = planar.contour(xx, yy, per_core[core], levels=[levels[str(mass)]])
            paths = [segment.copy() for segment in curve.allsegs[0] if len(segment) > 2]
            plt.close(auxiliary)
            for path in paths:
                ax.plot(path[:, 0], path[:, 1], zs=ground_z, color=PURPLE,
                        lw=width, ls=linestyle, alpha=.85, zorder=5)
            enclosed = float(per_core[core][per_core[core] >= levels[str(mass)]].sum() * cell_area)
            assert abs(enclosed - mass) < .004
            contours['AB'[core]][str(mass)] = dict(threshold_per_mm2=levels[str(mass)],
                enclosed_mass=enclosed, paths_mm=[path.tolist() for path in paths])

    # Every selected position is shown, including isolated low-loss candidates.
    ax.scatter(*best_xy.reshape(-1, 2).T, zs=ground_z, s=9, c='#574184',
               alpha=.9, edgecolors='white', linewidths=.2, depthshade=False, zorder=6)
    # Trim only unobserved margins; retain every evaluated position and contact.
    ticks = np.arange(0, 21, 5)
    zmax = float(z.max())
    ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), zlim=(0, zmax * 1.2),
           xticks=ticks[(ticks >= lower[0]) & (ticks <= upper[0])],
           yticks=ticks[(ticks >= lower[1]) & (ticks <= upper[1])],
           zticks=[], xlabel='x (mm)', ylabel='y (mm)')
    ax.view_init(elev=72, azim=-90.000001)
    ax.set_proj_type('ortho')
    ax.set_box_aspect((upper[0]-lower[0], upper[1]-lower[1], 6.0))
    ax.grid(False)
    ax.xaxis.labelpad = 2
    ax.yaxis.labelpad = 2
    ax.yaxis.set_rotate_label(False)
    ax.yaxis.label.set_rotation(90)
    ax.tick_params(axis='x', pad=-4)
    ax.tick_params(axis='y', pad=0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor((1, 1, 1, 0))
        axis.line.set_color('#88858E')
        axis.line.set_linewidth(.65)
        axis._axinfo['tick']['inward_factor'] = 0.
        axis._axinfo['tick']['outward_factor'] = .12
    ax.zaxis.line.set_visible(False)
    for core in range(2):
        iy, ix = np.unravel_index(np.argmax(per_core[core]), z.shape)
        ax.text(xx[iy, ix], yy[iy, ix]+(.28 if core == 0 else .85), z[iy, ix] + zmax*.07,
                'Core '+'AB'[core], fontsize=fonts['legend'], color='#403949',
                ha='center', va='bottom', zorder=8)

    # Use the projected ground plane rather than the invisible 3D axes box to
    # set the colorbar height and the compact export bounds.
    fig.canvas.draw()
    px, py, _ = proj3d.proj_transform(
        np.array([lower[0], upper[0], upper[0], lower[0]]),
        np.array([lower[1], lower[1], upper[1], upper[1]]), np.zeros(4), ax.get_proj())
    plane_pixels = ax.transData.transform(np.column_stack([px, py]))
    plane_fraction = fig.transFigure.inverted().transform(plane_pixels)
    lo, hi = plane_fraction.min(axis=0), plane_fraction.max(axis=0)
    cax = fig.add_axes([hi[0]+.027, lo[1], .016, hi[1]-lo[1]])
    bar = fig.colorbar(surface, cax=cax, orientation='vertical')
    bar.set_alpha(1)
    bar._draw_all()
    bar.set_ticks([x for x in (0., .1, .2, .3, .4, .5) if x <= zmax])
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=2.3, width=.6, pad=3, labelsize=fonts['tick'], colors='#56515D')
    bar.set_label(r'Position density (mm$^{-2}$)', fontsize=fonts['label'], labelpad=8, color='#403949')
    legend = fig.legend(handles=[
        Line2D([], [], marker='o', color='none', mfc='#A6A3AE', mec='none', ms=4, label='All evaluated'),
        Line2D([], [], marker='o', color='none', mfc='#574184', mec='none', ms=4, label='Best 20%'),
        Line2D([], [], color=PURPLE, lw=1.1, label='50% density'),
        Line2D([], [], color=PURPLE, lw=.85, ls=(0, (3, 2)), label='90% density')],
        bbox_to_anchor=(hi[0]-.012, hi[1]-.014), bbox_transform=fig.transFigure,
        loc='upper right', ncol=1, frameon=False, fontsize=fonts['legend'],
        borderaxespad=0, labelspacing=.6, handlelength=1.65, handletextpad=.7)

    labels = [ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels()]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    plane_bbox = Bbox.from_extents(*plane_pixels.min(axis=0), *plane_pixels.max(axis=0))
    bar_bbox = cax.get_window_extent(renderer)
    np.testing.assert_allclose([bar_bbox.y0, bar_bbox.y1], [plane_bbox.y0, plane_bbox.y1], atol=1e-7, rtol=0)
    boxes = [plane_bbox, cax.get_tightbbox(renderer), legend.get_window_extent(renderer),
             *[t.get_window_extent(renderer) for t in labels if t.get_visible() and t.get_text()]]
    crop = Bbox.union(boxes).transformed(fig.dpi_scale_trans.inverted()).padded(.08)
    assert crop.width < 8 and crop.height < 7
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(out / f'figures/{STEM}.{ext}', dpi=320, bbox_inches=crop)
    plt.close(fig)
    write(out / 'density_contours.json', dict(method='Per-core highest-density mass contours; not confidence intervals.',
                                             sigma_mm=snapshot['sigma_mm'], contours=contours))
    return dict(stem=STEM, display_limits_mm=[lower.tolist(), upper.tolist()],
        all_evaluated_configurations=len(records), plotted_background_positions=all_xy.size//2,
        selected_configurations=len(best_xy), plotted_selected_positions=best_xy.size//2,
        per_core_density_integrals=(per_core.sum(axis=(1, 2))*cell_area).tolist(),
        displayed_mixture_integral=integral, sigma_mm=snapshot['sigma_mm'], zmax=zmax,
        z_is_density=True, negative_values=False, color_normalization='linear', surface_alpha=1.,
        density_smoothing_includes_all_selected=True, no_position_jitter=True,
        xy_plane_height_mm=plane_bbox.height / fig.dpi * 25.4,
        colorbar_height_mm=bar_bbox.height / fig.dpi * 25.4,
        colorbar_alignment_error_pixels=float(max(abs(bar_bbox.y0-plane_bbox.y0), abs(bar_bbox.y1-plane_bbox.y1))),
        title_or_caption_count=len(fig.texts), typography=dict(font_family='DejaVu Sans',
            xy_label_pt=xy_label_pt, tick_pt=fonts['tick'], legend_pt=fonts['legend'],
            reference=str(registry_path), reference_sha256=sha(registry_path)),
        legend_labels=[t.get_text() for t in legend.get_texts()])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT / 'results/paper-ready-figure/fig4/candidates/core_position_search_context_20261008')
    args = parser.parse_args()
    out = args.out.resolve()
    for folder in ('source/scores', 'source/selected_audits', 'source/producers', 'figures', 'qa'):
        (out / folder).mkdir(parents=True, exist_ok=True)
    snapshot = selection.freeze(out)
    metrics = draw(out, snapshot)
    selected = snapshot['selections']['0.2']
    for producer in (Path(__file__), Path(selection.__file__), Path(kde.__file__)):
        shutil.copy2(producer, out / 'source/producers' / producer.name)
    write(out / 'figure_registry.json', dict(status='CANDIDATE_PENDING_AUTHOR_VISUAL_REVIEW',
        generated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
        source_snapshot_time=snapshot['asof'], pool_completed=snapshot['pool_completed'],
        pool_scorable=snapshot['pool_scorable'], main_selection=selected, rendering=metrics,
        producer=str(Path(__file__)), producer_sha256=sha(__file__),
        source_snapshot_sha256=sha(out / 'source/snapshot.json'),
        interpretation='Descriptive concentration of low-loss evaluated configurations relative to observed search coverage; not independent convergence, a posterior, or an objective-function landscape.',
        formal_current_figure_replaced=False,
        outputs={p.name: sha(p) for p in sorted((out / 'figures').glob('*')) if p.suffix in ('.png', '.pdf', '.svg')}))
    (out / 'figures/README.md').write_text(
        f"### {STEM}.png\n"
        f"灰点展示{snapshot['pool_completed']}个已完成配置的Core A/B位置，紫点和三维密度展示{snapshot['pool_scorable']}个可评分配置中最低20%的{selected['count']}个；同源PDF/SVG可直接导出。"
        f"每个配置贡献两个位置，均采用原始坐标且不抖动；低损失子集含{selected['historical']}个历史、{selected['new']}个扩围配置，统一按冻结J_joint排序（阈值{selected['cutoff_J_joint']:.6f}）。"
        "紫色曲面为两核等权位置密度，σ=0.35 mm；实线与虚线分别为每核50%和90%高密度包络，含义不是置信区间，远处优解也保留为紫点及低密度峰。\n"
        "**关注点**：比较真实搜索覆盖与低损失位置集中程度；同一拓扑／噪声下的候选并非独立优化重复，不据此声称固定位置或唯一收敛解。\n")
    print(out, flush=True)


if __name__ == '__main__':
    main()
