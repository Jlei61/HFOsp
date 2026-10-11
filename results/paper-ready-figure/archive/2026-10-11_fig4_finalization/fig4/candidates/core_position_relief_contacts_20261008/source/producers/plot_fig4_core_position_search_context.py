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
# Load the current geometry helpers before historical figure producers add
# their archived worktree modules to the package search path.
from scripts.paper_figures import plot_fig2_e1146_template_projection_composite as projection
from scripts.paper_figures import plot_fig4_spatial_edge_flow_validation as spatial
from scripts.paper_figures import plot_fig4_best_joint_positions as selection
from scripts.paper_figures import build_fig4_joint_core_density as kde
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LightSource, LinearSegmentedColormap, Normalize, to_rgb
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


def straight_electrode_display(out, geometry):
    """Reuse the Fig2A/Fig4 3-D shaft fit, registered to this frozen sheet."""
    path = out / 'source/electrode_straight_display.json'
    if not path.exists():
        context = spatial._figure2a_context_geometry({'static': {
            'contact_names': geometry['contact_names'],
            'contact_xy_mm': np.asarray(geometry['contact_xy_mm'])}})
        for key, value in list(context.items()):
            if isinstance(value, np.ndarray):
                context[key] = value.tolist()
        context['groups'] = [group.tolist() for group in context['groups']]
        context['method'] = 'Per-shaft 3-D SVD line fit using all local contacts, then shared-plane projection and affine registration to the frozen SNN sheet; display only.'
        context['producer_sources'] = {}
        for producer in (Path(spatial.__file__), Path(projection.__file__)):
            shutil.copy2(producer, out / 'source/producers' / producer.name)
            context['producer_sources'][str(producer)] = sha(producer)
        write(path, context)
    context = read(path)
    indices = [context['names'].index(name) for name in geometry['contact_names']]
    measured = np.asarray(context['measured_sheet_xy'])[indices]
    fitted = np.asarray(context['rod_sheet_xy'])[indices]
    np.testing.assert_allclose(measured, geometry['contact_xy_mm'], rtol=0, atol=1e-6)
    return fitted, dict(method=context['method'], source=str(path), source_sha256=sha(path),
        registration_max_abs_error_mm=context['registration_max_abs_error_mm'],
        straight_shaft_fit=context['straight_shaft_fit'],
        contact_names=geometry['contact_names'], displayed_contact_xy_mm=fitted.tolist(),
        max_display_displacement_mm=float(np.linalg.norm(fitted-measured, axis=1).max()),
        simulation_contact_coordinates_changed=False)


def draw(out, snapshot, simplify=False, peak_guides=False, peak_labels_only=False, relief=False):
    peak_layout = peak_guides or peak_labels_only
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
    rod_contacts, rod_display = straight_electrode_display(out, geometry)
    limit_points = np.vstack([all_xy.reshape(-1, 2), contacts])
    lower = np.maximum(0, np.floor(limit_points.min(axis=0) - .7))
    upper = np.minimum(20, np.ceil(limit_points.max(axis=0) + .7))
    assert np.all(limit_points >= lower) and np.all(limit_points <= upper)
    assert np.all(rod_contacts >= lower) and np.all(rod_contacts <= upper)
    # mplot3d does not clip a surface at x/y limits. Slice its display mesh so
    # the zero-density margin cannot cover tick labels outside the plane.
    view = np.ix_((coords >= lower[1]) & (coords <= upper[1]),
                  (coords >= lower[0]) & (coords <= upper[0]))

    current = read(ROOT / 'results/paper-ready-figure/fig4/current_version.json')
    registry_path = ROOT / current['registry']
    reference = read(registry_path)
    fonts = dict(reference['font_pt'])
    typography_path = out / 'source/typography.json'
    typography = read(typography_path) if typography_path.exists() else {}
    for name in ('tick', 'legend'):
        fonts[name] = typography.get(name+'_pt', fonts[name])
    xy_label_pt = typography.get('xy_label_pt',
        reference.get('original_panel_checks', {}).get('C_axis_label_font_pt', fonts['label']))
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
    common_z_extent = 10.5 if relief else (9.0 if peak_layout else 6.0)
    elevation = 64 if relief else 72
    ground_z = .00005
    background_xy = all_xy.reshape(-1, 2)
    surface_z = z[view]
    display_floor = 0.
    if simplify:
        # Coverage is the set of tested locations. Repeated coordinates do not
        # need repeated ink; every configuration still contributes to the KDE.
        background_xy = np.unique(background_xy, axis=0)
        gx, gy = np.meshgrid([lower[0], upper[0]], [lower[1], upper[1]])
        ax.plot_surface(gx, gy, np.zeros_like(gx), color=cmap(0.), shade=False,
                        linewidth=0, antialiased=False, zorder=.5)
        ax.scatter(*background_xy.T, zs=ground_z, s=3.2, c='#B6B2BF',
                   alpha=.36, linewidths=0, depthshade=False, zorder=1)
        # Draw only imperceptible Gaussian tails as the neutral base, so the
        # density surface can occlude background search dots.
        display_floor = float(z.max()) * 1e-4
        surface_z = np.where(surface_z >= display_floor, surface_z, np.nan)
    # Keep the same linear density mapping and a common geometric height scale.
    # Soft illumination supplies slope cues without smoothing or rescaling B.
    surface_colors = dict(cmap=cmap, norm=norm)
    if relief:
        rgba = cmap(norm(z))
        light = LightSource(azdeg=315, altdeg=50)
        rgba[..., :3] = light.shade_rgb(rgba[..., :3],
            z * common_z_extent / (float(z.max()) * 1.2),
            dx=float(coords[1]-coords[0]), dy=float(coords[1]-coords[0]),
            blend_mode='soft', fraction=.65)
        surface_colors = dict(facecolors=rgba[view])
    surface = ax.plot_surface(xx[view], yy[view], surface_z, rcount=len(coords), ccount=len(coords),
        **surface_colors, linewidth=0, edgecolor='none',
        antialiased=not simplify, shade=False, rasterized=True, zorder=2)
    color_mappable = ScalarMappable(norm=norm, cmap=cmap) if relief else surface
    if not simplify:
        ax.scatter(*background_xy.T, zs=ground_z, s=7, c='#A6A3AE',
                   alpha=.5, linewidths=0, depthshade=False, zorder=3)

    rod_display['collinearity_max_residual_mm'] = {}
    rod_zorder = 7 if relief else (1.5 if simplify else 4)
    rod_color = '#948F9D' if relief else ('#BCBAC3' if simplify else '#85828B')
    rod_display.update(overlay_above_density=relief, ground_z=ground_z,
                       artist_zorder=rod_zorder,
                       displayed_contact_count=len(rod_contacts))
    for shaft in ('SCL', 'ICL'):
        ii = sorted(np.flatnonzero(np.char.startswith(names, shaft)),
                    key=lambda i: int(names[i][len(shaft):]), reverse=True)
        points = rod_contacts[ii]
        direction = points[-1] - points[0]
        direction /= np.linalg.norm(direction)
        along = (points-points[0]) @ direction
        residual = np.linalg.norm(points-points[0]-along[:, None]*direction, axis=1)
        assert residual.max() < 1e-10
        rod_display['collinearity_max_residual_mm'][shaft] = float(residual.max())
        # A shaft is a single fitted segment. Carrier contact markers lie on
        # that display line; the original sampling coordinates stay frozen.
        ax.plot(*points[[0, -1]].T, zs=ground_z, color=rod_color,
                lw=.6 if relief else (.55 if simplify else .85), zorder=rod_zorder)
        ax.plot(*points.T, zs=ground_z, linestyle='none', marker='o',
                ms=2.6 if relief else (2 if simplify else 2.8),
                mfc='white', mec=rod_color,
                mew=.65 if relief else (.5 if simplify else .65), zorder=rod_zorder+.1)

    contours = {}
    isolated_xy = []
    for core in range(2):
        levels = mass_levels(per_core[core], cell_area)
        contours['AB'[core]] = {}
        specifications = ((.9, '-', .6),) if simplify else ((.9, (0, (3, 2)), .85), (.5, '-', 1.1))
        for mass, linestyle, width in specifications:
            # Extract true 2D KDE contours, then project the paths onto z=0.
            auxiliary, planar = plt.subplots()
            curve = planar.contour(xx, yy, per_core[core], levels=[levels[str(mass)]])
            paths = [segment.copy() for segment in curve.allsegs[0] if len(segment) > 2]
            plt.close(auxiliary)
            for path in paths:
                ax.plot(path[:, 0], path[:, 1], zs=ground_z,
                        color='#9784B6' if simplify else PURPLE,
                        lw=width, ls=linestyle, alpha=.8 if simplify else .85, zorder=5)
            enclosed = float(per_core[core][per_core[core] >= levels[str(mass)]].sum() * cell_area)
            assert abs(enclosed - mass) < .004
            contours['AB'[core]][str(mass)] = dict(threshold_per_mm2=levels[str(mass)],
                enclosed_mass=enclosed, paths_mm=[path.tolist() for path in paths])
        if simplify:
            values = kde.density(best_xy[:, core, 0], best_xy[:, core, 1],
                                 best_xy[:, core], sigma=snapshot['sigma_mm'])
            isolated_xy.extend(best_xy[values < levels['0.9'], core].tolist())

    # All selected positions remain in the surface, including isolated ones.
    if not simplify:
        ax.scatter(*best_xy.reshape(-1, 2).T, zs=ground_z, s=9, c='#574184',
                   alpha=.9, edgecolors='white', linewidths=.2, depthshade=False, zorder=6)
    else:
        # Sparse exceptions should remain identifiable even when their KDE
        # peaks are small on the common scale; do not add dots over the peaks.
        isolated_xy = np.unique(np.asarray(isolated_xy).reshape(-1, 2), axis=0)
        ax.scatter(*isolated_xy.T, zs=ground_z, s=12, facecolors='none',
                   edgecolors=PURPLE, linewidths=.65, depthshade=False, zorder=6)
    # Trim only unobserved margins; retain every evaluated position and contact.
    ticks = np.arange(0, 21, 5)
    zmax = float(z.max())
    ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), zlim=(0, zmax * 1.2),
           xticks=ticks[(ticks >= lower[0]) & (ticks <= upper[0])],
           yticks=ticks[(ticks >= lower[1]) & (ticks <= upper[1])],
           zticks=[], xlabel='x (mm)', ylabel='y (mm)')
    ax.view_init(elev=elevation, azim=-90.000001)
    ax.set_proj_type('ortho')
    ax.set_box_aspect((upper[0]-lower[0], upper[1]-lower[1], common_z_extent))
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
    peak_annotations = []
    peak_coordinates = {}
    for core in range(2):
        iy, ix = np.unravel_index(np.argmax(per_core[core]), z.shape)
        peak = [float(xx[iy, ix]), float(yy[iy, ix]), float(z[iy, ix])]
        peak_coordinates['AB'[core]] = dict(x_mm=peak[0], y_mm=peak[1],
            displayed_density_per_mm2=peak[2], per_core_density_per_mm2=float(per_core[core, iy, ix]))
        if peak_layout:
            if peak_guides:
                ax.scatter([peak[0]], [peak[1]], [peak[2]], s=15,
                           facecolors='#574184', edgecolors='white', linewidths=.6,
                           depthshade=False, zorder=8)
            px, py, _ = proj3d.proj_transform(*peak, ax.get_proj())
            peak_annotations.append(ax.annotate('Core '+'AB'[core], xy=(px, py),
                xytext=(0, 17 if core == 0 else 21), textcoords='offset points',
                fontsize=fonts['legend'], color='#403949', ha='center', va='bottom',
                arrowprops=(dict(arrowstyle='-', color='#786294', lw=.6, shrinkA=3, shrinkB=3)
                            if peak_guides else None),
                annotation_clip=False, zorder=9))
        else:
            ax.text(peak[0], peak[1]+(.28 if core == 0 else .85), peak[2] + zmax*.07,
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
    bar = fig.colorbar(color_mappable, cax=cax, orientation='vertical')
    bar.set_alpha(1)
    bar._draw_all()
    bar.set_ticks([x for x in (0., .1, .2, .3, .4, .5) if x <= zmax])
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=2.3, width=.6, pad=3, labelsize=fonts['tick'], colors='#56515D')
    bar.set_label(r'Position density (mm$^{-2}$)', fontsize=fonts['label'], labelpad=8, color='#403949')
    handles = [
        Line2D([], [], marker='o', color='none', mfc='#A6A3AE', mec='none', ms=4, label='All evaluated'),
        Line2D([], [], marker='o', color='none', mfc='#574184', mec='none', ms=4, label='Best 20%'),
        Line2D([], [], color=PURPLE, lw=1.1, label='50% density'),
        Line2D([], [], color=PURPLE, lw=.85, ls=(0, (3, 2)), label='90% density')]
    if simplify:
        handles = [Line2D([], [], marker='o', color='none', mfc='#B6B2BF', mec='none',
                          ms=2.5, label='Search positions'),
                   Line2D([], [], color='#9784B6', lw=.6, label='90% density'),
                   Line2D([], [], marker='o', color='none', mfc='none', mec=PURPLE,
                          mew=.65, ms=3.2, label='Outlying best 20%')]
        if peak_layout:
            handles = [handles[0], handles[2], handles[1]]
    legend = fig.legend(handles=handles,
        bbox_to_anchor=(hi[0]-.012, hi[1]-.014), bbox_transform=fig.transFigure,
        loc='upper right', ncol=1, frameon=peak_layout, fontsize=fonts['legend'],
        fancybox=False, framealpha=1., facecolor='white', edgecolor='#B7B2BF',
        borderaxespad=0, borderpad=.65, labelspacing=.6, handlelength=1.65, handletextpad=.7)
    if peak_layout:
        legend.get_frame().set_linewidth(.65)

    labels = [ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels()]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    plane_bbox = Bbox.from_extents(*plane_pixels.min(axis=0), *plane_pixels.max(axis=0))
    bar_bbox = cax.get_window_extent(renderer)
    np.testing.assert_allclose([bar_bbox.y0, bar_bbox.y1], [plane_bbox.y0, plane_bbox.y1], atol=1e-7, rtol=0)
    boxes = [plane_bbox, cax.get_tightbbox(renderer), legend.get_window_extent(renderer),
             *[text.get_window_extent(renderer) for text in peak_annotations],
             *[t.get_window_extent(renderer) for t in labels if t.get_visible() and t.get_text()]]
    crop = Bbox.union(boxes).transformed(fig.dpi_scale_trans.inverted()).padded(.08)
    assert crop.width < 8 and crop.height < 7
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(out / f'figures/{STEM}.{ext}', dpi=320, bbox_inches=crop)
    plt.close(fig)
    write(out / 'density_contours.json', dict(method='Per-core highest-density mass contours; not confidence intervals.',
                                             sigma_mm=snapshot['sigma_mm'], contours=contours))
    return dict(stem=STEM, display_limits_mm=[lower.tolist(), upper.tolist()],
        all_evaluated_configurations=len(records), plotted_background_positions=len(background_xy),
        background_repeated_coordinates_collapsed=simplify,
        selected_configurations=len(best_xy), plotted_selected_positions=len(isolated_xy) if simplify else best_xy.size//2,
        selected_positions_in_density=best_xy.size//2,
        isolated_selected_marker_rule='Selected positions outside their per-core 90 percent KDE mass contour' if simplify else None,
        isolated_selected_positions_mm=isolated_xy.tolist() if simplify else [],
        display_mode='simplified' if simplify else 'full', electrode_display=rod_display,
        visible_contour_masses=[.9] if simplify else [.5, .9],
        display_density_floor_per_mm2=display_floor,
        mass_above_display_floor=float(z[z >= display_floor].sum()*cell_area),
        per_core_density_integrals=(per_core.sum(axis=(1, 2))*cell_area).tolist(),
        displayed_mixture_integral=integral, sigma_mm=snapshot['sigma_mm'], zmax=zmax,
        z_is_density=True, negative_values=False, color_normalization='linear', surface_alpha=1.,
        peak_guides=peak_guides, peak_labels_only=peak_labels_only,
        peak_locations=peak_coordinates, common_z_box_extent=common_z_extent,
        camera_elevation_deg=elevation, camera_azimuth_deg=-90.000001,
        surface_illumination=(dict(azimuth_deg=315, altitude_deg=50, blend_mode='soft', fraction=.65,
            note='Display-only lighting applied to the linear density colormap; colorbar shows the unlit density scale.')
            if relief else None),
        peak_definition='Per-core KDE maximum on the unchanged 0.0625-mm grid; not the best-scoring candidate or a confidence estimate.',
        density_smoothing_includes_all_selected=True, no_position_jitter=True,
        xy_plane_height_mm=plane_bbox.height / fig.dpi * 25.4,
        colorbar_height_mm=bar_bbox.height / fig.dpi * 25.4,
        colorbar_alignment_error_pixels=float(max(abs(bar_bbox.y0-plane_bbox.y0), abs(bar_bbox.y1-plane_bbox.y1))),
        title_or_caption_count=len(fig.texts), typography=dict(font_family='DejaVu Sans',
            xy_label_pt=xy_label_pt, tick_pt=fonts['tick'], legend_pt=fonts['legend'],
            reference=typography.get('reference', str(registry_path)),
            reference_sha256=typography.get('reference_sha256', sha(registry_path))),
        legend_labels=[t.get_text() for t in legend.get_texts()], legend_frame_visible=legend.get_frame_on())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT / 'results/paper-ready-figure/fig4/candidates/core_position_search_context_20261008')
    parser.add_argument('--snapshot-dir', type=Path, help='Reuse an existing frozen source pool.')
    parser.add_argument('--simplify', action='store_true', help='Remove selected dots and the inner contour; put coverage behind the surface.')
    parser.add_argument('--relief', action='store_true', help='Show surface slope with a lower camera and soft illumination; overlay the ground-plane electrode carriers.')
    peak_style = parser.add_mutually_exclusive_group()
    peak_style.add_argument('--peak-guides', action='store_true', help='Identify KDE summits and group point legend entries inside a frame.')
    peak_style.add_argument('--peak-labels-only', action='store_true', help='Keep the summit layout and framed legend, with text labels but no summit markers or leaders.')
    args = parser.parse_args()
    out = args.out.resolve()
    if args.snapshot_dir is not None and not (out / 'source/snapshot.json').exists():
        shutil.copytree(args.snapshot_dir / 'source', out / 'source')
    for folder in ('source/scores', 'source/selected_audits', 'source/producers', 'figures', 'qa'):
        (out / folder).mkdir(parents=True, exist_ok=True)
    if args.snapshot_dir is not None and not (out / 'source/typography.json').exists():
        prior_style = read(args.snapshot_dir / 'figure_registry.json')['rendering']['typography']
        write(out / 'source/typography.json', prior_style)
    snapshot = selection.freeze(out)
    metrics = draw(out, snapshot, simplify=args.simplify, peak_guides=args.peak_guides,
                   peak_labels_only=args.peak_labels_only, relief=args.relief)
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
    summary = (f"浅灰小点展示{snapshot['pool_completed']}个已完成配置的搜索位置，重复坐标只画一次；紫色曲面保留{snapshot['pool_scorable']}个有效评分中前20%的{selected['count']}个配置，同源PDF/SVG可导出。"
               if args.simplify else
               f"灰点展示{snapshot['pool_completed']}个已完成配置的Core A/B位置，紫点和三维密度展示{snapshot['pool_scorable']}个可评分配置中最低20%的{selected['count']}个；同源PDF/SVG可直接导出。")
    density_description = (f"紫色曲面为两核等权位置密度，σ=0.35 mm且与上一版数值完全一致；仅保留每核90%高密度细轮廓，去掉聚集区优解散点和50%内圈，背景搜索点由前景曲面遮挡，仅轮廓外{metrics['plotted_selected_positions']}个优解位置保留细空心标记以免漏掉远处优解。\n"
                           if args.simplify else
                           "紫色曲面为两核等权位置密度，σ=0.35 mm；实线与虚线分别为每核50%和90%高密度包络，含义不是置信区间，远处优解也保留为紫点及低密度峰。\n")
    (out / 'figures/README.md').write_text(
        f"### {STEM}.png\n"
        + summary +
        f"每个配置贡献两个位置，均采用原始坐标且不抖动；低损失子集含{selected['historical']}个历史、{selected['new']}个扩围配置，统一按冻结J_joint排序（阈值{selected['cutoff_J_joint']:.6f}）。"
        + density_description
        + "浅灰电极杆沿用Fig2A及早期Fig4的逐杆三维SVD直线拟合后投影，空心触点采用拟合后的显示坐标；仿真采样坐标及core密度数据保持不变。\n"
        + ("视角改为64°俯仰并适度增加共同垂直显示比例，两核统一采用柔和坡面光照；色条表示光照前的线性密度映射，明暗辅助辨认坡度，未单独放大Core B。电极杆和全部15个触点作为底面位置标注叠加显示，不受曲面遮挡，XY和底面Z坐标未移动。\n" if args.relief else "")
        + ("峰顶实心标记及引线标出每核高斯核密度的最大值位置，两核共用同一垂直比例；图例加细边框，搜索点与Outlier点相邻，90%轮廓线置于最后。\n" if args.peak_guides else "")
        + ("Core A/B仅保留文字标签，去掉峰顶圆点和指示线；两核共用同一垂直比例，图例保留细边框，搜索点与Outlier点相邻，90%轮廓线置于最后。\n" if args.peak_labels_only else "") +
        "**关注点**：比较真实搜索覆盖与低损失位置集中程度；同一拓扑／噪声下的候选并非独立优化重复，不据此声称固定位置或唯一收敛解。\n")
    print(out, flush=True)


if __name__ == '__main__':
    main()
