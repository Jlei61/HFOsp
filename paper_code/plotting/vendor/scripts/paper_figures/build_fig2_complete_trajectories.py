"""Complete Figure 2: frozen latest A/B/D, with supported rank trajectories in C."""
from __future__ import annotations
from paper_paths import CODE_ROOT, INPUT_ROOT, OUTPUT_ROOT
import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path
ROOT = INPUT_ROOT
pass
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator
import numpy as np
import pymupdf as pdf
from scipy.ndimage import label
from scipy.spatial import cKDTree
from plotting.vendor.scripts.paper_figures.build_fig2_sequence_field_layout import export, place
from plotting.vendor.scripts.paper_figures.plot_fig2_exemplar_trajectory import display_trajectory, mpl_path, plain, TEMPLATE_COLORS
from plotting.vendor.scripts.paper_figures.plot_fig2_shared_field_reversal_row import apply_common_display_window
from plotting.vendor.scripts.plot_topic5_interictal_template_ab_fields import build_interictal_ab_panel_payloads, draw_interictal_rank_field_panel
from plotting.vendor.scripts.plot_contact_plane_static import VIS_MASK_REL, _smooth_rank_field_mm
from plotting.vendor.src.topic5_field_trajectory import sampling_hull, gaussian_rank_value_gradient, rank_gradient_trajectory
A_SOURCE = ROOT / 'results/paper-ready-figure/fig2/candidates/balanced_mri_view_20261012'
BASE = ROOT / 'results/paper-ready-figure/fig2/candidates/sequence_to_spatial_field_20261009'
OUT = ROOT / 'results/paper-ready-figure/fig2/candidates/complete_spatial_trajectories_20261012'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def freeze(output):
    snapshot = output / 'source_snapshot'
    snapshot.mkdir(parents=True, exist_ok=True)
    cmeta = json.loads((BASE / 'source_snapshot/fig2_panele_metadata.json').read_text())
    sources = {'fig2-panela.pdf': A_SOURCE / 'figures/fig2-panela-trajectories.pdf', 'fig2-panela.png': A_SOURCE / 'figures/fig2-panela-trajectories.png', 'fig2-panelb.pdf': BASE / 'figures/fig2-panelb.pdf', 'fig2-panelb.png': BASE / 'figures/fig2-panelb.png', 'fig2-panelc-original.pdf': BASE / 'figures/fig2-panelc.pdf', 'fig2-paneld.pdf': BASE / 'figures/fig2-paneld.pdf', 'fig2-paneld.png': BASE / 'figures/fig2-paneld.png', 'panelc_source_metadata.json': BASE / 'source_snapshot/fig2_panele_metadata.json', 'panela_trajectory_metadata.json': A_SOURCE / 'trajectory_metadata.json', 'panela_MRI_metadata.json': A_SOURCE / 'embedded_plane_metadata.json', 'panelb_source_metadata.json': BASE / 'source_snapshot/fig2_panelb_metadata.json', 'paneld_source_metadata.json': BASE / 'source_snapshot/fig2_panelf_metadata.json'}
    for e in cmeta['examples']:
        sources[f"{e['display_id']}_fields.json"] = Path(cmeta['input_root']) / 'per_subject' / f"{e['subject_id']}.json"
    manifest_path = snapshot / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for name, entry in manifest.items():
            assert sha(snapshot / name) == entry['sha256'], name
    else:
        manifest = {}
        for name, source in sources.items():
            shutil.copy2(source, snapshot / name)
            manifest[name] = dict(source=str(source), sha256=sha(snapshot / name))
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    return (snapshot, json.loads((snapshot / 'panelc_source_metadata.json').read_text()))

def shared_seeds(payloads, grids):
    """Use the existing hull centroid, or one centroid per disconnected support.

    The fallback reads only geometry and TA/TB support, never template ranks or
    expected direction. Both templates get the same seeds. No gap is bridged.
    """
    points = np.c_[payloads[0]['xs'], payloads[0]['ys']]
    centroid, hull = sampling_hull(points)
    X, Y = grids[0][:2]
    xy = np.c_[X.ravel(), Y.ravel()]
    inside = ((xy @ hull.equations[:, :2].T + hull.equations[:, 2]).max(1) <= 1e-08).reshape(X.shape)
    common = grids[0][4] & grids[1][4] & inside
    components, count = label(common)
    if count == 1:
        return ([centroid], dict(rule='sampling_convex_hull_area_centroid', components=1))
    seeds, details = ([], [])
    for component in range(1, count + 1):
        cells = np.c_[X[components == component], Y[components == component]]
        if len(cells) < 3:
            raise ValueError('Unresolved common support component')
        center = cells.mean(0)
        seed = cells[np.argmin(np.sum((cells - center) ** 2, axis=1))]
        seeds.append(seed)
        details.append(dict(component=component, cells=len(cells), area_centroid_mm=center.tolist(), seed_mm=seed.tolist()))
    return (seeds, dict(rule='nearest_supported_grid_point_to_each_common_support_component_area_centroid_within_sampling_hull', components=count, grid_shape=list(X.shape), components_detail=details, uses_template_ranks=False, same_seeds_for_TA_TB=True))

def compute_pair(display_id, record, output, *, sigma_mm=6.0, display_limits=None):
    a, b, mode = build_interictal_ab_panel_payloads(record, display_sigma_mm=sigma_mm)
    assert mode == 'shared'
    apply_common_display_window(a, b, x_span_mm=70, y_span_mm=70)
    if display_limits is not None:
        for payload in (a, b):
            payload['frame'].update(xlim=display_limits[0], ylim=display_limits[1])
    points = np.c_[a['xs'], a['ys']]
    np.testing.assert_allclose(points, np.c_[b['xs'], b['ys']], atol=1e-12)
    grids = [_smooth_rank_field_mm(*points.T, p['vals'], p['sup'], p['frame']['xlim'], p['frame']['ylim'], sigma_mm) for p in (a, b)]
    seeds, seed_meta = shared_seeds((a, b), grids)
    paths, records = ({}, {})
    for template, payload, grid in zip(('TA', 'TB'), (a, b), grids):
        X, Y, T, S, mask = grid
        floor = VIS_MASK_REL * float(S.max())
        sample = [(j, k) for j in range(0, T.shape[0], 11) for k in range(0, T.shape[1], 11) if mask[j, k]]
        field_error = max((abs(gaussian_rank_value_gradient([X[j, k], Y[j, k]], points, payload['vals'], payload['sup'], sigma_mm)[0] - T[j, k]) for j, k in sample))
        assert field_error < 1e-12
        paths[template], records[template] = ([], [])
        for index, seed in enumerate(seeds):
            kwargs = dict(sigma_mm=sigma_mm, support_floor=floor, seed=seed)
            trajectory = rank_gradient_trajectory(points, payload['vals'], payload['sup'], **kwargs)
            refined = rank_gradient_trajectory(points, payload['vals'], payload['sup'], step_mm=0.04, **kwargs)
            error = max(cKDTree(refined['points_mm']).query(trajectory['points_mm'])[0].max(), cKDTree(trajectory['points_mm']).query(refined['points_mm'])[0].max())
            assert error < 0.09, (display_id, template, error)
            assert np.all(np.diff(trajectory['rank01']) > 0)
            assert np.all(trajectory['support_mass'] >= floor)
            shown, trim = display_trajectory(trajectory, points)
            paths[template].append(shown)
            filename = f'{display_id}_{template.lower()}_trajectory_{index + 1}.csv'
            with (output / 'trajectories' / filename).open('w') as f:
                writer = csv.writer(f)
                writer.writerow(['x_mm', 'y_mm', 'rank01', 'support_mass', 'arc_length_mm', 'displayed'])
                for i, (q, rank, mass, arc) in enumerate(zip(trajectory['points_mm'], trajectory['rank01'], trajectory['support_mass'], trajectory['arc_length_mm'])):
                    writer.writerow([*q, rank, mass, arc, int(trim['first_computed_point_index'] <= i <= trim['last_computed_point_index'])])
            records[template].append(dict(trajectory=trajectory, display_trim=trim, displayed_points_mm=shown, csv=filename, validation=dict(rank_strictly_increases=True, inside_display_support=True, integrator_half_step_max_distance_mm=float(error), canonical_field_max_abs_error=float(field_error))))
    return ((a, b), paths, dict(display_id=display_id, subject_id=record['subject_id'], input_fingerprint_sha256=record['interictal_field']['fingerprint_sha256'], shared_seed_rule=seed_meta, xlim_mm=a['frame']['xlim'], ylim_mm=a['frame']['ylim'], contact_order=a['names'], points_mm=points, display_sigma_mm=sigma_mm, trajectories=records))

def build_c(snapshot, source_metadata, output, dpi):
    rc = {'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.linewidth': 0.7, 'pdf.fonttype': 42}
    metadata = []
    with plt.rc_context(rc):
        width, height = (660.0, 322.0)
        fig = plt.figure(figsize=(width / 72, height / 72), facecolor='white')
        for column, example in enumerate(source_metadata['examples']):
            display_id = example['display_id']
            record = json.loads((snapshot / f'{display_id}_fields.json').read_text())
            payloads, paths, meta = compute_pair(display_id, record, output)
            metadata.append(meta)
            for row, (template, payload) in enumerate(zip(('TA', 'TB'), payloads)):
                left, top, side = (54.0 + column * 143.0, 20.0 + row * 145.0, 120.0)
                ax = fig.add_axes([left / width, 1 - (top + side) / height, side / width, side / height])
                draw_interictal_rank_field_panel(ax, payload, template, compact=True, panel_title=display_id if row == 0 else '', contact_size=20, contact_outline_lw=0.75, show_template_tag=False)
                ax.set_title(display_id if row == 0 else '', fontsize=10.5, fontweight='bold', pad=3)
                ax.xaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))
                ax.tick_params(axis='both', labelsize=7.5, length=2.5, width=0.6, pad=1.5)
                if column == 0:
                    ax.set_ylabel('y (mm)', fontsize=8.5, labelpad=2)
                    ax.text(-0.36, 0.5, f'{template} trajectory', transform=ax.transAxes, rotation=90, ha='center', va='center', color=TEMPLATE_COLORS[template], fontsize=9, fontweight='bold', clip_on=False)
                for path in paths[template]:
                    mpl_path(ax, path, TEMPLATE_COLORS[template], scale=0.85)
                for collection in ax.collections:
                    collection.set_zorder(9)
                meta.setdefault('axes_rect_pt', {})[template] = [left, top, left + side, top + side]
        cax = fig.add_axes([620 / width, (height - 285) / height, 6 / width, 265 / height])
        cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap='viridis'), cax=cax, ticks=[0, 1])
        cb.ax.set_yticklabels(['early', 'late'])
        cb.ax.tick_params(labelsize=8, length=2, pad=2)
        cb.ax.set_title('rank', fontsize=8.5, pad=4)
        cb.outline.set_linewidth(0.7)
        fig.text(323 / width, 1 - 316 / height, 'Shared TA axis (mm)', ha='center', va='bottom', fontsize=10)
        for ext in ('png', 'pdf', 'svg'):
            fig.savefig(output / f'figures/fig2-panelc.{ext}', dpi=dpi, facecolor='white')
        plt.close(fig)
    return metadata

def build(output, dpi=600):
    output.mkdir(parents=True, exist_ok=True)
    (output / 'figures').mkdir(exist_ok=True)
    (output / 'trajectories').mkdir(exist_ok=True)
    snapshot, cmeta = freeze(output)
    for letter in 'abd':
        for ext in ('png', 'pdf'):
            shutil.copy2(snapshot / f'fig2-panel{letter}.{ext}', output / f'figures/fig2-panel{letter}.{ext}')
        with pdf.open(snapshot / f'fig2-panel{letter}.pdf') as doc:
            (output / f'figures/fig2-panel{letter}.svg').write_text(doc[0].get_svg_image(text_as_path=True))
    c_results = build_c(snapshot, cmeta, output, dpi)
    complete = pdf.open()
    page = complete.new_page(width=1240, height=870)
    docs = {letter: pdf.open(output / f'figures/fig2-panel{letter}.pdf') for letter in 'abcd'}
    layout = {'a': place(page, docs['a'], (28, 52, 680, 438), docs['a'][0].rect), 'b': place(page, docs['b'], (710, 60, 1221, 420)), 'c': place(page, docs['c'], (25, 490, 811, 852), docs['c'][0].rect), 'd': place(page, docs['d'], (839, 498, 1216, 849))}
    page.insert_font(fontname='FigureTwoPanelBold', fontfile=str(INPUT_ROOT / 'fonts' / 'DejaVuSans-Bold.ttf'))
    for letter, x, y in [('A', 12, 33), ('B', 695, 33), ('C', 12, 474), ('D', 823, 474)]:
        page.insert_text((x, y), letter, fontname='FigureTwoPanelBold', fontsize=24, color=(0, 0, 0))
    export(complete, output / 'figures/fig2-complete-layout', dpi)
    page.get_pixmap(matrix=pdf.Matrix(1800 / 1240, 1800 / 1240), alpha=False).save(output / 'figures/fig2-complete-layout-preview.png')
    for doc in [complete, *docs.values()]:
        doc.close()
    metadata = dict(schema_version='fig2_complete_spatial_trajectories_v1', human_visual_acceptance='PENDING', source_contract='all_event_timing_plus_space_no_hard_qc_v1', panela_source=str(A_SOURCE), panel_mapping=dict(A='latest E10 sequence, MRI and trajectories', B='frozen original B', C='E1, E12, E5, Y9 frozen fields with local gradient trajectories', D='frozen original F'), complete_layout=layout, canvas_pt=[1240, 870], panelc_examples=c_results, trajectory_method='RK4 of normalized analytic gradient of the original 6 mm support-weighted Gaussian rank display field, early to late; stop at sampling hull or support boundary', disconnected_support_rule='same geometry/support-only component centroid seed for TA/TB; no trajectory bridges an unsupported gap', trajectory_colors=TEMPLATE_COLORS, trajectory_outer_halo='#FFFFFF', panelc_style=dict(rows=2, columns=4, field_frame_pt=120, contact_size=20, contact_outline_pt=0.75, arrow_scale=0.85), scientific_boundary='Descriptive gradient streamlines of interpolated ranks, not measured tissue trajectories, velocities or unique paths. Existing cohort statistics unchanged.', validation=dict(frozen_panels_a_b_d_byte_identical=True, canonical_figure_replaced=False, agent_png_review='PENDING', agent_pdf_review='PENDING'))
    (output / 'figure2_layout_metadata.json').write_text(json.dumps(metadata, default=plain, ensure_ascii=False, indent=2) + '\n')
    for source in (Path(__file__), ROOT / 'src/topic5_field_trajectory.py', ROOT / 'scripts/paper_figures/plot_fig2_exemplar_trajectory.py'):
        shutil.copy2(source, snapshot / source.name)
    (output / 'figures/README.md').write_text('# Figure 2：完整轨迹候选\n\n### fig2-complete-layout.png / .pdf / .svg\n\n完整A–D版式：A沿用最新E10、B保留方向验证、C给四例TA/TB二维场增加梯度轨迹、D保留18人统计。所有10张二维rank场均有轨迹，E5每张场的两个分离区域各有一条，共12条；颜色统一为TA深红、TB蓝色与白色外沿。**关注点**：局部轨迹表示插值rank递增方向，不是直接测得的组织运动或速度；待作者目视验收。\n\n### fig2-panela.png / .pdf / .svg\n\n沿用作者选定B视角与SP向R移12 mm的最新A，左侧白色方向箭头、右侧深红/蓝色轨迹。PNG/PDF与本次冻结A逐字节一致。**关注点**：维持1D Rank SEQ到二维Spatial Trajectory的阅读关系。\n\n### fig2-panelb.png / .pdf / .svg\n\n原B完整保留，包括E10/E14方向示例及26名患者留出得分。PNG/PDF与冻结来源逐字节相同。**关注点**：方向验证的样本、统计和标签均未改变。\n\n### fig2-panelc.png / .pdf / .svg\n\n沿用E1、E12、E5、Y9及原70×70 mm视野、6 mm核、rank/support和viridis色标；继续使用公共场renderer，并扩大数据框供轨迹与电极共同显示。E5按TA/TB共同有效support与采样凸包的连通区域分别选几何中心，TA/TB共用种子，逐步积分并在支持边界停止，不跨空白连接。完整计算轨迹与显示端点裁短信息保存在CSV和metadata中。**关注点**：曲线逐点rank递增，显示末端仅为避让触点；四例是形态展示，完整队列推断在D。\n\n### fig2-paneld.png / .pdf / .svg\n\n保留18名患者共享场相关、15/18负相关比例及原空间零模型。PNG/PDF与原F逐字节相同。**关注点**：新增轨迹不改变统计定义、队列或显著性。\n\n### fig2-complete-layout-preview.png\n\n完整图的1800像素宽预览，与同目录PDF对应。**关注点**：检查C中方向尖端与白色外沿、电极的可读性，以及各面板边距。\n')
    return metadata
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--dpi', type=int, default=600)
    args = parser.parse_args()
    result = build(args.output.resolve(), args.dpi)
    print(json.dumps(dict(output=str(args.output), examples=[r['display_id'] for r in result['panelc_examples']], new_trajectories=sum((len(t) for r in result['panelc_examples'] for t in r['trajectories'].values()))), indent=2))
