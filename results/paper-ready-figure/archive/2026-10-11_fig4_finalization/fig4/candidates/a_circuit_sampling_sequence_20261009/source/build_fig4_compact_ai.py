#!/usr/bin/env python3
"""Render the author-requested three-row Fig4 A-I from frozen current inputs.

The first row reserves space for a subsequent mechanism redesign. This step
changes panel selection and layout, never fits a model or advances an experiment.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.paper_figures import build_fig4_core_count_candidate as core
from scripts.paper_figures import build_fig4_readable_complete as plate
from scripts.paper_figures import build_fig4_template_recovery_by_mode as recovery
from scripts.paper_figures import fig4_parameter_response_assets as parameters
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.transforms import Bbox
import numpy as np

DEST = ROOT / 'results/paper-ready-figure/fig4'
PARENT = ROOT / 'results/paper-ready-figure/archive/2026-10-09_pre_compact_ai_fig4/fig4'
PREVIOUS = ROOT / 'results/paper-ready-figure/archive/2026-10-09_pre_tighter_columns_fig4/fig4'
OUT = DEST / 'candidates/compact_ai_tighter_columns_20261009'
PRODUCER = 'scripts/paper_figures/build_fig4_compact_ai.py'
VERSION = 'compact_ai_tighter_columns_larger_squares_v4'
W, H = 300., 232.
# Data-axis boxes in mm: paired row edges are explicit, not inferred from crops.
BOXES = {
    'B': (18, 89, 50, 50), 'Bbar': (69.5, 89, 1.6, 50),
    'C': (93, 89, 50, 50), 'D': (159, 89, 50, 50),
    'E': (233, 89, 50, 50), 'Ebar': (285, 89, 1.4, 50),
    'F': (18, 17, 84, 50), 'G': (116, 17, 27, 50),
    'H': (159, 17, 50, 50), 'Hbar': (211, 17, 1.6, 50),
    'I': (233, 17, 50, 50),
}
TITLES = {
    'A': '局部 E/I 回路与患者空间基底；右侧预留机制连接图',
    'B': '先验开发后的三类误差：原第6–16阶段194个工作点',
    'C': 'E1146一至四core最佳训练loss：4次优化重复均值±样本标准差',
    'D': 'E→E角度的Mean rank／Order／Participation误差响应',
    'E': '413个可评分配置中低J_joint前20%（83个）的位置密度',
    'F': '连续30–80 Hz虚拟接触活动及MTA／MTB事件标记',
    'G': '模型与患者的平均传播rank',
    'H': '模型—患者触点交叉匹配矩阵',
    'I': '25位患者TA–MTA／TB–MTB模板相似度',
}


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def table(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def move(ax, key):
    plate.pos(ax, *BOXES[key])


def validate_sources():
    pointer = plate.read(PARENT / 'current_version.json')
    record = plate.read(PARENT / 'figure4_panel_registry.json')
    for key in ('panel_b', 'panel_c', 'panel_d'):
        assert pointer[key] == record[key], key
    assert pointer['panel_b']['version'] == 'post_prior_stages_6_16_v1'
    assert pointer['panel_c']['core_counts'] == [1, 2, 3, 4]
    assert pointer['panel_d']['version'] == 'three_error_responses_density_v1'
    density = ROOT / pointer['panel_d']['density_source_package']
    assert plate.sha(density / 'source/snapshot.json') == pointer['panel_d']['density_snapshot_sha256']
    assert plate.sha(PARENT / 'source/core_count_input_snapshot.json') == pointer['panel_c']['snapshot_sha256']
    return pointer, record, density


def draw_a(fig):
    with plate.group(fig, 'A'):
        plate.panel_a(fig)
    group = plate.GROUPS['A']
    old_positions = [ax.get_position().frozen() for ax in group['axes']]
    local, spatial = group['axes']
    # Preserve A's physical size while closing the two inter-row gaps.
    old = old_positions[0]
    ox, oy = old.x0 * W, old.y0 * H
    scale = 1.23
    def transform(x, y):
        return ((x * W - ox) * scale + 7) / W, ((y * H - oy) * scale + 161) / H
    for ax, pos in zip(group['axes'], old_positions):
        x, y = transform(pos.x0, pos.y0)
        ax.set_position([x, y, pos.width * scale, pos.height * scale])
    for text in group['texts']:
        text.set_position(transform(*text.get_position()))
    # The old raster contains the SCL9 callout and parts of its connectors.
    # Re-render the same frozen E/I/contact substrate without those annotations.
    old_meta = plate.read(plate.freeze(ROOT / 'results/paper-ready-figure/archive/'
        '2026-09-29_pre_template_recovery_fig4/fig4/figures/fig4-panela-metadata.json'))
    clean, source = plate.combined._render_clean_panel_b(include_legend=False)
    expected = old_meta['sources']['clean_patient_substrate']
    assert source['figdata_path'] == expected['figdata_path']
    assert source['figdata_sha256'] == expected['figdata_sha256']
    plate.freeze(ROOT / source['figdata_path'])
    spatial.images[0].set_data(np.asarray(clean))
    for line in group['artists']:
        line.remove()
    group['artists'].clear()
    # Keep the example left of the electrode shafts so neither connector crosses them.
    center = np.array([-8.5, 0.])
    fov = np.array(old_meta['layout']['zoom_field_of_view_mm'])
    lo, hi = center - fov / 2, center + fov / 2
    frame = Rectangle(lo, *fov, fill=False, edgecolor='#4a4a4a',
                      lw=.75, linestyle=(0, (3, 2)), zorder=6)
    spatial.add_patch(frame)
    local_frame = local.patches[0].get_bbox()
    for a, b in [((local_frame.x1, local_frame.y1), (lo[0], hi[1])),
                 ((local_frame.x1, local_frame.y0), (lo[0], lo[1]))]:
        line = ConnectionPatch(a, b, coordsA=local.transAxes, coordsB=spatial.transData,
            arrowstyle='-', color='#4a4a4a', lw=.65, linestyle=(0, (3, 2)),
            clip_on=False, zorder=4)
        fig.add_artist(line)
        group['artists'].append(line)
    contacts = np.asarray(source['contact_xy_mm'])
    nearest_distance = float(np.linalg.norm(contacts - np.clip(contacts, lo, hi), axis=1).min())
    assert nearest_distance > 2.5
    write(plate.SOURCE / 'A_zoom_in.json', dict(source=source,
        center_mm=center.tolist(), field_of_view_mm=fov.tolist(),
        nearest_contact_to_box_mm=nearest_distance,
        meaning='Representative local circuit anywhere in the spatial model; not restricted to SEEG contacts.'))
    plate.CHECKS['A_revision'] = {
        'status': 'LEFT_ZOOM_CLEAR_OF_ELECTRODES_ADDITIONAL_MECHANISM_DIAGRAM_PENDING',
        'artwork_scale': scale,
        'zoom_center_mm': center.tolist(), 'zoom_field_of_view_mm': fov.tolist(),
        'nearest_contact_to_box_mm': nearest_distance,
        'old_contact_callout_removed': True, 'frozen_spatial_geometry_verified': True,
        'zoom_metadata': 'source/A_zoom_in.json',
        'reserved_right_region_mm': [180, 155, 100, 70],
        'next_redraw': ['show physical anisotropy direction', 'show possible core locations',
                        'add right-side physical connectivity mechanism'],
    }


def draw_b(fig):
    rows = table(PARENT / 'source/B_visible_workpoints.csv')
    for row in rows:
        row['stage'] = int(row['stage'])
        for key in plate.base.KEYS:
            row[key] = float(row[key])
    assert len(rows) == 194 and {r['stage'] for r in rows} == set(range(6, 17))
    with plate.group(fig, 'B'):
        ax = plate.panel_b(fig, rows, [], profile_only=True)
    bar = plate.GROUPS['B']['axes'][1]
    move(ax, 'B'); move(bar, 'Bbar')
    # Use exactly the three category positions: no empty tails before/after the data.
    ax.set_xlim(0, 2)
    ax.spines['bottom'].set_bounds(0, 2)
    colors = bar._colorbar.cmap.colors[5:]
    cmap = plate.matplotlib.colors.ListedColormap(colors)
    norm = plate.matplotlib.colors.BoundaryNorm(np.arange(5.5, 17), len(colors))
    bar._colorbar.update_normal(plt.cm.ScalarMappable(norm=norm, cmap=cmap))
    bar._colorbar.set_ticks([6, 10, 16])
    assert not ax.texts and len(ax.lines) == 194
    for line, row in zip(ax.lines, sorted(rows, key=lambda r: (r['stage'], r['candidate']))):
        np.testing.assert_array_equal(line.get_ydata(), [row[k] for k in plate.base.KEYS])
        np.testing.assert_allclose(line.get_color(), cmap(norm(row['stage'])))
    plate.CHECKS['B_values_colors_exact_no_showcase_labels'] = True
    plate.CHECKS['B_horizontal_spacing'] = dict(x_limits=[0, 2],
        first_category_at_left_axis=True, last_category_at_right_axis=True,
        data_to_axis_padding_mm=[0, 0], axis_to_colorbar_mm=1.5)


def draw_c(fig):
    case, _ = core.frozen_case(PARENT / 'source/core_count_input_snapshot.json',
                               core_counts=(1, 2, 3, 4), summary_style='mean_std')
    with plate.group(fig, 'C'):
        ax = fig.add_axes(plate.rect(*BOXES['C']))
        core.progress.draw(ax, case, show_title=False, paper_style=plate.FONT,
                           show_legend=True, summary_style='mean_std')
        ax.set_yticks([0, 2, 4, 6, 8])
    means = table(PARENT / 'source/core_count_mean_std_curves.csv')
    for k, line in zip((1, 2, 3, 4), ax.lines):
        expected = [r for r in means if int(r['core_count']) == k
                    and r['figure'] == 'e1146_one_two_three_four_core_loss']
        np.testing.assert_allclose(line.get_ydata(), [float(r['mean']) for r in expected], atol=1e-12)
        np.testing.assert_allclose(np.std(case['arrays'][k][:, :31, 0], axis=0, ddof=1),
                                   [float(r['std']) for r in expected], atol=1e-12)
    plate.CHECKS['C_frozen_mean_sample_sd_exact'] = True


def draw_d(fig):
    rows = sorted([r for r in table(PARENT / 'source/D_connection_response.csv')
                   if int(r['axis']) == 5], key=lambda r: float(r['parameter_value']))
    assert len(rows) == 13
    with plate.group(fig, 'D'):
        ax = fig.add_axes(plate.rect(*BOXES['D']))
        parameters.error_curves(plate, ax, [float(r['parameter_value']) for r in rows],
            [[float(r[k]) for r in rows] for k in parameters.METRICS],
            'E→E angle (°)', [-20, 0, 20], references=[])
        # The compact row uses the same axis typography as B/C.
        ax.xaxis.label.set_fontsize(plate.FONT['label'])
        ax.yaxis.label.set_fontsize(plate.FONT['label'])
    assert len(ax.lines) == 3 and not ax.texts
    plate.CHECKS['D_angle_values_exact_no_dangling_showcase_references'] = True


def draw_e(fig, density):
    with patch.object(parameters, 'DENSITY', density):
        arrays, geometry = parameters.position_data(plate)
    pixels, cmap, norm, labels = parameters.completed_density_layer(arrays, geometry)
    with plate.group(fig, 'E'):
        ax = fig.add_axes(plate.rect(*BOXES['E']))
        ax.imshow(pixels, extent=(1, 19, 1, 19), origin='upper', aspect='equal')
        ax.set(xlim=(1, 19), ylim=(1, 19), xticks=[5, 10, 15], yticks=[5, 10, 15],
               xlabel='x (mm)', ylabel='y (mm)')
        plate.axis_type(ax)
        for xy, label in zip(labels, ['Core A', 'Core B']):
            ax.text(*xy, label, fontsize=plate.FONT['legend'], ha='center', va='bottom', color='#403949')
        handles = [Line2D([], [], marker='o', ls='none', mfc='#B6B2BF', mec='none', ms=2.5),
                   Line2D([], [], marker='o', ls='none', mfc='none', mec='#65509A', mew=.65, ms=3.2),
                   Line2D([], [], color='#9784B6', lw=.6)]
        ax.legend(handles, ['Search', 'Outlier', '90% mass'], loc='upper right',
                  fontsize=plate.FONT['legend'], frameon=False, borderpad=.2, labelspacing=.1,
                  handlelength=.7, handletextpad=.25, borderaxespad=.1)
        bar = fig.add_axes(plate.rect(*BOXES['Ebar']))
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=bar, ticks=[0, .2, .4])
        cb.outline.set_visible(False); plate.axis_type(bar)
        cb.set_label(r'Position density (mm$^{-2}$)', fontsize=plate.FONT['label'], labelpad=1)
        bar.tick_params(pad=1)


def draw_bottom(fig, parent_record):
    plate.panels_fgh(fig)
    with np.load(PARENT / 'source/waveform_arrays.npz') as before, np.load(plate.SOURCE / 'waveform_arrays.npz') as after:
        assert before.files == after.files
        for key in before.files:
            np.testing.assert_array_equal(before[key], after[key])
    assert plate.CHECKS['GH_frozen_matrix'] == parent_record['original_panel_checks']['GH_frozen_matrix']
    for key in ('F', 'G', 'H'):
        move(plate.GROUPS[key]['axes'][0], key)
    move(plate.GROUPS['H']['axes'][1], 'Hbar')
    with plate.group(fig, 'I'):
        ax = fig.add_axes(plate.rect(*BOXES['I']))
        plate.axis_type(ax)
        summary = plate.read(PARENT / 'source/template_recovery_summary.json')
        metadata = recovery.draw_modes(ax, summary, legend_loc='upper right')
    assert metadata['groups'] == parent_record['panel_j']['groups']
    plate.CHECKS['I_all_50_subject_values_and_quartiles_unchanged'] = True
    return metadata


def verify_layout(fig):
    fig.canvas.draw()
    actual = {}
    for key in 'BCDEFGHI':
        actual[key] = (np.array(plate.GROUPS[key]['axes'][0].get_position().bounds)
                       * [W, H, W, H]).tolist()
        np.testing.assert_allclose(actual[key], BOXES[key], atol=1e-8, rtol=0)
    for row in ('BCDE', 'FGHI'):
        boxes = np.array([actual[key] for key in row])
        np.testing.assert_allclose(boxes[:, 1], boxes[0, 1], atol=1e-8)
        np.testing.assert_allclose(boxes[:, 1] + boxes[:, 3], boxes[0, 1] + boxes[0, 3], atol=1e-8)
    for top, lower in (('B', 'F'), ('D', 'H'), ('E', 'I')):
        assert abs(actual[top][0] - actual[lower][0]) < 1e-8
        if top in 'DE':
            np.testing.assert_allclose(np.array(actual[top])[[0, 2]],
                                       np.array(actual[lower])[[0, 2]], atol=1e-8)
    assert abs(sum(actual['C'][::2]) - sum(actual['G'][::2])) < 1e-8
    for key in 'BCDEHI':
        np.testing.assert_allclose(actual[key][2:], [50, 50], atol=1e-8)
    b = plate.GROUPS['B']['axes'][0]
    first, last = b.transData.transform([[0, 0], [2, 0]])[:, 0]
    np.testing.assert_allclose([first, last], [b.bbox.x0, b.bbox.x1], atol=1e-8)
    renderer = fig.canvas.get_renderer()
    bounds = {}
    for key, group in plate.GROUPS.items():
        boxes = [a.get_tightbbox(renderer) for a in group['axes']]
        boxes += [t.get_window_extent(renderer) for t in group['texts']]
        bound = Bbox.union([box for box in boxes if box is not None])
        bounds[key] = (np.array(bound.extents) / fig.dpi * 25.4).tolist()
        assert min(bounds[key][:2]) >= 0 and bounds[key][2] <= W and bounds[key][3] <= H, (key, bounds[key])
    for row in ('BCDE', 'FGHI'):
        for left, right in zip(row, row[1:]):
            assert bounds[left][2] + .5 < bounds[right][0], (left, right, bounds[left], bounds[right])
    row_gaps = [bounds['A'][1] - max(bounds[k][3] for k in 'BCDE'),
                min(bounds[k][1] for k in 'BCDE') - max(bounds[k][3] for k in 'FGHI')]
    previous = plate.read(PREVIOUS / 'figure4_panel_registry.json')['checks']['layout']['visible_bounds_mm']
    horizontal_gaps = {}
    previous_horizontal_gaps = {}
    for row in ('BCDE', 'FGHI'):
        for left, right in zip(row, row[1:]):
            pair = left + right
            horizontal_gaps[pair] = bounds[right][0] - bounds[left][2]
            previous_horizontal_gaps[pair] = previous[right][0] - previous[left][2]
            assert 2.5 < horizontal_gaps[pair] < previous_horizontal_gaps[pair], pair
    assert horizontal_gaps['CD'] < 4.5 and horizontal_gaps['GH'] < 5.5
    old_gaps = [previous['A'][1] - max(previous[k][3] for k in 'BCDE'),
                min(previous[k][1] for k in 'BCDE') - max(previous[k][3] for k in 'FGHI')]
    # Text extents shift slightly when square axes grow; retain gaps within 0.1 mm.
    np.testing.assert_allclose(row_gaps, old_gaps, atol=.1, rtol=0)
    assert min(row_gaps) > 5
    spatial = plate.GROUPS['A']['axes'][1]
    source = plate.read(plate.SOURCE / 'A_zoom_in.json')['source']
    contacts = spatial.transData.transform(source['contact_xy_mm'])
    minimum_connector_distance = np.inf
    for line in plate.GROUPS['A']['artists']:
        endpoints = line.get_path().transformed(line.get_transform()).vertices[[0, -1]]
        expected = [line.coords1.transform(line.xy1), line.coords2.transform(line.xy2)]
        np.testing.assert_allclose(endpoints, expected, atol=1e-7)
        start, end = endpoints
        delta = end - start
        fractions = np.clip((contacts - start) @ delta / (delta @ delta), 0, 1)
        distance = np.linalg.norm(contacts - (start + fractions[:, None] * delta), axis=1).min()
        minimum_connector_distance = min(minimum_connector_distance, distance / spatial.bbox.width * 20)
    assert minimum_connector_distance > source['seeg_sampling_overlay']['displayed_radius_mm']
    plate.CHECKS['A_revision']['connector_minimum_contact_distance_mm'] = float(minimum_connector_distance)
    plate.CHECKS['A_revision']['connectors_clear_of_electrode_footprints'] = True
    plate.CHECKS['A_revision']['connectors_attached_to_both_frames'] = True
    plate.CHECKS['layout'] = dict(axis_boxes_mm=actual, visible_bounds_mm=bounds,
        row_edges_aligned=True, paired_column_edges_aligned=True,
        no_panel_overlap=True, all_text_inside_canvas=True,
        square_axis_panels=list('BCDEHI'), square_axis_side_mm=50,
        horizontal_visible_gaps_mm=horizontal_gaps,
        previous_horizontal_visible_gaps_mm=previous_horizontal_gaps,
        horizontal_gaps_measured_including_labels_and_colorbars=True,
        all_adjacent_horizontal_gaps_reduced=True, BC_outer_edges_match_FG=True,
        DH_and_EI_both_edges_aligned=True,
        row_gaps_mm=row_gaps, previous_row_gaps_mm=old_gaps,
        row_gap_change_mm=(np.array(row_gaps) - old_gaps).tolist(), compact_row_gaps_preserved=True)


def export(fig, letters):
    positions = {ax: ax.get_position().frozen() for ax in fig.axes}
    def save(stem, box=None, dpi=320, extensions=('png', 'pdf', 'svg')):
        for ext in extensions:
            fig.savefig(plate.FIG / f'{stem}.{ext}', dpi=dpi, bbox_inches=box)
            for ax, pos in positions.items(): ax.set_position(pos)
    save('fig4-complete-layout')
    save('fig4-complete-layout-preview', dpi=135, extensions=('png',))
    for letter in letters: letter.set_visible(False)
    for key in 'ABCDEFGHI':
        for other, group in plate.GROUPS.items():
            for artist in group['axes'] + group['texts'] + group['artists']:
                artist.set_visible(other == key)
        fig.canvas.draw(); renderer = fig.canvas.get_renderer()
        group = plate.GROUPS[key]
        boxes = [ax.get_tightbbox(renderer) for ax in group['axes']]
        boxes += [t.get_window_extent(renderer) for t in group['texts']]
        box = Bbox.union([b for b in boxes if b is not None]).transformed(
            fig.dpi_scale_trans.inverted()).padded(1.2 * plate.MM)
        save(f'fig4-panel{key.lower()}', box)


def render():
    pointer, parent_record, density = validate_sources()
    current_image_sha = plate.sha(DEST / 'figures/fig4-complete-layout.png')
    plate.W, plate.H = W, H
    plate.OUT, plate.FIG, plate.SOURCE = OUT, OUT / 'figures', OUT / 'source'
    plate.FIG.mkdir(parents=True, exist_ok=True); plate.SOURCE.mkdir(exist_ok=True)
    plate.GROUPS.clear(); plate.CHECKS.clear(); plate.SOURCES.clear()
    plate.base.OUT = plate.SOURCE
    for name in ('B_visible_workpoints.csv', 'B_prior_workpoints.csv', 'B_display_contract.json',
                 'core_count_mean_std_curves.csv', 'D_connection_response.csv', 'template_recovery_summary.json'):
        shutil.copy2(PARENT / 'source' / name, plate.SOURCE / name)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.labelsize': 10,
        'xtick.labelsize': 9, 'ytick.labelsize': 9, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .7, 'legend.frameon': False})
    fig = plt.figure(figsize=(W * plate.MM, H * plate.MM), dpi=160, facecolor='white')
    for name, func in [('A', draw_a), ('B', draw_b), ('C', draw_c), ('D', draw_d)]:
        func(fig); print('Rendered', name, flush=True)
    draw_e(fig, density); print('Rendered E', flush=True)
    bottom = draw_bottom(fig, parent_record); print('Rendered F-I', flush=True)
    verify_layout(fig)
    letters = []
    for key in 'ABCDEFGHI':
        x, y = ((5, 226) if key == 'A' else (BOXES[key][0] - 11, 147 if key in 'BCDE' else 75))
        letters.append(plate.label(fig, x, y, key, fontsize=plate.FONT['letter'], weight='bold', va='top'))
    export(fig, letters); plt.close(fig)
    for module in (plate, plate.combined, core.progress, parameters, recovery):
        shutil.copy2(module.__file__, plate.SOURCE / Path(module.__file__).name)
    shutil.copy2(__file__, plate.SOURCE / Path(__file__).name)
    panel_b = dict(pointer['panel_b'], showcase_labels=False)
    panel_d = dict(version='angle_three_error_response_v1', source_package=str(PARENT.relative_to(ROOT)),
                   source_table='source/D_connection_response.csv', parent_panel='D',
                   parameter='EE_angle_offset_deg', conditions=13, showcase_references=False)
    panel_e = dict(pointer['panel_d'], version='completed_position_density_standalone_v1', parent_panel='D')
    record = dict(status='CURRENT_AUTHOR_DESIGNATED', human_visual_acceptance='PENDING',
        producer=PRODUCER, layout_version=VERSION, updated_on='2026-10-09',
        panel_letters=list('ABCDEFGHI'), titles=TITLES, whole_canvas_mm=[W, H],
        parent_package=str(PARENT.relative_to(ROOT)), base_published_complete_sha256=current_image_sha,
        panel_mapping={'A':'A', 'B':'B', 'C':'C',
        'D':'D angle response', 'E':'D position density', 'F':'G', 'G':'H', 'H':'I', 'I':'J'},
        omitted_parent_panels=['E propagation showcase', 'F propagation showcase',
                               'D outward EE response', 'D within-core EE response'],
        panel_a=plate.CHECKS['A_revision'], panel_b=panel_b, panel_c=pointer['panel_c'],
        panel_d=panel_d, panel_e=panel_e, panel_i=bottom, checks=plate.CHECKS,
        source_files=plate.SOURCES, scientific_claims='Unchanged frozen observations; no new simulations.',
        source_metadata_correction=parent_record['source_metadata_correction'],
        cohort_interpretation=parent_record['panel_j_interpretation'],
        outputs={str(p.relative_to(OUT)):plate.sha(p) for p in plate.FIG.iterdir()
                 if p.suffix in ('.png', '.pdf', '.svg')})
    write(OUT / 'figure4_candidate_registry.json', record)
    write(OUT / 'visual_qa.json', dict(status='LAYOUT_AND_VALUE_CHECKS_PASS_VISUAL_PENDING',
        human_visual_acceptance='PENDING', checks=plate.CHECKS))
    notes = ['# Figure 4：A–I 重排版', '', 'PNG、PDF、SVG同源导出；本轮只调整展示与排版，未重跑实验。', '',
        '### fig4-complete-layout.png',
        '首行A扩大并预留右侧机制连接图，中排B–E、底排F–I采用共同边界对齐。旧E/F传播showcase移出，底排数据保留；另有PDF/SVG及预览。',
        '**关注点**：收紧相邻横向留白后，B/C/D/E/H/I均扩大到50×50 mm；D/H与E/I逐列对齐，B/C整体边界与F/G组合对齐，紧凑行距保留。整图待作者目视检查。', '']
    for key, title in TITLES.items():
        focus = ('小框位于(-8.5,0) mm，虚线避开电极并连接两框。局部回路适用于整个模型空间，方向椭圆及右侧物理连接另待补充。' if key == 'A' else
                 'Mean rank／Participation分别贴合左右数据边界，色条间距1.5 mm；早期标签辅助开发仍是先验来源。' if key == 'B' else
                 '每组4次优化重复，均值±样本标准差，ddof=1，共同前31个epoch。' if key == 'C' else
                 '三条曲线沿用原冻结13点；已移除指向旧showcase的E/F参考线及标签。' if key == 'D' else
                 '描述已搜索低损失配置的位置密度；不是参数后验或唯一收敛证据。' if key == 'E' else
                 '沿用原底排的数值、事件、通道顺序与统计口径，仅调整位置和宽度。')
        notes += [f'### fig4-panel{key.lower()}.png', f'{title}。另有同源PDF与可编辑SVG。', f'**关注点**：{focus}', '']
    (plate.FIG / 'README.md').write_text('\n'.join(notes))
    mapping = ['# 当前 Figure 4：A–I', '', '| Panel | 内容 |', '|---|---|']
    mapping += [f'| {key} | {title} |' for key, title in TITLES.items()]
    mapping += ['', 'A小框位于(-8.5,0) mm，虚线避开电极；相邻横向留白收紧，B/C/D/E/H/I均为50×50 mm，D/H、E/I两侧对齐，B/C整体与F/G组合对齐，紧凑行距保持。右侧机制图仍待补充；旧完整A–J包保存在归档，底排旧G/H/I/J对应新F/G/H/I。',
                '', f'重建：`python {PRODUCER}`。']
    (OUT / 'panel_map.md').write_text('\n'.join(mapping) + '\n')
    print(plate.FIG / 'fig4-complete-layout-preview.png', flush=True)


def publish():
    record = plate.read(OUT / 'figure4_candidate_registry.json')
    current = plate.read(DEST / 'current_version.json')
    assert record['layout_version'] == VERSION and record['panel_letters'] == list('ABCDEFGHI')
    assert current['producer'] in (PRODUCER, 'scripts/paper_figures/build_fig4_current_aj.py')
    assert plate.sha(DEST / 'figures/fig4-complete-layout.png') in (
        record['base_published_complete_sha256'], record['outputs']['figures/fig4-complete-layout.png']
    ), 'Current figure changed; inspect before publication.'
    for name, digest in record['outputs'].items(): assert plate.sha(OUT / name) == digest
    shutil.copytree(OUT / 'figures', DEST / 'figures', dirs_exist_ok=True)
    shutil.copytree(OUT / 'source', DEST / 'source', dirs_exist_ok=True)
    # The former J is now I; keep historical J in the complete parent archive.
    for ext in ('png', 'pdf', 'svg'):
        stale = DEST / 'figures' / f'fig4-panelj.{ext}'
        if stale.exists():
            assert (PARENT / 'figures' / stale.name).exists()
            stale.unlink()
    shutil.copy2(OUT / 'panel_map.md', DEST / 'panel_map.md')
    record.update(published_from=str(OUT.relative_to(ROOT)), previous_version_archive=str(PREVIOUS.relative_to(ROOT)))
    write(DEST / 'figure4_panel_registry.json', record)
    write(DEST / 'figure4_candidate_registry.json', record)
    shutil.copy2(OUT / 'visual_qa.json', DEST / 'visual_qa.json')
    pointer = dict(schema_version=2, figure='Fig4', status=record['status'],
        human_visual_acceptance='PENDING', author_designated_on='2026-10-09', updated_on='2026-10-09',
        panel_letters=list('ABCDEFGHI'), asset_id='patient_geometry_prior_snn_compact_a_i',
        layout_version=VERSION, producer=PRODUCER, package=str(DEST.relative_to(ROOT)),
        documentation='docs/current_figure4.md', registry=str((DEST / 'figure4_panel_registry.json').relative_to(ROOT)),
        visual_qa=str((DEST / 'visual_qa.json').relative_to(ROOT)),
        complete_layout={ext:str((DEST / 'figures' / f'fig4-complete-layout.{ext}').relative_to(ROOT)) for ext in ('png','pdf','svg')},
        preview=str((DEST / 'figures/fig4-complete-layout-preview.png').relative_to(ROOT)),
        previous_version_archive=str(PREVIOUS.relative_to(ROOT)), published_from=str(OUT.relative_to(ROOT)),
        outputs_sha256={str((DEST / name).relative_to(ROOT)):digest for name,digest in record['outputs'].items()},
        **{key:record[key] for key in ('panel_a','panel_b','panel_c','panel_d','panel_e','panel_i')})
    for path, digest in pointer['outputs_sha256'].items(): assert plate.sha(ROOT / path) == digest
    temporary = DEST / 'current_version.json.tmp'; write(temporary, pointer)
    temporary.replace(DEST / 'current_version.json')
    (DEST / 'README.md').write_text('# 当前 Figure 4：A–I 重排版\n\n'
        '[完整预览](figures/fig4-complete-layout-preview.png) · [PDF](figures/fig4-complete-layout.pdf) · [编号说明](panel_map.md)\n\n'
        '按2026-10-09作者草图重排；B保留原第6–16阶段，C保留四core均值±标准差，D/E拆分角度响应与最新位置密度。'
        '旧E/F传播showcase不再展示，底排旧G–J改为F–I。横向留白收紧，B/C/D/E/H/I扩大到50×50 mm；D/H与E/I逐列对齐，F/G组合匹配B/C的整体边界。B两端无空白延伸、色条间距1.5 mm；A框位于(-8.5,0) mm，虚线避开电极。紧凑行距保留，右侧机制连接图仍待补充，整图待作者目视检查。\n\n'
        f'重建：`python {PRODUCER}`；本次前一版保存在`{PREVIOUS.relative_to(ROOT)}`，旧A–J数据包保存在`{PARENT.relative_to(ROOT)}`。\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish', action='store_true', help='Publish the already-rendered and checked layout.')
    args = parser.parse_args()
    if args.publish: publish()
    else: render()
