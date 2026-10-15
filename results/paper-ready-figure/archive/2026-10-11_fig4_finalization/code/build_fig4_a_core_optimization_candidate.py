#!/usr/bin/env python3
"""Add the author's schematic multicore position updates to current Figure 4A."""
from pathlib import Path
import hashlib
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.paper_figures import build_fig4_compact_ai as compact
from scripts.paper_figures import build_fig4_a_mechanism_candidate as current
from scripts.paper_figures import draw_fig4_a_spatial_readout as mechanism
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
import numpy as np
from PIL import Image

BASE = ROOT / 'results/paper-ready-figure/fig4'
OUT = BASE / 'candidates/a_core_search_central_shading_20261011'
VERSION = 'compact_ai_v4_a_core_search_central_shading_v15'
PRODUCER = 'scripts/paper_figures/build_fig4_a_core_optimization_candidate.py'
SPEC = dict(
    interpretation='Author-requested optimization schematic. Updated core locations match the two E-panel density modes; candidate regions and the update arrow are illustrative.',
    candidate_region_count=3, updated_core_count=2,
    core_count_context='C compares one through four cores. Three possible regions and two resulting cores here are schematic, not a demonstrated three-to-two count optimization.',
    candidate_centres_mm=[[-6.4, 7.0], [6.65, 3.6], [-3.85, -7.2]],
    updated_centres_mm=None,  # Read the current E-panel density modes below.
    candidate_radius_mm=1.15, updated_radius_mm=1.05,
    candidate_fill='#AC8BC9', candidate_fill_alpha=.32, candidate_edge='#B8A3CF',
    updated_fill='#805AA6', updated_edge='#76519D',
    arrow_color='#9479B2', arrow_dash_pattern=[3.0, 2.5],
    search_direction_arrows_mm=None,
    arrows='Two illustrative dashed OPTM arrows: the upper candidates point to Core A and Core B; the bottom candidate remains unconnected.',
    arrow_label='OPTM', arrow_label_xy_mm=[[-4.7, 4.35], [8.15, -.4]],
    arrow_target_core_indices=[0, 1],
    sampling_frame_shading=dict(color='#93C4A7', alpha=.22,
        locations=['Central zoom-location frame'],
        interpretation='Central zoom-frame background only. The right inset retains its original localized Gaussian sampling weights without a whole-frame background.'),
    question_mark_in_each_candidate_region=True,
    candidate_regions_have_light_purple_shading=True,
    regions_and_question_marks_below_neuron_markers=True,
    update_arrows_are_not_propagation_paths=True,
    simulations_and_readout_unchanged=True,
)


def draw_a(fig):
    with compact.plate.group(fig, 'A'):
        first_axis = len(fig.axes)
        record = mechanism.draw(fig)
        spatial = fig.axes[first_axis + 1]
        shade = SPEC['sampling_frame_shading']
        sampling_fov = np.array(record['sampling_zoom_field_of_view_mm'])
        sampling_lo = np.array(record['sampling_zoom_center_overview_mm']) - sampling_fov / 2
        spatial.add_patch(Rectangle(sampling_lo, *sampling_fov,
            fc=to_rgba(shade['color'], shade['alpha']), ec='none', zorder=.2))
        record.pop('central_sampling_region_removed', None)
        record['central_gaussian_sampling_weights_removed'] = True
        record['sampling_frame_background_shading'] = shade
        record['right_sampling_whole_frame_shading'] = False
        record['sampling_weight_display']['central_sheet'] = (
            'Light-green zoom-frame background only; Gaussian sampling weights remain in the right inset.')
        for point in SPEC['candidate_centres_mm']:
            spatial.add_patch(Circle(point, SPEC['candidate_radius_mm'],
                fc=to_rgba(SPEC['candidate_fill'], SPEC['candidate_fill_alpha']),
                ec=SPEC['candidate_edge'], lw=.7, ls=(0, (2.5, 2)), zorder=.6))
            spatial.text(*point, '?', color='#82579F', fontsize=10.5,
                ha='center', va='center', zorder=.8)
        # Draw regions beneath the original neuron scatter (zorder=1) so that
        # both candidate and updated regions retain every neuronal marker.
        for point in SPEC['updated_centres_mm']:
            spatial.add_patch(Circle(point, SPEC['updated_radius_mm'],
                fc=SPEC['updated_fill'], ec=SPEC['updated_edge'], alpha=.50, lw=.8, zorder=.5))
        for start, end in SPEC['search_direction_arrows_mm']:
            spatial.add_patch(FancyArrowPatch(start, end,
                arrowstyle='-|>', mutation_scale=7.5,
                color=SPEC['arrow_color'], lw=.85, linestyle=(0, SPEC['arrow_dash_pattern']),
                shrinkA=0, shrinkB=0, zorder=.7))
        for xy in SPEC['arrow_label_xy_mm']:
            spatial.text(*xy, SPEC['arrow_label'], color='#785398',
                fontsize=8, ha='center', va='center', zorder=2)
        record['schematic_core_optimization'] = SPEC
    compact.plate.CHECKS['A_revision'] = record


def main():
    pointer = json.loads((BASE / 'current_version.json').read_text())
    assert pointer['layout_version'] == 'compact_ai_v4_a_left_circuit_spacing_v10'
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    before = {name: sha(ROOT / name) for name in pointer['outputs_sha256']}
    shutil.copytree(BASE / 'source', OUT / 'source', dirs_exist_ok=True)
    density_path = BASE / 'source/D_position_density.npz'
    with np.load(density_path) as density:
        reference_centres = []
        for component in density['per_core_density_per_mm2']:
            iy, ix = np.unravel_index(np.argmax(component), component.shape)
            reference_centres.append([float(density['x_mm'][ix]), float(density['y_mm'][iy])])
    SPEC['updated_centres_mm'] = (np.array(reference_centres) - 10.).tolist()
    SPEC['core_position_source'] = dict(panel='E', snapshot='source/D_position_density.npz',
        snapshot_sha256=sha(density_path), reference_centres_mm=reference_centres,
        coordinate_shift_mm=[-10., -10.],
        rule='Maxima of the two frozen per-core position-density fields, in physical ground-plane coordinates; exclude the display height of the E relief.',
        interpretation='Two marginal density modes used as schematic locations, not a jointly selected fit or proof of unique recovery.')
    start = np.array(SPEC['candidate_centres_mm'][0]) - [0., SPEC['candidate_radius_mm'] + .15]
    end = np.array(SPEC['updated_centres_mm'][0]) + [0., SPEC['updated_radius_mm'] + .2]
    SPEC['search_direction_arrows_mm'] = [[start.tolist(), end.tolist()]]
    start = np.array(SPEC['candidate_centres_mm'][1]) - [0., SPEC['candidate_radius_mm'] + .15]
    end = np.array(SPEC['updated_centres_mm'][1]) + [0., SPEC['updated_radius_mm'] + .2]
    SPEC['search_direction_arrows_mm'].append([start.tolist(), end.tolist()])
    mechanism.SOURCE = OUT / 'source'
    compact.OUT, compact.VERSION, compact.PRODUCER = OUT, VERSION, PRODUCER
    compact.TITLES['A'] = '局部回路、多个core位置优化示意与SEEG采样读出'
    compact.draw_a, compact.verify_layout = draw_a, current.verify_layout
    compact.render()
    unchanged = {}
    for letter in 'bcdefghi':
        name = f'figures/fig4-panel{letter}.png'
        with Image.open(BASE / name) as a, Image.open(OUT / name) as b:
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        unchanged[letter.upper()] = 'PIXEL_IDENTICAL_TO_CURRENT_V10'
    record = json.loads((OUT / 'figure4_candidate_registry.json').read_text())
    for key in ('panel_b', 'panel_c', 'panel_d', 'panel_e', 'panel_i'):
        assert pointer[key] == record[key], key
    with Image.open(BASE / 'figures/fig4-complete-layout.png') as a, Image.open(OUT / 'figures/fig4-complete-layout.png') as b:
        aa, bb = np.asarray(a), np.asarray(b)
        assert aa.shape == bb.shape
        changed = np.any(aa != bb, axis=2)
        allowed = np.zeros(changed.shape, bool)
        x, y, w, h = record['panel_a']['sheet_axes_mm']
        left, right = int(np.floor(x / 300 * aa.shape[1])), int(np.ceil((x + w) / 300 * aa.shape[1]))
        top, bottom = int(np.floor((232 - y - h) / 232 * aa.shape[0])), int(np.ceil((232 - y) / 232 * aa.shape[0]))
        allowed[top:bottom, left:right] = True
        assert changed.any() and not np.any(changed & ~allowed)
    for name, digest in before.items():
        assert sha(ROOT / name) == digest, name
    record.update(status='CANDIDATE_PENDING_AUTHOR_VISUAL_REVIEW', updated_on='2026-10-11',
        base_layout_version=pointer['layout_version'], canonical_figure_replaced=False,
        unchanged_panels=unchanged)
    validation = dict(status='PASS', unchanged_panels=unchanged,
        all_pixels_outside_central_sheet_identical_to_v10=True,
        right_local_sampling_identical_to_v10=True,
        changed_pixels_in_central_sheet=int(changed.sum()),
        canonical_output_hashes_unchanged=True,
        candidate_coordinates_are_illustrative=True,
        updated_core_coordinates_match_E_density_modes=True,
        schematic_core_optimization=SPEC,
        human_visual_acceptance='PENDING')
    record['checks']['core_optimization_revision'] = validation
    compact.write(OUT / 'figure4_candidate_registry.json', record)
    compact.write(OUT / 'mechanism_validation.json', dict(record['panel_a'], **validation))
    qa = json.loads((OUT / 'visual_qa.json').read_text())
    qa['checks']['core_optimization_revision'] = validation
    compact.write(OUT / 'visual_qa.json', qa)
    compact.write(OUT / 'source/core_optimization_schematic.json', SPEC)
    shutil.copy2(__file__, OUT / 'source' / Path(__file__).name)
    mapping = (OUT / 'panel_map.md').read_text().replace('右侧机制图仍待补充；',
        'A中央保留三个问号区域，上方两个以OPTM箭头分别指向两个core，下方不加箭头；仅中央取样框增加浅绿色底纹；')
    (OUT / 'panel_map.md').write_text(mapping)
    description = ('中央二维sheet保留三个问号候选区域，加入淡紫色底纹；两个紫色core对齐E图的两个密度峰。'
        'E坐标为(3.3125,12.0625)/(16.4375,5.4375) mm，减去(10,10)后用于A；不使用三维浮雕抬高后的屏幕位置。'
        '上方两个候选区域带OPTM虚线箭头，分别指向两个core；下方问号区域不加箭头或OPTM文字。'
        '浅绿底纹仅在中央取样小框显示；右侧Local sampling无整框底纹，原σ=0.25 mm局部高斯采样权重保持。'
        '候选区域和优化方向为示意；底纹与问号位于神经元点下层。'
        '两个边际密度峰不表示一组联合最优配置，箭头不是实际优化轨迹、真实病灶移动或传播路径；'
        'core数量的实验比较仍见C，两组波形仍来自原冻结工作点。左侧回路、右侧采样几何与波形、B–I保持当前v10。')
    (OUT / 'README.md').write_text('# Figure 4A v15：简化优化箭头与中央取样框底纹\n\n' + description + '\n\n'
        '当前正式v10保留，本候选已输出独立A和完整拼版的PNG/PDF/SVG，待作者目视检查。'
        '全图中央sheet之外像素完全一致，包含右侧Local sampling；B–I逐像素一致，正式导出哈希不变。\n\n'
        f'重建：`python {PRODUCER}`。\n')
    (OUT / 'figures/README.md').write_text('### fig4-panela.png / .pdf / .svg\n\n' + description + '\n'
        '**关注点**：保留上方两条OPTM箭头，下方问号区域无箭头；浅绿底纹只出现在中央取样小框。\n\n'
        '### fig4-complete-layout.png / .pdf / .svg / -preview.png\n\n'
        '沿用当前三行A–I紧凑布局，A展示多core位置优化示意，仅中央取样框增加浅绿色底纹。'
        '原始数据、其他面板和所有轴的位置不变。\n'
        '**关注点**：实际整图尺寸下问号和虚线箭头的可读性；待作者目视检查。\n')
    print('Candidate complete; all pixels outside central sheet unchanged.', flush=True)


if __name__ == '__main__':
    main()
