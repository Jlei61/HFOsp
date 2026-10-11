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
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np
from PIL import Image

BASE = ROOT / 'results/paper-ready-figure/fig4'
OUT = BASE / 'candidates/a_core_search_simplified_20261010'
VERSION = 'compact_ai_v4_a_core_search_simplified_v12'
PRODUCER = 'scripts/paper_figures/build_fig4_a_core_optimization_candidate.py'
SPEC = dict(
    interpretation='Author-requested schematic parameter updates, not measured core locations or a fitted optimization trajectory.',
    candidate_region_count=3, updated_core_count=2,
    core_count_context='C compares one through four cores. Three possible regions and two resulting cores here are schematic, not a demonstrated three-to-two count optimization.',
    candidate_centres_mm=[[-6.4, 7.0], [6.65, 3.6], [-3.85, -7.2]],
    updated_centres_mm=[[-3.1, 2.2], [4.3, -1.4]],
    candidate_radius_mm=1.15, updated_radius_mm=1.05,
    candidate_fill='none', candidate_edge='#B8A3CF',
    updated_fill='#805AA6', updated_edge='#76519D',
    arrow_color='#9479B2', arrow_dash_pattern=[3.0, 2.5],
    search_direction_arrows_mm=[[[-5.8, 4.4], [-3.9, 3.2]], [[5.8, 2.5], [4.7, -.15]]],
    arrows='Two short direction cues, with no one-to-one connection from every candidate region to a final core.',
    question_mark_in_each_candidate_region=True,
    candidate_regions_unfilled=True,
    regions_and_question_marks_below_neuron_markers=True,
    update_arrows_are_not_propagation_paths=True,
    simulations_and_readout_unchanged=True,
)


def draw_a(fig):
    with compact.plate.group(fig, 'A'):
        first_axis = len(fig.axes)
        record = mechanism.draw(fig)
        spatial = fig.axes[first_axis + 1]
        for point in SPEC['candidate_centres_mm']:
            spatial.add_patch(Circle(point, SPEC['candidate_radius_mm'],
                fc='none', ec=SPEC['candidate_edge'], lw=.7, ls=(0, (2.5, 2)), zorder=.6))
            spatial.text(*point, '?', color='#82579F', fontsize=10.5,
                ha='center', va='center', zorder=.8)
        # Draw regions beneath the original neuron scatter (zorder=1) so that
        # both candidate and updated regions retain every neuronal marker.
        for point in SPEC['updated_centres_mm']:
            spatial.add_patch(Circle(point, SPEC['updated_radius_mm'],
                fc=SPEC['updated_fill'], ec=SPEC['updated_edge'], alpha=.50, lw=.8, zorder=.5))
        for start, end in SPEC['search_direction_arrows_mm']:
            spatial.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=7.5,
                color=SPEC['arrow_color'], lw=.85, linestyle=(0, SPEC['arrow_dash_pattern']),
                shrinkA=0, shrinkB=0, zorder=.7))
        record['schematic_core_optimization'] = SPEC
    compact.plate.CHECKS['A_revision'] = record


def main():
    pointer = json.loads((BASE / 'current_version.json').read_text())
    assert pointer['layout_version'] == 'compact_ai_v4_a_left_circuit_spacing_v10'
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    before = {name: sha(ROOT / name) for name in pointer['outputs_sha256']}
    shutil.copytree(BASE / 'source', OUT / 'source', dirs_exist_ok=True)
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
        left, right = int(np.ceil(x / 300 * aa.shape[1])), int(np.floor((x + w) / 300 * aa.shape[1]))
        top, bottom = int(np.ceil((232 - y - h) / 232 * aa.shape[0])), int(np.floor((232 - y) / 232 * aa.shape[0]))
        allowed[top:bottom, left:right] = True
        assert changed.any() and not np.any(changed & ~allowed)
    for name, digest in before.items():
        assert sha(ROOT / name) == digest, name
    record.update(status='CANDIDATE_PENDING_AUTHOR_VISUAL_REVIEW', updated_on='2026-10-10',
        base_layout_version=pointer['layout_version'], canonical_figure_replaced=False,
        unchanged_panels=unchanged)
    validation = dict(status='PASS', unchanged_panels=unchanged,
        all_pixels_outside_central_sheet_identical_to_v10=True,
        changed_pixels_in_central_sheet=int(changed.sum()),
        canonical_output_hashes_unchanged=True,
        source_coordinates_are_illustrative=True,
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
        'A中央为三个无填充问号候选区域、两个紫色core及少量虚线方向提示；')
    (OUT / 'panel_map.md').write_text(mapping)
    description = ('中央二维sheet改为三个无填充的浅紫色虚线候选区域，中心放问号，另画两个紫色更新core。'
        '不再逐一配对；只保留两条短虚线方向提示，所有新元素均位于原神经元点下层。'
        '位置为作者要求的示意坐标，不是实际优化轨迹、真实病灶位置或传播路径；'
        'core数量的实验比较仍见C，两组波形仍来自原冻结工作点。左侧回路、右侧采样与波形、B–I全部保持当前v10。')
    (OUT / 'README.md').write_text('# Figure 4A v12：简化core搜索区域\n\n' + description + '\n\n'
        '当前正式v10保留，本候选已输出独立A和完整拼版的PNG/PDF/SVG，待作者目视检查。'
        '全图中央sheet之外像素完全一致，B–I逐像素一致，正式导出哈希不变。\n\n'
        f'重建：`python {PRODUCER}`。\n')
    (OUT / 'figures/README.md').write_text('### fig4-panela.png / .pdf / .svg\n\n' + description + '\n'
        '**关注点**：三个问号区域不遮挡神经元点，两个紫色core清楚，视觉不形成逐一配对。\n\n'
        '### fig4-complete-layout.png / .pdf / .svg / -preview.png\n\n'
        '沿用当前三行A–I紧凑布局，只在A的中央二维图中增加多core位置更新示意。'
        '原始数据、其他面板和所有轴的位置不变。\n'
        '**关注点**：实际整图尺寸下问号和虚线箭头的可读性；待作者目视检查。\n')
    print('Multicore candidate complete; every pixel outside the central sheet unchanged.', flush=True)


if __name__ == '__main__':
    main()
