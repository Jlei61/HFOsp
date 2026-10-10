#!/usr/bin/env python3
"""Rebuild current Figure 4 into its review package, preserving panels B-I."""
from pathlib import Path
import hashlib
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.paper_figures import build_fig4_compact_ai as compact
from scripts.paper_figures import draw_fig4_a_spatial_readout as mechanism
from scripts.paper_figures import build_fig4_left_circuit_detail as left_detail
import numpy as np
from PIL import Image
from matplotlib.text import Text
from matplotlib.transforms import Bbox

plate = compact.plate
BASE = ROOT / 'results/paper-ready-figure/fig4'
OUT = mechanism.OUT


def draw_a(fig):
    with plate.group(fig, 'A'):
        record = mechanism.draw(fig)
    plate.CHECKS['A_revision'] = record


def verify_layout(fig):
    fig.canvas.draw(); renderer = fig.canvas.get_renderer()
    previous = json.loads((BASE / 'visual_qa.json').read_text())['checks']['layout']
    bounds = {}
    for key, group in plate.GROUPS.items():
        boxes = [ax.get_tightbbox(renderer) for ax in group['axes']]
        boxes += [text.get_window_extent(renderer) for text in group['texts']]
        box = Bbox.union([b for b in boxes if b is not None])
        bounds[key] = (np.array([box.x0, box.y0, box.x1, box.y1]) / fig.dpi * 25.4).tolist()
    for key in 'BCDEFGHI':
        np.testing.assert_allclose(bounds[key], previous['visible_bounds_mm'][key], atol=1e-7)
    for a, b in zip('BCDFGH', 'CDEGHI'):
        assert bounds[b][0] > bounds[a][2]
    for obj in fig.findobj(Text):
        if not obj.get_visible() or not obj.get_text() or not obj.get_in_layout(): continue
        # Axis labels/ticks outside unused tick locations are not drawn.
        box = obj.get_window_extent(renderer)
        if obj.axes is not None and not obj.axes.axison: continue
        if obj.axes is not None and obj not in obj.axes.texts: continue
        assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height, obj.get_text()
    gap = bounds['A'][1] - max(bounds[k][3] for k in 'BCDE')
    assert 5 < gap < 13, gap
    plate.CHECKS['layout'] = dict(previous, visible_bounds_mm=bounds,
        approved_v4_B_through_I_bounds_preserved=True,
        row_gaps_mm=[gap, previous['row_gaps_mm'][1]],
        A_uses_previously_reserved_mechanism_space=True)


def main():
    pointer = json.loads((BASE / 'current_version.json').read_text())
    assert pointer['layout_version'] in (
        'compact_ai_tighter_columns_larger_squares_v4',
        'compact_ai_v4_a_left_circuit_spacing_v10',
    )
    base_hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in pointer['outputs_sha256']}
    left_detail.main()
    compact.OUT = OUT
    compact.PRODUCER = 'scripts/paper_figures/build_fig4_a_mechanism_candidate.py'
    compact.VERSION = 'compact_ai_v4_a_left_circuit_spacing_v10'
    compact.TITLES['A'] = '仅调整左侧中央E/I连线间距和z下降节点的细密虚线边框'
    compact.draw_a = draw_a
    compact.verify_layout = verify_layout
    compact.render()
    # Enforce the requested scope directly on the delivered panels.
    comparisons = {}
    for letter in 'bcdefghi':
        name = f'figures/fig4-panel{letter}.png'
        with Image.open(BASE / name) as a, Image.open(OUT / name) as b:
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        comparisons[letter.upper()] = 'PIXEL_IDENTICAL_TO_APPROVED_V4'
    prior = BASE / 'candidates/a_local_sampling_sigma_20261010'
    detail = json.loads((OUT / 'source/left_circuit_detail_validation.json').read_text())
    current_a = json.loads((OUT / 'figure4_candidate_registry.json').read_text())['panel_a']
    geometry = json.loads((OUT / 'source/legacy_a_geometry.json').read_text())['axes'][0]
    x0, y0, x1, y1 = detail['changed_bounds_pixels']
    x, y, w, h = current_a['left_circuit_axes_mm']
    # Map the verified native detail footprint into the full figure. Padding
    # covers antialiasing from the established raster placement, not other art.
    data_x = np.array([x0, x1]) + geometry['image_extent'][0]
    data_y = np.array([y0, y1]) + geometry['image_extent'][3]
    mm_x = x + (data_x - geometry['xlim'][0]) / np.ptp(geometry['xlim']) * w
    mm_y = y + h - (data_y - geometry['ylim'][1]) / np.ptp(geometry['ylim']) * h
    allowed_mm = [mm_x[0] - .3, mm_y[1] - .3, mm_x[1] + .3, mm_y[0] + .3]
    with Image.open(prior / 'figures/fig4-complete-layout.png') as a, Image.open(OUT / 'figures/fig4-complete-layout.png') as b:
        aa, bb = np.asarray(a), np.asarray(b)
        assert aa.shape == bb.shape
        changed = np.any(aa != bb, axis=2)
        allowed = np.zeros(changed.shape, bool)
        left = int(np.floor(allowed_mm[0] / 300 * aa.shape[1]))
        right = int(np.ceil(allowed_mm[2] / 300 * aa.shape[1]))
        top = int(np.floor((232 - allowed_mm[3]) / 232 * aa.shape[0]))
        bottom = int(np.ceil((232 - allowed_mm[1]) / 232 * aa.shape[0]))
        allowed[top:bottom, left:right] = True
        assert changed.any() and not np.any(changed & ~allowed)
    for name in ['burst_readout_arrays.npz', 'waveform_arrays.npz', 'rigid_contact_geometry.npz',
                 'spatial_reference.npz', 'overview_neuron_samples.npz']:
        assert (prior / 'source' / name).read_bytes() == (OUT / 'source' / name).read_bytes()
    last_a = json.loads((prior / 'mechanism_validation.json').read_text())
    for key in ['local_population_coordinates_mm', 'kernel_centres_E_coordinates_mm',
                'sampling_contact_coordinates_overview_mm', 'sampling_zoom_field_of_view_mm',
                'sheet_axes_mm', 'left_circuit_axes_mm', 'sampling_weight_display']:
        assert last_a[key] == current_a[key], key
    # The three adjacent channels must be real crops from the frozen F source,
    # not the old spaced channels under different labels.
    meta = json.loads((OUT / 'source/mechanism_source.json').read_text())
    with np.load(OUT / 'source/waveform_arrays.npz') as frozen, np.load(OUT / 'source/burst_readout_arrays.npz') as burst:
        rows = [list(frozen['contact_names']).index(n) for n in burst['names']]
        for mode, event in enumerate(meta['mode_showcases']['events']):
            start, end = event['window_absolute_ms']
            selected = (frozen['absolute_time_ms'] >= start) & (frozen['absolute_time_ms'] <= end)
            np.testing.assert_array_equal(burst['waveforms'][mode], frozen['filtered_contact_activity'][rows][:, selected])
            np.testing.assert_array_equal(burst['time_ms'], frozen['absolute_time_ms'][selected] - start)
        np.testing.assert_array_equal(burst['names'], ['ICL3', 'ICL4', 'ICL5'])
        np.testing.assert_array_equal(burst['display_labels'], ['SL3', 'SL4', 'SL5'])
    for p, digest in base_hashes.items():
        assert hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == digest
    registry_path = OUT / 'figure4_candidate_registry.json'
    registry = json.loads(registry_path.read_text())
    registry['panel_a']['left_detail_validation'] = detail
    registry['panel_a']['left_detail_allowed_bounds_mm'] = allowed_mm
    registry['panel_a']['all_pixels_outside_left_detail_identical_to_v9'] = True
    registry['panel_a']['burst_axes_and_labels_pixel_identical_to_v9'] = True
    registry['panel_a']['sampling_geometry_and_circuits_preserved_from_v9'] = True
    registry['panel_a']['consecutive_contact_crops_verified_against_frozen_F'] = True
    registry['panel_a']['frozen_geometry_and_full_F_waveform_snapshot_unchanged'] = True
    registry.update(status='CANDIDATE_PENDING_AUTHOR_VISUAL_REVIEW',
        base_layout_version=pointer['layout_version'], canonical_figure_replaced=False,
        unchanged_panels=comparisons, original_left_circuit_artwork_preserved_except_requested_details=True)
    compact.write(registry_path, registry)
    compact.write(OUT / 'mechanism_validation.json', dict(registry['panel_a'],
        unchanged_panels=comparisons, canonical_output_hashes_unchanged=True))
    qa = json.loads((OUT / 'visual_qa.json').read_text())
    qa['checks']['A_revision'] = registry['panel_a']
    qa['checks']['unchanged_panels'] = comparisons
    qa['checks']['canonical_output_hashes_unchanged'] = True
    compact.write(OUT / 'visual_qa.json', qa)
    mapping = (OUT / 'panel_map.md').read_text().replace('# 当前 Figure 4：A–I', '# Figure 4A 机制重绘候选：A–I')
    mapping = mapping.replace('右侧机制图仍待补充；', 'A仅加宽左侧中央E/I连线间距并细化z下降边框，其余像素保持v9，待作者目视检查；')
    (OUT / 'panel_map.md').write_text(mapping)
    for script in [Path(__file__), Path(mechanism.__file__)]:
        shutil.copy2(script, OUT / 'source' / script.name)
    (OUT / 'README.md').write_text("""# Figure 4A 机制重绘候选 v10：左侧回路两处细节

仅修改左侧中央E/I之间的两条竖向连线，以及z↓节点的虚线边框。
中央红色E→I下行箭头左移0.07示意单位，蓝色I→E抑制连线及T端右移0.07单位，
两条竖线的间距由0.08增至0.22单位，在当前拼版中约由0.66增至1.80 mm。
E/I神经元、其他连接及z文字位置不变；蓝色通向z的短分支随主干起点更新。
z↓圆框线宽由原生绘图的1.60 pt减至0.60 pt，虚线节距改为1.5/1.5，形成更细密的边框。
标题、左侧框、比例尺、图尺寸及布局不变。

冻结原始Matplotlib绘图函数并通过原裁切、缩放和比例尺合成链生成左侧图。
修改前的重建与v9的整块左图逐像素一致；修改后仅中央连线和z边框附近的33230个源像素变化。
完整拼版进一步核对：该局部足迹及抗锯齿边缘之外的所有像素与v9一致。
中央sheet、右侧Local sampling、σ=0.25 mm绿色权重及两组波形保持，B–I逐像素保持正式v4。

继承v9的科学口径和来源：三连续源触点ICL3/ICL4/ICL5显示为SL3/SL4/SL5；
两组120 ms波形仍为F事件10/8的冻结截取，MTB前两峰在2 ms分辨率下相同，不移峰或逐通道放大。
绿色区域仅在右侧显示现有虚拟触点的高斯采样权重；中央只留无填充灰色定位框。
局部椭圆为连接核示意，读出为SNN发放密度代理，不表示新增的电位前向模型或实测SEEG电压。

独立A和完整拼版已导出PNG/PDF/SVG，待作者目视检查；正式v4输出未覆盖。
完整重排：`python scripts/paper_figures/build_fig4_a_mechanism_candidate.py`，会先重建左侧细节；
单独重建左图：`python scripts/paper_figures/build_fig4_left_circuit_detail.py`；
仅A重新导出：`python scripts/paper_figures/draw_fig4_a_spatial_readout.py`。
""")
    (OUT / 'figures' / 'README.md').write_text("""### fig4-panela.png / .pdf / .svg

左侧中央红色下行箭头和蓝色抑制连线稍微拉开，z↓外圈改为更细密的虚线。
图内其他内容及布局保持v9，修改范围之外的完整拼版像素一致。
**关注点**：两条竖线是否清楚分开、z↓边框是否足够纤细；待作者目视检查。

### fig4-complete-layout.png / .pdf / .svg / -preview.png

本次只修改A左侧的两处局部细节，中央、右侧和B–I保持。
正式入口的pending_panel_a_revision登记此候选，正式输出未覆盖。
**关注点**：实际版面大小下，左侧细节的可读性。
""")
    print('A candidate complete; B-I pixel-identical; canonical package untouched.', flush=True)


if __name__ == '__main__':
    main()
