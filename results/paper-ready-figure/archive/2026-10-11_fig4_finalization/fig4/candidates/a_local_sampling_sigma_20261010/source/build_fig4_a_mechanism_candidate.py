#!/usr/bin/env python3
"""Replace only A in the approved v4 layout; retain a reviewable candidate."""
from pathlib import Path
import hashlib
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.paper_figures import build_fig4_compact_ai as compact
from scripts.paper_figures import draw_fig4_a_spatial_readout as mechanism
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
    assert pointer['layout_version'] == 'compact_ai_tighter_columns_larger_squares_v4'
    base_hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in pointer['outputs_sha256']}
    compact.OUT = OUT
    compact.PRODUCER = 'scripts/paper_figures/build_fig4_a_mechanism_candidate.py'
    compact.VERSION = 'compact_ai_v4_a_local_sampling_sigma_v9'
    compact.TITLES['A'] = '仅右侧Local sampling显示实际sigma的绿色采样权重，中央只留无填充定位框'
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
    prior = BASE / 'candidates/a_larger_left_circuit_20261010'
    with Image.open(prior / 'figures/fig4-complete-layout.png') as a, Image.open(OUT / 'figures/fig4-complete-layout.png') as b:
        aa, bb = np.asarray(a), np.asarray(b)
        assert aa.shape == bb.shape
        # Preserve the enlarged left artwork/title, excluding outgoing connectors.
        cutoff_x = int(83 / 300 * aa.shape[1])
        cutoff_y = int(87 / 232 * aa.shape[0])
        np.testing.assert_array_equal(aa[:cutoff_y, :cutoff_x], bb[:cutoff_y, :cutoff_x])
    last = BASE / 'candidates/a_consecutive_contacts_20261010'
    with Image.open(last / 'figures/fig4-complete-layout.png') as a, Image.open(OUT / 'figures/fig4-complete-layout.png') as b:
        aa, bb = np.asarray(a), np.asarray(b)
        # The actual burst axes, labels and titles remain unchanged; exclude the
        # changing gray zoom connector in the blank upper-left margin.
        for x0, x1, y0, y1 in [(195, 295, 145, 177.8), (217, 295, 177.8, 186.5)]:
            left, right = [int(v / 300 * aa.shape[1]) for v in (x0, x1)]
            top, bottom = [int((232 - v) / 232 * aa.shape[0]) for v in (y1, y0)]
            np.testing.assert_array_equal(aa[top:bottom, left:right], bb[top:bottom, left:right])
    assert (last / 'source/burst_readout_arrays.npz').read_bytes() == (OUT / 'source/burst_readout_arrays.npz').read_bytes()
    last_a = json.loads((last / 'mechanism_validation.json').read_text())
    current_a = json.loads((OUT / 'figure4_candidate_registry.json').read_text())['panel_a']
    for key in ['local_population_coordinates_mm', 'kernel_centres_E_coordinates_mm',
                'sampling_contact_coordinates_overview_mm', 'sampling_zoom_field_of_view_mm',
                'sheet_axes_mm', 'left_circuit_axes_mm']:
        assert last_a[key] == current_a[key], key
    for name in ['waveform_arrays.npz', 'rigid_contact_geometry.npz',
                 'spatial_reference.npz', 'overview_neuron_samples.npz', 'legacy_a_components.npz']:
        assert (prior / 'source' / name).read_bytes() == (OUT / 'source' / name).read_bytes()
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
    registry['panel_a']['left_artwork_and_title_pixel_identical_to_v6'] = True
    registry['panel_a']['burst_axes_and_labels_pixel_identical_to_v8'] = True
    registry['panel_a']['sampling_geometry_and_circuits_preserved_from_v8'] = True
    registry['panel_a']['consecutive_contact_crops_verified_against_frozen_F'] = True
    registry['panel_a']['frozen_geometry_and_full_F_waveform_snapshot_unchanged'] = True
    registry.update(status='CANDIDATE_PENDING_AUTHOR_VISUAL_REVIEW',
        base_layout_version=pointer['layout_version'], canonical_figure_replaced=False,
        unchanged_panels=comparisons, original_left_circuit_artwork_preserved=True)
    compact.write(registry_path, registry)
    compact.write(OUT / 'mechanism_validation.json', dict(registry['panel_a'],
        unchanged_panels=comparisons, canonical_output_hashes_unchanged=True))
    qa = json.loads((OUT / 'visual_qa.json').read_text())
    qa['checks']['A_revision'] = registry['panel_a']
    qa['checks']['unchanged_panels'] = comparisons
    qa['checks']['canonical_output_hashes_unchanged'] = True
    compact.write(OUT / 'visual_qa.json', qa)
    mapping = (OUT / 'panel_map.md').read_text().replace('# 当前 Figure 4：A–I', '# Figure 4A 机制重绘候选：A–I')
    mapping = mapping.replace('右侧机制图仍待补充；', 'A只在右侧采样放大图显示sigma=0.25 mm绿色权重，中央只留无填充定位框；三个连续触点、局部回路与双burst保持，待作者目视检查；')
    (OUT / 'panel_map.md').write_text(mapping)
    for script in [Path(__file__), Path(mechanism.__file__)]:
        shutil.copy2(script, OUT / 'source' / script.name)
    (OUT / 'README.md').write_text("""# Figure 4A 机制重绘候选 v9：右侧采样权重按实际sigma显示

Local sampling保留3.8×1.9 mm小视野及上一版局部回路的坐标和大小，框内恰好三个连续触点。
上下统一采用作者指定的SL3、SL4、SL5标签，对应冻结数据中的ICL3、ICL4、ICL5；
映射只用于A的示意显示，不重命名底层数据或其他面板。几何保留真实刚性直杆间距，
放大框中心为两端触点的中点，框内没有额外触点，也不隔一个取一个。
仅右侧Local sampling显示三个触点的绿色高斯权重，采用实际sigma=0.25 mm，越远越淡，不画硬边界圈。
中央sheet不画采样权重、范围圈或绿色填充；中性灰色空心虚线框和连线只定位放大视野。
绿色渐变绘制到4 sigma并由小视野裁切，不挪动触点或缩小circuits来塞入完整权重尾部。
理论二维95%质量半径为0.61194 mm，仅在元数据记录，图中不加公式、半径或额外文字。
原左侧灰框继续位于(-8.5,0) mm、远离电极。通向SEEG readout的小箭头不再出现。
五个倾斜椭圆继续上二下三错落排列，含6–7个空心红E三角和1–2个内部蓝I圆；
恢复v6的符号大小与3.8×1.9 mm物理视野，从同一冻结坐标选取当前邻域的神经元。
中央sheet沿用v7的70.64 mm正方形及中心位置，左侧已放大20%的原回路保持。

下方两组120 ms波形保持v8的实际冻结波形，仍取F的事件10/8和同一时间窗；轴、标签和波形逐像素一致。
SL3/SL4/SL5按作者指定顺序自上而下排列。MTA窗口4310–4430 ms，峰时4386/4378/4372 ms；
MTB窗口3710–3830 ms，峰时3756/3756/3762 ms。MTB的前两峰在2 ms采样分辨率下同一时刻，
SL3振幅较弱；没有为了展示严格顺序而移动峰或逐通道归一化。
两模式共用幅度比例，直接截取F已冻结30–80 Hz信号，不重滤波、不锐化、不压缩时间。
验证逐样本等于对应源通道和源窗口。F仍保留完整15通道长记录，B–I逐像素保持正式v4。

椭圆继续使用同一xy_left_20参考E→E抽样核，长/短半轴尺度0.5374/0.4031 mm、方向−22.805°，
外轮廓rho=0.7；它们不是拟合core边界。绿色区域严格按原有σ=0.25 mm读出算子显示，权重形式与冻结A_zoom_in及当前sample_envelopes核定义核对。
SEEG readout是底层SNN的虚拟触点发放密度代理，不是实测电压，未新增电位前向模型。

独立A和完整拼版已导出PNG/PDF/SVG，待作者目视检查；正式v4输出未覆盖。
完整重排：`python scripts/paper_figures/build_fig4_a_mechanism_candidate.py`；
仅A重画：`python scripts/paper_figures/draw_fig4_a_spatial_readout.py`。
""")
    (OUT / 'figures' / 'README.md').write_text("""### fig4-panela.png / .pdf / .svg

采样小框内只显示三个连续触点，按作者要求标为SL3、SL4、SL5，下方展示对应三通道的冻结波形。
绿色采样权重仅画在右侧，sigma=0.25 mm；中央只保留无填充的灰色定位虚线框。
局部circuits、神经元、中央sheet与左侧回路大小不变，下方两组波形保持v8。
**关注点**：右侧绿色区域能否清楚表达触点周围的距离权重，待作者目视检查。

### fig4-complete-layout.png / .pdf / .svg / -preview.png

本次仅修改A采样权重及定位框的显示，B–I与正式v4逐像素一致，左侧回路和下方两组波形保持。
正式入口的pending_panel_a_revision登记本候选，正式图输出未覆盖。
**关注点**：A的比例和局部回路可读性、三行整体布局。
""")
    print('A candidate complete; B-I pixel-identical; canonical package untouched.', flush=True)


if __name__ == '__main__':
    main()
