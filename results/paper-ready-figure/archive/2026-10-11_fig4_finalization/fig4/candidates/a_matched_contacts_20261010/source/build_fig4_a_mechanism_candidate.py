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
    compact.VERSION = 'compact_ai_v4_a_matched_contacts_v7'
    compact.TITLES['A'] = 'Local sampling与MTA/MTB统一标注ICL8、ICL6、ICL4，中央sheet放大6%'
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
        # Include all burst axes, ticks and titles; omit the changing callout
        # connector in the blank upper-left margin of the readout block.
        for x0, x1, y0, y1 in [(195, 295, 145, 177.8), (217, 295, 177.8, 186.5)]:
            left, right = [int(v / 300 * aa.shape[1]) for v in (x0, x1)]
            top, bottom = [int((232 - v) / 232 * aa.shape[0]) for v in (y1, y0)]
            np.testing.assert_array_equal(aa[top:bottom, left:right], bb[top:bottom, left:right])
    for name in ['burst_readout_arrays.npz', 'waveform_arrays.npz', 'rigid_contact_geometry.npz',
                 'spatial_reference.npz', 'overview_neuron_samples.npz', 'legacy_a_components.npz']:
        assert (prior / 'source' / name).read_bytes() == (OUT / 'source' / name).read_bytes()
    for p, digest in base_hashes.items():
        assert hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == digest
    registry_path = OUT / 'figure4_candidate_registry.json'
    registry = json.loads(registry_path.read_text())
    registry['panel_a']['left_artwork_and_title_pixel_identical_to_v6'] = True
    registry['panel_a']['burst_axes_and_labels_pixel_identical_to_v6'] = True
    registry['panel_a']['frozen_geometry_and_waveform_snapshots_unchanged'] = True
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
    mapping = mapping.replace('右侧机制图仍待补充；', 'A采样图与下方双burst统一标注ICL8/ICL6/ICL4，移除小箭头、中央sheet放大6%，仍待作者目视检查；')
    (OUT / 'panel_map.md').write_text(mapping)
    for script in [Path(__file__), Path(mechanism.__file__)]:
        shutil.copy2(script, OUT / 'source' / script.name)
    (OUT / 'README.md').write_text("""# Figure 4A 机制重绘候选 v7：采样触点与读出对应

Local sampling明确标出ICL8、ICL6、ICL4，与MTA/MTB三行波形的名称和顺序一致。
绿色放大框和右侧视野同步改为以ICL6为中心的8×4 mm范围，以容纳三者的真实直杆间距。
保留中间两个未标注触点，三处对应触点用深色边框和绿色高斯采样范围突出；不把旧ICL2触点直接改名。
右侧绘图区域仍为60×30 mm。五个倾斜椭圆保持上二下三错落结构、同一物理尺度和方向，
每个内部保留6–7个空心红色E三角及1–2个蓝色I圆；从同一冻结坐标中选取当前视野内的示意群体。
神经元符号随视野扩大缩小以避免拥挤。移除Local sampling与SEEG readout之间的小绿色箭头。
中央2D sheet保持v6的中心位置，宽高各放大6%至70.64 mm；左侧已放大20%的原回路保持。
两组burst的轴位置、尺寸、名称、波形均不变，B–I八个面板逐像素保持正式v4。

中间网络与右侧采样共用E图登记的刚性直杆显示坐标，所有触点严格共线。
显示投影不改变仿真坐标或重算读出。原灰色回路框仍位于(-8.5,0) mm并避开电极；
绿色框表示当前三个电极触点附近的采样视野，两种放大用途分开。
椭圆采用xy_left_20工作点的E→E连接抽样核，长/短半轴尺度为0.5374/0.4031 mm、方向−22.805°，
外轮廓仍为rho=0.7；它们不是拟合core边界，示意符号不表示已重建具体邻接边。
绿色圈仍是σ=0.25 mm二维高斯的95%质量半径，算子没有该半径的硬截断。

两组120 ms burst复用F的事件10/8，直接截取冻结30–80 Hz波形，不锐化、不重滤波、不压缩时间。
MTA窗口4310–4430 ms，三通道峰时4360/4368/4378 ms；MTB窗口3710–3830 ms，峰时3786/3770/3756 ms。
两种模式共用幅度比例，不移动通道时间或逐通道归一化。A展示两个事件的相反到达顺序，F保留15通道长记录。
SEEG readout表示虚拟触点对底层SNN传播的发放密度代理读出，不是实测电压，也未新增电位前向模型。

完整拼版和独立A已导出PNG/PDF/SVG，待作者目视检查；正式v4输出未覆盖。
完整重排：`python scripts/paper_figures/build_fig4_a_mechanism_candidate.py`；
仅A重画：`python scripts/paper_figures/draw_fig4_a_spatial_readout.py`。
""")
    (OUT / 'figures' / 'README.md').write_text("""### fig4-panela.png / .pdf / .svg

Local sampling三处触点标注ICL8、ICL6、ICL4，与下方两组burst逐行对应；视野保留真实直杆间距。
移除小绿色箭头，中央sheet围绕原中心放大6%，保留左侧原回路及右侧五个错落的E/I群体。
**关注点**：采样位置、触点名称和传播读出的对应是否清楚，待作者目视检查。

### fig4-complete-layout.png / .pdf / .svg / -preview.png

本次修改仅涉及A的中央sheet和采样图，B–I与正式v4逐像素一致。
左侧回路与两组burst的像素区域保持v6；正式输出仍未替换。
**关注点**：中央sheet放大后的比例、三行整体可读性。
""")
    print('A candidate complete; B-I pixel-identical; canonical package untouched.', flush=True)


if __name__ == '__main__':
    main()
