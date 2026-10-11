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
    compact.VERSION = 'compact_ai_v4_a_larger_left_circuit_v6'
    compact.TITLES['A'] = '左侧局部E/I回路放大20%，保留居中sheet与右侧采样及SEEG读出'
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
    prior = BASE / 'candidates/a_centered_sparse_circuits_20261010'
    with Image.open(prior / 'figures/fig4-complete-layout.png') as a, Image.open(OUT / 'figures/fig4-complete-layout.png') as b:
        aa, bb = np.asarray(a), np.asarray(b)
        # Exclude the enlarged circuit and its changed connector segments.
        # The rest of the sheet and the full right block must remain exact.
        cutoff_x = int(116 / 300 * aa.shape[1])
        cutoff_y = int(87 / 232 * aa.shape[0])
        np.testing.assert_array_equal(aa[:cutoff_y, cutoff_x:], bb[:cutoff_y, cutoff_x:])
    for p, digest in base_hashes.items():
        assert hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == digest
    registry_path = OUT / 'figure4_candidate_registry.json'
    registry = json.loads(registry_path.read_text())
    registry['panel_a']['A_region_from_x116_mm_pixel_identical_to_v5'] = True
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
    mapping = mapping.replace('右侧机制图仍待补充；', 'A左侧原回路宽高各放大20%，保留居中sheet、右侧错落E/I采样与SEEG双burst，仍待作者目视检查；')
    (OUT / 'panel_map.md').write_text(mapping)
    for script in [Path(__file__), Path(mechanism.__file__)]:
        shutil.copy2(script, OUT / 'source' / script.name)
    (OUT / 'README.md').write_text('# Figure 4A 机制重绘候选 v6：放大左侧局部回路\n\n'
        '左侧Local E/I circuit沿用原图内容，宽高各放大20%至77.40×60.88 mm，保持左边界和垂直中心。'
        '标题重新居中，放大虚线重新连接；图内0.5 mm比例尺随原图等比例放大，物理含义不变。'
        '中央sheet与右侧区域保持v5位置、尺寸与内容；旧放大框仍位于(-8.5,0) mm。'
        'Local sampling由68×34收至60×30 mm；MTA/MTB各由47收至34 mm宽，120 ms信号时间轴不变。'
        '中间网络与右上Local sampling共用E图已登记的直杆投影坐标，各杆触点严格共线。'
        '中间933个E和269个I示意点保留旧图的精确抽样坐标；触点直杆投影只修正示意绘制，'
        '不修改仿真坐标或重算读出。ICL2附近绿色虚线框加粗并加白描边，绿色采样范围加深，'
        '与右上Local sampling一一对应放大。'
        '五个倾斜椭圆以上方2个、下方3个错落放置，保留同一物理方向和尺度。'
        '每个区域中从冻结E/I坐标随机抽取6–7个E和1–2个I；蓝圆全部位于区域内部，数量小于E。'
        '所有局部E及外周E统一为空心红三角，I为空心蓝圆；移除原混淆的突触箭头、填实三角和重复近触点符号。'
        '不加公式、长短轴符号和下方说明文字；'
        'B–I八张独立PNG与正式版逐像素一致。完整拼版与独立A待作者目视检查。\n\n'
        'Local sampling下方标题改为SEEG readout，表示虚拟触点对底层SNN传播的读出。'
        '并列MTA/MTB两个burst，复用F已展示的事件10/8。'
        '三通道始终按ICL8、ICL6、ICL4物理顺序排列；MTA窗口4310–4430 ms、'
        '峰时4360/4368/4378 ms，MTB窗口3710–3830 ms、峰时3786/3770/3756 ms。'
        '直接截取F已冻结的30–80 Hz虚拟触点波形，以120 ms短窗保留burst前后基线，'
        '不人为锐化、不重滤波、不压缩时间。F仍保留15通道长时程结果；'
        '两种模式共用幅度比例，未移动通道时间或逐通道归一化。'
        '三通道片段仅说明两个冻结事件的相反到达顺序，不替代F的全部15通道结果。\n\n'
        '第三图与中间网络使用同一冻结示意神经元及直杆显示坐标，位置和视野严格对应；'
        '连接核及读出来自xy_left_20工作点。连接核长/短半轴尺度分别为'
        '0.5374/0.4031 mm，方向−22.805°。红色椭圆表示E→E连接抽样核，不是core边界，'
        '外轮廓采用rho=0.7的等距离水平；区域内E/I符号是局部群体示意，未重建具体邻接边。'
        '绿色圈为σ=0.25 mm二维高斯的95%质量半径，'
        '实际算子没有这个半径的硬截断。SEEG readout是虚拟触点发放密度代理的滤波读出，'
        '不是实测临床电压，未新增电磁场或电位前向模型。\n\n'
        '完整重排：`python scripts/paper_figures/build_fig4_a_mechanism_candidate.py`；'
        '仅A重画：`python scripts/paper_figures/draw_fig4_a_spatial_readout.py`。\n')
    (OUT / 'figures' / 'README.md').write_text(
        '### fig4-panela.png / .pdf / .svg\n\n'
        '左侧原回路等比例放大20%，标题居中、虚线重新连接；中央sheet和右侧区域保持上一版。'
        '五个椭圆以上二下三错落排布，每个内部都有空心红三角和少量蓝圆，移除会与神经元混淆的箭头。'
        '两个120 ms burst和刚性直杆保持，待作者目视检查。\n\n'
        '**关注点**：左侧回路放大后是否与中央sheet形成合适比例。\n\n'
        '### fig4-complete-layout.png / .pdf / .svg / -preview.png\n\n'
        '仅A左侧回路放大，B–I逐像素一致，中央和右侧图的位置及科学数据保持。'
        '正式current_version入口仍指向v4。\n\n'
        '**关注点**：右上采样与下方读出的层次、整图可读性。\n')
    print('A candidate complete; B-I pixel-identical; canonical package untouched.', flush=True)


if __name__ == '__main__':
    main()
