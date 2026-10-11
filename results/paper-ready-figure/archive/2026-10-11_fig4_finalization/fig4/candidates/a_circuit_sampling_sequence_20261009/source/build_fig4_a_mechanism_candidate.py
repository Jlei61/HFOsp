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
    compact.VERSION = 'compact_ai_v4_a_circuit_sampling_sequence_v2'
    compact.TITLES['A'] = '原局部回路和空间网络、物理连接与采样合图、五触点传播序列'
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
    with Image.open(BASE / 'figures/fig4-complete-layout.png') as a, Image.open(OUT / 'figures/fig4-complete-layout.png') as b:
        aa, bb = np.asarray(a), np.asarray(b)
        cutoff_x = int(156.4 / 300 * aa.shape[1])
        cutoff_y = int(83 / 232 * aa.shape[0])
        np.testing.assert_array_equal(aa[:cutoff_y, :cutoff_x], bb[:cutoff_y, :cutoff_x])
    plate.CHECKS['A_revision']['original_first_two_components_pixel_identical'] = True
    for p, digest in base_hashes.items():
        assert hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == digest
    registry_path = OUT / 'figure4_candidate_registry.json'
    registry = json.loads(registry_path.read_text())
    registry.update(status='CANDIDATE_PENDING_AUTHOR_VISUAL_REVIEW',
        base_layout_version=pointer['layout_version'], canonical_figure_replaced=False,
        unchanged_panels=comparisons, original_first_two_components_pixel_identical=True)
    compact.write(registry_path, registry)
    compact.write(OUT / 'mechanism_validation.json', dict(registry['panel_a'],
        unchanged_panels=comparisons, canonical_output_hashes_unchanged=True))
    mapping = (OUT / 'panel_map.md').read_text().replace('# 当前 Figure 4：A–I', '# Figure 4A 机制重绘候选：A–I')
    mapping = mapping.replace('右侧机制图仍待补充；', 'A保留原左侧回路，第三图合并物理椭圆连接与采样，右侧呈现五触点传播序列，仍待作者目视检查；')
    (OUT / 'panel_map.md').write_text(mapping)
    for script in [Path(__file__), Path(mechanism.__file__)]:
        shutil.copy2(script, OUT / 'source' / script.name)
    (OUT / 'README.md').write_text('# Figure 4A 机制重绘候选 v2\n\n'
        '恢复正式v4左侧局部E/I回路与完整网络，前两个组件逐像素保持；第三组件合并二维椭圆连接、'
        '邻近神经元和代表性ICL3触点的高斯采样，统一物理尺度。去掉新增公式、长短轴符号和下方说明文字；'
        'B–I八张独立PNG与正式版逐像素一致。完整拼版与独立A待作者目视检查。\n\n'
        '右侧使用已展示于F的MTA事件10及ICL5至ICL1连续五触点，波峰绝对时刻依次为'
        '4372、4378、4386、4392、4398 ms；黑点及虚线直接标记这些真实波峰，未移动各通道时间、'
        '未做逐通道幅度归一化。它是这一示例事件的传播序列，不表示所有事件都按此顺序传播。\n\n'
        '第三图的神经元、触点坐标与连接核来自同一xy_left_20工作点：长/短半轴尺度分别为'
        '0.5374/0.4031 mm，方向−22.805°。红色椭圆表示E→E连接抽样核，不是core边界，'
        '红色连接箭头为抽样关系示意，未重建具体邻接边。绿色圈为σ=0.25 mm二维高斯的95%质量半径，'
        '实际算子没有这个半径的硬截断。显示的是发放密度读出，未计算电磁场或电位前向模型。\n\n'
        '前两图继续用原冻结示意基底；第三、四图用F真实工作点和同一已展示事件。'
        '完整重排：`python scripts/paper_figures/build_fig4_a_mechanism_candidate.py`；'
        '仅A重画：`python scripts/paper_figures/draw_fig4_a_spatial_readout.py`。\n')
    (OUT / 'figures' / 'README.md').write_text(
        '### fig4-panela.png / .pdf / .svg\n\n'
        '恢复原左侧回路，将物理椭圆连接与电极局部采样合并在第三张图，去掉公式和下方说明。'
        '右侧沿用F同一事件的五触点真实波形，点与虚线显示波峰序列；待作者目视检查。\n\n'
        '**关注点**：合图中的连接范围与采样范围是否直观，五通道延迟是否清楚。\n\n'
        '### fig4-complete-layout.png / .pdf / .svg / -preview.png\n\n'
        '仅A右侧采用新候选，原左侧两个组件及B–I逐像素保持。正式current_version入口仍指向v4。\n\n'
        '**关注点**：A四部分尺度与整图可读性。\n')
    print('A candidate complete; B-I pixel-identical; canonical package untouched.', flush=True)


if __name__ == '__main__':
    main()
