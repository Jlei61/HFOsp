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
    compact.VERSION = 'compact_ai_v4_a_spatial_readout_candidate_v1'
    compact.TITLES['A'] = '二维椭圆E→E连接核、空间网络、局部高斯采样与虚拟触点读出'
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
    for p, digest in base_hashes.items():
        assert hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == digest
    registry_path = OUT / 'figure4_candidate_registry.json'
    registry = json.loads(registry_path.read_text())
    registry.update(status='CANDIDATE_PENDING_AUTHOR_VISUAL_REVIEW',
        base_layout_version=pointer['layout_version'], canonical_figure_replaced=False,
        unchanged_panels=comparisons)
    compact.write(registry_path, registry)
    compact.write(OUT / 'mechanism_validation.json', dict(registry['panel_a'],
        unchanged_panels=comparisons, canonical_output_hashes_unchanged=True))
    mapping = (OUT / 'panel_map.md').read_text().replace('# 当前 Figure 4：A–I', '# Figure 4A 机制重绘候选：A–I')
    mapping = mapping.replace('右侧机制图仍待补充；', 'A已补充二维连接方向及局部采样读出，仍待作者目视检查；')
    (OUT / 'panel_map.md').write_text(mapping)
    for script in [Path(__file__), Path(mechanism.__file__)]:
        shutil.copy2(script, OUT / 'source' / script.name)
    (OUT / 'README.md').write_text('# Figure 4A 机制重绘候选\n\n'
        '基于已同步远端的50 mm方形A–I版式，仅重绘A。当前正式图与B–I数据保持；'
        'B–I八张独立PNG与正式版逐像素一致。完整拼版与独立A均待作者目视检查。\n\n'
        'A将椭圆连接核放在顶视二维平面，参考长轴角为−22.805°，在局部和空间网络中一致；'
        '红色椭圆表示E→E连接抽样核，不是core边界。左侧示例框保留在(−8.5,0) mm，'
        '椭圆表示对称的邻接倾向，不将长轴箭头解释为单向传播。下方保留E/I与适应反馈。\n\n'
        '右侧绿色高斯权重以σ=0.25 mm汇集邻近E发放活动；虚线为连续二维高斯的95%质量半径示意，'
        '代码本身没有该半径硬截断，也不保证离散神经元的95%质量恰在圆内。'
        '最右三通道直接复用F冻结30–80 Hz信号与统一幅度尺度。A空间基底仍是原有示意来源，'
        '不能把该基底认作F这一事件的精确网络；本图未实现电磁场、体积传导或电位前向模型。\n\n'
        '完整重排：`python scripts/paper_figures/build_fig4_a_mechanism_candidate.py`（需原分析工作区依赖）；'
        '仅A重画：`python scripts/paper_figures/draw_fig4_a_spatial_readout.py`（只需本包冻结数组及NumPy/Matplotlib）。\n')
    (OUT / 'figures' / 'README.md').write_text(
        '### fig4-panela.png / .pdf / .svg\n\n'
        'A重绘候选：二维椭圆连接方向、完整空间基底、邻近神经元高斯采样及三触点活动读出。'
        '实际信号沿用F，椭圆和采样圈含义不同；待作者目视检查。\n\n'
        '**关注点**：长轴方向是否直观，局部框与电极是否分开，采样权重是否清楚。\n\n'
        '### fig4-complete-layout.png / .pdf / .svg / -preview.png\n\n'
        '仅A采用候选机制图，B–I保持已确认的紧凑方形布局，八张独立PNG逐像素一致。'
        '正式current_version入口仍指向已发布版。\n\n'
        '**关注点**：首行四部分的视觉衔接与全文字号比例。\n')
    print('A candidate complete; B-I pixel-identical; canonical package untouched.', flush=True)


if __name__ == '__main__':
    main()
