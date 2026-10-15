"""Two pooled order observables on one paired-axis parameter-response figure.

Exactly the existing single-network statistics; equal numerical scales on both
axes. No smoothing, label conditioning, new simulation or inferred uncertainty.
"""
from pathlib import Path
import csv
import json
import hashlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path('/data/hfosp/topic4_sef_hfo/label_free_dense_parameter_search_20260915')
COLORS = {'rank': '#0072B2', 'local_order': '#D55E00'}


def render(root=ROOT):
    root = Path(root)
    source = root / 'key_results/overall_acceptance_20260916/source_data.csv'
    rows = list(csv.DictReader(source.open()))
    selected = [r for r in rows if r['observable'] in COLORS]
    assert len(selected) == 156 and all(r['mode'] == 'ALL' for r in selected)
    plan = json.loads((root / 'plan.json').read_text())
    anchor = json.loads((root / 'candidates' / f"{plan['reference_id']}.json").read_text())
    reference = [0., 0., 0., 0., anchor['parameters']['EE_core_to_out_scale'],
                 anchor['parameters']['EE_angle_offset_deg']]
    dest = root / 'key_results/combined_order_acceptance_20260916'
    figures = dest / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Noto Sans CJK JP', 'DejaVu Sans'],
        'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10,
        'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.fontsize': 9,
        'axes.linewidth': .7, 'axes.spines.top': False, 'legend.frameon': False,
        'svg.fonttype': 'none', 'pdf.fonttype': 42, 'savefig.facecolor': 'white',
        'axes.unicode_minus': True,
    })
    fig, axes = plt.subplots(2, 3, figsize=(183 / 25.4, 115 / 25.4))
    fig.subplots_adjust(left=.12, right=.88, bottom=.12, top=.84, wspace=.28, hspace=.62)
    settings = [
        (0, '左核 x', '中心位移 (mm)', [-1.5, 0, 1.5]),
        (1, '左核 y', '中心位移 (mm)', [-1.5, 0, 1.5]),
        (4, '核内→核外 EE', '权重倍率', [1., 1.25, 1.5]),
        (2, '右核 x', '中心位移 (mm)', [-1.5, 0, 1.5]),
        (3, '右核 y', '中心位移 (mm)', [-2, -1, 0]),
        (5, '全局 EE 方向', '轴偏移 (°)', [-20, 0, 20]),
    ]
    panels = []
    for k, (j, title, xlabel, ticks) in enumerate(settings):
        left = axes.ravel()[k]
        right = left.twinx()
        for ax, metric, marker in [(left, 'rank', 'o'), (right, 'local_order', 's')]:
            line = sorted([r for r in selected if int(r['axis']) == j and r['observable'] == metric],
                          key=lambda r: float(r['parameter_value']))
            x = np.array([float(r['parameter_value']) for r in line])
            y = np.array([float(r['value']) for r in line])
            assert len(x) == 13 and np.all(np.diff(x) > 0)
            assert np.all(np.isfinite(y)) and np.all((0 <= y) & (y < .42))
            ax.plot(x, y, color=COLORS[metric], marker=marker, markersize=3.2,
                    linewidth=1.25, markeredgewidth=.4, zorder=3)
            ax.set_ylim(0, .42)
            ax.set_yticks([0, .1, .2, .3, .4])
            ax.tick_params(axis='y', colors=COLORS[metric], length=3, width=.65, pad=3)
            panels.append(dict(parameter_axis=j, metric=metric, points=len(x),
                               parameter_values=x.tolist(), values=y.tolist()))
        left.spines['right'].set_visible(False)
        left.spines['left'].set_color(COLORS['rank'])
        right.spines['left'].set_visible(False)
        right.spines['right'].set_visible(True)
        right.spines['right'].set_color(COLORS['local_order'])
        right.spines['bottom'].set_visible(False)
        left.tick_params(axis='y', labelleft=k % 3 == 0)
        right.tick_params(axis='y', labelright=k % 3 == 2)
        left.axvline(reference[j], color='#999999', linestyle=':', linewidth=.8, zorder=1)
        left.set(title=title, xlabel=xlabel, xticks=ticks)
        left.tick_params(axis='x', direction='out', length=3, width=.65, pad=3)
        left.margins(x=.04)
    fig.text(.027, .49, '平均归一化 rank 误差 ↓', rotation=90,
             ha='center', va='center', color=COLORS['rank'], fontsize=10)
    fig.text(.976, .49, '杆内先后概率误差 ↓', rotation=90,
             ha='center', va='center', color=COLORS['local_order'], fontsize=10)
    handles = [
        Line2D([], [], color=COLORS['rank'], marker='o', markersize=3.5, linewidth=1.25,
               label='平均传播顺序（左轴）'),
        Line2D([], [], color=COLORS['local_order'], marker='s', markersize=3.5, linewidth=1.25,
               label='杆内相对顺序（右轴）'),
        Line2D([], [], color='#999999', linestyle=':', linewidth=.8, label='参考参数'),
    ]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .985), ncol=3,
               columnspacing=1., handlelength=1.8, handletextpad=.45)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            box = text.get_window_extent(renderer)
            assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.x1 + 1 and box.y1 <= fig.bbox.y1 + 1, text.get_text()
    png = figures / 'parameter_order_combined.png'
    fig.savefig(png, dpi=400)
    fig.savefig(figures / 'parameter_order_combined.svg')
    plt.close(fig)
    with (dest / 'source_data.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(selected)
    metadata = dict(
        archetype='quantitative grid', claim='Parameter effects differ across two pooled order summaries.',
        backend='Python matplotlib', topology_seed=2511, dynamics_seed=847401, duration_ms=60000,
        selected_source_rows=156, available_source_rows=len(rows),
        selection='All pooled rank and local-order rows; participation retained as an auxiliary figure.',
        uncertainty='One trajectory per point; no cross-network or cross-noise confidence interval.',
        axis_scales='Both axes use exactly 0 to 0.42 in every panel; no per-panel scale fitting.',
        colors='Blue: mean normalized rank error. Orange: within-shaft pairwise order-probability error.',
        aggregation='Original metrics unchanged: within-shaft averaging, then equal weight for SCL and ICL.',
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        physical_size_mm=[183,115], png_dpi=400, editable_svg=True, panels=panels,
        smoothing=False, pdf='Not requested; omitted.', agent_visual_review='PENDING', human_visual_review='PENDING')
    (dest / 'figure_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    (figures / 'README.md').write_text(
        '### parameter_order_combined.png\n'
        '六格依次对应左核x、左核y、核内→核外EE权重、右核x、右核y、全局EE轴角度；每格13个实测参数点。蓝色左轴为平均归一化rank误差，橙色右轴为同杆触点对先后概率误差，均先杆内平均再对两杆等权，越低越接近患者。所有左右轴固定为0–0.42；竖虚线为共同参考参数，位置以相对该参考点的位移显示。**关注点**：两种顺序摘要是否随同一参数一起改善，或出现取舍；双轴等刻度，不靠独立缩放制造趋势一致。\n\n'
        '每点是固定拓扑2511、噪声847401的60秒轨迹，剔除前1.5秒；无额外seed，点间连线仅引导阅读，无平滑。全体事件合并，不使用TA/TB分组；不改变训练损失或科学恢复判断。参与概率原图保留为辅助结果，其敏感性较弱不等于拟合已好。PNG为对话验收版，SVG保留可编辑文字，不生成PDF或图册。\n', encoding='utf-8')
    return str(png)


if __name__ == '__main__':
    print(render())
