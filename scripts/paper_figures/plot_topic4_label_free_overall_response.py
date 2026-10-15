"""Overall-only parameter response review: frozen statistics, three triptychs.

Each point is one fixed-network, fixed-noise trajectory, not an independent
network replicate. Connected marks guide the eye; no smoothing or interpolation.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

DEFAULT_ROOT = Path('/data/hfosp/topic4_sef_hfo/label_free_dense_parameter_search_20260915')
VERSION = 'overall_acceptance_20260916'
COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#AA7600', '#7654A6']
MARKERS = ['o', 's', '^', 'D', 'o', 'o']
LABELS = ['左核 x', '左核 y', '右核 x', '右核 y']
OBSERVABLES = [
    ('rank', '平均归一化 rank 误差', '平均传播顺序'),
    ('local_order', '杆内先后概率误差', '杆内相对顺序'),
    ('participation', '触点参与概率误差', '两杆触点参与'),
]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def render(table, baseline, root=DEFAULT_ROOT):
    root = Path(root)
    dest = root / 'key_results' / VERSION
    figures = dest / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    # Reuse the previously delivered first-row statistics exactly. Pooling is
    # performed upstream over events, never by averaging the TA/TB summaries.
    selected = [r for r in table if r['mode'] == 'ALL']
    assert len(selected) == 3 * 6 * 13
    assert all(np.isfinite(r['value']) for r in selected)
    assert all(0 <= r['value'] <= 1 for r in selected)
    selected_ids = sorted({r['candidate'] for r in selected})
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Noto Sans CJK JP', 'DejaVu Sans'],
        'font.size': 9, 'axes.labelsize': 9.5, 'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5, 'legend.fontsize': 8.5,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': 0.65, 'xtick.major.width': 0.65,
        'ytick.major.width': 0.65, 'legend.frameon': False,
        'svg.fonttype': 'none', 'pdf.fonttype': 42,
        'savefig.facecolor': 'white', 'axes.unicode_minus': True,
    })
    paths = []
    audit_panels = []
    for key, ylabel, title in OBSERVABLES:
        rows = [r for r in selected if r['observable'] == key]
        maximum = max(r['value'] for r in rows)
        ymax = np.ceil(maximum * 1.08 / .05) * .05
        fig, axes = plt.subplots(1, 3, figsize=(183 / 25.4, 68 / 25.4), sharey=True)
        fig.subplots_adjust(left=.092, right=.986, bottom=.25, top=.76, wspace=.22)
        common = None
        for col, js in enumerate(([0, 1, 2, 3], [4], [5])):
            ax = axes[col]
            for j in js:
                line = sorted((r for r in rows if r['axis'] == j), key=lambda r: r['parameter_value'])
                assert len(line) == 13
                x = np.array([r['parameter_value'] for r in line])
                y = np.array([r['value'] for r in line])
                assert np.all(np.diff(x) > 0)
                reference_x = 0. if j < 4 else float(baseline[j])
                reference = np.flatnonzero(np.isclose(x, reference_x, atol=1e-10))
                assert len(reference) == 1
                reference_y = float(y[reference[0]])
                if common is None:
                    common = reference_y
                assert np.isclose(reference_y, common, rtol=0, atol=1e-12)
                ax.plot(x, y, color=COLORS[j], marker=MARKERS[j], markersize=3.2,
                        markeredgewidth=.35, linewidth=1.15,
                        linestyle='--' if j in [1, 3] else '-',
                        label=LABELS[j] if j < 4 else None, zorder=3)
            ax.axhline(common, color='#B9B9B9', linewidth=.65, linestyle=(0, (3, 3)), zorder=1)
            ref_x = 0. if col == 0 else float(baseline[js[0]])
            ax.axvline(ref_x, color='#AAAAAA', linewidth=.65, linestyle=':', zorder=1)
            ax.plot(ref_x, common, marker='D', markersize=4, color='#222222', zorder=5)
            ax.set_ylim(0, ymax)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
            ax.tick_params(direction='out', length=3, pad=3)
            ax.margins(x=.035)
            if col == 0:
                ax.set_ylabel(ylabel + ' ↓', labelpad=7)
                ax.set_xlabel('核中心位移 (mm)', labelpad=5)
                ax.set_xticks([-2, -1, 0, 1])
            elif col == 1:
                ax.set_xlabel('核内→核外 EE\n权重倍率', labelpad=5)
                ax.set_xticks([1., 1.25, 1.5])
            else:
                ax.set_xlabel('全局 EE 轴偏移 (°)', labelpad=5)
                ax.set_xticks([-20, -10, 0, 10, 20])
            audit_panels.append(dict(observable=key, column=col, axes=js, points=13*len(js),
                                     y_limits=[0, float(ymax)], reference_y=common))
        axes[0].legend(loc='lower left', bbox_to_anchor=(-.04, 1.02), ncol=2,
                       handlelength=1.8, handletextpad=.45, columnspacing=.9, borderaxespad=0)
        fig.legend(handles=[Line2D([], [], color='#222222', marker='D', linestyle='none',
                                   markersize=4, label='共同参考工作点')],
                   loc='upper right', bbox_to_anchor=(.989, .976), borderaxespad=0)
        # Single-row panels carry no headline, candidate code, TA/TB split or
        # significance annotation. Definitions and statistical unit are below.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bounds = fig.bbox
        outside = []
        for item in fig.findobj(matplotlib.text.Text):
            if item.get_visible() and item.get_text():
                b = item.get_window_extent(renderer)
                if b.x0 < -1 or b.y0 < -1 or b.x1 > bounds.x1 + 1 or b.y1 > bounds.y1 + 1:
                    outside.append(item.get_text())
        assert not outside, outside
        path = figures / f'parameter_{key}.png'
        fig.savefig(path, dpi=400)
        fig.savefig(figures / f'parameter_{key}.svg')
        paths.append(str(path))
        plt.close(fig)
    columns = ['observable', 'axis', 'parameter_value', 'candidate', 'value', 'N', 'mode']
    with (dest / 'source_data.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(selected)
    write(dest / 'figure_metadata.json', dict(
        schema=VERSION, archetype='quantitative grid', backend='Python matplotlib',
        claim='Fixed-network parameter changes alter pooled patient-model summary errors.',
        scientific_acceptance='PARTIAL_PROPAGATION_FEATURE_RECOVERY',
        source_rows=len(table), displayed_rows=len(selected), unique_conditions=len(selected_ids),
        exclusions='Only TA/TB display rows removed at user request; all pooled observations retained.',
        topology_seed=2511, dynamics_seed=847401, duration_ms=60000, burnin_ms=1500,
        uncertainty='No cross-seed interval: exactly one fixed topology and noise per condition.',
        aggregation='Mean absolute discrepancy within each shaft, then equal weight across SCL and ICL.',
        reference='Common one-parameter scan anchor; not each axis optimum and not the final nominated candidate.',
        patient_target='Frozen FIT events, pooled without TA/TB labels; zero denotes matching this summary only.',
        metric_is_training_loss=False, thresholds_or_p_values=None,
        width_mm=183, height_mm=68, png_dpi=400, editable_svg_text=True,
        pdf='Not generated: user requested PNG figures without a recurring PDF report.',
        smoothing=False, panels=audit_panels, human_visual_review='PENDING',
        agent_visual_review='PENDING', script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    descriptions = {
        'rank': '患者与模型各触点平均归一化rank的绝对差，先在杆内平均，再对SCL与ICL等权平均。',
        'local_order': '比较同杆内所有触点对的先后概率：仅在两触点共同参与的事件中估计，先在杆内平均绝对误差，再对两杆等权。',
        'participation': '比较每个触点参与事件的概率，先计算每杆触点平均绝对误差，再对两杆等权。',
    }
    (figures / 'README.md').write_text('\n\n'.join(
        f'### parameter_{key}.png\n{descriptions[key]}只显示所有事件合并后的统计，三列依次为核位置、核内→核外EE权重、全局EE方向；每个标量保留13个实测点。黑菱形及灰参考线为同一出发工作点；颜色区分参数，不表示模式或seed。**关注点**：误差越低越接近患者该项摘要，但整体平均接近不等于两种条件传播分布已恢复。'
        for key, _, _ in OBSERVABLES) + '\n\n每点为固定拓扑2511、固定噪声847401的一次60秒轨迹，剔除前1.5秒；无跨seed置信带。三张图只删去旧图的TA/TB两行，不改变数值、观测定义、损失、提名或仿真。位置横轴为相对共同工作点的位移，颜色与标记对应左右核的x/y；方向正负沿用执行器角度定义。SVG保留可编辑文字；用户未要求PDF，本版不生成PDF或图册。\n', encoding='utf-8')
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    plan = read(args.root / 'plan.json')
    candidate = read(args.root / 'candidates' / f"{plan['reference_id']}.json")
    p = candidate['parameters']
    baseline = np.r_[np.asarray(candidate['centers_mm']).ravel(), p['EE_core_to_out_scale'], p['EE_angle_offset_deg']]
    table = read(args.root / 'analysis/parameter_metric_table.json')['rows']
    print(json.dumps(render(table, baseline, args.root), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
