"""Curve-first EE x threshold-spread bifurcation diagram.

Consumes the accepted critical-point table, not unfinished continuation folders.
State labels are attached to sampled parameter points; no interpolated basin or
stability region is inferred from the mere existence of a critical curve.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/topic4_sef_hfo/core_heterogeneity_bifurcation_20260918'
FIG = OUT / 'figures'
NAME = 'heterogeneity_parameter_boundaries'
STYLES = {
    'SN': ('#292929', '-', 2.0),
    'LPC_onset': ('#c48924', '-', 2.0),
    'LP1': ('#ba444e', '-', 2.0),
    'PD1': ('#2868ac', '--', 1.6),
    'PD2': ('#2868ac', '--', 1.6),
    'PD3': ('#2868ac', '--', 1.6),
}
BOXES = {'b': (1.118, 1.179, .954, 1.001),
         'c': (1.315, 1.400, .958, 1.001)}
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.labelsize': 12, 'axes.titlesize': 12,
    'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})


def read(path):
    return json.loads(path.read_text())


def inputs():
    table = read(OUT / 'bifurcation_curves.json')
    curves = {key: sorted([r for r in table if r['label'] == key],
                          key=lambda r: r['h']) for key in STYLES}
    for key, rows in curves.items():
        assert len(rows) >= 2 and all(r['status'] == 'REFINED' for r in rows)
        assert len(set(r['h'] for r in rows)) == len(rows)
        for row in rows:
            assert (ROOT / row['source']).is_file()
            for field in ['residual', 'orbit_residual', 'critical_residual', 'null_residual']:
                if field in row:
                    assert abs(row[field]) < 1e-7, (key, field, row[field])
        if key in ('LPC_onset', 'LP1'):
            assert all(r['curve_status'] == 'PARTIAL_UNRESOLVED_ENDPOINT' for r in rows)
        else:
            assert rows[0]['h'] == 0 and rows[-1]['h'] == 1

    states = read(OUT / 'state_map_summary.json')
    samples = []
    for g, expected, label in [(1.10, 0, 'Rest'), (1.24, 2, 'Both cores burst'),
                               (1.36, 3, 'A burst / B high'),
                               (1.42, 5, 'High-background oscillations')]:
        found = [r for r in states if r['h'] == .5 and abs(r['g']-g) < 1e-7
                 and r['direction'] == 'up']
        assert len(found) == 1 and found[0]['state_index'] == expected
        row = found[0]
        if expected:
            assert row['kind'] == 'periodic_candidate' and row['recurrence_error_hz'] < .1
        else:
            assert row['kind'] == 'equilibrium_candidate'
        samples.append(dict(row, display_label=label))

    coexist = []
    for direction in ['up', 'down']:
        p = OUT / 'validated_attractors' / f'h0.00000_{direction}_g1.38000.json'
        d = read(p)
        assert d['stable_periodic_attractor']
        assert all(v['max_transverse'] < 1 for v in d['poincare'])
        coexist.append((p, d))
    return curves, samples, coexist


def draw_curves(ax, curves, keys=None):
    for key in keys or STYLES:
        rows = curves[key]
        color, ls, lw = STYLES[key]
        xy = np.array([(r['g'], r['h']) for r in rows])
        line, = ax.plot(xy[:, 0], xy[:, 1], color=color, ls=ls, lw=lw,
                        zorder=3, solid_capstyle='round')
        # Display joins only accepted adjacent points, without extrapolation.
        assert np.array_equal(line.get_xydata(), xy)
        if rows[0]['curve_status'].startswith('PARTIAL'):
            ax.plot(*xy[0], 'o', ms=5.5, mfc='white', mec=color, mew=1.3, zorder=7)


def label(ax, text, point, position, color='#363636', fontsize=10.5, ha='center'):
    return ax.annotate(text, xy=point, xytext=position, ha=ha, va='center',
                       color=color, fontsize=fontsize,
                       bbox=dict(fc='white', ec='none', alpha=.94, pad=1.5),
                       arrowprops=dict(arrowstyle='-', lw=.8, color=color,
                                       shrinkA=4, shrinkB=4), zorder=9)


def build(curves, samples):
    fig = plt.figure(figsize=(12.0, 7.8))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.7, 1.0],
                           left=.09, right=.975, bottom=.23, top=.93,
                           wspace=.34, hspace=.60)
    main = fig.add_subplot(grid[:, 0])
    onset = fig.add_subplot(grid[0, 1])
    cycles = fig.add_subplot(grid[1, 1])
    main.set_title('a   Parameter plane', loc='left', fontweight='bold', pad=12)
    main.set(xlim=(1.07, 1.455), ylim=(-.025, 1.035),
             xlabel=r'Core EE coupling $J_{\mathrm{EE,core}}$',
             ylabel=r'Core A threshold heterogeneity $h=\sigma_A/\sigma_{A,0}$')
    main.set_xticks([1.1, 1.2, 1.3, 1.4])
    main.set_yticks([0, .25, .5, .75, 1.])
    draw_curves(main, curves)

    # Labels identify finite-time attraction examples, not exhaustive regions.
    for row in samples:
        main.plot(row['g'], row['h'], 'o', color='#343434', ms=3.8, zorder=8)
    label(main, 'Rest', (1.10, .5), (1.109, .60), fontsize=12)
    label(main, 'Both cores\nburst', (1.24, .5), (1.245, .64), fontsize=12)
    label(main, 'A burst\nB high', (1.36, .5), (1.287, .36), fontsize=11)
    label(main, 'Both high\noscillating', (1.42, .5), (1.422, .65), fontsize=10)
    main.plot(1.38, 0, 'D', color='#773d83', ms=5.5, zorder=8)
    label(main, 'Two stable cycles\n(same parameters)', (1.38, 0),
          (1.279, .115), color='#773d83', fontsize=10.5)
    main.text(1.179, .23, 'SN', rotation=90, va='center', fontsize=11)
    for key, y in [('PD1', .72), ('PD3', .79), ('PD2', .26)]:
        rows = curves[key]
        x = np.interp(y, [r['h'] for r in rows], [r['g'] for r in rows])
        offset, align = (.004, 'left') if key == 'PD2' else (-.004, 'right')
        main.text(x+offset, y, key, color=STYLES[key][0], rotation=90,
                  ha=align, va='center', fontsize=10)

    for tag, ax, keys, title in [
            ('b', onset, ['SN', 'LPC_onset'], 'b   Onset boundaries'),
            ('c', cycles, ['LP1', 'PD1', 'PD3', 'PD2'], 'c   Cycle bifurcations')]:
        x0, x1, y0, y1 = BOXES[tag]
        main.add_patch(Rectangle((x0, y0), x1-x0, 1-y0, fill=False,
                                 ec='#aaaaaa', lw=.8, ls=':', zorder=5))
        main.text((x0+x1)/2, 1.012, tag, color='#666666',
                  ha='center', va='bottom', fontsize=10)
        ax.set(xlim=(x0, x1), ylim=(y0, y1), ylabel=r'$h$',
               xlabel=r'$J_{\mathrm{EE,core}}$')
        ax.set_title(title, loc='left', fontweight='bold', pad=9)
        ax.set_yticks([.96, .98, 1.]);ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        draw_curves(ax, curves, keys)

    onset.set_xticks([1.12, 1.14, 1.16, 1.18])
    onset.text(1.153, .987, 'SN', fontsize=11)
    onset.text(1.136, .958, 'Rest stable', fontsize=10, color='#525252')
    label(onset, 'Cycle\nfold', (1.12167, .986525), (1.132, .982),
          color=STYLES['LPC_onset'][0], fontsize=10)
    # In the low-h part of SN, B rather than A leads the first low-rate loss.
    label(onset, 'B leads', (1.17455576, .960), (1.165, .959), fontsize=8.5)

    cycles.set_xticks([1.32, 1.34, 1.36, 1.38, 1.40])
    cycles.text(1.335, .983, 'LP1', color=STYLES['LP1'][0], fontsize=10.5)
    for key, x, y in [('PD1', 1.3493, .967), ('PD3', 1.3778, .967),
                       ('PD2', 1.3884, .985)]:
        cycles.text(x+.0012, y, key, color=STYLES[key][0], rotation=90,
                    va='center', fontsize=9)

    legend = [
        Line2D([], [], color=STYLES['SN'][0], lw=2, label='SN: equilibrium fold'),
        Line2D([], [], color=STYLES['LPC_onset'][0], lw=2, label='Onset cycle fold'),
        Line2D([], [], color=STYLES['LP1'][0], lw=2, label='LP1: cycle fold'),
        Line2D([], [], color=STYLES['PD1'][0], lw=1.6, ls='--', label='PD: period doubling'),
        Line2D([], [], color='#777777', marker='o', mfc='white', ls='', ms=5,
               label='Unresolved continuation endpoint'),
        Line2D([], [], color='#343434', marker='o', ls='', ms=3.8,
               label='Sampled state'),
    ]
    fig.legend(handles=legend, loc='lower center', bbox_to_anchor=(.54, .08),
               frameon=False, ncol=3, fontsize=9.5, columnspacing=1.8)
    fig.text(.09, .040, 'Mean threshold fixed; only Core A threshold spread changes. '
             'Lines mark critical conditions of specific solution branches.', fontsize=9)
    fig.text(.09, .016, 'State labels refer to the marked samples; uncomputed boundaries '
             'and coexistence regions are not filled in.', fontsize=9)
    FIG.mkdir(exist_ok=True)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(FIG/f'{NAME}.{ext}', dpi=230, bbox_inches='tight')
    plt.close(fig)


def documentation(curves, samples, coexist):
    sources = [OUT/'bifurcation_curves.json', OUT/'state_map_summary.json']
    sources += [p for p, _ in coexist]
    metadata = dict(
        figure=NAME, producer=str(Path(__file__).relative_to(ROOT)),
        x='J_EE,core: simultaneous AA and BB recurrent E-to-E weight multiplier',
        y='h: Core A E threshold standard deviation divided by reference standard deviation',
        fixed='Core A threshold mean; all other cell thresholds, connections, inputs and delays',
        presentation='Accepted bifurcation curves in the parameter plane; no heatmap or state-space trajectories',
        interpolation='Straight segments between adjacent accepted continuation points; no extrapolation',
        line_styles='Solid: fold; dashed: period doubling. These do not encode branch stability.',
        open_endpoints='Continuation stopped here; not a classified codimension-two bifurcation',
        region_scope='State annotations are sampled attraction outcomes, not an exhaustive region partition',
        curves={k: dict(n=len(v), h_range=[v[0]['h'], v[-1]['h']],
                        status=v[0]['curve_status']) for k, v in curves.items()},
        samples=samples,
        coexistence=dict(h=0, J_EE_core=1.38, validated_stable_cycles=2),
        sources=[dict(path=str(p.relative_to(ROOT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                 for p in sources],
        human_visual_acceptance='PENDING',
    )
    (OUT/f'{NAME}.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False)+'\n')
    note = '''# 参数平面的分岔曲线图

用户明确要求两轴均为参数。本版恢复横轴核内 EE 倍率、纵轴 Core A 阈值标准差比例，固定平均阈值；主体为已求解的分岔曲线，不用状态空间的 E/I 轨迹，也不以扫描色块作为主要展示。

主图展示关键 EE 范围，两幅局部图放大接近原异质性设置时的 onset 和周期分岔。SN 是低率平衡态的首次 fold；周期 fold 与 PD 分别针对对应周期解分支。虚线表示 PD 类型，不表示该参数区域内所有解均不稳定。A/B 谁先失稳的切换使 SN 曲线出现转折，但没有将它标成已经分类的双零或 cusp 点。

图中状态文字用细引线连到实际扫描点：静息、双 core burst、A burst / B 高背景，以及双 core 高背景振荡。普通点来自同一升序 EE 扫描的有限时间吸引态候选；紫色菱形为两个周期吸引子都经过精确周期解与横向 Floquet 谱检验的共存实例。没有据此推断整个共存区边界。

只读取 `bifurcation_curves.json` 中已接受的曲线点，不重新收集 `boundaries/` 下包含失败或拒绝候选的文件。相邻已求解点用直线连接，局部周期 fold 的空心末端保留，不向未求解区域外推；原有极窄 LP0/PD0 系列仍未完成双参数延拓。周期分支可以共存，所以不能把每条 PD 线都解释成唯一吸引态的切换线。

本图为六群体降阶延迟模型的分岔结果，不是原生空间 SNN 或患者组织的完整状态分类。数值输入与上一版相同，此次改变的是展示方式；未启动新的参数扫描。

图：`figures/heterogeneity_parameter_boundaries.png/pdf/svg`。复现：`python scripts/topic4_core_heterogeneity_v11/parameter_portrait.py`。源数据、已覆盖区间和图中实例记录在同名 JSON；待用户人工目视检查。
'''
    (OUT/f'{NAME}.md').write_text(note)
    readme = FIG/'README.md'
    heading = f'### {NAME}.pdf'
    text = readme.read_text() if readme.exists() else '# 双参数分析候选图\n'
    if heading not in text:
        text += f'''\n{heading}

按用户要求，两轴均为参数：核内 EE 强度 × Core A 阈值异质性；主体为 SN、周期 fold 和 PD 曲线，右侧放大局部边界。状态标注连接到实际采样点，空心端点表示延拓尚未完成；同名 PNG/SVG 为同一版本。**关注点**：分岔曲线在参数平面中的位置和已知范围，不将有限时间扫描色块当作精确边界。
'''
        readme.write_text(text)


if __name__ == '__main__':
    curves, samples, coexist = inputs()
    build(curves, samples)
    documentation(curves, samples, coexist)
    print(FIG/f'{NAME}.png')
