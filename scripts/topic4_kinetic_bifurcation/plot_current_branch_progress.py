"""Show existing numerical points and independent spatial observations.

This status figure assigns neither branch connectivity nor stability.  It is
deliberately separate from the requested, still incomplete bifurcation figure.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/topic4_sef_hfo/kinetic_population_bifurcation_20260916'


def main():
    cycle_paths = [OUT / f'curve_point_readouts/{name}/readout.json'
                   for name in ('D225_order5', 'D2375_order5')]
    equilibrium_paths = [OUT / name / 'status.json' for name in (
        'corrected_rate_sections/rE430.00000000_degree6_dv0.125_high_from_D4_local',
        'corrected_equilibria/D0.40000000_degree6_dv0.125_high_continuation_from_D5_refined',
        'corrected_equilibria/D0.50000000_degree6_dv0.125_high_anchor_refined')]
    cycles = [json.loads(p.read_text()) for p in cycle_paths]
    equilibria = [json.loads(p.read_text()) for p in equilibrium_paths]
    assert all(x['status'] == 'FULL_MAP_EQUILIBRIUM_CORRECTED' for x in equilibria)
    assert all(x['weighted_return_residual'] < 1e-8 for x in cycles)
    spatial = json.loads((OUT / 'transition_spatial_anchors_D245.json').read_text())
    with np.load(OUT / 'transition_spatial_anchors_D245.npz') as z:
        frames = z['frames']
    with np.load(OUT / 'operators/selected_g40_theta0.25/geometry.npz') as z:
        centers = z['centers_mm']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig = plt.figure(figsize=(10.6, 8.0), layout='constrained')
    grid = fig.add_gridspec(2, 4, height_ratios=[1.7, 1])
    ax = fig.add_subplot(grid[0, :])
    ax.plot([x['D'] for x in equilibria], [x['mean_E_hz'] for x in equilibria],
            linestyle='none', marker='o', color='#252525', ms=7,
            label='Equilibrium point')
    ds = [x['D'] for x in cycles]
    ax.plot(ds, [x['mean_rate_hz'][0] for x in cycles], linestyle='none',
            marker='D', color='#cc8a23', ms=7, label='Periodic-orbit candidate: mean')
    for key, label in [('maximum_native_rate_hz', 'Periodic-orbit candidate: extrema'),
                       ('minimum_native_rate_hz', None)]:
        ax.plot(ds, [x[key][0] for x in cycles], linestyle='none', marker='x',
                color='#218571', ms=9, mew=1.8, label=label)
    ax.set(xlim=(.20, .52), ylim=(0, 530),
           xlabel=r'$D=1-\langle Z_E\rangle$',
           ylabel='Global E rate (Hz / neuron)')
    ax.set_yscale('symlog', linthresh=.1)
    ticks = [0, .1, 1, 10, 100, 500]
    ax.set_yticks(ticks, labels=['0', '0.1', '1', '10', '100', '500'])
    ax.set_xticks([.20, .225, .25, .30, .35, .40, .45, .50])
    ax.legend(loc='center right', frameon=False, fontsize=11)
    ax.text(-.065, 1.02, 'A', transform=ax.transAxes, weight='bold', fontsize=17)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    axes = []
    for i, (frame, row) in enumerate(zip(frames, spatial['rows'])):
        a = fig.add_subplot(grid[1, i]); axes.append(a)
        im = a.imshow(frame, origin='lower', extent=(0, 20, 0, 20),
                      cmap='inferno', vmin=0, vmax=500, interpolation='nearest')
        for j, (x, y) in enumerate(centers):
            a.add_patch(Circle((x, y), 1.45, fill=False, color='#20c7d2', lw=1.1))
            a.text(x, y+1.8, 'AB'[j], ha='center', color='#20c7d2', fontsize=10)
        a.text(0, 1.08, f'{"BCDE"[i]}   $D={row["D"]:g}$', transform=a.transAxes)
        a.set(xlabel='x (mm)', xticks=[0, 10, 20], yticks=[0, 10, 20])
        if i == 0:
            a.set_ylabel('y (mm)')
        else:
            a.set_yticklabels([])
    fig.colorbar(im, ax=axes, ticks=[0, 250, 500], shrink=.75, pad=.02).set_label('E rate (Hz)')
    folder = OUT / 'figures'
    stem = 'fig_current_branch_progress'
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(folder / f'{stem}.{ext}', dpi=200)
    plt.close(fig)
    metadata = dict(
        status='PROGRESS_FIGURE_ONLY',
        numerical_point_sources=[str(p) for p in equilibrium_paths+cycle_paths],
        spatial_source=str(OUT / 'transition_spatial_anchors_D245.json'),
        current_display_D_range=[.20, .52],
        physical_parameter_domain=[0, 1],
        stability='NOT_ASSIGNED', bifurcation_type='NOT_CLASSIFIED',
        branch_connectivity='NOT_ESTABLISHED; no lines connect the points',
        model='Autonomous density approximation: frozen spatial Z, dynamic M, zero shared OU fields',
        scope='Panel A contains corrected numerical points at degree 6, dv=0.125 mV. '
              'Resolution qualification is pending. B-E are independent finite-trajectory '
              '50-ms observations, not spatial eigenmodes or certified cycle annotations.',
        human_visual_acceptance='PENDING')
    (OUT / 'current_branch_progress_figure.json').write_text(json.dumps(metadata, indent=2)+'\n')
    readme = folder / 'README.md'
    content = readme.read_text()
    if f'### {stem}.' not in content:
        content += (f'\n### {stem}.png / .pdf / .svg\n\n'
            '用于回答当前进度：上方仅画已校正的三个高活动平衡点、两个周期候选的均值与极值，'
            '未赋予稳定性且没有连线；没有画出的参数区间尚未由这些点覆盖，不能解释成分支不存在。'
            '下方复用 D=0.2375、0.24 的两个相邻峰及 0.245 首次持续活动段的真实空间帧，'
            '它们是独立有限时间观测，并非已验证的临界空间模态。'
            '此图是进度汇总，不能当作目标分岔主图；PNG/PDF/SVG 同次生成，待人工检查。'
            '**关注点**：周期候选与高活动平衡点之间的缺口尚未接通，临界类型与稳定性尚未判定。\n')
        readme.write_text(content)
    print(folder / f'{stem}.png')


if __name__ == '__main__':
    main()
