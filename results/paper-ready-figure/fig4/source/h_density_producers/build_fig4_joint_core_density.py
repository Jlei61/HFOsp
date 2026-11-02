#!/usr/bin/env python3
"""Display core centers from the three frozen J/J_joint optimization batches.

Reuse the current Fig4 renderer. Write a review candidate without changing the
author-designated current version. The Gaussian kernels are a display of the
24 evaluated candidates, not a posterior or independent optimization repeats.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.paper_figures import build_fig4_readable_complete as plate
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

OUT = ROOT / 'results/paper-ready-figure/fig4/candidates/core_position_joint_density_20260928'
RUN = Path('/data/hfosp/topic4_sef_hfo/geometry_axis_joint_distribution_20260917')
PARENT = plate.OUT
SIGMA = .35
BATCHES = [(14, '#549AAB', 'o'), (15, '#AE8B53', 's'), (16, '#7461AD', '^')]
WINDOWS = [(0., 6., 9., 15.), (13., 19., 2., 8.)]
DATA = {}
ORIGINAL_PANEL_C = plate.panel_c


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def density(x, y, points, sigma=SIGMA):
    """Equal-weight isotropic Gaussian mixture, normalized over R^2, in mm^-2."""
    d2 = ((x[..., None] - points[:, 0]) ** 2 +
          (y[..., None] - points[:, 1]) ** 2)
    return np.exp(-d2 / (2 * sigma ** 2)).mean(axis=-1) / (2 * np.pi * sigma ** 2)


def load_data():
    table_path = plate.freeze(plate.OLD / 'source/all_workpoints_source.csv')
    history = plate.read(plate.freeze(plate.OLD / 'source/history.json'))
    rows = list(csv.DictReader(table_path.open()))
    rows = [r for r in rows if int(r['stage']) in [14, 15, 16]]
    plan = plate.read(plate.freeze(RUN / 'plan.json'))
    assert len(rows) == 24 and len({r['candidate'] for r in rows}) == 24
    assert plan['objective']['method'] == 'original three-group Doff and frozen same-event joint Doff; minimize both'
    for stage, _, _ in BATCHES:
        phase = next(p for p in history['phases'] if p['stage'] == stage)
        proposal = plate.read(plate.freeze(Path(phase['source'])))
        assert set(proposal['ids']) == {r['candidate'] for r in rows if int(r['stage']) == stage}
        assert proposal['kinds'] == ['qLogNEHVI'] * 8
    records = []
    for r in rows:
        source = plate.freeze(Path(r['source']))
        assert plate.sha(source) == r['source_sha256']
        run = plate.read(source)
        assert run['status'] == 'COMPLETE' and run['actual_duration_ms'] == 60000
        assert run['job']['topology_seed'] == 2511 and run['job']['dynamics_seed'] == 847401
        actual = plate.read(plate.freeze(source.parents[1] / 'applied_physics.json'))
        vector = np.array(json.loads(r['vector']))
        centers = np.array(actual['threshold']['centers_mm'])
        np.testing.assert_allclose(centers, vector[:4].reshape(2, 2), atol=1e-12, rtol=0)
        assert actual['threshold']['n_raised'] == 0
        score = plate.read(plate.freeze(RUN / 'scores' / f"{r['candidate']}.json"))
        np.testing.assert_allclose([score['J'], score['J_joint']],
                                   [float(r['J']), float(r['J_joint'])], atol=1e-12, rtol=0)
        assert score['N'] == int(r['N']) and score['N'] >= 16
        records.append(dict(candidate=r['candidate'], stage=int(r['stage']),
            batch=int(r['stage'])-13, centers_mm=centers.tolist(),
            J=float(r['J']), J_joint=float(r['J_joint']),
            outward_EE_gain=float(vector[4]), angle_deg=float(vector[5]),
            source=str(source), source_sha256=r['source_sha256']))
    lower = np.array(plan['bounds']['lower'][:4]).reshape(2, 2)
    upper = np.array(plan['bounds']['upper'][:4]).reshape(2, 2)
    for r in records:
        xy = np.array(r['centers_mm'])
        assert np.all(xy >= lower-1e-10) and np.all(xy <= upper+1e-10)
    winners = {str(stage): min((r for r in records if r['stage'] == stage),
                 key=lambda r: (r['J_joint'], r['candidate']))['candidate'] for stage, _, _ in BATCHES}
    means = np.array([np.array([r['centers_mm'] for r in records if r['stage'] == stage]).mean(0)
                      for stage, _, _ in BATCHES])
    stats = {}
    for core in range(2):
        xy = np.array([r['centers_mm'][core] for r in records])
        m = means[:, core]
        boundary = np.any(np.isclose(xy, lower[core], atol=1e-8, rtol=0) |
                          np.isclose(xy, upper[core], atol=1e-8, rtol=0), axis=1)
        stats['AB'[core]] = dict(batch_means_mm=m.tolist(),
            largest_batch_mean_shift_mm=float(np.linalg.norm(m[:, None] - m[None, :], axis=2).max()),
            candidate_coordinate_min_mm=xy.min(0).tolist(), candidate_coordinate_max_mm=xy.max(0).tolist(),
            candidates_on_any_search_boundary=int(boundary.sum()),
            distinct_coordinates=len(np.unique(xy, axis=0)))
    data = dict(schema='fig4_joint_search_core_center_density_v1', subject='E1146',
        records=records, batches=[14, 15, 16], n_candidates=24, candidates_per_batch=8,
        best_joint_per_batch=winners, search_lower_mm=lower.tolist(), search_upper_mm=upper.tolist(),
        summary=stats, coordinate_frame='Original 20 x 20 mm model sheet, unchanged x/y; no recentering.',
        selection='All 24 evaluated qLogNEHVI proposals from B stages 14-16; no score threshold or post-hoc subset.',
        statistical_unit='One evaluated parameter candidate; adaptive dependence, same topology/noise. Not independent fits.',
        optimizer='Simultaneous minimization of J and J_joint, two-output GP / qLogNEHVI.',
        smoothing=dict(method='Equal-weight Gaussian kernel mixture within each batch and core',
                       sigma_mm=SIGMA, normalization='Each batch/core integrates to 1 on the full plane.',
                       interpretation='Descriptive density of evaluated centers; not posterior, confidence region, or tissue excitability.',
                       boundary='Uncorrected display kernels can extend beyond the dotted allowed-center rectangle.'),
        scientific_status='DESCRIPTIVE_LOCAL_SEARCH_POSITION_DISTRIBUTION_ONLY', new_simulations=0)
    write(OUT / 'source/core_position_data.json', data)
    fields = ['candidate', 'stage', 'batch', 'A_x_mm', 'A_y_mm', 'B_x_mm', 'B_y_mm', 'J', 'J_joint']
    with (OUT / 'source/core_position_data.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in records:
            w.writerow({**{k:r[k] for k in fields if k in r},
                        **dict(zip(fields[3:7], np.array(r['centers_mm']).ravel()))})
    return data


def core_map(ax, core):
    ax.clear()
    xmin, xmax, ymin, ymax = WINDOWS[core]
    xs, ys = np.linspace(xmin, xmax, 241), np.linspace(ymin, ymax, 241)
    xx, yy = np.meshgrid(xs, ys)
    lo, hi = np.array(DATA['search_lower_mm'])[core], np.array(DATA['search_upper_mm'])[core]
    saved = dict(x_mm=xs, y_mm=ys)
    for stage, color, marker in BATCHES:
        rows = [r for r in DATA['records'] if r['stage'] == stage]
        points = np.array([r['centers_mm'][core] for r in rows])
        z = density(xx, yy, points)
        dxdy = (xs[1]-xs[0]) * (ys[1]-ys[0])
        assert abs(z.sum() * dxdy - 1.) < .001
        ordered = np.sort(z.ravel())[::-1]
        cumulative = np.cumsum(ordered) * dxdy
        levels = [float(ordered[np.searchsorted(cumulative, mass)]) for mass in [.8, .5]]
        ax.contourf(xx, yy, z, levels=[levels[0], z.max()+1e-8], colors=[color], alpha=.10)
        ax.contour(xx, yy, z, levels=levels, colors=[color], linewidths=[.7, 1.0],
                   linestyles=['--', '-'], zorder=3)
        # Exact coincident centers retain their coordinates; do not jitter them.
        ax.scatter(*points.T, s=15, marker=marker, facecolors='none', edgecolors=color, linewidths=.65, zorder=5)
        winner = next(r for r in rows if r['candidate'] == DATA['best_joint_per_batch'][str(stage)])
        ax.scatter(*winner['centers_mm'][core], s=60, marker='*', facecolors=color,
                   edgecolors='white', linewidths=.45, zorder=7)
        saved[f'density_stage{stage}_per_mm2'] = z
        saved[f'contour_stage{stage}_80_50_percent_levels'] = np.array(levels)
    ax.add_patch(Rectangle(lo, *(hi-lo), fill=False, edgecolor='#777777',
                           linestyle=':', linewidth=.85, zorder=6))
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), aspect='equal', xlabel='x (mm)', ylabel='y (mm)',
           xticks=[xmin, (xmin+xmax)/2, xmax], yticks=[ymin, (ymin+ymax)/2, ymax])
    plate.axis_type(ax)
    ax.tick_params(axis='y', labelleft=True)
    for axis in [ax.xaxis, ax.yaxis]:
        axis.label.set_fontsize(plate.MIDDLE_LABEL_PT); axis.labelpad=2
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_color('#B3B3B3'); spine.set_linewidth(.6)
    if core == 0:
        handles = [Line2D([], [], color=color, marker=marker, markerfacecolor='none',
                          lw=.8, ms=3, label=f'Batch {stage-13}') for stage, color, marker in BATCHES]
        handles.append(Line2D([], [], color='#555555', marker='*', lw=0, ms=6,
                              label=r'Best $J_{\mathrm{joint}}$'))
        ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(.015, .015),
                  fontsize=7.4, handlelength=1.3, labelspacing=.08,
                  handletextpad=.4, frameon=False, borderaxespad=.2)
    np.savez_compressed(OUT / f'source/core_{"AB"[core]}_density.npz', **saved)


def panel_c(fig, curves, geometry):
    na, nt = len(fig.axes), len(fig.texts)
    ORIGINAL_PANEL_C(fig, curves, geometry)
    axes = fig.axes[na:]
    for core, index in enumerate([4, 6]):
        core_map(axes[index], core)
        axes[index+1].set_visible(False)
        title = fig.texts[nt+core]
        title.set_text('Core ' + 'AB'[core]); title.set_color('#30383C')
    plate.CHECKS.pop('C_position_conditions', None)
    plate.CHECKS['C_palette'] = dict(curves=plate.ERROR_COLORS,
        density_batches={str(stage-13):color for stage, color, _ in BATCHES})
    plate.CHECKS.update(C_position_candidates=24, C_position_B_stages=[14, 15, 16],
        C_density_sigma_mm=SIGMA, C_local_maps_equal_scale_mm=[6, 6],
        C_centers_verified_against_applied_physics=True,
        C_search_bounds_visible=True, C_star_rule='Minimum J_joint within each of the three batches',
        C_density_is_not_confidence_or_posterior=True, C_position_summary=DATA['summary'])


def spatial_view(geometry):
    """Full sheet with two descriptive density hills and an exact 2-D readout."""
    fig = plt.figure(figsize=(10.4, 4.8), dpi=160, facecolor='white')
    left = fig.add_axes([.015, .15, .44, .73], projection='3d')
    right = fig.add_axes([.565, .18, .31, .67])
    xs = np.linspace(0., 20., 251); xx, yy = np.meshgrid(xs, xs)
    all_xy = np.array([r['centers_mm'] for r in DATA['records']])
    z = sum(density(xx, yy, all_xy[:, core]) for core in range(2)) / 2
    surface = left.plot_surface(xx, yy, z, cmap='plasma', vmin=0, vmax=float(z.max()),
                                rcount=126, ccount=126, linewidth=0, antialiased=True)
    left.set(xlim=(0, 20), ylim=(0, 20), zlim=(0, z.max()*1.15),
             xticks=[0, 10, 20], yticks=[0, 10, 20], zticks=[0, .15, .3],
             xlabel='x (mm)', ylabel='y (mm)', zlabel=r'Density (mm$^{-2}$)')
    left.view_init(elev=38, azim=-66); left.set_box_aspect((1, 1, .46))
    left.grid(False)
    for axis in [left.xaxis, left.yaxis, left.zaxis]:
        axis.pane.fill=False; axis.pane.set_edgecolor('#D5D5D5')
    names = np.array(geometry['contact_names']); contacts = np.array(geometry['contact_xy_mm'])
    for shaft in ['SCL', 'ICL']:
        ix = sorted(np.flatnonzero(np.char.startswith(names, shaft)),
                    key=lambda i:int(names[i][len(shaft):]), reverse=True)
        left.plot(*contacts[ix].T, zs=.002, color='#A4A4A4', lw=1, marker='o', ms=2.4,
                  markerfacecolor='white')
        right.plot(*contacts[ix].T, color='#9A9A9A', lw=.8, marker='o', ms=3,
                   markerfacecolor='white', zorder=1)
    for core in range(2):
        center=all_xy[:, core].mean(0)
        core_z=density(xx, yy, all_xy[:, core])/2
        iy,ix=np.unravel_index(np.argmax(core_z), core_z.shape)
        left.text(xx[iy,ix], yy[iy,ix], core_z[iy,ix]+.025,
                  'Core '+'AB'[core], ha='center', fontsize=10, zorder=20)
        right.text(center[0], center[1]+2.4, 'Core '+'AB'[core], ha='center', fontsize=10)
        lo=np.array(DATA['search_lower_mm'])[core]; hi=np.array(DATA['search_upper_mm'])[core]
        right.add_patch(Rectangle(lo, *(hi-lo), fill=False, edgecolor='#777777', ls=':', lw=.8))
        for stage, color, marker in BATCHES:
            xy=np.array([r['centers_mm'][core] for r in DATA['records'] if r['stage']==stage])
            zz=density(xx, yy, xy)
            right.contourf(xx, yy, zz, levels=[float(zz.max())*.15, float(zz.max())+1e-8],
                           colors=[color], alpha=.08)
            right.scatter(*xy.T, s=13, marker=marker, facecolors='none', edgecolors=color, lw=.7)
    right.set(xlim=(0, 20), ylim=(0, 20), aspect='equal', xticks=[0, 10, 20],
              yticks=[0, 10, 20], xlabel='x (mm)', ylabel='y (mm)')
    fig.legend(handles=[Line2D([], [], color=c, marker=m, markerfacecolor='none', ls='',
                  label=f'Batch {s-13}') for s,c,m in BATCHES],
                  loc='lower center', bbox_to_anchor=(.55, .015), ncol=3, frameon=False)
    bar=fig.colorbar(surface, cax=fig.add_axes([.92, .28, .017, .44]))
    bar.set_label(r'Density (mm$^{-2}$)'); bar.outline.set_visible(False)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(OUT / f'figures/core-position-spatial-density.{ext}', dpi=240)
    plt.close(fig)


def finish():
    record=plate.read(OUT/'figure4_candidate_registry.json')
    for key in ['author_designated_on', 'current_version_document']:
        record.pop(key, None)
    record.update(status='CANDIDATE_PENDING_AUTHOR_REVIEW',
        producer=str(Path(__file__).relative_to(ROOT)), parent_package=str(PARENT.relative_to(ROOT)),
        data_changes='C lower maps: evaluated core centers from stages 14-16 replace position-error grids; upper curves and all other panels retain their frozen data.',
        source_version_mix='Parent A-I plate, with C densities from exactly the 24 existing B stages 14-16 candidates.',
        scientific_claim='Local candidate centers across J/J_joint search batches; constrained, dependent search points, not independent optimized solutions.',
        current_version_pointer_changed=False, new_simulations=0,
        source_files=plate.SOURCES,
        outputs={str(p.relative_to(OUT)):plate.sha(p) for p in (OUT/'figures').iterdir()
                 if p.suffix in ['.png', '.pdf', '.svg']})
    write(OUT/'figure4_candidate_registry.json', record)
    texts=[
        '### fig4-complete-layout.png\n复用当前 A–I Fig4，仅将 C 下方位置—误差地图改为联合目标搜索中 core A/B 中心的分布；各批全部八个候选等权，其他 panel 数据和工作点保留。完整图与单独 C 的 PNG/PDF/SVG 使用同次渲染的对象。\n**关注点**：本次候选尚待作者目视检查，未替换当前版本；E/F/G/H 原选点规则仍是原版规则。',
        '### fig4-panelc.png\n上方连接参数响应保持原样；下方在相同 6×6 mm、相同纵横比例的绝对模型坐标中比较 core A 和 B，颜色对应 B 的第 14/15/16 阶段。空心点为实际候选中心，星号为各批 J_joint 最低点，虚线/实线轮廓分别围住高斯核密度的 80%/50% 质量；灰色点线框为搜索允许范围。\n**关注点**：σ=0.35 mm 的核平滑不是单高斯分布假设、置信区间或组织易激性场；核尾部可超出允许范围，但没有在那里观测到候选。',
        '### core-position-spatial-density.png\n左侧为同一 20×20 mm 平面中两核候选中心的高斯核密度山峰，两个 core 等权、每批八个候选等权；右侧叠加实际电极和三批中心位置。高度与色条表示位置密度，不是 core 强度 h，未借用旧版模型场。\n**关注点**：用于整体空间定位；具体批次比较以 fig4-panelc 的局部等比例地图为准。']
    for s in 'ABDEFGHI':
        texts.append(f'### fig4-panel{s.lower()}.png\n直接复用当前版 {s} 的绘图函数与冻结数据，并随整图同次输出 PNG/PDF/SVG。\n**关注点**：本次仅更改 C，不把原工作点重新解释为 J_joint 最优解。')
    (OUT/'figures/README.md').write_text('\n\n'.join(texts)+'\n')
    a,b=DATA['summary']['A'],DATA['summary']['B']
    (OUT/'scientific_note.md').write_text(f'''# Fig4C：联合目标搜索下的 core 位置分布

问题是同一优化范式下不同参数候选的 core 中心是否保留大体空间位置。数据来自 B 的第 14–16 阶段，共 3 批×8 个候选，同时最小化原无标签 J 与同事件 J_joint；全部 24 个条件完整保留，不按图形集中程度或 loss 阈值筛点。统计单位为一个已评估参数候选，均使用同一拓扑和噪声，批次依赖前批反馈，不是 24 次独立拟合。

显示采用等权高斯核密度，所有 core/批次统一 σ={SIGMA} mm。不拟合一个强制单峰高斯，也不将高斯宽度解释为模型 core 半径。星号选择各批 J_joint 最低点；最后一批为 `{DATA['best_joint_per_batch']['16']}`，与原 B 按 J 选择的代表点不同。

实测三批均值的最大距离：core A 为 {a['largest_batch_mean_shift_mm']:.3f} mm，core B 为 {b['largest_batch_mean_shift_mm']:.3f} mm。core A 较集中，core B 有明显批次偏移和边界聚集：A/B 分别有 {a['candidates_on_any_search_boundary']}/24 与 {b['candidates_on_any_search_boundary']}/24 个候选至少一坐标位于搜索边界。core B 实际只有 {b['distinct_coordinates']} 个不同 XY，均为允许范围的角点；重合点未添加随机抖动，密度反映重复候选的频数。因此只能描述该局部搜索下仍处于两片原有区域，不能据此证明位置对优化条件不敏感或已唯一识别；搜索范围本身限制了变化。

全部坐标与每条 trajectory 的 applied_physics 核中心逐一核对，两个 score 与原结果一致。沿用原模型 0–20 mm 坐标，不平移、不交换 XY，不把旧图的连续兴奋性场冒充本次密度。原 A 的居中坐标显示、其他 panel 及主版本入口均未改动。

复现：`python scripts/paper_figures/build_fig4_joint_core_density.py`。完整数组、候选 CSV、选择规则和统计均位于 source/；没有新增仿真或患者拟合，候选待人工目视验收。
''')
    (OUT/'panel_map.md').write_text('# Figure 4 联合搜索位置分布候选\n\n'
        '继承 complete_readability_20260923 的 A–I 身份，仅 C 下排改为三批联合搜索的 core 中心密度。'
        '原 B 的全部 226 点及其历史目标变更仍保留，不能把整个 B 说成统一 J_joint 拟合。'
        '原 E/F/G/H 不重新选点；本次也不替换 current_version.json。\n\n'
        '数据定义、选择与科学边界见 scientific_note.md；渲染说明见 figures/README.md。\n')


def main():
    global DATA
    (OUT/'source').mkdir(parents=True, exist_ok=True)
    DATA=load_data()
    plate.OUT=OUT; plate.FIG=OUT/'figures'; plate.SOURCE=OUT/'source'
    plate.base.OUT=plate.SOURCE
    plate.panel_c=panel_c
    plate.write_current_version=lambda record: None
    plate.main()
    spatial_view(plate.read(plate.EXPANDED/'source/position_map_source.json'))
    plate.freeze(Path(__file__))
    finish()


if __name__=='__main__':
    main()
