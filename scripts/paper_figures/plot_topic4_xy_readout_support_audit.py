#!/usr/bin/env python3
"""Diagnostic figures from the completed offline audit; no selection or simulation."""
from pathlib import Path
import sys
import json
import hashlib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = ROOT / 'results/topic4_sef_hfo/vth_dual_core_xy_research'
AUDIT = OUT / 'readout_support_audit'
COLORS = ['#784578', '#358777', '#3976a9', '#c9823a', '#999999']


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    summary = read(AUDIT / 'summary.json')
    rows = read(AUDIT / 'candidate_views.json')['rows']
    modes = ['frozen_causal_family', 'full_same_window_event_bar',
             'full_detector_all_event_bar', 'full_detector_compound_event_bar',
             'shifted_same_window_event_bar']
    labels = ['Causal\nfamily', 'Full activity\nsame windows', 'All population\nfragments',
              'Compound\nfragments only', 'SCL shifted\nsame windows']
    byview = {m: {r['candidate_id']: r for r in rows if r['view'] == m} for m in modes}
    ids = sorted(byview[modes[0]])
    if len(ids) != 132 or any(set(byview[m]) != set(ids) for m in modes):
        raise RuntimeError('expected the complete paired round-1 audit')
    base = byview[modes[0]]
    projected = {c: read(AUDIT / 'per_subject' / f'{c}_sample_projection.json')['rows'] for c in ids}
    units = [read(AUDIT / 'per_subject' / f'{c}_{seed}.json') for c in ids for seed in (2511, 2512)]
    projections = []
    for k in range(4):
        values = np.array([projected[c][k]['conditional_gate_probability'] for c in ids])
        projections.append({'event_count_range': [min(projected[c][k]['n'] for c in ids),
                                                 max(projected[c][k]['n'] for c in ids)],
                            'probability_q0_q50_q100': np.quantile(values, [0, .5, 1]).tolist(),
                            'n_probability_at_least_0_8': int(np.sum(values >= .8)),
                            'n_probability_zero': int(np.sum(values == 0))})
    diagnostic = {
        'status': 'OFFLINE_DIAGNOSTIC_ONLY_NO_ACCEPTANCE_CHANGE',
        'paired_full_minus_family_median': float(np.median([
            byview[modes[1]][c]['both_shafts_fraction'] - base[c]['both_shafts_fraction'] for c in ids])),
        'n_shifted_greater_than_real_same_windows': sum(
            byview[modes[4]][c]['both_shafts_fraction'] > byview[modes[1]][c]['both_shafts_fraction'] for c in ids),
        'n_geometries_with_fewer_than_22_ever_observed_cross_pairs': sum(base[c]['positive_cross_pairs'] < 22 for c in ids),
        'compound_fraction_per_run_q0_q50_q100': np.quantile([u['compound_fraction'] for u in units], [0, .5, 1]).tolist(),
        'sample_projections': projections,
        'inference_unit_note': '132 geometries are design points for one patient, not 132 independent patients.',
        'bootstrap_note': '128 finite-event bootstrap draws per geometry, equally stratified across two seeds. No new events or long-run confirmation.',
        'shift_note': 'Three circular SCL offsets of 1, 2, 3 s, conditional on the same original family windows; not independent replicates or a patient null.',
    }
    (AUDIT / 'diagnostic_summary.json').write_text(json.dumps(diagnostic, indent=2) + '\n')
    directory = AUDIT / 'figures'; directory.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.titlesize': 12, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 10.2))
    fig.subplots_adjust(left=.075, right=.975, top=.87, bottom=.21, wspace=.28, hspace=.52)
    fig.suptitle('Why round 1 stopped: observation support and the search gate', fontsize=18, weight='bold', y=.965)
    fig.text(.075, .915, 'Same 132 geometries and 264 saved runs; fixed VTH depth, no new SNN simulations.', fontsize=11)
    ax = axes[0, 0]
    values = [[100 * byview[m][c]['both_shafts_fraction'] for c in ids] for m in modes]
    rng = np.random.default_rng(20260905)
    boxes = ax.boxplot(values, positions=np.arange(5), widths=.5, patch_artist=True, showfliers=False,
                       medianprops={'color': 'black'}, boxprops={'linewidth': 1})
    for i, (v, patch) in enumerate(zip(values, boxes['boxes'])):
        patch.set(facecolor=COLORS[i], alpha=.28)
        ax.scatter(i + rng.uniform(-.15, .15, len(v)), v, s=8, color=COLORS[i], alpha=.28, rasterized=True)
        ax.text(i, 88, f'{np.median(v):.1f}%', ha='center', fontsize=10, color=COLORS[i], weight='bold')
    patient = 100 * summary['patient']['both_shafts_fraction']
    ax.axhline(patient, color='#202020', ls='--', lw=1)
    ax.text(.02, .985, f'Patient training: {patient:.1f}%', transform=ax.transAxes, va='top', fontsize=9)
    ax.set(title='A  Joint recruitment depends on the readout', ylabel='Events recruiting both shafts (%)',
           ylim=(-3, 109), xticks=np.arange(5), xticklabels=labels)
    ax.tick_params(axis='x', labelsize=8)
    ax = axes[0, 1]
    x = np.array(values[0]); y = np.array(values[1])
    ax.plot([0, 100], [0, 100], color='.65', ls=':', lw=1)
    ax.scatter(x, y, s=23, color=COLORS[1], alpha=.65, edgecolor='white', linewidth=.3)
    ax.text(.04, .94, 'One point per geometry\nEvent windows and count held fixed', transform=ax.transAxes, va='top')
    ax.set(title='B  Removing the family mask changes support', xlim=(-2, 102), ylim=(-2, 102),
           xlabel='Causal family: both shafts (%)', ylabel='Full activity, same windows: both shafts (%)')
    ax = axes[1, 0]
    counts = [summary['views'][m]['conditional_gate_pass'] for m in modes[:4]]
    bars = ax.bar(np.arange(4), counts, color=COLORS[:4], alpha=.8, width=.6)
    for b, n in zip(bars, counts): ax.text(b.get_x()+b.get_width()/2, n+3, f'{n}/132', ha='center', fontsize=10)
    ax.set(title='C  The conditional-support gate is readout sensitive', ylabel='Geometries passing support gate',
           xticks=np.arange(4), xticklabels=labels[:4], ylim=(0, 155))
    ax.tick_params(axis='x', labelsize=8)
    ax.text(.02, .91, 'Support gate only; not accepted patient fit', transform=ax.transAxes, fontsize=9)
    ax = axes[1, 1]
    counts = [p['n_probability_at_least_0_8'] for p in projections]
    bars = ax.bar(np.arange(4), counts, color=COLORS[0], width=.55, alpha=.75)
    for b, n in zip(bars, counts): ax.text(b.get_x()+b.get_width()/2, n+1.5, f'{n}/132', ha='center')
    ax.set(title='D  Repeating existing event patterns rarely suffices', ylabel='Geometries with bootstrap P(pass) ≥ 0.8',
           xticks=np.arange(4), xticklabels=['Observed\n23–60', '80', '160', '320'],
           xlabel='Resampled causal-family events', ylim=(0, 32))
    ax.text(.04, .94, 'Empirical projection, not longer simulations\n111/132 have <22 cross pairs ever observed',
            transform=ax.transAxes, va='top', fontsize=9)
    fig.text(.075, .065, 'All panels are diagnostics for one patient. Geometries are not independent biological replicates.\n'
             'Full activity includes concurrent roots; joint recruitment does not establish a shared causal lineage.\n'
             'The original acceptance result remains 0/132. Fig. 5 follow-up remains stopped.', fontsize=10, linespacing=1.5)
    for ext in ('png', 'pdf', 'svg'): fig.savefig(directory / f'xy_readout_failure_audit.{ext}', dpi=180)
    plt.close(fig)

    contract = read(OUT / 'direction_objective_v2.json')
    xy = np.array(contract['contact_xy_mm']); names = contract['contact_names']
    shaft = np.array([n.startswith('ICL') for n in names])
    angle = contract['structural_ee_axis_deg']; vector = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
    show = [('control_historical_matched', 'Historical axis-aligned control'),
            ('control_old_edge', 'Previously used bottom cores'),
            ('xy_whole_sheet_045', 'Lowest finite score: whole 045'),
            ('xy_whole_sheet_019', 'Near-axis diagnostic: whole 019')]
    fig, axes = plt.subplots(1, 4, figsize=(16.2, 6.0))
    fig.subplots_adjust(left=.05, right=.985, top=.78, bottom=.32, wspace=.25)
    fig.suptitle('Core positions are free; none of these geometries is qualified', fontsize=18, weight='bold', y=.96)
    fig.text(.05, .875, f'Global E→E kernel: {angle:.1f}°, aspect ratio 2.  Patient training axis: {contract["patient_direction_summary"]["axial_angle_deg"]:.1f}°.', fontsize=11)
    for ax, (cid, title) in zip(axes, show):
        r = base[cid]
        with np.load(OUT / 'global/workers' / f'{cid}_seed_2511.npz') as z:
            pos = z['positions_E']; active = z['h'] > 0
        for x0 in (3, 7, 11, 15, 19):
            for y0 in (3, 7, 11, 15, 19):
                a = np.array([x0, y0]); line = np.array([a - .6*vector, a + .6*vector])
                ax.plot(line[:, 0], line[:, 1], color='.8', lw=1, zorder=0)
        ax.scatter(pos[active, 0], pos[active, 1], s=1.2, color='#508458', alpha=.25, rasterized=True)
        centers = np.array(r['centers_mm']); ax.plot(centers[:, 0], centers[:, 1], 'o--', color='#315e37', ms=4, lw=1)
        for mask, color in [(shaft, '#346999'), (~shaft, '#b2633d')]:
            ax.scatter(xy[mask, 0], xy[mask, 1], s=22, facecolor='white', edgecolor=color, lw=1.3, zorder=5)
        ax.text(15.5, 5.4, 'ICL', color='#346999', fontsize=8)
        ax.text(9.4, 16.4, 'SCL', color='#b2633d', fontsize=8)
        d = centers[1] - centers[0]; core_angle = np.degrees(np.arctan2(d[1], d[0]))
        ax.set(title=title, xlim=(0, 20), ylim=(0, 20), aspect='equal', xticks=[0, 10, 20], yticks=[0, 10, 20], xlabel='x (mm)')
        ax.text(.0, -.32, f'Core line {core_angle:.1f}°; score {r["J_round1"]:.2f}\n'
                f'Both shafts {100*r["both_shafts_fraction"]:.1f}%; {r["n_events"]} events', transform=ax.transAxes, fontsize=10)
    axes[0].set_ylabel('y (mm)')
    fig.text(.05, .045, 'Green points: actual VTH-modulated E neurons, seed 2511 (1,499 per geometry). Gray ticks: global kernel orientation.\n'
             'Scores and event fractions pool seeds 2511/2512. Finite scores here are diagnostic; no local refinement or final freeze occurred.', fontsize=10)
    for ext in ('png', 'pdf', 'svg'): fig.savefig(directory / f'xy_geometry_audit.{ext}', dpi=180)
    plt.close(fig)
    (directory / 'README.md').write_text('''### xy_readout_failure_audit.png / .pdf / .svg
用第一轮 132 个几何、264 条原有轨迹比较原始单因果家族读出、相同窗口的全部活动读出、群体片段及混合片段读出。A、B 显示读出对双电极杆共同招募的影响；C 仅统计条件指标支持门槛，不能当成患者拟合通过；D 是有限事件的重采样诊断，不能当成新长时程证据。全部几何仅属于一个患者，图中点不能作为患者样本做显著性检验。
**关注点**：读出差异能解释部分失败，但全部活动读出仍明显低于患者；原始验收仍是 0/132，Fig.5 未启动。

### xy_geometry_audit.png / .pdf / .svg
并排比较历史手放对照、旧底部双核、最低有限分数候选和接近训练轴的诊断候选。绿色散点是 seed 2511 中实际接受 VTH 调制的 E 神经元，短灰线仅标记全局 E→E 椭圆核方向，不表示每个位置真实产生的传播方向。分数和双杆共同招募比例汇总两个拟合网络，四个几何都没有通过正式验收。
**关注点**：新搜索允许两个 core 离开底部；最低有限分数的位置也不能当作已优化、已冻结的最终模型。
''')
    input_paths = [AUDIT / 'summary.json', AUDIT / 'candidate_views.json',
                   AUDIT / 'patient_training_mask_audit.json', OUT / 'direction_objective_v2.json']
    code_paths = [Path(__file__), ROOT / 'src/topic4_xy_readout_audit.py',
                  ROOT / 'scripts/audit_topic4_xy_round1_readout.py',
                  ROOT / 'scripts/audit_topic4_xy_patient_training_mask.py',
                  ROOT / 'tests/test_topic4_xy_readout_audit.py']
    metadata = {'producer': str(Path(__file__)), 'upstream_acceptance_modified': False,
                'source_hashes': {str(p): sha(p) for p in input_paths + code_paths},
                'outputs': {p.name: sha(p) for p in directory.iterdir() if p.suffix in ('.png', '.pdf', '.svg')},
                'tests': '9 passed; original readout parity includes both simulation boundaries',
                'visual_acceptance': 'DIAGNOSTIC_PENDING_AUTHOR_ACCEPTANCE'}
    (AUDIT / 'figure_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(diagnostic, indent=2))


if __name__ == '__main__':
    main()
