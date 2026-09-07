#!/usr/bin/env python3
"""Milestone review figure for the V4 component-coverage cycle (diagnostic, not paper Fig5).

Three independent questions, one panel each:
  a. Where did V4 actually search (V4 core positions by anchor vs the reused V3 pool)?
  b. Did component coverage relieve the participation/timing trade-off (eight-network candidates)?
  c. How did the joint-score reference move across V3 and V4 rounds?
Reads only audited artifacts; writes results/.../v4/figures/ and README.md.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_topic4_joint_xy_component_search as run
from src.topic4_xy_replicated_anchors import incumbent

OUT = run.OUT; V3 = run.OLD_V3
ANCHOR_COLORS = {'replicated_r008_003': '#d95f02', 'xy_whole_sheet_007': '#1b9e77', 'replicated_r001_004': '#7570b3',
                 'replicated_r007_009': '#e7298a', 'component_r001_015': '#e6ab02'}
ANCHOR_LABELS = {'replicated_r008_003': 'joint/timing anchor', 'xy_whole_sheet_007': 'participation anchor',
                 'replicated_r001_004': 'rank anchor', 'replicated_r007_009': 'timing anchor (round 1)',
                 'component_r001_015': 'joint fill anchor (rounds 2-4)'}


def v3_trajectory(plan):
    pool = run.read(V3 / 'baseline_scores.json')['candidates']; traj = [incumbent(pool, plan)['joint_distance']]
    for n in range(1, run.read(V3 / 'cycle_closure_review.json')['rounds'] + 1):
        stage = V3 / 'rounds' / f'{n:03d}'
        pool.extend(run.read(stage / 'scores.json')['candidates'])
        for cid in run.read(stage / 'race_nomination.json')['candidate_ids']:
            row = run.read(stage / f'combined_{cid}.json')['candidates'][0]
            pool = [row if r['candidate_id'] == cid else r for r in pool]
        traj.append(incumbent(pool, plan)['joint_distance'])
    return traj


def main():
    plan = run.read(run.CONFIG)
    review_path = OUT / 'cycle_closure_review.json'
    if not review_path.exists():
        review_path = OUT / 'partial_cycle_replay_review.json'
    review = run.read(review_path)
    table = list(csv.DictReader((OUT / 'component_coverage_milestone_eight_network_table.csv').open()))
    cal = run.read(OUT / 'patient_calibration.json')
    obj = run.KernelObjective(run.v1, OUT, run.KERNEL); xy = obj.xy
    v3_pool = run.read(OUT / 'baseline_scores.json')['candidates']
    proposals = []
    for rv in review['round_reviews']:
        for c in run.read(OUT / 'rounds' / f"{rv['round']:03d}" / 'design.json')['candidates']:
            proposals.append((rv['round'], c['anchor_candidate_id'], np.asarray(c['node_field']['centers_mm'])))
    anchors = {a['candidate_id']: np.asarray(a['centers_mm']) for rv in review['round_reviews'] for a in rv['local_anchors']}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15, 5.8), gridspec_kw={'width_ratios': [1.05, 1.1, 1.0]})
    fig.subplots_adjust(left=.05, right=.99, bottom=.3, top=.92, wspace=.3)
    # (a) search coverage map
    v3c = np.concatenate([np.asarray(r['candidate']['node_field']['centers_mm']) for r in v3_pool])
    a.scatter(v3c[:, 0], v3c[:, 1], s=6, c='#c8c8c8', label=f'V3 reused cores ({len(v3_pool)} geometries)', zorder=1)
    for key, color in ANCHOR_COLORS.items():
        pts = np.concatenate([p[2] for p in proposals if p[1] == key]) if any(p[1] == key for p in proposals) else None
        if pts is not None:
            a.scatter(pts[:, 0], pts[:, 1], s=16, c=color, label=f'{ANCHOR_LABELS[key]} children', zorder=3)
    fresh = np.concatenate([p[2] for p in proposals if p[1] is None])
    a.scatter(fresh[:, 0], fresh[:, 1], s=16, facecolors='none', edgecolors='black', linewidths=.7, label='V4 fresh uniform restarts', zorder=3)
    for key, color in ANCHOR_COLORS.items():
        if key in anchors:
            a.scatter(anchors[key][:, 0], anchors[key][:, 1], s=90, marker='*', c=color, edgecolors='black', linewidths=.5, zorder=4)
    a.scatter(xy[:, 0], xy[:, 1], s=12, c='black', marker='s', label='virtual contacts', zorder=2)
    a.set(xlim=(0, 20), ylim=(0, 20), aspect='equal', xlabel='x (mm)', ylabel='y (mm)', title='a  V4 core positions by local anchor')
    a.legend(fontsize=7, loc='upper center', bbox_to_anchor=(.5, -.12), ncol=2, frameon=False)
    # (b) participation vs timing trade-off, eight-network candidates
    for cycle, color, marker, label in (('V3', '#9a9a9a', 'o', 'V3 cycle (54)'), ('V4_expanded_old_geometry', '#1f78b4', 's', 'V4: old geometry expanded'),
                                        ('V4_new_geometry', '#e6550d', 'o', 'V4: new geometry')):
        rows = [r for r in table if r['cycle'] == cycle]
        if not rows:
            continue
        b.scatter([float(r['kernel_support_ratio']) for r in rows], [float(r['kernel_timing_space_ratio']) for r in rows],
                  s=[34 if r['anchor_eligible'] == 'True' else 14 for r in rows], c=color, marker=marker, alpha=.85,
                  edgecolors='black', linewidths=.3, label=f'{label} (n={len(rows)})')
    for key, color in ANCHOR_COLORS.items():
        r = next((x for x in table if x['candidate_id'] == key), None)
        if r:
            b.scatter(float(r['kernel_support_ratio']), float(r['kernel_timing_space_ratio']), s=130, marker='*', c=color, edgecolors='black', linewidths=.5, zorder=5)
    b.axhline(1, color='black', ls='--', lw=.8); b.axvline(1, color='black', ls='--', lw=.8)
    b.set(xscale='log', yscale='log', xlabel='participation kernel distance / patient q95', ylabel='timing kernel distance / patient q95',
          title='b  Component trade-off after eight networks')
    b.set_xlim(left=.8); b.set_ylim(bottom=.8)
    b.text(10, 1.06, 'patient tolerance (dashed)', fontsize=7, va='bottom', ha='center')
    b.legend(fontsize=7, loc='lower center', bbox_to_anchor=(.5, .1), frameon=False)
    # (c) reference trajectory
    v3 = v3_trajectory(plan)
    v4 = [review['round_reviews'][0]['incumbent_before_round']['joint_distance']] + [rv['incumbent_after_round']['joint_distance'] for rv in review['round_reviews'] if rv.get('incumbent_after_round')]
    c.plot(range(len(v3)), v3, 'o-', c='#9a9a9a', label='V3 reference (joint-score anchors)')
    c.plot(np.arange(len(v4)) + len(v3) - 1, v4, 's-', c='#e6550d', label='V4 reference (component anchors)')
    best_round = [min(x['joint_distance'] for x in rv['expanded']) for rv in review['round_reviews'] if rv.get('expanded')]
    c.plot(np.arange(1, len(best_round) + 1) + len(v3) - 1, best_round, 'v', c='#e6550d', mfc='none', label='V4 best candidate of the round')
    c.axhline(cal['samples']['128']['q95']['joint_distance'], color='black', ls='--', lw=.8)
    c.text(0, cal['samples']['128']['q95']['joint_distance'] * 1.1, 'patient tolerance (N in 65-128)', fontsize=7)
    c.set(yscale='log', xlabel='completed common-seed rounds', ylabel='joint kernel distance',
          title='c  Joint-score reference across cycles')
    c.set_xticks(range(len(v3) + len(v4) - 1)); c.set_xticklabels([f'V3 {i}' for i in range(len(v3))] + [f'V4 {i}' for i in range(1, len(v4))], rotation=60, fontsize=7)
    c.legend(fontsize=7, frameon=False, loc='lower left', bbox_to_anchor=(0, .12))
    for ax in (b, c):
        ax.spines[['top', 'right']].set_visible(False)
    folder = OUT / 'figures'; folder.mkdir(exist_ok=True)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(folder / f'v4_component_coverage_milestone.{ext}', dpi=170)
    plt.close(fig)
    (folder / 'README.md').write_text('''### v4_component_coverage_milestone.png / .pdf / .svg
V4 四轮分量覆盖实验的审阅图，三个面板各答一个独立问题。a：V4 的 96 个新位置（每个几何两个核）实际落在哪里，按产生它的局部参考着色（星号为参考位置，空心圆为全局随机起点，灰点为复用的 V3 347 个几何的核），黑方块是 15 个虚拟触点。b：全部八网络候选的参与核距离与时差核距离各自除以患者容忍度（按各自事件数档），虚线为 1；灰点为 V3 周期完成的候选，橙色为 V4 新几何，蓝色为 V3 旧几何在 V4 才补算者，星号为局部参考。c：每个完整补算轮次结束后的联合分数参考位置，V3 八轮与 V4 四轮接续画，倒三角为 V4 各轮最好候选，虚线为 N 在 65–128 时的患者容忍度。
**关注点**：a 里参与参考（绿色）周围是否真的被细化；b 里是否出现同时接近左下角的候选——目前所有点都在时差 ≥ 6.9 倍处；c 里 V4 四轮参考是否移动。所有候选均未通过患者分布门槛，本图不是模型验收。
''')
    meta = {'inputs': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [review_path, OUT / 'component_coverage_milestone_eight_network_table.csv', OUT / 'patient_calibration.json', Path(__file__)]},
            'outputs': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('v4_component_coverage_milestone.*')},
            'v3_reference_trajectory': v3, 'v4_reference_trajectory': v4, 'v4_best_of_round': best_round, 'author_acceptance': False, 'scientific_qualification': False}
    (folder / 'v4_component_coverage_milestone_metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps({'v3': [round(v, 6) for v in v3], 'v4': [round(v, 6) for v in v4], 'best_of_round': [round(v, 6) for v in best_round]}))


if __name__ == '__main__':
    main()
