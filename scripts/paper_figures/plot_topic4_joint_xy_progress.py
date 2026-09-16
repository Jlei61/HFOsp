#!/usr/bin/env python3
"""Refresh reproducible diagnostics after each completed joint-XY search round."""
from pathlib import Path
import argparse
import json
import hashlib
import sys
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True); args = parser.parse_args()
    out = args.out; base = read(out/'baseline_scores.json')['candidates']
    reports = [read(p) for p in sorted((out/'rounds').glob('*/analysis.json'))]
    best0 = min(base, key=lambda r:r['exploration_score'])
    best = reports[-1]['best_metrics'] if reports else best0
    from scripts.run_topic4_xy_research import training_contract
    from src.topic4_xy_direction import onset_directions, direction_histogram
    training, _ = training_contract()
    old = ROOT/'results/topic4_sef_hfo/vth_dual_core_xy_research'
    direction = read(old/'direction_objective_v2.json'); xy = np.array(direction['contact_xy_mm'])
    plt.rcParams.update({'font.size': 10, 'font.family': 'DejaVu Sans', 'pdf.fonttype': 42,
                         'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 10.2))
    fig.subplots_adjust(left=.09, right=.97, bottom=.19, top=.87, hspace=.6, wspace=.3)
    fig.suptitle('Joint rank / space fitting: randomized dual-core search', y=.965, fontsize=17, weight='bold')
    fig.text(.09, .915, 'One patient; VTH depth and E→E kernel fixed. No axis-angle penalty or two-mode gate.', fontsize=11)
    ax = axes[0, 0]
    ax.plot([0]+[r['round'] for r in reports], [best0['joint_distance']]+[r['best_metrics']['joint_distance'] for r in reports],
            'o-', color='#37745a')
    ax.set(title='A  Best joint-distribution loss', xlabel='Completed randomized batch (0 = reused runs)', ylabel='Joint sliced Wasserstein distance')
    ax = axes[0, 1]
    worker = read(Path(best['units'][0]['worker_path']))
    with np.load(worker['arrays']['path']) as z: pos = z['positions_E']; selected = z['h'] > 0
    ax.scatter(pos[selected, 0], pos[selected, 1], s=2, color='#37745a', alpha=.3, rasterized=True)
    ax.scatter(xy[:, 0], xy[:, 1], facecolor='white', edgecolor='#365f94', s=24)
    centers = np.array(best['candidate']['node_field']['centers_mm']); ax.plot(*centers.T, 'o--', color='#37745a', lw=1)
    ax.set(title='B  Current training candidate (not accepted)', xlim=(0, 20), ylim=(0, 20), aspect='equal',
           xlabel='x (mm)', ylabel='y (mm)')
    ax.text(.02, .97, best['candidate_id'], transform=ax.transAxes, va='top', fontsize=9)
    ax = axes[1, 0]
    patient_mask = np.isfinite(training['onsets_ms'])
    model = best['support']['recruitment_per_contact']
    ax.plot(patient_mask.mean(axis=0), 'o-', color='#333333', label='Patient training', ms=4)
    ax.plot(model, 's-', color='#37745a', label='Model groups', ms=4)
    ax.set(title='C  Contact participation distribution', ylim=(0, 1.05), ylabel='Recruitment probability',
           xticks=range(len(xy)), xticklabels=training['contact_names'])
    ax.tick_params(axis='x', rotation=60, labelsize=8); ax.legend(frameon=False, fontsize=9)
    ax = axes[1, 1]
    hp = direction_histogram(onset_directions(training['onsets_ms'], xy))
    hm = np.asarray(best['direction']['histogram']); angle = (np.arange(24)*15+180)%360-180; order = np.argsort(angle)
    ax.plot(angle[order], hp[:-1][order], color='#333333', label='Patient training')
    ax.plot(angle[order], hm[:-1][order], color='#37745a', label='Model groups')
    ax.set(title='D  Direction derived from rank / timing and space', xlabel='Earlier-to-later direction (degrees)',
           ylabel='Coherent mass / all group events', xticks=[-180, -90, 0, 90, 180])
    ax.text(.02, .96, f'Unresolved mass: patient {hp[-1]:.2f}, model {hm[-1]:.2f}', transform=ax.transAxes, va='top', fontsize=9)
    fig.text(.09, .055, f'{best["n_events"]} observed model groups; candidate selection uses patient training only.\n'
             'Shared group-event packing is applied to a firing-density proxy, not a validated HFO/LFP forward model.\n'
             'A low training loss does not establish a unique geometry or a qualified Fig. 5 substrate.', fontsize=10, linespacing=1.5)
    directory = out/'figures'; directory.mkdir(exist_ok=True)
    for ext in ('png', 'pdf', 'svg'): fig.savefig(directory/f'joint_xy_search_progress.{ext}', dpi=180)
    plt.close(fig)
    (directory/'README.md').write_text('''### joint_xy_search_progress.png / .pdf / .svg
此图在每个随机搜索批次完成后更新，展示联合分布损失、当前训练候选的实际双核、接触点参与分布，以及由时间和空间推导出的方向分布。模型读出采用与患者相同的群体打包规则，但输入是放电密度代理，不等于真实 HFO 或 LFP；模型读出的多个因果家族可以进入同一观测事件。图中展示当前训练候选，并不意味着已通过独立网络确认或找到了唯一解。
**关注点**：参与、rank/时滞和空间关系是否共同改善；不能以 core 连线变得好看或某一个方向峰更高来验收。
''')
    (out/'figure_metadata.json').write_text(json.dumps({'producer_sha256': sha(Path(__file__)),
        'baseline_sha256': sha(out/'baseline_scores.json'),
        'round_analysis_hashes': {str(p): sha(p) for p in sorted((out/'rounds').glob('*/analysis.json'))},
        'outputs': {p.name: sha(p) for p in directory.iterdir() if p.suffix in ('.png', '.pdf', '.svg')},
        'author_acceptance': False}, indent=2)+'\n')


if __name__ == '__main__': main()
