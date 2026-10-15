"""Read-only reconstruction of archived hand-placed Fig4 event coverage."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

MAIN = Path('/home/honglab/leijiaxin/HFOsp')
OUT = Path(__file__).resolve().parents[1] / 'results/topic4_sef_hfo/handplaced_two_core_history_audit'
RAW = MAIN / 'results/topic4_sef_hfo/field_swap_subject_snn'
ARCHIVE = MAIN / 'results/paper-ready-figure/archive/2026-08-13_non_main_figure_packages/fig4_subject_snn_e1146'
TAG = 'epilepsiae_1146_gradient_shared_corefrozen_cr1p5_s5_20260722'

def main():
    (OUT / 'figures').mkdir(parents=True, exist_ok=True)
    audit = json.loads((ARCHIVE / 'fig4_working_point_audit.json').read_text())
    run = json.loads((RAW / f'readout_{TAG}.json').read_text())
    z = np.load(RAW / f'figdata_{TAG}.npz', allow_pickle=True)
    rows = [json.loads((RAW / f'readout_{tag}.json').read_text())
            for tag in audit['candidates'][1]['input_tags']]
    rows.sort(key=lambda r: r['seed'])
    matrix = np.array([[np.nan if e['ranks'][str(n)] is None else e['ranks'][str(n)]
                        for e in run['events']] for n in z['names']], float)
    for i in range(matrix.shape[1]):
        x = matrix[:, i]; valid = np.isfinite(x)
        if valid.any():
            span = np.ptp(x[valid]); x[valid] = (x[valid] - x[valid].min()) / (span if span else 1)
    counts = np.array([[sum(e['sign'] == s for e in r['events']) for s in (1, -1, None)] for r in rows])
    assert counts.sum() == sum(r['n_events'] for r in rows)
    assert np.array_equal(counts[:, :2].sum(0), [29, 99])
    assert np.all(~np.isfinite(matrix[:4]))
    fig = plt.figure(figsize=(13.4, 7.7), layout='constrained')
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 2.2], height_ratios=[1.2, 1])
    ax = fig.add_subplot(gs[0, 0])
    for shaft, color in [('ICL', '#dd8b2b'), ('SCL', '#319cac')]:
        ix = np.array([str(n).startswith(shaft) for n in z['names']])
        xy = z['contacts'][ix]
        ax.plot(*xy.T, '-o', color=color, ms=4, lw=1, label=shaft)
        for n, p in zip(z['names'][ix], xy):
            if str(n) in ['SCL9', 'ICL9', 'ICL11', 'ICL1', 'ICL2', 'ICL3']:
                ax.annotate(str(n), p, xytext=(0, 7), textcoords='offset points', fontsize=7, ha='center')
    for i, p in enumerate(z['foci']):
        ax.add_patch(Circle(p, float(z['core_r']), fill=False, color='#ac3955', lw=1.8))
        ax.text(*p, f'C{i+1}', ha='center', va='center', fontsize=9, color='#ac3955')
    ax.plot(*z['foci'].T, '--', color='#ac3955', alpha=.5, lw=1)
    ax.set(xlim=(0, 20), ylim=(0, 20), aspect='equal', xlabel='x (mm)', ylabel='y (mm)', title='A   Archived Fig4B geometry')
    ax.legend(loc='upper right', frameon=False, fontsize=8)
    ax = fig.add_subplot(gs[0, 1])
    cmap = plt.get_cmap('viridis').copy(); cmap.set_bad('#e5e5e5')
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect='auto', interpolation='nearest')
    ax.set_yticks(range(15), z['names'], fontsize=8)
    ax.set_xticks(range(0, 34, 3), range(1, 35, 3))
    ax.set(xlabel='Detected event in time order (all 34; no direction filtering)', title='B   Contact participation and relative rank')
    ax.axhline(3.5, color='white', lw=1.5)
    cb = fig.colorbar(im, ax=ax, pad=.02, fraction=.025)
    cb.set_ticks([0, 1], labels=['Early', 'Late']); cb.set_label('Within-event rank')
    ax.text(.02, .93, 'Gray = nonparticipating; all four SCL contacts absent in this run', transform=ax.transAxes, fontsize=8)
    ax = fig.add_subplot(gs[1, :])
    colors = ['#d57636', '#438cb7', '#d0d0d0']
    labels = ['Positive direction', 'Negative direction', 'Direction not assigned']
    bottom = np.zeros(len(rows))
    for k, (color, label) in enumerate(zip(colors, labels)):
        ax.bar(range(len(rows)), counts[:, k], bottom=bottom, color=color, label=label, width=.8)
        bottom += counts[:, k]
    ax.set_xticks(range(len(rows)), [r['seed'] for r in rows])
    ax.set(xlabel='Legacy network seed (topology and dynamics coupled); 8 s per run', ylabel='Detected events',
           title='C   Earlier template-source geometry: all 688 events across 26 spontaneous dual-core runs')
    ax.legend(loc='upper right', frameon=False, fontsize=9)
    fig.suptitle('Hand-placed cores: bidirectional access was present; full patient-pattern recovery was not established', fontsize=12)
    for ext in ['png', 'pdf']:
        fig.savefig(OUT / f'figures/handplaced_geometry_and_all_events.{ext}', dpi=200)
    plt.close(fig)
    summary = dict(
        role='Historical diagnostic; distinct montage registrations are not pooled',
        representative_input=str(RAW / f'readout_{TAG}.json'),
        cohort_inputs=[str(RAW / f'readout_{t}.json') for t in audit['candidates'][1]['input_tags']],
        representative_detected_events=len(run['events']), representative_SCL_participating_events=0,
        cohort_detected_events=int(counts.sum()), cohort_positive=int(counts[:, 0].sum()),
        cohort_negative=int(counts[:, 1].sum()), cohort_unassigned=int(counts[:, 2].sum()),
        cohort_both_observed=int(np.sum(np.all(counts[:, :2] > 0, axis=1))),
        interpretation='Legacy axis signs are not validated patient TA/TB labels; missing direction is not proof of a missing mechanism.',
        core_source_contact_distances_mm={str(i+1): {str(n):float(np.linalg.norm(p-c)) for n,p in zip(z['names'],z['contacts'])}
                                          for i,c in enumerate(z['foci'])})
    (OUT / 'audit.json').write_text(json.dumps(summary, indent=2))
    (OUT / 'figures/README.md').write_text('### handplaced_geometry_and_all_events.png / .pdf\n'
        '上排重建存档 Fig4B 的双核与 SEEG 布局，并显示该次自发运行全部 34 个检测事件的参与和事件内归一化 rank；灰色表示未参与，紫到黄表示早到晚。'
        '下排统计更早 template-source 布局的 26 次自发运行全部 688 个事件；橙、蓝表示旧读出的正、反方向，灰色表示未能赋予方向，不能直接称为患者 TA/TB。'
        '上、下排的坐标注册版本不同，未将事件合并为同一条件；未运行新仿真，尚待用户目视审阅。\n'
        '**关注点**：自发正反方向确实存在，但存档示例的 SCL 完全未参与；9/26 次观测到双方向是有限时长的描述，不是其余网络缺乏第二模式的判定。\n')
    print(OUT)

if __name__ == '__main__':
    main()
