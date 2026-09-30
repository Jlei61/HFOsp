#!/usr/bin/env python3
"""Exit-field response and the sign of natural recovery, same native means."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from campaign import ROOT, NATIVE, read, write, sha


def main():
    source = ROOT / 'exit_return_probes/extended_analysis_summary.json'
    actual = [r for r in read(source)['rows'] if '_fields16p7_' in r['name']]
    assert len(actual) == 4 and all(r['full_horizon'] for r in actual)
    common = [r for r in read(NATIVE / 'extended_analysis_summary.json')['rows']
              if r['job']['local_cut'] == 'exit' and r['job']['target_K'] in [6, 9, 12]]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
        'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 7), sharex='col', sharey='row', layout='constrained')
    colors = ['#74398f', '#d34e99', '#249ac1']
    records = []
    for col, history in enumerate(['high', 'recovery']):
        ax[0, col].set_title(history.capitalize() + ' initial history')
        for label, rows, style, marker in [('Common t20 fields', common, '-', 'o'),
                                            ('Actual-exit t16.7 fields', actual, '--', 's')]:
            rows = sorted([r for r in rows if r['job']['source_history'] == history], key=lambda r: r['job']['target_K'])
            x = [r['job']['target_K'] for r in rows]
            for j, color in enumerate(colors):
                ax[0, col].plot(x, [r['tail_mean_Hz'][j] for r in rows], color=color, ls=style, marker=marker, lw=1.4, ms=5)
                ax[1, col].plot(x, [r['counterfactual_drift_mean_allE_A_B_other'][j][0] for r in rows],
                                color=color, ls=style, marker=marker, lw=1.4, ms=5)
            for r in rows:
                records.append(dict(name=r['name'], field_family=label, history=history,
                    mean_Z=r['job']['target_Z'], mean_K=r['job']['target_K'],
                    rate_Hz=r['tail_mean_Hz'][:3], drift=r['counterfactual_drift_mean_allE_A_B_other'][:3],
                    tail_brief_events=r['tail_brief_events'], joint_quiet_fraction=r['tail_joint_quiet_fraction'],
                    censoring=r['censoring']))
        ax[0, col].set_ylim(-8, 510)
        ax[1, col].set_ylim(-.065, .19)
        ax[1, col].axhline(0, color='#777777', lw=.8)
        ax[1, col].set_xlabel('Held all-E mean K (gK/gL)')
        for a in ax[:, col]:
            a.set_xlim(5.6, 12.4); a.set_xticks([6, 9, 12]); a.grid(axis='y', alpha=.15)
    ax[0, 0].set_ylabel('Native E rate (Hz)')
    ax[1, 0].set_ylabel('Mean natural dZ/dt (s$^{-1}$)')
    handles = [Line2D([], [], color=c, label=l) for c, l in zip(colors, ['All E', 'Core A', 'Core B'])]
    handles += [Line2D([], [], color='#444444', ls=s, marker=m, label=l) for l, s, m in
                [('Common t20 Z/K fields', '-', 'o'), ('Actual-exit t16.7 Z/K fields', '--', 's')]]
    fig.legend(handles=handles, ncol=3, loc='outside upper center', frameon=False)
    fig.supxlabel('Same mean Z = 0.21 and mean K; both held spatial fields change together within each history.\n'
                  'Final 10 s of paired 30-s native branches; drift is measured with clamps held. No certified bifurcation.', fontsize=9)
    out = ROOT / 'figures'
    for ext in ['png', 'svg']:
        fig.savefig(out / f'exit_spatial_response.{ext}', dpi=180)
    plt.close(fig)
    write(out / 'exit_spatial_response_metadata.json', dict(source=str(source), rows=records,
        producer_sha256=sha(__file__), human_review='PENDING', agent_visual_review='PENDING', formal_bifurcation=False,
        interpretation='Z/K spatialfields change together at fixedmeans, intrinsic/G/Mhistory and futureinput. No single-field attribution and no autonomous-exitcount. Guide lines are finite-window responses, not equilibrium branches.'))
    p = out / 'README.md'; text = p.read_text()
    marker = '### exit_spatial_response.png / exit_spatial_response.svg\n'
    section = marker + '固定平均Z=0.21和各点K，只把共同t20的Z/K场一起换为真实16.7秒退出附近的空间场；左右为高态史和恢复史，各自保持相同未来输入与其余初态。上排显示末10秒率，下排读取钳制条件下的自然Z漂移，两个场同时改变不能归因于其中一个。线条只连接30秒原生条件响应，不代表认证的平衡分支或自主退出。\n**关注点**：K=9处双核持续活动与Z净漂移是否随空间场改变，K=12是否仍能压入低活动。\n'
    if marker in text:
        before, after = text.split(marker, 1); tail = after.find('\n### ')
        text = before + section + (after[tail:] if tail >= 0 else '')
    else:
        text = text.rstrip() + '\n\n' + section
    p.write_text(text)
    print('EXIT_SPATIAL_RESPONSE_GENERATED', flush=True)


if __name__ == '__main__':
    main()
