"""Plot the explicitly local membrane-recovery diagnostic, not a bifurcation."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/topic4_sef_hfo/fig5_zm_runaway_mechanism_20260918'


def main():
    source = OUT / 'membrane_recovery_diagnostic'
    assert json.loads((source / 'independent_audit.json').read_text())['status'] == 'COUNT_AND_MOMENT_AUDIT_PASS'
    z = np.load(source / 'response.npz')
    t = z['phase_centres'] * float(z['T_ms'])
    plt.rcParams.update({'font.size': 14, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(3, 2, figsize=(11.2, 8.4), sharex=True, sharey='row')
    fig.subplots_adjust(left=.13, right=.98, top=.84, bottom=.09, hspace=.16, wspace=.17)
    for j in range(2):
        reset = float(z['pars'][j, 21])
        mean = z['moments_mean'][j]
        sem = z['moments_sem'][j]
        ax = axes[0, j]
        ax.plot(t, z['predicted_hz'][j, 1], color='#c77c24', lw=1.8, label='Rate response')
        ax.plot(t, z['measured_hz'][j], color='#96479b', lw=1.8, label='LIF population')
        ax.fill_between(t, z['measured_hz'][j]-2*z['sem_hz'][j],
                        z['measured_hz'][j]+2*z['sem_hz'][j], color='#96479b', alpha=.22, lw=0)
        ax.set_ylim(0, 500); ax.set_yticks([0, 250, 500])
        axes[1, j].axhline(0, color='black', lw=.8, ls='--')
        axes[1, j].plot(t, mean[:, 0]-reset, color='#96479b', lw=1.8)
        axes[1, j].fill_between(t, mean[:, 0]-reset-2*sem[:, 0], mean[:, 0]-reset+2*sem[:, 0],
                                color='#96479b', alpha=.22, lw=0)
        axes[1, j].set_ylim(-90, 10); axes[1, j].set_yticks([-80, -40, 0])
        axes[2, j].plot(t, 100*mean[:, 2], color='#96479b', lw=1.8)
        axes[2, j].fill_between(t, 100*(mean[:, 2]-2*sem[:, 2]), 100*(mean[:, 2]+2*sem[:, 2]),
                                color='#96479b', alpha=.22, lw=0)
        axes[2, j].set_ylim(0, 100); axes[2, j].set_yticks([0, 50, 100])
        axes[2, j].set_xlabel('Time in cycle (ms)')
        for i in range(3):
            a = axes[i, j]
            a.spines[['top', 'right']].set_visible(False)
            a.set_xlim(0, float(z['T_ms'])); a.set_xticks([0, 100, 200])
            a.text(-.11, 1.08, 'ABCDEF'[2*i+j], transform=a.transAxes, fontsize=17, fontweight='bold')
    axes[0, 0].set_ylabel('E rate\n(Hz / neuron)')
    axes[1, 0].set_ylabel('Mean V − Vreset\n(mV)')
    axes[2, 0].set_ylabel('Refractory cells\n(%)')
    axes[0, 0].text(.5, 1.08, 'Original input', transform=axes[0, 0].transAxes, ha='center')
    axes[0, 1].text(.5, 1.08, 'Mean varies; variance fixed', transform=axes[0, 1].transAxes, ha='center')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.55, .98), frameon=False, ncol=2)
    dest = OUT / 'figures'; stem = 'fig_local_membrane_recovery'
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(dest / f'{stem}.{ext}', dpi=180)
    plt.close(fig)
    meta = dict(source=str(source / 'response.npz'), audit=str(source / 'independent_audit.json'),
        group=39, population='Previously selected surround E subgroup',
        rows=['Rate with history-conditioned response prediction', 'Mean post-update voltage relative to reset, all cells', 'Post-update refractory fraction'],
        columns=['Original strong input', 'Same mean waveform; both variance intensities fixed at their cycle means'],
        uncertainty='Plus/minus 2 SEM across 8192 independent noise paths; repeated cycles combined within each path',
        voltage='Model voltage relative to model reset; negative values are allowed LIF voltages, not a negative resource D.',
        scope='Driven local population diagnostic, not an autonomous network, a native-network trajectory, or a bifurcation diagram.',
        agent_visual_review='PENDING', human_visual_acceptance='PENDING')
    (dest / f'{stem}.json').write_text(json.dumps(meta, indent=2)+'\n')
    path = dest / 'README.md'; body = path.read_text(); heading = f'### {stem}.png / .pdf / .svg'
    if heading not in body:
        path.write_text(body+'\n'+heading+'\n'
            '固定此前选定的外围E子群体，左列为原始强输入，右列仅保留均值随时间变化、两种方差固定。'
            '三行依次给出率响应与LIF均值、相对复位电位的群体平均膜电位、不应期占比；色带为8192条独立噪声路径的±2 SEM。'
            '记录器与原计数逐路径逐相位完全一致，负膜电位表示模型中的超极化，不是负资源D；本图不属于自主网络或分岔图。'
            '**关注点**：模型提前给出明显放电时，LIF膜电位仍在恢复且不应期占比很小，支持检查强输入后的膜电位记忆，但不能单凭此图确定新闭合方程。\n')
    print(dest / f'{stem}.png')


if __name__ == '__main__':
    main()
