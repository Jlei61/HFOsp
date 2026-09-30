"""Local strong-input diagnostic, explicitly separate from bifurcation figures."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/topic4_sef_hfo/fig5_zm_runaway_mechanism_20260918'


def main():
    source = OUT / 'factorial_waveform'
    read = lambda p: json.loads(p.read_text())
    assert read(source / 'independent_audit.json')['status'] == 'COUNT_LEVEL_AUDIT_PASS'
    result = read(source / 'result.json')
    z = np.load(source / 'response.npz')
    t = z['phase_centres'] * float(z['T_ms'])
    selected = [i for i, row in enumerate(result['rows']) if row['source_label'] == 'Surround E']
    assert [result['rows'][i]['condition'] for i in selected] == ['full', 'mean_only', 'variances_only']
    assert {result['rows'][i]['source_group'] for i in selected} == {39}
    plt.rcParams.update({'font.size': 11, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    fig.subplots_adjust(left=.09, right=.985, bottom=.20, top=.78, wspace=.18)
    for label, ax, index in zip('ABC', axes, selected):
        ax.plot(t, z['predicted_hz'][index, 0], color='black', lw=1.3, label='Original response')
        ax.plot(t, z['predicted_hz'][index, 1], color='#c77c24', lw=1.5, label='History response')
        ax.plot(t, z['measured_hz'][index], color='#96479b', lw=1.5, label='Driven LIF population')
        ax.fill_between(t, z['measured_hz'][index] - 2 * z['sem_hz'][index],
                        z['measured_hz'][index] + 2 * z['sem_hz'][index], color='#96479b', alpha=.2, lw=0)
        ax.text(-.08, 1.06, label, transform=ax.transAxes, fontsize=14, fontweight='bold')
        ax.set_xlim(0, float(z['T_ms']))
        ax.set_ylim(0, 500)
        ax.set_xticks([0, 100, 200])
        ax.set_yticks([0, 250, 500])
        ax.set_xlabel('Time in cycle (ms)')
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Surround E subgroup\n(Hz / neuron)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.55, .995), ncol=3, frameon=False)
    dest = OUT / 'figures'
    stem = 'fig_local_mean_variance_counterfactual'
    for suffix in ['png', 'pdf', 'svg']:
        fig.savefig(dest / f'{stem}.{suffix}', dpi=190)
    plt.close(fig)
    metadata = dict(source=str(source / 'response.npz'), audit=str(source / 'independent_audit.json'),
                    selected_group=39, selection='Previously selected surround E subgroup; no new subgroup search.',
                    panels={'A': 'Original time-dependent mean and variance intensities',
                            'B': 'Original mean waveform; both variance intensities held at their own cycle means',
                            'C': 'Original variance intensity waveforms; mean held at its cycle mean'},
                    displayed_models='Both use corrected voltage units; no new coefficients fitted.',
                    uncertainty='Mean plus/minus 2 SEM across 8192 independent noise paths; cycles combined within each path.',
                    scope='Imposed-input local population assay, not autonomous SNN, a whole-surround average, or a bifurcation diagram.',
                    human_visual_acceptance='PENDING')
    (dest / f'{stem}.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    path = dest / 'README.md'
    body = path.read_text()
    heading = f'### {stem}.png / .pdf / .svg'
    if heading not in body:
        path.write_text(body + '\n' + heading + '\n'
                        '固定此前选定的外围E群体，A为原始均值与方差共同变化，B仅均值变化，C仅方差变化；被固定的量取各自整周期均值。'
                        '黑色和橙色分别为电压单位已修正的原响应与历史响应，紫色为8192条新噪声路径下的LIF均值，色带为±2 SEM。'
                        '这是局部输入检验，不是自主网络、外围整体平均或分岔图，图内没有总体标题。'
                        '**关注点**：仅强均值变化已保留外围波形失配；仅方差变化在本条件通过，不表示方差近似在所有联合变化下均准确。\n')
    print(dest / f'{stem}.png')


if __name__ == '__main__':
    main()
