#!/usr/bin/env python3
"""Read-only paired native evidence for the order of exit and Z recovery."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from campaign import ROOT, read, write, sha
from analyze_feedback_tail import SOURCE, load


def main():
    analysis = read(ROOT / 'feedback_tail_mechanism/analysis.json')
    assert analysis['status'] == 'COMPLETE_EXISTING_NATIVE_RECORDS'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
        'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(4, 2, figsize=(11.5, 8.6), sharex='col', sharey='row', layout='constrained')
    delayed, instant, limit_color = '#74519a', '#797979', '#bd7739'
    records = []
    for col, seed in enumerate([9108402, 9108403]):
        row = next(r for r in analysis['rows'] if r['seed'] == seed and r['tau_G_s'] == .5)
        zero = row['first_R_below5_for100ms_s']
        unblock = row['first_G_below_recovery_block_after_R5_s'] - zero
        assert zero is not None
        for tau, color in [(0., instant), (.5, delayed)]:
            folder = SOURCE / 'runs' / f'G30_response{tau:g}_s{seed}'
            assert read(folder / 'result.json')['status'] == 'COMPLETE'
            r = load(folder, 'mechanism_chunks', ['time_ms', 'global_E_rate_Hz', 'global_raw_conductance_ratio'])
            k = load(folder, 'intrinsic_adaptation_chunks', ['time_ms', 'sahp_mean_conductance_ratio'])
            b = load(folder, 'z_budget_chunks', ['time_ms', 'values'])
            with np.load(sorted((folder / 'z_budget_chunks').glob('*.npz'))[0]) as z:
                assert z['keys'][4] == 'net_per_s'
                assert z['region_names'][1:3].tolist() == ['core_A', 'core_B']
            assert np.array_equal(r['time_ms'], k['time_ms'])
            t = r['time_ms'] / 1000. - zero
            use = (t >= -2) & (t <= 3)
            for j, v in enumerate([r['global_E_rate_Hz'], r['global_raw_conductance_ratio'], k['sahp_mean_conductance_ratio']]):
                ax[j, col].plot(t[use], v[use], color=color, lw=1.15, rasterized=False)
            tb = b['time_ms'] / 1000. - zero
            take = (tb >= -2) & (tb <= 3)
            # Both cores have positive net recovery iff their minimum is positive.
            net = b['values'][:, 1:3, 4].min(1)
            ax[3, col].plot(tb[take], net[take], color=color, lw=1.3)
        ax[0, col].set_title(f'Seed {seed}', fontsize=12)
        ax[0, col].set_yscale('symlog', linthresh=5)
        ax[0, col].set_ylim(0, 600)
        ax[0, col].set_yticks([0, 5, 50, 200, 500], ['0', '5', '50', '200', '500'])
        ax[0, col].axhline(5, color=limit_color, lw=.8, ls=':')
        ax[1, col].axhline(row['G_block_threshold'], color=limit_color, lw=.9, ls='--')
        ax[1, col].set_ylim(-.4, 31)
        ax[2, col].set_ylim(-.3, 17)
        ax[3, col].axhline(0, color='#555555', lw=.8)
        ax[3, col].set_ylim(-.21, .21)
        for a in ax[:, col]:
            a.axvspan(0, unblock, facecolor='#ebcaa6', alpha=.28, zorder=-10)
            a.axvline(0, color='#aaaaaa', lw=.7)
            a.axvline(unblock, color=limit_color, lw=.7, ls=':')
            a.set_xlim(-2, 3)
            a.grid(axis='y', alpha=.12)
        ax[1, col].text(.97, .94, f'G recovery block: {row["G_block_threshold"]:.2f}',
                        transform=ax[1, col].transAxes, ha='right', va='top', color=limit_color, fontsize=9)
        ax[3, col].text(.97, .95, f'G unblocks at +{unblock:.2f} s',
                        transform=ax[3, col].transAxes, ha='right', va='top', color=limit_color, fontsize=9)
        ax[3, col].set_xlabel('Time from delayed-G low activity (s)')
        records.append(dict(seed=seed, absolute_alignment_s=zero, G_unblocking_relative_s=unblock,
                            paired_same_absolute_time=True, displayed_relative_window_s=[-2, 3]))
    for j, label in enumerate(['Causal E rate\n'+r'$R_G$ (Hz)', 'Raw global\n'+r'conductance $G$',
                               r'Mean adaptation $K$', 'Core net recovery\n'+r'min $\Delta Z/\Delta t$ (s$^{-1}$)']):
        ax[j, 0].set_ylabel(label)
    handles = [Line2D([], [], color=instant, label='Instant G'),
               Line2D([], [], color=delayed, label='G response 0.5 s')]
    fig.legend(handles=handles, ncol=2, loc='outside upper center', frameon=False)
    fig.supxlabel('Shading: activity is low but G still blocks Z recovery. Paired controls use the same absolute time.\n'
                  'Net Z uses exact 20-ms budgets; zero marks sampled R ≤ 5 Hz for 100 ms, a readout only.', fontsize=9)
    out = ROOT / 'figures'; out.mkdir(exist_ok=True)
    for ext in ['png', 'svg']:
        fig.savefig(out / f'feedback_tail_recovery.{ext}', dpi=180)
    plt.close(fig)
    write(out / 'feedback_tail_recovery_metadata.json', dict(
        source=str(ROOT / 'feedback_tail_mechanism/analysis.json'), producer_sha256=sha(__file__),
        records=records, statistical_unit='Two native paired noise seeds; reused original runs, not additional simulations.',
        interpretation='The delayed G tail coexists with low activity and later decays below the necessary threshold for Z recovery. The response-time intervention is paired; this trace alone does not identify a unique mediator or formal bifurcation. First low activity and first net recovery are not sufficient/full recovery; the8403rate rises again near+2.9s in this view.',
        formal_bifurcation=False, human_review='PENDING', agent_visual_review='PENDING'))
    path = out / 'README.md'
    text = path.read_text() if path.exists() else ''
    marker = '### feedback_tail_recovery.png / feedback_tail_recovery.svg\n'
    section = marker + '两列为原生噪声种子8402/8403，只比较即时G与0.5秒响应的既有配对仿真；横轴按延迟G轨迹首次持续低活动对齐，对照保持相同绝对时间。依次显示因果全E率、原始全局电导、K和两核净Z恢复速率的较小值，阴影标出低活动已经出现但G仍阻断Z恢复的区间。8403在约+2.9秒又有活动，不能把首次低活动/净恢复当成充分恢复；本图是退出机制诊断，不替换Fig.5主图。\n**关注点**：退出、解除Z恢复阻断和两核净恢复的先后；不能把G降低活动直接等同于G补充Z。\n'
    if marker in text:
        before, after = text.split(marker, 1); tail = after.find('\n### ')
        text = before + section + (after[tail:] if tail >= 0 else '')
    else:
        text = text.rstrip() + '\n\n' + section
    path.write_text(text)
    print('FEEDBACK_TAIL_RECOVERY_GENERATED', flush=True)


if __name__ == '__main__':
    main()
