"""Whole-trajectory and spatial correspondence; no bifurcation labels."""
from common import OUT, BASE, model, np, read, write, log
from transient_response_network import DEST, LABEL, PARENT
from scipy.ndimage import uniform_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import argparse


def main(prescribed=False):
    destination = OUT/'transient_native_Z_path_20260923' if prescribed else DEST
    assert read(destination/'independent_comparison.json')['status'] == 'READOUT_AUDIT_PASS'
    s = model(40)
    native = np.load(BASE/'native_reference/seed9108401_readouts.npz')
    checkpoints = read(BASE/'native_reference/checkpoint_projections.json')
    native_slow = OUT/'transient_native_Z_path_20260923/native_slow_observations.npz'
    sources = [None, DEST/LABEL, destination/LABEL] if prescribed else [None, PARENT, DEST/LABEL]
    names = ['Native SNN', 'Rate: free Z', 'Rate: native Z(t)'] if prescribed else ['Native SNN', 'Parent rate model', 'Transient correction']
    colors = ['#202020', '#865ba6', '#27867b']
    plt.rcParams.update({'font.size':10, 'pdf.fonttype':42, 'svg.fonttype':'none',
        'axes.spines.top':False, 'axes.spines.right':False})
    fig = plt.figure(figsize=(10.6, 11))
    gs = fig.add_gridspec(5, 3, left=.09, right=.86, top=.955, bottom=.065,
        hspace=.56, wspace=.32, height_ratios=[.7, .7, .65, 1., 1.])
    snapshots = []
    for col, (source, name, color) in enumerate(zip(sources, names, colors)):
        if source is None:
            t, rate, field = native['t'], native['allE'], native['rate_cells']
            ts = np.array(sorted(map(int, checkpoints)))
            D = np.array([checkpoints[str(k)]['D'] for k in ts])
            M = np.array([checkpoints[str(k)]['mean_M']*.0005 for k in ts])
            ls, marker = 'none', 'o'
            if native_slow.exists():
                slow = np.load(native_slow)
                ts, D, M = slow['time_ms'], 1-slow['Z'], slow['M_feedback_mV']
                ls, marker = '-', None
        else:
            z = np.load(source/'trajectory.npz')
            t, rate, field = z['time_ms'], z['global_E_hz'], z['field_E_hz']
            ts, D = z['state_time_ms'], z['D']
            M = z['M_current'][:, s.E]@s.mean_weights
            ls, marker = '-', None
        smooth = uniform_filter1d(rate, 10, mode='nearest')
        for row, lim, ylim in [(0, (.5, 3), (0, 160)), (1, (0, 12.5), (0, 510))]:
            ax = fig.add_subplot(gs[row, col])
            ax.plot(t/1000, smooth, color=color, lw=.65)
            ax.set(xlim=lim, ylim=ylim, xlabel='Time (s)')
            if row == 0:
                ax.text(.5, 1.15, name, ha='center', transform=ax.transAxes)
            if col == 0:
                ax.set_ylabel('Global E rate (Hz)')
                ax.text(-.31, 1.1, 'AB'[row], fontsize=16, fontweight='bold', transform=ax.transAxes)
            else:
                ax.tick_params(labelleft=False)
            if row == 1:
                for tm in [4.025, 9.870]:
                    ax.axvline(tm, color='black', ls=':', lw=.55)
        ax = fig.add_subplot(gs[2, col])
        ax.plot(ts/1000, D, color='#202020', ls=ls, marker=marker, ms=3, lw=1.)
        other = ax.twinx()
        other.spines['right'].set_visible(True)
        other.plot(ts/1000, M, color='#ad5c28', ls=ls, marker=marker, ms=3, lw=1.)
        ax.set(xlim=(0, 12.5), ylim=(0, .7), xlabel='Time (s)', yticks=[0, .3, .6])
        other.set(ylim=(0, .3), yticks=[0, .15, .3])
        if col == 0:
            ax.set_ylabel(r'$D=1-\langle Z_E\rangle$')
            ax.text(-.31, 1.1, 'C', fontsize=16, fontweight='bold', transform=ax.transAxes)
        else:
            ax.tick_params(labelleft=False)
        if col == 2:
            other.set_ylabel(r'$\eta_M\langle M\rangle$ (mV)', color='#ad5c28')
        else:
            other.tick_params(labelright=False)
        other.tick_params(axis='y', colors='#ad5c28')
        for row, tm in [(3, 4025.), (4, 9870.)]:
            ax = fig.add_subplot(gs[row, col])
            mask = (t >= tm-25) & (t < tm+25)
            assert mask.sum() == 50
            im = ax.imshow(field[mask].mean(0).reshape(20, 20), origin='lower',
                extent=[0, 20, 0, 20], cmap='magma', vmin=0, vmax=500, interpolation='nearest')
            ax.set(xticks=[0, 10, 20], yticks=[0, 10, 20])
            if row == 4:
                ax.set_xlabel('x (mm)')
            else:
                ax.tick_params(labelbottom=False)
            if col == 0:
                ax.set_ylabel(f'{tm/1000:.3f} s\ny (mm)')
                ax.text(-.31, 1.03, 'DE'[row-3], fontsize=16, fontweight='bold', transform=ax.transAxes)
            else:
                ax.tick_params(labelleft=False)
            for center, letter in zip(s.geo['centers_mm'], 'AB'):
                ax.add_patch(Circle(center, 1.75, fill=False, color='#20c4cf', lw=.9))
                ax.text(center[0], center[1]+2.2, letter, color='#20c4cf', ha='center', fontsize=9)
            snapshots.append(dict(condition=name, window_ms=[tm-25, tm+25], samples=50))
    cax = fig.add_axes([.91, .065, .012, .35])
    fig.colorbar(im, cax=cax, ticks=[0, 250, 500], label='E rate (Hz)')
    folder = destination/'figures'
    folder.mkdir(exist_ok=True)
    stem = 'fig_transient_native_Z_path' if prescribed else 'fig_transient_response_network'
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(folder/f'{stem}.{ext}', dpi=190)
    plt.close(fig)
    write(folder/f'{stem}.json', dict(sources=[str(x) if x else str(BASE/'native_reference/seed9108401_readouts.npz') for x in sources],
        snapshots=snapshots, Z_modes=['dynamic', 'dynamic', 'prescribed'] if prescribed else ['dynamic']*3,
        M_dynamic_in_all=True, time_alignment='Original common external clock; no event or onset alignment.',
        native_slow=str(native_slow) if native_slow.exists() else 'Exact saved checkpoints only.',
        scope=('Native Z(t) supplied to the third column; D agreement is by construction. M and fast rates free. Conditional diagnostic only.' if prescribed else 'Whole-network diagnostic of a fixed local transient-response change.')+' Local validation remains failed; no model promotion or bifurcation labels.',
        agent_PNG_PDF_check='PENDING', human_visual_acceptance='PENDING'))
    introduction = ('三列比较原生SNN、新版率模型自由Z演化，以及供给原生完整Z空间轨迹的同一率模型；所有M均动态。第三列D与原生一致是外部供给的结果，不能算自主验证。'
        if prescribed else '三列比较原生SNN、上一版率模型和加入已锁定瞬态响应修正后的率模型，所有自由轨迹Z/M均动态。')
    (folder/'README.md').write_text(f'### {stem}.png / .pdf / .svg\n'+introduction+
        'A/B为早期和全程全局率，C为耗减D及M反馈；D/E是相同外部时钟4.025和9.870秒、50ms窗的空间活动，未按事件相位对齐。'
        '原生慢变量来自真实5–10ms观测及最终检查点；本图是对应检验，不是分岔图。**关注点**：早期自限事件、资源耗减与后期空间招募是否同时改善，不能用某一时刻的空间快照代替完整事件传播统计。\n')
    log('TRANSIENT NETWORK FIGURE', folder/f'{stem}.png')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--prescribed-Z', action='store_true')
    main(p.parse_args().prescribed_Z)
