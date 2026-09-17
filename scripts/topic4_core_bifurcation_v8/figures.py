"""Display-only refinement of the frozen v2-v7 bifurcation results.

No new dynamical equations, trajectories, or stability assignments are fitted.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import csv
import hashlib
import json
import sys
import numpy as np
from scipy.signal import resample, find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/topic4_core_bifurcation_v2'))
from rate_paths import saved_path
BASE = ROOT / 'results/topic4_sef_hfo'
V2 = BASE / 'core_burst_bifurcation_v2_20260915'
V5 = BASE / 'core_branch_connections_v5_20260915'
V6 = BASE / 'core_observable_bifurcation_v6_20260915'
V7 = BASE / 'core_network_bifurcation_v7_20260916'
OUT = BASE / 'core_bifurcation_types_v8_20260916'
FIG = OUT / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
J = r'$J_{\mathrm{EE,core}}$'
COL = dict(eq='#286aa4', mean='#c8831d', hi='#258253', lo='#329cac')
GROUP_COL = ['#286aa4', '#9553a3', '#28865c', '#df8243', '#c05f88', '#80804a']
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14,
    'axes.titlesize': 16, 'font.family': 'DejaVu Sans', 'pdf.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.fontsize': 11})
MANIFEST = []
BOOK = None

def read(p):
    return json.loads(Path(p).read_text())

def write(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')

SEQ = read(V7 / 'displayed_curve_sequences.json')
CP = list(csv.DictReader((V7 / 'critical_points.csv').open()))
upper_fold = read(V2 / 'upper_fold.json')
CP.append(dict(label='Unstable equilibrium fold', JEE_core=upper_fold['g'],
    A_mean_hz=upper_fold['r_hz'][0], B_mean_hz=upper_fold['r_hz'][1]))

def point(label):
    return next(r for r in CP if r['label'] == label)

def xy(label, group):
    r = point(label)
    return float(r['JEE_core']), float(r['AB'[group] + '_mean_hz'])

def save(fig, name, caption, focus):
    assert all(not ax.child_axes for ax in fig.axes)
    fig.canvas.draw()
    for ext in ('png', 'pdf'):
        fig.savefig(FIG / f'{name}.{ext}', dpi=200)
    if BOOK is not None:
        BOOK.savefig(fig)
    with Image.open(FIG / f'{name}.png') as im:
        im.load()
        pixels = list(im.size)
    MANIFEST.append(dict(name=name, caption=caption, focus=focus, pixels=pixels,
        axes=len(fig.axes), scales=[a.get_yscale() for a in fig.axes]))
    plt.close(fig)

def scale(ax, ylim=(-.02, 480)):
    ax.set_yscale('symlog', linthresh=1, linscale=.65, base=10)
    ticks = [0, .3, 1, 3, 10, 30, 100, 300]
    ticks = [x for x in ticks if ylim[0] <= x <= ylim[1]]
    ax.set_yticks(ticks, [f'{x:g}' for x in ticks])
    ax.set_ylim(*ylim)
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.grid(axis='y', color='#e6e6e6', linewidth=.6)
    ax.set_axisbelow(True)

def curves(ax, group):
    fold = read(V2 / 'fold.json')
    eq = read(V2 / 'equilibrium_spectrum.json')
    for direction, ls in [(-1, '-'), (1, '--')]:
        rows = [r for r in eq if r['direction'] == direction]
        # Preserve the continuation order, including all genuine turnbacks.
        xx = [fold['g']] + [r['g'] for r in rows]
        yy = [fold['r_hz'][group]] + [r['r_hz'][group] for r in rows]
        ax.plot(xx, yy, ls=ls, color=COL['eq'], lw=1.7)
    for rows in SEQ:
        assert len({r['stable'] for r in rows}) == 1
        stable = rows[0]['stable']
        ax.plot([r['g'] for r in rows], [r['mean'][group] for r in rows],
                color=COL['mean'], lw=2, ls='-' if stable else '--')
        if stable:
            for key in ('hi', 'lo'):
                ax.plot([r['g'] for r in rows], [r[key][group] for r in rows],
                        color=COL[key], lw=1.25)

def mark(ax, label, group, text, pos, marker='o', align=None):
    g, rate = xy(label, group)
    ax.plot(g, rate, marker=marker, ms=7, mfc='white', mec='#202020', mew=1.3, zorder=6)
    ax.annotate(text, (g, rate), xytext=pos, textcoords='axes fraction',
                ha=align or ('right' if pos[0] > .65 else 'left'), va='center', fontsize=11.5,
                arrowprops=dict(arrowstyle='-', color='#333333', lw=.8),
                bbox=dict(facecolor='white', edgecolor='none', pad=1.4), zorder=7)

def legend(ax):
    ax.legend(handles=[
        Line2D([], [], color=COL['eq'], label='Equilibrium'),
        Line2D([], [], color=COL['mean'], label='Periodic mean'),
        Line2D([], [], color=COL['hi'], label='Stable periodic maximum'),
        Line2D([], [], color=COL['lo'], label='Stable periodic minimum'),
        Line2D([], [], color='#333333', ls='-', label='Stable'),
        Line2D([], [], color='#333333', ls='--', label='Unstable')],
        loc='upper left', frameon=False)

def main_figures():
    for group in (0, 1):
        fig, ax = plt.subplots(figsize=(8.8, 8.8))
        fig.subplots_adjust(left=.13, right=.975, bottom=.11, top=.925)
        curves(ax, group)
        scale(ax)
        ax.set(xlim=(.45, 1.62), xlabel=J, ylabel=f'Core {"AB"[group]} E rate (Hz / cell)',
               title=f'Core {"AB"[group]} — bifurcation types')
        legend(ax)
        mark(ax, 'Unstable equilibrium fold', group,
             'Unstable equilibrium fold\n' + r'$\lambda=0$; already unstable',
             (.07, .53) if group == 0 else (.06, .31), 's')
        mark(ax, 'Low-rate equilibrium fold', group,
             'Equilibrium fold\n' + r'$\lambda=0$', (.38, .15), 's')
        mark(ax, 'Cycle fold', group, 'Cycle fold\n' + r'$\mu=+1$', (.39, .47))
        mark(ax, 'LP0c', group, 'Cycle folds + PD\nSurround recruitment', (.43, .66))
        if group == 0:
            positions = [(.70, .74), (.40, .86), (.97, .58), (.46, .975)]
        else:
            positions = [(.70, .74), (.52, .98), (.97, .64), (.45, .82)]
        for lab, pos, txt in zip(['LP1','PD1','PD2','PD3'], positions,
            ['LP1: cycle fold\n' + r'$\mu=+1$',
             'PD1: supercritical PD\n' + r'$\mu=-1$, decreasing $J$',
             'PD2: subcritical PD\n' + r'$\mu=-1$, increasing $J$',
             'PD3: supercritical PD\n' + r'$\mu=-1$, decreasing $J$']):
            mark(ax, lab, group, txt, pos, 'o' if lab == 'LP1' else '^')
        save(fig, f'00_core_{"AB"[group]}_bifurcation',
             '同一冻结六群体确定性模型的平衡点与周期解，保留v7曲线数值及延拓顺序。纵轴在0–1 Hz线性、1 Hz以上对数，谷值未用任意正数替代；黑色原生SNN编号菱形已全部删除。只标关键临界点，周边招募微区合并指示；HC极限另见起始局部图。',
             '方块为平衡点折点，圆圈为周期折点，三角为倍周期；主图无法分辨极窄的PD子支。')

        fig, ax = plt.subplots(figsize=(8.8, 8.8))
        fig.subplots_adjust(left=.13, right=.975, bottom=.11, top=.925)
        curves(ax, group)
        scale(ax, (25, 480))
        ax.set(xlim=(1.315, 1.435), xlabel=J, ylabel=f'Core {"AB"[group]} E rate (Hz / cell)',
               title=f'Core {"AB"[group]} — cycle fold and period doubling')
        pos = ([ (.13,.41), (.05,.16), (.93,.16), (.06,.9) ] if group == 0 else
               [ (.15,.37), (.04,.92), (.96,.50), (.72,.73) ])
        for label, where, words in zip(['LP1','PD1','PD2','PD3'], pos,
            ['LP1: cycle fold\n' + r'$\mu=+1$',
             'PD1: supercritical PD\nStable 2T branch toward lower J',
             'PD2: subcritical PD\nUnstable 2T branch\nat lower J',
             'PD3: supercritical PD\nStable 2T branch toward lower J']):
            mark(ax, label, group, words, where, 'o' if label == 'LP1' else '^')
        save(fig, f'01_core_{"AB"[group]}_right_types',
             '右侧周期支放大，沿用主图的颜色和稳定性线型，纵轴可见范围全部处于对数段。PD1、PD3向降低J方向超临界；PD2向增大J方向亚临界，其不稳定2T子支位于较小J侧。',
             '沿轨道延拓顺序读连接，不能按J从小到大将所有临界点串成唯一演化路径。')

def onset():
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2))
    fig.subplots_adjust(left=.08, right=.98, bottom=.17, top=.85, wspace=.3)
    for ax in axes:
        curves(ax, 0)
        scale(ax, (-.02, 25))
        ax.set(ylabel='Core A E rate (Hz / cell)', xlabel=J)
    axes[0].set(xlim=(1.119, 1.128), title='Equilibrium loss and separate cycle fold')
    mark(axes[0], 'Low-rate equilibrium fold', 0, 'Equilibrium fold\n' + r'$\lambda=0$', (.53,.35), 's')
    mark(axes[0], 'Cycle fold', 0, 'Cycle fold\n' + r'$\mu=+1$', (.42,.79))
    axes[1].set(xlim=(1.121813, 1.121841), title='Narrow coexistence window')
    axes[1].ticklabel_format(axis='x', style='sci', scilimits=(0, 0), useOffset=1.1218)
    mark(axes[1], 'Cycle fold', 0, 'Cycle fold', (.08,.92))
    mark(axes[1], 'HC limit estimate', 0, 'Homoclinic limit estimate\n' + r'$T\rightarrow\infty$', (.38,.42), '*')
    axes[1].axvline(float(point('HC limit estimate')['JEE_core']), color='#333333', ls=':', lw=.9)
    fig.suptitle('Low-rate equilibrium fold and burst coexistence', fontsize=17)
    save(fig, '02_onset_fold_and_homoclinic',
         '左图分开低率平衡点鞍结和更早出现的周期轨道鞍结；右图解析J约1.12182的低率/稳定周期共存窗口。星号是有限周期延拓支持的同宿极限估计，未画虚构曲线把有限周期端点强行连接到鞍点。',
         '周期折点出现时低率态尚稳定；同宿判断有强数值证据，但尚未直接求无限时间连接轨道。')

def modes():
    labels = ['Cycle fold', 'LP0a', 'LP0b', 'LP0c', 'PD0', 'PD0 (2T → 4T)', 'LP1', 'PD1', 'PD2', 'PD3']
    right, left, rows = [], [], []
    for label in labels:
        z = np.load(saved_path(point(label)['source']))
        components = []
        for key in (('mode','left_mode') if 'mode' in z else ('right_null','left_null')):
            v = z[key]
            if v.ndim == 1:
                v = v[:-1].reshape(len(z['r']), 6)
            f = (abs(v)**2).sum(0)
            components.append(100*f/f.sum())
        right.append(components[0]); left.append(components[1])
        rows.append(dict(label=label, J=float(z['g']), right_percent=components[0].tolist(), left_percent=components[1].tolist(), source=point(label)['source']))
    fig, axes = plt.subplots(1, 2, figsize=(13, 8.4))
    fig.subplots_adjust(left=.255, right=.875, top=.88, bottom=.13, wspace=.20)
    ylabels = [lab + ('  |  +1: cycle fold' if not lab.startswith('PD') else '  |  −1: PD') for lab in labels]
    for i, (ax, values) in enumerate(zip(axes, [right, left])):
        im = ax.imshow(values, cmap='Blues', vmin=0, vmax=100, aspect='auto')
        ax.set(xticks=range(6), xticklabels=['A E','B E','S E','A I','B I','S I'],
               yticks=range(len(labels)), yticklabels=ylabels if i == 0 else [],
               title=['Right critical rate mode', 'Adjoint critical rate mode'][i])
        for a in range(len(labels)):
            for b in range(6):
                n = values[a][b]
                if n >= 1:
                    ax.text(b, a, f'{n:.1f}', ha='center', va='center', color='white' if n>55 else '#172c42', fontsize=10)
    fig.suptitle('Which population expresses the critical mode?', fontsize=18)
    cax = fig.add_axes([.905, .24, .018, .49])
    fig.colorbar(im, cax=cax, label='Rate-mode squared norm (%)')
    save(fig, '03_critical_modes_and_types',
         '同一联合网络临界点的右率模和伴随率模，行名同时给出周期折点的+1或倍周期的−1判据。S表示周边；百分比为六个每神经元率坐标的平方范数，既不按细胞数加权，也不表示因果贡献。',
         'LP1集中于B E，PD1于B I，PD2于A E，PD3于A I；早期LP0a/b和PD0主要在周边E。')
    write('critical_mode_components.json', rows)

def doubled_waveforms():
    specs = [('PD1', 'mixed_lower_period2', 4, 'Stable 2T; supercritical toward lower J'),
             ('PD2', 'mixed_period2', 0, 'Unstable 2T; subcritical toward higher J'),
             ('PD3', 'tonic_lower_period2', 3, 'Stable 2T; supercritical toward lower J')]
    metrics = []
    fig, axes = plt.subplots(3, 2, figsize=(12, 11))
    fig.subplots_adjust(left=.09, right=.98, bottom=.08, top=.91, hspace=.55, wspace=.3)
    for row, (label, family, group, status) in enumerate(specs):
        source = V5 / 'periodic' / family / 'amp0.12_N4096.npz'
        z = np.load(source)
        rate = resample(z['r'], 65536, axis=0)*1000
        period = float(z['T']); half = len(rate)//2
        time = np.arange(len(rate))*period/len(rate)
        ax, bx = axes[row]
        for k in (0,1):
            ax.plot(time, rate[:,k], color=GROUP_COL[k], lw=1.3, label=f'Core {"AB"[k]} E')
        ax.axvline(period/2, color='#333333', ls=':', lw=.9)
        ax.set(xlabel='Time within complete orbit (ms)', ylabel='Rate (Hz / cell)',
               title=f'{label}: complete 2T = {period:.3f} ms', ylim=(0,420))
        ax.legend(frameon=False, fontsize=10)
        difference = rate[half:] - rate[:half]
        for k in dict.fromkeys((0,1,group)):
            bx.plot(time[:half], difference[:,k], color=GROUP_COL[k], lw=1.4,
                    label=['A E','B E','S E','A I','B I','S I'][k])
        bx.axhline(0, color='#333333', lw=.6)
        bx.set(xlabel='Phase within first half (ms)', ylabel=r'$r(t+T)-r(t)$ (Hz)', title=status)
        bx.legend(frameon=False, fontsize=10)
        intervals = None
        if label in ('PD1', 'PD2'):
            # A has isolated bursts here. B / PD3 remain high between local peaks,
            # so their peaks are deliberately NOT treated as burst events.
            rr = np.tile(rate[:,0],3)
            ii,_ = find_peaks(rr, prominence=1, distance=int(10/period*len(rate)))
            ii = ii[(ii>=len(rate)) & (ii<2*len(rate))] - len(rate)
            assert len(ii) == 2
            intervals = (np.diff(np.r_[ii,ii[0]+len(rate)])*period/len(rate)).tolist()
        stab = read(V5/'poincare'/family/'amp0.12_N4096/rk4_orthogonal_dt0.05.json')
        metrics.append(dict(label=label, source=str(source), J=float(z['g']),
            full_repeat_ms=period, A_burst_peak_intervals_ms=intervals,
            max_half_difference_hz=np.max(abs(difference),axis=0).tolist(),
            min_hz=rate.min(0).tolist(), max_transverse=stab['max_transverse'],
            stability_source=stab['source']))
    fig.suptitle('Period doubling changes the repeating pattern, not necessarily each burst interval', fontsize=16)
    save(fig, '04_period_doubling_waveform_changes',
         '使用三个已经求得并检查稳定性的真实2T子轨道。左列为A/B E完整重复周期，右列放大相邻半周期差，可见普通率轴下不明显的交替调制；PD2子轨道不稳定，不能作为持续吸引态。',
         'PD1的A相邻burst仍约187.5 ms，完整模式约375.1 ms才重复；高背景振荡的每个局部峰不能自动叫一次burst。')
    write('period_doubling_waveform_metrics.json', metrics)

def main():
    global BOOK
    with PdfPages(FIG / 'core_bifurcation_types.pdf') as book:
        BOOK = book
        main_figures(); onset(); modes(); doubled_waveforms()
        BOOK = None
    write('figure_manifest.json', MANIFEST)
    (FIG/'README.md').write_text('# Core 分岔类型与对数纵轴 v8\n\n' + '\n\n'.join(
        f'### {r["name"]}.png / .pdf\n{r["caption"]}\n**关注点**：{r["focus"]}' for r in MANIFEST)
        + '\n\n### core_bifurcation_types.pdf\n汇集本版七页主图与解释图，数值解均来自已保存的v2–v7结果。未增加原生SNN示例编号，也未更换模型。\n**关注点**：图已自查，仍待用户目视检查。\n')
    transform = matplotlib.scale.SymmetricalLogTransform(10, 1, .65)
    z = transform.transform_non_affine(np.array([0., 100., 480.]))
    write('display_validation.json', dict(status='PASS', model_changed=False,
        numbered_native_markers=0, scale='symlog', linthresh_hz=1, linscale=.65,
        zero_replaced=False, source_sequences=len(SEQ),
        source_periodic_points=sum(map(len,SEQ)),
        source_curve_sha256=hashlib.sha256((V7/'displayed_curve_sequences.json').read_bytes()).hexdigest(),
        fraction_vertical_range_below_100_hz=float((z[1]-z[0])/(z[2]-z[0])),
        figures=len(MANIFEST), human_acceptance='PENDING'))
    print('Generated', len(MANIFEST), 'figures in', FIG)

if __name__ == '__main__':
    main()
