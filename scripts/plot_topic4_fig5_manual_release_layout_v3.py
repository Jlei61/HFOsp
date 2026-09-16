#!/usr/bin/env python3
"""Display revision: state colors, readout-aligned snapshot, sparse native 3D paths."""
from pathlib import Path
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.text import Annotation
from mpl_toolkits.mplot3d import proj3d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import plot_topic4_fig5_manual_release_layout_v2 as previous

ROOT = Path(__file__).resolve().parents[1]
BASE = previous.BASE
OUT = BASE / 'layout_v3'
FIG = OUT / 'figures'
BLUE = '#347dad'
AMBER = '#d18a28'
RED = '#cb3c36'
GREEN = '#2b9276'
DARK_RED = '#a51d2d'
PALETTE = [BLUE, AMBER, RED, GREEN, DARK_RED]
ACTIVITY_CMAP = LinearSegmentedColormap.from_list(
    'activity_dark_to_red', plt.get_cmap('magma')(np.linspace(0, .69, 256)))


class Label3D(Annotation):
    def __init__(self, text, xyz, offset, color):
        super().__init__(text, xy=(0, 0), xytext=offset, textcoords='offset points',
                         ha='center', va='center', fontsize=14, color=color, weight='bold',
                         bbox=dict(boxstyle='circle,pad=.2', fc='white', ec=color, lw=1.2),
                         arrowprops=dict(arrowstyle='-', color=color, lw=.9),
                         annotation_clip=False, zorder=30)
        self.xyz = xyz

    def draw(self, renderer):
        xx, yy, _ = proj3d.proj_transform(*self.xyz, self.axes.get_proj())
        self.xy = (xx, yy)
        super().draw(renderer)


def write(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / (name + '.png'), dpi=180, bbox_inches='tight', facecolor='white')
    fig.savefig(FIG / (name + '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)


def windows(a, run, t, e):
    win = previous.selected_windows(t, e, run)
    for w, color in zip(win, PALETTE):
        w['color'] = color
    w = win[3]
    before = w['time']
    lt = a['lfp_time_ms'] / 1000
    ids = previous.source.choose_contacts(a)
    # Align the same event to the primary observable in A, not to a spatial pattern.
    y = a['lfp_effective'][:, ids].mean(axis=1)
    ix = np.flatnonzero(abs(lt - before) < .125)
    k = ix[np.argmax(y[ix])]
    w['time'] = float(lt[k])
    center_ms = int(round(w['time'] * 1000))
    w['spatial_window_ms'] = [center_ms - 25, center_ms + 25]
    w['spatial_center_s'] = center_ms / 1000
    alignment = dict(previous_time_s=before, readout_peak_s=w['time'],
                     shift_ms=(w['time'] - before) * 1000,
                     selection='Maximum mean applied-current proxy over the same eight A contacts, within the previous event +/-125 ms.',
                     spatial_pattern_used_in_selection=False)
    return win, alignment


def high_bands(run):
    return [(run['first_trigger_ms'] / 1000, run['restore_start_ms'] / 1000, RED),
            (run['all_trigger_times_ms'][1] / 1000, run['duration_ms'] / 1000, DARK_RED)]


def time_markings(ax, run, win):
    start = run['restore_start_ms'] / 1000
    release = run['release_ms'] / 1000
    for lo, hi, color in high_bands(run):
        ax.axvspan(lo, hi, color=color, alpha=.105, lw=0, zorder=0)
    ax.axvspan(start, release, color=GREEN, alpha=.10, lw=0, zorder=0)
    ax.axvline(release, color=GREEN, lw=1, ls='--', alpha=.85)
    for w in win:
        lo, hi = np.asarray(w['spatial_window_ms']) / 1000
        ax.axvspan(lo, hi, color=w['color'], alpha=.17, lw=0, zorder=0)
        ax.axvline(w['spatial_center_s'], color=w['color'], lw=.95, ls=':', alpha=.8)
    ax.set_xlim(0, run['duration_ms'] / 1000)
    ax.set_xticks(np.arange(0, 26, 5))
    ax.tick_params(axis='x', labelbottom=True)


def left_panels(fig, spec, a, run, win):
    # Reuse the exact aligned continuous readout/raster implementation.
    previous.time_markings = time_markings
    before = len(fig.axes)
    meta = previous.left_panels(fig, spec, a, run, win)
    axes = fig.axes[before:]
    for ax in axes:
        for im in ax.images:
            im.set_cmap(ACTIVITY_CMAP)
    for line in list(axes[0].lines):
        if len(line.get_xdata()) < 100:
            continue
        x = np.asarray(line.get_xdata())
        y = np.asarray(line.get_ydata())
        for lo, hi, color in high_bands(run):
            ix = (x >= lo) & (x < hi)
            axes[0].plot(x[ix], y[ix], color=color, lw=.7, rasterized=True)
    meta.update(sample_times_unchanged_from_v1=False,
                changed_snapshot_number=4,
                activity_color_scale='Same 0-500 Hz / PowerNorm gamma 0.6; magma truncated at 0.69 so highest rate is red.',
                high_bands=[dict(time_s=[lo, hi], color=c) for lo, hi, c in high_bands(run)])
    return meta


def rdp(points, tolerance=.0018):
    """Polyline simplification in normalized 3D coordinates; retains bends."""
    if len(points) < 3:
        return np.arange(len(points), dtype=int)
    keep = {0, len(points) - 1}
    work = [(0, len(points) - 1)]
    while work:
        lo, hi = work.pop()
        if hi <= lo + 1:
            continue
        delta = points[hi] - points[lo]
        q = points[lo + 1:hi]
        denom = np.dot(delta, delta)
        if denom < 1e-15:
            d = np.linalg.norm(q - points[lo], axis=1)
        else:
            u = np.clip(((q - points[lo]) @ delta) / denom, 0, 1)
            d = np.linalg.norm(q - (points[lo] + u[:, None] * delta), axis=1)
        j = int(np.argmax(d))
        if d[j] > tolerance:
            mid = lo + 1 + j
            keep.add(mid)
            work.extend([(lo, mid), (mid, hi)])
    return np.array(sorted(keep), dtype=int)


def sparse_paths(run, t, e, i, z, win):
    ee = gaussian_filter1d(e, 1)
    ii = gaussian_filter1d(i, 1)
    xyz = np.column_stack([ee, ii, z])
    norm = xyz / [500, 650, .7]
    release = run['release_ms'] / 1000
    restore = run['restore_start_ms'] / 1000
    first = run['first_trigger_ms'] / 1000
    second = run['all_trigger_times_ms'][1] / 1000
    peaks, _ = find_peaks(ee, prominence=20, distance=30)
    events = []
    paths = []
    # Retain four complete event-centred snippets in each native low-activity epoch.
    # Equal ordinal spacing, not selection by route, shape, or fit.
    for lo, hi, color, phase in [(win[0]['time'] - .01, win[1]['time'] - .3, BLUE, 'first'),
                                  (release + .5, second - .65, GREEN, 'second')]:
        pp = peaks[(t[peaks] >= lo) & (t[peaks] < hi)]
        chosen = pp[np.unique(np.round(np.linspace(0, len(pp) - 1, 4)).astype(int))]
        for p in chosen:
            ix = np.flatnonzero(abs(t - t[p]) <= .125)
            paths.append(dict(indices=ix, color=color, kind='selected_event', phase=phase))
            events.append(dict(peak_s=float(t[p]), window_s=[float(t[ix[0]]), float(t[ix[-1]])], phase=phase))
    quiet = np.flatnonzero((t > second - 2) & (t < second - .2) & (e < 1))
    second_entry = .5 * (float(t[quiet[-1]] + .0025) + second - .2) if len(quiet) else second - .6
    stages = [(win[1]['time'] - .2, first, AMBER, 'first', 'entry'),
              (first, restore, RED, 'first', 'high'),
              (restore, release, GREEN, 'refill', 'refill'),
              (second_entry - .2, second, AMBER, 'second', 'entry'),
              (second, float(t[-1]), DARK_RED, 'second', 'high')]
    summaries = []
    for lo, hi, color, phase, kind in stages:
        ix = np.flatnonzero((t >= lo) & (t <= hi))
        sel = rdp(norm[ix])
        paths.append(dict(indices=ix[sel], color=color, kind=kind, phase=phase))
        summaries.append(dict(kind=kind, time_s=[lo, hi], raw_points=len(ix), retained_points=len(sel), phase=phase))
    return xyz, paths, dict(selected_events=events, transition_paths=summaries,
                            simplification_tolerance_normalized=.0018,
                            normalization=[500, 650, .7], context_sample_ms=100,
                            sampling_scope='Display only. Disjoint event snippets are never connected across omitted intervals.')


def draw_3d(ax, run, t, xyz, paths, win, phases=None, overview=True, view=(22, -78)):
    if phases is None:
        phases = {'first', 'refill', 'second'}
    # Sparse, unconnected context preserves evidence of omitted native excursions.
    context = np.arange(0, len(t), 20)
    restore = run['restore_start_ms'] / 1000
    release = run['release_ms'] / 1000
    allowed = np.zeros(len(t), bool)
    if 'first' in phases:
        allowed |= t < restore
    if 'refill' in phases:
        allowed |= (t >= restore) & (t < release)
    if 'second' in phases:
        allowed |= t >= release
    context = context[allowed[context]]
    ax.scatter(*xyz[context].T, s=.9, c='#8e9ca8', alpha=.16, depthshade=False)
    for path in paths:
        if path['phase'] not in phases:
            continue
        ix = path['indices']
        width = (1.0 if path['kind'] == 'selected_event' else 1.7) if overview else 1.15
        ax.plot(*xyz[ix].T, color=path['color'],
                lw=2.1 if path['kind'] == 'refill' and overview else width,
                ls='--' if path['kind'] == 'refill' else '-', alpha=.92)
        # Arrows use original neighbouring samples, independent of line simplification.
        if path['kind'] == 'selected_event':
            jump = np.linalg.norm(np.diff(xyz[ix] / [500, 650, .7], axis=0), axis=1)
            js = [int(np.argmax(jump))]
        else:
            js = np.unique(np.linspace(0, max(0, len(ix)-2), 3, dtype=int))
        for j in js:
            k = ix[j]
            end = min(k + 3, ix[-1])
            ax.add_artist(previous.source.TrajectoryArrow3D(xyz[k], xyz[end], path['color']))
    offsets = {1: (-35, 22), 2: (-40, -22), 3: (35, 8), 4: (38, 28), 5: (18, -16)}
    if not overview:
        offsets = {1: (-15, 20), 2: (-25, -15), 3: (18, 10), 4: (25, 22), 5: (10, -15)}
    for w in win:
        phase = 'first' if w['number'] <= 3 else 'second'
        if phase not in phases:
            continue
        point = np.array([np.interp(w['time'], t, xyz[:, j]) for j in range(3)])
        ax.scatter(*point, s=32 if overview else 20, facecolor='white', edgecolor=w['color'],
                   lw=1.4, depthshade=False, zorder=20)
        label = Label3D(str(w['number']), point, offsets[w['number']], w['color'])
        if not overview:
            label.set_fontsize(12)
        ax.add_artist(label)
    ax.set(xlim=(-15, 500), ylim=(-20, 650), zlim=(.3, 1.035),
           xticks=[0, 250, 500], yticks=[300, 600], zticks=[.4, .6, .8, 1.])
    ax.set_xlabel(r'E rate, $r_E$ (Hz)', fontsize=14 if overview else 11, labelpad=6 if overview else 0)
    ax.set_ylabel(r'I rate, $r_I$ (Hz)', fontsize=14 if overview else 11, labelpad=7 if overview else 0)
    ax.set_zlabel('Mean Z', fontsize=14 if overview else 11, labelpad=8 if overview else 0)
    ax.tick_params(labelsize=12 if overview else 10, pad=0)
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_proj_type('ortho')
    ax.set_box_aspect((1.2, 1.0, 1.05), zoom=1 if overview else .86)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((.97, .98, .99, .45))
        axis._axinfo['grid'].update(color=(.6, .65, .7, .24), linewidth=.5)


def right_panels(fig, spec, run, t, xyz, paths, win):
    gs = spec.subgridspec(2, 1, height_ratios=[1.55, 1], hspace=.05)
    ax = fig.add_subplot(gs[0], projection='3d')
    draw_3d(ax, run, t, xyz, paths, win)
    ax.set_title('E  Native E–I–Z state trajectory', fontsize=16, loc='left', weight='bold', pad=15)
    legend = [Line2D([], [], color=BLUE, lw=2, label='Self-limited'),
              Line2D([], [], color=AMBER, lw=2, label='Entry'),
              Line2D([], [], color=RED, lw=2, label='High activity'),
              Line2D([], [], color=GREEN, lw=2, ls='--', label='Z refill'),
              Line2D([], [], color=GREEN, lw=2, label='Resumed events')]
    ax.legend(handles=legend, loc='upper left', bbox_to_anchor=(-.03, 1.04), ncol=2,
              fontsize=11, frameon=False, handlelength=1.4, columnspacing=1.)
    lower = gs[1].subgridspec(1, 2, wspace=.12)
    for j, (phases, title) in enumerate([({'first'}, 'First passage: 1 → 2 → 3'),
                                        ({'second'}, 'After refill: 4 → 5')]):
        ax = fig.add_subplot(lower[j], projection='3d')
        draw_3d(ax, run, t, xyz, paths, win, phases=phases, overview=False, view=(24, -72))
        ax.set_title(title, fontsize=12, pad=6)


def main():
    a, run, t, e, i, z = previous.source.load_main()
    win, alignment = windows(a, run, t, e)
    xyz, paths, sparse = sparse_paths(run, t, e, i, z, win)
    fig = plt.figure(figsize=(23, 14))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1], left=.06, right=.965,
                         top=.925, bottom=.065, wspace=.18)
    left = left_panels(fig, gs[0], a, run, win)
    right_panels(fig, gs[1], run, t, xyz, paths, win)
    fig.suptitle('Slow inhibitory depletion and spatial recruitment in a dual-core SNN', fontsize=19, y=.99)
    save(fig, 'fig5_manual_core_release_layout_v3')
    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(1, 1, left=.09, right=.935, top=.965, bottom=.065)
    standalone_left = left_panels(fig, gs[0], a, run, win)
    save(fig, 'continuous_seeg_raster_z_spatial')
    fig = plt.figure(figsize=(12, 13))
    gs = fig.add_gridspec(1, 1, left=.055, right=.92, top=.94, bottom=.04)
    right_panels(fig, gs[0], run, t, xyz, paths, win)
    save(fig, 'sparse_native_e_i_z_3d')
    write('figure_metadata.json', dict(source_result=str(BASE), simulation_rerun=False,
          windows=win, snapshot4_alignment=alignment, composite_left=left,
          standalone_left=standalone_left, sparse_phase=sparse,
          phase_coordinates=['All-E rate (Hz)', 'All-I rate (Hz)', 'Mean E-target Z'],
          phase_rate_bin_ms=5, phase_smoothing_sigma_ms=5,
          parameter_panels_removed=True, human_acceptance='PENDING_USER_REVIEW'))
    arr = {f'path_{k:02d}_indices': p['indices'] for k, p in enumerate(paths)}
    np.savez_compressed(OUT / 'phase_display_arrays.npz', time_s=t, full_xyz=xyz, **arr)
    write('producer_manifest.json', {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in [Path(__file__), Path(previous.__file__), Path(previous.source.__file__)]})
    (FIG / 'README.md').write_text('''### fig5_manual_core_release_layout_v3.png / .pdf
连续A/B/C沿用上一版时间对齐，③首次持续高活动和⑤再次高活动用红色及深红色，④保留同一次补回后事件并移至八通道平均虚拟SEEG电流读出的峰值。D统一使用暗色到红色的绝对Hz色标；参数比例/时间图已从本版移除，右侧恢复三维E率/I率/Z轨迹。
**关注点**：仅调整显示和④的18.5毫秒取样偏移，没有重新仿真；高活动仍按原200Hz/200毫秒判据标注，不增加发作或Hopf机制结论。

### continuous_seeg_raster_z_spatial.png / .pdf
左侧放大图保留80个固定采样神经元的全26秒raster、相同的未滤波电流读出和Z统计。高活动阶段使用淡红色底纹，所有空间图共用0–500Hz、gamma=0.6的同一颜色映射，最大值对应红色。
**关注点**：①②③⑤时刻不变；④只按原示例附近125毫秒范围内的平均电极读出峰值选择，不使用空间图形筛选。

### sparse_native_e_i_z_3d.png / .pdf
三维总览呈现原生耗竭、进入高活动、外部补回和释放后再次进入高态，下方分开显示第一次和第二次经过。每个原生自限事件阶段按事件序号等间距选4段250毫秒轨迹，背景每100毫秒显示一个淡点；进入、补回及高态保留完整时间区间，仅用归一化三维误差0.0018的折线简化去掉冗余点。
**关注点**：所有坐标均来自原生SNN的5毫秒率分箱和原5毫秒高斯平滑；未显示的事件段不跨空档连接、不平移轨迹、不构造极限环。番号由同一时刻插值得到，引线仅用于标签排版。
''', encoding='utf-8')
    write('delivery_status.json', dict(status='RENDERED_PENDING_AGENT_VISUAL_REVIEW',
          simulation_rerun=False, human_acceptance='PENDING_USER_REVIEW'))
    print(json.dumps({'out': str(OUT), 'snapshot4': alignment, 'events_shown': len(sparse['selected_events'])}))


if __name__ == '__main__':
    main()
