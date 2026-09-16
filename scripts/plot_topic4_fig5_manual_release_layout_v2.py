#!/usr/bin/env python3
"""Display-only revision: continuous aligned raster and separated phase paths.

Consumes the completed v1 native-SNN arrays; never launches a simulation.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, PowerNorm
from scipy.ndimage import gaussian_filter1d
import plot_topic4_fig5_manual_release as source

ROOT = Path(__file__).resolve().parents[1]
BASE = source.OUT
OUT = BASE / 'layout_v2'
FIG = OUT / 'figures'
E_COLOR = '#357ca8'
I_COLOR = '#d5813d'
Z_COLOR = '#7c418c'
I_CMAP = LinearSegmentedColormap.from_list(
    'inhibitory_rate', ['#326699', '#439984', '#c0a44c', '#a93740'])
I_NORM = Normalize(0, 650)
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 13,
    'axes.titlesize': 15, 'axes.labelsize': 15,
    'xtick.labelsize': 13, 'ytick.labelsize': 13,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False,
})


def write(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / (name + '.png'), dpi=180, bbox_inches='tight', facecolor='white')
    fig.savefig(FIG / (name + '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)


def selected_windows(t, e, run):
    # Remove only the quiet-window display; retain every other v1 time unchanged.
    win = [dict(w) for w in source.windows(t, e, run)
           if w['label'] != 'Quiet interval']
    labels = ['Self-limited', 'Entry', 'High activity', 'After release', 'Later high']
    for k, (w, label) in enumerate(zip(win, labels), 1):
        w['number'] = k
        w['short_label'] = label
    return win


def time_markings(ax, run, win):
    start = run['restore_start_ms'] / 1000
    release = run['release_ms'] / 1000
    ax.axvspan(start, release, color='#2e8c78', alpha=.11, lw=0, zorder=0)
    ax.axvline(release, color='#2e8c78', lw=1, ls='--', alpha=.9)
    for w in win:
        lo, hi = np.asarray(w['spatial_window_ms']) / 1000
        ax.axvspan(lo, hi, color=w['color'], alpha=.2, lw=0, zorder=0)
        ax.axvline(w['spatial_center_s'], color=w['color'], lw=.8, ls=':', alpha=.7)
    ax.set_xlim(0, run['duration_ms'] / 1000)
    ax.set_xticks(np.arange(0, 26, 5))
    ax.tick_params(axis='x', labelbottom=True)


def left_panels(fig, spec, a, run, win):
    gs = spec.subgridspec(5, 1, height_ratios=[1.15, 1.22, 1.04, .08, .86], hspace=.30)
    ax = fig.add_subplot(gs[0])
    lt = a['lfp_time_ms'] / 1000
    contacts = source.choose_contacts(a)
    raw = a['lfp_effective'][:, contacts]
    x = raw - np.median(raw[(lt >= .5) & (lt < 1)], axis=0)
    gain = max(float(np.max(np.quantile(x, .995, axis=0) - np.quantile(x, .005, axis=0))), 1e-12)
    offset = np.arange(len(contacts))[::-1]
    displayed = x / gain * .82 + offset
    for k, j in enumerate(contacts):
        color = '#d27943' if str(a['shaft_ids'][j]) in ('ICL', '0', 'A') else '#27929d'
        ax.plot(lt, displayed[:, k], lw=.48, color=color, rasterized=True)
    ax.set_yticks(offset)
    ax.set_yticklabels(a['contact_names'][contacts], fontsize=12)
    ax.set_ylim(-.25, len(contacts) + .25)
    ax.set_ylabel('Virtual SEEG\ncurrent proxy (a.u.)', fontsize=14)
    ax.set_title('A  Unfiltered contact readout', loc='left', weight='bold', pad=14)
    time_markings(ax, run, win)
    for w in win:
        ax.text(w['spatial_center_s'], .975, str(w['number']),
                transform=ax.get_xaxis_transform(), ha='center', va='top',
                fontsize=14, color=w['color'], weight='bold',
                bbox=dict(facecolor='white', edgecolor='none', alpha=.9, pad=.5))
    start = run['restore_start_ms'] / 1000
    release = run['release_ms'] / 1000
    ax.text((start + release) / 2, 1.04, 'Refill Z', color='#287b66', fontsize=12,
            transform=ax.get_xaxis_transform(), ha='center')
    ax.text(release + .65, 1.04, 'Release Z', color='#287b66', fontsize=12,
            transform=ax.get_xaxis_transform())

    ra = fig.add_subplot(gs[1], sharex=ax)
    # Exactly the v1 neuron subset, now displayed across all 26 s. No time thinning.
    selected = np.r_[np.arange(0, 60, 3), np.arange(60, 120, 3),
                     np.arange(120, 240, 6), np.arange(240, 300, 3)]
    spike_matrix = a['sample_spikes'][:, selected]
    st, sn = np.where(spike_matrix)
    ra.scatter(st * .0001, sn, s=.72, marker='.',
               c=np.where(selected[sn] < 240, E_COLOR, I_COLOR),
               linewidths=0, rasterized=True)
    ra.set_ylim(-1, 80)
    ra.set_yticks([10, 30, 50, 70])
    ra.set_yticklabels(['Core A E', 'Core B E', 'Other E', 'I'], fontsize=12)
    for y in (19.5, 39.5, 59.5):
        ra.axhline(y, color='#d2d2d2', lw=.7)
    ra.set_title('B  Continuous spike raster', loc='left', weight='bold', pad=12)
    time_markings(ra, run, win)

    sg = gs[2].subgridspec(2, 1, height_ratios=[2.2, 1], hspace=.08)
    za = fig.add_subplot(sg[0], sharex=ax)
    zt = a['z_time_ms'] / 1000
    zs = a['z_stats']
    za.fill_between(zt, zs[:, 2], zs[:, 4], color=Z_COLOR, alpha=.14, lw=0)
    za.plot(zt, zs[:, 0], color=Z_COLOR, lw=1.7, label='E mean Z')
    za.plot(zt, zs[:, 5], color='#ca548f', lw=.8, label='Core A')
    za.plot(zt, zs[:, 6], color='#3e9ec3', lw=.8, label='Core B')
    za.set(ylabel='Resource Z', ylim=(.25, 1.04), yticks=[.4, .7, 1.])
    za.set_title('C  Slow inhibitory resource', loc='left', weight='bold', pad=12)
    time_markings(za, run, win)
    za.tick_params(axis='x', labelbottom=False, length=0)
    za.legend(loc='lower left', ncol=3, fontsize=11, frameon=False, handlelength=1.5)
    za.text(.99, .92, 'M off', transform=za.transAxes, ha='right', va='top',
            fontsize=11, color='#666666')
    ga = fig.add_subplot(sg[1], sharex=ax)
    ga.plot(zt, zs[:, 8], color='#a96b32', lw=.75)
    ga.set(xlabel='Time, t (s)', ylim=(-.06, 1.08), yticks=[0, 1])
    ga.set_ylabel('$I_{GABA} \u2265 I_{th}$\nE fraction', fontsize=12)
    time_markings(ga, run, win)

    dh = fig.add_subplot(gs[3])
    dh.axis('off')
    dh.text(0, .45, 'D  Spatial activity (50-ms windows)', transform=dh.transAxes,
            fontsize=15, weight='bold', va='center')
    fg = gs[4].subgridspec(1, 5, wspace=.23)
    field_axes = []
    for k, w in enumerate(win):
        lo, hi = w['spatial_window_ms']
        field = a['field_e_count_1ms'][lo:hi].sum(0) / a['cell_e_counts'] / .05
        fa = fig.add_subplot(fg[k])
        im = fa.imshow(field.reshape(20, 20), origin='lower', extent=[0, 20, 0, 20],
                       cmap='magma', norm=PowerNorm(.6, vmin=0, vmax=500), interpolation='nearest')
        xy = a['centers_mm']
        fa.scatter(xy[:, 0], xy[:, 1], s=25, facecolors='none', edgecolors='#42d5d5', lw=1.)
        fa.set(xticks=[0, 20], yticks=[0, 20], xlabel='x (mm)')
        fa.tick_params(labelsize=12)
        if k:
            fa.set_yticklabels([])
        else:
            fa.set_ylabel('y (mm)')
        fa.set_title(f'{w["number"]}  {w["short_label"]}\n{w["time"]:.2f} s',
                     color=w['color'], fontsize=12, pad=7)
        field_axes.append(fa)
    cbax = field_axes[-1].inset_axes([1.045, 0, .04, 1])
    cb = fig.colorbar(im, cax=cbax, ticks=[0, 250, 500])
    cb.set_label('E rate (Hz)', fontsize=12, labelpad=3)
    cb.ax.tick_params(labelsize=11)
    # Verify after drawing: data-to-page mappings must match, not just time limits.
    fig.canvas.draw()
    axes = [ax, ra, za, ga]
    xb = np.array([[p.get_position().x0, p.get_position().x1] for p in axes])
    align = bool(np.allclose(xb, xb[0], rtol=0, atol=1e-10))
    assert align, xb
    assert int(spike_matrix.sum()) == len(st)
    assert displayed.min() >= ax.get_ylim()[0] and displayed.max() <= ax.get_ylim()[1]
    return dict(continuous_time_s=[0, 26],
                horizontal_alignment_pass=align, axes_horizontal_bounds=xb.tolist(),
                raster_neuron_count=len(selected), raster_indices=selected.tolist(),
                displayed_spike_count=len(st), temporal_thinning=False,
                contact_names=a['contact_names'][contacts].tolist(), readout_gain=gain,
                spatial_window_ms=50, spatial_max_hz=500,
                sample_times_unchanged_from_v1=True, quiet_snapshot_removed=True)


def phase_segments(run):
    start = run['restore_start_ms'] / 1000
    release = run['release_ms'] / 1000
    end = run['duration_ms'] / 1000
    return [(0, start, 'Native depletion'), (start, release, 'External refill'),
            (release, end, 'Native Z after release')]


def phase_panels(fig, spec, run, t, e, i, z, win, letter=True):
    gs = spec.subgridspec(3, 2, width_ratios=[1, .028], wspace=.07, hspace=.40)
    ee = gaussian_filter1d(e, 1)
    ii = gaussian_filter1d(i, 1)
    all_axes = []
    paths = []
    labels_at = {1: (125, .94), 2: (245, .85), 3: (410, .77),
                 4: (175, .94), 5: (380, .43)}
    for k, (lo, hi, title) in enumerate(phase_segments(run)):
        ax = fig.add_subplot(gs[k, 0])
        ix = np.flatnonzero((t >= lo) & (t < hi))
        points = np.column_stack([ee[ix], z[ix]])
        segments = np.stack([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=I_CMAP, norm=I_NORM, linewidths=.85, alpha=.85)
        lc.set_array((ii[ix[:-1]] + ii[ix[1:]]) / 2)
        ax.add_collection(lc)
        # Arc-length-spaced arrows show actual traversal without crowding quiet states.
        distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points / [500, .7], axis=0), axis=1))]
        arrow_indices = np.unique(np.searchsorted(distance, np.linspace(.12, .90, 5) * distance[-1]))
        for j in arrow_indices:
            end = min(j + 3, len(ix) - 1)
            if np.linalg.norm((points[end] - points[j]) / [500, .7]) < .004:
                continue
            ax.annotate('', xy=points[end], xytext=points[j],
                        arrowprops=dict(arrowstyle='-|>', mutation_scale=11, color='#303b48', lw=.75))
        for w in win:
            if not lo <= w['time'] < hi:
                continue
            j = int(np.argmin(abs(t - w['time'])))
            local = ix[abs(t[ix] - w['time']) < .125]
            ax.plot(ee[local], z[local], color=w['color'], lw=1.7, zorder=4)
            ax.scatter(ee[j], z[j], s=24, color='white', edgecolors=w['color'], lw=1.3, zorder=6)
            ax.annotate(str(w['number']), xy=(ee[j], z[j]), xytext=labels_at[w['number']],
                        color=w['color'], fontsize=14, weight='bold', ha='center', va='center',
                        bbox=dict(boxstyle='circle,pad=.18', fc='white', ec=w['color'], lw=1.1),
                        arrowprops=dict(arrowstyle='-', color=w['color'], lw=.8), zorder=7)
        ax.set(xlim=(-10, 500), ylim=(.3, 1.02), yticks=[.4, .6, .8, 1.],
               xticks=[0, 100, 200, 300, 400, 500], ylabel='Mean Z')
        ax.set_title(f'{title}  ·  {lo:g}–{hi:g} s', loc='left', fontsize=12, pad=6)
        if k == 2:
            ax.set_xlabel(r'E firing rate, $r_E$ (Hz)')
        else:
            ax.tick_params(axis='x', labelbottom=False)
        if k == 0:
            ax.text(0, 1.22, ('E  ' if letter else '') + 'Phase trajectory: E rate, Z and I rate',
                    transform=ax.transAxes, fontsize=15, weight='bold')
        all_axes.append(ax)
        paths.append(dict(stage=title, time_s=[lo, hi], plotted_samples=len(ix)))
    cb = fig.colorbar(lc, cax=fig.add_subplot(gs[:, 1]), ticks=[0, 200, 400, 600])
    cb.set_label(r'I firing rate, $r_I$ (Hz)', fontsize=14, labelpad=7)
    cb.ax.tick_params(labelsize=12)
    return dict(coordinates=['E rate (Hz)', 'mean E-target Z'], color='I rate (Hz)',
                rate_bin_ms=5, smoothing_sigma_ms=5, stage_paths=paths,
                direction='native time order', limits_same_across_stages=True,
                landmark_overlay='The same numbered 250-ms local trajectory in snapshot color; other segments use I-rate color.',
                full_trajectory_drawn=True, mathematical_vector_field_inferred=False)


def parameter_panels(fig, spec, rows, letter=True):
    meta = source.heatmaps(fig, spec, rows, letter=letter)
    # The original numeric calculation is retained; only display sizes change.
    for ax in fig.axes[-4:]:
        ax.tick_params(labelsize=12)
        ax.xaxis.label.set_size(14)
        ax.yaxis.label.set_size(13)
        ax.title.set_size(12)
    return meta


def main():
    a, run, t, e, i, z = source.load_main()
    win = selected_windows(t, e, run)
    rows = source.latency_rows()
    fig = plt.figure(figsize=(22.5, 14))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.85, 1], left=.065, right=.96,
                         top=.925, bottom=.065, wspace=.26)
    left = left_panels(fig, gs[0], a, run, win)
    right = gs[1].subgridspec(2, 1, height_ratios=[2.4, 1], hspace=.34)
    phase = phase_panels(fig, right[0], run, t, e, i, z, win)
    latency = parameter_panels(fig, right[1], rows)
    fig.suptitle('Slow inhibitory depletion and spatial recruitment in a dual-core SNN',
                 fontsize=19, y=.995)
    save(fig, 'fig5_manual_core_release_layout_v2')

    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(1, 1, left=.09, right=.935, top=.965, bottom=.065)
    left_standalone = left_panels(fig, gs[0], a, run, win)
    save(fig, 'continuous_seeg_raster_z_spatial')

    fig = plt.figure(figsize=(10, 12))
    gs = fig.add_gridspec(1, 1, left=.12, right=.88, top=.90, bottom=.07)
    phase_panels(fig, gs[0], run, t, e, i, z, win, letter=False)
    save(fig, 'separated_e_z_phase_trajectories')

    write('figure_metadata.json', dict(source_result=str(BASE), simulation_rerun=False,
          source_run=run, windows=win, composite_left=left, standalone_left=left_standalone,
          phase=phase, latency=latency,
          changes=['Continuous raster with exactly aligned time axes', 'Quiet snapshot omitted',
                   'Phase paths separated by protocol stage; I rate encoded by color',
                   'Larger labels and ticks; no lower-left footnote'],
          human_acceptance='PENDING_USER_REVIEW'))
    write('producer_manifest.json', {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in [Path(__file__), Path(source.__file__)]})
    (FIG / 'README.md').write_text('''### fig5_manual_core_release_layout_v2.png / .pdf
按用户要求重排已完成的同一条SNN轨迹：A为未滤波虚拟SEEG电流proxy，B为连续26秒raster，C为原生Z及耗竭驱动，三者严格共用时间坐标及高亮线。D删除安静期快照，保留五个原时间位置的50毫秒空间活动；坐标文字放大，移除左下脚注。
**关注点**：仅改显示，没有重跑仿真、改动读出公式或参数统计；M仍关闭，正式图验收待用户审阅。

### continuous_seeg_raster_z_spatial.png / .pdf
左侧独立放大版，连续raster沿用上一版固定的80个采样神经元，显示所有已记录脉冲，不进行时间抽稀。空间快照按时间排序，编号与A/B/C上的高亮线对应；五个空间图共用0–500Hz色标。
**关注点**：D作为离散快照允许横向等间距排版，A/B/C轴框及秒坐标严格对齐。删去安静期快照不删除连续时间序列中的安静段。

### separated_e_z_phase_trajectories.png / .pdf
把三维重叠轨迹展开成三个阶段的E率–Z相平面，使用同一坐标范围，轨迹颜色显示I率，箭头表示真实时间方向。全量轨迹保留，编号附近250毫秒另用与左侧相同的标记色加粗，并用引线标注，避免编号被轨迹遮挡。
**关注点**：5毫秒率分箱及5毫秒高斯平滑与上一版相同；这是轨迹显示，不新增闭合方向场、nullcline或分岔证明。第三段的持续高率没有被改画成极限环。
''', encoding='utf-8')
    write('delivery_status.json', dict(status='RENDERED_PENDING_AGENT_VISUAL_REVIEW',
          simulation_rerun=False, alignment_qa='PASS', spike_count_qa='PASS',
          human_acceptance='PENDING_USER_REVIEW'))
    print(json.dumps({'output': str(OUT), 'alignment': left['horizontal_alignment_pass'],
                      'raster_spikes': left['displayed_spike_count']}))


if __name__ == '__main__':
    main()
