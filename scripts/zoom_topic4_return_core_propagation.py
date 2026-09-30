#!/usr/bin/env python3
"""Observation-only zoom of the existing Fig.5 return; no simulation or gates changed."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path('/data/hfosp/topic4_sef_hfo/fig5_global_feedback_response_20260923')
NAME = 'G30_response0.5_s9108405'
OUT = ROOT / 'return_propagation_zoom_20260924'
COLORS = ['#74398f', '#d34e99', '#249ac1', '#37698b']
LABELS = ['All E', 'Core A', 'Core B', 'Other E']


def load_native():
    keys = ['spikes_1ms', 'regions_1ms', 'field_5ms', 'raster', 'slow_time_ms', 'Z']
    parts = {k: [] for k in keys}
    end = 0
    for path in sorted((ROOT / 'runs' / NAME / 'chunks').glob('*.npz')):
        with np.load(path) as f:
            assert int(f['start_step']) == end
            end = int(f['end_step'])
            for k in keys:
                parts[k].append(f[k])
    data = {k: np.concatenate(v) for k, v in parts.items()}
    with np.load(ROOT / 'geometry.npz') as f:
        geo = {k: f[k] for k in f.files}
    counts = np.r_[32000, geo['region_counts'][:3]]
    raw = np.c_[data['spikes_1ms'][:, 0], data['regions_1ms'][:, :3]]
    data['rates'] = raw.reshape(-1, 5, 4).sum(1) / counts / .005
    data['field_rates'] = data['field_5ms'] / geo['cell_e_counts'] / .005
    assert np.array_equal(data['field_5ms'].sum(1), data['spikes_1ms'][:, 0].reshape(-1, 5).sum(1))
    assert np.array_equal(data['regions_1ms'][:, :3].sum(1), data['spikes_1ms'][:, 0])
    with np.load(ROOT / 'references/native_s9108405.npz') as f:
        for k in ['spikes_1ms', 'regions_1ms', 'field_5ms', 'raster']:
            assert np.array_equal(data[k][:len(f[k])], f[k]), k
    # The original observer uses radius1.75mm, around the physical radius1.5mm core.
    pos = geo['positions_e']
    cell = np.floor(pos[:, 0]).astype(int) + 20*np.floor(pos[:, 1]).astype(int)
    assert np.array_equal(np.bincount(cell, minlength=400), geo['cell_e_counts'])
    core_counts = [int((np.linalg.norm(pos-c, axis=1) < 1.75).sum()) for c in geo['centers_mm']]
    assert core_counts == geo['region_counts'][:2].tolist()
    return data, geo


def cores(ax, geo):
    for label, c in zip('AB', geo['centers_mm']):
        ax.add_patch(Circle(c, float(geo['core_radius_mm']), fill=False, color='#36dbd3', lw=1.2))
        ax.add_patch(Circle(c, 1.75, fill=False, color='#36dbd3', lw=.55, ls=':'))
        ax.text(c[0], c[1]+2.0, label, color='#36dbd3', ha='center', fontsize=9, weight='bold')


def field(ax, values, geo):
    im = ax.imshow(values.reshape(20, 20), origin='lower', extent=(0, 20, 0, 20),
                   interpolation='nearest', cmap='magma', vmin=0, vmax=500)
    cores(ax, geo)
    ax.set(xticks=[0, 10, 20], yticks=[0, 10, 20])
    return im


def trace(ax, data, lo, hi, relative=False):
    a, b = round(lo*200), round(hi*200)
    t = (np.arange(a, b)+.5)*.005
    for j, (color, label) in enumerate(zip(COLORS, LABELS)):
        ax.plot((t-lo)*1000 if relative else t, data['rates'][a:b, j], color=color,
                lw=1.2 if j in (1, 2) else .85, label=label)
    ax.set(ylim=(0, 510), xlim=(0, (hi-lo)*1000) if relative else (lo, hi), ylabel='E rate (Hz)')


def temporal_zoom(data, row):
    fig, axes = plt.subplots(3, 2, figsize=(13, 7.5), sharex='col',
                             gridspec_kw={'height_ratios': [1.6, 1, .8]})
    windows = [(.5, 3.5), (100.6, 103.6)]
    for col, (lo, hi) in enumerate(windows):
        a, b = round(lo*10000), round(hi*10000)
        it, ix = np.where(data['raster'][a:b])
        for lower, upper, color in [(0, 20, COLORS[1]), (20, 40, COLORS[2]),
                                     (40, 60, COLORS[3]), (60, 80, '#bd7736')]:
            mask = (ix >= lower) & (ix < upper)
            axes[0, col].scatter((it[mask]+a)*.0001, ix[mask], s=3, c=color, lw=0)
        axes[0, col].set(yticks=[9.5, 29.5, 49.5, 69.5],
                         yticklabels=['Core A E', 'Core B E', 'Other E', 'I'], ylim=(-1, 87),
                         title=['Original interictal activity', 'Returned interictal activity'][col])
        trace(axes[1, col], data, lo, hi)
        t = data['slow_time_ms']/1000
        m = (t >= lo) & (t < hi)
        for i, c in [(5, COLORS[1]), (6, COLORS[2])]:
            axes[2, col].plot(t[m], data['Z'][m, i], color=c, lw=1.2)
        axes[2, col].set(ylim=(.7, 1.01), ylabel='Core Z', xlabel='Time (s)', xlim=(lo, hi))
    for lo, hi, label in [(100.86, 100.92, '1'), (101.42, 101.48, '2'), (101.81, 101.97, '3')]:
        for ax in axes[:, 1]:
            ax.axvspan(lo, hi, color='#b6b6b6', alpha=.20, zorder=0)
        axes[0, 1].text((lo+hi)/2, 82, label, ha='center', va='bottom', fontsize=11)
    axes[1, 0].legend(frameon=False, ncol=4, fontsize=9, loc='upper center', bbox_to_anchor=(.5, 1.22))
    fig.subplots_adjust(left=.085, right=.99, bottom=.09, top=.91, wspace=.28, hspace=.28)
    fig.savefig(OUT/'figures/return_transition_zoom.png', dpi=180)
    plt.close(fig)


def storyboard(data, geo, examples, filename):
    # Windows are chronological and identical in width; no peak alignment or warping.
    fig = plt.figure(figsize=(17, 2.9*len(examples)))
    grid = fig.add_gridspec(len(examples), 8, left=.07, right=.925, bottom=.10, top=.89,
                            wspace=.30, hspace=.72, width_ratios=[1.45, 1.45, 1, 1, 1, 1, 1, 1])
    for ri, (label, start) in enumerate(examples):
        start = round(start*200)/200
        ax = fig.add_subplot(grid[ri, :2])
        trace(ax, data, start, start+.15, relative=True)
        ax.set_title(label+f'\nt = {start:.3f} s', fontsize=11)
        if ri == len(examples)-1:
            ax.set_xlabel('Time from window start (ms)')
        if ri == 0:
            ax.legend(frameon=False, fontsize=8, ncol=2, loc='upper left')
        for ci, offset in enumerate([0, 25, 50, 75, 100, 125]):
            ax = fig.add_subplot(grid[ri, ci+2])
            index = round(start*200)+offset//5
            im = field(ax, data['field_rates'][index], geo)
            ax.set_title(f'+{offset}\u2013{offset+5} ms', fontsize=10)
            if ci:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel('y (mm)')
            if ri == len(examples)-1:
                ax.set_xlabel('x (mm)')
    cax = fig.add_axes([.94, .26, .013, .48])
    fig.colorbar(im, cax=cax, label='Native E rate (Hz; 5-ms bins)')
    fig.savefig(OUT/'figures'/filename, dpi=170)
    plt.close(fig)


def event_metrics(data, event):
    a, b = round(event['start_s']*200), round(event['end_s']*200)
    rates = data['rates'][a:b]
    peak = rates.max(0)
    order = {}
    for threshold in [25., 50., 100.]:
        hits = []
        for j in [1, 2, 3]:
            mask = rates[:, j] >= threshold
            ii = np.flatnonzero(mask[:-1] & mask[1:])
            hits.append(float(ii[0]*5) if len(ii) else None)
        core = [t for t in hits[:2] if t is not None]
        first = min(core) if core else None
        # Threshold order of population averages is descriptive, not source identification.
        order[str(int(threshold))] = dict(onset_ms_A_B_other=hits,
            core_precedes_other=(first < hits[2]) if first is not None and hits[2] is not None else None)
    return dict(start_s=event['start_s'], end_s=event['end_s'], peak_5ms_Hz=peak.tolist(),
                core_to_other_peak_ratio=float(max(peak[1:3])/max(peak[3], 1e-12)),
                core_peak_over100=bool(max(peak[1:3]) >= 100), order=order)


def summarize(items):
    p = np.array([x['peak_5ms_Hz'] for x in items])
    order = dict(A_first=0, B_first=0, same_5ms_bin=0, both_reach100Hz_for10ms=0)
    for item in items:
        a, b, _ = item['order']['100']['onset_ms_A_B_other']
        if a is not None and b is not None:
            order['both_reach100Hz_for10ms'] += 1
            order['A_first' if a < b else 'B_first' if b < a else 'same_5ms_bin'] += 1
    return dict(n=len(items), core_over100=sum(x['core_peak_over100'] for x in items),
                median_peak_5ms_Hz=np.median(p, axis=0).tolist(),
                median_core_to_other_peak_ratio=float(np.median([x['core_to_other_peak_ratio'] for x in items])),
                core_recruitment_order=order)


def paired_gif(data, geo):
    starts = [.5, 100.6]
    n = 600  # 3 real seconds at native 5-ms resolution, displayed at 25 fps.
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), gridspec_kw={'height_ratios': [2.1, 1]})
    ims, cursors = [], []
    for col, start in enumerate(starts):
        ims.append(field(axes[0, col], data['field_rates'][round(start*200)], geo))
        axes[0, col].set(xlabel='x (mm)', ylabel='y (mm)')
        trace(axes[1, col], data, start, start+3)
        cursors.append(axes[1, col].axvline(start, color='black', lw=1))
        axes[1, col].set_xlabel('Time (s)')
    axes[1, 0].legend(frameon=False, fontsize=8, ncol=2)
    fig.subplots_adjust(left=.09, right=.87, top=.94, bottom=.09, hspace=.35, wspace=.3)
    fig.colorbar(ims[0], cax=fig.add_axes([.91, .48, .018, .4]), label='E rate (Hz)')
    def update(i):
        for col, start in enumerate(starts):
            k = round(start*200)+i
            ims[col].set_data(data['field_rates'][k].reshape(20, 20))
            t = start+(i+.5)*.005
            cursors[col].set_xdata([t, t])
            axes[0, col].set_title(('Original' if col == 0 else 'Returned')+f' | {t:.4f} s')
        return ims+cursors
    FuncAnimation(fig, update, frames=n, interval=40).save(OUT/'figures/original_vs_return_native_5ms.gif',
        writer=PillowWriter(fps=25), dpi=95)
    plt.close(fig)


def main(make_animation=True):
    OUT.joinpath('figures').mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    row = json.loads((ROOT/'analysis'/f'{NAME}.json').read_text())
    data, geo = load_native()
    before = [e for e in row['native_rhythm']['strict_pre']['brief_events'] if .5 <= e['start_s'] and e['end_s'] <= 8]
    results = {'original_0p5_8s': [event_metrics(data, e) for e in before]}
    for i, e in enumerate(row['absolute_Z_recovery']['episodes'], 1):
        if e['sustained_return_after_absolute_recovery']:
            results[f'return_after_exit{i}'] = [event_metrics(data, v) for v in e['post_absolute_recovery_events']['brief_events']]
    meta = dict(source=str(ROOT/'runs'/NAME), seed=9108405, no_new_simulation=True,
        user_acceptance='Existence and spatial mechanism of return; not all seeds must exit.',
        native_prefix_bitwise=True, counts_conserved=True, geometry_and_core_counts_verified=True,
        temporal_field_bin_ms=5, spatial_grid_mm=1, interpolation=False,
        statistic_unit='individual detected brief event within one trajectory, not independent seeds',
        core_definition='Physical radius1.5mm solid circles; original rate observer uses radius1.75mm dotted circles, counts754/786. Observer unchanged.',
        event_selection='All previously detected20-200ms brief events, unchanged; baseline0.5-8s and all4complete return windows.',
        onset_order_caveat='Population average threshold crossings, sustained10ms, are not causal source localization; smaller cores and large surround have different dilution.',
        physical_low_activity_rule='tauK5s at causal global E rate<=5Hz, otherwise0.5s; no30s timer or Z target release',
        displayed_windows_s=[[.5, 3.5], [100.6, 103.6]],
        summaries={k: summarize(v) for k, v in results.items()}, events=results,
        human_review='PENDING', spatial_mechanism='DESCRIPTIVE_REVIEW_NOT_CAUSAL_ABLATION')
    (OUT/'event_comparison.json').write_text(json.dumps(meta, indent=2)+'\n')
    temporal_zoom(data, row)
    storyboard(data, geo, [('Original: first brief event after 0.5 s', .57),
        ('Original: next event', .81), ('Original: third event', 1.06)], 'original_first_three_events.png')
    storyboard(data, geo, [('Return 1: weak event (no core burst)', 100.82),
        ('Return 2: core recruitment', 101.395), ('Return 3: outside activity then cores', 101.79)],
        'returned_first_three_events.png')
    storyboard(data, geo, [('First full return: first event', 49.30),
        ('Original: first brief event after 0.5 s', .57)], 'first_return_vs_original.png')
    print(json.dumps(meta['summaries'], indent=2), flush=True)
    if make_animation:
        paired_gif(data, geo)
    notes = {
        'return_transition_zoom.png': '同一种子原间期0.5–3.5秒与第二次完整返回100.6–103.6秒并排放大；固定细胞raster、5ms区域率及原Z均来自已有轨迹。右侧标出连续前三个事件，包含最先出现的弱核外事件。',
        'original_first_three_events.png': '原间期0.5秒之后连续前三个短事件的区域率和原生空间帧。每帧5ms、1mm网格、色标0–500Hz；不插值、不平滑，青色实圈为物理core半径1.5mm，虚圈为原区域率观测半径1.75mm。',
        'returned_first_three_events.png': '约30秒低活动结束后连续前三个短事件的区域率和原生空间帧，与原间期使用相同尺度；行号指事件顺序，不是完整返回次数。第一弱事件不删去，第二、第三事件保留核外活动与两核招募的先后。',
        'first_return_vs_original.png': '另展示首次完整返回49.31秒的第一个事件，与原间期第一个事件比较。两个事件均按绝对时间取原生场，不做逐位置对齐或时间拉伸。',
        'original_vs_return_native_5ms.gif': '左侧原间期0.5–3.5秒，右侧第二次完整返回100.6–103.6秒，以5ms原生空间帧逐帧播放。两段各3秒仿真时间，25fps慢放约24秒；相同播放进度不代表事件配对。',
    }
    (OUT/'figures/README.md').write_text('\n'.join(f'### {name}\n\n{text}\n\n**关注点**：区分core起燃、core参与放大及核外传播；本图为候选，待用户目视验收。\n' for name, text in notes.items()))
    print('Finished native zoom and event comparison; animation '+('rendered' if make_animation else 'left unchanged')+'; no simulation launched.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-animation', action='store_true')
    main(not parser.parse_args().skip_animation)
