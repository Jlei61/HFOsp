#!/usr/bin/env python3
"""Retain the complete blue 1-to-2 native path in the single 3D summary."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import plot_topic4_fig5_manual_release_layout_v4 as prior

ROOT = Path(__file__).resolve().parents[1]
BASE = prior.BASE
OUT = BASE / 'layout_v6'
FIG = OUT / 'figures'


def write(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / (name + '.png'), dpi=180, bbox_inches='tight', pad_inches=.16)
    fig.savefig(FIG / (name + '.pdf'), bbox_inches='tight', pad_inches=.16)
    plt.close(fig)


def complete_paths(t, xyz, run, windows):
    definitions = [
        (windows[0]['time'] - .125, windows[1]['time'], prior.old.BLUE,
         '1 → 2: recurrent events / Z depletion'),
        (windows[1]['time'], run['first_trigger_ms'] / 1000, prior.old.AMBER,
         'Entry after 2'),
        (run['first_trigger_ms'] / 1000, run['restore_start_ms'] / 1000,
         prior.old.RED, 'High activity'),
        (run['restore_start_ms'] / 1000, run['release_ms'] / 1000,
         prior.FORCED, 'Forced Z refill'),
        (run['release_ms'] / 1000, windows[3]['time'] + .125,
         prior.RETURN, 'After release'),
    ]
    paths = []
    for start, end, color, name in definitions:
        tt, points = prior.exact_path(t, xyz, start, end)
        paths.append(dict(time=tt, coords=points, color=color, name=name))
    return paths


def plot_summary(ax, t, xyz, paths, run, windows):
    for k, p in enumerate(paths):
        width = .95 if k == 0 else 3.2 if k == 3 else 1.7
        if k == 3:
            ax.plot(*p['coords'].T, c='white', lw=width + 1.5, alpha=.95)
        ax.plot(*p['coords'].T, c=p['color'], lw=width, alpha=.8 if k == 0 else .97)
        if k == 0:
            # Local arrows use adjacent native samples; do not bridge across cycles.
            for time in [2., 4., 6., 8., 9.9]:
                ix = np.flatnonzero(np.abs(p['time'] - time) < .12)
                if len(ix) > 1:
                    j = int(ix[np.argmax(p['coords'][ix, 2])])
                    j = min(j, len(p['time']) - 2)
                    prior.arrow(ax, p['coords'][j], p['coords'][j + 1],
                                p['color'], size=10, lw=.8)
        else:
            prior.traverse_arrows(ax, p['coords'], p['color'], count=4 if k == 3 else 2,
                                  size=17 if k == 3 else 13)
    offsets = {1: (-35, 30), 2: (-28, -22), 3: (24, 16), 4: (18, -26)}
    for w in windows[:4]:
        point = prior.point_at(t, xyz, w['time'])
        ax.scatter(*point, s=34, facecolor='white', edgecolor=w['color'],
                   lw=1.4, depthshade=False)
        prior.label(ax, str(w['number']), point, offsets[w['number']], w['color'])
    for time, text, offset in [
        (run['restore_start_ms'] / 1000, 'Refill starts\n11.18 s', (155, 10)),
        (run['release_ms'] / 1000, 'Z released\n12.18 s', (-10, 100)),
    ]:
        point = prior.point_at(t, xyz, time)
        ax.scatter(*point, marker='s', s=28, c=prior.FORCED, depthshade=False)
        prior.label(ax, text, point, offset, prior.FORCED, size=11, circle=False)
    ax.set(xlim=(.63, 1.03), ylim=(-15, 500), zlim=(-10, 410),
           xticks=[.65, .8, 1.], yticks=[0, 250, 500], zticks=[0, 200, 400])
    prior.style_3d(ax)
    ax.set_title('E  Recurrent events, high-state entry and forced return',
                 fontsize=16, loc='left', weight='bold', pad=17)
    handles = [Line2D([], [], color=p['color'], lw=3 if k == 3 else 1.7,
                      label=p['name']) for k, p in enumerate(paths)]
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(-.03, .975),
              ncol=2, fontsize=10, frameon=False, handlelength=1.7, columnspacing=1.)
    ax.set_anchor('N')


def main():
    source_metadata = json.loads((prior.OUT / 'figure_metadata.json').read_text())
    windows = source_metadata['windows']
    a, run, *_ = prior.old.previous.source.load_main()
    t, xyz = prior.native_coordinates(a)
    paths = complete_paths(t, xyz, run, windows)

    # These are data/display checks, not an additional dynamics experiment.
    checks = {}
    for k, p in enumerate(paths):
        ix = (t > p['time'][0]) & (t < p['time'][-1])
        checks[f'path{k}_all_native_5ms_samples_retained'] = bool(
            np.array_equal(p['time'][1:-1], t[ix]) and
            np.array_equal(p['coords'][1:-1], xyz[ix]))
        if k:
            checks[f'boundary{k}_time_and_state_continuous'] = bool(
                p['time'][0] == paths[k-1]['time'][-1] and
                np.array_equal(p['coords'][0], paths[k-1]['coords'][-1]))
    checks['blue_to_orange_boundary_is_exactly_marker2'] = bool(
        paths[0]['time'][-1] == windows[1]['time'] and
        np.array_equal(paths[0]['coords'][-1],
                       prior.point_at(t, xyz, windows[1]['time'])))
    original = np.load(prior.OUT / 'return_path_arrays.npz')
    checks['high_refill_release_paths_bitwise_unchanged_from_v4'] = all(
        np.array_equal(paths[k+1]['time'], original[f'path{k}_time_s']) and
        np.array_equal(paths[k+1]['coords'], original[f'path{k}_Z_I_E'])
        for k in range(1, 4))
    all_coords = np.concatenate([p['coords'] for p in paths])
    checks['all_displayed_samples_within_axis_limits'] = bool(
        np.all(all_coords >= [.63, -15, -10]) and
        np.all(all_coords <= [1.03, 500, 410]))
    assert all(checks.values()), checks

    fig = plt.figure(figsize=(11.5, 10.5))
    ax = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(left=.065, right=.91, top=.93, bottom=.055)
    plot_summary(ax, t, xyz, paths, run, windows)
    assert len(fig.axes) == 1
    save(fig, 'single_native_state_trajectory')

    fig = plt.figure(figsize=(23, 14))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], left=.06, right=.952,
                         top=.925, bottom=.065, wspace=.2)
    left = prior.old.left_panels(fig, gs[0], a, run, windows)
    ax = fig.add_subplot(gs[1], projection='3d')
    plot_summary(ax, t, xyz, paths, run, windows)
    assert len([axis for axis in fig.axes if axis.name == '3d']) == 1
    fig.suptitle('Slow inhibitory depletion and spatial recruitment in a dual-core SNN',
                 fontsize=19, y=.99)
    save(fig, 'fig5_manual_core_release_layout_v6')

    arrays = {}
    for k, p in enumerate(paths):
        arrays[f'path{k}_time_s'] = p['time']
        arrays[f'path{k}_Z_I_E'] = p['coords']
    np.savez_compressed(OUT / 'trajectory_arrays.npz', **arrays)
    values = [dict(number=w['number'], time_s=w['time'],
                   Z_I_E=prior.point_at(t, xyz, w['time']).tolist()) for w in windows[:4]]
    write('figure_metadata.json', dict(source=str(prior.OUT), simulation_rerun=False,
          windows=windows, left=left, local_event_panels=False, phase_axis_count=1,
          landmark_values=values, coordinates=['Mean E-target Z', 'All-I rate (Hz)', 'All-E rate (Hz)'],
          rate_bin_ms=5, smoothing_sigma_ms=5, view=list(prior.VIEW),
          paths=[dict(name=p['name'], color=p['color'], time_range_s=p['time'][[0,-1]].tolist(),
                      sample_count=len(p['time'])) for p in paths],
          intermediate_repeated_events_omitted_for_clarity=False,
          interpretation_1_to_2='Repeated self-limited events with net Z depletion; all intervening native 5-ms samples are retained. Blue ends exactly at marker 2, where orange begins.',
          human_acceptance='PENDING_USER_REVIEW'))
    scripts = [Path(__file__), Path(prior.__file__), Path(prior.old.__file__),
               Path(prior.old.previous.__file__), Path(prior.old.previous.source.__file__)]
    write('producer_manifest.json', {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in scripts})
    (FIG / 'README.md').write_text('''### single_native_state_trajectory.png / .pdf
补齐①至②之间的全部连续轨迹，统一使用与①相同的蓝色；蓝色从①前125毫秒开始，在②约10.3075秒处与橙色精确相接。保留原5毫秒分箱及5毫秒高斯平滑后的所有轨迹点，以较细线宽呈现反复事件，③至外部补回再至④的轨迹与v4逐点一致。
**关注点**：①到②是多次事件伴随Z净耗竭的连续过程，不再省略中间事件，也未用直线替代；仍只保留一张三维图，待用户目视审阅。

### fig5_manual_core_release_layout_v6.png / .pdf
Figure 5左侧沿用原连续读出、raster、Z曲线和空间取样窗口，右侧更新为包含完整蓝色①至②轨迹的单一三维图。未改变仿真或重新选择编号对应时刻。
**关注点**：检查蓝色到橙色在②处的连续性及紫色外部补回路径的可读性；本版为候选，不自动替换正式paper-ready图。
''', encoding='utf-8')
    write('artifact_qa.json', dict(numeric_checks=checks, numeric_status='PASS',
          agent_visual_review='PENDING', human_acceptance='PENDING_USER_REVIEW'))
    write('delivery_status.json', dict(status='RENDERED_PENDING_AGENT_VISUAL_REVIEW',
          simulation_rerun=False, local_event_panels=False, human_acceptance='PENDING_USER_REVIEW'))
    print(json.dumps(dict(out=str(OUT), paths=[dict(name=p['name'], samples=len(p['time']),
          interval_s=p['time'][[0,-1]].tolist()) for p in paths], checks=checks)))


if __name__ == '__main__':
    main()
