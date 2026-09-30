#!/usr/bin/env python3
"""Single native 3D summary; no local event panels in this or subsequent layout use."""
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
OUT = BASE / 'layout_v5'
FIG = OUT / 'figures'


def write(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / (name + '.png'), dpi=180, bbox_inches='tight', pad_inches=.16)
    fig.savefig(FIG / (name + '.pdf'), bbox_inches='tight', pad_inches=.16)
    plt.close(fig)


def plot_summary(ax, t, xyz, paths, run, windows):
    prior.plot_cycle(ax, t, xyz, paths, run, windows)
    ax.set_title('E  Recurrent events, high-state entry and forced return',
                 fontsize=16, loc='left', weight='bold', pad=17)
    ax.get_legend().remove()
    handles = [Line2D([], [], color=prior.old.BLUE, lw=1.7,
                      label='1 → 2: recurrent events / Z depletion'),
               Line2D([], [], color=prior.old.AMBER, lw=1.7, label='Entry'),
               Line2D([], [], color=prior.old.RED, lw=1.7, label='High activity'),
               Line2D([], [], color=prior.FORCED, lw=3, label='Forced Z refill'),
               Line2D([], [], color=prior.RETURN, lw=1.7, label='After release')]
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(-.03, .975),
              ncol=2, fontsize=10, frameon=False, handlelength=1.7, columnspacing=1.)
    ax.set_anchor('N')


def main():
    # Consume the already displayed v4 sample times exactly; do not reselect windows.
    source_metadata = json.loads((prior.OUT / 'figure_metadata.json').read_text())
    windows = source_metadata['windows']
    a, run, *_ = prior.old.previous.source.load_main()
    t, xyz = prior.native_coordinates(a)
    paths = prior.cycle_paths(t, xyz, run, windows)

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
    save(fig, 'fig5_manual_core_release_layout_v5')

    values = [dict(number=w['number'], time_s=w['time'],
                   Z_I_E=prior.point_at(t, xyz, w['time']).tolist()) for w in windows[:4]]
    original = np.load(prior.OUT / 'return_path_arrays.npz')
    exact = all(np.array_equal(p['time'], original[f'path{k}_time_s']) and
                np.array_equal(p['coords'], original[f'path{k}_Z_I_E']) for k, p in enumerate(paths))
    assert exact
    write('figure_metadata.json', dict(source=str(prior.OUT), simulation_rerun=False,
          windows=windows, left=left, local_event_panels=False, phase_axis_count=1,
          return_path_bitwise_equal_to_v4=exact, landmark_values=values,
          interpretation_1_to_2='Repeated self-limited events while mean E-target Z declines from about 0.953 to 0.754. Points are distinct sampled times, not adjacent states of one event or established bifurcation locations.',
          intermediate_repeated_events_omitted_for_clarity=True,
          human_acceptance='PENDING_USER_REVIEW'))
    scripts = [Path(__file__), Path(prior.__file__), Path(prior.old.__file__),
               Path(prior.old.previous.__file__), Path(prior.old.previous.source.__file__)]
    write('producer_manifest.json', {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in scripts})
    (FIG / 'README.md').write_text('''### single_native_state_trajectory.png / .pdf
用户本轮确认：此后该相图仅保留单一三维总图，不再附下方两个局部事件图。①约1.13秒、平均Z约0.953，②约10.31秒、平均Z约0.754；其间是反复自限事件伴随抑制资源净耗竭，图例已注明这一含义。
**关注点**：为保持可读性，①至②之间的大多数重复事件未在本相图展开；编号是同一连续仿真的取样时刻，不是同一次事件的相邻阶段或已经确定的分岔点。③到外部补回再到④的连续轨迹与v4逐点一致。

### fig5_manual_core_release_layout_v5.png / .pdf
整体Figure5沿用已对齐的A/B/C时间轴与D空间快照，右侧仅保留一个三维总图。删除两个局部图，没有修改仿真、读出、取样时刻或③至④的完整返回轨迹。
**关注点**：本版落实单一相图的显示约定；仍为待用户审阅的候选，不自动替换正式paper-ready图。
''', encoding='utf-8')
    write('delivery_status.json', dict(status='RENDERED_PENDING_AGENT_VISUAL_REVIEW',
          return_path_qa='PASS', local_event_panels=False, human_acceptance='PENDING_USER_REVIEW'))
    print(json.dumps({'out':str(OUT),'landmarks':values,'return_path_unchanged':exact}))


if __name__ == '__main__':
    main()
