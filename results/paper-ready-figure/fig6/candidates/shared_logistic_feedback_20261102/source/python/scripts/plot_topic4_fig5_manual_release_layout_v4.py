#!/usr/bin/env python3
"""Full high-state-to-return path, permuted native 3D axes, local event zooms."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d
import plot_topic4_fig5_manual_release_layout_v3 as old

ROOT = Path(__file__).resolve().parents[1]
BASE = old.BASE
OUT = BASE / 'layout_v4'
FIG = OUT / 'figures'
FORCED = '#953cc0'
RETURN = '#18866b'
VIEW = (22, -125)


def write(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / (name + '.png'), dpi=180, bbox_inches='tight', pad_inches=.16, facecolor='white')
    fig.savefig(FIG / (name + '.pdf'), bbox_inches='tight', pad_inches=.16, facecolor='white')
    plt.close(fig)


def native_coordinates(a, bins=50, sigma=1):
    # Native dt = 0.1 ms. The coordinate permutation is x=Z, y=I, z=E.
    e = a['rate_e_hz'].reshape(-1, bins).mean(1)
    i = a['rate_i_hz'].reshape(-1, bins).mean(1)
    t = (np.arange(len(e)) + .5) * bins * .0001
    e = gaussian_filter1d(e, sigma)
    i = gaussian_filter1d(i, sigma)
    z = np.interp(t, a['z_time_ms'] / 1000, a['z_stats'][:, 0])
    return t, np.column_stack([z, i, e])


def point_at(t, coords, time):
    return np.array([np.interp(time, t, coords[:, j]) for j in range(3)])


def exact_path(t, coords, start, end):
    ix = np.flatnonzero((t > start) & (t < end))
    # Interpolated boundary points ensure adjacent protocol stages meet exactly.
    tt = np.r_[start, t[ix], end]
    xyz = np.vstack([point_at(t, coords, start), coords[ix], point_at(t, coords, end)])
    return tt, xyz


def arrow(ax, start, end, color, size=13, lw=1.1):
    artist = old.previous.source.TrajectoryArrow3D(start, end, color)
    artist.set_mutation_scale(size)
    artist.set_linewidth(lw)
    artist.set_zorder(25)
    ax.add_artist(artist)


def traverse_arrows(ax, points, color, count=3, size=13):
    normalized = points / [.4, 500, 400]
    distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(normalized, axis=0), axis=1))]
    if distance[-1] <= 1e-10:
        return
    for f in np.linspace(.16, .85, count):
        k = int(np.searchsorted(distance, f * distance[-1]))
        j = min(int(np.searchsorted(distance, distance[k] + .065)), len(points) - 1)
        if j > k:
            arrow(ax, points[k], points[j], color, size=size)


def label(ax, text, point, offset, color, size=14, circle=True):
    item = old.Label3D(text, point, offset, color)
    item.set_fontsize(size)
    if not circle:
        item.set_bbox(dict(boxstyle='round,pad=.2', fc='white', ec=color, lw=.8, alpha=.94))
    ax.add_artist(item)


def style_3d(ax, view=VIEW, zoom=1, local=False):
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_proj_type('persp', focal_length=1.)
    ax.set_box_aspect((1.2, 1., 1.05), zoom=zoom)
    size = 13 if local else 15
    ax.set_xlabel('Mean Z', fontsize=size, labelpad=7 if not local else 1)
    ax.set_ylabel(r'I rate, $r_I$ (Hz)', fontsize=size, labelpad=9 if not local else 2)
    ax.set_zlabel(r'E rate, $r_E$ (Hz)', fontsize=size, labelpad=9 if not local else 2)
    ax.tick_params(labelsize=11 if local else 13, pad=1)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((.94, .96, .985, .62))
        axis.pane.set_edgecolor((.55, .60, .67, .5))
        axis._axinfo['grid'].update(color=(.60, .65, .72, .36), linewidth=.65)


def cycle_paths(t, coords, run, win):
    first = run['first_trigger_ms'] / 1000
    start = run['restore_start_ms'] / 1000
    release = run['release_ms'] / 1000
    end = win[3]['time'] + .125
    definitions = [(9.80, first, old.AMBER, 'Entry'),
                   (first, start, old.RED, 'High activity'),
                   (start, release, FORCED, 'Forced Z refill'),
                   (release, end, RETURN, 'Native activity resumes')]
    paths = []
    for lo, hi, color, name in definitions:
        tt, points = exact_path(t, coords, lo, hi)
        paths.append(dict(time=tt, coords=points, color=color, name=name))
    return paths


def plot_cycle(ax, t, coords, paths, run, win, view=VIEW, title=True):
    # One early event gives context; all data from entry through event 4 are retained.
    for lo, hi, alpha in [(win[0]['time'] - .125, win[0]['time'] + .125, .92)]:
        tt, points = exact_path(t, coords, lo, hi)
        ax.plot(*points.T, c=old.BLUE, lw=1.05, alpha=alpha)
        if alpha > .8:
            traverse_arrows(ax, points, old.BLUE, count=2)
    for p in paths:
        width = 3.2 if p['name'] == 'Forced Z refill' else 1.7
        # White underlay keeps the intervention visible where it crosses other paths.
        if p['name'] == 'Forced Z refill':
            ax.plot(*p['coords'].T, c='white', lw=width + 1.5, alpha=.95)
        ax.plot(*p['coords'].T, c=p['color'], lw=width, alpha=.97)
        traverse_arrows(ax, p['coords'], p['color'], count=4 if p['name'] == 'Forced Z refill' else 2,
                        size=17 if p['name'] == 'Forced Z refill' else 13)
    offsets = {1: (-35, 30), 2: (-28, -22), 3: (24, 16), 4: (18, -26)}
    for w in win[:4]:
        point = point_at(t, coords, w['time'])
        ax.scatter(*point, s=34, facecolor='white', edgecolor=w['color'], lw=1.4, depthshade=False)
        label(ax, str(w['number']), point, offsets[w['number']], w['color'])
    for time, text, offset in [(run['restore_start_ms']/1000, 'Refill starts\n11.18 s', (155, 10)),
                               (run['release_ms']/1000, 'Z released\n12.18 s', (-10, 100))]:
        point = point_at(t, coords, time)
        ax.scatter(*point, marker='s', s=28, c=FORCED, depthshade=False)
        label(ax, text, point, offset, FORCED, size=11, circle=False)
    ax.set(xlim=(.63, 1.03), ylim=(-15, 500), zlim=(-10, 410),
           xticks=[.65, .8, 1.], yticks=[0, 250, 500], zticks=[0, 200, 400])
    style_3d(ax, view=view)
    if title:
        ax.set_title('E  High-state entry and forced return: 3 → 4',
                     loc='left', fontsize=16, weight='bold', pad=17)
        handles = [Line2D([], [], c=old.BLUE, lw=1.7, label='Self-limited event'),
                   Line2D([], [], c=old.AMBER, lw=1.7, label='Entry'),
                   Line2D([], [], c=old.RED, lw=1.7, label='High activity'),
                   Line2D([], [], c=FORCED, lw=3., label='Forced Z refill'),
                   Line2D([], [], c=RETURN, lw=1.7, label='After release')]
        ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(-.03, .975),
                  ncol=2, fontsize=10, frameon=False, handlelength=1.7, columnspacing=1.)


def plot_event(ax, time, coords, w):
    quiet = (coords[:, 2] < .2) & (coords[:, 1] < .5)
    before = np.flatnonzero((time < w['time']) & (time > w['time']-.25) & quiet)
    after = np.flatnonzero((time > w['time']) & (time < w['time']+.3) & quiet)
    assert len(before) and len(after), 'The displayed event needs quiet boundaries.'
    tt, points = exact_path(time, coords, float(time[before[-1]]), float(time[after[0]]))
    ax.plot(*points.T, c=w['color'], lw=1.8)
    # Local coordinates are not shifted; only the Z axis has a narrower range.
    span = max(np.ptp(points[:, 0]), .002)
    low = float(points[:, 0].min() - span * .14)
    high = float(points[:, 0].max() + span * .14)
    ax.set(xlim=(low, high), ylim=(-5, 180), zlim=(-2, 80),
           xticks=[round(low+.15*(high-low),3), round(high-.15*(high-low),3)],
           yticks=[0, 90, 180], zticks=[0, 40, 80])
    normalized = (points - points.min(0)) / np.maximum(np.ptp(points, axis=0), 1e-9)
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(normalized, axis=0), axis=1))]
    for f in [.22, .55, .82]:
        j = int(np.searchsorted(d, f*d[-1]))
        end = min(j+3, len(points)-1)
        arrow(ax, points[j], points[end], w['color'], size=12)
    ax.scatter(*points[0], s=17, c='white', edgecolor=w['color'], depthshade=False)
    ax.scatter(*points[-1], s=17, c=w['color'], depthshade=False)
    style_3d(ax, view=VIEW, zoom=.88, local=True)
    ax.set_title(f'Event {w["number"]} · local Z scale\n1-ms rates, 1-ms smoothing', fontsize=11, pad=9)
    return dict(number=w['number'], window_s=[float(tt[0]),float(tt[-1])],
                samples=len(tt), z_limits=[low,high], rate_bin_ms=1, smoothing_sigma_ms=1,
                display_window_selection='Nearest surrounding quiet samples within -250/+300 ms of the snapshot; E<0.2 Hz and I<0.5 Hz. Display boundaries only, not a new event detector.')


def right_panels(fig, spec, a, run, win):
    t, xyz = native_coordinates(a)
    paths = cycle_paths(t, xyz, run, win)
    gs = spec.subgridspec(2, 1, height_ratios=[1.52,1], hspace=.07)
    ax = fig.add_subplot(gs[0], projection='3d')
    plot_cycle(ax, t, xyz, paths, run, win)
    lower = gs[1].subgridspec(1,2,wspace=.14)
    fine_t, fine_xyz = native_coordinates(a,bins=10,sigma=1)
    event_meta=[]
    for j,w in enumerate([win[0],win[3]]):
        ax=fig.add_subplot(lower[j],projection='3d')
        event_meta.append(plot_event(ax,fine_t,fine_xyz,w))
    meta=dict(coordinates=['Mean E-target Z','All-I rate (Hz)','All-E rate (Hz)'],
              continuous_interval_s=[float(paths[0]['time'][0]),float(paths[-1]['time'][-1])],
              complete_3_to_4_path=True, simplification=False,
              main_rate_bin_ms=5,main_smoothing_sigma_ms=5,local_events=event_meta,
              focus='First high-state entry, external restoration, and return to event 4; later state 5 remains in A-D.',
              previous_gap_s=[12.18,12.8175], view=list(VIEW))
    return meta,paths


def preview(a,run,win):
    t,xyz=native_coordinates(a);paths=cycle_paths(t,xyz,run,win)
    fig=plt.figure(figsize=(16,13))
    for k,view in enumerate([(27,-52),(22,-125),(28,38),(20,-75)]):
        ax=fig.add_subplot(2,2,k+1,projection='3d')
        plot_cycle(ax,t,xyz,paths,run,win,view=view,title=False)
        ax.set_title(f'x=Z, y=I, vertical=E | elev={view[0]}, azim={view[1]}')
    fig.savefig('/tmp/fig5_return_view_choices.png',dpi=130,bbox_inches='tight')
    plt.close(fig)


def main():
    a,run,t,e,i,z=old.previous.source.load_main()
    win,align=old.windows(a,run,t,e)
    if '--preview' in sys.argv:
        preview(a,run,win);return
    fig=plt.figure(figsize=(23,14))
    gs=fig.add_gridspec(1,2,width_ratios=[1.5,1],left=.06,right=.952,
                        top=.925,bottom=.065,wspace=.2)
    left=old.left_panels(fig,gs[0],a,run,win)
    phase,paths=right_panels(fig,gs[1],a,run,win)
    fig.suptitle('Slow inhibitory depletion and spatial recruitment in a dual-core SNN',fontsize=19,y=.99)
    save(fig,'fig5_manual_core_release_layout_v4')
    fig=plt.figure(figsize=(12,13))
    gs=fig.add_gridspec(1,1,left=.065,right=.915,top=.94,bottom=.04)
    right_panels(fig,gs[0],a,run,win)
    save(fig,'continuous_high_state_to_return_3d')
    arrays={}
    for k,p in enumerate(paths):
        arrays[f'path{k}_time_s']=p['time'];arrays[f'path{k}_Z_I_E']=p['coords']
    np.savez_compressed(OUT/'return_path_arrays.npz',**arrays)
    write('figure_metadata.json',dict(source=str(BASE),simulation_rerun=False,
          windows=win,snapshot4_alignment=align,left=left,phase=phase,
          human_acceptance='PENDING_USER_REVIEW'))
    write('producer_manifest.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
          for p in [Path(__file__),Path(old.__file__),Path(old.previous.__file__),Path(old.previous.source.__file__)]})
    (FIG/'README.md').write_text('''### fig5_manual_core_release_layout_v4.png / .pdf
左侧沿用v3的连续读出/raster/Z与空间快照及配色。右侧交换轴序为水平Z、纵深I率、竖直E率，以透视视角突出③高态到外部补回再到④的连续返回过程；粗紫线和箭头单独标明外部Z补回。
**关注点**：9.80–13.086秒的全部5毫秒轨迹点均保留，并在协议边界插值使线段精确相接；修复上一版12.18–12.8175秒遗漏，不再用离散事件片段替代完整返回。右图集中展示第一次进入及返回，⑤仍在左侧。

### continuous_high_state_to_return_3d.png / .pdf
三维相轨迹放大版，下方单独展示①和④各一次事件，以1毫秒分箱/1毫秒高斯平滑读取原始spikes，并局部放大Z坐标以显示快活动的往返。两小图E/I范围相同，Z保留各自实际数值，起点为空心、终点为实心。
**关注点**：局部Z轴放大和时间分辨率已在图上说明，不移动或人为闭合轨迹；事件的往返曲线不等同于自治极限环或分岔证明。本次只改读图方式，不重跑或改动动力学。
''',encoding='utf-8')
    write('delivery_status.json',dict(status='RENDERED_PENDING_AGENT_VISUAL_REVIEW',simulation_rerun=False,
          human_acceptance='PENDING_USER_REVIEW'))
    print(json.dumps({'out':str(OUT),'phase':phase}))


if __name__=='__main__':main()
