#!/usr/bin/env python3
"""Native E-rate / applied inhibitory current / Z trajectory, same run and times."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import plot_topic4_fig5_manual_release_layout_v7 as previous

prior = previous.prior
ROOT = previous.ROOT
OUT = previous.previous.BASE / 'layout_v8'
FIG = OUT / 'figures'


def write(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f'{name}.png', dpi=180, bbox_inches='tight', pad_inches=.16)
    fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight', pad_inches=.16)
    plt.close(fig)


def current_coordinates(a):
    # Preserve the displayed E-rate and Z values exactly. The recorded current is
    # a population mean of neuron-wise products, NOT mean(Z) * mean(GABA).
    t, old_coords = prior.native_coordinates(a)
    current_time = a['z_time_ms'] / 1000
    applied = a['currents_5ms'][:, 2]
    # Same symmetric 5-ms Gaussian sigma as displayed firing rates; interpolate
    # from actual 5-ms snapshot times to the existing 5-ms rate-bin centers.
    smoothed = gaussian_filter1d(applied, 1)
    h = np.interp(t, current_time, smoothed)
    xyz = np.column_stack([old_coords[:, 0], h, old_coords[:, 2]])
    return t, xyz, old_coords


def plot_summary(ax, t, xyz, paths, run, windows, panel_letter=False):
    color = previous.plot_summary(ax, t, xyz, paths, run, windows,
                                   panel_letter=panel_letter)
    hmax = max(float(p['coords'][:, 1].max()) for p in paths)
    upper = float(np.ceil(hmax / 100) * 100)
    ax.set_ylim(-.025 * upper, 1.04 * upper)
    ax.set_yticks([0, upper / 2, upper])
    ax.set_ylabel(r'Applied inhibition, $H_E$ (mV equiv.)', fontsize=14, labelpad=12)
    ax.set_title(('E  ' if panel_letter else '') + 'Firing and inhibition during Z depletion and refill',
                 fontsize=16, loc='left', weight='bold', pad=17)
    return color


def main():
    metadata = json.loads((previous.OUT / 'figure_metadata.json').read_text())
    windows = metadata['windows']
    a, run, *_ = prior.old.previous.source.load_main()
    t, xyz, old_xyz = current_coordinates(a)
    paths = previous.previous.complete_paths(t, xyz, run, windows)
    saved = np.load(previous.previous.OUT / 'trajectory_arrays.npz')
    checks = dict(
        E_rate_and_Z_bitwise_unchanged=bool(np.array_equal(xyz[:, [0, 2]], old_xyz[:, [0, 2]])),
        current_snapshot_times_are_5ms=bool(np.allclose(np.diff(a['z_time_ms']), 5)),
        applied_inhibition_nonnegative_and_bounded_by_raw=bool(
            np.all(a['currents_5ms'][:, 2] >= 0) and
            np.all(a['currents_5ms'][:, 2] <= a['currents_5ms'][:, 1] + 1e-10)),
        all_original_path_times_retained=all(np.array_equal(p['time'], saved[f'path{k}_time_s'])
                                            for k, p in enumerate(paths)),
        all_original_path_E_and_Z_retained=all(np.array_equal(p['coords'][:, [0, 2]],
            saved[f'path{k}_Z_I_E'][:, [0, 2]]) for k, p in enumerate(paths)),
        path_boundaries_continuous=all(np.array_equal(p['coords'][-1], q['coords'][0]) and
            p['time'][-1] == q['time'][0] for p, q in zip(paths[:-1], paths[1:])),
    )
    assert all(checks.values()), checks

    fig = plt.figure(figsize=(12, 10.5))
    ax = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(left=.065, right=.90, top=.93, bottom=.055)
    color = plot_summary(ax, t, xyz, paths, run, windows)
    assert len(fig.axes) == 1 and len(ax.child_axes) == 1
    all_points = np.concatenate([p['coords'] for p in paths])
    limits = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
    checks['all_points_within_axis_limits'] = bool(np.all(all_points >= limits[:, 0]) and
                                                  np.all(all_points <= limits[:, 1]))
    assert checks['all_points_within_axis_limits']
    save(fig, 'single_native_inhibition_state_trajectory')

    fig = plt.figure(figsize=(23, 14))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], left=.06, right=.952,
                         top=.925, bottom=.065, wspace=.2)
    left = prior.old.left_panels(fig, gs[0], a, run, windows)
    ax = fig.add_subplot(gs[1], projection='3d')
    plot_summary(ax, t, xyz, paths, run, windows, panel_letter=True)
    fig.suptitle('Slow inhibitory depletion and spatial recruitment in a dual-core SNN',
                 fontsize=19, y=.99)
    save(fig, 'fig5_manual_core_release_layout_v8')

    arrays = dict(time_s=t, coordinates_Z_H_E=xyz,
                  current_snapshot_time_s=a['z_time_ms'] / 1000,
                  applied_current_unsmoothed=a['currents_5ms'][:, 2])
    for k, p in enumerate(paths):
        arrays[f'path{k}_time_s'] = p['time']
        arrays[f'path{k}_Z_H_E'] = p['coords']
    np.savez_compressed(OUT / 'trajectory_arrays.npz', **arrays)
    write('figure_metadata.json', dict(source=str(previous.OUT), simulation_rerun=False,
        source_run=str(previous.previous.BASE / 'runs/continuous_refill_release.npz'),
        windows=windows, left=left, local_event_panels=False, phase_axis_count=1,
        coordinates=['Mean E-target Z', 'Mean applied inhibitory current to E, H_E (mV equivalent)',
                     'All-E firing rate (Hz)'],
        current_definition='H_E(t) = mean_{j in E}[Z_j(t) * I_GABA,j(t)]',
        current_source='currents_5ms[:, 2], recorded before the membrane update',
        current_producer='scripts/run_topic4_fig5_manual_release.py: currents.append',
        current_units='Voltage-equivalent current in the native current-based LIF update; not measured amperes.',
        rate_bin_ms=5, rate_smoothing_sigma_ms=5, current_snapshot_interval_ms=5,
        current_smoothing_sigma_ms=5,
        current_alignment='Linear interpolation from actual snapshots at 0,5,... ms to rate-bin centers 2.5,7.5,... ms; no chosen phase shift.',
        color_encoding=color, view=list(prior.VIEW), limits_Z_H_E=limits.tolist(),
        landmark_values=[dict(number=w['number'], time_s=w['time'],
            Z_H_E=prior.point_at(t, xyz, w['time']).tolist()) for w in windows[:4]],
        interpretation='Observed rate-current-resource projection of the unchanged native SNN; open event loops do not establish an autonomous limit cycle or bifurcation.',
        human_acceptance='PENDING_USER_REVIEW'))
    scripts = [Path(__file__), Path(previous.__file__), Path(previous.previous.__file__),
               Path(prior.__file__), Path(prior.old.__file__), Path(prior.old.previous.__file__),
               Path(prior.old.previous.source.__file__), ROOT / 'scripts/run_topic4_fig5_manual_release.py']
    write('producer_manifest.json', {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in scripts})
    (FIG / 'README.md').write_text('''### single_native_inhibition_state_trajectory.png / .pdf
同一条手放双核SNN的单张三维轨迹：水平轴为平均Z，纵深轴替换成E神经元实际承受的平均抑制电流H_E，竖直轴仍为全E放电率。H_E直接读取逐神经元Z与GABA电流乘积的群体均值，单位为模型膜方程中的mV等效电流；不是平均Z乘平均GABA。
保留原5毫秒分箱放电率、5毫秒平滑、完整轨迹和①–④时刻；电流原始记录每5毫秒一次，经同尺度平滑后按真实时间插值，未人为移动相位。颜色继续按实际时间连续渐变，补回与释放标注保留。
**关注点**：观察上升和回落时相同放电率是否对应不同抑制电流，以及这些往返如何随着Z改变；轨迹回环本身不能证明自主极限环或分岔。

### fig5_manual_core_release_layout_v8.png / .pdf
左侧连续读出、raster、Z曲线和空间窗口沿用上一版，右侧采用上述放电率—实际抑制电流—Z轨迹。此次只替换观察坐标，没有新增状态变量或重跑模型。
**关注点**：比较原E/I率投影与当前电流投影的可读性；本候选仍待用户目视审阅，不替换正式paper-ready图。
''', encoding='utf-8')
    write('artifact_qa.json', dict(numeric_checks=checks, numeric_status='PASS',
          agent_visual_review='PENDING', human_acceptance='PENDING_USER_REVIEW'))
    write('delivery_status.json', dict(status='RENDERED_PENDING_AGENT_VISUAL_REVIEW',
          simulation_rerun=False, human_acceptance='PENDING_USER_REVIEW'))
    print(json.dumps(dict(out=str(OUT), checks=checks, axis_limits=limits.tolist())))


if __name__ == '__main__':
    main()
