#!/usr/bin/env python3
"""Spatially resolved verified equilibria and a separately labelled SNN comparison."""
from analyze_topic4_prescribed_z_phase import OUT, REFERENCE, source, field_at, read, write, MixedTimescaleSystem
from plot_topic4_prescribed_z_phase import save
from topic4_spatial_boundary_common import ROOT
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def main():
    s = MixedTimescaleSystem(quadrature=33)
    m = s.m
    native, ext, fields = source()
    branches = np.load(OUT / 'branches.npz')
    spectra = read(OUT / 'spectrum_status.json')['rows']
    centers = np.asarray(read(ROOT / 'config/topic4_rate_model_dynamics_validation_v1.json')['candidate']['node_field']['centers_mm'])
    selected = [('low', 0), ('low', 2000), ('high', 10500)]
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), layout='constrained')
    rows = []

    def show(ax, values, title, **kwargs):
        im = ax.imshow(np.asarray(values).reshape(10, 10), origin='lower', extent=[0, 20, 0, 20], interpolation='nearest', **kwargs)
        ax.scatter(centers[:, 0], centers[:, 1], c='white', edgecolors='black', marker='o', s=32, linewidths=.8)
        for label, xy in zip(('A', 'B'), centers):
            ax.annotate(label, xy, xytext=(5, -12), textcoords='offset points', color='white', fontsize=9,
                        bbox={'facecolor': 'black', 'alpha': .55, 'pad': 1, 'edgecolor': 'none'})
        ax.set(title=title, xlabel='x (mm)', ylabel='y (mm)', xticks=[0, 10, 20], yticks=[0, 10, 20])
        return im

    for col, (branch, tm) in enumerate(selected):
        r = branches[f'{branch}_{tm}']; z = field_at(fields, tm)
        err = float(np.max(abs(s.F(r, z))))
        assert err < 1e-8
        spec = next(v for v in spectra if v['branch'] == branch and v['time_ms'] == tm)
        assert spec['root_count']['status'] == 'PASS'
        unstable = spec['root_count']['unstable_roots']
        label = 'Stable' if unstable == 0 else f'Unstable: {unstable} modes'
        e, i = r[:100] * 1000, r[100:] * 1000
        em = float(np.average(e, weights=m.count_e)); im = float(np.average(i, weights=m.count_i))
        iz = show(axes[0, col], z, f'Z from {tm / 1000:g} s | {label}', cmap='viridis', vmin=0, vmax=1)
        for row, v, mean, pop in ((1, e, em, 'E'), (2, i, im, 'I')):
            ir = show(axes[row, col], np.maximum(v, 1e-3), f'{pop} equilibrium | mean {mean:.3g} Hz',
                      cmap='magma', norm=LogNorm(vmin=1e-3, vmax=1000))
        rows.append({'Z_path_time_ms': tm, 'branch': branch, 'unstable_modes': unstable,
                     'fixed_point_residual_per_ms': err, 'E_mean_hz': em, 'I_mean_hz': im,
                     'E_range_hz': [float(e.min()), float(e.max())],
                     'I_range_hz': [float(i.min()), float(i.max())],
                     'E_neuron_fraction_in_cells_above_50Hz': float(np.average(e > 50, weights=m.count_e))})
    fig.colorbar(iz, ax=axes[0, :], label='E-target Z', shrink=.8)
    fig.colorbar(ir, ax=axes[1:, :], label='Equilibrium rate (Hz; shared log scale)', ticks=[.001, .01, .1, 1, 10, 100, 1000], shrink=.8)
    fig.suptitle('Spatial structure of full rate-model equilibria\nFrozen native Z fields; constant reference input; A/B mark core centres', fontsize=14)
    save(fig, 'fixed_point_spatial_activity',
         '三列分别是原Z路径0、2、10.5秒空间场下实际求出的完整rate固定点，上排Z、中排E率、下排I率；A/B仅标既有双核中心。稳定性来自完整延迟映射谱，E/I六张活动图使用共同对数色标，低于0.001 Hz仅在显示时取色标下限。',
         '这些是平衡活动的空间分布，不是传播顺序或实际轨迹。低率颜色差别仍可能全部低于1 Hz；10.5秒高率固定点不稳定，不能称为最终吸引子。8.8秒未可靠求解，未绘制。')

    # Rebin native E counts to the same 10x10 spatial cells without averaging
    # individual cell rates or silently changing the statistical unit.
    pair_counts = np.bincount(ext['cell_e'] * 400 + native['cell_e'], minlength=40000).reshape(100, 400)
    assert np.all((pair_counts > 0).sum(axis=0) == 1)
    aggregate = (pair_counts > 0).astype(float)
    assert np.array_equal(aggregate @ native['cell_e_counts'], m.count_e)
    observed = []
    for lo, hi in ((10495, 10505), (10450, 10550)):
        counts = aggregate @ native['field_e_count_1ms'][lo:hi].sum(axis=0).astype(float)
        rate = counts / m.count_e * 1000 / (hi - lo)
        raw_mean = float(native['rate_e_hz'][lo * 10:hi * 10].mean())
        assert np.isclose(np.average(rate, weights=m.count_e), raw_mean, rtol=1e-6, atol=1e-5)
        observed.append((rate, lo, hi))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), layout='constrained')
    eq = branches['high_10500'][:100] * 1000
    values = [eq] + [r[0] for r in observed]
    titles = ['Rate fixed point: UNSTABLE', 'Native SNN: 10-ms activity', 'Native SNN: 100-ms activity']
    for ax, val, title in zip(axes, values, titles):
        im = show(ax, val, f'{title}\nmean {np.average(val, weights=m.count_e):.1f} Hz', cmap='magma', vmin=0, vmax=500)
    fig.colorbar(im, ax=axes, label='E rate (Hz; shared linear scale)', shrink=.8)
    fig.suptitle('Around 10.5 s: equilibrium pattern and observed spiking pattern\nNative SNN retains evolving Z and OU input; it is not an equilibrium simulation', fontsize=13)
    save(fig, 'fixed_point_vs_native_spatial_activity',
         '左图为10.5秒原Z场对应的不稳定rate固定点，中、右分别为原SNN在该时刻中心10毫秒和100毫秒窗的E活动。原SNN真实计数严格聚合到同一10×10网格后按神经元数换算Hz，三图共用0–500 Hz色标。',
         '原SNN的Z和OU输入仍在演化，左图采用固定Z和参考输入，不能据空间相似直接验证固定点归因。改变计数窗会改变活动图，静态图仍不给出逐点传播顺序。')
    write(OUT / 'fixed_point_spatial_activity_summary.json', {'status': 'COMPLETE', 'rows': rows,
          'core_centers_mm': centers.tolist(), 'native_windows_ms': [[lo, hi] for _, lo, hi in observed],
          'sources': ['branches.npz', 'spectrum_status.json', str(REFERENCE / 'trajectory.npz')],
          'scope': 'Verified rate equilibria; not native SNN attractor identification or propagation-order evidence.'})
    print(rows)


if __name__ == '__main__':
    main()
