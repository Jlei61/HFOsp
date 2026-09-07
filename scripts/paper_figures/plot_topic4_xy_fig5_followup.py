#!/usr/bin/env python3
"""Regenerate Fig5 development diagnostics from verified new-substrate outputs."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from src.topic4_xy_fig5_followup import read, sha, verify_lock, write

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none'})


def export(fig, directory, stem):
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(directory / f'{stem}.{ext}', dpi=190, bbox_inches='tight')
    plt.close(fig)


def checked_rows(path):
    if not path.exists():
        return []
    rows = read(path)['results']
    for row in rows:
        if sha(row['arrays']['path']) != row['arrays']['sha256']:
            raise RuntimeError('plot input arrays changed')
    return rows


def scan_plot(out, directory, rows):
    plan = read(out / 'analysis_plan.json')
    factors = sorted(plan['threshold_factors'])
    taus = sorted(plan['tau_z_ms'])
    probability = np.full((len(factors), len(taus)), np.nan)
    latency = probability.copy()
    summary = []
    for cfg in read(out / 'parameter_design.json')['configs']:
        units = [r for r in rows if r['job']['config']['config_id'] == cfg['config_id']]
        valid = [r for r in units if r['trajectory']['classification'] != 'EARLY_STOP_UNRESOLVED']
        complete = len(valid) == len(plan['analysis_topology_seeds'])
        frac = float(np.mean([r['trajectory']['runaway'] for r in valid])) if complete else None
        # With one common administrative horizon this is the empirical restricted mean.
        restricted_mean = float(np.mean([r['trajectory']['latency_observed_ms'] / 1000 for r in valid])) if complete else None
        summary.append({'config': cfg, 'n': len(units), 'n_valid': len(valid),
                        'n_right_censored': sum(r['trajectory']['right_censored'] for r in units),
                        'runaway_fraction': frac, 'restricted_mean_latency_s': restricted_mean,
                        'critical_states': [r['critical_state'] for r in units if r['critical_state'] is not None]})
        if cfg['family'] == 'threshold_tau_z' and complete:
            i, j = factors.index(cfg['threshold_factor']), taus.index(cfg['tau_z'])
            probability[i, j] = frac; latency[i, j] = restricted_mean
    write(out / 'parameter_summary.json', {'rows': summary, 'horizon_ms': plan['duration_ms'],
          'unit': 'three topology/dynamics pairs, not patients',
          'latency_definition': 'Mean min(onset,20s); non-events right censored. Unresolved stops are N/A.',
          'critical_state_definition': 'regional Z/M just before the operational onset, not a bifurcation classification'})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, data, title, cmap, label, vmax in [
        (axes[0, 0], probability, 'A  Tonic runaway within 20 s', 'magma', 'Fraction (3 seeds)', 1),
        (axes[0, 1], latency, 'B  Restricted mean latency', 'viridis', 'Time (s); horizon = 20 s', 20)]:
        im = ax.imshow(np.ma.masked_invalid(data), origin='lower', aspect='auto', cmap=cmap, vmin=0, vmax=vmax)
        ax.set(xticks=np.arange(len(taus)), xticklabels=[f'{t/1000:g}' for t in taus],
               yticks=np.arange(len(factors)), yticklabels=[f'{v:g}' for v in factors],
               xlabel=r'$\tau_Z$ (s)', ylabel=r'$I_{th}$ / reference', title=title)
        for i in range(len(factors)):
            for j in range(len(taus)):
                value = data[i, j]
                ax.text(j, i, 'N/A' if np.isnan(value) else f'{value:.2g}', ha='center', va='center',
                        color='white' if np.isfinite(value) and value < vmax * .45 else 'black')
        fig.colorbar(im, ax=ax, label=label, shrink=.85)
    colors = ['#1f77b4', '#b84b19', '#65429b']
    for r, name in enumerate(('Core A', 'Core B', 'Surround')):
        xz, yz, xm, ym = [], [], [], []
        for row in rows:
            state = row['critical_state']
            if state is None or row['job']['config']['family'] == 'control':
                continue
            xz.append(row['job']['config']['I_th_EI'] / plan['zm_reference']['I_th_EI'])
            yz.append(state['region_z'][r])
            xm.append(row['job']['config']['tau_adp'] / 1000)
            ym.append(state['region_adaptation'][r])
        axes[1, 0].scatter(xz, yz, s=22, alpha=.65, color=colors[r], label=name)
        axes[1, 1].scatter(xm, ym, s=22, alpha=.65, color=colors[r], label=name)
    axes[1, 0].set(title='C  Z state at operational onset', xlabel=r'$I_{th}$ / reference', ylabel='Z', ylim=(-.03, 1.03))
    axes[1, 1].set(title='D  Adaptation current at operational onset', xlabel=r'$\tau_M$ (s)', ylabel=r'$\eta_M M$ (model current)')
    for ax in axes[1]:
        ax.legend(frameon=False, fontsize=9)
        if not any(r['critical_state'] is not None for r in rows):
            ax.text(.5, .5, 'No qualified tonic onset in scan', transform=ax.transAxes, ha='center')
    fig.suptitle('New XY substrate: Z/M parameter and state diagnostics\nDevelopment analysis; no bifurcation type established', fontsize=13)
    export(fig, directory, 'fig5_parameter_state_diagnostics')


def trajectory_plot(out, directory, rows):
    handoff = read(out / 'development_substrate.json')
    # Stable file/seed order; all rows retained, no best-looking-example selection.
    active = [r for r in rows if r['job']['config']['family'] != 'control']
    if not active:
        return False
    row = sorted(active, key=lambda r: (r['job']['topology_seed'], r['job']['dynamics_seed']))[0]
    with np.load(row['arrays']['path']) as handle:
        a = {k: handle[k] for k in handle.files}
    fig = plt.figure(figsize=(13, 10), constrained_layout=True)
    gs = fig.add_gridspec(4, 2, width_ratios=[1, 2.7])
    geom = fig.add_subplot(gs[:2, 0])
    geom.scatter(a['positions_E'][::10, 0], a['positions_E'][::10, 1], c=a['vtheta_E'][::10],
                 s=3, cmap='plasma', rasterized=True)
    centers = np.asarray(handoff['candidate']['node_field']['centers_mm'])
    geom.plot(centers[:, 0], centers[:, 1], 'k--o', lw=1)
    geom.scatter(a['contact_xy'][:, 0], a['contact_xy'][:, 1], marker='+', color='cyan', s=40)
    angle = np.deg2rad(handoff['candidate']['mechanisms']['ellipse_angle_deg'])
    direction = 5 * np.array([np.cos(angle), np.sin(angle)])
    geom.plot([10-direction[0], 10+direction[0]], [10-direction[1], 10+direction[1]], color='white', lw=2)
    geom.set(xlim=(0, 20), ylim=(0, 20), aspect='equal', xlabel='x (mm)', ylabel='y (mm)', title='Frozen development geometry')
    note = fig.add_subplot(gs[2:, 0]); note.axis('off')
    cfg = row['job']['config']
    note.text(0, .95, '\n'.join([handoff['candidate']['candidate_id'],
        f'Topology {row["job"]["topology_seed"]}', f'Dynamics {row["job"]["dynamics_seed"]}',
        f'I threshold = {cfg["I_th_EI"]:.3g}', f'tau Z = {cfg["tau_z"]/1000:g} s',
        f'tau M = {cfg["tau_adp"]/1000:g} s', f'eta M = {cfg["eta_m"]:.4g}', '',
        'White line: global EE kernel axis', 'Cyan: virtual contacts',
        'Readout: current-based LFP proxy', '', 'All validation seeds retained',
        'Development candidate', 'Author acceptance pending']), va='top', fontsize=10, linespacing=1.7)
    axes = [fig.add_subplot(gs[i, 1]) for i in range(4)]
    lfp = a['lfp']; centered = lfp - np.median(lfp[:min(1000, len(lfp))], axis=0)
    scale = max(float(np.percentile(np.abs(centered), 99)), 1e-9)
    for ch in range(lfp.shape[1]):
        axes[0].plot(a['lfp_time_ms'][::2] / 1000, centered[::2, ch] / scale + ch, color='.2', lw=.5)
    axes[0].set(yticks=np.arange(lfp.shape[1]), yticklabels=a['contact_names'].astype(str), title='A  Spontaneous virtual-contact trace')
    axes[0].tick_params(axis='y', labelsize=7)
    for i, name in enumerate(('Population', 'Core A', 'Core B', 'Surround')):
        axes[1].plot(a['time_ms'] / 1000, a['rates_hz'][:, i], label=name, lw=1)
    axes[1].set(ylabel='Rate (Hz)', title='B  Recruitment and rate change')
    axes[1].legend(ncol=4, fontsize=8, frameon=False)
    for i, name in enumerate(('Core A', 'Core B', 'Surround')):
        axes[2].plot(a['slow_time_ms'] / 1000, a['region_z'][:, i], label=name)
        axes[3].plot(a['slow_time_ms'] / 1000, a['region_adaptation'][:, i], label=name)
    axes[2].set(ylabel='Z', ylim=(-.03, 1.03), title='C  Inhibitory efficacy')
    axes[3].set(ylabel=r'$\eta_M M$', title='D  Adaptation current')
    for ax in axes:
        ax.set_xlabel('Time (s)')
        if row['trajectory']['onset_ms'] is not None:
            ax.axvline(row['trajectory']['onset_ms'] / 1000, color='#b00062', ls='--', lw=1)
    fig.suptitle('Same new-substrate trajectory: virtual readout, rates and spatial Z/M', fontsize=14)
    export(fig, directory, 'fig5_same_trajectory_diagnostics')
    write(out / 'trajectory_figure_source.json', {'job_id': row['job']['job_id'], 'arrays': row['arrays'],
                                                  'selection': 'lowest topology/dynamics seed among active validation units'})
    return True


def probe_plot(out, directory, rows):
    if not rows:
        return False
    handoff = read(out / 'development_substrate.json')
    all_rows = [dict(r, topology_seed=u['job']['topology_seed'], dynamics_seed=u['job']['dynamics_seed'])
                for u in rows for r in u['probe_rows']]
    write(out / 'spatial_probe_summary.json', {'rows': all_rows,
          'unit': 'paired topology/dynamics seed, conditional on qualified transition; all signed effects retained',
          'endpoint': 'spikes per E neuron over 300 ms, probe minus identical-checkpoint sham',
          'axis_claim': 'Spatial diagnostic only; anisotropy contribution requires a separate geometry null.'})
    coordinates = read(out / 'analysis_plan.json')['probe']['site_coordinates_mm']
    n = len(coordinates)
    data = []
    for state in ('baseline', 'pre_onset'):
        by_seed = [[r['signed_excess_spikes_per_E'] for r in u['probe_rows'] if r['state'] == state] for u in rows]
        data.append(np.mean(by_seed, axis=0).reshape(n, n))
    data.append(data[1] - data[0])
    bound = max(float(np.max(np.abs(data))), 1e-9)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), constrained_layout=True)
    centers = np.asarray(handoff['candidate']['node_field']['centers_mm'])
    for ax, values, title in zip(axes, data, ('Baseline', 'Pre-onset', 'Pre-onset minus baseline')):
        im = ax.imshow(values, origin='lower', extent=(1, 19, 1, 19), cmap='RdBu_r', vmin=-bound, vmax=bound)
        ax.plot(centers[:, 0], centers[:, 1], 'k--o', lw=1, ms=4)
        ax.set(title=title, xlabel='x (mm)', ylabel='y (mm)', xlim=(0, 20), ylim=(0, 20), aspect='equal')
    fig.colorbar(im, ax=axes, shrink=.7, label='Signed excess spikes / E neuron')
    fig.suptitle(f'Paired spatial perturbation: {len(rows)} validated seed trajectories\nFixed 16-E packet; common-noise sham at each state', fontsize=13)
    export(fig, directory, 'fig5_paired_spatial_perturbation')
    return True


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); out = args.out.resolve()
    verify_lock(out / 'runtime_lock.json')
    pilot = checked_rows(out / 'pilot_results.json')
    validation = checked_rows(out / 'validation_results.json')
    probes = checked_rows(out / 'probe_results.json')
    directory = out / 'figures'; directory.mkdir(exist_ok=True)
    scan_plot(out, directory, pilot)
    trajectory = trajectory_plot(out, directory, validation) if validation else False
    spatial = probe_plot(out, directory, probes)
    text = '''### fig5_parameter_state_diagnostics.png
展示新 XY 基底上的 Z/M 参数扫描：20 s 内 tonic runaway 比例、含右删失的限制平均潜伏期，以及跳变前的区域 Z 与适应电流。每个点含三组拓扑/动力学种子；N/A 表示提前停止而未满足判据，不算“未发作”。这里的状态变化尚不能确定分叉类型。
**关注点**：阈值和时间常数是扫描参数，Z/M 是轨迹状态；不能把两者混为同一跳变。
'''
    if trajectory:
        text += '''\n### fig5_same_trajectory_diagnostics.png
展示验证阶段按种子顺序取出的同一条轨迹，包括虚拟触点电流代理、群体与区域放电率、Z 和适应电流。几何来自本轮选中的新 XY，所有验证种子完整保存在结果表中；图形身份为开发诊断候选。
**关注点**：速率变化与 Z/M 的时间关系，以及相应的 core 和电极位置。
'''
    if spatial:
        text += '''\n### fig5_paired_spatial_perturbation.png
展示固定 16 个 E 神经元脉冲，在基线和跳变前的空间响应。每个扰动都减去同一 checkpoint、相同噪声与延迟环的 sham；完整保留正负响应和脉冲碰撞数。当前是条件于合格转变轨迹的 3×3 探索性网格，不能单凭长轴对齐宣称各向异性机制已被证实。
**关注点**：两种状态的响应差异及其空间分布，而非原始活动能量之差。
'''
    (directory / 'README.md').write_text(text + '\n每张图同时提供 PDF / SVG。作者目视验收尚未完成。\n')
    write(out / 'figure_metadata.json', {'status': 'FIG5_DEVELOPMENT_DIAGNOSTICS_GENERATED',
          'author_acceptance': False, 'runtime_lock_sha256': sha(out / 'runtime_lock.json'),
          'handoff_sha256': sha(out / 'development_substrate.json'),
          'assets': {str(p): sha(p) for p in sorted(directory.iterdir()) if p.is_file()},
          'claim_boundary': read(out / 'analysis_plan.json')['claim_boundary']})


if __name__ == '__main__':
    main()
