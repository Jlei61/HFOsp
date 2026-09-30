#!/usr/bin/env python3
"""Audit early recruitment timing, band power, CAR, and applied inhibition.

This is an existing-trajectory diagnostic, never a replacement for the frozen
first-global-entry E2 endpoint of the running M parameter batch.
"""
import json
from pathlib import Path
import numpy as np
from scipy.signal import spectrogram
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/topic4_sef_hfo/fig5_m_overnight_exploration_20260913/early_energy_timing'
SOURCE = ROOT / 'results/topic4_sef_hfo/fig5_manual_core_release_v1/weak_fast_z_refill_recurrence_v2/runs/weak_fast_z_refill_recurrence.npz'
PATIENT = ROOT / 'results/paper-ready-figure/fig3/fig3_panelc_metadata.json'


def spans(mask):
    d = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def bands(x, fs, hop=None):
    f, t, p = spectrogram(x, fs=fs, nperseg=fs,
        noverlap=fs // 2 if hop is None else fs - hop, axis=0,
        detrend='constant', scaling='density', window='hann')
    return t, np.maximum(p[(f >= 1) & (f <= 150)].sum(0).T, 1e-20)


def safe(x):
    if isinstance(x, dict): return {k: safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [safe(v) for v in x]
    if isinstance(x, np.ndarray): return safe(x.tolist())
    if isinstance(x, np.generic): return safe(x.item())
    if isinstance(x, float) and not np.isfinite(x): return None
    return x


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(SOURCE) as a:
        raw = a['lfp_raw']; effective = a['lfp_effective']
        regions = a['region_spikes_1ms'].reshape(-1, 10, 6).sum(1) / a['region_counts'] / .01
        global_rate = a['rate_e_hz'].reshape(-1, 100).mean(1)
        native = a['field_e_count_1ms']; nc = a['cell_e_counts']
        contacts = a['contact_xy']; names = list(a['contact_names'])
        centers = a['centers_mm']
        slow_t = a['z_time_ms'] / 1000
        z = a['z_stats'][:, 0]; adaptation = a['m_stats'][:, 0] * .02
    high = [(l * .01, h * .01) for l, h in spans(global_rate >= 200) if h - l >= 20]
    onset = high[0][0]
    local = []
    for j in range(2):
        for lo, hi in spans(regions[:, j] >= 200):
            if hi - lo >= 20 and lo * .01 < onset:
                local.append(dict(core=['A', 'B'][j], onset_s=lo * .01, end_s=hi * .01))
    local.sort(key=lambda v: v['onset_s'])
    earliest = local[0]['onset_s']
    persistent = min(v['onset_s'] for v in local if v['end_s'] > onset)
    windows = [('Before local excursion', earliest - 1),
               ('First local excursion', earliest),
               ('Local high continuing into global high', persistent),
               ('Global high entry', onset),
               ('Established high', onset + 1), ('After refill', 77.)]
    patient = json.loads(PATIENT.read_text())
    ix = [names.index(n) for n in patient['contact_order']]
    py = np.asarray(patient['raw_ictal_robust_z_mean'])
    results = {}; display = {}
    for key, data in [('original', raw), ('Z_applied', effective)]:
        for car in [False, True]:
            x = data - data.mean(1, keepdims=True) if car else data
            tag = key + ('_CAR' if car else '_unreferenced')
            _, pb = bands(x[2000:60000], 2000)
            b = np.log10(pb); med = np.median(b, axis=0)
            mad = 1.4826 * np.median(abs(b - med), axis=0)
            assert np.all(mad > 1e-12)
            ts, p = bands(x[round(68 * 2000):round(78 * 2000)], 2000, hop=100)
            curve_db = 10 * (np.log10(p) - med)
            rows = []
            for label, start in windows:
                _, pp = bands(x[round(start * 2000):round((start + 1) * 2000)], 2000)
                lp = np.log10(pp).mean(0); robust_z = (lp - med) / mad
                db = 10 * (lp - med)
                rows.append(dict(label=label, window_s=[start, start + 1],
                    n_increased=int(np.sum(robust_z > 0)), robust_z=robust_z,
                    db_relative_baseline=db,
                    median_db=float(np.median(db)),
                    patient_rho=float(spearmanr(robust_z[ix], py).statistic)))
            baseline_counts = np.sum(b > med, axis=1)
            results[tag] = dict(windows=rows, baseline_windows=len(b),
                baseline_windows_with_12_or_more_increased=int(np.sum(baseline_counts >= 12)),
                baseline_counts=baseline_counts)
            display[tag] = dict(t=ts + 68, db=curve_db)
    native_windows = []
    for label, start in windows:
        rate = native[round(start * 1000):round((start + 1) * 1000)] / nc * 1000
        _, power = bands(rate, 1000)
        native_windows.append(dict(label=label, start_s=start,
                                   mean_rate=rate.mean(0), bandpower=power.mean(0)))
    baseline_native = native[1000:30000] / nc * 1000
    _, bp = bands(baseline_native, 1000)
    native_med = np.median(np.log10(bp), axis=0)
    for row in native_windows:
        row['bandpower_db'] = 10 * (np.log10(row['bandpower']) - native_med)
        row['bandpower_db'][native_med <= np.log10(1e-18)] = np.nan
    plt.rcParams.update({'font.size':15, 'axes.labelsize':17, 'axes.titlesize':17,
        'xtick.labelsize':14, 'ytick.labelsize':14, 'pdf.fonttype':42,
        'axes.spines.right':False, 'axes.spines.top':False})
    fig = plt.figure(figsize=(19, 15))
    gs = fig.add_gridspec(5, 2, height_ratios=[1, .8, 1.1, 1.15, 1.15],
                         hspace=.62, wspace=.28, left=.075, right=.93, bottom=.07, top=.93)
    rate_ax = fig.add_subplot(gs[0, :]); t = np.arange(len(global_rate)) * .01 + .005
    use = (t >= 68) & (t <= 78)
    for x, label, c in [(regions[:, 0], 'Core A', '#b13d68'),
                         (regions[:, 1], 'Core B', '#2582a1'), (global_rate, 'All E', '#282828')]:
        rate_ax.plot(t[use], x[use], label=label, color=c, lw=1.2)
    rate_ax.set(ylabel='E firing rate (Hz)', xlim=(68, 78), ylim=(0, 550))
    rate_ax.legend(ncol=3, frameon=False, loc='upper left')
    marks = [(earliest, 'Local excursion', '#238bad'), (persistent, 'Persistent core high', '#965e98'),
             (onset, 'Global high entry', '#b12736')]
    for (tm, label, col), text_x in zip(marks, [70.9, 72.5, 74.25]):
        rate_ax.axvline(tm, color=col, ls='--', lw=1)
        rate_ax.annotate(f'{label}\n{tm:.2f} s', xy=(tm, 540), xytext=(text_x, 590),
            color=col, fontsize=12, ha='center', va='bottom', annotation_clip=False,
            arrowprops=dict(arrowstyle='-', color=col, lw=.8))
    dc_ax = fig.add_subplot(gs[1, :], sharex=rate_ax)
    qt = np.arange(len(raw)) / 2000; choose = (qt >= 68) & (qt < 78)
    dc_ax.plot(qt[choose][::10], np.median(raw[choose], axis=1)[::10],
               color='#565656', label='Original current proxy')
    dc_ax.plot(qt[choose][::10], np.median(effective[choose], axis=1)[::10],
               color='#c38135', label='Z-applied current proxy')
    dc_ax.set(ylabel='Contact median\nunfiltered proxy (a.u.)')
    dc_ax.legend(ncol=2, frameon=False, loc='upper left')
    axes = [rate_ax, dc_ax]
    for col, key in enumerate(['original', 'Z_applied']):
        ax = fig.add_subplot(gs[2, col]); axes.append(ax)
        for car, color, label in [(False, '#999999', 'Before CAR'), (True, '#245c8c', 'After CAR')]:
            v = display[key + ('_CAR' if car else '_unreferenced')]
            ax.plot(v['t'], np.median(v['db'], axis=1), color=color, label=label, lw=1.5)
        ax.axhline(0, color='#888888', lw=.7, ls=':')
        ax.set(xlim=(68, 78), ylim=(-55, 15), xlabel='Time (s)', ylabel='1–150 Hz power\nchange (dB)')
        ax.set_title('Original proxy' if key == 'original' else 'Z-applied proxy', loc='left')
        ax.legend(frameon=False, fontsize=13)
    for ax in axes:
        ax.axvspan(75.5, 76.5, color='#429b85', alpha=.13, lw=0)
    # Native fields use measured spike-rate power, not contact interpolation.
    sub = gs[3:, :].subgridspec(2, 4, width_ratios=[1,1,1,.045], hspace=.24, wspace=.18)
    chosen = [1, 3, 4]
    for col, wi in enumerate(chosen):
        row = native_windows[wi]
        for ri, (quantity, norm, cmap) in enumerate([
            ('mean_rate', matplotlib.colors.Normalize(0, 500), 'magma'),
            ('bandpower_db', TwoSlopeNorm(vmin=-20, vcenter=0, vmax=20), 'RdBu')]):
            ax = fig.add_subplot(sub[ri, col]); im = ax.imshow(row[quantity].reshape(20,20),
                origin='lower', extent=[0,20,0,20], norm=norm, cmap=cmap, interpolation='nearest')
            for i, center in enumerate(centers):
                ax.add_patch(Circle(center, 1.5, fill=False, ec='#53e1dc', lw=1.3))
                ax.text(*center, 'AB'[i], color='#53e1dc', fontsize=13, ha='center', va='center')
            ax.set(xticks=[0,10,20], yticks=[0,10,20])
            if ri == 0: ax.set_title(f'{row["label"]}\n{row["start_s"]:.2f}–{row["start_s"]+1:.2f} s', fontsize=15)
            else: ax.set_xlabel('x (mm)')
            if col == 0: ax.set_ylabel('y (mm)')
            else: ax.tick_params(labelleft=False)
            if col == 2:
                cb = fig.colorbar(im, cax=fig.add_subplot(sub[ri, 3]))
                cb.set_label('Mean E rate (Hz)' if ri == 0 else 'Native 1–150 Hz\npower change (dB)')
    figs = OUT / 'figures'; figs.mkdir(exist_ok=True)
    for ext in ['png', 'pdf']: fig.savefig(figs / f'early_energy_timing.{ext}', dpi=170)
    plt.close(fig)
    artifact = safe(dict(source=str(SOURCE), existing_trajectory=True, new_M_batch_result=False,
        baseline_s=[1,30], band_Hz=[1,150], window_s=1, global_onset_s=onset,
        local_high_intervals=local, windows=windows, contact_names=names,
        patient_reference=str(PATIENT), readout_results=results,
        native_windows=native_windows, clinical_energy_claim='NOT_ESTABLISHED',
        full_Fig5_acceptance=False, formal_running_batch_endpoint_unchanged=True,
        native_and_contact_power_are_different_observables=True,
        interpretation='Temporal recruitment may precede global high-rate threshold. Band-power decrease during high firing must not be hidden by shifting windows or calling a tonic plateau sustained oscillation.',
        agent_visual_review='PENDING', human_review='PENDING'))
    (OUT / 'analysis.json').write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + '\n')
    lines = ['# 旧工作点：早期能量窗口与高率平台', '',
        '这是已有240秒轨迹的离线诊断，不是新M搜索结果。未改动任何动力学或扫描终点。', '',
        '| 1秒窗口起点 | 含义 | CAR触点增强数/15 | 触点dB中位数 | 与固定Fig3C相关 |',
        '|---:|---|---:|---:|---:|']
    for row in results['original_CAR']['windows']:
        lines.append(f'| {row["window_s"][0]:.2f} | {row["label"]} | {row["n_increased"]} | {row["median_db"]:.2f} | {row["patient_rho"]:.3f} |')
    count = results['original_CAR']['baseline_windows_with_12_or_more_increased']
    total = results['original_CAR']['baseline_windows']
    lines += ['', f'基线自身的{total}个重叠1秒窗中，也有{count}个窗达到至少12个触点高于基线中位数。因此短窗增强本身不能区分间期事件与发作早期；这些窗不是独立样本。', '',
        '以全E≥200Hz连续200ms定义的进入，晚于局部核升级。比较外部Z补充前的局部阶段、全局进入及随后平台，可检验能量增强是否短暂发生在招募前沿。改变CAR或改用Z加权GABA读出均只作同一轨迹的观测敏感性，不重新标定到患者。', '',
        '若局部早期有增强而全局高率阶段没有，说明原先“高率等于带功率增强”的解释不成立。局部起始作为以后E2时间锚点仍需跨新M条件检查，并明确操作定义；不能事后选择最吻合患者的时间窗替换原终点。', '',
        '原生场由1毫秒E发放计数计算，每格按实际E细胞数归一化。地图分别显示平均率与去均值后的1–150Hz功率相对同一基线变化，二者不能相互替代。']
    (OUT / 'scientific_review.md').write_text('\n'.join(lines) + '\n')
    (figs / 'README.md').write_text('### early_energy_timing.png / .pdf\n用已有M-on完整轨迹对照局部升级、全局高率进入与外部Z补充。上半部保持连续真实时间，分开展示率、未滤波电流proxy和带功率变化；下半部对照真实二维平均发放率与带功率变化。\n**关注点**：高率不等于带功率增强；早期窗口未按患者匹配选择，原始扫描E2定义保持不变，图待用户目视审阅。\n')
    print(json.dumps(safe(dict(global_onset_s=onset, local_high_intervals=local,
        original_CAR=results['original_CAR'])), ensure_ascii=False))


if __name__ == '__main__': main()
