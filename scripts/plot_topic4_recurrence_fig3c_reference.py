#!/usr/bin/env python3
"""Restore the actual locked Fig3C reference without recomputing patient power.

This is a reference assembly, not a new model/patient concordance analysis.
The preceding SZ13 ER comparison remains an audit artifact in its own folder.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import matplotlib.pyplot as plt

import plot_topic4_weak_fast_recurrence as previous


ROOT = Path(__file__).resolve().parents[1]
OUT = previous.BASE / 'fig3c_reference_repair_v3'
FIG = OUT / 'figures'
CANONICAL = ROOT / 'results/paper-ready-figure/fig3'


def main():
    source = CANONICAL / 'figures/fig3-panelc.png'
    metadata_path = CANONICAL / 'fig3_panelc_metadata.json'
    ref = previous.read(metadata_path)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert ref['seizure_idx'] == 2
    assert ref['ictal_extraction']['clinical_window_sec'] == [0., 10.]
    assert ref['ictal_extraction']['band_hz'] == [1., 150.]
    assert ref['ictal_extraction']['reference'] == 'car'
    # Paste the accepted artifact itself: no re-extraction, new interpolation,
    # model-coordinate projection, seizure selection, normalization or clipping.
    reference_image = plt.imread(source)
    with np.load(previous.OUT / 'runs' / (previous.NAME + '.npz')) as f:
        arrays = {k: f[k] for k in f.files}
    run = previous.read(previous.OUT / 'runs' / (previous.NAME + '.json'))
    metric = previous.read(previous.OUT / 'analysis_summary.json')
    snapshots = previous.snapshots(metric, run)
    fig = plt.figure(figsize=(27.5, 18))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], left=.075,
                            right=.93, top=.95, bottom=.065, wspace=.34)
    previous.left(fig, outer[0], arrays, run, metric, snapshots)
    right = outer[1].subgridspec(2, 1, height_ratios=[1.2, 1], hspace=.33)
    previous.trajectory(fig, right[0], arrays, run, snapshots)
    reference = right[1].subgridspec(2, 1, height_ratios=[.12, 1], hspace=.04)
    title = fig.add_subplot(reference[0])
    title.set_axis_off()
    title.text(0, .25, 'E2  Fig. 3C · patient reference', fontsize=20,
               fontweight='bold')
    ax = fig.add_subplot(reference[1])
    ax.imshow(reference_image, interpolation='none')
    ax.set_axis_off()
    FIG.mkdir(parents=True, exist_ok=True)
    previous.FIG = FIG
    previous.save(fig, 'weak_fast_recurrence_fig3c_reference')
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    result = dict(
        status='REFERENCE_RESTORED_PENDING_USER_REVIEW',
        source_run=str(previous.OUT / 'runs' / (previous.NAME + '.npz')),
        reference_source=str(source), reference_sha256=digest,
        reference_metadata=str(metadata_path),
        reference_subject=ref['subject'], reference_seizure_idx=2,
        reference_display_label=ref['display_label'],
        reuse='Entire canonical Fig3C PNG, no change to its scientific content',
        pdf_reference_is_raster=True,
        reference_power_limits=ref['display']['ictal_colorbar_limits'],
        reference_extraction=ref['ictal_extraction'],
        model_patient_comparison='NOT_ESTABLISHED; E2 is the fixed patient reference',
        removed_comparison='SZ13, 0-1 s, power ratio ER; not canonical Fig3C',
        previous_rho_not_applicable_to_Fig3C=True,
        recurrence=dict(duration_s=metric['duration_s'],
                        high_intervals_s=metric['high_intervals_s'],
                        refill_s=[75.5, 76.5],
                        post_release_observation_s=metric['duration_s'] - 76.5,
                        second_high_observed=False),
        agent_visual_review='PENDING', human_acceptance='PENDING')
    previous.write(OUT / 'figure_metadata.json', result)
    (FIG / 'README.md').write_text(
        '### weak_fast_recurrence_fig3c_reference.png / .pdf\n'
        '左侧及E1保留同一弱快M轨迹至240秒；75.5–76.5秒只补充Z一次，'
        '随后释放原生Z，M及快状态连续。E2直接嵌入已定稿Fig3C原图，'
        '展示TA间期场及E1146/SZ3临床发作后0–10秒的1–150Hz能量场，'
        '保留原坐标、触点、核函数及robust-z色条；PDF中的这一原图为高分辨率位图。\n'
        '**关注点**：E2是患者参考，不能当作模型已匹配的结果；'
        '上一版SZ13的ER图及相关系数不适用于Fig3C。'
        '240秒内未观察到第二次高活动，L仍为后期有限事件；待用户目视审阅。\n')
    (OUT / 'correction_note.md').write_text(
        '# Fig3C参考纠正与复发观察边界\n\n'
        '正式参考是E1146/SZ3（seizure_idx=2），不是SZ13。'
        '本次直接复用已定稿Fig3C，未重新挑发作、重算患者能量或映射到模型坐标。'
        '上一版使用另一例发作的0–1秒功率比ER，偏离已确认参考；'
        '其ρ不能作为当前模型与Fig3C的比较结论。模型与固定参考的匹配仍待验证。\n\n'
        '弱快M为ηM=0.02、τM=2秒。首次高态73.48–76.01秒，'
        '75.5–76.5秒外部补充Z后释放；到240秒未再次进入既定高态。'
        '原生Z方程有恢复项：τZ dz_i/dt=1[raw GABA_i<Ith]−z_i。'
        '超过阈值时趋向0，低于阈值时趋向1，因此不是累计时间足够就必然耗尽。'
        '100–130秒与210–240秒的0.5秒采样平均Z分别为0.8340和0.8331；'
        'M持续开启，恢复早期的强适应已衰减，随后由重复事件维持。'
        '单条随机轨迹不能区分更稀少的跨越与稳定的事件态，不能据此断言永不复发。\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
