#!/usr/bin/env python3
"""
玉泉数据集深度探索脚本
分析HFO事件的时空分布特征

Author: Generated on 2026-01-12
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from yuquan_dataloader import YuquanDataset
from datetime import datetime


def analyze_event_temporal_distribution(ds: YuquanDataset, patient: str, record: str):
    """分析事件的时间分布"""
    print(f"\n{'='*70}")
    print(f"时间分布分析: {patient} - {record}")
    print('='*70)
    
    # 加载数据
    times = ds.load_event_times(patient, record)
    gpu_data = ds.load_gpu_detections(patient, record)
    
    # 基本统计
    event_starts = times[:, 0]
    event_durations = times[:, 1] - times[:, 0]
    
    print(f"\n事件时间统计:")
    print(f"  总事件数:       {len(times)}")
    print(f"  记录时长:       {event_starts.max():.1f} 秒 ({event_starts.max()/3600:.2f} 小时)")
    print(f"  平均事件持续:   {event_durations.mean()*1000:.1f} ms")
    print(f"  事件持续范围:   {event_durations.min()*1000:.1f} ~ {event_durations.max()*1000:.1f} ms")
    
    # 事件间隔
    if len(event_starts) > 1:
        inter_event_intervals = np.diff(event_starts)
        print(f"\n事件间隔统计:")
        print(f"  平均间隔:       {inter_event_intervals.mean():.3f} 秒")
        print(f"  中位间隔:       {np.median(inter_event_intervals):.3f} 秒")
        print(f"  最短间隔:       {inter_event_intervals.min():.3f} 秒")
        print(f"  最长间隔:       {inter_event_intervals.max():.3f} 秒")
    
    # 事件发生率随时间变化
    bin_size = 300  # 5分钟bins
    n_bins = int(np.ceil(event_starts.max() / bin_size))
    hist, bin_edges = np.histogram(event_starts, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 / 60  # 转换为分钟
    
    # 可视化
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    # 子图1: 事件发生率随时间
    ax1 = axes[0]
    ax1.plot(bin_centers, hist, linewidth=1, color='steelblue')
    ax1.fill_between(bin_centers, hist, alpha=0.3, color='steelblue')
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('Event Count (per 5 min)')
    ax1.set_title(f'{patient} - {record}: Event Rate Over Time')
    ax1.grid(alpha=0.3)
    
    # 子图2: 事件间隔分布
    ax2 = axes[1]
    if len(event_starts) > 1:
        valid_intervals = inter_event_intervals[inter_event_intervals < 60]  # 只看<60秒的
        ax2.hist(valid_intervals, bins=100, color='coral', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Inter-Event Interval (seconds)')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Inter-Event Interval Distribution (< 60s)')
        ax2.grid(axis='y', alpha=0.3)
    
    # 子图3: 每个通道的事件数
    ax3 = axes[2]
    events_per_ch = gpu_data['events_count']
    ch_names = gpu_data['chns_names']
    x = np.arange(len(events_per_ch))
    ax3.bar(x, events_per_ch, color='teal', alpha=0.7)
    ax3.set_xlabel('Channel Index')
    ax3.set_ylabel('Event Count')
    ax3.set_title('Events per Channel')
    ax3.set_yscale('log')
    ax3.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def analyze_lagpat_propagation(ds: YuquanDataset, patient: str, record: str):
    """分析事件的传播模式"""
    print(f"\n{'='*70}")
    print(f"传播模式分析: {patient} - {record}")
    print('='*70)
    
    # 加载滞后模式
    lag_data = ds.load_lagpat(patient, record, with_freq=True)
    lag_raw = lag_data['lagPatRaw']
    lag_freq = lag_data['lagPatFreq']
    core_chns = lag_data['chnNames']
    events_bool = lag_data['eventsBool']
    
    print(f"\n滞后模式矩阵:")
    print(f"  核心通道: {list(core_chns)}")
    print(f"  矩阵形状: {lag_raw.shape}")
    print(f"  有效事件数: {events_bool.sum(axis=0).sum()}")
    
    # 统计每个通道的滞后时间
    print(f"\n通道滞后时间统计 (相对于事件开始):")
    for i, ch in enumerate(core_chns):
        valid_lags = lag_raw[i, events_bool[i, :] > 0]
        if len(valid_lags) > 0:
            print(f"  {ch:4s}: mean={valid_lags.mean():6.3f}s, std={valid_lags.std():6.3f}s, "
                  f"range=[{valid_lags.min():6.3f}, {valid_lags.max():6.3f}]")
    
    # 统计频率分布
    print(f"\n通道频率分布:")
    for i, ch in enumerate(core_chns):
        valid_freqs = lag_freq[i, events_bool[i, :] > 0]
        if len(valid_freqs) > 0:
            print(f"  {ch:4s}: mean={valid_freqs.mean():6.1f}Hz, std={valid_freqs.std():5.1f}Hz, "
                  f"range=[{valid_freqs.min():6.1f}, {valid_freqs.max():6.1f}]")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 子图1: 滞后模式热图
    ax1 = axes[0, 0]
    # 只显示有效事件
    valid_mask = events_bool > 0
    lag_masked = np.where(valid_mask, lag_raw, np.nan)
    
    im1 = ax1.imshow(lag_masked[:, :200], aspect='auto', cmap='RdYlBu_r', 
                     interpolation='nearest', vmin=0, vmax=0.5)
    ax1.set_xlabel('Event Index')
    ax1.set_ylabel('Channel')
    ax1.set_yticks(range(len(core_chns)))
    ax1.set_yticklabels(core_chns)
    ax1.set_title('Lag Pattern (first 200 events)')
    plt.colorbar(im1, ax=ax1, label='Lag Time (s)')
    
    # 子图2: 频率热图
    ax2 = axes[0, 1]
    freq_masked = np.where(valid_mask, lag_freq, np.nan)
    im2 = ax2.imshow(freq_masked[:, :200], aspect='auto', cmap='viridis',
                     interpolation='nearest', vmin=80, vmax=500)
    ax2.set_xlabel('Event Index')
    ax2.set_ylabel('Channel')
    ax2.set_yticks(range(len(core_chns)))
    ax2.set_yticklabels(core_chns)
    ax2.set_title('Frequency Pattern (first 200 events)')
    plt.colorbar(im2, ax=ax2, label='Frequency (Hz)')
    
    # 子图3: 通道间滞后时间相关性
    ax3 = axes[1, 0]
    # 计算通道间滞后时间的相关性
    valid_events_all = valid_mask.all(axis=0)
    if valid_events_all.sum() > 10:
        lag_valid = lag_raw[:, valid_events_all]
        corr_matrix = np.corrcoef(lag_valid)
        
        im3 = ax3.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax3.set_xticks(range(len(core_chns)))
        ax3.set_yticks(range(len(core_chns)))
        ax3.set_xticklabels(core_chns, rotation=45)
        ax3.set_yticklabels(core_chns)
        ax3.set_title('Channel Lag Correlation')
        plt.colorbar(im3, ax=ax3)
        
        # 添加数值标注
        for i in range(len(core_chns)):
            for j in range(len(core_chns)):
                text = ax3.text(j, i, f'{corr_matrix[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=8)
    
    # 子图4: 频率分布直方图
    ax4 = axes[1, 1]
    for i, ch in enumerate(core_chns):
        valid_freqs = lag_freq[i, events_bool[i, :] > 0]
        if len(valid_freqs) > 10:
            ax4.hist(valid_freqs, bins=30, alpha=0.5, label=ch, range=(80, 500))
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Count')
    ax4.set_title('Frequency Distribution per Channel')
    ax4.legend(fontsize=8, ncol=2)
    ax4.grid(axis='y', alpha=0.3)
    
    # 标记Ripple和Fast Ripple边界
    ax4.axvline(250, color='red', linestyle='--', linewidth=2, label='Ripple/FR boundary')
    ax4.legend(fontsize=8, ncol=2)
    
    plt.tight_layout()
    return fig


def analyze_all_patients_summary(ds: YuquanDataset):
    """分析所有患者的HFO事件统计"""
    print(f"\n{'='*70}")
    print("所有患者HFO事件汇总分析")
    print('='*70)
    
    patient_stats = []
    
    for patient in ds.list_patients():
        try:
            summary = ds.load_patient_summary(patient)
            events_total = summary['events_count'].sum()
            events_per_ch = summary['events_count']
            
            # 核心通道信息
            try:
                selection = ds.load_channel_selection(patient)
                n_core = len(selection['pick_chns'])
                core_chns = list(selection['pick_chns'])
            except:
                n_core = 0
                core_chns = []
            
            patient_stats.append({
                'patient': patient,
                'total_events': events_total,
                'n_channels': len(events_per_ch),
                'n_core_channels': n_core,
                'core_channels': core_chns,
                'max_events_per_ch': events_per_ch.max(),
                'mean_events_per_ch': events_per_ch.mean(),
                'active_channels': (events_per_ch > 0).sum()
            })
            
        except FileNotFoundError:
            continue
    
    # 打印统计表
    print(f"\n{'患者':<20s} {'总事件数':>10s} {'核心通道数':>10s} {'活跃通道':>10s} {'最大/通道':>12s}")
    print('-' * 70)
    for stats in patient_stats:
        print(f"{stats['patient']:<20s} {int(stats['total_events']):>10,d} "
              f"{stats['n_core_channels']:>10d} {stats['active_channels']:>10d} "
              f"{int(stats['max_events_per_ch']):>12,d}")
    
    # 汇总统计
    total_events = sum(s['total_events'] for s in patient_stats)
    print('-' * 70)
    print(f"{'总计':<20s} {int(total_events):>10,d}")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 子图1: 每个患者的总事件数
    ax1 = axes[0, 0]
    patients = [s['patient'] for s in patient_stats]
    totals = [s['total_events'] for s in patient_stats]
    x = np.arange(len(patients))
    ax1.barh(x, totals, color='steelblue', alpha=0.7)
    ax1.set_yticks(x)
    ax1.set_yticklabels(patients, fontsize=8)
    ax1.set_xlabel('Total Event Count')
    ax1.set_title('HFO Events per Patient')
    ax1.grid(axis='x', alpha=0.3)
    
    # 子图2: 核心通道数分布
    ax2 = axes[0, 1]
    n_cores = [s['n_core_channels'] for s in patient_stats if s['n_core_channels'] > 0]
    ax2.hist(n_cores, bins=range(min(n_cores), max(n_cores)+2), 
             color='coral', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Number of Core Channels')
    ax2.set_ylabel('Patient Count')
    ax2.set_title('Distribution of Core Channel Count')
    ax2.grid(axis='y', alpha=0.3)
    
    # 子图3: 事件数 vs 核心通道数
    ax3 = axes[1, 0]
    valid_stats = [s for s in patient_stats if s['n_core_channels'] > 0]
    x_cores = [s['n_core_channels'] for s in valid_stats]
    y_events = [s['total_events'] for s in valid_stats]
    ax3.scatter(x_cores, y_events, s=100, alpha=0.6, color='teal')
    ax3.set_xlabel('Number of Core Channels')
    ax3.set_ylabel('Total Events')
    ax3.set_title('Events vs Core Channels')
    ax3.set_yscale('log')
    ax3.grid(alpha=0.3)
    
    # 添加患者标签
    for s in valid_stats:
        ax3.annotate(s['patient'][:8], 
                    (s['n_core_channels'], s['total_events']),
                    fontsize=6, alpha=0.7)
    
    # 子图4: 活跃通道比例
    ax4 = axes[1, 1]
    active_ratios = [s['active_channels'] / s['n_channels'] * 100 
                     for s in patient_stats]
    patients_sorted = [p for _, p in sorted(zip(active_ratios, patients))]
    ratios_sorted = sorted(active_ratios)
    x = np.arange(len(patients_sorted))
    ax4.barh(x, ratios_sorted, color='forestgreen', alpha=0.7)
    ax4.set_yticks(x)
    ax4.set_yticklabels(patients_sorted, fontsize=8)
    ax4.set_xlabel('Active Channel Ratio (%)')
    ax4.set_title('Percentage of Channels with Events')
    ax4.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig, patient_stats


# ============================================================================
# 主执行流程
# ============================================================================

if __name__ == '__main__':
    # 初始化数据集
    ds = YuquanDataset()
    
    # 选择一个患者进行深度分析
    patient = 'chengshuai'
    records = ds.get_patient_records(patient)
    record = records[0]  # 第一条记录
    
    print("\n" + "="*70)
    print(f"深度分析: {patient}")
    print("="*70)
    
    # 1. 时间分布分析
    fig1 = analyze_event_temporal_distribution(ds, patient, record)
    fig1.savefig(f'/Users/leijiaxin/Desktop/Pys/Epilepsia/{patient}_{record}_temporal.png',
                 dpi=150, bbox_inches='tight')
    print(f"✓ 已保存: {patient}_{record}_temporal.png")
    plt.close(fig1)
    
    # 2. 传播模式分析
    fig2 = analyze_lagpat_propagation(ds, patient, record)
    fig2.savefig(f'/Users/leijiaxin/Desktop/Pys/Epilepsia/{patient}_{record}_propagation.png',
                 dpi=150, bbox_inches='tight')
    print(f"✓ 已保存: {patient}_{record}_propagation.png")
    plt.close(fig2)
    
    # 3. 所有患者汇总
    fig3, stats = analyze_all_patients_summary(ds)
    fig3.savefig('/Users/leijiaxin/Desktop/Pys/Epilepsia/all_patients_summary.png',
                 dpi=150, bbox_inches='tight')
    print(f"✓ 已保存: all_patients_summary.png")
    plt.close(fig3)
    
    print("\n" + "="*70)
    print("分析完成!")
    print("="*70)
