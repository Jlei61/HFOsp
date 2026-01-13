#!/usr/bin/env python3
"""
可视化HFO事件的原始波形

Usage:
    python visualize_event_waveforms.py <patient> <record> <event_idx>
    
Example:
    python visualize_event_waveforms.py chengshuai FC10477Q 100
"""

import sys
import numpy as np
import mne
import matplotlib.pyplot as plt
from pathlib import Path


def load_event_waveform(raw, event_time, duration=0.5, tmin=-0.1, tmax=0.6):
    """
    提取事件的原始波形
    
    Parameters:
    -----------
    raw : mne.io.Raw
        原始信号
    event_time : float
        事件开始时间(秒)
    duration : float
        事件持续时间(秒)
    tmin, tmax : float
        相对于事件开始的时间窗(秒)
    
    Returns:
    --------
    data : np.ndarray
        (n_channels, n_samples)
    times : np.ndarray
        时间轴(秒)
    """
    sfreq = raw.info['sfreq']
    start_sample = int((event_time + tmin) * sfreq)
    end_sample = int((event_time + tmax) * sfreq)
    
    data, times = raw[:, start_sample:end_sample]
    times = times - tmin  # 相对于事件开始
    
    return data, times


def plot_all_channels(data, times, ch_names, core_channels, title, figsize=(15, 20)):
    """
    绘制所有通道，核心通道用红色
    
    Parameters:
    -----------
    data : np.ndarray
        (n_channels, n_samples)
    times : np.ndarray
        时间轴
    ch_names : list
        通道名
    core_channels : list
        核心通道名
    """
    n_channels = data.shape[0]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 计算通道间距（用于堆叠显示）
    spacing = np.median(np.abs(data)) * 5
    
    for i in range(n_channels):
        offset = -i * spacing
        color = 'red' if ch_names[i] in core_channels else 'black'
        alpha = 0.8 if ch_names[i] in core_channels else 0.3
        linewidth = 1.5 if ch_names[i] in core_channels else 0.5
        
        ax.plot(times * 1000, data[i, :] + offset, 
                color=color, alpha=alpha, linewidth=linewidth)
        
        # 标注通道名
        ax.text(-50, offset, ch_names[i], 
                va='center', ha='right', fontsize=6,
                color='red' if ch_names[i] in core_channels else 'gray',
                weight='bold' if ch_names[i] in core_channels else 'normal')
    
    ax.axvline(0, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Event Start')
    ax.axvline(500, color='blue', linestyle='--', linewidth=2, alpha=0.7, label='Event End')
    
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Channel', fontsize=12)
    ax.set_title(title, fontsize=14, weight='bold')
    ax.set_yticks([])
    ax.legend(loc='upper right')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_core_channels(data, times, core_ch_names, title, figsize=(15, 8)):
    """
    只绘制核心通道
    """
    n_channels = len(core_ch_names)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    spacing = np.median(np.abs(data)) * 5
    
    for i in range(n_channels):
        offset = -i * spacing
        ax.plot(times * 1000, data[i, :] + offset, 
                color='red', linewidth=1.5, alpha=0.8)
        
        ax.text(-50, offset, core_ch_names[i], 
                va='center', ha='right', fontsize=10,
                color='red', weight='bold')
    
    ax.axvline(0, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Event Start')
    ax.axvline(500, color='blue', linestyle='--', linewidth=2, alpha=0.7, label='Event End')
    
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Channel', fontsize=12)
    ax.set_title(title, fontsize=14, weight='bold')
    ax.set_yticks([])
    ax.legend(loc='upper right')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    if len(sys.argv) != 4:
        print("Usage: python visualize_event_waveforms.py <patient> <record> <event_idx>")
        print("Example: python visualize_event_waveforms.py chengshuai FC10477Q 100")
        sys.exit(1)
    
    patient = sys.argv[1]
    record = sys.argv[2]
    event_idx = int(sys.argv[3])
    
    base_dir = Path('/Volumes/Elements/yuquan_24h_edf')
    patient_dir = base_dir / patient
    
    print(f"\n加载数据: {patient} - {record}, 事件 #{event_idx}")
    print("="*70)
    
    # 1. 加载原始信号（不预加载到内存）
    print("读取EDF...")
    edf_path = patient_dir / f'{record}.edf'
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False, encoding='latin1')
    
    # 2. 加载GPU检测结果
    print("读取GPU检测结果...")
    gpu = np.load(patient_dir / f'{record}_gpu.npz', allow_pickle=True)
    all_ch_names = list(gpu['chns_names'])
    
    # 3. 加载lagPat和事件时间
    print("读取lagPat和事件时间...")
    lag = np.load(patient_dir / f'{record}_lagPat_withFreqCent.npz', allow_pickle=True)
    core_ch_names = list(lag['chnNames'])
    events_bool = lag['eventsBool']
    
    times_all = np.load(patient_dir / f'{record}_packedTimes.npy')
    
    if event_idx >= len(times_all):
        print(f"错误: 事件索引 {event_idx} 超出范围 (0-{len(times_all)-1})")
        sys.exit(1)
    
    event_time = times_all[event_idx, 0]
    
    # 检查这个事件在核心通道中的参与情况
    core_participation = events_bool[:, event_idx]
    participating_cores = [ch for i, ch in enumerate(core_ch_names) if core_participation[i] > 0]
    
    print(f"\n事件信息:")
    print(f"  事件索引: {event_idx}")
    print(f"  事件时间: {event_time:.3f}s")
    print(f"  核心通道总数: {len(core_ch_names)}")
    print(f"  参与的核心通道: {len(participating_cores)} - {participating_cores}")
    
    # 4. 提取所有通道的波形（只保留GPU检测的120通道）
    print(f"\n提取波形...")
    
    # 从原始145通道中选出GPU检测的120通道
    # 需要处理通道名格式差异: "A1" vs "EEG A1-Ref" or "POL A1"
    raw_ch_names = raw.ch_names
    
    def find_channel(ch_name, raw_ch_list):
        """匹配通道名"""
        # 直接匹配
        if ch_name in raw_ch_list:
            return ch_name
        # 尝试匹配带前缀的
        for raw_ch in raw_ch_list:
            if raw_ch.endswith(ch_name) or ch_name in raw_ch:
                return raw_ch
        return None
    
    matched_channels = []
    for ch in all_ch_names:
        matched = find_channel(ch, raw_ch_names)
        if matched:
            matched_channels.append(matched)
    
    if len(matched_channels) == 0:
        print("错误: 通道名不匹配")
        print(f"GPU通道示例: {all_ch_names[:5]}")
        print(f"EDF通道示例: {raw_ch_names[:5]}")
        sys.exit(1)
    
    print(f"  匹配到 {len(matched_channels)}/{len(all_ch_names)} 个通道")
    
    # 先选择通道，再加载这一小段数据
    raw_selected = raw.copy().pick(matched_channels)
    
    # 只加载事件周围的数据
    sfreq = raw_selected.info['sfreq']
    tmin, tmax = -0.1, 0.6
    start_sample = int((event_time + tmin) * sfreq)
    end_sample = int((event_time + tmax) * sfreq)
    
    # 加载这一小段
    data_all = raw_selected[:, start_sample:end_sample][0]
    times_vec = np.arange(data_all.shape[1]) / sfreq + tmin
    
    # 5. 提取核心通道的波形
    matched_core = []
    for ch in core_ch_names:
        matched = find_channel(ch, raw_ch_names)
        if matched:
            matched_core.append(matched)
    
    raw_core = raw.copy().pick(matched_core)
    data_core = raw_core[:, start_sample:end_sample][0]
    
    print(f"  所有通道波形: {data_all.shape}")
    print(f"  核心通道波形: {data_core.shape}")
    
    # 6. 绘图
    print(f"\n生成图像...")
    
    # 图1: 所有通道 - 总事件
    fig1 = plot_all_channels(
        data_all, times_vec, 
        [raw_selected.ch_names[i] for i in range(len(raw_selected.ch_names))],
        core_ch_names,
        f'{patient} - {record} - Event #{event_idx}\nAll Channels (Core in Red)',
        figsize=(15, 20)
    )
    
    # 图2: 核心通道 - 总事件
    fig2 = plot_core_channels(
        data_core, times_vec,
        [raw_core.ch_names[i] for i in range(len(raw_core.ch_names))],
        f'{patient} - {record} - Event #{event_idx}\nCore Channels Only',
        figsize=(15, 8)
    )
    
    # 图3: 所有通道 - 群体事件（只显示参与的核心通道为红色）
    if len(participating_cores) > 0:
        fig3 = plot_all_channels(
            data_all, times_vec,
            [raw_selected.ch_names[i] for i in range(len(raw_selected.ch_names))],
            participating_cores,  # 只标记参与的核心通道
            f'{patient} - {record} - Event #{event_idx}\nAll Channels (Participating Cores in Red)',
            figsize=(15, 20)
        )
        
        # 图4: 只显示参与的核心通道
        participating_indices = [i for i, ch in enumerate(core_ch_names) if core_participation[i] > 0]
        if len(participating_indices) > 0:
            data_participating = data_core[participating_indices, :]
            fig4 = plot_core_channels(
                data_participating, times_vec,
                [core_ch_names[i] for i in participating_indices],
                f'{patient} - {record} - Event #{event_idx}\nParticipating Core Channels',
                figsize=(15, max(6, len(participating_indices) * 0.8))
            )
        else:
            fig4 = None
    else:
        print("\n警告: 该事件没有核心通道参与")
        fig3 = fig1
        fig4 = None
    
    # 保存
    output_dir = Path('/Users/leijiaxin/Desktop/Pys/Epilepsia')
    
    fig1.savefig(output_dir / f'{patient}_{record}_event{event_idx}_all.png', 
                 dpi=150, bbox_inches='tight')
    print(f"✓ 保存: {patient}_{record}_event{event_idx}_all.png")
    
    fig2.savefig(output_dir / f'{patient}_{record}_event{event_idx}_core.png', 
                 dpi=150, bbox_inches='tight')
    print(f"✓ 保存: {patient}_{record}_event{event_idx}_core.png")
    
    fig3.savefig(output_dir / f'{patient}_{record}_event{event_idx}_group_all.png', 
                 dpi=150, bbox_inches='tight')
    print(f"✓ 保存: {patient}_{record}_event{event_idx}_group_all.png")
    
    if fig4:
        fig4.savefig(output_dir / f'{patient}_{record}_event{event_idx}_group_core.png', 
                     dpi=150, bbox_inches='tight')
        print(f"✓ 保存: {patient}_{record}_event{event_idx}_group_core.png")
    
    print("\n完成!")
    plt.close('all')


if __name__ == '__main__':
    main()
