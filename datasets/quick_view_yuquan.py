#!/usr/bin/env python3
"""
玉泉数据集快速查看脚本
用于快速浏览和验证数据

Usage:
    python quick_view_yuquan.py                    # 显示数据集概览
    python quick_view_yuquan.py chengshuai         # 显示某个患者的详情
    python quick_view_yuquan.py chengshuai FC10477Q  # 显示某条记录的详情
"""

import sys
from yuquan_dataloader import YuquanDataset
import numpy as np


def show_dataset_overview(ds):
    """显示数据集概览"""
    stats = ds.get_dataset_statistics()
    
    print("\n" + "="*70)
    print("玉泉24小时SEEG数据集")
    print("="*70)
    print(f"数据路径: /Volumes/Elements/yuquan_24h_edf")
    print(f"\n患者数量: {stats['n_patients']} (已处理: {stats['n_patients_processed']})")
    print(f"EDF文件: {stats['n_edf_files']}")
    print(f"HFO事件: {stats['total_events']:,}")
    print(f"平均事件率: {stats['total_events'] / (stats['n_gpu_files'] * 2):.0f} 事件/小时")
    
    print("\n已处理的患者 (按事件数排序):")
    
    # 获取每个患者的事件数
    patient_events = []
    for patient in stats['patients_processed']:
        try:
            summary = ds.load_patient_summary(patient)
            events = summary['events_count'].sum()
            patient_events.append((patient, int(events)))
        except:
            continue
    
    patient_events.sort(key=lambda x: x[1], reverse=True)
    
    for i, (patient, events) in enumerate(patient_events, 1):
        bar_len = int(40 * events / patient_events[0][1])
        bar = "█" * bar_len
        print(f"  {i:2d}. {patient:20s} {events:>8,d} {bar}")
    
    print("\n未处理的患者:")
    unprocessed = set(ds.list_patients()) - set(stats['patients_processed'])
    for patient in sorted(unprocessed):
        print(f"  - {patient}")
    
    print("\n" + "="*70)


def show_patient_detail(ds, patient):
    """显示患者详情"""
    print("\n" + "="*70)
    print(f"患者: {patient}")
    print("="*70)
    
    # 记录列表
    records = ds.get_patient_records(patient)
    print(f"\n记录数: {len(records)}")
    print(f"总时长: 约{len(records)*2}小时")
    
    # 汇总信息
    try:
        summary = ds.load_patient_summary(patient)
        events_count = summary['events_count']
        chns_names = summary['chns_names']
        
        print(f"\n总事件数: {int(events_count.sum()):,}")
        print(f"通道数: {len(chns_names)}")
        print(f"活跃通道: {(events_count > 0).sum()} ({(events_count > 0).sum()/len(events_count)*100:.1f}%)")
        
        # 核心通道
        try:
            selection = ds.load_channel_selection(patient)
            core_chns = selection['pick_chns']
            hist_vals = selection['hist_meanX']
            
            print(f"\n核心通道 ({len(core_chns)}个):")
            for ch, val in zip(core_chns, hist_vals):
                # 找到这个通道在events_count中的索引
                ch_idx = np.where(chns_names == ch)[0]
                if len(ch_idx) > 0:
                    n_events = int(events_count[ch_idx[0]])
                    print(f"  {ch:4s}: {n_events:>6,d} 事件, 质量分数={val:.3f}")
                else:
                    print(f"  {ch:4s}: 质量分数={val:.3f}")
        except FileNotFoundError:
            print("\n核心通道: 未筛选")
        
        # Top 10 最活跃通道
        top_indices = np.argsort(events_count)[-10:][::-1]
        print(f"\nTop 10 最活跃通道:")
        for i, idx in enumerate(top_indices, 1):
            ch_name = chns_names[idx]
            n_events = int(events_count[idx])
            pct = n_events / events_count.sum() * 100
            print(f"  {i:2d}. {ch_name:4s}: {n_events:>8,d} 事件 ({pct:5.2f}%)")
        
    except FileNotFoundError:
        print("\n⚠️  该患者尚未完成处理")
    
    # 记录列表
    print(f"\n所有记录:")
    for i, record in enumerate(records, 1):
        info = ds.get_record_info(patient, record)
        status = "✓" if info.has_gpu else "✗"
        events_str = f"{info.n_events:>6,d} 事件" if info.has_gpu else "未处理"
        print(f"  {status} {i:2d}. {record}: {events_str}")
    
    print("\n" + "="*70)


def show_record_detail(ds, patient, record):
    """显示记录详情"""
    print("\n" + "="*70)
    print(f"记录: {patient} - {record}")
    print("="*70)
    
    info = ds.get_record_info(patient, record)
    
    print(f"\n文件状态:")
    print(f"  EDF文件:      {'✓' if info.has_edf else '✗'}")
    print(f"  GPU检测:      {'✓' if info.has_gpu else '✗'}")
    print(f"  LagPat分析:   {'✓' if info.has_lagpat else '✗'}")
    
    if not info.has_gpu:
        print("\n⚠️  该记录尚未完成GPU检测")
        print("="*70)
        return
    
    print(f"\n基本信息:")
    print(f"  通道数: {info.n_channels}")
    print(f"  核心通道数: {info.n_core_channels}")
    print(f"  总事件数: {info.n_events:,}")
    print(f"  事件率: {info.n_events / 2:.0f} 事件/小时")
    
    # GPU检测详情
    gpu_data = ds.load_gpu_detections(patient, record)
    events_count = gpu_data['events_count']
    chns_names = gpu_data['chns_names']
    
    print(f"\n通道统计:")
    print(f"  最多事件/通道: {int(events_count.max()):,}")
    print(f"  平均事件/通道: {events_count.mean():.1f}")
    print(f"  中位事件/通道: {int(np.median(events_count))}")
    print(f"  零事件通道: {(events_count == 0).sum()}")
    
    # 事件时间分析
    times = ds.load_event_times(patient, record)
    event_starts = times[:, 0]
    event_durations = times[:, 1] - times[:, 0]
    
    print(f"\n时间分析:")
    print(f"  记录时长: {event_starts.max():.1f} 秒 ({event_starts.max()/3600:.2f} 小时)")
    print(f"  对齐事件数: {len(times)}")
    print(f"  平均事件持续: {event_durations.mean()*1000:.1f} ms")
    
    if len(event_starts) > 1:
        intervals = np.diff(event_starts)
        print(f"  平均事件间隔: {intervals.mean():.3f} 秒")
        print(f"  中位事件间隔: {np.median(intervals):.3f} 秒")
        print(f"  最短间隔: {intervals.min():.3f} 秒")
        print(f"  最长间隔: {intervals.max():.3f} 秒")
    
    # LagPat分析
    if info.has_lagpat:
        lag_data = ds.load_lagpat(patient, record, with_freq=True)
        core_chns = lag_data['chnNames']
        lag_freq = lag_data['lagPatFreq']
        events_bool = lag_data['eventsBool']
        
        print(f"\n核心通道分析:")
        print(f"  通道: {', '.join(core_chns)}")
        
        print(f"\n  频率分布:")
        for i, ch in enumerate(core_chns):
            valid_freqs = lag_freq[i, events_bool[i, :] > 0]
            if len(valid_freqs) > 0:
                print(f"    {ch:4s}: {valid_freqs.mean():6.1f} ± {valid_freqs.std():5.1f} Hz "
                      f"(范围: {valid_freqs.min():.1f} - {valid_freqs.max():.1f})")
        
        # 频段分类
        valid_mask = events_bool > 0
        all_freqs = lag_freq[valid_mask]
        ripple = (all_freqs >= 80) & (all_freqs < 250)
        fast_ripple = (all_freqs >= 250) & (all_freqs < 500)
        
        print(f"\n  频段统计:")
        print(f"    Ripple (80-250Hz):     {ripple.sum():>5d} ({ripple.sum()/len(all_freqs)*100:.1f}%)")
        print(f"    Fast Ripple (250-500Hz): {fast_ripple.sum():>5d} ({fast_ripple.sum()/len(all_freqs)*100:.1f}%)")
    
    print("\n" + "="*70)


def main():
    # 初始化数据集
    try:
        ds = YuquanDataset()
    except FileNotFoundError as e:
        print(f"\n❌ 错误: {e}")
        print("\n请确认数据集路径是否正确:")
        print("  /Volumes/Elements/yuquan_24h_edf")
        sys.exit(1)
    
    # 解析命令行参数
    if len(sys.argv) == 1:
        # 无参数: 显示概览
        show_dataset_overview(ds)
    
    elif len(sys.argv) == 2:
        # 1个参数: 显示患者详情
        patient = sys.argv[1]
        if patient not in ds.list_patients():
            print(f"\n❌ 错误: 患者 '{patient}' 不存在")
            print(f"\n可用患者: {', '.join(ds.list_patients()[:5])}...")
            sys.exit(1)
        show_patient_detail(ds, patient)
    
    elif len(sys.argv) == 3:
        # 2个参数: 显示记录详情
        patient = sys.argv[1]
        record = sys.argv[2]
        
        if patient not in ds.list_patients():
            print(f"\n❌ 错误: 患者 '{patient}' 不存在")
            sys.exit(1)
        
        records = ds.get_patient_records(patient)
        if record not in records:
            print(f"\n❌ 错误: 记录 '{record}' 不存在")
            print(f"\n可用记录: {', '.join(records[:5])}...")
            sys.exit(1)
        
        show_record_detail(ds, patient, record)
    
    else:
        print("\nUsage:")
        print("  python quick_view_yuquan.py                    # 数据集概览")
        print("  python quick_view_yuquan.py <patient>          # 患者详情")
        print("  python quick_view_yuquan.py <patient> <record> # 记录详情")
        sys.exit(1)


if __name__ == '__main__':
    main()
