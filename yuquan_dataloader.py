#!/usr/bin/env python3
"""
玉泉24小时SEEG数据集加载工具
数据集路径: /mnt/yuquan_data/yuquan_24h_edf

Author: Generated on 2026-01-12
"""

import numpy as np
import mne
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class RecordInfo:
    """单个记录文件的信息"""
    patient: str
    record_id: str
    has_edf: bool
    has_gpu: bool
    has_lagpat: bool
    n_events: int
    n_channels: int
    n_core_channels: int
    start_time: float
    

class YuquanDataset:
    """玉泉24小时SEEG数据集加载器"""
    
    def __init__(self, base_dir: str = '/mnt/yuquan_data/yuquan_24h_edf'):
        self.base_dir = Path(base_dir)
        if not self.base_dir.exists():
            raise FileNotFoundError(f"数据集目录不存在: {base_dir}")
        
        self.patients = sorted([d.name for d in self.base_dir.iterdir() if d.is_dir()])
        print(f"✓ 找到 {len(self.patients)} 个患者")
    
    def list_patients(self) -> List[str]:
        """列出所有患者ID"""
        return self.patients
    
    def get_patient_records(self, patient: str) -> List[str]:
        """获取某个患者的所有记录ID"""
        patient_dir = self.base_dir / patient
        if not patient_dir.exists():
            raise ValueError(f"患者不存在: {patient}")
        
        edf_files = list(patient_dir.glob('*.edf'))
        record_ids = sorted([f.stem for f in edf_files])
        return record_ids
    
    def get_record_info(self, patient: str, record_id: str) -> RecordInfo:
        """获取记录的元信息"""
        patient_dir = self.base_dir / patient
        base_path = patient_dir / record_id
        
        has_edf = (patient_dir / f'{record_id}.edf').exists()
        has_gpu = (patient_dir / f'{record_id}_gpu.npz').exists()
        has_lagpat = (patient_dir / f'{record_id}_lagPat_withFreqCent.npz').exists()
        
        n_events = 0
        n_channels = 0
        n_core_channels = 0
        start_time = 0.0
        
        if has_gpu:
            gpu = np.load(patient_dir / f'{record_id}_gpu.npz', allow_pickle=True)
            n_events = gpu['events_count'].sum()
            n_channels = len(gpu['chns_names'])
            start_time = float(gpu['start_time'])
        
        if has_lagpat:
            lag = np.load(patient_dir / f'{record_id}_lagPat_withFreqCent.npz', allow_pickle=True)
            n_core_channels = len(lag['chnNames'])
        
        return RecordInfo(
            patient=patient,
            record_id=record_id,
            has_edf=has_edf,
            has_gpu=has_gpu,
            has_lagpat=has_lagpat,
            n_events=n_events,
            n_channels=n_channels,
            n_core_channels=n_core_channels,
            start_time=start_time
        )
    
    def load_raw_edf(self, patient: str, record_id: str, preload: bool = False) -> mne.io.Raw:
        """
        加载原始EDF文件
        
        Parameters:
        -----------
        patient : str
            患者ID
        record_id : str
            记录ID
        preload : bool
            是否预加载数据到内存
        
        Returns:
        --------
        raw : mne.io.Raw
            原始信号对象
        """
        edf_path = self.base_dir / patient / f'{record_id}.edf'
        if not edf_path.exists():
            raise FileNotFoundError(f"EDF文件不存在: {edf_path}")
        
        raw = mne.io.read_raw_edf(edf_path, preload=preload, verbose=False, encoding='latin1')
        return raw
    
    def load_gpu_detections(self, patient: str, record_id: str) -> Dict:
        """
        加载GPU检测结果
        
        Returns:
        --------
        data : dict
            - 'whole_dets': (120,) 每个通道的事件列表 [[start, end], ...]
            - 'chns_names': (120,) 通道名
            - 'events_count': (120,) 每个通道的事件数
            - 'start_time': float Unix时间戳
        """
        gpu_path = self.base_dir / patient / f'{record_id}_gpu.npz'
        if not gpu_path.exists():
            raise FileNotFoundError(f"GPU检测文件不存在: {gpu_path}")
        
        data = np.load(gpu_path, allow_pickle=True)
        return {
            'whole_dets': data['whole_dets'],
            'chns_names': data['chns_names'],
            'events_count': data['events_count'],
            'start_time': data['start_time']
        }
    
    def load_lagpat(self, patient: str, record_id: str, with_freq: bool = True) -> Dict:
        """
        加载滞后模式矩阵
        
        Parameters:
        -----------
        with_freq : bool
            是否加载带频率信息的版本
        
        Returns:
        --------
        data : dict
            - 'lagPatRaw': (n_core_channels, n_events) 滞后时间矩阵
            - 'lagPatRank': (n_core_channels, n_events) 滞后排名矩阵
            - 'lagPatFreq': (n_core_channels, n_events) 频率矩阵 (if with_freq)
            - 'eventsBool': (n_core_channels, n_events) 事件掩码
            - 'chnNames': (n_core_channels,) 核心通道名
            - 'start_t': float Unix时间戳
        """
        suffix = '_lagPat_withFreqCent.npz' if with_freq else '_lagPat.npz'
        lag_path = self.base_dir / patient / f'{record_id}{suffix}'
        
        if not lag_path.exists():
            raise FileNotFoundError(f"LagPat文件不存在: {lag_path}")
        
        data = np.load(lag_path, allow_pickle=True)
        result = {
            'lagPatRaw': data['lagPatRaw'],
            'lagPatRank': data['lagPatRank'],
            'eventsBool': data['eventsBool'],
            'chnNames': data['chnNames'],
            'start_t': data['start_t']
        }
        
        if with_freq and 'lagPatFreq' in data:
            result['lagPatFreq'] = data['lagPatFreq']
        
        return result
    
    def load_event_times(self, patient: str, record_id: str) -> np.ndarray:
        """
        加载事件时间窗
        
        Returns:
        --------
        times : np.ndarray
            shape (n_events, 2), times[:, 0]=开始时间, times[:, 1]=结束时间
        """
        times_path = self.base_dir / patient / f'{record_id}_packedTimes.npy'
        if not times_path.exists():
            raise FileNotFoundError(f"PackedTimes文件不存在: {times_path}")
        
        times = np.load(times_path, allow_pickle=True)
        return times
    
    def load_patient_summary(self, patient: str) -> Dict:
        """
        加载患者级汇总信息
        
        Returns:
        --------
        data : dict
            - 'events_count': (120,) 所有记录累加的事件数
            - 'chns_names': (120,) 通道名
        """
        refine_path = self.base_dir / patient / '_refineGpu.npz'
        if not refine_path.exists():
            raise FileNotFoundError(f"患者汇总文件不存在: {refine_path}")
        
        data = np.load(refine_path, allow_pickle=True)
        return {
            'events_count': data['events_count'],
            'chns_names': data['chns_names']
        }
    
    def load_channel_selection(self, patient: str) -> Dict:
        """
        加载通道筛选结果
        
        Returns:
        --------
        data : dict
            - 'hist_meanX': (n_core_channels,) 统计量
            - 'pick_chns': (n_core_channels,) 筛选的核心通道
        """
        hist_path = self.base_dir / patient / 'hist_meanX.npz'
        if not hist_path.exists():
            raise FileNotFoundError(f"统计文件不存在: {hist_path}")
        
        data = np.load(hist_path, allow_pickle=True)
        return {
            'hist_meanX': data['hist_meanX'],
            'pick_chns': data['pick_chns']
        }
    
    def get_dataset_statistics(self) -> Dict:
        """获取整个数据集的统计信息"""
        total_edfs = 0
        total_gpus = 0
        total_events = 0
        patients_with_processing = []
        
        for patient in self.patients:
            records = self.get_patient_records(patient)
            total_edfs += len(records)
            
            patient_has_processing = False
            for record in records:
                info = self.get_record_info(patient, record)
                if info.has_gpu:
                    total_gpus += 1
                    total_events += info.n_events
                    patient_has_processing = True
            
            if patient_has_processing:
                patients_with_processing.append(patient)
        
        return {
            'n_patients': len(self.patients),
            'n_patients_processed': len(patients_with_processing),
            'n_edf_files': total_edfs,
            'n_gpu_files': total_gpus,
            'total_events': total_events,
            'patients_processed': patients_with_processing
        }
    
    def plot_patient_overview(self, patient: str, figsize=(15, 8)):
        """
        可视化某个患者的HFO事件分布
        
        Parameters:
        -----------
        patient : str
            患者ID
        figsize : tuple
            图像尺寸
        """
        # 加载患者汇总数据
        try:
            summary = self.load_patient_summary(patient)
        except FileNotFoundError:
            print(f"患者 {patient} 尚未完成处理")
            return
        
        events_count = summary['events_count']
        chns_names = summary['chns_names']
        
        # 加载核心通道
        try:
            selection = self.load_channel_selection(patient)
            core_chns = selection['pick_chns']
        except FileNotFoundError:
            core_chns = []
        
        # 创建图形
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # 子图1: 事件数分布
        ax1 = axes[0]
        x = np.arange(len(events_count))
        colors = ['red' if ch in core_chns else 'steelblue' for ch in chns_names]
        ax1.bar(x, events_count, color=colors, alpha=0.7)
        ax1.set_xlabel('Channel Index')
        ax1.set_ylabel('Event Count')
        ax1.set_title(f'Patient: {patient} - HFO Event Distribution\n(Red: Core Channels)')
        ax1.grid(axis='y', alpha=0.3)
        
        # 子图2: 对数尺度
        ax2 = axes[1]
        events_nonzero = np.where(events_count > 0, events_count, 1)
        ax2.bar(x, events_nonzero, color=colors, alpha=0.7, log=True)
        ax2.set_xlabel('Channel Index')
        ax2.set_ylabel('Event Count (log scale)')
        ax2.set_title('Log Scale View')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def print_summary(self):
        """打印数据集摘要"""
        stats = self.get_dataset_statistics()
        
        print("=" * 70)
        print("玉泉24小时SEEG数据集概览")
        print("=" * 70)
        print(f"患者总数:        {stats['n_patients']}")
        print(f"已处理患者:      {stats['n_patients_processed']}")
        print(f"EDF记录文件:     {stats['n_edf_files']}")
        print(f"GPU检测结果:     {stats['n_gpu_files']}")
        print(f"总HFO事件数:     {stats['total_events']:,}")
        print(f"平均每小时事件:   {stats['total_events'] / (stats['n_gpu_files'] * 2):.0f}")
        print("=" * 70)
        
        print("\n已处理的患者:")
        for i, patient in enumerate(stats['patients_processed'], 1):
            records = self.get_patient_records(patient)
            print(f"  {i:2d}. {patient:20s} ({len(records)} 记录)")


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == '__main__':
    # 初始化数据集
    ds = YuquanDataset()
    
    # 打印概览
    ds.print_summary()
    
    print("\n" + "=" * 70)
    print("示例: 加载第一个患者的第一条记录")
    print("=" * 70)
    
    patient = ds.list_patients()[0]
    records = ds.get_patient_records(patient)
    record = records[0]
    
    print(f"\n患者: {patient}")
    print(f"记录: {record}")
    
    # 获取记录信息
    info = ds.get_record_info(patient, record)
    print(f"\n记录信息:")
    print(f"  - 有EDF文件:    {info.has_edf}")
    print(f"  - 有GPU检测:    {info.has_gpu}")
    print(f"  - 有LagPat:     {info.has_lagpat}")
    print(f"  - 事件总数:     {info.n_events}")
    print(f"  - 通道数:       {info.n_channels}")
    print(f"  - 核心通道数:   {info.n_core_channels}")
    
    # 加载GPU检测结果
    if info.has_gpu:
        print("\n加载GPU检测结果...")
        gpu_data = ds.load_gpu_detections(patient, record)
        print(f"  通道名 (前10): {gpu_data['chns_names'][:10]}")
        print(f"  事件数 (前10): {gpu_data['events_count'][:10]}")
        print(f"  第一个通道的前3个事件:")
        first_ch_events = gpu_data['whole_dets'][0]
        for i, event in enumerate(first_ch_events[:3]):
            print(f"    事件{i+1}: {event[0]:.3f}s ~ {event[1]:.3f}s (持续{(event[1]-event[0])*1000:.1f}ms)")
    
    # 加载滞后模式
    if info.has_lagpat:
        print("\n加载滞后模式矩阵...")
        lag_data = ds.load_lagpat(patient, record)
        print(f"  核心通道: {lag_data['chnNames']}")
        print(f"  滞后矩阵形状: {lag_data['lagPatRaw'].shape}")
        print(f"  频率矩阵形状: {lag_data['lagPatFreq'].shape}")
        
        # 事件时间
        times = ds.load_event_times(patient, record)
        print(f"\n事件时间窗:")
        print(f"  总事件数: {len(times)}")
        print(f"  第一个事件: {times[0][0]:.3f}s ~ {times[0][1]:.3f}s")
        print(f"  最后事件:   {times[-1][0]:.3f}s ~ {times[-1][1]:.3f}s")
    
    # 可视化
    print("\n" + "=" * 70)
    print("生成可视化...")
    print("=" * 70)
    fig = ds.plot_patient_overview(patient)
    if fig:
        output_path = f'/Users/leijiaxin/Desktop/Pys/Epilepsia/{patient}_overview.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ 已保存图像: {output_path}")
        plt.close(fig)
