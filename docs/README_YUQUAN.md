# 玉泉24小时SEEG数据集分析工具包

**数据集**: 玉泉医院24小时连续SEEG记录 + HFO检测结果  
**路径**: `/Volumes/Elements/yuquan_24h_edf`  
**生成日期**: 2026-01-12

---

## 📊 数据集概览

- **21个患者**, 260个EDF记录 (约520小时)
- **14个患者已完成HFO检测** (176个记录, 352小时)
- **172万+HFO事件** (高频振荡)
- **采样率**: 2000 Hz
- **主要频段**: Ripple (80-250Hz)

---

## 🚀 快速开始

### 1. 查看数据集概览

```bash
python quick_view_yuquan.py
```

输出:
```
玉泉24小时SEEG数据集
患者数量: 21 (已处理: 14)
HFO事件: 8,349,218
平均事件率: 23719 事件/小时

已处理的患者 (按事件数排序):
   1. zhangjiaqi            445,472 ████████████████████
   2. huangwanling          309,152 ████████████████
   ...
```

### 2. 查看某个患者

```bash
python quick_view_yuquan.py chengshuai
```

### 3. 查看某条记录详情

```bash
python quick_view_yuquan.py chengshuai FC10477Q
```

输出包括:
- 通道统计
- 事件时间分布
- 核心通道频率分析
- Ripple vs Fast Ripple比例

---

## 📁 文件说明

### 工具脚本

| 文件 | 功能 | 用途 |
|------|------|------|
| `yuquan_dataloader.py` | 数据加载器 | 核心类 `YuquanDataset` |
| `quick_view_yuquan.py` | 快速查看工具 | 命令行浏览数据 |
| `yuquan_analysis.py` | 深度分析脚本 | 生成统计和可视化 |

### 文档

| 文件 | 内容 |
|------|------|
| `yuquan_24h_dataset_structure.md` | 数据结构详细说明 |
| `YUQUAN_ANALYSIS_REPORT.md` | 完整分析报告 |
| `README_YUQUAN.md` | 本文件 |

### 可视化结果

| 文件 | 内容 |
|------|------|
| `chengshuai_overview.png` | 患者事件分布 |
| `chengshuai_FC10477Q_temporal.png` | 事件时间演化 |
| `chengshuai_FC10477Q_propagation.png` | 滞后模式和频率 |
| `all_patients_summary.png` | 所有患者统计 |

---

## 💻 编程接口

### 基本用法

```python
from yuquan_dataloader import YuquanDataset

# 初始化
ds = YuquanDataset()

# 列出所有患者
patients = ds.list_patients()  # ['chengshuai', 'chenziyang', ...]

# 获取某个患者的记录
records = ds.get_patient_records('chengshuai')  # ['FC10477Q', 'FC10477R', ...]

# 获取记录信息
info = ds.get_record_info('chengshuai', 'FC10477Q')
print(f"事件数: {info.n_events}")
```

### 加载数据

```python
# 1. 加载GPU检测结果
gpu_data = ds.load_gpu_detections('chengshuai', 'FC10477Q')
events_per_channel = gpu_data['events_count']  # (120,) 每个通道的事件数
channel_names = gpu_data['chns_names']          # (120,) 通道名
whole_dets = gpu_data['whole_dets']            # (120,) 每个通道的事件时间列表

# 2. 加载滞后模式和频率
lag_data = ds.load_lagpat('chengshuai', 'FC10477Q', with_freq=True)
lag_matrix = lag_data['lagPatRaw']   # (n_core_ch, n_events) 滞后时间
freq_matrix = lag_data['lagPatFreq'] # (n_core_ch, n_events) 频率
core_channels = lag_data['chnNames'] # 核心通道名

# 3. 加载事件时间窗
times = ds.load_event_times('chengshuai', 'FC10477Q')
event_starts = times[:, 0]  # 事件开始时间
event_ends = times[:, 1]    # 事件结束时间

# 4. 加载患者汇总
summary = ds.load_patient_summary('chengshuai')
total_events = summary['events_count'].sum()
```

### 可视化

```python
# 生成患者概览图
fig = ds.plot_patient_overview('chengshuai')
fig.savefig('output.png')
```

---

## 🧠 EDF预处理与波形绘图（本项目）

我们不从 `*_gpu.npz` 反推EDF是否做过重参考。需要什么参考方式就显式指定：
- `reference='bipolar'`: 同一电极串相邻触点差分，**通道命名为`A1-A2`**（避免与单极`A1`混淆）
- `reference='car'`: 每串CAR
- `reference='none'`: 保持EDF原始参考

```python
from src.preprocessing import SEEGPreprocessor
from src.visualization import plot_from_result, plot_shaft_channels

edf = '/Volumes/Elements/yuquan_24h_edf/chengshuai/FC10477Q.edf'

# 1) Bipolar 全通道（100s）
bip = SEEGPreprocessor(reference='bipolar', crop_seconds=101).run(edf)
plot_from_result(bip, start_sec=0, duration_sec=100, channels='all')

# 2) CAR 全通道（100s）
car = SEEGPreprocessor(reference='car', crop_seconds=101).run(edf)
plot_from_result(car, start_sec=0, duration_sec=100, channels='all')

# 3) 单电极串（例：K）
plot_shaft_channels(bip.data, bip.sfreq, bip.ch_names, shaft='K', start_sec=0, duration_sec=30,
                    reference_type=bip.reference_type)
```

如果你需要“完全复现某个`*_gpu.npz`里的通道集合”，用显式通道表，不要硬编码“去掉末端N个触点”：

```python
import numpy as np
gpu = np.load('/Volumes/Elements/yuquan_24h_edf/chengshuai/FC10477Q_gpu.npz', allow_pickle=True)
include = [str(x) for x in gpu['chns_names']]
res = SEEGPreprocessor(reference='none', include_channels=include, crop_seconds=101).run(edf)
```

---

## 📈 数据结构

### 目录组织

```
/Volumes/Elements/yuquan_24h_edf/
├── chengshuai/                    # 患者1
│   ├── FC10477Q.edf              # 原始SEEG (2小时, 2000Hz, 145通道)
│   ├── FC10477Q_gpu.npz          # GPU检测: 120通道, 数万事件
│   ├── FC10477Q_lagPat_withFreqCent.npz  # 8通道 × 2601事件
│   ├── FC10477Q_packedTimes.npy  # 2601个事件的时间窗
│   ├── FC10477R.edf              # 下一个2小时
│   ├── ...
│   ├── _refineGpu.npz            # 患者级汇总
│   └── hist_meanX.npz            # 通道筛选结果
├── chenziyang/                    # 患者2
├── ...
└── zhourongxuan/                  # 患者21
```

### 文件类型

| 文件 | 内容 | 形状 |
|------|------|------|
| `*.edf` | 原始SEEG信号 | 2000Hz × 7200s × 145通道 |
| `*_gpu.npz` | HFO检测结果 | 120通道 × 变长事件列表 |
| `*_lagPat*.npz` | 滞后模式矩阵 | 8核心通道 × 2601事件 |
| `*_packedTimes.npy` | 事件时间窗 | 2601事件 × [start, end] |
| `_refineGpu.npz` | 患者汇总 | 120通道累计统计 |
| `hist_meanX.npz` | 通道筛选 | 8个核心通道的质量分数 |

**数据降维**: 145通道 → 120通道(有效) → 8通道(核心)

---

## 🔍 关键发现

### 时间特征

- **事件间隔**: 平均1-3秒, 呈长尾分布
- **集簇现象**: 事件成串出现(burst), 中间有静默期
- **事件持续**: 固定500ms窗口

### 空间特征

- **幂律分布**: 少数通道贡献大量事件 (最高可达3.7万/通道)
- **活跃通道**: 70-100%通道有事件
- **核心通道**: 筛选出4-41个高质量通道

### 频率特征

- **主要频段**: Ripple (80-250Hz), 集中在80-90Hz
- **频率稳定**: 标准差只有3-4Hz
- **Fast Ripple**: 在该数据集中极少 (<1%)

### 患者差异

- **事件数**: 5千 ~ 44.5万 (90倍差异)
- **空间分布**: 从高度局限(4核心通道)到广泛(41核心通道)
- **通道同步性**: 相邻通道高度相关(r>0.95)

---

## 📊 典型案例

### 患者: chengshuai, 记录: FC10477Q

```
时长: 2小时
原始通道: 145 → 有效通道: 120 → 核心通道: 8
原始事件: 46,738 → 对齐事件: 2,601

核心通道: ['E11', 'K3', 'K5', 'K6', 'K7', 'K8', 'K9', 'K10']

频率分布: 80-90Hz (Ripple低频段)
事件间隔: 中位1.9秒, 平均2.8秒

空间分布: 
  - 最活跃通道: 11,528事件
  - 10个通道无事件
  - 幂律分布明显
```
