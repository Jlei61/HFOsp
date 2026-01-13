# 玉泉24小时SEEG数据集结构分析

**分析日期**: 2026-01-12  
**数据路径**: `/Volumes/Elements/yuquan_24h_edf`

---

## 【核心判断】✅ 数据集结构清晰

这是一个**预处理完成的HFO检测数据集**，数据组织合理，每个文件职责明确。

---

## 1. 数据集概览

| 指标 | 数值 |
|------|------|
| 患者数量 | 21人 |
| EDF原始文件 | 260个 (约520小时记录) |
| GPU检测结果 | 176个 |
| 已检测HFO事件 | 约70万+ |
| 采样率 | 2000 Hz |
| 单文件时长 | 约2小时 |

**注意**: 有5个患者(pengzihang, songzishuo, zhangbichen, zhangkexuan, zhaochenxi, zhaojinrui, zhourongxuan)只有EDF原始文件，尚未完成GPU检测。

---

## 2. 目录结构

```
/Volumes/Elements/yuquan_24h_edf/
├── chengshuai/          # 患者1
│   ├── FC10477Q.edf                           # 原始SEEG信号 (2小时, 145通道)
│   ├── FC10477Q_gpu.npz                       # GPU检测的HFO事件
│   ├── FC10477Q_lagPat.npz                    # 滞后模式矩阵
│   ├── FC10477Q_lagPat_withFreqCent.npz       # 滞后模式+频率中心
│   ├── FC10477Q_packedTimes.npy               # 事件时间窗
│   ├── FC10477Q_packedTimes_withFreqCent.npy  # 事件时间窗+频率
│   ├── FC10477R.edf                           # 下一个2小时记录
│   ├── ...                                    # (共12个2小时记录)
│   ├── _refineGpu.npz                         # 患者级汇总
│   ├── hist_meanX.npz                         # 统计信息
│   ├── pick_chns.png                          # 筛选通道可视化
│   └── refine_hist.png                        # 统计直方图
├── chenziyang/          # 患者2
├── ...
└── zhourongxuan/        # 患者21
```

---

## 3. 文件类型详解

### 3.1 原始数据: `*.edf`

**内容**: SEEG侵入式脑电信号原始记录

```python
# 示例: FC10477Q.edf
采样率: 2000 Hz
通道数: 145个 (包含SEEG电极 + ECG/EMG)
时长: 7200秒 (2小时)
开始时间: 2019-09-20 22:19:56 (Unix: 1569017996.0)

通道命名:
- EEG A1-Ref, A2-Ref, ...    # 参考电极
- POL A3, A4, A5, ...        # 极性记录 (主要SEEG通道)
- POL K3, K5, K6, ...        # 不同电极串
- POL ECG, EMG1, EMG2        # 生理信号
```

---

### 3.2 HFO检测结果: `*_gpu.npz`

**内容**: GPU加速的高频振荡(HFO)事件检测结果

```python
键名: ['whole_dets', 'chns_names', 'events_count', 'start_time']

whole_dets:   # 每个通道检测到的事件时间段
  - shape: (120,)  # 120个有效通道
  - dtype: object
  - 内容: 每个元素是 [[start1, end1], [start2, end2], ...] (秒)
  - 示例: [[282.34875, 282.42375], [343.85875, 343.91125], ...]

chns_names:   # 通道名列表
  - shape: (120,)
  - dtype: <U3
  - 示例: ['A1', 'A2', 'A3', ..., 'K10']

events_count: # 每个通道的事件数量
  - shape: (120,)
  - dtype: int64
  - 示例: [13, 1, 1, 2, ..., 11528, 9801, ...]
  - 注意: 某些通道事件极多 (>10000), 可能是伪迹

start_time:   # Unix时间戳
  - scalar: 1569017996.0
```

**数据特点**:
- 只保留了**120个有效通道** (原始145个)
- 移除了参考电极和生理信号通道
- 事件数量在通道间差异极大: 0 ~ 11528

---

### 3.3 滞后模式矩阵: `*_lagPat.npz` / `*_lagPat_withFreqCent.npz`

**内容**: 用于分析事件传播模式的时空矩阵

```python
# *_lagPat.npz (基础版本)
键名: ['lagPatRaw', 'lagPatRank', 'eventsBool', 'chnNames', 'start_t']

lagPatRaw:    # 原始滞后值
  - shape: (8, 2601)  # 8个筛选通道 × 2601个事件
  - dtype: float64

lagPatRank:   # 滞后值排名
  - shape: (8, 2601)
  - dtype: int64

eventsBool:   # 事件布尔掩码
  - shape: (8, 2601)
  - dtype: float64

chnNames:     # 筛选后的通道
  - shape: (8,)
  - 示例: ['E11', 'K3', 'K5', 'K6', 'K7', 'K8', 'K9', 'K10']

# *_lagPat_withFreqCent.npz (增强版本)
额外包含:
lagPatFreq:   # 频率信息
  - shape: (8, 2601)
  - dtype: float64
```

**关键洞察**:
- 从120个通道**筛选出8个核心通道** → 降维90%
- 2601个事件是跨所有通道的**时间对齐事件池**
- `lagPatRaw[i, j]` 表示第i个通道在第j个事件中的滞后时间

---

### 3.4 事件时间窗: `*_packedTimes.npy`

**内容**: 所有检测到的事件时间边界

```python
shape: (2601, 2)  # 2601个事件 × [start, end]
dtype: float64

示例:
[[  0.342,   0.842],   # 事件1: 0.342s ~ 0.842s (持续500ms)
 [  3.194,   3.694],   # 事件2
 [  5.660,   6.160],   # 事件3
 ...
 [7199.xxx, 7199.xxx]] # 最后一个事件
```

**特点**:
- 事件窗长度固定为**500ms** (0.5秒)
- 时间相对于记录开始时刻 (start_time)

---

### 3.5 患者级汇总: `_refineGpu.npz`

**内容**: 整合该患者所有记录的统计信息

```python
键名: ['events_count', 'chns_names']

events_count: # 所有记录文件累加的事件数
  - shape: (120,)
  - dtype: int64

chns_names:   # 通道名
  - shape: (120,)
```

---

### 3.6 统计信息: `hist_meanX.npz`

**内容**: 通道筛选和质量控制的统计量

```python
键名: ['hist_meanX', 'pick_chns']

hist_meanX:   # 某种均值统计 (可能是事件密度/质量分数)
  - shape: (8,)
  - dtype: float64
  - 示例: [6.17, 6.19, 3.99, 1.00, 3.01, 3.01, 5.03, 6.14]

pick_chns:    # 筛选出的8个核心通道
  - shape: (8,)
  - 示例: ['E11', 'K3', 'K5', 'K6', 'K7', 'K8', 'K9', 'K10']
```

---

## 4. 数据处理流程推断

```
[原始EDF 2000Hz, 145通道]
         ↓
    带通滤波 (80-250Hz 或 250-500Hz)
         ↓
    GPU加速HFO检测 (阈值法?)
         ↓
   [*_gpu.npz: 120通道, 数千~数万事件]
         ↓
    通道质量评估 + 事件对齐
         ↓
   筛选出8个核心通道 + 2601个代表性事件
         ↓
   [*_lagPat.npz: 8×2601滞后模式矩阵]
         ↓
    频率中心计算
         ↓
   [*_lagPat_withFreqCent.npz]
```

---

## 5. 关键数据关系

```
whole_dets (120通道 × 变长事件列表)
    ↓ 时间对齐 + 筛选
packedTimes (2601个统一事件 × [start, end])
    ↓ 通道筛选 (120 → 8)
lagPatRaw (8通道 × 2601事件)
```
