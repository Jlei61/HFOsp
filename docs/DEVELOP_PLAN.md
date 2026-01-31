# HFOsp 开发计划

**项目目标**: 复现并扩展 HFO (高频振荡) 分析流程，验证 Source-Sink 理论  
**数据集**: 玉泉24小时SEEG数据集  
**数据路径**: `/mnt/yuquan_data/yuquan_24h_edf`  
**更新日期**: 2026-01-16

---

## 0. 核心架构原则

### 数据流设计哲学

> "Bad programmers worry about the code. Good programmers worry about data structures." — Linus Torvalds

**单一职责原则**：每个模块只做一件事并做好。

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              HFOsp 数据流架构                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  [原始EDF]                                                                      │
│      │                                                                          │
│      ▼                                                                          │
│  ┌─────────────────┐                                                            │
│  │ preprocessing   │ ──→ 预处理：重参考、滤波、通道选择                          │
│  └────────┬────────┘                                                            │
│           │ (data, sfreq, ch_names)                                             │
│           ▼                                                                     │
│  ┌─────────────────┐                                                            │
│  │  hfo_detector   │ ──→ HFO检测：单通道事件列表                                 │
│  └────────┬────────┘                                                            │
│           │ Dict[ch_name → [(start, end), ...]]                                 │
│           ▼                                                                     │
│  ┌─────────────────────────────────────────────────────┐                        │
│  │           group_event_analysis                       │                       │
│  │  ┌─────────────────────────────────────────────┐    │                        │
│  │  │ Step 1: 群体事件窗口构建                      │    │                       │
│  │  │   - build_windows_from_detections           │    │                        │
│  │  │   - 输出: EventWindow list                   │    │                       │
│  │  └──────────────────┬──────────────────────────┘    │                        │
│  │                     ▼                               │                        │
│  │  ┌─────────────────────────────────────────────┐    │                        │
│  │  │ Step 2: 预计算 Envelope + Bandpass 缓存       │    │                       │
│  │  │   - precompute_envelope_cache               │    │                        │
│  │  │   - 存储: *_envCache.npz                    │    │                        │
│  │  └──────────────────┬──────────────────────────┘    │                        │
│  │                     ▼                               │                        │
│  │  ┌─────────────────────────────────────────────┐    │                        │
│  │  │ Step 3: 质心+TF分析 → 存储中间结果            │    │                       │
│  │  │   - compute_group_analysis_results          │    │                        │
│  │  │   - 存储: *_groupAnalysis.npz               │    │ ← 关键新增！           │
│  │  └──────────────────┬──────────────────────────┘    │                        │
│  │                     │                               │                        │
│  └─────────────────────┼───────────────────────────────┘                        │
│                        │                                                        │
│           ┌────────────┴────────────┐                                           │
│           ▼                         ▼                                           │
│  ┌─────────────────┐      ┌─────────────────┐                                   │
│  │ network_analysis│      │  visualization  │ ← 只读取中间结果，不做计算！       │
│  └─────────────────┘      └─────────────────┘                                   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 中间结果存储规范

**存储位置**: `/mnt/yuquan_data/yuquan_24h_edf/<patient>/`（与EDF同目录）

| 文件名模式 | 内容 | 来源模块 |
|-----------|------|----------|
| `*_gpu.npz` | 历史GPU检测结果 | (已有) |
| `*_packedTimes.npy` | 事件窗口 | (已有) |
| `*_lagPat*.npz` | 历史lag矩阵 | (已有) |
| `*_envCache_{band}_{ref}.npz` | **Envelope+带通信号缓存** | `group_event_analysis` |
| `*_groupAnalysis.npz` | **质心+TF+lag 完整分析结果** | `group_event_analysis` |

### `*_groupAnalysis.npz` 结构（核心新增）

```python
{
    # === 元数据 ===
    'sfreq': np.array([2000.0]),           # 采样率
    'band': np.array(['ripple']),          # 频段
    'ch_names': np.array([...]),           # 核心通道名 (n_ch,)
    'window_sec': np.array([0.5]),         # 窗口长度
    'n_events': np.array([2601]),          # 事件数
    'n_channels': np.array([8]),           # 通道数
    
    # === 事件窗口 ===
    'event_windows': np.array([...]),      # (n_events, 2) [start, end] 秒
    
    # === 质心分析 ===
    'centroid_time': np.array([...]),      # (n_ch, n_events) 时间质心(相对窗口起始)
    'centroid_freq': np.array([...]),      # (n_ch, n_events) TF 2D质心的频率分量
    'events_bool': np.array([...]),        # (n_ch, n_events) 通道是否参与
    
    # === Lag/Rank 分析 ===
    'lag_raw': np.array([...]),            # (n_ch, n_events) 相对lag(对齐到最早通道)
    'lag_rank': np.array([...]),           # (n_ch, n_events) 排名 (0=最早)
    
    # === 可选：TF 变换结果 (用于高级可视化) ===
    'tf_power_per_event': np.array([...]), # (n_ch, n_events, n_freq, n_time) 可选
}
```

### Visualization 职责边界（铁律）

**visualization.py 只负责**：
1. 读取 `*_groupAnalysis.npz` 或 `*_envCache.npz`
2. 根据参数选择子集（通道、事件）
3. 调用 matplotlib 画图

**visualization.py 禁止**：
- ❌ 计算 STFT
- ❌ 计算质心
- ❌ 计算 Hilbert envelope
- ❌ 任何超过 100 行的数据处理

---

## 1. 数据集概览

| 指标 | 数值 |
|------|------|
| 患者数量 | 21人 (14人已处理) |
| EDF文件 | 260个 (约520小时) |
| 采样率 | 2000 Hz |
| 单文件时长 | 约2小时 |
| 通道数 | 145 (原始, 含非SEEG) → ~140 (SEEG) → ~130 (Bipolar参考后对) → 8 (核心, lagPat) |
| HFO事件 | ~170万+ |

**已有中间结果**:
- `*_gpu.npz`: 120通道 × 变长事件列表
- `*_lagPat.npz`: 8通道 × 2601事件 滞后矩阵  
- `*_packedTimes.npy`: 事件时间窗 [start, end]
- `hist_meanX.npz`: 核心通道筛选结果

---

## 2. 现有文件结构

```
HFOsp/
├── docs/
│   ├── DEVELOP_PLAN.md          # 本文件
│   ├── README_YUQUAN.md         # 数据集使用说明
│   └── yuquan_24h_dataset_structure.md  # 数据结构详解
├── src/
│   ├── __init__.py              # ✅ 模块导出
│   ├── preprocessing.py         # ✅ 预处理Pipeline
│   ├── hfo_detector.py          # ✅ HFO检测器
│   ├── group_event_analysis.py  # ✅ 群体事件分析（核心计算中心）
│   ├── network_analysis.py      # [待开发] 网络分析
│   ├── visualization.py         # ⚠️ 可视化（需重构：剥离计算逻辑）
│   └── utils/
│       └── bqk_utils.py         # 基础工具函数
├── datasets/
│   ├── quick_view_yuquan.py     # 数据快速查看
│   └── ...
├── scripts/
│   ├── visualize_event_waveforms.py
│   └── yuquan_analysis.py
├── notebook/
│   └── chengshuai_hfo_analysis.ipynb  # ✅ 示例notebook
├── yuquan_dataloader.py         # 数据加载器
└── requirements.txt             # [待创建]
```

---

## 3. 核心数据流

```
原始EDF (145通道, 2000Hz, 2h) 
    │
    ├─ 电极名称解析 (去POL/EEG前缀, 排除EMG/ECG)
    │
    ├─ Bipolar重参考 (同电极串相邻触点差分)
    │   └─ 严格规则: 仅同前缀 + 数字连续/相邻
    │   └─ 145 → 约120通道
    │
    ├─ 重采样 (Ripple: 1000Hz, FR: 保留2000Hz)
    │
    ├─ 滤波 (Notch 50Hz谐波 + Bandpass)
    │
    └─ 通道质量检查 (z-score异常, 方差检测)
         │
         ▼
    滤波后数据 (120通道, 1000/2000Hz)
         │
    ┌────┴────┐
    │         │
  Ripple    Fast-Ripple
 80-250Hz   250-500Hz
    │         │
    └────┬────┘
         │
    HFO事件检测 (Hilbert包络 + 双阈值)
         │
         ├─ Ictal段落检测 (能量爆发 + 持续>3s)
         ├─ 背景基线估计 (剔除Ictal后计算)
         └─ 分段自适应阈值
         │
         ▼
    单通道HFO事件列表
         │
    群体事件识别 (500ms窗口内多通道共现)
         │
         ├─ 滞后计算 (Hilbert包络互相关)
         ├─ 核心通道筛选 (密度/Out-Strength/首发)
         └─ 频率中心计算
         │
         ▼
    滞后矩阵 (8核心通道 × N事件)
         │
    有向网络构建
         │
         ├─ 容积传导剔除 (距离+滞后条件)
         ├─ Interictal/Ictal 分段网络
         └─ 图论指标计算
         │
         ▼
    Source-Sink 验证
```

---

## 4. 模块开发计划

### 模块1: src/preprocessing.py ✅ 完成（Phase 1 重构 2026-01-30）

| 功能 | 说明 | 状态 |
|------|------|------|
| 1.1 EDF读取 | 加载EDF, 清洗电极名称, encoding='latin1' | ✅ |
| 1.2 电极名称解析 | 正则提取 (prefix, number), 支持A'5格式 | ✅ |
| 1.3 重参考策略 | **显式** bipolar / car / none（auto=兼容别名→bipolar；不做任何"推断"） | ✅ |
| 1.4 通道选择 | **显式** include/exclude channels（用于复现GPU通道列表；不硬编码"去掉末端N个触点"） | ✅ |
| 1.5 重采样 | Ripple→1000Hz, FR→2000Hz | ✅ |
| 1.6 滤波 | Notch + 可选Bandpass, GPU加速支持 | ✅ |
| 1.7 通道质量检查 | z-score, 方差, 伪迹标记 | ✅ |
| **1.8 FilterBackend架构** | **抽象接口 + CPU/GPU实现分离** | ✅ **NEW** |

#### Phase 1 重构总结（2026-01-30）

**核心改进：消除重复的条件判断**

```python
# Before: 散落各处的 GPU 判断
if self.use_gpu and HAS_GPU:
    return self._apply_filters_gpu(...)  # 70+ lines
else:
    return self._apply_filters_cpu(...)  # 30+ lines

# After: 初始化时决定后端，运行时零判断
self.filter_backend = GpuFilterBackend() if use_gpu else CpuFilterBackend()

def _apply_filters(self, data, sfreq):
    if self.notch_freqs:
        data = self.filter_backend.apply_notch(data, sfreq, self.notch_freqs)
    if self.bandpass:
        data = self.filter_backend.apply_bandpass(data, sfreq, *self.bandpass)
    return data
```

**删除的废代码**（零破坏性）：
- ❌ `PreBipolarDetector` (15行) - 空壳类，总是返回 False
- ❌ `validate_against_gpu_results()` (12行) - 运行时抛异常的废函数
- ❌ `exclude_last_n` 参数 - 从所有类和函数签名中移除

**新增的架构组件**：

```python
# 抽象接口
class FilterBackend:
    def apply_notch(data, sfreq, freqs): ...
    def apply_bandpass(data, sfreq, low, high): ...

# CPU 实现（scipy filtfilt）
class CpuFilterBackend(FilterBackend): ...

# GPU 实现（CuPy + chunked processing）
class GpuFilterBackend(FilterBackend):
    def __init__(self, chunk_sec=20.0): ...
    # 自动处理 GPU OOM - 20s chunks + reflect padding
```

**扩展新后端的方法**：

```python
# 例如：添加 Apple Metal 支持
class MetalFilterBackend(FilterBackend):
    def apply_notch(self, data, sfreq, freqs):
        # 使用 PyTorch MPS 或 Metal Performance Shaders
        ...
    
    def apply_bandpass(self, data, sfreq, low, high):
        ...

# 使用：
preprocessor = SEEGPreprocessor(
    use_gpu=False,  # 不使用 CUDA
    # 手动替换 backend（高级用法）
)
preprocessor.filter_backend = MetalFilterBackend()
```

**收益**：
- ✅ 代码行数: 1223 → 1199 (-24行，质量提升显著)
- ✅ GPU 条件分支: 4处 → 0处
- ✅ 可扩展性: 添加新 backend (Metal/ROCm/OpenCL) 只需实现 2 个方法
- ✅ 可测试性: 每个 backend 可独立单元测试
- ✅ 零破坏性: 所有外部调用接口保持向后兼容

---

#### 关键技术决策（历史记录）

- **不再猜**：不再根据"某些contact缺失"去推断EDF是否已做bipolar；那是通道选择策略，不是重参考证据。
- **重参考策略（显式）**:
  - `'bipolar'`: 同shaft相邻触点差分；**命名为明确的`A1-A2`**，避免与单极通道混淆
  - `'car'`: 每个shaft内部做CAR
  - `'none'`: 保持EDF原始参考（单极/参考电极体系），不做任何推断
- **GPU通道差异来源（已确认，chengshuai/FC10477Q）**：
  - `*_gpu.npz` 的 `chns_names` 是 EDF 清洗后通道的子集；
  - 差异主要来自**通道选择/剔除策略**（例如每个shaft缺少末端若干contact），并非bipolar推断依据；
  - 若要复现GPU通道集合，使用 `include_channels=gpu['chns_names']`（显式）。

---

### 模块2: src/hfo_detector.py ✅ 完成（Phase 2 重构 2026-01-31）

| 功能 | 说明 | 状态 |
|------|------|------|
| 2.1 纯BQK算法 | 删除 `mad_hysteresis` 算法（-200行代码） | ✅ |
| 2.2 BQKDetector类 | 封装 `bqk_utils.py`，预计算滤波器系数 | ✅ |
| 2.3 多带包络 | 宽带分20Hz子带，Butterworth 3阶滤波 + Hilbert | ✅ |
| 2.4 并行化 | `joblib.Parallel` 并行计算子带包络 | ✅ |
| 2.5 双阈值检测 | `rel_thresh×local_median` ∧ `abs_thresh×global_median` | ✅ |
| 2.6 事件合并筛选 | `merge_timeRanges` + `min_last` 持续时间过滤 | ✅ |
| 2.7 Ripple/FR分离 | `band='ripple'/'fast_ripple'` | ✅ |

#### Phase 2 重构总结（2026-01-31）

**核心改进：封装 + 并行化（但注意适用场景）**

```python
# Before: 分散的函数调用
env = bqk.return_hil_enve_norm(data, fs, freqband)  # 内部循环K次滤波
events = bqk.find_high_enveTimes(env, ...)

# After: 清晰的类封装 + 可选并行
detector = BQKDetector(sfreq=fs, freqband=(80,250), n_jobs=1)  # 默认串行
env = detector.compute_envelope(data)  # 预计算滤波器系数
events = detector.detect_events(data)  # 端到端检测
```

**删除的代码**（零破坏性，-348行）：
- ❌ `mad_hysteresis` 算法及所有相关函数（~200行）
- ❌ GPU相关代码（`cupy_hilbert`, `_HAS_CUPY`）
- ❌ Ictal检测相关（`_detect_ictal_mask`, `_mad`, `_moving_average`, `_find_runs`）

**性能测试结果**（2026-01-31, cuda_env）：

| 场景 | n_jobs=1 | n_jobs=-1 | 结论 |
|------|----------|-----------|------|
| 小数据 (8ch×10s) | 0.078s | 0.902s | ❌ 并行慢12x (进程开销) |
| 大数据 (16ch×30s) | 0.457s | 1.100s | ❌ 并行慢2.4x |
| 数值一致性 | - | diff=1.81e-10 | ✅ 完美 |

**关键发现（Amdahl's Law陷阱）**：
```
并行开销 = joblib进程创建 (~500ms) + 数据序列化 (~200ms)
计算时间 = K × (滤波+Hilbert) ≈ 0.078s (K=9, 8ch×10s)

当 计算时间 < 并行开销 → 串行更快
```

**并行化适用场景**：
- ✅ **长时程无chunk**：`chunk_sec=None` + 单文件>2分钟 + K>20
- ❌ **默认chunked处理**：30s chunk → 计算 <200ms → n_jobs=1 更快

**推荐配置**：

```python
# 默认配置（推荐）：n_jobs=1
config = HFODetectionConfig(
    band='ripple',
    chunk_sec=30.0,  # 分块处理
    n_jobs=1,        # ← 串行避免进程开销
)

# 全文件处理（特殊场景）：
config = HFODetectionConfig(
    chunk_sec=None,  # ← 整个文件一次性处理
    n_jobs=-1,       # ← 可能有收益（需测试）
)
```

**收益**：
- ✅ 代码: 632行 → 284行 (-55%)
- ✅ 算法: 单一BQK路径
- ✅ 封装: `BQKDetector` 类，滤波器系数预计算
- ✅ 数值: 与原 `bqk_utils.py` 误差 <1e-9
- ⚠️ 并行: **仅长时程场景有效，默认场景反而变慢**

**关键技术决策**:
- **算法纯化**: 只保留BQK，删除所有非BQK代码
- **类封装**: `BQKDetector` 预计算滤波器系数（不再每chunk重复）
- **Chunked处理**: 30s chunk + 1s overlap，避免内存爆炸
- **并行化策略**: 默认 `n_jobs=1`（实测更快），避免joblib开销
- **双阈值策略**: `rel_thresh × local_median` ∧ `abs_thresh × global_median`

---

### 模块3: src/group_event_analysis.py ✅ 核心逻辑完成

> **设计原则**：这是整个流程的"计算中心"——所有中间结果都从这里产出，后续模块只读取。

#### 输入规范

| 输入 | 来源 | 数据结构 |
|------|------|----------|
| 预处理结果 | `preprocessing.py` | `PreprocessingResult` (data, sfreq, ch_names) |
| HFO事件 | `hfo_detector.py` 或 `*_gpu.npz` | `Dict[ch_name → np.ndarray (n,2)]` |
| 患者级汇总 | `_refineGpu.npz` | events_count (n_ch,), chns_names |
| 核心通道 | `hist_meanX.npz` | pick_chns (8,) |
| 事件窗口 | `*_packedTimes.npy` | (n_events, 2) [start, end] 秒 |

#### 输出规范

| 输出文件 | 内容 | 下游用户 |
|----------|------|----------|
| `*_envCache_{band}_{ref}.npz` | envelope + x_band + sfreq + ch_names | visualization (Fig1波形) |
| `*_groupAnalysis.npz` | **质心+TF质心+lag+rank+baseline池元数据** | visualization, network_analysis |

#### 功能分解

| 功能 | 说明 | 状态 |
|------|------|------|
| 3.1 窗口构建 | `build_windows_from_detections` | ✅ |
| 3.2 Envelope缓存 | `precompute_envelope_cache` | ✅ |
| 3.3 质心计算 | `compute_centroid_matrix_from_envelope_cache` | ✅ |
| 3.4 Lag/Rank | `lag_rank_from_centroids` | ✅ |
| 3.5 TF质心 | `compute_tf_centroids`（wavelet+动态基线） | ✅ |
| 3.6 结果存储 | `save_group_analysis_results`, `load_group_analysis_results` | ✅ |
| 3.7 一键API | `compute_and_save_group_analysis` | ✅ |
| 3.7 通道筛选 | `select_core_channels_by_event_count` | ✅ |
| 3.8 验证函数 | `validate_*` 系列 | ✅ |

**阶段性结论（2026-01-16）**:
- ✅ 核心通道筛选：`mean + 1*std` 可复现 `hist_meanX.npz` 的 `pick_chns`
- ✅ packedTimes 窗口长度：从 `packedTimes[:,1]-packedTimes[:,0]` 推断
- ✅ Step1 验证：`reference='bipolar'` + 别名通道 + GPU通道过滤 → 高覆盖率
- ✅ Step2-3 验证：`eventsBool` 100% 一致；相对 lag 达 ms 级误差
- ✅ TF质心计算：`compute_tf_centroids` 改为 **wavelet+动态基线**（非STFT）
- ✅ 基线池：2s窗/1s步长，排除ictal+HFO+高LL+高Ripple，存入 `baseline_pool_starts/indices`
- ✅ 统一存储：`save_group_analysis_results` + `load_group_analysis_results` 已实现
- ✅ 一键API：`compute_and_save_group_analysis` 可从 EDF 一站式生成所有中间结果

---

### 模块4: src/network_analysis.py (待开发)

| 功能 | 说明 |
|------|------|
| 4.1 容积传导剔除 | lag≈0+远距(>20mm)→剔除, 近距(<10mm)→保留 |
| 4.2 加权有向图构建 | 边权重 = 频次 × 滞后一致性 |
| 4.3 Interictal/Ictal分段 | 分别构建网络, 对比拓扑 |
| 4.4 图论指标 | In/Out-Strength, Eccentricity, Betweenness, Clustering |
| 4.5 传播速度验证 | 结合电极坐标, 验证生理范围 |
| 4.6 Source-Sink验证 | SI指标, 状态翻转检验 |

---

### 模块5: src/visualization.py ✅ 重构完成（2026-01-16）

> **铁律**：visualization 只负责读取已计算的中间结果并画图，**不做任何复杂计算**。

#### 职责边界

**✅ 允许**:
1. 读取 `*_groupAnalysis.npz` 或 `*_envCache.npz`
2. 根据参数选择子集（通道、事件）
3. 调用 matplotlib 画图
4. 简单的数据切片、reshape、颜色映射

**❌ 禁止**:
- 计算 STFT
- 计算质心
- 计算 Hilbert envelope
- 任何超过 50 行的数据处理逻辑

---

#### 功能总览：可视化产出与资源依赖

**A. 基础可视化（数据探索）**

| 函数 | 产出图表 | 必需资源 | 用途 |
|------|---------|---------|------|
| `plot_seeg_segment` | 多通道时序波形 | `PreprocessingResult` (内存对象) | 查看预处理后数据 |
| `plot_from_result` | 快速波形图 | `PreprocessingResult` | 便捷封装 |
| `plot_shaft_channels` | 单电极串波形 | `PreprocessingResult` | 定位异常电极 |
| `plot_preprocessing_comparison` | 前后对比图 | 2个 `PreprocessingResult` | 验证预处理效果 |
| `plot_raw_filtered_envelope` | Raw/滤波/包络三联图 | `PreprocessingResult` + 通道选择 | 检测器调试 |
| `plot_event_counts` | HFO事件计数柱状图 | `HFODetectionResult` | 快速伪迹筛查 |

**B. 论文级可视化（只读取中间结果）**

| 函数 | 产出图表 | 必需资源 | 可选资源 | 用途 |
|------|---------|---------|---------|------|
| `plot_paper_fig1_bandpassed_traces` | **Fig1**: 拼接事件窗口的带通波形 raster | `*_envCache.npz`<br>`*_packedTimes.npy` | - | 展示群体事件时序特征 |
| `plot_paper_fig2_normalized_spectrogram` | **Fig2**: TF功率背景 + 质心点 + 传播路径 | `*_envCache.npz`<br>`*_packedTimes.npy`<br>`*_groupAnalysis.npz` | `*_gpu.npz`(mask) | 展示TF域传播模式 |
| `plot_lag_heatmaps_from_group_analysis` | **3张热图**: Energy/Rank/Lag (channels×events) | `*_groupAnalysis.npz`<br>`*_envCache.npz`<br>`*_packedTimes.npy` | - | 量化传播滞后和能量分布 |
| `plot_lag_statistics` | **统计三联图**: Lag分布/Rank分布/通道参与率 | `*_groupAnalysis.npz` | - | 群体统计特征 |
| `plot_tf_centroid_statistics` | **TF质心统计**: 频率质心/时间质心分布 | `*_groupAnalysis.npz` | - | TF质心特征分析 |

**C. 底层工具函数**

| 函数 | 产出 | 输入 | 说明 |
|------|------|------|------|
| `detections_to_events` | 事件字典列表 | `HFODetectionResult` | 格式转换，供 `plot_seeg_segment` 叠加 |
| `plot_lag_heatmaps` | 3张热图 | numpy 数组 | 底层绘图函数，通常用高层封装 |

---

#### 核心资源文件速查

**中间结果文件（必需）**：

```python
# 1. Envelope + Bandpass 缓存
"<record>_envCache_<band>_<ref>.npz"
├─ env: (n_ch, n_samples) Hilbert envelope
├─ x_band: (n_ch, n_samples) bandpassed signal
├─ sfreq: 采样率
└─ ch_names: 通道名列表

# 2. 完整分析结果
"<record>_groupAnalysis.npz"
├─ centroid_time: (n_ch, n_events) 时间质心
├─ tf_centroid_time: (n_ch, n_events) TF 2D质心-时间分量
├─ tf_centroid_freq: (n_ch, n_events) TF 2D质心-频率分量
├─ lag_raw: (n_ch, n_events) 相对滞后(秒)
├─ lag_rank: (n_ch, n_events) 排名(0=最早)
├─ events_bool: (n_ch, n_events) 参与mask
└─ sfreq, band, ch_names, event_windows...

# 3. 事件窗口
"<record>_packedTimes.npy"
└─ (n_events, 2) [start, end] 秒

# 4. GPU检测结果（可选，用于mask）
"<record>_gpu.npz"
├─ whole_dets: List[(n_det, 2)] per channel
└─ chns_names: 通道名
```

**典型可视化流程**：

```python
# Step 1: 生成中间结果（一次性）
from src.group_event_analysis import compute_and_save_group_analysis
out_paths = compute_and_save_group_analysis(
    edf_path='FC10477Q.edf',
    output_dir='./output',
    output_prefix='FC10477Q',
    packed_times_path='FC10477Q_packedTimes.npy',
    gpu_npz_path='FC10477Q_gpu.npz',  # 可选
    band='ripple',
    reference='bipolar',
    save_env_cache=True,
)

# Step 2: 可视化（任意多次，只读取）
from src.visualization import (
    plot_paper_fig1_bandpassed_traces,
    plot_paper_fig2_normalized_spectrogram,
    plot_lag_heatmaps_from_group_analysis,
    plot_lag_statistics,
    plot_tf_centroid_statistics,
)

# Fig1: 波形 raster
fig1 = plot_paper_fig1_bandpassed_traces(
    cache_npz_path=out_paths['env_cache_path'],
    packed_times_path='FC10477Q_packedTimes.npy',
    channel_order=CORE_CHANNELS,
    event_indices=list(range(30)),
)

# Fig2: TF + 质心路径
fig2 = plot_paper_fig2_normalized_spectrogram(
    cache_npz_path=out_paths['env_cache_path'],
    packed_times_path='FC10477Q_packedTimes.npy',
    group_analysis_npz_path=out_paths['group_analysis_path'],  # 读取预计算TF质心
    channel_order=CORE_CHANNELS,
    event_indices=list(range(30)),
)

# 热图: Energy/Rank/Lag
figE, figR, figL = plot_lag_heatmaps_from_group_analysis(
    group_analysis_npz=out_paths['group_analysis_path'],
    env_cache_npz=out_paths['env_cache_path'],
    packed_times_npy='FC10477Q_packedTimes.npy',
    channel_names=CORE_CHANNELS,
    max_events=100,
)

# 统计图
fig_stats = plot_lag_statistics(
    group_analysis_npz=out_paths['group_analysis_path'],
    patient_id='chengshuai',
    record_id='FC10477Q',
)

# TF质心统计
fig_tf = plot_tf_centroid_statistics(
    group_analysis_npz=out_paths['group_analysis_path'],
)
```

---

#### 重构总结（2026-01-16）

✅ **完成的重构**：
1. ✅ 将 TF 质心计算从 `visualization.py` 迁移到 `group_event_analysis.py`
2. ✅ 添加 `compute_tf_centroids()` 计算 2D TF 质心（时间+频率）
3. ✅ `*_groupAnalysis.npz` 现在包含 `tf_centroid_time` 和 `tf_centroid_freq`
4. ✅ `plot_paper_fig2_*` 重构为读取预计算质心，不再内部计算 STFT
5. ✅ 新增便利函数：
   - `plot_lag_heatmaps_from_group_analysis()` - 一键读取+绘制热图
   - `plot_lag_statistics()` - 统计分析三联图
   - `plot_tf_centroid_statistics()` - TF质心分布统计
6. ✅ `chengshuai_hfo_analysis.ipynb` 重构为纯可视化终端（不做计算）

**架构验证**：
- ✅ visualization.py 不再有任何 STFT/质心计算逻辑
- ✅ 所有复杂计算在 `group_event_analysis.py`
- ✅ notebook 只调用 visualization 函数，不写内联绘图代码
- ✅ 中间结果可复用，避免重复计算

**关键技术决策**:
- **波形颜色**: `tableau_20_no_red` 避免与HFO红色标记冲突
- **事件标注**: 默认 `style='tick'` 细线，不遮挡波形
- **TF质心**: 使用 wavelet TFR + 动态基线log校正，再在该TF图上计算2D质心

---

## 5. 快速开发参考

### 5.1 preprocessing.py 快速上手

```python
from src.preprocessing import SEEGPreprocessor

# 标准用法（自动选择采样率）
preprocessor = SEEGPreprocessor(
    target_band='ripple',        # 'ripple' (1000Hz) or 'fast_ripple' (2000Hz)
    reference='bipolar',         # 'bipolar' / 'car' / 'none'
    use_gpu=True,                # 自动降级到 CPU 如果无 GPU
)
result = preprocessor.run('path/to/file.edf')

# 复现 GPU 通道列表（显式）
gpu_data = np.load('FC10477Q_gpu.npz', allow_pickle=True)
gpu_channels = [str(ch) for ch in gpu_data['chns_names']]

preprocessor = SEEGPreprocessor(
    reference='bipolar',
    include_channels=gpu_channels,  # 显式通道白名单
)

# 排除坏通道（显式）
preprocessor = SEEGPreprocessor(
    reference='bipolar',
    exclude_channels=['A1-A2', 'EMG1-EMG2'],  # 显式黑名单
)

# 添加自定义 FilterBackend（高级）
from src.preprocessing import FilterBackend

class MyCustomBackend(FilterBackend):
    def apply_notch(self, data, sfreq, freqs):
        # 你的实现
        return data
    
    def apply_bandpass(self, data, sfreq, low, high):
        # 你的实现
        return data

preprocessor = SEEGPreprocessor()
preprocessor.filter_backend = MyCustomBackend()
```

### 5.2 常见任务速查

| 任务 | 代码 |
|------|------|
| **预处理 + HFO检测** | `preprocessor.run(edf) -> detector.detect(result.data, result.sfreq)` |
| **完整分析流程** | `compute_and_save_group_analysis(edf_path, ...)` |
| **可视化论文图** | `plot_paper_fig1_bandpassed_traces(env_cache_npz, ...)` |
| **读取中间结果** | `load_group_analysis_results('*_groupAnalysis.npz')` |
| **Envelope 缓存** | `precompute_envelope_cache(data, sfreq, ch_names, ...)` |

### 5.3 未来重构计划（可选，低优先级）

**Phase 2: 职责分离（Loader/Processor）**
- 目标：拆分 EDF 加载和数据变换逻辑
- 收益：支持多种数据格式（BrainVision, Neuralynx, ...）
- 触发条件：需要支持 ≥3 种数据格式时

**Phase 3: 移动 seizure detection**
- 目标：`detect_seizure_onsets_from_data()` → `src/seizure_detection.py`
- 收益：模块职责更清晰（preprocessing 不应包含分析逻辑）
- 触发条件：需要扩展多种发作检测算法时

## 6. 关键技术陷阱与解决方案

| 陷阱 | 问题描述 | 解决方案 |
|------|----------|----------|
| "Pre-bipolar推断" | 通过"末端contact缺失"推断EDF已bipolar → **误解** | **禁止推断**；需要什么就显式 `reference='bipolar'/'car'/'none'` |
| Bipolar断桥 | 跨电极串差分产生伪信号 | 严格解析电极前缀, 仅同串相邻 |
| FR采样率 | 1000Hz下500Hz严重衰减 | FR分析保留2000Hz |
| Ictal阈值 | 发作期拉高全局阈值 | 剔除Ictal后计算基线 |
| 鸡生蛋问题 | HFO簇被误判为Ictal | 持续时间约束 >3秒 |
| 相位滑移 | XCorr cycle skipping | 对**Hilbert包络**做互相关 |
| 核心通道循环论证 | 高密度≠Source | 多策略对比筛选 |
| 图非连通 | Eccentricity计算失效 | Harmonic Centrality替代 |
| 零滞后离散化 | 近距快速传播被误杀 | 距离条件剔除 (仅远距lag=0) |
| **GPU不可用** | 服务器有GPU本地无 | 所有算法CPU/GPU双版本, 自动降级 |

---

## 7. 开发进度

- [x] 项目结构设计
- [x] 开发计划文档
- [x] **模块1: preprocessing.py** ✅ Phase 1 重构完成 (2026-01-30)
  - [x] 电极名称解析 (ElectrodeParser)
  - [x] 重参考（显式）:
    - [x] Bipolar (BipolarReferencer) — 命名 `A1-A2`
    - [x] CAR per shaft (CommonAverageReferencer)
    - [x] None（保持原始EDF）
  - [x] 通道选择（显式）: include/exclude channels（用于匹配GPU通道列表）
  - [x] 重采样 + Notch滤波
  - [x] **✅ FilterBackend 架构重构** - 消除所有 GPU if/else 分支
  - [x] **✅ 删除废代码** - PreBipolarDetector, validate_against_gpu_results, exclude_last_n
  - [x] GPU加速支持 (CuPy可选)
  - [x] 通道质量检查
  - [x] chengshuai/FC10477Q: EDF vs GPU 通道差异来源确认（GPU=显式通道子集；不用于推断重参考）
- [x] **模块2: hfo_detector.py** ✅ Phase 2 重构完成（2026-01-31）
  - [x] 删除 `mad_hysteresis` 算法（-200行）
  - [x] 封装 `BQKDetector` 类（预计算滤波器系数）
  - [x] 实现 joblib 并行化（多带包络计算）
  - [x] 性能测试：串行 vs 并行（发现小数据并行反而慢）
  - [x] 双阈值检测（rel_thresh × local_median ∧ abs_thresh × global_median）
  - [x] 事件合并筛选（min_gap + min_last）
  - [x] Ripple/FR分离
  - [x] Chunked处理（30s chunk + 1s overlap）
  - [x] 数值验证：与原 `bqk_utils.py` 误差 <1e-9
- [x] **模块3: group_event_analysis.py** ✅ 2026-01-16（核心逻辑完成）
  - [x] 窗口构建 (build_windows_from_detections)
  - [x] Envelope缓存 (precompute_envelope_cache)
  - [x] 质心计算 (compute_centroid_matrix_from_envelope_cache)
  - [x] Lag/Rank计算 (lag_rank_from_centroids)
  - [x] 通道筛选 (select_core_channels_by_event_count)
  - [x] 验证函数 (validate_* 系列)
  - [x] TF质心计算 (compute_tf_centroids) ✅
  - [x] 统一结果存储 (save_group_analysis_results) ✅
  - [x] 一键API (compute_and_save_group_analysis) ✅
- [x] **模块5: visualization.py** ✅ 2026-01-16（重构完成）
  - [x] 基础可视化（波形、事件标注、调试视图）
  - [x] Fig1: 带通波形 raster（读取 envCache）
  - [x] Fig2: TF功率+质心路径（读取 groupAnalysis 预计算质心）
  - [x] 热图：Energy/Rank/Lag（读取 groupAnalysis）
  - [x] 统计分析：Lag/Rank/Participation（读取 groupAnalysis）
  - [x] TF质心统计（读取 groupAnalysis）
  - [x] ✅ 架构重构：剥离所有 STFT/质心计算到 group_event_analysis
  - [ ] 传播动图（500ms窗口内能量传播动画）
  - [ ] 网络拓扑图（待模块4）
- [ ] 模块4: network_analysis.py
- [x] Notebook: chengshuai_hfo_analysis.ipynb ✅
- [ ] 验证与原结果一致性
- [ ] 完整Pipeline测试

---

## 8. 测试数据

**示例患者**: chengshuai  
**示例记录**: FC10477Q  
**数据路径**: `/mnt/yuquan_data/yuquan_24h_edf/chengshuai/FC10477Q.edf`

**预期输出**:
- Bipolar通道数: ~130（取决于每根电极串contact数量与缺失情况）
- GPU检测事件数: 46,738
- 核心通道: ['E11', 'K3', 'K5', 'K6', 'K7', 'K8', 'K9', 'K10']
- 对齐事件数: 2,601

**Notebook验证结果（100s crop）**:
- Detections: 10232 total
- Top channels: ['K15-K16', 'G13-G14', 'J9-J10', 'K6-K7', 'D13-D14', ...]
