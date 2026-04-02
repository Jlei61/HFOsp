# HFOsp 开发计划

**项目目标**: 复现并扩展 HFO (高频振荡) 分析流程，验证 Source-Sink 理论  
**数据集**: 玉泉24小时SEEG数据集  
**数据路径**: `/mnt/yuquan_data/yuquan_24h_edf`  
**更新日期**: 2026-03-05

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
│  │  │ Step 2: 预计算 Envelope 缓存（bandpass可选）   │    │                       │
│  │  │   - precompute_envelope_cache               │    │                        │
│  │  │   - 存储: *_envCache.npz                    │    │                        │
│  │  └──────────────────┬──────────────────────────┘    │                        │
│  │                     ▼                               │                        │
│  │  ┌─────────────────────────────────────────────┐    │                        │
│  │  │ Step 3: 质心分析（TF可选）→ 存储中间结果       │    │                       │
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
| `*_envCache_{band}_{ref}.npz` | **Envelope缓存（x_band可选）** | `group_event_analysis` |
| `*_groupAnalysis.npz` | **质心+lag（TF可选）分析结果** | `group_event_analysis` |

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
    'centroid_time': np.array([...]),      # (n_ch, n_events) 时间质心(相对窗口起始, 默认env)
    'tf_centroid_time': np.array([...]),   # (n_ch, n_events) TF质心时间分量（可选）
    'tf_centroid_freq': np.array([...]),   # (n_ch, n_events) TF质心频率分量（可选）
    'events_bool': np.array([...]),        # (n_ch, n_events) 通道是否参与
    
    # === Lag/Rank 分析 ===
    'lag_raw': np.array([...]),            # (n_ch, n_events) 相对lag(对齐到最早通道)
    'lag_rank': np.array([...]),           # (n_ch, n_events) 排名 (0=最早)

    # === Co-activation (ch × ch) ===
    'coact_event_count': np.array([...]),  # (n_ch, n_ch) 共同激活事件数
    'coact_event_ratio': np.array([...]),  # (n_ch, n_ch) 共同激活比例 (=count/n_events)
    'coact_time_ratio': np.array([...]),   # (n_ch, n_ch) 质心绝对时间对齐强度(0..1)
    'coact_rank_ratio': np.array([...]),   # (n_ch, n_ch) 质心相对rank对齐强度(0..1)
    
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

### 模块1: src/preprocessing.py ✅ 完成（Phase 1 重构 2026-01-30; PR1 扩展 2026-04-01）

| 功能 | 说明 | 状态 |
|------|------|------|
| 1.1 EDF读取 | 加载EDF, 清洗电极名称, encoding='latin1' | ✅ |
| 1.2 电极名称解析 | 正则提取 (prefix, number), 支持A'5格式 | ✅ |
| 1.3 重参考策略 | **显式** bipolar / car / none（auto=兼容别名→bipolar；不做任何"推断"） | ✅ |
| 1.4 通道选择 | **显式** include/exclude channels（用于复现GPU通道列表；不硬编码"去掉末端N个触点"） | ✅ |
| 1.5 重采样 | Ripple→1000Hz, FR→2000Hz | ✅ |
| 1.6 滤波 | Notch + 可选Bandpass, GPU加速支持 | ✅ |
| 1.7 通道质量检查 | z-score, 方差, 伪迹标记 | ✅ |
| **1.8 FilterBackend架构** | **抽象接口 + CPU/GPU实现分离** | ✅ |
| **1.9 EDF+ Annotation Parser** | `fast_read_edf_annotations()` — 二进制 TAL 解析；NFS 上改为 annotation-tail threaded `pread`，不依赖 MNE | ✅ **PR1** |
| **1.10 Seizure Annotation Parser** | `parse_seizure_onsets_from_annotations()` — 精确 label 匹配 + onset-END 自动配对；默认丢弃 orphan zero-duration，并合并重叠重复区间 | ✅ **PR1** |
| **1.11 Timezone Infrastructure** | `epoch_to_local_hour()` — `zoneinfo.ZoneInfo` 显式时区转换; `config/default.yaml` 新增 `dataset.timezone_default` | ✅ **PR1** |
| **1.12 Recording Timeline Helper** | `read_edf_record_info()` + `build_recording_timeline()` — header 驱动的 subject 连续时间轴；禁止硬编码 `12 files == 24h` | ✅ **PR1 hardening** |

#### PR1 真验收更新（2026-04-01）

- **PR1 现在可以作为真正验收，但验收边界要说清楚**：
  - ✅ 已验收：EDF+ annotation parsing、seizure interval extraction、timezone conversion、header-driven timeline foundation
  - ❌ 不能夸大：它还不是“临床 gold-standard seizure inventory”；EDF 原始标注本身存在重复 onset、孤立 onset、缺失 END 等脏数据
- 全量 Yuquan 审计（21 subjects / 260 EDF）后确认：
  - 原始命中 `32` 个 seizure-bearing EDF、`54` 个原始 intervals
  - 归一化后为 `25` 个 valid interval-bearing EDF、`30` 个 normalized intervals
  - 另外保留 `16` 个 orphan onset markers（有 onset、无可靠 offset）
  - 对 32 个 seizure-labeled EDF 的 offset 审计显示：有效 offset 全部来自后续 `END` 标签配对，`duration` 来源为 0；重复问题来自多 onset 共享同一个 END，而不是 offset 错配
  - 大多数 subject 是连续分段记录，但总时长并不固定在 24h；`litengsheng`、`zhangjiaqi` 存在真实缺口
- 结论：PR1 可作为 **后续 PR2/PR3 的可靠时间轴与人工标注入口**，但不是最终的发作真值来源。

#### PR1.5 Epilepsiae 数据契约调研（2026-04-02）

- 新增脚本：`scripts/survey_epilepsiae_dataset.py`
- 新增脚本：`scripts/run_epilepsiae_interictal_synchrony.py`
- 新增脚本：`scripts/aggregate_epilepsiae_interictal_synchrony.py`
- 新增正式接口：`src/epilepsiae_dataset.py`
- 新增正式接口：`src/interictal_synchrony_aggregation.py`
- 新增文档：`docs/epilepsiae_dataset_structure.md`
- 已确认：
  - `Epilepsiae` 原始数据合同是 `*.data + *.head + SQL`，不是 EDF
  - 挂载盘全量是 **27** 个 SQL subjects；其中只有 **20** 个有 `all_data_lns` 间期中间产物
  - 时间真值应优先信 SQL `recording / block / seizure`，`.head.start_ts` 只做块级校验
  - 当前挂载数据上，`.head.start_ts` 与 SQL `block.begin` 是 **0s 对齐**
  - 数据连续性不干净：`75` 个 recordings 里只有 `10` 个 block 级连续；`27` 个 subjects 里只有 `5` 个没有明显 inter-recording gap
  - seizure 标注整体可用，但不是每条 EEG interval 都完整；`vigilance` 不能直接当 day/night
  - 当前挂载数据的时区已经钉死为 `UKLFR -> Europe/Berlin`；`src.epilepsiae_dataset` 已内建 override 接口与 `08:00-20:00` day/night 规则
  - 已输出 `results/epilepsiae_sync_subject_manifest.csv`，按 `ready_full_artifacts / ready_partial_artifacts / missing_interictal_artifacts` 分层
  - `interictal_synchrony` 已接上 manifest，并实际跑完 `ready_full_artifacts` 的 `16` 个 subjects / `2962` 个 blocks
  - block-level synchrony 已进一步聚合成 `subject × seizure_interval × window_type` 分析表；聚合规则是严格整块归属，跨 seizure / post-ictal / day-night / gap 边界的 block 直接排除
  - 当前聚合实跑保留：`1903` blocks 能安全落进完整 seizure interval，`1742` blocks 能进入 `phase(post_ictal/interictal)` 聚合，主表产出 `409` 行
- 对后续 PR 的硬约束：
  - 若将 Epilepsiae 纳入同步性分析，必须消费 `results/epilepsiae_*_inventory.*` 形成的统一时间轴与 seizure inventory
  - subject 选择必须优先消费 manifest，而不是手工挑病人
  - 不能把“20 个 artifact subjects”误当成“全量数据集”
  - 不能把 Yuquan 的 EDF 路径直接套到 Epilepsiae 上
  - 不能把 1h block-level synchrony 假装成任意精细的临床窗口；当前聚合层明确拒绝半块归属

#### PR2 Streaming Seizure Detector 验收复盘（2026-04-02）

- 已完成（基础设施，可复用）：
  - `src/preprocessing.py`：二进制 EDF 流式读取与 channel-mean LL+RMS 检测主流程
  - `scripts/pr2_seizure_validation.py`：单 EDF 叠图、24h 总览、误差散点、审计 CSV
  - `tests/test_seizure_streaming.py`：`_flag_to_runs` / `_merge_close_runs` / `match_seizure_intervals` 合成数据测试
  - NFS 性能优化：单次顺序读取 + 特征缓存，支持 `n_jobs` 并行批跑
  - 手工标注评估修正：interval 与 onset-only 分开匹配，避免漏算人工标注
- 已验收（litengsheng）：
  - 峰值内存约 `110MB`（< `500MB`）
  - onset 中位误差约 `4.6s`（< `30s`）
  - recall 达到门槛附近（约 `80%+`）
- 未通过（跨 subject 泛化）：
  - 同一参数在 `sunyuanxin` 上出现明显漏检，无法同时兼顾低 FP 与高 recall
  - 根因不是阈值没调好，而是数据结构有损：先对多通道取均值再做特征，抹掉了空间招募信息
- 结论：
  - PR2 基础设施通过，可作为后续检测器与验证框架
  - channel-mean 检测器不再继续加补丁（如 `ignore_initial`）；进入 PR2.5 重构

#### PR2.5 空间招募检测器（第一性原理）（计划）

- 第一性原理约束：
  - 发作具有通道逐步招募特征（participation 上升）
  - 发作期存在大幅高频振荡（LL 对该特征敏感）
  - 发作有自限性（participation 回落形成 offset）
- 核心顺序修正：
  - 旧：`channels -> mean -> LL/RMS -> threshold`
  - 新：`channels -> per-channel LL -> per-channel z -> active-channel fraction -> threshold`
- 预期收益：
  - 消除 `combine_mode` / `ignore_initial` 这类补丁参数
  - 提升跨 subject 泛化，减少单通道伪迹导致的 FP 与均值稀释导致的 FN

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
| `*_envCache_{band}_{ref}.npz` | envelope + (可选x_band) + sfreq + ch_names | visualization (Fig1波形) |
| `*_groupAnalysis.npz` | **质心+lag+rank+baseline池元数据（TF可选）** | visualization, network_analysis |

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

**配置补充（默认行为）**：
- `centroid_source='env'`：lag/rank 默认基于包络质心
- `compute_tf_centroids=False`：TF质心默认不计算（可选开启）
- `save_bandpass=False`：x_band 默认不保存（仅可视化需要时开启）

**阶段性结论（2026-01-16）**:
- ✅ 核心通道筛选：`mean + 1*std` 可复现 `hist_meanX.npz` 的 `pick_chns`
- ✅ packedTimes 窗口长度：从 `packedTimes[:,1]-packedTimes[:,0]` 推断
- ✅ Step1 验证：`reference='bipolar'` + 别名通道 + GPU通道过滤 → 高覆盖率
- ✅ Step2-3 验证：`eventsBool` 100% 一致；相对 lag 达 ms 级误差
- ✅ TF质心计算：`compute_tf_centroids` 改为 **wavelet+动态基线**（非STFT），默认可关闭
- ✅ 基线池：2s窗/1s步长，排除ictal+HFO+高LL+高Ripple，存入 `baseline_pool_starts/indices`
- ✅ 统一存储：`save_group_analysis_results` + `load_group_analysis_results` 已实现
- ✅ 一键API：`compute_and_save_group_analysis` 可从 EDF 一站式生成所有中间结果

---

### 模块4: src/network_analysis.py (开发中)

> **核心目标**：从 HFO 群体事件构建下一代癫痫网络，实现 SOZ 定位与传播路径预测。

---

#### 4.0 设计哲学与批判性前提

**"宽建图 → 精剪枝 → 定方向" 策略 (Build-Prune-Direct Strategy)**

```
┌────────────────────────────────────────────────────────────────────────┐
│                    癫痫网络构建流水线 v2                                 │
│               "宽建图 → 精剪枝 → 定方向"                                │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  [全通道池] (n_all ≈ 120)                                              │
│      │                                                                 │
│      ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │  Step 1: 宽建图 (Broad Graph Construction)                  │       │
│  │  边权 = Simpson Index (归一化共激活)                         │       │
│  │  "不再用原始共激活计数——校正基础率偏差"                       │       │
│  │  + Surrogate 显著性检验 → 剔除随机重合                      │       │
│  └──────────────────────────┬──────────────────────────────────┘       │
│                             │                                          │
│                             ▼                                          │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │  Step 2: XYZ 多维剪枝 (Multi-Dimensional Pruning)          │       │
│  │  X = HFO Rate (活跃度) → 节点是病理活动发生者               │       │
│  │  Y = Connection Entropy (特异性) → 剔除全脑噪声/参考伪迹    │       │
│  │  Z = FR/R Ratio (致痫性) 或 谱聚类(XYZ距离度量)            │       │
│  └──────────────────────────┬──────────────────────────────────┘       │
│                             │                                          │
│                             ▼                                          │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │  Step 3: 方向注入 (Direction Injection)                     │       │
│  │  Wilcoxon + 一致性检验 → 有向边                             │       │
│  └──────────────────────────┬──────────────────────────────────┘       │
│                             │                                          │
│                             ▼                                          │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │  Step 4: 复合权重 + 物理约束                                │       │
│  │  Simpson × Consistency × Stability                          │       │
│  │  + 容积传导剔除 (<10mm, Phase B)                            │       │
│  │  + 传播速度验证 (0.1-10 m/s, Phase B)                       │       │
│  └──────────────────────────┬──────────────────────────────────┘       │
│                             │                                          │
│                             ▼                                          │
│  [加权有向图 G(V, E, W)]  →  图论指标  →  SOZ/传播路径                  │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

**为什么是 "宽建图 → 精剪枝" 而不是旧版 "选节点 → 建骨架"？**

| 策略 | 致命缺陷 |
|------|---------|
| 先选节点再建边 | 选节点用的 co-activation 本身被基础率偏差污染，垃圾进垃圾出 |
| 先建骨架再选节点 | 骨架的边权（原始 count/ratio）无法区分"真同步"和"随机重合" |
| **宽建图 → 精剪枝** | Simpson 归一化消除率偏差 → XYZ 多维独立剪枝，每步可审计可回溯 |

---

#### 4.1 现有资产盘点 (Asset Inventory)

**✅ 已有数据（groupAnalysis.npz）**

| 数据 | 形状 | 物理意义 | 网络用途 |
|------|------|----------|---------|
| `ch_names` | (n_ch,) | 核心通道名 | 节点标识 |
| `coact_all_ch_names` | (n_all,) | **全通道名** | 扩大节点池 |
| `coact_event_ratio` | (n_ch, n_ch) | 共激活概率 | **骨架构建** |
| `coact_all_event_ratio` | (n_all, n_all) | **全通道共激活** | 扩大节点池 |
| `lag_raw` | (n_ch, n_events) | 质心时间（相对窗口起点） | **方向计算** |
| `events_bool` | (n_ch, n_events) | 通道参与mask | 事件过滤 |
| `event_windows` | (n_events, 2) | 事件窗口 [start, end] | 时间分段 |

**关键数据结构洞察**：

```python
# lag_raw 存储的是每通道相对于窗口起点的质心时间
# 要获得通道对 (i, j) 在事件 k 中的时滞：
lag_ij_k = lag_raw[i, k] - lag_raw[j, k]  # 负值 = i 领先 j

# 这是紧凑存储：O(n_ch × n_events) vs O(n_ch² × n_events)
# 运行时计算差值，空间换时间
```

**⚠️ 缺失数据（需要扩展）**

| 数据 | 形状 | 来源 | 优先级 | 用途 |
|------|------|------|--------|------|
| `electrode_distance` | (n_all, n_all) | MNI坐标计算 | **Phase B 阻塞** | 容积传导剔除、传播速度验证 |
| `hfo_type_per_event` | (n_ch, n_events) | 检测器输出 | **Phase B** | 病理加权 (FR 比例) |
| `tissue_label` | (n_all,) | FreeSurfer | Phase C | 灰/白质过滤（不硬剔除） |
| `mni_coords` | (n_all, 3) | 配准结果 | **Phase B 阻塞** | 3D可视化、距离矩阵 |
| `lead_field_matrix` | (n_ch, n_sources) | BEM 前向建模 | Phase C | 源空间 LFM 概率投影 |
| `sc_matrix` | (n_regions, n_regions) | HCP tractography | Phase C | SC-FC 耦合图 |

---

#### 4.2 宽建图 (Broad Graph Construction) — Simpson Index 归一化共激活

> "建图宽进，剪枝严出。" 先把所有有意义的连接保留下来，用统计学上正确的指标度量，再在下一步精确剪枝。

##### 4.2.1 为什么不能直接用 Co-activation Count 建边？

**致命缺陷：基础率偏差 (Base Rate Bias)**

假设节点 A 只有 10 次 HFO，节点 B 有 1000 次。A 的 10 次**全部**伴随 B 发生（100% 必然跟随）：

| 指标 | 计算 | 结果 | 问题 |
|------|------|------|------|
| Raw Count | $\|E_A \cap E_B\| = 10$ | 10 | 被 B 的 1000 次淹没，看起来"不重要" |
| Jaccard | $\frac{10}{10 + 1000 - 10}$ | 1% | 分母被 B 的规模稀释 |
| Dice | $\frac{2 \times 10}{10 + 1000}$ | 2% | 同上，稍好但仍被稀释 |
| **Simpson** | $\frac{10}{\min(10, 1000)}$ | **100%** | 完美捕捉"A 必然跟随 B" |

**在癫痫网络中，"必然跟随"比"共同活跃"更重要**：
- 真正的"起搏器"可能发放率不高，但每次发放都必然带动下游
- 传播通路节点的特征是：它的每次 HFO 都伴随上游 Source 发放
- Simpson Index 天然捕捉这种不对称的包含关系

##### 4.2.2 推荐边权指标：Simpson Index

$$W_{ij}^{Simpson} = \frac{|E_i \cap E_j|}{\min(|E_i|, |E_j|)}$$

**备选**（供对比验证）：

$$W_{ij}^{Dice} = \frac{2 \cdot |E_i \cap E_j|}{|E_i| + |E_j|}$$

| 指标 | 公式 | 偏向 | 适用场景 |
|------|------|------|----------|
| **Simpson** (推荐) | $\frac{\|E_i \cap E_j\|}{\min(\|E_i\|, \|E_j\|)}$ | 捕捉包含/跟随关系 | 癫痫传播网络（不对称耦合） |
| Dice (备选) | $\frac{2\|E_i \cap E_j\|}{\|E_i\| + \|E_j\|}$ | 对称，温和归一化 | 一般共激活网络 |
| Jaccard | $\frac{\|E_i \cap E_j\|}{\|E_i \cup E_j\|}$ | 惩罚不对称对 | ❌ 不推荐：稀释低频节点 |
| Raw Count/Ratio | $\|E_i \cap E_j\|$ 或 $/ N$ | 随率缩放 | ❌ 不推荐：高频节点主导 |

**默认选择 Simpson 的理由**：
1. 癫痫网络的核心问题是识别"谁跟随谁"，Simpson 正是度量包含关系的指标
2. Simpson 的不对称偏差会被 Step 2 的 HFO Rate (X) 剪枝校正 — 率太低的节点会被剔除
3. Simpson 对"沉默的共犯"友好 — 低频但 100% 跟随的节点不会被遗漏

##### 4.2.3 数据来源与向量化实现

**关键洞察**：所有需要的数据已存在于 `*_groupAnalysis.npz`：

```python
# 数据来源映射
intersection = coact_all_event_count[i, j]   # |E_i ∩ E_j|
event_count_i = coact_all_event_count[i, i]  # |E_i| (对角线 = 自身事件数)
event_count_j = coact_all_event_count[j, j]  # |E_j|
```

**向量化实现** (N=120, <1ms)：

```python
def build_broad_graph(
    coact_event_count: np.ndarray,    # (n_all, n_all) 共激活事件计数矩阵
    method: str = 'simpson',          # 'simpson' | 'dice'
    significance_mask: Optional[np.ndarray] = None,  # surrogate 检验结果
) -> np.ndarray:
    """
    从共激活计数矩阵构建归一化边权图。

    Simpson: W_ij = |E_i ∩ E_j| / min(|E_i|, |E_j|)
    Dice:    W_ij = 2|E_i ∩ E_j| / (|E_i| + |E_j|)

    Returns: (n_all, n_all) 对称边权矩阵, 值域 [0, 1]
    """
    intersection = coact_event_count.astype(np.float64).copy()
    events_count = np.diag(coact_event_count).astype(np.float64)  # |E_i|
    np.fill_diagonal(intersection, 0.0)

    if method == 'simpson':
        denom = np.minimum.outer(events_count, events_count)
    elif method == 'dice':
        denom = np.add.outer(events_count, events_count) / 2.0
    else:
        raise ValueError(f"Unknown method: {method}. Use 'simpson' or 'dice'.")

    W = np.divide(
        intersection, denom,
        out=np.zeros_like(intersection),
        where=denom > 0,
    )

    # 对称化（Simpson 可能因浮点不完全对称）
    W = np.maximum(W, W.T)

    # 显著性门控（可选）
    if significance_mask is not None:
        W[~significance_mask] = 0.0

    np.fill_diagonal(W, 0.0)
    return W
```

##### 4.2.4 Surrogate 显著性检验（保留，逻辑不变）

> 共激活的"统计显著"不等于"物理真实"。即使用了 Simpson 归一化，也必须验证观测值是否显著高于随机。

```python
def surrogate_significance_test(
    events_bool: np.ndarray,       # (n_ch, n_events) 参与mask
    n_surrogates: int = 200,       # 替代数据集数量
    p_threshold: float = 0.05,     # 显著性阈值
) -> np.ndarray:
    """
    独立循环平移各通道事件序列生成替代数据集，
    验证真实 Simpson/Dice 共激活是否显著高于随机预期。

    Returns: (n_ch, n_ch) bool — 显著性 mask
    """
    # 实现逻辑同之前：circular shift → 重算 → p-value
    ...
```

##### 4.2.5 宽建图的设计约束

**⚠️ 关键原则**：

- ✅ **宽进**：此步不做任何节点剔除，保留所有有 HFO 的通道
- ✅ **归一化**：Simpson/Dice 消除基础率偏差
- ✅ **统计门控**：Surrogate 剔除随机重合边（可选但推荐）
- ❌ **不做阈值剪枝**：不设 `min_coact` — 那是 Step 2 的活
- ❌ **不做节点选择**：不在这里用谱聚类 — 那也是 Step 2 的活
- ❌ **不做距离约束**：Phase A 无 MNI 坐标，Phase B 再加

**输出**：`W_broad` — (n_all, n_all) 归一化的对称边权矩阵，值域 [0, 1]

---

#### 4.3 XYZ 多维剪枝 (Multi-Dimensional Pruning) — 从广泛图中提取病理网络

> "建图宽进，剪枝严出。" 三个正交维度，每个维度瞄准一类特定的噪声源。

##### 4.3.1 三维度框架总览

```
              高                    ┌──────────────────────┐
               │                    │  SOZ 核心 (保留)      │
               │       ┌────────────┤  高率 + 低熵 + 高 Z   │
    X: HFO    │       │            └──────────────────────┘
    Rate       │       │
   (活跃度)    │       │    ┌──────────────────────┐
               │       │    │  参考伪迹 (剔除)      │
               │       │    │  高率 + 高熵           │
              低       │    └──────────────────────┘
               ─────────┼──────────────────────────────→
              低        │                            高
                 Y: Connection Entropy (特异性)
```

| 维度 | 指标 | 物理意义 | 剪枝方向 | Phase |
|------|------|----------|----------|-------|
| **X (Activity)** | HFO Rate ($events/min$) | 节点是否是病理活动的活跃发生者 | 保留 $X > X_{min}$ | **A** |
| **Y (Specificity)** | Connection Entropy $\hat{H}_i$ | 连接是特异性的还是全脑弥散的 | 保留 $\hat{H} < H_{max}$ | **A** |
| **Z (Epileptogenicity)** | FR/R Ratio 或 谱聚类(XYZ距离) | 节点的致痫性特异度 | Phase A: 谱聚类; Phase B: FR比例 | **A/B** |

##### 4.3.2 维度 X — HFO Rate (活跃度)

$$X_i = \frac{|E_i|}{T_{recording}} \quad (\text{events/min})$$

- **物理意义**：节点是否产生足够多的 HFO 来被纳入网络分析
- **剪枝逻辑**：$X_i \geq X_{min}$
- **默认阈值**：`min_rate = 0.5 events/min`（每2分钟至少1次 HFO）
- **⚠️ 不要设太高**：真正的"起搏器"可能发放率不高但每次必然带动下游（Simpson 已捕捉这种关系）

##### 4.3.3 维度 Y — Connection Entropy (特异性) 🔑 核心创新

**定义**：给定节点 $i$ 在宽建图 $W$ 中的连接权重分布：

$$p_{ij} = \frac{W_{ij}}{\sum_{k \neq i} W_{ik}}, \quad H_i = -\sum_{j \neq i} p_{ij} \ln p_{ij}$$

**归一化熵**（映射到 [0, 1]）：

$$\hat{H}_i = \frac{H_i}{\ln(N_{neighbors,i})}$$

其中 $N_{neighbors,i}$ = 节点 $i$ 的非零连接数。

**物理解释**：

| $\hat{H}_i$ | 含义 | 网络角色 | 判定 |
|---|---|---|---|
| **≈ 0** | 连接集中于1-2个节点 | 高度特异的"共犯关系" | ✅ 保留（局灶性传播通路） |
| **0.3 - 0.6** | 中等分散 | 有选择性的 Hub | ✅ 保留 |
| **≈ 1.0** | 均匀连接所有节点 | 全脑同步（伪迹/噪声） | ❌ 剔除 |

**为什么 Connection Entropy 是剔除 Global Artifacts 的"神技"**：

Reference contamination 的数学特征：一个通道因共参考电极而与所有通道产生虚假"共激活"。在 Simpson 空间中，这个通道与每个其他通道的 Simpson 值都 > 0（因为它的每次 HFO 都"伴随"很多通道）。**但它的连接分布接近均匀** → $\hat{H} \approx 1.0$。

真正的病理通道只与网络内的特定"共犯"高度同步 → $\hat{H}$ 显著低于 1.0。

这比传统的"剔除与太多通道连接的节点"更精确——它不关心你连了多少通道，而关心你的连接是否有**选择性**。

**剪枝逻辑**：$\hat{H}_i < H_{max}$，默认 `max_entropy = 0.85`

```python
def compute_connection_entropy(W: np.ndarray) -> np.ndarray:
    """
    计算每个节点的归一化连接熵。

    Parameters
    ----------
    W : (n, n) 边权矩阵 (Simpson/Dice，对角线为0)

    Returns
    -------
    H_norm : (n,) 归一化熵，0=极度特异，1=均匀弥散
    """
    n = W.shape[0]
    H_norm = np.ones(n, dtype=np.float64)  # 默认最大熵（最坏情况）

    for i in range(n):
        w_i = W[i].copy()
        w_i[i] = 0.0
        total = w_i.sum()
        if total < 1e-10:
            continue  # 孤立节点，保持默认
        p = w_i / total
        nonzero = p > 0
        n_neighbors = nonzero.sum()
        if n_neighbors < 2:
            H_norm[i] = 0.0  # 只有1个连接 = 最大特异性
            continue
        H = -np.sum(p[nonzero] * np.log(p[nonzero]))
        H_max = np.log(n_neighbors)
        H_norm[i] = H / H_max if H_max > 0 else 1.0

    return H_norm
```

##### 4.3.4 维度 Z — Epileptogenicity (致痫性)

**Phase A（无 FR 分类数据）**：

在 X-Y 空间中用谱聚类，以 Simpson 连接权重为亲和度、以 XY 特征为辅助距离度量：

$$A_{ij}^{cluster} = W_{ij}^{Simpson} \times \exp\left(-\frac{(\hat{X}_i - \hat{X}_j)^2 + (\hat{Y}_i - \hat{Y}_j)^2}{2\sigma^2}\right)$$

- 谱聚类在此作为"自适应社区发现"工具
- Eigengap 自动确定聚类数（不硬编码 N=8）
- 小于 `min_cluster_size` 的孤立簇被标记为噪声

**Phase B（有 FR 分类数据后）**：

$$Z_i = \frac{N_{FR,i}}{N_{Ripple,i} + N_{FR,i}}$$

**更激进的 XYZ 距离度量**（Phase B）：

$$d_{ij}^{XYZ} = \sqrt{w_X(\hat{X}_i - \hat{X}_j)^2 + w_Y(\hat{Y}_i - \hat{Y}_j)^2 + w_Z(\hat{Z}_i - \hat{Z}_j)^2}$$

谱聚类使用 $A_{ij} = W_{ij}^{Simpson} \times \exp(-d_{ij}^{XYZ}/2\sigma^2)$ 作为亲和矩阵，同时编码**连接强度**和**病理特征相似性**。

##### 4.3.5 完整剪枝 API

```python
def compute_node_xyz(
    W_broad: np.ndarray,               # (n_all, n_all) Simpson 宽建图
    events_count: np.ndarray,           # (n_all,) 每通道 HFO 事件数
    recording_duration_min: float,      # 记录时长（分钟）
    fr_ratio: Optional[np.ndarray] = None,  # (n_all,) FR/(R+FR) (Phase B)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算每个节点的 XYZ 三维病理特征。

    Returns
    -------
    X : (n_all,) HFO Rate (events/min)
    Y : (n_all,) Normalized Connection Entropy (0=specific, 1=diffuse)
    Z : (n_all,) Epileptogenicity (Phase A: zeros; Phase B: FR ratio)
    """
    X = events_count.astype(np.float64) / max(recording_duration_min, 1e-6)
    Y = compute_connection_entropy(W_broad)
    Z = fr_ratio.copy() if fr_ratio is not None else np.zeros_like(X)
    return X, Y, Z


def prune_network(
    W_broad: np.ndarray,               # (n_all, n_all) 宽建图
    X: np.ndarray,                     # HFO Rate
    Y: np.ndarray,                     # Connection Entropy
    Z: np.ndarray,                     # Epileptogenicity
    *,
    min_rate: float = 0.5,             # X: 最低 HFO Rate (events/min)
    max_entropy: float = 0.85,         # Y: 最高归一化连接熵
    use_spectral: bool = True,         # Z: 在 XY+Simpson 空间做谱聚类
    min_cluster_size: int = 3,
    n_clusters: Optional[int] = None,  # None = Eigengap 自动
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    XYZ 多维剪枝：从宽建图中提取病理网络核心。

    Pipeline:
    1. X 门控 → 剔除低活跃度节点
    2. Y 门控 → 剔除高熵（全脑弥散）节点
    3. Z 门控 → 谱聚类 (Phase A) 或 FR 比例筛选 (Phase B)

    Returns
    -------
    selected_idx : (n_sel,) 入选节点的全通道池索引
    W_pruned : (n_sel, n_sel) 剪枝后的边权子图
    cluster_labels : (n_all,) 聚类标签 (-1=剔除)
    """
    n = W_broad.shape[0]
    labels = np.full(n, -1, dtype=np.int32)

    # Step 1: X 门控 — 活跃度
    x_pass = X >= min_rate

    # Step 2: Y 门控 — 特异性（剔除高熵 = 伪迹/弥散噪声）
    y_pass = Y <= max_entropy

    # 联合 mask
    node_mask = x_pass & y_pass
    candidate_idx = np.where(node_mask)[0]

    if len(candidate_idx) < min_cluster_size + 1:
        # 候选太少，退化为全部保留
        selected_idx = candidate_idx
        labels[candidate_idx] = 0
    elif use_spectral:
        # Step 3: 谱聚类 → 识别病理网络社区，剔除孤立噪声
        W_sub = W_broad[np.ix_(candidate_idx, candidate_idx)]
        sub_labels, _ = extract_network_clusters(
            W_sub, min_cluster_size, n_clusters,
        )
        # 映射回全通道索引
        for si, ci in enumerate(candidate_idx):
            labels[ci] = sub_labels[si]
        selected_idx = candidate_idx[sub_labels >= 0]
    else:
        selected_idx = candidate_idx
        labels[candidate_idx] = 0

    W_pruned = W_broad[np.ix_(selected_idx, selected_idx)]
    return selected_idx, W_pruned, labels
```

##### 4.3.6 典型案例：XYZ 如何区分真网络与伪迹

| 场景 | HFO Rate (X) | Entropy (Y) | FR Ratio (Z) | 判定 |
|------|---|---|---|---|
| SOZ 核心 | 高 (8/min) | **低** (0.2) | 高 (0.6) | ✅ 保留 — 高活跃、特异连接、高致痫性 |
| 传播通路 | 中 (2/min) | **低** (0.3) | 中 (0.3) | ✅ 保留 — 活跃且有选择性 |
| 参考伪迹 | 高 (10/min) | **极高** (0.95) | 低 (0.1) | ❌ Y 剔除 — 与所有通道均匀连接 |
| 生理性 HFO | 中 (1.5/min) | 中 (0.5) | **极低** (0.02) | ⚠️ X 保留, Y 保留, Z 低 → 被谱聚类标记为边缘/噪声 |
| 安静关键节点 | **低** (0.3/min) | **极低** (0.1) | 高 (0.5) | ❌ X 剔除 — 活跃度不足（Simpson 已记录其跟随关系，后续可回溯） |

##### 4.3.7 工程约束与退化策略

**⚠️ 关键约束**：
- ❌ 不要仅用白质标签剔除 — 灰质异位/脑室周围结节位于深部白质但 HFO 高发
- ❌ 不要硬编码通道数 — 让谱聚类的 Eigengap 或 XY 阈值自适应决定
- ✅ 所有阈值（min_rate, max_entropy）必须可配置 — 患者间差异大
- ✅ Eigengap 不稳定时，`n_clusters` 可手动覆盖
- ✅ 被剔除的节点信息保留在 `cluster_labels` 中，可随时回溯

**退化策略**（当 XYZ 剪枝过于激进时）：

```python
# 保底：只用 X 门控 + 弱 Y 门控
selected = np.where((X >= min_rate) & (Y <= 0.95))[0]
```

---

#### 4.4 方向注入 (Direction Injection) — 剪枝图升级为有向图

**核心改进：统计鲁棒性**

> **批判**：直接用中位数 Lag 定向是危险的。多峰分布（直接通路 5ms + 间接通路 25ms）的中位数 15ms 在物理上没有意义。

**鲁棒方向判定流程**：

```python
from scipy.stats import wilcoxon

def inject_direction(
    W_pruned: np.ndarray,           # (n_sel, n_sel) 剪枝后的 Simpson 边权图
    lag_raw: np.ndarray,            # (n, n_events) 质心时间
    events_bool: np.ndarray,        # (n, n_events) 参与mask
    min_events: int = 5,            # 最小样本量
    lag_thresh_ms: float = 5.0,     # 零滞后阈值
    consistency_thresh: float = 0.6, # 方向一致性阈值
    p_value_thresh: float = 0.05,   # 显著性阈值
) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回:
        adj_directed: (n, n) 有向邻接矩阵，A[i,j] = i→j 的权重
        edge_stats: dict 包含每条边的统计信息
    """
    n = skeleton.shape[0]
    adj = np.zeros((n, n), dtype=np.float64)
    
    for i in range(n):
        for j in range(i+1, n):
            if not skeleton[i, j]:
                continue
            
            # 提取共同事件的 lag
            mask = events_bool[i] & events_bool[j]
            if mask.sum() < min_events:
                continue
            
            lags = lag_raw[i, mask] - lag_raw[j, mask]  # 负 = i 领先
            
            # === 统计检验 ===
            
            # 1. Wilcoxon 检验：Lag 是否显著异于 0？
            try:
                _, p_val = wilcoxon(lags)
            except ValueError:  # 全零或样本太少
                continue
            
            if p_val > p_value_thresh:
                continue  # 零滞后同步，不定向
            
            # 2. 方向一致性检验
            median_lag = np.median(lags)
            if abs(median_lag) < lag_thresh_ms * 1e-3:
                continue  # 太接近零，不定向
            
            direction = np.sign(median_lag)
            consistency = np.mean(np.sign(lags) == direction)
            
            if consistency < consistency_thresh:
                continue  # 方向太乱，视为湍流
            
            # 3. 赋予方向
            if median_lag < 0:  # i 领先 j
                adj[i, j] = consistency
            else:              # j 领先 i
                adj[j, i] = consistency
    
    return adj
```

**关键统计保护**：

| 检验 | 目的 | 失败处理 |
|------|------|---------|
| Wilcoxon | Lag ≠ 0？ | 不定向（视为同步） |
| 一致性 | 方向稳定？ | 不定向（视为湍流） |
| 样本量 | n ≥ 5？ | 不建边（数据不足） |

---

#### 4.5 权重定义 (Weight Definition) — 多维复合权重

> 单一权重无法捕捉致痫网络的复杂性。必须融合因果性、稳定性与病理特异性。

##### 4.5.1 三维权重模型

$$W_{ij} = \underbrace{\text{Simpson}_{ij} \times \text{Consistency}_{ij}}_{\text{Causality（因果性）}} \times \underbrace{(1 - \text{CV}_{time}^{ij})}_{\text{Stability（稳定性）}} \times \underbrace{\left(1 + \alpha \cdot \frac{N_{FR}^{ij}}{N_{total}^{ij}}\right)}_{\text{Pathology（病理性）}}$$

| 维度 | 定义 | 数据来源 | Phase |
|------|------|----------|-------|
| **Causality** | $\text{Simpson}_{ij} \times \text{Consistency}_{ij}$ — Simpson 归一化共激活 × 方向一致性 | `W_pruned` + `lag_raw` | **A (立即可做)** |
| **Stability** | $1 - \text{CV}(\text{Connectivity}(t))$ — 连接的时间鲁棒性 | `event_windows` 按时间窗切片 | **A (立即可做)** |
| **Pathology** | $1 + \alpha \cdot \frac{N_{FR}}{N_{total}}$ — Fast Ripple 比例加权 | `hfo_type_per_event` | **B (需分类数据)** |

##### 4.5.2 Stability（稳定性）维度 — 时间鲁棒性

**核心思想**：癫痫网络应具有刻板性（Stereotypical），随机出现的连接是噪声。

```python
def compute_stability_weights(
    lag_raw: np.ndarray,           # (n_ch, n_events) 质心时间
    events_bool: np.ndarray,       # (n_ch, n_events) 参与mask
    event_times: np.ndarray,       # (n_events,) 事件时间戳
    window_sec: float = 300.0,     # 5分钟时间窗
    min_windows: int = 3,          # 最少窗口数
) -> np.ndarray:
    """
    计算每条边在多个时间窗内的连接方向一致性。
    
    Stability = 1 - CV(consistency_per_window)
    高稳定性 = 固定的病理通路；低稳定性 = 瞬态噪声
    """
    n_ch = lag_raw.shape[0]
    stability = np.full((n_ch, n_ch), np.nan)
    
    # 按时间窗切片
    t_min, t_max = event_times.min(), event_times.max()
    edges = np.arange(t_min, t_max, window_sec)
    if len(edges) < min_windows:
        return np.ones((n_ch, n_ch))  # 数据不够，退化为权重1
    
    window_consistencies = []
    for t_start in edges:
        t_end = t_start + window_sec
        win_mask = (event_times >= t_start) & (event_times < t_end)
        if win_mask.sum() < 5:
            continue
        
        # 每个时间窗内计算方向一致性
        cons = np.zeros((n_ch, n_ch))
        for i in range(n_ch):
            for j in range(i+1, n_ch):
                both = events_bool[i, win_mask] & events_bool[j, win_mask]
                if both.sum() < 3:
                    continue
                lags = lag_raw[i, win_mask][both] - lag_raw[j, win_mask][both]
                med = np.median(lags)
                cons[i, j] = np.mean(np.sign(lags) == np.sign(med))
                cons[j, i] = cons[i, j]
        window_consistencies.append(cons)
    
    if len(window_consistencies) < min_windows:
        return np.ones((n_ch, n_ch))
    
    stacked = np.stack(window_consistencies)
    mean_cons = np.nanmean(stacked, axis=0)
    std_cons = np.nanstd(stacked, axis=0)
    cv = np.where(mean_cons > 0, std_cons / mean_cons, 1.0)
    stability = 1.0 - np.clip(cv, 0, 1)
    
    return stability
```

##### 4.5.3 Pathology（病理性）维度 — 频率特异性

**设计理由**（参考文献：SpikewHFO更重要.pdf）：
- 叠加 HFO 的 Spike 比单纯 Spike 更能定位 SOZ
- Fast Ripple 比 Ripple 更具病理特异性
- 给高病理性传播事件更高投票权

**Phase A（立即可做）**：用 Coact × Consistency × Stability 三维权重

**Phase B（需 FR 分类数据后）**：加入 $(1 + \alpha \cdot FR_{ratio})$ 因子，$\alpha$ 建议 0.5-2.0

##### 4.5.4 进阶方向：频谱因果性（Phase C 研究前沿）

> 用频谱格兰杰因果 (Spectral GC) 或偏定向相干 (PDC) 替代 Lag-based 因果推断。

$$W_{ij}^{advanced} = \underbrace{\text{PDC}_{ij}(f_{HFO})}_{\text{频域因果}} \times \underbrace{(1 - \text{CV}_{time})}_{\text{稳定性}} \times \underbrace{\frac{SC_{ij}}{SC_{max}}}_{\text{解剖先验}} \times \underbrace{\text{PathScore}_i}_{\text{节点病理分}}$$

**为什么列为 Phase C**：PDC 需要模型阶数选择（AIC/BIC）、平稳性检验、$O(N^2 \times T \times p)$ 计算。对 50 通道 × 2h 数据虽然可行但调参复杂。先用 Lag-based 方法验证整体流程，再考虑替换为 PDC。

---

#### 4.6 图论指标计算 (Metric Calculation)

**使用 `networkx` 库**：

| 指标 | 公式 | 临床意义 |
|------|------|---------|
| **Net Outflow Index** | $\frac{OutDegree - InDegree}{OutDegree + InDegree}$ | **SOZ 定位**：值接近 +1 = Source |
| **Outflow Volatility** | $\text{Var}(\text{NetOutflow}_t)$ | 真正 SOZ 往往发作前突然爆发 |
| **Local Efficiency** | $E_{loc}(i) = \frac{1}{k_i(k_i-1)} \sum_{j,h \in N_i} \frac{1}{d_{jh}}$ | 致痫灶的紧密程度 |
| **Shortest Path Tree** | 从 Source 出发的最短路径 | 传播路径预测 |

```python
import networkx as nx

def compute_network_metrics(adj: np.ndarray, ch_names: List[str]) -> Dict:
    """
    计算核心图论指标。
    """
    G = nx.DiGraph()
    n = adj.shape[0]
    
    for i in range(n):
        for j in range(n):
            if adj[i, j] > 0:
                G.add_edge(ch_names[i], ch_names[j], weight=adj[i, j])
    
    metrics = {}
    
    # Net Outflow Index
    for node in G.nodes():
        out_deg = G.out_degree(node, weight='weight')
        in_deg = G.in_degree(node, weight='weight')
        total = out_deg + in_deg
        metrics[f'{node}_outflow'] = (out_deg - in_deg) / total if total > 0 else 0
    
    # Local Efficiency (需要转无向图)
    G_undirected = G.to_undirected()
    metrics['local_efficiency'] = nx.local_efficiency(G_undirected)
    
    # Betweenness Centrality
    metrics['betweenness'] = nx.betweenness_centrality(G, weight='weight')
    
    return metrics
```

---

#### 4.7 关键陷阱与防护 (Critical Pitfalls)

**陷阱1：容积传导的幽灵 (Volume Conduction)**

| 现象 | 物理距离 <10mm，Lag ≈ 0，Co-activation 极高 |
|------|-------------------------------------------|
| 原因 | 电场直接传导，非神经元传播 |
| 危害 | 网络被无意义短边主导 |
| **防护** | 强制剔除 `dist_matrix < 10mm` 的边 |
| **反向利用** | 保留局部连接强度作为 "Local Recruitment Score" |
| **零滞后陷阱** | 深部强源被两个远距电极同时记录 → 高同步但零延迟<br>必须用 PLI/wPLI（对零滞后不敏感）或 Wilcoxon 检验过滤 |

**陷阱2：采样偏差 (Sampling Bias)**

| 现象 | SEEG 仅覆盖不到 1% 的脑体积 |
|------|------------------------------|
| 危害 | 真正的源在未采样区，中继站被误判为源 |
| **防护** | 结论必须谨慎：<br>"在被监测的网络中，节点 X 表现出源的特征" |

**陷阱3：中位数陷阱 (The Median Trap)**

| 现象 | Lag 分布多峰（直接通路 5ms + 间接通路 25ms） |
|------|-------------------------------------------|
| 危害 | 中位数 15ms 在物理上不存在 |
| **防护** | 单峰性检验 (Hartigan's dip test) 或方差检查<br>高方差边标记为 "Unstable Connection"，降低权重 |

**陷阱4：静态网络的局限**

| 现象 | 24小时平均图抹杀时间维度 |
|------|-------------------------|
| 危害 | 间歇性喷发的 SOZ 被持续活跃的中继站掩盖 |
| **进阶方向** | 动态切片：每 5 分钟或每 100 事件计算一次<br>比较 Pre-ictal vs Interictal 网络拓扑 |

**陷阱5：生理性 HFO 混淆 (Physiological HFO Contamination)** 🔴 新增

| 现象 | 视觉/运动皮层和海马在 NREM 期间产生高发放率生理性 HFO |
|------|--------------------------------------------------|
| 危害 | 功能区被误判为致痫灶 → 手术导致功能缺损 |
| **防护** | 谱聚类 + 共激活过滤 — 生理性 HFO 往往是孤立的局部功能柱活动，不形成大尺度同步网络<br>Stability 权重 — 生理性 HFO 是任务/状态相关的瞬态，病理性更持续 |
| **补充** | 结合 Spike-HFO 共现特征：叠加 Spike 的 HFO 病理特异性更高 |

**陷阱6：Sink/Source 反转 (Sink Trap)** 🔴 新增

| 现象 | SOZ 在发作间期可能表现为 Sink（被抑制），发作期转为 Source |
|------|------------------------------------------------------|
| 危害 | 仅分析发作间期数据会将 SOZ 误判为"接收节点" |
| **防护** | 必须结合 Ictal 数据验证：寻找"间期 Sink → 发作期 Source"的动态反转节点<br>这种反转本身就是 EZ 的"指纹"特征 |
| **指标** | $\Delta \text{Outflow} = \text{Outflow}_{ictal} - \text{Outflow}_{interictal}$ — 反转幅度最大的节点 |

**陷阱7：SEEG 行波假设失效 (Traveling Wave Caveat)** 🔴 新增

| 现象 | HFO/IED 在皮层上表现为行波（Traveling Waves） |
|------|---------------------------------------------|
| 危害 | 在 Grid 电极上可直接拟合波峰梯度场计算传播速度矢量<br>**但 SEEG 是棒状深部电极**，穿过不同皮层层级，2D 平面波假设失效 |
| **防护** | 在 SEEG 中必须沿电极轴向（Axial）和跨电极（Cross-electrode）分别计算延迟<br>不可盲目拟合平面波 |
| **方向反转** | IED 传播方向通常**指向**致痫灶（Sink 特征）<br>Ictal Discharge 通常**背离**致痫灶传播<br>这一方向反转是重要的定位特征 |

---

#### 4.8 三阶段实施路线图 (Three-Phase Roadmap)

> "Theory and practice sometimes clash. Theory loses. Every single time." — 先用现有数据跑通全流程，再逐步加入高级特征。

##### Phase A：Channel-Scale MVP（数据已就绪，立即可做）

| Step | 任务 | 输入 | 输出 | 新增依赖 | 状态 |
|------|------|------|------|----------|------|
| A.1 | **宽建图 (Simpson Index)** | `coact_all_event_count` | `W_simpson/W_dice` (n_all, n_all) | — | ✅ |
| A.2 | 替代数据显著性检验 | `events_bool` | `sig_mask` | — | ✅ |
| A.3 | **XYZ 特征计算** | `W_broad`, `events_count`, `duration` | `X, Y, Z` per node | — | ✅ |
| A.4 | **XYZ 多维剪枝** | `W_broad`, `X`, `Y`, `Z` | `selected_idx`, `W_pruned` | `sklearn` (谱聚类) | ✅ |
| A.5 | 方向注入（Wilcoxon+一致性） | `W_pruned`, `lag_raw` | `adj_directed` | `scipy.stats` | ✅ |
| A.6 | Stability 权重 | `lag_raw`, `event_windows` | `stability_matrix` | — | ✅ |
| A.7 | 复合/融合权重（direction-first） | `assoc/B/D/S/lag` | `adj_weighted` | — | ✅ |
| A.8 | 图论指标 | `adj_weighted` | `metrics_dict` | `networkx` | ✅ |
| A.9 | 2D 网络拓扑图 + XY 散点诊断图 | `metrics`, `X`, `Y` | `network_plot.png` | `matplotlib` | ✅ |

**Phase A 的交付物**：
1. 一个完整的 Channel-scale 有向加权癫痫网络（Simpson 归一化 + XYZ 剪枝）
2. XY 散点诊断图：直观展示哪些节点被保留/剔除及原因
3. Net Outflow 排名（Source-Sink 预测）

##### Phase B：Channel-Scale + Geometry（需 MNI 坐标）

| Step | 任务 | 输入 | 输出 | 阻塞条件 | 状态 |
|------|------|------|------|----------|------|
| B.0 | 电极坐标获取 | MNI 配准结果 | `mni_coords.npy`, `dist_matrix.npy` | **需临床数据** | ⬜ |
| B.1 | 空间约束骨架 | `dist_matrix`, `coact_ratio` | `skeleton_spatial` | B.0 | ⬜ |
| B.2 | 容积传导剔除 | `dist_matrix < 10mm` | `skeleton_clean` | B.0 | ⬜ |
| B.3 | 传播速度验证 | `dist_matrix`, `lag_raw` | `velocity diagnostics` (0.1-10 m/s) | B.0 | 🔄 |
| B.4 | 病理加权（FR 比例） | `hfo_type_per_event` | `pathology_weight` | **需 FR 分类** | ⬜ |
| B.5 | 3D 脑图 | `metrics`, `mni_coords` | `outflow_brain_3d.html` | B.0 | ⬜ |
| B.6 | Ictal vs Interictal 对比 | `event_windows`, `seizure_onsets` | `delta_outflow` | — | 🔄 |

**Phase B 的交付物**：物理约束后的网络 + 3D 可视化 + Sink/Source 反转分析。

##### Phase C：Source-Scale 研究前沿（需影像学流水线）

| Step | 任务 | 输入 | 阻塞条件 | 状态 |
|------|------|------|----------|------|
| C.1 | 前向模型(BEM) | FreeSurfer 输出, 电极坐标 | 需 MRI 分割 + 配准 | ⬜ |
| C.2 | 导联场矩阵(LFM) | BEM 模型 | C.1 | ⬜ |
| C.3 | LFM 概率投影 | `LFM`, `channel_metrics` | C.2 | ⬜ |
| C.4 | SC-FC 耦合图 | HCP tractography | 需 DWI 数据 | ⬜ |
| C.5 | PDC/频谱格兰杰 | 原始时间序列 | 计算密集 | ⬜ |
| C.6 | NMM 验证 | 连接矩阵 | 独立研究课题 | ⬜ |

**Phase C 的交付物**：源空间级别的病理网络重构（研究论文级别）。

---

**核心 API 设计**：

```python
# src/network_analysis.py

@dataclass
class NetworkResult:
    """癫痫网络分析结果 (v3: direction-first causal)."""
    adj: np.ndarray
    node_names: List[str]
    node_weights: np.ndarray
    W_simpson: np.ndarray
    W_dice: np.ndarray
    W_pruned: np.ndarray
    pool_names: List[str]
    selected_idx: np.ndarray
    node_xyz: Dict[str, np.ndarray]
    skeleton: np.ndarray
    direction_mask: np.ndarray
    stability: np.ndarray
    cluster_labels: np.ndarray
    metrics: Dict[str, Any]
    edge_stats: List[Dict]
    n_pool_channels: int
    n_selected: int
    params: Dict[str, Any]

def build_hfo_network(
    group_analysis_npz: str,
    dist_matrix: Optional[np.ndarray] = None,
    *,
    detections_npz_path: Optional[str] = None,
    # — Step 1: 宽建图 —
    edge_method: str = 'simpson',     # 'simpson' | 'dice'
    run_surrogate: bool = True,
    n_surrogates: int = 200,
    # — Step 2: XYZ 剪枝 —
    min_rate: float = 0.5,            # X: 最低 HFO Rate (events/min)
    max_entropy: float = 0.85,        # Y: 最高归一化连接熵
    use_spectral: bool = True,        # Z: 谱聚类进一步剪枝
    min_cluster_size: int = 3,
    n_clusters: Optional[int] = None, # None = Eigengap 自动
    # — Step 3: 方向注入 —
    min_events: int = 5,
    lag_thresh_ms: float = 5.0,
    consistency_thresh: float = 0.6,
    p_value_thresh: float = 0.05,
    # — Step 4: 稳定性 / direction-first 融合 —
    stability_window_sec: float = 300.0,
    assoc_window_ms: float = 40.0,
    min_pair_events: int = 5,
    tau_assoc: float = 20.0,
    tau_lag_ms: float = 10.0,
    fusion_w_b: float = 0.35,
    fusion_w_d: float = 0.45,
    fusion_w_s: float = 0.20,
    d_strong: float = 0.35,
    b_min: float = 0.2,
    min_dist_mm: float = 10.0,
    lag_vc_ms: float = 3.0,
    sample_cap_per_edge: int = 50000,
) -> NetworkResult:
    """
    一站式构建癫痫网络 v3（direction-first causal）。

    流程：pairwise association → direction injection → fusion + physics →
         final pruning + node rescue → metrics

    Returns
    -------
    NetworkResult : 包含有向加权邻接矩阵、XYZ 特征和图论指标
    """
    ...
```

**可视化终极目标**：

```python
def plot_outflow_brain_map(
    network_result: NetworkResult,
    mni_coords: np.ndarray,
    output_path: str,
):
    """
    3D 脑图（Phase B 交付物）：
    - 节点颜色 = Net Outflow (红=Source, 蓝=Sink)
    - 节点大小 = Local Efficiency
    - 边颜色 = 传播方向
    - 边粗细 = 权重

    这是直接对话临床医生的"终极图表"。
    """
    ...

def plot_network_topology_2d(
    network_result: NetworkResult,
    output_path: str,
    layout: str = 'spring',
):
    """
    2D 网络拓扑图（Phase A 交付物）：
    - 节点颜色 = Net Outflow
    - 节点大小 = Node Weight (EigenCentrality × Rate)
    - 边粗细 = 复合权重
    - 布局 = spring / circular / spectral

    不需要 MNI 坐标，Phase A 即可生成。
    """
    ...
```

---

#### 4.9 功能清单 (Feature Checklist)

**Phase A — Channel-Scale MVP（立即可做）**

| 功能 | 说明 | 依赖 | 状态 |
|------|------|------|------|
| A.1 宽建图 (Simpson) | Simpson Index 归一化共激活 → 宽边权图 | `coact_all_event_count` | ✅ |
| A.2 替代数据检验 | Surrogate test 验证共激活显著性 | `events_bool` | ✅ |
| A.3 XYZ 特征计算 | X=Rate, Y=Connection Entropy, Z=placeholder | `W_broad` | ✅ |
| A.4 XYZ 多维剪枝 | X门控 + Y门控 + 谱聚类(XY距离) | `sklearn` | ✅ |
| A.5 方向注入 | Wilcoxon + 一致性 + 零滞后过滤 | `lag_raw`, `scipy.stats` | ✅ |
| A.6 Stability 权重 | 时间窗切片 + CV 计算 | `event_windows` | ✅ |
| A.7 复合权重 | direction-first 融合权重 | — | ✅ |
| A.8 图论指标 | Net Outflow, Local Efficiency, Betweenness | `networkx` | ✅ |
| A.9 2D 网络拓扑图 | Spring 布局 + XY 散点诊断图 | `matplotlib` | ✅ |
| A.10 传播路径 | 最短路径树 | `adj` | ✅ |

**Phase B — Channel + Geometry（需 MNI 坐标）**

| 功能 | 说明 | 阻塞条件 | 状态 |
|------|------|----------|------|
| B.1 空间约束骨架 | 距离惩罚 + 容积传导剔除 | MNI coords | ⬜ |
| B.2 传播速度验证 | 0.1-10 m/s 生理范围检查 | MNI coords | 🔄 |
| B.3 病理加权 | FR 比例加权 | HFO type 分类 | ⬜ |
| B.4 3D 脑图 | Outflow 颜色映射 | MNI coords | ⬜ |
| B.5 动态切片 | Pre-ictal vs Interictal 网络对比 | Seizure onsets | ⬜ |
| B.6 Sink/Source 反转 | $\Delta$Outflow (ictal - interictal) | B.5 | 🔄 |

**Phase C — Source Space 研究前沿**

| 功能 | 说明 | 阻塞条件 | 状态 |
|------|------|----------|------|
| C.1 前向模型(BEM/FEM) | 患者个性化导联场 | FreeSurfer + MRI | ⬜ |
| C.2 LFM 概率投影 | 灵敏度加权映射 | C.1 | ⬜ |
| C.3 SC-FC 耦合图 | 解剖先验约束 | HCP tractography | ⬜ |
| C.4 PDC/频谱格兰杰 | 频域因果性 | 模型阶数选择 | ⬜ |
| C.5 NMM 验证 | 分析-综合闭环 | 独立研究课题 | ⬜ |

---

#### 4.10 源空间构建远景 (Source Space Vision) — Phase C 理论基础

> 本节记录 Source-Scale 网络构建的理论基础和工程路径。**当前不实现**，作为研究前沿参考。

##### 4.10.1 粒度困境 (Granularity Dilemma)

| 尺度 | 分辨率 | 失效原因 |
|------|--------|---------|
| **脑区 (AAL/DK)** | ~100 区域 | HFO 生成器 <2mm，脑区平均化彻底淹没病理信号，SNR 指数下降 |
| **顶点 (Vertex)** | ~20k/半球 | SEEG 仅 100-200 触点 → 极度欠定逆问题 → 无数据区域的"插值幻觉" |
| **传感体积 (VOI)** | **5mm 半径** | ✅ 匹配 SEEG 宏电极传感半径 (~3-5mm)<br>✅ 包含 HFO 微观发生结构<br>✅ 避免过度插值 |

**结论**：源空间节点应定义为以电极触点为中心的 5mm VOI（Virtual Voxels），非均匀全脑网格。

##### 4.10.2 LFM 概率投影 (Lead-Field Weighted Projection)

> 无需求解复杂的源成像逆问题。利用导联场矩阵作为几何先验进行概率映射。

**核心公式**：

$$W_{ji} = \frac{L_{ij}^2}{\sum_{k \in \text{Channels}} L_{kj}^2}$$

$$\text{SourceMetric}_j = \sum_{i} W_{ji} \cdot \text{ChannelMetric}_i$$

其中 $L_{ij}$ 是导联场矩阵中源 $j$ 对通道 $i$ 的贡献（包含距离衰减和偶极子方向信息）。使用平方是因为功率/能量随距离平方衰减。

**优势**：
- 计算高效：一次性线性变换，非迭代反演
- 物理合理：自动处理距离加权 + 方向性
- 避免 Double Counting：归一化权重自然分配重叠区域

**对共激活矩阵的映射扩展**：

$$\text{SourceCoAct}_{jk} = \sum_{m,n} W_{jm} \cdot \text{ChannelCoAct}_{mn} \cdot W_{kn}$$

##### 4.10.3 SC-FC 耦合图 (Structure-Function Coupled Graph)

> 在源空间定义传播路径时，必须引入 HCP SC 作为贝叶斯先验。

$$P(E_{A \to B} | \text{Data}) \propto \text{FC}_{A \to B} \times \text{SC}_{A \to B}$$

**物理意义**：如果源 A 到源 B 的功能连接（FC）很强，但无白质纤维束直接连接（SC ≈ 0），则该"连接"极可能是间接的或虚假的。

**工程依赖链**：
1. FreeSurfer 皮层重建 → 高分辨率 mesh
2. BEM/FEM 前向建模 → 导联场矩阵 $G$
3. 电极定位 (LeadDBS/iElectrodes) → MNI 坐标
4. HCP tractography → 结构连接矩阵 SC
5. LFM 概率投影 → 源空间指标
6. SC × FC → 耦合图

**现实评估**：这是一条 6 个月的工程路径，每个环节都需要独立验证。但一旦建成，可以实现从电生理到解剖的无缝对接，这是最终的临床转化目标。

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
| `plot_group_event_tf_propagation_from_cache` | **Fig2**: 多通道TF传播图（金标准） | `*_groupTF_tiles.npz`<br>`*_groupAnalysis.npz` | - | 展示群体事件TF域传播模式 |
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
├─ x_band: (n_ch, n_samples) bandpassed signal（可选）
├─ sfreq: 采样率
└─ ch_names: 通道名列表

# 2. 完整分析结果
"<record>_groupAnalysis.npz"
├─ centroid_time: (n_ch, n_events) 时间质心
├─ tf_centroid_time: (n_ch, n_events) TF 2D质心-时间分量（可选）
├─ tf_centroid_freq: (n_ch, n_events) TF 2D质心-频率分量（可选）
├─ lag_raw: (n_ch, n_events) 相对滞后(秒)
├─ lag_rank: (n_ch, n_events) 排名(0=最早)
├─ coact_event_count: (n_ch, n_ch) 共同激活事件数
├─ coact_event_ratio: (n_ch, n_ch) 共同激活比例
├─ coact_time_ratio: (n_ch, n_ch) 质心绝对时间对齐强度
├─ coact_rank_ratio: (n_ch, n_ch) 质心相对rank对齐强度
├─ events_bool: (n_ch, n_events) 参与mask
└─ sfreq, band, ch_names, event_windows...

# 3. 事件窗口
"<record>_packedTimes.npy"
└─ (n_events, 2) [start, end] 秒

# 4. 群体TF谱图缓存（Fig2）
"<record>_groupTF_tiles.npz"
├─ power_db: (n_ch, n_events, n_freqs, n_time) 4D TF tiles (dB)
├─ freqs_hz: (n_freqs,) 对数频率轴
├─ event_indices: (n_events,) 对应事件索引
├─ channel_names: (n_ch,) 通道列表
├─ window_sec: float 事件窗口长度
└─ sfreq: float 采样率

# 5. GPU检测结果（可选，用于mask）
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
    plot_group_event_tf_spectrogram_from_cache,
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

# Fig2: 多通道TF传播（金标准）
fig2 = plot_group_event_tf_propagation_from_cache(
    tfr_tile_cache_npz_path=out_paths['group_tf_tile_cache_path'],
    group_analysis_npz_path=out_paths['group_analysis_path'],
    channel_order=CORE_CHANNELS,
    event_indices=list(range(30)),
    plot_window_sec=0.1,  # 100ms 时间窗口
    low_color="#1f4b99",  # 蓝色背景
    low_color_percentile=70.0,  # 低于70%设为背景
    cmap="Reds",  # 红色能量映射
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
- [x] **模块1: preprocessing.py** ✅ Phase 1 重构完成 (2026-01-30); PR1 扩展 (2026-04-01)
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
  - [x] **✅ PR1: EDF+ Annotation Parser** - `fast_read_edf_annotations()` annotation-tail threaded `pread`; 冷 NFS 文件从 ~113s 降到 ~10-14s/文件
  - [x] **✅ PR1: Seizure Annotation Parser** - `parse_seizure_onsets_from_annotations()` 精确 label 匹配 + onset-END 配对 + 重叠区间合并 + orphan zero-duration 丢弃
  - [x] **✅ PR1: Timezone Infrastructure** - `epoch_to_local_hour()` via `zoneinfo`; `config/default.yaml` 新增 `dataset.timezone_default/timezone_overrides`
  - [x] **✅ PR1: Recording Timeline Helper** - `read_edf_record_info()` / `build_recording_timeline()`，不再假设固定 24h
  - [x] **✅ PR1 验证**: litengsheng 16 EDF 全部解析；全量 Yuquan 21 subjects / 260 EDF 审计完成，确认重复标注、零时长 marker 与非固定 24h 的真实约束
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
- [ ] **模块4: network_analysis.py** (三阶段递进, v2: 宽建图→精剪枝→定方向)
  - [ ] **Phase A: Channel-Scale MVP**
    - [ ] A.1 宽建图 (Simpson Index 归一化共激活)
    - [ ] A.2 替代数据显著性检验 (Surrogate)
    - [ ] A.3-A.4 XYZ 特征计算 + 多维剪枝 (Rate × Entropy × 谱聚类)
    - [ ] A.5 方向注入 (Wilcoxon + 一致性)
    - [ ] A.6-A.7 Stability 权重 + 复合权重
    - [ ] A.8 图论指标 (Net Outflow, Local Efficiency, Betweenness)
    - [ ] A.9-A.10 2D 网络拓扑图 + XY 散点诊断图 + 传播路径
  - [ ] **Phase B: Channel + Geometry** (阻塞于 MNI 坐标)
    - [ ] B.0 电极坐标获取
    - [ ] B.1-B.3 空间约束 + 容积传导剔除 + 传播速度验证
    - [ ] B.4 病理加权 (FR 分类 → Z 维度升级)
    - [ ] B.5-B.6 3D 脑图 + Ictal vs Interictal 对比
  - [ ] **Phase C: Source Space** (研究前沿)
- [x] Notebook: chengshuai_hfo_analysis.ipynb ✅
- [x] 2026-02-01: toy timelag + env cache 接口测试（cuda_env）
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
