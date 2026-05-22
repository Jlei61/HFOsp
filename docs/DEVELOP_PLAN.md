# HFOsp 开发计划

**项目目标**: 复现并扩展 HFO (高频振荡) 分析流程，验证 Source-Sink 理论
**数据集**: 玉泉 24h SEEG + Epilepsiae *.data/*.head
**数据路径**: `/mnt/yuquan_data/yuquan_24h_edf` + `/mnt/epilepsia_data`
**最后大瘦身**: 2026-05-22（已归档大量已完成内容，见 §9）

> **本 doc 的定位**：保留**架构契约**（数据流、`*_groupAnalysis.npz` schema、visualization 铁律、关键技术陷阱）。**当前在做的科学工作**走 topic 主文档（`docs/topic{0,1,2,3,5}_*.md`），**已完成的 PR 复盘 / refactor postmortem** 走归档（见 §9）。

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

| 指标 | Yuquan | Epilepsiae |
|---|---|---|
| Subject 总数 | 21 | 27 (其中 20 有 `all_data_lns` 间期产物) |
| 数据合同 | EDF / EDF+ | `*.data + *.head + SQL` |
| 时区规则 | `Asia/Shanghai` | `UKLFR → Europe/Berlin`, `08:00–20:00 = day` |
| 详细 inventory | `results/yuquan_*_inventory.csv` | `results/epilepsiae_*_inventory.csv` |

更多 dataset 契约见 `docs/yuquan_24h_dataset_structure.md`、`docs/epilepsiae_dataset_structure.md`、AGENTS.md §Epilepsiae Contract。

---

## 2. 文件结构

参考 `AGENTS.md` §Current Code Map（按 module 列出当前 active 入口），不在此处复制 src/scripts/tests 树。

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

每个模块的当前状态摘要见下表。完整的 PR 复盘 / refactor postmortem / 数值结果走 archive（见 §9 归档索引）。

| 模块 | 状态 | 当前实现入口 | 归档 |
|---|---|---|---|
| **1. preprocessing.py** | ✅ Phase 1 重构完成 + PR1 EDF+ annotation 扩展 | `SEEGPreprocessor`, `fast_read_edf_annotations()`, `parse_seizure_onsets_from_annotations()`, `detect_seizure_by_spatial_extent[_epilepsiae]()` | [Phase 1 refactor + GPU backend 设计决策](archive/develop_plan_history/module_refactor_summaries_2026-01.md) |
| **2. hfo_detector.py** | ✅ Phase 2 重构完成 (BQK only) → ⏭️ 已被 **HFO Detector v2** (2026-05-05 canonical) 接管 | `src/hfo_detector.py::BQKDetector` (legacy)；v2 canonical: `results/hfo_detector_v2/` | [Phase 2 refactor postmortem](archive/develop_plan_history/module_refactor_summaries_2026-01.md); v2 specs: `docs/archive/hfo_detector_v2/` |
| **3. group_event_analysis.py** | ✅ 核心逻辑完成 (centroid + lag + TF 质心 + 统一存储 + 一键 API) | `compute_and_save_group_analysis()`, `compute_centroid_matrix_from_envelope_cache()`, `lag_rank_from_centroids()`, `compute_tf_centroids()` | (无独立 archive；当前实现是真值) |
| **4. network_analysis.py** | ⬜ **未启动**。Phase A/B/C 详细计划已归档；项目科学焦点已转向 Topic 1/2/3。 | (无) | [Module 4 v2 完整 1010 行计划](archive/develop_plan_history/module4_network_analysis_v2_2026-03-05.md) |
| **5. visualization.py** | ✅ 重构完成 (STFT/质心计算已剥离到 group_event_analysis) | `plot_paper_fig1_bandpassed_traces()`, `plot_group_event_tf_propagation_from_cache()`, `plot_lag_heatmaps_from_group_analysis()`, `plot_lag_statistics()`, `plot_tf_centroid_statistics()` | [Visualization 重构 + 资源依赖速查](archive/develop_plan_history/module_refactor_summaries_2026-01.md) |

### Seizure Detector 演化

完整 4 段复盘（PR1 真验收 → PR1.5 Epilepsiae 数据契约 → PR2 channel-mean 失败 → PR2.5 空间招募 v3）见 [seizure_detector_postmortem_2026-04-02.md](archive/develop_plan_history/seizure_detector_postmortem_2026-04-02.md)。**当前 canonical detector**: HFO Detector v2 (`docs/archive/hfo_detector_v2/v2_specification.md`)；空间扩张 seizure detector: `src/preprocessing.py::detect_seizure_by_spatial_extent[_epilepsiae]()`。

### 间期同步性 PR4–PR6

完整 6 个发现 + 指标层级判定 + 对后续工作约束见 [develop_plan_pr4_pr6_findings_2026-04-03.md](archive/topic1/synchrony/develop_plan_pr4_pr6_findings_2026-04-03.md)（**注意**：归档定格于 n=16；当前 cohort 已扩到 n=29，引用数字以 `docs/topic1_within_event_dynamics.md` 为准）。

---

## 5. 快速开发参考

### 常见任务速查

| 任务 | 代码 |
|------|------|
| **预处理 + HFO检测** | `preprocessor.run(edf) -> detector.detect(result.data, result.sfreq)` |
| **完整分析流程** | `compute_and_save_group_analysis(edf_path, ...)` |
| **可视化论文图** | `plot_paper_fig1_bandpassed_traces(env_cache_npz, ...)` |
| **读取中间结果** | `load_group_analysis_results('*_groupAnalysis.npz')` |
| **Envelope 缓存** | `precompute_envelope_cache(data, sfreq, ch_names, ...)` |

具体 API 签名以 `src/preprocessing.py`、`src/hfo_detector.py`、`src/group_event_analysis.py`、`src/visualization.py` 当前代码为准。

---

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

## 7. 当前进行中的工作

科学侧（按 topic doc 为准）：
- **Topic 0**: 方法学审计——lagPatRank phantom phase 0 已完成（2026-05-21）；其他 audit 持续；见 `docs/topic0_methodology_audits.md`
- **Topic 1**: 内事件动态学——within-event propagation + synchrony；PR-4B/4C/5/6/7 持续；见 `docs/topic1_within_event_dynamics.md`
- **Topic 2**: 事件间时序——~2Hz 周期性已归因；PR-3 stereotypy robustness pending；见 `docs/topic2_between_event_dynamics.md`
- **Topic 3**: 空间 SOZ 归属——PR-1 完成 (Yuquan n=9)；PR-2 Epilepsiae 三层 i/l/e gradient pending；见 `docs/topic3_spatial_soz_modulation.md`
- **Topic 4**: SEF-ITP attractor 框架——Phase 1 runner 刚 land；见 `docs/topic4_sef_itp_framework.md`
- **Topic 5**: Seizure subtyping (z-ER)；exploratory；见 `docs/topic5_seizure_subtyping.md`

工程侧 forward TODO：
- [ ] Module 4 network_analysis（**deferred indefinitely**——若重启动，见 archive plan）
- [ ] 完整 Pipeline integration test
- [ ] 验证与原结果一致性（按 topic doc 推进，不再单独维护清单）

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

---

## 9. 历史档案索引

### `docs/archive/develop_plan_history/` — DEVELOP_PLAN 内容归档

- [`module4_network_analysis_v2_2026-03-05.md`](archive/develop_plan_history/module4_network_analysis_v2_2026-03-05.md) — Module 4 完整 v2 设计（宽建图 → 精剪枝 → 定方向，Simpson Index，XYZ 剪枝，Phase A/B/C，源空间远景）；1010 行，**从未实施**
- [`seizure_detector_postmortem_2026-04-02.md`](archive/develop_plan_history/seizure_detector_postmortem_2026-04-02.md) — PR1 真验收 + PR1.5 Epilepsiae 契约 + PR2 channel-mean 失败 + PR2.5 v3 三大踩坑（合并 4 段）
- [`module_refactor_summaries_2026-01.md`](archive/develop_plan_history/module_refactor_summaries_2026-01.md) — Phase 1 preprocessing 重构 + Phase 2 hfo_detector 重构 + Module 5 visualization 重构（合并 3 段 postmortem）

### `docs/archive/topic1/synchrony/` — 同步性归档

- [`develop_plan_pr4_pr6_findings_2026-04-03.md`](archive/topic1/synchrony/develop_plan_pr4_pr6_findings_2026-04-03.md) — PR4–PR6 完整 event-level 链路 + 6 大科学发现；**n=16 时点定格**，当前 cohort 已 n=29

### 其他 canonical 归档

- HFO Detector v2 specs: `docs/archive/hfo_detector_v2/`
- Topic 0 audits: `docs/archive/topic0/`
- Topic 1 propagation / synchrony / PR-6 / PR-7 / PR-8: `docs/archive/topic1/`
- Topic 2 periodicity: `docs/archive/topic2/`
- Topic 3 spatial: `docs/archive/topic3/`
