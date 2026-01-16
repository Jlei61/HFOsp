# HFOsp 开发计划

**项目目标**: 复现并扩展 HFO (高频振荡) 分析流程，验证 Source-Sink 理论  
**数据集**: 玉泉24小时SEEG数据集  
**数据路径**: `/mnt/yuquan_data/yuquan_24h_edf`  
**更新日期**: 2026-01-14

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
│   ├── __init__.py              # [待创建]
│   ├── preprocessing.py         # ✅ 预处理Pipeline
│   ├── hfo_detector.py          # ✅ HFO检测器
│   ├── group_event_analysis.py  # [待开发] 群体事件分析
│   ├── network_analysis.py      # [待开发] 网络分析
│   ├── visualization.py         # ✅ 可视化工具（基础功能）
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

### 模块1: src/preprocessing.py ✅ 完成（已重构，拒绝"猜测式"分支）

| 功能 | 说明 | 状态 |
|------|------|------|
| 1.1 EDF读取 | 加载EDF, 清洗电极名称, encoding='latin1' | ✅ |
| 1.2 电极名称解析 | 正则提取 (prefix, number), 支持A'5格式 | ✅ |
| 1.3 重参考策略 | **显式** bipolar / car / none（auto=兼容别名→bipolar；不做任何"推断"） | ✅ |
| 1.4 通道选择 | **显式** include/exclude channels（用于复现GPU通道列表；不硬编码"去掉末端N个触点"） | ✅ |
| 1.5 重采样 | Ripple→1000Hz, FR→2000Hz | ✅ |
| 1.6 滤波 | Notch + 可选Bandpass, GPU加速支持 | ✅ |
| 1.7 通道质量检查 | z-score, 方差, 伪迹标记 | ✅ |

**关键技术决策**:
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

### 模块2: src/hfo_detector.py ✅ 完成

| 功能 | 说明 | 状态 |
|------|------|------|
| 2.1 Ictal段落检测 | `_detect_ictal_mask`：能量爆发 + 持续>3秒 | ✅ |
| 2.2 背景基线估计 | 剔除Ictal后MAD；bqk算法内部使用median | ✅ |
| 2.3 Hilbert包络 | `scipy.hilbert` + `cupy_hilbert`（GPU FFT）；宽带分20Hz子带求和 | ✅ |
| 2.4 双阈值检测 | `rel_thresh×local_median` ∧ `abs_thresh×global_median` | ✅ |
| 2.5 事件合并筛选 | `merge_timeRanges` (min_gap) + `min_last` 持续时间过滤 | ✅ |
| 2.6 Ripple/FR分离 | `band='ripple'/'fast_ripple'` | ✅ |
| 2.7 验证函数 | Notebook验证通过（10232 events，K电极高密度） | ✅ |

**关键技术决策**:
- **算法选择**: 默认 `algorithm='bqk'`，严格复用 `src/utils/bqk_utils.py` 的检测逻辑
- **Chunked处理**: 30s chunk + 1s overlap，避免内存爆炸；跨chunk事件正确合并
- **GPU加速**: Hilbert变换可选GPU FFT（`use_gpu=True`），滤波仍在CPU以保持数值一致性
- **双阈值策略**: 同时满足 `rel_thresh × local_median` **且** `abs_thresh × global_median`，能有效抑制噪声通道和全局伪迹

---

### 模块3: src/group_event_analysis.py (待开发)

| 功能 | 说明 |
|------|------|
| 3.1 群体事件窗口构建 | `build_windows_from_detections`（固定窗口，默认从`packedTimes`推断窗口长度；支持`min_channels`过滤；可与`packedTimes`做 overlap-based 对齐评估） |
| 3.2 事件内时间定位 | 质心法：对 Hilbert envelope 计算能量质心；支持 `align='first_centroid'` 得到相对传播时延；支持 tie-tolerant rank 评估（<2ms 视为并列 + pairwise concordance） |
| 3.3 核心通道筛选 | 在患者级 `_refineGpu.npz` 的 `events_count` 上用 `mean + 1*std` 复现 `hist_meanX.npz/pick_chns`；同时保留“全脑通道”分析路径 |
| 3.4 GPU 加速与缓存 | 对整段 crop 预先计算每通道 envelope（GPU filter+Hilbert+20Hz 子带求和）并保存到 `/mnt/yuquan_data/yuquan_24h_edf/<patient>/`，后续按 window 切片计算质心，避免重复 Hilbert |
| 3.5 验证函数 | Step1：检测→窗口 vs packedTimes；Step2-3：packedTimes→质心→lag/rank vs `*_lagPat.npz`（比较事件内相对 lag 与顺序一致性） |

**阶段性结论（2026-01-15，验证口径已对齐）**:
- 核心通道筛选：在患者级 `_refineGpu.npz` 的 `events_count` 上，用 `mean + 1*std` 可复现 `hist_meanX.npz` 的 `pick_chns`（与 `*_lagPat.npz` 的 `chnNames` 一致）。
- packedTimes 窗口长度：不同记录可能不同（例如 0.5s 与 0.3s 共存），不能硬编码窗口长度；应从 `packedTimes` 推断。
- Step1（检测→窗口 vs packedTimes）：使用 `reference='bipolar'` + “别名通道”(A1≈A1-A2) + “丢末端 contact”(按 GPU 通道集过滤 pair) 能显著提升 packedTimes 覆盖率；precision 下降是正常现象（packedTimes 是筛选后的事件池）。
- Step2-3（packedTimes→质心→相对lag/rank vs lagPat）：`eventsBool` 可达 100% 一致；相对 lag（对齐到最早通道）可到 ms 级误差；rank 对 ms 级抖动敏感，建议提供 tie-tolerant 评估（例如 <2ms 视为并列，并用 pairwise concordance 评估）。

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

### 模块5: src/visualization.py ✅ 基础功能完成

| 功能 | 说明 | 状态 |
|------|------|------|
| 5.1 多通道时序图 | `plot_seeg_segment`：单电极/全通道/核心电极视图 | ✅ |
| 5.2 Bipolar对比图 | `plot_preprocessing_comparison`：原始 vs Bipolar | ✅ |
| 5.3 HFO事件标注 | `_overlay_events` + `detections_to_events`：tick样式不遮波形 | ✅ |
| 5.4 群体事件图（Fig1/Fig2） | **Fig1**：`plot_group_events_band_raster(plot_style='trace', mode='bandpassed')` 画拼接窗口的**带通波形**（不是 block/heatmap）；**Fig2**：`plot_group_events_tf_centroids_per_channel` 对每通道做 STFT 并叠加 **TF(时间,频率) 质心** + colorbar；支持仅底部 x ticks、去 top/right 边框、紧凑布局 | ✅ |
| 5.5 滞后热图 | `plot_lag_heatmaps`：channels × events 的能量/秩/lag（用于 Step2-3 验证） | ✅ |
| 5.5 传播动图 | 500ms窗口内能量传播动画 | ⏳ 待模块3 |
| 5.6 网络拓扑图 | 节点=Strength, 边=权重, 颜色=SI | ⏳ 待模块4 |
| 5.7 状态对比图 | Interictal vs Ictal 网络 | ⏳ 待模块4 |

**关键技术决策**:
- **波形颜色**: `tableau_20_no_red` 避免与HFO红色标记冲突
- **事件标注**: 默认 `style='tick'` 细线，不遮挡波形
- **调试视图**: `plot_raw_filtered_envelope` 可视化 raw/filtered/envelope 验证检测器
- **Fig2 质心定义**: 在每个 event window 内、每通道对 STFT 功率 \(P(f,t)\) 计算 2D 质心：
  - \(t_c = \\frac{\\sum_{f,t} P(f,t)\\,t}{\\sum_{f,t} P(f,t)}\)
  - \(f_c = \\frac{\\sum_{f,t} P(f,t)\\,f}{\\sum_{f,t} P(f,t)}\)
  默认 `centroid_power='power2'`（强调能量峰），并可通过 `mask_by_detections` 控制是否仅对有 detection 的 (channel,event) 画点。

---

## 5. 关键技术陷阱与解决方案

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

## 6. 开发进度

- [x] 项目结构设计
- [x] 开发计划文档
- [x] **模块1: preprocessing.py** ✅ 2026-01-14（已重构：去除推断/硬编码）
  - [x] 电极名称解析 (ElectrodeParser)
  - [x] 重参考（显式）:
    - [x] Bipolar (BipolarReferencer) — 命名 `A1-A2`
    - [x] CAR per shaft (CommonAverageReferencer)
    - [x] None（保持原始EDF）
  - [x] 通道选择（显式）: include/exclude channels（用于匹配GPU通道列表）
  - [x] 重采样 + Notch滤波
  - [x] GPU加速支持 (CuPy可选)
  - [x] 通道质量检查
  - [x] chengshuai/FC10477Q: EDF vs GPU 通道差异来源确认（GPU=显式通道子集；不用于推断重参考）
- [x] **模块2: hfo_detector.py** ✅ 2026-01-14
  - [x] 算法选择：bqk (复用bqk_utils.py) / mad_hysteresis
  - [x] Ictal段落检测 (`_detect_ictal_mask`)
  - [x] Hilbert包络（宽带分20Hz子带 + 求和）
  - [x] 双阈值检测（rel_thresh × local_median ∧ abs_thresh × global_median）
  - [x] 事件合并筛选（min_gap + min_last）
  - [x] Ripple/FR分离
  - [x] Chunked处理（30s chunk + 1s overlap）
  - [x] GPU加速（cupy_hilbert）
  - [x] Notebook验证通过（10232 events，K电极高密度）
- [x] **模块5: visualization.py（基础功能）** ✅ 2026-01-14
  - [x] 多通道时序图
  - [x] HFO事件标注（tick样式）
  - [x] 调试视图（raw/filtered/envelope）
- [ ] 模块3: group_event_analysis.py ← 下一步
- [ ] 模块4: network_analysis.py
- [ ] 模块5: visualization.py（高级功能：滞后热图/传播动图/网络拓扑）
- [x] Notebook: chengshuai_hfo_analysis.ipynb ✅
- [ ] 验证与原结果一致性
- [ ] 完整Pipeline测试

---

## 7. 测试数据

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
