# 老代码(ReplayIED/yuquan_24h) vs 新代码(HFOsp/src) 算法逐环节对比

> 范围：仅 yuquan_24h 主线 subject，从原始 EDF 到 lagPat/网络
> 生成时间：2026-03-27
> 2026-03-27 状态更新：`bipolar_gap=1`、`side_thresh=1.5`、`max_last=200ms`、`rel/abs=2/2` 覆盖链、spectrogram 质心已在新代码中修复；本文保留“历史差异”并额外标注“当前状态/剩余差异”。

---

## 0. 全流程概览


| 阶段             | 老代码入口                                                        | 新代码入口                                                                                                |
| -------------- | ------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| 预处理            | `p16_cuda_24h_bipolar.py` 内联                                 | `src/preprocessing.py` → `SEEGPreprocessor`                                                          |
| HFO 检测         | 同上, `find_high_enveTimes_cu()`                               | `src/hfo_detector.py` → `HFODetector`                                                                |
| 同步性约束 / 通道筛选   | `p16_refine_chns_bySyn.py`                                   | `group_event_analysis.py` → `select_core_channels_by_event_count` + `filter_windows_by_min_channels` |
| 群体事件窗口         | `hfo_net.py` → `get_packedEventsTimes_overThresh`            | `group_event_analysis.py` → `build_windows_from_packed_times` / `build_windows_from_detections`      |
| 质心/时序 (lagPat) | `packGroupEvents*.py` → `return_massCenterPat` (spectrogram) | `group_event_analysis.py` → `compute_centroid_matrix_spectrogram` / `compute_centroid_matrix_from_envelope_cache` |
| lag → rank     | `np.argsort(np.argsort(x))` 内联                               | `lag_rank_from_centroids` → `compute_dense_rank`                                                     |
| 网络构建           | `diffnet_prepareTXT.py` → netRate外部工具                        | `src/network_analysis.py` → `build_hfo_network`                                                      |


---

## 1. 预处理

### 1.1 参考方式


| 参数     | 老代码                                   | 新代码                                                 | 差异影响                          |
| ------ | ------------------------------------- | --------------------------------------------------- | ----------------------------- |
| 参考类型   | 双极 (`data[i]-data[i+1]`)              | 双极 / CAR / none 可配置，默认 bipolar                      | 行为一致(bipolar模式)               |
| 双极间隔   | **固定相邻** (gap=1)                      | 可配 `bipolar_gap`，**当前默认已改为 1**                  | 已对齐；若手动改回 2 会重新引入物理信号差异 |
| 通道命名   | `A1` (取左触点名)                          | `A1-A2` 格式，配 `alias_bipolar_to_left=True` 时别名为 `A1` | 名义一致，需确保 alias 开启             |
| 噪声电极丢弃 | 硬编码 `subs_drop_info` → 先双极再 drop      | 配置 `core_channels` 或外部指定                            | 行为可对齐，但来源不同                   |
| 最外触点去除 | `return_valid_chan_index` 去掉每根轴最外面的触点 | 无显式逻辑                                               | **老代码多一步**                    |


### 1.2 重采样


| 参数    | 老代码                                                | 新代码                                                        |
| ----- | -------------------------------------------------- | ---------------------------------------------------------- |
| 目标采样率 | **800 Hz** (硬编码 `resample_to=800`)                 | 可配置 `resample_sfreq`，YAML 里可设 `auto`                       |
| 方法    | `cusignal.resample_poly(data, 2, round(2*fs/800))` | `scipy.signal.resample_poly` (CPU) 或 `mne.io.Raw.resample` |


### 1.3 滤波


| 参数  | 老代码                                           | 新代码                                       |
| --- | --------------------------------------------- | ----------------------------------------- |
| 陷波  | `cusignal.firwin` FIR, 50-250Hz 每50Hz, 带宽±2Hz | `scipy.signal.iirnotch` IIR, Q=30         |
| 带通  | `cusignal.firwin` 201阶 FIR                    | `scipy.signal.butter` 3阶 IIR + `filtfilt` |


**差异影响**：FIR vs IIR 的相位响应和边带特性不同。对于 HFO 检测这种包络-阈值方法，差异通常可忽略，但严格复现需要保持一致。

---

## 2. HFO 检测

### 2.1 多子带包络 (BQK 核心)

**两边完全一致的算法骨架**：

```
if bandwidth > 20Hz:
    把 [80,250] 拆成 [80,100],[100,120],...,[240,250]
    每个子带: bandpass → Hilbert → abs
    所有子带包络求和
else:
    直接单带 Hilbert 包络
```

### 2.2 阈值检测参数


| 参数             | 老代码 (yuquan_24h) | 新代码 (当前默认/实际生效值) | 差异 |
| -------------- | ---------------- | ------------------------ | ---- |
| `rel_thresh`   | **2.0**          | **2.0**                  | 已对齐 |
| `abs_thresh`   | **2.0**          | **2.0**                  | 已对齐 |
| `min_gap`      | 20 ms            | 20 ms                    | 一致 |
| `min_last`     | **50 ms**        | **50 ms**                | 已对齐 |
| `max_last`     | **200 ms**       | **200 ms**               | 已对齐 |
| `side_thresh`  | **1.5** (边带去噪)   | **1.5**                  | 已对齐 |
| `segment_time` | 200s             | `chunk_sec=50s` (overlap=2s) | 只影响内存与分块，不应改变理论判定结果 |
| frequency band | [80, 250]        | ripple=[80,250]          | 一致 |


> **状态更新**：`run_pipeline.py` 中对 `rel_thresh` 的偷偷覆盖已移除。`causal_tuning` 现在只用于网络层的因果参数，不再污染检测器。

### 2.3 side_thresh 边带去噪 (老代码独有)

```python
# p16_cuda_24h_bipolar.py L255-267
for tmp_tr in tmp_highTime:
    side_pre = tmp_enve[event前等长区间]
    side_after = tmp_enve[event后等长区间]
    side_mean = mean(concat(side_pre, side_after))
    pick_mean = mean(event区间内的包络)
    if pick_mean >= side_thresh * side_mean:   # side_thresh=1.5
        保留该事件
```

这一步现在已经补到 `src/utils/bqk_utils.py::find_high_enveTimes()` 和 `src/hfo_detector.py::HFODetectionConfig`。它的作用是排除那些“前后背景也很高”的伪事件。对于棘波叠加高频、宽带噪声通道，这个步骤有实际意义。

### 2.4 max_last 上限 (老代码独有)

老代码要求事件时长 `50ms < duration < 200ms`，排除持续过长的活动。这个约束现在也已经补到新代码，参数名为 `max_last_ms=200.0`。

---

## 3. 同步性约束 / 通道筛选

### 3.1 老代码 (`p16_refine_chns_bySyn.py`)

**两步处理**：

1. **初步通道筛选**：所有 `*_gpu.npz` 的事件计数求和 → `mean + pickChn_thresh × std` 阈值 → 得到 "picked channels"
2. **同步性约束 (refine)**：只用 picked channels 的检测结果 → 构建群体活动窗口 → 在窗口内重新计数每个通道（含非 picked 通道）→ 保存为 `_refineGpu.npz`

参数：`pickChn_thresh` = 1.0 (默认)，部分患者手动调整(如 xuxinyi=0.7, gaolan=1.9)

### 3.2 新代码

1. **通道筛选**：`select_core_channels_by_event_count` → 支持 `mean_std`/`log_mean_std`/`log_median_mad`/`top_n` 多种方法
2. **窗口过滤**：`filter_windows_by_min_channels(min_channels=6)` → 保留至少 6 个通道有事件的窗口

**关键差异**：

- 老代码的 refine 是一个**闭环**：先挑通道 → 建窗口 → 窗口内重新统计 → 第二次挑通道。新代码是**开环**：挑通道 → 挑窗口，没有反馈循环。
- 老代码 `pickChn_thresh` 对每个患者有手动微调。新代码用统一配置。
- 新代码 `min_channels=6` 是硬过滤，老代码 `chnsThr=0.5` 是 50% picked channels 有事件。

---

## 4. 群体事件窗口构建

### 4.1 老代码 (`hfo_net.py::get_packedEventsTimes_overThresh`)

```
1. 所有 picked channels 的事件时间，前后各扩展 ext=30ms
2. 在 500Hz 时间轴上累加为 index_array（每时刻有几个通道有事件）
3. index_array >= chnsThr × n_picked_chns 的区间提取出来
4. 取每个区间的中心点 ± packWinLen/2 作为最终窗口
5. 去除重叠窗口 (pick_noOverlap_timeRanges)
6. 去除 > 2s 的窗口
```

参数因患者而异：`packWinLen` = 200ms~500ms, `chnsThr` = 0.5

### 4.2 新代码 (`build_windows_from_detections`)

```
1. 所有通道的事件按 start 时间排序
2. 取最早未分配的事件 start → 窗口 = [start, start + window_sec)
3. 所有 start 落在窗口内的事件标记为已分配
4. 重复直到所有事件分配完毕
```

或者直接读取老代码生成的 `packedTimes.npy`：`build_windows_from_packed_times`。

**关键差异**：

- 老代码：**累积通道计数 → 阈值 → 中心对齐**，窗口是"大多数通道同步活动"的时刻
- 新代码：**最早事件驱动 → 贪心分配**，窗口是从第一个事件开始的固定长度窗
- 老代码的窗口含义更接近"群体同步事件"，新代码的窗口更像"检测事件的时间分箱"
- **如果新代码使用 `packed_times_path` 走 legacy 路径，则窗口定义完全一致**

---

## 5. 质心与 lagPat 生成

### 5.1 老代码 (`return_massCenterPat`)

```python
1. 对 picked channels 的高频滤波信号计算 spectrogram:
   - window='hamming', nperseg=0.05*fs, noverlap=0.04*fs, mode='magnitude'
   - 频率范围 50-300 Hz
   - Gaussian smooth σ=1.5
2. 每个通道每个窗口: **spec^3** → 归一化权重 → 加权时间质心
3. `lagPatRaw` = 质心精确时间矩阵 `(n_chns, n_events)`
4. `lagPatRank` = `argsort(argsort(lagPatRaw per event))`
5. 变体脚本 `*_withFreqCent.npz` 还会额外保存 `lagPatFreq`（频率质心）
```

### 5.2 新代码（当前默认已切到 spectrogram）

```python
1. `centroid_source=spectrogram` 时：
   - 对窗口内带通信号算 spectrogram
   - `window='hamming'`, `nperseg=0.05*fs`, `noverlap=0.8*nperseg`
   - 频率范围 `50-300Hz`
   - Gaussian `sigma=1.5`
   - `spec^3` → 加权时间质心
2. `centroid_source=env` 时仍可走 Hilbert 包络质心
3. 输出 `centroids (n_chns, n_events)` 与 `events_bool (n_chns, n_events)`
4. `lag_rank_from_centroids`：
   - `lag_raw` 保留精确时间（相对事件窗起点）
   - `lag_rank` 为逐事件排序，非参与通道 = `-1`
```

**关键差异**：


| 方面         | 老代码                             | 新代码（当前 spectrogram 模式）                       |
| ---------- | ------------------------------- | -------------------------------------------- |
| 质心计算域      | **时频图** (spectrogram, 50-300Hz) | **时频图** (spectrogram, 50-300Hz)              |
| 功率加权       | spec^**3**                      | spec^**3**                                   |
| 平滑         | Gaussian σ=1.5 on spectrogram   | Gaussian σ=1.5 on spectrogram                |
| 通道参与判断     | 基于 eventsBool (群体打包时计算)         | 基于 `_overlaps_any_event(detections, window)` |
| rank 非参与标记 | 不标记（全通道参与）                      | **-1** 标记                                    |
| align      | 无 align，直接 argsort              | `lag_raw` 保留精确时间；`lag_rank` 逐事件排序            |


**状态更新**：spectrogram 质心主路径已经对齐。仍然存在的差异主要不是质心公式，而是上游窗口定义、FIR/IIR 滤波、最外触点去除、闭环 refine 通道筛选。

---

## 6. 网络构建

### 6.1 老代码

- `diffnet_prepareTXT.py` → 提取 cascade 文本 → 外部 `netRate` 工具建 diffusion network
- `h2` 非线性相关 → 发作网络
- Spearman 相关比较 diffnet 与 ictal h2 网络

### 6.2 新代码 (`build_hfo_network`)

- Simpson/Dice 共现 → 代理检验 → 谱聚类 → 方向/稳定性 → 多证据融合
- 完全自包含，不依赖外部工具
- 参数 Tier-2 锁定 (`strict_baseline=true`)

**差异**：两套完全不同的网络构建范式。老代码依赖外部工具和 cascade 格式，新代码是统计驱动的纯内部实现。

---

## 6.5 老代码还做了哪些有意义的下游分析

### 6.5.1 精确 lag 时间统计（不是只有 rank）

老代码明确保留并使用 `lagPatRaw`：

- `plotting_fig4_pairedDelay.py`：基于 `lagPatRaw` 和 `eventsBool` 计算每个群体事件的传播总时长 `max(lag)-min(lag)`、相邻通道排序后的 `np.diff(sorted(lag))`、平均延迟，以及与参与通道数的关系
- `plotting_fig8_seizureRelated_synchron.py`：把 `lagPatRaw` 归一化到相位后计算 Kuramoto-like order parameter，比较发作相关窗口 vs 非发作窗口的同步性
- `plotting_fig4_durDelayStats.py`：使用 `lagPatRaw` + `lagPatFreq` 联合统计传播时长、频率中心、参与通道数、枚举延迟分布

结论：**老代码不只是存了精确时间，它确实在群体事件统计层面系统使用了这些精确 lag。**

### 6.5.2 频率质心统计

老代码有 `withFreqCent` 变体：

- `p16_packGroupEvents_per2h_showSpecs_bipolar_refine_bool_withFreqCenter.py`
- 保存 `lagPatFreq`
- `plotting_fig4_durDelayStats.py`、`plotting_fig4_seqsTemporalStats.py` 用于时间-频率联合分布和跨受试者比较

这说明老代码在某些分析里不是“只有传播顺序”，而是同时分析“传播时长 + 频率中心”。

### 6.5.3 模板稳定性 / 24h 动态

- `p16_quantify_MI.py`、`p16_quantify_MI_20210615.py`：事件间 propagation pattern 一致性（Matching Index）
- `p16_MI_24hChange.py`、`p16_MI_24hChange_entropy.py`：按 2h/全天跟踪 pattern 稳定性、模板 MI、rank 直方图熵
- `plotting_fig5_lagPat_variance.py`：每通道 rank 分布、均值模板、变异性

### 6.5.4 区域级传播统计

- `p16_region_lagPat_stats.py`：将电极映射到解剖区，统计 region-to-region 的 leader→follower 方向计数
- `plotting_fig4_regionHist.py`、`plotting_fig4_regionHist_median.py`：做区域层面的先导/滞后汇总

### 6.5.5 网络/发作关联分析

- `p16_lagPat_diffusionNet_comparison.py`：把 lag 模板顺序和 diffusion network 做对照
- `plotting_fig8_seizureRelated_synchron.py`：比较发作窗口与非发作窗口的 MI / 同步性
- `plotting_fig8_resortConnMatrix_withKuramoto.py` 等：把传播模式、连接矩阵、同步性一起看

---

## 7. 综合判断

### 7.1 参数合理性


| 环节                           | 判断                  | 理由                                        |
| ---------------------------- | ------------------- | ----------------------------------------- |
| rel/abs_thresh 2/2 vs 3/3    | **2/2 更适合间期，且现已对齐** | 间期异常活动幅值相对较弱，3/3 过严会漏检。文献中 2×median 是常见选择 |
| side_thresh                  | **有价值，不应丢弃**        | 对宽带噪声电极和棘波叠加高频的区分有实际作用                    |
| max_last=200ms               | **合理**              | HFO 的典型持续时间 50-500ms，200ms 上限排除了持续性噪声     |
| packWinLen 因患者调整             | **必要但不可扩展**         | 群体活动的时间窗宽依赖于具体网络的传播速度，统一值是妥协              |
| Hilbert 质心 vs Spectrogram 质心 | **Spectrogram 更鲁棒，且现已加入主路径** | 时频域质心对宽带噪声不敏感，且频率维度信息可辅助区分 HFO 和 spike |
| bipolar_gap=2 vs gap=1       | **必须对齐，且现已对齐**  | gap=2 意味着跳触点差分，物理意义完全不同 |


### 7.2 哪种方式更适合 SEEG 癫痫间期事件

**结论：当前新代码已经把最关键的“科学行为”对齐到老代码，但还没做到全链路等价。**

老代码的优势：

- `2/2` 阈值 + `side_thresh` + `max_last` 三重约束，对间期 HFO 的检测更精细
- 同步性约束的闭环 refine 更贴近实际临床需求
- Spectrogram 质心保留了频率维度信息
- 患者个性化参数调优（虽然不可扩展，但对论文结果是对的）

新代码的优势：

- 模块化、可配置、可测试
- `events_bool` 和 rank 的 -1 标记更规范
- 网络分析完全自包含

当前仍未完全对齐的点：

- 最外触点去除逻辑还没有补到新代码
- 老代码的闭环 refine 通道筛选还没有移植
- 若不用 `packedTimes.npy` 而走新窗口生成逻辑，群体事件定义仍不同
- 滤波实现仍是老代码 FIR vs 新代码 IIR
- `lagPatFreq` 仍未进入新代码标准输出

### 7.3 建议方案

**不大动历史代码的前提下，构建一个 bridge 脚本**：

1. **检测层**：直接复用老代码已经生成好的 `*_gpu.npz`（检测结果已存在）
2. **refine 层**：直接复用老代码已经生成好的 `_refineGpu.npz`（通道筛选结果已存在）
3. **群体事件层**：复用老代码已经生成的 `*_packedTimes.npy` 和 `*_lagPat.npz`
4. **网络层**：可以在新代码里用 `build_hfo_network` 重算，或保持老的 

如果需要**重新跑整条线**（比如参数调整）：

- 用新代码框架但把参数调成老代码的值
- `side_thresh` 和 `max_last` 已补到 `BQKDetector`
- `bipolar_gap` 已设为 1
- `rel_thresh`/`abs_thresh` 已恢复到 2.0/2.0，且覆盖链已清理
- 质心计算已切到 spectrogram 模式
- 剩下优先补：最外触点去除、闭环 refine、`lagPatFreq`

---

## 附录：参数速查表（yuquan_24h 论文复现用）

```yaml
# 对齐老代码的参数（当前新代码已基本采用）
preprocessing:
  reference: bipolar
  bipolar_gap: 1              # 老代码 gap=1
  resample_sfreq: 800         # 老代码 resample_to=800
  # 需要保留最外触点去除逻辑

hfo_detection:
  band: ripple                # [80, 250]
  rel_thresh: 2.0             # 老代码默认
  abs_thresh: 2.0             # 老代码默认
  min_gap_ms: 20
  min_last_ms: 50
  max_last_ms: 200
  side_thresh: 1.5

channel_selection:
  method: mean_std
  k: 1.0                      # 对应 pickChn_thresh=1

group_events:
  chnsThr: 0.5
  ext: 30ms
  packWinLen: 患者个性化       # 200-500ms

group_analysis:
  core:
    centroid_source: spectrogram
    centroid_power: 3.0
```

