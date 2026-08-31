# Group-Event State v0.1 — source audit 报告（2026-09-01）

机器可读产物：`results/epi_prssm/group_event_state/v0_1/`
（`source_audit.json` / `subject_inventory.csv` / `block_inventory.csv` /
`event_pointer_audit.json` / `contiguous_session_inventory.csv` / `band_availability.csv`）

## 1. 覆盖面

| | 数量 |
|---|---|
| 有 group-event artifact 的患者 | **41**（Epilepsiae 20 + Yuquan 21） |
| block | **3,896** |
| 群体事件（完整 artifact，非旧筛子集） | **1,774,188** |
| 其中 interictal（剔除真实 ictal 交集） | **1,746,959**（ictal 27,229 = 1.5%） |
| **每个事件可回到原生样本区间** | **1.0000（41/41 患者，3,896/3,896 blocks）** |
| contiguous coverage session | **344**（≥1k 事件 183 / ≥5k 77 / ≥10k 48） |
| 患者拥有 ≥10k 事件的单一连续段 | 18 |
| 患者拥有 ≥5k 事件的单一连续段 | **27**（v0.1 合格集） |

`epilepsiae_958` 单人 **165,577** 事件，高于旧线 v0.4 的 123,419——因为本线扫描的是
完整 artifact，而不是旧"确定无疑间期 block"子集。

## 2. 修复的两处覆盖缺口

### 2.1 Yuquan block inventory 只有 9 人

`results/dataset_inventory/yuquan_block_inventory.csv` 原本只覆盖 9 位 Yuquan 患者
（115 blocks），`chengshuai` 等不在其中。这不是"数据缺失"，是 inventory 生成时的
subject 列表默认值。已用同一 builder（`scripts/build_yuquan_block_inventory.py`，
新增 `--all-subjects` / `--pr1-dir` 两个开关 + 原子写）重建：

- **21 位患者、260 blocks**；对原 9 人子集**逐位复现**旧文件（`diff` 为空），确认
  builder 行为未变。
- seizure inventory 同步重建：**11 位患者共 54 次**检出发作；另外 10 位患者的
  `pr1_seizure_*.json` **存在但检出 0 次**——记为"未检出"，**不得**读作"无发作"。

### 2.2 83 个 Yuquan block 缺 `_gpu.npz`

7 位患者（`pengzihang` `songzishuo` `zhangbichen` `zhangkexuan` `zhaochenxi`
`zhaojinrui` `zhourongxuan`，合计 **181,160** 事件）的检测产物 `_gpu.npz` 不在盘上，
首轮审计因此把它们整体判为不可重建（waveform pointer fraction = 0）。

`_gpu.npz` 在这里只承担一件事：说明 detector 用的是哪种 montage。Yuquan 的规则是
**adjacent bipolar，lagPat 行名 = anode**。该规则可以从记录自身的通道表重建，且有三条
独立证据支持：

1. cohort 内**已记录**的 972 个 bipolar pair，**零例外**全是同杆 `(n, n+1)`；
2. `pengzihang` 唯一保留 `_gpu.npz` 的 block 上，重建 pair 与记录 pair **12/12 逐个一致**；
3. 7 位患者的**全部** lagPat 标签及其 `n+1` 邻居在各自 EDF 通道表中都存在（0 unresolvable）。

因此这些 block 的 montage 由规则重建，并**永久带标记**
`montage_provenance = "derived_from_label_rule_no_gpu"`（cohort：3,813 读取 / 83 重建）。
重建后 waveform pointer 覆盖率 **0.8996 → 1.0000**。

## 3. 数据语义的实证确认

| 检查 | 结果 |
|---|---|
| `start_t`（lagPat）vs `block_start_epoch`（SQL / EDF header） | 3,896/3,896 blocks 逐位相等（\|Δ\| ≤ 1e-3 s） |
| 事件内 `lagPatRaw` 相对延迟跨度 | 中位 **28.9 ms**（p10 11.6 / p90 82.3） |
| 该跨度超过 packed core 时长的比例 | **0.000%** |
| 非参与触点带有限 `lagPatRaw` 的比例 | 中位 **100%**（phantom 污染，必须过 `eventsBool`） |
| 每事件参与触点数 | 中位 5.0 |
| 每事件可分辨招募步（10 ms 容差） | 中位 **2.0**（p90 3.0） |

最后两行合起来说明：legacy 的整数 rank 在中位事件上把 5 个触点强行排成 5 个名次，
而 producer 的时间分辨率只支撑得起 **2 个**可分辨的招募步。

## 4. 端到端重建验证（承重）

从原生 `.data` / `.edf` 重建 detector 视图后，取 80–150 Hz 包络，比较 event core 峰值
与 core 外基线中位数：

| | 参与触点 | 非参与触点 | corr(重建包络峰时, producer 质心) |
|---|---|---|---|
| `epilepsiae_958`（200 事件） | **15.79×** | 4.57× | median r = **+0.929** |
| `yuquan_chengshuai`（200 事件） | **9.94×** | 3.60× | median r = **+0.566** |

指针、montage、事件对齐、时间轴——四者必须同时正确才会出现这个结果。

## 5. 频带可用性

| band | Hz | 支持患者 | 支持事件 |
|---|---|---|---|
| ied_low | 1–30 | 41/41 | 1,774,188 |
| gamma | 30–70 | 41/41 | 1,774,188 |
| low_ripple | 80–110 | 41/41 | 1,774,188 |
| ripple | 80–150 | 41/41 | 1,774,188 |
| **fast_ripple** | 150–250 | **39/41** | 1,684,687 |

`epilepsiae_253` / `epilepsiae_139` 为 512 Hz，fast ripple **标为 missing**，不填 0。

## 6. 一个静默丢事件的 bug（已修）

首版按 `round(t0·fs)` 与 `round(t1·fs)` 分别取整推 context 宽度，再要求宽度等于块内中位数。
`epilepsiae_1073/107300102_0014` 的 core 恒为 0.11 s，但 `0.11 × 1024 = 112.64`，
于是同一个恒定时长的事件会因 `t0` 的小数相位在 112/113 samples 之间跳变，
**该块 377 个事件里 141 个（37%）被判为不可用**。

修法：每块只取一个名义 core 宽度，context 从 core 起点推。同时把"波形不可用"
与"事件不存在"分开——滤波窗越出块边界的事件（实测每块约 1 个）保留在事件时钟上、
只标 `has_waveform=False`。丢掉它们会**静默缩短**状态模型要积分的 inter-event interval。
修复后该块 **376/377**。

## 7. v0.1 合格集（选择规则先于任何模型结果）

条件：pointer == 1.0 且最长连续段 ≥ 5,000 事件 → **27 位患者、1,663,394 事件**。
已缓存并一致化的 **23 位患者、1,506,867 interictal 事件**（Epilepsiae 9 + Yuquan 14；
其余 Epilepsiae 患者的缓存仍在生成）。逐患者分母见 `dataset_summary.json`。
