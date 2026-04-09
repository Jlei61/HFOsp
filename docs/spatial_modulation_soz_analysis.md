# 群体事件时序调制的空间归因分析

> 状态：**规划中** — PR-1 待实施
> 创建日期：2026-04-09
> 核心问题：**IEI 序列相关所反映的慢调制，发生在哪里？SOZ 与非 SOZ 通道是否存在可分离的调制特征？**
> 上游依赖：`docs/event_periodicity_analysis.md`（Phase 1–5 + PR-1/2/2.5）、`docs/interictal_population_event_methodological_review.md`

---

## 0. Topic 边界

本文档**只讨论 Where**——慢调制的空间归因。以下内容**不在**本文档范围：

| 已回答（其他文档） | 本文档不重复 |
|---|---|
| ~2 Hz 是不是内禀振荡？（否） | `event_periodicity_analysis.md` |
| IEI 分布是什么？（lognormal） | 同上 |
| 慢调制存不存在？（存在，30/30 正相关） | 同上 §PR-2 |
| 慢调制在什么时间尺度？（PR-2.5） | `event-periodicity-pr-plan.mdc` |
| IEI 和 n_participating 是否被同一状态变量调制？（PR-2.5） | 同上 |
| 传播 stereotype 是否真实？（PR-3） | 同上 |

本文档聚焦：

1. **在 per-channel 层面**，SOZ 通道与非 SOZ 通道的 IEI 时序结构是否有可分离的差异？
2. 差异是否超出事件率（event rate）本身可以解释的范围？
3. 如果差异显著且跨数据集可复现，是否有潜力成为 SOZ 辅助定位标准？

---

## 1. 动机：为什么 lagPat 框架回答不了 Where

### 1.1 当前 SOZ 对比的结构性缺陷

所有已完成的 SOZ 分层分析（PR-1 exp6B、PR-2 exp7D）都在 lagPat 框架内操作。这个框架有**三层嵌套的选择偏差**：

**第一层：通道宇宙被 refine 预截断**

`select_core_channels_by_event_count` 用 `mean + k*std`（默认 k=1）筛选 HFO 高发通道。`*_gpu.npz` 有 ~120 通道的完整检测，但 lagPat 的 `chnNames` 通常只保留 5–15 个——全是 refine 闭环后的高事件数通道。

SOZ 的临床定义恰好对应"HFO 最活跃的区域"，所以 refine 后的通道集天然富集 SOZ。这不是 bug，是 refine 的设计意图（聚焦高活动区域），但它让 SOZ vs non-SOZ 的空间对比在起点上就不公平。

**第二层：群体事件的 SOZ 标注是二值的**

当前 `_load_group_events_with_soz_labels` 把群体事件按"至少一个 SOZ 通道参与"分为 SOZ/non-SOZ。在一个 SOZ 富集的通道宇宙里，几乎所有群体事件都至少有一个 SOZ 通道参与。数据证实：**22/30 subject 的 non-SOZ 群体事件接近零**。

```
# 来自 PR-1 exp6B
排除原因             数量
SOZ 定义缺失           5
non-SOZ 事件 < 50     17  ← 结构性不足，不是运气
有效配对               8
```

**第三层：群体事件本身是 SOZ 现象**

群体事件要求多通道 HFO 时间重叠。当 lagPat 通道几乎都是 SOZ 或 SOZ 邻近，"群体性"本身就暗含了 SOZ 网络的协同激活。在这个框架里分离"SOZ 的调制"和"全局调制"，等于在一个被选择偏差污染的样本里做空间归因。

### 1.2 正确的做法

不是放弃 refine（refine 的存在有道理——SEEG 不采样全脑，全量 120 通道里包含大量白质、伪迹通道），而是：

1. **放松 refine 阈值**，纳入更多有 HFO 活动但未通过严格筛选的通道
2. **在 per-channel 层面做分析**，而非在群体事件层面做二值 SOZ/non-SOZ 标注
3. **用连续指标**（per-channel 事件率、serial correlation、dead-time）替代二值标签

这样：
- 统计单元从"subject 内两类群体事件的不平衡对比"变成"subject 内多个通道的空间对比"
- 非 SOZ 通道虽然事件少，但只要超过双阈值（min_count + min_rate）且通过质控（CV < 5.0、IEI ≥ 10ms），就能贡献一个有效的 serial correlation 估计
- 可以做 SOZ 距离/标签的剂量-效应分析（特别是 Epilepsiae 的 i/l/e 三值标注）

### 1.3 与已有 PR 的关系

| 已有工作 | 本文档的关系 |
|---|---|
| PR-1 exp6B（SOZ vs non-SOZ dead-time） | 本文档的分析将替代/扩展 exp6B 的 n=8 对结果 |
| PR-2 exp7D（SOZ vs non-SOZ serial corr） | 同上，n=9 对 → 期望显著扩大到 20+ 对 |
| PR-2.5 exp7B（多尺度去趋势） | 正交：PR-2.5 回答 when，本文档回答 where |
| PR-2.5 exp7C（n_participating 自相关） | 正交：exp7C 在群体事件层面，本文档在单通道层面 |
| PR-3（传播 stereotype 稳健性） | 弱关联：如果 SOZ 的 serial corr 更强，stereotype 的 SOZ 驱动性获得独立支撑 |

---

## 2. 数据基础设施

### 2.1 数据源

| 数据源 | 文件 | 通道数 | 覆盖 |
|--------|------|--------|------|
| Yuquan 原始检测 | `*_gpu.npz` → `whole_dets`, `chns_names`, `events_count`, `start_time` | ~120 双极 | 12 blocks × 2h = 24h |
| Epilepsiae 原始检测 | `*_gpu.npz`（可能损坏） | 变化 | 1h blocks |
| lagPat 通道集 | `*_lagPat.npz` → `chnNames` | 5–15 | refine 子集 |
| SOZ 标注（Yuquan） | `results/yuquan_soz_core_channels.json` | 二值 | 20 subjects 有非空 SOZ |
| SOZ 标注（Epilepsiae） | `results/epilepsiae_soz_core_channels.json` | 二值 | 来自 SQL `focus_rel == 'i'` |
| 区域标注（Epilepsiae） | `results/epilepsiae_electrode_focus_rel.json` | 三值 i/l/e | per-channel |

### 2.2 `*_gpu.npz` 内部结构（Yuquan 文档化）

```python
gpu = np.load("RECORD_gpu.npz", allow_pickle=True)
gpu["chns_names"]   # shape (N_ch,) — 全量通道名
gpu["whole_dets"]   # shape (N_ch,) object array — 每元素 (n_events, 2) float [start, end] 秒
gpu["events_count"] # shape (N_ch,) — 每通道事件数
gpu["start_time"]   # scalar — block 起始 Unix epoch
```

### 2.3 关键已有函数

| 函数 | 位置 | 作用 | 本文档是否复用 |
|------|------|------|--------------|
| `_try_load_gpu` | `event_periodicity.py:210` | 安全加载 gpu.npz，损坏返回 None | 复用 |
| `select_core_channels_by_event_count` | `group_event_analysis.py:470` | mean+k*std 筛通道 | **需要调整 k 参数** |
| `compute_iei` | `event_periodicity.py` | 从事件序列计算 IEI（排除跨 block） | 复用 |
| `compute_serial_correlation_decay` | `event_periodicity.py` | lag-k Pearson r on log IEI | 复用 |
| `compute_detrended_serial_correlation` | `event_periodicity.py` | 600s 去趋势后 lag-1 r | 复用 |
| `_split_events_by_block` | `event_periodicity.py` | 按 block_ranges 切分事件 | 复用 |

---

## 3. PR-1：Per-Channel SOZ 空间对比基础设施

### 3.1 执行计划

#### Step 0：数据审计（`scripts/audit_gpu_npz.py`）

**在写任何分析代码之前**，先确认数据可用性。审计脚本应该：

1. 遍历 Yuquan 和 Epilepsiae 的所有被试目录
2. 对每个 `*_gpu.npz` 文件：
   - 尝试加载（使用 `_try_load_gpu` 的逻辑）
   - 记录：文件大小、是否损坏、`chns_names` 长度、`events_count` 总和、非零通道数
3. 对每个被试汇总：
   - 有效 block 数 / 总 block 数
   - 总录制时长（小时）
   - 全量通道数 vs lagPat 通道数
   - 超过双阈值（min_count=100 且 min_rate=5 events/hour）的通道数
   - SOZ 标注是否存在、SOZ 通道与 gpu 通道的交集数（使用 alias_bipolar_to_any 匹配）
   - SOZ 通道中超过双阈值的数量 / non-SOZ 通道中超过双阈值的数量
4. 输出 CSV：`results/spatial_modulation/gpu_audit.csv`

**通过标准**：
- Yuquan：≥ 8/18 subjects 有完整 gpu.npz（全部 block 有效）
- Epilepsiae：≥ 10/20 subjects 至少有 1 个有效 gpu.npz block
- 每个通过审计的 subject，在 relaxed refine 下有 ≥ 10 个超过双阈值的通道
- 其中 SOZ 组和 non-SOZ 组各至少 3 个合格通道的 subject ≥ 15

如果审计不通过，**停下来讨论**，不要继续写分析代码。

#### Step 1：Relaxed Refine 通道选择

**思路**：不从头重写 refine 管线。利用已有的 `select_core_channels_by_event_count`，但降低阈值参数 k。

当前 legacy 使用 `method="mean_std", k=1.0`。放松策略：

- **方案 A**：`k=0.0`（threshold = mean），纳入所有事件数超过均值的通道
- **方案 B**：`k=-0.5`（threshold = mean − 0.5*std），进一步放松
- **方案 C**：`min_count` 阈值替代（例如 `min_count=100`，不做均值筛选）

**不单用方案 C 的原因**：不同被试的总事件数差异极大（从数百到数十万）。固定 min_count 在低事件被试上可能选到噪声通道，在高事件被试上可能仍然太严格。`mean_std` 方法天然适应不同被试的事件率分布。但 min_count 和 min_rate 作为**双重下限**仍然需要，用于兜底质控（见 Step 4a）。

**建议**：先在审计数据上对比 k=1.0 / k=0.5 / k=0.0 / k=-0.5 下的通道数分布，选择使中位通道数达到 15–30 的 k 值。**不要跳过这步直接选一个 k**。

**具体做法**：

```python
# 不修改 select_core_channels_by_event_count 的实现
# 只调用时传入不同的 k
from src.group_event_analysis import select_core_channels_by_event_count

# 对每个被试：跨 block 累加 events_count，计算 total_hours
# 在四个 k 值下分别调用，记录入选通道数
for k in [1.0, 0.5, 0.0, -0.5]:
    selected = select_core_channels_by_event_count(
        events_count=sum_events_count,
        ch_names=ch_names,
        method="mean_std",
        k=k,
        min_count=1,
    )
    # 再用 min_rate 做第二道过滤
    selected = [ch for ch in selected
                if (sum_events_count[ch] / total_hours) >= min_rate]
```

这一步的输出：每个被试在不同 k 下入选的通道列表。**不修改任何现有代码**，只是用不同参数调用。

#### Step 2：Per-Channel 事件加载器

新增函数（放在 `src/event_periodicity.py`）：

```python
def load_perchannel_events_relaxed(
    subject_dir: Path,
    dataset: str,
    refine_k: float = 0.0,
    min_count: int = 100,
    min_rate: float = 5.0,
) -> Dict:
    """从 *_gpu.npz 加载放松筛选后的 per-channel 事件序列。

    与 load_yuquan_subject_events 的关键差异：
    1. 通道集来自 gpu.npz 的 chns_names + relaxed refine，不来自 lagPat 的 chnNames
    2. 返回所有合格通道的事件，不仅是 lagPat 子集
    3. 同时返回 lagPat 通道标记，用于标识"原始 refine 通道"
    4. 使用双阈值（min_count + min_rate）控制通道质量

    Parameters
    ----------
    min_count : 绝对事件数下限（serial correlation 统计力需求）
    min_rate : 事件率下限 (events/hour)，归一化跨数据集录制时长差异

    Returns
    -------
    dict with keys:
        per_ch_events : dict[str, ndarray(N,2)]
            绝对时间 [start, end]，按 start 排序
        ch_names : list[str]
            relaxed refine 后的通道列表
        lagpat_channels : set[str]
            原始 lagPat 的 chnNames（严格 refine）
        block_ranges : list[tuple(float, float)]
        total_hours : float
            总录制时长（小时），用于 rate 计算
        events_count_all : dict[str, int]
            gpu.npz 中所有通道的事件计数（审计用）
    """
```

**实现要点**：

- 对 Yuquan：遍历 EDF → gpu.npz，对**每个 block 的 events_count** 做 cross-block 累加，然后对 **sum counts** 调用 `select_core_channels_by_event_count(k=refine_k)`
- 对 Epilepsiae：同 Yuquan，但需处理 gpu.npz 损坏的情况（使用 `_try_load_gpu`）
- **不重新跑 HFO 检测**，直接消费 legacy `*_gpu.npz`
- 返回 `lagpat_channels` 让调用方可以标记"这个通道是原始 refine 还是 relaxed 新增的"

#### Step 3：Per-Channel SOZ 标注

对 relaxed 通道集，标注每个通道的 SOZ 属性：

- Yuquan：binary（SOZ / non-SOZ），来自 `results/yuquan_soz_core_channels.json`
- Epilepsiae：三值（i / l / e），来自 `results/epilepsiae_electrode_focus_rel.json`
- 如果通道在 SOZ JSON 里找不到，标注为 `unknown`（不要丢弃，不要假设是 non-SOZ）

**双极通道的 SOZ 匹配（关键）：必须使用 alias_bipolar_to_any 逻辑**

当前 pipeline 的 `alias_bipolar_to_left: true` 约定把 `A1-A2` 映射为 `A1`。这在群体事件分析中勉强可用，但在 per-channel SOZ 对比中会造成**系统性 spatial leakage**：

> 假设 A2 是临床判定的 SOZ，A1 不是。双极通道 A1-A2 会记录到 A2 的病理高频放电。
> 如果用 alias_to_left，这个通道被映射到 A1，标注为 non-SOZ。
> 结果：non-SOZ 对照组混入了大量真实 SOZ 信号，两组差异被系统性抹平（假阴性偏倚）。

**正确做法**：对双极通道 `X-Y`，如果 X ∈ SOZ **或** Y ∈ SOZ，该通道标记为 SOZ。只有当 X ∉ SOZ **且** Y ∉ SOZ 时，才作为 non-SOZ 对照。

```python
def match_bipolar_soz(ch_name: str, soz_set: set) -> str:
    """对双极通道做 SOZ 匹配，任一触点在 SOZ 则整个通道标为 SOZ。"""
    parts = ch_name.split("-")
    normalized = [p.strip().upper() for p in parts]
    for p in normalized:
        if p in soz_set:
            return "soz"
    return "non_soz"
```

**通道名归一化流程**（按顺序）：

1. `strip()` 去除首尾空格
2. `upper()` 统一大小写
3. 去除 `EEG ` 等前缀
4. 对双极名 `X-Y`：拆分为 `[X, Y]`，两端分别查 SOZ set
5. 对无法匹配的通道，标注 `unknown` 并在审计报告中记录

#### Step 4：Per-Channel 时序指标计算

##### 4a. 质控预处理（在计算指标之前）

放松 refine 阈值后，边缘通道可能引入伪迹。在算 IEI 指标前必须加一层质控：

1. **IEI 物理下限过滤**：剔除 IEI < 10 ms 的连发对。HFO 事件的绝对不应期在 ~10 ms 量级，低于此值的"连发"100% 是检测器分频漏检或肌电伪迹。对该通道的 IEI 序列做 `iei = iei[iei >= 0.01]` 后再计算后续指标。

2. **爆发通道标记**：计算该通道 IEI 的变异系数 CV = σ/μ。如果 CV > 5.0（事件极度聚集在极短窗口，其余时间空白），将该通道标记为 `artifact_suspect = True`。这类通道的 serial correlation 被极端 outlier IEI 主导，不可信。在主分析中**排除** `artifact_suspect` 通道，但在审计报告中保留其指标供检查。

3. **事件率 + 事件数双阈值**：通道合格要求**同时**满足：
   - `n_events >= min_count`（默认 100）——serial correlation 的统计力取决于绝对事件对数
   - `event_rate >= min_rate`（默认 5 events/hour）——排除 24h 内只有几十个散落事件的噪声通道
   
   使用 min_rate 而非单纯 min_count 的原因：Yuquan 是 24h 连续记录，Epilepsiae 是 1h blocks。同一个 min_count=50 在 Yuquan 意味着 ~2 events/hour（极低频，长期截断误差大），在 Epilepsiae 意味着 50 events/hour（活跃通道）。min_rate 归一化了跨数据集的质量基准。

##### 4b. 指标计算

对每个通过质控的通道独立计算：

| 指标 | 函数 | 说明 |
|------|------|------|
| `event_rate` | `n_events / total_recording_hours` | 事件率 |
| `n_events` | 直接计数 | 绝对事件数 |
| `iei_lag1_r` | `compute_serial_correlation_decay` (取 k=1) | IEI 序列相关 |
| `iei_half_life` | 同上函数的 half_life_lag 输出 | 调制时间尺度 |
| `iei_detrended_r` | `compute_detrended_serial_correlation` | 去趋势后残差 |
| `detrend_fraction` | 同上 | 慢漂移占比 |
| `iei_p02` | `np.percentile(iei, 2)` | dead-time proxy |
| `iei_median` | `np.median(iei)` | 中位 IEI |
| `iei_cv` | `np.std(iei) / np.mean(iei)` | 变异系数（质控用） |
| `artifact_suspect` | `iei_cv > 5.0` | 伪迹标记 |

**不新写算法**，全部复用 PR-2 已验证的函数，只是喂入单通道事件序列而非群体事件序列。IEI 物理下限过滤和 CV 质控是新增的预处理步骤。

#### Step 5：Subject-Within SOZ vs non-SOZ 对比

统计设计（每个 subject 内部）：

1. 对 SOZ 通道和 non-SOZ 通道分别取各指标的中位数
2. 配对差值 = median(SOZ channels) − median(non-SOZ channels)
3. Cohort 统计：对 N 个有效 subject 的配对差值做 Wilcoxon signed-rank test

**有效 subject 的定义**：SOZ 组和 non-SOZ 组各至少 3 个通道通过双阈值（min_count + min_rate）且未被标记为 `artifact_suspect`。

**Epilepsiae 三值扩展**：i vs l、i vs e、l vs e 三组配对，Bonferroni 校正。

**事件率协变量控制**：

关键混淆因素是 SOZ 通道事件率更高 → 序列更长 → serial correlation 估计更稳定。控制方案：

- **简单方案**：报告 SOZ vs non-SOZ 的 `iei_lag1_r` 差异时，同时报告 `event_rate` 差异。如果方向一致（SOZ 更高 rate 且更高 r），需要进一步控制。
- **进阶方案**：在 mixed effects 框架下，metric ~ SOZ_label + log(event_rate) + (1|subject)，看 SOZ_label 系数是否在控制 event_rate 后仍显著。
- **匹配方案**：对每个 SOZ 通道，在同一 subject 内找 event rate 最接近的 non-SOZ 通道做 1:1 匹配，只在匹配对上比较。

PR-1 先用简单方案 + 进阶方案，匹配方案视结果需要再加。

**事件率支撑域重叠诊断（推断前必做）**：

在做任何推断统计之前，必须画一张散点图：X = log(event_rate)，Y = iei_lag1_r，颜色 = SOZ/non-SOZ。**只有当两组的 event_rate 分布有实质性重叠区间时**，在该重叠区间内做的统计推断才有效。

如果 SOZ 通道的 event_rate 分布是 [1000, 5000] events/day 而 non-SOZ 是 [50, 300] events/day，两者支撑域无交集，则：
- 1:1 匹配找不到配对
- ANCOVA 会用线性外推"控制" event_rate，这是统计学大忌
- 此时只能诚实报告："SOZ 的强序列相关与其高事件率在数学上不可解耦"

这张散点图是 `figures/event_rate_vs_lag1r_scatter.png` 的内容，也是决定后续统计方案的门控条件。

### 3.2 输出产物

```
results/spatial_modulation/
├── gpu_audit.csv                          ← Step 0
├── relaxed_refine_channel_counts.csv      ← Step 1（不同 k 的通道数对比）
├── per_channel_metrics/                   ← Step 4
│   ├── yuquan/
│   │   └── {subject}_perchannel.json
│   └── epilepsiae/
│       └── {subject}_perchannel.json
├── soz_comparison/                        ← Step 5
│   ├── cohort_soz_vs_nonsoz.csv
│   ├── cohort_statistics.json
│   └── figures/
│       ├── README.md                      ← 必须
│       ├── soz_vs_nonsoz_lag1r_paired.png
│       ├── soz_vs_nonsoz_deadtime_paired.png
│       ├── perchannel_lag1r_by_region.png  （Epilepsiae i/l/e）
│       └── event_rate_vs_lag1r_scatter.png （混淆控制可视化）
```

### 3.3 验证计划

**与已有结果的一致性校验**：

1. 对 lagPat 通道子集，新 loader 的 per-channel IEI 应该与 Phase 1 `run_subject_periodicity` 的 per-channel 结果一致（数值相等或差异 < 1%）
2. 对 lagPat 通道的 group-level serial correlation，应该与 PR-2 exp7 一致
3. 在 `k=1.0` 下，relaxed refine 的通道列表应该**包含**（但可能不完全等于）lagPat 的 chnNames

**legacy 代码维护性警告**：

修改 refine 和 gpu 的管线时需要特别谨慎。已知问题：

- Epilepsiae 的 `*_gpu.npz` 存在损坏 stubs（文件 < 500 bytes）
- legacy 通道命名不一致（大小写、空格、双极命名格式）
- `events_count` 在某些 gpu.npz 里可能是 0 维或形状异常
- `whole_dets` 是 object array，某些通道的子数组可能是空的或形状不规则

**因此**：Step 0 审计不是可选的，它是 PR-1 的通过条件。

### 3.4 不做的事情（PR-1 范围外）

- 不做 per-channel PSD / specparam（那是 Phase 1 已有的，不需要在 relaxed 集上重做）
- 不做群体事件重新 packing（放松 refine 后的群体事件定义变化太大，需要独立设计，属 PR-2+）
- 不做 ROC / AUC 预测分析（需要先确认效应存在，属 PR-3+）
- 不做 mixed effects 模型（PR-1 用配对 Wilcoxon 即可，模型属 PR-2+）
- 不修改 `select_core_channels_by_event_count` 的实现代码（只调不同参数）

---

## 4. 预期结论的可能方向

### 4.1 如果 SOZ 通道的 serial corr 显著更高

→ 支持"SOZ 网络有自主的状态调制"假说，与 PR-2 exp7D 的边缘趋势（p=0.055, n=9）方向一致但统计力更强。
→ 科学故事：SOZ 的兴奋性不仅体现为更高的事件率和更短的 dead-time（PR-1 exp6B 已示），还体现为更强的慢时间尺度状态依赖性。
→ 临床价值：如果效应量大到在个体水平可区分，per-channel lag-1 r 可以作为 SOZ 辅助定位的候选标志物。

### 4.2 如果 SOZ 和 non-SOZ 的 serial corr 无差异

→ 支持"调制是全局性的"（最可能是 sleep/wake 驱动的全脑兴奋性变化）。
→ 这个结果本身也有价值：它意味着慢调制不能区分 SOZ 和 non-SOZ，SOZ 的特异性只体现在事件率和 dead-time 上。

### 4.3 如果 Epilepsiae 呈现 i > l > e 的单调梯度

→ 最有文章价值的结果。建立"调制强度与致痫性的空间梯度"。
→ 可以对接到 Epileptor permittivity 变量的空间分布叙事。

---

## 5. 代码文件清单

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `scripts/audit_gpu_npz.py` | **新建** | Step 0 审计脚本 |
| `src/event_periodicity.py` | 新增函数 | `load_perchannel_events_relaxed` + 调用已有函数的胶水 |
| `scripts/run_spatial_modulation.py` | **新建** | PR-1 batch driver |
| `scripts/plot_spatial_modulation.py` | **新建** | PR-1 图表 |
| `src/group_event_analysis.py` | **不修改** | 只以不同 k 调用 `select_core_channels_by_event_count` |
| `config/default.yaml` | **不修改** | relaxed k 作为脚本参数，不改默认配置 |

---

## 6. 执行顺序

| 步骤 | 依赖 | 估计耗时 | 通过条件 |
|------|------|---------|---------|
| Step 0：审计 | 无 | 0.5 天 | Yuquan ≥8, Epilepsiae ≥10 通过审计 |
| Step 1：k 值选择 | Step 0 | 0.5 天 | 选定 k 使中位通道数 15–30 |
| Step 2：Loader | Step 1 | 1 天 | 与 Phase 1 per-channel 结果一致性校验通过 |
| Step 3：SOZ 标注 | Step 2 | 0.5 天 | 通道名匹配率 > 80% |
| Step 4：指标计算 | Step 2+3 | 1 天 | 30 subjects 全量运行 |
| Step 5：统计 + 图 | Step 4 | 1 天 | figures/README.md 生成 |

**总计**：~4.5 天。Step 0 如果不通过则整个 PR-1 暂停。

---

## 附录 A：`select_core_channels_by_event_count` 的 k 参数效果示意

legacy 默认 `method="mean_std", k=1.0`：只保留 counts > mean + 1*std 的通道。

假设 120 通道的事件数分布为重尾分布（少数 SOZ 通道事件极多，多数通道事件少）：

| k 值 | 大致含义 | 预期通道数 | 覆盖 |
|------|---------|-----------|------|
| 1.0 | 仅保留 top ~15% | ~10–15 | 基本只有 SOZ 和 SOZ 邻近 |
| 0.5 | 保留 top ~25% | ~20–30 | SOZ + 部分活跃 non-SOZ |
| 0.0 | 保留超过均值的 | ~30–50 | SOZ + 大部分活跃通道 |
| −0.5 | 保留 > mean−0.5*std | ~50–70 | 绝大部分有活动的通道 |

**以上是基于正态假设的粗估**。实际的重尾分布会使 k=0 仍然比较严格（均值被少数高事件通道拉高）。需要审计数据确认。

## 附录 B：通道名匹配的已知坑

1. **大小写**：gpu.npz 里可能是 `"A1"`，SOZ JSON 里可能是 `"a1"` 或 `"A'1"`
2. **双极 SOZ 匹配**：gpu 里 `"A1-A2"`，SOZ 里只写 `"A2"`。**必须**用 `alias_bipolar_to_any`——只要任一触点在 SOZ，整个双极通道就标为 SOZ（见 Step 3）。**不要**使用 `alias_bipolar_to_left`，那会导致 spatial leakage
3. **空格**：某些 legacy 文件的通道名末尾有空格
4. **前缀**：某些 Epilepsiae 通道名有 `EEG ` 前缀

匹配流程：strip → upper → 去除已知前缀 → 双极拆分 `X-Y` → 两端分别查 SOZ set → 任一命中即 SOZ → 全部未命中且能在 focus_rel JSON 中找到则标注具体类别 → 否则 `unknown`。

## 附录 C：质控阈值的选择依据

| 阈值 | 默认值 | 理由 |
|------|--------|------|
| `min_count` | 100 | lag-k decay 需要 n_events > 2 × max_lag（max_lag=50 时需 100）；低于此 serial corr 估计不稳定 |
| `min_rate` | 5 events/hour | 归一化跨数据集时长差异；Yuquan 24h × 5 = 120 events ≈ min_count；Epilepsiae 1h × 5 = 5 events（被 min_count 兜底） |
| `iei_min_threshold` | 10 ms | HFO 事件的绝对不应期下限；低于此的"连发"必为检测器伪迹 |
| `cv_threshold` | 5.0 | IEI CV > 5 意味着事件极度聚集在极短窗口；经验值，审计后可调整 |
