# Topic 1：间期事件内部时序结构

> 状态：当前正式入口
> 范围：只讨论单个间期群体事件内部的时序组织，包括传播刻板性与事件级同步性。

---

## 1. 这个 topic 只回答什么问题

本 topic 只回答两个问题：

1. 单个群体事件内部，不同通道的激活顺序是否稳定、是否刻板、是否存在多种主要传播模式。
2. 单个事件内部的同步性指标在发作前后是否表现出系统性变化。

它**不**回答：

- 事件与事件之间的 IEI / PSD / rate modulation：那是 `docs/topic2_between_event_dynamics.md`
- 慢调制发生在 SOZ 还是 non-SOZ：那是 `docs/topic3_spatial_soz_modulation.md`

---

## 2. 一句话当前结论

- **传播线**：单个群体事件内部的传播结构是真实存在的，但不是单一模板；`k=2` 是主导压缩，不是普适真相，少数 subject 存在更丰富的 `k=4` 到 `k=6` 多模态路径集合，而且这些模板在 split-half / blockwise 尺度上总体稳定（`23/30 strong`, `7/30 moderate`, `0 weak`）。`PR-4A` 进一步显示：在固定模板下，day/night occupancy 漂移整体很弱，但它仍是描述层读数，不宜单独承担后续发作邻近分析的正式结论。`PR-4B` 的三类读数分析已经完成（30 subjects）：模板混合 / 模式选择与模板内顺序一致性在 cohort level 均为 **null**；模板内相对时延结构在全 cohort 上无显著差异，但在 8 个 absolute-lag 高置信 subject（dom_r > 0.7）中 Pearson r 显著更高（p=0.016, 7/8 方向一致），表明高速率下同一模板的事件 timing 更一致。跨指标同向性强（24/29 subjects lag span 与 Pearson r delta 同号, binomial p=0.0003）。**H2 在高置信子集有探索性支持，在全 cohort 上证据不足。** `PR-4D` 已验收为描述层补强：它不再平滑 occupancy，而是直接展示固定模板分解后的绝对事件率（rate×type）。当前 cohort 中 dominant template rate fraction 中位数为 `0.584`（range `0.262-0.866`）；`25/30` subject 至少出现一次主导模板交叉，但只有 `17/30` 在中高 rate 区间出现交叉，`6/30` 出现反复交叉。因此它支持“主导模式切换确实存在”，但目前仍是**描述层现象**，不是正式统计结论。
- **同步性线**：cohort-level interictal synchrony 总体为 null，没有支持“post-ictal reset / pre-ictal resynchronization”；唯一探索性信号是 extra-focal `phase_e` 的 `pre > post`。

---

## 3. 核心证据链

### 3.1 内部传播刻板性

来自 `lagPatRank + eventsBool + chnNames` 的 cluster-aware 分析表明：

- `30/30` subject 的 pairwise Kendall `τ` 分布都呈多模态
- KMeans(`k=2`) 后，簇内 `τ` 中位数为 `0.250`，明显高于整体 `τ = 0.089`
- `29/30` subject 的 within-cluster `τ > overall τ`
- `30/30` legacy MI permutation test 显著，复现老论文结论

这说明老论文说的“传播路径可重复”没有错，但原来的“单一平均模板”口径太粗。真正合理的口径是：**一个 subject 常常有多条主要传播路径，而每条路径内部仍然是刻板的。**

### 3.1b 数据合同、聚类稳定性与跨时间复现（PR-2a/2b/2.5）

- PR-2a：`load_subject_propagation_events()` 按 `start_t` 排序 block，事件绝对时间从 `packedTimes[:, 0] + start_t` 重建。channel 跨 block 完全稳定。
- PR-2b：`compute_adaptive_cluster_stereotypy()` 对 k 扫描，multi-seed AMI 稳定性 + 最小簇比例约束。
  - 全量 `30/30` subject 都找到 `stable_k`，`0` fallback。
  - `stable_k` 分布：`27/30` 为 `k=2`，`2/30` 为 `k=4`，`1/30` 为 `k=6`。
  - Adaptive within-cluster `τ` 中位数为 `0.252`，相对整体 `τ = 0.089` 的 uplift 中位数为 `+0.100`。
  - `12/30` subject 出现 `candidate_forward_reverse` 对，共 `17` 对；`958` 的 `r = -0.915` 被精确复现，但 `candidate_forward_reverse` 仍只是描述标签，不是机制结论。
  - `stable_k` 的语义是“在当前事件云上的最佳稳定压缩”，不是“真实传播模式数”。
- PR-2.5：`compute_time_split_reproducibility()` 对同一 subject 做 split-half + odd/even block 模板复现。
  - cohort `30/30` 可用 subject 中，`23/30` 为 `strong`，`7/30` 为 `moderate`，`0` 为 `weak`。
  - split-half 中位模板相关为 `0.899`，中位 assignment agreement 为 `0.865`。
  - odd/even block 中位模板相关为 `0.985`，中位 assignment agreement 为 `0.882`。
  - `9` 个 k=2 subject 带有 forward/reverse 候选对（inter-cluster `r < -0.5`），其中 `8/9` 能在时间切片中复现同一匹配后的互逆关系（使用 first_half_second_half 或 odd_even_block 任一 split）。
  - 这说明我们现在不只是证明了**算法稳定性**，而是已经证明了**blockwise / split-half 尺度上的跨时间模板稳定性**；但这还不等于已经回答了 day/night、seizure proximity 或 occupancy 漂移。

### 3.1c PR-3 / PR-4A：固定模板可视化与 occupancy 漂移

- PR-3 已验收并回到主树：`raw / k=2 / stable_k` 的 per-subject 图和 cohort 可视化已经固定下来，`moderate` subject 在图上不再被伪装成"铁板一块"。
- PR-3 后续（2026-04-13）：cohort 6-panel 图完成论文级重设计。叙事逻辑为 a→c 证明 stereotypy 显著且多模态，d→f 证明模板稳定、反相关、identity-bias 组成。新增簇内 identity-bias 计算（`compute_within_cluster_centered_tau()`），median bias = 86%。Forward/reverse 复现统计修正为 8/9（使用 first_half_second_half 或 odd_even_block 任一 split）。
- PR-4A：`compute_temporal_cluster_dynamics()` 在**固定模板投射**前提下，按 block 覆盖时间窗做 occupancy timeline，避免把停录空窗误画成连续零占比。
- `30/30` subject 都得到 day/night summary。
- dominant fraction：day median `0.587`，night median `0.575`，subject-level Wilcoxon `p = 0.124`。
- normalized entropy：day median `0.960`，night median `0.974`，subject-level Wilcoxon `p = 0.245`。
- day-night total variation distance：median `0.019`，IQR `0.011-0.030`。
- fixed-template projection agreement 中位数为 `0.888`；只有 `3/30` subject 低于 `0.8`（`chengshuai`, `253`, `818`），说明大多数 subject 的 occupancy 轨迹并不是模板投射失真造出来的。
- 这层结果支持的最强口径是：**模板稳定，但占比的昼夜漂移整体较弱**。它是高质量描述性结果，不是强机制结论。
- 但这里有一个必须正视的技术边界：occupancy 本质上是时间窗内模板标签的样本比例。低 rate 时段的 `n_events` 很小，一两个事件就足以把曲线推到极端值，所以它更适合做描述层，而不适合直接充当 `PR-4C` 的主统计读数。`PR-4D` 已把这层补强成 **rate×type** 读数：上面板画每个固定模板的绝对事件率包络（events/hour），下面板画相同模板颜色的离散计数堆叠柱。它回答的是“总 rate 变化时，哪个模板在变”，而不是“比例是否平滑一点”。

### 3.2 Identity bias 不是小问题，在簇内水平更高

**整体水平：**

- Raw `τ` 中位数：`0.089`
- Centered `τ` 中位数：`0.023`
- Bias fraction 中位数：`0.652`

**簇内水平（PR-3 后续补充，2026-04-13）：**

- Within-cluster raw `τ` 中位数：`0.252`
- Within-cluster centered `τ` 中位数：`≈0.03`
- **Within-cluster bias fraction 中位数：`0.86`**

关键解读：在每个传播模式内部，约 86% 的通道排序一致性来自通道本身的固有激活位置（identity ordering），只有约 14% 是事件特异性传播动力学。这并不否定传播结构——通道身份排序本身反映网络拓扑约束。但量化口径必须更新：**stereotypy 主要由结构性通道排序驱动，而不是每次事件独立产生的传播动力学。**

### 3.3 Event-level synchrony 的正式统计口径

PR4–PR6 以 seizure interval 为统计单位，主指标是 `phase`：

- `phase_all` post vs pre：`p = 0.279`
- `phase_core` post vs pre：`p = 0.967`
- within-interval trajectory：`phase_all p = 0.589`，`phase_core p = 0.643`
- event rate：`p = 0.361`

所以 cohort level 没有支持“发作后去同步重置”或“发作前同步性恢复”。

### 3.4 Topic 1 中唯一值得继续追的 synchrony 信号

Epilepsiae 的区域分层分析中：

- `phase_i`：`p = 0.646`
- `phase_l`：`p = 0.543`
- `phase_e`：`p = 0.012`, `r = 0.31`

方向是 `pre > post`，而且 Bonferroni 校正后仍勉强保留。这是目前 synchrony 线中唯一可称为 exploratory-significant 的结果。

---

## 4. 当前最可信的结果

### 4.1 传播刻板性

- 多模态是普遍现象，不是例外；但主要压缩仍然是 `k=2`
- 模板在跨时间切片上总体稳定：`23/30 strong`，其余 `7/30 moderate`，没有 `weak`
- `PR-3` 已经把固定模板的 per-subject / cohort 图稳定下来，后续展示口径不再依赖临时绘图脚本
- `PR-4A` 显示 fixed-template occupancy 的 day/night 漂移很弱：dominant fraction 与 entropy 的 subject-level paired summary 都没有显著差异，TV distance 也很小（median `0.019`）
- `958` 的 forward/reverse 双模式可复现，不是孤例；`9` 个 inter-cluster `r < -0.5` 的 subject 中，`8/9` 能在 split-half 或 odd/even block 分析中复现该关系
- legacy MI 全部显著，老论文最硬的结果站得住
- 真正可信的定量指标应该是 cluster-aware `τ` 与 raw/centered 并列报告，而不是只给一个整体 MI
- 少数 subject 的 `k=4` / `k=6` 结构里，互逆关系仍能复现；但精确 cluster 边界在 `818`、`zhangjinhan` 这类高 k subject 上还不够稳，不能过度解读

### 4.2 同步性

- 主结论是 population-level null
- `phase` 是主指标，`legacy` 仅作兼容，`span` 仅作附录
- `phase_e` 的 `pre > post` 是唯一需要继续追的分层信号

---

## 5. 仍未解决的问题 / 风险点

- SOZ > non-SOZ 的传播优势仍偏弱，当前更像探索性趋势，不该写成定论。
- centered rank 可能过度校正；虽然当前 `soz_source_erased` 仅 `3/30`，但今后仍必须和 raw 结果并列报告。
- `PR-4A` 已经给了 day/night occupancy 的描述性答案，但 occupancy 在低 rate 时段天然高方差；它还没有回答 seizure proximity，也不应被直接升级为正式的强统计结论。
- `candidate_forward_reverse` 目前只是 `inter-cluster Spearman r < -0.5` 的描述标签。它可以提示互逆模式，但还不够资格直接写成生理机制。
- 少数 `k>2` subject 仍需要额外验证，确认高维多模态不是 `n_participating`、稀有事件或 channel identity 残差造成的假复杂度。当前最需要盯的是 `818` 与 `zhangjinhan`。
- 固定模板投射 agreement 虽然整体够高（median `0.888`），但 `chengshuai`、`253`、`818` 这 3 个 subject 仍应谨慎解释时间轨迹细节。
- synchrony 线最大的风险不是假阳性，而是“把 null 写得太花”。现在最诚实的说法就是：**总体 null，局部 extra-focal 线索待验证。**
- propagation 与 synchrony 都是 topic 1，但它们不是同一个统计对象，文档里必须并列而不能混写成一个指标体系。
- **PR-4B 的三类读数分析已完成（2026-04-14）。** 模板混合 / 模式选择为 null，模板内顺序一致性也为 null（identity-bias 限制灵敏度）；模板内相对时延结构的唯一显著信号来自 n=8 的高置信子集（Pearson r p=0.016）。Wilcoxon n=8 的最小可能 p 值为 `2^-8=0.004`，当前 p=0.016 是倒数第二小的可能值（W=1），统计功效极有限。该子集的选择标准（dom_r > 0.7）来自 Step 0 的数据质量判断（不是结果依赖选择），但 8/30 的样本量不足以做出强 population-level 结论。
- **模板内相对时延结构的 lag span 在全 cohort 上 p=0.135（18/30），高置信子集 p=0.055（6/8）。** 方向与 Pearson r 一致（24/29 同号, binomial p=0.0003），但单独不显著。lag span delta 正值意味着高速率状态下 lag 跨度**更大**（不是更压缩）。这与 Kuramoto K(t) 升高 → lag 压缩的最简单预测**相反**——高速率伴随的是更稳定但更展开的传播。
- **huangwanling 在“模板内相对时延结构”读数上完全 ineligible**（eligible_frac=0，仅 n_part=3–4 事件），所以这部分分析 n=29。

---

## 6. 传播模板受慢调控的三类读数框架

PR-4 系列的核心问题是：**固定传播模板受到什么慢调控？**

为了诚实地回答这个问题，正式文档里把读数分成三类，而不是继续用层号代称：

| 读数类别 | 科学问题 | 因变量 | 核心指标 | 数据来源 | 当前状态 |
| --- | --- | --- | --- | --- | --- |
| **模板混合 / 模式选择** | 慢调制改变了哪个模板更常出现，或者总 rate 的变化是由哪个模板贡献的？ | 固定模板在时间窗内的占比，或固定模板分解后的绝对事件率 | occupancy fraction（PR-4A）、template-decomposed rate envelope + stacked count histogram（PR-4D） | `lagPatRank` cluster labels | `PR-4A` 完成；`PR-4D` 已验收为描述层补强 |
| **模板内顺序一致性** | 慢调制改变了模板内部的 rank 一致性？ | within-cluster `τ` | Pairwise Kendall `τ`（high vs low rate） | `lagPatRank` | `PR-4B` 完成 |
| **模板内相对时延结构** | 慢调制改变了模板内部的相对时延几何？ | within-cluster lag span / Pearson `r` | fixed-template 内的 relative-lag 统计 | `lagPatRaw` | `PR-4B` 完成 |

### 为什么需要“模板内相对时延结构”读数

模板内顺序一致性（`τ`）是纯 ordinal metric，只看通道激活顺序，完全不看绝对时间。考虑以下场景：在高率状态下，rank order 完全不变（`τ` 不变），但所有通道的 lag 从 50ms 压缩到 5ms（所有通道更紧密地同步激活）。这在 Kuramoto 模型下是 coupling strength 增强的最自然预测（`K(t)` 升高 → 锁相更紧 → lag 压缩），但 **`τ` 对此完全无感**。

模板内相对时延结构读数需要 `lagPatRaw`（centroid 绝对时间，秒），而不是 `lagPatRank`。数据已经在 `lagPat.npz` 中，无需重新检测。但 `lagPatRaw` 存储的是拼接时间轴上的 centroid 绝对时间（跨事件单调递增），不是事件内相对 lag。做跨事件比较前必须先做 per-event min-subtraction：`relative_lag[ch, ev] = lagPatRaw[ch, ev] - min(lagPatRaw[participating, ev])`。这和 `lag_rank_from_centroids()` 的 `align='first_centroid'` 逻辑以及 synchrony 线 `_compute_sync_metrics` 的处理完全一致。

### 这类读数与 synchrony 线的关系

synchrony 线（`interictal_synchrony.py`）也使用 `lagPatRaw`，但它的 `sync_phase_global` 衡量的是**单个事件内部**通道 lag 的集中程度（within-event dispersion），不是**跨事件**的模式一致性（cross-event consistency）。把模板内相对时延结构引入传播线，是在已有的固定 cluster 结构上加一层 cardinal precision 维度，回答的问题比 synchrony 线更结构化：它是在固定模板框架内跟踪 timing precision，而不是对单个事件做 pre/post 比较。

### 这类读数不需要重做聚类（但 rank cluster 在 absolute-lag 空间只是近似有效）

Legacy `lagPatRank` 是 `lagPatRaw` 的全通道 argsort（rank 0 = 最早 centroid，**包含 non-participating channels**）。传播分析中的 `τ` 只使用 participating channels 子集，所以全通道 rank 语义不影响 `τ` 正确性。方案仍然是：保留 rank-based cluster labels，在已有簇内计算 absolute lag 统计。

**Step 0 实测降级（2026-04-13）**：原假设“rank-based clustering 在大多数情况下也把 absolute lag 结构聚好了”**只对 8/30 subjects 成立**（dominant cluster within-cluster Pearson `r > 0.7`）。Cohort median 为 0.601。但 Spearman(dominant `τ`, dominant `r`) = 0.910，说明 ordinal 好的簇在 cardinal 空间也紧，低 Pearson `r` 的 subject 本身 ordinal stereotypy 就弱。因此 rank cluster 仍然是合理的聚类基础，不需要在 absolute-lag 空间重新聚类，但这类读数的统计解释力度必须按 subject 的 `dominant_r` 分层。

### Pearson r 的 n_participating 门槛

Pearson `r` 在低 `n_participating` 下**完全不可靠**：模拟表明，纯随机 3 维向量有 51% 概率产生 `|r| > 0.7`，4 维有 29%，5 维 17%，8 维以上才降到 6% 以下。因此，模板内相对时延结构的 Pearson `r` 必须设 `min_participating >= 5` 的硬门槛。Step 0 验证报告中必须按 `n_participating` 分层（3-4 / 5-8 / 9+）报告 Pearson `r`，而不是笼统的 cohort median。部分 subject（如 `zhangjinhan`，`n_ch=5`, median `n_part=3`）在这类分析中可能因可用事件不足而无法参与。

### 为什么“模板内相对时延结构”比“模板内顺序一致性”更敏感

Within-cluster identity-bias 高达 86%，意味着模板内顺序一致性（within-cluster `τ`）中只有约 14% 是 event-specific dynamics。即使 H2 成立（coupling modulation 导致 event-specific `τ` 提升 50%），在总 `τ` 上的反映也只有 `14% × 50% ≈ 7%`，这个效应大小在 30 subjects 的 paired Wilcoxon 上很容易被噪声淹没。**模板内相对时延结构中的 lag span 不受 identity-bias 影响**（直接测量时间跨度，不做 rank centering），因此它是 H2 的主检测层；模板内顺序一致性只能作为辅助层。如果 `τ` 为 null 但 lag-span / Pearson `r` 有信号，仍然支持 H2。

### 为什么 MI 不适合这类读数

MI (`_mi_vector`) 本质上是 `np.sign()` 比较，纯 ordinal，输入 absolute lag 和输入 rank 得到完全相同的结果。模板内相对时延结构应使用 **Pearson `r`**（捕获 order + proportional timing，需 `n_part >= 5`）或 **lag span**（鲁棒标量，不受 `n_part` 限制），而不是沿用 MI。

---

## 7. 推荐的下一步验证

PR-3 和 PR-4A 已完成。模板在可视化和时间轨迹上已经固定下来，但 `PR-4A` 的 occupancy 仍只是描述层；低 rate 时段下它是高方差样本比例，不能直接承担后续发作邻近分析的主结论。

### 7.1 PR-4B：Rate state × stereotype coupling（已完成）

**科学问题**：慢 rate state 改变的是模板混合、模板内顺序一致性，还是模板内相对时延结构？对应 `layered_model_framework.md` 的 H1 vs H2 判别。

**Step 0 结论（DONE 2026-04-13）**：

- `lagPatRaw → relative_lag` per-event min-subtraction 正确：30/30 exact order match = 1.0
- Dominant cluster Pearson r cohort median = **0.601**；pooled eligible median = 0.453
- 8/30 subjects dominant_r > 0.7（high confidence）
- Spearman(dominant_τ, dominant_r) = **0.910**（p < 0.0001, n=29）
- huangwanling（eligible_frac = 0）在模板内相对时延结构读数上完全 ineligible → 后续这部分分析 n=29
- **关键洞察**：低 Pearson r 不是 bug，而是与弱 ordinal stereotypy 高度共变 — rank clustering 对这些 subject 本身就不够紧。后续相对时延结构的 rate-coupling 分析应做全 cohort + high-confidence (dom_r > 0.7) 两层报告

**Step 1 结论（模板内顺序一致性, DONE 2026-04-14）**：

1. 按 local event-rate 中位数分窗为 high/low（2h bin, min 20 events/bin），eligible bins 中位数 rate-split
2. **Matched subsampling**：每个 cluster 内，high 和 low 状态的 event 数量做随机下采样匹配（30/30 subjects `matched=True`），消除 event-count 不对称混杂
3. **raw τ**：high vs low delta 中位数 = **+0.003**，Wilcoxon **p = 0.349**（n=30），17/30 high > low
4. **centered τ**：high vs low delta 中位数 = **+0.003**，Wilcoxon **p = 0.221**（n=30），19/30 high > low
5. **结论**：模板内顺序一致性为 **null**。86% identity-bias 使这类读数灵敏度极低（~14% event-specific × effect → 总 tau 变化 <7%），即使 H2 成立（rate modulation 改变 timing precision），rank order 层面也几乎看不到。**这不排斥 H2——只是这类读数对 H2 的检测力不够。**
6. 结果文件：`results/interictal_propagation/pr4b_step1_rate_coupling.json`
7. Cohort summary：`results/interictal_propagation/pr1_cohort_summary.json` → `rate_state_coupling_analysis`

**Step 2–3 结论（模板内相对时延结构 + 模板混合, DONE 2026-04-14）**：

模板内相对时延结构分析：within-cluster lag span 和 Pearson `r` 在 high/low rate 状态间比较，**n_participating 精确匹配**后。

1. **lag span（全 cohort n=30）**：delta median = **+0.001**，18/30 high > low，Wilcoxon **p = 0.135**。方向一致但不显著
2. **lag span（高置信 n=8）**：delta median = **+0.004**，6/8 high > low，Wilcoxon **p = 0.055**
3. **Pearson r（探索性，n=29）**：delta median = **+0.033**，17/29 high > low，Wilcoxon **p = 0.265**
4. **Pearson r（高置信 n=8，dom_r > 0.7）**：delta median = **+0.083**，**7/8 high > low**，Wilcoxon **p = 0.016**
5. **跨指标同向性**：lag span delta 与 Pearson r delta 的 Spearman ρ = **0.628**（p=0.0003, n=29），24/29 subjects 两个 delta 同号（binomial p=0.0003）

模板混合分析：occupancy fraction 与 local rate 的 Spearman 相关。

1. **dominant cluster ρ median = −0.083**，13/30 正方向。Max |ρ| median = 0.275。模板混合为 **null**：事件率不系统性改变模板占比

**综合结论**：

- **模板混合为 null**：速率不改变模板选择
- **模板内顺序一致性为 null**（Step 1）：identity-bias 86% 使灵敏度结构性不足
- **模板内相对时延结构在全 cohort 上为 null**，但高置信子集 Pearson `r` 显著（p=0.016, 7/8）
- Lag span delta 正值 = 高速率下 lag 跨度更大（**不是**更压缩）。与 Pearson r 正值联合解读：高速率状态下同一模板的事件 timing pattern 更一致，但不是 Kuramoto 意义上的 lag 压缩
- H2（rate modulation of timing precision）在高置信子集有**探索性支持**，在全 cohort 上**证据不足**
- 高置信子集 p=0.016 的功效极有限：n=8 Wilcoxon 最小可能 p = 0.004，当前 W=1（仅 1 个 subject 反方向 zhangjinhan）

1. 结果文件：`results/interictal_propagation/pr4b_coupling_summary.json`（per-subject 模板混合 / 顺序一致性 / 相对时延结构），`results/interictal_propagation/pr1_cohort_summary.json` → `rate_state_coupling_analysis`

**前置验证（PR-4B Step 0 — DONE 2026-04-13）**：

1. **Per-event min-subtraction**：`relative_lag[ch] = lagPatRaw[ch] - min(lagPatRaw[participating])`，验证相对 lag 全部非负且 channel-order 与 lagPatRank 一致 → **30/30 exact order match = 1.0, pairwise concordance = 1.0**
2. 对 30 subject 计算 within-cluster Pearson r on relative absolute lag vectors，**按 n_participating 分层报告**（3-4 / 5-8 / 9+）
3. Dominant cluster cohort median Pearson r = **0.601**（range 0.213–0.925）；原始 pooled median = 0.453
4. **8/30 subjects** dominant cluster median_r > 0.7（pass）；**7/30 borderline**（0.6–0.7）；**14/30 weak**（< 0.6）；**1/30 ineligible**（huangwanling, eligible_frac = 0）
5. Dominant τ 与 dominant Pearson r 的 **Spearman ρ = 0.910（p < 0.0001）**：ordinal stereotypy 强的簇在 absolute-lag 空间也紧
6. 结果文件：`results/interictal_propagation/pr4b_lag_validation.json`

**统计单元**：subject。不做 pooled event-level p-value。

### 7.2 PR-4C：Seizure proximity（双轨口径）

**科学问题**：发作邻近是否改变了模板混合，还是改变了固定模板内部的结构？

**分析内容**：

1. 借用 Topic 2 PR-2.7 已有的 seizure-triggered rate framework 定义 pre-ictal / baseline / post-ictal 窗口
2. **次级描述层**：seizure-triggered **rate×type** trajectory —— 发作前后固定模板分解后的绝对事件率曲线与离散计数；先看总 rate 变化由哪个模板贡献，再看是否真的发生主导模式切换
3. **辅助层**：within-cluster `τ` trajectory — 发作邻近的模板内顺序一致性变化（灵敏度受 identity-bias 限制）
4. **主检测层**：within-cluster lag span trajectory + Pearson `r` — 发作邻近的模板内相对时延结构变化（仅 `n_part >= 5` 事件时再谈 Pearson `r`）
5. Subject-level paired comparison（baseline vs pre-ictal vs post-ictal），**lag span 比较需 n_participating 匹配**；`PR-4C` 不应再把 occupancy 单独当成主结论

**前置依赖**：PR-4B 已完成。PR-4B 提供了模板内顺序一致性与模板内相对时延结构的基础实现（`compute_rate_state_coupling`）、absolute lag 验证层（`validate_absolute_lag_clustering`）和 `n_participating` 匹配框架（`_match_event_indices_by_nparticipating`）。PR-4C 可直接复用这些实现，将 rate-state 分窗替换为 seizure-proximity 分窗；次级描述层则直接复用 `PR-4D` 的 gap-aware `rate×type` 读数。

### 7.3 PR-4D：Template-rate decomposition（PR-4A 的描述层补强，已验收）

**科学问题**：当总 event rate 随时间变化时，变化究竟来自哪个固定模板？

**当前接受口径**：

1. 不再去“平滑 occupancy”。正式保留的读数只有一个：**固定模板分解后的绝对事件率**。每个 subject 只保留一张图：上面板是各模板的 smoothed rate envelope（events/hour），下面板是同色离散计数堆叠柱。
2. 暴露时间只按真实 recording coverage 计算；gap 必须留白，不能画成零活动，也不能用连续折线跨过去。
3. 当前 cohort 的 dominant template rate fraction 中位数为 `0.584`（range `0.262-0.866`）。`25/30` subject 至少有一次主导模板交叉；若只看总 rate 高于各自 peak 的 `25%` 的区间，仍有 `17/30` subject 出现交叉，其中 `6/30` 出现反复交叉（`>=3` 次）。
4. 这说明**主导模式切换是存在的**，而且不全是低 rate 尾部噪声；但它目前仍是描述层现象，不应单独升级成正式 inferential 结论。
5. 当前全量 batch 中，PR-4D 没有出现模板投射丢失导致的 event drop（`n_events_used == adaptive_cluster_labels` for `30/30`），所以这层图不是由固定模板投射失败造出来的。

### 7.4 高 k subject 的鲁棒性复核

- 对 `818`、`zhangjinhan` 做 `n_participating` 匹配子样本、raw/centered 双版本模板比较
- 可与 PR-4B 并行，不阻塞

### 7.5 优先级

1. ~~**PR-4B**（P0）：rate × 模板内顺序一致性 + 模板内相对时延结构 + absolute lag validation~~ — **DONE（2026-04-14）**
2. **PR-4C**（P0）：seizure proximity 双轨分析 — 主读数是模板内相对时延结构，次级描述层直接复用 PR-4D 的 rate×type 图
3. ~~**PR-4D**（P1）：补强模板混合的连续时间与 rate-aware 口径~~ — **DONE / ACCEPTED（2026-04-16）**
4. **高 k 复核**（P1）：不阻塞，可并行

---

## 8. 代码与结果入口

### 内部传播

- 文档：`docs/archive/topic1/interictal_group_event_internal_propagation.md`
- 代码：`src/interictal_propagation.py`
- 脚本：`scripts/run_interictal_propagation.py`、`scripts/plot_interictal_propagation.py`
- 结果：`results/interictal_propagation/`

### 事件级同步性

- 文档：`docs/archive/topic1/interictal_synchrony_preliminary_report_2026-04-03.md`
- 代码：`src/interictal_synchrony.py`、`src/interictal_synchrony_aggregation.py`、`src/interictal_synchrony_analysis.py`
- 脚本：`scripts/pr6_interictal_sync_figures.py`
- 结果：`results/interictal_synchrony/analysis/combined/`

---

## 9. 与其他 topic 的边界

- 如果问题在问“`~2 Hz` 峰是不是真的”或“IEI serial correlation 说明什么”，跳到 `docs/topic2_between_event_dynamics.md`
- 如果问题在问“SOZ 和 non-SOZ 到底差在哪里”，跳到 `docs/topic3_spatial_soz_modulation.md`
- 如果问题同时涉及“传播是否真实”和“慢调制是否发生在 SOZ”，先分别读 topic 1 和 topic 3，不要混成一个问题

---

## 10. 历史文档索引

- `docs/archive/topic1/interictal_group_event_internal_propagation.md`
  - 这份是内部传播线的详细结果与合同文档
- `docs/archive/topic1/interictal_synchrony_preliminary_report_2026-04-03.md`
  - 这份是 PR4–PR6 的统计报告

这两份文档保留为历史事实来源；当前正式口径以本文件为准。