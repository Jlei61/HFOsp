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

- **传播线**：单个群体事件内部的传播结构是真实存在的，但不是单一模板；`k=2` 是主导压缩，不是普适真相，少数 subject 存在更丰富的 `k=4` 到 `k=6` 多模态路径集合，而且这些模板在 split-half / blockwise 尺度上总体稳定（`23/30 strong`, `7/30 moderate`, `0 weak`）。`PR-4A` 进一步显示：在固定模板下，day/night occupancy 漂移整体很弱，但它仍是描述层读数，不宜单独承担后续发作邻近分析的正式结论。`PR-4B` 的三类读数分析已经完成（30 subjects）：模板混合 / 模式选择与模板内顺序一致性在 cohort level 均为 **null**；模板内相对时延结构在全 cohort 上无显著差异，但在 8 个 absolute-lag 高置信 subject（dom_r > 0.7）中 Pearson r 显著更高（p=0.016, 7/8 方向一致），表明高速率下同一模板的事件 timing 更一致。跨指标同向性强（24/29 subjects lag span 与 Pearson r delta 同号, binomial p=0.0003）。**H2 在高置信子集有探索性支持，在全 cohort 上证据不足。** `PR-4D` 已验收为描述层补强：它不再平滑 occupancy，而是直接展示固定模板分解后的绝对事件率（rate×type）。当前 cohort 中 dominant template rate fraction 中位数为 `0.584`（range `0.262-0.866`）；`25/30` subject 至少出现一次主导模板交叉，但只有 `17/30` 在中高 rate 区间出现交叉，`6/30` 出现反复交叉。因此它支持“主导模式切换确实存在”，但目前仍是**描述层现象**，不是正式统计结论。`PR-4C` 三处实现合同问题已于 2026-04-19 完成 P0 修复（pair-wise window usability、候选枚举式事件归属、gap-aware rate denominator）并完成主+辅助两配置全量复跑（n_usable_windows 主 187→**360**，辅 245→**370**）。**修复后传播模式五指标 cohort Wilcoxon 仍然 null（主 1/15、辅 1/15 名义显著且跨配置不一致）→ 模板内部几何无稳健发作邻近调制，正式封板为阴性。** 唯一稳健发作邻近信号在 `rate_by_template`：post_ictal vs baseline 的主导模板事件率在主+辅两配置下均通过 Wilcoxon（主 p=0.0009，辅 p=0.0067，方向一致），baseline 真实 rate 修复后从误报 ~~152/h 降到 ~72/h（main）/~~109/h（aux），peri-ictal/baseline 倍数 ~~2.4×（main）/~~1.4×（aux）。详见 `docs/archive/topic1/pr4c_seizure_proximity_review_2026-04-17.md` §9。**PR-5（2026-04-20 立项，当前 P0）**：把这条 rate_by_template 信号正式化为 Topic 1 × Topic 2 桥接结论；分两阶段——先用 novel-template falsification gate 钉死"peri-ictal 仍属全局稳定模板库"前提（PR-5-A），再做 dominant template 的 recruitment shift 主分析（PR-5-B）；KONWAC v2 进阶建模延后到 PR-6 占位、不进 PR-5 工作量。完整计划见 `docs/archive/topic1/pr5_template_recruitment_plan_2026-04-20.md`，§7.8 / §7.9 是主文档入口。
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
- **PR-4C 三处合同问题已于 2026-04-19 完成 P0 修复并复跑（status：CLOSED）**。修复后 n_usable_windows 主 187→360、辅 245→370，传播模式五指标 cohort Wilcoxon 仍然 null（主 1/15、辅 1/15 名义显著且跨配置不一致）→ 模板内部几何无稳健发作邻近调制可正式封板。原本认为可能被合同 bug 掩盖的 `pre_ictal vs baseline raw_tau aux` 显著性在修复后（p=0.0005 → p=0.141）也消失，进一步印证旧版那条信号是 bug 制品。详见 `docs/archive/topic1/pr4c_seizure_proximity_review_2026-04-17.md` §9。
- **PR-5（已立项 / 待执行）的两条诚实风险**：(1) novel-template gate 的 `r / e / gap` 漂移阈值（`|Δr| ≤ 0.05`、`|Δe / e_baseline| ≤ 0.10`）是首次设定，若 gate 在某个方向被边缘性触发，必须按 archive 写死的口径走，不允许事后调阈值；(2) gate FAIL 不等于"peri-ictal 出现新模板"被证实，只等于"H_OOD 不能排除"，此时 PR-5-B 不启动、PR-6 同步冻结，PR-4C `rate_by_template` 维持描述层口径，而不是退守到"peri-ictal 模板不同"的另一个未验证叙事。
- **PR-5-B 的样本风险**：总 cohort 是 30 subjects，但只有 26 subjects 有 PR-4C usable seizure-proximity 输出；PR-5-A 再叠加 `per-state >= 30 events` 与 `min_participating_l3 = 5` 两道门槛后，预计可分析 cohort 可能掉到 ~20。这是已知代价；不允许通过降低 `min_state_events_for_gate` 或把 `n_part < 5` 事件塞回 PR-5-B 来补救。
- **PR-5-B dominant 选择的双口径风险**：保留 `dominant_global`（候选 A，与 PR-4D 一致）与 `dominant_per_window`（候选 B，每 window 重选）两条并行路径。代价：(a) forward/reverse 高对称 subject 上两条路径可能给出不同 dominant id，`dom_agreement` 接近 0.5；这种情况下按 §7.8 sensitivity gate 走 `medium` 路径，不允许回头挑一条更好看的当主结论；(b) 候选 B 功效天然低于候选 A，候选 B 不显著时不允许改其定义（如改成 per-seizure 或 per-day 重选）来补救。

---

## 6. 传播模板受慢调控的三类读数框架

PR-4 系列的核心问题是：**固定传播模板受到什么慢调控？**

为了诚实地回答这个问题，正式文档里把读数分成三类，而不是继续用层号代称：


| 读数类别            | 科学问题                                  | 因变量                                   | 核心指标                                                                                         | 数据来源                        | 当前状态                         |
| --------------- | ------------------------------------- | ------------------------------------- | -------------------------------------------------------------------------------------------- | --------------------------- | ---------------------------- |
| **模板混合 / 模式选择** | 慢调制改变了哪个模板更常出现，或者总 rate 的变化是由哪个模板贡献的？ | 固定模板在时间窗内的占比，或固定模板分解后的绝对事件率           | occupancy fraction（PR-4A）、template-decomposed rate envelope + stacked count histogram（PR-4D） | `lagPatRank` cluster labels | `PR-4A` 完成；`PR-4D` 已验收为描述层补强 |
| **模板内顺序一致性**    | 慢调制改变了模板内部的 rank 一致性？                 | within-cluster `τ`                    | Pairwise Kendall `τ`（high vs low rate）                                                       | `lagPatRank`                | `PR-4B` 完成                   |
| **模板内相对时延结构**   | 慢调制改变了模板内部的相对时延几何？                    | within-cluster lag span / Pearson `r` | fixed-template 内的 relative-lag 统计                                                            | `lagPatRaw`                 | `PR-4B` 完成                   |


### 为什么需要“模板内相对时延结构”读数

模板内顺序一致性（`τ`）是纯 ordinal metric，只看通道激活顺序，完全不看绝对时间。考虑以下场景：在高率状态下，rank order 完全不变（`τ` 不变），但所有通道的 lag 从 50ms 压缩到 5ms（所有通道更紧密地同步激活）。这在 Kuramoto 模型下是 coupling strength 增强的最自然预测（`K(t)` 升高 → 锁相更紧 → lag 压缩），但 `**τ` 对此完全无感**。

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

**当前状态（2026-04-19，P0 完成）**：主配置 `(-4, -1)/(-1, -0.25)/(0.25, 1) h` + 辅助配置 `(-2, -0.5)/(-0.5, -1/12)/(1/12, 1) h` 全量已完成两轮：第一轮（2026-04-17）识别出三处合同问题（window usability 过严 / nearest-seizure 归属错位 / rate denominator 未 gap-aware）；第二轮（2026-04-19）三处合同 TDD 修复后复跑（test 36 passed；n_usable_windows 主 187→360、辅 245→370）。**修复后 propagation pattern 五指标 cohort Wilcoxon 在主 + 辅两配置下仍然 null（主 1/15、辅 1/15 名义显著且跨配置不一致）→ 模板内部几何无稳健发作邻近调制可正式封板**。唯一稳健信号在 `rate_by_template` 层：post_ictal vs baseline 主导模板事件率主 p=0.0009、辅 p=0.0067，方向一致。后续 PR-4C 不再追加 propagation geometry 假设，而是把 rate-by-template 信号搬到 §7.6（Topic 1 × Topic 2 桥接）走正式分析。详见 `docs/archive/topic1/pr4c_seizure_proximity_review_2026-04-17.md` §9。

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
2. **PR-4C**（P0 已完成，2026-04-19）：seizure proximity 双轨分析两轮已跑完，三处合同问题已 TDD 修复并复跑。结论已封板：传播模式几何无稳健发作邻近调制；rate-by-template 是唯一稳健信号。**P0 状态：CLOSED**，不再在 PR-4C 内部继续扩 propagation geometry 假设，转入 §7.6 / §7.7 + Topic 3 §7 的后续可选方向
3. ~~**PR-4D**（P1）：补强模板混合的连续时间与 rate-aware 口径~~ — **DONE / ACCEPTED（2026-04-16）**
4. **高 k 复核**（P1）：不阻塞，可并行
5. **PR-5**（P0，**已立项 / 待执行**）：把 PR-4C 留下的唯一稳健信号（`rate_by_template` post_ictal vs baseline）正式化为 Topic 1 × Topic 2 桥接；先用 novel-template falsification gate 钉死前提（§7.8），再做 recruitment shift 主分析。计划合同：`docs/archive/topic1/pr5_template_recruitment_plan_2026-04-20.md`
6. **PR-6**（待定）：KONWAC v2 进阶建模（多模板 Hebbian 容量 + α(t) 慢调制 + 个体化推断），硬前置 = PR-5 gate PASS 且 recruitment shift 在主+辅两配置 Bonferroni 后通过；当前略写占位（§7.9）
7. ~~**§7.6 / §7.7 可选方向**（P2/P3）~~：§7.6 已被 PR-5 吸收为正式分析，§7.7 仍维持 exploratory 子集

---

### 7.6 后续可选方向：模板招募频率而非模板几何（Topic 1 × Topic 2）

> 来源：`docs/archive/topic1/pr4c_seizure_proximity_review_2026-04-17.md` §6.3 + §9.5。
> 状态：**已升级为 PR-5 正式分析**（2026-04-20）。本节保留科学问题陈述与最小工作合同；具体计划、假设、失败合同、测试合同与数据测试环境见 §7.8 与 `docs/archive/topic1/pr5_template_recruitment_plan_2026-04-20.md`。

**科学问题**：PR-4C P0 修复后的最稳健发作邻近信号在 `rate_by_template` 层（main + auxiliary 两个配置下 post_ictal vs baseline 主导模板事件率均通过 Wilcoxon，主 p=0.0009、辅 p=0.0067，方向一致）。这指向"哪个模板被招募的频率随发作邻近变化"，而不是"模板内部几何如何变形"。这条问题与 Topic 2 的 seizure-triggered rate framework（PR-2.7）天然衔接，把 Topic 1 的固定模板套到 Topic 2 已有的发作邻近窗口上。

**最小工作合同**：

1. 复用 PR-2.7 的 seizure-triggered rate 窗口定义（不新设窗口规则）
2. 把每个窗口里的事件按 PR-4D 的固定模板分解，但 **PR-5-B 只在与 PR-5-A 相同的 gate-eligible 事件池**（`min_participating_l3 = 5`）上重算 per-template event rate；不直接吃 PR-4C 原始 all-event `rate_by_template`
3. 主统计读数：subject-level paired 比较 dominant template 的 rate × 状态；次级读数：non-dominant templates 的 rate
4. 暴露分母**必须**走 PR-4D 的 gap-aware coverage（与 §5 列出的 `rate_by_template` 修复一致）
5. 不引入新 metric，不画新主图；只在已有 PR-4D rate envelope 上叠加发作邻近窗口

**判定边界**：这条线**不**回答"模板内部几何是否变形"——那由 PR-4C 主分析回答（PR-4C P0 已封板：geometry stable across baseline/pre/post）。这条线只回答"哪些模板更常被招募"。当前只能说：PR-4C 已封板几何稳定，PR-5 将检验 recruitment 信号能否升级为正式结论；在 PR-5 跑完前，不能把两条线混写成"已并存结论"。

### 7.7 后续可选方向：高置信子集的窄 exploratory 分支（Topic 1 内部）

> 来源：`docs/archive/topic1/pr4c_seizure_proximity_review_2026-04-17.md` §6.4。
> 性质：可选 / 后续；PR-4C P0 已 CLOSED（2026-04-19）。本方向是**全 cohort 阴性已封板**之后的 exploratory 子集分析，仅用于辅助叙事，不作为主结论。

**科学问题**：Topic 1 中信号最强的子集是 8/30 dominant_r > 0.7 的 absolute-lag 高置信 subject 与 8/9 forward/reverse 跨时间复现的 subject。这部分 subject 上模板本身就最稳定，刻板程度也最强。在他们身上做条件性的发作邻近分析，不强求 cohort-level 普适结论，作为**机制性 case-series**呈现。

**最小工作合同**：

1. subset 定义两种：(a) `dominant_r > 0.7` 高置信 (n=8)；(b) `inter-cluster r < -0.5` 且跨时间复现的 forward/reverse 对 (n=8/9)
2. 只跑 PR-4C 修复后的主读数（`lag_span` + `pearson_r`），不引入新 metric；本子集主分析与 cohort 主分析使用同一份 P0 修复后的 `compute_seizure_proximity_coupling` 输出
3. 单独报告：subject-level paired 比较 + 个案展示，不做 cohort-level inferential 论述
4. **必须**写明 selection criterion 是 Step 0 / PR-2.5 数据质量门槛，而非结果依赖选择，避免被读成 cherry-picking
5. 不要把 case-series 写成"全 cohort 阴性但子集阳性"叙事；这两条结论是不同 scope，不同 statistical scope 不互相否决

**判定边界**：本分支永远是 exploratory 层；PR-4C 主分析的 cohort 阴性结论已封板（2026-04-19），本分支不替代该结论，也不据此追加 propagation geometry 的新假设。

### 7.8 PR-5：Template Recruitment Around Seizures（当前 P0，PR-5-A 已完成 / PR-5-B 待执行）

> 完整计划合同（数据/假设/失败合同/代码入口/测试合同/工作量）：`docs/archive/topic1/pr5_template_recruitment_plan_2026-04-20.md`
> PR-5-A gate 全 cohort 跑数与判定中间报告：`docs/archive/topic1/pr5a_novel_template_gate_2026-04-20.md`
> 性质：正式入口；本节只保留科学问题、PR 拆分与启动条件，不重述具体阈值与 metric 定义（避免与 archive 双源漂移）。

**当前状态（2026-04-20 复跑）**：PR-5-A `overall_pass=True`。主配置 (retained `n=23`) 与辅助配置 (retained `n=22`) 在 r / e / gap × pre/post × baseline 6 条对比上全部满足 archive §3.5 阈值（最大 \|Δr\| = 0.0088、最大 \|Δe / e_baseline\| = 0.0169、所有 Wilcoxon p ≥ 0.21、gap 漂移方向均非"peri-ictal 显著低于 baseline"）；cohort inferential 层同时补齐 sign test，方向计数也未显示系统偏斜。最终口径：在 **PR-5-A gate-eligible retained cohort** 中，peri-ictal 事件**不能拒绝**"仍属全局稳定模板库 in-distribution"零假设；按预设 `r/e/gap` diagnostics，**未观察到支持** `H_OOD` 或 `H_assignment_drift` 的 cohort-level evidence。**PR-5-B (`--pr5-recruitment`) 解锁，可立即启动。**

**科学问题**：把 PR-4C 唯一稳健发作邻近信号（`rate_by_template` 的 post_ictal vs baseline 升高，主 p=0.0009 / 辅 p=0.0067）从 PR-4C 内部的次级描述层升级为正式的 Topic 1 × Topic 2 桥接结论；同时严格证伪一个直接的备择假设——peri-ictal 事件可能并不属于全局稳定模板库。

**两阶段拆分**：

1. **PR-5-A（Falsification gate）**：peri-ictal 事件能否被全局稳定模板库以与 baseline 同质的 best-template Spearman r、reconstruction error 与 assignment gap 解释？
   - 不重新聚类 peri-ictal 子集（重新聚类只作为探索性诊断进 archive，不作为 gate 主读数）
   - 实现上复用 PR-4C P0 修复后的同一套 window-building 合同（`_build_seizure_proximity_windows` + 同口径 `v_ranks/v_rel/v_bools`），并结合既有 `T_global = build_cluster_templates(...)`；不是从 PR-4C JSON 直接 reload `state_event_indices`
   - PASS / FAIL 阈值在 archive §3.5 写死，禁止事后调
2. **PR-5-B（Template recruitment shift）**：仅在 PR-5-A `overall_pass=True` 时启动。subject-level paired Wilcoxon on dominant template `events/hour`（gap-aware exposure），主+辅两配置同时跑。**PR-5-B 不直接复用 PR-4C 原始 all-event `rate_by_template` 数值**；它只复用 PR-4C 的 windows / usability / coverage 合同，并在与 PR-5-A **相同的 gate-eligible 事件池**（`min_participating_l3 = 5`）上重算每个 window / state 的 `counts_by_template` 与 `rate_by_template_per_hour`，保证“先 gate、再主分析”闭环。**dominant 定义保留两条并行路径**：候选 A `dominant_global`（与 PR-4D 一致，全程 occupancy 最高）、候选 B `dominant_per_window`（每 usable window baseline 内重选）。两条候选**分别**做 3 pair × 2 config = 6 主比较 → 候选内 Bonferroni alpha = 0.0083；两条之间不再二次 Bonferroni（同现象的不同算子，不是独立假设家族）。Sensitivity gate 三态**完全以 archive §4.4 为准**：`strong` = 候选 A 的 `post_vs_baseline` 在主+辅两配置下方向一致，且至少一个配置 Bonferroni 通过、另一个 nominal 0.05 通过，**且**候选 B 在主配置 nominal 0.05 同向；`medium` = 候选 A 满足上述条件但候选 B 未独立支持；`descriptive` = 候选 A 本身都过不了。`dom_agreement` = 候选 A 与候选 B 的 dominant id 在 usable windows 上的一致比例，作为描述性辅助。

**硬启动条件 / 失败行为**：

- gate FAIL → **不**启动 PR-5-B；Topic 1 主文档明确写"PR-5 在当前数据上未通过 gate，PR-4C `rate_by_template` 维持描述层口径"；此时 PR-6 也保持冻结
- recruitment shift sensitivity FAIL（主+辅方向不一致或一边 nominal 0.05 都过不了）→ 降级为 descriptive，不写 inferential 主结论

**与 PR-4C / PR-4D 的边界**：

- PR-4C 五指标几何 cohort null **保持封板**，PR-5 不重开
- PR-4D `rate×type` 描述层保持原状；PR-5-B 输出只在 PR-4D per-subject 图上加 baseline / pre / post inset，不开新主图
- PR-5 不涉及 SOZ 解剖锚定（属于 Topic 3 §7 独立 P1 候选，与 PR-5 并行，不进 PR-5 工作量）
- 不引入 KONWAC v2 / manifold / persistent homology — 见 §7.9

**代码入口**（详见 archive §5）：

- 复用：`compute_seizure_proximity_coupling`、`build_cluster_templates`、`assign_events_to_templates`、`SEIZURE_PROXIMITY_CONFIGS`
- 新增：`compute_novel_template_gate`、`compute_template_recruitment_shift`（`src/interictal_propagation.py`）
- 脚本：`scripts/run_interictal_propagation.py --pr5-gate`（必跑） / `--pr5-recruitment`（gate PASS 后启动；脚本层面 fail-fast）
- 测试：PR-5 全计划目标仍是 archive §5.3 的 8 项；**截至 PR-5-A 验收** 已落地其中 5 项（3 个 gate 行为测试 + 2 个 cohort summary / threshold 测试），`tests/test_interictal_propagation.py` 现为 36→41 项全绿；PR-5-B 再补剩余 3 项
- 数据测试环境：Python 3.10+ + 现有 venv（`scipy.stats.wilcoxon`、`scipy.optimize.linear_sum_assignment` 已在），CPU 即可，单 subject smoke = Epilepsiae `548`

**工作量**：7–8 工作日（≈1.5 周），前提 PR-4C 输出可直接复用、无需回头修

### 7.9 PR-6：KONWAC v2 进阶建模（占位，略写）

> 完整科学定位与候选可证伪问题：`docs/archive/topic1/pr5_template_recruitment_plan_2026-04-20.md` §9。
> 性质：占位；本节只锁 scope 与启动条件，**不**展开实现，**不**进入当前排期。

**为什么不在 PR-5 内做**：老 KONWAC（2024）已经把"框架能解释现象"做完。继续沿老 KONWAC 重跑没有论文价值边际。要做 v2，必须以 **PR-4 已钉死的事实** 与 **PR-5 若通过 gate / sensitivity 后才能成立的结果** 做强约束反过来约束模型，否则会复刻老论文。

**硬启动条件**：

- PR-5-A gate PASS
- PR-5-B 在主 + 辅两配置下 `post_vs_baseline` 通过 Bonferroni
- 任一不满足 → PR-6 冻结，Topic 1 不再排期

**候选可证伪问题（略写，详见 archive §9.3）**：多模板 Hebbian 容量 / 干扰（`zhangjinhan` n_ch=5 k=6 stress test）、慢 α(t) 调制能否同时复现 geometry stable + 候选 recruitment shift、forward/reverse 在多模板叠加下的天然出现、86% identity-bias 用作 noise/coupling 比例硬约束。

**预估工作量**：6–10 周（建模 + 个体化推断），单独立 PR；工程入口规划放 `src/interictal_propagation_model/`，与分析层隔离。

**几何 / manifold / persistent homology 方向**：不进 PR-5/PR-6 主线；如果 PR-6 成立后模型与数据稳定，再考虑作为 PR-7 的可选 discussion 层引入，**当前不立项**。

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
- `docs/archive/topic1/pr4c_seizure_proximity_review_2026-04-17.md`
  - 这份是 PR-4C 主+辅助配置全量结果的审阅归档：§1-§8 是 2026-04-17 第一轮审阅的 cohort 数值表 / 三处实现合同问题（含 line refs）/ 为什么阴性不能直接封板的判断 / P0/P1/P2/P3 后续路线；§9 是 2026-04-19 P0 修复完成后的复跑数值与正式封板结论。Topic 1 §3.1c / §5 / §7.2 / §7.6 / §7.7 都引用本文件。
- `docs/archive/topic1/pr5_template_recruitment_plan_2026-04-20.md`
  - 这份是 PR-5 的完整计划合同：科学问题、主/备择假设、失败合同与 fail-fast、PR-5-A novel-template gate / PR-5-B recruitment shift 的 metric 定义与统计阈值、与 gate 同一事件池的主分析约束、复用与新增代码入口、8 项 TDD 测试合同、数据测试环境、工作量与时序，以及 §9 PR-6 略写占位（KONWAC v2 启动条件与候选可证伪问题）。Topic 1 §5（PR-5/PR-6 风险点）、§7.5（优先级）、§7.6（升级口径）、§7.8（PR-5 入口）、§7.9（PR-6 占位）均引用本文件。

这四份文档保留为历史事实来源；当前正式口径以本文件为准。