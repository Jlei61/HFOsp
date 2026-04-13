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

- **传播线**：单个群体事件内部的传播结构是真实存在的，但不是单一模板；`k=2` 是主导压缩，不是普适真相，少数 subject 存在更丰富的 `k=4` 到 `k=6` 多模态路径集合，而且这些模板在 split-half / blockwise 尺度上总体稳定（`23/30 strong`, `7/30 moderate`, `0 weak`）。`PR-4A` 进一步显示：在固定模板下，day/night occupancy 漂移整体很弱，当前更像轻微描述性变化，而不是强昼夜重排。
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
  - `12` 个带有 forward/reverse 候选对的 subject 中，`11` 个能在时间切片中复现同一匹配后的互逆关系；`huanghanwen` 未通过该关。
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
- `PR-4A` 已经给了 day/night occupancy 的描述性答案，但还没有回答 seizure proximity，也没有把 day/night 写成正式的强统计结论。
- `candidate_forward_reverse` 目前只是 `inter-cluster Spearman r < -0.5` 的描述标签。它可以提示互逆模式，但还不够资格直接写成生理机制。
- 少数 `k>2` subject 仍需要额外验证，确认高维多模态不是 `n_participating`、稀有事件或 channel identity 残差造成的假复杂度。当前最需要盯的是 `818` 与 `zhangjinhan`。
- 固定模板投射 agreement 虽然整体够高（median `0.888`），但 `chengshuai`、`253`、`818` 这 3 个 subject 仍应谨慎解释时间轨迹细节。
- synchrony 线最大的风险不是假阳性，而是“把 null 写得太花”。现在最诚实的说法就是：**总体 null，局部 extra-focal 线索待验证。**
- propagation 与 synchrony 都是 topic 1，但它们不是同一个统计对象，文档里必须并列而不能混写成一个指标体系。

---

## 6. 推荐的下一步验证

PR-3 和 PR-4A 已完成。模板在可视化和 occupancy 时间轨迹上都已经固定下来。

1. **高 k subject 的鲁棒性复核**
   - 对 `k>2` subject 和 forward/reverse 候选，做 `n_participating` 匹配子样本、raw/centered 双版本模板比较。
   - 目标是排除"复杂结构只是稀疏事件或 channel identity 偏差"的垃圾解释；当前最关键的对象是 `818` 和 `zhangjinhan`。
2. **PR-4B：和 Topic 2 的慢调制做固定模板 coupling**
   - 既然模板本身已经被验证稳定，下一步才能诚实地问：慢 rate state 改变的是模式占比，还是模式内部的 stereotype 强度。
3. **seizure proximity 的固定模板占比轨迹**
   - `PR-4A` 已经把 day/night 的描述层补上了，下一步若继续做时间上下文，应该优先问 seizure 邻近，而不是再重复一个昼夜散点图。

现在优先级最高的是 `1` 和 `2`。因为 `PR-3` 和 `PR-4A` 已经把"模板长什么样"和"占比会不会明显漂"这两层补上了，接下来该做的是盯高 `k` 风险点，以及把 Topic 1 和 Topic 2 的耦合问题收窄成一个硬分析。

---

## 7. 代码与结果入口

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

## 8. 与其他 topic 的边界

- 如果问题在问“`~2 Hz` 峰是不是真的”或“IEI serial correlation 说明什么”，跳到 `docs/topic2_between_event_dynamics.md`
- 如果问题在问“SOZ 和 non-SOZ 到底差在哪里”，跳到 `docs/topic3_spatial_soz_modulation.md`
- 如果问题同时涉及“传播是否真实”和“慢调制是否发生在 SOZ”，先分别读 topic 1 和 topic 3，不要混成一个问题

---

## 9. 历史文档索引

- `docs/archive/topic1/interictal_group_event_internal_propagation.md`
  - 这份是内部传播线的详细结果与合同文档
- `docs/archive/topic1/interictal_synchrony_preliminary_report_2026-04-03.md`
  - 这份是 PR4–PR6 的统计报告

这两份文档保留为历史事实来源；当前正式口径以本文件为准。
