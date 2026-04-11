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

- **传播线**：单个群体事件内部的传播结构是真实存在的，但不是单一模板，而是多模态且长期稳定的路径集合。
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

### 3.2 Identity bias 不是小问题，但也不是全部

- Raw `τ` 中位数：`0.089`
- Centered `τ` 中位数：`0.023`
- Bias fraction 中位数：`0.652`

也就是说，约 `65%` 的表观刻板性来自 channel identity bias。但这不等于传播结构不存在，因为 cluster-aware 后的 centered 结果仍保留了真实信号。

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

- 多模态是普遍现象，不是例外
- `958` 的 forward/reverse 双模式可复现，不是孤例
- legacy MI 全部显著，老论文最硬的结果站得住
- 真正可信的定量指标应该是 cluster-aware `τ` 与 raw/centered 并列报告，而不是只给一个整体 MI

### 4.2 同步性

- 主结论是 population-level null
- `phase` 是主指标，`legacy` 仅作兼容，`span` 仅作附录
- `phase_e` 的 `pre > post` 是唯一需要继续追的分层信号

---

## 5. 仍未解决的问题 / 风险点

- SOZ > non-SOZ 的传播优势仍偏弱，当前更像探索性趋势，不该写成定论。
- centered rank 可能过度校正；虽然当前 `soz_source_erased` 仅 `3/30`，但今后仍必须和 raw 结果并列报告。
- synchrony 线最大的风险不是假阳性，而是“把 null 写得太花”。现在最诚实的说法就是：**总体 null，局部 extra-focal 线索待验证。**
- propagation 与 synchrony 都是 topic 1，但它们不是同一个统计对象，文档里必须并列而不能混写成一个指标体系。

---

## 6. 代码与结果入口

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

## 7. 与其他 topic 的边界

- 如果问题在问“`~2 Hz` 峰是不是真的”或“IEI serial correlation 说明什么”，跳到 `docs/topic2_between_event_dynamics.md`
- 如果问题在问“SOZ 和 non-SOZ 到底差在哪里”，跳到 `docs/topic3_spatial_soz_modulation.md`
- 如果问题同时涉及“传播是否真实”和“慢调制是否发生在 SOZ”，先分别读 topic 1 和 topic 3，不要混成一个问题

---

## 8. 历史文档索引

- `docs/archive/topic1/interictal_group_event_internal_propagation.md`
  - 这份是内部传播线的详细结果与合同文档
- `docs/archive/topic1/interictal_synchrony_preliminary_report_2026-04-03.md`
  - 这份是 PR4–PR6 的统计报告

这两份文档保留为历史事实来源；当前正式口径以本文件为准。
