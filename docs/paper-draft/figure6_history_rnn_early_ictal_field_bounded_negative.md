# HistoryRNN next-event proxy：provisional bounded-negative 写作母稿

> 当前状态：Supplementary candidate；本页只记录 v0.1 target-blind next-event proxy。独立 v0.2 已直接运行 early-ictal transfer 并得到 bounded negative，见 `docs/archive/topic5/history_rnn_direct_early_ictal_transfer_v0_2_result_2026-08-02.md`。

## 安全结果口径

To test whether interictal events accumulated into a real-time state, we separated within-event rank encoding from an across-event recurrent branch. The event encoder was reset for every population event, whereas the candidate history state was carried within continuous recordings and decayed according to the observed inter-event intervals. In a target-blind next-event proxy task, the chronological model was compared with a parameter-matched nonrecurrent model that received the same static contact prior, event embeddings, unordered history summary, last event, event count and elapsed-time covariates. Across 31 development-excluded patients and three frozen seeds, chronological history did not improve held-out next-event contact-field prediction (median difference in participation BCE, −0.00019 nats per contact-decision; 15/31 patients positive; one-sided Wilcoxon p=0.101), and the direction was inconsistent across datasets and seeds. The recurrent state was nevertheless order-dependent: reordering the 64 most recent preceding events within the identical causal prefix, while preserving the event set, the timestamp slots and the last event, reliably worsened the chronological model itself (median cost 0.00016 nats; 26/31 patients positive; p=3.9×10⁻⁵; consistent in both datasets and all three seeds). Because that perturbation also presents the recurrent branch with input statistics it was never trained on, order sensitivity alone does not establish that event order carries predictive information; the direct test of that claim is the comparison with the order-blind matched model, which was null. Thus, under the current next-event objective, the recurrent branch did not identify a reproducible chronology-specific increment beyond static and unordered history information. Early-ictal values were not evaluated in v0.1, so these results neither support nor refute direct history-to-early-ictal transfer; that question is addressed under a separate v0.2 contract.

## 图注母稿

**Target-blind next-event evaluation of an inter-event history model. A,** Each interictal population event was encoded by an EventRNN that reset at event onset. A separate candidate HistoryRNN state was carried across events within continuous recording segments and decayed according to the observed inter-event interval. The early-ictal endpoint was not evaluated in this v0.1 proxy analysis. **B,** Patient-level held-out participation BCE difference between the parameter-matched unordered model (M1) and chronological HistoryRNN (M2), after median aggregation across three frozen seeds. Positive values favor M2. The primary development-excluded cohort did not show a chronological-state increment (31 patients; median −0.00019; 15/31 positive; one-sided Wilcoxon p=0.101). **C,** Cost of shuffling the 64 most recent preceding events within the same causal prefix while preserving the event set, timestamp slots and last event. This strict perturbation reliably worsened the chronological model (median 0.00016; 26/31 positive; p=3.9×10⁻⁵), showing that the fitted state is order-dependent; panels B and C answer different questions, and C does not imply that order carries information the unordered model lacks. Red and blue denote Epilepsiae and Yuquan patients, respectively; horizontal bars show dataset medians. Direct early-ictal transfer was subsequently evaluated under v0.2 rather than being inferred from this proxy.

## 红线

- 不写 `HistoryRNN predicted the early-ictal field`；
- 不写 `early-ictal prediction failed`；
- 不把 G2/G3 写成 fail；
- 不把 learned 2 h persistence 写成 biological time constant；
- 不以 seed1 或单患者轨迹支撑跨事件 state；
- 不把顺序置换显著写成 `event order carries predictive information`（它只说明 state 对顺序敏感；直接检验是 B 图的 `M2−M1`，为零）；
- 不把 `static → matched` 的 +0.0035 写成 `recent events carry information`（同一臂还多了一个可学习的 contact 读出，本轮没有把两者分开的对照）；
- 不把 static scaffold 和 within-event short-range order 的既有结果一并否定。

完整证据与数值见 `docs/archive/topic5/history_rnn_early_ictal_field_v0_1_closeout_2026-08-02.md`。
