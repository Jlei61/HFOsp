# Topic 5 stateful event-sequence RNN v2.7 验收

## 一句话结论

v2.7 已按 repair-only 合同完整验收：early stopping 不再把 epoch −1 静态初始化当成
trained checkpoint，但 34 人主效应与 v2.6 逐患者完全相同。最终证据仍是“模型确实使用短程
事件历史，但没有超过简单 EWMA，也没有 chronology-specific 证据”，不能写网络塑造。

## 完整性

- validation screen 34/34，epoch-boundary audit 34/34；4 人触发延长预算，2 人验证选择改变；
- formal test 34 人 × 3 seeds = 102 runs，全部 finite，trained checkpoint 均来自训练后 epoch；
- dense、state reset、memory curve、block shuffle、time reversal 全部 34/34；
- H40 34 份产物，其中 32 人 eligible、2 人按合同 ineligible；
- v2.6 config/module/runner 和冻结结果哈希未变化；旧 heldout20 未进入；
- acceptance 状态：`DERIVED_ACCEPTANCE_COMPLETE`。

## 科学结果

误差越低越好，因此模型减对照为负表示模型更好。

1. 相对静态 repertoire，RNN 中位差 −0.0619，25/34 人更好，单侧 Wilcoxon
   `p=0.00385`。支持度越高结果越稳，≥50 formal windows 时中位 −0.0826、17/20、
   `p=0.000293`。
2. 相对固定 EWMA，formal 中位差 −0.0248，18/34，`p=0.0764`；≥20 windows 后
   方向翻为 +0.0378，`p=0.607`。dense test 中位 +0.0294，`p=0.163`。
3. 每场事件清零 state 的中位惩罚 +0.0257，25/34，`p=0.00505`；memory curve 的
   主要差异集中在最近约 1–10 场，至 100 场时中位惩罚为 0。
4. 真实 chronology 没有超过 source-coherent block shuffle（true−null 中位
   +0.0176，预设方向 `p=0.967`）或 source-level time reversal（−0.00882，
   `p=0.513`）。block-shuffle 的反方向名义 `p=0.0348` 只说明两臂不可交换，不能写成
   “真实顺序有害”。
5. H40 相对 EWMA 中位 −0.00298，16/32，`p=0.195`，没有释放更长程优势。

## v2.6→v2.7 的含义

v2.7 与 v2.6 的两项患者级主比较逐位相同，34/34 配对差均为 0。这不是重新使用旧
checkpoint：v2.7 重新完成了验证筛选、102 次正式训练和所有 controls。其含义是，旧 bug
确实会让部分训练过早停在最低预算，但这些 run 的最佳 trained checkpoint 没有因此改变，故
主效应数值不变。

## 允许与禁止表述

允许：

> 同一患者稳定的间期传播 repertoire 存在可由短程事件历史追踪的慢变表达；清零历史状态会
> 降低未来窗口预测，但一个简单的 leaky recency observer 已解释可检测的增量。

禁止：

- 精确事件顺序驱动未来传播；
- 跨 source、跨天长期记忆；
- evolving graph、activity-dependent shaping 或 causal plasticity；
- 将该模型重新解释为事件内 next-rank mechanism。

## 冻结产物

- `results/topic5_stateful_event_sequence_rnn/v2_7/acceptance/ACCEPTANCE_STATE.json`
- `results/topic5_stateful_event_sequence_rnn/v2_7/STATEFUL_TEST_STATE.json`
- `results/topic5_stateful_event_sequence_rnn/v2_7/validation_screen/FROZEN_VALIDATION_STATE.json`
- `results/topic5_stateful_event_sequence_rnn/v2_7/{dense_test_sensitivity,state_reset_ablation,memory_curve,chronology_null,h40_sensitivity}/`

