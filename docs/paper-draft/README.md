# Paper draft 工作区

本目录用于整理进入论文的正式口径。这里的文字是写作母稿，不替代各 Topic 的方法合同、结果 artifact 或审计记录。

## Methods

- [Methods 中文当前修订稿](methods_revised_draft.md)：当前主要修改入口；按确认口径逐轮吸收修改，未处理的审阅注继续保留。
- [Methods 中文原文与段后审阅](methods_original_annotated.md)：上一轮逐段审阅，完整保留作者原文。
- [Methods 纯原文](methods_original_verbatim.md)：两份附件原文的完整合并副本，不含任何改写或审阅。
- [D_AB 三维梯度轴替换稿](methods_axis_gradient_rewrite.md)：新的无端点、无 k 共同病理传播轴方法。
- [上一版压缩工作母稿](methods_working_draft.md)：保留作历史对照，不再作为当前主稿。
- [Methods 待确认问题](methods_open_questions.md)：按阻断程度整理伦理、队列、统计定义和 SNN 分层问题。

## Core 1 与队列表

- [Core 1：间期 HFO 传播骨架](core1_interictal_hfo_propagation_scaffold.md)
- [Figure 1：间期 HFO 时序组织](figure1_interictal_hfo_temporal_scaffold.md)
- [Cohort contract 与 Supplementary Tables](cohort_contract_and_supplementary_tables.md)

## 计算模型

- [Static-anchored HistoryRNN field refinement v0.4 正式结果](../archive/topic5/history_conditioned_field_refinement_v0_4_result_2026-08-03.md)：冻结论文已有 A/B 静态间期场，使用相同 sign-free maxAB 训练 causal-history residual。15 人/31 次发作中，静态 M0 保持超过 matched all-contact null，但联合 RNN M3 未改善 M0，也不优于冻结 state 或非递归时间汇总；真实顺序和正确 seizure-history 配对均无优势。当前定位为 Supplementary bounded result：静态跨状态 scaffold 保留，逐发作 history refinement 未建立。
- [HistoryRNN next-event proxy 的暂定 bounded-negative](figure6_history_rnn_early_ictal_field_bounded_negative.md)：3 seeds × 34 LOSO 完成，真实时间 chronological branch 未超过匹配无序历史；该结果只约束 next-event proxy。
- [HistoryRNN direct early-ictal transfer v0.2 最终结果](../archive/topic5/history_rnn_direct_early_ictal_transfer_v0_2_result_2026-08-02.md)：15 名 primary 患者的 causal history state → early-ictal field frozen transfer。按合同实现的全前缀顺序打乱下，真实事件顺序在 c10 与 c30 两个预算下都没有优势；c10 的 RNN 相对增量与 zero-state 敏感性未在 c30 复现；两种预算均未超过 absolute channel-shuffle，也未通过正确/错误 seizure pairing。场只在每患者 6–16 个骨架触点上评分。最终定位为 Supplementary training-sensitive boundary，不支持稳定的 seizure-specific latent state 或因果“塑造”。
- [Figure 6：structured path-mode RNN 的正式 34 人阴性边界](figure6_persistent_path_mode_rnn_bounded_negative.md)：510 个 LOSO runs 已完成；局部 next-set NLL 可学，但自由生成触点分布与结构必要性均未过门，clinical-onset target 保持封存。

## 使用规则

1. 主文方法必须能回溯到当前 producer、配置和结果 artifact。
2. 新定义若尚未重跑，只能写为待选方案，不能写成已经执行的方法。
3. TBC 清零前，不把本目录文字视为投稿终稿。
4. 慢变量 SNN 与当前主模型分开写；阴性或边界性探索不升级为已验证机制。
