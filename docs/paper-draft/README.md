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

- [Figure 5 候选：E1146 SNN state-dependent readout](../fig5_snn_state_readout_spec.md)：锁定同一连续轨迹、单次间期事件 order、early-runaway energy、E1146 montage 与 claim boundary。
- [Figure 6：structured path-mode RNN 的正式 34 人阴性边界](figure6_persistent_path_mode_rnn_bounded_negative.md)：510 个 LOSO runs 已完成；局部 next-set NLL 可学，但自由生成触点分布与结构必要性均未过门，clinical-onset target 保持封存。

## 使用规则

1. 主文方法必须能回溯到当前 producer、配置和结果 artifact。
2. 新定义若尚未重跑，只能写为待选方案，不能写成已经执行的方法。
3. TBC 清零前，不把本目录文字视为投稿终稿。
4. 慢变量 SNN 与当前主模型分开写；阴性或边界性探索不升级为已验证机制。
