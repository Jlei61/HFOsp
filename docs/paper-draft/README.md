# Paper draft 工作区

本目录用于整理进入论文的正式口径。这里的文字是写作母稿，不替代各 Topic 的方法合同、结果 artifact 或审计记录。

## Introduction

- [Introduction 中文当前修订稿](introduction_revised_draft.md)：当前论证与 claim boundary 已对齐的中文母稿；引用键待最终 bibliography 核对。

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
- [Supplementary Figures S1–S6 中文 Overleaf 图注](supplementary_figure_legends_s1_s6_zh.tex)：按 `FigS1`–`FigS6` 编号整理的完整 LaTeX figure blocks；S1 明确为 legacy 人工标注事件验证集。
- [Supplementary Figures S2–S6 英文图注](supplementary_figure_legends_s2_s6.md)：按 Nature Computational Science 投稿语法整理的可直接粘贴版本；逐 panel 定义样本单位、图形元素、统计检验和缩写。
- [Cohort contract 与 Supplementary Tables](cohort_contract_and_supplementary_tables.md)

## 计算模型

- [Figure 4：被试特异性 SNN](figure4_subject_specific_snn.md)
- [当前 Supplementary Figure 6：static contact topography 与 bounded-history audit](figure6_static_contact_topography_bounded_result.md)：当前唯一 manuscript-facing 版本。间期 participation scaffold 稳定，真实顺序相对 rank-shuffle 有 heldout 增益且有效历史集中于最近 2–3 个 rank set；early-ictal energy 只支持 reused-target 的 orientation-free spatial correspondence，fixed positive direction、unbounded-history necessity 和 GRU-specific static increment 均未建立。
- [RNN 整体综合验收](../archive/topic5/rnn_overall_integrated_acceptance_2026-07-28.md)：full-rank、low-rank、path-mode、axis/competition/source、internal-state、static transfer 与 H1/H2/H3 的统一 claim hierarchy；该文档是所有 RNN 分支的最终总入口。
- [历史 Figure 6：structured path-mode RNN 的正式 34 人阴性边界](figure6_persistent_path_mode_rnn_bounded_negative.md)：510 个 LOSO runs 已完成；局部 next-set NLL 可学，但自由生成触点分布与结构必要性均未过门。保留作模型谱系，不再作为当前 Figure 6 文稿。
- [历史计算补充：symmetric-axis propagation-state RNN v2.2.1 的阴性边界](figure6_symmetric_axis_propagation_state_bounded_negative.md)：Markov 在 22 人中保留稳定的一阶 transition information，但 full/isotropic 线性传播模型均低于 node-bias；保留作模型谱系。
- [计算补充：interictal transition signal decomposition](interictal_transition_signal_decomposition.md)：Markov 信号主要为 symmetric、跨局部几何且含 ordered multi-step history；axis residual可检出但符号混合，source-conditioned增益很小。只允许设计 v2.3，不开放发作期 target。
- [历史 Figure 6 候选：competitive propagation RNN 的正式边界](figure6_competitive_propagation_rnn_bounded_result.md)：categorical next-contact 任务证明 contact-rank sequence 可预测且依赖历史，但 delayed competition、physical axis 与 source-conditioned direction 均未过门；不再作为当前 Figure 6 文稿。

## 使用规则

1. 主文方法必须能回溯到当前 producer、配置和结果 artifact。
2. 新定义若尚未重跑，只能写为待选方案，不能写成已经执行的方法。
3. TBC 清零前，不把本目录文字视为投稿终稿。
4. 慢变量 SNN 与当前主模型分开写；阴性或边界性探索不升级为已验证机制。
