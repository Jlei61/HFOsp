# H2b v0.3 少 gate、多探索执行附录

> **状态：SUPERSEDED。** 当前用户指定的 H2b 验收意见恢复严格串行
> gate：A1 没有足够 `state_qualified` checkpoint 时立即停止 downstream；
> A2 power 不足时不得解释真实阴性。本文仅保留为历史执行记录，不再授权
> `all_frozen` hazard、geometry 或 phenotype override。

## 决定

本附录响应 2026-08-31 的用户指令，优先于原合同中“某一科学层阴性就停止全部 downstream”的执行措辞，但不修改冻结数据、估计量、30 min 主 endpoint、因果时间边界或 claim ladder。

只保留五类硬门：上游 source purity、嵌套 estimand、因果时间/分区、原子可复现产物、资源安全。它们失败会让数字本身无效，因此仍必须停止并修复。

A1 state qualification、A2 assay power、T、M、D、IED-source ablation 与 phenotype 不再是整个 v0.3 的串行总 gate。它们分别控制对应措辞：

- A1 不通过：不能称 persistent state，但仍运行 `all_frozen` 的 hazard/lag/geometry 诊断；
- A2 power 不足：不能把真实阴性写成生物学阴性，但仍可运行并公开探索结果；
- T 不通过：不能称 transferable representation，但不阻断 M/D、source ablation 或 phenotype 的预注册诊断；
- M 不通过：只撤回 persistent-memory 主张；
- D 不通过：只撤回 seizure-entry specificity；
- source ablation 和 phenotype 可在 development 中探索，只有相应上游证据齐全时才升级措辞。

所有探索必须同时报告 `all_frozen` 与可用的 `state_qualified` 层，保留患者和 OOF lead seizure 的真实分母。阴性结果不删除、不成为全项目 blocker；工程 PASS 也不升级成科学支持。

机器附录：`config/topic5_continuous_marked_state_h2b_v0_3_exploration_policy.json`，冻结后写入 v0.3 结果根的 `exploration_policy.json`。
