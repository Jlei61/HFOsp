# Figure 5 本版来源

本版 F 唯一绘图输入为 `interim_core_count_summary.json`，经 `cohort_selection.json` 的固定 17 人名单筛选后写入 `displayed_core_count_summary.json`；`interim_selection.json` 为本版约束选点，`ta_preferred_selection_audit.json` 逐患者记录按 J 检查至首个合格候选的过程。新工作点的事件表与模型/患者模板在 summary 的 event_export 指向目录。

`unconstrained_core_count_summary.json`、`unconstrained_interim_selection.json`、`interim_subjects.csv` 为先前无约束快照，`joint_matching_reference_statistics.json` 为历史 Stronger/Weaker 诊断，均不参与本版 F 的画点或统计。当前画点 CSV 在上一级 `core_count_subjects.csv`。其余冻结示例资产服务 A–E，旧 fig4-paneli/template_recovery_summary 仅为历史来源。
