# 2026-09-16 近期结果与远端同步

本次范围为上次同步后的9月11–16日，以文件实际修改时间和结果包日期定位。原工作树、暂存区、冲突、删除记录及运行进程均保留；没有重跑实验或更换正式Figure5。代码按源工作树发布为独立快照分支，main收录本索引和可阅读结果包，不将未经整体验证的探索代码直接覆盖main实现。

## 先看这四条线

| 研究线 | 本次保存的结果边界 | 归档入口 |
|---|---|---|
| 几何、核向外EE与全局方向的传播拟合 | 现有最终报告记录72/72可评分槽位及8/8新拓扑确认；部分分布改善，但TB时差和TA局部顺序仍有缺口，尚不接受患者双模式恢复或Fig5基底 | [最终优化报告](../../../../results/recent_sync/2026-09-16/data/geometry_ee_axis_three_observable_optimization_20260914/overnight_20260914/final_optimization_report.pdf) |
| 无标签事件分布拟合 | 归档时status仍为RUNNING；保存的是某一时刻状态，不按旧报告推断最终完成 | [状态快照](../../../../results/recent_sync/2026-09-16/data/label_free_dense_parameter_search_20260915/status.json) |
| Core网络分岔v7 | 保存20条件原生SNN图谱、周期折点和倍周期等数值结果；报告明确原生SNN未复现简化模型相同阈值突跳，全局分支连接尚未穷尽 | [科学报告](../../../../results/recent_sync/2026-09-16/local/core_network_bifurcation_v7_20260916/scientific_report.md) · [Core A](../../../../results/recent_sync/2026-09-16/local/core_network_bifurcation_v7_20260916/figures/00_core_A_bifurcation.png) · [Core B](../../../../results/recent_sync/2026-09-16/local/core_network_bifurcation_v7_20260916/figures/00_core_B_bifurcation.png) |
| k100×恢复时间尺度的复发探索 | 现有报告未找到自主闭环；外部干预分支与自主轨迹分开，部分矩阵条件观察时间截尾 | [科学审阅](../../../../results/recent_sync/2026-09-16/data/fig5_interictal_recurrence_k100_matrix_20260916/scientific_review.md) · [矩阵表](../../../../results/recent_sync/2026-09-16/data/fig5_interictal_recurrence_k100_matrix_20260916/matrix_summary.md) |

以上为已有报告和数据状态的整理，不构成本次独立复验或新的科学接受。原图说明与既有验收状态随文件保存；候选图仍待用户目视审阅。复制的历史文档可能含绝对路径和指向大型原始数组的链接；阅读用文件见下表，未复制数据仍在原路径。

## 源码与复现边界

- `codex/recent-results-20260916-root`：`9a0d0e8a64d1013a5fd9ae04a051a4f74e6ea6a4`，404个近期或本地导入依赖文件；父提交`bcafd00088429fa11313ca030f67fedc8da73bd7`。
- `codex/recent-results-20260916-continuous-core-state-r1`：`b3bd36da0bb81ac26d036f2dc16e60748abff8cc`，135个近期或本地导入依赖文件；父提交`f01db07cdb1dbb28c2695f54ce9742915747e093`。

源码快照保留各自原分支的模型背景，并含逐文件哈希和来源。Python静态导入依赖已追踪；动态路径、共享工作树、患者数据与大数组继续依赖现有环境，不声称脱离原数据即可完整重演。源工作树中4个既有冲突未纳入覆盖，删除与9月11日前无关积压未顺带提交。近期模型合同和进度文档同时保存在source_context/。

## 文件与验证

本次保存1977个结果文件，合计191.91 MiB。逐文件稳定读取、SHA256、JSON解析、PNG/GIF容器校验、PDF头和SVG XML检查均通过；源码Python语法解析通过。未把这些检查称为数值实验重新通过。

[完整结果映射](artifact_manifest.json)、[源码分支及文件哈希](source_branches.json)、[检查记录](validation.json)、[工作树盘点](worktree_inventory.json)。

Git同步范围是报告、顶层/分析汇总表、关键PNG/PDF/GIF、QA元数据及对应源码快照。完整worker数组、模型checkpoint、环境、逐条件大数据和历史快照不进入本次Git提交，保留源盘；这不是整块数据盘备份。每个文件的采集UTC时间、原路径和哈希均在清单中，运行中结果不宣称跨文件原子快照。

## 各结果包

| 来源与包名 | 已同步文件 | 阅读入口 |
|---|---:|---|
| data/core_multiseed_response_curves_20260913 | 61 | [查看](../../../../results/recent_sync/2026-09-16/data/core_multiseed_response_curves_20260913/status.json) |
| data/core_propagation_recovery_20260911 | 114 | [查看](../../../../results/recent_sync/2026-09-16/data/core_propagation_recovery_20260911/scientific_review.md) |
| data/core_recruitment_tradeoff_followup_20260912 | 64 | [查看](../../../../results/recent_sync/2026-09-16/data/core_recruitment_tradeoff_followup_20260912/status.json) |
| data/core_response_review_20260913 | 26 | [查看](../../../../results/recent_sync/2026-09-16/data/core_response_review_20260913/scientific_report.md) |
| data/core_shape_output_response_20260911 | 94 | [查看](../../../../results/recent_sync/2026-09-16/data/core_shape_output_response_20260911/status.json) |
| data/current_best_inline_review_20260913 | 17 | [查看](../../../../results/recent_sync/2026-09-16/data/current_best_inline_review_20260913/artifact_qa.json) |
| data/fig5_interictal_recurrence_20260915 | 32 | [查看](../../../../results/recent_sync/2026-09-16/data/fig5_interictal_recurrence_20260915/scientific_review.md) |
| data/fig5_interictal_recurrence_k100_matrix_20260916 | 101 | [查看](../../../../results/recent_sync/2026-09-16/data/fig5_interictal_recurrence_k100_matrix_20260916/scientific_review.md) |
| data/fig5_log_m_entry_extension_20260915 | 12 | [查看](../../../../results/recent_sync/2026-09-16/data/fig5_log_m_entry_extension_20260915/README.md) |
| data/fig5_m_overnight_exploration_20260913 | 81 | [查看](../../../../results/recent_sync/2026-09-16/data/fig5_m_overnight_exploration_20260913/status.json) |
| data/geometry_ee_axis_three_observable_optimization_20260914 | 70 | [查看](../../../../results/recent_sync/2026-09-16/data/geometry_ee_axis_three_observable_optimization_20260914/README.md) |
| data/global_axis_residual_probe_20260913 | 40 | [查看](../../../../results/recent_sync/2026-09-16/data/global_axis_residual_probe_20260913/status.json) |
| data/label_free_dense_parameter_search_20260915 | 33 | [查看](../../../../results/recent_sync/2026-09-16/data/label_free_dense_parameter_search_20260915/README.md) |
| data/label_free_objective_audit_20260915 | 3 | [查看](../../../../results/recent_sync/2026-09-16/data/label_free_objective_audit_20260915/README.md) |
| data/local_geometry_ee_axis_reference_20260914 | 6 | [查看](../../../../results/recent_sync/2026-09-16/data/local_geometry_ee_axis_reference_20260914/README.md) |
| data/m_parameter_modes_fig5_20260913 | 61 | [查看](../../../../results/recent_sync/2026-09-16/data/m_parameter_modes_fig5_20260913/scientific_review.md) |
| data/overnight_exploration_20260913 | 99 | [查看](../../../../results/recent_sync/2026-09-16/data/overnight_exploration_20260913/README.md) |
| data/parameter_window_response_update_20260914 | 101 | [查看](../../../../results/recent_sync/2026-09-16/data/parameter_window_response_update_20260914/artifact_qa.json) |
| data/scl_internal_template_review_20260914 | 19 | [查看](../../../../results/recent_sync/2026-09-16/data/scl_internal_template_review_20260914/adjacent_covariances.csv) |
| local/autonomous_recovery_exploration_20260914 | 78 | [查看](../../../../results/recent_sync/2026-09-16/local/autonomous_recovery_exploration_20260914/README.md) |
| local/burst_regime_map_20260914 | 33 | [查看](../../../../results/recent_sync/2026-09-16/local/burst_regime_map_20260914/scientific_report.md) |
| local/core_branch_connections_v5_20260915 | 35 | [查看](../../../../results/recent_sync/2026-09-16/local/core_branch_connections_v5_20260915/scientific_report.md) |
| local/core_burst_bifurcation_states_v3_20260915 | 43 | [查看](../../../../results/recent_sync/2026-09-16/local/core_burst_bifurcation_states_v3_20260915/scientific_report.md) |
| local/core_burst_bifurcation_v2_20260915 | 35 | [查看](../../../../results/recent_sync/2026-09-16/local/core_burst_bifurcation_v2_20260915/scientific_report.md) |
| local/core_burst_onset_brunel_v1_20260915 | 73 | [查看](../../../../results/recent_sync/2026-09-16/local/core_burst_onset_brunel_v1_20260915/scientific_report.md) |
| local/core_burst_right_branch_v4_20260915 | 29 | [查看](../../../../results/recent_sync/2026-09-16/local/core_burst_right_branch_v4_20260915/scientific_report.md) |
| local/core_network_bifurcation_v7_20260916 | 226 | [查看](../../../../results/recent_sync/2026-09-16/local/core_network_bifurcation_v7_20260916/scientific_report.md) |
| local/core_observable_bifurcation_v6_20260915 | 41 | [查看](../../../../results/recent_sync/2026-09-16/local/core_observable_bifurcation_v6_20260915/scientific_report.md) |
| local/fig5_fixed_zm_termination_20260915 | 48 | [查看](../../../../results/recent_sync/2026-09-16/local/fig5_fixed_zm_termination_20260915/scientific_review.md) |
| local/fig5_interictal_recurrence_20260915 | 1 | [查看](../../../../results/recent_sync/2026-09-16/local/fig5_interictal_recurrence_20260915/README.md) |
| local/fig5_log_m_kinetics_20260914 | 11 | [查看](../../../../results/recent_sync/2026-09-16/local/fig5_log_m_kinetics_20260914/README.md) |
| local/fig5_preentry_event_audit_20260914 | 29 | [查看](../../../../results/recent_sync/2026-09-16/local/fig5_preentry_event_audit_20260914/README.md) |
| local/fig5_resting_recovery_layout_20260914 | 5 | [查看](../../../../results/recent_sync/2026-09-16/local/fig5_resting_recovery_layout_20260914/M_parameter_first_entry_complete.json) |
| local/fig5_single_transition_20260914 | 9 | [查看](../../../../results/recent_sync/2026-09-16/local/fig5_single_transition_20260914/README.md) |
| local/fig5_z_bifurcation_preview_20260915 | 45 | [查看](../../../../results/recent_sync/2026-09-16/local/fig5_z_bifurcation_preview_20260915/scientific_report.md) |
| local/fig5_z_branch_extension_20260915 | 52 | [查看](../../../../results/recent_sync/2026-09-16/local/fig5_z_branch_extension_20260915/scientific_report.md) |
| local/liou_original_design_20260915 | 45 | [查看](../../../../results/recent_sync/2026-09-16/local/liou_original_design_20260915/scientific_review.md) |
| local/liou_spatial_feedback_20260915 | 13 | [查看](../../../../results/recent_sync/2026-09-16/local/liou_spatial_feedback_20260915/scientific_review.md) |
| local/m_on_z_kinetics_20260912 | 13 | [查看](../../../../results/recent_sync/2026-09-16/local/m_on_z_kinetics_20260912/scientific_review.md) |
| local/model_collaborator_report_v8_2026-09-11 | 31 | [查看](../../../../results/recent_sync/2026-09-16/local/model_collaborator_report_v8_2026-09-11/current.json) |
| local/overnight_key_results_20260911 | 5 | [查看](../../../../results/recent_sync/2026-09-16/local/overnight_key_results_20260911/current.json) |
| local/reset_state_diagnosis_20260911 | 15 | [查看](../../../../results/recent_sync/2026-09-16/local/reset_state_diagnosis_20260911/status.json) |
| local/weaker_M_onset_paired_20260914 | 28 | [查看](../../../../results/recent_sync/2026-09-16/local/weaker_M_onset_paired_20260914/scientific_review.md) |
