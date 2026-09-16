# 连续 core 状态 R1

实际进度：[status.json](status.json)。各运行的 `.progress.json` 给出已模拟时长；`logs/` 保留执行日志。

冻结方案：[execution_plan.md](execution_plan.md)；配置：[design.json](design.json)。完整物理资格：[qualification.json](qualification.json)。分析链使用历史包络的独立测试：[analysis_integration_check.json](analysis_integration_check.json)；测试图和虚拟状态不属于本轮实验。

代码版本位于 `/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-continuous-core-state-r1`，branch `codex/topic4-continuous-core-state-r1`。基底为历史手放位置在当前 v2.1 引擎的实现，未采用 old_joint 或先前最佳评分候选。

队列自动执行 18 个单元，最多 4 个同时运行；完成后自动调用 `scripts/analyze_topic4_continuous_core_state_r1.py`，生成 `scientific_report.md`、逐事件/逐运行 CSV 和诊断图。`ROUND_COMPLETE_PENDING_SCIENTIFIC_REVIEW` 表示运行及自动汇总完成，不代表患者传播已通过科学验收。控制器仍在运行时不要重复派发。

失败项与缺失模式会保留。若工程失败，先修复其原因再续跑；已有完成项仅在配置和数组核对一致时复用。不得根据第一轮曲线自动调剂量、换基底或增加重复。

代表性图与科学解读：[representative_review/scientific_note.md](representative_review/scientific_note.md)。包含同网络四列图、原生事件帧及全部 OU 事件的患者 FIT 对照。

全局状态改名与下一版时间建模：[患者事件序列约束的漂移—扩散方案](/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-continuous-core-state-r1/docs/archive/topic4/sef_hfo/global_state_s_drift_diffusion_design_2026-09-09.md)。状态为 DESIGN_READY_NOT_FITTED；本次图面统一使用 s，未改变 R1 参数。
