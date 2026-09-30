# Fig5边界附近的3000秒单种子实验

固定seed9108401，共19条（8条从1000秒续跑，11个新参数点）。已完成19/19，进入7条，其中1000秒后进入1条，3000秒未进入12条。

[实时状态](status.json) · [参数清单](parameter_points.csv) · [执行方案](execution_plan.md) · [连续边界候选图](figures/long_boundary.png) · [逐点结果](first_entry_points.csv) · [检查点核对](continuation_qa.json)

原59点1000秒paper-ready快照保留；本目录每10分钟或新终点刷新候选。实线表示1000秒边界，紫色虚线仅连接具有双侧合格支持的3000秒边界；colorbar保持原1–1000秒，晚进入另行标数值。19条完成后停止，不增加种子、参数或观察窗。候选待人工检查。

## 2026-09-28并入Fig5

本批19/19条已全部完成并从原始计数复核。更新后的[paper-ready Fig5](../../paper-ready-figure/fig5/README.md)纳入70个唯一参数组合，按照作者最新要求仅显示连续色面、保留原log色条，不叠加采样点或边界；本目录原long_boundary图保留为历史分析图。真实晚进入时间与不同随访终点见[长时补充结论](../../paper-ready-figure/fig5/long_followup_summary.md)。
