# 当前 Figure 6：原 Figure 5 的 Z/M 图

2026-10-24 按作者要求将旧 Figure 5 登记为 **Figure 6**。原数值、图层和验收状态保持；主入口仍是旧顶层空间 Z/M 图，10 月 11 日两列重排仍为待目视检查的候选。

- [完整 PDF](../results/paper-ready-figure/fig6/figures/fig6-complete-layout.pdf) · [PNG](../results/paper-ready-figure/fig6/figures/fig6-complete-layout.png)
- [10 月 11 日重排候选](../results/paper-ready-figure/fig6/candidates/reference_layout_20261011/figures/fig6-complete-layout.pdf)
- [版本入口](../results/paper-ready-figure/fig6/current_version.json) · [迁移前说明](../results/paper-ready-figure/fig6/source/renumber_20261024/previous_current_figure5.md)

原包整体迁入 `fig6/`，历史文件中的 Fig5 编号只指迁移前身份。冻结正文未修改。新 Figure 5 专用于原 Fig4 拆出的传播复现与单/双 core 对照。`build_fig6_current.py` 校验或恢复已登记导出，不重跑仿真。

## 2026-11-02 周报 Figure 6 与平滑反馈分析

当前讨论及[周报](weekly_reports/2026-11-02_epilepsy_model_figures/2026-11-02_癫痫文章_Figure4-6周报.md)使用 `selected_r08_r09_normalized_onsets_20261102` 候选：A–D 来自 `/data/hfosp/topic4_sef_hfo/adaptation_history_20261031/runs/seed9108404` 的同一 120 s 连续仿真，E 保留历史队列结果。该仿真的全局抑制与适应增量均使用 $q(R)=\mathrm{clip}[(R-200)/300,0,1]$；图件重新排版没有更换仿真来源。

平滑全局反馈在独立的 `unified_feedback_phase_20261030` 分析中实现：$\tau_G\dot G=H(R)-G$，$H(R)=30(R/350)^{14/3}/[1+(R/350)^{14/3}]$。这里只将 $G$ 的驱动改为 Hill 型 S 形函数，$M$ 放电增量中的 $q(R)$ 仍为 clip。该模型已完成独立工作点验证，但未替换上述周报 Figure 6 的数据。模型适应状态仍记为 $M$，有效适应强度记为 $\eta_M M$。

图片来源、冻结代码与保存的 $q(R)$ 采样核对见[反馈函数核对记录](weekly_reports/2026-11-02_epilepsy_model_figures/source/fig6_feedback_function_audit.json)。此处补充候选与分析分支的对应关系；正式版本登记和验收状态沿用上方入口。
