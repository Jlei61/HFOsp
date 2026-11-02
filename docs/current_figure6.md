# 当前 Figure 6：原 Figure 5 的 Z/M 图

2026-10-24 按作者要求将旧 Figure 5 登记为 **Figure 6**。原数值、图层和验收状态保持；主入口仍是旧顶层空间 Z/M 图，10 月 11 日两列重排仍为待目视检查的候选。

- [完整 PDF](../results/paper-ready-figure/fig6/figures/fig6-complete-layout.pdf) · [PNG](../results/paper-ready-figure/fig6/figures/fig6-complete-layout.png)
- [10 月 11 日重排候选](../results/paper-ready-figure/fig6/candidates/reference_layout_20261011/figures/fig6-complete-layout.pdf)
- [版本入口](../results/paper-ready-figure/fig6/current_version.json) · [迁移前说明](../results/paper-ready-figure/fig6/source/renumber_20261024/previous_current_figure5.md)

原包整体迁入 `fig6/`，历史文件中的 Fig5 编号只指迁移前身份。冻结正文未修改。新 Figure 5 专用于原 Fig4 拆出的传播复现与单/双 core 对照。`build_fig6_current.py` 校验或恢复已登记导出，不重跑仿真。

## 2026-11-02 周报 Figure 6：G 与 M 共享 logistic

当前讨论及[周报](weekly_reports/2026-11-02_epilepsy_model_figures/2026-11-02_癫痫文章_Figure4-6周报.md)使用新候选 `shared_logistic_feedback_20261102`。A–D 已替换为同一网络、同一噪声种子下重新运行的 120 s 原生仿真，**全局抑制 G 和适应增量 M 中的两处 clip 均已替换**：

$$
S(R;R_0,k)=\frac{1}{1+\exp[-(R-R_0)/k]},\qquad
\tau_G\dot G=G_{\max}S-G,
$$

$$
\dot M_i=-\frac{M_i}{\tau_M(R)}+
\frac{a_0/\tau_M(R)+a_1S}{\eta_M}\sum_n\delta(t-t_i^n).
$$

$R_0$ 为半最大响应发放率，$k>0$ 为平滑宽度；公式保留通用可调参数。适应状态仍记为 $M$，有效适应强度记为 $\eta_M M$。当前图的具体配置为 $R_0=350$ Hz、$k=40$ Hz；网络、噪声种子、G/M 幅度及时间常数、Z 负荷路由均与 clip 对照保持。原有 Z 阈值与低活动适应保留规则不变。

新轨迹在 22.04 s 进入持续高活动，27.09 s 退出，34.40 s 两处 core 的 Z 恢复参考值；之后有 48 次短事件覆盖 14.54 s，71.81 s 再次进入。这里的退出采用原 observer 的两段 1 s 低活动判据，与首次完全静默不同。另一个较窄的 k=25 Hz 也能产生完整恢复循环，但最初退出后先快速反跳；k=50/75 Hz 在 60 s 筛查内没有进入。所有宽度及自适应选择过程均保留在[原生验证报告](../results/topic4_sef_hfo/logistic_feedback_20261102/README.md)。这是单网络、单噪声的存在性验证，尚未检验跨网络或跨噪声的稳健性。

- [新候选 PNG](../results/paper-ready-figure/fig6/candidates/shared_logistic_feedback_20261102/figures/fig6-complete-layout.png) · [PDF](../results/paper-ready-figure/fig6/candidates/shared_logistic_feedback_20261102/figures/fig6-complete-layout.pdf)
- [图注及来源](../results/paper-ready-figure/fig6/candidates/shared_logistic_feedback_20261102/README.md) · [当前公式与参数](../results/topic4_sef_hfo/logistic_feedback_20261102/effective_protocol.json)
- 原生数据：`/data/hfosp/topic4_sef_hfo/logistic_feedback_20261102/runs/logistic_r350_k40_s9108404`；producer：`scripts/run_topic4_logistic_feedback.py`；重绘：`scripts/paper_figures/build_fig6_logistic_feedback.py`。

A/C 的采样时刻、B 的完整 0–120 s 轨迹和 D 的模型场及患者最佳匹配均从新数据重算。E 仍为历史模型的 13 患者、140 次发作队列结果，尚未使用新模型重算。当前候选为 `PENDING_AUTHOR_VISUAL_REVIEW`，正式版本登记与验收状态仍沿用上方入口。

### 替换前的两个版本

此前周报的 `selected_r08_r09_normalized_onsets_20261102` 来自 `adaptation_history_20261031/runs/seed9108404`，两处门控均为 clip。独立的 `unified_feedback_phase_20261030` 仅将 G 的驱动改为 Hill 型 S 形函数，M 增量仍为 clip；这两个版本都不是当前共享 logistic 候选的数据来源。历史冻结代码、图片哈希与实际门控采样核对保留在[反馈函数核对记录](weekly_reports/2026-11-02_epilepsy_model_figures/source/fig6_feedback_function_audit.json)。
