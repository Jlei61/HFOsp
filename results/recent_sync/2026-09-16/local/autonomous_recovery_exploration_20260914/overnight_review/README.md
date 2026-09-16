> **2026-09-15：本路线已撤回为历史探索。** 新增10秒全局池及关联rho候选不再作为当前模型；原始证据保留，详见[撤回说明](../WITHDRAWAL.md)。

# 夜间探索结果入口

三套完整Fig5来自不同参数的实际连续SNN轨迹；均无外部reset。完整短图不能替代后续60秒审阅，正式验收仍待用户目视。

- [原生Z，κ200/τG10：短程高—安静—高 · PNG](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/autonomous_recovery_exploration_20260914/native_field_candidates_recurrence/resource_rho0_k200_tau10_s9108401/full_fig5/figures/fig5_autonomous_recurrence.png)
- [原生Z，κ200/τG10：短程高—安静—高 · PDF](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/autonomous_recovery_exploration_20260914/native_field_candidates_recurrence/resource_rho0_k200_tau10_s9108401/full_fig5/figures/fig5_autonomous_recurrence.pdf)
- [rho=.25，κ200/τG10：高态后反复广泛短爆发 · PNG](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/autonomous_recovery_exploration_20260914/native_field_candidates_recurrence/resource_rho0.25_k200_tau10_s9108401/full_fig5/figures/fig5_autonomous_recurrence.png)
- [rho=.25，κ200/τG10：高态后反复广泛短爆发 · PDF](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/autonomous_recovery_exploration_20260914/native_field_candidates_recurrence/resource_rho0.25_k200_tau10_s9108401/full_fig5/figures/fig5_autonomous_recurrence.pdf)
- [rho=.25，κ50/τG10：首噪声安静后再入 · PNG](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/autonomous_recovery_exploration_20260914/native_field_candidates_recurrence/resource_rho0.25_k50_tau10_s9108401/full_fig5/figures/fig5_autonomous_recurrence.png)
- [rho=.25，κ50/τG10：首噪声安静后再入 · PDF](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/autonomous_recovery_exploration_20260914/native_field_candidates_recurrence/resource_rho0.25_k50_tau10_s9108401/full_fig5/figures/fig5_autonomous_recurrence.pdf)

- [科学审阅](scientific_review.md)
- [逐轨迹原始计数审计](trajectory_ledger.csv)
- [完整定义和端点JSON](trajectory_ledger.json)

逐轨迹表为最终版：48/48端点和原始计数核查通过、0运行失败。46个参数/噪声条件与2条续跑分开标记；门曾通过与末段是否仍能返回分别审阅。

模型原生动画（用于核查真实空间招募，不是患者TA/TB验证）：
- [autonomous_quiet_reentry](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/autonomous_recovery_exploration_20260914/preserved_global_gain_round7/runs/resource_rho0.25_k50_tau10_s9108401/native_movies/autonomous_quiet_reentry.gif)
- [post_high_finite_bursts](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/autonomous_recovery_exploration_20260914/preserved_global_gain_round7/runs/resource_rho0.25_k200_tau10_s9108401/native_movies/post_high_finite_bursts.gif)

此前独立的宽范围log-M时延扫描已经完成，其图保持原Z基底，不能挪作新增全局/rho模型的参数图：
- [原版Fig5，噪声9108401](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/fig5_preentry_event_audit_20260914/clean_panels_v2/eta0.0005_s9108401/figures/fig5.png)
- [原版Fig5，噪声9108402](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/fig5_preentry_event_audit_20260914/clean_panels_v2/eta0.0005_s9108402/figures/fig5.png)

F主图现在按固定原窗口中较大的有符号对应系数选择模型A/B类别，全部比较保留；选后示例不算独立验证。此前B-only诊断完整保存在每个候选的full_fig5_B_diagnostic目录；不是把负结果删除。

先看长程的正反对照：

- [同状态60秒续跑对照 · PNG](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/autonomous_recovery_exploration_20260914/long_recovery_contrast/figures/long_recovery_contrast.png)
- [同状态60秒续跑对照 · PDF](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/autonomous_recovery_exploration_20260914/long_recovery_contrast/figures/long_recovery_contrast.pdf)
- [后期连续三次全场爆发 · GIF](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/autonomous_recovery_exploration_20260914/revisedZ_continuation_to60/runs/revisedZ_k200_tau10_continuation60_s9108401/native_movies/late_three_global_bursts.gif)

长程左列原生Z最终进入持续高率，右列新增rho保持可终止宽爆发；右列也没有恢复原先的局部间期事件。其后30秒全部11次完整爆发均为350–410ms且全场招募，详见[逐事件复核](late_bout_review.json)。安静期仍驱动Z耗竭的[电流下界诊断](quiet_resource_threshold_bound.json)只解释已保存轨迹，不是提高阈值后的仿真结果。
