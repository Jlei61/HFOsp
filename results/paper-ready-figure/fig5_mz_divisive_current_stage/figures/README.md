# MZ-divisive current-stage visual diagnostics
### fig5_candidate_E1146_mz_divisive_failure_summary.png / .pdf

这张图只消费已经完成的 v2/v3 单 seed 轨迹。左图显示高状态选择性的慢除法器把 delayed runaway 改成约 5 Hz 的有限窗 recruited bursting；中图显示同一窗口里 `z` 仍下降、`T_G` 仍上升，因而它不是 settled branch；右图显示线性 M 的所有非零档都在进入前压低活动，没有先建立发作态再终止。

**关注点**：当前结构在无 kick 条件下跨过了预注册的持续招募操作阈值，并出现约 5 Hz 主导调制，但缺失稳定高态与回到同一间期 basin 的 exit。右图的非零 M 结果应读成 prevention/containment，而不是 termination；centered 250-ms envelope 的 onset 不是因果分岔时刻。

### fig5_candidate_E1146_mz_divisive_current_stage.png / .pdf

上方为同一条 20 s 自发轨迹的连续 virtual-SEEG；中间将 population-rate 定义的 recruited onset 与 `z/T_G` 慢漂移对齐；下方分别显示 onset 前一个机器选择的 returning event、onset 后 recruited window 的真实 E-neuron 空间读出、完整 source→sink 轴时空场，以及 onset 附近的放大图。returning-event 颗粒颜色是逐神经元 first-spike latency，菱形颜色是触点 30–80 Hz envelope-peak latency；两者共用毫秒色标但不是同一测量。所有颗粒、触点与空间热图都来自同一次 capture，未用一维 rate 伪造空间结果。

操作性 onset 附近，48/48 个轴向 bin 在 60 ms 内依次跨过因果 50-ms activity 门，轴向 Spearman rho=+0.96；这对应约数百 mm/s 的 fast event sweep，而不是秒级的 ictal tissue-recruitment front。onset 后首个 1 s 中 47.4% 的 E 神经元发放，其中 18.3% 的全体 E 神经元超过 100 Hz。

**关注点**：当前模型并非没有空间结构；它保留 returning-event 的空间颗粒，并在操作性转变处产生快速有序轴向波。真正缺少的是一个慢速、局部改变组织状态的 recruitment front，以及其后的 refractory wake、stall/annihilation 和 return。高率细胞尾部也说明 population mean 的约 60 Hz 不能单独证明一个生理性有界高态。

两张图都是 current-stage diagnostic，不是正式锁定的 Figure 5，也不支持 seizure lifecycle、limit cycle、患者机制或 cohort inference。
