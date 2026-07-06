# Topic 5 network-extension three-way statistic

### topic5_network_extension_three_way_comparison.png / .pdf

正式三联版。每个频段放在同一组里：`Core-field prediction`、`Hidden own order`、`Channel-shuffle null`。`Core-field prediction` 是 core-only interictal field 对 hidden contacts seizure-energy pattern 的 per-subject median |r|；`Hidden own order` 是 hidden contacts 自身间期顺序 C1 对同一发作能量的预测；`Channel-shuffle null` 是同一 subject、同一 hidden-contact set、同一发作集合下的通道打乱 null median。

三条 bracket 对应三个问题：`Core-field prediction` vs `Channel-shuffle null` = network extension；`Hidden own order` vs `Channel-shuffle null` = hidden 自身间期顺序是否也有预测力；`Core-field prediction` vs `Hidden own order` = 核心外推是否有 added advantage。

**关注点：null 对比**：Broadband energy：Core-field prediction > channel-shuffle null Wilcoxon one-sided p=0.0008，13/16 subjects above null；formal subject-pass 9/16，q<1e-4；Broadband energy：Hidden own-order > channel-shuffle null Wilcoxon one-sided p=0.0002，13/16 subjects above null；HFA energy：Core-field prediction > channel-shuffle null Wilcoxon one-sided p=0.0053，12/16 subjects above null；formal subject-pass 9/16，q<1e-4；HFA energy：Hidden own-order > channel-shuffle null Wilcoxon one-sided p=0.0005，15/16 subjects above null。Core-field 和 hidden own-order 都显著高于 channel-shuffle null。

**关注点：added advantage**：Broadband energy：Core>Own/Own>Core/Tie=7/5/4 (tie=|Δ|≤0.03)，Core-field > Own-order Wilcoxon one-sided p=0.2968；HFA energy：Core>Own/Own>Core/Tie=5/8/3 (tie=|Δ|≤0.03)，Core-field > Own-order Wilcoxon one-sided p=0.872。Core-field 没有系统性赢过 hidden own-order，但这不是严格等价性检验。

### topic5_network_extension_core_vs_null_and_own_order.png / .pdf

兼容上一版 combined 文件名，内容与正式三联版相同。

### topic5_network_extension_channel_null.png / .pdf

兼容旧文件名，内容与正式三联版相同。

### topic5_network_extension_three_way_comparison_summary.json

Machine-readable core-field/null/own-order medians, paired Wilcoxon statistics, formal subject-pass binomial/FDR results, and per-subject rows.
