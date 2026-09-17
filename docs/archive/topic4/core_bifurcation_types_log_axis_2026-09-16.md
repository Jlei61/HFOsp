# Core 分岔图 v8：对数纵轴、类型标注与放电行为

用户要求放大 100 Hz 以下结构、移除原生 SNN 编号菱形，并明确 LP 与 PD 及 A/B 关系。本轮沿用已保存的 v2–v7 数值解，仅重绘并重新读取实际 2T 波形，不更换模型。

- [报告](../../../results/topic4_sef_hfo/core_bifurcation_types_v8_20260916/scientific_report.md)
- [Core A 主图](../../../results/topic4_sef_hfo/core_bifurcation_types_v8_20260916/figures/00_core_A_bifurcation.png)
- [Core B 主图](../../../results/topic4_sef_hfo/core_bifurcation_types_v8_20260916/figures/00_core_B_bifurcation.png)
- [七页图册](../../../results/topic4_sef_hfo/core_bifurcation_types_v8_20260916/figures/core_bifurcation_types.pdf)

纵轴 0–1 Hz 线性、以上对数；0–100 Hz 约占八成显示高度。LP1 是周期轨道鞍结（额外 μ=+1），PD1/PD3 为向降低 J 方向超临界倍周期，PD2 为向增大 J 方向亚临界倍周期（μ=−1）。实际 PD1 子解完整模式 375.074 ms 才重复，A 相邻 burst 仍约 187.5 ms；右侧还有高背景振荡，不能统一解释为 regular burst。

所有分岔解释属于联合确定性率闭合。A/B 无直接跨核连接、通过周边间接耦合；模态定位不是通路因果消融。完整全局周期族仍未穷尽，图候选待用户目视检查。
