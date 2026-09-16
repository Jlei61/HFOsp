# 核内分岔投影、尖点与A/B对照 v6 — 2026-09-15

本版承接v5，核查用户指出的1.176附近尖点、右侧均值/极值交叠，以及A/B联合动力学的区别。模型仍为冻结的六群体确定性延迟率闭合系统，未更换几何或原生SNN四状态样本。

1. 左侧确认LP0a/b/c周期折点、两稳定周期态共存；进一步发现并用反周期零模、加倍网格、独立Floquet和实际2T子支确认PD0，J约1.1763002486428，向增大J方向超临界。其临界右率模约99.5%在周边E分量。
2. J约1.1762741938处，A同周期内两峰交换最高峰身份，最大值曲线出现折角而轨道仍稳定。
3. J约1.3795519528处，不稳定轨道A均值与另一稳定轨道A谷值恰好相等；完整周期、波形和稳定性不同，交叠不等于分岔。
4. J=1.38的两条稳定共存轨道，A谷值分别接近0与206.63Hz，而B均保持高活动背景。A/B图共享对应联合轨道的临界J和谱，但读出及模态分量不同。

[完整报告与数值边界](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/core_observable_bifurcation_v6_20260915/scientific_report.md) · [数值核查](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/core_observable_bifurcation_v6_20260915/numerical_validation.json)

- [A/B相同坐标对照](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/core_observable_bifurcation_v6_20260915/figures/00_core_AB_bifurcation.png)
- [左侧招募区与峰身份交换](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/core_observable_bifurcation_v6_20260915/figures/04_recruitment_region.png)
- [极窄折点—PD0及特征乘子](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/core_observable_bifurcation_v6_20260915/figures/10_hidden_fold_flip_sequence.png)
- [A/B均值、最大值、最小值](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/core_observable_bifurcation_v6_20260915/figures/06_AB_mean_maximum_minimum.png)
- [相同J下右侧两种稳定波形](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/core_observable_bifurcation_v6_20260915/figures/07_same_J_right_waveforms.png)
- [完整图目录及逐图说明](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/core_observable_bifurcation_v6_20260915/figures/README.md)

左侧两族的全部远端不稳定连接及PD0子支再次失稳后的4T支仍未穷尽；本版没有将它们宣称为已完成的全局分岔清单。原生SNN的不规则burst不能直接用这些确定性倍周期结果定性。候选图待用户目视检查，旧版保留。
