# Core 低率到 burst 的经典分岔候选图

本次按用户要求完成平衡分支、完整时延特征谱、左右特征向量、自由周期轨道与 Floquet 稳定性；范围仅为 core burst 起始。

当前网络投影出的六群体确定性闭合，在核内 EE 倍率 **g=1.12541641329**、core A E 率 **0.454004672 Hz** 处有非退化 saddle-node。九个已求解周期点、周期的逆平方根标度和临界参数处的有限时间回返支持 SNIC 型起始。数值精度检查通过，图仍待用户目视检查。

该闭合使用实际连接矩、阈值经验分布和全部时延，但没有匹配当前原生 SNN 的低率工作点与动态响应；**原 SNN 的临界点和中间态分岔尚未确证**。不将此候选图替换为当前原生模型的已接受机制结论，也不沿用旧 q 分岔数值。

- [完整科学报告与方程](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/core_burst_bifurcation_v2_20260915/scientific_report.md)
- [经典分岔主图](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/core_burst_bifurcation_v2_20260915/figures/01_classical_bifurcation.png)
- [特征值与左右特征向量](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/core_burst_bifurcation_v2_20260915/figures/02_eigenvalues_and_eigenvectors.png)
- [完整图册](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/core_burst_bifurcation_v2_20260915/figures/core_burst_classical_bifurcation_booklet.pdf)
- [数值验证](/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/core_burst_bifurcation_v2_20260915/validation.json)

Producer：`/home/honglab/leijiaxin/HFOsp/scripts/topic4_core_bifurcation_v2/`。原始网络系数、延拓结果、周期解和未采用的早期试算均保留于上述结果目录；原 SNN 执行器和旧结果未修改。
