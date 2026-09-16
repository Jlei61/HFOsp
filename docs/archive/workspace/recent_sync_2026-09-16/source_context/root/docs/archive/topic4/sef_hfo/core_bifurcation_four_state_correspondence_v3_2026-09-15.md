# Core分岔与原生四状态的编号对照 v3（2026-09-15）

2026-09-15 后续布局确认：上方所有曲线合并为单一坐标轴，去掉内嵌窗。当前[单坐标轴四状态图](../../../../results/topic4_sef_hfo/core_burst_bifurcation_states_v3_20260915/figures/01_single_axis_bifurcation_four_states.png)；[曲线来源与同次A/B仿真比较](../../../../results/topic4_sef_hfo/core_burst_bifurcation_states_v3_20260915/line_construction_and_core_comparison.md)。

按用户要求将核内EE倍率统一为 `J_{\mathrm{EE,core}}`，主图改成近方形，以简短 Fold 标记折点，并在下方配①低活动、②不规则burst、③中间态、④规则burst的原生波形与固定30细胞raster。右侧周期峰率、均率和谷值由实际周期解补至1.175；每个新增点均经周期方程与Floquet稳定性核验，更右侧未收敛尝试不当作新分岔。

**关键科学边界**：四个原生例子分别在0.5、0.7、0.85、1.0，均处于当前六群体闭合折点1.12541641329的左侧，连规则例子也如此。编号点按原生2–20秒实际平均率绘制，不能作为降阶平衡点或四种已证实吸引子。闭合的局部saddle-node及SNIC型周期起始证据保留，但尚未准确解释原SNN的状态边界或中间表型的分岔类型。

A/B是同一实际网络的两个空间核，A核720个E、B核742个E；无直接跨核连接，通过周边E/I群体间接耦合。临界小扰动模态首先在A核E，非线性周期burst可涉及两核；1.15周期解中的B峰晚于A峰约64 ms，不能单凭峰序推断因果。

- [近方形主图](../../../../results/topic4_sef_hfo/core_burst_bifurcation_states_v3_20260915/figures/01_square_bifurcation_four_states_inset.png)
- [特征值与左右向量](../../../../results/topic4_sef_hfo/core_burst_bifurcation_states_v3_20260915/figures/02_square_fold_eigenvalues_vectors.png)
- [两核关系图](../../../../results/topic4_sef_hfo/core_burst_bifurcation_states_v3_20260915/figures/03_core_a_b_relationship.png)
- [7页矢量图册，含四个状态的大幅raster](../../../../results/topic4_sef_hfo/core_burst_bifurcation_states_v3_20260915/figures/core_bifurcation_four_states_booklet.pdf)
- [完整科学说明及数值表](../../../../results/topic4_sef_hfo/core_burst_bifurcation_states_v3_20260915/scientific_report.md)
- [数值验证](../../../../results/topic4_sef_hfo/core_burst_bifurcation_states_v3_20260915/validation.json)；[图形自查](../../../../results/topic4_sef_hfo/core_burst_bifurcation_states_v3_20260915/visual_review.json)

Producer：`scripts/topic4_core_bifurcation_states_v3/`，数值方程继承v2；未新增原生SNN仿真。旧v2输出完整保留，本版为待用户目视检查的候选图。
