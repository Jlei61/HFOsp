# Axis-only 敏感性：endpoint 轴 vs gradient 主口径（七频带）— 结果与讨论

date: 2026-07-19
spec: `docs/superpowers/specs/2026-07-19-axis-only-seven-band-endpoint-vs-gradient-design.md`
calc: `results/topic5_ictal_recruitment/field_concordance_grid_endpoint_axis/n161_endpoint/`
compare: `.../field_concordance_grid_endpoint_axis/axis_only_endpoint_vs_gradient_{per_band.csv,summary.json}`
figure: `results/paper-ready-figure/fig3_ictal_field_concordance_grid_method_sensitivity/axis_only/`

## 测了什么

老的 endpoint 版（F2）七频带里"6/7 频段过 FWER"，现在 gradient 版只"2/7"。到底是不是**换了轴**
造成的？我们在**完全相同的管线**下，只把投影轴从 gradient（全触点传播梯度）换成 endpoint（源→汇
端点核，k=3），其余一切不动，然后**直接配对**比两个轴每个被试每个频段的"余量"（真实 − 随机零假设
中位）差。

## 怎么测的

- **held constant（逐比特相同已验证）**：同 17 人 / 167 次发作、同 `[0,10]s` 七频带能量、同共同
  mask（含 BB150-anchor 约束）、同 σ 规则、同 N=161 网格、同 coherent all-contact 1000 次置换。
  **两个轴的置换映射哈希逐事件完全相同**（`identical_pipeline_verified_perm_hashes=True`），所以差
  异只可能来自轴/投影。
- **changed**：只有投影轴。endpoint 用 source/sink 端点核定义（`build_endpoint_cores` k=3 →
  `compute_axis_frame` → 同一个 `make_normalized_plane`），per-template A/B。
- **confound（已在图/文标明）**：endpoint 是 per-template A/B，gradient 主口径是 shared-else-own。
  所以这一对比是"endpoint 套餐 vs gradient 主口径套餐"，**混了轴 + routing 两个因素，不是纯轴**。
- **直接检验**：每个被试每个频段 `endpoint 余量 − gradient 余量`，配对双侧 Wilcoxon + 被试符号翻转；
  再把七频段折叠到被试一层做一次总体配对检验。**这才是判据，不是"谁星多"。**

## 揭示了什么

1. **在完全相同的管线下，endpoint 轴只过 3/7（δ/α/FR），gradient 过 2/7（δ/θ）——远不是老版本的
   6/7。** 所以老"6/7→2/7"的落差**主要不是轴**，而是当初 endpoint 版和现在 gradient 版之间的**其它
   方法差异**（时间窗、分母、网格、共同 mask、镜像规则、null 那一整套）。把这些都对齐后，换轴带来的
   过 FWER 频段数只从 2 变到 3。

2. **轴本身的直接效应很小、且频段特异**。直接配对检验（endpoint − gradient 余量）：
   - **β：+0.040，配对 Wilcoxon p=0.031（显著）**、符号翻转 p=0.015；
   - α：+0.037，p=0.051（边缘）；
   - δ/θ/γ/R/FR：中位差都为正但都不显著（p 0.35–0.85）。
   - 七频段折叠到被试：中位 endpoint−gradient **+0.0046**、17 人里 **11 人** endpoint 略高，但配对
     **p=0.19（整体不显著）**。

3. 所以对审阅第 1 条"新旧绝不只是换了轴"的判断**成立且被量化**：换轴在同管线下只给了一点（主要在
   β、其次 α）的提升，撑不起老版本的 6/7；而且这点提升还**混着 routing**（endpoint per-template vs
   gradient shared-else-own），不是纯轴效应。整体上两个轴给出的七频带图景高度相似。

4. **纪律**：gradient 仍是 primary，endpoint 只作 sensitivity。**不能**用"endpoint 3 星、gradient 2
   星"推断"轴更好"（显著性差异谬误）——要看直接配对（只有 β 显著）。要真正把"轴"从"routing"里拆
   出来，需要把两边都固定成 per-template A/B 的纯 axis-only（本轮 out of scope，可作后续）。

## 一句话

老"6/7→2/7"**主要来自方法学口径的整体收紧，不是换轴**；换轴本身在完全对齐的管线下只带来 β（及边缘
α）频段的小幅、混着 routing 的提升，整体七频带一致性差异不显著（折叠 p=0.19）。endpoint 轴保留为
sensitivity，gradient 保持 primary。

（内部归档代号：endpoint build_endpoint_cores k=3 / compute_axis_frame / make_normalized_plane,
gradient shared-else-own primary, coherent_cohort_spatial_null_p, seven_band_maxt_pfwer, direct paired
margin contrast, band→subject fold, axis+routing confound, identical-pipeline perm-hash invariant）
