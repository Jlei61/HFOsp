# endpoint-package vs gradient-primary 七频带敏感性 — 结果与讨论（NOT axis-only）

date: 2026-07-19 (rev2, 审阅收紧)
spec: `docs/superpowers/specs/2026-07-19-axis-only-seven-band-endpoint-vs-gradient-design.md`
calc: `results/topic5_ictal_recruitment/field_concordance_grid_endpoint_axis/n161_endpoint/`
compare: `.../field_concordance_grid_endpoint_axis/endpoint_package_vs_gradient_primary_{per_band,subject_contrast}.csv, _summary.json`
figure: `results/paper-ready-figure/fig3_ictal_field_concordance_grid_method_sensitivity/axis_only/`

> **定位**：诊断性 sensitivity。**不是纯轴效应，也不是 paper-ready 主结果。** gradient 保持 primary，
> endpoint 只作对照。

## 测了什么

老 endpoint 版七频带"6/7 过 FWER"，现在 gradient 版只"2/7"。这是不是"换了轴"造成的？我们在**完全
相同的管线**下，把投影从 gradient（全触点传播梯度）换成 endpoint（源→汇端点核，k=3），各算一遍七频带
一致性，然后**直接配对**比两个轴每个被试的余量差，并做多重比较校正。

## 怎么测的（含一处重要限制）

- **held constant，且脚本 fail-closed 核验**：同 17 人 / 167 次发作、同共同 mask、同 seed、同 N=161、同
  coherent all-contact 1000 次置换——**两个轴的置换映射哈希逐事件完全相同**（不一致就 raise，不再静默）。
- **⚠️ 这不是纯 axis-only**：换轴的同时也换了 (a) **routing**（endpoint 全部 per-template A/B；gradient
  主口径是 7 shared + 10 own-fallback）、(b) **σ 数值**（规则同，但 σ 在投影平面上估，endpoint/gradient
  比值范围约 0.17–1.03）、(c) A/B grid 也随之变。所以右图测的是 **endpoint package − gradient-primary
  package**，**不能叫"轴效应"**。
- **判据 = 校正后的直接配对**：每频段 `endpoint 余量 − gradient 余量` 配对 Wilcoxon，**7 频段做 Holm +
  同步 subject sign-flip maxT** 校正；星号用校正后 p。另报 routing 分层（own-fallback 是最接近纯轴的内部
  对照）。

## 揭示了什么（收紧后）

1. **同管线下 endpoint 不再复现旧 6/7**：endpoint 过自身七带 maxT 的只有 3/7（δ/α/FR）、gradient 2/7
   （δ/θ）。所以旧 6/7 明显依赖旧的时间窗、分母、mask、镜像、null 那整套方法——**旧 6/7 不能归因于
   endpoint 轴本身**。

2. **总体没有检出 package 差异**：七频段折叠到被试一层，中位 endpoint−gradient **+0.0046**、11/17 略高、
   配对 **Wilcoxon p=0.19**、sign-flip p=0.14。

3. **没有任何频段过直接七带校正**：名义 β 原始 p=0.031 → **Holm 0.214**；α 原始 0.051 → **Holm 0.303**；
   sign-flip maxT 也 **0/7**（β≈0.39、α≈0.23）。**所以图上不打 β 星，只标为名义未校正。**

4. **名义 β/α 提升是 routing 混杂，不是轴**：分层看——
   - **own-fallback（10 人，两侧都 per-template = 最接近纯轴）**：折叠差 **+0.00016，5/10，p=0.62（≈0）**；
     β +0.004（5/10）、α −0.0045（4/10）都不显著；
   - **shared（7 人，routing 发生改变）**：折叠差 +0.029，β +0.066（7/7）、α +0.064（7/7）——名义提升
     全在这里。
   两侧口径统一的那一批（own）差不多是零，所以 β/α 是 routing 换了带来的，不是轴。

## 安全结论（正式口径）

> 在完全更新的 17 人 / 167 次发作合同下，endpoint 方法不再复现旧 6/7 结果；endpoint package 与
> gradient-primary 的总体七频带差异**未检出**，也**没有任何频带在直接七带校正后显示可靠差异**。名义
> β/α 提升集中在 routing 发生改变的 shared 分层、在最接近纯轴的 own-fallback 分层里≈0，故属 routing
> 混杂而非轴效应。gradient 继续作 primary，endpoint 保留为 sensitivity。**未做等价检验，故不主张两者
> 等价。**

若论文确实需要**纯轴**裁决：把 gradient 也固定成 per-template A/B（并决定是否冻结同一 σ），再重比——
本轮 out of scope。

（内部归档代号：endpoint build_endpoint_cores k=3 / compute_axis_frame / make_normalized_plane; gradient
shared-else-own primary; direct paired margin contrast + Holm + sign-flip maxT; routing-stratified own vs
shared; identical-pipeline perm-hash fail-closed; package-not-axis confound）
