# D_AB 梯度轴 sensitivity：轴代表性 + F2 换轴复算（2026-07-18）

> tier = **exploratory sensitivity**（不进主分析、不改主结论）；这条线已收尾、不再继续开发。
> D_AB 轴定义本身（`src.topic5_scaffold_ab_contrast.build_D_AB` / `src.dab_gradient_axis` / `docs/paper-draft/methods_axis_gradient_rewrite.md`）已在主线合并里收尾；本文件只记录两项**围绕该轴的敏感性复算**。

## 朴素话摘要

我们把"间期传播轴"的定义从旧的"两个端点连线（source→sink）"换成"连续的 D_AB 三维梯度场"，然后问两件事：

1. **这条梯度轴是不是一条真的方向轴？** —— 看它能不能代表单次事件各自的传播方向。做法：把每次事件的方向和拟合出的轴比余弦，再和"把模板排名在同一批触点上打乱、用同样方法重拟一条轴"的零假设比。如果轴只是估计器的产物，观测余弦应该和打乱后差不多（≈0）。**实测：26 个被试的中位 signed cosine = 0.45，零假设 ≈ 0，中位 rank-shuffle p = 0.007** —— 观测远高于打乱基线，多数被试显著。也就是说这条梯度轴确实指向单事件方向、不是估计器假象。（少数被试 p 到 0.33，非全体通过。）

2. **换了这条轴，原来的主结论会不会翻？** —— 把原来的 7 频带 F2 队列统计（间期传播场 ↔ 发作早期能量场对齐）的"场读出"从旧端点轴换成 D_AB 冻结梯度场，其余（窗口/事件资格、fold、空间 null 层级、1000 抽 Westfall–Young FWER 家族）全部保留。**稳健的不变量：ripple（`hg_low_ripple` / `ripple_high`）在所有变体里都不过 FWER** —— 即"对齐不是 ripple 特异"的主结论换轴后依然成立。低频带里哪几条过 FWER 随平滑/routing 变体而变（见下），属敏感性范围，不作独立主张。

## 怎么测（合同要点）

- **簇1 轴代表性** `run_topic5_axis_representativeness.py` + `src/topic5_axis_representativeness.py`
  - 统计单位 = 被试；被试内 fold = TA/TB 等权平均。
  - primary metric = 观测 mean signed cosine − 模板 rank-shuffle null 中位。
  - 事件门：mapped contacts≥6、shafts≥2、effective rank≥2、LOCO valid≥0.8、median signed cosine≥0.8；每模板 ≥20 events。
  - strict frozen-axis stability 只作 sensitivity（13/… ），非纳入门。
  - cohort flow：40 请求 → 28 可溯源 → 56 梯度模板行 → 52 合格 → 26 被试行。
- **簇2 F2 换轴复算** `run_topic5_gradient_multiband_*.py`
  - `field_routing = shared_a/shared_b if complete else own_a/own_b`，`routing_is_outcome_independent = True`。
  - 5 个控制变体：base（`original_f2`）、每被试单一固定 sigma（`fixed_sigma`，平滑审计承重控制）、`shared_only`、`cohort_matched`，外加 7 频带显著性重建（`significance`）。
  - 出图复用 `scripts/plot_topic5_v2_phase1_figures.py::fig2_null_perband`（主 Fig3-Sup1 的同一出图器；本次字号默认上调，统计不变）。

## 各变体 FWER 通过 band（cohort 层，供参考，非主张）

| 变体 | 过 FWER | 不过 |
|---|---|---|
| `original_f2_fixed_sigma` | δ_HYP_slow / θ_preictal_PAC / β_LVFA_low | alpha / gamma / hg_low_ripple / ripple_high |
| `original_f2_fixed_sigma_shared_only` | δ / alpha / β / gamma | theta / hg_low_ripple / ripple_high |
| `onset_0_10s` | gamma_LVFA | δ / θ / alpha / β / hg_low_ripple / ripple_high |

**不变量**：ripple 两带在所有变体均不过；低频带通过与否随变体漂移（敏感性）。

## 文件

- 代码：`src/topic5_axis_representativeness.py`、`scripts/run_topic5_axis_representativeness.py`、`scripts/paper_figures/plot_axis_representativeness.py`、`scripts/run_topic5_gradient_multiband_original_f2{,_fixed_sigma,_fixed_sigma_cohort_matched,_fixed_sigma_shared_only}.py`、`scripts/run_topic5_gradient_multiband_significance.py`；测试 `tests/test_topic5_axis_representativeness.py`、`tests/test_topic5_gradient_multiband_original_f2{,_shared_only}.py`、`tests/test_topic5_gradient_multiband_significance.py`。
- 结果：`results/interictal_propagation_masked/axis_representativeness/`、`results/paper-ready-figure/fig_axis_representativeness/`、`results/topic5_ictal_recruitment/gradient_multiband_*`、`results/paper-ready-figure/fig3_gradient_multiband_significance/`。

## 禁止 claim

不写"D_AB 轴证明发作沿间期方向重放 / timing-order replay / 机制"；不把某个变体过 FWER 的低频带升级成 cohort 主张；轴代表性支持"轴是真方向轴"，不等于"发作沿此轴传播"。
