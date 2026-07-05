# M3B 谱相图 / W-场阶段性结论（2026-06-28）

> Scope: M3 阶段 B 线（谱相图 / W-场）收口。主文档 `docs/topic4_m3_stage.md`；姊妹 A 线
> `m3a_stage_conclusion_2026-06-27.md`。证据来自 `src/topic4_m3b_spectral_phase.py`（84+ TDD 测试绿，
> commit `32ba62d`→当前）+ ignored `results/topic4_sef_hfo/m3b_spectral_phase_map/`（由
> `scripts/build_m3b_spectral_outputs.py` 再生，含 STATUS.md / verdict_inputs.json / 8 artifacts / figures）。
> 这是模型 / 线性算子层面的机制 screen，**不是发作机制 validation**。

## 一句话判断

> 把带核薄片线性化、扫"核兴奋度 × 全局去抑制"相图：**最先持续长大的本征模式永远是全局的**（齐次色散最高
> 在 k=0，没有有限-k 峰）；但**按计划 §5 用非正规瞬态读，核扰动会被瞬态放大、沿 E→E 轴铺开再自限**——
> 这条"间期自限轴向传播"信号在 8/8 未饱和点全出现、且**方向骨架特异**（各向同性对照轴向≈0）。
> 判决 = **SPM-PASS frozen map**（线性算子层面，无 SNN/M3A/几何零模型三道桥）。

## 路线与已验证机器（TDD-0..15 + path(a)）

整条机器全部 test-first 建出并验证（相图/Jacobian/本征对/度量/读出/判决闸）：

- **有限 Jacobian（6 场 [rE,rI,sEE,sEI,sIE,sII]）**：A(k) 块矩阵经 3 路独立推导 + 数值对账锁定
  （JVP 对有限差分到 1e-12；主导 rate-branch 本征值对解析色散到机器精度）。归约到
  `src/sef_hfo_lif._char_det`（δ=0）。设计锁档：`m3b_jacobian_design_LOCKED_2026-06-27.md`。
  陷阱：6×6 永远带两个突触极点 −1/τ_AMPA、−1/τ_GABA(=−0.0556，在慢簇)，所以 α₁ 取 **rate-branch**
  + `synaptic_pole_floor_active` flag，cross-check 用 membership 不用裸 max。
- **工作点**：改用"把真实 6 场率方程积分到稳态"（这是抑制稳定网络，朴素 Picard 会振荡）。齐次极限归约到
  `mean_field`。关键物理：高外驱**不**饱和（抑制跟着升、把驱动抵消）；失控来自去抑制（q≤0.8）或高 w_ee。
- **左右本征对**：biorthonormal，残差 1e-15、双正交误差 2e-15。非正规诊断（`finite_time_gain`、
  `core_controllability` 用**左**本征向量）为**主**指标（计划 §5），不是补充。
- **模式度量**：elongation_axis vs phase_gradient_axis 保持 90° 可区分；globality；off_axis。

## 主结论（§5 主读法）

- **主导本征模式全局**：齐次色散沿 E→E 轴在 k=0 最高、随 k 先降到 k≈1.65 的谷再略回升（**非严格单调**，
  但最高点在 k=0、**无有限-k 峰**）。AR2 各向异性核在有限 k 有方向偏好（沿轴 vs 横轴增长不同），AR1 各向
  同性则没有（对照）。
- **§5 非正规瞬态 = 自限轴向**（机器已三重验证）：core kick 的 `‖e^{JT} b_core‖/‖b_core‖` 在 ~10ms 冲到约
  **2 倍**（放大本身**不挑骨架**，AR2/AR1 都有），沿轴拉伸峰更靠后（**~30ms、max≈0.45**），随后衰减
  （**自限**）。**轴向才骨架特异**：AR1 各向同性对照 max_axis≈0。在 8/8 未饱和点全成立。
  → 间期自限轴向传播**活在非正规瞬态里，不在主导花样里**。
- **谱增长率 α₁ ≠ 非线性饱和**：相图 `runaway` 格全部来自工作点非线性饱和（op_status=saturated，
  α₁ 实际负≈−0.05），不是 α₁>0 线性失稳。两者文档里必须分开。
- **对照消融特异**：无核 → 核局域≈0（有核 0.057）；AR1 → 色散各向异性=0（AR2>0）；打散核 < 连续核；
  off-axis 核跟着核走但更弱。

## SNN 口径（**已纠正**——这是本 doc 的重点更正）

B 线脚本里内置一个 tiny-SNN 抽查（`run_snn_spotcheck_grid`，L=0.5mm、~500 神经元 × 6 点 × 3 seeds），
它给出"招募轴向≈0"，曾被错误写成"§5 轴向不在 spiking 层复现"。

> **这个 in-line grid 不是有效的轴向测试，其"轴向≈0"是伪影，已撤回。** 原因两条：(1) 片子太小
> （kick 盘几乎盖满整片）；(2) 用的是**放电空间拉伸**（错仪器）。

**轴向在 spiking 层的正确验证 = A 线 A2-P 源空间逐细胞 onset 梯度**（`m3a_a2_abbott_lg_pilot_recap_2026-06-26.md`
§6.2）：40000 神经元（L=20mm）SNN，高许可度大态**是**一条相干单源**轴向招募波**——onset~位置回归
**R²≈0.87**、梯度沿轴 **align≈1.0**、方向可读率 **1.0**。所以 **§5 线性轴向与 SNN 一致**；B 线判决停在
frozen-map 是因为**它本身是线性算子结果**，不是因为 spiking 失败。`snn_spotcheck_grid.png` 现作为
"小片+错仪器假阴"的反面教材保留，`verdict_inputs.json::snn_axial_validated_in_M3A_A2=True`。

## Round-1 桥（已完成，现下游）

kick-rate-field 仪器探针支持间期 scaffold 桥（model-to-real median field corr ~0.844、placement ~74%，
打赢 channel/within-shaft 零模型）；ictal-early 腿只 placement（~0.420/~72%，**没**打赢几何零模型）；
"same field two gains" 扫不出 graded recruitment。这条是 M3B-R2 谱相图的上游动机。

## 判决闸（fail-closed）

`m3b_verdict()` 显式闸：`SPM-PASS frozen map` 需 `controls_pass AND non_normal_axial_pass`（缺一→
`SPM-BOUNDED negative`）；`spontaneous mechanism` 另需 `snn_grid_pass`；`full bridge` 再需
`m3a_overlay_pass AND readout_null_pass`。本轮真值（`verdict_inputs.json`）：

- `controls_pass=True`、`non_normal_axial_pass=True`、`model_matches_dynamics(ratefield)=True` → frozen-map ✓
- `snn_grid_pass=False`（B 线内置 grid 无效；**轴向已在 A2-P 验证为正**，但不是 B 线的有效 SNN 闸）→ 不升 spontaneous
- `m3a_overlay_pass=False`（M3A 5 产物缺失，overlay `refused`；相图是 raw-knob，`m3a_overlay_consumable=False`）
- `readout_null_pass=False`（几何零模型 `not_run`，verdict=`projection_only`，非 `placement_only`）→ 无 full bridge

## 能写 / 不能写

**能写**：线性算子层面，核扰动触发**骨架特异的自限轴向瞬态**（§5 主读法，8/8 点）；它与 A 线 A2-P 的源空间
轴向招募波互相印证。

**不能写**：`W causes seizure` / `α₁>0 = 临床发作` / `平面波 k 模式解释固定核事件`（无有限-Jacobian 证据）/
`M3B 证明慢变量致发作`（无有效 M3A 轨迹 + SNN validation）/ `§5 轴向不在 spiking 层复现`（错仪器假阴）/
`已完成 model-to-patient bridge`（读出只 projection_only）。

## 下一步

1. 若要把 §5 轴向升到 spontaneous-mechanism：用**正确口径**重做（A2-P 源空间 onset 梯度、proper-scale SNN），
   不是 B 线 tiny grid。
2. 若要升 bridge：补真实 cohort 几何零模型（当前 `geometry_null_status=not_run`）。
3. A→B overlay：等 A2 trajectory/export schema 过测试 + 接口合同（normalized phase coords）满足；当前
   raw-knob 相图不可被 overlay 消费。
