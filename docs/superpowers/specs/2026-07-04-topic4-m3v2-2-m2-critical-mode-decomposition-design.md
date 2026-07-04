# Topic 4 — M3-v2.2 Approach-Criticality — Milestone 2: Dense α₀ Crossing + Axis/Global/Non-axis Critical Mode Decomposition · Design

date 2026-07-04 · 状态 **design rev1** · 分支 `topic4-criticality-m2`（worktree, base `codex/topic4-criticality`@1207e85, off M1）· 前置：**M1 (frozen-Jacobian verdict instrument) COMPLETE** — 真实 v2.2 仿真轨迹 verdict=`unresolved_operating_point`，子原因 `alpha0_crossing_between_sampled_trajectory_points`（低支 α₁ 在两抽样点间穿 0 至 +0.189，采样漏采）。

> **方法学 base** = M1 spec `docs/superpowers/specs/2026-07-02-topic4-m3v2-2-approach-criticality-design.md` + M1 code（`src/topic4_criticality.py`, `src/topic4_m3b_spectral_phase.py`）。M2 **复用** M1 判读器 + 特征模指标，只新加：dense α₀ localization、axis/global/nonaxis basis + projection、projected gain/leak、minimal rate-field perturbation。
> **执行 gate**：M2 全模型侧，不消费 topic5 phase2。Topic5 correspondence 留 M3。

---

## 0. 摘要（朴素话）+ 锁定口径

M1 的尺子在真实 v2.2 **仿真**轨迹上说"看不清"：抽样快照上恢复速率 α₁ 还明显为负，但分支延续补检发现低支 α₁ 在两个抽样点之间穿过 0（最高 +0.189）。M2 做两件事：**(1) 把这个穿零点加密定位出来**（在真实三维慢状态里递归二分，精确找 α₁=0 crossing）；**(2) 判穿零处那个变软/失稳的模态是什么空间形状**——压在间期 HFO 轴上（轴向）、全场同步（全局）、还是逃到轴外残差网络（非轴向）。

核心科学问题：**失控前的临界模态，是沿间期轴变软，还是轴向减弱、非轴向/全局模态接管。**

怎么判：定义三个空间子空间——轴（2D：沿轴参与 + early→late 梯度）、全局（全场均匀）、非轴残差（投掉轴和全局剩下的）；把穿零处的领头模态投影到三个子空间看能量落哪；再用投影算子的 finite-time gain/leak 看扰动被放大/泄漏的方向；最后用最小的 rate-field 扰动 spot-check 验证这套线性化读数真能预测扰动响应。

**锁定口径（抗审稿）**：(a) **global ≠ nonaxis**——全场同步 vs 真正离轴残差网络是不同生物学解释，JSON/图必须拆 subtype。(b) 分类用**能量**不用符号（特征向量/复共轭对相位会翻转）。(c) 判决预注册三类 {axial_supported / off_axis_global_supported / unresolved} + 强制 subtype，且有 **base gate** 前置。(d) M2b 与 M2a 矛盾 → `linear_mode_candidate_unvalidated`。(e) tier=`model_side_preliminary`，从不声称"模型证明发作/CSD"。

（内部归档代号：`e_axis_occupancy`/`e_axis_signed`/`e_global`/`e_nonaxis`；`P_axis`/`P_global`/`P_nonaxis`；`E_axis`/`E_global`/`E_nonaxis`；`alpha0_crossing`；`gain_{axis,global,nonaxis}_to_{...}`/leak；`nonaxis_source_policy=critical_residual_direction`；verdict∈{axial_supported, off_axis_global_supported, unresolved}；subtype∈{axial_occupancy, axial_signed_gradient, global, nonaxis_residual, mixed_axis_global, mixed_axis_nonaxis, mixed_global_nonaxis, unclassified}；dominance≥0.50/margin≥0.15。）

---

## 1. 核心问题 + verdict + subtype

**Q**：穿零处的临界模态落在 轴 / 全局 / 非轴 的哪个子空间？

**Top-level verdict（预注册 3 类）**：
- `axial_supported` — 穿零处模态主要压在间期轴上。
- `off_axis_global_supported` — 全局同步 或 非轴残差 主导。
- `unresolved` — 定位不了 / 分支模糊 / 无清楚 dominance / 线性预测与扰动矛盾。

**Mandatory subtype（JSON + figure 必须拆，global ≠ nonaxis）**：
`axial_occupancy` / `axial_signed_gradient` / `mixed_axis`（轴内 occupancy+signed 都贡献）/ `global` / `nonaxis_residual` / `mixed_axis_global` / `mixed_axis_nonaxis` / `mixed_global_nonaxis` / `unclassified`。

**报告量**：`alpha0_crossing_time_ms`、`alpha0_crossing_slow_state`(q_I/g_K/h_G)、`crossing_width_ms`、`E_axis`/`E_global`/`E_nonaxis`（+ 轴内 `E_occupancy`/`E_signed`）、`mode_class_at_alpha0`、projected gains/leaks、perturbation spot-check outcomes、`threshold_sensitivity`。

---

## 2. 投影基（Q1，承重）

三个子空间在 **E-node 空间**定义：

- **Global** `G = span(e_global)`，`e_global = normalize(uniform over E nodes)`。全场同步基准，**单独报，不并入 nonaxis**。
- **Axis** `A = span(e_axis_occupancy, e_axis_signed)`（2D 子空间，`Q_axis = orthonormalize([occupancy, signed])`）：
  - `e_axis_occupancy`（无符号沿轴参与）：`raw = axis_mask(E-subset) as float`；`e_axis_occupancy = normalize((I − P_global)·raw)`。问：模态是不是压在 corridor/axis 上。
  - `e_axis_signed`（有符号 early→late 梯度）：`s_i = (x_i − x_center)·axis_unit`；`raw_i = axis_mask_i · centered(s_i)`；`e_axis_signed = normalize((I − P_global − P_occupancy)·raw)`。**sign convention：+ = early→late（锁死）**。问：模态是不是沿 early→late 梯度偏移。
- **Non-axis** `N`：`P_nonaxis = I − P_global − P_axis`，`P_axis = Q_axis Q_axisᵀ`。投掉全局和轴，剩下算非轴。

**主指标（和≈1）**：`E_axis = ‖P_axis·loading‖²/‖loading‖²`、`E_global`、`E_nonaxis`。轴内再分 `E_occupancy`/`E_signed`。

**分类用能量、不用符号**（特征向量/复对相位翻转）；方向性分析（谁 early 谁 late）才用 signed projection。

**Full-state 嵌入**（承重，别假装是 full-state basis）：基在 E-node 空间，Jacobian 是 full 6N state。
- eigenmode 分类：用 M1 已有的 rE-field loading / complex-pair invariant-subspace loading（`mode_e_field` / `pair_loading`）。
- projected gain/leak：input `B_X = embed_rE(Q_X)`（rE block = 方向，其余 5 个 state block = 0）；output = read rE block of `exp(JT)·B_X`，再投影到目标子空间。明说这些是 **E-rate observable subspaces**。

---

## 3. M2a — dense α₀ localization + mode projection + projected gain/leak

### 3.1 dense α₀ localization（残留#1 已定）
在 M1 的 last-qualified low-branch 点 与 first-saturated 之间，**递归二分**慢轨迹（线性插值 slow state (q_I, g_K, h_G)）：
- 每层：在中点慢状态重解 low branch（`solve_branches` warm-start）、算 α₁、检查 qualified + branch identity。
- **停止**：`crossing_width`（夹住 α₁=0 的相邻两点的时间间隔）< `crossing_width_ms_tol`（默认 **1.0 ms**）或达 `max_bisect_levels`（默认 **8**）。
- 输出：`alpha0_crossing_time_ms`、`alpha0_crossing_slow_state`、`crossing_width_ms`、`branch_identity_near_crossing`（穿零附近是否全 low_branch、无 ambiguous）、`quality_status_near_crossing`（两侧 `qualify_point` reason）。
- **base gate**：穿零附近 branch identity 混乱（ambiguous）或 quality 两侧都不合格 → `not_cleanly_localized` → 下游 `unresolved`。

### 3.2 mode projection at crossing
在 `alpha0_crossing` 慢状态求 op + Jacobian + `rate_eigenpairs`；取 leading subspace（complex pair 当 2D 不变子空间，用 `leading_subspace_indices`）；算 rE-loading（`pair_loading`），投影到 axis/global/nonaxis：`E_axis`(+ occupancy/signed 分)、`E_global`、`E_nonaxis`、`mode_class_at_alpha0`。**复对用 invariant-subspace loading，不用单向量符号。**

### 3.3 projected operator gain/leak
在预注册 horizons（复用 M1 `[10,25,50,100,250,500]` ms）对穿零处（+ 可选 pre/post-crossing 点）算 9 个 source→target directional gain：
`gain_{axis,global,nonaxis}_to_{axis,global,nonaxis}(T) = ‖P_target · rE(exp(JT)·embed_rE(e_source))‖ / ‖e_source‖`。复用 M1 `transient_gain`（matrix-free `exp(JT)·b`）。
`leak_axis_to_nonaxis`/`leak_axis_to_global`/`return_nonaxis_to_axis` 是其中的 off-diagonal。
**nonaxis source = `critical_residual_direction`**（穿零领头模态在 N 子空间的投影方向）；artifact 明写 `nonaxis_source_policy=critical_residual_direction`，**不冒充** full `‖P_N exp(JT) P_N‖`。

---

## 4. M2b — minimal rate-field perturbation spot-check（残留#2 已定）

验证 frozen-Jacobian 读数真能预测扰动响应（审稿必问）。只取 **3-4 个点**：`early_stable` / `last_sampled_qualified` / `just_before_alpha0` / `just_after_alpha0`（末者若可读）。

扰动方向（同 E-rate norm）：`critical_eigenspace` / `axis_occupancy` / `axis_signed` / `global` / `nonaxis_critical_residual` / `random_orthogonal_control`。ε sweep = `[small, medium]`（config）。

**积分器（残留#2 已定）**：复用 `solve_operating_point` 的 rate-field 动力学（`_moments` + `_phi_field`），**新建一个小积分器**：从 `z* + ε·v` 起步、积分非线性 rate field、测 `observed_peak_gain`、`observed_recovery_time`（回到 op 的时间）、`return_to_op_success`、`escape_probability`（是否跑飞）、`observed_{axis,global,nonaxis}_leak`。（rate-field 层，**非 SNN**。）

**验收（弱：方向一致，非严格 τ）**：`predicted_tau↑ ↔ observed_recovery_time↑`；`predicted gain_nonaxis/global↑ ↔ observed nonaxis/global amplification↑`。M2b 与 M2a 强烈矛盾（线性说 X dominant 但扰动完全不支持）→ verdict `linear_mode_candidate_unvalidated`（不写 supported）。

---

## 5. 验收门（Q3，预注册，跑前锁 — "门编码结论非存在"）

### 5.0 base gate（先过，否则 unresolved）
`alpha0 crossing cleanly localized in actual slow-space`（§3.1）∧ `branch identity clean near crossing` ∧ `quality gate passes both sides`（或明确可解释边界）∧ `complex-pair policy applied` ∧ `mode projection from rE-field loading / invariant-subspace loading`。

### 5.1 `axial_supported` iff
`E_axis dominant at α₀` ∧ `E_axis − max(E_global, E_nonaxis) >= margin` ∧ projected axis gain 不被 global/nonaxis gain 反驳 ∧ rate-field perturbation 不反驳 axis dominance。subtype：`axial_occupancy` / `axial_signed_gradient` / `mixed_axis`（轴内 occupancy vs signed）。

### 5.2 `off_axis_global_supported` iff
(`E_global or E_nonaxis dominant at α₀`) or (`projected gain_global/gain_nonaxis exceeds axis gain near α₀`) ∧ rate-field perturbation 不反驳。**subtype MUST split**：`global` / `nonaxis_residual` / `mixed_global_nonaxis` / `mixed_axis_global` / `mixed_axis_nonaxis`。

### 5.3 `unresolved` iff（任一）
α₀ 未紧致定位 ∨ branch identity ambiguous ∨ mode class 在 tiny continuation 变化下不稳 ∨ 无清楚 E dominance ∨ projected gain 与 eigenmode projection 强烈冲突 ∨ M2b 与 frozen-Jacobian 预测相反。**unresolved 非失败，是防强解释。**

### 5.4 dominance thresholds（primary + sensitivity）
- primary：`dominant iff top_energy >= 0.50 ∧ top_energy − second_energy >= 0.15`。
- mixed：`top >= 0.35 ∧ second >= 0.25 ∧ gap < 0.15`。else `unclassified`。
- `threshold_sensitivity`：`dominance ∈ [0.45, 0.50, 0.60] × margin ∈ [0.10, 0.15, 0.20]`。primary 翻则文字写 "X-dominant but threshold-sensitive"，不只靠单阈值。

---

## 6. 3 个残留点（已定，锁进 spec）
1. **densification**：递归二分，停 `crossing_width_ms < 1.0` 或 max **8** 层（config）；每层重解 low branch + branch/quality check。
2. **perturbation integrator**：复用 `solve_operating_point` rate-field 动力学，新建 `z*+εv → integrate → recovery/gain/escape` 小积分器（rate-field，非 SNN）。
3. **substrate**：`subject1146`（同 M1 轨迹；`axis_unit`/`axis_mask`/`pos` 就绪）。

---

## 7. 红线 / tier
`model_side_preliminary`；单本征值≠发作；global runaway≠真发作；谱=机制地图非表型证明；有核不声称平面波 k 模；**禁"模型证明发作/CSD"**；**禁"真数据"（用 actual v2.2 SIMULATION trajectory）**；**global ≠ nonaxis**；分类用能量非符号；M2b 矛盾→`linear_mode_candidate_unvalidated`。
**DEFER（非 M2）**：SNN perturbation；slow-var attribution（∂E/∂{q_I,g_K,h_G}）；full controls（no-core/isotropic-AR1/shuffled-core/axis-rank-shuffled/axis-mask-rotated/h_G-off/ramp-rate）；Topic5 correspondence → **M2.5 / M3**。

---

## 8. config / results / 复用
- **config**：新 `config/topic4_criticality_m2.yaml`（basis：sign_convention=early_to_late、embedding=rE_block；densification：crossing_width_ms_tol=1.0、max_bisect_levels=8；gain：horizons 复用 M1；perturbation：points、directions、epsilon=[small,medium]；verdict：dominance/margin + sweep）。
- **results**：`results/topic4_criticality_m2/`（`dense_trajectory_verdict.json`、`alpha0_crossing.json`、`continuation_trace.csv`、`mode_decomposition.json`、`perturbation_spotcheck.json`、`figures/` + README 中文 + FIGURE_INDEX）。
- **复用 M1**：`solve_branches`/`build_jacobian_dense`/`rate_eigenpairs`/`leading_subspace_indices`/`pair_loading`/`mode_e_field`/`elongation_axis_score`/`off_axis_score`/`globality`/`transient_gain`/`numerical_abscissa`/`check_low_branch_continuation_between`/`qualify_point`；`subject1146` layout（`axis_unit`/`axis_mask`/`pos`）。

---

## 9. tasks 概览（plan 细化）
- **T0** config + basis 定义 + sign lock（pure fns + tests：三投影能量和≈1、`Q_axis` orthonormal、energy≥0、embed/readout 往返）。
- **T1** dense α₀ localization（递归二分 + branch/quality check + crossing outputs + base gate）。
- **T2** mode projection at crossing（rE-loading → E_axis/global/nonaxis + occupancy/signed 分，complex-pair）。
- **T3** projected gain/leak（9 source→target via `transient_gain`，`nonaxis_source_policy`）。
- **T4** rate-field perturbation integrator + spot-check（复用 rate-field 动力学）。
- **T5** verdict（base gate + dominance + subtype + sensitivity + M2b-contradiction）+ artifacts + Figure（穿零处 axis/global/nonaxis 能量条 + projected gain 矩阵 + τ-crossing；中文 README）。
- 每步 pre-registered gate 编码为阈值 + 坏数据回归；TDD。

---

## 10. Self-Review
1. **Placeholder**：所有阈值在 §5/§8 锁（crossing_width 1.0、max 8、dominance 0.50、margin 0.15、sweep grid、epsilon config）；无 TBD。
2. **一致性**：basis(§2) 能量分类 + global≠nonaxis 贯穿 §1/§5；M2a(§3) 复用 M1 机器；M2b(§4) 弱验收非严格 τ；verdict(§5) base gate 前置 + 三类中性 + subtype 强制拆。
3. **Scope**：单 spec T0-T5，model-side；SNN/attribution/controls/Topic5 明确 defer。
4. **Ambiguity**：分类用能量（非符号）；`nonaxis_source_policy` 明写不冒充 operator norm；embedding=E-rate observable（非 full-state basis）；densification 停止准则明确；perturbation 是 rate-field 非 SNN。
