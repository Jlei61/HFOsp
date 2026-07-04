# Topic 4 — M3-v2.2 Approach-Criticality — Milestone 2: Dense α₀ Crossing + Axis/Global/Non-axis Critical Mode Decomposition · Design

date 2026-07-04 · 状态 **design rev1.1**（rev1 → rev1.1：折入用户设计评审 P1-1/P1-2/P1-3 + 10 点 required edits + **A' 决定**）· 分支 `topic4-criticality-m2`（worktree, base `codex/topic4-criticality`@1207e85, off M1）· 前置：**M1 (frozen-Jacobian verdict instrument) COMPLETE** — real-v2.2 verdict=`unresolved_operating_point`（低支 α₁ 在两抽样点间穿 0 至 +0.189，采样漏采）。

> **方法学 base** = M1 spec `2026-07-02-topic4-m3v2-2-approach-criticality-design.md` + M1 code。M2 **复用** M1 判读器 + 特征模指标，只新加：dense α₀ localization、THETA_EE 模态分类的显式持久化、gain/leak 方向向量、minimal rate-field perturbation。
> **执行 gate**：M2 全模型侧，不消费 topic5 phase2。Topic5 correspondence 留 M3。
> **A'（核实后锁定）**：本征模住在 **m3b 率场网格 `Grid(n=6)`（单核, THETA_EE=π/4 轴）**，**不在** subject1146 SNN/电极空间。**mode 分类用 THETA_EE 二阶矩 shape 分数（复用 M1）**；**gain/leak 用 rate-field state space 里归一化的方向向量**。**`phase_gradient_axis_score` 是无向 wavevector 轴对齐、不是 early→late signed**（`|F|²`+`cos(2(β−θ))` 对 180° 对称）——禁写 "signed early→late"；真要方向性需另做相位/时序定义（**deferred**）。

---

## 0. 摘要（朴素话）+ 锁定口径

M1 的尺子在真实 v2.2 **仿真**轨迹上说"看不清"：抽样快照上恢复速率 α₁ 还明显为负，但分支延续补检发现低支 α₁ 在两个抽样点之间穿过 0（+0.189）。M2 做两件事：**(1) 把穿零点加密定位出来**（真实三维慢状态里递归二分）；**(2) 判穿零处那个变软/失稳模态的空间形状**——沿模型传播轴（轴向）、全场同步（全局）、还是离轴残差（非轴向）。

**核心科学问题**：失控前的临界模态，是沿传播轴变软，还是轴向减弱、非轴向/全局模态接管。

**怎么测**：本征模在一张 m3b 率场网格上（有一条内建的 E→E 传播轴 THETA_EE）。判"轴/离轴/全局"用**模态的空间形状分数**（沿轴拉长 / 离轴拉长 / 全场均匀度 / 波矢轴对齐——都是无向的二阶矩量，M1 已有）；判"往哪个方向扰动会被放大"用**归一化方向向量**的有限时间增益/泄漏。最后用最小 rate-field 扰动 spot-check 验证这套线性化读数真能预测扰动响应。

**锁定口径（抗审稿）**：(a) **global ≠ nonaxis**——全场同步 vs 离轴残差是不同生物学解释，JSON/图必须拆 subtype。(b) **分类用二阶矩 shape 分数，不用线性投影残差能量**（后者对轴向拉长钝感、且 nonaxis 高维残差 raw energy 天然偏大 → 维度污染）。(c) **分类只在 m3b `Grid(n=6)`/THETA_EE 空间做，不碰 subject1146 电极轴**。(d) **无向即无向**：wavevector 对齐 ≠ early→late 方向。(e) 判决预注册三类 {axial_supported / off_axis_global_supported / unresolved} + 强制 subtype + base gate 前置。(f) M2b 与 M2a 矛盾 → `verdict=unresolved`, `unresolved_subreason=linear_mode_candidate_unvalidated`（**不做第四顶层 verdict**）。(g) tier=`model_side_preliminary`，从不声称"模型证明发作/CSD"。

（内部归档代号：`axis_elongation`(=elongation_axis_score)/`axis_wavevector_alignment`(=phase_gradient_axis_score, 无向)/`off_axis`/`globality`/`core_overlap`；gain 方向 `e_global`/`e_axis_gradient`/`e_nonaxis`；`nonaxis_source_policy=critical_residual_direction`|`unavailable_low_residual_energy`；verdict∈{axial_supported, off_axis_global_supported, unresolved}；subtype∈{axial_elongation, axial_wavevector, mixed_axis, global, nonaxis_residual, mixed_axis_global, mixed_axis_nonaxis, mixed_global_nonaxis, unclassified}。）

---

## 1. 核心问题 + verdict（结构化）

**Q**：穿零处临界模态的空间形状 = 轴向 / 离轴 / 全局？

**结构化 verdict（保留 M2a 信息 + 不破坏三类预注册）**：
```
linear_mode_shape_verdict   # 仅从 shape 分数：axial / off_axis / global / mixed_* / unclassified
projected_gain_verdict      # 从方向向量 gain/leak：谁被放大 / axis→X leak
perturbation_validation_status  # supportive / neutral_or_underpowered / contradicted / failed
final_verdict               # ∈ {axial_supported, off_axis_global_supported, unresolved}  ← 唯一预注册三类
subtype                     # 强制拆，global ≠ nonaxis（见 §5）
unresolved_subreason        # null | linear_mode_candidate_unvalidated | multiple_alpha0_crossings | not_cleanly_localized | ...
```
`linear_mode_candidate_unvalidated`（M2b 反驳 M2a）是 `final_verdict=unresolved` 的一个 `unresolved_subreason`，**不是**第四个顶层 verdict。

---

## 2. 基（A'，承重）— 分类用 shape 分数、gain 用方向向量

### 2.1 分类空间 = m3b `Grid(n=6)` / THETA_EE（不碰 subject1146）
本征模的 rE-loading（`mode_e_field` / 复对 `pair_loading`）在 `Grid(n=6, L)` 上；轴 = `kernels.theta = THETA_EE = π/4`。**subject1146 的 `axis_unit`/`axis_mask`/`theta_rad` 只喂仿真、产生慢状态轨迹，不进本征模分类。**

### 2.2 模态 shape 分数（复用 M1，全部无向；M2 必须显式**计算 + 持久化**）
| 代号 | M1 函数（THETA_EE） | 测 | 范围 |
|---|---|---|---|
| `axis_elongation` | `elongation_axis_score` | loading 沿 THETA_EE 的空间拉长 | [-1,1]（+沿轴 / −垂直） |
| `axis_wavevector_alignment` | `phase_gradient_axis_score` | wavevector 沿 THETA_EE（**无向**） | [-1,1] |
| `off_axis` | `off_axis_score` | 垂直 THETA_EE 的拉长功率 | [0,1] |
| `globality` | `globality` | participation ratio（1=全场均匀） | [0,1] |
| `core_overlap` | `core_overlap` | E-power 落核内比例 | [0,1] |
**P1-2**：M1 eval 当前只算 `elongation_axis`+`off_axis`、只持久化 `mode_class/core_overlap/globality`；**M2 必须补算 `axis_wavevector_alignment` 并把 5 个连续分数全部写进 JSON**（不从 `mode_class` 反推）。复对：用 invariant-subspace loading，不用单向量符号。

### 2.3 gain/leak 方向向量（分开的一套，rate-field state space，归一化）
- `e_global` = normalize(uniform over grid E cells)。
- `e_axis_gradient` = normalize(沿 THETA_EE 的线性坐标梯度 `s = x·cosθ + y·sinθ`, centered)。（这是**方向**，与 §2.2 的 shape 分数是两回事：shape 分数判形状、方向向量判扰动响应。）
- `e_nonaxis` = normalize(leading-mode rE-loading − 投到 span(e_global, e_axis_gradient) 的分量)（临界残差方向）。**低范数即 invalid**：`if ‖residual‖ < nonaxis_direction_min_norm → nonaxis_source_policy=unavailable_low_residual_energy, nonaxis gain=NaN`（**不用 random 填**；如需 control 另报 `random_nonaxis_control_gain`，非主）。
- **Full-state 嵌入**：`B_X = embed_rE(e_X)`（rE block=方向，其余 5 个 state block=0）；readout = `exp(JT)·B_X` 的 rE block 再投影。明说是 **E-rate observable subspaces**。

---

## 3. M2a — dense α₀ localization + shape 分类 + projected gain/leak

### 3.1 dense α₀ localization（残留#1 + P0-crossing/nonmonotone + quality 拆两层）
- **coarse 预扫**：M1 last-qualified↔first-saturated bracket 内先均匀取 K 个点（config，默认 5）算 α₁。若 sign change > 1 → `crossing_status=multiple_alpha0_crossings`（取第一个做定位，但 `final_verdict` 不能 supported 除非模态类在各 crossing 稳定；否则 `unresolved_subreason=multiple_alpha0_crossings`）。
- **递归二分**（线性插值 slow state q_I/g_K/h_G）：每层中点重解 low branch（`solve_branches` warm-start）、算 α₁、检查 branch identity。停止：`crossing_width_ms < crossing_width_ms_tol`（默认 1.0）或达 `max_bisect_levels = min(hard_cap=16, max(8, ceil(log2(initial_width_ms/tol))+2))`。
- **quality 拆两层（P0）**：
  - `op_solve_quality`：converged ∧ residual ok ∧ branch identity clean ∧ 非 solver artifact。
  - `stability_read_quality`：α₁<0 ∧ τ defined ∧ quasi-static recovery 可解释。
  - τ 等"恢复量"**只在 α₁<0 侧解读**。
- 输出：`alpha0_crossing_time_ms`、`alpha0_crossing_slow_state`、`crossing_width_ms`、`alpha_left/alpha_right`、`branch_identity_clean`、`op_solve_quality_left/right`、`crossing_status`。

### 3.2 mode shape 分类 at crossing
在 α₀ crossing 慢状态求 op + Jacobian + `rate_eigenpairs`；取 leading subspace（complex pair 当 2D 不变子空间 `leading_subspace_indices`）；算 §2.2 的 5 个连续 shape 分数（复对用 `pair_loading`）+ `mode_class`。全部持久化。

### 3.3 projected operator gain/leak（方向向量）
预注册 horizons（复用 M1 `[10,25,50,100,250,500]` ms）。对穿零处（+ 可选 pre/post-crossing）算 source∈{axis_gradient, global, nonaxis} × target∈{axis_gradient, global, nonaxis} 的 directional gain：
`gain_X_to_Y(T) = ‖P_Y · rE(exp(JT)·embed_rE(e_X))‖ / ‖e_X‖`（复用 M1 `transient_gain`；`e_X` 已归一）。off-diagonal 即 leak（`leak_axis_to_nonaxis` / `leak_axis_to_global` / `return_nonaxis_to_axis`）。
- axis source 就是 `e_axis_gradient`（1 个方向；shape 的 occupancy/wavevector 是分类概念、不是扰动方向）。
- nonaxis source = `e_nonaxis`（临界残差方向）；低范数 → invalid（§2.3）。artifact 明写 `nonaxis_source_policy`，**不冒充** full `‖P_N exp(JT) P_N‖`。

---

## 4. M2b — minimal rate-field perturbation spot-check（残留#2 + P1-3 硬门）

### 4.1 field_rhs 一致性硬门（P1-3，M2b 前置）
`field_rhs`（`m3b:670`）的 Jacobian 就是 `build_jacobian_dense`——但当前 `field_rhs` **缺 g_K/h_G shift**（T2.5 审查遗留 Minor）。**M2 T4 第一步**：给 `field_rhs` 补 `gK_field`/`hG_scalar`（与 `solve_operating_point` 同源），并加**硬门测试**：同一 shifted op 上，`(field_rhs(z*+εu)−field_rhs(z*−εu))/(2ε) ≈ build_jacobian_dense@u`（finite-diff JVP 匹配）。不过此门，M2b 不许跑。

### 4.2 扰动 spot-check
点：`early_stable` / `last_sampled_qualified` / `just_before_alpha0` / `just_after_alpha0`（末者若 op_solve_quality clean）。
方向（同 E-rate norm）：`critical_eigenspace` / `axis_gradient` / `global` / `nonaxis_critical_residual` / `random_orthogonal_control`。复对 `critical_eigenspace`：实向量 → ±Re(v)；复对 → 沿归一 Re(v)、Im(v)、以及最大短时 rE 增益的相位方向，summary 取 max + median。
积分器（复用 §4.1 修好的 `field_rhs`）：从 `z*+ε·v` 起步、积分、测 `observed_peak_gain`/`observed_recovery_time`/`return_to_op_success`/`escape_probability`/`observed_{axis,global,nonaxis}_leak`。**跑 ±ε**（nonlinear 可能不对称）。config：`epsilon_rel=[0.01,0.05]`、`max_time_ms=1000`、`dt_ms=0.1`、`recovery_radius_rel=0.05`、`escape_rate_khz=_SAT_RATE_KHZ`(=M1 饱和阈)、`polarities=[-1,1]`。

### 4.3 perturbation_validation_status（定量，P1/§11）
```
supportive: 预测主导方向的 observed_peak_gain 在 top-2；且不比 axis/global/nonaxis 备选低 > margin
contradicted: 预测主导方向在 observed_peak_gain 或 recovery_time 里垫底（两 ε × 两 polarity 都是）
neutral_or_underpowered: ε/polarity 间不一致 或 所有响应都小
failed: 积分数值失败 / 无有效恢复量
```
`contradicted` → `final_verdict=unresolved, unresolved_subreason=linear_mode_candidate_unvalidated`（保留 `linear_mode_shape_verdict` + `projected_gain_verdict`）。

---

## 5. 验收门（预注册，跑前锁 — "门编码结论非存在"）

### 5.0 base gate（先过，否则 unresolved）
`alpha0 crossing localized in actual slow-space`（§3.1）∧ `op_solve_quality clean on both bracketing sides` ∧ `branch identity remains low/approach near crossing`（no ambiguous）∧ `α₁ sign change bracketed`（且非 multiple crossings 未解释）∧ complex-pair policy applied ∧ 5 个 shape 分数已算。（**不要求** crossing 后侧仍 stable-recovery qualified；τ 只在 α₁<0 侧解读。）

### 5.1 shape 分类（linear_mode_shape_verdict，决策树而非能量分区）
> shape 分数量纲不同、不自然求和为 1，故用决策树 + 阈值 + sensitivity（非 raw energy dominance）。
- `global` if `globality >= global_thresh`（默认 0.5）∧ 各向异性低（`|axis_elongation|` 与 `off_axis` 都 < iso_thresh 0.2）。
- `axial` if `axis_elongation >= axis_thresh`（默认 0.3）∧ `axis_elongation − off_axis >= margin`（默认 0.15）。subtype：`axial_elongation` / `axial_wavevector`（看 `axis_wavevector_alignment` 是否也高）/ `mixed_axis`。
- `off_axis`(→nonaxis_residual) if `off_axis >= offaxis_thresh`（默认 0.3）∧ `off_axis − max(0, axis_elongation) >= margin`。
- 两条同时满足 / 边界 → `mixed_axis_global` / `mixed_axis_nonaxis` / `mixed_global_nonaxis`；都不满足 → `unclassified`。
- `threshold_sensitivity`：`global_thresh∈[0.45,0.5,0.6]`、`axis/offaxis_thresh∈[0.25,0.3,0.4]`、`margin∈[0.10,0.15,0.20]`；分类翻则文字写 "X but threshold-sensitive"。

### 5.2 final_verdict（合 shape + gain + perturbation）
- `axial_supported` iff `linear_mode_shape_verdict=axial` ∧ projected axis gain 不被 global/nonaxis gain 反驳（§5.4）∧ `perturbation_validation_status ≠ contradicted`。
- `off_axis_global_supported` iff (`linear_mode_shape_verdict ∈ {global, off_axis/nonaxis}`) or (`projected gain_global/gain_nonaxis 明显超 axis gain near α₀`) ∧ `perturbation_validation_status ≠ contradicted`。**subtype MUST split** global vs nonaxis_residual vs mixed_*。
- 否则 / 冲突 / 未定位 / multiple crossings / M2b contradicted → `unresolved`（带 subreason）。

### 5.3 unresolved（任一）
α₀ 未紧致定位 ∨ branch ambiguous ∨ multiple crossings 未解释 ∨ shape 分类 unclassified ∨ shape 与 gain 冲突（§5.4）∨ M2b contradicted ∨ shape class 在 tiny continuation 变化下不稳。

### 5.4 gain_conflict（定量，P1/§12）
`gain_conflict if`：`linear_mode_shape_verdict = X` 但 `projected self-gain / dominant response class = Y ≠ X` 且 `gain_Y − gain_X >= gain_conflict_margin`（默认 0.20）且跨 `>= gain_conflict_min_horizons`（默认 2，取近恢复尺度的 horizons）。

---

## 6. 已定残留 + 评审 required edits（全折入）
1. densification：递归二分，停 `crossing_width<1.0ms` 或动态 `max_bisect_levels`（hard_cap 16）；每层重解 + branch/quality check + coarse 预扫多穿零。
2. perturbation integrator：复用 **修好的 `field_rhs`**（§4.1，含 g_K/h_G shift + JVP 硬门）；config 数值全给（§4.2）。
3. substrate：`subject1146`（同 M1 轨迹）。
4. **A'**：THETA_EE 二阶矩 shape 分类 + 方向向量 gain；不碰 subject1146 电极轴。
5. `axis_signed` → `axis_wavevector_alignment`（**无向**）；禁 "early→late signed"；directional 需另做（deferred）。
6. **持久化 5 个连续 shape 分数**（非只 mode_class）。
7. base gate 拆 `op_solve_quality` / `stability_read_quality`；α₀ 后侧不要求 stable-recovery qualified。
8. `linear_mode_candidate_unvalidated` = `unresolved` subreason（非第四顶层）。
9. nonaxis 方向低范数 → invalid（非高维 raw energy）。
10. gain_conflict + perturbation_validation 定量化；basis sanity 图。

---

## 7. 红线 / tier
`model_side_preliminary`；单本征值≠发作；global runaway≠真发作；谱=机制地图；**禁"模型证明发作/CSD"**；**禁"真数据"（用 actual v2.2 SIMULATION trajectory）**；**global ≠ nonaxis**；分类用二阶矩 shape 分数非线性投影残差；**wavevector 对齐 ≠ early→late 方向**；分类只在 THETA_EE 网格空间、不碰 subject1146 电极轴；M2b 矛盾→unresolved(subreason)。
**DEFER（非 M2）**：directional early→late（相位/时序定义）；SNN perturbation；slow-var attribution；full controls；Topic5 correspondence → M2.5 / M3。

---

## 8. config / results / 复用
- **config** 新 `config/topic4_criticality_m2.yaml`：`basis`(theta=THETA_EE, embedding=rE_block, nonaxis_direction_min_norm=1e-3)；`densification`(coarse_K=5, crossing_width_ms_tol=1.0, max_bisect_hard_cap=16)；`gain`(horizons 复用 M1, gain_conflict_margin=0.20, gain_conflict_min_horizons=2)；`perturbation`(epsilon_rel=[0.01,0.05], max_time_ms=1000, dt_ms=0.1, recovery_radius_rel=0.05, escape_rate_khz=_SAT_RATE_KHZ, polarities=[-1,1])；`classify`(global_thresh=0.5, iso_thresh=0.2, axis_thresh=0.3, offaxis_thresh=0.3, margin=0.15, sweep grids)。
- **results** `results/topic4_criticality_m2/`：`dense_trajectory_verdict.json`（结构化，§1）、`alpha0_crossing.json`、`continuation_trace.csv`、`mode_decomposition.json`、`perturbation_spotcheck.json`、`figures/`(+ README 中文 + FIGURE_INDEX)。
- **复用 M1**：`solve_branches`/`build_jacobian_dense`/`rate_eigenpairs`/`leading_subspace_indices`/`pair_loading`/`mode_e_field`/`elongation_axis_score`/`off_axis_score`/`phase_gradient_axis_score`/`globality`/`core_overlap`/`transient_gain`/`check_low_branch_continuation_between`/`qualify_point`/`field_rhs`(补 shift)。

---

## 9. tasks 概览（plan 细化，逐步 TDD + 门编码为阈值）
- **T0** config + gain 方向向量基（`e_global`/`e_axis_gradient`/`e_nonaxis`，归一 + 低范数 invalid）+ **shape-score sanity tests**：①合成 loading 沿 THETA_EE 拉长 → `axis_elongation` 高、`off_axis` 低；②垂直拉长 → `off_axis` 高；③各向同性/全局 → `globality` 高、axis/offaxis 不乱报。
- **T1** dense α₀ localization（coarse 预扫多穿零 + 递归二分 + quality 两层 + base gate + crossing outputs）。
- **T2** shape 分类 at crossing（补算 `axis_wavevector_alignment`；持久化 5 连续分数；复对 invariant-subspace）。
- **T3** projected gain/leak（方向向量, transient_gain, `nonaxis_source_policy`, gain_conflict）。
- **T4** `field_rhs` 补 g_K/h_G shift + **JVP 硬门** → rate-field perturbation integrator + spot-check（±ε, 复对方向, perturbation_validation_status）。
- **T5** verdict（base gate + 决策树分类 + subtype + sensitivity + gain_conflict + M2b-contradiction → 结构化 JSON §1）+ 图（穿零处 shape 分数条 + gain 矩阵 + τ-crossing + **basis sanity panel**：axis 方向/e_axis_gradient/critical loading/critical residual）+ README 中文。

---

## 10. Self-Review
1. **Placeholder**：阈值全在 §5/§8 锁；无 TBD。
2. **一致性**：A'（shape 分类 vs 方向 gain）贯穿 §2/§3/§5；`axis_wavevector_alignment` 无向、§1/§2/§7 一致；`linear_mode_candidate_unvalidated`=subreason 非顶层（§1/§4/§5）；global≠nonaxis 强制拆（§1/§5）。
3. **Scope**：单 spec T0-T5 model-side；directional early→late / SNN / attribution / controls / Topic5 defer。
4. **Ambiguity**：分类空间=THETA_EE 网格（非 subject1146）；shape 分数≠扰动方向；nonaxis 方向低范数 invalid；field_rhs shift + JVP 硬门；分类用决策树+sensitivity（非能量分区）。
5. **评审闭环**：P1-1（rename+无向）、P1-2（持久化连续分数）、P1-3（field_rhs JVP 硬门）+ 10 required edits 全折入 §2/§4/§5/§6。
