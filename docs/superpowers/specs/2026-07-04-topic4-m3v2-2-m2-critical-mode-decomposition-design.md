# Topic 4 — M3-v2.2 Approach-Criticality — Milestone 2: Dense α₀ Crossing + Two-Stage Linear-Ignition / Nonlinear-Spread Readout · Design

date 2026-07-04（rev2.3 2026-07-05）· 状态 **design rev2.3 — in execution (SDD)**（rev2.2 → rev2.3（T3 review）：off_axis sentinel 用**渐近尾 horizon 一致规则**（`sentinel_min_horizon_ms=250`，§2.3）替代单一 horizon；gain 形式 **sign-off = full-state self-gain**（M1 precedent，§3.3），E-rate-projected 精细化 deferred）（rev1.1 → rev2：两段式 ignition/spread 重构；rev2 → rev2.1：折入用户 GO-review 4 项 P1 契约钉死——① pilot 语言降调（exploratory de-risk，正式待 T2/T4/T5 注册）② nonaxis gain 三态阈值 ③ epsilon pass/fail 锁死 ④ ignition/spread enum 显式化；**rev2.1 → rev2.2**（T1 review，用户 2026-07-05）：`op_solve_quality` 改 **fold-appropriate 残差容差**（`op_residual_tol=1e-2`）而非严格 1e-9 `converged`（near-fold op 稳定但不收敛到 1e-9；否则 §5.0 ignition gate 恒 fail 废掉 core_localized 读数）+ `branch_identity_clean` 纳入 T1）· 分支 `topic4-criticality-m2`（worktree, base `codex/topic4-criticality`@1207e85, off M1）· 前置：**M1 (frozen-Jacobian verdict instrument) COMPLETE** — real-v2.2 verdict=`unresolved_operating_point`（低支 α₁ 在两抽样点间穿 0 至 +0.189，采样漏采）。**用户 2026-07-05 审阅=90/100 GO、无 P0；建议 M1 PR #6 先合、M2 impl 以 M1-merged main 为 base（避免 rebase 混科学+工程）。**

> **方法学 base** = M1 spec `2026-07-02-topic4-m3v2-2-approach-criticality-design.md` + M1 code。M2 **复用** M1 判读器 + 特征模指标，只新加：dense α₀ localization、two-core ignition 确认、nonlinear-footprint spread readout、gain/leak 方向向量（nonaxis 降为 sentinel）。
> **执行 gate**：M2 全模型侧，不消费 topic5 phase2。Topic5 correspondence 留 M3。
> **rev2 pilot 依据（exploratory de-risk scouts，方向承重）**：见 `docs/superpowers/specs/2026-07-04-topic4-m2-pilot-findings.md` §6（round-1）+ §7（round-2）。两轮 pilot 是**探索性 de-risk scout、非正式 milestone 结论**——**在当前衬底、当前 pilot 范围内稳定指向**：穿零处线性临界模态 core-localized（核心点火），两核条件下也 symmetry-break 到单核、走廊全暗；"往哪扩"要看非线性足迹、稳定指向沿轴起→全场收/自限、`off_axis≈0`（未见侧向）。**正式结论待 T2/T4/T5 注册流程确认**。据此判断 rev1.1 的"线性临界模态 = axial/global/nonaxis 三分类当主 verdict"是**切错层级**，rev2 改两段式。
> **A'（核实后锁定，rev2 role 调整）**：本征模住在 **m3b 率场网格 `Grid(n=6)`（单核, THETA_EE=π/4 轴）**，**不在** subject1146 SNN/电极空间。THETA_EE 二阶矩 shape 分数（复用 M1）在 rev2 里**不再当轴/全局/离轴分类器**，而是：core_overlap+globality → ignition class；axis_elongation/off_axis/wavevector → ignition 模态**描述量 + off_axis sentinel**。**`phase_gradient_axis_score` 是无向 wavevector 轴对齐、不是 early→late signed**（`|F|²`+`cos(2(β−θ))` 对 180° 对称）——禁写 "signed early→late"。

---

## 0. 摘要（朴素话）+ 锁定口径

M1 的尺子在真实 v2.2 **仿真**轨迹上说"看不清"：抽样快照上恢复速率 α₁ 还明显为负，但分支延续补检发现低支 α₁ 在两个抽样点之间穿过 0（+0.189）。**这个"看不清"（CSD 欠采样）结论 M2 不改**——M2 只把穿零点加密定位出来，并新增一套**两段式判读**回答两个**不同层级**的问题：

1. **线性临界模态判"在哪点火"**（ignition locus）。pilot（探索性 de-risk 范围内）**稳定指向**：**`core_localized`**（缩在被去抑制的核心里），两核条件下也 symmetry-break 到单核、中间走廊全暗（双几何一致）。**升为一等、可报告字段**——但它**不取代** M1 的 CSD `unresolved`，两个读数并存；**正式 core_localized 结论待 T2 注册流程确认**。
2. **非线性足迹判"往哪扩"**（spread character）。pilot（探索性 de-risk 范围内）**稳定指向**：**沿轴起（`onset=axial`）→ 要么自限缩回、要么低幅全场招募（`endgame=self_limited | global_flooding`），`off_axis` 全程≈0（未见侧向）**，且自限还是漫开**取决于过临界多深**（`depth_dependent`）。这是"往哪扩"的**主裁**，从线性模态形状挪到非线性足迹；**正式 spread 结论待 T4/T5 注册流程确认**。

**核心科学问题（rev2）**：失控 = **在哪点火** + **往哪扩** 两问。线性模态答前者（核心点火）；非线性足迹答后者（沿轴→全局/自限）。原来想在**一个线性模态**上贴 axial/global/nonaxis 标签是切错层级——那个模态在 pilot 范围内稳定是核心点火，会稳健地对不上、落 unclassified、把真答案藏起来。

**锁定口径（抗审稿）**：
- (a) **两段式**：`linear_ignition`（点火位置）与 `nonlinear_spread`（铺开方式）是两个独立读数；**不**把 spread 判读贴到线性模态上。
- (b) **`linear_ignition=core_localized` 是新增字段、不取代 M1 CSD `unresolved`**。M2 输出同时带 `csd_verdict=unresolved_operating_point`（M1 结论不变）与 `linear_ignition_mode=core_localized`（新）。
- (c) **nonaxis 从主判据降为 sentinel / negative-control**（`off_axis: absent/present`）。若算 `e_nonaxis gain`，**必须注明**它在 core-localized 模态里代表"核心紧致残差"、**不是**侧向传播（防与数据侧"离轴/非轴向招募"混写）。
- (d) **"沿轴 vs 全场"是时间相位、不是单帧/单本征模的 static 标签**：`spread_onset` / `spread_endgame` 分两个字段沿 t 报。
- (e) **global ≠ nonaxis** 仍拆；但两者在本模型侧都被 pilot 证伪为"主铺开方向"（线性核紧致、非线性 off_axis≈0）。
- (f) **near-fold caveat（承重）**：两核 own-crossing 的 α₁=+0.189 **不是 α₀≈0 临界形状**，是 fold 后第一段正值。two-core 确认的精确陈述 = "two-core low-branch crossing / near-fold 后 leading mode 仍 core-localized"，**不写成**精确临界形状。
- (g) **对称去抑制近似（承重）**：两核共享一个 `q_core` 标量。故 two-core 证的是"**即使给轴向双核机会，也不自然铺走廊**"，**不是** subject1146 真实双源慢变量完全复现。
- (h) **shape 分数不当分类器**：core_overlap+globality 判 ignition class；axis_elongation 的核内符号**不稳**（单核 +0.55 / 双核 −0.99 / 足迹 +0.1..0.8 抖），只作描述量、不承重。
- (i) tier=`model_side_preliminary`，从不声称"模型证明发作/CSD"；global runaway ≠ 真发作。

（内部归档代号：`axis_elongation`(=elongation_axis_score)/`axis_wavevector_alignment`(=phase_gradient_axis_score, 无向)/`off_axis`/`globality`/`core_overlap`；ignition `class`∈{core_localized, delocalized, ambiguous}、`delocalized_subtype`∈{corridor_lit, global_like, multi_core}；spread `onset`∈{axial, core_only, global_first, off_axis, undetermined}/`endgame`∈{self_limited, global_flooding, marginal, undetermined}/`off_axis`∈{absent, present, undetermined}/`depth_dependent`/`epsilon_sensitivity`∈{pass, epsilon_sensitive}；gain 方向 `e_global`/`e_axis_gradient`/`e_nonaxis(sentinel)`；`csd_verdict=unresolved_operating_point`(M1, 不变)。）

---

## 1. 两段式 verdict（结构化）+ M1 CSD 并存

**Q**：失控前 (1) 在哪点火？(2) 往哪扩？

**结构化 verdict（rev2 两段式）**：
```
csd_verdict                  # = unresolved_operating_point  ← M1 结论，M2 不改，仅并存展示
                             #   (α₀ 穿零欠采样；M2 dense-localize 但不翻此结论)

linear_ignition:             # 「在哪点火」— 冻结-Jacobian 临界模态 @ α₀ crossing（单核网格）
  class                      # core_localized(primary) | delocalized | ambiguous
  delocalized_subtype        # null(if core_localized) | corridor_lit | global_like | multi_core
  core_overlap               # 高 → 局域（pilot: 0.98–0.995）
  globality                  # 低 → 局域（pilot: 0.03–0.06）
  two_core_symmetry_break    # bool：two-core 确认——模态缩单核、走廊暗
  corridor_power             # ≈0 确认非轴向走廊（pilot: 0.000）
  shape_descriptors:         # 次级描述量（承重 caveat：核内 elong 符号不稳，仅描述）
    axis_elongation          # 描述用，不承重（sign unstable on coarse grid）
    off_axis                 # sentinel：应 ≈0
    axis_wavevector_alignment
  near_fold_note             # two-core crossing α₁ 为 fold 后首段正值、非精确 α₀≈0

nonlinear_spread:            # 「往哪扩」— field_rhs 非线性足迹积分（spread 主裁）
  onset                      # axial | core_only | global_first | off_axis | undetermined
  endgame                    # self_limited | global_flooding | marginal | undetermined
  off_axis                   # absent(sentinel: off_axis<tol 全程) | present | undetermined
  depth_dependent            # bool：自限 vs 漫开取决于过临界深度
  footprint_trajectory       # active_frac(t)/core_overlap(t)/elongation(t)/off_axis(t)/globality(t)
  control_minus_kick         # bool：已扣 v=0 控制残漂（必 true 否则该段 undetermined）
  epsilon_sensitivity        # pass | epsilon_sensitive（§4.3 pass/fail 锁死）

interpretation               # 自然语言合成，e.g.
                             # "core ignition followed by axial transient and possible global endgame"

base_gate_passed             # bool（§5.0）
unresolved_subreason         # null | alpha0_not_localized | branch_ambiguous |
                             #   multiple_alpha0_crossings | jvp_gate_failed |
                             #   ignition_not_localized | unresolved_nonlinear_spread
```

**要点**：
- `csd_verdict` 与 `linear_ignition`/`nonlinear_spread` **是三个不同问题**——CSD（α₁ 是否平滑穿零）、ignition（在哪点火）、spread（往哪扩）。M2 回答后两个、并存展示第一个（不改）。
- **不再有** `final_verdict ∈ {axial_supported / off_axis_global_supported}` 这个线性三分类顶层——它被两段式取代。
- ignition 或 spread 任一段过不了 base/quality 门 → 该段落 `class/onset=…undetermined`，`unresolved_subreason` 记原因；**另一段仍可报**（两段解耦）。

---

## 2. 基（A'，rev2 role 调整）— shape 分数作描述量+sentinel、gain 方向 nonaxis 降 sentinel

### 2.1 分析空间 = m3b `Grid(n=6)` / THETA_EE（不碰 subject1146）
本征模的 rE-loading（`mode_e_field` / 复对 `pair_loading`）在 `Grid(n=6, L)` 上；轴 = `kernels.theta = THETA_EE = π/4`。**subject1146 的 `axis_unit`/`axis_mask`/`theta_rad` 只喂仿真、产生慢状态轨迹，不进本征模分析。**

### 2.2 模态 shape 分数（复用 M1，全部无向；M2 计算 + 持久化，但 role = 描述量/sentinel 非分类器）
| 代号 | M1 函数（THETA_EE） | 测 | rev2 role |
|---|---|---|---|
| `core_overlap` | `core_overlap` | E-power 落核内比例 | **ignition class 主判据**（高→局域）|
| `globality` | `globality` | participation ratio（1=全场均匀）| **ignition class 主判据**（低→局域）|
| `off_axis` | `off_axis_score` | 垂直 THETA_EE 的拉长功率 | **sentinel**（应≈0；线性&非线性都作离轴 negative-control）|
| `axis_elongation` | `elongation_axis_score` | loading 沿 THETA_EE 的拉长 | **描述量，不承重**（核内符号不稳）|
| `axis_wavevector_alignment` | `phase_gradient_axis_score` | wavevector 沿 THETA_EE（**无向**）| 描述量 |
**持久化**：5 个连续分数全部写进 JSON（不从 class 反推）。复对：用 invariant-subspace loading（`pair_loading`），不用单向量符号。

### 2.3 gain/leak 方向向量（rate-field state space，归一化；nonaxis 降 sentinel）
- `e_global` = normalize(uniform over grid E cells)。
- `e_axis_gradient` = normalize(沿 THETA_EE 的线性坐标梯度 `s = x·cosθ + y·sinθ`, centered)。
- `e_nonaxis` = normalize(leading-mode rE-loading − 投到 span(e_global, e_axis_gradient) 的分量)（临界残差方向）。**rev2 role = sentinel / negative-control**（阈值全 config，跑前锁 §8）：
  - **`off_axis` 三态判据（钉死）**：`present` **仅当**两门都破——`off_axis_score ≥ off_axis_score_tol`（默认 0.05）**且** nonaxis gain 显著超 axis/global（`gain_nonaxis − max(gain_axis, gain_global) ≥ nonaxis_gain_excess_tol`（默认 0.10）**且** `gain_nonaxis / max(gain_axis, gain_global) ≥ nonaxis_gain_ratio_tol`（默认 1.25））；两门都不破 → `absent`；只破一门 / 边界 → `undetermined`。
  - **horizon 稳健化（rev2.3，T3 review）**：gain 门在**渐近尾 horizons**（config `sentinel_min_horizon_ms=250` 及以上，即 [250,500]）各评一次 `_off_axis_decision`；**尾内全一致** → 该结局，不一致 → `undetermined`。短 horizons（10–100ms）处紧致 `e_nonaxis` 会瞬态 out-gain axis/global = §2.3 已警告的"核心紧致误读为传播"伪迹，故 sentinel **只读渐近尾且要求尾内一致**（镜像 §4.3 epsilon_sensitivity 的 across-sweep 一致规则），避免单一任意 horizon 的脆弱性 / result-shopping。
  - **未达"present"两门时，只能写 `off_axis: absent` 或 `undetermined`，禁写任何侧向/离轴传播结论**（防 §6.3 pronoun 误扩）。
  - **若报 `e_nonaxis gain`，JSON+图必须并写注记**：`"nonaxis_residual = core-compactness residual in a core-localized mode, NOT sideways propagation"`。
  - 低范数即 invalid：`if ‖residual‖ < nonaxis_direction_min_norm(1e-3) → off_axis=absent, nonaxis gain=NaN`（不用 random 填；如需 control 另报 `random_nonaxis_control_gain`）。
- **Full-state 嵌入**：`B_X = embed_rE(e_X)`（rE block=方向，其余 5 state block=0）；readout = `exp(JT)·B_X` 的 rE block 再投影。明说是 **E-rate observable subspaces**。

---

## 3. M2a — dense α₀ localization + linear_ignition readout

### 3.1 dense α₀ localization（carried verbatim from rev1.1；穿零真实、欠采样、需定位）
- **coarse 预扫**：M1 last-qualified↔first-saturated bracket 内先均匀取 K 个点（config，默认 5）算 α₁。若 sign change > 1 → `crossing_status=multiple_alpha0_crossings`（取第一个做定位；`unresolved_subreason=multiple_alpha0_crossings` 除非 ignition class 在各 crossing 稳定）。
- **递归二分**（线性插值 slow state q_I/g_K/h_G）：每层中点重解 low branch（`solve_branches` warm-start）、算 α₁、检查 branch identity。停止：`crossing_width_ms < crossing_width_ms_tol`（默认 1.0）或达 `max_bisect_levels = min(hard_cap=16, max(8, ceil(log2(initial_width_ms/tol))+2))`。
- **quality 拆两层（carried）**：
  - `op_solve_quality`：`residual ≤ op_residual_tol`（**fold-appropriate 残差容差，非严格 converged 旗**——穿零 near-fold op 的解达不到 solver 内部 1e-9 `converged` 硬线（残差~1e-3，pilot 已知性质）但 α₁/模态读数跨 bracket 稳定；故用残差容差 gate 掉真发散解、保住 near-fold 穿零可报 core_localized。rev2.2 决定，用户 2026-07-05）∧ ¬saturated ∧ resolved spectrum ∧ `branch_identity_clean`（低支跨穿零 bracket 连续，用 M1 `check_low_branch_continuation_between`）∧ 非 solver artifact。
  - `stability_read_quality`：α₁<0 ∧ τ defined ∧ quasi-static recovery 可解释。τ 等"恢复量"**只在 α₁<0 侧解读**。
- 输出：`alpha0_crossing_time_ms`、`alpha0_crossing_slow_state`、`crossing_width_ms`、`alpha_left/alpha_right`、`branch_identity_clean`、`op_solve_quality_left/right`、`crossing_status`。

### 3.2 linear_ignition readout at crossing（rev2：ignition 非分类）
在 α₀ crossing 慢状态求 op + Jacobian + `rate_eigenpairs`；取 leading subspace（complex pair 当 2D 不变子空间 `leading_subspace_indices`）；算 §2.2 的 5 个连续 shape 分数（复对用 `pair_loading`）。
- **ignition class**：`core_localized` if `core_overlap ≥ core_localized_overlap_thresh`（默认 0.8）∧ `globality ≤ core_localized_globality_thresh`（默认 0.3）；否则 `delocalized` / `ambiguous`（见 §5.1）。
- **two-core 确认（新）**：用 `make_core_mask(kind="two", radius=0.9, separation=2.4)`（两核沿 THETA_EE、中间走廊在轴上），在 two-core **自己的** low-branch crossing/near-fold 慢状态重解 + 取 leading mode，算逐区功率（coreA/coreB/corridor）。`two_core_symmetry_break = (max(coreA,coreB) ≥ two_core_single_core_thresh 0.9) ∧ (corridor_power ≤ corridor_dark_thresh 0.05)`。**near_fold_note 必写**：该点 α₁ 为 fold 后首段正值、非精确 α₀≈0（§0-f）；对称去抑制近似（§0-g）。
- 全部持久化到 `linear_ignition`（§1）。

### 3.3 projected operator gain/leak（方向向量；nonaxis=sentinel）
预注册 horizons（复用 M1 `[10,25,50,100,250,500]` ms）。对穿零处算各方向 source∈{axis_gradient, global, nonaxis(sentinel)} 的 **full-state self-gain**：
`gain_X(T) = ‖exp(JT)·embed_rE(e_X)‖ / ‖e_X‖`（**复用 M1 `transient_gain`，全 6-field 态范数——与 M1 `finite_time_gain` 同惯例**）。
- **gain 形式 sign-off（rev2.3，spec-owner，T3 review）**：spec 早稿写过 E-rate-projected `‖P_Y·rE(…)‖` 形式，但 sentinel 是 **negative-control**（主判据 = `off_axis_score` 门；gain 门为**次级相对比较**）；full-state self-gain 足够且与 M1 precedent 一致，采纳之。**E-rate-block-projected 精细化 = DEFERRED**——仅当未来某穿零 `off_axis_score ≥ off_axis_score_tol` 打开 score 门、gain 门真正承重时再评（届时紧致 `e_nonaxis` 与弥散 `e_global` 向非-rE 场的耦合差异才可能影响比较公平性）。
- `e_nonaxis` gain **只作 sentinel**（§2.3）：报 `off_axis: absent/present` + 强制注记"核心紧致残差非侧向传播"；**不**冒充 full `‖P_N exp(JT) P_N‖`、**不**当主铺开判据。

---

## 4. M2b — nonlinear-footprint spread readout（rev2：spread 主裁，升级）

### 4.1 field_rhs 一致性硬门（P1-3，carried，M2b 前置）
`field_rhs`（`m3b:670`）的 Jacobian 就是 `build_jacobian_dense`——但当前 `field_rhs` **缺 g_K/h_G shift**（T2.5 审查遗留 Minor）。**M2 T4 第一步**：给 `field_rhs` 补 `gK_field`/`hG_scalar`（与 `solve_operating_point` 同源），并加**硬门测试**：同一 shifted op 上，`(field_rhs(z*+εu)−field_rhs(z*−εu))/(2ε) ≈ build_jacobian_dense@u`（finite-diff JVP 匹配）。**不过此门，M2b 不许跑**（`unresolved_subreason=jvp_gate_failed`）。pilot 确认本轨迹 hG≈0、`field_rhs` shift-gap 不阻塞，但正确性要求此门（M2b 是通用工具）。

### 4.2 footprint 积分 = spread 主裁（rev2 升级）
- **点**：`at_crossing`（α₀）+ `just_past`（过 fold 一点）**至少两个过临界深度**（判 `depth_dependent`）；可选 `just_before`。
- **扰动方向**（同 E-rate norm）：`core_kick`（核内单位扰动）为主；`critical_eigenspace`（leading Re(v)）+ `random_orthogonal_control` 为辅。复对：沿归一 Re(v)、Im(v)、最大短时 rE 增益相位方向，summary 取 max+median。
- **积分器**（复用 §4.1 修好的 `field_rhs`）：从 `z*+ε·v` 起步、前向积分，**同跑 v=0 控制、报 kick−control 的 δrE(t)**（扣工作点自身残漂——近 fold op-solve 不完全收敛 `fixedpoint_residual≈1–4e-3）。**跑 ±ε**（nonlinear 不对称）。
- **每采样时刻算 footprint 量**：`active_frac`、`core_overlap`、`elongation_axis`、`off_axis`、`globality`、`peak_δRE`；escape = max rE > `_SAT_RATE_KHZ`。
- config：`epsilon_rel=[0.01,0.05]`、`max_time_ms=300`（自限/漫开在此尺度已分清，pilot）、`dt_ms=0.1`、`recovery_radius_rel=0.05`、`escape_rate_khz=_SAT_RATE_KHZ`、`polarities=[-1,1]`、`footprint_sample_ms=[2,5,10,20,30,50,75,100,200,300]`。

### 4.3 nonlinear_spread verdict（从 footprint 读，定量；enum + pass/fail 全钉死）
- **onset ∈ {axial, core_only, global_first, off_axis, undetermined}**（每组 ε×polarity 各判一次）：
  - `axial` if 扩张窗（`active_frac` 上升段）内 `elongation_axis > axial_onset_thresh`（默认 0.2）∧ `off_axis < off_axis_score_tol`（默认 0.05）；
  - `core_only` if `active_frac` 无显著上升（峰值 − 初值 < `expand_active_delta`（默认 0.1））；
  - `global_first` if 早期 `globality` 即高（≥ `global_thresh` 0.5）且无轴向优先（`elongation_axis ≤ axial_onset_thresh`）；
  - `off_axis` if `off_axis ≥ off_axis_score_tol`（sentinel 破，§2.3 门）；
  - 否则 → `undetermined`。
- **endgame ∈ {self_limited, global_flooding, marginal, undetermined}**：`global_flooding` if `active_frac(t_max) ≥ flood_active_thresh`（默认 0.9）；`self_limited` if 峰后 `active_frac ≤ self_limit_active_thresh`（默认 0.1）且未 escape；两者皆不满足 → `marginal`。
- **off_axis ∈ {absent, present, undetermined}**：`absent` if `off_axis` 全程 < `off_axis_score_tol`；`present`/`undetermined` 按 §2.3 两门规则（未破两门禁写传播结论）。
- **depth_dependent**：≥2 深度的 endgame 不同（如 at_crossing 自限、just_past 漫开）→ true。
- **epsilon_sensitivity（pass/fail 锁死）**：跑全 `epsilon_rel=[0.01,0.05]` × `polarities=[-1,+1]`（4 组）。**pass** = `onset` ∧ `off_axis` 在 4 组里**全一致**（`epsilon_onset_agreement=all`）**且** `endgame` **majority ≥3/4 一致**（`epsilon_endgame_agreement=majority`；endgame 允许 marginal 抖动）；否则 **fail** → `epsilon_sensitivity=epsilon_sensitive`，`nonlinear_spread` 段判 undetermined、`unresolved_subreason=unresolved_nonlinear_spread`（ignition 段不受影响仍报）。
- **control_minus_kick**：必 true（否则 spread undetermined）。

---

## 5. 验收门（预注册，跑前锁 — "门编码结论非存在"）

### 5.0 base gate（先过，否则该段 undetermined）
- **ignition 段**：`alpha0 crossing localized in actual slow-space`（§3.1）∧ `op_solve_quality clean on crossing side` ∧ `branch identity remains low/approach near crossing` ∧ 5 个 shape 分数已算。
- **spread 段**：`field_rhs JVP 硬门 pass`（§4.1）∧ `control_minus_kick` ∧ 积分数值未 fail。

### 5.1 linear_ignition gate（rev2：ignition class，非轴/全局/离轴分类）
- `core_localized` iff `core_overlap ≥ 0.8` ∧ `globality ≤ 0.3`（阈见 §8）。
- `two_core_symmetry_break` iff 单核占比 ≥ 0.9 ∧ `corridor_power ≤ 0.05`（§3.2）。
- `off_axis`（ignition sentinel）：`absent` iff `off_axis_score < 0.05`。
- `delocalized` if `globality ≥ 0.5` ∧ core_overlap 低；`delocalized_subtype`：`corridor_lit`（两核间走廊功率 ≥ `corridor_lit_thresh` 0.2）/ `global_like`（globality 高 ∧ `|axis_elongation|`+`off_axis` 都 < iso_thresh 0.2）/ `multi_core`（loading 有 ≥2 个分离核峰）。`ambiguous` 否则。
- `ignition_sensitivity`：`core_localized_overlap_thresh∈[0.7,0.8,0.9]`、`globality_thresh∈[0.2,0.3,0.4]`；class 翻则文字写 "core_localized but threshold-sensitive"。
- **near_fold_note / 对称去抑制近似** 必随 two_core_symmetry_break 一并写（§0-f/g）。

### 5.2 nonlinear_spread gate（rev2：spread 主裁 = footprint）
- `onset` / `endgame` / `off_axis` / `depth_dependent` 按 §4.3 定量阈判。
- 要求：`control_minus_kick=true` ∧ `epsilon_sensitivity=pass`（§4.3 全一致/majority）∧ ≥2 深度已跑。任一不满足 → spread 段 undetermined（`unresolved_subreason=unresolved_nonlinear_spread`），**但 ignition 段仍报**。

### 5.3 interpretation 合成
`interpretation` 由 `linear_ignition.class` + `nonlinear_spread.{onset,endgame,off_axis}` 机械拼：
`"{class} ignition followed by {onset} transient and {endgame}; off_axis {off_axis}"`。**禁**把 spread 结论回贴到 linear mode（"临界模态是 axial"）；**禁**把 core_localized 说成"取代 CSD unresolved"。

### 5.4 undetermined（任一段）
ignition：α₀ 未紧致定位 ∨ branch ambiguous ∨ multiple crossings 未解释 ∨ core_overlap/globality 边界不稳。
spread：JVP 门 fail ∨ control 未扣 ∨ ε/polarity 不一致 ∨ <2 深度 ∨ 积分 fail。
两段解耦：一段 undetermined 不拖累另一段。

---

## 6. 已定改动 + pilot 折入（rev2）
1. **两段式 verdict**：`csd_verdict`(M1, 不变) + `linear_ignition`(点火) + `nonlinear_spread`(铺开) + `interpretation`；弃线性 {axial/global/nonaxis} 顶层三分类。
2. **linear_ignition = core_localized 一等字段**（core_overlap+globality）+ **two-core 对称破缺确认**（走廊暗）；**不取代 M1 CSD unresolved**。
3. **nonlinear_spread 主裁 = field_rhs 足迹**（onset/endgame/off_axis/depth_dependent + control-minus-kick + ε 敏感）；spread 判读从线性模态挪走。
4. **nonaxis 降 sentinel / negative-control**（`off_axis: absent/present`）+ 强制"核心紧致残差非侧向传播"注记。
5. **"沿轴 vs 全场" = 时间相位**（onset/endgame 两字段），非单帧 static 标签。
6. **near-fold caveat**（two-core α₁=+0.189 非 α₀≈0）+ **对称去抑制近似** 必写。
7. dense α₀ localization（§3.1）、base gate quality 两层拆、complex-pair invariant-subspace、5 连续 shape 分数持久化、field_rhs shift + JVP 硬门 —— 全 carried from rev1.1。
8. `axis_signed` → `axis_wavevector_alignment`（无向）；禁 "early→late signed"；directional 需另做（deferred）。

---

## 7. 红线 / tier
`model_side_preliminary`；单本征值≠发作；global runaway≠真发作；谱=机制地图；**禁把 round-1/round-2 pilot 当正式 milestone 结论**（探索性 de-risk scout；core_localized / axial-onset 正式结论待 T2/T4/T5 注册流程确认）；**禁"模型证明发作/CSD"**；**禁"真数据"（用 actual v2.2 SIMULATION trajectory）**；**禁"core_localized 取代/翻转 M1 CSD unresolved"**（并存，非替换）；**禁把 spread 结论回贴线性模态**；**禁"放弃 nonaxis"**（是降 sentinel/negative-control）；**禁把 two-core α₁=+0.189 写成精确 α₀≈0 临界形状**（fold 后首段正值）；**禁"two-core 复现 subject1146 真实双源慢变量"**（对称去抑制近似）；**global ≠ nonaxis**；**wavevector 对齐 ≠ early→late 方向**；分类/模态只在 THETA_EE 网格空间、不碰 subject1146 电极轴。
**与数据的关系（边界）**：本模型侧支持 **core ignition → axial transient → global/endgame**；**不支持**真正 sideways/off-axis 传播。**若真实数据关键现象是离轴招募 → 当前衬底未解释完**，后续动 structural scaffold / D_EE / subject-specific geometry，**非**继续调这个线性分类器。
**DEFER（非 M2）**：directional early→late（相位/时序定义）；SNN perturbation；slow-var attribution；full controls；Topic5 correspondence → M2.5 / M3。

---

## 8. config / results / 复用
- **config** 新 `config/topic4_criticality_m2.yaml`：
  - `basis`(theta=THETA_EE, embedding=rE_block, nonaxis_direction_min_norm=1e-3, **off_axis_score_tol=0.05, nonaxis_gain_excess_tol=0.10, nonaxis_gain_ratio_tol=1.25**)；
  - `densification`(coarse_K=5, crossing_width_ms_tol=1.0, max_bisect_hard_cap=16, **op_residual_tol=1.0e-2**（fold-appropriate；pilot near-fold 残差 1e-3–4e-3 过关、真发散解仍拦；rev2.2）)；
  - `ignition`(core_localized_overlap_thresh=0.8, core_localized_globality_thresh=0.3, sweep [0.7,0.8,0.9]×[0.2,0.3,0.4]; **delocalized**: globality_thresh=0.5, iso_thresh=0.2, corridor_lit_thresh=0.2)；
  - `two_core_confirm`(kind=two, radius=0.9, separation=2.4, single_core_thresh=0.9, corridor_dark_thresh=0.05)；
  - `spread`(axial_onset_thresh=0.2, **expand_active_delta=0.1, global_thresh=0.5**, flood_active_thresh=0.9, self_limit_active_thresh=0.1, footprint_sample_ms=[2,5,10,20,30,50,75,100,200,300]; **epsilon_onset_agreement=all, epsilon_endgame_agreement=majority**)；
  - `gain`(horizons 复用 M1；**sentinel_min_horizon_ms=250**（off_axis sentinel 只读 ≥ 此的渐近尾 horizons + 要求尾内一致，rev2.3）)；
  - `perturbation`(epsilon_rel=[0.01,0.05], max_time_ms=300, dt_ms=0.1, recovery_radius_rel=0.05, escape_rate_khz=_SAT_RATE_KHZ, polarities=[-1,1])。
- **results** `results/topic4_criticality_m2/`：`ignition_spread_verdict.json`（两段式，§1）、`alpha0_crossing.json`、`continuation_trace.csv`、`linear_ignition.json`（含 two-core 逐区）、`nonlinear_spread.json`（footprint 轨迹）、`figures/`(+ README 中文 + FIGURE_INDEX)。
- **复用 M1**：`solve_branches`/`build_jacobian_dense`/`rate_eigenpairs`/`leading_subspace_indices`/`pair_loading`/`mode_e_field`/`elongation_axis_score`/`off_axis_score`/`phase_gradient_axis_score`/`globality`/`core_overlap`/`transient_gain`/`check_low_branch_continuation_between`/`qualify_point`/`make_core_mask`/`field_rhs`(补 shift)。pilot 脚本 `results/topic4_criticality_m2/pilots/m2_pilots{,_round2}.py` 是 ignition/spread 读数的 reference 实现。

---

## 9. tasks 概览（plan 细化，逐步 TDD + 门编码为阈值）
- **T0** config + gain 方向向量基（`e_global`/`e_axis_gradient`/`e_nonaxis` sentinel，归一 + 低范数 invalid）+ **shape-score sanity tests**：①合成 loading 沿 THETA_EE 拉长 → `axis_elongation` 高、`off_axis` 低；②垂直拉长 → `off_axis` 高（sentinel present）；③各向同性/全局 → `globality` 高。
- **T1** dense α₀ localization（coarse 预扫多穿零 + 递归二分 + quality 两层 + base gate + crossing outputs）。carried from rev1.1。
- **T2** linear_ignition readout（crossing 单核 mode → core_overlap/globality → class；补算持久化 5 连续分数；复对 invariant-subspace）+ **two-core 对称破缺确认**（逐区功率、走廊暗、near_fold_note、对称近似注记）。
- **T3** projected gain/leak（`e_axis_gradient`/`e_global` gain；`e_nonaxis` **只作 sentinel** + 强制注记）。
- **T4** `field_rhs` 补 g_K/h_G shift + **JVP 硬门（前置，不过不跑）** → nonlinear-footprint integrator + spread readout。**spread 主裁**。**跑前锁死**：4 组 `epsilon_rel×polarity` 全跑、`onset`/`endgame`/`off_axis` enum（§4.3）、`epsilon_sensitivity` pass/fail（onset+off_axis 全一致 ∧ endgame majority ≥3/4；fail→`unresolved_nonlinear_spread`）、control-minus-kick 必扣、≥2 过临界深度判 depth_dependent。
- **T5** 两段式 verdict（base gate 两段 + ignition class + two-core 确认 + spread footprint 判读 + interpretation 机械合成 + sensitivity → 结构化 JSON §1，并存 `csd_verdict`）+ 图（**ignition panel**：crossing mode loading + two-core 逐区 + core_overlap/globality；**spread panel**：footprint active_frac(t)/off_axis(t)/elongation(t) 多深度；basis sanity）+ README 中文。

---

## 10. Self-Review
1. **Placeholder**：阈值全在 §5/§8 锁；无 TBD。
2. **一致性**：两段式（ignition/spread）贯穿 §0/§1/§3/§4/§5；`csd_verdict` 并存不改（§0-b/§1/§7）；nonaxis=sentinel（§0-c/§2.3/§3.3/§5.1/§7）；沿轴vs全局=时间相位（§0-d/§4.3/§5.3）；near-fold + 对称近似（§0-f/g/§3.2/§5.1/§7）；`axis_wavevector_alignment` 无向（§2.2/§7）。
3. **Scope**：单 spec T0-T5 model-side；directional early→late / SNN / attribution / controls / Topic5 defer。
4. **Ambiguity**：分析空间=THETA_EE 网格（非 subject1146）；shape 分数=描述量/sentinel 非分类器；ignition class 由 core_overlap+globality；spread 由 footprint；nonaxis 低范数 invalid；field_rhs shift + JVP 硬门；两段解耦（一段 undetermined 不拖另一段）。
5. **pilot 闭环**：round-1（core-localized headline）+ round-2（two-core 稳健 + footprint axial→global never off-axis）作为**探索性 de-risk 方向证据**折入 §0/§1/§3/§4/§5/§7（正式结论待 T2/T4/T5 注册）；用户 2026-07-05 两轮审阅（5 改动 + 精确 wording + 4 项 P1 契约钉死）全折入。
