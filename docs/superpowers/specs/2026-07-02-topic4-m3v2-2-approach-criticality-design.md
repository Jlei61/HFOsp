# Topic 4 — M3-v2.2 approach-to-runaway 临界性：谱相图 + 轨迹 overlay（path a）· Design

date 2026-07-02 · 状态：design **rev2**（并入 review 6 阻断点 + 命名/度量 + provenance + SNN spot-check + 3×3 correspondence + virtual-SEEG estimator 合同 + Topic5 接口向量）· 分支 `codex/topic4-criticality`（worktree，base `codex/topic4-m3a-v2-2`@e01c08b）· rev1 已 commit `6c376d0`

> **方法学 base = M3B-next 设计** `docs/superpowers/specs/2026-06-27-sef-hfo-m3b-spectral-phase-map-design.md`；机器 `src/topic4_m3b_spectral_phase.py`。本 spec 应用到 M3-v2.2 approach-to-runaway 回答"临界慢化"，**执行只认 rev2**。
> **执行 gate**：T4（correspondence）等 topic5 phase2；T1–T3d 不消费 phase2，可并行 prep（模型侧，不写 correspondence 结论）。

> **代码现实（已核实，承重——分清"接线" vs"需扩 M3B 模块"）**：
> - **`J` 是连续时间**（`build_jacobian_dense` 对角 `−1/TAU_ME`、`−1/TAU_AMPA`；matvec = `dz/dt`）→ `α₁=Re(λ₁)` **per-ms**，`τ=−1/α₁` ms。**operator_type=continuous_jacobian 已确认**。
> - **`solve_operating_point` 现无 init 参数**（单一 `mean_field` 播种）→ **branch 协议（阻断 4）需给 solver 加 warm-start `(rE,rI)` init**（扩展，非接线）。
> - **`leading_rate_eigenpair` 只取单个右本征向量**（无 left-eigvec、无 complex-pair loading、无 next-distinct gap）→ **阻断 5 + core_controllability 需扩本征对/度量层**（扩展）。
> - **已存在可复用**：`finite_time_gain`、dt-independent `residual`、`saturated`/`converged`、rate-branch 本征选择、`core_overlap`/`globality`。**需新建**：slow-var 有限差分、mode-shaped 扰动注入器（spot-check）、virtual-SEEG estimator import（复用 topic5）。

---

## 0. 摘要（朴素话）

我们看 M3-v2.2 从**安静**逼近**全场失控**那条路上，系统"受一下扰动后要多久平复"是不是越来越长（临界慢化）——**不预设它会**。**怎么测**：不用方差/自相关代理（数据侧才用），而是每个慢状态点线性化、**直接算**领头本征值实部 `α₁`；且**只在系统确实停在准静态平衡点附近、能被线性化的点上读**（不然算子不是真恢复算子）；有固定核、无平移对称，**不能用 Brunel 平面波**，整张异质 Jacobian 数值求本征模。**揭示什么**：三种预注册结局（平滑变脆 / 硬跳 / 不可测），若变脆是哪个模式（局部/沿轴/全局，M3B 示全局）、哪个慢变量推、是否伴非正规瞬态放大；再与 topic5 phase2 对照，分清"模型能算的本体"和"稀疏电极代理看不看得到"。

**锁定口径（rev2，抗审稿）**：本 spec **不把 `α₁→0` 当预期结论**，而作三类预注册 verdict 之一。`smooth_CSD` 要求在 **branch-aware、quasi-static、非饱和** operating point 上，**continuous-time** leading real-part eigenvalue 平滑接近 0，**且经 SNN 扰动 spot-check 支持**。`hard_jump_no_CSD` 要求**最后合格 low-branch 点仍与 0 有明确 margin**、随后 simulation 进入 saturated/runaway、**且 branch continuation 未发现被跳过的 low-branch `α₁≈0`**。若 operating point / branch identity / adiabatic 条件不成立，判 `unresolved_operating_point`，**不当阴性**。

（内部归档代号：`operator_type∈{continuous_jacobian,discrete_transition}`；`α₁=Re(λ₁)` per-ms / `ρ=max|eig(A)|` / `α=log ρ/dt`；verdict∈{smooth_CSD,hard_jump_no_CSD,unresolved_operating_point}；quality gate converged∧¬saturated∧residual_rms∧rate_mismatch_rel∧slow_mismatch_rel∧adiabatic_index；branch∈{low,high,saturated,ambiguous}；`finite_time_gain`/numerical_abscissa；`∂α₁/∂{q_I,g_K,h_G}`；mode_observability=‖G_seeg·v‖/‖v‖；`classify_mode` line954 saturated⇒runaway。）

---

## 1. Q1/Q2 + verdict + 报告量（命名已修）

**Q1（三类预注册 verdict）**：`smooth_CSD` / `hard_jump_no_CSD` / `unresolved_operating_point`（定义见 §0 锁定口径 + §4 严格判据）。

**报告量（命名修 3.1/3.2）**：
- `alpha1_closest_to_zero_pre_onset`（= onset 前合格点里 α₁ 的 **max**，因 α₁<0 时最接近 0 是最大值；**不叫 min**）
- `last_stable_alpha1`；`jump_distance_to_alpha0 = abs(last_stable_alpha1)`（正数）
- `n_qualified_points` / `qualified_fraction`
- `tau_ms = −1/α₁` **仅当 α₁<0**；`α₁≥0 → tau=NaN`，另存 `instability_growth_time = 1/α₁ (α₁>0)`

**Q2（哪个 mode / 谁推 / 有无非正规放大）**：mode class（§6）；slow-var 归因（§9）；非正规瞬态（§7）。

---

## 2. operator 单位合同（阻断 1，承重）

主指标 `α₁` **仅指 continuous-time frozen-Jacobian 的 leading real-part eigenvalue**（已核实本仓 `J` 连续时间，`α₁` per-ms）。若某实现输入是**离散更新矩阵** `A`：`ρ=max|eig(A)|`；`α=log(ρ)/dt`；`τ=−1/α`。`τ=−1/α` **只在 α<0 且 continuous-time normalized 后**定义。每个 spectrum artifact **强制存**：`operator_type`、`dt_ms`、`eig_raw`、`alpha1_per_ms`、`alpha1_per_s`、`tau_ms`、`tau_s`、`stability_margin`。**理由**：模型 α₁ / 数据 VAR λmax / DMD eigenvalue 都叫"leading eigenvalue"但单位不同，不锁死会在 correspondence 里静默混淆。

---

## 3. operating-point 质量门（阻断 2，承重）

每个轨迹点存：`op_status`、`converged`、`residual_rms`、`rate_mismatch_rms`、`rate_mismatch_rel`、`slow_mismatch_rel`、`slow_speed`、`adiabatic_index`（或 `alpha_drift_index`）、`saturated`。

- **归一化 mismatch（非裸 L2）**：`rate_mismatch_rms=sqrt(mean((rate_sim−z_star)²))`；`rate_mismatch_rel=rate_mismatch_rms/(median(|z_star|)+eps)`。
- **slow_mismatch**：op 在某 slow-state 上解出；若 `q_I/g_K/h_G` 读数 ≠ sim state → `slow_mismatch_rel` 超标即不合格（只比 rate 不够）。
- **adiabatic gate**：`slow_speed=‖d slow/dt‖`；`adiabatic_index=slow_speed·tau_fast/slow_scale`（`tau_fast=−1/α₁, α₁<0`）或 `alpha_drift_index=|dα₁/dt|/(α₁²+eps)`。太大 → 系统被 ramp 拖着跑，**不是**准静态小扰动恢复 → 标 `trajectory_not_quasistatic`。

**qualified = converged ∧ ¬saturated ∧ residual_rms<res_tol ∧ rate_mismatch_rel<mismatch_tol ∧ slow_mismatch_rel<slow_mismatch_tol ∧ adiabatic_index<adiabatic_tol**。**CSD 趋势只读 qualified 点**；不合格点带 reason（`nonconverged/saturated/high_residual/rate_mismatch/slow_mismatch/not_quasistatic`），归入 `trajectory_not_linearizable`（≠ `hard_jump_no_CSD`：前者测不了、后者测得了且没慢化）。

---

## 4. hard_jump_no_CSD 严格判据（阻断 3）

`hard_jump_no_CSD` **iff**：last `N_pre` qualified **low-branch** 点存在 ∧ `last_stable_alpha1 < −alpha_margin_hard` ∧ 无 α₁ 趋势达 `alpha_near_zero_tol` ∧ sim 在 `jump_window` 内进入 saturated/runaway ∧ **branch continuation 确认未跳过合格 low-branch α₁≈0 点**。否则（solver 直接跳 high-branch、或轨迹已离 op）→ `unresolved_operating_point`，**不判 hard jump**。

---

## 5. branch-aware operating point（阻断 4，需扩 solver）

M3B 带饱和 + 高/低放电支 → 多平衡点。**需给 `solve_operating_point` 加 init 参数**（现无），每格/每轨迹点从 `{low_rate, previous_point, high_rate, random_small(opt)}` 求解 → 按 rate 距离聚类 → 标 `{low_branch, high_branch, saturated_branch, ambiguous_branch}`。每点存 `n_branches_found/branch_id/branch_rate_mean/branch_saturated/branch_alpha1/branch_residual/branch_selected_reason`。**CSD 主分析只读 `low_branch`/`approach_branch`**。low branch 消失而 high branch 稳定 = fold/hysteresis/hard-jump 关键信息。

---

## 6. 本征模 + mode class（阻断 5，需扩本征对层）

- **complex conjugate pair 当一个实不变 2D 子空间**：`mode_loading_i=sqrt(|v_pair1_i|²+|v_pair2_i|²)`；mode class（`core_overlap`/`axis_score`/`globality`）算在**非负 loading / 不变子空间能量**上，**不用**本征向量正负号。
- **谱隙**：`alpha_gap = alpha1 − alpha_next_distinct`；`alpha_next_distinct` **不能**是 conjugate pair 的另一成员（否则 gap 人工=0）。
- **left eigenvector**（需新建）：`core_controllability = |ψ_m^T b_core|`（核扰动能否激发该模）。

---

## 7. non-normality 必报（阻断 6，进 T3a 验收）

稳定系统（所有 Re(λ)<0）仍可因非正规性短时放大。**必报**：`numerical_abscissa = max eig((J+Jᵀ)/2)`；`henrici_departure` 或 `commutator_norm`；`G_T=‖exp(J·T)‖₂`，`G_max` over `T∈{10,25,50,100,250,500}ms`，`T_at_G_max`。verdict 加 secondary tag `transient_amplification_present`。四类机制解释：

| α₁ | finite-time gain | 机制 |
|---|---|---|
| →0 | 高 | smooth CSD + 非正规放大 |
| →0 | 低 | 经典 smooth CSD |
| <0 | 高 | stable-but-amplifying / 非正规 approach-to-runaway |
| <0 | 低 | hard jump / 无被动预警 |

**对 Topic5 关键**：数据侧 VAR λmax 阴性**不排除**模型侧短时 gain 或 axis→nonaxis 瞬态放大。

---

## 8. 两 prereq + provenance（§2 + §5 硬化）

- **P1 v2.2→interface export**：`build_handoff_from_sim(sim, events, dt_ms, mapping_id="m3a_v2_2_approach", gk_enabled=..., hG_enabled=...)`+`write_handoff_artifacts`；照 `run_a2_axisbreak_sweep.py`。**T1 两层 fail-closed**：fixture **必**过 `phase_map_trajectory`；真实 v2.2 不过 sign-cal/rate-matched/Gate A → `refused/mechanism_candidate_only`+原因，**不放水**。
- **P2 normalized grid（合同 D1）**：轴 normalized `phase_x_core×phase_y_global∈[0,1]`，`axes_built_from_slow_to_rate_mapping_id`=P1 同 id。**若 `phase_x×phase_y` 不能唯一定 op（因 h_G 被投影）→ atlas 明确命名 `conditional_2d_atlas_at_phase_recovery=…`，不冒充完整 phase map。**
- **provenance（必存）**：P1 `mapping_id/mapping_hash/source_branch/source_commit/sim_config_hash/events_hash/phase_coord_ranges_hash/gk_enabled/hG_enabled/slow_var_names/slow_var_units/rate_transform/axis_normalization_version`；P2 `axes_built_from_slow_to_rate_mapping_id/_hash/phase_{x,y,recovery}_definition/grid_extent/grid_resolution`。

---

## 9. slow-var 归因：partial + trajectory contribution（§4 两层）

- **A local partial sensitivity**：`∂α₁/∂q_I`、`∂α₁/∂g_K`、`∂α₁/∂h_G`，**central difference** `(α(x+δ)−α(x−δ))/(2δ)`，**两侧 op 都 qualified** 否则该导数 `invalid`。
- **B trajectory contribution**：`contrib_x=(∂α₁/∂x)·(dx/ds)` → 答"哪个慢变量**实际**把系统推向 softening/hard-jump"（敏感度大但轨迹上变化小 ≠ 实际贡献大）。
- **Figure**：`α₁(s)` + partial sensitivities(s) + trajectory contributions(s)。比单纯 facet 更直接（facet 作 sensitivity 补充）。

---

## 10. α₁=0 contour mask 纪律（3.3）

相图三层 mask：`qualified_low_branch` / `saturated` / `nonconverged_or_invalid`。**`α₁=0` contour 只画在 `qualified_low_branch` 内**（否则 contour 穿 saturated/nonconverged 区，看似机制边界实为 solver artifact）。

---

## 11. SNN spot-check（§6，T3b，需 mode-shaped 扰动注入器）

frozen-Jacobian 是否真解释原模型的扰动恢复——**审稿必问**。每条轨迹选 4 类点：`early_stable` / `closest_to_zero` / `last_qualified` / `post_jump_saturated`（仅描述，不做线性恢复）。每 qualified 点沿 `{leading, axis, nonaxis/global, random_orthogonal}` mode **扰动**（需新建 mode-shaped 注入器：按 mode loading 空间加权注入 δ），SNN 量 `observed_recovery_time/observed_peak_gain/return_to_op_success/nonlinear_escape_probability`。**验收**：`predicted_tau` vs `observed_recovery_time` 单调；`predicted_finite_time_gain` vs `observed_peak_gain` 正相关。

---

## 12. 控制（三类 + branch/ramp 两类，§9）

已有 `no-core homogeneous` / `isotropic AR=1` / `shuffled core thresholds`；**加** `branch-control`（同格 low/high init 求解，验 α₁ 图非 solver-init artifact；出 low/high/selected atlas）+ `ramp-rate control`（slow/original/fast ramp：smooth_CSD 若本体，慢 ramp 更易见 α₁→0 + τ↑；原 ramp 太快则本体有 softening 但轨迹代理看不到 → 解释 `unresolved`/`not_quasistatic`）。

---

## 13. correspondence 3×3（阻断 §7，含 unresolved 单列）

| 模型 verdict | 数据代理 阳性 | 数据代理 阴性 | 数据代理 不可判读 |
|---|---|---|---|
| **smooth_CSD** | 可观测 CSD 支持 | 需 virtual-SEEG 判可检测性 | 数据不足 |
| **hard_jump_no_CSD** | 数据另机制 / 代理 confound | 一致支持短预警窗 / 硬跳 | 数据不足 |
| **unresolved_operating_point** | 不做 correspondence | 不做 correspondence | 不做 correspondence |

`unresolved` **不是**阴性、**不并入** "模型无 α₁→0"；它=不可测。

---

## 14. virtual-SEEG proxy（阻断 §8，复用 topic5 同一 estimator）

**必须**调用 topic5 phase2 **同一** AR1/VAR-λmax/branching/avalanche code path，同 window/sampling/envelope/channel-count/contact-mask/surrogate/subject-level aggregation——**不允许**写"模型侧简化版"（否则分不清差异来自模型/观测层/estimator 实现）。三读出：`source_all_nodes_proxy` / `virtual_SEEG_all_available_contacts_proxy` / `virtual_SEEG_matched_10ch_proxy`（**matched_10ch 必须有**，Topic5 就是稀疏 SEEG：模型本体 α₁→0 但 10ch 看不到 → 数据阴性不能反驳模型）。**mode observability**：`mode_observability=‖G_seeg·v_mode‖/‖v_mode‖`（`axis_/global_/nonaxis_`），`G_seeg`=虚拟-SEEG gain matrix → 可明说"模型有 smooth CSD 但 leading mode 对当前 montage 低可观测，故数据代理阴性可解释"。
*（依赖：import topic5 estimator code；T4 gated on phase2，届时该 code 应可得。）*

---

## 15. Topic5 接口预测向量（§13，喂数据侧下一轮 axial-weakening + non-axis-amplification）

模型侧除 mode class 外，输出可直接喂 Topic5 的：`model_leading_mode_loading` / `model_axis_projection` / `model_nonaxis_projection` / `model_globality` / `model_axis_to_nonaxis_gain` / `model_nonaxis_gain` / `model_mode_observability_under_SEEG`。使 Topic5 用同一 `e_axis`/`e_nonaxis` 测"模型预测的可放大 mode 是否在真实 SEEG 的 non-axis criticality 指标里出现"（否则模型说 global、数据说 non-axis 难对齐）。

---

## 16. 红线（继承 M3B §10 + rev2）

单本征值≠发作；`α₁>0` 不证临床发作；runaway≠真发作；谱相图=机制地图，SNN=表型检验；有核不声称平面波 k 模；CSD 在已知会失稳系统里出现=本体度量成立非"证明预测发作"；**不预设 α₁→0**（runaway 是饱和归类，α₁ 可仍为负）。

---

## 17. Tasks（rev2 结构；标注 wiring vs 扩展）

- **T0** config + terminology lock（operator/units、verdict/quality-gate/branch/finite-time-gain/h_G-step/virtual-SEEG estimator 合同；§18 YAML）。
- **T1**（wiring）v2.2→interface export；fixture 必过 `phase_map_trajectory`；真实 fail-closed；provenance hash。
- **T2**（扩：normalized 轴）normalized phase grid，mapping_id/hash 对齐 T1；2D atlas + phase_recovery/h_G conditional 层；invalid/saturated/nonconverged masks。
- **T3a**（扩：solver init + 本征层 + 非正规）branch-aware op（**加 solve_operating_point init**）+ qualified low-branch mask（§3）+ `α₁/gap/mode/finite-time-gain/numerical_abscissa`（§6/§7）+ verdict（§4）。**Figure**：`α₁=0`+mode-class 相图 + v2.2 overlay。
- **T3b**（新建：mode-shaped 注入器）linear-response SNN spot-check（§11）。
- **T3c**（新建：finite-diff）slow-var attribution partial + trajectory contribution（§9）。
- **T3d** controls（no-core/isotropic/shuffled-core/branch-control/ramp-rate，§12）。
- **T4**（gated on phase2）3×3 correspondence（§13）+ virtual-SEEG proxy 复用 topic5 estimator（§14）+ mode observability + Topic5 接口向量（§15）+ honest-ceiling 文本。
- **milestone 建议**：T1+T2+T3a = 第一里程碑（先出 branch-aware 相图 + verdict），再 T3b/c/d；避免 all-of-T3 才有结果。**依赖**：T1‖(T3c/T3b 的纯函数部分)；T3a 依赖 T1+T2；T4 依赖 T3 + phase2。

---

## 18. T0 config（`config/topic4_criticality.yaml`）

```yaml
operator: {type: continuous_jacobian, dt_ms: null, alpha_units: per_ms, tau_units: ms}
quality_gate:
  residual_rms_tol: ...        # dt-independent max |rhs|
  rate_mismatch_rel_tol: ...
  slow_mismatch_rel_tol: ...
  adiabatic_index_tol: ...
  alpha_drift_index_tol: ...
  min_qualified_points: ...
  min_qualified_fraction: ...
verdict:
  alpha_near_zero_tol: ...
  alpha_margin_hard: ...
  jump_window_ms: ...
  smooth_min_tau_growth_ratio: ...
  smooth_min_alpha_spearman: ...
  unresolved_if_branch_ambiguous: true
branching:
  solve_inits: [low_rate, previous_point, high_rate, random_small]
  branch_cluster_rate_tol: ...
  selected_branch: approach_low_branch
mode:
  complex_pair_policy: invariant_subspace_loading
  axis_score_definition: ...
  globality_definition: participation_ratio
  core_overlap_definition: ...
  spectral_gap_policy: next_distinct_real_part
finite_time_gain: {horizons_ms: [10,25,50,100,250,500], norm: weighted_l2, report_numerical_abscissa: true}
slow_sensitivity:
  finite_difference: central
  step_fraction_qI: ...
  step_fraction_gK: ...
  step_fraction_hG: ...
  require_both_sides_qualified: true
virtual_seeg:
  use_topic5_estimator_code: true
  channel_sets: [source_all_nodes, virtual_all_contacts, matched_10ch]
  same_windows_as_topic5: true
  same_surrogates_as_topic5: true
```

---

## 19. 执行 gate / results / commit

- **gate**：T4 等 topic5 phase2；T1–T3d 并行 prep（模型侧、不写 correspondence 结论）。
- **results**：新 `results/topic4_criticality/`（引用旧 `results/topic4_sef_hfo/m3b_spectral_phase_map/` 作 provenance，T4 correspondence 不埋回 M3B 目录）。
- **commit**：rev1 已 commit（`6c376d0`）；本 rev2 随即 commit；执行只认 rev2。

---

## 20. Self-Review

1. Placeholder：所有 tol/阈值在 §18 config 锁；`...` 是 config 待标定值非遗漏。**OK**。
2. 一致性：operator 单位（§2）confirmed continuous；quality gate（§3）用现有 residual/saturated/converged + 新 rate/slow/adiabatic；hard_jump（§4）需 branch continuation（§5）；complex/left（§6）+ 非正规（§7）标注需扩；correspondence 3×3（§13）unresolved 单列；virtual-SEEG（§14）复用 topic5；verdict 三类中性（§0/§1）grounded classify_mode。**OK**。
3. Scope：单 plan T0–T4（wiring + M3B 模块定点扩展，已在"代码现实"标清）；不含数据侧（topic5 拥有）、不含 reduced/Epileptor（retracted）。**OK**。
4. Ambiguity：α₁=continuous per-ms；不预设 α₁→0；branch-aware low-branch only；complex-pair invariant-subspace loading；correspondence 分本体/代理/不可测。**OK**。
