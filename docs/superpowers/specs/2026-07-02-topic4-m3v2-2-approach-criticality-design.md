# Topic 4 — M3-v2.2 approach-to-runaway 临界性：谱相图 + 轨迹 overlay（path a）· Design

date 2026-07-02 · 状态：design **post-review rev1**（并入 P1-1..P1-4）· 分支 `codex/topic4-criticality`（worktree，base `codex/topic4-m3a-v2-2`@e01c08b）· 未 commit

> **方法学 base = 现有 M3B-next 设计**：`docs/superpowers/specs/2026-06-27-sef-hfo-m3b-spectral-phase-map-design.md`（冻结-Jacobian 谱相图 + M3A→M3B overlay 合同）。机器已实现于 `src/topic4_m3b_spectral_phase.py`；本 spec 只把它**应用到 M3-v2.2 approach-to-runaway 轨迹**回答"临界慢化"问题。
> **rev1 review 修（承重）**：(P1-1) `α₁→0` **不预设**为 runaway 机制——预注册三类 verdict；(P1-2) 冻结-Jacobian 加 operating-point 质量门；(P1-3) `h_G`/recovery 不能被 2D 图投影掉——加 facet 或 `∂α₁/∂slow` 导数；(P1-4) T1 两层 fail-closed + correspondence 改 2×2 + virtual-SEEG proxy 可检测性对照。
> **执行 gate**：T4（correspondence）**等 topic5 phase2** 落地。T1–T3（两 prereq + 相图 + CSD 读数）不消费 phase2，可并行 prep。

---

## 0. 摘要（朴素话）

我们看 M3-v2.2 从**安静**逼近**全场一起猛烈放电、停不下来**（下称"失控"）那条路上，系统"受一下扰动后要多久才平复"这个恢复时间，是不是越来越长（临界慢化）——**但我们不预设它一定会**。

**怎么测**：不用"活动抖动变大/自相关变大"去**估**（那是数据侧的代理），而是**直接算**——把这张带固定低阈值核的有限网络，在逼近路上每个慢状态点线性化，求领头本征值实部 `α₁`（恢复时间 `τ=−1/α₁`）。**两条纪律**：① **只在系统确实停在某个平衡点附近、能被线性化的点上读**（不然线性化出来的算子不是真正的"受扰恢复"算子，见 §1 质量门）；② 有固定核 → 无平移对称 → **不能用 Brunel 平面波**，必须整张异质 Jacobian 数值求本征模。

**揭示什么**：**预注册三种结局**——平滑变脆（`α₁` 平滑趋 0=临界慢化）、**硬跳**（工作点非线性饱和、直接跳到高放电支，`α₁` 还没到 0 就跳过去=没有慢化）、不可线性化（逼近太快/无稳定平衡点）。**关键背景**：M3B 已知"失控格是按**放电饱和**归类的、那里 `α₁` 可仍为负"（`classify_mode`：`saturated⇒runaway`，与 `α₁` 符号无关；且有 `α₁<0` 的点正是间期自限瞬态）——所以**硬跳是活跃假设，不是要去证 `α₁→0`**。若真有变脆：是哪个空间模式（局部/沿轴/**全局**，M3B 示全局）、哪个慢变量（`q_I`/`g_K`/`h_G`）在推。与 topic5 phase2 对照时，还要分清"模型能算的本体"和"数据只能用的代理"能不能在稀疏电极上被看到。

（内部归档代号：runaway=`saturated` op（`_SAT_RATE_KHZ`）；frozen-Jacobian `α₁=Re(λ₁)`/`τ=−1/α₁`；verdict∈{smooth_CSD, hard_jump_no_CSD, unresolved_operating_point}；op 质量门 op_status/converged/residual/`‖rate_sim−z_star‖`/saturated；mode class local/axial/global；`∂α₁/∂{q_I,g_K,h_G}`；M3A→M3B interface D1 normalized phase grid；`build_handoff_from_sim`；`classify_mode` line 954；α₁<0 interictal transient line 1548。）

---

## 1. 核心问题 + 预注册 verdict + operating-point 质量门

**Q1（有没有 / 哪一类）** — **预注册三类 verdict（P1-1，不预设 α₁→0）**：
- `smooth_CSD`：合格点上 `α₁(s)` 平滑趋 0、`τ(s)` 发散 = 临界慢化本体。
- `hard_jump_no_CSD`：工作点非线性饱和/跳到高放电支时 `α₁` 仍明显 < 0（跳前没趋 0）= 硬跳、被动预警窗≈0。
- `unresolved_operating_point`：approach 期无稳定平衡点可线性化（见质量门）。

**报告量（不只"是否单调趋 0"，P1-1）**：`min_alpha1_pre_onset`（onset 前 α₁ 最接近 0 的值）、`last_stable_alpha1`（最后一个合格点的 α₁）、`jump_distance_to_alpha0`（跳变发生时 α₁ 离 0 还有多远）、`n_qualified_points`。

**Q2（哪个 mode / 谁推）**：`α₁→0`（若有）的是哪个 mode（`core_overlap`/`axis_score`/`globality` 判 local/axial/global）；谱隙 `α₁−α₂` 是否收缩；非正规 `finite_time_gain` 沿轴如何；**哪个 slow-var 推 α₁**（见 §3 h_G 层）。

**operating-point 质量门（P1-2，承重）**：每个轨迹点存 `op_status`、`converged`、fixed-point `residual`、`rate_mismatch=‖rate_sim(t)−z_star‖`、`saturated`（这些字段 `OperatingPoint` 已有：converged/residual/saturated）。**CSD 只在 quasi-static 合格点读**：`converged ∧ ¬saturated ∧ residual<res_tol ∧ rate_mismatch<mismatch_tol`。不合格点标 `trajectory_not_linearizable`，**不进** α₁ 趋势（ramp/onset 期系统远离不动点时，围绕 z_star 的 J ≠ 真实扰动恢复算子）。`trajectory_not_linearizable` 与 `hard_jump_no_CSD` 是**不同** verdict（前者=测不了，后者=测得了且没慢化）。

---

## 2. 两个 prereq（path a 真实工作量；含 T1 fail-closed）

现有 M3B 相图 **不是 overlay-ready**：`build_m3b_spectral_outputs.py` 是 **raw-knob atlas**（`mu_core×q_global`），显式标 `m3a_overlay_consumable=False`（合同 D1 要 normalized grid）。故 path (a) 需：

- **P1 — M3-v2.2 → interface export（复用现成 exporter）**：v2.2 continuous 协议（`_simulate_continuous`/`run_transition`）→ `sim+events` → `src/sef_hfo_m3a_export.py::build_handoff_from_sim(sim, events, dt_ms, mapping_id="m3a_v2_2_approach", gk_enabled=...)` + `write_handoff_artifacts(...)`（产 5 件合同 artifact），照 `scripts/run_a2_axisbreak_sweep.py` 调法。
  - **T1 两层验收（P1-4，fail-closed）**：(a) **fixture** 必须能拿到 `overlay_verdict==phase_map_trajectory`（证机器通）；(b) **真实 v2.2 artifact** 若过不了 sign-cal / rate-matched control / Gate A，**必须 fail-closed**：`overlay_verdict∈{mechanism_candidate_only, refused}` + 写明阻断原因，**绝不放水**降 gate 去凑 T3。
- **P2 — normalized phase grid 重建（合同 D1）**：相图轴改 normalized `phase_x_core × phase_y_global ∈ [0,1]`（extent=`phase_coord_ranges.json`），`axes_built_from_slow_to_rate_mapping_id` = P1 同一 mapping id；`m3a_overlay_consumable=True`。

**两 prereq 都不消费 phase2** → 可并行 prep。

---

## 3. 谱相图 + 轨迹 overlay + h_G 层（path a 主体）

1. 用 `topic4_m3b_spectral_phase.py`（`solve_operating_point`/`build_jacobian_dense`/`rate_eigenpairs`/`spectral_gap`/`finite_time_gain`/`core_overlap`/`globality`/`classify_mode`）在 **normalized** 网格建相图；每格存领头 5–10 eigenpair + `α₁=0` 等值线 + mode class + 谱隙 + finite-time-gain。
2. overlay P1 的 `phase_trajectory.csv`（**仅当** T1 真实 gate 通过；否则不画正式 overlay，只保留 mechanism-candidate 说明 — P1-4 fail-closed）。
3. 沿合格轨迹点读 Q1 verdict + 报告量 + Q2 mode。
4. **h_G / recovery 层（P1-3，承重）** — 2D 图 `phase_x_core×phase_y_global` 会把 recovery 投影掉，而 `h_G` 正是 v2.2 核心新增；故加**最小一层**二选一（或都做）：
   - **facet**：在几个固定 `phase_recovery`/`h_G` 档位重算相图；
   - **导数（更直接归因）**：沿合格轨迹点有限差分 `∂α₁/∂q_I`、`∂α₁/∂g_K`、`∂α₁/∂h_G`（各扰一个 slow-var 重解 op + 重算 α₁），报最大 `|∂α₁/∂·|` 与符号 = **谁推临界 / 谁改软硬**。否则只答 q/global 轨迹，答不了 h_G。
5. **控制**（M3B design §7）：no-core homogeneous、isotropic `AR=1`、shuffled core thresholds 作 CSD 归因坏数据回归（至少这三）。

---

## 4. 与 topic5 phase2 的 correspondence（2×2，分本体 vs 代理可检测性 — P1-4）

数据侧（phase2，并行 session，目前偏阴）用**代理**（variance/AR1/VAR λmax/branching）；模型侧用**本体**（frozen-Jacobian `α₁`/mode）。**不能只按同阴/同阳判"对应"**，必须分清模型本体与数据代理的**可观测性**：

| | 数据代理 阳性 | 数据代理 阴性 |
|---|---|---|
| **模型 `α₁→0`** | 支持"可观测的全局模临界慢化" | 观测窗/SEEG 通道/代理不够敏感 **或** 模型-数据错位 → **必须**跑 virtual-SEEG proxy 对照才能分辨 |
| **模型 无 `α₁→0`** | 真数据可能走另一机制 / 代理受 confound | 一致阴性 → 支持硬跳变 / 短预警窗 |

**virtual-SEEG proxy 可检测性对照（P1-4，T4 内小对照）**：在**同一条**模型轨迹上，把活动经虚拟 SEEG 读出层（M3B Round-1 已有）→ 跑**数据侧那套** AR1/VAR-λmax/branching estimator → 看模型的 `α₁→0`（若有）在 ~10ch 包络代理上**看不看得到**。这把"模型本体有 vs 数据代理测得到"分开，让 2×2 可判读。**这不是**之前 retract 的大 program，只是 T4 里的一个观测性对照。

---

## 5. 红线 / honest ceilings（继承 M3B design §10）

- 单个本征值 ≠ 发作；`α₁>0` 不证临床发作起始；runaway ≠ 真发作。
- 谱相图=机制地图，SNN 行为才是表型检验（`α₁→0` 预测须 SNN spot-check 佐证）。
- 有核 → 不声称平面波 `k` 模解释固定核事件。
- CSD 在已知会失稳系统里出现 = 本体度量成立，**非**"证明模型预测发作"。
- **不预设 α₁→0**（P1-1）：runaway 在 M3B 是**饱和**归类，`α₁` 可仍为负；三类 verdict 中性预注册。

---

## 6. Tasks（lean；机器已存在，主要 wiring + 应用）

- **T0**：本 spec + `config/topic4_criticality.yaml`（相图网格、mapping_id、`res_tol`/`mismatch_tol`、verdict 阈值）。
- **T1（P1，两层 fail-closed）**：v2.2→interface export runner（复用 `build_handoff_from_sim`+`write_handoff_artifacts`）；TDD (a) fixture→`phase_map_trajectory`；(b) 真实 gate 不过→`refused/mechanism_candidate_only`+原因。
- **T2（P2）**：normalized phase grid 重建（同 mapping_id）；`m3a_overlay_consumable=True`。
- **T3**：谱相图（normalized）+ overlay（仅真实 gate 通过）+ 沿合格点 verdict/报告量/mode（**含 op 质量门 P1-2**）+ **h_G facet 或 `∂α₁/∂slow` 导数（P1-3）** + 控制（no-core/isotropic/shuffled-core）。**Figure**：`α₁=0` + mode-class 相图 + v2.2 轨迹 overlay + slow-var 敏感度。
- **T4（gated on phase2）**：2×2 correspondence + **virtual-SEEG proxy 可检测性对照** + honest-ceiling 措辞。
- **依赖**：T1‖T2 并行（不需 phase2）；T3 依赖 T1+T2；T4 依赖 T3 + phase2。

---

## 7. Self-Review

1. Placeholder：网格/`res_tol`/`mismatch_tol`/verdict 阈值在 T0 config 锁；mode-class 用 `classify_mode` metric 定义。**OK**。
2. 一致性：verdict 三类（§0/§1）中性、grounded 在 `classify_mode` saturated⇒runaway；质量门（§1）用 `OperatingPoint` 现有字段 + rate_mismatch；`trajectory_not_linearizable`≠`hard_jump_no_CSD`；h_G 导数（§3）答 Q2；T1 fail-closed（§2）与 §4 2×2 一致。**OK**。
3. Scope：单 plan（T0–T4，机器已存在，主要 wiring）；不含数据侧（topic5 拥有）、不含 reduced/Epileptor（retracted）。**OK**。
4. Ambiguity：path=a；CSD 本体=α₁（非代理）；不预设 α₁→0；机制归因=slow-var 导数 + mode class；correspondence=2×2 分本体/代理。**OK**。

---

## 8. 执行 gate / results / commit

- **gate**：T4 等 topic5 phase2。**T1–T3 不消费 phase2，可并行 prep**（模型侧，不写 correspondence 结论）。
- **results 目录（rev1 定）**：新建 `results/topic4_criticality/`（approach-to-criticality 项目，非仅 M3B phase map）；normalized atlas provenance 记于此，需要时**引用**旧 `results/topic4_sef_hfo/m3b_spectral_phase_map/`，但 **T4 correspondence 不埋回 M3B 目录**。
- **commit 策略（待用户，rev1 提出）**：spec/plan 现为 worktree untracked。建议**执行前把 spec + plan commit 到本分支**（显式路径），避免"以为记录了但没进 git"。待用户点头。
