# MZ early-field bridge — archive report (2026-07-19)

Branch `codex/topic4-mz-slowvars`. Design contract:
`docs/superpowers/specs/2026-07-19-topic4-mz-early-field-bridge-design.md`.
Overnight autonomous run. Local commits only.

## Abstract（朴素话）

**测了什么**：同一块按「病人 E1146 电极布局」摆的模型脑组织，安静时会自己一小簇一小簇地放电，每簇事件里
15 个虚拟电极被点亮的先后顺序跨很多次事件大致固定——这是这块组织的空间「指纹」，而且指纹有两个相反方向
（一头先亮，或另一头先亮）。然后我们只动一个旋钮：让「抑制刹车」随强活动慢慢磨损（去抑制），同一块组织就
慢慢滑向一次刹不住的失控放电（是模型代理，**不是临床发作**）。核心问题：在**跨过失控阈值（t120）之前**那段
早期招募窗里（相对 t_recruit 的 0–50 毫秒），能量最高的电极，是不是就是间期指纹里最早被点亮的那批。

**怎么测的**：先只用安静段的事件搭出两个方向的「最早→最晚」排序模板，并用「留一半事件出来验证」确认它不是
碰巧（留出的事件几乎每次都长一样，相似度约 0.99）。再拿失控点火那一瞬的能量分布跟两个模板比「先后↔能量」
相关，取相关更高的那个方向。如果这只是电极杆几何造成的假象，把能量在同一根杆内部随机重排一万次也该达到同样
高的相关；实测三个随机种子的真值都比随机高——其中两个明显更高（越过随机的概率约 0.0004 和 0.001），第三个
只是勉强（约 0.09，弱阳性）。这是**电极层面的主结果**。换成直接看神经元格子（源空间）三个种子都明显（约
0.009–0.017），但源空间只作**方向无关的「轴被调用」补充诊断**，不与电极合并成「跨尺度同方向」。（去核对照
这一轮不成立：没有一个电极落在核阈值内、实际一个都没删除，所以这条检验没提供信息，**不能写成「不靠贴核」**。）

**揭示了什么**：在这个模型尺度上看起来是——这块固定组织提供了一条可反复调用的**双向病理轴**，去抑制把整体
增益推高、让早期招募能量沿这条轴的时序指纹分布；每个种子的匹配（**方向无关的 maxAB**）都是正的，但**从哪一端
起始由噪声和当时网络状态决定，A→B 和 B→A 都是合法实现，我们不把某个固定方向在跨种子里稳定当成成功标准，也不把
某一端当成固定发作灶**。也就是「同一支架、不同状态」在这个观测层面说得通。但这只是「看得见相关」这一层，**还没到
因果层**：两次仿真是同噪声重放、不是真正的状态分叉，无法区分「局部去抑制图案」和「整体增益上移」哪个才是原因；
而且电极层面的统计强度不稳（三个种子里两强一弱）。要往因果走，下一步得把整个网络状态精确存档再分叉，做「原样/
抹平/打乱/复位去抑制」的状态匹配对照。

（内部归档代号：候选 `zA_q75_tz5000`；`B_to_A` direction；`rho_maxAB`；within-shaft / toroidal-shift null；
held-out 模板；`t_recruit`/`t120` onset；design §6–§9；completion level 4 = bridge supported diagnostic。）

## 1. Decision and question (§1)

The next MZ experiment is a **direct field bridge**, not another slow-variable search. Question:
does the spatial ORDER in which the 15 virtual-SEEG contacts light up during stable interictal-like
events on the fixed E1146 scaffold predict the early ENERGY field of a `z`-driven operational runaway?

Mandatory chain: same-seed slow-off returning events → held-out interictal timing templates →
`z`-only delayed operational runaway → onset-locked early activation/energy field →
template↔field association + spatial null + three-seed consistency.

No new global denominator, no new slow variable, no broad `z+m` scan, no spectral atlas were added.

## 2. Frozen candidate and seeds (§5)

- Primary: `zA_q75_tz5000` = `{use_z:true, use_m:false, I_th_EI:95.19851312666987, tau_z:5000.0}`
  (verbatim from `results/topic4_sef_hfo/mz_slowvars/p3_candidates.json`).
- Seeds 1/3/4, T=15000 ms. Multiseed had already confirmed runaway all 3 seeds
  (`runaway_ms` 9293.6 / 9499.3 / 9757.9; slow-off returning events 38/40/39).
- Sensitivity `zA_q50_tz10000` only if all primary deliverables done + >90 min left.

## 3. Methods (as implemented in `src/topic4_mz_early_field_bridge.py`)

Every function maps to a spec clause; the 10 required contract invariants are guarded by
`tests/test_topic4_mz_early_field_bridge.py` on synthetic fixtures.

- **Fixed-bar detector (§6)** — `compute_event_bar`: `floor=P95(af in [5,50]ms)`,
  `bar=floor+0.5*(max(af_slowoff)-floor)`, frozen ONCE from slow-off and reused for slow-off + native
  via `sef_hfo_events.detect_events(af, bin_w, event_on_frac=bar)`. Never recomputed from the target's
  own `af.max()` (the bug in `run_topic4_mz_slowvars._events_from_res`). Test 1 proves the frozen bar
  yields a different native event inventory than the target's own bar (freezing is load-bearing).
- **Interictal timing templates (§7.2/§7.3)** — 30–80 Hz butter(4) zero-phase + Hilbert envelope
  (`burst_envelope`, mirrors the accepted M3 readout). Per returning event: window
  `[t_on, min(t_off+40ms, next event, record end)]`; readable contact = event peak envelope exceeds its
  **slow-off quiet-envelope median by 5·MAD** AND excess peak ≥ 10% of the largest contact excess in the
  event; ≥6 readable; PEAK latency → ordinal ranks; **missing contacts never imputed** (test 3). Direction
  from Spearman(contact along-axis coord, latency rank): `A_to_B` ≥ +0.30, `B_to_A` ≤ −0.30 (sign→endpoint
  mapping from `src_xy/snk_xy/axis_unit`, written to metadata, never read off a plot). Chronological
  odd/even held-out split (no leakage, test 4); full-data template used for the §9 association.
- **Onset markers (§8.1)** — `t120` via the exact reused `run_m4_dynamic_qi._first_sustained(_smooth(rate_E))`
  (first 100 ms with ≥80% of 20 ms-smoothed E-rate ≥120 Hz). `theta_recruit=P99.9(20ms-smoothed slow-off
  rate)`; `t_recruit` = start of the native supra-theta component (≤5 ms gaps) that **contains t120** (test 8);
  else `onset_unresolved` (no early-field claim).
- **Early fields (§8.2/§8.3/§8.4)** — contact energy = mean-sq positive 30–80 Hz envelope excess over the
  **slow-off quiet median** in windows relative to `t_recruit` (reuses `early_energy_field`; incomplete window
  fails closed, test 2). Primary `0–50 ms`; sensitivities `0–100/0–25/25–50/50–100`. Source-grid = fixed 24×24
  bin mapping (mirrors `_spatial_movie`), per-bin first-spike latency (timing, ≥5 active E/bin) and per-bin
  early-window rate excess² over slow-off quiet mean (energy).
- **Association + nulls (§9)** — `earliness_energy_spearman = corr(-template_rank, energy)` + field cosine +
  quartile contrast + top-k (`compare_arrival_to_energy` reuse). `rho_maxAB=max(rho_A,rho_B)`, eligible only
  when BOTH held-out-validated direction templates are eligible. Primary contact null = **within-shaft**
  energy permutation recomputing `max(rho_A,rho_B)` inside each permutation (test 5), preserving shaft
  membership (test 6); plus unrestricted shuffle. Source null = non-zero **toroidal shifts** of the energy
  field recomputing maxAB (test 7). All associations on all-support AND direct-core-excluded support
  (source: exclude bins with any low-V_th-core E neuron; contact: Gaussian core-loading ≥ threshold; fails
  closed below 6 kept, no fallback). Local-tissue participation audit (fraction of E within 1.5 mm firing).
- **Three-seed reporting** — median/range of every effect + sign count; **no n=3 cohort p-value**.

## 4. Reuse map (§0/§5)

- Ported wholesale into this worktree (generic, numpy-only): `src/early_recruitment_readout.py` +
  `tests/test_early_recruitment_readout.py` (the upstream m2-integrator test was dropped — out of scope).
- Reused in-place: `run_m4_phaseplane.build_substrate`, `run_topic4_mz_slowvars.{run_mz_cell,build_core_masks}`,
  `run_sef_hfo_snn_cm_spontaneous_readout.{active_fraction,BIN_MS,BASELINE_MS,CAL_FRAC}`,
  `sef_hfo_events.detect_events`, `run_m4_dynamic_qi.{_smooth,_first_sustained}`, `snn_engine.lfp.LFPRecorder`.
- Written new (absent from the library): fixed-bar detector, 5·MAD readable rule, `maxab` observed + null,
  source toroidal-shift null, quartile contrast, `t_recruit`-contains-`t120` logic, source-grid fields.
- **No edits to the 6 guarded engine files** → no engine re-bless. (An off-by-default `snapshot_steps`
  observer already sits uncommitted in `src/snn_engine/mz_slow_vars.py`; byte-parity preserved, only used
  if the optional §10 z-snapshot decomposition is reached.)

## 5. Known-invalid artifact quarantine (§3)

Arm C `z+m` discovery is **not** consumed as evidence: the nominal 3×3 collapses to two unique z
configurations with all three m levels identical. No `9/9`, no `weak/mid/strong`, no dose-response.

## 6. Claim boundaries (§15)

Allowed if supported: a fixed patient-specific scaffold expresses reproducible interictal timing fields;
`z`-mediated loss of inhibitory efficacy moves the same model into an operational runaway whose pre-t120 early
recruitment energy field is concordant with the fixed **bidirectional** interictal timing axis (mirror-invariant
maxAB; A→B and B→A both legitimate); a model-side observation-level feasibility bridge for "same scaffold,
different state." Forbidden: calling the runaway a clinical seizure; claiming termination / recovery / a full
cycle; claiming `z_i` is the unique biological mechanism; claiming interictal events causally trigger the
transition (needs event-deletion); claiming local-z causality (needs snapshot/resume); using Arm C as
dose-response; interpreting virtual-LFP energy as clinical broadband power; choosing a direction/window/
candidate/seed for the strongest correlation; **claiming a fixed direction stable across seeds or a fixed
seizure focus**; **claiming the result is not core-driven** (the exclusion removed no contact this run);
**equating a contact virtual-LFP hotspot with preferential local-neuron recruitment**.

## 7. Completion levels (§14)

1. engineering complete — fixed-bar detector, reusable readout, tests, resumable artifacts.
2. numerically eligible — held-out template + complete non-degenerate early field exist.
3. scientific observation — direction, effect sizes, nulls, seed consistency reported (any sign).
4. bridge supported — ≥2/3 seeds eligible held-out + positive mirror-invariant contact `maxAB`; source not
   contradictory (as direction-free axis engagement). (Diagnostic criterion, NOT cohort proof, NOT seizure.
   Note: the spec's "not dependent only on direct-core loading" sub-condition was UNTESTABLE this run —
   n_kept=15, no contact fell inside the core threshold.)

## 8. Results (seeds 1/3/4, T=15000)

**Verdict: observation-level bridge SUPPORTED.** On the fixed E1146 SNN scaffold, the **bidirectional**
contact-level timing axis defined by slow-off interictal events predicts the early recruitment energy
gradient in the **pre-t120** window of the z-only disinhibition trajectory: the mirror-invariant contact
`rho_maxAB` is positive in all 3 seeds (0.945 / 0.735 / 0.924); the within-shaft null is significant in 2/3
(seed3 weak / null-overlap, p=0.086). This supports a model-side **"same scaffold, different state"** bridge.
It does **NOT** support causal mechanism, a complete seizure cycle, or a fixed-direction seizure focus.

**Framing (2026-07-19 review corrections).** (a) The scaffold defines a **bidirectional** pathological axis;
which end each event/seed starts from is noise/state-determined — A→B and B→A are both legitimate, and a fixed
direction stable across seeds is **not** a success condition. Primary statistic = the mirror-invariant `maxAB`
(max over the two direction templates, re-selected inside every permutation). (b) **Contact is primary; source
is a supplementary mechanism diagnostic** reported only as *direction-free axis engagement*, never merged with
contact into a "cross-scale same-direction" claim. (c) The 0–50 ms window is **pre-t120 early recruitment /
transition energy** (t_recruit precedes t120 by 139–215 ms), not post-onset or seizure energy. (d)
**Direct-core exclusion is uninformative this run**: n_kept=15 in every seed means NO contact was actually
removed, so it cannot support "not core-driven" — that claim is dropped. (e) Denominator = **one E1146 model
scaffold × 3 random seeds, not 3 patients**. (f) A readable contact virtual-LFP hotspot is **not** equated with
preferential local-neuron recruitment (participation audit incomplete, Q9).

Cohort (pre-t120 0–50 ms window, contact all-support): mirror-invariant `rho_maxAB` median **0.924**, range
**[0.735, 0.945]**, n_positive **3/3** (`cohort_summary.{json,csv}`).

1. **Fixed slow-off bar reused across states?** Yes. `compute_event_bar` freezes floor+bar once from slow-off
   and passes it to `detect_events` for BOTH slow-off and native; never recomputed from the native max
   (seed1 frozen bar=0.0259 on floor=9.4e-5, af_max_slowoff=0.0517). Test 1 proves the freeze changes the
   native event inventory.
2. **A/B train/held-out counts + reproducibility.** All 3 seeds: BOTH direction templates eligible. Held-out
   Spearman medians — seed1: A_to_B 0.361 (7 ev, 4tr/3ho), B_to_A **0.995** (26 ev, 13/13); seed3: A_to_B
   **1.000** (27 ev), B_to_A **0.999** (12 ev); seed4: A_to_B 0.743 (16 ev), B_to_A **1.000** (20 ev).
   Interictal is direction-imbalanced and the majority direction differs by seed (seed1/4 B_to_A-majority,
   seed3 A_to_B-majority) — consistent with a bidirectional axis whose per-seed balance is noise-set, not a
   fixed direction.
3. **t_recruit / t120 / window completeness.** All onsets eligible. seed1 t120=9293.6 / t_recruit=9078.3 (Δ215);
   seed3 9499.3 / 9360.1 (Δ139); seed4 9757.9 / 9559.3 (Δ199). Primary + all sensitivity windows complete
   (~5 s of post-onset trace at T=15000).
4. **Pre-t120 early recruitment field support / dynamic range.** Contact support 15/15 all seeds,
   non-degenerate (dyn-range seed1≈18.2, seed4≈5, seed3≈4). Source support 576/576 bins, non-degenerate.
5. **rho_A / rho_B / mirror-invariant maxAB + nulls (pre-t120 0–50 ms).** Primary = maxAB; the winning
   direction is descriptive only, not a success criterion. seed1 maxAB **0.945** (rho_a −0.565, rho_b +0.945),
   within-shaft p **4.0e-4**; seed3 **0.735** (rho_a −0.526, rho_b +0.735), p **0.086** (weak / null-overlap);
   seed4 **0.924** (rho_a −0.812, rho_b +0.924), p **1.0e-3**. Quartile contrast positive (seed1 +1.39). The
   maxAB winner is B_to_A in all three THIS run — per the review this is NOT reported as a stable phenotype.
6. **Consistency / source (supplementary).** Mirror-invariant contact maxAB is positive in all 3 seeds.
   Source is a supplementary *direction-free axis engagement* diagnostic (rho_maxAB 0.651 / 0.546 / 0.585,
   toroidal p **0.0087 / 0.012 / 0.017**, significant 3/3), reported separately and NOT combined with contact
   into a direction-agreement claim.
7. **Pre-runaway within-trajectory audit.** Eligible all 3 (25 / 29 / 30 pre-runaway returning events under
   the frozen bar) — secondary within-trajectory support present (not used as the primary template, §7.1).
8. **Observation vs mechanism layer.** Observation-layer only. slow-off and native are common-random-number
   replays from t=0 (not exact state forks); establishes association + a broad z-necessity boundary, NOT that
   a local pre-transition z pattern (vs a uniform gain shift) causes the early gradient.
9. **Optionals / incomplete.** §10 z global/local decomposition: **not_run**. q50/tz10 sensitivity:
   **not_run**. M3B projected propagator: **not_run**. Direct-core-excluded test: **uninformative** (n_kept=15,
   nothing removed → cannot claim not-core-driven). Local-tissue participation audit: **not_recomputed** — the
   contact readout was float-window-patched from saved LFP (`--readout-only`) and the native raster was not
   persisted, so a contact virtual-LFP hotspot is NOT claimed as preferential local-neuron recruitment.
   Follow-up: persist the early-window raster slice → recompute participation; add contacts inside the core so
   core-exclusion becomes testable.
10. **Largest gap + next step.** Gap: (a) contact-level significance is seed-fragile (within-shaft sig 2/3;
    seed3 weak); source-space direction-free axis engagement is 3/3; (b) causality unproven (CRN replay).
    **Next step (onset-dynamics phase)**: MZ equation crosswalk (z/m vs qI/J_K) + an onset state observer
    (orthogonal G/X/D coordinates, no ratio parameter) + slow-off/z-only/m-only/z+m phase portraits; then, once
    the observer is complete and checkpoint/resume is proven bit-identical, native / uniform-mean / shuffled /
    reset-z (and m freeze/reset) state-matched counterfactuals with maxAB kept direction-invariant.

## 9. Provenance

HEAD at run start `66a4d93`. Engine SHAs recorded in each per-seed `bridge_metrics.json::provenance`
and in `results/topic4_sef_hfo/mz_early_field_bridge/provenance.json`. Config snapshot:
`results/topic4_sef_hfo/mz_early_field_bridge/config_snapshot.yaml`.

## 10. V1 整体复核与口径冻结（2026-07-19）

### 一句话判断

early-field bridge 已完成一个合格的**观测层桥接**：正式 contact 分析、空间 null、三 seed 重复和
Figure 5 连续轨迹示例相互一致。最大缺口不是图，而是**因果状态分叉尚未完成**；因此当前不能从
“同一支架在两种状态下读出相似空间场”升级为“局部 `z_i` 耗竭导致该转变”。

### 完成度

**完成度：82/100（V1 observation/readout 目标）**。

- 已完成：fixed-bar 事件合同、held-out 双向模板、pre-t120 contact 场、maxAB-matched spatial null、
  seeds 1/3/4、连续 Figure 5 visual grammar、可复现 artifact/provenance。
- 扣分：contact 显著性仅 2/3 seed；local-tissue participation 与有效 core-exclusion 未完成；CRN replay
  不是 checkpoint 后的 state-matched fork；cross-seed 扩展只有 3 个目标场且只验证了本次胜出的 B→A 分支。

### P0 / P1 关键问题

没有阻断 V1 观测结论的 P0。三个 P1 必须写入口径：

1. **不能把跨 seed 的 3×3 当 n=9。** 9 个格子共享 3 个目标能量场；重复单位是 target seed，矩阵只作
   描述性 transfer diagnostic。
2. **不能把跨 seed 结果写成双向模板整体稳定。** 9/9 格子的 `maxAB` 都由 B→A 分支取胜；安全结论是
   “本次被调用的预测分支跨 seed 可迁移”，不是“A/B 两分支均为 seed-invariant scaffold property”。
3. **不能把 post-selection summary 当独立指标验证。** 原 field cosine 仍是相关量，quartile contrast 又是在
   Spearman 胜出方向上读取；它们已从独立证据口径中删除。若需要 metric-robustness，必须预先冻结方向/指标，
   或把方向选择完整嵌入相应 null。

### 跨 seed 补充诊断的安全结论

迁移矩阵为：

```text
template seed 1: 0.945  0.753  0.929
template seed 3: 0.943  0.735  0.938
template seed 4: 0.944  0.722  0.924
                 target seeds 1 / 3 / 4
```

同一 target 下换 template seed 的平均散度约 0.007；target 场均值之间的散度约 0.095。更直接的
same-seed maxAB 减去 foreign-template 中位数约为 +0.002 / −0.002 / −0.010。也就是说，本数据里没有
same-seed replay 优势；这**削弱**了纯同噪声巧合解释，但不构成 scaffold causality proof。

### 当前冻结 claim

> 在固定 E1146 模型支架上，held-out 双向间期样时序轴能够预测三个噪声种子中 operational-runaway
> 阈值前的 virtual-contact 早期能量分布，支持“同一支架、状态依赖读出”的观测层可行性。

必须同时保留：一块模型底物而非患者队列；operational runaway 而非 clinical seizure；virtual 30–80 Hz
energy 而非临床 broadband power；相关/迁移而非因果；contact hotspot 不等于局部神经元招募。

### V2 更新锁

本 V1 保留为独立 observation-layer artifact。等待 MZ onset-dynamics / state-conditioned 结果正式验收后，
再做 **V2 integrated bridge**，并且只在以下项目齐全时升级机制口径：

1. 把本分析冻结的 `t_recruit`、`t120`、模板和能量窗注册到 `D_z`、`q_eff`、`A_m`；
2. checkpoint/resume 逐比特一致，并完成 native / uniform-mean / shuffled / reset-z 的 state-matched fork；
3. 分开报告整体去抑制、空间化易感性、线性增益与 nonlinear ignition threshold；
4. 补 early-window raster 后审计 local tissue participation；
5. V2 另出动力学图，不向当前 Figure 5 塞入相图或未验收机制。

若只新增相图但没有 state-matched counterfactual，下一版仍是动力学描述，不是 causal bridge。
