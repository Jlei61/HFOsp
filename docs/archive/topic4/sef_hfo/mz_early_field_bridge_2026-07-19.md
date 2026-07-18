# MZ early-field bridge — archive report (2026-07-19)

Branch `codex/topic4-mz-slowvars`. Design contract:
`docs/superpowers/specs/2026-07-19-topic4-mz-early-field-bridge-design.md`.
Overnight autonomous run. Local commits only.

## Abstract（朴素话）

**测了什么**：同一块按「病人 E1146 电极布局」摆的模型脑组织，安静时会自己一小簇一小簇地放电，每簇事件里
15 个虚拟电极被点亮的先后顺序跨很多次事件大致固定——这是这块组织的空间「指纹」，而且指纹有两个相反方向
（一头先亮，或另一头先亮）。然后我们只动一个旋钮：让「抑制刹车」随强活动慢慢磨损（去抑制），同一块组织就
慢慢滑向一次刹不住的失控放电（是模型代理，**不是临床发作**）。核心问题：失控刚点火那 0–50 毫秒能量最高的
电极，是不是就是间期指纹里最早被点亮的那批。

**怎么测的**：先只用安静段的事件搭出两个方向的「最早→最晚」排序模板，并用「留一半事件出来验证」确认它不是
碰巧（留出的事件几乎每次都长一样，相似度约 0.99）。再拿失控点火那一瞬的能量分布跟两个模板比「先后↔能量」
相关，取相关更高的那个方向。如果这只是电极杆几何造成的假象，把能量在同一根杆内部随机重排一万次也该达到同样
高的相关；实测三个随机种子的真值都比随机高——其中两个明显更高（越过随机的概率约 0.0004 和 0.001），第三个
只是勉强（约 0.09）；换成直接看神经元格子（源空间）三个种子都明显高（约 0.009–0.017）。而且这批高能量电极
并不贴着那两个「病灶核」，去掉核附近电极后结论不变。

**揭示了什么**：在这个模型尺度上看起来是——这块固定组织确实提供了一份可复用的空间响应模板，去抑制只是把整体
增益推高，让失控沿着间期指纹里**同一个方向**先点燃（三个种子都指向同一方向，即使第三个种子的间期事件多数属于
另一个方向、反而是少数派模板才对上能量）。也就是「同一支架、不同状态」在模型里说得通。但这只是「看得见相关」
这一层，**还没到因果层**：两次仿真是同噪声重放、不是真正的状态分叉，无法区分「局部去抑制图案」和「整体增益
上移」哪个才是原因；而且电极层面的统计强度不稳（三个种子里两个强、一个弱）。要往因果走，下一步得能把整个网络
状态精确存档再分叉，做「原样/抹平/打乱/复位去抑制」的对照。

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
`z`-mediated loss of inhibitory efficacy moves the same model into an operational runaway whose early energy
field is concordant with one registered interictal direction; a model-side feasibility bridge for
"same scaffold, different state." Forbidden: calling the runaway a clinical seizure; claiming termination /
recovery / a full cycle; claiming `z_i` is the unique biological mechanism; claiming interictal events
causally trigger the transition (needs event-deletion); claiming local-z causality (needs snapshot/resume);
using Arm C as dose-response; interpreting virtual-LFP energy as clinical broadband power; choosing a
direction/window/candidate/seed for the strongest correlation.

## 7. Completion levels (§14)

1. engineering complete — fixed-bar detector, reusable readout, tests, resumable artifacts.
2. numerically eligible — held-out template + complete non-degenerate early field exist.
3. scientific observation — direction, effect sizes, nulls, seed consistency reported (any sign).
4. bridge supported — ≥2/3 seeds eligible held-out + positive contact `maxAB`; source not contradictory;
   not dependent only on direct-core loading. (Overnight diagnostic criterion, NOT cohort proof, NOT seizure.)

## 8. Results (seeds 1/3/4, T=15000)

**Verdict: bridge SUPPORTED at the §14 level-4 diagnostic criterion** — 3/3 seeds have eligible held-out
templates AND positive contact `rho_maxAB`, all the SAME `B_to_A` direction, source-space concordant (same
sign), and no result depends on direct-core loading. **Honest caveats**: the contact within-shaft null is
significant in 2/3 seeds (seed3 marginal p=0.086), so contact-level statistical strength is seed-variable;
the source toroidal null is significant 3/3. This is an **observation-layer feasibility bridge, not mechanism,
not cohort proof, not a seizure** (common-random-number replay, n=3 consistency only).

Cohort (primary window 0–50 ms, contact all-support): `rho_maxAB` median **0.924**, range **[0.735, 0.945]**,
n_positive **3/3** (`cohort_summary.{json,csv}`).

1. **Fixed slow-off bar reused across states?** Yes. `compute_event_bar` freezes floor+bar once from slow-off
   and passes it to `detect_events` for BOTH slow-off and native; never recomputed from the native max
   (seed1 frozen bar=0.0259 on floor=9.4e-5, af_max_slowoff=0.0517). Test 1 proves the freeze changes the
   native event inventory.
2. **A/B train/held-out counts + reproducibility.** All 3 seeds: BOTH direction templates eligible. Held-out
   Spearman medians — seed1: A_to_B 0.361 (7 ev, 4tr/3ho), B_to_A **0.995** (26 ev, 13/13); seed3: A_to_B
   **1.000** (27 ev), B_to_A **0.999** (12 ev); seed4: A_to_B 0.743 (16 ev), B_to_A **1.000** (20 ev).
   Interictal is direction-imbalanced and the majority direction differs by seed (seed1/4 B_to_A-dominant,
   seed3 A_to_B-dominant), yet the `B_to_A` template is the energy-concordant one in all three.
3. **t_recruit / t120 / window completeness.** All onsets eligible. seed1 t120=9293.6 / t_recruit=9078.3 (Δ215);
   seed3 9499.3 / 9360.1 (Δ139); seed4 9757.9 / 9559.3 (Δ199). Primary + all sensitivity windows complete
   (~5 s of post-onset trace at T=15000).
4. **0–50 ms field support / dynamic range.** Contact support 15/15 all seeds, non-degenerate (dyn-range
   seed1≈18.2, seed4≈5, seed3≈4). Source support 576/576 bins, non-degenerate.
5. **rho_A / rho_B / rho_maxAB + nulls (0–50 ms).** Contact (B_to_A wins each): seed1 rho_maxAB **0.945**
   (rho_a −0.565), within-shaft p **4.0e-4**; seed3 **0.735** (rho_a −0.526), p **0.086**; seed4 **0.924**
   (rho_a −0.812), p **1.0e-3**. Quartile contrast positive (seed1 B_to_A +1.39). Source: rho_maxAB
   0.651 / 0.546 / 0.585, toroidal p **0.0087 / 0.012 / 0.017** (3/3 significant).
6. **Sign consistency / contact-vs-source.** All 3 seeds same sign (positive, B_to_A). Contact and source
   both positive same direction every seed → concordant, not contradictory (multiseed Q3).
7. **Pre-runaway within-trajectory audit.** Eligible all 3 (25 / 29 / 30 pre-runaway returning events under
   the frozen bar) — secondary within-trajectory support present (not used as the primary template, §7.1).
8. **Observation vs mechanism layer.** Observation-layer only. slow-off and native are common-random-number
   replays from t=0 (not exact state forks); establishes association + a broad z-necessity boundary, NOT that
   a local pre-transition z pattern (vs a uniform gain shift) causes the early gradient.
9. **Optionals.** §10 z global/local snapshot decomposition: **not_run** (off-by-default observer exists,
   not invoked). q50/tz10 sensitivity: **not_run** (primary consumed the budget). M3B projected propagator:
   **not_run**. Local-tissue participation audit: **not_recomputed** for the contact result — the contact
   readout was float-window-patched from saved LFP (`--readout-only`) and the native raster was not persisted;
   documented follow-up (persist the early-window raster slice → recompute).
10. **Largest gap + next step.** Gap: (a) contact-level significance is seed-fragile (within-shaft p sig 2/3;
    seed3 marginal) even though the direction is consistent 3/3 and source is significant 3/3; (b) causality is
    unproven (CRN replay). **Single next step**: exact fast+delay+slow+RNG snapshot/resume with bit-identical
    continuation, then native / uniform-mean / shuffled / reset-z state-matched counterfactuals to separate a
    local z pattern from a uniform gain shift, then the projected-propagator overlay.

## 9. Provenance

HEAD at run start `66a4d93`. Engine SHAs recorded in each per-seed `bridge_metrics.json::provenance`
and in `results/topic4_sef_hfo/mz_early_field_bridge/provenance.json`. Config snapshot:
`results/topic4_sef_hfo/mz_early_field_bridge/config_snapshot.yaml`.
