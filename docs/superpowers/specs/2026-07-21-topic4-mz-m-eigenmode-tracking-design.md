# Topic 4 — MZ full-SNN state-aligned finite-time spatial mode tracking (design contract)

Status: BINDING. Autonomous task 2026-07-21. Branch `codex/topic4-mz-m-eigenmode-tracking`,
worktree `.worktrees/topic4-mz-m-eigenmode-tracking`, base commit `2c4f82b`
(`codex/topic4-mz-direct-spatial-modes`).

This spec transcribes the autonomous-agent prompt and adds the pre-registered numeric lock
(§11). It MUST NOT self-expand: no new mechanisms, no extra registered experiments, no scope
beyond what is written here. All §11 numbers are frozen BEFORE any spatial-response / operator /
mode result is viewed (spec §V/§VI of the prompt).

---

## 0. One-line scientific object

The empirical finite-time spatial response of the **complete current-based MZ spiking network**
(≈40 000 E/I LIF neurons) at the accepted **z+m plateau** work point, and how that response — and
its identifiable finite-time response modes — reorganize along the slow-state trajectory once the
adaptation variable `m` genuinely participates in the dynamics.

This is **not** a repeat of the z-only spatial perturbation, **not** a rate-field Jacobian, **not**
an exact full-SNN eigenanalysis, **not** a clinical seizure-mechanism proof.

Tier = model-side mechanism analysis. Every phenotype is a detection label. A bounded-negative
outcome (sparse / unidentifiable operator, honestly represented) is a valid completion.

### Naming discipline (BINDING)
The object is the **empirical finite-time SNN response operator** `K_T`. Its SVD gives V1 / U1 /
`sigma_hat_1`. These are **empirical finite-time singular modes**, NOT exact eigenvectors;
`sigma_hat_1` is NOT a Jacobian eigenvalue (units = output Hz / input current fraction). Forbidden
labels: exact eigenmode / eigenvalue / `Re(λ)` crossing / Hopf / fold / Floquet / critical-point
crossing.

---

## 1. Locked work point (READ, never re-estimated)

Substrate = `run_m4_phaseplane.build_substrate(seed)`: subject `epilepsiae_1146`, montage `narrow`,
placement `template_source`, lesion `twoend_equal`, L=20 mm, density 100/mm² (NE≈32k / NI≈8k),
E→E AR=2 along source→sink, g=3.6, ν_ext_ratio=0.6, dt=0.1 ms, two low-V_th cores r=1.5 mm at
17.5±1.0 mV, background 18 mV, spontaneous (no native kick).

**z+m plateau (primary, locked):** `use_z=True, use_m=True, I_th_EI=95.19851312666987,
tau_z=5000 ms, tau_adp=2000 ms, A_target=0.001, peak_m_tau2000=36.6036014019694`,
`eta_m = eta_m_from_frac(A_target, I_EE_scale, peak_m_tau2000) = 0.007451594355587098`
(`I_EE_scale = 272.75518960107513`). Seeds 1, 3, 4. **eta_m MUST be computed by calling
`src.topic4_mz_slowvars.eta_m_from_frac`, never hardcoded.**

This point is chosen because m genuinely participates AND it is the most cross-seed-stable m-active
point: all 3 seeds are gap-dynamics `bounded_elevated_plateau`, `D_max ≈ 0.0558 / 0.0600 / 0.0541`,
0/3 runaway, 0/3 recovery (`results/topic4_sef_hfo/mz_onset_dynamics/gap_dynamics_summary.json`,
`target_frac=0.001`). It represents "m holds the network in a controlled elevated plateau" — NOT a
terminated seizure, NOT a recovery cycle, NOT a clinical seizure.

**Forbidden substitutions:** pure m-only; z-only main result; tau_adp 500/1000 ms unstable points;
qi/gk/STD/conductance/any unregistered mechanism; tuning eta_m / kick strength / state times to get
a prettier mode.

### Relation to prior work (do NOT re-do)
This is the proper redo of the z-only line's **withdrawn P1 control** (z+m plateau vs D-matched
z-only, retracted 2026-07-20 for using an unsettled plateau checkpoint + transition D). Its redo
requirement — "settled plateau + avoid transition D" — is discharged by §3 here. Do NOT repeat:
z-only baseline/midpoint/pre_onset fixed-kick maps or low-k audit; rate-field frozen-J eigenmode;
fold/Hopf continuation; early-field bridge / virtual-SEEG; post-runaway runaway classification; new
mechanism search.

The ONLY new information this task adds: **at the registered, m-active, cross-seed-stable z+m
plateau, how the SNN's finite-time spatial response and its identifiable modes reorganize along the
slow-state trajectory.**

---

## 2. Reuse surface (do NOT reinvent; do NOT copy the whole runner)

Import and reuse verbatim (all import-safe / output-dir-agnostic):
- `run_m4_phaseplane.build_substrate`.
- `src.topic4_mz_onset_dynamics.{MZOnsetProbe, run_loop, LoopState, score_runaway, _loop_consts,
  natural_zm_trajectory}` — full-state checkpoint/fork, bit-identical resume, D/a/rate trajectory.
- `src.topic4_mz_direct_spatial_modes.*` — `MZSpatialProbe`, grid readout, 144-dim real Fourier
  basis, `balanced_lowk_indices`, `build_empirical_operator`, field geometry, fixed-kick readouts,
  `robust_identifiability_gate`, `select_epsilon`, `right_censoring_label`.
- From the baseline runner `scripts/run_topic4_mz_direct_spatial_modes.py` (imported as `DSM`):
  `build_S`, `grid_region_masks`, `replay_checkpoints`-pattern, `_replay_traj`, `_resting_mask`,
  `_capture_at`, `_ensure_flat`, `_fork_run`, `_fork_Y`, `fixed_kick_state`, `linearity_audit_state`,
  `operator_for_state`, `corrected_operator_audit`, `_realization_state`, `_op_task`, `_corr_task`,
  `_parallel_map`, `_time_stack`. These reference no output path except through `cfg` / args.

Do NOT modify: the 6 guarded engine files, `src/snn_engine/mz_slow_vars.py`, `MZ*Probe`,
`src.topic4_state_conditioned_susceptibility`, `src.topic4_mz_direct_spatial_modes`, the baseline
runner, any existing onset-dynamics / direct-spatial-modes artifact or figure. If an engine change
were unavoidable it must be off-by-default with a parity test — but none is expected here.

New files ONLY: `config/topic4_mz_m_eigenmode_tracking.yaml`,
`src/topic4_mz_m_eigenmode_tracking.py`, `scripts/run_topic4_mz_m_eigenmode_tracking.py`,
`scripts/paper_figures/plot_figure5_mz_m_eigenmode_tracking.py`,
`tests/test_topic4_mz_m_eigenmode_tracking.py`, this spec, the archive doc.

---

## 3. State registration (frozen BEFORE any perturbation result)

The upstream trajectory NPZ holds only aggregate D/a/rate, not full neuron state, so each seed is
re-replayed under the identical config to capture full recoverable checkpoints. Registration uses
ONLY raw slow variables + population rate + time — never a perturbation, U1, `sigma_hat_1`, or figure.

Per seed, register 5 states: `baseline, approach_25, approach_50, approach_75, settled_plateau`.

1. Replay the z+m plateau once for `replay_ms` (§11) with a no-schedule `MZSpatialProbe` (natural
   z+m). Accumulate `trace_z_mean`, `trace_adap_current`, population E-rate → via
   `natural_zm_trajectory(..., downsample_ms)` obtain `D_allE = 1 - z̄`, `a_allE = A_abs/I_EE_scale`,
   `rate_E_hz`.
2. **Parity gate (§V.7):** prove the replayed (D, a, rate) match the upstream NPZ
   `traj_zA_q75_tz5000_A0.001_seed{1,3,4}.npz` within tolerance (§11). Same code / substrate / seed
   ⇒ expect near-bit-identical; a relative deviation over `parity_rel_tol` is a STOP-and-report
   discrepancy, not a silent proceed.
3. Estimate `D_base` = D at the baseline anchor (resting step nearest `baseline_ms` within
   `baseline_search_halfwidth_ms`); `D_plateau` = median D over resting steps in the settled tail
   window `settle_tail_ms`.
4. `approach_f` (f = 0.25/0.50/0.75): the FIRST time D crosses `D_base + f·(D_plateau − D_base)`;
   within a locked forward window `approach_search_ms` of that first crossing pick the lowest
   population-rate (most resting) step — so a checkpoint never lands on a fast-event peak. Resting =
   `DSM._resting_mask` (20 ms-smoothed population rate ≤ P20 + 0.3·(P99−P20)).
5. `settled_plateau`: the resting step in the tail window whose D is nearest the tail median.
   **Settled gate:** accept only if over the tail resting steps `ptp(D) < settled_D_ptp_max` AND
   `ptp(a) < settled_a_ptp_max` AND the tail has ≥ `settled_min_resting_frac` resting fraction AND
   `D_plateau` sits in the elevated band (`> 0.5 · D_onset_ref`, `D_onset_ref = 0.0873`). If any
   fails → mark the seed's `settled_plateau` `unresolved`; NEVER force a plausible-looking point.
6. Capture the full `LoopState` checkpoint at each registered step via segmented replay + resume
   (`run_loop(..., capture_final=True)`); persist with sha256 hash + the fast-state fingerprint.
7. Write `state_registration.json`: per seed × state → `branch_step, time_ms, D, a, rate_hz,
   checkpoint_sha, settled_gate` + the frozen §11 rule block + parity report.

Checkpoints are NEVER moved because a later state's mode looks nicer.

---

## 4. Analysis order (P0 → P4)

**P0 — state + checkpoint contract.** Full replay, checkpoint/resume parity, freeze verification.
Prove: fork-time V / spike-history / synaptic state / z / m / RNG all identical pre-fork; z & m
unchanged post-freeze; all perturbation forks use common random numbers. Tests first.

**P1 — fixed-kick tracking (MAIN).** At all 5 states, the identical source-localized kick (§11
fixed_kick): output at 5/15/30/50 ms → 2-D response maps + axial kymograph + corridor / matched
off-axis response + distal recruitment; arrival-time-vs-distance fit ONLY if the response clears the
absolute floor. Zero-response / constant-arrival / empty-mask → fail closed (`DSM.fixed_kick_state`
already enforces this). Kymograph supports spatiotemporal recruitment shape only; "stepwise
propagation" requires a positive-slope, adequate-range, qualified arrival fit — a direct remote
response is NEVER written as a continuous wavefront.

**P2 — low-k empirical operator tracking.** Reuse `DSM.corrected_operator_audit`: balanced
symmetric low-k basis (k_max=1 → 9 modes), per-grid RMS aligned to the fixed kick, ±ε paired,
1×/2× amplitude scaling, 16 independent continuation-noise futures, two independent 8-future halves,
CRN, saturation fail-closed. **Strict identifiability gate (all four + no saturation):** full-N
discrepancy ≤ 0.15 AND half-A ≤ 0.15 AND half-B ≤ 0.15 AND cross-half operator instability ≤ 0.15
(= `robust_identifiability_gate`). NOT identifiable → that state's V1/U1/`sigma_hat_1` are blank /
`unresolved`; the full-16 mean is NEVER used to bypass the split-half gate; missing modes are NEVER
interpolated on a time curve. Identifiable → at T = 10/30/50 ms save K_T, singular spectrum,
V1/U1/`sigma_hat_1`, `sigma_hat_1/sigma_hat_2` gap, U1/V1 axis alignment, U1 globality, U1 corridor
fraction, the two-half operator discrepancy, sign-invariant U1/V1 half-overlap.

**P3 — cross-state mode tracking.** ONLY between adjacent states that BOTH pass the strict gate:
sign-invariant adjacent U1/V1 |cos| overlap, principal angle(s), |field|²-weighted spatial-centroid
displacement, axis-alignment change, `sigma_hat_1` change. If `sigma_hat_1/sigma_hat_2` is below the
locked `degeneracy_ratio`, do NOT chase a single U1 — track the leading singular SUBSPACE via
principal angles / Procrustes. A seed with < 2 adjacent identifiable states supports no mode
trajectory (report fixed-kick + sparse identifiability only).

**P4 — minimal m-mechanism controls.** ONLY at `baseline, approach_75, settled_plateau`. Four
conditions sharing the SAME fast state / z / kick / RNG future: (a) native z+m; (b) m_reset (m→0);
(c) m_uniform (keep mean(m), flatten spatial pattern); (d) m_shuffle (keep m distribution, permute
location). Readout = the fixed kick (P1), NOT the full audit. Interpretation bounds: m_reset is a
short off-manifold counterfactual (immediate m-current contribution only); m_uniform vs native
separates mean brake from spatial pattern; m_shuffle is a spatial-pattern sensitivity control.
These short forks are NEVER written as long natural trajectories. Operator-level m-contrast is
discussed only where P2 passed the gate; the fixed-kick contrast always stands.

---

## 5. cheap-first + stop rules (BINDING)

First: seed1, 5 states, fixed kick, corrected audit with a 4-future SMOKE (plus a tiny-net smoke for
pure mechanics). Confirm checkpoint / input normalization / response shape / gate / resource before
the final 16-future 3-seed run. Expand low-k k_max 1→2 ONLY if ≥2 seeds each have ≥2 adjacent
strictly-identifiable states AND half-to-half U1 / leading-subspace is stable. NEVER auto-expand to
the full 144-dim operator — if it is ever warranted, STOP and give the user a compute-budget +
scientific-benefit estimate. Resource discipline: one seed's full SNN at a time; workers ≤ 8;
OMP/MKL=1; check RAM + other worktrees before each heavy stage; checkpoint/resume every long task;
never enlarge an experiment to fill time.

---

## 6. Allowed / forbidden claims

**Allowed:** the same m-active plateau scaffold shows a specific finite-time spatial susceptibility;
the fixed kick becomes / does not stably become more axial along the slow trajectory; m acts mainly
as a global brake / mainly reshapes space / both; the z-only pre-onset axial susceptibility is
preserved / weakened / lost in the z+m plateau; the operator is / is not identifiable at each state.

**Forbidden:** plateau = clinical seizure; a full interictal→ictal→recovery cycle; V1/U1 are exact
eigenmodes; `sigma_hat_1 > 1` = net amplification; kymograph proves a travelling wave; Hopf / fold /
Floquet / eigenvalue crossing; a cross-seed-consistent axial operator claim without ≥2 same-state
strictly-identifiable seeds agreeing; writing a short m-control fork as a natural trajectory; tuning
state / seed / ε / basis / T / eta_m to rescue a conclusion.

---

## 7. Engineering invariants (each = a TDD test)

- **E1** eta_m computed via `eta_m_from_frac` (= 0.007451594355587098 at the locked args).
- **E2** state registration independent of any perturbation outcome (registration on D/a/rate only).
- **E3** replay ↔ upstream NPZ parity within tolerance (§11).
- **E4** full-state checkpoint/resume parity: segmented replay + resume == single continuous replay
  (V, ref, s/I currents, rings, xi, RNG, z, m all equal at the branch).
- **E5** freeze holds z and m constant over the fork window.
- **E6** m_reset changes ONLY m (m[:NE]→0; z, V, currents, RNG untouched).
- **E7** m_uniform preserves mean(m[:NE]) and flattens (ptp→0).
- **E8** m_shuffle preserves the multiset (sorted(m[:NE]) unchanged), changes ordering.
- **E9** common random numbers: +ε / −ε / no-probe / m-control forks share the checkpoint RNG.
- **E10** low-k basis symmetry + orthonormality (9 modes for k_max=1; nested for k_max=2).
- **E11** 1×/2× amplitude pairing (`amps = [base, base/2]`).
- **E12** 16 futures + two 8-halves are genuinely independent RNG states.
- **E13** strict identifiability gate = 4-term AND + no saturation (`robust_identifiability_gate`).
- **E14** saturation fail-closed (a saturated fork ⇒ not identifiable).
- **E15** zero-response arrival fail-closed (constant / empty / below-floor ⇒ ineligible).
- **E16** mode sign ambiguity: overlap and principal angle sign-invariant.
- **E17** degenerate-subspace tracking (gap < degeneracy_ratio ⇒ subspace, principal angles).
- **E18** resume idempotency (a completed cell is not recomputed; atomic writes).
- **E19** provenance + checkpoint hash recorded.
- **E20** plotting fails closed on an unresolved / missing sidecar (blank, never gain=0).

Resource: record `free -h` + nprocs; ≤ 8 COW-shared fork workers; no nested multiprocessing;
`--resume`; atomic JSON writes; known scipy LIF roundoff warning logged not hidden; no new hidden
warnings.

---

## 8. Results tree + figures

Science root `results/topic4_sef_hfo/mz_m_eigenmode_tracking/`: `STATUS.md`,
`state_registration.json`, `checkpoint_manifest.json`, `fixed_kick_summary.json`,
`operator_tracking_summary.json`, `controls_summary.json`, `numerical_audit.json`,
`provenance.json`, `per_seed/`, `figures/README.md`. Raw per-fork arrays / checkpoint pickles may be
gitignored; manifests, hashes, summaries, figures, README are auditable. (Per the direct-spatial
convention this results tree is a gitignored working tree; commits carry code + spec + archive.)

Paper-ready candidate `results/paper-ready-figure/fig5_mz_m_eigenmode_tracking_candidate/figures/`
with PNG **and** PDF, `README.md` (Chinese). Do NOT touch the z-only Figure 5 directories or
FIGURE_INDEX / main_figure_plan before user visual acceptance.

**Two candidate figures only.** Nature main-panel style: width ≈7.2 in, 300 dpi, editable PDF text
(`fonttype=42`), no suptitle, no background grid, no explanatory paragraph on canvas, short
left-aligned panel letters, concise axis labels/titles; baseline = neutral grey, plateau = a
distinct accent — NEVER template-A/B red/blue; unresolved states left blank (never drawn as 0).
Eyeball every PNG; never confirm by file existence.

- **Figure A (state-aligned fixed-kick tracking):** top D/a trajectory with the 5 checkpoints; body
  = baseline / approach_75 / settled_plateau response maps at 5/15/30/50 ms; bottom = corridor vs
  off-axis + distal recruitment + a qualified arrival slope (if any). No long caption.
- **Figure B (finite-time mode tracking):** identifiability strip; U1 insets for robust states only;
  `sigma_hat_1(T)`; axis alignment; adjacent overlap / subspace principal angle; the minimal
  m_reset / m_uniform / m_shuffle contrast.

---

## 9. Pre-registration lock (§11 — FROZEN 2026-07-21 before any spatial result)

| block | key | value |
|---|---|---|
| replay | replay_ms | 20000.0 |
| replay | downsample_ms | 5.0 |
| parity | parity_rel_tol | 0.02 |
| parity | parity_report_abs | true (raw max-abs-diff logged; bit-identical expected) |
| resting | resting_win_ms / resting_k | 20.0 / 0.3 (via `DSM._resting_mask`) |
| baseline | baseline_ms / baseline_search_halfwidth_ms | 1000.0 / 250.0 |
| approach | approach_fracs | [0.25, 0.50, 0.75] |
| approach | approach_search_ms | 300.0 |
| settled | settle_tail_ms | 1000.0 |
| settled | settled_D_ptp_max | 0.015 |
| settled | settled_a_ptp_max | 8.0e-5 |
| settled | settled_min_resting_frac | 0.30 |
| settled | D_onset_ref (elevated-band ref) | 0.0873 |
| operator | k_max_start | 1 (9 modes) |
| operator | n_realizations / smoke | 16 / 4 |
| operator | strength_frac | 0.02 |
| operator | realization_base_seed | 91021 |
| operator | linearity_tol | 0.15 |
| operator | degeneracy_ratio | 1.05 |
| operator | T_windows_ms | [10.0, 30.0, 50.0] |
| fixed_kick | center / sigma_norm / frac | source / 0.6 / 0.01 |
| fixed_kick | local_map_centers_ms / width_ms | [5,15,30,50] / 5.0 |
| fixed_kick | kymograph_band_norm / n_pos | 0.5 / 12 |
| fixed_kick | arrival_thresh_fracs / min_peak_hz / r2_min | [0.05,0.10,0.20] / 2.0 / 0.5 |
| window | window_ms / current_dur_ms / freeze_zm | 50.0 / 1.0 / true |
| saturation | runaway_hz / saturation_dur_ms | 120.0 / 20.0 |
| controls | m_control_states | [baseline, approach_75, settled_plateau] |
| controls | m_control_conditions | [native_zm, m_reset, m_uniform, m_shuffle] |
| controls | m_shuffle_seed | 20260721 |
| resource | workers | 8 |
| seeds | seeds | [1, 3, 4] |

These are frozen. Changing any to rescue a conclusion is a spec violation.
