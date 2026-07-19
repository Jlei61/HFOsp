# Topic 4 — MZ full-SNN direct spatial mode dynamics (design contract)

Status: BINDING. Autonomous task 2026-07-19. Branch `codex/topic4-mz-direct-spatial-modes`,
worktree `.worktrees/topic4-mz-direct-spatial-modes`, base commit `6c878ae`
(`codex/topic4-mz-onset-dynamics`).

This spec transcribes the autonomous-agent prompt 1:1. It MUST NOT self-expand: no new
mechanisms, no extra registered experiments, no scope beyond what is written here.

---

## 0. One-line scientific object

The empirical finite-time spatial response of the **complete current-based MZ spiking
network** (≈40 000 E/I LIF neurons) to a brief per-E-neuron current perturbation, measured at
different slow states (z-only trajectory). This is **not** a rate-field surrogate, **not** an
exact full-SNN eigenanalysis, and **not** a clinical seizure-mechanism proof.

Tier = model-side mechanism analysis. Every phenotype is a detection label.

---

## 1. Substrate + accepted work point (READ, never re-estimated)

- `PP.build_substrate(seed)` from `scripts/run_m4_phaseplane.py`: subject `epilepsiae_1146`,
  montage `narrow`, placement `template_source`, lesion `twoend_equal`, L=20 mm,
  density=100/mm² (NE≈32 000 / NI≈8 000), E→E AR=2 along source→sink, g=3.6, ν_ext_ratio=0.6,
  dt=0.1 ms, two low-V_th cores r=1.5 mm at 17.5±1.0 mV, background V_th=18 mV, spontaneous
  (no native kick). Returns `posE, src_xy, snk_xy, axis_unit, vth, center, p, net, NE, N`.
- Slow variables: E-cell `z_i` (inhibitory efficacy) + `m_i` (adaptation) only, per
  `src/snn_engine/mz_slow_vars.py`. All historic mechanisms (q_I / g_K / S_G pool / shunt /
  STD / conductance MZ) are OFF and out of scope.

### Primary z-only candidate (locked)
`label = zA_q75_tz5000`, `use_z=True, use_m=False, I_th_EI=95.19851312666987, tau_z=5000` ms.
Seeds 1, 3, 4. **Locked operational-runoff onset (READ from config, never re-estimated):**
seed1 9293.6 ms, seed3 9499.3 ms, seed4 9757.9 ms.

### Primary temporal states
1. `baseline` = 1000 ms
2. `midpoint` = 0.50 × locked runoff
3. `pre_onset` = locked runoff − 100 ms

Main figure directly compares **baseline vs pre_onset only**. `midpoint` is for the mode
trajectory only — never added to the main figure to inflate panel count. The `onset`
checkpoint is NOT a primary state; if any no-probe control runs away inside its analysis
window, label the cell `right_censored_native_transition` (never gain=0, never escalate
amplitude).

### Secondary controls
- **z+m plateau** (P1): `zA_q75_tz5000`, `A_target=0.001`, `tau_adp=2000` ms. Choose one
  plateau checkpoint in the last 500 ms of a stable segment, NOT inside a detected event; the
  selection rule (D nearest the window-median at a non-event time) is frozen before viewing the
  spatial response.
- **D-matched z-only** (P1): in the same seed's z-only trajectory, the earliest non-event time
  whose D is closest to the plateau D. Selection uses only D + event label + time, never the
  spatial response.
- **q50/tz10000 fixed-kick sensitivity** (P2): `zA_q50_tz10000`,
  `I_th_EI=1.6652801609959704, tau_z=10000`, fixed-kick only.

Execution priority: **P0** (primary baseline + midpoint + pre_onset) > **P1** (plateau +
D-matched) > **P2** (q50 fixed-kick). If near the 8 h limit, finish P0 + figures + report;
never leave P0 half-done for P2.

---

## 2. Direct-SNN perturbation numerics

### 2.1 Branch state & checkpoint fork (REUSE — do not reinvent)
Reuse `src.topic4_mz_onset_dynamics.run_loop` + `LoopState`, which already captures the full
engine state at a branch step (`V`, `ref`, `s_E/I_E/s_I/I_I`, `ring_sE/ring_sI`, OU `xi`,
`rng_state`, slow object) and resumes it bit-identically. Each seed replays its native
trajectory **once** (segmented replay with resume to capture baseline / midpoint / pre_onset
checkpoints). All perturbations FORK from a captured checkpoint — never re-run from t=0 per
pattern.

**Primary analysis freezes z/m for the 50 ms fork window** (fast-subsystem susceptibility
isolation), via the inherited `MZOnsetProbe.set_branch(branch_step, freeze=True)` armed in the
captured checkpoint's slow object. `native_dynamic` (freeze=False) forks are a **secondary**
sanity check only.

### 2.2 Perturbation carrier (NEW, off-by-default)
Do NOT modify `src/snn_engine/kick_probe.py` or `src/snn_engine/mz_slow_vars.py`. In the new
module, subclass `MZOnsetProbe` as `MZSpatialProbe` adding an off-by-default per-E-neuron
additive-current schedule:
- acts on E cells only (indices `[:NE]`);
- duration 1.0 ms (10 steps at dt=0.1);
- pattern may be positive or negative;
- **when the schedule is unset, the native path is bit-identical** (parity gate);
- amplitude expressed as a fraction of `I_EE_scale = 272.75518960107513` (NOT rate-field proxy
  units).

Injection point: the current is added to the `I_net` returned by `apply_currents`, which the
engine consumes AFTER both per-step RNG draws (`standard_normal` for OU, `poisson` for external
drive). Therefore the schedule **cannot perturb the RNG draw order** → common random numbers
hold for free.

Amplitude ladder (pre-registered): `[0.001, 0.0025, 0.005, 0.01] × I_EE_scale`. One final
amplitude is chosen for ALL seeds / states / patterns. Selection criterion = linear regime +
numerical identifiability ONLY, never "produces the expected axial result".

### 2.3 Linearity audit
- Compare the central-difference response at ε vs ε/2.
- Normalized discrepancy ≤ 15 % qualifies a candidate ε for the empirical operator.
- Immediate spike fraction and output must not saturate.
- If several ε qualify, use the **largest** qualifying ε.
- If **no** registered ε qualifies: STOP the operator SVD (do NOT widen the ladder); complete
  the fixed-kick nonlinear response and report `nonlinear_response_only`.

### 2.4 Common random numbers
For each checkpoint × pattern × ε: `+ε`, `−ε`, and `no-probe` forks all fork from the identical
`LoopState` + RNG state (run_loop resets `net["rng"]` to `ck.rng_state`), so future random draws
are common. No pattern inherits a RNG already advanced by a previous pattern. Save RNG
provenance/hash.

### 2.5 Output space
The SNN keeps ≈40 000 neurons. A 12×12 grid is an input/output READOUT only, never the
dynamical model. Save only `time_bin × 12 × 12` E-spike counts/rates; never full `N × T` spike
bool. Raw time bin 1 ms; response maps = 5 ms local windows up to 5/15/30/50 ms; operator
output `Y_T` = mean E firing rate per grid cell over `[0, T]`, `T ∈ {10, 30, 50}` ms; each cell
divided by its actual E-neuron count; save empty-cell check + total spike mass-conservation
audit. Reuse `normalize_subject_coordinates` + `coarse_cell_index` from
`src.topic4_state_conditioned_susceptibility` for cell assignment (same convention as the
rate-field work). Do NOT import its rate-field operator functions for our results.

---

## 3. Three input classes (kept strictly separate)

1. **Fixed localized kick** — one pre-registered source-core Gaussian **positive** current kick;
   identical location, width, RMS, timing across all states.
   `DeltaY_fixed = Y(+kick) − Y(no-probe)`. Answers "same stimulus, different states". Never
   conflated with a per-state-re-optimized V1.

2. **Full empirical operator basis** — a complete real orthonormal 2-D Fourier basis on the
   12×12 coarse input space (**all 144 dims**, `Q.T @ Q ≈ I`, not a few low-k modes). Map each
   coarse pattern to E-neuron coordinates, apply ±ε, and for each T:
   `K_T[:, j] = [Y_T(+ε p_j) − Y_T(−ε p_j)] / (2ε)`; then `K_T = U S Vᵀ`. Report V1 input, U1
   output, `sigma_hat_1(T)`, gap `s1/s2`, and a top-r subspace when `s1/s2 ≈ 1`. Units of
   `sigma_hat_1` = output Hz / input current fraction — **not** dimensionless gain, so never
   write "sigma>1 = net amplification"; only compare across states under identical units + B/C
   definitions.

3. **Fourier/Gabor probe scan** (comparison to the old `G(k∥,k⊥)`) — a fixed source-centered
   phase-paired cos/sin Gabor dictionary (`p_max=4`, equal RMS), identical for baseline and
   pre_onset. This is a **dictionary probe**, named `phase-paired probe susceptibility`; its
   maximum is NEVER called the real `sigma_hat_1` unless its input space equals the full 144-dim
   operator (it does not).

---

## 4. Readouts

**Fixed kick:** 5/15/30/50 ms response maps; whole-field response norm; source / axis-corridor /
remote-sink response; cumulative remote/source response energy; axis kymograph; first-arrival
time vs source→sink distance; threshold sensitivity at 5/10/20 % of state-specific peak. The
kymograph shows axial spatiotemporal recruitment only — NOT a proven continuous wavefront.
First-arrival regression needs ≥4 valid axial positions; report slope, velocity proxy, R²,
n_points, threshold. If remote responds near-simultaneously or slope≈0, write "compatible with
direct remote recruitment", never a forced wavefront.

**Empirical modes:** V1/U1 field; `sigma_hat_1(T)`; U1 axis alignment; U1 globality;
adjacent-state mode/subspace overlap; singular gap; phase-paired k∥/k⊥ preference. For signed
V1/U1, axis + globality use `|field|²` loading; SVD overall sign is meaningless → all comparisons
sign-invariant. If `s1/s2` too close so V1 direction is unstable, downgrade to the leading
subspace — never pick the best-looking single vector.

---

## 5. Required controls

1. no-probe fork; 2. ε/2 linearity audit; 3. common-RNG reproducibility; 4. baseline vs
pre_onset; 5. three native seeds; 6. z+m plateau vs D-matched z-only; 7. Gabor probe maximum ≤
full-space `sigma_hat_1` upper-bound check; 8. onset / right-censoring check. No AR1 rebuild,
no rotated scaffold, no large network sweep in this task. An optional seed-1 AR1 diagnostic is
P2-exploratory and must not delay the primary three-seed result.

---

## 6. Difference from adjacent lines (do NOT cross)

- **vs MZ onset temporal dynamics** — already answered (z-only runoff ≈9.3–9.8 s; z+m sub-onset
  plateau; runoff corridor D≈0.087). Do not sweep eta_m/tau_adp, do not redraw the D–a temporal
  phase diagram, do not discuss recovery mechanisms.
- **vs frozen-q rate-field susceptibility** — bypass the `state_operator → G_μ*C Jacobian →
  eigenvalue/eigenmode/exp(JT)/SVD` closure. FORBIDDEN objects for our results: `state_operator`,
  M3B operating point, `build_jacobian_dense`, `exp(JT)`, frozen-q rate-field eigenvalue,
  proxy-normalized mu_core/a mapping. Old results may be read for qualitative comparison only,
  never mixed numerically into the direct-SNN operator.
- **vs early-field bridge** — no electrode projection, no template similarity, no cross-seed
  contact-field transfer.
- **vs conductance MZ** — use only the current-based MZ at commit `6c878ae`.
- **vs exact eigenmode** — the full SNN (threshold/spike/reset/refractory/delay/noise) is a
  hybrid system; name the object `Empirical finite-time SNN response operator`. Forbidden
  labels: exact full-SNN eigenmode / eigenvalue / `Re(λ)` crossing / Hopf/fold/Floquet.

---

## 7. Engineering + contract clauses (the invariants the code must hold)

New files (do not exceed): `config/topic4_mz_direct_spatial_modes.yaml`,
`src/topic4_mz_direct_spatial_modes.py`, `scripts/run_topic4_mz_direct_spatial_modes.py`,
`scripts/paper_figures/plot_figure5_mz_direct_spatial_modes.py`,
`tests/test_topic4_mz_direct_spatial_modes.py`, this spec.

Do NOT modify: `src/snn_engine/kick_probe.py`, `src/snn_engine/mz_slow_vars.py`,
`src/topic4_m3b_spectral_phase.py`, `src/topic4_state_conditioned_susceptibility.py`,
existing onset-dynamics artifacts, existing rate-field Figure 5 Supplementary 1/2, Topic 5 /
Methods / main docs.

**Multi-clause invariants (each = a TDD test):**
- C1 parity: `MZSpatialProbe` with no schedule ⇒ byte-identical native replay
  (`run_loop == simulate_kick`, `E_spk_bool` + `rate_E` array-equal).
- C2 pre-branch identity: segmented replay + resume == single continuous replay.
- C3 common RNG: `+ε` / `−ε` forks share the checkpoint RNG state; no-probe fork reproducible.
- C4 E-only + window + RMS: current pattern acts on `[:NE]` only, inside `[lo,hi)` only; RMS
  normalization exact; I cells untouched.
- C5 basis: real 2-D Fourier basis has 144 dims and `Qᵀ Q = I` (atol 1e-10).
- C6 mass conservation: spike-bin sum == total spikes; empty cells flagged.
- C7 synthetic operator: a known linear operator is recovered (V1/U1/sigma1) through the full
  ±ε fork→SVD pipeline.
- C8 `sigma_hat_1 ≥` any probe gain in the same input space.
- C9 singular-vector sign invariance.
- C10 near-degenerate subspace handling (`s1/s2 ≈ 1` ⇒ subspace, not single vector).
- C11 onset-branch right-censoring label.
- C12 interrupted-run resume / idempotency (completed cell not recomputed; atomic writes).
- C13 plotting fails closed when a sidecar is missing.
- C14 no existing rate-field Figure 5 file is overwritten.

Resource discipline: record `free -h` + nproc first; default ≤1–2 full-density workers; no
nested multiprocessing; build ≈13 GB connectivity once per seed (COW share across fork workers);
runner `--resume`; completed cells never recomputed; atomic writes (no shared-JSON race). Known
scipy LIF integration roundoff warning may be logged but not hidden; no new warnings hidden.

---

## 8. Results tree + figures

Science root: `results/topic4_sef_hfo/mz_direct_spatial_modes/` with `STATUS.md`,
`provenance.json`, `checkpoint_manifest.json`, `linearity_audit.json`, `fixed_kick_arrays.npz`,
`fixed_kick_summary.json`, `empirical_operator_arrays.npz`, `empirical_operator_summary.json`,
`probe_scan_summary.json`, `controls_summary.json`, `numerical_audit.json`, `per_seed/`,
`figures/`, `figures/README.md`. Raw per-pattern arrays may be gitignored, but manifests, hashes,
summaries, figures, README are auditable.

Do NOT overwrite `results/paper-ready-figure/fig5_mz_spatial_dynamics_supplementary/`. New
candidate → `results/paper-ready-figure/fig5_mz_direct_snn_spatial_modes_candidate/figures/`
with `figure5_supplementary_1_direct_snn_spatial_response.{png,pdf}`,
`figure5_supplementary_2_direct_snn_empirical_modes.{png,pdf}`, `README.md`. No edits to
FIGURE_INDEX / main_figure_plan / old Figure 5 directory before user visual acceptance.

**Figure visual contract (Nature main-panel):** width ≈7.2 in, 300 dpi, editable PDF text
(`fonttype=42`), no suptitle, no background grid, no explanatory paragraph on the canvas, short
left-aligned panel letters, concise axis labels/titles, baseline = neutral grey `#555555`,
pre_onset = ochre `#C88719`, NO red/blue (reserved for template A/B semantics). Eyeball the PNG,
never confirm by file existence.

- **Supplementary 1** = fixed-kick response: baseline vs pre_onset for the SAME localized kick;
  5/15/30/50 ms maps sharing one symmetric colorbar; shared-colorbar kymograph; response norm; a
  compact remote/source-energy or first-arrival panel; NO operator gain envelope; caption states
  fixed-kick.
- **Supplementary 2** = empirical modes: baseline + pre_onset V1 input fields, U1 output fields,
  `sigma_hat_1(T)` across registered states, U1 axis/globality/overlap; plateau + D-matched drawn
  as independent symbols (not a time trajectory); NO `Re(λ)` / oscillation freq / "zero
  crossing"; title "Empirical response modes", never "Full-SNN eigenmodes".

README (Chinese, per figure) states this is the direct current-based MZ SNN, not a rate-field
surrogate.

---

## 9. Allowed / forbidden claims

**Allowed:** same MZ spiking scaffold shows different finite-time spatial susceptibility across
slow states; pre-runoff state gives stronger/broader/more persistent response to the same local
stimulus; empirical optimal output becomes more axial (or does not stably become more axial);
z+m plateau changes the spatial response vs D-matched z-only; direct SNN agrees / partly agrees
/ disagrees with the old frozen-q rate-field result.

**Forbidden:** operational runoff = clinical seizure onset; a full interictal→ictal→recovery
cycle reproduced; V1/U1 are exact full-SNN eigenmodes; `sigma_hat_1>1` = net amplification;
kymograph proves a travelling wave; results prove Hopf/fold/Floquet; changing
state/seed/ε/basis/T to rescue a conclusion.

If direct SNN disagrees with the rate field: direct SNN is the model-body result; the rate field
stays a theoretical closure; explain the difference (spike/reset/delay/noise/nonlinearity); do
NOT tune parameters to force agreement. Positive, negative, seed-inconsistent, right-censored,
and linearity-fail outcomes are all valid completions.
