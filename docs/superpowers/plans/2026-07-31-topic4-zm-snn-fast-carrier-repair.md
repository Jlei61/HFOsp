# Topic 4 Phase D — Z/M SNN-native fast-carrier repair implementation plan

**Date:** 2026-07-31

**Status:** execution in progress; source migration gate passed

**Spec:** `docs/superpowers/specs/2026-07-31-topic4-zm-snn-fast-carrier-repair-design.md`

**Pre-result calibration amendment:**
`docs/superpowers/specs/2026-07-31-topic4-zm-fast-carrier-baseline-anchor-amendment.md`

## Goal

On the correct per-neuron Z/M spatial SNN, test whether a unit-safe
conductance membrane, local/weak-global GABA decomposition, and a preregistered
60–160 ms dynamic threshold can create a sustained, bounded, spatially
structured non-tonic carrier. E→E is immutable. Entry, autonomous offset and
recovery remain conditional on a confirmed fast carrier.

Phase-D success at the first gate means `fast_carrier_supported`, not an ictal
lifecycle.

## Global locks

- Canonical substrate: `epilepsiae_1146`, `twoend_equal`,
  \(N_E=32000,N_I=8000\), two original low-threshold cores.
- Correct slow variables: per-neuron \(z_i,m_i\), with starting
  \(\tau_z=5000\) ms, \(\tau_m=500\) ms and the locked Phase-C calibration.
- E→E graph, weights, kernel, anisotropy, orientation, STD and plasticity are
  immutable. Hash them in every manifest and fail on drift.
- Phase-C source states and raw results are read-only. A Phase-D counterfactual
  fork never rewrites or silently re-labels them.
- Arms:
  - A: exact current-based frozen Z/M+\(S_G\) baseline;
  - B: conductance Z/M, local GABA, \(\gamma=0\), \(\phi=0\);
  - C: conductance Z/M, local+weak-global GABA, \(\gamma=1/6\), \(\phi=0\);
  - D: arm C plus the six locked
    \((\tau_\phi,f_\phi)\in\{60,100,160\}\times\{0.15,0.30\}\).
- Conductance arms never divide recurrent E by the old \(S_G\), never call
  `slow.apply_currents()` for the E membrane, and apply \(z,m\) exactly once.
- Primary virtual-SEEG in conductance arms uses
  \(g_E(E_E-V)+g_I(E_I-V)\); the old current proxy is diagnostic only.
- Carrier definitions reuse
  `carrier_gate_v2.1_revised_2026-07-24`; no outcome-dependent thresholds.
- \(H,P\), reset, persistence actuator, E→E tuning and large slow-state grids
  are forbidden.
- Every run uses
  `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=NUMEXPR_NUM_THREADS=1`.
- Reserve at least 96 GB `MemAvailable`; no Phase-D worker may use swap; at
  most 12 full-SNN workers.

---

## Task 1 — Immutable Phase-D contract and source-state compatibility

**Files**

- Create `src/topic4_zm_fast_carrier_contract.py`
- Create `tests/test_topic4_zm_fast_carrier_contract.py`
- Create `scripts/lock_topic4_zm_fast_carrier.py`

**Steps**

- [ ] Lock the spec SHA, starting git SHA and all producer SHAs.
- [ ] Load the Phase-C manifest, coordinate manifest and machine futility
  verdict; require their self-hashes and the 59-run evidence-set hashes to
  validate.
- [ ] Resolve the exact seed-1 source-state panel: one real
  `pre_entry__natural` checkpoint plus rising/peak checkpoints for
  `bounded_mid` and `bounded_late`. Pre-entry uses two locked future-noise
  repeats; it is never relabelled as a nonexistent rising/peak state.
- [ ] Record source config SHA, source engine SHA, state hash, anatomy hash,
  connectivity hash, threshold-field hash, future-noise-bank hash and
  checkpoint time.
- [ ] Define a new Phase-D arm-config SHA. Source-state provenance and
  intervention-config provenance are separate fields; never overwrite the
  source config SHA with the new config.
- [ ] Define an explicit compatible-state migration table:
  `V,ref,s_E,I_E,s_I,I_I,s_E_rec,I_E_rec,ring_sE,ring_sI,xi,rng,t`,
  all Z/M/slow state, and the new deterministic \(\phi=0\) insertion.
- [ ] Require exact dtype, shape, state timing and hash on every carried field.
  Reject any unclassified simulator state.
- [ ] Prove the migrated arm-A state gives exact current-path continuation.
- [ ] Lock analytic reversal choices, scale-factor bounds, gamma panel,
  threshold panel, perturbation calibration, gates and resource policy.
- [ ] Serialize `phaseD_input_manifest.json` first, then write a single
  content-addressed `phaseD_manifest.json` that binds the input and every
  producer. Both are write-once.
- [ ] Mutation-test wrong source engine, missing delay-ring field, altered
  threshold field, cross-seed state, stale noise, nonzero inserted phi and
  E→E drift.

**Gate**

No engine edit or Phase-D run is allowed until state migration is lossless for
arm A and fail-closed for every incompatible fixture.

---

## Task 2 — Pure conductance membrane mathematics

**Files**

- Create `src/snn_engine/zm_conductance.py`
- Create `tests/test_zm_conductance.py`

**Interfaces**

- `@dataclass ZMConductanceConfig`
- `analytic_anchor(V_ref, V_th_median, V_reset, eta_m)`
- `decompose_conductances(I_E, I_I, z, m, cfg, is_E)`
- `conductance_currents(V, g_E, g_I_eff, g_Mm, cfg)`
- `conductance_membrane_step(V, I_E, I_I, z, m, decay_V, is_E, cfg)`

**Steps**

- [x] Implement \(g_L=1,C=\tau_{m,E}\),
  \(E_L=E_K=0,E_I=V_{\rm reset},E_E=2V_\theta-V_{\rm reset}\).
- [x] Implement the analytic current-tangent anchor at \(V_{\rm ref}\):
  \(\kappa_E^0=(E_E-V_{\rm ref})^{-1}\),
  \(\kappa_I^0=(V_{\rm ref}-E_I)^{-1}\),
  \(g_M^0=\eta_m/(V_{\rm ref}-E_K)\).
- [x] Validate \(E_E>V_{\rm ref}>E_I\ge E_K\), finite non-negative inputs,
  gamma in `[0,1]`, and calibration scales in `[0.8,1.2]`.
- [x] Use already-filtered local GABA:
  \(g_I^G=\langle g_I^L\rangle_E\), with no second low-pass.
- [x] Implement primary all-GABA z scaling and the registered local-only-z
  sensitivity as distinct configs.
- [x] Return \(V_\infty,\tau_{\rm eff},I_{\rm exc},I_{\rm inh},I_{\rm sAHP}\)
  for observation without retaining full time-by-neuron arrays.
- [x] Preserve the current update exactly on I cells.
- [x] Test the analytic tangent against
  `I_E-I_I-eta_m*m` at \(V_{\rm ref}\).
- [x] Test signs and monotonicity:
  more GABA lowers \(V_\infty\) and \(\tau_{\rm eff}\);
  \(z\downarrow\) disinhibits; \(m\uparrow\) lowers \(V_\infty\).
- [x] Test uniform-state local/global budget identity and heterogeneous-state
  spatial-rank difference.
- [x] Test exact exponential update against the closed-form scalar solution.
- [x] Mutation-test direct insertion of mV drives into the denominator,
  double filtering of global GABA and double z/m application.

**Gate**

The pure module must pass unit, sign, tangent and exact-update tests before it
is reachable from the SNN loop.

---

## Task 3 — Dynamic-threshold increment in the Z/M slow object

**Files**

- Modify `src/snn_engine/slow_field.py`
- Modify `src/topic4_zm_checkpoint.py`
- Modify `src/topic4_zm_fork_state.py`
- Create `tests/test_zm_dynamic_threshold.py`

**Steps**

- [ ] Add off-by-default `use_phi`, `tau_phi`, `delta_phi` config fields.
- [ ] Allocate an E-only threshold-increment vector \(\phi_i\), initialized to
  zero; I entries remain zero.
- [ ] Return `V_th_base + phi` while preserving the heterogeneous base
  threshold exactly.
- [ ] Update after spikes using
  \(\phi\leftarrow\phi e^{-dt/\tau_\phi}+\Delta_\phi S\), with exact decay.
- [ ] Add `phi` to checkpoint capture/restore and simulator-state tables.
  It is dynamic in frozen-Z/M carrier forks and must not be frozen by
  `FreezeWrapper`.
- [ ] Add streaming mean/max/core/surround phi traces only when enabled.
- [ ] Test single-spike increment, exponential recovery, E-only behavior,
  heterogeneous-threshold preservation and exact-resume parity.
- [ ] Prove `use_phi=False` preserves all existing spike rasters byte-for-byte.

**Gate**

No Phase-D arm may use the legacy absolute-threshold `SlowVars.phi` path. The
new variable is always an increment.

---

## Task 4 — Minimal guarded-engine conductance hook

**Files**

- Modify `src/snn_engine/kick_probe.py`
- Extend `src/snn_engine/slow_field.py`
- Create `tests/test_zm_conductance_engine_hook.py`
- Update the Phase-D engine-version lock only after review

**Steps**

- [ ] Capture the pre-edit current-path golden outputs for:
  slow=None, current Z/M, Z/M+\(S_G\), exact resume, and current virtual-SEEG.
- [ ] Add `uses_zm_conductance()` and a pure delegation method on the Z/M slow
  object. No other slow implementation enters this path.
- [ ] In conductance arms, pass raw `I_E,I_I,I_E_rec,V,z,m` directly to the
  conductance hook. Do not call `apply_currents()` first.
- [ ] Track `I_E_rec` when either old \(S_G\) or conductance mode is on, but
  assert it is not divided in conductance mode.
- [ ] In arms B–D, restore the source \(S_G\) value for provenance/readout but
  disable its old recurrent-E divisor and pool evolution. The only global
  inhibitory term is the instantaneous spatial mean of already-filtered GABA.
- [ ] Stash raw pre-z `I_I` for the Z sensor before `slow.step`, including the
  later dynamic-Z/M stage; never infer the sensor from effective conductance.
- [ ] Keep the original branch literally unchanged when conductance is off.
- [ ] Compute conductance virtual-SEEG from synaptic currents with the same
  electrode weights. Record the old current proxy separately.
- [ ] Record sAHP separately and never add it to synaptic carrier energy.
- [ ] Add fixed-panel streaming diagnostics for
  \(V_\infty,\tau_{\rm eff},g_E,g_I^L,g_I^G,g_Mm,\phi\).
- [ ] Prove no new RNG draw, delay-ring change or scatter-order change.
- [ ] Run all historic baseline-SHA tests and exact-resume tests.
- [ ] Review the guarded diff line-by-line; only then write a new Phase-D
  engine-version lock. Do not rewrite old Phase-C manifests or artifacts.

**Gate**

Default-off and arm-A outputs must be byte-identical to their pre-edit goldens.
A mere statistical match is insufficient.

---

## Task 5 — Counterfactual state-fork migration

**Files**

- Create `src/topic4_zm_fast_carrier_state.py`
- Create `tests/test_topic4_zm_fast_carrier_state.py`

**Steps**

- [ ] Load each source state under its original Phase-C contract first.
- [ ] Copy the exact fast, synaptic, delay, refractory, Z/M and RNG state into
  the Phase-D state schema.
- [ ] Insert only new deterministic state: `phi_increment=zeros(N)`.
- [ ] Preserve absolute time and future-noise alignment.
- [ ] Reconstruct arm A and require exact continuation against the original
  runner for 500 ms.
- [ ] Construct arms B–D from the same migrated bytes and record the first
  intentional divergence step.
- [ ] Require identical pre-divergence spikes and observables for any
  delayed-onset diagnostic arm.
- [ ] Save source-state hash, migrated-state hash, transformation payload and
  arm-config hash in every part.

**Gate**

If arm A cannot be reproduced exactly after migration, the Phase-D experiment
is invalid and stops.

---

## Task 6 — Analytic anchor and baseline-only calibration

**Files**

- Create `src/topic4_zm_fast_carrier_calibration.py`
- Create `tests/test_topic4_zm_fast_carrier_calibration.py`
- Create `scripts/calibrate_topic4_zm_fast_carrier.py`

**Steps**

- [ ] Measure the locked free-E voltage distribution in the pre-entry state,
  before candidate runs. Report reversal-crossing fractions explicitly.
- [ ] Build the positive distribution-magnitude anchor from
  \(D_E=\operatorname{median}(E_E-V)\),
  \(D_I=\operatorname{median}|V-E_I|\), and
  \(D_M=\operatorname{median}|V-E_K|\), then construct the deterministic
  `[0.8,1.2]^3` scale-factor lattice. Do not claim signed pointwise current
  equivalence below a reversal.
- [ ] Evaluate only slow-off baseline criteria:
  returning-event count/rate, source ordering, two-source geometry,
  \(V_\infty\), E/I effective charge ratio, \(\tau_{\rm eff}\), prevention and
  all-sheet plateau.
- [ ] Verify that active/free-E \(V_\infty>E_I\) during the carrier-supporting
  portions of the reference. If this is not true, report the fraction and do
  not interpret increasing GABA conductance as a uniformly hyperpolarizing
  feedback; reversal-clamped shunting may raise sub-reversal \(V_\infty\)
  while still shortening \(\tau_{\rm eff}\).
- [ ] Use lexicographic objective order and deterministic distance-to-
  `(1,1,1)` tie-break from the spec.
- [ ] Reject settings that suppress returning events, create a baseline
  plateau or fail any hard constraint.
- [ ] Write one content-addressed, write-once calibration lock before opening
  bounded-mid/late candidate outcomes.
- [ ] Test deterministic selection, no candidate-data access, fail-closed no
  solution, and immunity to directory iteration order.

**Gate**

No valid calibration means `NO_GO_baseline_calibration_failed`; do not widen
the scale bounds after observing failure.

---

## Task 7 — Fixed perturbation and carrier observation contract

**Files**

- Create `src/topic4_zm_fast_carrier_observation.py`
- Create `src/topic4_zm_fast_carrier_verdict.py`
- Create corresponding tests

**Steps**

- [ ] Calibrate one 50 ms E-threshold pulse on arm A bounded-mid before arms
  B–D are inspected: smallest amplitude causing 50–70% paired core-spike
  reduction without ≥100 ms all-sheet rest.
- [ ] Freeze pulse amplitude, mask, onset and duration.
- [ ] Reuse corrected Phase-C source morphology and the exact
  `carrier_gate_v2.1_revised_2026-07-24` observation semantics.
- [ ] Implement all run gates: all-sheet runaway, active area, rest dwell,
  occupancy, tail escalation, modulation, cycles/bursts, CV, participating
  cells, \(\rho_{80}\), 30–80 Hz macroepisode, HFO-train rejection, spatial
  first passage and phase/latency reproducibility.
- [ ] Require paired no-pulse and pulse continuations; quantify return within
  1 s to the same carrier class, frequency and spatial ordering.
- [ ] Add synthetic fixtures for tonic plateau, HFO-like burst train,
  synchronized flash, bounded periodic carrier, clonic carrier, metastable
  survival without return, and valid perturbation return.
- [ ] Mutation-test core/all-sheet rate swaps, raw-current/conductance-current
  proxy swaps and use of sAHP in the spectral carrier readout.
- [ ] Emit independent fields for fast carrier, spatial pattern, virtual-SEEG,
  perturbation return, entry, offset and recovery.

**Gate**

No waveform can pass solely by lowering the tonic mean rate or increasing HFO
burst energy.

---

## Task 8 — Crash-safe atomic runner and resource controller

**Files**

- Create `scripts/run_topic4_zm_fast_carrier_cell.py`
- Create `scripts/run_topic4_zm_fast_carrier.py`
- Create `scripts/analyze_topic4_zm_fast_carrier.py`
- Reuse `src/topic4_zm_phasec_resources.py`
- Create runner/coordinator tests

**Steps**

- [ ] One atomic part per arm × checkpoint × fast phase × noise × perturbation.
- [ ] Validate manifest, state, engine, config, calibration, pulse and noise
  hashes before every run.
- [ ] Publish content-addressed observables, terminal part JSON and adjacent
  resource receipt atomically.
- [ ] Resume only exact matching terminal parts; quarantine partial or
  mismatched outputs under `invalidated/`.
- [ ] Add `--smoke`, `--confirm-run`, `--resume`, `--max-workers`,
  `--arms`, `--checkpoints` and `--T`.
- [ ] Measure one complete full-SNN worker before choosing concurrency.
- [ ] Compute safe workers from measured RSS, keeping 96 GB memory reserve,
  eight logical CPUs and a hard cap of 12 workers.
- [ ] Stop launches on low memory, any Phase-D worker swap, receipt failure or
  peer-process ambiguity. Never kill peer worktree processes.
- [ ] Log worker PID ownership, pre-publish self-snapshot, sample count/max RSS,
  `MemAvailable`, system swap baseline/delta and cleanup.
- [ ] Analyzer requires the exact expected matrix; partial evidence receives
  `blocked_evidence`, never a default verdict.

**Gate**

Smoke artifacts live outside production and cannot satisfy production
coverage.

---

## Task 9 — Preflight, calibration and seed-1 cheap screen

**Production order**

1. Lock Phase-D inputs and source-state compatibility.
2. Run arm-A migration parity.
3. Run one full worker, choose safe concurrency and calibrate baseline.
4. Freeze the perturbation pulse.
5. Run seed-1 arms A–D on bounded-mid and bounded-late, two fast phases,
   replayed future noise, 4 s.
6. Analyze without modifying settings or thresholds.

**Immediate stopping rule**

- If no D setting passes all run-level gates in both fast phases at either
  tonic checkpoint: `NO_GO_fast_carrier_not_repaired`; stop.
- If D yields only rest-separated HFO/HYP bursts:
  `NO_GO_hfo_like_burst_train`; stop.
- If one or more D settings pass, lock all passing settings. Do not add a
  near-miss or interpolate a new \(\Delta_\phi\).

---

## Task 10 — Conditional multi-seed and numerical confirmation

Only locked seed-1 candidates proceed.

- [ ] Seeds 3/4, two fast phases, three future-noise continuations, 8 s.
- [ ] Independent native \(dt/2\) on seeds 1/3.
- [ ] Require ≥5/6 passes per seed, both phases represented, native seeds 1/3
  concordant and third seed non-opposite.
- [ ] Require frequency agreement within 20% under \(dt/2\).
- [ ] Run B/C/D and local/global/phi ablations on the same states/noise.
- [ ] Estimate E/I phase lag, Poincaré map, perturbation return and a
  finite-difference reduced monodromy/Jacobian.
- [ ] Locate a frozen fast-carrier boundary before any dynamic Z/M run.

**Gate**

Only this task can emit `fast_carrier_supported`.

---

## Task 11 — Conditional dynamic Z/M lifecycle test

Only `fast_carrier_supported` authorizes this task.

- [ ] Re-enable dynamic \(z,m,\phi\) from the original returning-interictal
  substrate.
- [ ] Test spontaneous repeated-event entry across the frozen carrier boundary.
- [ ] Require endogenous offset without \(H,P\), actuator, reset or clamp.
- [ ] Require return to the same irregular intermittent-event basin; no fixed
  interictal rhythm.
- [ ] Test early-retrigger suppression and late-retrigger recovery.
- [ ] Replicate across the locked seeds and noise streams.

**Claim gate**

Only entry + bounded carrier + autonomous offset + irregular recovery supports
`recoverable_ictal_lifecycle_candidate`. Otherwise report the failed leg
explicitly.

---

## Task 12 — Figures, archive, tests, commits and push

- [ ] Create
  `results/topic4_sef_hfo/zm_fast_carrier_repair/figures/README.md`.
- [ ] First diagnostic: arm A–D source rate, E/I lag, virtual-SEEG energy,
  spatial kymograph, conductances, \(\phi,z,m\), and perturbation return.
- [ ] Do not create a Figure-5 lifecycle layout unless Task 11 passes.
- [ ] Write one machine verdict with separate carrier/entry/offset/recovery/
  spatial/virtual-SEEG/claim-boundary fields.
- [ ] Write a Chinese archive with tested question, method, evidence, caveats,
  allowed claims and exact next gate.
- [ ] Update `docs/topic4_sef_hfo.md`, archive index and figure index.
- [ ] Run focused tests, all Z/M/Phase-C regressions, baseline SHA tests,
  exact-resume tests, `git diff --check`, process/resource audit and artifact
  hash audit.
- [ ] Commit in logical batches and push only this branch.

## Final completion boundary

This plan is complete when the registered Phase-D branch has either:

1. stopped honestly at baseline calibration or seed-1 fast-carrier futility; or
2. produced a replicated `fast_carrier_supported` result and completed only
   the conditional downstream stages it authorizes.

Engineering green alone is not scientific success, and fast-carrier success is
not lifecycle success.
