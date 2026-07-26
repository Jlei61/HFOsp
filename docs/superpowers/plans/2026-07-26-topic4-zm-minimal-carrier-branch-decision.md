# Z/M minimal-carrier-subsystem branch decision — implementation plan

> **Spec:** `docs/superpowers/specs/2026-07-26-topic4-zm-minimal-carrier-branch-decision-design.md`
> revision 3.1.
> **Scope:** Phase 0–3 branch decision only. No Branch T/F/M/P/A implementation.

**Goal:** Determine, on the unchanged anisotropic 40k-neuron Z/M SNN, whether a
bounded carrier exists in the smallest subsystem E/I, E/I+\(M\), E/I+\(S_G\), or
E/I+\(M+S_G\); distinguish a missing carrier from a slow trajectory that misses
a nearby carrier window; then measure slow-coordinate functional rank, map
\(Z\)-entry and existing-slow-coordinate offset boundaries, and
select the next mechanism branch.

**Architecture:** Add a minimal off-by-default checkpoint/freeze hook to the
canonical guarded simulator, keeping a single integration loop; serialize every
dynamic state; generate matched future noise; run an early end-to-end vertical
slice; lock empirical carrier/readout thresholds; run multi-seed ×
natural-fast-phase minimal-subsystem forks; perform coarse and spatial
slow-manifold audits; then run functional-rank before modal and entry/offset
analyses. A pure fail-closed adjudicator writes one branch verdict.

**Non-negotiable constraints:**

- only `src/snn_engine/kick_probe.py` may receive the minimal guarded
  checkpoint/freeze hook; connectivity, neuron, parameter, and LFP engine files
  remain untouched;
- record the old guarded SHA and diff, prove default-path byte parity, then
  update only the `kick_probe.py` guard entry; never claim byte parity removes
  the need to re-bless;
- do not change E→E;
- no H, q_I/g_K fallback, new exit current, or actuator;
- seed 1 is discovery only; no Branch F from one seed;
- no production run before state parity and the relevant target lock pass; if
  early-ictal reference artifacts are blocked, source-space forks may proceed
  under an explicit `observation_layer_blocked` status;
- source-only carrier may continue diagnostics but cannot authorize an actuator;
- use `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1`;
- one full SNN worker initially; at most two after measured RSS, with
  `MemAvailable >= 96 GB` and no swap growth.

## Global file map

New modules:

- `src/topic4_zm_fork_state.py` — config/state schema, serialization, freeze policy;
- `src/topic4_zm_checkpoint.py` — schema/serializer/controller used by the
  canonical simulator hook;
- `src/topic4_zm_noise_bank.py` — paired external-noise streams;
- `src/topic4_zm_empirical_carrier.py` — reference resolver and data-locked gate;
- `src/topic4_zm_anchor_states.py` — state-bin/fast-phase selection;
- `src/topic4_zm_minimal_carrier.py` — fork matrix and survival/stationarity metrics;
- `src/topic4_zm_neighbourhood.py` — slow-trajectory PCA/local continuation;
- `src/topic4_zm_modal_operator.py` — perturbation/operator audit;
- `src/topic4_zm_boundaries.py` — entry/offset probability boundaries;
- `src/topic4_zm_effective_rank.py` — dimensionless sensitivity/SVD;
- `src/topic4_zm_exit_drivers.py` — matched offline driver comparison;
- `src/topic4_zm_branch_verdict.py` — pure fail-closed branch adjudicator.

New orchestration/plotting:

- `scripts/run_topic4_zm_branch_decision.py`;
- `scripts/plot_topic4_zm_branch_decision.py`;
- `scripts/topic4_zm_resource_monitor.py`.

Results root:

- `results/topic4_sef_hfo/zm_branch_decision/`.

No result file is committed before it exists and passes provenance validation.

---

### Task 1 — Canonical config and dynamic-state inventory gate

**Files**

- Create `src/topic4_zm_fork_state.py`
- Create `tests/test_topic4_zm_fork_state.py`
- Create `scripts/audit_topic4_zm_dynamic_state.py`

**Deliverables**

- versioned `CanonicalConfigV1`;
- versioned `SimulationStateV1`;
- machine-readable inventory rows:
  `name/category/shape/dtype/time_scale/simulator_or_observer/dt_dependent/`
  `snapshot/freeze_semantics/current_effect`;
- `canonical_config.json`, `state_inventory.json`;
- SHA256 of all six guarded engine files and all effective config fields.

**Steps**

- [ ] Write failing tests that require every mutable variable in the canonical loop
  to appear exactly once in the inventory.
- [ ] Test that observer-only fields cannot be marked as membrane-current state.
- [ ] Test that an unknown current-affecting field makes the audit fail closed.
- [ ] Resolve the canonical Z/M+\(S_G\) config from the real builder, not duplicated
  constants.
- [ ] Include \(V\), refractory counters, synaptic gates/currents, recurrent-E
  currents, delay rings/cursor, OU state, external-noise state, complete RNG,
  \(z/m\), last \(I_I^E\), \(r_E^\mathrm{fast}/\mu_G/S_G\), optional feature
  switches, and observer buffers.
- [ ] Run:
  `python scripts/audit_topic4_zm_dynamic_state.py --write-lock`.
- [ ] Stop with `blocked_state_inventory` if any state is unresolved.
- [ ] Commit:
  `feat(topic4): lock ZM fork state inventory and canonical config`.

**Hard gate:** Task 2 cannot start unless the inventory is complete.

---

### Task 2 — Minimal canonical-simulator checkpoint hook

**Files**

- Modify `src/snn_engine/kick_probe.py` (only guarded engine edit)
- Create `src/topic4_zm_checkpoint.py`
- Create `tests/test_topic4_zm_checkpoint_hook.py`
- Modify only the `src/snn_engine/kick_probe.py` key in
  `results/topic4_sef_hfo/snn_heterogeneity/engine_versions.json`, after tests

**Design**

Keep one integration loop. Add a fully gated controller to `simulate_kick`
supporting:

- `initial_state`;
- `start_step`;
- `return_final_state`;
- externally supplied noise bank;
- per-coordinate freeze policy;
- snapshot callback.

When every new argument is default/off, the original operation order, RNG draws,
allocations, and floating-point graph must remain unchanged.

**Steps**

- [ ] Record the pre-edit `kick_probe.py` SHA and save a pre-edit callable/fixture.
- [ ] Write failing tests for a short nontrivial state with nonempty delay buffers,
  active Z/M/\(S_G\), nonzero recurrent current, and refractory neurons.
- [ ] Implement only gated capture/restore/freeze calls; state packing lives in
  `topic4_zm_checkpoint.py`, not in the timestep mathematics.
- [ ] Compare pre-edit vs post-edit default paths for spike raster, source/current
  traces, and final RNG state using byte equality.
- [ ] Add a test proving a hook-enabled run with no snapshot request does not add
  RNG draws or change results.
- [ ] Add a mutation test that changes one update order and prove parity fails.
- [ ] Keep existing `BASELINE_SHA=da5fc18c27d5340a` tests green.
- [ ] Run:
  `pytest -q tests/test_topic4_zm_checkpoint_hook.py tests/test_snn_shunting.py tests/test_snn_gates.py tests/test_zm_slow_field_parity.py tests/test_a1c_feedback.py`.
- [ ] Only after parity passes, update the single guarded SHA entry and record
  old SHA, new SHA, exact diff, tests, and reason in the Phase-0 manifest.
- [ ] Commit:
  `feat(topic4): add guarded parity-locked SNN checkpoint hook`.

**Hard gate:** no snapshot/fork run and no re-bless unless default-path parity
passes. Any second copied simulator loop is a P0 rejection.

---

### Task 3 — Exact snapshot/restore and freeze semantics

**Files**

- Modify `src/topic4_zm_fork_state.py`
- Modify `src/topic4_zm_checkpoint.py`
- Create `tests/test_topic4_zm_exact_resume.py`

**Interfaces**

- `save_state_npz(state, manifest, path)`;
- `load_state_npz(path, expected_config_sha, expected_engine_sha)`;
- `FreezePolicy(freeze_z, freeze_m, freeze_sg_family)`;
- `apply_freeze(policy, reference_state, evolving_state)`.

No pickle or object arrays.

**Steps**

- [ ] Split a run at a nontrivial state and require continuous vs restored
  continuation byte equality.
- [ ] Test three fork times: trough, rise, and peak.
- [ ] Test that frozen \(z_i/m_i\) retain their spatial fields and current effects.
- [ ] Test that primary frozen \(S_G\) freezes
  \(r_E^\mathrm{fast},\mu_G,S_G\) together.
- [ ] Test that frozen output with drifting internal pool has a distinct diagnostic
  policy and cannot be labelled primary.
- [ ] Test schema/version/config/engine/state-hash mismatch rejection.
- [ ] Test `dt` mismatch rejection.
- [ ] Run exact-resume fixture twice to exclude accidental RNG alignment.
- [ ] Commit:
  `feat(topic4): implement exact ZM state resume and explicit freeze policy`.

**Hard gate:** scientific fork runner stops at `blocked_exact_resume` on any failure.

---

### Task 3.5 — Mandatory end-to-end vertical-slice smoke

**Kill question:** Can one real seed-1 state travel through
anchor → checkpoint → restore → `freeze_all` → continuation → source/current
metrics without state, readout, or provenance discontinuity?

**Files**

- Create `tests/test_topic4_zm_vertical_slice.py`
- Extend `scripts/run_topic4_zm_branch_decision.py`

**Steps**

- [ ] Run a short seed-1 anchor to a naturally nontrivial state.
- [ ] Snapshot and restore it through the canonical simulator hook.
- [ ] Continue one short `freeze_all` arm.
- [ ] Produce source-rate, current-vSEEG, rest-distance, and preliminary carrier
  metric schemas.
- [ ] Verify observer filters remain continuous across restore.
- [ ] Verify config/state/noise/engine hashes round-trip into the output.
- [ ] Write only under `smoke/`; assert smoke cannot update production summary.
- [ ] Treat any interface mismatch as a redesign stop before Tasks 4–14.
- [ ] Commit:
  `test(topic4): close checkpoint-to-carrier vertical slice`.

This smoke carries no scientific carrier evidence.

---

### Task 4 — Paired future-noise bank

**Files**

- Create `src/topic4_zm_noise_bank.py`
- Create `tests/test_topic4_zm_noise_bank.py`

**Interfaces**

- `build_noise_bank(config, seed, start_step, n_steps, replicate)`;
- replicates `noise_replay`, `noise_resample_1`, `noise_resample_2`,
  `mean_input_only`;
- independent simulator and observer RNG streams.

**Steps**

- [ ] Test arm-order invariance: the same bank yields identical external input
  regardless of execution order.
- [ ] Test replay reproduces the anchor's future stream exactly.
- [ ] Test resamples preserve mean/variance/autocorrelation within locked tolerance.
- [ ] Test `mean_input_only` preserves external mean while removing fluctuations.
- [ ] Test that disabling noise without restoring mean is rejected.
- [ ] Store bank SHA in every fork manifest.
- [ ] Commit:
  `feat(topic4): add matched future-noise continuations`.

---

### Task 5 — Empirical carrier/readout lock

**Kill question:** Can the locked readout distinguish early-ictal broadband
recruitment from both sharp harmonic pulse trains and a stationary global
low-frequency oscillator?

**Files**

- Create `src/topic4_zm_empirical_carrier.py`
- Create `tests/test_topic4_zm_empirical_carrier.py`
- Create `scripts/lock_topic4_zm_carrier_target.py`

**Inputs**

Resolve immutable artifacts for:

- real returning interictal group events;
- real early-ictal windows;
- matched synthetic sharp pulse trains;
- E1146 current-based readout geometry/kernel.

If any mandatory input is unavailable, write
`blocked_reference_artifacts.json` and stop. Do not synthesize a replacement
early-ictal distribution from the model.

**Metrics**

- duration, duty/occupancy, energy, spatial extent;
- independent-contact count using kernel-width separation;
- harmonic-comb concentration;
- spectral entropy/broadband continuity;
- instantaneous-frequency drift;
- burst-interval CV and temporal phase coherence;
- wavefront-velocity variability;
- spatial phase entropy;
- axial first passage;
- multivariate \(d_\mathrm{rest}\);
- current-based primary and rate-proxy sensitivity.

**Steps**

- [ ] Write synthetic tests: sharp pulse comb fails broadband continuity;
  continuous noisy broadband passes; adjacent contacts from one hotspot count as
  one independent contact.
- [ ] Add a mean-rate/frequency/energy-matched synchronized global-oscillator null;
  a fixed ~5 Hz whole-field rhythm must fail the temporal/spatial organisation gate.
- [ ] Test all thresholds are read from the lock; no historical `0.8/2s` literal
  can silently become primary.
- [ ] Lock null quantiles, early-ictal empirical intervals, multiplicity,
  missing-data rule, sample-count minimum, rest-distance/dwell rule.
- [ ] Verify current-based and rate-based readouts are labelled separately.
- [ ] Write `carrier_target_lock.json` with input SHA256s.
- [ ] Commit:
  `feat(topic4): lock empirical source and observation carrier targets`.

**Hard gate:** no observation-matched claim without this lock. Source-space forks
may proceed only if the state gates are green.

---

### Task 6 — Multi-seed anchors and natural fast-phase state selection

**Kill question:** Do at least three locked primary seeds provide bounded
trajectory states with naturally sampled trough/rise/peak microstates, without
retuning or manual fast-state resets?

**Files**

- Create `src/topic4_zm_anchor_states.py`
- Create `tests/test_topic4_zm_anchor_states.py`
- Extend `scripts/run_topic4_zm_branch_decision.py`

**Protocol**

- discovery seed: 1;
- primary seeds: `{1,3,4}`;
- bins: `pre_entry`, `onset_adjacent`, `bounded_early/mid/late`;
- phases per bounded bin: `trough`, `rising`, `peak`.

**Steps**

- [ ] Test slow-trajectory arc-length/quantile bin selection on synthetic
  trajectories with unequal time speed.
- [ ] Test fast-phase classifier uses local temporal derivatives and event context,
  not fixed clock time.
- [ ] Require candidate snapshots within a locked slow-distance tolerance.
- [ ] Reject manual membrane/synaptic resets.
- [ ] Run seed-1 anchor with durable resource monitoring.
- [ ] If seed 1 forms no bounded anchor, record it; do not retune.
- [ ] Run seeds 3 and 4 only after measured seed-1 RSS leaves the required margin.
- [ ] Write per-snapshot NPZ + manifest + hash and
  `phase1_anchor_lock.json`.
- [ ] If fewer than three bounded anchors exist, top-level status =
  `insufficient_bounded_anchors`; no Branch F.
- [ ] Commit code and locks separately; large ignored results remain on disk with
  durable manifests.

---

### Task 7 — Minimal carrier-subsystem fork matrix and stationarity verdict

**Kill question:** Does any natural visited slow state support a statistically
persistent active regime independent of lifecycle drift, and what is its
smallest dynamic subsystem?

**Files**

- Create `src/topic4_zm_minimal_carrier.py`
- Create `tests/test_topic4_zm_minimal_carrier.py`
- Extend `scripts/run_topic4_zm_branch_decision.py`

**Arms**

- `dynamic_replay`;
- `freeze_z`;
- `freeze_zm`;
- `freeze_zsg`;
- `freeze_all`;
- `dynamic_z_only`.

Each is crossed with
`noise_replay/noise_resample_1/noise_resample_2`.

**Steps**

- [ ] Unit-test the exact arm freeze table.
- [ ] Test burn-in:
  `max(250 ms, 2*tau_max_dynamic_carrier_variable)`.
- [ ] Implement streaming survival, drift, lifetime, rest-distance, source gate,
  observation gate, instantaneous-frequency drift, burst-interval CV, phase
  coherence, wavefront-velocity variation, and spatial phase-entropy metrics.
- [ ] Synthetic-test all taxonomy outcomes, including plateau/runaway,
  pulse-train false broadband, fixed global oscillator, isolated candidate,
  stable carrier, metastable carrier, transient carrier-like regime, and
  probabilistically indeterminate.
- [ ] Estimate beta-binomial \(P_\mathrm{carrier}(8s)\) across natural
  fast-phase × future-noise replicas with intervals.
- [ ] Apply locked classes:
  stable \(P>0.8\) with bounded variance;
  metastable \(0.3<P\le0.8\) with lifetime beyond matched IEDs;
  transient \(P\le0.3\) or IED-like lifetime;
  HFO train = repeated rest-basin reset.
- [ ] Require compatible stable/metastable support in two adjacent slow bins and
  convergence across at least two natural fast phases.
- [ ] Require confirmation in at least two of three eligible primary seeds for a
  positive carrier-window verdict; seed 1 alone remains isolated/discovery.
- [ ] Require three eligible bounded seeds for a formal negative.
- [ ] Determine smallest positive subsystem by partial order; retain multiple
  minimal positives if tied.
- [ ] Run seed-1 cheap forks first; stop expensive expansion on
  `runaway/plateau/no_evidence` implementation failures, not on a scientific
  negative.
- [ ] Confirm central positive candidates for 20 s.
- [ ] Run an independent \(dt/2\) anchor and homologous fork; never reuse a
  \(dt\) snapshot.
- [ ] Commit:
  `feat(topic4): adjudicate minimal dynamic carrier subsystem`.

**Decision gate**

- carrier at visited states → Tasks 9A–11;
- no carrier/isolated only → Task 8;
- source-only carrier → Tasks 9A–11 allowed, Task 12 actuator selection blocked
  from implementation.

---

### Task 8 — Local slow-trajectory neighbourhood audit and Branch T/F split

**Kill question:** Is the carrier absent from the local slow-state manifold, or
does the actual trajectory merely miss a nearby spatial slow-field direction?

**Files**

- Create `src/topic4_zm_neighbourhood.py`
- Create `tests/test_topic4_zm_neighbourhood.py`
- Extend runner/plotter.

**Steps**

- [ ] Build locked feature vector
  `[z_core,z_surround,dz_axis,m_core,m_surround,dm_axis,SG]`.
- [ ] Robust-standardize and fit the two-dimensional coarse decision PCA on
  locked primary trajectories only.
- [ ] Fit a separate full-field PCA on vectorized `[z_i,m_i,SG]`, retain and map
  the first three spatial modes.
- [ ] Compute preregistered pathology-axis parallel/perpendicular projections,
  axial gradients, and core-boundary displacement.
- [ ] Test PCA sign/order determinism.
- [ ] Lock a small \((a,b)\) lattice along \(u_1,u_2\), no more than one robust
  trajectory SD and preferably interpolation between visited states.
- [ ] Reconstruct full \(z_i/m_i/S_G\) fields through trajectory interpolation/PCA;
  reject independent arbitrary scalar combinations.
- [ ] Run matched sensitivities along the first three field modes and pathology-axis
  directions after the primary coarse audit.
- [ ] Run the same minimal-subsystem and paired-noise gate on neighbourhood states.
- [ ] Pure verdict:
  local positive in at least two of three eligible primary seeds →
  `branch_T_slow_trajectory_repair`;
  adequately replicated local negative with no representation disagreement →
  `branch_F_fast_carrier_repair`;
  coarse/spatial disagreement → `representation_sensitive_no_branch`;
  otherwise `no_evidence`.
- [ ] Mutation-test that missing seeds or missing noise replicas can never default
  to Branch F.
- [ ] Commit:
  `feat(topic4): separate slow-path miss from fast-carrier absence`.

**Stopping point:** A Branch T or F verdict ends execution. Do not implement the branch.

---

### Task 9A — Slow-coordinate functional-rank audit

**Kill question:** Do \(Z/M/S_G\) provide more than one locally independent
dynamical direction, or are they functionally rank-1 near the carrier?

**Files**

- Create `src/topic4_zm_effective_rank.py`
- Create `tests/test_topic4_zm_effective_rank.py`

**Steps**

- [ ] Implement central finite differences under paired future noise.
- [ ] Standardize inputs/outputs by robust trajectory scales.
- [ ] Compute static-observable and impulse-response matrices separately at at
  least three states.
- [ ] Bootstrap over seeds and natural microstates.
- [ ] Report singular values and uncertainty intervals.
- [ ] Synthetic-test unit invariance: rescaling \(m\), rate, or energy units cannot
  change inferred rank.
- [ ] Label near-rank-1 as local functional collinearity only.
- [ ] Route near-rank-1 directly to joint existing-coordinate offset audit; do not
  over-interpret an M-alone atlas.
- [ ] Commit:
  `feat(topic4): add unit-invariant slow-coordinate rank diagnostic`.

---

### Task 9B — Trajectory-conditioned modal/operator audit

**Kill question:** Is the pathological axis the softest eigenmode, the optimal
finite-time amplification direction, or neither?

**Files**

- Create `src/topic4_zm_modal_operator.py`
- Create `tests/test_topic4_zm_modal_operator.py`
- Extend runner/plotter.

**Steps**

- [ ] Coarse-grain E/I activity on a locked grid and verify rate reconstruction.
- [ ] Define equal-energy axial, transverse, isotropic, core, and random
  perturbations.
- [ ] Test at least three amplitudes and identify a local linear range.
- [ ] Fit on a subset of perturbations; require held-out prediction error below a
  pre-locked tolerance derived from synthetic fixtures.
- [ ] Fixed state → eigen analysis; periodic carrier → stroboscopic/Floquet;
  stochastic carrier → DMD/linear-response plus finite-time SVD.
- [ ] Test tool routing from carrier type; averaging a periodic carrier then using a
  fixed-state Jacobian must raise.
- [ ] Output \(\alpha(q)\), \(G(T,q)\), axial–transverse contrast, left/right modes,
  optimal input, axis angle, speed ratio, prediction error, and linearity range.
- [ ] Validate mode-axis computation on rotated synthetic operators.
- [ ] Commit:
  `feat(topic4): add trajectory-conditioned modal and gain audit`.

This task is explanatory and does not by itself override the carrier verdict.

---

### Task 10 — Phase 2A: Z-entry probability boundary

**Kill question:** Does the actual \(Z\) trajectory cross a reproducible carrier
entry boundary, rather than merely increase burst density?

**Files**

- Create `src/topic4_zm_boundaries.py`
- Create `tests/test_topic4_zm_boundaries.py`
- Extend runner/plotter.

**Steps**

- [ ] From matched rest/interictal states, interpolate actual \(z_i(t)\) fields
  along the trajectory manifold.
- [ ] Hold the non-entry coordinates consistently with the smallest carrier
  subsystem.
- [ ] Use locked IED-like perturbation or matched natural background; do not tune
  kick amplitude per state.
- [ ] Run paired future-noise continuations and estimate
  \(P_\mathrm{enter}\) with bootstrap intervals.
- [ ] Locate `P_enter=0.5` only when bracketed; otherwise report censored/no boundary.
- [ ] Test actual trajectory crossing direction.
- [ ] Synthetic-test monotonic, nonmonotonic, unbracketed, and hysteretic cases.
- [ ] Commit:
  `feat(topic4): map trajectory-manifold Z entry boundary`.

---

### Task 11 — Phase 2B: existing slow-coordinate offset boundary

**Kill question:** Can any existing slow direction—\(M\) alone, \(M+S_G\), or
\(M+Z\)-recovery coupling—cross a distinct offset surface without preventing
carrier formation?

**Files**

- Modify `src/topic4_zm_boundaries.py`
- Extend `tests/test_topic4_zm_boundaries.py`
- Extend runner/plotter.

**Steps**

- [ ] Use actual `bounded_early/mid/late` joint \(z_i/m_i/S_G\) states.
- [ ] Run nested families: M alone; M+\(S_G\); M+\(Z\)-recovery coupling, with
  \(S_G\) treated according to the minimal carrier subsystem.
- [ ] Interpolate joint fields along trajectory or locked PCA directions; scalar
  whole-field scaling is sensitivity only.
- [ ] Start every state from active-carrier and matched-low initial conditions.
- [ ] Estimate \(P_\mathrm{remain}\), bootstrap `P_remain=0.5`, and crossing
  direction.
- [ ] Compare onset and offset surfaces/hysteresis.
- [ ] Detect whether dynamic \(M\) is required by the minimal carrier subsystem.
- [ ] Lock a small near-boundary extension before inspecting outcomes.
- [ ] Adjudicate:
  `existing_slow_offset_reached`,
  `M_sufficient_and_reached`,
  `M_SG_joint_offset_reached`,
  `M_Z_recovery_offset_reached`,
  `M_boundary_near_but_unreached`,
  `M_boundary_far_unreached`,
  `M_is_carrier_component`,
  `M_shapes_but_no_offset_surface`,
  `no_M_evidence`.
- [ ] Permit Phase 3 only after every valid existing-coordinate family fails;
  route narrowly unreached M-alone boundary to Branch M-calibration.
- [ ] Commit:
  `feat(topic4): map probabilistic existing slow-coordinate offset surfaces`.

---

### Task 12 — Matched offline exit-driver selection

**Kill question:** Does any candidate driver carry information beyond matched
duration, spikes, energy, peak rate, active-neuron count, and mean activity?

**Files**

- Create `src/topic4_zm_exit_drivers.py`
- Create `tests/test_topic4_zm_exit_drivers.py`
- Extend runner/plotter.

**Gate**

Run only if:

- at least a source-space carrier exists;
- all valid existing slow-coordinate families lack a usable offset;
- state/readout provenance is complete.

No actuator is implemented.

**Steps**

- [ ] Compute `D_mean` as negative control, plus local load, area integral,
  positive area-flux integral, and suprathreshold-duration candidates.
- [ ] Match/reweight carrier and IED windows on duration, total spikes, energy,
  peak rate, and active-neuron count.
- [ ] Add spatial and temporal shuffle controls.
- [ ] Test incremental information beyond mean rate/total spike count using a
  locked nested-model or conditional-information metric.
- [ ] Implement lexicographic gate:
  separation → trough persistence → within-carrier accumulation → incremental
  information → recovery-compatible decay → three-seed direction → shuffle loss.
- [ ] Test that a duration-only integrator fails matched controls.
- [ ] Keep area and flux mechanistically separate.
- [ ] Output at most one simplest passing mechanism family; tie =
  `driver_selection_ambiguous`.
- [ ] Commit:
  `feat(topic4): select independent exit observable with matched controls`.

**Stopping point:** write a later spec for one selected actuator. Do not build it.

---

### Task 13 — Pure branch adjudicator, orchestration, and crash-safe resume

**Files**

- Create `src/topic4_zm_branch_verdict.py`
- Create `tests/test_topic4_zm_branch_verdict.py`
- Complete `scripts/run_topic4_zm_branch_decision.py`
- Complete `scripts/topic4_zm_resource_monitor.py`

**Steps**

- [ ] Implement the exact top-level verdict vocabulary from spec §13.
- [ ] Write one fixture per verdict and adversarial missing-field fixtures.
- [ ] Require seed/phase/noise counts before any negative branch.
- [ ] Ensure a source-only carrier cannot authorize actuator implementation.
- [ ] Implement write-once locks, per-arm/per-state resume, config/state/noise SHA
  validation, `--smoke` isolation, and atomic output writes.
- [ ] Record wall time, peak RSS, `MemAvailable`, swap, PID, and command.
- [ ] Add stop rules after every phase; no later task launches when a terminal
  branch is reached.
- [ ] Run all new unit tests and existing 121-test Z/M/M4 regression set.
- [ ] Commit:
  `feat(topic4): orchestrate fail-closed ZM branch decision`.

---

### Task 14 — Figures, archive, and final handoff

**Files**

- Complete `scripts/plot_topic4_zm_branch_decision.py`
- Create `results/topic4_sef_hfo/zm_branch_decision/figures/README.md`
- Update `results/FIGURE_INDEX.md`
- Create
  `docs/archive/topic4/sef_hfo/zm_minimal_carrier_branch_decision_2026-07-XX.md`
- Update `docs/topic4_sef_hfo.md` and `docs/archive/topic4/INDEX.md`

**Minimum diagnostic figure set**

1. state inventory/exact-resume parity;
2. slow-trajectory bins with natural fast-phase snapshots;
3. minimal subsystem × slow bin × posterior carrier-probability matrix;
4. source-space/current-vSEEG carrier plus frequency-drift/global-oscillator diagnostics;
5. coarse/full-field/pathology-axis neighbourhood maps and Branch T/F location, if run;
6. standardized slow-coordinate rank spectrum, if carrier exists;
7. modal/gain panel, if carrier exists;
8. entry/joint-offset probability surfaces, if run;
9. matched driver comparison, if run.

**Steps**

- [ ] Plot only completed phases; missing phases are labelled “not run by stop
  rule,” never blank-pass.
- [ ] Chinese figure README: 2–4 sentences plus `**关注点**：` per figure.
- [ ] Archive sections: what was tested, how, result, allowed/forbidden claim,
  engineering evidence, exact branch verdict, next spec allowed.
- [ ] Run `git diff --check` and markdown-link checks.
- [ ] Verify worktree clean after intended commits and no residual process.
- [ ] Final commit:
  `docs(topic4): close minimal-carrier branch decision`.

## Final execution order and resource ladder

```text
Tasks 1–3     state/config/checkpoint gates
Task 3.5      mandatory seed-1 vertical-slice smoke; no scientific claim
Tasks 4–5     paired-noise + empirical/readout locks
Task 6        seed 1 anchor with one worker
Task 7        seed 1 probabilistic minimal-carrier discovery
              ├─ implementation/no-evidence problem -> stop/fix
              └─ valid result -> seeds 3/4, max two workers after RSS check
Task 8        only when visited states lack carrier
              └─ Branch T/F -> terminal stop
Task 9A       functional rank immediately after carrier verdict
Task 9B       trajectory-conditioned modal/operator audit
Tasks 10–11  entry + existing-slow-coordinate offset geometry
Task 12       only when all valid existing-coordinate offsets are inadequate
Tasks 13–14   adjudicate and archive only completed phases
```

No task is allowed to “rescue” a negative result by changing E→E, adding H/P/A,
or expanding into an unregistered global grid.
