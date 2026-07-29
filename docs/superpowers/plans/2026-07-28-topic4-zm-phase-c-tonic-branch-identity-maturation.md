# Z/M Phase C — tonic-branch identity and carrier maturation implementation plan

> **Scope:** implement and execute Phase C0–C1 from
> `docs/superpowers/specs/2026-07-28-topic4-zm-phase-c-tonic-branch-identity-maturation-design.md`.
> No E→E change, no \(H/P/A\), no exit actuator, and no lifecycle attempt.

## Goal

Determine, fail closed and across seeds `{1,3,4}`, whether the trajectory-visited
frozen tonic branch is:

- a non-saturated asynchronous/irregular tonic candidate;
- a refractory-limited saturated branch;
- mixed/indeterminate;

and whether a bounded non-tonic carrier window exists in the locked primary
convex slow-field neighbourhood or only in the separately labelled secondary
shell.

Phase-C success means the identity/neighbourhood question is adjudicated under
the locked contract. It does not mean the seizure-lifecycle mechanism succeeds.

---

## Task 1 — Preflight, worktree safety, and immutable Phase-C lock

**Files**

- Create `src/topic4_zm_phase_c_contract.py`
- Create `tests/test_topic4_zm_phase_c_contract.py`
- Create `scripts/lock_topic4_zm_phase_c.py`

**Steps**

- [ ] Verify the branch/worktree and record the clean starting SHA.
- [ ] Inventory peer worktrees and running processes without changing them.
- [ ] Resolve upstream canonical configs and state hashes for seeds 1/3/4.
- [ ] Assert `bounded_mid__rising` and `bounded_mid__peak` states exist for all
  seeds.
- [ ] Lock two phases × three future-noise streams × 8 s + 500 ms burn-in.
- [ ] Lock neuron/current panels by config-SHA hashing.
- [ ] Lock all C0 thresholds, bootstrap settings, C1 coordinates, physical
  bounds, and resource guards from the spec.
- [ ] Serialize a non-production `phasec_input_manifest.json` first; coordinate
  manifests point only to it, and the final production-authorized v1.3
  manifest then locks the input plus native/\(dt/2\) coordinate file and
  semantic SHAs. Test the two-stage chain for hash-cycle absence.
- [ ] Lock independent \(dt/2\) configs/anchors/states for seeds 1/3, while
  reusing native anatomy-panel IDs with the parent native config SHA.
- [ ] Move the stale pre-amendment `4b0f9a76…` final manifest and its
  `61df061a…`/`3dd9cff1…` coordinate locks under `invalidated/`; rebuild the
  complete input → coordinate → final coverage-attestation chain from the
  stable live producers before production.
- [ ] Test that any missing/mutated seed, phase, noise, state, bound, or threshold
  fails closed.
- [ ] Test that old aggregate traces cannot satisfy a raw-observable requirement.

**Gate**

No engine or production action is allowed until Task 1 tests pass and the
manifest is immutable.

---

## Task 2 — Streaming source observables and exact ceiling metrics

**Files**

- Create `src/topic4_zm_tonic_identity.py`
- Create `tests/test_topic4_zm_tonic_identity.py`
- Minimally extend the canonical simulator observer/checkpoint interface if
  required.

**Steps**

- [ ] Implement the exact refractory-ceiling calculation from engine update
  semantics.
- [ ] Add a fixture proving the E ceiling under the locked refractory update.
- [ ] Reduce the guarded engine's one transient full-E raster in the atomic
  child process; never persist it or retain it in the parent coordinator.
  Measure a complete identity child's peak RSS before selecting concurrency.
- [ ] Compute active-core/all-core/all-E `rho80`, active fractions, and
  refractory occupancy.
- [ ] Record sparse ISIs and membrane/current traces only for locked panels.
- [ ] Save \(f_{\rm ref}\) as per-500 ms block ×
  `{core,surround}` ISI numerator/denominator counts. Use the pooled
  active-core ratio for the decisive gate; surround/all-panel ratios are
  supportive and per-neuron fraction medians are forbidden substitutes.
- [ ] Compute local CV2, 5/20/100 ms Fano factors, and threshold-distance
  distributions.
- [ ] Compute 5 ms pairwise correlations and circular-shift nulls on the locked
  panel, with exactly 100 draws saved and fail-closed checked.
- [ ] Treat the activity-independent pair panel as a fixed dependent design
  census: do not bootstrap overlapping pair indices; retain the original
  stratum median-minus-shift-null-q97.5 estimator. Reduce fixed pairs within
  each sampled block first, then blocks; for the null, take q97.5 only after
  the same block aggregation along each of the 100 draw columns.
- [ ] Compute effective E/I/recurrent-E/net-current balance and lag.
- [ ] Compute 1–2 ms E/I rates, PSD, virtual-SEEG metrics, active area, entropy,
  centroid motion, and kymograph.
- [ ] Define active area from the locked 16×16 local E-rate grid
  (`rate>=5 Hz`) over anatomy-occupied bins, at 25 ms for C0 and 2 ms for C1;
  never substitute active-neuron fraction.
- [ ] Synthetic-test asynchronous renewal, periodic refractory ceiling,
  synchronized oscillation, low-active-fraction hotspot, whole-sheet plateau,
  and pulse-train impostors.
- [ ] Mutation-test core/all-sheet denominator swaps and Hz/ms unit errors.
- [ ] Prove observer-disabled byte parity against the accepted engine.

**Gate**

The synthetic saturated process must pass only saturation; the asynchronous
renewal process must pass only AI-supportive metrics. A fixture that passes both
is P0.

---

## Task 3 — Paired local-gain probe

**Files**

- Create `src/topic4_zm_phase_c_gain.py`
- Create `tests/test_topic4_zm_phase_c_gain.py`
- Reuse the existing off-by-default threshold-perturbation hook; do not edit
  the guarded engine.

**Steps**

- [ ] Implement source-core E-threshold probes at ±0.05 and ±0.10 mV.
- [ ] Ensure the existing hook changes only E threshold and is outside
  Z/M/\(S_G\), recurrent E→E, and the external-noise law.
- [ ] Reuse the exact future-noise stream; the hook draws no new RNG values.
- [ ] Compute paired central slopes in carrier and seed-matched pre-entry states.
- [ ] Compute `G_rel` and amplitude-linearity checks.
- [ ] Synthetic-test preserved gain, collapsed gain, sign inconsistency, and
  nonlinear amplitude response.
- [ ] Make zero/undefined pre-entry slope yield `gain_unresolved`.
- [ ] Prove hook-disabled byte parity and exact-resume parity.
- [ ] Fail if a guarded engine file changes; Phase C does not re-bless it.

**Gate**

An unresolved or nonlinear gain is never coerced to zero.

---

## Task 4 — C0 pure analyzer and fail-closed taxonomy

**Files**

- Extend `src/topic4_zm_tonic_identity.py`
- Extend `tests/test_topic4_zm_tonic_identity.py`
- Create `scripts/analyze_topic4_zm_phase_c0.py`

**Steps**

- [ ] Implement the 5,000-draw hierarchical bootstrap.
- [ ] Resample 500 ms blocks, CV2 analysis-panel neurons, null draws, and
  continuations, while holding the overlapping fixed pair panel/strata as a
  census to avoid pair pseudoreplication. Recompute active-core
  \(f_{\rm ref}\) from the sampled count numerator/denominator rather than
  resampling or taking medians of per-neuron fractions.
- [ ] Produce seed-specific CIs without treating neurons/time bins as seeds.
- [ ] Implement exact saturation thresholds:
  `LCB(rho80)>=0.50` plus
  `UCB(G_rel)<=0.20` or `LCB(ref_lock)>=0.80`.
- [ ] Implement exact AI thresholds:
  `UCB(rho80)<=0.20`,
  `LCB(G_rel)>=0.50`,
  `LCB(median CV2)>=0.70`,
  correlation below shift-null 97.5%, absolute median correlation below 0.10,
  active area below 0.50, and no rest/runaway/plateau.
- [ ] Require both fast phases and at least 2/3 noises per phase.
- [ ] Implement every gray-zone, missing-field, nonlinear-gain, metric-conflict,
  and seed-conflict outcome.
- [ ] Require 2/3 seed agreement with no opposite third seed.
- [ ] Synthetic-test every per-seed and aggregate verdict.
- [ ] Mutation-test that majority voting cannot hide an opposite third seed.

**Gate**

The analyzer must emit `no_evidence`/blocked for every incomplete adversarial
fixture. There is no default identity.

---

## Task 5 — Crash-safe C0 runner and smoke

**Files**

- Create `scripts/run_topic4_zm_phase_c.py`
- Create `scripts/run_topic4_zm_phase_c_cell.py`
- Create `scripts/merge_topic4_zm_phase_c_parts.py`
- Create `scripts/topic4_zm_phase_c_resource_monitor.py`
- Create runner/merge tests.

**Steps**

- [ ] Implement one atomic part per seed × state/phase × noise × diagnostic arm.
- [ ] Validate manifest/config/state/noise SHA before every run.
- [ ] Publish immutable content-addressed NPZ files and use JSON as the atomic
  completion marker, so an NPZ orphan cannot block resume.
- [ ] Reject duplicate rows with different hashes.
- [ ] Require `--resume` to reuse exact terminal cells; resume only missing or
  explicitly invalid technical cells.
- [ ] Bind every production part to an immutable adjacent resource receipt and
  require analyzers/final adjudication to revalidate the complete receipt
  index. If a crash leaves a part without its receipt, move the part,
  observables, and partial receipt together to `invalidated/` before rerunning;
  never reconstruct a receipt after the worker has exited unless the same
  coordinator still holds that launch token's complete live audit and the
  worker-published pre-publication self-snapshot.
- [ ] Add `--smoke`, `--manifest`, `--cell`, `--resume`, and explicit
  production-confirmation gates.
- [ ] Run one short seed-1 smoke.
- [ ] Verify units, shapes, panel IDs, current signs, ceiling calculation,
  future-noise reuse, and memory estimate.
- [ ] Keep smoke artifacts in a non-production namespace.

Production order is fixed:

1. Write the input lock, build both coordinate locks, then write the final
   Phase-C lock.
2. Run and analyze native C0; run dt/2 only when the native two-seed gate
   requires it, then finalize the C0 resolution gate.
3. Run C1 base, analyze once to create the base atlas, and write the
   gain-trigger manifest even when it is an explicit closed-empty selection.
4. Run conditional C1 gain and analyze C1 a second time. The first-pass
   `C1_gain_trigger_not_locked` state is routing metadata, never a scientific
   verdict.
5. Run and analyze the locked C1 dt/2 subset only when required, finalize the
   C1 resolution gate, then perform modal and final adjudication.

**Gate**

All Tasks 1–5 tests pass before any production launch.

---

## Task 6 — Execute the complete C0 production matrix

**Coverage**

- seeds `{1,3,4}`;
- phases `{bounded_mid__rising, bounded_mid__peak}`;
- noises `{replay,resample_1,resample_2}`;
- identity continuation plus locked gain arms and matched pre-entry gain control.

**Steps**

- [ ] Start one full SNN worker and measure RSS.
- [ ] Compute `W_max` from the spec, reserving 96 GB and eight logical CPUs.
- [ ] Launch the maximum safe number of independent cells up to `W_max`.
- [ ] Monitor RSS, `MemAvailable`, swap, progress, logs, and artifact growth.
- [ ] Stop new launches below 96 GB `MemAvailable`, on any sampled Phase-C
  worker `VmSwap>0`, or when shared-host swap growth exceeds the separately
  logged 64 MiB jitter tolerance. Lock worker polling to at most 5 s, retain
  per-PID sample counts/maxima plus a self-snapshot immediately before each
  terminal-part publish, and do not kill peer-worktree processes. Report only
  sampled/pre-publish absence of worker swap; never call it an unobserved
  kernel peak, and never replace the pre-publish snapshot with a post-exit
  zero.
- [ ] Resume crashed technical cells with identical hashes.
- [ ] Merge only after the expected-cell matrix is complete.
- [ ] Run the C0 analyzer without changing any threshold.
- [ ] If C0 supports an identity natively, require seeds 1 and 3 to be among
  its native supporters before running their independent homologous \(dt/2\)
  confirmations. A `{1,4}` or `{3,4}` native support pair is
  `resolution_confirmation_unavailable`/insufficient evidence, not a
  scientific negative or resolution-sensitive result.

**Stopping rules**

- Technical block → fix only the technical defect, invalidate affected outputs,
  and rerun exact cells.
- Scientific AI/saturation/mixed/heterogeneous result → retain it and continue
  to C1.
- No threshold or grid change is permitted.

---

## Task 7 — C1 primary and shell coordinate builder

**Files**

- Create `src/topic4_zm_phase_c_neighbourhood.py`
- Create `tests/test_topic4_zm_phase_c_neighbourhood.py`
- Create `scripts/build_topic4_zm_phase_c_neighbourhood.py`

**Steps**

- [ ] Load complete seed-local full-field Z/M/\(S_G\) anchors only.
- [ ] Build six exact early/mid/late × rising/peak primary fields.
- [ ] Build four 50:50 same-phase early–mid/mid–late convex interpolants.
- [ ] Fit seed-specific robust full-field bases using locked anchors only.
- [ ] Align signs to forward trajectory/pathology axes deterministically.
- [ ] Define the seven summaries using true pathology-axis z/m field
  projections, not core-minus-surround differences.
- [ ] Build the fixed ±0.25 robust-SD secondary shell along the locked
  non-tangent/pathology directions.
- [ ] Build native seeds 1/3/4 and independent \(dt/2\) seeds 1/3 separately
  from each resolution's own six full-field anchors; never interpolate native
  fields/checkpoints to \(dt/2\).
- [ ] Store all coordinate/basis floats as float64 and prove round-trip
  semantic slow-state hashes are unchanged.
- [ ] Record per-cell reconstruction error, standardized distance from the
  piecewise anchor manifold, and the sign-alignment rule/fallback.
- [ ] Apply hard/intrinsic and empirical physical bounds without clipping.
- [ ] Record invalid cells and reconstruction distances.
- [ ] Treat exact observed anchors as non-reconstructed empirical cells:
  require exact source-state/hash identity, zero anchor distance, finite values,
  and intrinsic hard-domain validity, but do not apply the circular empirical
  quantile envelope fitted from those same anchors.
- [ ] Keep the empirical quantile envelope unchanged for convex midpoints and
  the fixed shell; never waive genuine shell \(m<0\) or \(z\) hard-bound
  violations.
- [ ] Before any C1 SNN launch, fail closed unless native primary coverage is
  30/30, independent \(dt/2\) primary coverage is 20/20, and at least one
  homologous adjacent primary pair is available in both \(dt/2\) seeds.
- [ ] Recompute and lock shell coverage. The pre-production audit found only
  4/24 valid native and 2/16 valid \(dt/2\) shell cells; if the final relock
  confirms incomplete coverage, preclude a complete shell-negative verdict
  while retaining the valid cells as extrapolative sensitivity.
- [ ] Synthetic-test convexity, PCA sign determinism, intrinsic bounds,
  empirical envelopes, no-clipping, and seed-local basis separation.
- [ ] Mutation-test accidental cross-seed PCA fitting and independent scalar
  field scaling.
- [ ] Write the full coordinate manifest before any C1 SNN result exists.
- [ ] Use one canonical write-once C1 \(dt/2\) selection payload through the
  dedicated lock, coordinator enumeration, cell validation, and analyzer; add
  a true end-to-end integration test across all four stages.

**Gate**

Primary and secondary coordinate lists cannot change after the first C1
production result.

---

## Task 8 — C1 phenotype analyzer

**Files**

- Create `src/topic4_zm_phase_c_maturation.py`
- Create `tests/test_topic4_zm_phase_c_maturation.py`
- Create `scripts/analyze_topic4_zm_phase_c1.py`

**Steps**

- [ ] Implement the common bounded carrier gate.
- [ ] Implement AI-tonic, periodic non-tonic, clonic/bursting, and spatial-relay
  labels.
- [ ] Implement the locked cycle/burst counts, modulation, occupancy,
  rest-dwell, interval-CV, and first-passage-null thresholds.
- [ ] Require the joint refractory-saturation thresholds
  `rho80>=0.50` and refractory-ISI fraction `>=0.80`.
- [ ] Require each separated spatial zone to have occupancy `>=0.80`, and use
  each seed's canonical `params.Rr` as the readout-kernel separation scale.
- [ ] Require cross-fast-phase relative period agreement `<=0.20` for a
  periodic carrier.
- [ ] Require at least 5/6 run support, posterior median >0.8, and both fast
  phases for a positive cell.
- [ ] Require two adjacent primary cells within seed for a window.
- [ ] Require the same aligned direction in at least 2/3 seeds with no opposite
  third-seed outcome.
- [ ] Require both native seeds 1 and 3 to support a positive primary or shell
  window before \(dt/2\) confirmation. Other native 2/3 support combinations
  are `resolution_confirmation_unavailable`, not scientific negatives and not
  resolution-sensitive outcomes.
- [ ] Adjudicate the secondary shell separately: require the same non-tonic
  phenotype at the same locked basis-direction/sign cell in at least 2/3
  seeds; never apply primary `path_index+1` adjacency to shell points.
- [ ] Keep primary-convex and secondary-shell verdicts separate.
- [ ] Emit independent `primary_gate` and `shell_gate` resolution closures
  (`confirmed|contradicted|indeterminate|blocked|not_required`); each final
  input consumes only its own layer gate, while the top-level C1 verdict is
  summary-only.
- [ ] Implement the complete 3-seed/10-primary-cell negative-coverage contract.
- [ ] Make invalid/missing shell cells block only the shell-negative claim.
- [ ] Implement representation-sensitive and seed-heterogeneous outcomes.
- [ ] Synthetic-test every phenotype, isolated candidate, contiguous window,
  shell-only candidate, bounded negative, and incomplete matrix.
- [ ] Mutation-test that shell positives cannot become primary reachable
  positives.
- [ ] Regression-test that primary confirmation cannot silently complete an
  indeterminate shell, and that mixed C0 plus primary-negative preserves a
  closed shell positive/isolated/heterogeneous sensitivity result.

**Gate**

No incomplete or representation-conflicted atlas can produce a positive
reachable-window or complete negative verdict.

---

## Task 9 — Execute C1 primary production

**Coverage per seed**

- ten locked primary slow-field cells;
- two canonical fast initial phases;
- three future-noise continuations;
- 8 s post-burn continuations.

**Steps**

- [ ] Recompute the safe worker count from fresh RSS/resource state.
- [ ] Run crash-safe independent cells at the maximum safe concurrency.
- [ ] Monitor continuously and preserve at least 96 GB `MemAvailable`.
- [ ] Merge only an exact expected-cell matrix.
- [ ] Run C1 analysis with the immutable C0/C1 thresholds.
- [ ] Do not stop or expand based on whether a positive appears early.

---

## Task 10 — Execute the preregistered secondary shell

**Steps**

- [ ] Run every physically valid locked shell cell regardless of the primary
  scientific outcome.
- [ ] Preserve invalid cells as explicit coverage entries.
- [ ] Use the same two fast phases, three future noises, duration, metrics, and
  resource rules.
- [ ] Adjudicate shell-only positives separately.
- [ ] Never expand amplitudes beyond 0.25 robust SD.

**Stopping rule**

The shell remains sensitivity/extrapolation evidence. It cannot override a
primary-convex negative or authorize slow-path reachability. Any invalid locked
shell cell makes the shell layer coverage-limited and blocks
`no_maturation_in_tested_secondary_shell`; it does not block a complete
primary-convex conclusion.

---

## Task 10A — Conditional C1 AI gain closure

**Files**

- Create `scripts/lock_topic4_zm_phasec1_gain_triggers.py`
- Extend the C1 runner/analyzer and their tests.

**Steps**

- [ ] Finish the complete primary and shell base atlas before evaluating a
  gain trigger.
- [ ] Apply the locked spike-only AI screen to every cell.
- [ ] Require at least 5/6 screen passes and at least 2/3 in each fast phase.
- [ ] Write one immutable trigger manifest containing the base-atlas SHA,
  Phase-C manifest SHA, slow-field SHA, triggering part SHAs, every expected
  carrier gain arm, and reused C0 pre-entry denominator SHAs.
- [ ] For every triggered cell, run
  `0, ±0.05, ±0.10 mV` across both phases and all three future noises.
- [ ] Use the C0 gain linearity, plateau/runaway, \(G_{\rm rel}\), bootstrap,
  and provenance rules unchanged.
- [ ] Label valid AI as `balanced_AI_tonic_cell`; keep it separate from
  periodic/clonic maturation.
- [ ] Label nonlinear/sign-inconsistent gain as
  `tonic_gain_indeterminate`; missing/hash/truncation as
  `C1_blocked_conditional_gain`.
- [ ] Never trigger a new cell after the trigger manifest is written.

**Gate**

Conditional gain is not required to certify an already complete non-tonic
maturation window.  Otherwise, any unresolved trigger blocks a complete
negative or complete C1 identity atlas; it never defaults to `tonic_non_AI`.

---

## Task 11 — Seed-specific modal/operator audit

**Files**

- Extend `src/topic4_zm_modal_operator.py` or create a Phase-C wrapper.
- Extend modal tests.
- Create `scripts/analyze_topic4_zm_phase_c_modal.py`.

**Steps**

- [ ] Route each seed independently from its C0/C1 phenotype.
- [ ] AI/stochastic → DMD/Koopman/finite-time singular gain.
- [ ] Periodic/clonic → phase-conditioned/Poincaré/Floquet.
- [ ] Saturated → local gain/refractory sensitivity only.
- [ ] Indeterminate → descriptive response only.
- [ ] Fit and test operators on held-out perturbation shapes/time windows.
- [ ] Compare spatial subspace angles and pathology-axis alignment across
  seeds without pooling eigenvalues.
- [ ] Keep `class_disagreement` visible at aggregate level.

**Gate**

Modal evidence cannot change the C0/C1 phenotype verdict.

---

## Task 12 — Pure Phase-C adjudicator

**Files**

- Create `src/topic4_zm_phase_c_verdict.py`
- Create `tests/test_topic4_zm_phase_c_verdict.py`
- Create `scripts/adjudicate_topic4_zm_phase_c.py`

**Steps**

- [ ] Implement exactly the spec verdict vocabulary.
- [ ] Require all C0 and C1 input SHA/provenance fields.
- [ ] Keep primary and secondary neighbourhood layers separate.
- [ ] Carry source identity, seed-specific modal, observation block, and
  lifecycle boundaries as explicit fields.
- [ ] Hard-code:
  `entry=not_tested`,
  `offset=not_tested`,
  `recovery_lifecycle=not_established`,
  `phase_c2_authorized=false`,
  `actuator_authorized=false`.
- [ ] Add one fixture per verdict and adversarial missing/conflicting fixtures.
- [ ] Ensure no default fall-through to AI, saturation, maturation, negative,
  or a next mechanism branch.

---

## Task 13 — Figures and visual scientific QA

**Files**

- Create `scripts/plot_topic4_zm_phase_c.py`
- Create
  `results/topic4_sef_hfo/zm_phase_c_tonic_identity/figures/README.md`
- Update `results/FIGURE_INDEX.md` after figures exist.

**Minimum figures**

1. core/all-sheet ceiling occupancy and paired gain;
2. CV2, refractory locking, Fano, and pairwise shift-null;
3. E/I currents, membrane-distance, fine-rate PSD, and current-vSEEG;
4. primary-convex phenotype atlas with exact coverage;
5. separately styled secondary-shell atlas;
6. representative spatiotemporal traces and kymographs;
7. seed-specific modal panels;
8. Phase-C status and claim boundary.

**Steps**

- [ ] Plot denominators, seed points, intervals, and missing/invalid cells.
- [ ] Never show pooled neurons as independent seed replication.
- [ ] Label shell panels `extrapolated sensitivity`.
- [ ] Label every figure `source-space identity/maturation; not lifecycle`.
- [ ] Inspect figures at full resolution.
- [ ] Write the Chinese README only after plots exist, using 2–4 sentences and
  one `**关注点**：` line per figure.

---

## Task 14 — Acceptance, archive, tests, and handoff

**Files**

- Create
  `docs/archive/topic4/sef_hfo/zm_phase_c_tonic_identity_maturation_2026-07-XX.md`
- Update `docs/topic4_sef_hfo.md`
- Update `docs/archive/topic4/INDEX.md`

**Steps**

- [ ] Run all Phase-C unit/synthetic/mutation tests.
- [ ] Run the complete upstream Z/M/M4/checkpoint regression suite.
- [ ] Rerun the pure adjudicator from immutable machine artifacts.
- [ ] Verify production coverage, input SHAs, resource logs, and exact expected
  cells.
- [ ] Verify no residual Phase-C process, no observed `VmSwap` at the locked
  per-worker samples/immediate pre-publish self-snapshots, and bounded/logged
  shared-host jitter; do not relabel sampled absence as a kernel peak
  measurement or use a post-exit zero as evidence.
- [ ] Run `git diff --check` and markdown-link checks.
- [ ] Archive: question, methods, C0 identity, C1 primary, C1 shell,
  seed-specific modal results, allowed/forbidden claims, engineering evidence,
  exact verdict, and next-spec boundary.
- [ ] Report primary and shell layers separately. If shell validity remains
  incomplete, state `secondary_shell_incomplete`/coverage-limited explicitly
  and never write a shell-negative conclusion.
- [ ] Report `resolution_confirmation_unavailable` as insufficient evidence
  whenever a native positive lacks joint seed-1/seed-3 support; do not convert
  it to a scientific negative or resolution sensitivity.
- [ ] Read reporting priority from the layer-local resolution gates. Never let
  a mixed C0 summary hide a closed secondary-shell structure, and never let
  that shell structure authorize primary reachability.
- [ ] State explicitly that C0/C1 acceptance does not establish lifecycle.
- [ ] Commit in logical code/test, production-verdict, and docs/figure batches.

**Terminal acceptance**

This plan is complete when the locked C0/C1 matrices are either:

- fully adjudicated into one authorized Phase-C verdict; or
- fail-closed with a specific technical/coverage blocker that cannot be solved
  without changing the preregistration.

Scientific saturation, AI, mixed identity, maturation, or bounded negative are
all valid completions. None authorizes Phase C2, \(H/P/A\), an E→E change, or a
lifecycle claim.

---

## Execution order

```text
Tasks 1–4   contract, observables, gain, pure taxonomy tests
Task 5      crash-safe runner + non-claim smoke
Task 6      complete C0 production + conditional dt/2 confirmation
Tasks 7–8   lock C1 coordinates + pure phenotype tests
Task 9      complete primary-convex production
Task 10     complete fixed secondary shell
Task 10A    lock and execute conditional C1 AI gain, if triggered
Task 11     seed-specific modal audit
Task 12     fail-closed adjudication
Tasks 13–14 figures, archive, regression, process cleanup
```

No later task may rescue a scientific result by modifying the locked substrate,
thresholds, grid, seeds, or mechanism.
