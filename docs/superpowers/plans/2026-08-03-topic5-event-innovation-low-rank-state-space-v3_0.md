# Topic 5 event-innovation impulse response and cumulative displacement v3.0 plan

This plan implements the matching v3.0 association specification in three scientific Goals.
It does not reopen event-internal next-rank prediction and does not use architecture search
as a substitute for identification. Phases 0–4 use only synthetic data and the human
train/validation partitions; no human test outcome is read before Phase 5.

## Phase 0 — provenance, continuity and frozen schemas

1. Reuse the canonical train80 event inventory; do not regenerate events or read old
   heldout20, ictal, SNN, geometry, SOZ or axis labels.
2. Build the continuity-unit manifest from absolute source times, session metadata, gaps and
   montage/channel compatibility. Join only verified artificial file splits; reset true
   independent sessions.
3. Save event, contact, source and continuity-unit hashes plus the real-time spans of every
   planned pre/future window.
4. Freeze the rank/precedence primary schema, mode secondary schema and participation
   tertiary schema.
5. Freeze dense-train source-balanced weights, non-overlap validation/test anchors and dense
   sensitivity anchors before fitting outcomes.

**Deliverables:** `source_continuity_manifest.csv`, `event_inventory.json`,
`anchor_contract.json`, `feature_schema.json`.

## Phase 1 — Goal 1: measurable state and valid innovation

1. Fit the train-only stable rank backbone and estimate contact/tie reliability.
2. Fit low-rank rank-field bases with validation candidates `K=1..4`; reserve `K=6`, `K=8`
   and maximum/full rank for mandatory diagnostic comparison.
3. Quantify raw and contact-residualized rank reliability, pairwise precedence reliability,
   and secondary mode/participation reliability.
4. Implement the frozen past-only observer ladder: pre20, pre40, pre80, four 20-event lag
   bins, then time/event-rate nuisances.
5. Construct training innovations only through source-level or blocked chronological
   cross-fitting. Keep rank/precedence, mode and participation innovations separate.
6. Test innovation whiteness/calibration against blocked nulls. If past-only predictors still
   predict a residual, expand the observer only along the frozen ladder and mark unresolved
   families explicitly.
7. At every rung, test the residual against the full available past-only ladder in fixed
   contact-rank coordinates. Use continuity-sequence coherent blocks and require at least two
   permutable blocks per contributing sequence; never use a cross-fit fold as the null group.

**Stop/interpretation rule:** invalid rank/precedence innovation blocks an innovation claim,
but does not erase the stable backbone or observer result.

**Deliverables:** `state_reliability.json`, `innovation_validity.json`,
`crossfit_provenance.parquet`, `dimension_diagnostic.csv`.

## Phase 2 — synthetic and negative-control calibration

1. Simulate a stable backbone with observation noise and autonomous drift but no event-driven
   update; event terms must not receive credit.
2. Simulate known low-rank event-driven impulses and recover the sign/subspace of the
   observable rank/precedence response.
3. Simulate repeated aligned, cancelling and decaying innovations; recover dose,
   cancellation and persistence.
4. Verify masks, ties, continuity resets/carries, non-overlap anchors, safe shifts, donor
   derangements, source-balanced weights and cross-fitting.

**Deliverables:** `synthetic_identifiability_state.json`, `synthetic_impulse_recovery.csv`,
`synthetic_accumulation_recovery.csv`.

## Phase 3 — Goal 2: multi-horizon local impulse response

1. Estimate strictly pre-event states and disjoint post-event states at
   `h=5,10,20,40`; keep `h=20` primary and report the full frozen response profile.
2. Fit local projections of autonomous future-state residual on the primary
   rank/precedence innovation with frozen nuisance covariates.
3. Convert every response to observable contact-rank and pairwise-precedence changes; do not
   rank or interpret raw latent weights.
4. Run duration-matched future-versus-past tests.
5. Run state/source-progress-matched innovation donors, source-coherent block sizes
   `1/2/5/10/20/40`, and safe lag shifts `2h/3h/4h` without wrap-around. Save state distance,
   source-progress difference, IEI/event-rate difference, donor reuse and eligible-anchor
   fraction for every matched-null draw.
6. Save per-patient continuous effects and uncertainty; do not classify patients with a
   conjunctive multi-Gate rule.

**Deliverables:** `local_projection_state.json`, `observable_impulse_response.parquet`,
`local_response_nulls.json`, `patient_local_effects.csv`.

## Phase 4 — Goal 3: repeated-innovation accumulation

1. Build causal cumulative innovations for `m=5,10,20,40` using event-count weights.
2. Use anchors separated by at least `max(m,h)` and compare the post-accumulation future state
   with the autonomous forecast from a disjoint state ending before the accumulation window;
   no innovation-window event may appear in either endpoint state estimate.
3. Test magnitude/dose response, directional alignment, cancellation and persistence across
   future horizons.
4. Run order-shuffled, direction/sign-matched and state-matched cumulative nulls.
   For the equal-weight primary sum, replace the algebraically invariant within-window order
   shuffle with matched complete-exposure reassignment. Reserve within-window order for the
   IEI-decay sensitivity where weights are unequal.
5. Add validation-selected IEI-decay weighting only as a frozen sensitivity and report the
   implied real-time spans without claiming a biological recovery constant.
6. Before any V3.0 human test is read, emit a train/validation-only `V3_1_HANDOFF_STATE.json`
   applying the exact cohort-median Goal 2/Goal 3 rules in spec Section 9 and recording all
   eligible-patient support and dataset-specific directions.

**Deliverables:** `cumulative_response_state.json`, `dose_response.csv`,
`alignment_cancellation.csv`, `iei_decay_sensitivity.json`, `V3_1_HANDOFF_STATE.json`.

## Phase 5 — frozen 34-patient exploratory execution

1. Record the Sections 14 and 16 release checklist, all indices and hashes before reading human test
   outcomes. V2.7 completion is not a release condition.
2. Reconstruct the train-only basis, refit the frozen observer on train plus validation, and
   refit response coefficients on cross-fitted train plus future-blind validation rows using
   the already selected ridge; no test-dependent selection is allowed.
3. Run all 34 patients as one exploratory cohort; do not define a post hoc 6/28 confirmation
   split.
4. Use dense, source-balanced training; non-overlap validation/test for primary inference;
   dense test with continuity-unit moving-block bootstrap as sensitivity.
5. Combine folds within patient and perform patient-first cohort inference.
6. Report the rank-plus-precedence propagation endpoint as the V3.0 primary. Do not import
   v2.7 mode/participation descriptors into V3.0 because this rank-field model has no such
   heads.
7. Assign the cohort evidence level from 0 to 2 without changing models, thresholds or
   subgroups after test inspection.

**Deliverables:** `HUMAN_EXPLORATORY_STATE.json`, `patient_summary.csv`,
`cohort_inference.json`, `evidence_level.json`.

## Phase 6 — acceptance

1. Archive a Chinese acceptance report with allowed/forbidden wording for the achieved
   evidence level.
2. Report negative and mixed outcomes without expanding the architecture family.
3. Keep the V3.1 handoff fixed at its pre-test train/validation decision; V3.0 human outcomes
   cannot reopen or redesign V3.1.
4. Freeze the complete pipeline before any independent cohort is opened.
5. Keep SNN independent; compare observable propagation principles only after both lines are
   frozen.

## Execution order and parallelism

- V2.7 repair and V3.0 Phases 0–4 may run in parallel because they answer different
  questions.
- Phase 5 requires the v3.0 design checklist and synthetic/engineering validity, not a
  favorable V2.7 scientific outcome.
- V3.1 code and synthetic calibration may run in parallel, but V3.1 human model selection is
  not part of V3.0 and cannot be used to rescue a null V3.0 test.
- Do not branch into GRU/Transformer sweeps, contact-graph recovery, multiple autonomous
  fields or SNN-conditioned models.
- Any failed identification experiment lowers the evidence level; it does not trigger
  unplanned capacity expansion.
