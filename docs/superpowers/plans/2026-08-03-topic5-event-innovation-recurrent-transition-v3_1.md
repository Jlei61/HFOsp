# Topic 5 matched event-innovation recurrent transition v3.1 plan

## Phase 0 — inherit and lock

1. Read the frozen V3.0 continuity, feature, innovation, anchor and dimension artifacts.
2. Implement one shared observation/filter module used by every continuous transition arm.
3. Register parameter names, shapes, preprocessing, optimizer, regularization and budget;
   assert equality between T1 and T2 outside the single `B nu_e` transition term.
4. Implement T0 fixed, T1 observer-only, T2 event-driven and one T3 switching control.

## Phase 1 — synthetic identification

1. Calibrate autonomous hidden-state data, where T1 must be sufficient.
2. Calibrate known event-driven transitions and recover the observable rank/precedence
   impulse.
3. Calibrate discrete switching, where T3 must win or tie.
4. Verify state-matched donor, block, shift, future/past and nuisance null behavior.
5. Write a machine-readable engineering/identifiability acceptance state.

Current implementation evidence is written under
`results/topic5_event_innovation_state_space/v3_1/synthetic_calibration/`. The synthetic
acceptance must remain human-data-free and records equality of shared transition,
observation and filter parameters; only `event_transition_B` may differ between T1 and T2.

Synthetic implementation may run in parallel with V2.7 and V3.0.

## Phase 2 — V3.0 handoff audit

1. Read only the frozen V3.0 train/validation handoff, not human test outcomes used for model
   redesign.
2. Confirm a stable rank/precedence signal in Goal 2 or Goal 3 under the predeclared rule.
3. If absent, close V3.1 human execution as `NOT_TRIGGERED`; do not expand model capacity.
4. If present, lock all model and decision hashes before opening human test.

## Phase 3 — matched train/validation fit

1. Fit T0/T1/T2/T3 on identical dense source-balanced training anchors.
2. Select shared dimension and regularization on non-overlap validation anchors only.
3. Confirm training adequacy and parameter equality outside `B nu_e`.
4. Freeze one configuration per patient before test.

## Phase 4 — one-shot exploratory human test

1. Evaluate T1 versus T2 on non-overlap rank/precedence primary targets.
2. Compare T2 with T3 and fixed T0; report mode and participation separately.
3. Run the frozen state-matched, chronology, shift, future/past and nuisance controls.
4. Compare the T2 observable impulse with the frozen V3.0 local-projection response.
5. Aggregate seeds/folds within patient and perform patient-first cohort inference.
6. Run dense-anchor moving-block sensitivity without treating anchors as independent samples.

## Phase 5 — acceptance

1. Emit `TRANSITION_ACCEPTANCE_STATE.json`, per-patient effects and a Chinese acceptance
   report.
2. Classify the result as observer-only, discrete switching, predictive-unidentified or
   event-associated low-rank update.
3. Do not tune architecture after test and do not use SNN as a rescue or Gate.
4. Reserve shaping language for frozen independent replication combined with the V3.0
   cumulative evidence.
