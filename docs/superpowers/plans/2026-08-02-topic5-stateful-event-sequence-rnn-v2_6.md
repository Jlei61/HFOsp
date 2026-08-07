# Topic 5 stateful event-sequence RNN v2.6 execution plan

**Execution status:** completed and frozen on 2026-08-03. This file now records the executed
plan; it is not the active implementation plan for successor work.

## Phase 1 — continuous sequence data contract

1. Reuse train80-only event ranks, participation, event times and source mapping.
2. Fit templates from training sources only.
3. Build source-continuous tokens and next-20-event targets without materializing overlapping
   fixed histories.
4. Save formal non-overlapping validation/test anchors and raw event indices.

## Phase 2 — recurrent engine

1. Implement state-carrying tanh-RNN, GRU and LSTM.
2. Implement TBPTT with hidden carry and detach, source-boundary reset and configurable update
   accumulation.
3. Save best trained and nested-static checkpoints separately.
4. Implement full-source inference and hidden-state export.

## Phase 3 — calibration and tests

1. Verify future targets, split isolation and train-only normalization.
2. Verify chunked and unchunked forward parity.
3. Verify at least two cells learn the long-memory same-composition synthetic task.
4. Verify state reset damages the task.

## Phase 4 — 34-patient validation screen

1. Run architecture stage for every patient.
2. Run training/TBPTT refinement around each patient's best architecture.
3. Extend validation-only training for leading profiles that hit the 40-epoch boundary.
4. Freeze one patient-specific profile and its training budget using validation only.
5. Record profile distribution and training adequacy before reading test results.

## Phase 5 — untouched test execution

1. Run three seeds for each frozen patient profile.
2. Fold seeds within patient.
3. Report trained RNN, static initialization and fixed EWMA comparator.
4. Export continuous hidden-state trajectories and state-norm diagnostics.

## Phase 6 — chronology and sensitivity

1. Run source-coherent block shuffle and source-level time reversal with rebuilt targets.
2. Run horizon 40 and TBPTT-length sensitivity without changing the primary test result.
3. Review dataset, support grade and selected-cell heterogeneity.

## Phase 7 — scientific acceptance

1. Engineering audit: state carry, checkpoints, leakage, finite training.
2. Optimization audit: cell/profile distribution, boundary choices, seed spread.
3. Scientific audit: test gain, coherent nulls, occupancy/rank/participation separation.
4. Interpretation audit: event-history state versus biological plasticity.

## Phase 8 — final adjudication and handoff

1. Accept V2.6 as an event-history **state-tracking** result: the recurrent state uses recent
   cross-event history and improves over a fixed repertoire.
2. Reject any V2.6 claim of event-driven network updating, evolving contact graph, biological
   plasticity or causal shaping.
3. Treat the matched EWMA result as the minimal supported dynamics — a low-dimensional leaky
   observer — rather than as a reason to abandon evolving-state system identification.
4. Freeze all V2.6 code, checkpoints and nine state files. Do not repair the early-stopping
   bug in place.
5. Route the engineering correction to the V2.7 repair-only plan and the new scientific test
   to the V3.0 event-innovation low-rank state-space plan.

Final state: `ACCEPTED_AS_STATE_TRACKING_PRECURSOR_WITH_KNOWN_TRAINING_BIAS`.
