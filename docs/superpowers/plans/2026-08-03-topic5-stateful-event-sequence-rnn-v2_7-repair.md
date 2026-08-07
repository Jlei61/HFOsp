# Topic 5 stateful event-sequence RNN v2.7 repair execution plan

**Execution status:** complete on 34 patients; derived acceptance frozen.

## Phase 1 — isolated repair

1. Copy the v2.6 core/runner/config namespace to v2.7 and change only trained-checkpoint
   patience bookkeeping.
2. Add the three early-stopping regression tests from the spec.
3. Point every artifact to the parallel `v2_7/` root and save v2.6 parent hashes.
4. Run the full existing unit-test set before any cohort fit.

## Phase 2 — validation-only rerun

1. Recreate the same 748 architecture/refinement fits on all 34 patients.
2. Repeat the same epoch-boundary rule and freeze one profile plus budget per patient.
3. Audit minimum-budget runs, selected-cell distribution, best epochs and finite training.
4. Do not inspect test scores until the 34-profile freeze state is complete.

## Phase 3 — formal test and controls

1. Run the same three seeds per frozen patient profile.
2. Recompute formal and dense RNN/static/EWMA comparisons.
3. Rerun state-reset and memory-curve analyses on the repaired checkpoints.
4. Rerun block-order and reversal nulls, because profile selection and checkpoints may change.
5. Rerun H40 with the repaired frozen H20 profile and budget.

## Phase 4 — acceptance

1. Produce a read-only derived acceptance state with support strata and V2.6 paired deltas.
2. Verify code/config/runner hashes, artifact counts, finite values and forbidden-input flags.
3. Replace manuscript-facing v2.6 effect sizes only after the V2.7 acceptance is complete.
4. Close V2.7 as the final observer/state-tracking implementation; do not tune it further.

## Parallel scientific preparation

V2.7 and V3.0 answer different scientific questions. V3.0 synthetic, train/validation and
implementation work may run while V2.7 trains. V3.0 human test execution depends on its own
frozen state/innovation, cumulative-response, matched-transition, null and inference
contracts; it does not depend on a favorable or completed V2.7 scientific result.
The V3.1 shared-filter transition implementation may also be calibrated on synthetic data;
its human test follows the frozen V3.0 train/validation handoff, not the V2.7 outcome.
