# Topic 5 ordered-history conditional-information and architecture audit plan

## Milestone A — temporal semantics and target-blind pairing audit

1. Fingerprint the frozen 34-patient rank dataset.
2. Verify that current RNN state resets between group events.
3. Join strict seizure indices to SQL clinical-onset timestamps.
4. Count distinct causal pre-seizure histories and quantify last-event gaps.
5. Freeze the across-event branch as primary, exploratory, or infeasible
   before reading target arrays.

## Milestone B — exact architecture implementation

1. Add linear state-space and vanilla tanh RNN cells to the shared
   contact-query implementation.
2. Expose low-rank leaky RNN ranks 0/1/2/4 through the same trainer.
3. Add a control selector so new models can be trained without rerunning
   accepted static/GRU controls.
4. Add tests for event resets, no-future leakage, recurrent rank, stable
   parameterization, output shapes and finite gradients.

## Milestone C — development smoke and freeze

Run the three established development patients:

- `epilepsiae_1073`
- `epilepsiae_1146`
- `yuquan_chenziyang`

Use the smoke only for correctness, convergence, runtime, OOM and output
schema. Do not choose the architecture based on target results.

## Milestone D — formal target-sealed architecture ladder

Run 34 held-out-patient folds and three seeds for:

- linear state-space;
- vanilla RNN;
- low-rank leaky RNN ranks 0/1/2/4.

Reuse the already frozen static, unordered, last-set, GRU and rank-shuffle GRU
artifacts after fingerprint validation. Run at most three GPU workers, each
with a bounded memory fraction, and retain per-fold logs/checkpoints.

After the primary fixed-hidden-size ladder, run a target-sealed
matched-parameter sensitivity for linear state `h=64` and vanilla RNN `h=48`.
Do not use this sensitivity to reselect the target-facing model.

## Milestone E — information and intervention analysis

1. Collapse seeds within patient.
2. Compare every ordered architecture with static, unordered and last-set.
3. Compare GRU and the best non-GRU recurrence with matched within-event rank
   shuffle.
4. Reuse the accepted H1/H2/H3 matched-window analysis to fix the effective
   history-depth interpretation; do not rerun it.
5. On frozen checkpoints, reverse prefixes, reset after ranks 1/2/3 and remove
   the earliest rank set.
6. Separate predictive benefit from low-dimensionality; report effective
   dimension only alongside shuffled and architecture controls.
7. For the selected nongated model, summarize the local state Jacobian after
   projection through the fitted contact-logit readout. Interpret only in
   rank-step units.

## Milestone F — early-ictal conditional readout

1. Freeze the candidate ordered residual before loading early-ictal arrays.
2. Evaluate increment over static participation and unordered regularized
   fields in the existing 16-patient/106-seizure strict cohort.
3. Use patient-first all-contact/channel-shuffle statistics.
4. If distinct pre-seizure histories are sufficient, run the grouped
   across-event circular-shift sensitivity; otherwise report it as
   metadata-blocked rather than imputing histories.

## Milestone G — closeout

Produce:

- machine-readable acceptance JSON;
- patient-level architecture and intervention tables;
- resource/log audit;
- fixed-hidden-size and matched-parameter architecture audits kept separate;
- paper-ready multi-panel figure with `figures/README.md`;
- Chinese integrated report distinguishing supported, negative, exploratory
  and blocked claims;
- reproducible manifest and clean, scoped commit.
