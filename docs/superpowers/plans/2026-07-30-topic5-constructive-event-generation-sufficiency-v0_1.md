# Topic 5 constructive event-generation sufficiency v0.1 execution plan

## Milestone A — Contract and artifact audit

1. Verify 34 dataset records, 102 selected linear checkpoints and frozen
   train/heldout fingerprints.
2. Inventory heldout source groups, geometry completeness, train-axis support
   and train-only KMeans read-back eligibility.
3. Freeze the paired conditions, metrics and Gate A/B/C in the companion spec.

## Milestone B — Source-conditioned generator

1. Implement deterministic inverse-CDF categorical sampling from shared
   uniforms.
2. Implement source initialization and no-repeat candidate masks.
3. Residualize frozen linear-state contact logits against their no-prefix
   field.
4. Estimate train80 static scaffold and progress hazard.
5. Implement `full`, `static_only`, `static_shuffle`, H1, H2,
   `constant_stop` and `no_termination`.
6. Add toy tests for source retention, paired uniforms, termination and
   component isolation.

## Milestone C — Posterior predictive read-back

1. Compute heldout human and generated event grammar, suffix participation,
   rank, precedence and transition metrics.
2. Fit train80-only masked \(k=2\) read-back and classify heldout/generated
   events without generated-data refitting.
3. Derive the train-only displacement PCA axis and compute signed/unsigned
   heldout/generated endpoints.
4. Build heldout split-half empirical error floors.

## Milestone D — Development smoke

Run one seed for:

- `epilepsiae_1073`;
- `epilepsiae_1146`;
- `yuquan_chenziyang`.

Only engineering, leakage, runtime and schema may be changed after this
stage. No scientific threshold or condition may be added.

## Milestone E — Formal human run

1. Run 34 patients × 3 seeds.
2. Use bounded GPU batches and per-process memory fraction; persist one
   patient/seed atomically.
3. Record logs, runtime, peak RSS/GPU memory and completion state.
4. Monitor expected 102/102 cells and retry only engineering failures.

## Milestone F — Gate analysis

1. Collapse seeds within patient.
2. Evaluate Gate A, then B, then C without pooling failed/locked endpoints.
3. Report Epilepsiae and Yuquan separately as sensitivity.
4. Stop before SNN unless both human scientific gates pass.

## Milestone G — Delivery

1. Machine-readable acceptance JSON.
2. Patient-level CSV and component-ablation statistics.
3. Paper-ready main figure plus diagnostic figure and Chinese README.
4. Chinese archive report with allowed/forbidden manuscript wording.
5. Tests, output fingerprints and figure visual QA.
