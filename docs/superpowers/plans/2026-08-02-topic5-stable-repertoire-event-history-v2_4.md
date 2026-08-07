# Topic 5 stable-repertoire event-history v2.4 — implementation plan

## Phase 0 — closure and immutable inputs

1. Preserve all v2.3.1 code/results as historical development evidence.
2. Freeze the six development patients and explicit 28-patient extension list.
3. Record hashes for `dataset_v0_4`, source mappings, spec and configs.

## Phase 1 — test-first P0 repair

1. Add a v2.4 dataset object carrying raw event indices, sequence positions, times and null
   donor provenance.
2. Implement source-level 20-event block permutation and rebuild all overlapping windows.
3. Implement safe circular target pairing with synchronized values/indices/positions/times,
   no history overlap and at least one-horizon separation.
4. Add adversarial tests that fail the v2.3 row-wise shuffle and stale-metadata circular code.

## Phase 2 — matched forecasting ladder

1. Add first/last/random equal-count descriptors.
2. Add full-token EWMA, event-descriptor EWMA and four-bin distributed-lag ridge.
3. Select all decay/alpha/dimension choices only on validation primary propagation score.
4. Fit low-dimensional leaky state and compare it with the validation-selected B4/B5 model.
5. Add time/IEI nuisance features without using them as a primary biological clock.

## Phase 3 — score, reliability and audit

1. Estimate score scaling on train targets only.
2. Report propagation, participation and full repertoire scores separately.
3. Report raw and train-mean-residualized split-half reliability.
4. Save history/target durations, IEI, event rate, source progress and independent-window
   counts for every patient/split.

## Phase 4 — six-patient repair rerun

1. Run H=20 from scratch for all B0–B6, R1 and both coherent null families.
2. Run H=40 as a complete sensitivity, without changing the primary configuration.
3. Audit all event indices and null donor maps directly from artifacts.
4. Apply the frozen development release gates and write a fail-closed release JSON.

## Phase 5 — locked extension

If and only if Phase 4 releases the extension:

1. hash-lock code/config/spec;
2. run the remaining 28 patients once;
3. do not change model, grids, endpoint, horizon or inclusion based on extension results;
4. keep ineligible/insufficient-window patients in the denominator audit and fail closed per
   patient rather than silently dropping them.

## Phase 6 — acceptance and handoff

1. Produce patient-first primary/sensitivity tables and dataset-stratified effects.
2. Run Wilcoxon, sign test and patient bootstrap on the locked 28 only.
3. Produce a separate 34-patient descriptive table with the six development patients marked.
4. Write the allowed claim according to the frozen stopping rules.
5. Run scoped tests, artifact reload, hash verification and `git diff --check` before handoff.

