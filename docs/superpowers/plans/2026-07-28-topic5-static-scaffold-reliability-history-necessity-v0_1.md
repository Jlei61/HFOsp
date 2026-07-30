# Execution plan: static-scaffold reliability and history necessity v0.1

## A. Freeze and audit

1. Snapshot the v0.4 dataset fingerprints, selected hyperparameters, formal
   result root, seeds, and target-seal status.
2. Confirm 34 valid patients and non-overlapping chronological train80 /
   heldout20 indices.
3. Write a standalone config and output root.

## B. Static reliability

1. Implement event-first participation fields and reliability metrics.
2. Add within-shaft circular nulls and event-count saturation.
3. Run all 34 patients without loading any seizure artifact.
4. Save per-patient, saturation, null, and patient-first summary tables.

## C. Finite-history implementation

1. Add `WindowedHistorySequenceGRU` with causal prefix masks and fixed history
   windows.
2. Add equivalence, invariance, mask, shape, and gradient tests.
3. Build a dedicated one-fold trainer for history 1/2/3 only, reusing the
   accepted v0.4 training/evaluation functions.

## D. Smoke and resource validation

1. Run three heterogeneous patients at one seed with reduced coverage.
2. Check finite losses, deterministic rerun, fingerprints, target seal,
   per-fold peak memory, and resume behaviour.
3. Inspect GPU/RAM before formal launch.

## E. Formal execution

1. Run 34 LOSO folds × 3 seeds × 3 finite-history conditions.
2. Use six GPU worker processes, at most 48 CPU threads in aggregate.
3. Skip completed folds on restart and log every process under the result root.
4. Keep a 30-second GPU/RAM/progress watcher in tmux.
5. After the primary finite-window run, run the matched history-3
   rank-shuffle sensitivity under the same 34 × 3-seed contract.

## F. Aggregation and figures

1. Merge new finite-history metrics with the frozen full-history and control
   metrics from `formal_multiseed_20260725_v1`.
2. Collapse seeds within patient before all group statistics.
3. Produce:
   - scaffold split reliability and structured-null comparison;
   - event-count saturation curves;
   - heldout NLL by history depth;
   - paired incremental-history contrasts and per-patient heterogeneity.
4. Write `figures/README.md` after the figures exist.

## G. Acceptance and scientific closeout

1. Audit 102/102 new folds, coverage, fingerprints, target seal, NaN, OOM,
   watcher log, and exact score denominator.
2. State the static-scaffold and history-necessity conclusions independently.
3. Decide whether the RNN contributes a defensible recurrent-history result,
   while keeping early-ictal target work outside this experiment.
