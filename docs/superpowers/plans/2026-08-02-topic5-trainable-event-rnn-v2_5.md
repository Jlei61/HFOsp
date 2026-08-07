# Topic 5 trainable event RNN v2.5 — implementation and execution plan

## Phase 0 — close the old claim boundary

1. Amend the v2.4 acceptance report so its negative claim applies only to the fixed
   PCA + scalar-decay leaky filter.
2. Preserve the frozen v2.4 code/config hashes and artifacts.

## Phase 1 — 34-patient denominator and split audit

1. Count eligible events, source records and `L=20/40/80, H=20` formal windows.
2. Implement window-balanced chronological source splits.
3. Save per-patient split strategy, source IDs, raw event indices, window count and support
   grade. Attempt all 34 patients.

## Phase 2 — trainable model implementation

1. Implement a shared residual decoder for `RNN`, `GRU` and `LSTM` cells.
2. Add event/block recurrent steps, normalization modes, LayerNorm, optimizer, learning-rate,
   batch-size, weight-decay and gradient-clip controls.
3. Use train-only target scales and primary propagation loss for checkpoint selection.
4. Save reproducible checkpoints and full training diagnostics.

## Phase 3 — tests and synthetic calibration

1. Test split chronology, train80-only indices and formal target non-overlap.
2. Test normalization is train-only and block aggregation cannot read future events.
3. Test all cells/optimizers forward, train and reload correctly.
4. Run a same-composition order task; recurrent state must beat unordered prediction.

## Phase 4 — six-patient global screen

1. Screen frozen training profiles with a fixed GRU architecture.
2. Freeze the best optimizer/LR/batch/normalization profile by median validation propagation.
3. Screen frozen architecture profiles across cell, hidden size, layers, LayerNorm and step.
4. Freeze one primary profile and one compact sensitivity profile; freeze epoch budget from
   the median best epoch.

## Phase 5 — complete 34-patient run

1. Fit the frozen matched baseline and recurrent profile for all 34 patients and three seeds.
2. Run source-coherent block and safe circular controls with the same profile.
3. Run `L=40/80` sensitivity where windows exist; never substitute it for `L=20` primary.
4. Save per-patient predictions, checkpoints, diagnostics, contract audit and denominator.

## Phase 6 — multiround review

1. Engineering review: leakage, provenance, finite training, checkpoint reload parity.
2. Optimization review: seed dispersion, learning curves, boundary-selected hyperparameters.
3. Scientific review: matched baseline, coherent nulls, propagation vs participation.
4. Denominator review: all 34 attempted, low-support influence and six/28 boundary.
5. Interpretation review: recurrent prediction versus plasticity/evolving graph claims.

The run stops only after all five reviews have written artifacts, including a bounded-negative
report if the trainable recurrent family is not selected.
