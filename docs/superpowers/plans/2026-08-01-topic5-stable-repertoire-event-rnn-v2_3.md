# Topic 5 stable-repertoire event RNN v2.3 — implementation plan

## Phase A — freeze and test the data contract

1. Load the audited six-patient `dataset_v0_4` event arrays and v2.2 source mapping.
2. Restrict to the existing `train80`; split whole source recordings 60/20/20 in chronological order.
3. Fit fixed `K=2` masked templates on train events only and serialize their hashes, occupancy and mode descriptors.
4. Build `L=80` history / `H=20` future-window samples with source-boundary resets and non-overlapping formal targets.
5. Unit-test future blindness, source disjointness, source resets, template train-only provenance and old-heldout exclusion.

## Phase B — build in order

1. R0 static target mean.
2. R1 recent-window ridge with validation-selected ridge penalty.
3. R2 first-order discrete switching with analytic `H`-step occupancy and train-mode descriptors.
4. R3 leaky linear event-state; select hidden size and regularization on validation only.
5. Run the six-patient R0–R3 pilot and audit convergence before adding R4.
6. R4 GRU with exactly the same token, target, history, decoder and loss.

R4 training adequacy is fail-closed: a candidate whose validation optimum remains at
the final allowed epoch is not scientifically scored. Optimizer-only repair is allowed
before the six-patient R4 run, but may not change the split, target, history, model grid
or test endpoint. The development repair changed AdamW learning rate `0.001 → 0.01`
after the first patient's curves had not plateaued; with the repaired setting all three
ordered seeds peaked at epochs 31–43 under a 100-epoch cap.

For the predeclared `H=40` sensitivity, the 100-epoch cap was insufficient in two
patients. The cap was increased to 200 using validation-curve adequacy only; a probe in
the first affected patient reached its validation optimum at epoch 114. No input,
architecture, test target or scientific threshold changed.

## Phase C — chronology and review

1. Re-fit R3/R4 on within-history shuffled samples.
2. Re-fit R3/R4 after a within-source circular input-target shift.
3. Aggregate seeds within patient, then compare patients.
4. Run `H=40` as a declared sensitivity only after the primary artifacts are frozen.
5. Write one review separating stable-backbone evidence, future-repertoire prediction, chronology specificity and recurrent-cell necessity.

## Stop rules

- A failed model does not invalidate the previously established stable templates.
- If R0–R2 are invalid, repair the common dataset/estimator before neural models.
- If R3 does not beat the strongest baseline, R4 is still run once as the predeclared nonlinear test, but no width/depth sweep follows.
- If ordered R3/R4 does not beat the order controls, close as a marginal-distribution result rather than history-dependent dynamics.
