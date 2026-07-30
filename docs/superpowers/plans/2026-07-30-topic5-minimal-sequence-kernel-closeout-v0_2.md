# Topic 5 minimal sequence-kernel closeout v0.2 execution plan

## Milestone A — freeze and inventory

1. Fingerprint the 34-patient v0.4 dataset and all reused checkpoints.
2. Record the exact tie/cardinality inventory.
3. Freeze Epilepsiae as development and Yuquan as confirmation for the new
   endpoints; reverse direction is sensitivity.
4. Write a machine-readable run manifest before formal target access.

## Milestone B — exact scoring repair

1. Add an exact joint-softmax decomposition into contact-choice, continue and
   terminal STOP terms.
2. Add tests proving exact reconstruction of the original NLL.
3. Export one row per heldout decision with event, step, action type,
   candidate-mask fingerprint and all loss components.
4. Re-evaluate unordered, H1/H2/H3/full, selected linear state, rank shuffle
   and retained interventions on identical decision keys.
5. Remove the raw-stop-logit sigmoid metric from scientific summaries.

## Milestone C — lag-kernel analysis

1. Extract patient-specific contact-space \(K_0\ldots K_5\) and STOP kernels
   from each selected linear-state checkpoint.
2. Run lag 0/1/2/3+ contact-token ablations while preserving causal scalar
   covariates and candidate masks.
3. Compute seed stability and patient-level lag contributions.
4. Build the finite-horizon Hankel matrix and summarize its singular spectrum.

## Milestone D — explicit FIR-H3

1. Implement a three-lag, no-recurrent-state residual model over a frozen
   unordered baseline.
2. Add tests for no future access, exact three-lag horizon, frozen baseline
   parameters and matching candidate masks.
3. Run three engineering patients without scientific selection.
4. If all engineering checks pass, run 34 folds and three seeds with bounded
   GPU memory and full logs.
5. Compare FIR-H3 and selected linear state separately for contact-choice and
   STOP.

## Milestone E — confirmation and robustness

1. Train shared ordered models on Epilepsiae and freeze the model/endpoints.
2. Calibrate only Yuquan train80 contact offsets and evaluate Yuquan heldout20.
3. Repeat Yuquan-to-Epilepsiae as sensitivity.
4. Report event-count, contact-count and event-length heterogeneity.
5. Re-encode heldout decisions at 1/2/5/10 ms tolerance without retuning and
   audit whether conclusions depend on near-tie grouping.

## Milestone F — matched contexts and `when` feasibility

1. Count repeated unordered-prefix contexts with different recent order.
2. Run the observed-context outcome test only if the frozen support threshold
   is met.
3. Audit early-ictal target structure for patient-mean and seizure-residual
   reliability.
4. Audit absolute event times, block continuity and causal next-event
   denominators for the inter-event Gate 1 baselines.
5. Do not train a new recurrent architecture in this milestone.

## Milestone G — closeout

1. Run focused and regression tests in `cuda_env`.
2. Produce patient-level CSV, cohort JSON, resource log and acceptance file.
3. Generate the revised six-panel paper-ready figure plus Chinese
   `figures/README.md`.
4. Write a detailed Chinese report separating supported, negative,
   unidentifiable and future-gated claims.
5. Update Topic 5 archive and figure indices without overwriting prior
   bounded-result artifacts.
