# Topic 4 rev21 implementation plan

## Stage 0: freeze inputs and seed semantics

1. Freeze hashes for rev20 config, candidate manifest, selected-candidate file,
   confirmation aggregate, patient-training interictal target, classifier and
   held-out endpoint.
2. Freeze `dc_both_scale_1p25` as the only substrate candidate.
3. Add separate `topology_seed` and `dynamics_seed` arguments without changing
   historical one-seed behavior.
4. Regression-test that `topology_seed == dynamics_seed == old_seed` reproduces
   the historical network, external drive and event arrays exactly.

## Stage 1: Z/M-off seed-factorization audit

1. Run `3 topology x 4 dynamics` for the frozen substrate with Z/M off.
2. Use common dynamics seeds across all topologies and common topology seeds
   across all dynamics.
3. Score complete distribution, natural KMeans and OOD with the rev20 contract.
4. Report two-way variance decomposition and every matrix cell; do not collapse
   the result to one pooled number.
5. Render a compact seed audit. Continue to Z/M as development-only even if
   topology dependence is large, but downgrade ensemble-stable wording.

## Stage 2: unified transition worker

1. Extend the complete causal-family worker rather than the old detector-
   fragment Z/M worker.
2. Enable MZ slow variables and record at least 1,200 ms after transition.
3. Derive and store 20 ms population activity, local occupied-bin recruitment,
   contact readout, rate traces, Z/M traces and complete pre-transition families.
4. Add tests for repeated-burst rejection, continuous broad-state acceptance,
   frequency increase, post-window completeness and pre-onset family filtering.
5. Run one Z/M-off parity job and one historical reference-Z/M smoke job before
   any grid.

## Stage 3: coarse Z/M access map

The screen contains the Z/M-off candidate plus all 16 active candidates on the
same `2 topology x 2 dynamics` cells. Report paired active-minus-off effects.
Separately compare candidate aggregate endpoints with the q05--q95 support from
Stage 2; use the explicit one-sided retention rule in the spec. Do not read the
patient held-out artifact in this stage.

1. Generate the frozen 4 x 4 `(s_I,s_M)` grid.
2. Run a small fixed paired seed set under systemd/nohup.
3. Aggregate model-ictal morphology and the three separate interictal endpoints.
4. Do not use patient ictal data and do not add grid points after seeing output.
5. Prefer a formally eligible, interictal-retained candidate. If none exists,
   carry exactly one interictal-retained, operational and numerically safe
   near-state point into the already frozen timescale grid by its weakest-clause
   shortfall. Do not call that point eligible and do not relax the final gate.

## Stage 4: timescale refinement

1. Around the frozen coarse candidate, run the 3 x 3
   `(tau_z,tau_adp)` grid while preserving integrated M strength.
2. Apply the frozen lexicographic selection rule and the original full state
   criteria. Near-state status from Stage 3 has no confirmatory standing.
3. Freeze one finalist before confirmation only if it is formally eligible and
   retains all three interictal endpoints.

## Stage 5: independent confirmation and controls

1. Run the finalist on a fresh `3 topology x 4 dynamics` matrix.
2. Run matched Z/M-off, Z-only and M-only controls on the same matrix.
3. Report paired network/dynamics contrasts and estimability, with no event as a
   valid outcome.
4. Freeze `WORKPOINT_FROZEN.json` only if model-ictal and interictal-retention
   requirements both hold.

## Stage 6: post-freeze bridge and Fig.5

1. Open the clinical target only after the work-point hash exists.
2. Compute energy, absolute spatial, incremental spatial and time endpoints.
3. Produce one uninterrupted representative trajectory plus multi-seed panels.
4. Generate PNG/PDF/GIF, metadata and Chinese README; visually inspect all.

## Execution safety

- All long jobs use `systemd-run --user` plus `nohup`.
- Numerical thread count is one per worker.
- The controller computes slots from measured peak RSS and keeps at least
  32 GiB available memory.
- Status is written atomically; completed JSON/NPZ hashes make jobs resumable.
- Monitor interval is 600 s and desktop notification fires on completion or
  failure.
- Existing rev20 and historical Z/M artifacts are immutable and never deleted.
