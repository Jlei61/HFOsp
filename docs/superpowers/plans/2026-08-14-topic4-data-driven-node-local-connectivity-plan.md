# Topic 4 rev11-NLC execution plan

**Spec:** `docs/superpowers/specs/2026-08-14-topic4-data-driven-node-local-connectivity-design.md`

## NLC0: current-network audit

- [x] Confirm target/source naming for all four pathways.
- [x] Quantify one frozen 40k-neuron network: edge counts, fixed in-degree,
  source out-degree and distance geometry.
- [x] Freeze a machine-readable baseline audit with cache/config hashes.

## NLC1: continuous local mapper

- [x] Implement separate target-normalized E->E and E->I transforms.
- [x] Preserve pathway-specific incoming budgets, topology, delay bins and all
  GABA matrices.
- [x] Test exact no-op, pathway isolation, finite values and off-field source
  participation.
- [ ] Add NLC2 structural summaries for field/background source-target flow and
  weighted distance.

## NLC2: capacity canary

- [x] Freeze candidate coefficient bounds before simulations.
- [x] Run paired `Node`, `Node+EE`, `Node+EtoI`, `Node+EE+EtoI` arms.
- [x] Use slow-off dynamics, common detector and common random numbers.
- [x] Keep only non-finite, late runaway and statistic-not-evaluable as hard
  invalid states.
- [x] Rescore a small Pareto set using natural KMeans, cross-fitted patient
  geometry, weakest-mode error and event yield.

## NLC3: joint search and confirmation

- [x] Release all 12 low-frequency whole-sheet Node perturbation coordinates.
- [x] Jointly fit Node and retained E->E/E->I coefficients around the two
  complementary NLC1 centres.
- [x] Separate fit, selection and confirmation network pools.
- [x] Produce canonical direct-readout and KMeans figures.
- [x] Report network-level paired effects and all negative arms.

## NLC3R: post-hoc calibration of the confirmation statistics

Review of 2026-08-15, `docs/archive/topic4/sef_hfo/rev11_nlc_frozen_substrate_review_2026-08-15.md`.

- [x] Calibrate balanced alignment against a within-network direction-label
  permutation null; the fixed 0.5 threshold sits below the null median.
- [x] Calibrate the cross-fit margin against a contact-correspondence
  permutation null, free and within-shaft.
- [x] Restore the two D5.2 pooled diagnostics this rev dropped: seed-stratified
  permutation p and the patient-matched purity benchmark.
- [x] Replay both fixed-threshold statements on every arm; record that the
  Node-only control also clears them.
- [x] Rebuild Fig.4 so the acceptance quantities, the Node-only control and the
  matrix of the plotted clusters are all on the canvas.
- [ ] Carry the calibrated form of each acceptance statement into the NLC4 spec
  before any Z/M transfer claim.

## NLC4: seizure-unification transfer

- [ ] Freeze the selected static substrate before enabling Z/M.
- [ ] Transfer Node/connectivity unchanged to active Z/M.
- [ ] Run >=20 s with late-runaway invalid and lifecycle readouts.
- [ ] Do not retune Z/M to improve interictal KMeans.
