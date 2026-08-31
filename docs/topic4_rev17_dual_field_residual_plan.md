# Topic 4 rev17 execution plan

## Phase 0: representation repair

1. Freeze the accepted mean and dispersion spline fields from the rev16 source
   manifest.
2. Generate 15 observation-invariant cosine residuals on each channel with
   antithetic signs and retain an explicit exact-zero anchor.
3. Reconstruct seeds 2351--2353 without stepping the SNN and require exact
   `h` and `Delta Vtheta` parity with archived exact-off workers.

## Phase 1: fit-network response atlas

1. Freeze the 61-candidate manifest at a clean commit.
2. Launch 61 candidates by three fit networks through managed user services,
   numerical threads fixed to one and measured-RSS worker limits.
3. Monitor at 600-s intervals; stop new launches on memory, disk, non-finite,
   provenance or late-runaway failure. Completed artifacts are resumable.
4. Aggregate complete patient-training endpoints with equal network weight.

## Phase 2: joint local directions

1. Estimate central finite differences separately for mean and dispersion.
2. Audit antithetic curvature and discard coordinates whose response is too
   nonlinear for a local model.
3. Construct a small set of regularized joint directions for weak-mode,
   maximin and support-protected objectives.
4. Project candidate amplitudes to a bounded local RMS neighborhood around the
   exact dual anchor; zero remains exactly reconstructable.

## Phase 3: fresh-network selection

Run exact anchor plus nominated directions on a disjoint three-network pool.
Select only by complete training-target score, weakest mode, mode-B protection
and per-network support. Do not open natural KMeans, held-out patient blocks or
figures during selection.

## Phase 4: Node confirmation and causal validation

1. Freeze one Node candidate before confirmation.
2. Run unseen-network natural KMeans, frozen classifier, complete held-out
   distribution and mode-specific source topology.
3. From the same checkpoints, perform crossed hotspot suppression/relocation
   controls and test mode-selective changes in onset density and event rate.
4. Render the two Fig.4 acceptance views only from the frozen confirmation and
   intervention artifacts.

## Stop boundaries

- Zero-residual parity failure: repair representation; run no SNN.
- Universal runaway or missing event support: invalidate the affected
  coordinate, do not reinterpret it as mode failure.
- No transferable dual-channel response: close this local static Node family
  before considering connectivity or Z/M.
- Node not frozen: EE, E-to-I, Z/M, ictal targets and hotspot claims remain
  closed.
