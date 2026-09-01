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

If no coordinate passes the frozen local-linearity audit, record
`NO_LOCALLY_LINEAR_COORDINATE` as a completed response result rather than an
input error. Use the already measured antithetic endpoints as a discrete
fallback: retain only candidates that improve complete `J14` and weak mode A
in all three fit networks, protect B within 10%, and keep both effective
supports at least three. Reconstruct each nominated field bit-for-bit from its
source atlas coordinate. Do not fit a gradient, combine coordinates or rerun
the fit networks in this fallback.

## Phase 3: fresh-network selection

Run exact anchor plus nominated directions or measured discrete endpoints on a
disjoint three-network pool.
Select only by complete training-target score, weakest mode, mode-B protection
and per-network support. Do not open natural KMeans, held-out patient blocks or
figures during selection.

## Phase 4: unseen-network confirmation

1. Freeze one Node candidate before confirmation.
2. Run the candidate and paired exact dual anchor on unseen networks 2381--2383.
3. Re-establish the complete training target, weak-mode improvement, mode-B
   protection, support and safety independently in every network. Do not rerank
   on failure and do not yet open held-out patient blocks.

## Phase 5: frozen-candidate read-only acceptance

1. On the exact confirmation workers, require both modes in every network and
   run natural masked-rank KMeans without patient labels in the fit.
2. Require at least three events per supervised mode and natural cluster,
   per-network AMI at least 0.8, and a pooled positive-diagonal/negative-crossed
   model--patient matrix.
3. Only after KMeans acceptance, open patient held-out once. Relative to the
   paired exact anchor require improvement in eventwise prototype `R2`, both
   mode and weakest-mode losses, and both mode-specific and weakest event-cloud
   losses.
4. Require candidate source-topology separation above the occupancy-preserving
   within-network label-permutation q95 and better than the exact dual anchor.
   These checks may reject but never rerank the frozen candidate.

## Phase 6: same-checkpoint causal validation and freeze

1. In each confirmation network select one native event per mode by the frozen
   joint source-topology/contact-rank medoid rule.
2. Construct mode-contrast hotspots leave-one-network-out from the other two
   networks and a separated off-template control matched over the actual pulse
   footprint for Node field, `Delta Vtheta`, E count and baseline E rate.
3. Branch sham, MTA hotspot/control and MTB hotspot/control from the identical
   checkpoint and random stream. Apply the frozen +20 mV, 70 ms threshold pulse.
4. Test event survival before onset delay and require at least one mode's
   predicted hotspot to exceed both its opposite-mode effect and matched control
   in at least two of three networks.
5. Set `REV17_NODE_FIELD_FROZEN` only if this intervention is selective. Render
   the two Fig.4 acceptance views and causal-validation panels only from these
   frozen confirmation/intervention artifacts.

## Stop boundaries

- Zero-residual parity failure: repair representation; run no SNN.
- Universal runaway or missing event support: invalidate the affected
  coordinate, do not reinterpret it as mode failure.
- No transferable dual-channel response: close this local static Node family
  before considering connectivity or Z/M.
- Any confirmation, KMeans, held-out/topology or intervention failure rejects
  the one frozen candidate without returning to the atlas ranking.
- Node not frozen: EE, E-to-I, Z/M, ictal targets and hotspot claims remain
  closed.
