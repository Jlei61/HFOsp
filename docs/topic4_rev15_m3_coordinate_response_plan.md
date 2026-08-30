# Topic 4 rev15 execution plan

## Phase 0: close rev14 replication

1. Freeze the 18-run equal-network aggregate and its per-mode component
   decomposition.
2. Record `NO_USABLE_M3_ANCHOR` and do not launch the registered rev14 CMA-ES
   or shell-only M4 search.

## Phase 1: freeze the complete M3 coordinate atlas

1. Generate 28 unit coordinate directions from the unchanged paired-phase M3
   basis.
2. Normalize each on the physical sheet and freeze both signs at RMS 0.8.
3. Add nonselectable uniform and `exact_off` references.
4. Verify 58 unique candidate identities, exact sign pairs, field mass,
   signed-depth hash and mechanism-off invariants.
5. Prepare one full-duration uniform prewarm before queue launch.

## Phase 2: run and aggregate

1. Run all 58 fields on seed 2331 for 20 s with common random numbers.
2. Use systemd plus nohup, one numerical thread per worker and a 600-s monitor.
3. Derive concurrency from measured peak RSS. Hard cap is 14 workers; retain at
   least 64 GiB available memory before launching more jobs and never consume
   the emergency 48-GiB reserve.
4. Score every valid run with the frozen training-only `J14_v1` producer.
5. Produce the coordinate response table and A/B Pareto plot. Do not compute
   natural KMeans or read held-out data.

## Phase 3: freeze combinations

1. Build the four registered response-derived directions from Phase 2 using
   the formulas frozen in the specification.
2. Evaluate RMS 0.6/0.8/1.0 after deduplication.
3. Add the deterministic best-overall and best-A/B single-coordinate controls
   at RMS 0.8.
4. Freeze all 15 candidate identities (`exact_off`, 12 combinations and two
   single-coordinate controls) before any combination simulation.

## Phase 4: independent combination replication

1. Run combinations plus `exact_off` on seeds 2332--2333.
2. Aggregate seeds 2331--2333 only when the same candidate exists on all three;
   otherwise use 2332--2333 and label the atlas seed as construction-only.
3. Continue to full-M3 local optimization only if a usable two-mode anchor is
   obtained. Otherwise stop and revise the field family or objective from the
   observed A residual; do not rescue with EE, E-to-I or Z/M.

## Phase 5: post-freeze acceptance

Only after a Node candidate is frozen: fresh-network selection, natural KMeans,
the two Fig.4 figures, one-time developmental held-out evaluation and
same-checkpoint hotspot interventions follow the rev14 contract unchanged.

## Phase 4b: multinetwork coordinate tensor

This phase replaces direct progression to post-freeze acceptance because Phase
4 produced no usable anchor.

1. Copy the exact 58-candidate Phase-2 manifest into a new frozen manifest;
   candidate coefficients and hashes must remain identical.
2. Run the complete Cartesian product on seeds 2332--2333: 116 runs, 20 s each,
   CRN within each network, Node-only, all pathways and Z/M off.
3. Use up to nine one-thread workers, preserve at least 64 GiB available memory,
   stop below 48 GiB or 40 GiB free disk, and monitor every 600 s.
4. Combine these runs with the existing seed-2331 atlas only after validating
   every worker artifact and paired same-seed `exact_off`.
5. Estimate cross-network response consistency and construct a robust direction
   from all three training networks. Do not choose from natural KMeans, held-out
   data or rendered figures.
6. Freeze any robust direction before fresh selection-network simulation. If no
   common direction exists, close the local M3 response strategy rather than
   relaxing the mode-support requirement.

## Phase 5a: fresh Node-only post-selection

1. Run every frozen robust candidate and paired `exact_off` on seeds
   2341--2343 for 20 s with common random numbers.
2. Select only from the frozen training objective: A improves on 3/3, B is
   protected on 3/3, and equal-network A/B effective support is at least six.
3. For the single selected candidate, use complete returned non-overlapping
   causal families to audit supervised A/B support and independent masked-rank
   natural KMeans.
4. Require both modes in every network, natural-KMeans AMI at least 0.8 in every
   network, and positive-diagonal/negative-crossed pooled patient-profile
   geometry.
5. Render the two Fig.4-style figures only after the numerical audit. Figures
   cannot change the candidate.

## Phase 5b: one-time held-out and source-topology audit

1. Freeze the accepted candidate, the exact worker hashes and the paired
   `exact_off` workers before opening held-out.
2. Compute complete-event-cloud held-out `R2` and held-out weakest/A/B mode
   losses. Training-target losses are reported separately and cannot satisfy
   the final clauses.
3. Compute mode-conditioned source-topology quality on each network and compare
   it with 4096 within-network label permutations preserving each network's
   mode occupancy.
4. Advance only when held-out `R2` is positive and better than `exact_off`, the
   weakest and both individual modes improve, and topology exceeds the matched
   null q95.
5. Do not re-rank or resume fitting after held-out is opened.

## Phase 6: same-checkpoint hotspot intervention

1. Select a representative seed and one event per mode algorithmically from
   the frozen three-network source maps.
2. Branch from checkpoints 40 ms before native onset with identical state and
   random streams: sham, dominant mode hotspot suppression, secondary hotspot
   suppression and matched off-template suppression.
3. Require exact sham replay and pre-pulse spike parity. Report event survival,
   latency, mode identity, rank displacement and source-topology displacement.
4. Freeze Node only if at least one predicted hotspot shows a mode-selective
   effect not reproduced by the matched controls. Otherwise retain the field as
   an exploratory Node candidate and keep EE, E-to-I and Z/M closed.
