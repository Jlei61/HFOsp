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

1. Build the four registered response-derived directions from Phase 2.
2. Evaluate RMS 0.6/0.8/1.0 after deduplication.
3. Freeze the manifest and ranking rule before any combination simulation.

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

