# Topic 4 rev13 execution plan

## Phase 0: finish the patient event-window audit

1. Expose event end times from the canonical propagation loader.
2. Replace start-to-start `gap` with `next_start - previous_end` for mechanism
   timing claims; retain start-to-start only as a named diagnostic.
3. Add burst-separated, event-size/IEI-conditioned and local-occupancy nulls,
   plus recording-block bootstrap.
4. Freeze the result as short-lag clustering or burst-separated memory. The
   completed status is `PATIENT_MODE_MEMORY_NOT_ROBUST_TO_GAP_CONTROL`; neither
   a short-lived recovery law nor a persistent state is patient-authorized. In
   no case may this audit select `tau`, `A_ref` or a runtime mode state.

## Phase 1: implementation and parity

1. Implement the bounded controller in a new module with `threshold`, `step`,
   checkpoint and trace interfaces.
2. Add `node_accessibility=None` to the engine as an independent keyword-only
   mechanism. Preserve the literal old branch when absent.
3. Upgrade checkpoint capture/restore with backward-compatible controller-off
   loading and strict enabled/disabled pairing checks.
4. Add unit tests for exact-off parity, zero-sum error, amplitude bound,
   no-RNG behavior, step causality, checkpoint round-trip and controller+OU
   resume. Z/M coexistence is tested for engineering compatibility even though
   Z/M is off in the scientific canary.
5. Do not modify the frozen rev12 worker. Add a dedicated rev13 worker and
   freezer that reuse the causal-family event producer.
6. In worker preflight, reconstruct all candidates from the frozen config and
   hashed Stage-AK/AL inputs and require exact equality with the manifest.
   Explicitly hash every execution module that can change a number.
7. Run a real same-seed 10-s `exact_off` prefix against the historical Stage-AK
   artifact and require bitwise equality of whole-sheet activity, contact
   envelope and complete interior causal-family events.

## Phase 2: one-network model-internal canary

1. Freeze the primary Stage-AK substrate hash and six arms from the spec.
2. Use one unused network/noise seed with common random numbers for six runs.
3. Run 20 s. Historical replay showed that 10-s prefixes contain only 13--29
   clean families and positive held-out K2 evidence in 3/6 networks, whereas
   20 s provides 27--59 families. Zero events and runaway remain valid negative
   outcomes.
4. Before launch, run one 2-s `/usr/bin/time -v` sentinel. Set one numerical
   thread per worker and choose the worker count from measured peak RSS while
   retaining at least 48 GiB available memory; cap at 14 workers. Require at
   least 40 GiB free disk. Monitor no more often than every 600 s; no busy
   polling.
5. Formal K2 evidence excludes all time-overlapping causal families and uses
   event-count-matched, contiguous-time held-out K1/K2
   density on signed causal-family displacement. Whole-sheet onset-map KMeans
   is descriptive. Three blocks are equal-duration. Require at least 24
   isolated families, at least six per direction, opposite directions,
   within-cluster consistency, temporal-block recurrence and the frozen
   overlap/transition nulls.
6. The one-network result is engineering-only. Proceed to the two additional
   networks unless there is an engine/event-unit failure, all active arms are
   runaway, or all active arms produce no causal families. Do not stop because
   seed 2311 alone has negative or non-evaluable K2.

## Phase 3: fresh-network capacity replication

1. If Phase 2 yields at least one model-internally admissible zero-sum arm,
   freeze at most two arms without opening patient scores.
2. Run the primary substrate, paired exact-off controls and all six frozen arms
   on two additional unused networks: 12 runs, 18 primary trajectories total.
   Use the sentinel-derived worker count under the same memory and disk reserve.
3. Require both directions within individual networks. Different networks
   contributing different directions is not a pass.
4. Require `zero_sum_c020` to outperform raise-only and spatial-shift on
   causal-family integrity and natural two-branch support.
5. Treat `c=0.1` and `c=0.4` as signal/dose arms only. If either is the only
   promising dose, freeze coefficient-matched raise-only and spatial-shift
   controls before it can pass.
6. Only after a primary pass, run the channel-swapped sensitivity substrate
   with exact off, the winner and its nearest amplitude or time-scale neighbor
   on the same three seeds: nine additional runs.

## Phase 4: frozen patient evaluation

1. Open the frozen patient-training objective only for Phase-3 admissible arms.
2. Compare each arm with its paired static substrate on both modes and all four
   distribution components. Report natural KMeans separately.
3. A candidate advances only if its weakest complete-patient endpoint improves
   and the improvement is present in a majority of networks. No pooled-event
   rescue is allowed.
4. Patient held-out remains unopened. If no arm advances, close this Node
   recovery family rather than scanning more `tau` or amplitude values.

## Phase 5: confirmation and intervention

1. Only after a Node candidate passes Phase 4, run fresh-network confirmation
   with parameters and event rules frozen.
2. From one checkpoint per network, branch sham, global trace erase,
   activity-hotspot erase and matched low-trace-region erase. Rebuild one
   controller instance per branch and restore identical state.
3. Use causal-root families, not detector fragments, for all intervention
   endpoints.
4. Produce the standard Fig.4 pair: direct model waveform/dynamics and natural
   KMeans/patient cross-fit. Include transition/fragmentation diagnostics in a
   supplement, not as a replacement for the acceptance figures.

## Stop rules

- `PATIENT_EVENT_WINDOW_DEPENDENCE_UNRESOLVED`: event end/burst audit is not
  complete; do not infer a timescale.
- `ZERO_SUM_ENGINE_PARITY_FAIL`: no SNN canary starts.
- `ZERO_SUM_NODE_CAPACITY_NOT_OBSERVED`: model-internal canary fails; close the
  family.
- `DUALMODE_BY_GENERIC_ALTERNATION`: KMeans comes from forced alternation or
  fragmentation; reject.
- `ZERO_SUM_NODE_PATIENT_ALIGNMENT_FAIL`: capacity exists but complete patient
  modes do not improve; do not open EE/E-to-I/Z/M as a rescue.
- `ZERO_SUM_NODE_FROZEN`: fresh networks and same-checkpoint intervention pass;
  only then hand off to the later EE/E-to-I/Z/M ictal-transition line.
