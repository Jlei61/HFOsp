# Topic 4 rev12-ND execution plan

## Phase 0: freeze and audit

1. Snapshot the current Node field, 12-network baseline, patient target hashes,
   classifier and detector.
2. Record that patient held-out blocks have already been inspected and are
   development confirmation only.
3. Verify exact Node reconstruction, fixed mass, EE/E-to-I no-op and Z/M off.

## Phase 1: scoring implementation

1. Move event-feature, weighted `R2`, event-cloud and contrast metrics into a
   reusable module.
2. Implement the four-component per-mode distance, recording-block excess-noise
   calibration and weakest-mode LSE; retain raw and normalized components.
3. Implement event-level source maps and mode-template reliability.
4. Add the eight synthetic controls from the spec.
5. Freeze a versioned patient-scoring sidecar.

## Phase 1b: settled-episode correction (historical, insufficient)

1. Preserve the shared 12 ms detector-fragment contract for upstream activity
   detection.
2. Merge adjacent fragments across the detector's frozen 50 ms settling scale
   before contact readout, classification, KMeans or source-topology extraction.
3. Store fragment membership for every episode and verify exact parity when no
   fragments merge.
4. Re-score all completed development pools at 25/50/75/100 ms.  Do not use the
   already-opened confirmation pool to select the next field.
5. Reject any direct-readout figure that displays two fragments from one episode
   as separate modes.

## Phase 1c: population-excursion correction (current blocker)

1. Stop selection, confirmation, intervention and all new field optimization.
2. Define event boundaries from population active fraction only.  Start from the
   frozen high threshold; end only after activity stays below
   `RETURN_FRAC * high` for `5 * max(tau_m,E, tau_d,GABA) + max_delay`.
3. Recompute contact recruitment and rank on the whole excursion.  Store the
   high-threshold trigger, full analysis window, reset interval and constituent
   fragments separately.
4. Store a whole-run 1 mm neuron-activity movie at 2 ms resolution for canary
   workers, so any grouping can be audited without rerunning the SNN.
5. Run zero-simulation rescoring at four, five and six fast-state decay
   constants.  Candidate ranking and natural K=2 must be stable across all three.
6. Run one Node-only canary field on two development networks, generate full-
   excursion GIFs, and verify that every visually continuous wave packet belongs
   to the same statistical event.
7. Mark all 50 ms Stage-B/C rankings and figures invalid for field selection.
   Resume Phase 5 only after this correction is accepted.

## Phase 2: historical rescore

1. Inventory compatible V3-V6, D6 and rev11 Node artifacts.
2. Reconstruct frozen patient labels for every returned event.
3. Score complete event distribution and both modes with equal network weight.
4. Group only identical field hashes within the current spatial-OU stratum,
   deduplicate network seeds, then produce a field-level Pareto table.
5. Do not infer source topology where spike-level sidecars are absent.

## Phase 3: source-topology canary

1. Extend the Node worker to write 1 mm event onset maps without retaining the
   full spike matrix on disk.
2. Run current baseline on three canary networks.
3. Check mode-conditioned split-half reliability and one/two-source diagnostics.
4. Stop and repair the metric if stable synthetic templates and shuffled
   templates are not separated.

## Phase 4: same-checkpoint intervention

1. Select one evaluable TA and one evaluable TB event algorithmically.
2. Capture a checkpoint 40 ms before each event.
3. Replay sham, dominant region, secondary region and matched control with the
   same RNG and spatial OU state.
4. Confirm pre-intervention byte parity and quantify mode-selective effects.

## Phase 5: Node-only field search

1. Freeze current field and historical non-dominated fields as initial points.
2. Stage A: 4 x 4 smooth residual, common-random-number fit pool.
3. Select a small Pareto shortlist on fresh networks.
4. Stage B: optional 6 x 6 residual only around a Stage-A candidate that improves
   both patient modes without topology collapse.
5. Long runs use `systemd-run --user` plus `nohup`, one numerical thread per
   worker, memory sentinels and a 600 s monitor.  Worker count is selected from
   measured RSS while retaining at least 32 GiB available RAM.

## Phase 6: frozen confirmation and figures

1. Freeze one candidate before confirmation networks are opened.
2. Run paired current-baseline/candidate confirmation.
3. Generate the five figure products in the spec and inspect PNG/PDF/GIF.
4. Write a result report with safe claim, largest remaining gap and the exact
   handoff boundary for EE/E-to-I/Z/M.

## Phase 7: corrected local continuation (paused by Phase 1c)

The first confirmation candidate failed: complete held-out event-cloud R2 stayed
negative and the two modes did not both improve. Continue only on new fit,
selection and confirmation seed pools using the settled-episode worker. The
next local library may inherit fields from development pools only after
population-excursion stability is established, and never from the opened
confirmation results. EE, E-to-I and Z/M remain closed until the three minimal
scientific acceptance conditions in the spec are jointly met.

## Current execution order

The machine currently carries another high-load Topic 4 cohort.  Phases 0-2 are
zero-simulation and proceed immediately.  Phase 3 begins only when measured
resources allow at least three Node workers without reducing the reserved memory
margin.  The run is not accelerated by lowering duration or reusing selection
networks.
