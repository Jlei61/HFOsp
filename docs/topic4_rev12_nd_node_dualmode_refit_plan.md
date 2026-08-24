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

## Phase 1d: spatiotemporal-cascade correction (historical, still insufficient)

1. Retain population excursions only as contact-independent outer envelopes.
2. Label 2 ms by 1 mm sheet-activity nodes with a 3 x 3 spatial neighborhood
   across adjacent frames; do not use contacts or patient labels.
3. Group detector fragments only when they share a cascade component.  Mark
   multi-origin fragments as compound instead of assigning them a direction.
4. Run synthetic travelling-wave, detector-dip and simultaneous-origin controls.
5. Rescore the fresh canary at two/three active neurons per bin and dominance
   0.65/0.70/0.75.  Report event count, compound rate, natural KMeans alignment
   and complete held-out R2.
6. Freeze this event producer before creating another field library.  The new
   objective must use cascade events and retain compound/OOD fractions as
   explicit continuous penalties.

Observed canary result: both legacy fields were stable negative controls
under the cascade unit (patient-direction purity about 0.56-0.59; held-out R2
-0.35 to -0.44).  This closes the false dual-direction interpretation but does
not yet recover the patient repertoire.

## Phase 1e: directed-lineage correction (necessary but insufficient)

1. Preserve the frozen whole-sheet movie, detector fragments and contact readout;
   do not rerun the SNN for this correction.
2. Trace roots forward one movie frame at a time.  Partition multi-root patches
   by seeded watershed and retain separate roots after collision.
3. Count equidistant collision mass in the fragment denominator.  Keep the 70%
   root-dominance rule and retain compound fragments explicitly.
4. Require the frozen detector return decision for every constituent fragment;
   ending before the recording boundary alone is insufficient.
5. Recompute contact onset/rank and root first-arrival maps after event windows
   are frozen.  Never reuse source maps from the undirected event unit.
6. Zero-simulation resegment all 18 historical fields x 2 fit networks, then
   recompute matched patient loss, equal-network KMeans, OOD and compound rate.
   This completed with no historical field below the matched patient q95 floor,
   negative held-out R2 for all fields, and substantial compound detector
   fragments.  Before any new field fit, render one algorithmically selected
   representative from each natural KMeans cluster with all concurrent activity
   visible and the selected directed root explicitly outlined.
7. Recompute contact recruitment from activity assigned to that same root.  Audit
   the 1 mm binned Gaussian sampler against the frozen per-neuron full-contact
   envelope and require Pearson r >= 0.98 for every contact.  Re-score all 18
   historical fields again; only this lineage-restricted result may seed a new
   continuous-field search.
   Completed: all 36 workers passed contact-sampler parity, but every historical
   field remained above the patient q95 floor and had negative held-out R2.
8. Treat the undirected library's apparent 0.93 alignment as non-selective until
   the directed resegmentation reproduces it.

## Phase 1g: causal population episode with root topology (current blocker)

1. Formally invalidate every field ranking that drops multi-root detector events.
2. Use contact-independent whole-network active fraction and the frozen fast-state
   reset to define one complete statistical episode.  Contacts only read the
   already-frozen episode.
3. Retain all persistent roots inside each episode as topology annotations.  Do
   not split an episode by root and do not exclude multi-root events.
4. Verify exact detector-fragment conservation and split invariance with synthetic
   controls, including a detector dip inside one continuing population state.
5. Re-score the same three fields and two networks at four, five and six fast-state
   decay constants without rerunning the SNN.  Require stable episode partitions,
   mode assignments and natural-KMeans interpretation before reopening a fit.
6. Render one complete-episode GIF per natural cluster with every simultaneous
   sheet activation visible.  The figure must not outline only one selected root.
7. Only after this canary is accepted may Phase 5 regenerate a Node-field library.
   EE, E-to-I and Z/M remain frozen throughout.

Observed result: the population reset definition overmerged six or seven packets
into 0.7-0.8 s episodes and changed substantially across 4/5/6 time constants.
It remains an outer GIF diagnostic, not the scoring event.

## Phase 1h: persistent-root coactivity episode (current blocker)

1. Within every detector fragment, group roots only when their activity actually
   overlaps in a movie frame.  Split roots that are merely sequential inside a
   long detector window.
2. Across detector fragments, merge only through a shared persistent root.
3. Include every positive-mass root in the exact per-neuron contact readout; remove
   the 70% dominance exclusion from the patient objective.
4. Run synthetic simultaneous-root, sequential-root and shared-root detector-dip
   controls, with complete detector-fragment coverage as an invariant.
5. Run the same three fields x two seeds at 4/5/6 root memories.  Candidate
   summaries use worst patient loss and minimum KMeans alignment, never the best
   segmentation.
6. Render both complete outer excursions and root-coactivity events.  Accept the
   event producer only when the latter no longer hides simultaneous upper-sheet
   activity or splits one persistent wave into opposite modes.

Observed result: detector-window dependence was removed and no fragment was
dropped, but the inherited 4/5/6-tau root memory remained too broad.  More than
90% of events were multi-root, representative windows lasted 120-330 ms, and
event counts changed 20-30% across the sensitivity range.  This version is an
audit artifact and cannot reopen the field fit.

## Phase 1i: engine-derived excitatory root memory (current blocker)

1. Replace `c * max(tau_m,E, tau_d,GABA) + global maximum delay` with the
   simulated E-to-E AMPA-to-membrane PSP tail plus delay support from E-to-E
   edges inside the same 1 mm movie-parent cone.
2. Freeze 10% of PSP peak as primary and 20%/5% as sensitivities.  Use the
   maximum local compatible delay; do not inspect KMeans or patient loss when
   setting these values.
3. Add synthetic tests for the exact exponential-Euler PSP response, local-edge
   delay filtering, simultaneous roots, sequential roots and detector dips.
4. Run the same three frozen fields x two network seeds with exact per-neuron
   contact readout.  No field parameter is released and selection remains
   forbidden.
5. Report event count, partition stability, multi-root fraction, matched 6-vs-6
   patient loss and natural KMeans for all three tail definitions.
6. Render algorithmic complete-event representatives.  If long multi-packet
   episodes or segmentation-dependent A/B labels remain, keep Node fitting
   closed and revise the event producer rather than tuning the field.

Observed result: the canary remained invalid.  The primary events had median
duration 144-164 ms and 91-97% contained multiple roots.  Complete-event GIFs
showed successive hotspots inside both natural clusters.  The 10% PSP memory
was shorter than the old global formula, but transitive root coactivity still
merged distinct packets.

## Phase 1j: observable causal-root canary (completed)

1. Use one directed root as one model observation.  A detector fragment may
   establish observability but cannot extend latent root boundaries.
2. Keep mixed-root fragments as explicit compounds outside KMeans and report
   their fraction; never force them into A or B and never silently delete them.
3. Freeze half-maximum E-to-E PSP support plus maximum local delay as primary.
   Audit 80%/20% support and root dominance 0.60/0.70/0.80 one axis at a time.
4. Run the same three fields x two network seeds, with Node only and exact
   per-neuron root-restricted contact readout.
5. Report clean event counts, compound fraction, latent root duration, matched
   patient loss, natural KMeans alignment and partition sensitivity.
6. Render one algorithmic root representative per natural cluster with all
   concurrent sheet activity visible.  Reopen fitting only if the GIF and
   metrics agree that A/B are different causal propagation events.

Observed result: event identity passed but all three fields failed scientific
acceptance.  Every scored event contains exactly one latent root, detector
fragments no longer determine its boundaries, and the memory-tail variants are
stable.  However 40-58% of detector fragments are compound, matched patient loss
worsened to 1.17-1.27 after mixed observations were removed, and the two natural
clusters usually propagate in the same spatial direction on at least one seed.
The old field ranking therefore benefited from detector-window composition.

## Phase 1f: native-worker parity before refitting

1. Make `lineage_restricted_sheet_activity` the formal contact readout in the
   simulation worker, rather than a retrospective rescoring option.
2. Store the readout source and per-contact full-trace sampler parity in every
   worker JSON.  Fail the worker if any contact has Pearson r below 0.98.
3. Rerun `stage_c_r04_p` on seed 2201 with Node only and compare all shared
   arrays, event boundaries, root identifiers, returned status, onsets and ranks
   exactly against the frozen Stage-G zero-simulation rescore.
4. Do not release a new field parameter or start the fit queue unless this
   parity canary is exact.  The canary is an engineering equivalence test, not a
   new patient-fit result.

The historical-field canary passed, but the first new-field batch exposed a
support problem: two workers fell below r=0.98 because the 1 mm population-
averaged contact approximation is not invariant to new spatial activity
patterns.  Stage-I v1 is therefore invalidated without scoring.  The replacement
keeps the 1 mm/2 ms movie for root identity, assigns each neuron's binned spikes
to that root, and applies the original normalized 0.25 mm per-neuron Gaussian
contact sampler.  No threshold relaxation is allowed.

The exact-neuron canary then passed with the frozen activity trajectory, all 50
event boundaries and all directed root identifiers unchanged.  Ten of 50 events
changed their recruited-contact mask relative to the 1 mm approximation; the
minimum event-level Jaccard was 0.5.  This confirms that the correction is
scientifically material.  Stage K now reruns the identical 54 fields under the
exact-neuron readout; no field parameter is regenerated.

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

## Phase 5: causal-root Node-only field search (current runnable phase)

1. Freeze current field and historical non-dominated fields as initial points.
2. Stage I: use the three best final Stage-G fields, their pairwise midpoints and
   signed 4 x 4 whole-sheet smooth residuals at two amplitudes.  The residual
   basis is uniform over the sheet and never receives contact coordinates.
3. Rank on the componentwise worst case across the five frozen causal-root
   memory/dominance variants: matched patient-training loss, equal-network
   KMeans alignment, per-network K2-vs-K1 support, OOD and compound rate.
4. Add the root-onset-map directional endpoint: patient-labelled modes must have
   opposite early-to-late displacement along the training-only TA/TB contrast
   axis.  Protect the weaker mode and give networks equal weight.
5. Do not use held-out patient R2 for selection.  Event formation itself never
   sees contact geometry; the patient axis enters only after a root is frozen.
6. Run all 54 frozen fields on a two-network common-random-number fit pool.
7. Select a small Pareto shortlist on fresh networks.
8. Stage B: optional 6 x 6 residual only around a Stage-A candidate that improves
   both patient modes without topology collapse.
9. Long runs use `systemd-run --user` plus `nohup`, one numerical thread per
   worker, memory sentinels and a 600 s monitor.  Worker count is selected from
   measured RSS while retaining at least 32 GiB available RAM.

## Phase 6: frozen confirmation and figures

1. Freeze one candidate before confirmation networks are opened.
2. Run paired current-baseline/candidate confirmation.
3. Generate the five figure products in the spec and inspect PNG/PDF/GIF.
4. Write a result report with safe claim, largest remaining gap and the exact
   handoff boundary for EE/E-to-I/Z/M.

## Phase 7: corrected local continuation (paused until Phase 5 fit result)

The first confirmation candidate failed: complete held-out event-cloud R2 stayed
negative and the two modes did not both improve. Continue only on new fit,
selection and confirmation seed pools using the causal-root worker. The next
local library may inherit fields from development pools only after cascade
stability is established, and never from the opened
confirmation results. EE, E-to-I and Z/M remain closed until the three minimal
scientific acceptance conditions in the spec are jointly met.

## Current execution order

Phase 1j is accepted for event identity.  Freeze the 54-field Phase 5 library at
the current commit, run two common-random-number fit networks, aggregate with the
five-variant conservative event envelope and causal-direction term, and inspect
the shortlist before opening selection networks.  Intervention, confirmation,
EE, E-to-I and Z/M remain closed until a field improves both patient modes and
shows opposite causal propagation on the fit and fresh selection pools.
