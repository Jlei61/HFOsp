# Topic 4 rev4 implementation plan: interictal-to-model-ictal discovery audit

Date: 2026-08-20
Status: draft for review; do not launch new simulation before Task 5 review
Spec: `docs/superpowers/specs/2026-08-20-topic4-data-driven-zm-model-ictal-qualification-and-fig3-bridge-design.md`

## Execution principle

The work point is selected from model-internal state and interictal evidence only. Clinical Fig.3
ictal targets are read only after `WORKPOINT_FROZEN.json` is written. Existing trajectories and
saved event/contact data are exhausted before any SNN rerun.

Long runs, if later authorized, use `systemd-run --user` plus `nohup`, one numeric thread per
worker and one monitor/waiter at intervals of at least 600 s. Worker count is set from measured
peak RSS while retaining at least 32 GiB free memory. No continuous polling and no output file is
overwritten in place.

## Task 0: freeze the discovery boundary

**Create**

- `config/topic4_data_driven_zm_discovery_audit_v1.json`
- `results/topic4_sef_hfo/data_driven_zm_ictal_transition/discovery_audit_v1/discovery_boundary.json`
- `.../provenance.json`

**Actions**

1. Hash the Fig.4 substrate manifest, direction classifier, OOD reference, event detector,
   montage, candidate trajectory inventory and existing Fig.5 artifacts.
2. Record exact 15-contact names, shafts, positions and accepted display order.
3. Record analyst exposure: E1146 Fig.3 targets and 2%/5% morphology have already been viewed.
4. Encode the allowed pre-freeze inputs and forbidden clinical ictal inputs from spec Section 8.
5. Preserve the 2026-08-17 execution contract and old morphology artifacts unchanged.

**Acceptance**

- clean checkout resolves every immutable input or fails with the exact missing path;
- all result producers list their source hashes;
- status is `DEVELOPMENT_ONLY_RETROSPECTIVE_DISCOVERY_AUDIT`, never `BLIND`.

## Task 1: audit exact Fig.4 carry-over and parameter semantics

**Create**

- `scripts/audit_topic4_zm_exact_fig4_carryover.py`
- `tests/test_topic4_zm_exact_fig4_carryover.py`
- `.../exact_carryover_audit.json`

**Actions**

1. Reconstruct the accepted Fig.4 substrate and compare node field, thresholds, EE/EI
   coefficients, topology, delays, incoming budgets and montage by hash/value.
2. Define `exact_fig4_carryover`: all Fig.4 substrate and pathway parameters unchanged, only Z/M
   feedback active.
3. Define every candidate with changed `I_th_EI`, Z/M constants or pathway doses as
   `calibrated_transition`.
4. Verify `dose_local_connectivity_coefficients` scales coefficient rows only and preserves the
   frozen mapper's target-wise budgets.

**Acceptance**

- no calibrated arm is labelled "same Fig.4 substrate + only Z/M";
- every candidate manifest records changed parameters relative to exact carry-over;
- existing raw trajectories are not regenerated.

## Task 2: implement model-ictal qualification and sensitivities

**Create**

- `src/topic4_fig5_ictal_bridge.py`
- `tests/test_topic4_fig5_ictal_bridge.py`

Add `qualify_model_ictal_v2`; do not modify the historical
`classify_sustained_runaway` implementation.

**Required outputs**

```text
operational onset and complete analysis windows
F_E and F_sheet duty
population-rate ratio
contact spectral-centroid difference and ratio
numerical-safety status
70/80/90% duty sensitivities
0.4/0.5/0.6 activity sensitivities
0.5/1/2 mm bin and occupancy sensitivities
onset +/-100 ms sensitivity
```

**Tests**

1. short returned high-rate bursts fail;
2. sustained broad 85-90% duty passes primary morphology;
3. local high-frequency activity without broad sheet recruitment fails;
4. broad high-rate activity without contact-frequency increase fails the primary Fig.5
   morphology but remains visible in diagnostics;
5. sparse bins cannot inflate `F_sheet`;
6. incomplete 1 s post window is not evaluable;
7. current 2%/5% fixtures are classified mechanically and preserve raw metrics exactly;
8. plotting code cannot promote a failed candidate.

## Task 3: implement repertoire retention from frozen Fig.4 contracts

**Create**

- `src/topic4_fig5_cross_state.py`
- `tests/test_topic4_fig5_cross_state.py`

**Actions**

1. Reuse the frozen direction classifier, OOD thresholds and
   `INTERICTAL_REPERTOIRE_RETAINED` reference quantiles without refitting.
2. Score every returned pre-onset event; do not select events by appearance.
3. Emit event count/rate, A/B support, OOD, confidence, rank profiles, recruitment, shaft
   participation and spatial range.
4. Confirm the displayed event is simply the last algorithmically qualifying event.

**Tests**

- fewer than 20 events, missing one mode, excessive OOD and low natural-KMeans alignment each
  fail the correct clause;
- missing/SCL-censored contacts worsen the corresponding metrics rather than disappearing;
- A/B labels are never assigned to the runaway interval;
- event ordering does not change under plotting or file enumeration.

## Task 4: implement motif reuse and structural null audits

**Create**

- `src/topic4_fig5_motif_reuse.py`
- `scripts/audit_topic4_fig5_motif_nulls.py`
- focused tests

**Readouts**

```text
event rank vs early-ictal first-passage rank
contact-pair precedence reuse
interictal vs early-ictal recurrent-E edge-flow cosine
reuse score for every event versus time-to-transition
```

**Nulls**

- within-shaft contact-label permutation;
- onset-time circular shift;
- learned-edge gain permutation within pathway x delay x distance strata;
- matched off-motif node sets.

**Structural-null acceptance**

- topology and delays unchanged;
- incoming EE/EI budget error at floating tolerance;
- pathway gain and edge-distance distributions retained;
- source/target degree summaries retained;
- motif alignment destroyed on synthetic known-motif fixtures;
- all permutation seeds and finite permutation counts frozen.

Do not launch shuffled-connectivity SNNs in this task. It is a zero-simulation metric and
structure audit.

## Task 5: zero-simulation candidate inventory and model-only shortlist

**Create**

- `scripts/rescore_topic4_fig5_model_internal_candidates.py`
- `.../model_internal_candidate_rescore.{csv,json}`
- `.../model_internal_shortlist.json`

**Inventory**

- exact Fig.4 carry-over if a completed trajectory exists;
- original active-Z/M Joint work point;
- completed threshold/adaptation candidates;
- all completed E-to-I doses, including 2% and 5%;
- incomplete candidates listed explicitly as not evaluable.

**Outputs per candidate**

```text
arm identity and parameter delta from Fig.4
model-ictal clauses and sensitivities
repertoire-retention clauses and distributions
rank/precedence/edge-flow reuse versus matched nulls
parameter distance from exact carry-over
available seed count and missing evidence
```

**Decision**

1. Exclude only non-finite or non-model-ictal candidates from the model-ictal shortlist.
2. Prefer candidates passing repertoire retention and motif reuse.
3. Shortlist at most three candidates for fixed replication.
4. If none passes Layer 2, label the best eligible candidate `MODEL_ICTAL_ONLY`; do not use a
   clinical bridge score to rescue it.
5. Stop for collaborator review before any work-point replication.

No Fig.3 patient ictal artifact is loaded by this producer. A test injects a forbidden path and
requires a hard failure.

## Task 6: fixed model-internal replication and work-point freeze

This is the first task allowed to rerun full trajectories.

**Development replication**

- at most three shortlisted candidates;
- exactly the same three predeclared network/noise seed pairs for every candidate;
- identical horizon, detector and recorder;
- exact carry-over included whenever technically runnable.

The three-seed stage supports work-point stability and a representative figure. It is not used
for a population-level pathway claim.

**Selection**

Apply the spec's lexicographic model-only rule: eligible proportion, cross-state gates, lower
reuse bound, then parameter distance. Write:

- `.../workpoint_replication.json`
- `.../WORKPOINT_FROZEN.json`

The freeze file includes all parameters, seeds, hashes, event rules, metric versions and the
statement that clinical bridge scores were not read by the selector. After writing it, no
clinical outcome may trigger work-point retuning.

If the paper will make a replicated cross-state claim, approve and run the fixed 12-network set
before inference. That decision is based on claim scope, not the direction seen in three seeds.

## Task 7: freeze and evaluate the clinical Fig.3 bridge

Only this task may open the patient ictal target.

**Create**

- `scripts/freeze_topic4_fig5_clinical_target.py`
- `scripts/evaluate_topic4_fig5_postfreeze_bridge.py`
- focused tests
- `.../clinical_target.json`
- `.../clinical_bridge_postfreeze.json`

**Patient target**

1. Export exact-name baseline, `[-10,0] s` and `[0,10] s` contact robust-z vectors for the 24
   eligible E1146 seizures.
2. Re-score them through the frozen Fig.3 scorer and require parity.
3. Build leave-one-seizure-out early-field agreement and split-half reliability distributions.
4. Keep seizure 2 display-only and verify `0.719127` / `0.570884`.

**Model spectral estimator**

- primary `10-150 Hz`, at least 500 ms multitaper/Welch windows;
- no bootstrap of overlapping windows as independent units;
- uncertainty from independent network/noise seeds and non-overlapping time blocks;
- patient `10-150 Hz` and model `1-150 Hz` sensitivities.

**Bridge outputs**

```text
D_energy
S_absolute and its location in patient LOSO distribution
D_increment with the exact per-contact L1 formula
increment Spearman/cosine diagnostics
D_time
descriptive J_bridge
```

`J_bridge` cannot change `WORKPOINT_FROZEN.json`.

## Task 8: optional target-informed retrospective sensitivity

Run only if collaborators explicitly want to compare historical candidates against patient
ictal targets.

For every matched patient-target surrogate, repeat the entire candidate Pareto/ranking process
and record the null minimum `J_bridge`. Label all outputs
`TARGET_INFORMED_SENSITIVITY_NOT_DISCOVERY`. This task cannot alter Fig.5 candidate identity.

## Task 9: motif-conditioned perturbation for Panel D

**First canary**

```text
candidate    frozen work point
seed         representative frozen seed
states       validated low activity and pre-transition
sites        interictal motif sites plus matched off-motif sites
doses        64,80,96,112,128 E neurons
continuation 200 ms from exact paired checkpoints
```

**Rules**

- directly injected neurons and the injection-source bin do not count toward broad recruitment;
- sham is run from the identical checkpoint/noise state;
- sham transition is recorded as a competing latency;
- scalar `n_crit` requires monotone dose response;
- below/above-ladder thresholds are interval-censored;
- non-monotone sites show the complete dose ladder;
- motif/off-motif matching covariates are audited before simulation.

The six-site canary decides only whether the endpoint is identifiable. A 7x7 map is secondary and
is run only if it adds spatial resolution after motif-versus-control evidence exists.

## Task 10: counterfactual motif dependence

At identical checkpoints and noise states compare:

```text
sham continuation
motif-specific learned-edge attenuation
equal-budget matched random-edge attenuation
```

Primary outputs are restricted ictal-free time, transition probability within the fixed horizon,
onset location and early-ictal field change. This experiment supports model-internal mechanism
only; it does not identify patient biological causality.

## Task 11: render and audit Fig.5

**Modify**

- `scripts/paper_figures/plot_fig5_data_driven_zm_main.py`
- `scripts/paper_figures/plot_fig5_data_driven_zm_panels.py`
- corresponding tests

**Panels**

1. A: one uninterrupted 15-contact readout plus broad-recruitment/rate strip and at least 1 s of
   model ictal state.
2. B: projected Z/M trajectory.
3. C: algorithmically last qualifying interictal event and early model-ictal `10-150 Hz` field,
   with model-internal reuse value.
4. D: motif versus matched off-motif susceptibility; do not restore the failed linear response.

**Audit**

- one frozen trajectory supplies A-C;
- exact contact join and registered electrode direction;
- model current never labelled SEEG;
- metadata reproduces all visible values;
- supplementary output shows all qualifying events;
- PNG/PDF/SVG and Chinese `README.md` pass visual and numerical QA;
- no plotting code can render a non-eligible candidate as the accepted model-ictal figure.

## Task 12: connectivity factorization, only after Fig.5 freeze

First compare exact carry-over, Node-only, generic distance/budget-matched connectivity, shuffled
learned connectivity and learned connectivity. Structural audits must pass before those runs.

If a pathway claim is intended, predeclare and run 12 paired network seeds for:

```text
Node
Node + E-to-E
Node + E-to-I
Joint
```

Report model-ictal latency/eligibility, repertoire retention, motif reuse, post-freeze bridge
components and the factorial interaction per network. The representative three-seed canary may
debug runtime but may not decide whether to expand based on effect direction.

## Task 13: closeout

Update the Topic 4 archive and paper figure registry only after visual acceptance. Record:

- model-ictal verdict and threshold sensitivities;
- exact carry-over versus calibrated-arm result;
- repertoire and motif-reuse verdicts;
- whether counterfactual dependence was tested;
- post-freeze clinical bridge vector;
- representative versus replicated evidence;
- development-only and non-blind boundary;
- all unrun tasks and reasons.

## Stop rules

| Condition | Action |
|---|---|
| no model-ictal candidate | write morphology failure; no clinical ranking or broad sweep |
| model ictal but repertoire absent | render only as model ictal; close cross-state claim |
| motif metric fails synthetic/null audit | do not run motif perturbation |
| exact carry-over fails, calibrated passes | report calibration dependence explicitly |
| no monotone packet regime | show ladder; do not invent `n_crit` |
| clinical bridge poor after freeze | report negative bridge; do not retune work point |
| no unseen patient/seizure unit | retain development-only, non-generalization language |
