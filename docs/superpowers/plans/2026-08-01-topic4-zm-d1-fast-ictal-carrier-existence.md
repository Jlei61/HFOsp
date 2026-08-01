# Topic 4 Z/M fast dynamics → lifecycle vertical slice — implementation plan

**Date:** 2026-08-01
**Revision:** 2 — development-first execution
**Status:** READY FOR DEVELOPMENT EXECUTION
**Spec:** `docs/superpowers/specs/2026-08-01-topic4-zm-d1-fast-ictal-carrier-existence-design.md`

## Goal

Reduce the main scientific uncertainty quickly:

1. what does E-only dynamic threshold actually do to the Phase-C tonic branch?
2. is any resulting carrier-like state reachable when \(\phi,z,m,S_G\) all run
   from interictal baseline?
3. once reachable, does it show a native-exit tendency or a finite-control route
   back to returning events?

This plan deliberately separates discovery from confirmation.  It does not
build a new production evidence system before the first SNN result.

## Reuse before creating

Reuse these existing components:

- `src/snn_engine/slow_field.py` — tested `phi_increment` hook;
- `src/topic4_zm_fast_carrier_state.py` — old-checkpoint migration;
- `src/topic4_zm_fast_carrier_runtime.py` — frozen-state runtime helpers;
- `scripts/run_topic4_zm_phasec_cell.py` — SNN/state/noise/readout path;
- `src/topic4_zm_carrier_gate_v2.py` — operational carrier descriptors;
- Phase-C resource measurement and crash-safe per-cell patterns.

Create at most one thin runner, one thin analyzer/plotter and one focused test
file during Stages A/B:

- `scripts/run_topic4_zm_fast_lifecycle_development.py`;
- `scripts/analyze_topic4_zm_fast_lifecycle_development.py`;
- `tests/test_topic4_zm_fast_lifecycle_development.py`.

Do not create a separate observation package, general coordinator, multi-module
verdict framework or paper figure suite until Stage C is opened.

## Fixed resource discipline

- `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=NUMEXPR_NUM_THREADS=1`;
- measure one production worker first;
- compute worker count from measured RSS with the existing 1.25× margin;
- retain at least 96 GB `MemAvailable` and eight logical CPUs;
- never kill or modify peer-worktree processes;
- one atomic NPZ + minimal JSON per run;
- resume only missing or technical-invalid cells.

---

## Task 1 — Minimal mechanism sanity, then stop engineering

**Time/compute target:** less than one focused coding block; no production SNN.

### Required tests

- [ ] Verify

  `delta_phi = f_phi * gap / ((tau_phi_ms/1000) * r_core_ref_hz)`

  against a hand calculation; a factor-1000 error must fail.
- [ ] Verify `use_phi=False` parity using the existing historical test, not a new
  re-bless.
- [ ] Verify enabled phi affects E thresholds only; I entries remain exact zero.
- [ ] Verify one E spike gives one `delta_phi` jump and decay is
  `exp(-dt/tau_phi)`.
- [ ] Verify Arm-A freeze holds only \(z,m\) fixed; \(S_G\), phi, fast E/I,
  delays and future noise remain active.
- [ ] Verify Arm-A initial phi is zero and label it
  `branch_intervention_not_reachability` in every output.

### Minimal implementation

- [ ] Add only the wrapper code necessary to pass one of the six phi settings
  into the existing current-based runtime.
- [ ] Reject `use_zm_conductance=True` and any E→E semantic-hash change.
- [ ] Add a `--smoke` output root that cannot be consumed by development runs.

### Command

```bash
python -m pytest -q \
  tests/test_zm_dynamic_threshold.py \
  tests/test_topic4_zm_fast_carrier_state.py \
  tests/test_topic4_zm_fast_carrier_runtime.py \
  tests/test_topic4_zm_fast_lifecycle_development.py
```

**Hard checkpoint:** once these tests pass, start Task 2.  Do not add real-data,
verdict, archive or formal plotting infrastructure here.

## Task 2 — Immediately run the 24-cell phenotype discovery matrix

**Scientific question:** can \(\phi\) break the tonic branch, and what does it
produce instead?

### Locked matrix

- seed: 1;
- checkpoints: `bounded_mid__rising`, `bounded_mid__peak`,
  `bounded_late__rising`, `bounded_late__peak`;
- phi panel: `tau_phi={60,100,160} ms × f_phi={0.15,0.30}`;
- future noise: paired replay only;
- duration: 6 s, with 1 s switch-on transient and 5 s description;
- current-based membrane; dynamic \(S_G\); frozen \(z,m\); phi starts at zero.

### Runner outputs

- [ ] Reuse existing state/noise hashes and readout code.
- [ ] Save core/surround/E/I rate, all-sheet active fraction, refractory
  occupancy, phi/\(S_G\), spatial bins/kymograph and existing vSEEG proxy.
- [ ] Write one minimal provenance JSON: seed, checkpoint, noise, parameter row,
  code SHA, source hash, status and resource peak.
- [ ] Quarantine partial output; exact valid reruns are idempotent.

### Phenotype analyzer

- [ ] Classify each cell as `tonic`, `burst_train`,
  `spatially_relayed_carrier`, `metastable_carrier_like`, `silence`,
  `whole_sheet_oscillation`, `runaway` or `technical_invalid`.
- [ ] Report modulation, persistence/gaps, refractory occupancy, axial latency,
  spatial active fraction and v2.1 operational gate fields separately.
- [ ] Do not require perturbation return, real-data sidecar or AI.
- [ ] Produce one phenotype-matrix CSV/JSON and one compact diagnostic figure
  containing representative rate, phi, vSEEG and kymograph traces.
- [ ] Write a short Chinese `figures/README.md` after the figure exists.

### Execution

First measure one complete cell, then launch as many workers as the measured RSS
allows under the fixed reserve:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
python scripts/run_topic4_zm_fast_lifecycle_development.py \
  discovery --confirm-run --workers auto

python scripts/analyze_topic4_zm_fast_lifecycle_development.py discovery
```

### Required stage report

Before additional implementation, record:

- which phenotypes occurred;
- whether a tonic→burst/carrier→silence boundary is visible;
- top zero, one or two Stage-B candidates;
- whether one bounded local refinement direction is justified.

If every valid cell remains tonic/runaway/silent without a coherent boundary,
stop the phi-only line as

`NO_CARRIER_IN_REGISTERED_PHI_PANEL_AND_TESTED_STATES`.

Do not generalise this to all dynamic-threshold mechanisms.

## Task 3 — One bounded refinement round, only if earned

Open this task only if Task 2 finds a coherent boundary or carrier-like near
miss.  The decision and nearest parent cell must be written before each new run.

### Allowed neighbors

- tonic near active boundary → one stronger point with `f_phi=0.45`;
- silence next to an active phenotype → one weaker point with `f_phi=0.075`;
- burst-train time-scale boundary → one `tau_phi=40` or `240 ms` neighbor;
- total added combinations ≤4;
- no second refinement round.

If active cells are dominated specifically by whole-sheet synchrony, the only
backup is the predeclared GABA-decay diagnostic at the best phi cell:

- canonical `tau_d_GABA=18 ms`;
- diagnostic neighbors `12 ms` and `24 ms`;
- no E→E change and no broader inhibitory grid.

### Execution

```bash
python scripts/run_topic4_zm_fast_lifecycle_development.py \
  refine --decision-json <locked_decision.json> --confirm-run --workers auto
python scripts/analyze_topic4_zm_fast_lifecycle_development.py refine
```

### Selection

Promote at most two candidates, prioritising:

1. bounded non-tonic dynamics without refractory hard saturation;
2. spatial relay over common-mode synchrony;
3. sustained vSEEG occupancy/energy over isolated burst spikes;
4. distance from silence/runaway boundaries;
5. distinct phenotypes when two candidates survive.

Freeze only the selected candidates for Task 4.  Discovery neighbors cannot be
silently added to later confirmation panels.

## Task 4 — Run the end-to-end reachable lifecycle vertical slice

**This is the route-decision experiment.**  Run it before multi-seed, \(dt/2\),
real-data rebuilding or formal archive work.

### 4.1 Reachable trajectory

For each top candidate:

- [ ] start from the original interictal initial condition;
- [ ] enable \(\phi,z,m,S_G\) from \(t=0\) with fixed equations and parameters;
- [ ] do not load old fast states or initialise phi at an ictal checkpoint;
- [ ] run through the original escalation window and at least 10 s after the
  first sustained high-activity episode, capped at 30 s for development;
- [ ] keep rolling/full checkpoints sufficient to fork a matched reachable
  carrier state without changing the uncontrolled trajectory.

### 4.2 Baseline assessment

Hard failure only for:

- persistent silence;
- pre-carrier runaway or whole-sheet plateau;
- complete loss of returning events from both cores;
- loss of pathology-axis geometry or simultaneous whole-sheet flash.

Report event count, intervals, duration, amplitude, all-sheet rate, core balance
and phi carryover as continuous deviations.  Do not apply the old ±20% or 10%
phi-decay hard gates during development.

### 4.3 Identity and slow-flow assessment

- [ ] Determine whether the reachable trajectory enters the same phenotype
  family as the old-checkpoint discovery arm.
- [ ] Report \(z,m,\phi,S_G\) trajectories and derivatives through interictal,
  entry, maintenance and decline.
- [ ] An Arm-A candidate not reproduced here is
  `unreachable_frozen_phenotype`, irrespective of replication on old states.
- [ ] If no candidate episode occurs by 30 s, label it
  `no_reachable_entry_within_development_horizon`, not global unreachability.

### 4.4 Native and controlled exit branches

From identical reachable carrier checkpoints/RNG state:

**Native:** continue without intervention.

**Controlled:** apply a 50 ms E-threshold uplift with

`dose/gap={0,0.05,0.10,0.20}`

at the active pathological core and at all E cells.  Record return, exit,
permanent silence, rebound, recovery time and returning events.  This is a
developmental susceptibility map, not an efficacy claim.

For each site/dose, use replay plus the two existing resampled future-noise
continuations from the same reachable state.  Report descriptive fractions over
these three continuations; do not attach inferential confidence or call them
control-efficacy probabilities.

Do not require a candidate to return after the pulse.  Estimate/describe
`P(return|u)` and `P(exit|u)` across the paired forks.  A controllable
metastable episode is allowed to survive route selection.

### 4.5 Vertical-slice outcomes

Publish exactly one of:

- `unreachable_frozen_phenotype`;
- `no_reachable_entry_within_development_horizon`;
- `reachable_carrier_no_exit_route`;
- `reachable_native_offset_no_recovery`;
- `suppression_without_recovery`;
- `spontaneous_onset_with_controllable_termination_candidate`;
- `autonomous_lifecycle_candidate`;
- `lifecycle_compatible_candidate`.

The last three open Task 5.  A candidate with controlled exit but no native
offset is scientifically distinct from an autonomous lifecycle and must remain
labelled that way.

### Execution

```bash
python scripts/run_topic4_zm_fast_lifecycle_development.py \
  vertical-slice --candidate-json <selected_candidates.json> \
  --confirm-run --workers auto
python scripts/analyze_topic4_zm_fast_lifecycle_development.py vertical-slice
```

## Task 5 — Conditional locked confirmation

Do not implement or execute this task unless Task 4 opens it.

### Lock before confirmation

- [ ] candidate equations and parameter values;
- [ ] seeds 1/3/4 and future-noise panel;
- [ ] full observation and phenotype definitions;
- [ ] representative E1146 seizure-7 Fig3-A sidecar and provenance;
- [ ] native/control/sham/matched-energy conditions;
- [ ] ablations and \(dt/2\) cells;
- [ ] resource/coverage policy and claim vocabulary.

### Confirmation experiments

- [ ] reproduce the reachable carrier/lifecycle phenotype across seeds/noise;
- [ ] confirm at \(dt/2\);
- [ ] compare pre- and post-event interictal distributions on longer runs;
- [ ] run vSEEG real-data comparison;
- [ ] ablate phi, dynamic \(S_G\), and any selected fast-I backup;
- [ ] test native versus controlled exit with sham and matched energy;
- [ ] only now build the immutable manifest, formal verdict, archive and
  paper-facing diagnostic figure set.

The exact confirmatory thresholds must be written after Task 4 candidate
selection but before any validation seed/noise output is opened.  Discovery data
cannot count as confirmation.

## Commit checkpoints

1. `docs(topic4): revise fast-carrier route to development vertical slice`
2. `feat(topic4): run phi phenotype discovery on frozen tonic states`
3. conditional `feat(topic4): map reachable carrier exit susceptibility`
4. conditional `docs(topic4): lock lifecycle candidate confirmation`

Do not push or merge unless explicitly authorised.  A green Task-1 test suite is
engineering readiness; Task-2 and Task-4 results determine scientific progress.
