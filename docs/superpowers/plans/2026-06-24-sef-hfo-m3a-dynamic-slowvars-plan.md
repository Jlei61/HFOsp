# M3A-A2 dynamic slow-variable mechanism plan

> Status: new hard-boundary plan, 2026-06-24.
> Scope: M3A only. This plan tests whether endogenous slow-variable dynamics can make the SNN spontaneously transition between interictal-like and seizure-like discharge phenotypes.
> Dependency: run after M3A-A1 quasi-static slow-state scan.

## 0. Hard Boundary

M3A-A2 answers:

```text
activity history -> slow-variable trajectory -> spontaneous phenotype transition
```

It does not test whether W is a field readout. It does not use `h(W)` to set threshold. W-derived load is deferred to M3B after M3A has a real slow-state variable.

Primary evidence remains no-kick spontaneous activity. Kick basin is secondary and cannot define seizure onset.

## 1. Scientific Goal

Test whether the SNN can endogenously move through states resembling:

```text
resting / sparse IID-like events -> pre-ictal susceptibility -> R4a sustained recruitment
```

The plan does not require a full clinical seizure reproduction. It only asks whether biologically motivated slow variables create distinguishable spontaneous event regimes.

## 2. Starting Mechanisms

Use A1 to choose the candidate range. If A1 is not available yet, default priority is:

1. chloride / GABA reversal axis: dynamic `e_GABA` or `z` disinhibition;
2. `phi` adaptive threshold as a fast self-limiting or refractoriness term;
3. `g_K` sAHP as slow outward protection / termination.

Do not run all combinations first. Start with single-variable dynamics, then add two-variable combinations only when the single-variable failure mode is clear.

## 3. Dynamic State Contract

Every dynamic run must emit slow-state traces. Required state samples:

```text
s(t) full trace or binned trace
s_pre_event
s_onset
s_peak
s_end
s_post_50ms
s_post_200ms
s_post_1s
```

Required event-history fields:

```text
event_count_so_far
time_since_previous_event
cumulative_active_mass_tau_1s
cumulative_active_mass_tau_5s
pre_event_state_slope
post_event_state_delta
```

If a state is not implemented, write `NA`, not 0.

## 4. Tasks

### Task 0: Import A1 candidate and failure boundary

- [ ] Read the A1 recap.
- [ ] Pick one primary slow variable and one negative/control variable.
- [ ] Freeze the candidate parameter range before long runs.
- [ ] Write the expected failure mode in `STATUS.md`.

### Task 1: Dynamic slow-variable recorder

- [ ] Add recorder support for `z`, `phi`, `g_K`, and `e_GABA` where available.
- [ ] Keep recorder off or cheap by default.
- [ ] Add tests that recorder output aligns with simulation time and event times.
- [ ] Add provenance fields for initial state, time constants, amplitudes, and update equations.

### Task 2: Single-variable dynamic smoke

For each candidate, run a tiny no-kick spontaneous smoke:

```text
T: 8-20 s
seeds: 3
one variable on
other slow variables off
```

Readouts:

- event rate;
- event size/duration;
- return probability;
- R2/R3/R4a/R4b fractions;
- pre-event slow state vs event class;
- post-event slow-state change.

Pass condition for expansion: any monotone or threshold-like relation between slow state and phenotype beyond event rate alone.

### Task 3: History and accumulation test

This is the core A2 question: do repeated interictal-like events move the slow state toward a more seizure-prone regime?

Required analyses:

- event index vs slow-state level;
- cumulative active mass vs subsequent slow-state change;
- time-since-last-event vs recovery of slow state;
- early events vs late events in the same simulation;
- shuffled event-time control.

Success pattern:

```text
repeated returned events
  -> slow state drifts or accumulates
  -> later events have larger size/duration or lower return probability
  -> R4a probability increases
```

Rate-only changes without state accumulation are not enough.

### Task 4: Dynamic parameter sweep

Only after Task 2/3 shows a candidate:

```text
time constant: low / nominal / high
amplitude: low / nominal / high
clearance/recovery: low / nominal / high
seeds: >= 5
T: enough to see several events or a documented silent failure
```

Do not tune detector thresholds after seeing the outcome.

### Task 5: R4a verification

Every candidate R4 event must be split:

- R4a: sustained/recurrent recruitment with spatial front or structured propagation;
- R4b: tonic saturation / full-field runaway.

Only R4a can support the M3A bridge. R4b is a failure or toxicity mode.

Required figure set:

- representative R4a event raster / active mass / slow-state trace;
- representative R4b tonic event if present;
- state-vs-phenotype scatter;
- event-history accumulation plot.

### Task 6: A2 verdict

Answer:

1. Does the slow state move before phenotype changes?
2. Do repeated returned events predict slow-state accumulation?
3. Does slow-state accumulation predict larger/longer/lower-return events?
4. Does R4a appear, or only R4b?
5. Is the effect stronger than shuffled event-history controls?
6. Is this worth passing to M3B for `W_eff(s)` analysis?

## 5. Outputs

Canonical output root:

```text
results/topic4_sef_hfo/m3a_slowvars/dynamic/
```

Required docs:

- `STATUS.md`: six-question verdict.
- `dynamic_slowvars_summary.json`.
- `per_event.csv`.
- `slow_state_trace_summary.csv`.
- `figures/README.md`.
- archive recap: `docs/archive/topic4/sef_hfo/m3a_dynamic_slowvars_recap_<date>.md`.

## 6. Handoff to M3B

M3B can consume A2 only if A2 provides:

- a slow-state scalar or vector `s_slow(t)`;
- per-event state labels or state bins;
- event classes R2/R3/R4a/R4b;
- pre-event and post-event state samples;
- a verdict on whether slow state changes phenotype.

If A2 is negative, M3B may still define W and data bridge, but must not draw a slow-state phase-map claim.

