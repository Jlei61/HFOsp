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

For M3B-R2 handoff, A2 must not emit only raw slow traces. It must emit a **phase-map trajectory**
computed with the A1 `slow_to_rate_mapping.json`. Without that mapping, M3B can inspect A2
phenotypes but cannot claim the slow state moved through the spectral phase map.

## 1. Scientific Goal

Test whether the SNN can endogenously move through states resembling:

```text
resting / sparse IID-like events -> pre-ictal susceptibility -> R4a sustained recruitment
```

The plan does not require a full clinical seizure reproduction. It only asks whether biologically motivated slow variables create distinguishable spontaneous event regimes.

After the M3B-R2 phase-map redesign, A2 has two nested gates:

- **Gate A: trajectory gate for M3B-R2 overlay.** The slow state must move through calibrated
  `(phase_x_core, phase_y_global, phase_recovery)` coordinates and show source-space phenotype
  movement beyond event-rate-only heating. This can pass even if the event remains axial, as long
  as the trajectory and phenotype shift are real, rate-matched, and self-limiting.
- **Gate B: seizure-like phenotype gate for an M3A mechanism claim.** This is stricter: global or
  off-axis recruitment, low tonic fraction, sustained/recurrent recruitment, and return to baseline.

Do not let Gate B block a valid Gate A handoff. Conversely, do not promote a Gate A trajectory into a
seizure-like mechanism claim.

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

Required M3B-R2 phase-map fields:

```text
slow_to_rate_mapping_id
phase_x_core(t)
phase_y_global(t)
phase_recovery(t)
phase_coord_valid(t)
phase_coord_out_of_range(t)
phase_coord_source
rho_model_coord(t) optional
```

`rho_model_coord(t)` may reuse the compact coordinate
`lgr / (q_core q_global)` when q-resources are the active mechanism, but it is a diagnostic
coordinate. The handoff to M3B is the explicit `(phase_x_core, phase_y_global, phase_recovery)`
trajectory plus validity/range flags.

Minimum dynamic artifact schemas:

`phase_trajectory.csv`

```text
time_ms,event_id,event_stage,
q_core_L,q_core_R,q_core_mean,q_core_min,q_global,
gK_core_L,gK_core_R,gK_surround,
phi_core_L,phi_core_R,phi_surround,
rho_source,rho_mean,rho_min,
phase_x_core,phase_y_global,phase_recovery,
phase_coord_valid,phase_coord_out_of_range,phase_coord_source,
global_E_rate,core_L_rate,core_R_rate,surround_rate,active_E_fraction
```

`event_phase_samples.csv`

```text
event_id,t_onset,t_peak,t_end,event_stage,source_core,
phase_x_core,phase_y_global,phase_recovery,
phase_coord_valid,phase_coord_out_of_range,
R_class,returned,tail_to_baseline_ratio,tonic_fraction,duration_ms,
n_fired_E,r95_mm,reach_axis,reach_perp,isotropy,grad_r2,grad_align,
off_axis_score,collision,dual_core_onset_lag_ms,peak_lag_span_ms,
rate_matched_group,phenotype_label
```

`dynamic_slowvars_summary.json`

```json
{
  "mapping_id": "m3a_a1_<date>_<hash>",
  "gate_A_trajectory": "PASS|FAIL|INCONCLUSIVE",
  "gate_B_seizure_like": "PASS|FAIL|INCONCLUSIVE",
  "trajectory_robustness": "robust|seed_fragile|runaway_prone|quiet_prone|not_tested",
  "rate_matched_control": "passed|failed|not_run",
  "out_of_range_fraction": 0.0,
  "forbidden_claims": []
}
```

If a field is unavailable because a mechanism is not enabled, write `NA`; if it is unknown because
the runner failed to measure it, fail the export rather than writing 0.

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
- [ ] Load the A1 `slow_to_rate_mapping.json` and attach its `mapping_id` to every run.

### Task 1b: Phase-map coordinate exporter

- [ ] Convert dynamic slow traces into `phase_x_core(t)`, `phase_y_global(t)`, and
  `phase_recovery(t)` using the A1 mapping.
- [ ] Emit range flags relative to `phase_coord_ranges.json`.
- [ ] Refuse M3B-ready export if the mapping is missing, uncalibrated, or sign tests failed.
- [ ] Add tests:
  - `test_phase_export_refuses_missing_mapping`
  - `test_phase_export_flags_out_of_range_samples`
  - `test_phase_export_preserves_event_stage_samples`
  - `test_q_resource_trace_maps_to_expected_rho_direction`
  - `test_phase_export_fails_closed_on_uncalibrated_mapping`
  - `test_phase_export_writes_NA_not_zero_for_disabled_mechanisms`

### Task 1c: Interface artifact TDD before parameter sweeps

- [ ] Implement schema validators for `slow_to_rate_mapping.json`, `phase_coord_ranges.json`,
  `phase_trajectory.csv`, `event_phase_samples.csv`, and `dynamic_slowvars_summary.json`.
- [ ] Add tests:
  - `test_dynamic_summary_has_gate_A_and_gate_B_verdicts`
  - `test_event_phase_samples_have_pre_onset_peak_end_post`
  - `test_tail_to_baseline_is_absolute_not_relative`
  - `test_rate_matched_group_is_present_when_gate_A_claimed`
  - `test_phase_trajectory_contains_per_core_q_when_two_core_substrate`
  - `test_m3b_ready_flag_requires_mapping_and_phenotype_movement`
  - `test_phenotype_positive_without_mapping_is_mechanism_candidate_only`

No broad dynamic sweep is allowed until Task 1c passes. This is the lock that prevents M3B from
borrowing an uncalibrated slow trace as a spectral phase-map trajectory.

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
7. Is there a valid phase-map trajectory export for M3B-R2?

Verdict split:

- **Gate A PASS** requires: calibrated phase trajectory; source-space phenotype movement across
  low/high slow-state windows; rate-matched spatial/globality difference; absolute return; multiple
  seeds with consistent direction; no R4b-only tonic saturation.
- **Gate B PASS** additionally requires: global/off-axis or sustained recruitment phenotype,
  discrete self-limiting bouts, and R4a-like structure rather than only axial R3 waves or R4b runaway.

## 5. Outputs

Canonical output root:

```text
results/topic4_sef_hfo/m3a_slowvars/dynamic/
```

Required docs:

- `STATUS.md`: seven-question verdict.
- `dynamic_slowvars_summary.json`.
- `per_event.csv`.
- `event_table.csv`.
- `slow_state_trace_summary.csv`.
- `slow_state_trace.parquet` if dependencies are available; otherwise `slow_state_trace.csv`.
- `phase_trajectory.csv`.
- `event_phase_samples.csv`.
- `freeze_samples.jsonl`.
- `phenotype_summary.json`.
- `slow_to_rate_mapping.json` or a pointer to the A1 mapping artifact.
- `figures/README.md`.
- archive recap: `docs/archive/topic4/sef_hfo/m3a_dynamic_slowvars_recap_<date>.md`.

`freeze_samples.jsonl` contains event-aligned frozen points for M3B spot checks:

```json
{
  "sample_id": "...",
  "event_id": "...",
  "stage": "pre|onset|peak|end|post|baseline",
  "q_core_L": "NA",
  "q_core_R": "NA",
  "q_global": "NA",
  "gK_core": "NA",
  "phi_core": "NA",
  "phase_x_core": 0.0,
  "phase_y_global": 0.0,
  "phase_recovery": 0.0,
  "phase_coord_valid": true,
  "phenotype_label": "local_axial|larger_axial|mixed_global|global_recruitment|runaway|recovery",
  "operating_point_source": "SNN_pre_event_baseline_average"
}
```

## 6. Handoff to M3B

> **2026-06-27 — CANONICAL CONTRACT SUPERSEDES THE HANDOFF DETAIL.** See
> `docs/superpowers/specs/2026-06-27-sef-hfo-m3-interface-contract.md` §5/§6 (mirror
> `src/sef_hfo_m3_interface.py`, TDD `tests/test_sef_hfo_m3_interface.py`). A2 corrections:
> - every phase-coord-bearing artifact (`phase_trajectory.csv`, `event_phase_samples.csv`,
>   `freeze_samples.jsonl`) carries `slow_to_rate_mapping_id`, all three phase coords,
>   `phase_coord_valid`, `phase_coord_out_of_range` (REQUIRED — absence is an export failure, never
>   silently in-range), and a canonical `event_stage`;
> - `event_phase_samples.csv` uses `return_to_baseline` (rename from `returned`) and carries
>   `R_class`; `freeze_samples.jsonl` also carries `R_class`;
> - `event_stage` ∈ the one canonical enum {baseline, pre, onset, peak, end, post_50ms, post_200ms,
>   post_1s, post, inter_event};
> - `dynamic_slowvars_summary.json` carries `m3b_ready` + `m3b_ready_reason`
>   (= gate_A PASS AND all on-axis coords calibrated AND rate_matched_control passed) and gate fields
>   validated against their enums; `gate_A_trajectory == PASS` requires `rate_matched_control ==
>   passed` AND `rate_matched_group` recorded;
> - a disabled mechanism writes `NA` (never `0.0`), and a derived coord whose contributors are all
>   disabled is `NA`;
> - `rho_model_coord = lgr/(q_core q_global)` is an OPTIONAL A2-local diagnostic named `rho_resource`
>   (not the required handoff, and not M3B's spectral `rho(M)`); "lgr" is UNDEFINED and must be
>   expanded or dropped before any claim — see contract §9 open items;
> - `phase_x_core`/`phase_y_global` are the Gate-A overlay coordinates; `phase_recovery` is carried
>   losslessly but projected out of the 2-axis map (recorded in the audit). The overlay is Gate-A
>   tier only; a Gate-B seizure-like claim requires `gate_B_seizure_like == PASS`.

M3B can consume A2 only if A2 provides:

- a slow-state scalar or vector `s_slow(t)`;
- a calibrated `slow_to_rate_mapping.json`;
- a phase trajectory with `phase_x_core`, `phase_y_global`, `phase_recovery`, and range flags;
- per-event state labels or state bins;
- event classes R2/R3/R4a/R4b;
- pre-event and post-event state samples;
- a verdict on whether slow state changes phenotype.

If A2 is phenotype-positive but lacks a calibrated phase trajectory, M3B may call it an M3A
mechanism candidate but must not overlay it on the spectral map. If A2 is negative, M3B may still
define the frozen phase map and data bridge, but must not draw a slow-state phase-map claim.

## 7. Follow-on Execution Ladder After the 2026-06-27 Interface Update

This replaces the older single-goal framing where `off_axis_SELF_LIMITING` was the only useful next
target.

1. **T0 status reset**: write current M3A status as screen-level only. A1 gives state topography;
   A2 single `q` depletion is stay/runaway; `q+g_K` is a seed-fragile slow-fast candidate; current
   source-space waves are mostly axial recruitment amplification, not the defined global seizure state.
2. **T1 schema first**: implement and test the mapping/trajectory artifacts before more sweeps.
3. **T2 L1 two-tank / axis-break sweep**: engage `q_global`; classify `axial_only`,
   `off_axis_oneshot`, `off_axis_TONIC`, `off_axis_SELF_LIMITING`, `quiet_or_tiny`.
4. **T3 L2 recovery tuning**: sweep `g_K`, `tau_K`, `tau_rec_global`, `q_min`, and `k_use` only after
   L1 shows whether the gap is axis-break or termination.
5. **T4 frozen phenotype plane**: freeze trajectory points in `(phase_x_core, phase_y_global,
   phase_recovery)` and color by measured SNN phenotype, not by `rho` alone.
6. **T5 rate-matched propagation/globality gate**: compare low/high slow-state windows at matched peak
   rate or matched fired mass.
7. **T6 robustness**: 3-seed pilot, 5-seed confirmation for candidates, and `T >= 30-50s` or at least
   `8-10 * max(tau_K, tau_rec)`.
8. **T7 L3 only after L1/L2 exhaustion**: add E->E short-term depression only if the system can become
   global/off-axis but cannot self-terminate. This needs engine TDD and re-bless before any sweep.
9. **T8 M3B handoff**: emit `phase_trajectory.csv`, `event_phase_samples.csv`,
   `freeze_samples.jsonl`, and `dynamic_slowvars_summary.json`. Without these, M3B-R2 must not draw a
   slow-state trajectory overlay.
