# Stage 3 Event-Triggered Axial Intervention — Implementation Plan

> File name keeps the earlier `deadzone` slug for continuity. The plan now targets an event-triggered axial intervention strategy, not a static wall result.

## Goal

Build a TDD implementation that answers the user's actual question:

> Can an intervention applied after a large Stage 3 interictal-like event has already started spreading along the propagation axis stop further axial propagation?

v1 uses an idealized **E-only dynamic threshold shutoff** as the controllable abstraction of a silencing stimulation. This is a feasibility/strategy probe, not a claim about real refractory physiology.

## Key Design Choice

v1 is **oracle replay-triggered**:

1. Run no-intervention baseline for a fixed seed.
2. Detect an eligible event that has source-core onset and would cross the midline.
3. Rerun the same seed/network/RNG.
4. At `source_onset + trigger_delay_ms`, dynamically clamp E cells in the axial target band for `duration_ms`.
5. Verify the replay is bit-identical to baseline before intervention onset.

This directly tests "event already started -> intervention stops spread" while avoiding a brittle online trigger state machine in v1. A true online detector is post-gate.

## Non-Negotiable Constraints

- Do not edit `src/snn_engine/*`.
- Do not edit `scripts/run_sef_hfo_snn_cm_spontaneous_readout.py`.
- Do not re-bless `engine_versions.json`.
- Do not use readout `sign` as source.
- v1 primary mode is E-only clamp, canonical montage, no dense montage, no E+I.
- Static dead-zone is only an upper-bound placement control.
- Stop after pilot JSON summary; no formal long run and no figure-as-evidence.

## Deliverables

- `src/sef_hfo_axial_intervention.py`
- `scripts/run_stage3_axial_intervention_probe.py`
- `scripts/summarize_stage3_axial_intervention_pilot.py`
- `tests/test_sef_hfo_axial_intervention.py`
- `tests/test_run_stage3_axial_intervention_probe.py`
- `tests/test_summarize_stage3_axial_intervention_pilot.py`
- Output root: `results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/axial_intervention_probe/`

---

## Task 1 — Pure Geometry And Source Helpers

### Purpose

Lock the geometry and source-label contracts before touching simulation.

### Tests To Write First

Create `tests/test_sef_hfo_axial_intervention.py`.

Required tests:

- `test_band_mask_perpendicular_to_axis`
  - Given x-axis propagation and center `(5, 5)`, a thickness-1 band includes contacts with `|x-5| <= 0.5`.
- `test_split_near_target_far_orients_by_source`
  - For source at +axis side, far side is negative-axis side; for source at -axis side, far side flips.
- `test_core_source_raw_independent_of_readability`
  - `core_source_raw(10, 60, 30) == "neg"`.
  - `core_source_raw(60, 10, 30) == "pos"`.
  - `core_source_raw(10, 25, 30) == "collision"`.
  - single-core onset gives that source.
  - both missing gives `"none"`.
- `test_participation_ratio_excludes_clamped_denominator`
  - With `valid=free`, clamped cells are removed from denominator.
- `test_exclude_target_contacts`
  - Contact validity mask removes target-band contacts without mutating input.

### Implementation

Create `src/sef_hfo_axial_intervention.py` with:

- `band_mask(coords, normal_unit, center_point, thickness)`
- `split_near_target_far(coords, axis_unit, center, source_focus, target_thickness)`
- `core_source_raw(on_neg, on_pos, delta_onset)`
- `participation_ratio(participate, region_mask, valid=None)`
- `exclude_target_contacts(valid, target_mask)`

### Verification

Run:

```bash
python -m pytest tests/test_sef_hfo_axial_intervention.py -q
```

Expected: all Task 1 tests pass.

---

## Task 2 — Target Masks And Dynamic Clamp Schedule

### Purpose

Represent the intervention target and its time window as pure data.

### Tests To Write First

Append tests:

- `test_dynamic_vth_E_only_only_inside_window`
  - E cells in target are clamped only for `on <= t < off`.
  - I cells are never clamped.
  - input `vth` is not mutated.
- `test_dynamic_vth_off_window_identity`
  - before and after the intervention window, output equals input exactly.
- `test_count_matched_off_axis_target_avoids_cores`
  - off-axis target clamps the same number of E cells as on-axis target.
  - if candidate target overlaps either focus core, raise `ValueError`.
- `test_static_target_is_upper_bound`
  - static target is equivalent to dynamic clamp with `on=0` and `off=inf`.

### Implementation

Add helpers:

- `CLAMP_LEVEL = 1e6`
- `intervention_vth_at_time(base_vth, target_mask, is_E, t_ms, on_ms, off_ms, clamp_level=CLAMP_LEVEL)`
- `make_on_axis_target(pos, is_E, axis_unit, center, thickness)`
- `make_off_axis_target(pos, is_E, axis_unit, center, thickness, n_match, core_masks, rng, L, mode="lateral")`
- `make_static_deadzone_schedule()`

### Verification

Run:

```bash
python -m pytest tests/test_sef_hfo_axial_intervention.py -q
```

Expected: all Task 1-2 tests pass.

---

## Task 3 — Baseline Event Extraction And Replay Schedule

### Purpose

Find events that are worth intervening on, then build an oracle replay schedule.

### Tests To Write First

Append tests:

- `test_baseline_eligibility_requires_cross_midline_opportunities`
  - eligible only if `n_returned >= 20`, `n_neg >= 3`, `n_pos >= 3`, `n_cross_midline >= 5`, and `n_trigger_opportunity >= 5`.
- `test_select_first_eligible_event_prefers_single_source_cross_midline`
  - skips `collision`, `none`, and non-crossing events.
  - returns first event with `core_source_raw in {neg,pos}` and `oracle_far_ratio > 0.05`.
- `test_build_replay_schedule_starts_after_source_onset`
  - schedule `on = source_onset + trigger_delay_ms`.
  - schedule `off = on + duration_ms`.
  - rejects schedules where `on >= far_onset_time` unless `allow_late=True`.
- `test_late_schedule_marks_late_control`
  - late control uses `far_onset_time + late_delay_ms`.

### Implementation

Add helpers:

- `baseline_eligibility(summary, min_events=20, min_per_end=3, min_cross_midline=5, min_trigger_opportunity=5, cross_midline_frac=0.05)`
- `select_first_eligible_event(events, cross_midline_frac=0.05)`
- `build_replay_schedule(event, trigger_delay_ms=8.0, duration_ms=40.0, allow_late=False)`
- `build_late_schedule(event, late_delay_ms=8.0, duration_ms=40.0)`

Event dicts must include:

- `event_id`
- `core_source_raw`
- `core_onset_neg`
- `core_onset_pos`
- `source_onset`
- `far_onset_time`
- `oracle_far_ratio`

### Verification

Run:

```bash
python -m pytest tests/test_sef_hfo_axial_intervention.py -q
```

Expected: all helper tests pass.

---

## Task 4 — Runner-Local Simulation Adapter With Dynamic Vth

### Purpose

Copy the existing `simulate_kick` integration loop into the new runner/module and add only one behavior: time-dependent `V_th_per_neuron`.

### Tests To Write First

Create `tests/test_run_stage3_axial_intervention_probe.py`.

Required tests:

- `test_dynamic_adapter_no_intervention_matches_static_vth`
  - On a tiny simulation, dynamic adapter with no schedule equals canonical `simulate_kick(..., V_th_per_neuron=base_vth)` for `E_spk_bool` and `rate_E`.
- `test_dynamic_adapter_pre_intervention_parity`
  - With an intervention scheduled at 100 ms, outputs are identical to no-intervention through the step before 100 ms.
- `test_dynamic_adapter_clamps_after_onset`
  - A target cell's effective threshold is high during the intervention window. This can be tested by returning `intervention_active` or `n_target_clamped_by_step`.

### Implementation

Place in `src/sef_hfo_axial_intervention.py` or directly in the runner with a private name:

- `simulate_dynamic_vth(...)`

Implementation rule:

- Start from `src/snn_engine/kick_probe.py::simulate_kick`.
- Keep recurrent dynamics, RNG path, LFP recorder, and recorders identical.
- At each time step, set `V_th_eff = intervention_vth_at_time(base_vth, target_mask, is_E, tm, on_ms, off_ms)` when schedule exists.
- If no schedule exists, the function must be parity-equivalent to canonical `simulate_kick`.

### Verification

Run:

```bash
python -m pytest tests/test_sef_hfo_axial_intervention.py tests/test_run_stage3_axial_intervention_probe.py -q
```

Expected: dynamic adapter tests pass. If no-intervention parity fails, stop and fix before continuing.

---

## Task 5 — Probe Runner And JSON Schema

### Purpose

Build the user-facing runner that can run baseline, choose event, replay intervention, and emit event-level metrics.

### Tests To Write First

Add subprocess smoke tests:

- `test_runner_baseline_smoke_writes_schema`
  - Run tiny `--arm baseline`.
  - JSON has `arm`, `events`, `n_returned`, `baseline_eligibility`.
- `test_runner_dynamic_on_axis_smoke_writes_intervention_fields`
  - Run tiny `--arm dynamic_on_axis --schedule-json <toy_schedule>`.
  - Run tiny `--arm dynamic_on_axis --baseline-json <baseline_summary>` and verify the runner can derive the schedule from the selected baseline event.
  - JSON has `intervention_on`, `intervention_off`, `pre_intervention_parity`.
- `test_runner_rejects_dynamic_without_schedule_or_baseline_event`
  - Dynamic arm without a schedule or selected event exits nonzero with clear message.

### Implementation

Create `scripts/run_stage3_axial_intervention_probe.py`.

CLI:

```text
--arm {baseline,static_deadzone,dynamic_on_axis,dynamic_off_axis,late_on_axis,wall_only}
--seed
--T
--core-mean
--core-std
--sep-frac
--drive
--target-thickness
--trigger-delay-ms
--duration-ms
--schedule-json
--baseline-json
--tag
--out
```

Runner responsibilities:

1. Import canonical spontaneous runner via importlib.
2. Build the same network, foci, `base_vth`, montage, and LFP recorder.
3. For `baseline`, run no intervention and compute per-event metrics.
4. For dynamic arms, load a baseline-selected event schedule or create it from a passed baseline JSON.
5. Rerun same seed/network/RNG with dynamic schedule.
6. Compute per-event metrics:
   - `core_source_raw`
   - `source_onset`
   - `far_onset_time`
   - `oracle_far_ratio`
   - `oracle_near_ratio`
   - `oracle_reach_mm`
   - `instr_far_ratio`
   - `instr_far_ratio_excl_target_contacts`
   - `trigger_status`
7. Emit JSON only. Do not generate figures.

JSON summary must include:

- `arm`
- `tag`
- `config`
- `schedule`
- `selected_baseline_event`
- `baseline_json`
- `pre_intervention_parity`
- `n_returned`
- `n_neg`
- `n_pos`
- `n_collision`
- `n_none`
- `collision_rate`
- `events`

### Verification

Run:

```bash
python -m pytest tests/test_sef_hfo_axial_intervention.py tests/test_run_stage3_axial_intervention_probe.py -q
```

Expected: all tests pass.

---

## Task 6 — Pilot Driver And Hard Stop Summary

### Purpose

Run the minimum real-scale pilot and stop with a concise JSON/table summary for user review.

### Tests / Checks First

Create `tests/test_summarize_stage3_axial_intervention_pilot.py`.

Required tests:

- `test_summary_groups_by_arm_and_seed`
  - Given toy per-arm JSON files, summary CSV/JSON contains one row per `arm,seed`.
- `test_summary_preserves_fail_guard_fields`
  - Summary includes `pre_intervention_parity`, `collision_rate`, `selected_event_id`, and `selected_source`.
- `test_summary_uses_excluded_target_contact_metric`
  - Summary reports `instr_far_ratio_excl_target_contacts`; it must not silently substitute raw `instr_far_ratio`.

Runtime checks:

- all tests green
- baseline eligibility checked before dynamic arms
- dynamic replay reports `pre_intervention_parity=True`
- no result files committed

### Commands

1. Full tests:

```bash
python -m pytest \
  tests/test_sef_hfo_axial_intervention.py \
  tests/test_run_stage3_axial_intervention_probe.py \
  tests/test_summarize_stage3_axial_intervention_pilot.py \
  -q
```

2. Baseline smoke at predeclared hot point:

```bash
python scripts/run_stage3_axial_intervention_probe.py \
  --arm baseline --seed 1 --T 3000 \
  --core-mean 17.0 --core-std 1.5 --sep-frac 0.6 \
  --tag baseline_m17p0_sep0p6_s1
```

Expected:

- JSON written under `results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/axial_intervention_probe/`
- `baseline_eligibility.eligible` true

If false, only try this fallback order:

```bash
# fallback 1
python scripts/run_stage3_axial_intervention_probe.py \
  --arm baseline --seed 1 --T 3000 \
  --core-mean 16.5 --core-std 1.5 --sep-frac 0.6 \
  --tag baseline_m16p5_sep0p6_s1

# fallback 2, only if fallback 1 ineligible
python scripts/run_stage3_axial_intervention_probe.py \
  --arm baseline --seed 1 --T 3000 \
  --core-mean 16.5 --core-std 1.5 --sep-frac 0.5 \
  --tag baseline_m16p5_sep0p5_s1
```

If still ineligible, stop and report `baseline_ineligible`.

3. If eligible, build schedule from selected baseline event and run v1 arms for seeds 1-3:

```bash
for s in 1 2 3; do
  python scripts/run_stage3_axial_intervention_probe.py --arm baseline --seed $s --T 3000 \
    --core-mean 17.0 --core-std 1.5 --sep-frac 0.6 --tag baseline_s${s}

  python scripts/run_stage3_axial_intervention_probe.py --arm dynamic_on_axis --seed $s --T 3000 \
    --core-mean 17.0 --core-std 1.5 --sep-frac 0.6 --tag dynamic_on_axis_s${s} \
    --baseline-json results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/axial_intervention_probe/baseline_s${s}.json

  python scripts/run_stage3_axial_intervention_probe.py --arm dynamic_off_axis --seed $s --T 3000 \
    --core-mean 17.0 --core-std 1.5 --sep-frac 0.6 --tag dynamic_off_axis_s${s} \
    --baseline-json results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/axial_intervention_probe/baseline_s${s}.json

  python scripts/run_stage3_axial_intervention_probe.py --arm late_on_axis --seed $s --T 3000 \
    --core-mean 17.0 --core-std 1.5 --sep-frac 0.6 --tag late_on_axis_s${s} \
    --baseline-json results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/axial_intervention_probe/baseline_s${s}.json

  python scripts/run_stage3_axial_intervention_probe.py --arm static_deadzone --seed $s --T 3000 \
    --core-mean 17.0 --core-std 1.5 --sep-frac 0.6 --tag static_deadzone_s${s}

  python scripts/run_stage3_axial_intervention_probe.py --arm wall_only --seed $s --T 3000 \
    --core-mean 17.0 --core-std 1.5 --sep-frac 0.6 --tag wall_only_s${s}
done
```

4. Run the summary helper. It writes a small summary JSON/CSV, not a figure:

```bash
python scripts/summarize_stage3_axial_intervention_pilot.py \
  --input-dir results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/axial_intervention_probe/ \
  --out-prefix results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/axial_intervention_probe/pilot_summary
```

Summary fields by arm and seed:

- median `oracle_far_ratio`
- median `oracle_reach_mm`
- median `instr_far_ratio_excl_target_contacts`
- `n_neg`, `n_pos`, `n_collision`
- `collision_rate`
- `pre_intervention_parity`
- selected event id and source

### Hard Stop

After Task 6, stop. The final report to the user should say only:

- whether implementation/tests passed
- whether baseline was eligible
- whether dynamic on-axis directionally reduced far spread compared with baseline/off-axis/late
- whether any FAIL guard fired

Do not:

- run longer simulations
- run wall-thickness sweeps
- add E+I
- add dense montage
- write a paper-style conclusion
- update topic docs

---

## Agent Execution Notes

- Use a clean branch or worktree. The current `topic4-event-extent-audit` worktree has unrelated dirty/untracked files.
- Commit spec/plan first if requested, then commit each task after tests pass.
- If a test fails because the plan's pasted code is stale relative to actual APIs, update the implementation and the test together, but preserve the scientific contract above.
- If no-intervention parity fails, stop immediately. Do not continue to pilot.
- If baseline is ineligible after the two predeclared fallbacks, stop and report; do not tune further.
