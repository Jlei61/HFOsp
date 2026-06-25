# Stage 3 Event-Triggered Axial Intervention — Design Spec

- Date: 2026-06-26
- Status: design locked for implementation planning
- Supersedes: the earlier static `dead-zone / excitability-clamp barrier` framing
- Upstream: Stage 3 `twoend_equal` cm-SNN spontaneous dual-focus runs

## Plain Goal

The scientific question is now intervention, not static placement:

> After a large interictal-like event has already started propagating along the Stage 3 axis, can an event-triggered intervention on the propagation corridor stop further axial spread?

The first implementation is not a biophysical stimulation proof. The current engine has no synaptic depletion and no calibrated spike-frequency adaptation. Therefore v1 uses an idealized **event-triggered E-only threshold shutoff** as the controllable abstraction of a silencing stimulation. It tests whether the propagation axis has an intervention point where shutting down excitable E cells after onset can stop far-side recruitment.

Static dead-zone remains only as an **upper-bound placement control**: if an always-on inexcitable band cannot block spread, no triggered stimulation strategy at that target is worth pursuing.

## Scope

### In Scope

- New helper module: `src/sef_hfo_axial_intervention.py`.
- New runner: `scripts/run_stage3_axial_intervention_probe.py`.
- Reuse canonical Stage 3 construction by importlib:
  - `build_lesion_vth`
  - `montage`
  - `read_event`
  - `active_fraction`
  - `snn_event_envelope`
  - `build_sidecar`
- Add a runner-local simulation adapter copied from the existing `simulate_kick` loop, with exactly one functional addition: a time-dependent per-neuron E-threshold intervention schedule.
- Primary v1 mode: `dynamic_clamp`, E-only, canonical montage.
- Controls:
  - `baseline`: no intervention.
  - `static_deadzone`: always-on E-only axial band; placement upper bound.
  - `dynamic_on_axis`: event-triggered E-only shutoff at the axial target.
  - `dynamic_off_axis`: same number of E cells shut off, but away from the propagation corridor.
  - `late_on_axis`: same on-axis shutoff but delayed until after far-side recruitment should have happened.
  - `wall_only`: no dual low-threshold foci, intervention target present; artifact sanity.

### Out of Scope

- No `src/snn_engine/*` edits.
- No `engine_versions.json` re-bless.
- No E+I clamp in v1.
- No dense montage in v1.
- No wall-thickness sweep in v1.
- No claim that real stimulation induces refractory/depletion.
- No formal long run or figure as evidence before pilot review.

## Key Contracts

### 1. Source Labels

Use `core_source_raw`, not readout sign.

- `core_source_raw = neg | pos | collision | none`.
- It depends only on the two focus-core onset times.
- It must not require `n_part >= 7` or finite `axis_err`.
- Existing Stage 3 `hidden_source_label` remains compatibility output only.

Reason: a successful intervention may deliberately make an event unreadable globally. If source labeling depends on global readability, successful stops disappear into `ambiguous`.

### 2. Trigger Logic

The intervention is event-triggered.

For each event, source side is determined by `core_source_raw`. A target band is placed on the propagation axis between source and far side. The trigger should fire only after the source-side event is real but before far-side recruitment is already established.

**v1 trigger mode is oracle replay-triggered, not fully online closed-loop.**

Procedure:

1. Run the no-intervention baseline with a fixed seed.
2. Identify eligible single-source cross-midline events and their source-core onset times.
3. Rerun the same seed/network/RNG with an intervention schedule at `source_onset + trigger_delay_ms` for the selected event.
4. Verify pre-intervention parity against the baseline trace up to `intervention_on`.

This still tests the requested causal question: the event has already started before intervention begins. It avoids prematurely building an online trigger state machine before the feasibility of axial shutoff is known. A true online detector is post-gate.

Post-gate trigger options:

- `source_core`: trigger at source-core onset plus `trigger_delay_ms`.
- `trigger_band`: trigger when an axial trigger band reaches `trigger_frac` active E participation.

Default v1: oracle `source_core + trigger_delay_ms=8 ms`, because it is deterministic and easy to parity test. `trigger_band` is implemented/tested only if time permits; it is not required for v1 smoke.

Every event must report:

- `trigger_time`
- `intervention_on`
- `intervention_off`
- `trigger_status = fired | no_source | collision | late | no_trigger`

### 3. Intervention

Primary intervention is an idealized E-only shutoff:

- `dynamic_clamp`: during `[intervention_on, intervention_off)`, E cells inside the intervention target have `V_th = CLAMP_LEVEL`.
- I cells are unchanged.
- Outside the intervention window, the threshold field equals the baseline dual-focus `vth`.

This is a controllable abstraction of a silencing strategy, not a model of electrical stimulation physics.

### 4. Target And Controls

The primary target is an on-axis band between the foci, centered near the midline unless explicitly shifted.

Controls:

- `dynamic_off_axis` must clamp the same number of E cells as `dynamic_on_axis`, must not overlap either focus core, and must not lie on the source-to-sink corridor.
- `late_on_axis` tests timing specificity. If late intervention works as well as timely intervention, the result is likely global suppression or measurement artifact, not propagation stopping.
- `static_deadzone` is the upper-bound placement control.
- `wall_only` should produce no event source and no readable propagation.

### 5. Metrics

Primary metrics are event-level and source-stratified:

- `oracle_far_ratio`: far-side E participation excluding clamped cells from denominator.
- `oracle_reach_mm`: maximum axial reach beyond the source-side focus.
- `far_onset_time`: first far-side activation time, if any.
- `instr_far_ratio`: canonical montage far-side contact participation.
- `instr_far_ratio_excl_target_contacts`: same after excluding contacts inside the intervention target.

Secondary metrics:

- `oracle_near_ratio`
- event `duration`
- `collision_rate`
- `n_neg`, `n_pos`, `n_collision`, `n_none`
- trigger opportunity counts

### 6. Eligibility Gates

Before comparing dynamic interventions, the baseline must have something to stop:

- `n_returned >= 20`
- `n_neg >= 3`
- `n_pos >= 3`
- `n_cross_midline >= 5`, where cross-midline means a single-source event with `oracle_far_ratio > 0.05`.
- `n_trigger_opportunity >= 5`, where the event has source onset and would trigger before far-side onset or before far-side recruitment crosses `0.05`.

If baseline is ineligible, stop and report the reason. A hotter fallback may be tested only in the pre-declared order:

1. `core_mean=16.5`, same `sep_frac=0.6`, same `drive=0.6`.
2. If still ineligible, `core_mean=16.5`, `sep_frac=0.5`, same `drive=0.6`.
3. If still ineligible, stop. Do not tune further without user review.

### 7. Pilot Verdict

Pilot PASS is not a scientific claim. It only means the strategy is worth a formal follow-up.

Pilot directionally supports the intervention if:

- `dynamic_on_axis` lowers median `oracle_far_ratio` versus baseline and `dynamic_off_axis`.
- `dynamic_on_axis` lowers median `oracle_reach_mm` versus baseline and `dynamic_off_axis`.
- `dynamic_on_axis` does not suppress source/near-side ignition to zero.
- `late_on_axis` is weaker than timely `dynamic_on_axis`.
- `wall_only` has no meaningful events.
- Excluding target contacts does not reverse the instrument conclusion.

Mandatory FAIL guards:

- `dynamic_off_axis` suppresses far spread as much as `dynamic_on_axis`.
- `late_on_axis` suppresses far spread as much as timely on-axis.
- source-side ignition disappears under on-axis intervention.
- collision rate increases.
- wall-only produces events.
- contact-exclusion flips the instrument result.

## Implementation Order

1. Pure helper TDD.
2. Runner-local simulation adapter TDD with no-intervention parity against canonical `simulate_kick`.
3. Baseline eligibility smoke.
4. If eligible, four-control short pilot.
5. Stop and report JSON summary only. No formal run, no figure-as-evidence, no stronger claim.
