# Topic 4 patient-like off-axis surround stimulation — Plan

> **For agentic workers:** implement with a cheap-first TDD flow. Do not run long simulations until the geometry screen and smoke tests pass and the user has reviewed the candidate list.

## Goal

Test a new stimulation strategy on patient-like electrode layouts: after identifying the pathological propagation axis, stimulate **non-axis surrounding contacts** rather than the axis corridor or the source core, then quantify whether propagation / off-axis recruitment is reduced during stimulation and whether activity returns after stimulation is removed.

This is not the same claim as the prior on-axis barrier figure. The scientific question is:

> In a real-patient-like planar montage, can off-axis surround stimulation suppress the spread or lateral recruitment of interictal-like events around the pathological axis?

## Preliminary Survey

Current reusable pieces:

- `scripts/run_sef_hfo_subject_snn.py` already builds subject-specific SNN substrates from real contact geometry and source/sink endpoints, then saves `figdata_*.npz` with registered contacts, `reg`, source/sink cores, E positions, `vth`, and readout metadata.
- `scripts/paper_figures/plot_fig_subject_snn_stimulation.py` already demonstrates subject-level stimulation using `simulate_dynamic_vth` and a contact-based E-cell mask.
- `src/sef_hfo_axial_intervention.py::simulate_dynamic_vth` is the preferred dynamic-threshold engine path because it is parity-tested against no-stim simulation before stimulation onset.
- Existing real-data geometry evidence supports a pathological long-axis scaffold, but does not yet settle event-level axial-vs-lateral footprint. Therefore this plan must report off-axis surround stimulation as exploratory.

Important boundary:

- Do **not** claim that the chosen subject is clinically confirmed ECoG unless separate metadata proves it. Existing Epilepsiae documentation says the legacy code can distinguish intracranial vs auxiliary channels, but not SEEG vs ECoG reliably. For v1, select **ECoG-like / planar broad-coverage** subjects by geometry.

Quick candidate screen from existing `figdata_*.npz` suggests these are good first candidates because they have broad two-dimensional contact spread, enough non-axis contacts, and readable directional events:

| Priority | Candidate | Why |
| --- | --- | --- |
| 1 | `epilepsiae_916` broad | broad off-axis span, enough off-axis candidate contacts, bidirectional readout in existing subject-SNN artifact |
| 2 | `epilepsiae_590` narrow | planar enough, enough off-axis contacts, bidirectional readout |
| 3 | `epilepsiae_442` narrow | usable off-axis contacts and bidirectional readout |
| Backup | `epilepsiae_1150` broad | broad layout but needs baseline event eligibility check |
| Backup | `yuquan_zhaojinrui` / `yuquan_zhaochenxi` | strong geometry/readout backups, but less aligned with the "ECoG-like Epilepsiae" motivation |

## Claim Scope

Allowed:

- "In this model and patient-like electrode geometry, off-axis surround stimulation reduces / does not reduce event spread during the stimulation window."
- "The effect is selective only if axis-readable or local events remain while off-axis participation drops."
- "This is an exploratory within-model stimulation strategy."

Forbidden:

- "ECoG stimulation treats seizures."
- "Non-axis stimulation proves a clinical mechanism."
- "All events are eliminated" if the result is only reduced readout or local confinement.
- "This is true ECoG" without external channel-type confirmation.

## Experimental Arms

All arms use the same subject substrate, same random seed, same run duration, and matched contact count `N`.

Pairing boundary: arms must be identical before `stim_on`; after `stim_on`, trajectories are allowed to diverge. Therefore post-stim comparisons are window-level statistics, not one-to-one paired event comparisons.

| Arm | Target | Purpose |
| --- | --- | --- |
| `baseline` | no stimulation | pre/during/post windows from the same unstimulated run; establishes event availability |
| `offaxis_surround` | non-axis contacts flanking the pathological axis | main new strategy |
| `onaxis_corridor` | same number of contacts inside the axis corridor | positive/reference comparator; should resemble prior axis-blocking logic |
| `empty_or_far` | no E cells or far non-participating contacts, if available | sham/control for analysis artifacts |
| `core_partial` | same number of source/core-adjacent contacts, optional | comparator to the "打灶" strategy; include only if contact budget is fair and not full-core coverage |

v1 can stop after `baseline + offaxis_surround + onaxis_corridor` if the geometry does not support a clean sham/core arm.

## Geometry Definitions

Use the registered 2-D sheet coordinate system saved in subject `figdata_*.npz`.

- Pathological axis: vector from registered source centroid to sink centroid.
- Along-axis coordinate: projection onto the pathological axis.
- Off-axis coordinate: projection onto the perpendicular axis.
- Axis corridor: contacts with `abs(off_axis_mm) <= corridor_halfwidth_mm` and lying between / near the source-sink interval.
- Non-axis surround contacts: contacts with `abs(off_axis_mm) >= offaxis_min_mm`, not in source/sink core contacts, not inside the axis corridor, and with along-axis positions near the source-sink interval.
- Stim target mask: E cells within `stim_radius_mm` of selected contacts, using the same style as `_electrode_stim_target`.
- Core contact mask: prefer saved source/sink contact labels if present; otherwise derive by distance to saved `foci` and `core_r` in `figdata_*.npz`.

Default v1 thresholds, to be locked after helper tests:

- `N = 4` contacts per stimulation arm.
- `corridor_halfwidth_mm = max(1.5, 0.15 * inter_core_distance_mm)`.
- `offaxis_min_mm = max(2.5, corridor_halfwidth_mm)`.
- `stim_radius_mm = 2.0`.
- Off-axis contacts should be balanced across the two sides of the axis when possible: `N/2` above, `N/2` below.

Eligibility gates:

- planar/broad geometry: `PCA_minor / PCA_major >= 0.45` or externally labeled ECoG.
- off-axis span at least `8 mm`.
- at least `N` eligible off-axis contacts after excluding source/sink core and axis corridor.
- baseline readable directional events: at least `6` clean events over the full run, and at least `2` before stimulation onset.
- no runaway / tonic saturation before the stimulation window.

If a subject fails eligibility, do not tune stimulation to force a positive result; move to the next predeclared candidate.

## Timing

Use a continuous spontaneous subject-SNN run, not event-triggered closed-loop stimulation.

Default:

- `T = 5000 ms` for pilot.
- `stim_on = 1500 ms`.
- `stim_off = 3500 ms`.
- windows: `pre = [0, stim_on)`, `during = [stim_on, stim_off)`, `post = [stim_off, T)`.

If a candidate has too few pre-stim events, run one cheaper baseline-only screen with longer `T` before accepting or rejecting the subject. Do not adjust stimulation timing based on the observed event onset times in the same run.

## Metrics

Compute metrics per event and aggregate by window (`pre`, `during`, `post`).

Primary:

- clean propagation event count per window.
- off-axis participation fraction: participating contacts outside the axis corridor divided by all participating contacts.
- transverse spread: max-min off-axis coordinate of participating contacts.
- axis readability: whether event still has a coherent along-axis direction / gradient.

Secondary:

- local-only event count during stimulation: events with activity near source or stim contacts but below clean propagation threshold.
- far-side participation fraction.
- event duration and peak LFP envelope.
- total E spike count and max population rate, to detect global suppression.

Interpretation contract:

- `offaxis_surround` is a plausible selective effect only if off-axis participation or transverse spread drops during stimulation while some local/axis-readable activity remains.
- If all activity disappears, report it as global suppression, not selective non-axis control.
- If `offaxis_surround` matches `empty_or_far`, report no effect.
- If `onaxis_corridor` works but `offaxis_surround` does not, report that the patient-like layout supports axis blocking but not off-axis surround suppression.

## Outputs

Write outputs under:

- `results/topic4_sef_hfo/offaxis_surround_stim/candidate_screen.json`
- `results/topic4_sef_hfo/offaxis_surround_stim/pilot_<subject>.json`
- `results/topic4_sef_hfo/offaxis_surround_stim/pilot_<subject>_events.csv`
- optional figure after pilot review: `results/paper-ready-figure/fig_topic4_offaxis_surround_stim/figures/`

If a figure is generated, also write `figures/README.md` in Chinese, with the exact pre/during/post counts and a clear statement that the figure is model-only and exploratory.

## TDD Tasks

### Task 1 — Pure geometry helpers

Files:

- Create `src/topic4_offaxis_surround_stim.py`
- Create `tests/test_topic4_offaxis_surround_stim.py`

Required helpers:

- `axis_frame(source_xy, sink_xy) -> dict`
- `project_contacts(contacts, frame) -> DataFrame or dict`
- `classify_axis_corridor(contacts, frame, corridor_halfwidth_mm, along_pad_mm) -> mask`
- `select_offaxis_surround_contacts(contacts, frame, core_contact_mask, N, corridor_halfwidth_mm, offaxis_min_mm) -> indices`
- `select_onaxis_corridor_contacts(contacts, frame, core_contact_mask, N, corridor_halfwidth_mm) -> indices`
- `electrode_e_mask(posE, contacts, indices, radius_mm) -> bool array`

Tests:

- synthetic planar grid selects balanced off-axis contacts on both sides of the axis.
- selected off-axis contacts never overlap core contacts or axis corridor contacts.
- selected on-axis contacts have the same `N` as off-axis contacts.
- insufficient off-axis contacts raises a clear error.
- E-cell mask radius behaves deterministically on a toy point cloud.

### Task 2 — Candidate screen

Create `scripts/screen_topic4_offaxis_surround_candidates.py`.

Inputs:

- `results/topic4_sef_hfo/field_swap_subject_snn/figdata_*.npz`
- matching `readout_*.json`

Output:

- `candidate_screen.json` with geometry metrics, readout counts, selected contacts for each arm, and eligibility status.

Acceptance:

- screen runs without SNN simulation.
- it ranks the predeclared candidates and explains rejection reasons.
- it does not require clinical ECoG labels.

### Task 3 — Dynamic stimulation runner

Create `scripts/run_topic4_offaxis_surround_stim.py`.

Behavior:

- Rebuild or load the same subject-SNN substrate used by `run_sef_hfo_subject_snn.py`.
- If rebuild code is duplicated, first factor a behavior-preserving `build_subject_snn_model(...)` helper out of `run_sef_hfo_subject_snn.py`; do not change the canonical subject-SNN defaults or saved artifact schema.
- Use `simulate_dynamic_vth` for all stimulation arms.
- Use identical seed and same random stream across arms.
- Assert pre-stim parity: `baseline` and stimulated arms must match before `stim_on`.
- Write event-level metrics and summary JSON.

Acceptance:

- one short smoke run (`T <= 800 ms`) completes for one candidate.
- no-stim dynamic path matches canonical no-stim path before any stimulation.
- all selected masks are E-only and nonempty.

### Task 4 — Event metrics

Add metric helpers to `src/topic4_offaxis_surround_stim.py`.

Required helpers:

- `split_events_by_window(events, stim_on, stim_off, T)`
- `event_axis_offaxis_metrics(event, contact_coords, axis_frame, corridor_halfwidth_mm)`
- `summarize_windows(events, metrics, windows)`

Tests:

- events on window boundaries are assigned consistently.
- off-axis fraction is zero for corridor-only events.
- transverse spread increases when contacts on both sides of the axis participate.
- summary reports `n_clean`, `n_local`, `offaxis_fraction_median`, `transverse_span_median`, `axis_readable_fraction`.

### Task 5 — Pilot execution checkpoint

Run only after Tasks 1-4 pass.

Pilot sequence:

1. Run candidate screen.
2. Pick top eligible Epilepsiae ECoG-like candidate, default `epilepsiae_916` if it passes.
3. Run `baseline`, `offaxis_surround`, and `onaxis_corridor`.
4. Stop and summarize pre/during/post metrics.

Hard stop conditions:

- no candidate has enough off-axis contacts.
- baseline has fewer than 2 pre-stim readable events.
- stimulated arms are not parity-identical before `stim_on`.
- off-axis target overlaps source/sink core or axis corridor.
- the run hits tonic saturation before `stim_on`.

### Task 6 — Optional figure only after pilot review

Do not generate a paper-ready figure until the pilot summary is reviewed.

If approved, render a single-row figure:

1. left: patient-like contact map with pathological axis, axis corridor, and selected off-axis surround contacts.
2. middle: representative pre-stim propagation event.
3. middle/right: representative during-stim event or local-only activity.
4. right: readout trace with `stim_on`/`stim_off` shading and pre/during/post counts.

The figure must not imply clinical efficacy.

## Minimal Success Criteria

Engineering success:

- helper tests pass.
- candidate screen writes reproducible JSON.
- runner has pre-stim parity.
- pilot summary has complete pre/during/post metrics.

Scientific pilot success:

- there is at least one eligible patient-like planar candidate.
- baseline produces readable events before stimulation.
- off-axis surround stimulation changes either clean propagation count, off-axis participation, or transverse spread during stimulation.
- post-stim window is included to check reversibility / return.

Negative but useful outcome:

- no eligible geometry exists in current artifacts.
- off-axis surround has no effect while on-axis corridor works.
- off-axis surround only causes global suppression.
- baseline events are too sparse for within-run pre/during/post comparison.

All of these should be archived as valid exploratory outcomes, not treated as failed engineering.
