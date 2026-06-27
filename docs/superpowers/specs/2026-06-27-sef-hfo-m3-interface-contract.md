# M3A ↔ M3B-R2 interface contract (CANONICAL)

> Status: CANONICAL CONTRACT, 2026-06-27.
> This is the single source of truth for the M3A → M3B-R2 handoff. The M3A-A1/A2 plans and the
> M3B-R2 spec/plan all REFERENCE this file for field names, enums, and the overlay gate; they do
> not redefine them locally.
> Executable mirror: `src/sef_hfo_m3_interface.py` (imported by BOTH the M3A exporter and the M3B
> axis-builder/overlay so the two lines cannot drift).
> TDD: `tests/test_sef_hfo_m3_interface.py` (contract-layer) + the runner-layer tests listed in §7.
> Origin: 4-lens adversarial review 2026-06-27 found 8 fail-open blockers in the previously-written
> interface; this contract closes them. See §8 for the blocker→fix map.

## 0. Why this contract exists (plain language)

The whole job of this handoff is to stamp "this slow-variable trajectory is ALLOWED to be drawn on
the spectral phase map." Without a strict stamp, M3A and M3B silently borrow each other's
conclusions: M3B draws a trajectory that was never calibrated, or M3A claims its trace "walks the
phase diagram" when M3B never agreed the coordinates match. The review showed the stamp could be
bypassed 8 ways. This contract makes every stamp condition fail-closed: a missing/empty/default
value REFUSES the overlay, it never silently permits it.

## 1. Decisions locked 2026-06-27 (user sign-off)

- **D1 — coordinate space**: the M3B phase-map axes ARE the normalized phase coordinates
  `phase_x_core`, `phase_y_global` ∈ [0, 1]; axis extent = `phase_coord_ranges.json`. There is no
  raw-physical-unit axis. `axis_space == "normalized_unit"` is the only allowed value this round.
  (The M3B spec §3 "examples" of raw-unit axes are demoted to *descriptions of the underlying slow
  variables*, not axis units.)
- **D2 — out-of-range tolerance**: `OUT_OF_RANGE_FRACTION_MAX = 0.05`. Overlay is permitted only if
  ≤ 5 % of trajectory samples are out-of-range AND every out-of-range sample carries its flag.
  Above 5 % (or a missing fraction) → not a trajectory.
- **D3 — recovery dimension**: the overlay is LOSSLESS — `slow_trajectory_overlay.csv` always
  carries all three phase coordinates. The audit records which two are on-axis
  (`on_axis_coords = [phase_x_core, phase_y_global]`) and which is projected out
  (`projected_out_coords = [phase_recovery]`).
- **D4 — scope of the 2026-06-27 build**: shared module + contract-layer TDD + this canonical doc +
  reconcile the 4 plan/spec interface sections + scaffold the broader M3B TDD-0..13.
- **D5 — two_core_reduction (2026-06-27 user)**: equal cores (`stage3_twoend_equal`) collapse to
  `phase_x_core` by AVERAGING — `two_core_reduction == "mean_q"`. The exporter averages
  `q_core_L`/`q_core_R` before `evaluate_phase_coord`; `source_core` / `min_q` are NOT used (they
  break the left/right symmetry of the equal-core substrate) and remain unimplemented (raise).
- **D6 — absolute return-to-baseline (2026-06-27 user)**: `tail_to_baseline_ratio` is ABSOLUTE —
  tail activity vs a FIXED quiet baseline window, never the event's own peak. Operational definition:
  baseline window = `BASELINE_MS = (5, 50)` ms, tail window = 200 ms after event end, returned if
  ratio ≤ 1.5. Pure helper: `src/sef_hfo_a2.tail_to_baseline_absolute`.
- **D7 — peak landmark (2026-06-27 user)**: the event `peak` sample is the real activity-fraction
  peak inside the event window (`src/sef_hfo_a2.event_peak_ms`), NOT the window midpoint. The midpoint
  is a recorded fallback only when no peak is supplied.

Fail-closed defaults adopted without a separate decision (the only correct answer is "fail closed"):
closed-enum transforms (no `eval` of free-text formulas), single canonical `slow_to_rate_mapping_id`,
schema'd `m3a_interface_audit.json`, sign-tests must be non-empty-and-all-pass, out-of-range checked
against the INPUT domain (so output clipping cannot hide extrapolation), overlay triggered by gate
VERDICT not by file availability.

## 2. Canonical field names (resolve all naming drift)

| Concept | CANONICAL name (use everywhere) | Was also called |
|---|---|---|
| mapping identity | `slow_to_rate_mapping_id` | `mapping_id` (JSON top-level key is now `slow_to_rate_mapping_id`) |
| baseline-return boolean | `return_to_baseline` | `returned` (A2 `event_phase_samples.csv`) |
| out-of-range flag | `phase_coord_out_of_range` | (absent from A1 handoff / freeze_samples — now REQUIRED) |
| validity flag | `phase_coord_valid` | — |
| M3B grid axes provenance | `axes_built_from_slow_to_rate_mapping_id` | (absent — now REQUIRED on `finite_jacobian_grid.json`) |

`slow_to_rate_mapping_id` is a REQUIRED column on `phase_trajectory.csv`, `event_phase_samples.csv`,
`freeze_samples.jsonl`, and a top-level key on `slow_to_rate_mapping.json`, `phase_coord_ranges.json`,
`dynamic_slowvars_summary.json`, and `m3a_interface_audit.json`. All must be byte-identical for an
overlay. `phase_coord_out_of_range` is REQUIRED on every phase-coord-bearing artifact; its absence is
an export failure / overlay refusal, never silently "in range".

## 3. Canonical enums (hosted in `src/sef_hfo_m3_interface.py`)

```text
event_stage          : {baseline, pre, onset, peak, end, post_50ms, post_200ms, post_1s, post, inter_event}
                       ("post" is the documented coarse rollup of post_50ms/post_200ms/post_1s;
                        "inter_event" is for trajectory rows with no parent event)
gate_A_trajectory    : {PASS, FAIL, INCONCLUSIVE}
gate_B_seizure_like  : {PASS, FAIL, INCONCLUSIVE}
trajectory_robustness: {robust, seed_fragile, runaway_prone, quiet_prone, not_tested}
rate_matched_control : {passed, failed, not_run}
calibration_status   : {passed, failed, not_applicable}
overlay_verdict      : {phase_map_trajectory, mechanism_candidate_only, refused}
transform.type       : {identity, affine, reciprocal_affine}   # closed enum, evaluated deterministically, NO eval
axis_space           : {normalized_unit}                        # D1
mode_class  (M3B)    : {stable, local, axial, mixed, global, runaway, unresolved}
R_class              : {R0, R1, R2, R3, R4a, R4b}
phenotype_label      : {local_axial, larger_axial, mixed_global, global_recruitment, runaway, recovery}
```

A field that takes one of these enums must hold a MEMBER value; mere key-presence is not enough
(closes the "verdict could be `maybe`/`null`" hole).

## 4. `slow_to_rate_mapping.json` (M3A-A1 owns; closes B3, B4, M9, M16)

```json
{
  "slow_to_rate_mapping_id": "m3a_a1_<date>_<hash>",
  "source": "M3A-A1 quasi-static SNN calibration",
  "substrate": "stage3_twoend_equal",
  "axis_space": "normalized_unit",
  "two_core_reduction": "source_core | min_q | mean_q",     // required when substrate is two-core (D-science, recorded as data)
  "coordinates": {
    "phase_x_core":   { "...": "coordinate spec, see below" },
    "phase_y_global": { "...": "coordinate spec" },
    "phase_recovery": { "...": "coordinate spec" }
  }
}
```

Per-coordinate spec (closed-enum transform — NO free-text formula evaluated):

```json
{
  "transform": {
    "type": "reciprocal_affine",          // identity | affine | reciprocal_affine
    "input_var": "q_global",
    "a": 1.0, "b": 0.0,                     // out = clip(a/input + b)  (reciprocal_affine)
    "clip": [0.0, 1.0],
    "input_min": 0.2, "input_max": 1.0,    // calibrated INPUT domain; out-of-range checked here too
    "expected_direction": "decreasing_in_input"   // increasing_in_input | decreasing_in_input
  },
  "units": "dimensionless",
  "valid_range": [0.0, 1.0],
  "variables": ["q_global"],
  "shunt_path_active": true,               // REQUIRED when input_var/variables include e_GABA
  "calibration_status": "passed",          // passed | failed | not_applicable
  "sign_tests": [
    { "name": "phase_y_global_decreasing_in_q_global",
      "coord": "phase_y_global", "input_var": "q_global",
      "expected_direction": "decreasing_in_input",
      "observed_slope_sign": -1, "passed": true, "engine_sha": "<sha>" }
  ]
}
```

Fail-closed rules:
- `calibration_status == "passed"` is permitted ONLY if `sign_tests` is non-empty AND every entry
  has `passed == true`. Empty / missing / `not_applicable` `sign_tests` on a coordinate used as a
  plotted axis ⇒ that coordinate is treated as **failed** (B3).
- The sign test is a SIGNED-slope test, not a monotonicity test: evaluating `transform` over
  `[input_min, input_max]`, the coordinate must be STRICTLY monotone in `expected_direction`. A
  backwards-encoded axis (e.g. `phase_y_global` increasing in `q_global`) fails (B4).
- Required directions (from the A1 mapping table): `phase_y_global` decreasing in `q_global`/`z_global`;
  `phase_x_core` decreasing in `q_core`/`z_core` (more disinhibition ⇒ higher coordinate);
  `phase_recovery` increasing in `phi` and `g_K`, decreasing in `x_EE` (more available E→E resource
  = less protection). `x_EE` may appear in `phase_recovery` only with a reciprocal/decreasing term;
  `phi`/`g_K` must not appear as plain positive disinhibition terms (M6, M7).
- `e_GABA` is assigned to exactly one coordinate, recorded in `variables`; its disinhibition export
  requires `shunt_path_active == true`, else `calibration_status != "passed"` (M8).
- out-of-range is checked against the INPUT domain `[input_min, input_max]` as well as the output
  `valid_range`; if the raw input is outside its domain the sample is out-of-range even when the
  clipped output lands in [0, 1] (M16).

`phase_coord_ranges.json` carries the same `slow_to_rate_mapping_id` and per-coordinate
`{min, max, source}` for the axis extent.

## 5. `phase_trajectory.csv` / `event_phase_samples.csv` / `freeze_samples.jsonl` (M3A-A2 owns)

- Both CSVs and `freeze_samples.jsonl` carry `slow_to_rate_mapping_id`, all three phase coords,
  `phase_coord_valid`, `phase_coord_out_of_range`, and `event_stage` (canonical enum).
- `event_phase_samples.csv` carries `return_to_baseline` (canonical name) plus `tail_to_baseline_ratio`
  and `R_class`. `freeze_samples.jsonl` also carries `R_class` (M11) and `phenotype_label`.
- Two-core substrate ⇒ `phase_trajectory.csv` carries `q_core_L`, `q_core_R` and the mapping's
  `two_core_reduction` defines how they collapse to `phase_x_core` (M19).
- `phase_coord_valid(sample) := AND over the axes the sample uses of (calibration_status=="passed"
  AND sign tests passed)` — i.e. "is the mapping trustworthy here". VALIDITY (calibration) and RANGE
  are ORTHOGONAL: `phase_coord_out_of_range == true` does NOT force `phase_coord_valid == false` (a
  calibrated-but-extrapolated sample is valid AND out_of_range). Range is handled separately by
  `cond3`; calibration is handled by `cond1`. Any sample with `phase_coord_valid == false` REFUSES the
  overlay (`cond3` is false) — invalid samples are not silently dropped (M1). `cond3` is therefore:
  trajectory schema-valid AND every row `phase_coord_valid` AND out-of-range fraction ≤ 5 %.
- A disabled mechanism writes `NA` (never `0.0`) for its raw slow-var columns; a derived phase
  coordinate whose contributors are ALL disabled is `NA` (not `0.0`) (M12).
- `rho_*` columns (`rho_model_coord = lgr/(q_core q_global)`) are OPTIONAL A2-LOCAL diagnostics, not
  part of the required handoff. "lgr" is currently undefined and MUST be expanded (or the columns
  dropped) before they enter any claim; the resource coordinate is named `rho_resource` to avoid
  colliding with M3B's spectral `rho(M)` (M10). [open: lgr definition — see §9]

## 6. The overlay gate — `m3a_interface_audit.json` (M3B owns; closes B1, B2, B6, B7, M13, M14)

The audit is ALWAYS written when an M3A handoff is attempted (verdict records the outcome). The
overlay artifacts are written ONLY when `overlay_verdict == phase_map_trajectory` (B2 — trigger on
verdict, never on file availability).

```json
{
  "audited_slow_to_rate_mapping_id": "m3a_a1_<date>_<hash>",
  "cond1_sign_tests_passed":           true,   // every plotted-axis coord: calibration passed AND sign_tests non-empty all-pass
  "cond1_source": "slow_to_rate_mapping.json",
  "cond2_same_mapping_and_ranges":     true,   // axes id == trajectory id == ranges id AND axis_space + transform descriptors identical
  "cond2_source": "finite_jacobian_grid.axes_built_from_slow_to_rate_mapping_id",
  "cond3_in_range_or_flagged":         true,   // schema-valid AND every row phase_coord_valid (M1) AND out_of_range_fraction <= 0.05
  "cond3_out_of_range_fraction":       0.0,    // REQUIRED, no default; missing => cond3 false
  "cond4_phenotype_movement_beyond_rate": true,  // STRICT: A2 dynamic rate_matched_control=="passed" AND gate_A_trajectory=="PASS"
  "cond4_source": "dynamic_slowvars_summary.json",
  "on_axis_coords":      ["phase_x_core", "phase_y_global"],
  "projected_out_coords":["phase_recovery"],
  "gate_used": "A",                            // overlay is Gate-A only (tier guard, M13)
  "overlay_verdict": "phase_map_trajectory",   // = compute_overlay_verdict(cond1..cond4)
  "overlay_allowed": true                      // == (overlay_verdict == "phase_map_trajectory")
}
```

`compute_overlay_verdict(c1, c2, c3, c4)` (pure logic, fail-closed; None/missing ⇒ False):
- all four True ⇒ `phase_map_trajectory`;
- `c4` True but NOT(`c1` and `c2` and `c3`) ⇒ `mechanism_candidate_only`
  (the phenotype is real but the calibration/provenance/range is not — overlay refused, candidate kept);
- else ⇒ `refused` (no phenotype movement, or missing files).

`cond4` reads ONLY the A2 dynamic summary (never the A1 quasi-static "beyond rate-only heating"
answer — resolution-level trap, B6), by STRICT equality: `not_run` / `INCONCLUSIVE` / missing ⇒ False.
`cond2` requires both the id triple-equality AND identical `axis_space` + transform descriptors
(id match alone is necessary-not-sufficient, B5/B8).

The four `condN_*` booleans are REQUIRED; a missing condition cannot default to true (B1). The
present-but-failed cases (failed sign tests, id mismatch, gate_A not PASS, partial calibration,
out_of_range fraction over threshold) each individually drive the verdict away from
`phase_map_trajectory` and have explicit refusal tests (M14).

`dynamic_slowvars_summary.json` additionally carries `m3b_ready` (bool) + `m3b_ready_reason`:
`m3b_ready == (gate_A_trajectory=="PASS" AND all axis calibration_status=="passed" AND
rate_matched_control=="passed")`. It is necessary-not-sufficient: M3B still independently re-checks
`cond2`/`cond3` (M15).

## 6.1 `slow_trajectory_overlay.csv` (M3B output; closes M18)

Written ONLY when `overlay_verdict == phase_map_trajectory`. Columns = the §6.2 min-columns set
PLUS `phase_coord_out_of_range`, `slow_to_rate_mapping_id`, `in_map` (bool), and the phase-map
readout at the point (`leading_mode_class`, `alpha_1`). All three phase coords are carried (D3,
lossless). When the verdict is not `phase_map_trajectory`, the file is absent (or 0 rows with a
verdict marker) and no overlay figure is drawn.

The builder (`build_slow_trajectory_overlay`) is bound to the audited artifact and does not trust its
inputs: it `validate_interface_audit`s the audit, gates on `overlay_allowed` (not just the verdict
string), and INDEPENDENTLY re-checks the rows it actually draws — every drawn row's
`slow_to_rate_mapping_id` must equal `audited_slow_to_rate_mapping_id`, no sample may be invalid, and
the out-of-range fraction must be within D2 — raising otherwise (TOCTOU guard). Required phase
coordinates are hard-indexed, never NA-defaulted.

## 6.2 Resolved min-columns (M3B consumed record; closes M5)

The 11 minimum columns are resolved by joining `phase_trajectory ⋈ event_phase_samples` on
`event_id` (sentinel `event_id == -1` / `NA` for inter-event rows):

```text
time_ms, event_id, event_stage, phase_x_core, phase_y_global, phase_recovery,
phase_coord_valid, phase_coord_out_of_range, slow_to_rate_mapping_id, R_class, return_to_baseline
```

A contract test asserts the full set is resolvable from sample A2 artifacts after the documented
join, failing if any column is missing or misnamed.

## 7. Contract-layer vs runner-layer test partition

Contract-layer (pure JSON/CSV, hosted in `tests/test_sef_hfo_m3_interface.py`, GREEN without the
SNN): all schema validators, signed sign-direction checks on the declared transform, mapping_id
consistency, `phase_coord_valid` AND-logic, out-of-range input+output check, `compute_overlay_verdict`
truth table, the present-but-failed refusal cases, the missing-file refusals, the min-columns join,
`m3b_ready` truth table, and the lossless overlay schema.

Runner-layer (need the SNN; live in the M3A-A2 worktree TDD, NOT faked here — they must emit
bool/enum that the contract-layer tests then assert on):
- `test_m3a_mapping_signs_match_rate_helpers` — JSON sign vs actual LIF engine response.
- A1 Task-2 engine sign tests (lower z weakens inhibition; higher phi raises threshold; higher g_K
  suppresses; depolarized e_GABA less protective only in shunt path).
- `test_q_resource_trace_maps_to_expected_rho_direction` — needs a dynamic trace + a DEFINED "lgr".
- phenotype-movement determination half of `test_m3b_ready_flag_requires_mapping_and_phenotype_movement`.
- `test_tail_to_baseline_is_absolute_not_relative` — absolute definition is PINNED (§1 D6) and a
  pure/helper test already EXISTS (`src/sef_hfo_a2.tail_to_baseline_absolute`,
  `tests/test_m3a_tail_and_peak.py`); the SNN event-table wiring (per-event `tail_to_baseline_ratio`
  into `event_phase_samples.csv`) is pending R-class.
- `gate_A_trajectory` / `gate_B_seizure_like` / `rate_matched_control` determinations.
- M3B TDD-12 SNN spot-check classification.

## 8. Blocker → fix map (from the 2026-06-27 review)

| Blocker | Fix (this contract) |
|---|---|
| B1 audit has no schema (fail-open by omission) | §6 schema: 4 required booleans + `overlay_verdict` = pure AND |
| B2 overlay triggered by file availability | §6/§6.1 trigger on `overlay_verdict == phase_map_trajectory` |
| B3 sign_tests vacuously passed | §4 `calibration_status=="passed"` ⇒ non-empty all-pass; element schema |
| B4 monotonicity passes a sign-flipped axis | §4 signed-slope test in `expected_direction` |
| B5 mapping_id named two ways, absent from CSVs | §2 canonical `slow_to_rate_mapping_id` on all artifacts + `axes_built_from_*` |
| B6 cond4 fail-open / wrong resolution level | §6 STRICT A2-only `rate_matched_control=="passed" AND gate_A=="PASS"` |
| B7 out-of-range OR always satisfiable + 0.0 default | §1 D2 numeric gate 5 %; `cond3_out_of_range_fraction` required no default |
| B8 axes vs trajectory different coordinate space | §1 D1 normalized [0,1]; shared `evaluate_phase_coord`; cond2 transform-identity |

## 9. Still-open items requiring a later USER/science decision (NOT invented here)

These do not block the contract-layer build (each is fail-closed by default), but must be pinned
before the corresponding runner-layer test or claim:

- `e_GABA` axis assignment (core vs global) — decided at A1 calibration; contract only enforces it is
  recorded and shunt-gated.
- "lgr" definition in `rho_model_coord` — undefined; expand (with units/sign) or drop the optional
  `rho_*` columns. Must be A2-locally computable (no M3B spectral input) or it couples the two lines.
- classification crosswalk `phenotype_label ↔ R_class ↔ mode_class` — `classification_crosswalk()`
  ships a documented table; confirm its rows. Wiring `R_class` into `event_phase_samples.csv` is the
  next M3A-A2 step.

**Resolved 2026-06-27 (moved to §1 locked decisions):** `two_core_reduction = "mean_q"` (D5); the
absolute `tail_to_baseline` operational definition (D6); the activity-fraction peak landmark (D7).
