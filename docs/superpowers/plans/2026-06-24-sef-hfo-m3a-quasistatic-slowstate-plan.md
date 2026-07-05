# M3A-A1 quasi-static slow-state mechanism plan

> Status: new hard-boundary plan, 2026-06-24.
> Scope: M3A only. This plan tests whether fixed slow-variable states can make the SNN spontaneously produce distinguishable interictal-like vs seizure-like event phenotypes.

## 0. Hard Boundary

M3A answers only this question:

```text
slow-variable state s  ->  spontaneous event phenotype
```

M3A does not define `W`, does not use `h(W)`, and does not use `W`-coupled threshold permissivity as a mechanism. The old static `V_th_eff = V_th0 - delta * mu * h(W)` result is treated as negative evidence: it raised event rate but did not change event size, duration, or R-class in a mechanistically useful way.

Primary input is no-kick spontaneous activity. Kick or finite-pulse probes are allowed only as secondary stability checks, not as the definition of seizure-like transition.

## 1. Scientific Goal

Test whether a biologically motivated slow state can move the SNN through at least two distinguishable regimes:

- interictal-like returned finite events: R2/R3, return-to-baseline preserved;
- seizure-like sustained recruitment: R4a only, with spatial/recruitment structure still present;
- tonic full-field runaway: R4b, explicitly not accepted as seizure-like bridge.

M3A-A1 is quasi-static. It freezes or clamps slow-state values and asks: if the tissue were already at this slow state, would spontaneous events change phenotype?

A1 also owns the **slow-to-rate calibration interface** for M3B-R2. That does not make A1 a
spectral analysis. It means every frozen slow state that A1 passes downstream must have an explicit
interpretation as a coarse rate-field parameter change: local core excitability, global
disinhibition, or recovery/protection. M3B must not infer those signs by itself.

## 2. Mechanism Candidates

Priority order:

1. `e_GABA` / depolarizing GABA / chloride accumulation proxy.
2. `z` disinhibition: current-based `I_net = I_E - z * I_I`, where lower `z` weakens inhibition.
3. `phi` adaptive threshold.
4. `g_K` sAHP outward current.

`src/snn_engine/slow_vars.py` contains placeholder parameter values. The plan must not draw biological conclusions from defaults. Defaults are only smoke-test starting points.

## 3. Engine Path Gate

Before any scientific run:

- confirm the active worktree has `SlowVars` hooks in the execution path being used;
- current mainline `src/snn_engine/model.py` supports `simulate(..., slow=...)`;
- current mainline `src/snn_engine/kick_probe.py` also exposes `simulate_kick(..., slow=...)`, but this must be rechecked in the target worktree;
- `slow=None` must be bit-parity with the previous no-slow baseline;
- if `model.simulate()` lacks full event readouts needed for R0-R4 classification, using `simulate_kick(KICK_BOOST=0, t_kick=1e9, slow=...)` is acceptable, but the run must be documented as no-kick spontaneous.

Do not implement new engine semantics until the path audit says which hook is missing. If a hook must be added, it is off by default and requires parity tests.

### Engine-path audit RESULT (2026-06-24, this worktree; src/snn_engine is git-tracked so it carries over)

- `kick_probe.py::simulate_kick(p, net, KICK_BOOST, slow=None, ...)` **DOES wire slow** (calls
  `slow.apply_currents / slow.threshold / slow.step`, lines 230-251); `model.py::simulate()` also does.
  Either path works for no-kick spontaneous (`simulate_kick(KICK_BOOST=0.0, t_kick=0.0, slow=...)`).
- `SlowVars` (slow_vars.py) `apply_currents/threshold/step` are **implemented** (z/φ/g_K equations present);
  only the *parameters* are PLACEHOLDER (must calibrate).
- **Smoke-verified (L8)**: `slow=None` is BYTE-IDENTICAL to no-slow-arg (bit-parity ✓);
  `slow=SlowVars(z=0.3)` changes spikes (125 → 404988 → z works but z=0.3 is uncalibrated runaway);
  static depolarized `e_gaba` (`shunt_gaba=True, e_gaba=...`) changes activity with **no engine change**.
- **Therefore**: Task 1 bit-parity gate is already met for `simulate_kick(slow=)`. The existing
  `scripts/run_m3_static_mu_spontaneous.py` (= `simulate_kick(KICK_BOOST=0)`) only needs `--slow-mode /
  --e-gaba / --shunt-gaba` pass-through to become the A1 runner — NO new engine semantics for
  quasi-static z/φ/g_K or static e_GABA. **Only DYNAMIC e_GABA (Cl⁻ accumulation, the prime A2 candidate)
  needs a new SlowVars e_GABA state + per-neuron time-varying e_gaba in membrane_step (off-by-default).**
- Concrete A1 first step (lowest effort, no engine change): **quasi-static e_GABA scan = sweep the
  `e_gaba` param (`shunt_gaba=True`)** over Task-4's levels via the spontaneous runner; z/φ/g_K
  quasi-static = `SlowVars` with frozen values. Calibrate before any conclusion (z=0.3 already runaway).

## 4. Outputs

Canonical output root:

```text
results/topic4_sef_hfo/m3a_slowvars/quasistatic/
```

Required files:

- `config.json`: engine SHA, Params, substrate, slow-state values, detector thresholds.
- `per_event.csv`: one row per spontaneous event.
- `slow_state_samples.csv`: state at `pre`, `onset`, `peak`, `end`, and post-event windows.
- `slow_to_rate_mapping.json`: calibrated mapping from SNN slow variables to M3B rate-field
  coordinates / parameter effects.
- `phase_coord_ranges.json`: valid A1 coordinate ranges for M3B phase-map axes.
- `summary.json`: event rate, size, duration, return probability, R-class fractions.
- `figures/README.md`: Chinese description for every generated figure.

Required per-event fields:

```text
event_id, seed, state_label, onset_ms, end_ms, duration_ms,
size_bins, active_mass, return_to_baseline,
R_class, sustained_front_score,
z_pre, z_onset, z_peak, z_end,
phi_pre, phi_onset, phi_peak, phi_end,
gK_pre, gK_onset, gK_peak, gK_end,
e_gaba_pre, e_gaba_onset, e_gaba_peak, e_gaba_end
```

Fields can be `NA` only when the mechanism is not active or not implemented; do not write unknown as 0.

Required state-level / event-level M3B handoff fields:

```text
phase_x_core, phase_y_global, phase_recovery,
phase_coord_valid, phase_coord_source,
slow_to_rate_mapping_id
```

`phase_x_core` and `phase_y_global` are not new biology; they are documented coordinates used by
M3B-R2 to place frozen states on a phase map. If a mechanism has no calibrated mapping, write
`phase_coord_valid=false` and do not hand it to M3B as a trajectory/range.

### M3A -> M3B slow-to-rate mapping contract

> **2026-06-27 — CANONICAL CONTRACT SUPERSEDES THE SCHEMA DETAIL BELOW.** See
> `docs/superpowers/specs/2026-06-27-sef-hfo-m3-interface-contract.md` §4 (executable mirror
> `src/sef_hfo_m3_interface.py`, TDD `tests/test_sef_hfo_m3_interface.py`). Corrections A1 must honor:
> - each coordinate's transform is a CLOSED-ENUM descriptor
>   `{type ∈ {identity, affine, reciprocal_affine}, input_var, a, b, clip, input_min, input_max,
>   expected_direction}` — NOT a free-text `formula` string (no `eval`);
> - the sign tests are SIGNED-slope tests (`phase_y_global` strictly decreasing in `q_global`;
>   `phase_x_core` strictly decreasing in `q_core`; `phase_recovery` increasing in `phi`/`g_K`,
>   decreasing in `x_EE`), NOT mere monotonicity — a backwards-encoded axis must FAIL;
> - `calibration_status == "passed"` is permitted only with non-empty, all-pass `sign_tests`
>   (element schema `{name, coord, input_var, expected_direction, observed_slope_sign, passed,
>   engine_sha}`); empty / `not_applicable` on a plotted axis fails closed;
> - the top-level id key is `slow_to_rate_mapping_id` (not `mapping_id`); `axis_space` ==
>   `"normalized_unit"`; `e_GABA` records `shunt_path_active` and may not export calibrated
>   disinhibition when it is false; two-core substrates record `two_core_reduction`;
> - out-of-range is checked against the INPUT domain `[input_min, input_max]` too, so output clipping
>   to [0,1] cannot hide extrapolation.
> The mapping-interface TDD listed at the end of this section is implemented at the CONTRACT layer in
> `tests/test_sef_hfo_m3_interface.py`; the ENGINE-sign tests (lower z weakens inhibition, etc.)
> remain runner-layer A1 work.

| SNN slow/control variable | Rate-field parameter effect | Phase-map coordinate role | Required sign check |
|---|---|---|---|
| `q_global` or global `z` | scales E-target inhibitory input globally | `phase_y_global ~ 1/q_global` or `1/z_global` | lower value weakens inhibition and should not be encoded with the opposite sign |
| `q_core` or core-local `z` | extra inhibition scale on core E cells | `phase_x_core ~ 1/q_core` or local disinhibition component | affects core E cells more than background E cells |
| `phi` / threshold adaptation | raises effective `V_theta` or reduces local drive | recovery/protection coordinate; may lower `phase_x_core` if core-local | higher `phi` suppresses firing / local gain |
| `g_K` | subtracts effective E drive / outward recovery current | recovery/protection coordinate | higher `g_K` suppresses activity and finite-time gain |
| static or dynamic `e_GABA` | changes effective inhibitory current via shunting/reversal path | global or core disinhibition only after calibration | depolarized `e_GABA` makes inhibition less protective in the active shunt path |
| static core threshold / drive | changes core E excitability | primary `phase_x_core` axis | lower threshold / higher drive increases core event susceptibility |

`slow_to_rate_mapping.json` must include formula, units, valid range, calibration status, and sign
tests for every coordinate it exposes. M3B consumes this file; it should not re-derive the mapping
from raw slow traces.

Minimum `slow_to_rate_mapping.json` schema:

```json
{
  "mapping_id": "m3a_a1_<date>_<hash>",
  "source": "M3A-A1 quasi-static SNN calibration",
  "substrate": "stage3_twoend_equal",
  "coordinates": {
    "phase_x_core": {
      "formula": "...",
      "units": "dimensionless",
      "valid_range": [0.0, 1.0],
      "variables": ["q_core", "z_core", "g_K_core", "phi_core"],
      "calibration_status": "passed|failed|not_applicable",
      "sign_tests": []
    },
    "phase_y_global": {
      "formula": "...",
      "units": "dimensionless",
      "valid_range": [0.0, 1.0],
      "variables": ["q_global", "z_global"],
      "calibration_status": "passed|failed|not_applicable",
      "sign_tests": []
    },
    "phase_recovery": {
      "formula": "...",
      "units": "dimensionless",
      "valid_range": [0.0, 1.0],
      "variables": ["g_K", "phi", "x_EE"],
      "calibration_status": "passed|failed|not_applicable",
      "sign_tests": []
    }
  }
}
```

Minimum `phase_coord_ranges.json` schema:

```json
{
  "mapping_id": "same-as-slow_to_rate_mapping",
  "phase_x_core": {"min": 0.0, "max": 1.0, "source": "A1 sweep range"},
  "phase_y_global": {"min": 0.0, "max": 1.0, "source": "A1 sweep range"},
  "phase_recovery": {"min": 0.0, "max": 1.0, "source": "A1 sweep range"}
}
```

TDD for the mapping interface is part of A1, not M3B:

- `test_slow_to_rate_mapping_schema_required_keys`
- `test_phase_coord_ranges_reference_same_mapping_id`
- `test_q_global_or_z_global_maps_monotonically_to_phase_y_global`
- `test_q_core_or_z_core_maps_monotonically_to_phase_x_core`
- `test_recovery_variables_are_not_mislabeled_as_disinhibition`
- `test_uncalibrated_variable_exports_phase_coord_invalid`
- `test_mapping_sign_tests_fail_closed`

## 5. Tasks

### Task 0: Freeze the M3A-A1 contract

- [ ] Write a short `STATUS.md` saying A1 excludes W/h(W), excludes external kick as primary, and treats static vth-mu as negative evidence.
- [ ] Record the old negative boundary: rate-only increase is not success.
- [ ] Record R4a vs R4b definitions before running.

### Task 1: Engine-path and bit-parity audit

- [ ] Add or run tests proving `slow=None` is bit-identical for the chosen no-kick execution path.
- [ ] Confirm `SlowVars` can alter dynamics when enabled.
- [ ] Confirm the runner can emit full event traces needed by R0-R4 classification.
- [ ] If the target worktree lacks `slow` support in `simulate_kick`, add only an off-by-default hook and re-bless parity.

Pass condition: no-slow path unchanged; slow-on path measurably changes either currents, threshold, or spikes in a small smoke case.

### Task 2: Quasi-static state helper

- [ ] Implement a small helper that can freeze or clamp slow-state values per neuron/bin without using W.
- [ ] Support at least:
  - fixed `z`;
  - fixed `phi` offset;
  - fixed `g_K`;
  - fixed `e_GABA` if the conductance-shunt path is available.
- [ ] Add unit tests for sign semantics:
  - lower `z` weakens inhibition;
  - higher `phi` raises threshold;
  - higher `g_K` suppresses excitability;
  - depolarized `e_GABA` makes inhibition less protective only in the shunting path.
- [ ] Add phase-coordinate / slow-to-rate mapping tests:
  - `q_global` / global `z` maps monotonically to `phase_y_global`;
  - `q_core` / core `z` maps monotonically to `phase_x_core` or a documented core-disinhibition term;
  - `phi` and `g_K` are exported as recovery/protection coordinates, not mislabeled as disinhibition;
  - unmapped variables are exported with `phase_coord_valid=false`.

### Task 3: No-kick spontaneous detector reuse

- [ ] Reuse existing event detection / R0-R4 classification where possible.
- [ ] Ensure the detector reads raw spontaneous activity, not kick-minus-sham fields.
- [ ] Ensure R4a requires sustained/recurrent recruitment with nontrivial front or spatial structure.
- [ ] Ensure R4b tonic saturation is reported but never counted as seizure-like success.

### Task 4: Tiny quasi-static sweep

Run a small, cheap pilot before any full grid:

```text
substrate: one accepted Stage-3/M3 base substrate
T: 8-20 s per seed
seeds: 3-5
states:
  e_GABA: baseline plus 3-5 depolarized levels, if available
  z: 1.0, 0.8, 0.6, 0.4
  phi: baseline plus 2-3 offsets
  g_K: 0 plus 2-3 levels
```

Only one slow variable is varied at a time. Other mechanisms remain off or baseline.

### Task 5: A1 verdict

Answer these seven questions:

1. Does event rate change?
2. Does event size distribution shift?
3. Does duration distribution shift?
4. Does return probability drop?
5. Does R-class composition change toward R4a?
6. Does the effect differ from simple rate-only heating?
7. Which slow-state coordinates are M3B-ready, with valid sign/calibration ranges?

Success requires more than event-rate increase. A minimal A1 candidate needs size/duration or return/R-class movement in addition to rate change.
M3B-readiness additionally requires `slow_to_rate_mapping.json` and `phase_coord_ranges.json` with
valid sign tests. A1 may find a biological phenotype candidate that is **not yet** M3B-ready if
its rate-field coordinate mapping is unclear.

Failure modes:

- rate increases but size/duration/R-class stay flat: repeat of static vth-mu negative result;
- only R4b appears: tonic runaway, not seizure-like bridge;
- detector threshold changes alone explain the R-class shift: invalid;
- all variables are silent or unstable: no quasi-static mechanism support.

## 6. Stop Rule

Stop after A1 tiny pilot and write a recap before A2. Do not start dynamic slow-variable runs until a slow-state range has either:

- a positive phenotype-shift candidate, or
- a clearly documented negative boundary worth testing dynamically for history effects.
