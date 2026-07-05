"""Shared M3A <-> M3B-R2 interface contract — single source of truth for the handoff gate.

Canonical contract doc: docs/superpowers/specs/2026-06-27-sef-hfo-m3-interface-contract.md
Contract-layer TDD:     tests/test_sef_hfo_m3_interface.py

This module is imported by BOTH the M3A exporter (writes phase_trajectory / mapping / summary) and
the M3B axis-builder + overlay (reads them) so the two lines cannot drift. Every gate condition is
fail-closed: a missing / empty / default value REFUSES the overlay, it never silently permits it.

Pure-data only: no SNN, no eigensolver, no pandas. Artifacts are dicts (JSON) and list[dict] (CSV
rows). The four-condition overlay gate lives in `audit_m3a_interface` / `compute_overlay_verdict`.

Origin: 2026-06-27 4-lens adversarial review found 8 fail-open blockers in the prior interface;
this module closes them (see contract doc §8 blocker->fix map).
"""
from __future__ import annotations

import math
from typing import Iterable, Optional

# ---------------------------------------------------------------------------
# Canonical coordinates / enums (contract doc §2, §3)
# ---------------------------------------------------------------------------
PHASE_COORDS = ("phase_x_core", "phase_y_global", "phase_recovery")
ON_AXIS_COORDS = ("phase_x_core", "phase_y_global")          # D1: the two normalized phase-map axes
PROJECTED_OUT_COORDS = ("phase_recovery",)                   # D3: recovery is lossless-carried, off-axis

CANONICAL_EVENT_STAGES = frozenset({
    "baseline", "pre", "onset", "peak", "end",
    "post_50ms", "post_200ms", "post_1s", "post", "inter_event",
})

GATE_A_VALUES = frozenset({"PASS", "FAIL", "INCONCLUSIVE"})
GATE_B_VALUES = frozenset({"PASS", "FAIL", "INCONCLUSIVE"})
RATE_MATCHED_VALUES = frozenset({"passed", "failed", "not_run"})
CALIBRATION_STATUS_VALUES = frozenset({"passed", "failed", "not_applicable"})
TRAJECTORY_ROBUSTNESS_VALUES = frozenset(
    {"robust", "seed_fragile", "runaway_prone", "quiet_prone", "not_tested"})
OVERLAY_VERDICT_VALUES = frozenset(
    {"phase_map_trajectory", "mechanism_candidate_only", "refused"})
TRANSFORM_TYPES = frozenset({"identity", "affine", "reciprocal_affine"})
EXPECTED_DIRECTIONS = frozenset({"increasing_in_input", "decreasing_in_input"})
AXIS_SPACE_VALUES = frozenset({"normalized_unit"})           # D1
R_CLASS_VALUES = ("R0", "R1", "R2", "R3", "R4a", "R4b")
PHENOTYPE_LABELS = frozenset({
    "local_axial", "larger_axial", "mixed_global",
    "global_recruitment", "runaway", "recovery"})
MODE_CLASSES = frozenset({
    "stable", "local", "axial", "mixed", "global", "runaway", "unresolved"})

OUT_OF_RANGE_FRACTION_MAX = 0.05    # D2
NA_SENTINEL = "NA"

# Variable role sets (contract doc §4: recovery vars must not sit on a disinhibition axis)
RECOVERY_VARS = frozenset({"phi", "g_K", "gK", "x_EE"})

MIN_COLUMNS = (
    "time_ms", "event_id", "event_stage",
    "phase_x_core", "phase_y_global", "phase_recovery",
    "phase_coord_valid", "phase_coord_out_of_range",
    "slow_to_rate_mapping_id", "R_class", "return_to_baseline",
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def is_na(x) -> bool:
    """NA sentinel: explicit "NA" string, None, or float NaN (never 0.0)."""
    if x is None:
        return True
    if isinstance(x, str):
        return x == NA_SENTINEL
    if isinstance(x, float):
        return math.isnan(x)
    return False


def canonical_event_stages() -> frozenset:
    return CANONICAL_EVENT_STAGES


def gate_enums() -> dict:
    return {
        "gate_A_trajectory": GATE_A_VALUES,
        "gate_B_seizure_like": GATE_B_VALUES,
        "trajectory_robustness": TRAJECTORY_ROBUSTNESS_VALUES,
        "rate_matched_control": RATE_MATCHED_VALUES,
    }


def required_min_columns() -> tuple:
    return MIN_COLUMNS


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


# ---------------------------------------------------------------------------
# Transform evaluation (closed enum, NO eval of free-text formulas) — contract §4, B8/M9
# ---------------------------------------------------------------------------
def _apply_transform(t: dict, value: float) -> float:
    ttype = t["type"]
    a = float(t.get("a", 1.0))
    b = float(t.get("b", 0.0))
    if ttype == "identity":
        out = float(value)
    elif ttype == "affine":
        out = a * float(value) + b
    elif ttype == "reciprocal_affine":
        out = a / float(value) + b
    else:
        raise ValueError(f"unknown transform.type {ttype!r}; allowed {sorted(TRANSFORM_TYPES)}")
    lo, hi = t.get("clip", [0.0, 1.0])
    return float(min(max(out, lo), hi))


def evaluate_phase_coord(mapping: dict, coord: str, slow_values: dict) -> float:
    """The single deterministic slow-value -> normalized [0,1] coordinate transform.

    Imported by BOTH the M3A exporter and the M3B axis-builder so they cannot diverge (B8).
    """
    t = mapping["coordinates"][coord]["transform"]
    return _apply_transform(t, slow_values[t["input_var"]])


def check_sign_direction(mapping: dict, coord: str, n: int = 21) -> bool:
    """Signed-slope test (NOT mere monotonicity, B4): evaluating the declared transform over the
    calibrated input domain, the coordinate must be STRICTLY monotone in `expected_direction`."""
    c = mapping["coordinates"][coord]
    t = c["transform"]
    lo, hi = float(t["input_min"]), float(t["input_max"])
    if hi <= lo:
        return False
    xs = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    ys = [_apply_transform(t, x) for x in xs]
    diffs = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
    direction = t["expected_direction"]
    eps = 1e-9
    if direction == "increasing_in_input":
        return all(d > eps for d in diffs)
    if direction == "decreasing_in_input":
        return all(d < -eps for d in diffs)
    return False


# ---------------------------------------------------------------------------
# slow_to_rate_mapping.json validation + fail-closed sign predicate (contract §4; B3,B4,M6,M7,M8,M9)
# ---------------------------------------------------------------------------
_COORD_REQUIRED = ("transform", "units", "valid_range", "variables",
                   "calibration_status", "sign_tests")
_TRANSFORM_REQUIRED = ("type", "input_var", "clip", "input_min", "input_max", "expected_direction")
_SIGN_TEST_REQUIRED = ("name", "coord", "input_var", "expected_direction",
                       "observed_slope_sign", "passed", "engine_sha")


def validate_slow_to_rate_mapping(mapping: dict) -> None:
    _require(isinstance(mapping, dict), "mapping must be a dict")
    for k in ("slow_to_rate_mapping_id", "axis_space", "coordinates"):
        _require(k in mapping, f"mapping missing required key {k!r}")
    _require(mapping["axis_space"] in AXIS_SPACE_VALUES,
             f"axis_space must be one of {sorted(AXIS_SPACE_VALUES)} (D1)")
    coords = mapping["coordinates"]
    for coord in PHASE_COORDS:
        _require(coord in coords, f"coordinates missing {coord!r}")
        c = coords[coord]
        for k in _COORD_REQUIRED:
            _require(k in c, f"{coord} missing required key {k!r}")
        t = c["transform"]
        for k in _TRANSFORM_REQUIRED:
            _require(k in t, f"{coord}.transform missing {k!r}")
        _require(t["type"] in TRANSFORM_TYPES,
                 f"{coord}.transform.type {t['type']!r} not in {sorted(TRANSFORM_TYPES)} "
                 "(closed enum — no free-text formula is evaluated)")
        _require(t["expected_direction"] in EXPECTED_DIRECTIONS,
                 f"{coord}.transform.expected_direction invalid")
        _require(len(c["valid_range"]) == 2, f"{coord}.valid_range must be [min,max]")
        _require(c["calibration_status"] in CALIBRATION_STATUS_VALUES,
                 f"{coord}.calibration_status invalid")
        # role check: a recovery variable must not sit on a disinhibition/excitability axis (M6/M7)
        if coord in ON_AXIS_COORDS:
            bad = set(c["variables"]) & RECOVERY_VARS
            _require(not bad, f"{coord} (disinhibition axis) carries recovery vars {sorted(bad)}; "
                              "recovery vars belong on phase_recovery with a suppressive sign")
        else:  # phase_recovery: only recovery vars
            bad = set(c["variables"]) - RECOVERY_VARS
            _require(not bad, f"phase_recovery carries non-recovery vars {sorted(bad)}")
        # e_GABA disinhibition is only trustworthy in the active shunt path (M8)
        if "e_GABA" in c["variables"]:
            _require("shunt_path_active" in c,
                     f"{coord} uses e_GABA but does not record shunt_path_active")
            _require(not (c["calibration_status"] == "passed" and not c["shunt_path_active"]),
                     f"{coord} claims calibrated e_GABA disinhibition while shunt_path_active=False")
        # calibration_status=='passed' demands non-empty, all-pass sign_tests (B3)
        if c["calibration_status"] == "passed":
            st = c["sign_tests"]
            _require(len(st) > 0,
                     f"{coord}.calibration_status=='passed' but sign_tests is empty (fail-open, B3)")
            for s in st:
                for k in _SIGN_TEST_REQUIRED:
                    _require(k in s, f"{coord} sign_test missing {k!r}")
                _require(s["passed"] is True, f"{coord}.calibration_status=='passed' but a sign_test failed")
            # signed-direction must actually hold for a 'passed' axis (B4)
            _require(check_sign_direction(mapping, coord),
                     f"{coord} declares calibration passed but transform is not strictly "
                     f"{t['expected_direction']} over its input domain (sign-flip, B4)")


def mapping_sign_tests_passed(mapping: Optional[dict], coord: Optional[str] = None) -> bool:
    """Fail-closed predicate. coord=None -> all plotted (on-axis) coords.

    True only if: calibration_status=='passed' AND sign_tests non-empty all-pass AND the signed
    direction actually holds AND (if e_GABA) the shunt path is active. Anything missing/None -> False.
    """
    if not isinstance(mapping, dict):
        return False
    coords = [coord] if coord is not None else list(ON_AXIS_COORDS)
    for cn in coords:
        try:
            c = mapping["coordinates"][cn]
        except (KeyError, TypeError):
            return False
        if c.get("calibration_status") != "passed":
            return False
        st = c.get("sign_tests") or []
        if len(st) == 0 or not all(s.get("passed") is True for s in st):
            return False
        if "e_GABA" in c.get("variables", []) and not c.get("shunt_path_active", False):
            return False
        try:
            if not check_sign_direction(mapping, cn):
                return False
        except (KeyError, TypeError, ZeroDivisionError):
            return False
    return True


# ---------------------------------------------------------------------------
# out-of-range (input domain AND output range, M16) + ranges validation
# ---------------------------------------------------------------------------
def coord_out_of_range(mapping: dict, ranges: dict, coord: str, slow_values: dict) -> bool:
    t = mapping["coordinates"][coord]["transform"]
    v = float(slow_values[t["input_var"]])
    if v < float(t["input_min"]) or v > float(t["input_max"]):
        return True   # input extrapolates beyond calibrated sweep (even if clipped output lands in [0,1])
    out = _apply_transform(t, v)
    rng = ranges[coord]
    return out < float(rng["min"]) or out > float(rng["max"])


def validate_phase_coord_ranges(ranges: dict, mapping_id: Optional[str] = None) -> None:
    _require("slow_to_rate_mapping_id" in ranges, "ranges missing slow_to_rate_mapping_id")
    if mapping_id is not None:
        _require(ranges["slow_to_rate_mapping_id"] == mapping_id,
                 "phase_coord_ranges.slow_to_rate_mapping_id must equal the mapping id")
    for coord in PHASE_COORDS:
        _require(coord in ranges, f"ranges missing {coord!r}")
        for k in ("min", "max", "source"):
            _require(k in ranges[coord], f"ranges[{coord}] missing {k!r}")


def sample_phase_coord_valid(mapping: dict, ranges: dict, slow_values: dict,
                             axes_used: Iterable[str]) -> bool:
    """phase_coord_valid := AND over the axes the sample uses of (calibration passed AND sign tests
    passed). Calibration gates validity; range is ORTHOGONAL (out_of_range does not force invalid)."""
    return all(mapping_sign_tests_passed(mapping, ax) for ax in axes_used)


# ---------------------------------------------------------------------------
# mapping_id consistency across all artifacts (B5)
# ---------------------------------------------------------------------------
def _extract_id(artifact):
    if isinstance(artifact, dict):
        if "slow_to_rate_mapping_id" in artifact:
            return artifact["slow_to_rate_mapping_id"]
        if "axes_built_from_slow_to_rate_mapping_id" in artifact:
            return artifact["axes_built_from_slow_to_rate_mapping_id"]
        raise ValueError("artifact carries no slow_to_rate_mapping_id")
    if isinstance(artifact, list):
        ids = {r.get("slow_to_rate_mapping_id") for r in artifact}
        _require(len(ids) == 1 and None not in ids,
                 "row artifact has missing or non-uniform slow_to_rate_mapping_id")
        return next(iter(ids))
    raise ValueError(f"cannot extract mapping id from {type(artifact)}")


def assert_mapping_id_consistent(*artifacts) -> str:
    ids = [_extract_id(a) for a in artifacts]
    _require(len(set(ids)) == 1,
             f"slow_to_rate_mapping_id differs across artifacts: {sorted(set(ids))}")
    return ids[0]


# ---------------------------------------------------------------------------
# trajectory / event-sample / summary schema (contract §5; M2,M3,M4,m2)
# ---------------------------------------------------------------------------
_TRAJ_REQUIRED = ("time_ms", "event_id", "event_stage",
                  "phase_x_core", "phase_y_global", "phase_recovery",
                  "phase_coord_valid", "phase_coord_out_of_range", "slow_to_rate_mapping_id")


def validate_phase_trajectory(rows, two_core: bool = False, mapping: Optional[dict] = None) -> None:
    _require(isinstance(rows, list) and len(rows) > 0, "phase_trajectory must be a non-empty list")
    for r in rows:
        for k in _TRAJ_REQUIRED:
            _require(k in r, f"phase_trajectory row missing required column {k!r}")
        _require(r["event_stage"] in CANONICAL_EVENT_STAGES,
                 f"event_stage {r['event_stage']!r} not in canonical enum")
    if two_core:
        _require(mapping is not None and "two_core_reduction" in mapping,
                 "two-core substrate requires mapping.two_core_reduction (M19)")
        for r in rows:
            for k in ("q_core_L", "q_core_R"):
                _require(k in r, f"two-core trajectory row missing {k!r}")


_EVT_REQUIRED = ("event_id", "event_stage",
                 "phase_x_core", "phase_y_global", "phase_recovery",
                 "phase_coord_valid", "phase_coord_out_of_range", "slow_to_rate_mapping_id",
                 "return_to_baseline", "tail_to_baseline_ratio", "R_class")


def validate_event_phase_samples(rows) -> None:
    _require(isinstance(rows, list) and len(rows) > 0, "event_phase_samples must be a non-empty list")
    for r in rows:
        for k in _EVT_REQUIRED:
            _require(k in r, f"event_phase_samples row missing required column {k!r} "
                             "(note canonical name is return_to_baseline, not 'returned')")
        _require(r["event_stage"] in CANONICAL_EVENT_STAGES, "event_stage not canonical")
        _require(r["R_class"] in R_CLASS_VALUES, f"R_class {r['R_class']!r} invalid")


def validate_dynamic_slowvars_summary(summary: dict) -> None:
    req = ("slow_to_rate_mapping_id", "gate_A_trajectory", "gate_B_seizure_like",
           "trajectory_robustness", "rate_matched_control", "out_of_range_fraction",
           "forbidden_claims")
    for k in req:
        _require(k in summary, f"summary missing required key {k!r}")
    _require(summary["gate_A_trajectory"] in GATE_A_VALUES, "gate_A_trajectory not in enum")
    _require(summary["gate_B_seizure_like"] in GATE_B_VALUES, "gate_B_seizure_like not in enum")
    _require(summary["trajectory_robustness"] in TRAJECTORY_ROBUSTNESS_VALUES,
             "trajectory_robustness not in enum")
    _require(summary["rate_matched_control"] in RATE_MATCHED_VALUES,
             "rate_matched_control not in enum")
    # claiming Gate-A requires the rate-matched control to have actually passed + the group recorded
    if summary["gate_A_trajectory"] == "PASS":
        _require(summary["rate_matched_control"] == "passed",
                 "gate_A_trajectory==PASS requires rate_matched_control=='passed'")
        _require("rate_matched_group" in summary,
                 "gate_A_trajectory==PASS requires rate_matched_group to be recorded")


def assert_disabled_mechanisms_na(row: dict, disabled_vars, derived: Optional[dict] = None) -> None:
    """A disabled mechanism writes NA (never 0.0); a derived coord with ALL contributors disabled
    is NA too (M12). 0.0 is the silent fail-open value and is rejected."""
    disabled = set(disabled_vars)
    for v in disabled:
        if v in row:
            _require(is_na(row[v]), f"disabled mechanism {v!r} must be NA, not {row[v]!r}")
    for coord, contributors in (derived or {}).items():
        if set(contributors) <= disabled and coord in row:
            _require(is_na(row[coord]),
                     f"derived {coord!r} has all contributors disabled -> must be NA, not {row[coord]!r}")


# ---------------------------------------------------------------------------
# The overlay gate (contract §6; B1,B2,B6,B7,M13,M14,M15)
# ---------------------------------------------------------------------------
def compute_overlay_verdict(cond1, cond2, cond3, cond4) -> str:
    """Pure fail-closed verdict. None/missing -> False.

    all four True            -> phase_map_trajectory
    cond4 True, others not all-> mechanism_candidate_only (phenotype real, calibration/provenance not)
    else (no phenotype move)  -> refused
    """
    c1, c2, c3, c4 = (bool(x) for x in (cond1, cond2, cond3, cond4))
    if c1 and c2 and c3 and c4:
        return "phase_map_trajectory"
    if c4 and not (c1 and c2 and c3):
        return "mechanism_candidate_only"
    return "refused"


def _out_of_range_fraction(rows) -> Optional[float]:
    """Out-of-range fraction for reporting. None if rows empty or any row lacks the flag (fail-closed)."""
    if not rows:
        return None
    flags = []
    for r in rows:
        if not isinstance(r, dict) or "phase_coord_out_of_range" not in r:
            return None   # malformed row / missing flag -> fail closed
        flags.append(bool(r["phase_coord_out_of_range"]))
    return sum(flags) / len(flags)


def _mapping_ok(mapping) -> bool:
    """cond1: the mapping is schema-valid (axis_space membership F5, recovery-var roles F6, e_GABA
    shunt gating, signed direction on ALL coords F4) AND every on-axis coord passes its sign tests."""
    if not isinstance(mapping, dict):
        return False
    try:
        validate_slow_to_rate_mapping(mapping)
    except (ValueError, TypeError, KeyError):   # type-confused values must fail closed, not crash the audit
        return False
    return mapping_sign_tests_passed(mapping, None)


def _same_mapping_and_ranges(mapping, ranges, trajectory_rows, axes_meta) -> bool:
    """cond2: ranges schema-valid (F7) AND id triple-equality AND identical axis_space (a D1 member,
    F5) + transform descriptors (B5/B8). id match alone is necessary-not-sufficient."""
    try:
        mid = mapping["slow_to_rate_mapping_id"]
        validate_phase_coord_ranges(ranges, mapping_id=mid)             # F7
        if mapping["axis_space"] not in AXIS_SPACE_VALUES:              # F5
            return False
        if {r["slow_to_rate_mapping_id"] for r in trajectory_rows} != {mid}:
            return False
        if axes_meta["axes_built_from_slow_to_rate_mapping_id"] != mid:
            return False
        if axes_meta["axis_space"] != mapping["axis_space"]:
            return False
        for c in ON_AXIS_COORDS:
            if axes_meta["axis_transforms"][c] != mapping["coordinates"][c]["transform"]:
                return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def _trajectory_refusal_ok(rows) -> bool:
    """cond3: trajectory is schema-valid, EVERY row is phase_coord_valid (invalid samples count as
    refused, not silently dropped — contract §5 M1 / F9), and out-of-range fraction <= D2 (B7)."""
    try:
        validate_phase_trajectory(rows)
    except (ValueError, TypeError, KeyError):   # type-confused rows must fail closed, not crash the audit
        return False
    if not rows:
        return False
    if any(not bool(r["phase_coord_valid"]) for r in rows):
        return False
    out = sum(bool(r["phase_coord_out_of_range"]) for r in rows)
    return (out / len(rows)) <= OUT_OF_RANGE_FRACTION_MAX


def _summary_phenotype_ok(summary) -> bool:
    """cond4: summary is schema-valid (F8) AND (STRICT, A2-only) rate_matched_control=='passed' AND
    gate_A_trajectory=='PASS' (resolution-level trap, B6)."""
    if not isinstance(summary, dict):
        return False
    try:
        validate_dynamic_slowvars_summary(summary)
    except (ValueError, TypeError, KeyError):   # type-confused summary must fail closed, not crash the audit
        return False
    return summary.get("rate_matched_control") == "passed" \
        and summary.get("gate_A_trajectory") == "PASS"


def audit_m3a_interface(*, mapping, ranges, trajectory_rows, summary, axes_meta) -> dict:
    """Compute the four overlay conditions and the verdict, WIRING IN the full schema validators so a
    malformed input fails CLOSED (drives its condition to False) instead of being silently stamped.
    Always returns an audit dict; overlay artifacts are built only when overlay_verdict ==
    'phase_map_trajectory' (B2). The D2 tolerance is fixed — no override parameter (F10)."""
    cond1 = _mapping_ok(mapping)                                          # B3/B4/F4/F5/F6
    cond2 = _same_mapping_and_ranges(mapping, ranges, trajectory_rows, axes_meta)   # B5/B8/F7
    cond3 = _trajectory_refusal_ok(trajectory_rows)                       # B7/F1/F9
    cond4 = _summary_phenotype_ok(summary)                                # B6/F8
    frac = _out_of_range_fraction(trajectory_rows)
    verdict = compute_overlay_verdict(cond1, cond2, cond3, cond4)
    return {
        "audited_slow_to_rate_mapping_id": mapping["slow_to_rate_mapping_id"]
        if isinstance(mapping, dict) and "slow_to_rate_mapping_id" in mapping else None,
        "cond1_sign_tests_passed": cond1,
        "cond1_source": "slow_to_rate_mapping.json",
        "cond2_same_mapping_and_ranges": cond2,
        "cond2_source": "finite_jacobian_grid.axes_built_from_slow_to_rate_mapping_id",
        "cond3_in_range_or_flagged": cond3,
        "cond3_out_of_range_fraction": frac,
        "cond4_phenotype_movement_beyond_rate": cond4,
        "cond4_source": "dynamic_slowvars_summary.json",
        "on_axis_coords": list(ON_AXIS_COORDS),
        "projected_out_coords": list(PROJECTED_OUT_COORDS),
        "gate_used": "A",
        "overlay_verdict": verdict,
        "overlay_allowed": verdict == "phase_map_trajectory",
    }


_AUDIT_REQUIRED = (
    "audited_slow_to_rate_mapping_id",
    "cond1_sign_tests_passed", "cond1_source",
    "cond2_same_mapping_and_ranges", "cond2_source",
    "cond3_in_range_or_flagged", "cond3_out_of_range_fraction",
    "cond4_phenotype_movement_beyond_rate", "cond4_source",
    "on_axis_coords", "projected_out_coords", "gate_used",
    "overlay_verdict", "overlay_allowed",
)


def validate_interface_audit(audit: dict) -> None:
    for k in _AUDIT_REQUIRED:
        _require(k in audit, f"m3a_interface_audit missing required key {k!r} "
                             "(a missing condition must NOT default to true, B1)")
    for k in ("cond1_sign_tests_passed", "cond2_same_mapping_and_ranges",
              "cond3_in_range_or_flagged", "cond4_phenotype_movement_beyond_rate", "overlay_allowed"):
        _require(isinstance(audit[k], bool), f"{k} must be an explicit bool")
    _require(audit["overlay_verdict"] in OVERLAY_VERDICT_VALUES, "overlay_verdict not in enum")
    _require(audit["overlay_allowed"] == (audit["overlay_verdict"] == "phase_map_trajectory"),
             "overlay_allowed must equal (overlay_verdict == 'phase_map_trajectory')")
    _require(audit["gate_used"] == "A", "overlay is Gate-A tier only (M13)")


# ---------------------------------------------------------------------------
# overlay output (contract §6.1; B2,M18) + min-columns join (M5)
# ---------------------------------------------------------------------------
def build_slow_trajectory_overlay(trajectory_rows, audit, readout_fn=None) -> list:
    """Build slow_trajectory_overlay.csv rows, BOUND to the audited artifact. Returns [] (no claim
    drawn) unless the audit is schema-valid AND overlay_allowed. Raises if the rows handed in do not
    match the audited mapping, contain an invalid sample, or exceed the D2 gate — the builder
    re-checks what it actually DRAWS rather than trusting it equals the audited rows (TOCTOU, F2/F3).
    Required phase coords are hard-indexed, never NA-defaulted (F1). All three phase coords carried
    (lossless, D3) + flags + the phase-map readout at the point."""
    validate_interface_audit(audit)                                  # F2: forged/incomplete audit refuses
    if not audit["overlay_allowed"]:
        return []                                                    # structural no-claim (M14)
    validate_phase_trajectory(trajectory_rows)                       # F1: required coords are hard
    audited_id = audit["audited_slow_to_rate_mapping_id"]
    _require({r["slow_to_rate_mapping_id"] for r in trajectory_rows} == {audited_id},
             "overlay rows carry a slow_to_rate_mapping_id != the audited mapping (TOCTOU, F3)")
    _require(not any(not bool(r["phase_coord_valid"]) for r in trajectory_rows),
             "overlay rows contain an invalid sample (phase_coord_valid False)")
    out_frac = sum(bool(r["phase_coord_out_of_range"]) for r in trajectory_rows) / len(trajectory_rows)
    _require(out_frac <= OUT_OF_RANGE_FRACTION_MAX,
             f"overlay rows out-of-range fraction {out_frac} exceeds D2 {OUT_OF_RANGE_FRACTION_MAX}")
    out = []
    for r in trajectory_rows:
        mode_class, alpha_1 = readout_fn(r) if readout_fn else ("unresolved", float("nan"))
        row = {k: r[k] for k in _TRAJ_REQUIRED}                      # required coords: hard-indexed (F1)
        for k in ("R_class", "return_to_baseline"):
            row[k] = r.get(k, NA_SENTINEL)                           # per-event fields legitimately absent on a trajectory row
        row["in_map"] = not bool(r["phase_coord_out_of_range"])
        row["leading_mode_class"] = mode_class
        row["alpha_1"] = alpha_1
        out.append(row)
    return out


def join_trajectory_and_event_samples(traj_rows, evt_rows) -> list:
    """Resolve the min-columns record by joining trajectory ⋈ event_phase_samples on event_id
    (sentinel event_id == -1 / NA for inter-event rows; per-event fields are NA there) (M5)."""
    evt_by_id = {r["event_id"]: r for r in evt_rows}
    merged = []
    for r in traj_rows:
        row = dict(r)
        eid = r.get("event_id")
        ev = evt_by_id.get(eid) if (eid is not None and not is_na(eid) and eid != -1) else None
        for k in ("R_class", "return_to_baseline"):
            row[k] = ev[k] if ev is not None else NA_SENTINEL
        for k in MIN_COLUMNS:
            row.setdefault(k, NA_SENTINEL)
        merged.append(row)
    return merged


# ---------------------------------------------------------------------------
# m3b_ready (M15), tier guard (M13), classification crosswalk (M11)
# ---------------------------------------------------------------------------
def m3b_ready(summary: dict, mapping: Optional[dict]) -> tuple:
    """Necessary-not-sufficient producer-side readiness flag: gate_A PASS AND all on-axis coords
    calibrated AND rate-matched control passed. M3B still independently re-checks cond2/cond3."""
    reasons = []
    if summary.get("gate_A_trajectory") != "PASS":
        reasons.append("gate_A_trajectory != PASS")
    if summary.get("rate_matched_control") != "passed":
        reasons.append("rate_matched_control != passed")
    if not (isinstance(mapping, dict) and all(
            mapping.get("coordinates", {}).get(c, {}).get("calibration_status") == "passed"
            for c in ON_AXIS_COORDS)):
        reasons.append("an on-axis coordinate is not calibrated")
    return (len(reasons) == 0, "; ".join(reasons) if reasons else "ready")


def overlay_is_seizure_like_claim(audit: dict, summary: dict) -> bool:
    """An overlay is a Gate-A trajectory artifact. A seizure-like (Gate-B) claim is licensed ONLY
    when gate_B_seizure_like == 'PASS' — the overlay alone never licenses it (M13 tier guard)."""
    return summary.get("gate_B_seizure_like") == "PASS"


def classification_crosswalk() -> dict:
    """Documented crosswalk between the three classification vocabularies (M11). Rows pending
    confirmation (contract doc §9) — used to keep R_class / phenotype_label / mode_class from being
    silently conflated across the handoff."""
    r_to = {
        "R0":  {"phenotype_label": "recovery",           "mode_class": "stable"},
        "R1":  {"phenotype_label": "local_axial",        "mode_class": "local"},
        "R2":  {"phenotype_label": "local_axial",        "mode_class": "local"},
        "R3":  {"phenotype_label": "larger_axial",       "mode_class": "axial"},
        "R4a": {"phenotype_label": "mixed_global",       "mode_class": "mixed"},
        "R4b": {"phenotype_label": "runaway",            "mode_class": "runaway"},
    }
    return {
        "R_class": r_to,
        "phenotype_label": sorted(PHENOTYPE_LABELS),
        "mode_class": sorted(MODE_CLASSES),
    }
