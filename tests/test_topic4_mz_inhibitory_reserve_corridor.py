from copy import deepcopy
from pathlib import Path

import yaml

from scripts.run_topic4_mz_inhibitory_reserve_corridor import (
    _aggregate_outcome,
    _fiber_summary,
)
from scripts.run_topic4_mz_inhibitory_reserve_corridor_r0b import _build_r0b_gates
from scripts.run_topic4_mz_inhibitory_reserve_boundary_sentinel import (
    _summarize_boundary_sentinel,
)


def _rows(q, additive, outcomes):
    return [
        {"q": q, "additive_mv": additive, "outcome": outcome}
        for outcome in outcomes
    ]


def test_aggregate_outcome_is_fail_closed_across_phases():
    assert _aggregate_outcome([{"outcome": "LLL"}] * 4) == "all_LLL"
    assert _aggregate_outcome(
        [{"outcome": "LLL"}] * 3 + [{"outcome": "physical_or_numerical_failure"}]
    ) == "any_failure"
    assert _aggregate_outcome(
        [{"outcome": "LLL"}] * 3 + [{"outcome": "bounded_CCO"}]
    ) == "phase_mixed_or_unresolved"


def test_fiber_margin_requires_a_strict_safe_point_beyond_exit():
    q = 0.84
    source = [{"q": q, "outcome": "bounded_CCO"}] * 4
    exit_rows = []
    exit_rows += _rows(q, 0.0, ["bounded_CCO"] * 4)
    exit_rows += _rows(q, 0.24, ["LLL"] * 4)
    exit_rows += _rows(q, 0.26, ["LLL"] * 4)
    row = _fiber_summary(q, {"additive_mv": 0.235}, source, exit_rows, 0.02)
    assert row["fiber_safe_discovery"]
    assert row["coarse_exit_additive_mv"] == 0.24

    exit_rows += _rows(q, 0.25, ["physical_or_numerical_failure"] * 4)
    failed = _fiber_summary(q, {"additive_mv": 0.235}, source, exit_rows, 0.02)
    assert not failed["fiber_safe_discovery"]


def _complete_r0b_rows():
    cfg = yaml.safe_load(
        Path("config/topic4_mz_inhibitory_reserve_corridor_r0b.yaml").read_text(
            encoding="utf-8"
        )
    )
    confirm_q = list(map(float, cfg["r0b"]["confirm_q_axis"]))
    stress_q = list(map(float, cfg["r0b"]["ramp_stress_q_axis"]))
    phases = list(map(float, cfg["r0b"]["relative_phase_fractions"]))
    dts = list(map(float, cfg["integration"]["dt_ms"]))
    offsets = list(map(float, cfg["r0b"]["threshold_offsets_from_low_fold_mv"]))

    def base(outcome):
        row = {"outcome": outcome}
        for patch in ("core", "annulus", "bath"):
            row[f"{patch}_support_violations"] = 0
            row[f"{patch}_bound_violations"] = 0
        return row

    source = [
        {"q": q, "phase": phase, "dt_ms": dt, **base("bounded_CCO")}
        for q in sorted(set(confirm_q + stress_q))
        for phase in phases
        for dt in dts
    ]
    step = [
        {
            "q": q,
            "phase": phase,
            "dt_ms": dt,
            "offset_from_low_fold_mv": offset,
            **base("bounded_CCO" if offset < 0.0 else "LLL"),
        }
        for q in confirm_q
        for phase in phases
        for dt in dts
        for offset in offsets
    ]
    ramp = [
        {
            "q": q,
            "phase": phase,
            "dt_ms": dt,
            "max_additive_mv": 0.30,
            "low_fold_additive_mv": 0.20,
            "max_abs_fixed_q_error": 0.0,
            "first_support_failure_ms": None,
            "first_nonfinite_ms": None,
            **base("LLL"),
        }
        for q in sorted(set(confirm_q + stress_q))
        for phase in phases
        for dt in dts
    ]
    recovery = [
        {
            "source_q": q,
            "phase": phase,
            "dt_ms": dt,
            **base("LLL"),
        }
        for q in confirm_q
        for phase in phases
        for dt in dts
    ]
    return cfg, source, step, ramp, recovery


def test_r0b_gates_require_complete_observed_cartesian_products():
    cfg, source, step, ramp, recovery = _complete_r0b_rows()
    gates, diagnostics = _build_r0b_gates(source, step, ramp, recovery, cfg)
    assert all(gates.values())
    assert all(diagnostics["cartesian_complete"].values())

    incomplete, _ = _build_r0b_gates(source, step[:-1], ramp, recovery, cfg)
    assert not incomplete["tables_form_complete_cartesian_products"]


def test_r0b_safe_strip_is_reconstructed_from_failclosed_rows():
    cfg, source, step, ramp, recovery = _complete_r0b_rows()
    corrupted = deepcopy(step)
    target = next(
        row for row in corrupted
        if row["q"] == 0.84 and row["offset_from_low_fold_mv"] == 0.025
    )
    target["core_support_violations"] = 1
    gates, diagnostics = _build_r0b_gates(
        source, corrupted, ramp, recovery, cfg
    )
    assert not gates["formal_rows_have_zero_failclosed_violations"]
    assert not gates["continuous_safe_q_strip_from_outcomes_meets_gate"]
    assert diagnostics["safe_by_q"]["0.84"] is False


def test_r0b_margin_gate_uses_registered_config_not_hardcoded_offset():
    cfg, source, step, ramp, recovery = _complete_r0b_rows()
    cfg["formal_r0_gate"]["minimum_additive_margin_mv"] = 0.03
    gates, diagnostics = _build_r0b_gates(source, step, ramp, recovery, cfg)
    assert diagnostics["registered_margin_offset_mv"] is None
    assert not gates["step_registered_margin_reaches_LLL"]


def test_boundary_sentinel_reports_confirmed_anchors_not_a_resolved_boundary():
    cfg, _, _, ramp, _ = _complete_r0b_rows()
    phases = list(map(float, cfg["r0b"]["relative_phase_fractions"]))
    dts = list(map(float, cfg["integration"]["dt_ms"]))

    def row(q, phase, dt, outcome):
        result = {"q": q, "phase": phase, "dt_ms": dt, "outcome": outcome}
        for patch in ("core", "annulus", "bath"):
            result[f"{patch}_support_violations"] = 0
            result[f"{patch}_bound_violations"] = 0
        return result

    sentinel = [
        row(q, phase, dt, "LLL")
        for q in map(float, cfg["r0b"]["lower_boundary_sentinel_q_axis"])
        for phase in phases
        for dt in dts
    ]
    known = [item for item in ramp if item["q"] == 0.825]
    for item in known:
        item["outcome"] = "physical_or_numerical_failure"
    summary = _summarize_boundary_sentinel(sentinel, known, cfg)
    assert summary["status"] == "R0B_LOWER_RAMP_CONFIRMED_ANCHOR_BRACKET"
    assert summary["highest_confirmed_failing_q"] == 0.825
    assert summary["lowest_confirmed_safe_q"] == 0.83
    assert summary["unresolved_source_q"] == 0.8275

    incomplete = _summarize_boundary_sentinel(sentinel[:-1], known, cfg)
    assert incomplete["status"] == "R0B_LOWER_RAMP_ANCHOR_BRACKET_NOT_RESOLVED"
