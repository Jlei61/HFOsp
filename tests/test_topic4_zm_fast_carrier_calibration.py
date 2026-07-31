"""Calibration invariants for the Phase-D baseline-only screen."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from src import topic4_zm_fast_carrier_calibration as C


def _row(scales, **overrides):
    row = {
        "scale_E": scales[0],
        "scale_I": scales[1],
        "scale_M": scales[2],
        "data_scope": "pre_entry_only",
        "baseline_reference_sha256": "a" * 64,
        "median_e_rate_ratio": 1.0,
        "returning_event_count_ratio": 1.0,
        "returning_event_count": 10,
        "event_order_preserved": True,
        "two_source_geometry_readable": True,
        "vinf_error_mv": 0.0,
        "charge_ratio_relative_error": 0.0,
        "tau_eff_ratio": 0.8,
        "prevention": False,
        "whole_sheet_plateau": False,
    }
    row.update(overrides)
    return row


def test_reference_anchor_uses_only_free_E_and_records_crossing():
    state = {
        "V": np.array([8.0, 10.0, 12.0, 14.0, 5.0, 7.0]),
        "ref": np.array([0, 0, 2, 0, 0, 0]),
    }
    out = C.build_reference_anchor(
        state, n_e=4, v_th_median=18.0, v_reset=11.0, eta_m=0.001
    )
    assert out["n_free_e"] == 3
    assert out["candidate_outcomes_accessed"] is False
    assert out["diagnostics"]["fraction_V_above_EI"] == pytest.approx(1 / 3)
    assert len(out["scale_lattice"]) == 27


def test_candidate_config_applies_only_locked_literal_scales():
    state = {"V": np.array([8.0, 10.0, 12.0]), "ref": np.zeros(3)}
    ref = C.build_reference_anchor(
        state, n_e=3, v_th_median=18.0, v_reset=11.0, eta_m=0.001
    )
    cfg = C.candidate_config(ref, (0.8, 1.2, 1.0))
    assert cfg["kappa_E"] == pytest.approx(0.8 * ref["base_config"]["kappa_E"])
    assert cfg["kappa_I"] == pytest.approx(1.2 * ref["base_config"]["kappa_I"])
    with pytest.raises(C.CalibrationError, match="outside lock"):
        C.candidate_config(ref, (0.7, 1.0, 1.0))


def test_selector_is_lexicographic_before_distance_tie_break():
    rows = [_row(scales, median_e_rate_ratio=1.10) for scales in C.scale_lattice()]
    near = rows[C.scale_lattice().index((1.0, 1.0, 1.0))]
    near["median_e_rate_ratio"] = 1.05
    far = rows[C.scale_lattice().index((0.8, 1.2, 0.8))]
    far["median_e_rate_ratio"] = 1.01
    out = C.select_calibration(list(reversed(rows)))
    assert out["verdict"] == "baseline_calibration_passed"
    assert out["selected_scales"] == [0.8, 1.2, 0.8]


@pytest.mark.parametrize(
    "override, reason",
    [
        ({"returning_event_count": 0, "prevention": True}, "returning_events_prevented"),
        ({"whole_sheet_plateau": True}, "baseline_whole_sheet_plateau"),
        ({"vinf_error_mv": 0.51}, "vinf_error_over_0p5mV"),
        ({"charge_ratio_relative_error": 0.16}, "charge_ratio_outside_15pct"),
        ({"tau_eff_ratio": 0.2}, "tau_eff_ratio_outside_0p25_1p0"),
        ({"event_order_preserved": False}, "event_order_lost"),
    ],
)
def test_hard_constraints_reject_false_baseline_preservation(override, reason):
    out = C.adjudicate_row(_row((1.0, 1.0, 1.0), **override))
    assert out["valid"] is False
    assert reason in out["reasons"]


def test_selector_fails_closed_on_no_solution():
    rows = [_row(scales, prevention=True, returning_event_count=0) for scales in C.scale_lattice()]
    out = C.select_calibration(rows)
    assert out["verdict"] == "NO_GO_baseline_calibration_failed"
    assert out["selected_scales"] is None


def test_incomplete_duplicate_or_candidate_leak_fails_closed():
    rows = [_row(scales) for scales in C.scale_lattice()]
    with pytest.raises(C.CalibrationError, match="incomplete"):
        C.select_calibration(rows[:-1])
    duplicated = copy.deepcopy(rows)
    duplicated[-1] = copy.deepcopy(duplicated[0])
    with pytest.raises(C.CalibrationError, match="duplicate"):
        C.select_calibration(duplicated)
    leaked = copy.deepcopy(rows)
    leaked[0]["data_scope"] = "bounded_mid"
    with pytest.raises(C.CalibrationError, match="leaked"):
        C.select_calibration(leaked)
