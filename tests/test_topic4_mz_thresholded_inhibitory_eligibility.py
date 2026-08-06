from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.run_topic4_mz_thresholded_inhibitory_eligibility import (
    NO_ROOT_IN_DOMAIN,
    _cycle_safety,
    _save_json,
    _status,
    classify_schedule,
    eligibility_gate,
    eligibility_trace,
    largest_edge_component,
    root_brackets,
    root_resolution_gates,
    solve_periodic_q_reserve,
    synthesize_schedule,
)


ROOT = Path(__file__).resolve().parents[1]


def test_eligibility_gate_has_registered_midpoint_and_is_monotone() -> None:
    h = np.array([0.0, 0.018, 0.020, 0.022, 0.1])
    gate = eligibility_gate(h, theta_h=0.020, width=0.002)

    assert gate[2] == pytest.approx(0.5)
    assert np.all(np.diff(gate) > 0.0)
    assert gate[0] < 1.0e-6
    assert gate[-1] > 1.0 - 1.0e-12


def test_periodic_constant_use_has_constant_h_and_exact_q_hold_solve() -> None:
    durations = np.full(8, 10.0)
    use = np.full(8, 0.2)
    h = eligibility_trace(
        durations,
        use,
        tau_h_ms=10_000.0,
        theta_h=0.020,
        gate_width=0.002,
        substeps=2,
        periodic=True,
    )

    np.testing.assert_allclose(h["h"], 0.2, rtol=0.0, atol=2.0e-13)
    assert h["map_alpha"] == pytest.approx(np.exp(-80.0 / 10_000.0))
    solved = solve_periodic_q_reserve(
        h["durations_ms"],
        h["weighted_use"],
        q_hold=0.8425,
        q_rest=0.9,
        tau_recovery_ms=20_000.0,
        tau_depletion_ms=200.0,
        integrated_returns=8,
        q_reserve_search=(0.70, 0.84249),
        tolerance=1.0e-12,
    )

    assert 0.70 < solved["q_reserve"] < 0.8425
    assert solved["q_mean"] == pytest.approx(0.8425, abs=1.0e-12)
    assert solved["closure_error"] < 1.0e-12


def test_synthesized_schedule_and_locked_entry_classification() -> None:
    template = np.array([0.0, 0.4, 0.2, 0.0])
    time, use = synthesize_schedule(
        [2.0, 7.0], template, dt_ms=1.0, stop_after_last_ms=4.0
    )

    assert time[-1] == 11.0
    np.testing.assert_allclose(use[2:6], template)
    np.testing.assert_allclose(use[7:11], template)

    classified = classify_schedule(
        {
            "time_ms": np.array([0.0, 10.0, 20.0, 29.0, 30.0, 31.0, 40.0]),
            "q": np.array([0.90, 0.88, 0.87, 0.87, 0.86, 0.84, 0.84]),
        },
        [10.0, 20.0, 30.0],
        entry_fold_q=0.855,
        final_target_q=0.850,
        pre_last_margin_q=0.005,
    )

    assert classified["outcome"] == "target_entry_event_3"
    assert classified["locked_last_event_pass"] is True


def test_root_and_edge_adjacency_contracts() -> None:
    assert root_brackets([1.0, 0.4, -0.2, -0.1, 0.3]) == [(1, 2), (3, 4)]
    assert root_brackets([1.0, 0.0, -1.0]) == [(1, 1)]
    assert largest_edge_component([(0, 0), (0, 1), (2, 2)]) == 2
    assert largest_edge_component([(0, 0), (1, 1)]) == 1


def test_cycle_safety_uses_locked_q_multiplier_key() -> None:
    cfg = {
        "gates": {
            "minimum_periodic_q": 0.8325,
            "maximum_periodic_q": 0.85,
            "maximum_abs_periodic_mean_error_q": 0.00125,
            "maximum_q_per_cycle_multiplier": 0.90,
        }
    }
    result = {
        "q_min": 0.84,
        "q_max": 0.845,
        "q_mean_residual": 0.0,
        "per_cycle_multiplier": 0.89,
        "closure_error": 0.0,
    }

    assert _cycle_safety(result, cfg) is True
    result["per_cycle_multiplier"] = 0.90
    assert _cycle_safety(result, cfg) is False


def test_supported_status_never_claims_an_autonomous_lifecycle() -> None:
    supported = _status(True, True)
    assert supported.endswith("SHORT_COUPLED_ARM_ONLY")
    assert "AUTONOMOUS" not in supported
    assert _status(False, True).endswith("CLEAN_NO_GO_REGISTERED_ROBUSTNESS_GATES")
    assert _status(False, False).endswith("NUMERICALLY_UNRESOLVED")


def test_no_root_cell_is_resolved_and_does_not_fail_found_root_validity() -> None:
    rows = [
        {
            "root_found": True,
            "root_scan_monotone": True,
            "root_bracket_count": 1,
            "numeric_error": False,
        },
        {
            "root_found": False,
            "root_scan_monotone": True,
            "root_bracket_count": 0,
            "numeric_error": False,
            "calibration_status": NO_ROOT_IN_DOMAIN,
        },
    ]
    gates = root_resolution_gates(
        rows, observed_scan_rows=192, expected_scan_rows=192
    )

    assert all(gates.values())
    rows[1]["numeric_error"] = True
    assert root_resolution_gates(
        rows, observed_scan_rows=192, expected_scan_rows=192
    )["no_numeric_calibration_errors"] is False


def test_strict_json_writer_uses_null_and_rejects_nan(tmp_path: Path) -> None:
    path = tmp_path / "strict.json"
    _save_json(path, {"missing": None, "value": 1.0})
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "missing": None,
        "value": 1.0,
    }
    with pytest.raises(ValueError, match="Out of range float"):
        _save_json(path, {"invalid": np.nan})


def test_canonical_threshold_artifact_is_complete_strict_clean_no_go() -> None:
    result = ROOT / "results/topic4_sef_hfo/mz_thresholded_inhibitory_eligibility"
    summary_text = (result / "thresholded_eligibility_summary.json").read_text(encoding="utf-8")

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant: {value}")

    summary = json.loads(summary_text, parse_constant=reject_constant)
    assert summary["status"].endswith("CLEAN_NO_GO_REGISTERED_ROBUSTNESS_GATES")
    assert summary["root_scan_row_count"] == 864
    assert summary["expected_root_scan_row_count"] == 864
    assert summary["safe_cell_count"] == 0
    assert summary["theta_sensitivity_pass_cell_count"] == 0
    assert summary["numeric_error_cell_count"] == 0

    with (result / "thresholded_eligibility_root_scan.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        scan_rows = list(csv.DictReader(handle))
    assert len(scan_rows) == 864
    no_root = [
        row for row in summary["grid_cells"]
        if row["calibration_status"] == NO_ROOT_IN_DOMAIN
    ]
    assert len(no_root) == 1
    assert no_root[0]["root_scan_monotone"] is True
    assert no_root[0]["root_bracket_count"] == 0
    assert no_root[0]["numeric_error"] is False
