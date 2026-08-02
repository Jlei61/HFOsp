import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


R = _load("topic4_zm_control_panel", "scripts/run_topic4_zm_lifecycle_control_panel.py")
A = _load("topic4_zm_control_analysis", "scripts/analyze_topic4_zm_lifecycle_control.py")


def _candidate():
    return {
        "config_id": "c0", "arm": "combined", "tau_D_ms": 500.0,
        "d_star": 0.7, "strength_scale": 1.0, "tau_aI_ms": 150.0,
        "f_aI": 0.05, "g_M": 3.0, "tau_M_ms": 2000.0, "g_Z": 1.0,
        "onset_ms": 500.0,
    }


def test_control_manifests_lock_onset_plus_1500_and_six_doses():
    selection = {"rows": [_candidate()]}
    calibration = R.build_calibration_manifest(selection)
    assert len(calibration["rows"]) == 5
    assert {row["control_t0_ms"] for row in calibration["rows"]} == {2000.0}
    assert calibration["rows"][0]["uncontrolled_onset_ms"] == 500.0
    assert calibration["rows"][0]["uncontrolled_offset_ms"] is None
    assert calibration["rows"][0]["uncontrolled_duration_right_censored"] is True
    decision = {"calibration_decisions": [{"selection_rank": 0, "u_ref_mV": 1.0}]}
    dose = R.build_dose_manifest(selection, decision)
    assert len(dose["rows"]) == 6
    assert {(row["dose_multiplier"], row["control_duration_ms"]) for row in dose["rows"]} == {
        (0.5, 50.0), (0.5, 200.0), (1.0, 50.0),
        (1.0, 200.0), (1.5, 50.0), (1.5, 200.0),
    }


def test_control_response_and_u_ref_enforce_drop_without_long_silence():
    core = np.full(1000, 100.0)
    all_e = np.full(1000, 20.0)
    core[500:550] = 40.0
    got = A.control_response(core, all_e, t0_ms=1000.0, duration_ms=50.0)
    assert 0.5 <= got["fractional_core_drop"] <= 0.7
    assert got["calibration_target_met"] is True
    rows = [{"control_uplift_mV": 1.0, "control_response": got}]
    assert A.choose_u_ref(rows)["u_ref_mV"] == 1.0


def test_control_waves_are_interleaved_across_candidates():
    candidates = [_candidate() for _ in range(4)]
    for index, row in enumerate(candidates):
        row["config_id"] = f"c{index}"
        row["onset_ms"] = 500.0 + 100.0 * index
    calibration = R.build_calibration_manifest({"rows": candidates})
    assert {row["selection_rank"] for row in calibration["rows"][:4]} == {0, 1, 2, 3}
    assert {row["control_uplift_mV"] for row in calibration["rows"][:4]} == {0.25}
    assert [row["uncontrolled_source_candidate_id"] for row in calibration["rows"][:4]] == [
        "c0", "c1", "c2", "c3",
    ]
    assert [row["uncontrolled_onset_ms"] for row in calibration["rows"][:4]] == [
        500.0, 600.0, 700.0, 800.0,
    ]


def test_control_exit_requires_advance_over_paired_uncontrolled_trajectory():
    base = {
        "onset_ms": 500.0,
        "control_t0_ms": 2000.0,
        "uncontrolled_onset_ms": 500.0,
        "uncontrolled_offset_ms": None,
    }
    assert A.paired_control_effect({**base, "offset_ms": 3000.0}) == {
        "status": "offset_vs_censored_uncontrolled",
        "causal_control_exit_candidate": True,
        "duration_advance_ms": None,
    }
    late_natural = {
        **base,
        "uncontrolled_offset_ms": 5000.0,
        "offset_ms": 4500.0,
    }
    assert A.paired_control_effect(late_natural)["causal_control_exit_candidate"] is False
    advanced = {**late_natural, "offset_ms": 3000.0}
    assert A.paired_control_effect(advanced)["causal_control_exit_candidate"] is True


def test_control_exit_before_pulse_is_not_causal():
    got = A.paired_control_effect({
        "onset_ms": 500.0,
        "offset_ms": 1500.0,
        "control_t0_ms": 2000.0,
        "uncontrolled_onset_ms": 500.0,
        "uncontrolled_offset_ms": None,
    })
    assert got["status"] == "offset_precedes_control"
    assert got["causal_control_exit_candidate"] is False
