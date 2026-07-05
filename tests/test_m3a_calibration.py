"""M3A-A2 mapping calibration: ground the q->excitability direction in the engine response.

Pure sign-evaluation + calibration-application logic. The engine sign test (contract §7)
confirms a coordinate's declared phase direction matches the engine's actual firing
response to the slow variable; a passing test flips calibration_status to 'passed' so the
overlay audit's cond1 can pass. (The engine measurement itself is exercised separately.)
"""
import sys, os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.sef_hfo_m3_interface import validate_slow_to_rate_mapping, mapping_sign_tests_passed  # noqa: E402
from src.sef_hfo_m3a_export import default_precalib_mapping_and_ranges  # noqa: E402
from src.sef_hfo_m3a_calibration import (  # noqa: E402
    evaluate_engine_sign_test, apply_calibration, calibrate_axisbreak_mapping,
)


def _q_core_sign(firing, sha="abc"):
    return evaluate_engine_sign_test([0.4, 0.7, 1.0], firing, variable="q_core",
                                     coord="phase_x_core", expected_direction="decreasing_in_input",
                                     engine_sha=sha)


def _q_global_sign(firing, sha="abc"):
    return evaluate_engine_sign_test([0.4, 0.7, 1.0], firing, variable="q_global",
                                     coord="phase_y_global", expected_direction="decreasing_in_input",
                                     engine_sha=sha)


def test_sign_test_passes_when_engine_matches_declared_decreasing():
    # phase_x_core ~ 1/q_core: firing DECREASES as q_core increases -> consistent with decreasing
    st = _q_core_sign([5.0, 3.0, 1.0])
    assert st["observed_slope_sign"] == -1
    assert st["passed"] is True
    for k in ("name", "coord", "input_var", "expected_direction",
              "observed_slope_sign", "passed", "engine_sha"):
        assert k in st


def test_sign_test_fails_when_engine_contradicts_declared_direction():
    # firing INCREASES with q_core -> contradicts "decreasing" -> fail closed
    st = _q_core_sign([1.0, 3.0, 5.0])
    assert st["observed_slope_sign"] == 1
    assert st["passed"] is False


def test_sign_test_fails_closed_on_flat_response():
    st = _q_core_sign([2.0, 2.0, 2.0])  # no slope -> ambiguous -> fail closed
    assert st["passed"] is False


def test_apply_calibration_flips_status_and_passes_audit_cond1():
    mapping, _ = default_precalib_mapping_and_ranges("m3a_a2_cal")
    cal = apply_calibration(mapping, {
        "phase_x_core": _q_core_sign([5.0, 3.0, 1.0]),
        "phase_y_global": _q_global_sign([6.0, 4.0, 2.0]),
    })
    assert cal["coordinates"]["phase_x_core"]["calibration_status"] == "passed"
    assert cal["coordinates"]["phase_y_global"]["calibration_status"] == "passed"
    validate_slow_to_rate_mapping(cal)                   # calibrated mapping is schema-valid
    assert mapping_sign_tests_passed(cal, None) is True  # on-axis coords pass -> audit cond1 can pass


def test_apply_calibration_marks_failed_sign_test_failed():
    mapping, _ = default_precalib_mapping_and_ranges("m3a_a2_cal")
    cal = apply_calibration(mapping, {"phase_x_core": _q_core_sign([1.0, 3.0, 5.0])})  # wrong direction
    assert cal["coordinates"]["phase_x_core"]["calibration_status"] == "failed"
    assert mapping_sign_tests_passed(cal, None) is False  # fail-closed


def test_calibrate_axisbreak_mapping_orchestration_with_injected_measure():
    # inject a synthetic engine whose firing falls as q rises (correct physics) -> calibrates
    mapping, _ = default_precalib_mapping_and_ranges("m3a_a2_cal")

    def fake_measure(a, q, which):
        return 10.0 * (1.0 - q)   # q=0.4 -> 6, 0.7 -> 3, 1.0 -> 0 (decreasing in q)

    cal, sts = calibrate_axisbreak_mapping(None, mapping, measure_fn=fake_measure)
    assert sts["phase_x_core"]["passed"] is True
    assert sts["phase_y_global"]["passed"] is True
    assert cal["coordinates"]["phase_x_core"]["calibration_status"] == "passed"
    assert mapping_sign_tests_passed(cal, None) is True   # calibrated -> audit cond1 can pass
