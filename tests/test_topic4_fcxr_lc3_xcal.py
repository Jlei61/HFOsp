import numpy as np

from src.topic4_fcxr_lc3_xcal import (
    choose_calibration_family,
    relay_x_inf,
    return_brackets,
    select_x_candidates,
)


def test_return_bracket_requires_same_d_high_start_return_and_survival():
    rows = [
        dict(point_id="H1_ts1.25_r025", state_kind="high", d_label="D50",
             a_x=0.1, resolved_label="INTERICTAL_WORKPOINT"),
        dict(point_id="H1_ts1.25_r025", state_kind="high", d_label="D50",
             a_x=0.3, resolved_label="INTERICTAL_WORKPOINT"),
        dict(point_id="H1_ts1.25_r025", state_kind="high", d_label="D50",
             a_x=0.5, resolved_label="FINITE_HIGH_FIXED"),
        dict(point_id="H1_ts1.25_r025", state_kind="low", d_label="D50",
             a_x=0.5, resolved_label="INTERICTAL_WORKPOINT"),
    ]
    got = return_brackets(rows, {"D50": 0.08})
    assert got == [dict(d_label="D50", mean_D=0.08, a_return_max=0.3,
                        a_survive_min=0.5, a_off_midpoint=0.4, bracket_width=0.2)]


def test_calibration_family_distinguishes_asymptote_from_speed():
    assert choose_calibration_family(
        observed_x=0.8, inferred_x_inf=0.7,
        a_return_max=0.3, a_survive_min=0.5) == "SENSOR_GATE_AND_HILL_MIDPOINT"
    assert choose_calibration_family(
        observed_x=0.8, inferred_x_inf=0.2,
        a_return_max=0.3, a_survive_min=0.5) == "HILL_MIDPOINT_AND_RISE_TIME"
    assert choose_calibration_family(
        observed_x=0.4, inferred_x_inf=0.2,
        a_return_max=0.3, a_survive_min=0.5) == "BOUNDARY_ALREADY_REACHED_NO_RECALIBRATION_NEEDED"


def test_relay_inf_and_candidate_gate():
    x = relay_x_inf(np.array([0.0, 10.0, 100.0]), y_gate=5.0, K_y=5.0,
                    hill_n=4, x_min=0.1)
    assert x[0] == 1.0 and x[-1] < x[1]
    rows = [
        dict(candidate_id="a", numerical_safe=True, ied_mean_a_x=0.95,
             crossing_time_ms=1800.0, high_returned_to_low=True),
        dict(candidate_id="b", numerical_safe=True, ied_mean_a_x=0.91,
             crossing_time_ms=2200.0, high_returned_to_low=True),
        dict(candidate_id="bad", numerical_safe=True, ied_mean_a_x=0.89,
             crossing_time_ms=2000.0, high_returned_to_low=True),
    ]
    assert [row["candidate_id"] for row in select_x_candidates(rows)] == ["a", "b"]
