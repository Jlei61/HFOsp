import numpy as np

from src.topic4_fcxr_lc3_xcal import (
    choose_calibration_family,
    lifecycle_candidate_gate,
    multivariate_statistical_return,
    postictal_suppression_gate,
    relay_x_inf,
    return_brackets,
    select_x_candidates,
)


def test_postictal_suppression_compares_population_rate_to_population_rate():
    gate = postictal_suppression_gate(
        early_post_population_rate_hz=1.9, pre_population_rate_hz=4.0)
    assert gate["pass_"] is True
    assert gate["threshold_population_rate_hz"] == 2.0
    assert gate["comparable_quantity"] == "mean_population_rate_hz_per_E_cell"


def test_postictal_suppression_rejects_missing_or_nonpositive_pre_rate():
    assert postictal_suppression_gate(
        early_post_population_rate_hz=0.0, pre_population_rate_hz=0.0)["pass_"] is False


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
             low_label="INTERICTAL_WORKPOINT", n_low_returning_events=4,
             crossing_time_ms=1800.0, high_returned_to_low=True),
        dict(candidate_id="b", numerical_safe=True, ied_mean_a_x=0.91,
             low_label="INTERICTAL_WORKPOINT", n_low_returning_events=3,
             crossing_time_ms=2200.0, high_returned_to_low=True),
        dict(candidate_id="bad", numerical_safe=True, ied_mean_a_x=0.89,
             low_label="INTERICTAL_WORKPOINT", n_low_returning_events=5,
             crossing_time_ms=2000.0, high_returned_to_low=True),
    ]
    assert [row["candidate_id"] for row in select_x_candidates(rows)] == ["a", "b"]


def test_lifecycle_gate_requires_multivariate_return_not_mean_rate_alone():
    pre = dict(n_events=5, event_rate_hz=1.0, median_iei_ms=900.0,
               median_duration_ms=20.0, median_participation=0.1,
               median_compactness_mm=2.0, fraction_A=0.5)
    post = dict(pre)
    rest = multivariate_statistical_return(pre, post)
    assert rest["pass_"]
    gate = lifecycle_candidate_gate(
        lifecycle_label="RECOVERED_INTERICTAL", onset_ms=9000.0,
        high_duration_ms=3000.0, x_activates_after_onset=True,
        postictal_suppression=True, statistical_return=rest,
        numerical_unsafe=False, refractory_ceiling_fraction=0.0)
    assert gate["pass_"]
    post["median_compactness_mm"] = 8.0
    failed_rest = multivariate_statistical_return(pre, post)
    failed = lifecycle_candidate_gate(
        lifecycle_label="RECOVERED_INTERICTAL", onset_ms=9000.0,
        high_duration_ms=3000.0, x_activates_after_onset=True,
        postictal_suppression=True, statistical_return=failed_rest,
        numerical_unsafe=False, refractory_ceiling_fraction=0.0)
    assert not failed["pass_"]
