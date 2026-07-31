"""HYB2 adjudication tests (plan sections 2, 3, 6, 7)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.topic4_fcxr_hyb2 as H2        # noqa: E402
import src.topic4_fcxr_hyb1 as H1        # noqa: E402


# --------------------------------------------------------------- locked constants
def test_locked_constants_match_the_plan():
    assert (H2.DT_R_MS, H2.Q_BG, H2.EPS_S_FRAC, H2.EPS_Q_FRAC, H2.Q_ON_MARGIN) == \
        (0.5, 0.99, 0.10, 0.10, 1.10)
    assert H2.I_R_MAX == pytest.approx(4.134151260609386, rel=0, abs=0)
    assert H2.T_EVENT_GUARD_MS == 22.0 and H2.RESIDUAL_TARGET == 0.01
    assert (H2.TAU_Z_DOWN_MS, H2.TAU_Z_UP_MS, H2.T_CAL_MS_MIN) == (5000.0, 20000.0, 15000.0)


def test_the_force_anchor_reproduces_from_the_accepted_E_K_implementation():
    from src.topic4_fcxr_ion import E_K, K_O0, K_I0, E_K_0
    assert float(E_K(K_O0 + 0.6715, K_I0) - E_K_0) == H2.I_R_MAX


# --------------------------------------------------------------- plan 3: GAP not IEI
def test_gap_is_onset_minus_previous_OFFSET_not_previous_onset():
    on = np.array([0.0, 200.0, 500.0])
    off = np.array([20.0, 230.0, 510.0])
    g = H2.event_gaps(on, off)
    assert np.allclose(g, [180.0, 270.0])          # NOT [200, 300]


def test_gap_is_always_shorter_than_the_corresponding_IEI():
    rng = np.random.default_rng(0)
    on = np.cumsum(rng.uniform(120.0, 400.0, 40))
    off = on + rng.uniform(8.0, 22.0, 40)
    assert np.all(H2.event_gaps(on, off) < np.diff(on))


def test_gap_rejects_overlapping_events():
    with pytest.raises(ValueError):
        H2.event_gaps(np.array([0.0, 10.0]), np.array([50.0, 60.0]))


def test_gap_rejects_an_offset_before_its_own_onset():
    with pytest.raises(ValueError):
        H2.event_gaps(np.array([0.0, 100.0]), np.array([-5.0, 110.0]))


def test_tau_R_is_the_geometric_midpoint_of_the_feasible_interval():
    t = H2.tau_R_from_timescale(22.0, 147.5)
    assert t["feasible"]
    assert t["interval"][1] == pytest.approx(147.5 / np.log(100.0))
    assert t["tau_R_ms"] == pytest.approx(np.sqrt(22.0 * 147.5 / np.log(100.0)))
    assert t["tau_R_ms"] == pytest.approx(26.55, abs=0.01)


def test_the_headroom_is_reported_because_the_interval_is_narrow():
    t = H2.tau_R_from_timescale(22.0, 147.5)
    assert t["headroom_ms"] == pytest.approx(10.03, abs=0.05)


def test_using_IEI_instead_of_GAP_would_have_given_a_wider_and_WRONG_interval():
    """Regression on the P0-2 error: IEI overstates the decay budget by ~one event duration."""
    wrong = H2.tau_R_from_timescale(22.0, 169.5)["interval"][1]
    right = H2.tau_R_from_timescale(22.0, 147.5)["interval"][1]
    assert wrong == pytest.approx(36.81, abs=0.01) and right == pytest.approx(32.03, abs=0.01)
    assert right < wrong


def test_an_empty_timescale_interval_blocks_the_design():
    v = H2.adjudicate_calibration(dict(T_event_guard_ms=40.0, gap_05_ms=147.5,
                                       Q_on=1.0, Q_scale=1.0))
    assert v["status"] == "DESIGN_BLOCKED_EVENT_TIMESCALE"


def test_degenerate_thresholds_are_CALIBRATION_INVALID_not_a_silent_pass():
    for q_on, q_sc in ((0.0, 1.0), (1.0, 0.0), (-1.0, 1.0)):
        v = H2.adjudicate_calibration(dict(T_event_guard_ms=22.0, gap_05_ms=147.5,
                                           Q_on=q_on, Q_scale=q_sc))
        assert v["status"] == "CALIBRATION_INVALID"


def test_calibration_lock_reports_the_short_gap_tail_not_only_GAP_05():
    v = H2.adjudicate_calibration(dict(T_event_guard_ms=22.0, gap_05_ms=147.5, gap_01_ms=100.0,
                                       gap_min_ms=40.0, Q_on=3.0, Q_scale=3.0))
    assert v["status"] == "CALIBRATION_LOCKED"
    r = v["residual_tail"]
    assert r["gap_05_ms"] < 0.01 and r["gap_01_ms"] > r["gap_05_ms"] and r["gap_min_ms"] > 0.1
    assert v["eps_q"] == pytest.approx(0.3)


# --------------------------------------------------------------- plan 2: background / eps
def test_background_excludes_unoccupied_voxels_and_makes_them_never_source():
    load = np.stack([np.full(200, 5.0), np.full(200, 50.0), np.zeros(200)], axis=1)
    b = H2.background_envelope(load, np.array([True, True, False]))
    assert np.isfinite(b[0]) and np.isfinite(b[1]) and np.isinf(b[2])


def test_eps_s_uses_the_median_of_OCCUPIED_backgrounds_only():
    b = np.array([2.0, 4.0, np.inf])
    assert H2.eps_s_from_background(b) == pytest.approx(0.3)


def test_q_on_carries_the_registered_margin_above_the_calibration_maximum():
    assert H2.q_on_from_event_peaks([1.0, 3.0, 2.0]) == pytest.approx(3.3)


# --------------------------------------------------------------- plan 7.1: Gate B0
def _b0(**over):
    m = dict(active_occupancy=0.002, pre_onset_residual_frac=0.003, q_floor_drift=0.001,
             event_stats_in_band=True, clip_frac_max=0.0, numerical_unsafe=False)
    m.update(over)
    return m


def test_gate_B0_passes_when_every_clause_holds():
    assert H2.adjudicate_gate_B0(_b0())["status"] == "BASELINE_INVISIBLE"


def test_gate_B0_marks_clauses_1_and_2_as_NOT_independent_evidence():
    """Because Q_on = 1.10 * max event peak, both hold by construction."""
    c = H2.adjudicate_gate_B0(_b0())["checks"]
    assert c["active_occupancy"]["independent"] is False
    assert c["pre_onset_residual"]["independent"] is False
    assert c["q_floor_drift"]["independent"] is True
    assert c["event_stats_in_band"]["independent"] is True


def test_gate_B0_fails_on_the_q_v_RATCHET_which_is_exactly_where_HYB1_failed():
    v = H2.adjudicate_gate_B0(_b0(q_floor_drift=0.25))
    assert v["status"] == "STOP_ELR_BASELINE_VISIBLE" and not v["checks"]["q_floor_drift"]["ok"]


def test_gate_B0_drift_is_a_DIFFERENCE_so_a_near_zero_first_floor_cannot_explode_it():
    """A ratio would be 0/0 or divide by a tiny number; the plan uses (last-first)/Q_on."""
    assert "DIFFERENCE" in H2.adjudicate_gate_B0(_b0())["checks"]["q_floor_drift"]["rule"]


def test_gate_B0_fails_when_the_interictal_event_statistics_move():
    assert H2.adjudicate_gate_B0(_b0(event_stats_in_band=False))["status"] == \
        "STOP_ELR_BASELINE_VISIBLE"


def test_gate_B0_fails_on_any_clip():
    assert H2.adjudicate_gate_B0(_b0(clip_frac_max=1e-9))["status"] == "STOP_ELR_BASELINE_VISIBLE"


def test_gate_B0_forbids_the_over_strong_wording():
    v = H2.adjudicate_gate_B0(_b0())
    assert "OBSERVED" in v["allowed_wording"]
    assert "shown not to disturb" in v["forbidden_wording"]


# --------------------------------------------------------------- plan 7.2: Gate A0 three-way
def _a0(**over):
    off = dict(window_participants=8000, recruitment_radius_mm=7.0, participant_voxels=500,
               end_rate_hz=40.0, early_stopped=False)
    on = dict(window_participants=9200, recruitment_radius_mm=8.1, participant_voxels=560)
    m = dict(crossed_Q_on=True, ms_after_t_gate=1500.0, n_E=32000, n_occupied_voxels=577,
             off=off, on=on, max_R_evt=3.0, clip_frac_max=0.0, finite=True)
    m.update(over)
    return m


def test_A0_input_insufficient_when_the_threshold_was_never_crossed():
    v = H2.adjudicate_gate_A0(_a0(crossed_Q_on=False))
    assert v["status"] == "A0_INPUT_INSUFFICIENT"


def test_A0_input_insufficient_when_the_post_gate_window_is_too_short():
    v = H2.adjudicate_gate_A0(_a0(ms_after_t_gate=400.0))
    assert v["status"] == "A0_INPUT_INSUFFICIENT"


def test_A0_ceiling_confounded_when_the_off_arm_already_recruits_almost_everything():
    v = H2.adjudicate_gate_A0(_a0(off=dict(window_participants=31000, recruitment_radius_mm=9.0,
                                           participant_voxels=560, end_rate_hz=40.0,
                                           early_stopped=False)))
    assert v["status"] == "A0_CEILING_CONFOUNDED" and v["ceiling"]["participants"]


def test_A0_ceiling_confounded_when_the_off_arm_ran_away():
    v = H2.adjudicate_gate_A0(_a0(off=dict(window_participants=8000, recruitment_radius_mm=7.0,
                                           participant_voxels=500, end_rate_hz=452.8,
                                           early_stopped=False)))
    assert v["status"] == "A0_CEILING_CONFOUNDED" and v["ceiling"]["runaway"]


def test_A0_effective_needs_at_least_two_of_three_extent_measures_up_by_10pc():
    v = H2.adjudicate_gate_A0(_a0())
    assert v["status"] == "A0_RECRUITMENT_EFFECTIVE" and v["n_measures_up"] >= 2


def test_A0_one_measure_alone_is_not_enough():
    v = H2.adjudicate_gate_A0(_a0(on=dict(window_participants=9200, recruitment_radius_mm=7.05,
                                          participant_voxels=505)))
    assert v["status"] == "NO_GO_EVENT_LIMITED_ACTUATOR" and v["n_measures_up"] == 1


def test_A0_a_NO_GO_is_only_reachable_after_eligibility():
    """The whole point of the three-way: 'ineffective' must not absorb 'the input was wrong'."""
    for bad in (dict(crossed_Q_on=False), dict(ms_after_t_gate=100.0)):
        assert H2.adjudicate_gate_A0(_a0(**bad))["status"] != "NO_GO_EVENT_LIMITED_ACTUATOR"


def test_A0_fails_if_the_actuator_exceeded_its_own_amplitude_anchor():
    v = H2.adjudicate_gate_A0(_a0(max_R_evt=H2.I_R_MAX * 1.01))
    assert v["status"] == "NO_GO_EVENT_LIMITED_ACTUATOR" and not v["bounded"]


def test_A0_negative_wording_forbids_attributing_the_loss_to_cross_event_memory():
    v = H2.adjudicate_gate_A0(_a0(on=dict(window_participants=8010, recruitment_radius_mm=7.01,
                                          participant_voxels=501)))
    assert v["status"] == "NO_GO_EVENT_LIMITED_ACTUATOR"
    assert "THIS short-memory" in v["allowed_wording"]
    assert "cross-event" in v["forbidden_wording"]


# --------------------------------------------------------------- plan 6: S_Z axis
def test_C_analytic_matches_the_recorded_values():
    for T, want in ((3000.0, 0.248), (5000.0, 0.368), (15000.0, 0.683), (24000.0, 0.793)):
        assert H2.c_analytic(T, H2.TAU_Z_DOWN_MS) == pytest.approx(want, abs=1e-3)


def test_a_short_T_cal_is_refused_because_S_Z_would_degenerate_to_h_Z():
    S = np.full((int(3000.0 / 100.0), 50), 10.0)
    with pytest.raises(ValueError, match="degenerate"):
        H2.s_z_response(S, np.ones(50), 5.0, dt_ms=100.0)


def test_S_Z_is_proportional_to_a_p_when_no_cell_ever_crosses_the_threshold():
    """The frozen-replay identity: constant above/below status -> S_Z = a_p * C, i.e. NO new
    information beyond the t=0 hazard.  What S_Z adds is only the crossing cells."""
    n, nt = 100, 240
    S = np.zeros((nt, n))
    S[:, :30] = 10.0                                   # always above; the rest always below
    got = H2.s_z_response(S, np.ones(n), 5.0, dt_ms=100.0)
    assert got == pytest.approx(0.30 * H2.c_analytic(nt * 100.0, H2.TAU_Z_DOWN_MS), rel=2e-2)


def test_S_Z_differs_from_a_p_only_through_cells_that_CROSS_the_threshold():
    n, nt = 100, 240
    fixed = np.zeros((nt, n)); fixed[:, :30] = 10.0
    cross = fixed.copy(); cross[nt // 2:, 30:60] = 10.0      # 30 more cells cross halfway
    assert H2.s_z_response(cross, np.ones(n), 5.0, dt_ms=100.0) > \
        H2.s_z_response(fixed, np.ones(n), 5.0, dt_ms=100.0)


def test_S_Z_uses_the_p_weights():
    n, nt = 4, 240
    S = np.zeros((nt, n)); S[:, :2] = 10.0
    hi = H2.s_z_response(S, np.array([9.0, 9.0, 1.0, 1.0]), 5.0, dt_ms=100.0)
    lo = H2.s_z_response(S, np.array([1.0, 1.0, 9.0, 9.0]), 5.0, dt_ms=100.0)
    assert hi > lo


def test_S_Z_rejects_mismatched_weights():
    with pytest.raises(ValueError):
        H2.s_z_response(np.zeros((240, 10)), np.ones(9), 1.0, dt_ms=100.0)


def _sensor_bracketing_both_anchors(nt=240, n=600, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.lognormal(mean=np.log(8.0), sigma=1.8, size=n)
    return np.tile(base, (nt, 1)) * rng.uniform(0.9, 1.1, (nt, n))


def test_z_response_axis_locks_and_places_three_interior_points():
    S = _sensor_bracketing_both_anchors()
    v = H2.adjudicate_z_response_axis(S, np.ones(S.shape[1]), dt_ms=100.0)
    assert v["status"] == "Z_RESPONSE_AXIS_LOCKED"
    lv = [v["levels"][k]["I_th_EI"] for k in ("S25", "S50", "S75")]
    assert H2.I_TH_Q50 <= min(lv) and max(lv) <= H2.I_TH_Q75
    assert v["S_Z_q50"] > v["S_Z_q75"]


def test_z_response_axis_is_blocked_when_S_Z_does_not_separate_the_anchors():
    S = np.full((240, 100), 1e-6)                     # everything below both thresholds
    v = H2.adjudicate_z_response_axis(S, np.ones(100), dt_ms=100.0)
    assert v["status"] == "DESIGN_BLOCKED_Z_RESPONSE_AXIS"


def test_the_axis_scope_statement_forbids_reading_it_as_self_limitation():
    v = H2.adjudicate_z_response_axis(_sensor_bracketing_both_anchors(), np.ones(600), dt_ms=100.0)
    assert "does NOT measure self-limitation" in v["scope"]


# --------------------------------------------------------------- HYB1 gates reused verbatim
def test_the_seven_gates_and_bad_data_regressions_are_reused_not_reimplemented():
    assert callable(H1.adjudicate_lifecycle)
    assert (H1.PRE_MIN_MS, H1.BOUT_MIN_MS, H1.BOUT_MAX_MS, H1.POST_MIN_MS) == \
        (8000.0, 1000.0, 5000.0, 8000.0)
    q75 = dict(kick_boost=0.0, t_kick_ms=1e9, onset_detected=True, pre_interictal_ms=10000.0,
               bout_ms=None, bounded=True, clip_frac_max=0.0, finite=True, numerical_unsafe=False,
               end_rate_hz=8.853, x_activation_delay_ms=None, post_return_ms=0.0,
               label="DENSE_EVENT_TRAIN", recruit_contacts=14, onset_gradient_r2=0.6,
               post_iei_cv=0.0, band_event_rate=(0.3, 3.0), band_duration=(10.0, 90.0),
               band_participation=(0.01, 0.2), post_event_rate_hz=0.0, post_duration_ms=0.0,
               post_participation=0.0, post_silent=True)
    v = H1.adjudicate_lifecycle(q75, spatial_leg="PASS")
    assert v["status"] == "NOT_A_CANDIDATE" and "3_bounded_high_state" in v["failed"]
