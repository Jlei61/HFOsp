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
             revt_occupancy_by_segment=[0.0, 0.001, 0.002, 0.002],
             event_stats_in_band=True, clip_frac_max=0.0, numerical_unsafe=False)
    m.update(over)
    return m


def test_gate_B0_passes_when_every_clause_holds():
    assert H2.adjudicate_gate_B0(_b0())["status"] == "BASELINE_PRACTICALLY_INVISIBLE"


def test_gate_B0_gates_at_the_membrane_and_keeps_the_q_measures_as_diagnostics():
    """q_v is a hidden sensor: nothing about it reaches a membrane until it crosses Q_on and
    becomes R_evt.  So B0 gates on R_evt occupancy, per-segment R_evt occupancy, the interictal
    event statistics and numerical safety; the two q_v measures are reported but do not vote."""
    v = H2.adjudicate_gate_B0(_b0())
    assert set(v["checks"]) == {"active_occupancy", "revt_no_sustained_rise",
                                "event_stats_in_band", "numerically_safe"}
    assert all(c["gating"] for c in v["checks"].values())
    assert set(v["diagnostics"]) == {"q_pre_onset_residual", "q_floor_drift"}
    assert not any(d["gating"] for d in v["diagnostics"].values())


def test_a_q_v_ratchet_alone_no_longer_vetoes_the_gate():
    """HYB1's ratchet was a delta-K floor that entered the membrane DIRECTLY.  Transplanting that
    veto onto a hidden q floor is not a level-preserving translation, so a large q drift with a
    quiet membrane must pass -- and must still be reported."""
    v = H2.adjudicate_gate_B0(_b0(q_floor_drift=0.25))
    assert v["status"] == "BASELINE_PRACTICALLY_INVISIBLE"
    assert v["diagnostics"]["q_floor_drift"]["value"] == 0.25


def test_a_creeping_R_evt_floor_DOES_veto_the_gate():
    """The membrane-level counterpart of HYB1's failure: if the actuator itself creeps up, the
    later segments breach the same 1% bound and the gate must stop."""
    v = H2.adjudicate_gate_B0(_b0(revt_occupancy_by_segment=[0.0, 0.004, 0.02, 0.05]))
    assert v["status"] == "STOP_ELR_BASELINE_VISIBLE"
    assert not v["checks"]["revt_no_sustained_rise"]["ok"]


def test_gate_B0_stops_when_the_segment_profile_is_missing_rather_than_passing_blind():
    v = H2.adjudicate_gate_B0(_b0(revt_occupancy_by_segment=None))
    assert v["status"] == "STOP_ELR_BASELINE_VISIBLE"


def test_a_broken_pre_onset_residual_cannot_veto_the_gate():
    """The residual samples the NEXT event's local build-up, not the previous event's remnant --
    measured gap-resolved, the envelope clears to 0.0029-0.0051 of Q_on 30-75 ms before onset."""
    v = H2.adjudicate_gate_B0(_b0(pre_onset_residual_frac=0.18))
    assert v["status"] == "BASELINE_PRACTICALLY_INVISIBLE"
    assert v["diagnostics"]["q_pre_onset_residual"]["value"] == 0.18


def test_gate_B0_fails_when_the_interictal_event_statistics_move():
    assert H2.adjudicate_gate_B0(_b0(event_stats_in_band=False))["status"] == \
        "STOP_ELR_BASELINE_VISIBLE"


def test_gate_B0_fails_on_any_clip():
    assert H2.adjudicate_gate_B0(_b0(clip_frac_max=1e-9))["status"] == "STOP_ELR_BASELINE_VISIBLE"


def test_gate_B0_forbids_the_over_strong_wording():
    v = H2.adjudicate_gate_B0(_b0())
    assert "PRE-REGISTERED calibration-half Q_on" in v["allowed_wording"]
    assert "very rare" in v["allowed_wording"]
    assert "never fired at all" in v["forbidden_wording"]
    assert "bit-exactly" in v["forbidden_wording"]


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


# --------------------------------------------------------------- Gate B0 reduction (pure)
def _b0_metrics():
    """Load the runner's pure reduction by source, without importing the heavy engine modules."""
    import types
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "scripts", "run_topic4_fcxr_hyb2.py")).read()
    seg = src[src.index("def _b0_metrics("):src.index("def cmd_gate_b0(")]
    ns = dict(np=np, H1_IEI_CV_MIN=H1.IEI_CV_MIN)
    exec(compile(seg, "<b0_metrics>", "exec"), ns)
    return ns["_b0_metrics"]


def _b0_inputs(**over):
    rng = np.random.default_rng(0)
    on = np.cumsum(rng.uniform(200.0, 900.0, 34))
    ev = [dict(dur_ms=10.0 + rng.uniform(-2, 2), peak_ext=0.045) for _ in on]
    d = dict(active_occupancy=0.002,
             q_stats=dict(pre_onset_residual_frac=0.003, q_floor_drift=0.001,
                          revt_occupancy_by_segment=[0.0, 0.001, 0.002, 0.002]),
             on_events=ev, on_onsets=on,
             off_durations_ms=np.full(34, 10.0), off_onsets_ms=on.copy(),
             band=dict(event_rate_lo=0.154, event_rate_hi=4.478), T_ms=24000.0,
             numerical=dict(clip_frac_max=0.0, numerical_unsafe=False),
             label="INTERICTAL_BASELINE")
    d.update(over)
    return d


def test_b0_reduction_runs_without_a_keyword_collision():
    """Regression: `in_band` already carried an `iei_cv` key and the caller passed `iei_cv=` again,
    raising TypeError AFTER a 41 minute simulation had already finished.  The reduction is now a
    pure function, so the same mistake is caught in milliseconds."""
    m = _b0_metrics()(**_b0_inputs())
    assert set(m) == {"active_occupancy", "pre_onset_residual_frac", "q_floor_drift",
                      "revt_occupancy_by_segment", "event_stats_in_band", "event_stats_detail",
                      "clip_frac_max", "numerical_unsafe"}
    # the membrane-level profile must survive the reduction: adjudicate_gate_B0 stops without it
    assert H2.adjudicate_gate_B0(m)["checks"]["revt_no_sustained_rise"]["ok"]
    assert set(m["event_stats_detail"]["clauses"]) == {
        "event_rate", "iei_cv", "duration", "participation", "not_silent"}
    assert m["event_stats_detail"]["iei_cv"] == m["event_stats_detail"]["off_iei_cv"]


def test_b0_reduction_output_feeds_the_adjudicator_unchanged():
    m = _b0_metrics()(**_b0_inputs())
    assert H2.adjudicate_gate_B0(m)["status"] in ("BASELINE_INVISIBLE",
                                                  "STOP_ELR_BASELINE_VISIBLE")


def test_b0_reduction_fails_the_event_clause_when_the_train_thins_out():
    inp = _b0_inputs()
    inp["on_events"] = inp["on_events"][:8]
    inp["on_onsets"] = inp["on_onsets"][:8]
    m = _b0_metrics()(**inp)
    assert m["event_stats_detail"]["n_events"] == 8
    assert m["event_stats_detail"]["event_rate_hz"] < m["event_stats_detail"]["off_event_rate_hz"]


def test_b0_reduction_fails_the_iei_clause_on_a_clock_like_train():
    inp = _b0_inputs(on_onsets=np.arange(34) * 500.0)
    m = _b0_metrics()(**inp)
    assert not m["event_stats_detail"]["clauses"]["iei_cv"] and not m["event_stats_in_band"]


def test_b0_reduction_reuses_the_HYB1_IEI_CV_floor():
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "scripts", "run_topic4_fcxr_hyb2.py")).read()
    seg = src[src.index("def _b0_metrics("):src.index("def cmd_gate_b0(")]
    # the CV FLOOR must come from HYB1; the 0.5 that does appear is the +-50% band tolerance
    assert "H1_IEI_CV_MIN" in seg
    assert "cv >= 0.5" not in seg and "cv>=0.5" not in seg
    assert "cv >= H1_IEI_CV_MIN" in seg


# ---------------------------------------------------------------- Q_on / clause 2-3 contract
# Both regressions come from Gate B0's first run (2026-07-31), where the shipped code deviated from
# the locked contract in two independent ways and the deviations only surfaced after a 35 min run.

def test_validation_half_events_must_not_enter_Q_on():
    """plan 4.2 step 1 + spec 207: Q_on is locked on the CALIBRATION half alone, so that the
    validation half stays out-of-sample for the false-activation check (spec 207 step 3).

    This test previously asserted the opposite.  Using the whole record let a validation-half peak
    set the threshold it was supposed to be tested against -- seed1's Q_on went 112.505 -> 169.846,
    above the validation maximum of 154.405, so "the actuator never fires interictally" became
    circular.  The 40 Hz peak below sits in the validation half and must be excluded.
    """
    n_t, n_v = 400, 3
    q = np.zeros((n_t, n_v))
    q[10, 0] = 5.0                        # calibration half
    q[300, 1] = 40.0                      # validation half -- must NOT set the threshold
    occ = np.array([True, True, False])
    peaks = H2.event_peak_values(q, [5.0, 150.0], occ, tau_R_ms=1.0, dt_ms=0.5,
                                 calibration_end_ms=100.0)
    assert max(peaks) == pytest.approx(5.0)
    assert H2.q_on_from_event_peaks(peaks) == pytest.approx(5.5)


def test_event_peak_values_has_no_default_calibration_end():
    """A default would silently restore the leaking path for any caller that forgets the argument
    -- the same 'default=None restores the buggy path' failure mode as the PR-6 valid_mask."""
    with pytest.raises(TypeError):
        H2.event_peak_values(np.zeros((10, 2)), [0.0], np.array([True, True]), 1.0, 0.5)


def test_revt_activation_profile_is_per_segment_membrane_occupancy():
    # 8 blocks in 4 segments = 2 blocks each; the last holds 4 active voxel-samples out of 2*4
    prof = H2.revt_activation_profile([0, 0, 0, 0, 0, 0, 2, 2], n_occupied=4, n_segments=4)
    assert prof == [0.0, 0.0, 0.0, pytest.approx(0.5)]


def test_revt_activation_profile_catches_a_creeping_floor():
    """A rising R_evt floor must fail the same 1% bound that the scalar occupancy uses."""
    blocks = [0] * 400 + [1] * 400
    prof = H2.revt_activation_profile(blocks, n_occupied=10)
    assert max(prof) > H2.B0_ACTIVE_OCCUPANCY_MAX
    assert prof[0] == 0.0 and prof[-1] == pytest.approx(0.1)


def test_revt_activation_profile_rejects_empty_input():
    with pytest.raises(ValueError):
        H2.revt_activation_profile([], n_occupied=4)


def test_event_peak_values_ignores_unoccupied_voxels():
    q = np.zeros((100, 3))
    q[4, 2] = 999.0
    peaks = H2.event_peak_values(q, [1.0], np.array([True, True, False]), tau_R_ms=1.0,
                                 dt_ms=0.5, calibration_end_ms=1e9)
    assert max(peaks) == pytest.approx(0.0)


def test_b0_envelope_statistics_is_a_joint_quantile_not_a_spatial_max():
    """plan 7.1 clause 2 asks for the q99 across (events x occupied voxels).  Reducing each block to
    max_v q_v first and taking the q99 of THAT is a different, much larger statistic -- the same
    error class as the B2.1 amplitude clause.  Here 1 voxel in 100 is hot, so the joint q99 must sit
    near the cold value while a spatial max would return the hot one.

    The hot fraction is 0.2% (1 voxel in 500), deliberately WELL below the 1% the q99 cuts at: at
    exactly 1% the quantile interpolates across the cold/hot boundary and the test would assert a
    number that depends on numpy's interpolation rule rather than on the statistic being right.
    """
    pre = []
    for _ in range(8):
        w = np.full((4, 500), 1.0)
        w[:, 0] = 500.0
        pre.append(w)
    out = H2.b0_envelope_statistics(pre, Q_on=1000.0)
    assert out["pre_onset_residual_frac"] == pytest.approx(1.0 / 1000.0, rel=1e-6)
    assert out["pre_onset_residual_frac"] < 0.1        # a spatial max would give 0.5
    assert out["n_joint_samples"] == 8 * 4 * 500


def test_b0_envelope_statistics_drift_is_a_difference_over_the_same_statistic():
    pre = [np.full((4, 50), 10.0) for _ in range(8)] + [np.full((4, 50), 30.0) for _ in range(8)]
    out = H2.b0_envelope_statistics(pre, Q_on=100.0)
    assert out["q_floor_drift"] == pytest.approx(0.20, rel=1e-6)
    assert out["floor_first"] == pytest.approx(10.0)
    assert out["floor_last"] == pytest.approx(30.0)


def test_b0_envelope_statistics_reports_insufficient_below_four_events():
    out = H2.b0_envelope_statistics([np.ones((4, 10))] * 3, Q_on=1.0)
    assert out["insufficient"] and np.isnan(out["pre_onset_residual_frac"])


def test_b0_envelope_statistics_rejects_nonpositive_Q_on():
    with pytest.raises(ValueError):
        H2.b0_envelope_statistics([np.ones((4, 10))] * 8, Q_on=0.0)
