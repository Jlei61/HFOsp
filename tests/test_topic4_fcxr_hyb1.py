"""FCXR-HYB1 gate and bad-data-regression tests.

plan 5: "classifier 必须先用这些坏数据回归，再有资格判新候选" -- a classifier that cannot fail
q75 / q50 / q50-without-X is not allowed to pass anything.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.topic4_fcxr_hyb1 as H          # noqa: E402


# --------------------------------------------------------------- plan 2.2 deadband
def test_deadband_is_exactly_zero_at_and_below_background():
    u = np.array([-5.0, -1e-9, 0.0])
    assert np.all(H.deadband_positive(u, eps=0.1) == 0.0)


def test_deadband_is_C1_at_the_origin_so_the_source_has_no_kink():
    eps, h = 0.1, 1e-6
    d = (H.deadband_positive(np.array([h]), eps)[0]
         - H.deadband_positive(np.array([0.0]), eps)[0]) / h
    assert d == pytest.approx(0.0, abs=1e-4)


def test_deadband_is_asymptotically_the_shifted_identity():
    eps = 0.1
    u = 1000.0
    assert H.deadband_positive(np.array([u]), eps)[0] == pytest.approx(u - eps, rel=1e-3)


def test_a_softplus_style_leak_would_be_caught():
    """Guard against silently swapping in softplus: it is positive everywhere, which pours a
    source into every interictal voxel at every step and destroys the structural fixed point."""
    leak = float(np.log1p(np.exp(-5.0 / 0.1)) * 0.1)      # what softplus would give at u=-5
    assert H.deadband_positive(np.array([-5.0]), 0.1)[0] == 0.0
    assert leak >= 0.0            # softplus is never exactly zero -- that is the failure mode


def test_background_envelope_is_per_voxel_not_global():
    rng = np.random.default_rng(0)
    load = np.stack([rng.normal(1.0, 0.01, 400), rng.normal(9.0, 0.01, 400)], axis=1)
    b = H.background_envelope(load)
    assert b.shape == (2,) and b[0] < 2.0 < b[1]


def test_deadband_eps_scales_with_the_measured_background():
    assert H.deadband_eps(np.array([2.0, 4.0])) == pytest.approx(0.3)


# --------------------------------------------------------------- plan 3 Z hazard axis
def _survival_fixture(n=4000, seed=0):
    """A monotone sensor distribution wide enough to bracket both anchors."""
    rng = np.random.default_rng(seed)
    I = rng.lognormal(mean=np.log(6.0), sigma=1.6, size=n)
    p = np.full(n, 1.0 / n)
    return I, p


def test_hazard_is_the_exact_t0_slope_of_D_Z():
    """dD_Z/dt|0 = a_p/tau_z is an identity, not a fit: every z_i starts at 1."""
    I = np.array([10.0, 10.0, 0.0, 0.0])
    p = np.ones(4)
    h, a_p = H.hazard_from_survival(I, p, theta=5.0, tau_z_ms=5000.0)
    assert a_p == pytest.approx(0.5)
    assert h == pytest.approx(0.5 / 5.0)          # tau_z = 5 s


def test_hazard_uses_the_p_weights_not_a_plain_count():
    I = np.array([10.0, 0.0])
    h_a, a_a = H.hazard_from_survival(I, np.array([9.0, 1.0]), 5.0, 1000.0)
    h_b, a_b = H.hazard_from_survival(I, np.array([1.0, 9.0]), 5.0, 1000.0)
    assert a_a == pytest.approx(0.9) and a_b == pytest.approx(0.1)


def test_hazard_rejects_mismatched_weights():
    with pytest.raises(ValueError):
        H.hazard_from_survival(np.zeros(4), np.zeros(3), 1.0, 1000.0)


def test_survival_inversion_round_trips_to_the_requested_hazard():
    I, p = _survival_fixture()
    for target in (0.05, 0.10, 0.20):
        th = H.invert_survival_for_theta(I, p, target)
        _, got = H.hazard_from_survival(I, p, th, H.TAU_Z_DOWN_MS)
        assert got <= target + 1.0 / I.size


def test_three_hazard_levels_are_strictly_between_the_two_observed_anchors():
    lo, hi = H.H_LO_HI
    assert lo < hi
    for v in H.H_TARGETS.values():
        assert lo < v < hi


def test_hazard_levels_are_geometrically_spaced_so_no_level_hugs_an_anchor():
    v = [H.H_TARGETS[k] for k in ("H_LO", "H_MID", "H_HI")]
    assert v[1] / v[0] == pytest.approx(v[2] / v[1], rel=1e-9)


def test_the_anchor_observable_is_the_D_Z_SLOPE_not_the_run_average():
    """Regression on the error the first pre-registered attempt made.  D_Z saturates, so its 24 s
    average is ~3x below its initial slope; comparing an exact t=0 prediction against that average
    is a dimensional mismatch, not a finding about the model."""
    q75 = H.Z_ANCHORS["q75_seed1"]["h_Z"] / H.Z_ANCHORS_SUPERSEDED_AVG["q75_seed1"]
    q50 = H.Z_ANCHORS["q50_seed1"]["h_Z"] / H.Z_ANCHORS_SUPERSEDED_AVG["q50_seed1"]
    # q75 saturates hard within the run (slope ~3x its own average); q50 barely does (~1.04x).
    # That asymmetry is exactly why the mis-specified comparison failed on q75 alone -- so the
    # failure was never evidence about the survival curve.
    assert q75 > 2.5 and q50 == pytest.approx(1.0, abs=0.1)
    for k in H.Z_ANCHORS:
        assert "slope" in H.Z_ANCHORS[k]["source"]


def test_the_seed3_anchor_is_provenance_only_not_an_identifiability_check():
    """A seed-1 survival curve cannot predict a seed-3 substrate; keeping it as a gate would fail
    the axis for a reason that has nothing to do with identifiability."""
    assert "q75_seed3" not in H.Z_ANCHORS
    assert "q75_seed3" in H.Z_ANCHORS_PROVENANCE_ONLY


def test_the_corrected_bracket_is_much_narrower_than_the_superseded_one():
    """Recorded because it changes what the screen can learn: on the correct statistic q75 and q50
    differ by 1.6x in INITIAL hazard, not 4.3x -- what really differs is how fast the depletion
    self-limits."""
    assert H.H_LO_HI[1] / H.H_LO_HI[0] == pytest.approx(1.638, abs=0.01)


def test_z_axis_is_blocked_when_the_probe_cannot_reproduce_the_anchors():
    """A sensor distribution concentrated far below both anchors predicts ~zero hazard for each,
    which must block the axis instead of silently producing three thresholds."""
    I = np.full(1000, 1e-3)
    v = H.adjudicate_z_axis(I, np.ones(1000))
    assert v["status"] == "DESIGN_BLOCKED_Z_AXIS"


def test_z_axis_survival_must_be_monotone_in_the_threshold():
    I, p = _survival_fixture()
    v = H.adjudicate_z_axis(I, p)
    assert v["checks"]["monotone"]["ok"]


# --------------------------------------------------------------- plan 2.5 baseline preservation
def _base_ok(**over):
    m = dict(dk_duty=0.005, dk_q99_mM=0.001, event_rate_in_band=True, iei_cv_in_band=True,
             iei_cv=0.9, duration_in_band=True, participation_in_band=True,
             clip_frac_max=0.0, numerical_unsafe=False)
    m.update(over)
    return m


def test_baseline_preserved_when_every_clause_holds():
    assert H.adjudicate_baseline_preservation(_base_ok())["status"] == "BASELINE_PRESERVED"


def test_baseline_disturbed_when_the_excess_field_is_active_too_often():
    v = H.adjudicate_baseline_preservation(_base_ok(dk_duty=0.25))
    assert v["status"] == "STOP_BASELINE_DISTURBED" and not v["checks"]["dk_duty"]["ok"]


def test_baseline_disturbed_when_interictal_potassium_rises():
    v = H.adjudicate_baseline_preservation(_base_ok(dk_q99_mM=0.4))
    assert v["status"] == "STOP_BASELINE_DISTURBED"


def test_baseline_disturbed_when_the_interictal_train_becomes_regular():
    """A periodic train has a low IEI CV; it must not count as a preserved baseline even if the
    event RATE is unchanged."""
    v = H.adjudicate_baseline_preservation(_base_ok(iei_cv=0.05))
    assert v["status"] == "STOP_BASELINE_DISTURBED"


def test_baseline_disturbed_on_any_conductance_clip():
    assert H.adjudicate_baseline_preservation(
        _base_ok(clip_frac_max=1e-9))["status"] == "STOP_BASELINE_DISTURBED"


# --------------------------------------------------------------- plan 5 the seven gates
def _run(**over):
    """A synthetic run that passes every gate; each regression perturbs exactly one thing."""
    m = dict(kick_boost=0.0, t_kick_ms=1e9, onset_detected=True, pre_interictal_ms=9000.0,
             bout_ms=2500.0, bounded=True, clip_frac_max=0.0, finite=True, numerical_unsafe=False,
             end_rate_hz=9.0, recruit_contacts=14, onset_gradient_r2=0.6,
             x_activation_delay_ms=400.0, post_return_ms=9000.0, label="RECOVERED_INTERICTAL",
             post_iei_cv=0.8, post_event_rate_hz=1.0, band_event_rate=(0.3, 3.0),
             post_duration_ms=40.0, band_duration=(10.0, 90.0),
             post_participation=0.05, band_participation=(0.01, 0.2), post_silent=False)
    m.update(over)
    return m


def test_a_fully_passing_run_is_a_candidate_only_with_a_passing_spatial_leg():
    assert H.adjudicate_lifecycle(_run(), spatial_leg="PASS")["status"] == "LIFECYCLE_CANDIDATE"


def test_an_unresolved_spatial_leg_downgrades_rather_than_passes_or_fails():
    v = H.adjudicate_lifecycle(_run(), spatial_leg="UNRESOLVED")
    assert v["status"] == "LIFECYCLE_CANDIDATE_SPATIAL_UNRESOLVED"
    assert not v["gates"]["4_spatial"]["ok"]


def test_spatial_leg_must_be_one_of_three_labels():
    with pytest.raises(ValueError):
        H.adjudicate_lifecycle(_run(), spatial_leg="probably")


# --- the bad-data regressions: the classifier must FAIL these before it may pass anything -------
def test_BAD_DATA_q75_must_fail_gate_3_no_bounded_high_state():
    """LC1 q75: keeps a long interictal run and stays bounded, but the dense episodes
    self-extinguish, so there is no 1-5 s ictal-like bout and X never engages."""
    q75 = _run(pre_interictal_ms=10000.0, bout_ms=None, label="DENSE_EVENT_TRAIN",
               end_rate_hz=8.853, x_activation_delay_ms=None, post_return_ms=0.0)
    v = H.adjudicate_lifecycle(q75, spatial_leg="PASS")
    assert v["status"] == "NOT_A_CANDIDATE"
    assert "3_bounded_high_state" in v["failed"]
    assert "persistence" in v["failure_layer"]


def test_BAD_DATA_q50_must_fail_gate_2_and_gate_6():
    """LC1 q50 + X: a real sustained bout that X terminates, but onset arrives at 3 s and Z is
    left depleted, so there is neither an 8 s prelude nor a statistical recovery."""
    q50 = _run(pre_interictal_ms=3000.0, bout_ms=2000.0, label="PERMANENT_SILENCE",
               post_return_ms=0.0, post_iei_cv=0.0, post_event_rate_hz=0.0, post_silent=True,
               end_rate_hz=4.1)
    v = H.adjudicate_lifecycle(q50, spatial_leg="PASS")
    assert v["status"] == "NOT_A_CANDIDATE"
    assert "2_pre_interictal" in v["failed"] and "6_statistical_recovery" in v["failed"]
    assert set(v["failure_layer"]) >= {"onset", "recovery"}


def test_BAD_DATA_q50_without_X_must_fail_gate_7():
    """LC1 q50 Z-only: 452.8 Hz at the end, bounded=False -- the runaway the guard exists for."""
    q50nx = _run(pre_interictal_ms=3000.0, bout_ms=None, bounded=False, end_rate_hz=452.8430625,
                 label="ICTAL_LIKE_BOUNDED", post_return_ms=0.0, x_activation_delay_ms=None)
    v = H.adjudicate_lifecycle(q50nx, spatial_leg="PASS")
    assert v["status"] == "NOT_A_CANDIDATE" and "7_numerical" in v["failed"]


def test_permanent_silence_can_never_be_read_as_recovery():
    v = H.adjudicate_lifecycle(_run(label="PERMANENT_SILENCE", post_silent=True,
                                    post_event_rate_hz=0.0, post_iei_cv=0.0), spatial_leg="PASS")
    assert not v["gates"]["6_statistical_recovery"]["ok"]


def test_rapid_relapse_can_never_be_read_as_recovery():
    v = H.adjudicate_lifecycle(_run(label="RAPID_RELAPSE", post_return_ms=1500.0),
                               spatial_leg="PASS")
    assert not v["gates"]["6_statistical_recovery"]["ok"]


def test_a_fixed_periodic_train_is_not_a_recovered_neighbourhood():
    """The recovery target is a statistical neighbourhood that keeps producing sparse IRREGULAR
    events -- a clock-like train has the right rate and the wrong CV."""
    v = H.adjudicate_lifecycle(_run(post_iei_cv=0.04), spatial_leg="PASS")
    assert not v["gates"]["6_statistical_recovery"]["ok"]


def test_recovery_needs_the_post_rate_inside_the_baseline_band_not_merely_nonzero():
    v = H.adjudicate_lifecycle(_run(post_event_rate_hz=25.0), spatial_leg="PASS")
    assert not v["gates"]["6_statistical_recovery"]["ok"]


def test_a_kicked_run_can_never_be_a_spontaneous_lifecycle():
    v = H.adjudicate_lifecycle(_run(kick_boost=6.0, t_kick_ms=2500.0), spatial_leg="PASS")
    assert not v["gates"]["1_spontaneous"]["ok"] and "onset" in v["failure_layer"]


def test_X_engaging_before_onset_is_not_a_termination_mechanism():
    v = H.adjudicate_lifecycle(_run(x_activation_delay_ms=-50.0), spatial_leg="PASS")
    assert not v["gates"]["5_x_after_onset"]["ok"] and "termination" in v["failure_layer"]


def test_X_engaging_inside_the_sensor_resolution_is_not_yet_causal_ordering():
    v = H.adjudicate_lifecycle(_run(x_activation_delay_ms=40.0), spatial_leg="PASS")
    assert not v["gates"]["5_x_after_onset"]["ok"]


def test_a_bout_longer_than_five_seconds_is_not_the_target_phenotype():
    assert not H.adjudicate_lifecycle(_run(bout_ms=9000.0),
                                      spatial_leg="PASS")["gates"]["3_bounded_high_state"]["ok"]


# --------------------------------------------------------------- plan 5.1 spatial separation
def test_spatial_leg_is_unresolved_when_recruitment_saturates_in_both_classes():
    """HEO2.1 measured recruit >=13/15 in 48/48 working points including the synchronous tonic
    state, so recruitment alone cannot separate structured events from the control."""
    v = H.adjudicate_spatial_separation(
        structured=dict(recruit=[15, 14, 15], onset_gradient_r2=[0.31, 0.28, 0.35]),
        synchronous=dict(recruit=[15, 15, 15], onset_gradient_r2=[0.30, 0.33, 0.36]))
    assert v["leg"] == "UNRESOLVED" and not v["recruit_separates"]


def test_spatial_leg_passes_only_when_a_leg_actually_separates():
    v = H.adjudicate_spatial_separation(
        structured=dict(recruit=[15, 15], onset_gradient_r2=[0.71, 0.68]),
        synchronous=dict(recruit=[15, 15], onset_gradient_r2=[0.10, 0.12]))
    assert v["leg"] == "PASS" and v["onset_gradient_separates"]


# --------------------------------------------------------------- locked constants
def test_locked_constants_match_the_plan():
    assert (H.Q_BG, H.EPS_FRAC, H.G_DELTA_K, H.ETA_PUMP) == (0.99, 0.10, 1.0, 0.0)
    assert H.Q_K == pytest.approx(0.013615797289152352)
    assert (H.TAU_Z_DOWN_MS, H.TAU_Z_UP_MS) == (5000.0, 20000.0)
    assert (H.PRE_MIN_MS, H.BOUT_MIN_MS, H.BOUT_MAX_MS, H.POST_MIN_MS) == (8000., 1000., 5000., 8000.)
    assert (H.RECRUIT_MIN, H.X_DELAY_MIN_MS, H.IEI_CV_MIN) == (12, 100.0, 0.5)
