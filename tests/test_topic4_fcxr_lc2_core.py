"""Synthetic gate tests for the LC2-Core sensor adjudication."""
import os
import sys

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import run_topic4_fcxr_lc2_core as R  # noqa: E402
from src.topic4_fcxr_lc2_core import (  # noqa: E402
    sustained_latch_score, empirical_false_latch_threshold, pareto_mask,
    select_sensor_candidates,
)


def test_event_peak_matrix_uses_returning_ieds_only():
    h = np.zeros((100, 3), float)
    h[10:21] = [1.0, 2.0, 3.0]
    h[40:51] = 99.0
    events = [
        dict(t_on_ms=10.0, t_off_ms=20.0, returned=True),
        dict(t_on_ms=40.0, t_off_ms=50.0, returned=False),
    ]
    got = R._event_peak_matrix(h, events)
    np.testing.assert_array_equal(got, np.array([[1.0, 2.0, 3.0]]))


def test_high_trough_is_temporal_q10_per_cell_then_population_q10():
    h = np.tile(np.arange(1.0, 5.0), (300, 1))
    h[:30] = 0.0  # excluded by the locked established window
    got = R._high_trough_by_cell(h, 50.0, 250.0)
    np.testing.assert_array_equal(got, np.arange(1.0, 5.0))
    assert np.quantile(got, 0.10) == 1.3


def test_bootstrap_gate_separates_clean_synthetic_low_and_high():
    low = np.full((8, 256), 1.0)
    h1 = np.full(256, 3.0)
    h2 = np.full(256, 2.5)
    b = R._bootstrap_bounds(low, h1, h2, tau_index=0)
    assert b["L_upper95"] == 1.0
    assert b["HEO1_lower95"] == 3.0
    assert b["HEO2_lower95"] == 2.5


def test_r1_scoped_adjudication_does_not_turn_gap_failure_into_loop_no_go():
    sensor = {
        "status": "H_SENSOR_NOT_SEPARABLE",
        "rows": [
            dict(tau_ms=100.0, L_upper95=2.0, HEO1_lower95=1.0, HEO2_lower95=0.2),
            dict(tau_ms=300.0, L_upper95=1.0, HEO1_lower95=1.5, HEO2_lower95=0.3),
        ],
    }
    replays = {"heo2": {"events": [
        dict(t_on_ms=100.0, t_off_ms=500.0, returned=True),
        dict(t_on_ms=1500.0, t_off_ms=1700.0, returned=True),
    ]}}
    got = R._r1_scoped_adjudication(sensor, replays)
    assert got["canonical_status"] == "R1_IMPLEMENTATION_ACCEPTED"
    assert "R1_SUSTAINED_CONTROL_SEPARABLE" in got["labels"]
    assert "R1_LONG_REST_GAP_NOT_BRIDGED" in got["labels"]
    assert "R1_CLOSED_LOOP_H_GEOMETRY_UNTESTED" in got["labels"]
    assert "H_LOOP_NO_BISTABILITY" not in got["labels"]
    assert got["heo2_first_rest_like_gap_ms"] == 1000.0


def test_sustained_latch_score_rejects_a_one_sample_peak():
    x = np.array([0.0, 5.0, 2.0, 2.0, 2.0, 0.0])
    assert sustained_latch_score(x, 3) == 2.0


def test_empirical_threshold_respects_discrete_event_resolution():
    scores = np.arange(1.0, 10.0)
    t0, p0 = empirical_false_latch_threshold(scores, 0.0)
    t10, p10 = empirical_false_latch_threshold(scores, 0.10)
    t25, p25 = empirical_false_latch_threshold(scores, 0.25)
    assert (t0, p0) == (9.0, 0.0)
    assert (t10, p10) == (9.0, 0.0)  # 9 events cannot resolve 5% or 10%
    assert (t25, p25) == (7.0, 2 / 9)


def test_pareto_and_role_selection_are_deterministic_and_unique():
    rows = [
        dict(tau_ms=20.0, theta=1.0, false_latch_fraction=0.0,
             heo1_support_duty=0.5, heo2_active_support_duty=0.4, gap_persistence=0.1),
        dict(tau_ms=100.0, theta=2.0, false_latch_fraction=0.0,
             heo1_support_duty=0.9, heo2_active_support_duty=0.8, gap_persistence=0.4),
        dict(tau_ms=500.0, theta=3.0, false_latch_fraction=0.2,
             heo1_support_duty=0.95, heo2_active_support_duty=0.9, gap_persistence=0.8),
        dict(tau_ms=200.0, theta=4.0, false_latch_fraction=0.2,
             heo1_support_duty=0.2, heo2_active_support_duty=0.1, gap_persistence=0.05),
    ]
    got = pareto_mask(rows, ["false_latch_fraction"],
                      ["heo1_support_duty", "heo2_active_support_duty", "gap_persistence"])
    assert got.tolist() == [False, True, True, False]  # 100 ms dominates the weaker same-p row
    chosen = select_sensor_candidates(rows, max_n=3)
    ids = [(r["tau_ms"], r["theta"]) for r in chosen]
    assert len(ids) == len(set(ids)) == 2  # fewer than max_n is legal when the Pareto set is smaller
    assert ids[0] == (100.0, 2.0)  # locked fastest non-dominated p_false=0 role
