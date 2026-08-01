"""Synthetic gate tests for the LC2-Core sensor adjudication."""
import os
import importlib.util
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


def test_closed_loop_exploration_cli_imports_all_runtime_dependencies():
    """Catch stale module names before a 40k stage is submitted."""
    path = os.path.join(ROOT, "scripts", "run_topic4_fcxr_lc2_explore.py")
    spec = importlib.util.spec_from_file_location("_lc2_explore_import_smoke", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.cmd_screen_manifest)
    assert module.SCREEN_WORKER_CHOICES == (1, 2, 3, 4)
    rows = [dict(index=i, candidate_id=c, rho_fraction=r, k_ratio=k)
            for i, (c, r, k) in enumerate((("H1", .2, .05), ("H2", .1, .05),
                                            ("H1", .1, .05), ("H1", .1, .1)))]
    got = module.screen_submission_order(rows)
    assert [(x["rho_fraction"], x["k_ratio"], x["candidate_id"]) for x in got] == [
        (.1, .05, "H1"), (.1, .05, "H2"), (.1, .1, "H1"), (.2, .05, "H1")]


def _load_fork_runner():
    path = os.path.join(ROOT, "scripts", "run_topic4_fcxr_lc2_forks.py")
    spec = importlib.util.spec_from_file_location("_lc2_fork_import_smoke", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_fork_runner_imports_and_locks_arm_specific_duration():
    f = _load_fork_runner()
    assert f.MAX_FINALISTS_UNDER_MEASURED_BUDGET == 2
    assert f._duration_ms(500.0, "C") == 2500.0
    assert f._duration_ms(2000.0, "C") == 5000.0
    assert f._duration_ms(500.0, "D1") == 3000.0
    assert f._duration_ms(2000.0, "D2") == 8000.0
    assert callable(f.cmd_provisional_manifest)


def test_frozen_fork_finalist_selection_includes_an_adjacent_survivor():
    f = _load_fork_runner()
    base = dict(label="screen_survivor", candidate_id="H1", tau_ms=500.0,
                false_latch_fraction=0.0, refractory_ceiling_fraction=0.0,
                tail_gH_mean=2.0, tail_gA_mean=4.0, k_ratio=0.05)
    rows = [
        dict(base, run_id="anchor", rho=2.16, rho_fraction=0.10),
        dict(base, run_id="adjacent", rho=4.32, rho_fraction=0.20),
        dict(base, run_id="far", rho=10.8, rho_fraction=0.50),
    ]
    got = f.select_finalists(rows, max_n=3)
    assert [r["run_id"] for r in got[:2]] == ["anchor", "adjacent"]


def test_frozen_fork_window_classifier_separates_low_and_high():
    f = _load_fork_runner()
    low = f._window_metrics(np.full(100, 5.0), np.zeros((100, 4), bool),
                            np.full(100, 0.2), theta=1.0, tau_ref=2.0,
                            dt=1.0, window_ms=100.0)
    high_spk = np.zeros((100, 4), bool)
    high_spk[::5] = True
    high = f._window_metrics(np.full(100, 40.0), high_spk,
                             np.full(100, 2.0), theta=1.0, tau_ref=2.0,
                             dt=1.0, window_ms=100.0)
    assert low["low_like"] and not low["high_like"]
    assert high["high_like"] and not high["low_like"]


def _load_dynamic_runner():
    path = os.path.join(ROOT, "scripts", "run_topic4_fcxr_lc2_dynamic.py")
    spec = importlib.util.spec_from_file_location("_lc2_dynamic_import_smoke", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dynamic_runner_locks_accepted_x_authority_and_q75_entry():
    d = _load_dynamic_runner()
    assert d.Z_CFG == dict(regime="q75", I_th_EI=95.19851312666987, tau_z=5000.0)
    assert d.X_CFG["tau_x_down"] < d.Z_CFG["tau_z"] <= d.X_CFG["tau_x_up"]
    assert d.MAX_DYNAMIC_CANDIDATES == 1


def test_dynamic_lifecycle_readout_finds_sustained_interval_not_brief_pulse():
    d = _load_dynamic_runner()
    rate = np.zeros(4000)  # dt=1 ms
    rate[200:300] = 100.0  # too brief
    rate[1000:2600] = 40.0
    got = d.lifecycle_readout(rate, 1.0, [])
    assert len(got["high_intervals_ms"]) == 1
    assert got["onset_ms"] is not None and got["offset_ms"] is not None


def test_dynamic_recovery_requires_statistics_not_one_late_event():
    d = _load_dynamic_runner()
    baseline = dict(n_returning=10, T=10000.0,
                    event_durations_ms=[10.0] * 10, event_participation=[0.05] * 10)
    one = [dict(t_on_ms=11000.0, dur_ms=10.0, peak_ext=0.05, returned=True)]
    got = d.recovery_stats(one, offset_ms=2000.0, T_ms=20000.0, baseline=baseline)
    assert not got["statistical_neighbourhood_match"]
    many = [dict(t_on_ms=float(t), dur_ms=10.0, peak_ext=0.05, returned=True)
            for t in (3000, 3400, 5000, 5400, 7000, 7400, 9000, 9400,
                      11000, 11400, 13000, 13400, 15000, 15400, 17000, 17400, 19000, 19400)]
    got = d.recovery_stats(many, offset_ms=2000.0, T_ms=20000.0, baseline=baseline)
    assert got["statistical_neighbourhood_match"]
