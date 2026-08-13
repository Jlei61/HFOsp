import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "lc5v2_prefix", ROOT / "scripts/run_topic4_fcxr_lc5v2_natural_prefix.py"
)
PREFIX = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREFIX)


def test_prefix_tags_are_unambiguous():
    assert PREFIX._tag(0.0).endswith("milli000")
    assert PREFIX._tag(0.001).endswith("milli001")
    assert PREFIX._tag(0.003).endswith("milli003")
    assert PREFIX._tag(0.005, "q099").endswith("milli005")
    with pytest.raises(ValueError):
        PREFIX._tag(0.0015)
    assert PREFIX._tag(0.001, "q099").startswith("u3_prefix_q099_")
    assert "tau3" in PREFIX._tag(0.010, "q099", 3000.0)
    assert "tau15" in PREFIX._tag(0.010, "q099", 15000.0)
    assert PREFIX._tag(0.010, "q099", 3000.0) != PREFIX._tag(0.010, "q099", 15000.0)
    with pytest.raises(ValueError):
        PREFIX._tag(0.010, "q099", 6000.0)
    assert PREFIX._tag(0.010, "q099", 3000.0, "lc5v2p1_map").startswith(
        "lc5v2p1_map_q099_"
    )
    with pytest.raises(ValueError):
        PREFIX._tag(0.010, "q099", 3000.0, "LC5 bad")


def test_prefix_is_fresh_u0_always_online_contract():
    source = Path(PREFIX.__file__).read_text()
    assert "pump_u_init_E=np.zeros" in source
    assert 'use_pump=True' in source
    assert 'runtime_semantics\": \"fresh_t0_u0_always_online_no_step' in source
    assert "pump_tau_ms=float(tau_ms)" in source


def test_outcome_never_calls_no_onset_an_offset():
    outcome, onset, offset = PREFIX._outcome(
        {"label": "INTERICTAL_BASELINE"}, ["INTERICTAL"] * 20, saturated=False
    )
    assert outcome == "NO_NATURAL_ONSET"
    assert onset is None and offset is None


def test_disabled_optional_current_has_zero_peak():
    assert PREFIX._safe_peak([]) == 0.0
    assert PREFIX._safe_peak([0.0, 0.0]) == 0.0
    assert PREFIX._safe_peak([0.0, 2.5, 1.0]) == pytest.approx(2.5)


def test_u_tail_diagnostic_reports_quantiles_slope_and_release_time():
    out = PREFIX._u_tail_diagnostics(
        [1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0], 3000.0,
        [0.0, 0.1, 0.2, 0.3], 500.0,
    )
    assert out["u_q50_q90_q99_max"][0] == pytest.approx(2.5)
    assert out["u_q50_q90_q99_max"][-1] == pytest.approx(4.0)
    assert out["release_time_s_q50_q90_q99_max"][0] == pytest.approx(7.5)
    assert out["u_mean_slope_last_1s_per_s"] > 0


def test_active_protocol_allows_event_aligned_limits_but_legacy_defaults_remain_fixed():
    assert PREFIX._target_end_ms(11000.0, 18000.0, 25000.0, 7000.0) == 18000.0
    assert PREFIX._target_end_ms(15000.0, 18000.0, 25000.0, 7000.0) == 22000.0
    assert PREFIX._target_end_ms(20000.0, 18000.0, 25000.0, 7000.0) == 25000.0
    assert PREFIX._target_end_ms(None, 18000.0, 25000.0, 7000.0) == 25000.0


def test_q99_boundary_dose_is_analytic_scaling_not_refit():
    summary = {
        "Imax_by_gamma": {"0.005": 2.5},
        "recurrent_force_integral_median_ms": 100.0,
        "selected_excess_integral_median_ms": 2.0,
    }
    assert PREFIX._q99_imax(summary, 0.005) == pytest.approx(2.5)
    assert PREFIX._q99_imax(summary, 0.007) == pytest.approx(0.007 * 100.0 / 2.0)
    bad = dict(summary, selected_excess_integral_median_ms=0.0)
    with pytest.raises(ValueError, match="finite and positive"):
        PREFIX._q99_imax(bad, 0.007)
