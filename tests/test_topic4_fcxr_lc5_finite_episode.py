import numpy as np
import pytest

from src.topic4_fcxr_lc5 import SparseSpikeStream, replay_sparse_loads
from src.topic4_fcxr_lc5_finite_episode import (
    calibrate_episode_dose,
    classify_u2_excursion,
    coarsen_sparse_stream,
    estimate_shrunken_p0,
    replay_finite_load,
    solve_a_for_window_target,
)


def _stream(n_steps=1000, n_cells=8):
    steps = np.arange(10, n_steps, 20, dtype=np.int64)
    cells = np.arange(steps.size, dtype=np.int64) % n_cells
    order = np.lexsort((cells, steps))
    return SparseSpikeStream(steps[order], cells[order], n_steps, n_cells)


def test_coarsen_preserves_spike_count_and_clock():
    s = _stream(1000, 8)
    c = coarsen_sparse_stream(s, source_dt_ms=0.05, target_dt_ms=1.0, stop_ms=50.0)
    assert c.n_steps == 50
    assert c.steps.size == s.steps.size
    assert np.array_equal(np.sort(c.cells), np.sort(s.cells))


def test_coarsen_rejects_duplicate_cell_in_bin():
    s = SparseSpikeStream(np.array([0, 10]), np.array([1, 1]), 100, 3)
    with pytest.raises(ValueError, match="twice"):
        coarsen_sparse_stream(s, source_dt_ms=0.05, target_dt_ms=1.0)


def test_finite_replay_is_monotone_in_a_load():
    s = _stream()
    kw = dict(dt_ms=0.05, tau_ms=3000.0, blocks={"e": (400, 1000)}, target_block="e")
    lo = replay_finite_load(s, a_load=0.01, **kw).target_phi_median
    hi = replay_finite_load(s, a_load=0.05, **kw).target_phi_median
    assert hi > lo


def test_zero_load_stays_zero():
    s = _stream()
    out = replay_finite_load(s, dt_ms=0.05, tau_ms=3000.0, a_load=0.0)
    assert np.array_equal(out.u_final, np.zeros(s.n_cells))


def test_replay_matches_registered_exact_implementation_on_small_stream():
    s = _stream(500, 8)
    a, tau = 0.03, 3000.0
    exact = replay_sparse_loads(
        s, candidates={"x": {"a_load": a, "tau_ms": tau, "h": 3}}, dt_ms=0.05
    )["x"]["u_final"]
    got = replay_finite_load(s, dt_ms=0.05, tau_ms=tau, a_load=a).u_final
    assert np.array_equal(got, exact)


def test_bisection_hits_finite_window_target():
    s = _stream(3000, 8)
    out = solve_a_for_window_target(
        s, dt_ms=0.05, tau_ms=3000.0, target_window=(1000, 3000),
        target=0.20, sample_every_steps=5, tolerance=2e-3,
    )
    assert out["a_load"] > 0.0
    assert abs(out["achieved_target"] - 0.20) <= 2e-3


def test_p0_shrinkage_uses_only_aligned_baseline_blocks():
    blocks = np.vstack([np.linspace(0.01 + k * 0.001, 0.2 + k * 0.001, 20) for k in range(4)])
    rates = np.linspace(0.0, 20.0, 20)
    out = estimate_shrunken_p0(blocks, rates)
    assert out["p0"].shape == rates.shape
    assert 0.0 <= out["weight"] <= 1.0
    assert np.all(np.isfinite(out["p0"]))


def test_dose_calibration_is_linear_in_gamma():
    out = calibrate_episode_dose(
        unit_excess_integral_ms=np.array([1.0, 2.0, 3.0]),
        recurrent_force_integral_ms=np.array([10.0, 20.0, 30.0]),
        gammas=(0.1, 0.25, 0.4),
    )
    vals = out["Imax_by_gamma"]
    assert vals["0.25"] / vals["0.1"] == pytest.approx(2.5)
    assert vals["0.4"] / vals["0.1"] == pytest.approx(4.0)


def test_large_load_is_kept_not_dropped():
    s = SparseSpikeStream(np.arange(100), np.zeros(100, dtype=np.int64), 100, 2)
    out = replay_finite_load(s, dt_ms=1.0, tau_ms=3000.0, a_load=1.0)
    assert out.u_final[0] > 90.0
    assert out.u_final[1] == 0.0


def test_u2_classifier_requires_terminal_two_second_low_tail():
    # A one-second rolling band needs roughly one extra raw second before it can establish a full
    # two-second terminal low tail.
    rate = np.r_[np.full(3000, 60.0), np.zeros(3000)]
    out = classify_u2_excursion(
        rate, dt_ms=1.0, interictal_upper_hz=10.0, saturated=False,
    )
    assert out["label"] == "FINITE_EXCURSION_OFFSET"
    assert out["terminal_low_ms"] >= 2000.0


def test_u2_classifier_does_not_call_a_brief_trough_offset():
    rate = np.r_[np.full(2000, 60.0), np.zeros(1000), np.full(2000, 60.0)]
    out = classify_u2_excursion(
        rate, dt_ms=1.0, interictal_upper_hz=10.0, saturated=False,
    )
    assert out["label"] == "CONTAINED_HIGH_NO_OFFSET"
    assert out["terminal_offset"] is False


def test_u2_classifier_preserves_saturation_as_distinct_failure():
    out = classify_u2_excursion(
        np.full(7000, 300.0), dt_ms=1.0, interictal_upper_hz=10.0, saturated=True,
    )
    assert out["label"] == "ESCALATING_SATURATION"
