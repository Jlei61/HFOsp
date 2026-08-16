"""Contracts for the LC6B outcome-distance adjudicator."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.topic4_fcxr_lc6b_outcome import (  # noqa: E402
    PHASE_ALIGN_WINDOW_MS, RATE_BIN_MS, coarse_spatial_map, outcome_distance,
    per_cell_rate_vector, phase_aligned_correlation, population_rate,
)


def _train(period_bins=5, n_bins=100, amplitude=200.0, phase=0):
    """A burst train: one loud bin per period, silence otherwise."""
    rate = np.zeros(n_bins)
    rate[(np.arange(n_bins) + phase) % period_bins == 0] = amplitude
    return rate


def _case(rate_hz, area, cells, pop, spatial):
    return {"final_second_rate_hz": rate_hz, "median_active_area_mm2": area,
            "per_cell_rate_vector": cells, "population_rate": pop, "coarse_spatial_map": spatial}


# ------------------------------------------------------------------ readouts

def test_per_cell_rate_vector_counts_only_the_tail():
    steps = np.array([0, 1, 199_000, 199_500], np.int64)      # 0.05 ms steps -> last two are late
    cells = np.array([0, 1, 2, 2], np.int64)
    vector = per_cell_rate_vector(steps, cells, n_steps=200_000, n_cells=4, tail_ms=1000.0)
    assert vector.tolist() == [0.0, 0.0, 2.0, 0.0]


def test_population_rate_is_in_hz_per_cell():
    # one spike from each of 10 cells inside a single 20 ms bin of a 1 s tail
    steps = np.full(10, 199_990, np.int64)
    rate = population_rate(steps, n_steps=200_000, n_cells=10, tail_ms=1000.0, bin_ms=20.0)
    assert rate.size == 50
    assert rate[-1] == pytest.approx(1.0 / 10 * 10 / 0.020)   # 10 spikes / 10 cells / 20 ms
    assert rate[:-1].sum() == 0.0


def test_coarse_spatial_map_normalises_by_occupancy():
    steps = np.array([199_000, 199_100, 199_200], np.int64)
    cells = np.array([0, 1, 2], np.int64)
    cell_bins = np.array([0, 0, 1, 1])          # cells 0,1 -> bin 0 ; cells 2,3 -> bin 1
    occupancy = np.array([2.0, 2.0])
    field = coarse_spatial_map(steps, cells, cell_bins, occupancy, n_steps=200_000, tail_ms=1000.0)
    assert field.tolist() == [1.0, 0.5]


# ------------------------------------------------------------------ phase alignment

def test_zero_lag_correlation_is_negative_for_the_same_train_at_a_different_phase():
    """This is why the adjudicator cannot use a zero-lag comparison.

    Every non-saturated outcome here is a burst train; two runs of one field settle on the same
    rhythm at different phase, and a zero-lag rule would call that a different end state.
    """
    a, b = _train(phase=0), _train(phase=2)
    assert np.corrcoef(a, b)[0, 1] < 0
    r, lag = phase_aligned_correlation(a, b, window_ms=PHASE_ALIGN_WINDOW_MS)
    assert r == pytest.approx(1.0)
    assert lag != 0.0


def test_phase_alignment_still_separates_genuinely_different_trains():
    """Guards the test above: alignment must not make everything look identical."""
    a = _train(period_bins=5)
    b = np.full_like(a, a.mean())                 # same mean, no burst structure at all
    r, _lag = phase_aligned_correlation(a, b)
    assert r < 0.2


def test_phase_alignment_rejects_a_window_shorter_than_the_lag_search():
    with pytest.raises(ValueError, match="window too short"):
        phase_aligned_correlation(np.zeros(10), np.zeros(10), window_ms=200.0, max_lag_ms=200.0)


# ------------------------------------------------------------------ verdict

def test_same_regime_at_different_phase_is_not_reported_as_a_split():
    cells = np.linspace(1.0, 5.0, 64)
    spatial = np.linspace(0.5, 2.0, 32)
    low = _case(25.0, 90.0, cells, _train(phase=0), spatial)
    high = _case(26.7, 89.0, cells * 1.01, _train(phase=2), spatial * 1.01)
    out = outcome_distance(low, high)
    assert out["population_rate_zero_lag_correlation"] < 0
    assert out["phase_aligned_population_rate_correlation"] > 0.9
    assert out["same_outcome_regime"] is True
    assert out["verdict"] == "NO_MACROSCOPIC_INITIALISATION_SPLIT_DETECTED"
    assert out["failed_checks"] == []


def test_a_genuine_difference_is_reported_as_a_candidate_not_a_demonstration():
    cells = np.linspace(1.0, 5.0, 64)
    spatial = np.linspace(0.5, 2.0, 32)
    low = _case(25.0, 90.0, cells, _train(phase=0), spatial)
    high = _case(250.0, 400.0, cells[::-1] * 8, np.full(100, 250.0), spatial[::-1] * 8)
    out = outcome_distance(low, high)
    assert out["same_outcome_regime"] is False
    assert out["verdict"] == "INITIALISATION_SPLIT_CANDIDATE_PENDING_PERTURBATION_AND_SECOND_STREAM"
    assert "final_second_rate" in out["failed_checks"]
    assert "median_active_area" in out["failed_checks"]


def test_a_per_cell_difference_alone_is_enough_to_flag():
    """Two runs can share every scalar and still recruit different cells."""
    pop = _train(phase=0)
    spatial = np.linspace(0.5, 2.0, 32)
    cells = np.linspace(1.0, 5.0, 64)
    low = _case(25.0, 90.0, cells, pop, spatial)
    high = _case(25.0, 90.0, np.random.default_rng(3).permutation(cells), pop, spatial)
    out = outcome_distance(low, high)
    assert out["failed_checks"] == ["per_cell_rate_vector"]
    assert out["same_outcome_regime"] is False


def test_the_verdict_states_that_common_noise_cannot_be_separated():
    cells = np.linspace(1.0, 5.0, 64)
    spatial = np.linspace(0.5, 2.0, 32)
    out = outcome_distance(_case(25.0, 90.0, cells, _train(), spatial),
                           _case(25.0, 90.0, cells, _train(), spatial))
    assert "common-noise" in out["claim_boundary"]
    assert "no macroscopic split was detected" in out["claim_boundary"]


def test_thresholds_are_declared_in_the_verdict():
    cells = np.linspace(1.0, 5.0, 64)
    spatial = np.linspace(0.5, 2.0, 32)
    out = outcome_distance(_case(25.0, 90.0, cells, _train(), spatial),
                           _case(25.0, 90.0, cells, _train(), spatial))
    assert set(out["thresholds"]) == {
        "tail_ms", "rate_bin_ms", "phase_align_window_ms", "max_lag_ms",
        "rate_relative_tolerance", "area_relative_tolerance", "per_cell_correlation_floor",
        "phase_aligned_correlation_floor", "spatial_map_correlation_floor"}
    assert out["thresholds"]["rate_bin_ms"] == RATE_BIN_MS
