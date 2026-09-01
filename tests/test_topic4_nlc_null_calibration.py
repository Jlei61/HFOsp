"""Tests for the rev11-NLC null calibration helpers."""
from __future__ import annotations

import numpy as np
import pytest

from src.topic4_d6_natural_kmeans import (
    crossfit_patient_readout,
    normalize_event_ranks,
    patient_profiles,
)
from src.topic4_nlc_null_calibration import (
    contact_permutation_draws,
    crossfit_margin,
    direction_label_permutation_draws,
    equal_network_null,
)


def _synthetic(seed, *, n_events=40, n_contacts=15, missing=0.15):
    rng = np.random.default_rng(seed)
    axis = np.linspace(0.0, 1.0, n_contacts)
    ranks = np.empty((n_events, n_contacts))
    for event in range(n_events):
        sign = 1.0 if event % 2 == 0 else -1.0
        ranks[event] = sign * axis + rng.normal(0.0, 0.25, n_contacts)
    ranks[rng.random(ranks.shape) < missing] = np.nan
    return ranks


def _patient(seed, *, n_events=60, n_contacts=15):
    rng = np.random.default_rng(seed + 1000)
    axis = np.linspace(0.0, 1.0, n_contacts)
    ranks = np.empty((n_events, n_contacts))
    labels = np.zeros(n_events, int)
    for event in range(n_events):
        labels[event] = event % 2
        sign = 1.0 if labels[event] == 0 else -1.0
        ranks[event] = sign * axis + rng.normal(0.0, 0.15, n_contacts)
    return ranks, labels


FOLDS = (np.arange(0, 15, 2), np.arange(1, 15, 2))


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_fast_crossfit_margin_matches_reference(seed):
    """The vectorised margin must equal the audited reference implementation."""
    ranks = _synthetic(seed)
    patient_ranks, patient_labels = _patient(seed)
    reference = crossfit_patient_readout(
        ranks, patient_ranks, patient_labels, FOLDS,
    )["signed_margin"]
    fast = crossfit_margin(
        normalize_event_ranks(ranks),
        patient_profiles(patient_ranks, patient_labels),
        FOLDS,
    )
    assert reference is not None and fast is not None
    assert fast == pytest.approx(reference, abs=1e-12)


def test_contact_permutation_null_is_centered_below_the_observed_signal():
    """A model that shares the patient axis must beat its own contact null."""
    ranks = _synthetic(7)
    patient_ranks, patient_labels = _patient(7)
    observed = crossfit_margin(
        normalize_event_ranks(ranks),
        patient_profiles(patient_ranks, patient_labels),
        FOLDS,
    )
    draws = contact_permutation_draws(
        ranks, patient_ranks, patient_labels, FOLDS, draws=120, seed=11,
    )
    assert len(draws) >= 100
    assert float(np.median(draws)) < observed


def test_contact_permutation_null_accepts_a_shaft_restriction():
    ranks = _synthetic(5)
    patient_ranks, patient_labels = _patient(5)
    shaft_ids = np.asarray(["A"] * 8 + ["B"] * 7)
    draws = contact_permutation_draws(
        ranks, patient_ranks, patient_labels, FOLDS, draws=40, seed=3,
        shaft_ids=shaft_ids,
    )
    assert len(draws) >= 30


def test_direction_label_null_is_above_one_half():
    """Best-of-two matching makes the null of balanced alignment exceed 0.5."""
    rng = np.random.default_rng(4)
    cluster = rng.integers(0, 2, 48)
    direction = rng.integers(0, 2, 48)
    draws = direction_label_permutation_draws(
        cluster, direction, draws=400, seed=9,
    )
    assert len(draws) == 400
    assert float(np.median(draws)) > 0.5


def test_direction_label_null_preserves_unlabelled_events():
    cluster = np.array([0, 0, 1, 1, 0, 1])
    direction = np.array([-1, 0, 1, -1, 0, 1])
    draws = direction_label_permutation_draws(
        cluster, direction, draws=25, seed=2,
    )
    assert len(draws) == 25
    assert np.all(draws <= 1.0) and np.all(draws >= 0.0)


def test_equal_network_null_uses_equal_network_weighting():
    observed = {"1": 0.9, "2": 0.5}
    draws = {
        "1": np.full(100, 0.4),
        "2": np.full(100, 0.2),
    }
    summary = equal_network_null(observed, draws)
    assert summary["n_networks"] == 2
    assert summary["observed_equal_network_mean"] == pytest.approx(0.7)
    assert summary["null_median"] == pytest.approx(0.3)
    assert summary["observed_above_null_q95"] is True
    assert summary["one_sided_p"] == pytest.approx(1.0 / 101.0)


def test_equal_network_null_reports_a_failing_observation():
    observed = {"1": 0.1, "2": 0.1}
    draws = {"1": np.linspace(0.0, 1.0, 100), "2": np.linspace(0.0, 1.0, 100)}
    summary = equal_network_null(observed, draws)
    assert summary["observed_above_null_q95"] is False
    assert summary["one_sided_p"] > 0.5
