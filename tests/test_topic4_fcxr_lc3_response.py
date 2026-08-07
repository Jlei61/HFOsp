"""Tests for the empirical response operator.

Two failures are guarded. The first is claiming a dominant mode from a response that has none --
the same error the spatial-mode reader had to be protected from. The second is quieter: if the
paired difference is dropped, the noise between two runs swamps the perturbation entirely, and the
operator measures the generator rather than the network.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.topic4_fcxr_lc3_response import (
    alignment,
    bin_response,
    gaussian_basis,
    paired_difference,
    response_operator,
)

GRID, L = 8, 20.0
NB = GRID * GRID


def _pos(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(0, L, n), rng.uniform(0, L, n)])


def _basis_grid(pos, n_side=3):
    P, _ = gaussian_basis(pos, n_side=n_side, sigma_mm=2.5, L=L)
    return np.column_stack([bin_response(P[:, k], pos, grid=GRID, L=L)
                            for k in range(P.shape[1])]), P


def test_the_basis_is_smooth_normalised_and_covers_the_sheet():
    P, centres = gaussian_basis(_pos(), n_side=3, sigma_mm=2.5, L=L)
    assert P.shape[1] == 9 and centres.shape == (9, 2)
    assert np.allclose(np.linalg.norm(P, axis=0), 1.0)
    assert centres.min() > 0 and centres.max() < L


def test_a_known_operator_is_recovered_from_its_own_responses():
    pos = _pos()
    Pg, _ = _basis_grid(pos)
    rng = np.random.default_rng(1)
    K_true = rng.normal(size=(NB, NB)) * 0.01
    out = response_operator(K_true @ Pg, Pg)
    assert np.allclose(out["operator"] @ Pg, K_true @ Pg, atol=1e-8)


def test_the_leading_direction_is_the_one_that_is_amplified():
    """A rank-one operator must report its own input direction and its own output shape."""
    pos = _pos()
    Pg, _ = _basis_grid(pos)
    v = Pg[:, 0] / np.linalg.norm(Pg[:, 0])
    u = Pg[:, -1] / np.linalg.norm(Pg[:, -1])
    K_true = 12.0 * np.outer(u, v)
    out = response_operator(K_true @ Pg, Pg)
    assert out["leading_share"] > 0.99
    assert abs(np.corrcoef(out["response_pattern"], u)[0, 1]) > 0.99


def test_an_operator_with_no_dominant_direction_is_not_reported_as_having_one():
    pos = _pos()
    Pg, _ = _basis_grid(pos)
    K_true = np.eye(NB) * 3.0                 # isotropic: every direction equally amplified
    out = response_operator(K_true @ Pg, Pg)
    assert out["leading_share"] < 0.5


def test_the_shapes_must_match_rather_than_broadcast_silently():
    pos = _pos()
    Pg, _ = _basis_grid(pos)
    with pytest.raises(ValueError, match="share a shape"):
        response_operator(Pg[:, :2], Pg)


def test_paired_differencing_removes_the_noise_the_two_runs_share():
    """Without pairing the common noise dominates; with it the signal survives."""
    rng = np.random.default_rng(7)
    signal = rng.normal(size=NB) * 0.5
    common = rng.normal(size=NB) * 50.0            # 100x the signal, shared by both runs
    eps = 0.25
    plus = common + eps * signal
    minus = common - eps * signal
    recovered = paired_difference(plus, minus, eps)
    assert np.corrcoef(recovered, signal)[0, 1] > 0.999
    unpaired = (plus - common * 0.0) / eps          # what a single run would have given
    assert np.corrcoef(unpaired, signal)[0, 1] < 0.5


def test_binning_conserves_the_spikes_it_is_given():
    pos = _pos(n=500, seed=3)
    spikes = np.zeros((10, 500), bool)
    spikes[:, ::5] = True
    binned = bin_response(spikes, pos, grid=GRID, L=L)
    assert binned.sum() == pytest.approx(spikes.sum())


def test_alignment_is_sign_free_and_bounded():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert alignment(a, a) == pytest.approx(1.0)
    assert alignment(a, -a) == pytest.approx(1.0)
    assert 0.0 <= alignment(a, np.array([4.0, 1.0, 3.0, 2.0])) <= 1.0


def test_alignment_refuses_a_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        alignment(np.zeros(4), np.zeros(5))
