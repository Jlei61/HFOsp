"""Pure-metric contract tests for src/sef_hfo_b1_validation.py (M3 B1b/B1d).

Shape-comparison + spatial-geometry metrics used by the matched-shape equivalence (B1d)
and the axis/anisotropy analysis (B1b). No SNN.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import sef_hfo_b1_validation as b1  # noqa: E402


# --- shape similarity ------------------------------------------------------- #
def test_cosine_identical_is_one_orthogonal_is_zero():
    a = np.array([0.0, 0.7, 0.3, 0.0])
    assert b1.shape_similarity(a, a, "cosine") == pytest.approx(1.0)
    assert b1.shape_similarity(np.array([1.0, 0, 0]), np.array([0, 1.0, 0]),
                               "cosine") == pytest.approx(0.0)


def test_pearson_and_spearman_run():
    a = np.array([1.0, 2, 3, 4]); b = np.array([1.0, 2, 3, 5])
    assert b1.shape_similarity(a, b, "pearson") > 0.9
    assert b1.shape_similarity(a, b, "spearman") == pytest.approx(1.0)  # monotone


# --- weighted centroid ------------------------------------------------------ #
def test_weighted_centroid():
    pos = np.array([[0.0, 0.0], [2.0, 0.0]])
    w = np.array([1.0, 1.0])
    assert np.allclose(b1.weighted_centroid(w, pos), [1.0, 0.0])
    w2 = np.array([3.0, 1.0])
    assert np.allclose(b1.weighted_centroid(w2, pos), [0.5, 0.0])


# --- principal axis / anisotropy ------------------------------------------- #
def test_principal_axis_along_45_degrees():
    # weight concentrated along the 45-degree diagonal -> angle ~45, anisotropy >> 1
    pos = np.array([[-2.0, -2.0], [-1.0, -1.0], [0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    w = np.ones(5)
    angle, aniso = b1.principal_axis(w, pos)
    assert abs(angle - 45.0) < 5.0
    assert aniso > 10.0


def test_principal_axis_isotropic_has_ratio_near_one():
    # 4 corners equal weight -> isotropic -> anisotropy ~1
    pos = np.array([[1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]])
    _, aniso = b1.principal_axis(np.ones(4), pos)
    assert aniso == pytest.approx(1.0, abs=0.05)


def test_axis_angle_diff_is_mod_180():
    assert b1.axis_angle_diff(45.0, 225.0) == pytest.approx(0.0)   # axis undirected
    assert b1.axis_angle_diff(45.0, 135.0) == pytest.approx(90.0)
    assert b1.axis_angle_diff(10.0, 170.0) == pytest.approx(20.0)  # circular on 180


# --- top-k overlap ---------------------------------------------------------- #
def test_top_k_overlap():
    a = np.array([0.9, 0.8, 0.1, 0.0])   # top2 = {0,1}
    b = np.array([0.1, 0.7, 0.6, 0.0])   # top2 = {1,2}
    assert b1.top_k_overlap(a, b, 2) == pytest.approx(0.5)
    assert b1.top_k_overlap(a, a, 2) == pytest.approx(1.0)


# --- split-half / cross similarity distributions ---------------------------- #
def test_split_half_similarity_identical_rows_is_high():
    rows = np.tile(np.array([0.0, 0.6, 0.4, 0.0]), (8, 1))
    d = b1.split_half_similarity(rows, metric="cosine", n_splits=50, rng_seed=0)
    assert d["median"] == pytest.approx(1.0, abs=1e-6)


def test_cross_subsample_similarity_deterministic():
    rng = np.random.default_rng(0)
    A = rng.random((6, 5)); A /= A.sum(1, keepdims=True)
    B = rng.random((6, 5)); B /= B.sum(1, keepdims=True)
    d1 = b1.cross_subsample_similarity(A, B, metric="cosine", n_sub=30, rng_seed=3)
    d2 = b1.cross_subsample_similarity(A, B, metric="cosine", n_sub=30, rng_seed=3)
    assert d1["median"] == d2["median"]
