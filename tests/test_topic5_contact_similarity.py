"""TDD tests for topic5_contact_similarity.kernel_smooth_at_contacts.

Task 1: grid-free Gaussian contact kernel proven equal to smooth_field.
Task 2: polarity-free maxAB similarity (raw + same-plane kernel, mirror-faithful).
"""
import numpy as np
import pytest
from scipy.stats import pearsonr
from src.topic5_contact_similarity import (
    kernel_smooth_at_contacts,
    contact_corr,
    polarity_free_maxab,
)
from src.propagation_contact_plane_readout import smooth_field, make_plane_grid


def _toy_pts():
    # 3 contacts, irregular spacing on the plane
    pts = np.array([[0.0, 0.0], [0.4, 0.1], [1.0, -0.2]])
    vals = np.array([0.0, 1.0, 2.0])
    sup = np.ones(3)
    return pts, vals, sup


def test_kernel_matches_smooth_field_on_grid():
    """R2 kernel ≡ R3 field math: evaluating the contact kernel at the grid
    points reproduces smooth_field's grid (at finite, well-supported pixels)."""
    pts, vals, sup = _toy_pts()
    sigma = 0.3
    X, Y = make_plane_grid()
    grid_pts = np.column_stack([X.ravel(), Y.ravel()])   # (N,2): column_stack of two 1D arrays is correct
    record = {"channels": [{"x_norm": float(p[0]), "y_norm": float(p[1]),
                            "support": float(s), "typical_rank": float(v)}
                           for p, s, v in zip(pts, sup, vals)]}
    field = smooth_field(record, X, Y, sigma_xy=sigma, s_thresh=0.0)  # real sig: record-first, returns {"T","S"}
    T = field["T"]
    mine = kernel_smooth_at_contacts(vals, pts, grid_pts, sup, sigma).reshape(X.shape)
    m = np.isfinite(T) & np.isfinite(mine)
    assert m.sum() > 100
    assert np.allclose(T[m], mine[m], atol=1e-9)


def test_kernel_reduces_to_self_value_as_sigma_to_zero():
    """σ→0: each eval point (=its own contact) returns its own value."""
    pts, vals, sup = _toy_pts()
    out = kernel_smooth_at_contacts(vals, pts, pts, sup, sigma=1e-4)
    assert np.allclose(out, vals, atol=1e-6)


# ---------------------------------------------------------------------------
# Task 2: polarity-free maxAB similarity
# ---------------------------------------------------------------------------

def test_raw_mode_is_plain_abs_pearson():
    rng = np.random.default_rng(0)
    rank = rng.random(8); val = 2 * rank + rng.normal(0, 0.1, 8)
    pts = rng.random((8, 2)); sup = np.ones(8)
    got = contact_corr(rank, val, mode="raw", source_pts=pts, support=sup, sigma=0.3)
    assert np.isclose(abs(got), abs(pearsonr(rank, val)[0]))


def test_kernel_sigma_to_zero_equals_raw():
    rng = np.random.default_rng(1)
    rank = rng.random(10); val = rng.random(10)
    pts = rng.random((10, 2)); sup = np.ones(10)
    raw = contact_corr(rank, val, mode="raw", source_pts=pts, support=sup, sigma=0.3)
    ker = contact_corr(rank, val, mode="kernel", source_pts=pts, support=sup, sigma=1e-4)
    assert np.isclose(raw, ker, atol=1e-4)


def test_maxab_takes_better_template():
    rng = np.random.default_rng(2)
    val = rng.random(12)
    rank_a = rng.random(12)              # unrelated to val
    rank_b = val + rng.normal(0, 0.01, 12)  # strongly related
    pts = rng.random((12, 2)); sup = np.ones(12)
    mab = polarity_free_maxab(rank_a, rank_b, val, mode="raw",
                              source_pts=pts, support=sup, sigma=0.3)
    assert mab > 0.9   # picks template B


def test_maxab_sign_free_reverse_passes():
    """Sign-free: a perfectly reversed rank is a true positive (|corr|=1)."""
    rng = np.random.default_rng(3)
    val = rng.random(12); rank_a = -val   # reversed
    pts = rng.random((12, 2)); sup = np.ones(12)
    mab = polarity_free_maxab(rank_a, None, val, mode="raw",
                              source_pts=pts, support=sup, sigma=0.3)
    assert np.isclose(mab, 1.0, atol=1e-6)
