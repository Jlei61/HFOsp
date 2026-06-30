"""TDD tests for topic5_contact_similarity.kernel_smooth_at_contacts.

Task 1: grid-free Gaussian contact kernel proven equal to smooth_field.
"""
import numpy as np
import pytest
from src.topic5_contact_similarity import kernel_smooth_at_contacts
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
