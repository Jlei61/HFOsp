"""Task 8: local slow-state neighbourhood representations and fail-closed branch split."""
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import src.topic4_zm_neighbourhood as NBH  # noqa: E402


def test_pca_sign_and_order_are_deterministic():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((20, 7))
    a = NBH.fit_pca(X, n_modes=3)
    b = NBH.fit_pca(X, n_modes=3)
    np.testing.assert_array_equal(a["components"], b["components"])
    for row in a["components"]:
        assert row[np.argmax(np.abs(row))] > 0
    assert np.all(np.diff(a["singular_values"]) <= 0)


def test_lattice_never_exceeds_the_locked_trajectory_budget():
    X = np.column_stack([np.linspace(-2, 2, 20), np.sin(np.linspace(0, 3, 20)),
                         np.zeros((20, 5))])
    pca = NBH.fit_pca(X, n_modes=2)
    scale = NBH.trajectory_scale(X)
    rows = NBH.build_lattice(X[10], pca, scale)
    assert rows
    assert all(r["magnitude"] <= NBH.MAX_SD * scale + 1e-12 for r in rows)
    for r in rows:
        d = np.asarray(r["q"]) - X[10]
        np.testing.assert_allclose(d / np.linalg.norm(d), r["direction"], atol=1e-12)


def test_field_reconstruction_and_split_keep_spatial_fields():
    nE = 4
    states = [
        dict(z=np.linspace(0.3, 0.9, nE) + 0.01 * i,
             m=np.linspace(0.0, 2.0, nE) + 0.2 * i, S_G=0.1 + 0.05 * i)
        for i in range(5)
    ]
    pca = NBH.full_field_representation(states, n_modes=3)
    v = NBH.reconstruct_full_field(pca, [0.1, -0.2, 0.05])
    out = NBH.split_full_field(v, nE)
    assert out["z"].shape == (nE,) and out["m"].shape == (nE,)
    assert np.all((out["z"] >= 0) & (out["z"] <= 1))
    assert np.all(out["m"] >= 0)
    assert 0 <= out["S_G"] <= 1


def test_branch_F_requires_complete_three_seed_negative_evidence():
    partial = NBH.branch_verdict(
        False, [], [1, 3, 4], True, local_negative_seeds=[1, 3],
        evidence_complete=False)
    assert partial["verdict"] == "no_evidence"
    complete = NBH.branch_verdict(
        False, [], [1, 3, 4], True, local_negative_seeds=[1, 3, 4],
        evidence_complete=True)
    assert complete["verdict"] == "branch_F_fast_carrier_repair"


def test_representation_disagreement_blocks_branch_F_even_with_complete_negatives():
    out = NBH.branch_verdict(
        False, [], [1, 3, 4], False, local_negative_seeds=[1, 3, 4],
        evidence_complete=True)
    assert out["verdict"] == "representation_sensitive_no_branch"


def test_two_seed_local_positive_selects_branch_T():
    out = NBH.branch_verdict(
        False, [1, 3], [1, 3, 4], True, local_negative_seeds=[4],
        evidence_complete=True)
    assert out["verdict"] == "branch_T_slow_trajectory_repair"
