import numpy as np
import pytest
from src.topic4_core_field import axis_coords, axial_basis_centers, partition_of_unity


def test_axis_coords_projects_onto_axis_and_perpendicular():
    pos = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
    s, r = axis_coords(pos, np.array([0.0, 0.0]), np.array([1.0, 0.0]))
    assert np.allclose(s, [1.0, 0.0, 2.0])
    assert np.allclose(np.abs(r), [0.0, 1.0, 2.0])


def test_axis_coords_axis_flip_negates_s_and_preserves_abs_r():
    rng = np.random.default_rng(0)
    pos = rng.uniform(-5, 5, size=(50, 2))
    center, u = np.array([0.3, -0.2]), np.array([0.6, 0.8])
    s1, r1 = axis_coords(pos, center, u)
    s2, r2 = axis_coords(pos, center, -u)
    assert np.allclose(s2, -s1)
    assert np.allclose(np.abs(r2), np.abs(r1))


def test_partition_of_unity_rows_sum_to_one():
    kappa = axial_basis_centers((-8.0, 8.0), M=9)
    s = np.linspace(-8.0, 8.0, 200)
    Phi = partition_of_unity(s, kappa, 1.2 * (kappa[1] - kappa[0]))
    assert Phi.shape == (200, 9)
    assert np.allclose(Phi.sum(axis=1), 1.0, atol=1e-12)


def test_uniform_weights_give_a_flat_axial_profile():
    """Why partition-of-unity is required: unnormalised Gaussians sag where fewer
    bases overlap, which would make `uniform_axial` a broad peak, not a corridor."""
    kappa = axial_basis_centers((-8.0, 8.0), M=9)
    s = np.linspace(-8.0, 8.0, 200)
    profile = partition_of_unity(s, kappa, 1.2 * (kappa[1] - kappa[0])) @ np.full(9, 1 / 9)
    assert (profile.max() - profile.min()) / profile.mean() < 1e-6


from src.topic4_core_field import (
    EPS, TAU_H, build_vth, core_thresholds, project_to_budget,
    sample_core_quantiles, signed_depth,
)


def test_core_thresholds_match_the_truncated_normal_moments():
    v = core_thresholds(sample_core_quantiles(200_000, seed=7))
    assert v.min() >= 11.0
    assert abs(v.mean() - 17.5) < 0.02
    assert abs(v.std() - 1.0) < 0.02


def test_signed_depth_keeps_the_negative_third():
    """About 31% of 'core' neurons sit ABOVE baseline; max(0,.) would drop them
    and break parity with the accepted manual core (spec C1)."""
    d = signed_depth(core_thresholds(sample_core_quantiles(200_000, seed=7)))
    assert 0.28 < (d < 0).mean() < 0.34
    assert abs(d.mean() - 0.5) < 0.02


def test_budget_projection_hits_the_target_count():
    q = np.random.default_rng(1).uniform(EPS, 1.0, size=32_000)
    h, lam = project_to_budget(q, target_count=1131.0)
    assert np.isfinite(lam) and (h >= 0).all() and (h <= 1).all()
    assert abs(h.sum() - 1131.0) / 1131.0 < 1e-6


def test_budget_projection_is_monotone_in_lambda():
    """Strictly decreasing, so the root is unique. Budgeting on sum(h*d) would
    NOT be monotone once a third of d is negative."""
    from scipy.special import expit
    q = np.random.default_rng(2).uniform(EPS, 1.0, size=5_000)
    lq = np.log(q + EPS)
    totals = [expit((lq - lam) / TAU_H).sum() for lam in np.linspace(-8, 2, 25)]
    assert all(b < a for a, b in zip(totals, totals[1:]))


@pytest.mark.parametrize("target", [0.0, -1.0, 32_000.0, 40_000.0])
def test_budget_projection_rejects_an_out_of_range_target(target):
    q = np.random.default_rng(3).uniform(EPS, 1.0, size=32_000)
    with pytest.raises(ValueError):
        project_to_budget(q, target_count=target)


def test_budget_projection_rejects_non_finite_input():
    q = np.random.default_rng(4).uniform(EPS, 1.0, size=100)
    q[3] = np.nan
    with pytest.raises(ValueError):
        project_to_budget(q, target_count=10.0)


def test_budget_projection_does_not_overflow_on_an_extreme_field():
    """expit, not 1/(1+exp(-x)): the naive form overflows for large |x|."""
    q = np.concatenate([np.full(500, 1e-12), np.full(500, 1e6)])
    with np.errstate(over="raise"):
        h, _ = project_to_budget(q, target_count=500.0)
    assert np.isfinite(h).all()


def test_build_vth_places_baseline_outside_and_core_distribution_inside():
    n_E, n_total = 1000, 1250
    d = signed_depth(core_thresholds(sample_core_quantiles(n_E, seed=3)))
    h = np.zeros(n_E); h[:100] = 1.0
    vth = build_vth(h, d, n_total=n_total, n_E=n_E)
    assert vth.shape == (n_total,)
    assert np.allclose(vth[100:], 18.0)
    assert np.allclose(vth[:100], 18.0 - d[:100])
