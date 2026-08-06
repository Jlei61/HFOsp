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


from src.topic4_core_field import (
    ARM_NAMES, arm_h, manual_mask, preflight_shape, shape_metrics,
)

SEP = 6.0


def _mock_sheet(n=32_000, L=20.0, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, L, size=(n, 2))
    return pos, pos[:, 0] - L / 2.0, pos[:, 1] - L / 2.0


def _geom():
    return dict(sep=SEP, s_support=(-8.0, 8.0), M=9, sigma_perp=1.5, shift_mm=3.0)


def _mask(s, r, core_r=1.5):
    """Cores sit at +-sep/2, matching two_core_q."""
    return (np.minimum((s - SEP / 2) ** 2, (s + SEP / 2) ** 2) + r ** 2) <= core_r ** 2


def test_there_are_eight_arms_including_both_manual_variants():
    assert len(ARM_NAMES) == 8
    for name in ("manual_hard", "manual_projected", "manual_smooth"):
        assert name in ARM_NAMES


def test_manual_projected_is_exactly_the_hard_mask():
    """spec 4.3.1: manual_projected changes the DRAWS, not the mask. If it were a
    smoothed field, comparison A would move three things at once."""
    _, s, r = _mock_sheet()
    m = _mask(s, r)
    h = arm_h("manual_projected", s, r, _geom(), float(m.sum()), manual_mask_E=m)
    assert np.array_equal(h, m.astype(float))


def test_manual_smooth_is_close_to_but_not_identical_to_the_hard_mask():
    _, s, r = _mock_sheet()
    m = _mask(s, r)
    h = arm_h("manual_smooth", s, r, _geom(), float(m.sum()), manual_mask_E=m)
    assert np.corrcoef(h, m.astype(float))[0, 1] >= 0.9
    assert not np.array_equal(h, m.astype(float))


def test_all_arms_hit_the_same_budget():
    _, s, r = _mock_sheet()
    m = _mask(s, r)
    for name in ARM_NAMES:
        if name == "manual_hard":
            continue
        h = arm_h(name, s, r, _geom(), float(m.sum()), manual_mask_E=m)
        assert abs(h.sum() - m.sum()) / m.sum() < 1e-6, name


def test_width_arms_reshape_rather_than_blur():
    """spec 4.4: bare sigma_perp only blurs the edge because the budget pins the
    area; rho reshapes it at fixed a*b."""
    _, s, r = _mock_sheet()
    m = _mask(s, r); target = float(m.sum())
    wide = arm_h("width_wide", s, r, _geom(), target, manual_mask_E=m)
    narrow = arm_h("width_narrow", s, r, _geom(), target, manual_mask_E=m)
    assert shape_metrics(wide, s, r)["rms_transverse"] > \
           2.5 * shape_metrics(narrow, s, r)["rms_transverse"]


def test_transverse_arms_are_mirror_images():
    _, s, r = _mock_sheet()
    m = _mask(s, r); target = float(m.sum())
    plus = arm_h("transverse_plus", s, r, _geom(), target, manual_mask_E=m)
    minus = arm_h("transverse_minus", s, -r, _geom(), target, manual_mask_E=_mask(s, -r))
    assert np.corrcoef(plus, minus)[0, 1] > 0.999


def test_preflight_covers_shape_comparisons_only_and_excludes_the_equivalence_arms():
    """P0-2: manual_hard and manual_projected SHOULD be near-identical. An
    all-pairs correlation gate would reject the correct implementation."""
    _, s, r = _mock_sheet()
    m = _mask(s, r); target = float(m.sum())
    h_by_arm = {n: arm_h(n, s, r, _geom(), target, manual_mask_E=m)
                for n in ARM_NAMES if n != "manual_hard"}
    h_by_arm["manual_hard"] = m.astype(float)
    rep = preflight_shape(h_by_arm, s, r, target)
    assert rep["ok"] is True
    assert set(rep["checks"]) == {"B1", "B2", "B3", "B4"}


def test_preflight_fails_when_a_shape_arm_collapses_onto_the_baseline():
    _, s, r = _mock_sheet()
    m = _mask(s, r); target = float(m.sum())
    h_by_arm = {n: arm_h(n, s, r, _geom(), target, manual_mask_E=m)
                for n in ARM_NAMES if n != "manual_hard"}
    h_by_arm["manual_hard"] = m.astype(float)
    h_by_arm["uniform_axial"] = h_by_arm["manual_smooth"].copy()   # collapse B1
    rep = preflight_shape(h_by_arm, s, r, target)
    assert rep["ok"] is False
    assert rep["checks"]["B1"]["ok"] is False
