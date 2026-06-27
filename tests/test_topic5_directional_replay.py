import numpy as np
import pytest
from src.topic5_directional_replay import (
    TWO_PI, plane_fit_direction, coord_aspect, cluster_directions_k2, silhouette_unit,
    kappa_from_R, unimodal_null_pvalue, bootstrap_label_stability, two_class_eligible,
    axis_quality_tier, angular_distance, best_pair_residual, best_pair_rotation_null,
    nearest_template_gap, cohort_alignment_rotation_test)


# ---- Task 1: geometry ----
def test_plane_fit_direction_increasing_x():
    x = np.array([0, 1, 2, 0, 1, 2], float)
    y = np.array([0, 0, 0, 1, 1, 1], float)
    vals = x.copy()                       # increases along +x -> angle ~ 0
    ang, gnorm, r2, n = plane_fit_direction(x, y, vals)
    assert n == 6
    assert gnorm > 0
    assert r2 == pytest.approx(1.0, abs=1e-6)
    assert min(abs(ang - 0.0), abs(ang - 2 * np.pi)) < 1e-6


def test_plane_fit_direction_degenerate_constant():
    x = np.array([0, 1, 2], float)
    y = np.array([0, 1, 2], float)
    ang, gnorm, r2, n = plane_fit_direction(x, y, np.array([5.0, 5.0, 5.0]))
    assert np.isnan(ang)
    assert n == 3


def test_plane_fit_direction_too_few_points():
    ang, gnorm, r2, n = plane_fit_direction([0, 1], [0, 1], [1.0, 2.0])
    assert np.isnan(ang)
    assert n == 2


def test_coord_aspect_square_vs_line():
    sq_x = np.array([0, 1, 0, 1], float); sq_y = np.array([0, 0, 1, 1], float)
    assert coord_aspect(sq_x, sq_y) == pytest.approx(1.0, abs=1e-6)
    ln_x = np.array([0, 1, 2, 3], float); ln_y = np.array([0, 0, 0, 0], float)
    assert coord_aspect(ln_x, ln_y) < 0.05


# ---- Task 2: clustering ----
def test_cluster_directions_k2_two_clear_poles():
    rng = np.random.default_rng(0)
    a = np.concatenate([rng.normal(0.0, 0.1, 10), rng.normal(np.pi, 0.1, 8)])
    res = cluster_directions_k2(a, seed=0)
    assert res["n"] == 18
    assert sorted(res["sizes"]) == [8, 10]
    assert min(res["class_R"]) > 0.9
    d = abs(res["means"][0] - res["means"][1]) % TWO_PI
    assert min(d, TWO_PI - d) > 2.5


def test_silhouette_unit_clean_split_high():
    rng = np.random.default_rng(1)
    a = np.concatenate([rng.normal(0.0, 0.1, 10), rng.normal(np.pi, 0.1, 10)])
    res = cluster_directions_k2(a, seed=0)
    assert silhouette_unit(res["angles"], res["labels"]) > 0.5


# ---- Task 3: kappa + unimodal (contaminated mode+background) null ----
def test_kappa_from_R_monotone_and_edges():
    assert kappa_from_R(0.0) == pytest.approx(0.0, abs=1e-6)
    assert kappa_from_R(0.3) < kappa_from_R(0.6) < kappa_from_R(0.9)
    assert np.isfinite(kappa_from_R(0.999))


def test_unimodal_null_rejects_single_mode():            # P0 regression (命脉 #1)
    rng = np.random.default_rng(7)
    a = rng.vonmises(0.6, 4.0, 24)                        # single mode + noise
    p, s = unimodal_null_pvalue(a, B=300, seed=20260627)
    assert p > 0.1                                        # must NOT be called two-class


def test_unimodal_null_passes_true_bimodal():
    rng = np.random.default_rng(8)
    a = np.concatenate([rng.vonmises(1.0, 12, 15), rng.vonmises(1.0 + np.pi, 12, 9)])
    p, s = unimodal_null_pvalue(a, B=300, seed=20260627)
    assert p < 0.05


# ---- Task 4: bootstrap stability ----
def test_bootstrap_stability_high_for_clean_bimodal():
    rng = np.random.default_rng(3)
    a = np.concatenate([rng.normal(0.0, 0.08, 14), rng.normal(np.pi, 0.08, 12)])
    assert bootstrap_label_stability(a, B=200, seed=20260627) > 0.7


def test_bootstrap_stability_nan_for_tiny_n():
    assert np.isnan(bootstrap_label_stability(np.array([0.1, 0.2, 0.3]), B=50))


# ---- Task 5: two_class gate + anti-deception ----
def test_two_class_eligible_all_pass():
    ok, reasons = two_class_eligible(10, [7, 3], 0.01, 0.6)
    assert ok and reasons == []


def test_two_class_eligible_each_failure_reason():
    assert two_class_eligible(5, [3, 2], 0.01, 0.9)[1].count("n_sz<6") == 1
    assert "min_class<3" in two_class_eligible(11, [9, 2], 0.01, 0.9)[1]
    assert "p_bimodal>=alpha" in two_class_eligible(10, [5, 5], 0.2, 0.9)[1]
    assert "stability<min" in two_class_eligible(10, [5, 5], 0.01, 0.3)[1]
    assert two_class_eligible(5, [3, 2], 0.2, 0.3)[0] is False


def test_unimodal_with_scattered_outliers_not_two_class():   # P1 review anti-deception (命脉 #2)
    rng = np.random.default_rng(9)
    main = rng.vonmises(0.4, 10, 20)
    scatter = rng.uniform(0, 2 * np.pi, 4)
    a = np.concatenate([main, scatter])                       # one dominant direction + scatter
    p, _ = unimodal_null_pvalue(a, B=500, seed=20260627)      # contaminated null -> p ~ 0.27 (high)
    clus = cluster_directions_k2(a, seed=0)
    stab = bootstrap_label_stability(a, B=200, seed=20260627)
    eligible, _ = two_class_eligible(clus["n"], clus["sizes"], p, stab)
    assert eligible is False


# ---- Task 6: axis quality ----
def test_axis_quality_tier_boundaries():
    assert axis_quality_tier(np.radians(147), 10, 10) == "interpretable"
    assert axis_quality_tier(np.radians(120), 10, 10) == "interpretable"
    assert axis_quality_tier(np.radians(119), 10, 10) == "weak_axis"
    assert axis_quality_tier(np.radians(60), 10, 10) == "weak_axis"
    assert axis_quality_tier(np.radians(59), 10, 10) == "diagnostic_only"
    assert axis_quality_tier(np.radians(6), 10, 10) == "diagnostic_only"


def test_axis_quality_tier_low_valid_forces_diagnostic():
    assert axis_quality_tier(np.radians(147), 5, 10) == "diagnostic_only"
    assert axis_quality_tier(np.nan, 10, 10) == "diagnostic_only"


# ---- Task 7: best-pair + rotation null ----
def test_best_pair_residual_picks_straight_and_is_exchange_invariant():
    r1 = best_pair_residual([0.1, 3.0], [0.0, 3.1])
    assert r1["pairing"] == "straight" and r1["sum"] == pytest.approx(0.2, abs=1e-6)
    assert sorted(r1["matched"]) == pytest.approx([0.1, 0.1], abs=1e-6)
    r2 = best_pair_residual([3.0, 0.1], [0.0, 3.1])   # swap c1<->c2
    assert r2["sum"] == pytest.approx(r1["sum"], abs=1e-9) and r2["pairing"] == "crossed"


def test_best_pair_residual_none_on_nan():
    assert best_pair_residual([np.nan, 1.0], [0.0, 3.0]) is None


def test_rotation_null_small_when_aligned():
    p = best_pair_rotation_null([0.1, np.pi + 0.1], [0.0, np.pi], B=2000, seed=20260627)
    assert p < 0.1


def test_rotation_null_large_when_orthogonal():
    p = best_pair_rotation_null([np.pi / 2, 3 * np.pi / 2], [0.0, np.pi], B=2000, seed=20260627)
    assert p > 0.8


# ---- cohort sign-free axis-alignment test ----
def test_nearest_template_gap():
    assert nearest_template_gap(0.1, 0.0, np.pi) == pytest.approx(0.1, abs=1e-9)
    assert nearest_template_gap(np.pi - 0.1, 0.0, np.pi) == pytest.approx(0.1, abs=1e-9)


def test_cohort_alignment_significant_when_aligned():
    mains = [0.2, 1.0, 2.0, 3.0, 0.5]
    pairs = [(m, m + np.pi) for m in mains]              # main == template A end -> gap 0
    r = cohort_alignment_rotation_test(mains, pairs, B=2000, seed=20260627)
    assert r["T_obs"] < np.radians(2)
    assert r["p"] < 0.01


def test_cohort_alignment_null_when_orthogonal():
    bases = [0.2, 1.0, 2.0, 3.0, 0.5]
    mains = [b + np.pi / 2 for b in bases]               # 90deg off the axis -> max gap
    pairs = [(b, b + np.pi) for b in bases]
    r = cohort_alignment_rotation_test(mains, pairs, B=2000, seed=20260627)
    assert r["p"] > 0.5
