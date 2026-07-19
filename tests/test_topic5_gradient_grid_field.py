"""Unit tests for the gradient-axis R3 dense-grid field scorer.

These lock the science-critical invariants of the Figure 3 recompute
(handoff docs/archive/topic5/fig3_ictal_gradient_r3_full_recompute_handoff_2026-07-18.md
§3-§5, §8). The module under test is a pure-math layer with no filesystem I/O.
"""
from __future__ import annotations

import numpy as np
import pytest

import src.topic5_gradient_grid_field as gg
from src.propagation_contact_plane_readout import (
    make_plane_grid,
    smooth_field,
    corr_pair_mirror_invariant_signed,
    S_THRESH,
)


# --------------------------------------------------------------------------
# overlap-min resolution formula (handoff §3.7)
# --------------------------------------------------------------------------
def test_overlap_min_matches_prescribed_formula():
    assert gg.overlap_min_for_n(81) == 25
    assert gg.overlap_min_for_n(161) == 99  # ceil(25/81**2 * 161**2)


# --------------------------------------------------------------------------
# adaptive grid (handoff §3.3)
# --------------------------------------------------------------------------
def test_adaptive_grid_is_y_symmetric_and_flip_is_mirror():
    pts = np.array([[0.0, 0.2], [1.0, -0.5], [0.4, 0.1]])
    grid = gg.make_adaptive_grid(pts, sigma=0.1, support_budget=3.0, n=81)
    X, Y = grid["X"], grid["Y"]
    assert X.shape == (81, 81) and Y.shape == (81, 81)
    # y axis strictly symmetric about 0 -> flip(axis=0) is exact y -> -y
    assert np.allclose(Y, -np.flip(Y, axis=0))
    assert np.allclose(X, np.flip(X, axis=0))
    # odd n contains the y=0 row
    assert np.isclose(Y[40, 0], 0.0)


def test_adaptive_grid_bounds_use_support_radius():
    pts = np.array([[0.0, 0.2], [1.0, -0.5]])
    sigma, budget = 0.1, 4.0
    grid = gg.make_adaptive_grid(pts, sigma=sigma, support_budget=budget, n=81)
    r = gg.support_radius(sigma, budget)
    assert np.isclose(grid["x_lo"], pts[:, 0].min() - r)
    assert np.isclose(grid["x_hi"], pts[:, 0].max() + r)
    assert np.isclose(grid["y_ext"], np.abs(pts[:, 1]).max() + r)


def test_support_radius_pushes_boundary_support_below_threshold():
    # By construction S(boundary) < S_THRESH so the S>=0.15 region cannot touch
    # the grid edge (handoff §3.3 assertion).
    pts = np.array([[0.0, 0.0], [0.8, 0.3], [0.3, -0.4]])
    support = np.array([1.5, 2.0, 1.0])
    sigma = 0.12
    budget = float(support.sum())
    grid = gg.make_adaptive_grid(pts, sigma=sigma, support_budget=budget, n=81)
    F, S = gg.build_grid_field(grid["X"], grid["Y"], pts, support,
                               np.array([0.0, 1.0, -1.0]), sigma)
    border = np.zeros_like(S, dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    assert np.all(S[border] < gg.S_THRESH)
    assert not gg.support_region_touches_boundary(S)


def test_support_region_touches_boundary_detects_contact_at_edge():
    grid = gg.make_adaptive_grid(np.array([[0.0, 0.0]]), sigma=0.1,
                                 support_budget=1.0, n=41)
    S = np.zeros((41, 41))
    S[0, 20] = 1.0  # a supported pixel sitting on the border
    assert gg.support_region_touches_boundary(S)


# --------------------------------------------------------------------------
# grid field construction (handoff §3.4)
# --------------------------------------------------------------------------
def test_build_grid_field_matches_hand_computed_two_contacts():
    X, Y = np.meshgrid(np.array([0.0, 1.0]), np.array([0.0]), indexing="ij")
    # grid pts: (0,0) and (1,0) laid on the X axis
    pts = np.array([[0.0, 0.0], [1.0, 0.0]])
    support = np.array([1.0, 1.0])
    values = np.array([2.0, 4.0])
    sigma = 1.0
    F, S = gg.build_grid_field(X, Y, pts, support, values, sigma)
    # at grid point (0,0): K to pts = [exp(0), exp(-1/2)] = [1, 0.60653]
    k = np.array([1.0, np.exp(-0.5)])
    expF00 = (k @ (support * values)) / (k @ support)
    assert np.isclose(F.ravel()[0], expF00)
    assert np.isclose(S.ravel()[0], k @ support)


def test_ictal_support_uses_finite_mask_but_interictal_uses_full():
    # A contact outside the common finite mask contributes to the interictal
    # field but NOT to the ictal field (handoff §3.4: S_inter >= S_ictal).
    pts = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    support = np.array([1.0, 1.0, 1.0])
    earliness = np.array([-1.0, 0.0, 1.0])
    activation = np.array([3.0, 5.0, np.nan])  # 3rd contact missing this event
    sigma = 0.3
    grid = gg.make_adaptive_grid(pts, sigma=sigma, support_budget=3.0, n=81)
    finite = np.isfinite(activation)
    w_ict = support * finite.astype(float)
    v_ict = np.where(finite, activation, 0.0)
    F_in, S_in = gg.build_grid_field(grid["X"], grid["Y"], pts, support, earliness, sigma)
    F_ic, S_ic = gg.build_grid_field(grid["X"], grid["Y"], pts, w_ict, v_ict, sigma)
    # every pixel: interictal support >= ictal support
    assert np.all(S_in + 1e-12 >= S_ic)
    # near the 3rd contact the interictal support strictly exceeds the ictal one
    near_c3 = np.argmin((grid["X"].ravel() - 1.0) ** 2 + (grid["Y"].ravel()) ** 2)
    assert S_in.ravel()[near_c3] > S_ic.ravel()[near_c3] + 1e-6


# --------------------------------------------------------------------------
# regression against the historical endpoint smooth_field (handoff §8 test 20)
# --------------------------------------------------------------------------
def test_grid_field_reproduces_legacy_smooth_field_on_fixed_grid():
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(0.0, 1.0, 7), rng.uniform(-0.5, 0.5, 7)])
    support = rng.uniform(0.3, 1.0, 7)
    values = rng.normal(size=7)
    sigma = 0.15
    record = {"channels": [
        {"x_norm": float(pts[i, 0]), "y_norm": float(pts[i, 1]),
         "typical_rank": float(values[i]), "support": float(support[i]),
         "uncertainty_rank": 0.0}
        for i in range(7)]}
    X, Y = make_plane_grid(81)
    legacy = smooth_field(record, X, Y, sigma_xy=sigma, scalar="rank",
                          s_thresh=S_THRESH)
    F, S = gg.build_grid_field(X, Y, pts, support, values, sigma)
    m = legacy["mask"]
    assert np.array_equal(m, S >= gg.S_THRESH)
    assert np.allclose(F[m], legacy["T"][m], equal_nan=True)
    assert np.allclose(S, legacy["S"])


# --------------------------------------------------------------------------
# mirror abs-max selection (handoff §3.5, §8 tests 8/12)
# --------------------------------------------------------------------------
def test_score_template_selects_abs_max_over_identity_and_mirror():
    # Construct an ictal field whose mirror correlates strongly NEGATIVELY with
    # the interictal template; abs-max must pick the mirror (signed r < 0).
    pts = np.array([[0.1, 0.6], [0.3, 0.5], [0.6, -0.55], [0.8, -0.5],
                    [0.45, 0.05], [0.5, -0.05]])
    support = np.ones(6)
    earliness = np.array([2.0, 1.5, -1.5, -2.0, 0.2, -0.2])
    sigma = 0.18
    grid = gg.make_adaptive_grid(pts, sigma=sigma, support_budget=6.0, n=81)
    F_in, S_in = gg.build_grid_field(grid["X"], grid["Y"], pts, support, earliness, sigma)
    # activation = mirror of earliness field source -> identity corr weak,
    # mirror corr strong. Use flipped-sign spatial arrangement.
    activation = np.array([-2.0, -1.5, 1.5, 2.0, -0.2, 0.2])
    res = gg.score_template_r3(grid, F_in, S_in, pts, support, activation,
                               np.isfinite(activation))
    assert res["mirror_choice"] in ("identity", "mirror")
    # abs-max rule: |signed_r| == max(|r_id|,|r_mir|)
    cand = [abs(res["r_identity"]) if res["r_identity"] is not None else -1,
            abs(res["r_mirror"]) if res["r_mirror"] is not None else -1]
    assert np.isclose(res["abs_r"], max(cand))


def test_score_template_matches_legacy_corr_pair_primitive():
    # Our per-template R3 score must equal the historical
    # corr_pair_mirror_invariant_signed on the same grid fields.
    pts = np.array([[0.0, 0.3], [0.4, -0.2], [0.7, 0.1], [1.0, -0.3],
                    [0.5, 0.25], [0.2, -0.15]])
    support = np.array([1.0, 0.8, 1.2, 0.6, 0.9, 1.1])
    earliness = np.array([1.0, 0.3, -0.4, -1.2, 0.6, 0.1])
    activation = np.array([0.9, 0.2, -0.5, -1.0, 0.7, 0.0])
    sigma = 0.2
    grid = gg.make_adaptive_grid(pts, sigma=sigma, support_budget=float(support.sum()), n=81)
    finite = np.isfinite(activation)
    F_in, S_in = gg.build_grid_field(grid["X"], grid["Y"], pts, support, earliness, sigma)
    w_ict = support * finite.astype(float)
    v_ict = np.where(finite, activation, 0.0)
    F_ic, S_ic = gg.build_grid_field(grid["X"], grid["Y"], pts, w_ict, v_ict, sigma)
    legacy = corr_pair_mirror_invariant_signed(F_in, S_in, F_ic, S_ic,
                                               s_thresh=gg.S_THRESH,
                                               overlap_min=gg.overlap_min_for_n(81))
    res = gg.score_template_r3(grid, F_in, S_in, pts, support, activation, finite)
    if legacy["signed_corr"] is None:
        assert res["signed_r"] is None or not np.isfinite(res["signed_r"])
    else:
        assert np.isclose(res["signed_r"], legacy["signed_corr"])
        assert res["mirror_choice"] == legacy["mirror_choice"]
        assert res["n_overlap"] == legacy["n_overlap"]


# --------------------------------------------------------------------------
# batch == per-draw reference (handoff §8 test 14: every draw rebuilt)
# --------------------------------------------------------------------------
def test_batch_scoring_equals_per_draw_reference():
    rng = np.random.default_rng(3)
    pts = np.column_stack([rng.uniform(0, 1, 8), rng.uniform(-0.4, 0.4, 8)])
    support = rng.uniform(0.4, 1.0, 8)
    earl_a = rng.normal(size=8)
    earl_b = rng.normal(size=8)
    sigma = 0.16
    finite = np.ones(8, bool)
    act_base = rng.normal(size=8)
    perms = np.array([rng.permutation(8) for _ in range(25)])
    acts = np.vstack([act_base] + [act_base[p] for p in perms])  # (26, 8)

    ev = gg.build_event_scorer(pts_a=pts, support_a=support, earliness_a=earl_a,
                               pts_b=pts, support_b=support, earliness_b=earl_b,
                               sigma=sigma, finite=finite, n=81)
    batch = gg.score_event_maxab_batch(ev, acts)  # (26,)
    ref = np.array([gg.score_event_maxab_single(ev, acts[i]) for i in range(acts.shape[0])])
    assert np.allclose(batch, ref, equal_nan=True)


def test_varmask_batch_equals_per_row_reference():
    # Fig3-C readout: each row (time window) may have a DIFFERENT missing-contact
    # pattern. The variable-mask batch must equal looping the single-activation
    # reference with each row's own finite mask.
    rng = np.random.default_rng(21)
    m = 9
    pts = np.column_stack([rng.uniform(0, 1, m), rng.uniform(-0.4, 0.4, m)])
    support = rng.uniform(0.4, 1.0, m)
    earl_a = rng.normal(size=m)
    earl_b = rng.normal(size=m)
    sigma = 0.16
    ev = gg.build_event_scorer(pts_a=pts, support_a=support, earliness_a=earl_a,
                               pts_b=pts, support_b=support, earliness_b=earl_b,
                               sigma=sigma, finite=np.ones(m, bool), n=81)
    rows = []
    for _ in range(12):
        act = rng.normal(size=m)
        drop = rng.random(m) < 0.25          # per-row missing contacts
        act[drop] = np.nan
        rows.append(act)
    acts = np.vstack(rows)
    batch = gg.score_event_maxab_batch_varmask(ev, acts)

    def ref_row(act):
        fin = np.isfinite(act)
        a = gg.score_template_r3(ev["grid_a"], ev["A"].F_inter, ev["A"].S_inter,
                                 ev["A"].points, ev["A"].support, act, fin)
        b = gg.score_template_r3(ev["grid_b"], ev["B"].F_inter, ev["B"].S_inter,
                                 ev["B"].points, ev["B"].support, act, fin)
        vals = [x["abs_r"] for x in (a, b) if x["abs_r"] is not None and np.isfinite(x["abs_r"])]
        return max(vals) if vals else np.nan

    ref = np.array([ref_row(r) for r in rows])
    assert np.allclose(batch, ref, equal_nan=True)


def test_maxab_reselected_per_draw():
    # Different draws can pick different winning template; maxab == max(absA,absB)
    rng = np.random.default_rng(7)
    pts = np.column_stack([rng.uniform(0, 1, 7), rng.uniform(-0.4, 0.4, 7)])
    support = np.ones(7)
    ev = gg.build_event_scorer(pts_a=pts, support_a=support, earliness_a=rng.normal(size=7),
                               pts_b=pts, support_b=support, earliness_b=rng.normal(size=7),
                               sigma=0.2, finite=np.ones(7, bool), n=81)
    act = rng.normal(size=7)
    detail = gg.score_event_detail_single(ev, act)
    finite_abs = [v for v in (detail["abs_a"], detail["abs_b"]) if v is not None and np.isfinite(v)]
    assert np.isclose(detail["maxab"], max(finite_abs))
    assert detail["best_template"] in ("A", "B")


# --------------------------------------------------------------------------
# seven-band maxT / pFWER (handoff §5.3)
# --------------------------------------------------------------------------
def test_seven_band_maxt_pfwer_matches_reference_formula():
    rng = np.random.default_rng(11)
    n_subj, n_band, n_draw = 6, 4, 200
    D = rng.normal(0.6, 0.1, (n_subj, n_band))
    D[:, 0] += 0.4  # band 0 clearly elevated
    N = rng.normal(0.5, 0.1, (n_subj, n_band, n_draw))
    out = gg.seven_band_maxt_pfwer(D, N)
    # reference formula
    Cobs = np.median(D, axis=0)
    Cnull = np.median(N, axis=0)                    # (band, draw)
    Zobs = Cobs - np.median(Cnull, axis=1)
    Znull = Cnull - np.median(Cnull, axis=1, keepdims=True)
    M = Znull.max(axis=0)                            # (draw,)
    pfwer = np.array([(1 + np.sum(M >= Zobs[b])) / (n_draw + 1) for b in range(n_band)])
    assert np.allclose(out["Cobs"], Cobs)
    assert np.allclose(out["Zobs"], Zobs)
    assert np.allclose(out["pFWER"], pfwer)
    # elevated band 0 has the smallest pFWER
    assert np.argmin(out["pFWER"]) == 0


def test_seven_band_per_subject_delta_and_cohort_bar():
    D = np.array([[0.8, 0.4], [0.6, 0.3]])
    N = np.zeros((2, 2, 3))
    N[..., 0] = np.array([[0.5, 0.2], [0.4, 0.1]])
    N[..., 1] = N[..., 0]
    N[..., 2] = N[..., 0]
    out = gg.seven_band_maxt_pfwer(D, N)
    nmed = np.median(N, axis=2)                      # (subj, band)
    delta = D - nmed
    assert np.allclose(out["per_subject_delta"], delta)
    assert np.allclose(out["cohort_delta_median"], np.median(delta, axis=0))


# --------------------------------------------------------------------------
# one-sided paired Wilcoxon (handoff §5.2)
# --------------------------------------------------------------------------
def test_one_sided_wilcoxon_greater_matches_scipy():
    from scipy.stats import wilcoxon
    data = np.array([0.9, 0.7, 0.6, 0.85, 0.5, 0.75])
    null = np.array([0.5, 0.6, 0.55, 0.5, 0.52, 0.5])
    p = gg.paired_one_sided_wilcoxon_greater(data, null)
    exp = wilcoxon(data, null, alternative="greater").pvalue
    assert np.isclose(p, exp)


# --------------------------------------------------------------------------
# direct band-specificity omnibus (handoff §5.4)
# --------------------------------------------------------------------------
def test_direct_band_omnibus_reports_friedman_kendall_and_calibrated_p():
    rng = np.random.default_rng(5)
    # 8 subjects, 4 bands, band 0 systematically largest -> non-null omnibus
    delta = rng.normal(0, 0.05, (8, 4))
    delta[:, 0] += 0.5
    out = gg.direct_band_omnibus(delta, n_perm=2000, seed=1)
    assert out["n_subjects"] == 8 and out["n_bands"] == 4
    assert np.isfinite(out["friedman_statistic"])
    assert 0.0 <= out["kendall_w"] <= 1.0
    assert out["calibrated_p"] <= 0.01          # strong effect
    assert out["n_permutations"] == 2000


def test_direct_band_omnibus_null_case_is_not_significant():
    rng = np.random.default_rng(9)
    delta = rng.normal(0, 0.1, (12, 5))          # exchangeable bands
    out = gg.direct_band_omnibus(delta, n_perm=2000, seed=2)
    assert out["calibrated_p"] > 0.05


def test_direct_band_contrasts_yield_21_pairs_with_holm():
    rng = np.random.default_rng(4)
    delta = rng.normal(0, 0.1, (10, 7))
    rows = gg.direct_band_contrasts(delta, band_labels=list("abcdefg"))
    assert len(rows) == 21                        # C(7,2)
    for r in rows:
        assert {"band_i", "band_j", "median_difference", "iqr_low", "iqr_high",
                "wilcoxon_p", "holm_p"} <= set(r)
    # Holm p-values are >= raw p-values and <= 1
    assert all(r["holm_p"] >= r["wilcoxon_p"] - 1e-12 for r in rows)
    assert all(r["holm_p"] <= 1.0 + 1e-12 for r in rows)


# --------------------------------------------------------------------------
# pure within-shaft with min_group_for_shaft = 4 (handoff §4.2)
# --------------------------------------------------------------------------
def test_within_shaft_marks_event_unavailable_when_a_shaft_too_small():
    names = ["A1", "A2", "A3", "A4", "B1", "B2"]   # shaft B has only 2 finite
    finite = np.ones(6, bool)
    res = gg.within_shaft_permutations(names, finite, n_perm=10, seed=1, min_group=4)
    assert res["eligible"] is False
    assert res["permutations"] is None


def test_within_shaft_permutes_within_shaft_only_when_eligible():
    names = ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"]
    finite = np.ones(8, bool)
    res = gg.within_shaft_permutations(names, finite, n_perm=50, seed=3, min_group=4)
    assert res["eligible"] is True
    perms = res["permutations"]
    assert perms.shape == (50, 8)
    # every permuted index stays inside its own shaft block
    shaft = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    for row in perms:
        assert np.array_equal(shaft[row], shaft)


def test_within_shaft_eligibility_uses_finite_contacts_only():
    names = ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"]
    finite = np.array([1, 1, 1, 1, 1, 1, 1, 0], bool)  # shaft B now only 3 finite
    res = gg.within_shaft_permutations(names, finite, n_perm=10, seed=1, min_group=4)
    assert res["eligible"] is False


# --------------------------------------------------------------------------
# permutation mapping hash audit (handoff §4.1, §7)
# --------------------------------------------------------------------------
def test_build_event_scorer_sigma_a_b_backward_compat():
    # sigma=X must be identical to sigma_a=X, sigma_b=X (frozen_per_model==subject_fixed
    # for shared route where both sigmas equal).
    rng = np.random.default_rng(31)
    pts = np.column_stack([rng.uniform(0, 1, 7), rng.uniform(-0.4, 0.4, 7)])
    sup, ea, eb = np.ones(7), rng.normal(size=7), rng.normal(size=7)
    common = dict(pts_a=pts, support_a=sup, earliness_a=ea, pts_b=pts, support_b=sup,
                  earliness_b=eb, finite=np.ones(7, bool), n=81)
    ev1 = gg.build_event_scorer(sigma=0.18, **common)
    ev2 = gg.build_event_scorer(sigma_a=0.18, sigma_b=0.18, **common)
    act = rng.normal(size=7)
    assert np.isclose(gg.score_event_maxab_single(ev1, act), gg.score_event_maxab_single(ev2, act))
    assert ev1["grid_a"]["sha256"] == ev2["grid_a"]["sha256"]
    assert ev1["sigma_a"] == ev2["sigma_a"] == 0.18


def test_build_event_scorer_frozen_per_model_own_route_uses_sigma_b_for_b():
    pts_a = np.array([[0.0, 0.1], [0.5, -0.2], [1.0, 0.3], [0.3, 0.0], [0.7, -0.1], [0.9, 0.2]])
    pts_b = np.array([[0.2, -0.3], [0.4, 0.4], [0.8, -0.1], [0.1, 0.2], [0.6, 0.0], [1.0, -0.4]])
    sa, sb = np.ones(6), np.ones(6)
    ev = gg.build_event_scorer(pts_a=pts_a, support_a=sa, earliness_a=np.arange(6.0),
                               pts_b=pts_b, support_b=sb, earliness_b=np.arange(6.0)[::-1],
                               sigma_a=0.10, sigma_b=0.20, finite=np.ones(6, bool),
                               n=81, shared_grid=False)
    # B grid must be built with sigma_b (its support radius differs from a sigma_a grid)
    ref_b = gg.make_adaptive_grid(pts_b, 0.20, float(sb.sum()), n=81)
    assert np.isclose(ev["grid_b"]["support_radius"], ref_b["support_radius"])
    assert ev["sigma_a"] == 0.10 and ev["sigma_b"] == 0.20


def test_build_event_scorer_shared_grid_requires_equal_sigma():
    pts = np.array([[0.0, 0.1], [0.5, -0.2], [1.0, 0.3], [0.3, 0.0], [0.7, -0.1], [0.9, 0.2]])
    with pytest.raises(ValueError):
        gg.build_event_scorer(pts_a=pts, support_a=np.ones(6), earliness_a=np.arange(6.0),
                              pts_b=pts, support_b=np.ones(6), earliness_b=np.arange(6.0),
                              sigma_a=0.10, sigma_b=0.20, finite=np.ones(6, bool),
                              n=81, shared_grid=True)


def test_own_route_builds_separate_grids_and_support_per_template():
    # own-fallback: A and B live on different planes with different support;
    # the two grids must be distinct and each template must keep its own support.
    pts_a = np.array([[0.0, 0.1], [0.5, -0.2], [1.0, 0.3], [0.3, 0.0], [0.7, -0.1], [0.9, 0.2]])
    pts_b = np.array([[0.2, -0.3], [0.4, 0.4], [0.8, -0.1], [0.1, 0.2], [0.6, 0.0], [1.0, -0.4]])
    support_a = np.array([1.0, 0.5, 0.9, 0.7, 0.4, 0.8])
    support_b = np.array([0.6, 0.8, 0.3, 0.5, 0.9, 0.7])
    ev = gg.build_event_scorer(pts_a=pts_a, support_a=support_a, earliness_a=np.arange(6.0),
                               pts_b=pts_b, support_b=support_b, earliness_b=np.arange(6.0)[::-1],
                               sigma=0.15, finite=np.ones(6, bool), n=81, shared_grid=False)
    assert ev["shared_grid"] is False
    assert ev["grid_a"]["sha256"] != ev["grid_b"]["sha256"]
    # each grid's support budget is that template's own support sum (not shared max)
    assert np.isclose(ev["grid_a"]["support_budget"], support_a.sum())
    assert np.isclose(ev["grid_b"]["support_budget"], support_b.sum())


def test_loo_contact_reconstruction_leaves_each_out():
    # Leave-one-out kernel reconstruction of contact earliness from the others.
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.8]])
    support = np.array([1.0, 1.0, 1.0])
    values = np.array([2.0, 4.0, -1.0])
    sigma = 0.5
    recon = gg.loo_contact_reconstruction(pts, support, values, sigma)
    # reference: for contact i, weighted mean of the OTHER contacts
    sig2 = 2 * sigma ** 2
    for i in range(3):
        w = np.array([np.exp(-((pts[i] - pts[j]) ** 2).sum() / sig2) * support[j]
                      for j in range(3) if j != i])
        v = np.array([values[j] for j in range(3) if j != i])
        assert np.isclose(recon[i], (w @ v) / w.sum())
    # a lone contact cannot be reconstructed -> nan
    assert np.isnan(gg.loo_contact_reconstruction(pts[:1], support[:1], values[:1], sigma)[0])


def test_permutation_mapping_hash_is_order_sensitive_and_stable():
    rng = np.random.default_rng(0)
    perms = np.array([rng.permutation(9) for _ in range(20)])
    h1 = gg.permutation_mapping_hash(perms)
    h2 = gg.permutation_mapping_hash(perms.copy())
    assert h1 == h2                                  # stable
    perms2 = perms.copy()
    perms2[0, [0, 1]] = perms2[0, [1, 0]]
    assert gg.permutation_mapping_hash(perms2) != h1  # sensitive
