"""Unit tests for src/sef_itp_direction_axis.py (Topic 4 SEF-ITP H2b).

Plan: docs/archive/topic4/sef_itp_direction_axis/phase_h2b_direction_axis_plan_2026-05-25.md

Each synthetic case locks one verdict label (axis_reversal / dual_source /
same_direction / degenerate_geometry / exit_no_universe) plus permutation null
sanity. Aim: catch silent verdict-rule drift before cohort run.
"""
from __future__ import annotations

import numpy as np
import pytest

from src import sef_itp_direction_axis as da


# --------------------------------------------------------------------------- #
# Module sanity
# --------------------------------------------------------------------------- #


def test_module_version():
    assert da.__version__ == "v1.0.0"


def test_locked_constants():
    """Framework-time locks. Adjusting these is a contract violation."""
    assert da.DEGENERACY_LAMBDA_RATIO_12_THRESHOLD == 0.10
    assert da.COS_AXIS_REVERSAL_THRESHOLD == 0.5
    assert da.DUAL_SOURCE_COS_ABS_MAX == 0.5
    assert da.R2_DUAL_SOURCE_MAX == 0.20
    assert da.SLOPE_PVALUE_THRESHOLD == 0.05
    assert da.AXIS_PERM_PVALUE_THRESHOLD == 0.05
    assert da.MIN_UNIVERSE_SIZE == 6
    assert da.MIN_EVENT_ELIGIBLE_FACTOR == 2
    assert da.DEFAULT_N_PERM == 1000
    assert da.DEFAULT_K_EVENT == 2


# --------------------------------------------------------------------------- #
# Helpers (synthetic builders)
# --------------------------------------------------------------------------- #


def _make_axis_reversal_subject(n=12, decision_k=3, seed=0):
    """n channels predominantly along X (the propagation axis) but with realistic
    3D shaft scatter in Y/Z (~3 mm SD). Cluster A rank ascends 0..n-1, B descends.

    The Y/Z scatter is what avoids near-1D PCA degeneracy; this matches real SEEG
    where shafts span multiple anatomical structures and aren't collinear.
    """
    rng = np.random.default_rng(seed)
    coords = np.column_stack([
        np.arange(n, dtype=float),
        rng.normal(0, 3.0, n),
        rng.normal(0, 3.0, n),
    ])
    universe_mask = np.ones(n, dtype=bool)
    rank_a = np.arange(n, dtype=float)
    rank_b = (n - 1 - np.arange(n)).astype(float)
    return coords, universe_mask, rank_a, rank_b, decision_k


def _make_dual_source_subject(seed=0):
    """16 channels in two orthogonal clouds; A axis along X, B axis along Y, no overlap."""
    rng = np.random.default_rng(seed)
    # cluster A endpoints on x-axis at x=0 (source) and x=20 (sink)
    # cluster B endpoints on y-axis at y=0 (source) and y=20 (sink)
    a_src = np.column_stack([rng.normal(0, 0.3, 3), rng.normal(0, 0.3, 3), rng.normal(0, 0.3, 3)])
    a_snk = np.column_stack([rng.normal(20, 0.3, 3), rng.normal(0, 0.3, 3), rng.normal(0, 0.3, 3)])
    b_src = np.column_stack([rng.normal(10, 0.3, 3), rng.normal(0, 0.3, 3), rng.normal(0, 0.3, 3)])
    b_snk = np.column_stack([rng.normal(10, 0.3, 3), rng.normal(20, 0.3, 3), rng.normal(0, 0.3, 3)])
    # other channels scattered in X-Y to fill universe
    other = np.column_stack([rng.uniform(5, 15, 4), rng.uniform(5, 15, 4), rng.normal(0, 0.3, 4)])
    coords = np.vstack([a_src, a_snk, b_src, b_snk, other])  # 16 channels
    n = coords.shape[0]
    universe_mask = np.ones(n, dtype=bool)
    # rank_a: low at a_src (rows 0,1,2), high at a_snk (rows 3,4,5), middle for others
    rank_a = np.zeros(n, dtype=float)
    rank_a[:3] = [0, 1, 2]
    rank_a[3:6] = [n - 3, n - 2, n - 1]
    other_ranks = np.arange(3, n - 3)
    rank_a[6:] = other_ranks
    # rank_b: low at b_src (rows 6,7,8), high at b_snk (rows 9,10,11)
    rank_b = np.zeros(n, dtype=float)
    rank_b[6:9] = [0, 1, 2]
    rank_b[9:12] = [n - 3, n - 2, n - 1]
    # fill a_src/a_snk and others with middle ranks for B
    rank_b[:6] = [3, 4, 5, 6, 7, 8]
    rank_b[12:] = [9, 10, 11, 12]
    decision_k = 3
    return coords, universe_mask, rank_a, rank_b, decision_k


def _make_same_direction_subject(n=12, decision_k=3, seed=0):
    """Cluster A and B share the same source/sink → cos(A,+B) ~ 1.

    Same Y/Z scatter as axis-reversal subject to keep PCA in 3D (not near-1D).
    """
    rng = np.random.default_rng(seed)
    coords = np.column_stack([
        np.arange(n, dtype=float),
        rng.normal(0, 3.0, n),
        rng.normal(0, 3.0, n),
    ])
    universe_mask = np.ones(n, dtype=bool)
    rank_a = np.arange(n, dtype=float)
    rank_b = np.arange(n, dtype=float) + rng.normal(0, 0.1, n)
    return coords, universe_mask, rank_a, rank_b, decision_k


def _make_single_shaft_subject(n=10, decision_k=3, seed=0):
    """All channels on one line; tiny noise in 2nd/3rd dims → near-1D PCA degenerate."""
    rng = np.random.default_rng(seed)
    coords = np.column_stack([
        np.arange(n, dtype=float),
        rng.normal(0, 1e-3, n),
        rng.normal(0, 1e-3, n),
    ])
    universe_mask = np.ones(n, dtype=bool)
    rank_a = np.arange(n, dtype=float)
    rank_b = (n - 1 - np.arange(n)).astype(float)
    return coords, universe_mask, rank_a, rank_b, decision_k


# --------------------------------------------------------------------------- #
# Layer 0 — universe and source/sink extraction
# --------------------------------------------------------------------------- #


def test_compute_universe_mask_intersection():
    jv = np.array([True, True, False, True, True], dtype=bool)
    mm = np.array([True, False, True, True, False], dtype=bool)
    u = da.compute_universe_mask(jv, mm)
    assert u.tolist() == [True, False, False, True, False]


def test_compute_universe_mask_shape_mismatch_raises():
    with pytest.raises(ValueError):
        da.compute_universe_mask(np.zeros(3, dtype=bool), np.zeros(4, dtype=bool))


def test_derive_source_sink_uses_lowest_and_highest_dense_rank():
    rank = np.array([5.0, 2.0, 9.0, 1.0, 0.0, 8.0, 7.0, 4.0, 6.0, 3.0])  # 10 ch
    universe = np.ones(10, dtype=bool)
    source, sink = da.derive_source_sink_within_universe(rank, universe, decision_k=3)
    # lowest 3 ranks: positions 4, 3, 1 (values 0, 1, 2); sorted asc: 1, 3, 4
    assert sorted(source.tolist()) == [1, 3, 4]
    # highest 3 ranks: positions 5, 6, 2 (values 8, 7, 9); sorted asc: 2, 5, 6
    assert sorted(sink.tolist()) == [2, 5, 6]


def test_derive_source_sink_raises_when_universe_too_small_for_2k():
    rank = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    universe = np.ones(5, dtype=bool)
    with pytest.raises(ValueError, match="universe size 5"):
        da.derive_source_sink_within_universe(rank, universe, decision_k=3)


# --------------------------------------------------------------------------- #
# Layer 4 — degeneracy detector
# --------------------------------------------------------------------------- #


def test_degeneracy_near_1d_fires_on_single_shaft():
    coords, universe, _, _, _ = _make_single_shaft_subject()
    deg = da.detect_degeneracy(coords, universe)
    assert deg.is_near_1d is True
    assert deg.lambda_ratio_12 < 0.10


def test_degeneracy_not_near_1d_on_3d_cloud():
    rng = np.random.default_rng(0)
    coords = rng.normal(0, 5, (20, 3))
    universe = np.ones(20, dtype=bool)
    deg = da.detect_degeneracy(coords, universe)
    assert deg.is_near_1d is False


# --------------------------------------------------------------------------- #
# Layer 1 — axis pair alignment + permutation null
# --------------------------------------------------------------------------- #


def test_compute_template_axis_v_axis_direction():
    coords = np.array([[0, 0, 0], [1, 0, 0], [10, 0, 0], [11, 0, 0]], dtype=float)
    ta = da.compute_template_axis(coords, source_indices=np.array([0, 1]),
                                  sink_indices=np.array([2, 3]), cluster_id=0)
    # v_axis = mean(sink) - mean(source) = (10.5, 0, 0) - (0.5, 0, 0) = (10, 0, 0)
    np.testing.assert_allclose(ta.v_axis, [10.0, 0.0, 0.0])
    assert ta.axis_length == pytest.approx(10.0)


def test_axis_pair_alignment_antipodal_axis_reversal():
    """Same-axis reversal → cos(v_A, -v_B) ≈ 1, permutation p ≈ 0."""
    coords, universe, rank_a, rank_b, k = _make_axis_reversal_subject()
    src_a, snk_a = da.derive_source_sink_within_universe(rank_a, universe, k)
    src_b, snk_b = da.derive_source_sink_within_universe(rank_b, universe, k)
    ta = da.compute_template_axis(coords, src_a, snk_a, cluster_id=0)
    tb = da.compute_template_axis(coords, src_b, snk_b, cluster_id=1)
    align = da.compute_axis_pair_alignment(coords, universe, ta, tb, n_perm=500, seed=0)
    assert align.cos_A_neg_B > 0.95
    assert align.cos_A_pos_B < -0.95
    assert align.p_one_sided_axis_reversal < 0.05
    assert align.n_perm_completed > 0


def test_axis_pair_alignment_dual_source_orthogonal():
    """Dual source → |cos(v_A, ±v_B)| ≈ 0."""
    coords, universe, rank_a, rank_b, k = _make_dual_source_subject()
    src_a, snk_a = da.derive_source_sink_within_universe(rank_a, universe, k)
    src_b, snk_b = da.derive_source_sink_within_universe(rank_b, universe, k)
    ta = da.compute_template_axis(coords, src_a, snk_a, cluster_id=0)
    tb = da.compute_template_axis(coords, src_b, snk_b, cluster_id=1)
    align = da.compute_axis_pair_alignment(coords, universe, ta, tb, n_perm=500, seed=0)
    assert abs(align.cos_A_neg_B) < 0.4
    assert abs(align.cos_A_pos_B) < 0.4


def test_axis_pair_alignment_same_direction():
    """Cluster A and B share direction → cos(v_A, +v_B) ≈ 1."""
    coords, universe, rank_a, rank_b, k = _make_same_direction_subject()
    src_a, snk_a = da.derive_source_sink_within_universe(rank_a, universe, k)
    src_b, snk_b = da.derive_source_sink_within_universe(rank_b, universe, k)
    ta = da.compute_template_axis(coords, src_a, snk_a, cluster_id=0)
    tb = da.compute_template_axis(coords, src_b, snk_b, cluster_id=1)
    align = da.compute_axis_pair_alignment(coords, universe, ta, tb, n_perm=500, seed=0)
    assert align.cos_A_pos_B > 0.9
    assert align.cos_A_neg_B < -0.9


def test_permutation_null_median_close_to_zero_in_axis_reversal():
    """Null distribution from role shuffle should have median near 0 (random axis direction)."""
    coords, universe, rank_a, rank_b, k = _make_axis_reversal_subject(n=14, decision_k=3, seed=42)
    src_a, snk_a = da.derive_source_sink_within_universe(rank_a, universe, k)
    src_b, snk_b = da.derive_source_sink_within_universe(rank_b, universe, k)
    ta = da.compute_template_axis(coords, src_a, snk_a, cluster_id=0)
    tb = da.compute_template_axis(coords, src_b, snk_b, cluster_id=1)
    align = da.compute_axis_pair_alignment(coords, universe, ta, tb, n_perm=1000, seed=0)
    # Null median should be near 0 (no axis-pair structure under role shuffle).
    # Allow generous tolerance because the union universe is small (12) and shaft
    # alignment can bias slightly.
    assert abs(align.null_cos_A_neg_B_median) < 0.40


# --------------------------------------------------------------------------- #
# Layer 3 — axis projection slope
# --------------------------------------------------------------------------- #


def test_slope_A_on_axisA_is_trivially_positive_monotone():
    """Sanity: cluster A's rank vs A's own axis projection → strong + slope."""
    coords, universe, rank_a, _, k = _make_axis_reversal_subject()
    src_a, snk_a = da.derive_source_sink_within_universe(rank_a, universe, k)
    ta = da.compute_template_axis(coords, src_a, snk_a, cluster_id=0)
    slope_AA = da.compute_axis_projection_slope(
        rank_a, universe, coords, ta.v_axis, cluster_id=0, n_perm=200, seed=0,
    )
    assert slope_AA.slope > 0
    assert slope_AA.r2 > 0.9


def test_slope_B_on_axisA_is_negative_in_axis_reversal():
    """Discriminative: cluster B's rank vs A's axis projection → strong NEGATIVE slope."""
    coords, universe, rank_a, rank_b, k = _make_axis_reversal_subject()
    src_a, snk_a = da.derive_source_sink_within_universe(rank_a, universe, k)
    ta = da.compute_template_axis(coords, src_a, snk_a, cluster_id=0)
    slope_BA = da.compute_axis_projection_slope(
        rank_b, universe, coords, ta.v_axis, cluster_id=1, n_perm=500, seed=0,
    )
    assert slope_BA.slope < 0
    assert slope_BA.r2 > 0.9
    assert slope_BA.p_one_sided_neg_slope < 0.05


def test_slope_B_on_axisA_near_zero_and_low_r2_in_dual_source():
    coords, universe, rank_a, rank_b, k = _make_dual_source_subject()
    src_a, snk_a = da.derive_source_sink_within_universe(rank_a, universe, k)
    ta = da.compute_template_axis(coords, src_a, snk_a, cluster_id=0)
    slope_BA = da.compute_axis_projection_slope(
        rank_b, universe, coords, ta.v_axis, cluster_id=1, n_perm=500, seed=0,
    )
    # With orthogonal axes the projection contains no rank info for cluster B
    assert slope_BA.r2 < 0.30


# --------------------------------------------------------------------------- #
# Layer 2 — event-level direction (optional)
# --------------------------------------------------------------------------- #


def test_event_direction_cos_centers_near_pos1_for_cluster_A():
    """Synthetic events of cluster A should have direction ≈ v_A → cos ≈ +1."""
    coords, universe, rank_a, _, k = _make_axis_reversal_subject(n=12, decision_k=3)
    n_ch = coords.shape[0]
    rng = np.random.default_rng(0)
    n_events = 30
    # Build ranks_full + bools_full: each event sees all channels, ranks ≈ rank_a + jitter
    ranks_full = np.zeros((n_ch, n_events), dtype=float)
    bools_full = np.ones((n_ch, n_events), dtype=bool)
    for e in range(n_events):
        jitter = rng.normal(0, 0.5, n_ch)
        ranks_full[:, e] = rank_a + jitter
    event_labels = np.zeros(n_events, dtype=int)
    src_a, snk_a = da.derive_source_sink_within_universe(rank_a, universe, k)
    ta = da.compute_template_axis(coords, src_a, snk_a, cluster_id=0)
    stats = da.compute_event_direction_stats(
        ranks_full, bools_full, event_labels, universe, coords, ta.v_axis,
        cluster_id=0, k_event=2,
    )
    assert stats.n_events_eligible > 0
    assert stats.cos_with_u_A_median > 0.9


def test_event_direction_k_event_2_vs_3_consistent_sign():
    """k=2 and k=3 should agree on direction sign in clean synthetic data."""
    coords, universe, rank_a, _, k_decision = _make_axis_reversal_subject(n=14, decision_k=3)
    n_ch = coords.shape[0]
    rng = np.random.default_rng(1)
    n_events = 40
    ranks_full = np.tile(rank_a[:, None], (1, n_events)).astype(float)
    ranks_full += rng.normal(0, 0.3, ranks_full.shape)
    bools_full = np.ones((n_ch, n_events), dtype=bool)
    event_labels = np.zeros(n_events, dtype=int)
    src_a, snk_a = da.derive_source_sink_within_universe(rank_a, universe, k_decision)
    ta = da.compute_template_axis(coords, src_a, snk_a, cluster_id=0)
    s2 = da.compute_event_direction_stats(
        ranks_full, bools_full, event_labels, universe, coords, ta.v_axis,
        cluster_id=0, k_event=2,
    )
    s3 = da.compute_event_direction_stats(
        ranks_full, bools_full, event_labels, universe, coords, ta.v_axis,
        cluster_id=0, k_event=3,
    )
    assert np.sign(s2.cos_with_u_A_median) == np.sign(s3.cos_with_u_A_median)
    assert s2.cos_with_u_A_median > 0
    assert s3.cos_with_u_A_median > 0


# --------------------------------------------------------------------------- #
# Layer 5 — SOZ relation
# --------------------------------------------------------------------------- #


def test_soz_relation_returns_distance_when_soz_present():
    coords = np.array([
        [0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0],
        [10, 0, 0], [11, 0, 0], [12, 0, 0], [13, 0, 0],
    ], dtype=float)
    universe = np.ones(8, dtype=bool)
    soz_mask = np.array([True, True, False, False, False, False, False, False])  # SOZ near source
    ta = da.TemplateAxis(0, [0, 1], [6, 7], coords[[0, 1]].mean(0), coords[[6, 7]].mean(0),
                         coords[[6, 7]].mean(0) - coords[[0, 1]].mean(0), 12.0)
    tb = da.TemplateAxis(1, [6, 7], [0, 1], coords[[6, 7]].mean(0), coords[[0, 1]].mean(0),
                         coords[[0, 1]].mean(0) - coords[[6, 7]].mean(0), 12.0)
    soz = da.compute_soz_relation(coords, universe, soz_mask, ta, tb,
                                  mapped_mask=np.ones(8, dtype=bool))
    assert soz.exit_reason is None
    assert soz.n_soz_in_universe == 2
    # source_A close to SOZ, sink_A far
    assert soz.d_source_A_to_SOZ < soz.d_sink_A_to_SOZ


def test_soz_relation_reports_both_mapped_full_and_joint_universe():
    """v1.0.2 audit fix #5: mapped-full and joint-universe SOZ subsets reported separately.

    Construct: 4 SOZ channels — 2 inside universe (joint-valid), 2 outside (mapped but not
    joint-valid). mapped_full should have n=4; joint_universe should have n=2.
    """
    coords = np.array([
        [0, 0, 0], [1, 0, 0],
        [10, 0, 0], [11, 0, 0],
        [50, 0, 0], [51, 0, 0],  # mapped-full SOZ channels OUTSIDE universe
        [60, 0, 0], [61, 0, 0],
    ], dtype=float)
    universe = np.array([True, True, True, True, False, False, False, False])  # first 4 only
    soz_mask = np.array([True, True, False, False, True, True, False, False])  # 4 SOZ total: 2 in universe + 2 outside
    mapped_mask = np.ones(8, dtype=bool)
    ta = da.TemplateAxis(0, [0, 1], [2, 3], coords[[0, 1]].mean(0), coords[[2, 3]].mean(0),
                         coords[[2, 3]].mean(0) - coords[[0, 1]].mean(0), 10.0)
    tb = da.TemplateAxis(1, [2, 3], [0, 1], coords[[2, 3]].mean(0), coords[[0, 1]].mean(0),
                         coords[[0, 1]].mean(0) - coords[[2, 3]].mean(0), 10.0)
    soz = da.compute_soz_relation(coords, universe, soz_mask, ta, tb, mapped_mask=mapped_mask)
    assert soz.mapped_full is not None
    assert soz.mapped_full.n == 4
    assert soz.joint_universe is not None
    assert soz.joint_universe.n == 2
    # joint-universe centroid is at mean of [0,0,0],[1,0,0] = (0.5, 0, 0)
    np.testing.assert_allclose(soz.joint_universe.centroid, [0.5, 0.0, 0.0], atol=1e-6)
    # mapped-full centroid pulled toward x=50,51 → much higher x
    assert soz.mapped_full.centroid[0] > 10.0
    # Back-compat top-level fields mirror joint_universe (not mapped_full)
    assert soz.n_soz_in_universe == 2


def test_soz_relation_exits_when_mask_none():
    coords = np.array([[0, 0, 0]] * 6, dtype=float)
    universe = np.ones(6, dtype=bool)
    ta = da.TemplateAxis(0, [0], [1], coords[0], coords[1], np.zeros(3), 0.0)
    tb = da.TemplateAxis(1, [1], [0], coords[1], coords[0], np.zeros(3), 0.0)
    soz = da.compute_soz_relation(coords, universe, None, ta, tb)
    assert soz.exit_reason == "no_soz_mask"
    assert soz.n_soz_in_universe == 0


# --------------------------------------------------------------------------- #
# Verdict
# --------------------------------------------------------------------------- #


def _build_full_pipeline(coords, universe, rank_a, rank_b, decision_k, n_perm=500):
    src_a, snk_a = da.derive_source_sink_within_universe(rank_a, universe, decision_k)
    src_b, snk_b = da.derive_source_sink_within_universe(rank_b, universe, decision_k)
    ta = da.compute_template_axis(coords, src_a, snk_a, cluster_id=0)
    tb = da.compute_template_axis(coords, src_b, snk_b, cluster_id=1)
    deg = da.detect_degeneracy(coords, universe)
    align = da.compute_axis_pair_alignment(coords, universe, ta, tb, n_perm=n_perm, seed=0)
    slope_BA = da.compute_axis_projection_slope(
        rank_b, universe, coords, ta.v_axis, cluster_id=1, n_perm=n_perm, seed=0,
    )
    return da.assess_subject_verdict(int(universe.sum()), deg, align, slope_BA)


def test_verdict_axis_reversal_on_synthetic():
    coords, universe, rank_a, rank_b, k = _make_axis_reversal_subject()
    verdict = _build_full_pipeline(coords, universe, rank_a, rank_b, k)
    assert verdict.label == "axis_reversal"


def test_verdict_dual_source_on_synthetic():
    coords, universe, rank_a, rank_b, k = _make_dual_source_subject()
    verdict = _build_full_pipeline(coords, universe, rank_a, rank_b, k)
    assert verdict.label == "dual_source"


def test_verdict_same_direction_on_synthetic():
    coords, universe, rank_a, rank_b, k = _make_same_direction_subject()
    verdict = _build_full_pipeline(coords, universe, rank_a, rank_b, k)
    assert verdict.label == "same_direction"


def test_verdict_degenerate_geometry_overrides_axis_reversal():
    """Even with strong axis reversal signal, near-1D layout must force degenerate label."""
    coords, universe, rank_a, rank_b, k = _make_single_shaft_subject(n=10, decision_k=3)
    verdict = _build_full_pipeline(coords, universe, rank_a, rank_b, k)
    assert verdict.label == "degenerate_geometry"


def test_verdict_exit_no_universe_on_small_universe():
    """n_universe < 6 → exit_no_universe."""
    coords = np.eye(5)
    universe = np.array([True, True, True, True, False])
    # decision_k=1 here only to allow source/sink derivation; verdict should still exit
    rank_a = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    rank_b = rank_a[::-1].copy()
    # We bypass derive_source_sink to construct a placeholder verdict
    n_universe = int(universe.sum())
    deg = da.detect_degeneracy(coords, universe)
    # Build dummy alignment + slope with NaN
    align = da.AxisPairAlignment(0, 0, 0, 0, 0, 0, 0, 0)
    slope = da.AxisProjectionSlope(1, 0, 0, 0, 0, 0, 0, 0)
    verdict = da.assess_subject_verdict(n_universe, deg, align, slope)
    assert verdict.label == "exit_no_universe"


def _build_descriptive_pipeline(coords, universe, rank_a, rank_b, decision_k, n_perm=300):
    """Same as _build_full_pipeline but returns descriptive_geometry instead of strict verdict."""
    src_a, snk_a = da.derive_source_sink_within_universe(rank_a, universe, decision_k)
    src_b, snk_b = da.derive_source_sink_within_universe(rank_b, universe, decision_k)
    ta = da.compute_template_axis(coords, src_a, snk_a, cluster_id=0)
    tb = da.compute_template_axis(coords, src_b, snk_b, cluster_id=1)
    deg = da.detect_degeneracy(coords, universe)
    align = da.compute_axis_pair_alignment(coords, universe, ta, tb, n_perm=n_perm, seed=0)
    slope_BA = da.compute_axis_projection_slope(
        rank_b, universe, coords, ta.v_axis, cluster_id=1, n_perm=n_perm, seed=0,
    )
    return da.assess_descriptive_geometry(int(universe.sum()), deg, align, slope_BA)


def test_locked_r2_descriptive_min():
    """v1.0.2 audit fix #3: descriptive shape labels require r² >= 0.20."""
    assert da.R2_DESCRIPTIVE_MIN == 0.20


def test_descriptive_geometry_axis_reversal_shaped():
    coords, universe, rank_a, rank_b, k = _make_axis_reversal_subject()
    d = _build_descriptive_pipeline(coords, universe, rank_a, rank_b, k)
    assert d.label == "axis_reversal_shaped"


def test_descriptive_geometry_axis_reversal_REJECTED_when_r2_low():
    """v1.0.2 audit fix #3: low-r² noise-driven negative slope must NOT enter axis_reversal_shaped.

    Construct an alignment with cos_neg >= 0.5 and slope_B < 0 but r² = 0.05 → fall to unclear.
    """
    coords, universe, _, _, _ = _make_axis_reversal_subject()
    deg = da.detect_degeneracy(coords, universe)
    align = da.AxisPairAlignment(
        cos_A_pos_B=-0.9, cos_A_neg_B=0.9,
        angle_A_pos_B_deg=180.0, angle_A_neg_B_deg=0.0,
        null_cos_A_neg_B_median=0.0, null_cos_A_neg_B_95=0.5,
        p_one_sided_axis_reversal=0.01, n_perm_completed=1000,
    )
    slope_BA_lowr2 = da.AxisProjectionSlope(
        cluster_id=1, slope=-0.05, intercept=0.0, r2=0.05, spearman_rho=-0.1,
        n_points=12, null_slope_median=0.0, p_one_sided_neg_slope=0.04,
    )
    d = da.assess_descriptive_geometry(int(universe.sum()), deg, align, slope_BA_lowr2)
    assert d.label == "unclear"
    assert "r2" in d.reason or "r²" in d.reason or "no descriptive pattern" in d.reason


def test_descriptive_geometry_dual_source_shaped():
    coords, universe, rank_a, rank_b, k = _make_dual_source_subject()
    d = _build_descriptive_pipeline(coords, universe, rank_a, rank_b, k)
    assert d.label == "dual_source_shaped"


def test_descriptive_geometry_same_direction_shaped():
    coords, universe, rank_a, rank_b, k = _make_same_direction_subject()
    d = _build_descriptive_pipeline(coords, universe, rank_a, rank_b, k)
    assert d.label == "same_direction_shaped"


def test_descriptive_geometry_unclear_on_near_1d():
    """Near-1D PCA degenerates should yield descriptive 'unclear' regardless of cos."""
    coords, universe, rank_a, rank_b, k = _make_single_shaft_subject()
    d = _build_descriptive_pipeline(coords, universe, rank_a, rank_b, k)
    assert d.label == "unclear"


def test_descriptive_geometry_ignores_perm_p_value():
    """Manually construct a case where strict verdict would gate on perm p but
    descriptive should still label by geometry alone."""
    # Re-use axis_reversal subject; build alignment manually with NaN perm p,
    # confirming descriptive still fires on cos + slope.
    coords, universe, rank_a, rank_b, k = _make_axis_reversal_subject()
    src_a, snk_a = da.derive_source_sink_within_universe(rank_a, universe, k)
    src_b, snk_b = da.derive_source_sink_within_universe(rank_b, universe, k)
    ta = da.compute_template_axis(coords, src_a, snk_a, cluster_id=0)
    tb = da.compute_template_axis(coords, src_b, snk_b, cluster_id=1)
    deg = da.detect_degeneracy(coords, universe)
    # Manually craft an alignment where perm p is high (no perm signal) but geometry strong.
    cos_neg = 0.95
    cos_pos = -0.95
    align = da.AxisPairAlignment(
        cos_A_pos_B=cos_pos, cos_A_neg_B=cos_neg,
        angle_A_pos_B_deg=180.0, angle_A_neg_B_deg=0.0,
        null_cos_A_neg_B_median=0.5, null_cos_A_neg_B_95=0.9,
        p_one_sided_axis_reversal=0.8,  # perm null can't reject
        n_perm_completed=1000,
    )
    slope_BA = da.compute_axis_projection_slope(
        rank_b, universe, coords, ta.v_axis, cluster_id=1, n_perm=200, seed=0,
    )
    strict = da.assess_subject_verdict(int(universe.sum()), deg, align, slope_BA)
    desc = da.assess_descriptive_geometry(int(universe.sum()), deg, align, slope_BA)
    # Strict gate forbidden by high perm p
    assert strict.label == "inconclusive"
    # Descriptive ignores perm p — labels by geometry
    assert desc.label == "axis_reversal_shaped"


def test_serialization_keys_present():
    """Sanity: serialized dicts contain expected keys (catches accidental field renames)."""
    coords, universe, rank_a, rank_b, k = _make_axis_reversal_subject()
    src_a, snk_a = da.derive_source_sink_within_universe(rank_a, universe, k)
    ta = da.compute_template_axis(coords, src_a, snk_a, cluster_id=0)
    d = da.serialize_template_axis(ta)
    for key in ["cluster_id", "source_indices", "sink_indices",
                "source_centroid", "sink_centroid", "v_axis", "axis_length"]:
        assert key in d
    deg = da.detect_degeneracy(coords, universe)
    sd = da.serialize_degeneracy(deg)
    assert {"lambda_eigenvalues", "lambda_ratio_12", "lambda_ratio_23",
            "is_near_1d", "is_near_2d"}.issubset(sd.keys())
