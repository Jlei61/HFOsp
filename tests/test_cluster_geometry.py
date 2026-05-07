"""Tests for src/cluster_geometry.py.

Design doc:
docs/archive/topic1/propagation/cluster_geometry_viz_plan_2026-05-06.md §7.3
(TDD contract).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.cluster_geometry import (
    classical_mds,
    compute_augmented_distance_matrix,
    compute_event_event_distances,
    compute_event_template_distances,
    compute_masked_distance,
    compute_per_event_silhouette,
    compute_subject_geometry,
    compute_template_template_distances,
    summarize_cohort_geometry,
)
from src.interictal_propagation import (
    assign_events_to_templates,
    build_cluster_templates,
)


# ---------------------------------------------------------------------------
# Test 1: compute_masked_distance basic properties
# ---------------------------------------------------------------------------


def test_masked_distance_self_zero_symmetric_nonneg():
    rng = np.random.default_rng(0)
    rank = rng.uniform(0, 5, 8)
    bool_v = np.ones(8, dtype=bool)
    d = compute_masked_distance(rank, rank, bool_v, bool_v)
    assert d == 0.0
    # Symmetry
    rank2 = rng.uniform(0, 5, 8)
    d_xy = compute_masked_distance(rank, rank2, bool_v, bool_v)
    d_yx = compute_masked_distance(rank2, rank, bool_v, bool_v)
    assert d_xy == pytest.approx(d_yx)
    # Non-negative
    assert d_xy >= 0.0


# ---------------------------------------------------------------------------
# Test 2: shared < min_shared → NaN
# ---------------------------------------------------------------------------


def test_masked_distance_min_shared_gate():
    rank = np.array([1.0, 2.0, 3.0, 4.0])
    bool_a = np.array([True, True, False, False])
    bool_b = np.array([False, True, True, False])
    # Only 1 shared channel
    d = compute_masked_distance(rank, rank, bool_a, bool_b, min_shared=3)
    assert np.isnan(d)
    # With min_shared=1, returns 0
    d2 = compute_masked_distance(rank, rank, bool_a, bool_b, min_shared=1)
    assert d2 == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 3: matches assign_events_to_templates internal distance
# ---------------------------------------------------------------------------


def test_distance_consistent_with_assign_events_to_templates():
    """sqrt(d_geometry²) should equal mean_sq from assign_events_to_templates
    on the SAME event-template pair.
    """
    rng = np.random.default_rng(42)
    n_ch = 6
    n_events = 30
    n_clusters = 2

    ranks = rng.uniform(0, n_ch - 1, (n_ch, n_events)).astype(float)
    bools = rng.random((n_ch, n_events)) > 0.3  # ~70% participation

    templates_real = np.full((n_clusters, n_ch), np.nan, dtype=float)
    templates_real[0] = rng.uniform(0, n_ch - 1, n_ch)
    templates_real[1] = rng.uniform(0, n_ch - 1, n_ch)

    # Event 0, template 0
    ev = 0
    tk = 0
    template_bool_k = np.isfinite(templates_real[tk])
    d = compute_masked_distance(
        ranks[:, ev],
        templates_real[tk],
        bools[:, ev],
        template_bool_k,
        min_shared=3,
    )

    # assign_events_to_templates computes mean_sq = nansum(sq_diff) / max(n_valid,1)
    # Replicate that exactly:
    masked_ranks = np.where(bools > 0, ranks, np.nan)
    diff = masked_ranks - templates_real[tk][:, None]
    sq_diff = diff ** 2
    valid = np.isfinite(sq_diff)
    n_valid = int(valid[:, ev].sum())
    if n_valid >= 3:
        mean_sq = float(np.nansum(sq_diff[:, ev]) / max(n_valid, 1))
        # geometry's d == sqrt(mean_sq)
        assert d == pytest.approx(float(np.sqrt(mean_sq)))


# ---------------------------------------------------------------------------
# Test 4: classical_mds reconstructs known Euclidean configuration
# ---------------------------------------------------------------------------


def test_classical_mds_reconstructs_euclidean_2d():
    """Embed 6 known 2D points; compute pairwise distances; recover via MDS;
    distances of recovered embedding should match input distances.
    """
    pts = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [-0.5, 0.5], [2.0, 1.0], [1.5, -1.0]]
    )
    diffs = pts[:, None, :] - pts[None, :, :]
    D = np.sqrt((diffs * diffs).sum(axis=-1))

    out = classical_mds(D, n_components=2)
    Y = out["Y"]
    diffs_Y = Y[:, None, :] - Y[None, :, :]
    D_Y = np.sqrt((diffs_Y * diffs_Y).sum(axis=-1))

    # MDS preserves pairwise distances up to rotation/reflection
    # so D_Y should match D exactly (within floating point) for an
    # already-Euclidean input.
    assert np.allclose(D, D_Y, atol=1e-8)
    # Stress should be near zero
    assert out["stress"] < 1e-6


# ---------------------------------------------------------------------------
# Test 5: classical_mds output shape + stress in [0, inf)
# ---------------------------------------------------------------------------


def test_classical_mds_output_shape_and_stress():
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(10, 3))
    diffs = pts[:, None, :] - pts[None, :, :]
    D = np.sqrt((diffs * diffs).sum(axis=-1))
    out = classical_mds(D, n_components=2)
    assert out["Y"].shape == (10, 2)
    assert out["eigvals"].shape == (2,)
    # Stress is non-negative; for 3D->2D embedding we expect > 0
    assert out["stress"] >= 0.0


# ---------------------------------------------------------------------------
# Test 6: anti-correlated templates → opposite ends in MDS
# ---------------------------------------------------------------------------


def test_anticorrelated_templates_opposite_in_mds():
    """Construct two clusters whose templates are exact rank reversals of
    each other; events sampled around each. After augmented MDS, the two
    template positions should be far apart, and event clouds should cluster
    around their assigned templates.
    """
    rng = np.random.default_rng(7)
    n_ch = 6
    n_per_cluster = 30

    template_a_rank = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    template_b_rank = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0])

    # Generate events: small perturbations of templates
    events_a = template_a_rank[:, None] + rng.normal(0, 0.5, (n_ch, n_per_cluster))
    events_b = template_b_rank[:, None] + rng.normal(0, 0.5, (n_ch, n_per_cluster))
    ranks = np.concatenate([events_a, events_b], axis=1)
    bools = np.ones((n_ch, 2 * n_per_cluster), dtype=bool)
    labels = np.concatenate(
        [np.zeros(n_per_cluster, dtype=int), np.ones(n_per_cluster, dtype=int)]
    )

    # Reuse build_cluster_templates to mimic the production path
    templates_real = build_cluster_templates(ranks, bools, labels, n_clusters=2)
    valid_idx = np.arange(2 * n_per_cluster)

    D_aug = compute_augmented_distance_matrix(
        ranks, bools, templates_real, valid_idx, min_shared=3
    )
    out = classical_mds(D_aug, n_components=2)
    Y = out["Y"]

    n_events = 2 * n_per_cluster
    template_a_xy = Y[n_events]
    template_b_xy = Y[n_events + 1]
    template_dist = float(np.linalg.norm(template_a_xy - template_b_xy))

    # Distances of cluster-A events to template A vs template B
    Y_events_a = Y[:n_per_cluster]
    Y_events_b = Y[n_per_cluster:n_events]
    a_to_ta = np.linalg.norm(Y_events_a - template_a_xy, axis=1).mean()
    a_to_tb = np.linalg.norm(Y_events_a - template_b_xy, axis=1).mean()
    b_to_ta = np.linalg.norm(Y_events_b - template_a_xy, axis=1).mean()
    b_to_tb = np.linalg.norm(Y_events_b - template_b_xy, axis=1).mean()

    # Each event cloud should land closer to its template than the other
    assert a_to_ta < a_to_tb
    assert b_to_tb < b_to_ta
    # Templates should be visibly separated
    assert template_dist > 0.0


# ---------------------------------------------------------------------------
# Test 7: silhouette boundary cases
# ---------------------------------------------------------------------------


def test_silhouette_boundary_cases():
    a = np.array([1.0, 1.0, 0.0, 1.0])
    b = np.array([1.0, 2.0, 0.0, np.nan])
    s = compute_per_event_silhouette(a, b)
    # a == b → 0
    assert s[0] == pytest.approx(0.0)
    # a < b → positive (= (2-1)/2 = 0.5)
    assert s[1] == pytest.approx(0.5)
    # both 0 → 0 (max(0,0) = 0 path)
    assert s[2] == pytest.approx(0.0)
    # NaN propagates
    assert np.isnan(s[3])


# ---------------------------------------------------------------------------
# Test 8: end-to-end subject pipeline on synthetic input
# ---------------------------------------------------------------------------


def test_subject_geometry_end_to_end_synthetic():
    rng = np.random.default_rng(11)
    n_ch = 8
    n_per_cluster = 40

    template_a_rank = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    template_b_rank = np.array([7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    events_a = template_a_rank[:, None] + rng.normal(0, 0.3, (n_ch, n_per_cluster))
    events_b = template_b_rank[:, None] + rng.normal(0, 0.3, (n_ch, n_per_cluster))
    ranks = np.concatenate([events_a, events_b], axis=1)
    bools = np.ones((n_ch, 2 * n_per_cluster), dtype=bool)
    labels = np.concatenate(
        [np.zeros(n_per_cluster, dtype=int), np.ones(n_per_cluster, dtype=int)]
    )
    valid_idx = np.arange(2 * n_per_cluster)

    out = compute_subject_geometry(
        ranks=ranks,
        bools=bools,
        channel_names=[f"ch{i}" for i in range(n_ch)],
        adaptive_labels=labels,
        chosen_k=2,
        valid_event_indices=valid_idx,
        min_shared=3,
        max_events_for_mds=200,
    )

    assert out["status"] == "ok"
    assert out["n_events_total"] == 2 * n_per_cluster
    assert out["chosen_k"] == 2
    # Synthetic data is highly separable → silhouette should be strongly positive
    assert out["silhouette_median"] > 0.5
    # KMeans-vs-matching agreement on synthetic separable data → 1.0 (or near)
    assert out["agreement_overall"] > 0.95
    # Each event has an MDS coord
    assert len(out["events"]) == 2 * n_per_cluster
    for ev in out["events"]:
        assert "mds_x" in ev
        assert "mds_y" in ev
        assert "silhouette" in ev


# ---------------------------------------------------------------------------
# Test 9: subsampling preserves all templates
# ---------------------------------------------------------------------------


def test_subject_geometry_subsamples_events_keeps_templates():
    rng = np.random.default_rng(13)
    n_ch = 6
    n_events = 1000  # > max_events_for_mds (set to 100)
    n_per_cluster = n_events // 2

    template_a = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    template_b = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    events_a = template_a[:, None] + rng.normal(0, 0.4, (n_ch, n_per_cluster))
    events_b = template_b[:, None] + rng.normal(0, 0.4, (n_ch, n_per_cluster))
    ranks = np.concatenate([events_a, events_b], axis=1)
    bools = np.ones((n_ch, n_events), dtype=bool)
    labels = np.concatenate(
        [np.zeros(n_per_cluster, dtype=int), np.ones(n_per_cluster, dtype=int)]
    )
    valid_idx = np.arange(n_events)

    out = compute_subject_geometry(
        ranks=ranks,
        bools=bools,
        channel_names=[f"ch{i}" for i in range(n_ch)],
        adaptive_labels=labels,
        chosen_k=2,
        valid_event_indices=valid_idx,
        min_shared=3,
        max_events_for_mds=100,  # forces subsample
    )
    assert out["subsampled"] is True
    assert out["n_events_used_for_mds"] == 100
    assert out["n_events_total"] == n_events
    # Both templates have non-NaN MDS coords (templates always retained)
    for tmpl in out["templates"]:
        assert np.isfinite(tmpl["mds_x"])
        assert np.isfinite(tmpl["mds_y"])
    # Non-subsampled events have NaN mds_x/mds_y; subsampled ones are finite
    finite_count = sum(
        1 for ev in out["events"] if np.isfinite(ev["mds_x"]) and np.isfinite(ev["mds_y"])
    )
    assert finite_count == 100


# ---------------------------------------------------------------------------
# Test 10: cohort summary aggregator handles excluded subjects
# ---------------------------------------------------------------------------


def test_cohort_summary_handles_mixed_subjects():
    per_subject = {
        "sub_a": {
            "status": "ok",
            "silhouette_median": 0.5,
            "silhouette_iqr": [0.3, 0.6],
            "agreement_overall": 0.92,
            "chosen_k": 2,
            "stress": 0.1,
            "imputed_fraction": 0.0,
            "imputation_warning": False,
            "stress_warning": False,
            "n_events_total": 500,
            "subsampled": False,
            "boundary_fraction_by_nparticipating": {"3-4": 0.2, "5-8": 0.05, "9+": 0.01},
        },
        "sub_b": {
            "status": "ok",
            "silhouette_median": 0.3,
            "silhouette_iqr": [0.1, 0.5],
            "agreement_overall": 0.78,
            "chosen_k": 4,
            "stress": 0.4,
            "imputed_fraction": 0.25,
            "imputation_warning": True,
            "stress_warning": True,
            "n_events_total": 1500,
            "subsampled": True,
            "boundary_fraction_by_nparticipating": {"3-4": 0.4, "5-8": 0.1, "9+": 0.02},
        },
        "sub_c": {
            "status": "excluded",
            "excluded_reason": "too_few_events",
            "n_events_total": 20,
        },
    }
    summary = summarize_cohort_geometry(per_subject)
    assert summary["n_subjects_included"] == 2
    assert summary["n_subjects_excluded"] == 1
    assert "sub_c" in summary["excluded"]
    assert "sub_b" in summary["subjects_high_stress"]
    assert "sub_b" in summary["subjects_high_imputation"]
    assert summary["joint_silhouette_vs_agreement_spearman"]["n"] == 2
    assert len(summary["boundary_fraction_by_nparticipating"]["3-4"]) == 2


# ---------------------------------------------------------------------------
# Test 11: all-NaN template triggers exclusion
# ---------------------------------------------------------------------------


def test_subject_geometry_all_nan_template_excluded():
    """If a cluster has zero participating channels across all its events,
    its template is all-NaN and the subject should be excluded.
    """
    n_ch = 5
    n_events = 60
    ranks = np.tile(np.arange(n_ch, dtype=float)[:, None], (1, n_events))
    bools = np.ones((n_ch, n_events), dtype=bool)
    # Cluster 1 has no events whose channels participate (we force bools=False
    # for all events labeled 1)
    labels = np.zeros(n_events, dtype=int)
    labels[: n_events // 2] = 0
    labels[n_events // 2 :] = 1
    bools[:, n_events // 2 :] = False  # cluster-1 events have empty bool

    out = compute_subject_geometry(
        ranks=ranks,
        bools=bools,
        channel_names=[f"ch{i}" for i in range(n_ch)],
        adaptive_labels=labels,
        chosen_k=2,
        valid_event_indices=np.arange(n_events),
        min_shared=3,
    )
    assert out["status"] == "excluded"
    assert out["excluded_reason"] == "all_nan_template"


# ---------------------------------------------------------------------------
# Test 12: too few events triggers exclusion
# ---------------------------------------------------------------------------


def test_subject_geometry_too_few_events_excluded():
    n_ch = 5
    n_events = 10  # below default n_min_events_total=50
    ranks = np.zeros((n_ch, n_events))
    bools = np.ones((n_ch, n_events), dtype=bool)
    labels = np.zeros(n_events, dtype=int)

    out = compute_subject_geometry(
        ranks=ranks,
        bools=bools,
        channel_names=[f"ch{i}" for i in range(n_ch)],
        adaptive_labels=labels,
        chosen_k=2,
        valid_event_indices=np.arange(n_events),
    )
    assert out["status"] == "excluded"
    assert out["excluded_reason"] == "too_few_events"
