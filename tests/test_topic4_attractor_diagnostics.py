"""Tests for src/topic4_attractor_diagnostics.py (Topic 4 Step 1)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.topic4_attractor_diagnostics import (  # noqa: E402
    _build_label_transition_count,
    _two_state_lambda2_from_count,
    build_rank_feature_matrix,
    compute_angle_to_kmeans_axis,
    compute_gof,
    compute_pr2_label_transition_sanity,
    fit_pca,
    fit_principal_curve,
    kmeans_centroids_from_labels,
    run_step1_subject,
)


# ---------- build_rank_feature_matrix ----------


def test_build_rank_feature_matrix_nan_to_zero_pr2_contract():
    """Inactive (bools=False) channels must be filled with 0.0 in X."""
    n_chan, n_event = 8, 200
    rng = np.random.default_rng(0)
    bools = rng.random((n_chan, n_event)) > 0.3
    bools[:, :30] = False  # force first 30 events to have n_part=0
    bools[:6, 30:80] = True
    bools[6:, 30:80] = False  # n_part=6 here
    bools[:, 80:] = True
    ranks = rng.uniform(1, n_chan + 1, size=(n_chan, n_event))
    ranks_with_nan = np.where(bools, ranks, np.nan)

    X, idx = build_rank_feature_matrix(ranks_with_nan, bools, min_participating=6)
    expected_eligible = int((bools.sum(axis=0) >= 6).sum())
    assert X.shape == (expected_eligible, n_chan)
    assert idx.shape == (expected_eligible,)
    for i, ev in enumerate(idx):
        for ch in range(n_chan):
            if not bools[ch, ev]:
                assert X[i, ch] == 0.0


def test_build_rank_feature_matrix_min_participating_gate():
    """min_participating=6 filters events with fewer participants."""
    n_chan, n_event = 10, 50
    bools = np.zeros((n_chan, n_event), dtype=bool)
    bools[:5, :20] = True
    bools[:8, 20:40] = True
    bools[:, 40:] = True
    ranks = np.broadcast_to(
        np.arange(1, n_chan + 1, dtype=float)[:, None], (n_chan, n_event)
    ).copy()
    ranks_with_nan = np.where(bools, ranks, np.nan)

    X6, idx6 = build_rank_feature_matrix(ranks_with_nan, bools, min_participating=6)
    assert idx6.size == 30  # last 30 events have n_part >= 6

    X3, idx3 = build_rank_feature_matrix(ranks_with_nan, bools, min_participating=3)
    assert idx3.size == 50

    X11, idx11 = build_rank_feature_matrix(ranks_with_nan, bools, min_participating=11)
    assert idx11.size == 0
    assert X11.shape == (0, n_chan)


# ---------- fit_pca ----------


def test_fit_pca_orthogonality_and_ratio_bounds():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 8))
    pca = fit_pca(X, n_components=3)
    assert pca["components"].shape == (3, 8)
    assert pca["scores"].shape == (300, 3)
    inner = pca["components"] @ pca["components"].T
    assert np.allclose(inner, np.eye(3), atol=1e-10)
    ratios = pca["explained_variance_ratio"]
    assert (ratios >= 0).all()
    assert (ratios <= 1).all()
    assert ratios.sum() < 1.0
    # Largest first
    assert (np.diff(pca["explained_variance"]) <= 1e-10).all()


def test_fit_pca_recovers_synthetic_low_rank():
    """If X is rank-1 with one dominant direction, PC1 ratio ~ 1.0."""
    rng = np.random.default_rng(2)
    direction = np.array([1.0, 1.0, 1.0, 0.0])
    direction /= np.linalg.norm(direction)
    t = rng.normal(size=400)
    X = t[:, None] * direction[None, :] + rng.normal(scale=0.001, size=(400, 4))
    pca = fit_pca(X, n_components=3)
    assert pca["explained_variance_ratio"][0] > 0.999


# ---------- fit_principal_curve ----------


def test_principal_curve_recovers_straight_line():
    rng = np.random.default_rng(3)
    n = 200
    t = np.linspace(0, 10, n)
    direction = np.array([1.0, 0.5, 0.2])
    direction /= np.linalg.norm(direction)
    points = t[:, None] * direction[None, :] + rng.normal(scale=0.05, size=(n, 3))
    curve = fit_principal_curve(points)
    total_var = float(np.var(points - points.mean(axis=0), axis=0).sum())
    var_explained = 1 - curve["residual_mean_sq"] / total_var
    assert var_explained > 0.95


def test_principal_curve_subsamples_when_n_exceeds_threshold():
    """When n > spline_max_n, fit uses stride-subsampled subset; projection uses full n."""
    rng = np.random.default_rng(7)
    n = 30000
    t = np.linspace(0, 10, n)
    direction = np.array([1.0, 0.5, 0.2])
    direction /= np.linalg.norm(direction)
    points = t[:, None] * direction[None, :] + rng.normal(scale=0.05, size=(n, 3))
    curve = fit_principal_curve(points, spline_max_n=5000)
    assert curve["spline_n_used"] <= 5001  # subsampled
    assert curve["s"].shape[0] == n        # but s computed for all n
    assert curve["residuals"].shape[0] == n
    total_var = float(np.var(points - points.mean(axis=0), axis=0).sum())
    var_explained = 1 - curve["residual_mean_sq"] / total_var
    assert var_explained > 0.95


def test_principal_curve_beats_pc1_on_arc():
    """For a 1D arc embedded in 3D, curve should explain more variance than PC1 alone."""
    rng = np.random.default_rng(4)
    n = 300
    theta = np.linspace(0, np.pi, n)
    radius = 5.0
    points = np.column_stack([
        radius * np.cos(theta),
        radius * np.sin(theta),
        np.zeros(n),
    ])
    points += rng.normal(scale=0.05, size=points.shape)

    pca = fit_pca(points, n_components=3)
    curve = fit_principal_curve(pca["scores"])
    total_pc_var = float(np.sum(pca["explained_variance"]))
    var_curve = 1 - curve["residual_mean_sq"] / total_pc_var
    pc1_ratio = float(pca["explained_variance_ratio"][0])

    assert var_curve > 0.9
    assert var_curve > pc1_ratio + 0.1


# ---------- compute_gof ----------


def test_compute_gof_perfect_fit():
    pc_var = np.array([4.0, 1.0, 0.25])
    residuals = np.zeros(100)
    gof = compute_gof(pc_var, residuals, n_events=100)
    assert gof["var_explained_curve"] == pytest.approx(1.0)


def test_compute_gof_consistent_scale():
    """var_unexplained == sample_var(residuals); var_explained = 1 - that / total."""
    rng = np.random.default_rng(5)
    n = 500
    pc_var = np.array([4.0, 1.0, 0.25])
    residuals = rng.normal(scale=0.5, size=n)
    gof = compute_gof(pc_var, np.abs(residuals), n_events=n)
    expected_unexplained = float(np.sum(residuals ** 2) / (n - 1))
    expected = 1.0 - expected_unexplained / float(np.sum(pc_var))
    assert gof["var_explained_curve"] == pytest.approx(expected, abs=1e-6)


# ---------- compute_angle_to_kmeans_axis ----------


def test_angle_aligned_pc1_fallback():
    n_features = 10
    components = np.eye(3, n_features)
    t0 = np.zeros(n_features)
    t1 = np.zeros(n_features)
    t1[0] = 1.0  # axis = PC1 direction
    angle = compute_angle_to_kmeans_axis(components, t0, t1, splines=None, s_eval=None)
    assert angle["angle_deg"] < 1.0


def test_angle_orthogonal_pc1_fallback():
    n_features = 10
    components = np.eye(3, n_features)
    t0 = np.zeros(n_features)
    t1 = np.zeros(n_features)
    t1[1] = 1.0  # axis = PC2 direction
    angle = compute_angle_to_kmeans_axis(components, t0, t1, splines=None, s_eval=None)
    assert 89 < angle["angle_deg"] < 91


def test_angle_returns_acute_only():
    """Angle should be in [0, 90] (sign of axis is arbitrary)."""
    n_features = 10
    components = np.eye(3, n_features)
    t0 = np.zeros(n_features)
    t1 = np.zeros(n_features)
    t1[0] = -1.0  # axis = -PC1; cosine = -1 but reported angle should be ~0
    angle = compute_angle_to_kmeans_axis(components, t0, t1, splines=None, s_eval=None)
    assert angle["angle_deg"] < 1.0


def test_angle_axis_mostly_outside_pc_subspace():
    """If axis lives in a feature dim outside the PC space, axis_explained_in_pc is small."""
    n_features = 10
    components = np.eye(3, n_features)  # only first 3 feature dims
    t0 = np.zeros(n_features)
    t1 = np.zeros(n_features)
    t1[5] = 1.0  # axis lives in feature dim 5, outside top-3 PC subspace
    angle = compute_angle_to_kmeans_axis(components, t0, t1, splines=None, s_eval=None)
    assert angle["axis_explained_in_pc"] < 0.01


# ---------- kmeans_centroids_from_labels ----------


def test_kmeans_centroids_from_labels():
    X = np.array(
        [[1.0, 0.0],
         [1.0, 1.0],
         [5.0, 5.0],
         [5.0, 6.0]]
    )
    labels = np.array([0, 0, 1, 1])
    centroids = kmeans_centroids_from_labels(X, labels, n_clusters=2)
    assert np.allclose(centroids[0], [1.0, 0.5])
    assert np.allclose(centroids[1], [5.0, 5.5])


# ---------- run_step1_subject end-to-end ----------


def _synth_two_cluster_ranks(rng, n_chan=10, n_event_per_cluster=300):
    """Two distinct rank patterns (forward vs reverse) with full participation."""
    base_forward = np.arange(1, n_chan + 1, dtype=float)
    base_reverse = base_forward[::-1].copy()

    ranks_list = []
    bools_list = []
    label_list = []
    for cluster_id, base in enumerate([base_forward, base_reverse]):
        for _ in range(n_event_per_cluster):
            jitter = base + rng.normal(scale=0.4, size=n_chan)
            r = (np.argsort(np.argsort(jitter)) + 1).astype(float)
            ranks_list.append(r)
            bools_list.append(np.ones(n_chan, dtype=bool))
            label_list.append(cluster_id)

    ranks = np.array(ranks_list).T  # (n_chan, n_event)
    bools = np.array(bools_list).T
    labels = np.array(label_list)
    return ranks, bools, labels


def test_run_step1_subject_smoke_two_cluster():
    rng = np.random.default_rng(42)
    ranks, bools, labels = _synth_two_cluster_ranks(rng)
    n_event = ranks.shape[1]
    valid_events = np.arange(n_event)

    out = run_step1_subject(
        ranks, bools,
        pr2_valid_events=valid_events, pr2_labels=labels, pr2_n_clusters=2,
        min_participating=6, n_components=3, gof_threshold=0.6,
    )
    assert "skipped" not in out
    assert out["n_events_eligible"] == n_event
    assert out["pca"]["cumulative_top_k"] > 0.5
    assert out["centroid_source"] == "pr2_labels_recomputed_in_X"
    # Synthetic jitter in rank space (re-ranked after noise injection) introduces
    # variance orthogonal to the cluster axis; 45° is well below the 90° random
    # baseline and sufficient as a smoke check.
    assert out["angle_to_kmeans_axis"]["angle_deg"] < 45
    assert out["gof_pass"] is True


def test_run_step1_subject_skips_when_too_few_events():
    rng = np.random.default_rng(43)
    ranks, bools, _ = _synth_two_cluster_ranks(rng, n_event_per_cluster=20)
    out = run_step1_subject(ranks, bools, min_participating=6)
    assert out.get("skipped") == "insufficient_eligible_events"


def test_run_step1_subject_excludes_on_label_length_mismatch():
    """If pr2_labels and pr2_valid_events have different lengths, subject is
    excluded from H3 main (no silent template fallback)."""
    rng = np.random.default_rng(45)
    ranks, bools, labels = _synth_two_cluster_ranks(rng)
    n_event = ranks.shape[1]
    valid_events = np.arange(n_event)
    # Simulate stale-PR-2 drift: labels longer than valid_events
    bad_labels = np.concatenate([labels, np.zeros(5, dtype=int)])
    out = run_step1_subject(
        ranks, bools,
        pr2_valid_events=valid_events, pr2_labels=bad_labels, pr2_n_clusters=2,
        min_participating=6,
    )
    assert "excluded_from_h3_main" in out
    assert "pr2_label_event_index_drift" in out["excluded_from_h3_main"]
    # angle should be nan (no axis available)
    assert not np.isfinite(out["angle_to_kmeans_axis"]["angle_deg"])
    assert out["centroid_source"] == "none"


# ---------- PR-2 label transition sanity ----------


def test_label_transition_count_drops_cross_block_pairs():
    labels = np.array([0, 1, 0, 1, 0, 1])
    block_ids = np.array([0, 0, 0, 1, 1, 1])
    M = _build_label_transition_count(labels, block_ids, n_clusters=2)
    # Within block 0: (0,1) (1,0) → M[0,1]+=1, M[1,0]+=1
    # Cross block 0->1: dropped
    # Within block 1: (1,0) (0,1) → M[1,0]+=1, M[0,1]+=1
    assert M[0, 0] == 0
    assert M[1, 1] == 0
    assert M[0, 1] == 2
    assert M[1, 0] == 2


def test_label_transition_count_drops_invalid_labels():
    labels = np.array([0, -1, 0, 1])
    block_ids = np.array([0, 0, 0, 0])
    M = _build_label_transition_count(labels, block_ids, n_clusters=2)
    # Only the (0,1) pair at the end is valid (others involve -1)
    assert M[0, 1] == 1
    assert M.sum() == 1


def test_two_state_lambda2_perfect_metastable():
    # Long dwell in each state, never switch
    M_count = np.array([[100, 0], [0, 100]])
    out = _two_state_lambda2_from_count(M_count)
    assert out["lambda_2"] == pytest.approx(1.0)
    assert out["stay_fracs"] == pytest.approx([1.0, 1.0])


def test_two_state_lambda2_random():
    # Equal stay/switch (chance-level for π = [0.5, 0.5])
    M_count = np.array([[50, 50], [50, 50]])
    out = _two_state_lambda2_from_count(M_count)
    assert out["lambda_2"] == pytest.approx(0.0)


def test_two_state_lambda2_anti_correlated():
    # Always switches (oscillation)
    M_count = np.array([[0, 100], [100, 0]])
    out = _two_state_lambda2_from_count(M_count)
    assert out["lambda_2"] == pytest.approx(-1.0)


def test_pr2_label_transition_sanity_metastable_signal():
    """Synthetic metastable: long runs of same label → λ_2 obs >> null."""
    rng = np.random.default_rng(123)
    n_blocks = 20
    block_size = 200
    labels_list = []
    block_ids_list = []
    for b in range(n_blocks):
        # Each block is 50/50 but generated as 4 long runs
        runs = rng.permutation([0, 1, 0, 1])
        run_len = block_size // 4
        block_labels = np.concatenate([np.full(run_len, r) for r in runs])
        labels_list.append(block_labels)
        block_ids_list.append(np.full(block_size, b))
    labels = np.concatenate(labels_list)
    block_ids = np.concatenate(block_ids_list)

    out = compute_pr2_label_transition_sanity(
        labels, block_ids, n_clusters=2, n_perm=200, rng_seed=0
    )
    obs_lam2 = out["obs"]["lambda_2"]
    assert obs_lam2 > 0.9
    assert out["z_lambda_2"] > 5
    assert out["p_empirical"] < 0.01


def test_pr2_label_transition_sanity_random_signal():
    """Synthetic IID labels: λ_2 obs ≈ null mean."""
    rng = np.random.default_rng(456)
    n_blocks = 20
    block_size = 200
    labels_list = []
    block_ids_list = []
    for b in range(n_blocks):
        labels_list.append(rng.integers(0, 2, size=block_size))
        block_ids_list.append(np.full(block_size, b))
    labels = np.concatenate(labels_list)
    block_ids = np.concatenate(block_ids_list)

    out = compute_pr2_label_transition_sanity(
        labels, block_ids, n_clusters=2, n_perm=200, rng_seed=0
    )
    # IID is the null distribution → z near 0, p near 0.5
    assert abs(out["z_lambda_2"]) < 3
    assert 0.05 < out["p_empirical"] < 0.95


def test_run_step1_subject_excludes_on_zero_label_overlap():
    """If PR-2 labels exist but cover no Topic 4 eligible events, subject is excluded."""
    rng = np.random.default_rng(46)
    ranks, bools, labels = _synth_two_cluster_ranks(rng)
    # PR-2 valid_events points at non-overlapping indices
    valid_events = np.arange(ranks.shape[1], ranks.shape[1] + len(labels))
    out = run_step1_subject(
        ranks, bools,
        pr2_valid_events=valid_events, pr2_labels=labels, pr2_n_clusters=2,
        min_participating=6,
    )
    assert "excluded_from_h3_main" in out
    assert "insufficient_pr2_label_overlap" in out["excluded_from_h3_main"]


def test_run_step1_subject_pr2_label_subset_alignment():
    """When PR-2 labels cover events with n_part >= 3 but Topic 4 only takes
    n_part >= 6, the centroid recompute should use the intersection."""
    rng = np.random.default_rng(44)
    n_chan = 10
    n_event = 600
    base = np.arange(1, n_chan + 1, dtype=float)
    base_rev = base[::-1].copy()

    ranks_list = []
    bools_list = []
    label_list = []
    for i in range(n_event):
        cid = i % 2
        b = base if cid == 0 else base_rev
        jitter = b + rng.normal(scale=0.4, size=n_chan)
        r = (np.argsort(np.argsort(jitter)) + 1).astype(float)
        # Half the events have only 4 active channels (below Topic 4 gate)
        bool_vec = np.ones(n_chan, dtype=bool)
        if i % 2 == 0 and i < 200:
            bool_vec[6:] = False  # n_part = 6 still; keep eligible
        if i >= 400:
            bool_vec[5:] = False  # n_part = 5, below gate
        # Keep ranks NaN for inactive
        r = np.where(bool_vec, r, np.nan)
        ranks_list.append(r)
        bools_list.append(bool_vec)
        label_list.append(cid)

    ranks = np.array(ranks_list).T
    bools = np.array(bools_list).T
    labels = np.array(label_list)
    valid_events = np.arange(n_event)  # PR-2 takes all (default min_part=3)

    out = run_step1_subject(
        ranks, bools,
        pr2_valid_events=valid_events, pr2_labels=labels, pr2_n_clusters=2,
        min_participating=6,
    )
    # First 400 events have n_part >= 6 (eligible); last 200 have n_part=5 (excluded)
    assert out["n_events_eligible"] == 400
    assert out["centroid_source"] == "pr2_labels_recomputed_in_X"
    assert sum(out["n_in_cluster"]) == 400
