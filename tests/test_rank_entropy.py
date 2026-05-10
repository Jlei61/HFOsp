"""TDD tests for PR-6-sup1 — first-rank entropy / symmetry-breaking diagnostic.

Plan-of-record (v3, 2026-05-10):
    docs/archive/topic1/pr6_template_anchoring/
        pr6_supplementary_rank_entropy_plan_2026-05-10.md

Coverage map (Tasks 1-4 of plan §11):
- Task 1: rank position entropy primitive (uniform / deterministic / raise)
- Task 2: endpoint vs middle delta (confluence toy / uniform / asymmetry)
- Task 3: N0 + N1 cluster + N1 subject nulls
    * N0 zero-mean delta on shuffled data
    * N1 cluster schema (8 fields)
    * N1 cluster confluence curve → endpoint_pair = max
    * N1 cluster uniform curve → p_N1 = 1.0
    * N1 cluster floor contract: n_valid=6 → min_attainable_p_N1 = 1/15
    * N1 subject joint enumeration
    * raise contracts on n_valid<4 / wrong cluster count
- Task 4: per-subject pipeline + participation filter
    * partial-participation events dropped
    * low-kept-events triggers excluded flag
    * high-drop-rate triggers warning (not exclude)
    * synthetic stable_k=2 integration
"""

from __future__ import annotations

import numpy as np
import pytest

from src.rank_displacement import (
    compute_endpoint_middle_entropy_delta,
    compute_rank_position_entropy,
    rank_entropy_null_N0,
    rank_entropy_null_N1_pseudo_endpoint,
    rank_entropy_null_N1_subject_level,
    run_subject_rank_entropy,
)


# =====================================================================
# Task 1 — compute_rank_position_entropy
# =====================================================================


def _random_rank_matrix(n_events: int, n_valid: int, rng_seed: int) -> np.ndarray:
    """Build (n_events, n_valid) integer rank matrix where each row is a
    uniformly-random permutation of {1, ..., n_valid}."""
    rng = np.random.default_rng(rng_seed)
    R = np.empty((n_events, n_valid), dtype=int)
    for e in range(n_events):
        R[e] = rng.permutation(n_valid) + 1
    return R


def test_rank_position_entropy_uniform_baseline():
    """Random rank vectors → H_p_norm ≈ 1.0 (cap on full entropy)."""
    R = _random_rank_matrix(n_events=10000, n_valid=10, rng_seed=0)
    H_p_norm = compute_rank_position_entropy(R, n_valid=10)
    assert H_p_norm.shape == (10,)
    assert np.all(H_p_norm > 0.95), H_p_norm
    assert np.all(H_p_norm <= 1.0 + 1e-9), H_p_norm


def test_rank_position_entropy_deterministic_baseline():
    """Every event has the same rank → H_p_norm = 0 ∀p."""
    deterministic = np.tile(np.arange(1, 11), (200, 1))
    H_p_norm = compute_rank_position_entropy(deterministic, n_valid=10)
    assert H_p_norm.shape == (10,)
    assert np.allclose(H_p_norm, 0.0, atol=1e-12), H_p_norm


def test_rank_position_entropy_raises_on_n_valid_below_4():
    R = _random_rank_matrix(n_events=10, n_valid=3, rng_seed=0)
    with pytest.raises(ValueError, match=r"n_valid"):
        compute_rank_position_entropy(R, n_valid=3)


def test_rank_position_entropy_raises_on_empty_matrix():
    R = np.empty((0, 10), dtype=int)
    with pytest.raises(ValueError, match=r"n_events"):
        compute_rank_position_entropy(R, n_valid=10)


# =====================================================================
# Task 2 — compute_endpoint_middle_entropy_delta
# =====================================================================


def test_endpoint_middle_delta_confluence_toy():
    """H_p_norm = high at endpoints, low in middle → Δ > 0."""
    H_p_norm = np.array([0.95, 0.10, 0.10, 0.10, 0.10, 0.95])
    delta, asymmetry = compute_endpoint_middle_entropy_delta(H_p_norm)
    assert delta > 0.5
    assert abs(asymmetry) < 1e-12  # H_1 == H_n


def test_endpoint_middle_delta_uniform_yields_zero():
    H_p_norm = np.full(10, 0.7)
    delta, asymmetry = compute_endpoint_middle_entropy_delta(H_p_norm)
    assert abs(delta) < 1e-12
    assert abs(asymmetry) < 1e-12


def test_endpoint_middle_delta_asymmetry_picks_up_source_only():
    """H_1 high, H_n low → asymmetry > 0 (source jitter, sink locked)."""
    H_p_norm = np.array([0.95, 0.10, 0.10, 0.10, 0.10, 0.10])
    delta, asymmetry = compute_endpoint_middle_entropy_delta(H_p_norm)
    assert asymmetry > 0.5


# =====================================================================
# Task 3 — Surrogate construction (N0 + N1 cluster + N1 subject)
# =====================================================================


def test_null_N0_returns_zero_mean_delta():
    """Per-event rank shuffle null → Δ_null mean ≈ 0."""
    R = _random_rank_matrix(n_events=2000, n_valid=10, rng_seed=42)
    deltas_null = rank_entropy_null_N0(R, n_valid=10, n_perm=200, base_seed=0)
    assert deltas_null.shape == (200,)
    assert abs(np.mean(deltas_null)) < 0.05


def test_null_N0_is_deterministic_under_same_seed():
    R = _random_rank_matrix(n_events=500, n_valid=10, rng_seed=7)
    a = rank_entropy_null_N0(R, n_valid=10, n_perm=50, base_seed=0)
    b = rank_entropy_null_N0(R, n_valid=10, n_perm=50, base_seed=0)
    assert np.allclose(a, b)


def test_null_N0_n_valid_below_4_raises():
    R = _random_rank_matrix(n_events=10, n_valid=3, rng_seed=0)
    with pytest.raises(ValueError, match=r"n_valid"):
        rank_entropy_null_N0(R, n_valid=3, n_perm=10, base_seed=0)


def test_null_N1_cluster_returns_full_schema():
    """All v3 fields present for a normal H_p_norm input."""
    H_p_norm = np.array([0.9, 0.6, 0.5, 0.5, 0.4, 0.4, 0.5, 0.5, 0.6, 0.9])
    out = rank_entropy_null_N1_pseudo_endpoint(H_p_norm)
    expected = {
        "delta_obs", "delta_pair_dist", "endpoint_pair_rank",
        "endpoint_pair_percentile", "is_endpoint_pair_max", "p_N1",
        "min_attainable_p_N1", "n_valid",
    }
    assert expected.issubset(out.keys())
    # n_valid=10 → C(10, 2) = 45
    assert out["n_valid"] == 10
    assert len(out["delta_pair_dist"]) == 45
    assert out["min_attainable_p_N1"] == 1.0 / 45


def test_null_N1_cluster_endpoint_special_under_confluence():
    """High endpoints + low middle → endpoint_pair is max, percentile=1.0."""
    H_p_norm = np.array([0.95, 0.10, 0.10, 0.10, 0.95])  # n_valid=5 → C(5,2)=10
    out = rank_entropy_null_N1_pseudo_endpoint(H_p_norm)
    assert out["is_endpoint_pair_max"] is True
    assert out["endpoint_pair_rank"] == 1
    assert out["endpoint_pair_percentile"] == pytest.approx(1.0)
    assert out["p_N1"] == pytest.approx(1.0 / 10)


def test_null_N1_cluster_uniform_curve_yields_p1():
    """Uniform H_p_norm → all Δ_pair = 0 → ties → p_N1 = 1.0."""
    H_p_norm = np.full(8, 0.5)
    out = rank_entropy_null_N1_pseudo_endpoint(H_p_norm)
    assert out["p_N1"] == pytest.approx(1.0)
    assert out["is_endpoint_pair_max"] is True  # tie includes endpoint


def test_null_N1_cluster_min_attainable_p_floor_at_n_valid_6():
    """Contract test: n_valid=6 → min_attainable_p_N1 = 1/15 ≈ 0.0667.
    Locks the floor higher than the conventional 0.05 threshold so future
    reviewers can't quietly reintroduce a `p_N1 < 0.05` gate that
    mechanically excludes n_valid=6 subjects."""
    H_p_norm = np.array([0.9, 0.5, 0.5, 0.5, 0.5, 0.9])
    out = rank_entropy_null_N1_pseudo_endpoint(H_p_norm)
    assert out["n_valid"] == 6
    assert out["min_attainable_p_N1"] == pytest.approx(1.0 / 15)
    assert out["min_attainable_p_N1"] > 0.05  # the whole point


def test_null_N1_cluster_n_valid_below_4_raises():
    H_p_norm = np.array([0.5, 0.5, 0.5])  # n_valid=3
    with pytest.raises(ValueError, match=r"n_valid"):
        rank_entropy_null_N1_pseudo_endpoint(H_p_norm)


def test_null_N1_subject_joint_enumeration_basic():
    """Two clusters with n_valid=[6, 6] → n_combos = 15 × 15 = 225."""
    confluence = np.array([0.9, 0.3, 0.3, 0.3, 0.3, 0.9])
    uniform = np.full(6, 0.5)
    out = rank_entropy_null_N1_subject_level([confluence, uniform])
    assert out["n_combos"] == 225
    assert out["min_attainable_p_N1_subject"] == pytest.approx(1.0 / 225)
    # Subject combo: endpoint_obs_subject = mean(Δ_obs_confluence, 0)
    # Confluence Δ_(1,6) = 0.6; uniform Δ = 0
    # Δ_obs_subject = 0.3
    assert out["delta_obs_subject"] == pytest.approx(0.3)


def test_null_N1_subject_floor_below_005_at_n_valid_6():
    """Contract test: subject-level Option B has min_attainable_p ≈ 0.004
    at n_valid=6 double-cluster, no longer floor-binding."""
    H_p_a = np.array([0.9, 0.3, 0.3, 0.3, 0.3, 0.9])
    H_p_b = np.array([0.9, 0.3, 0.3, 0.3, 0.3, 0.9])
    out = rank_entropy_null_N1_subject_level([H_p_a, H_p_b])
    assert out["min_attainable_p_N1_subject"] < 0.05  # the whole point


def test_null_N1_subject_wrong_cluster_count_raises():
    H_p = np.array([0.9, 0.3, 0.3, 0.3, 0.3, 0.9])
    with pytest.raises(ValueError, match=r"clusters?"):
        rank_entropy_null_N1_subject_level([H_p])  # only 1 cluster
    with pytest.raises(ValueError, match=r"clusters?"):
        rank_entropy_null_N1_subject_level([H_p, H_p, H_p])  # 3 clusters


# =====================================================================
# Task 4 — per-subject pipeline + participation filter
# =====================================================================


def _make_subject(
    n_events_per_cluster_full: int,
    n_events_per_cluster_partial: int,
    n_valid: int,
    cluster_a_pattern: str,  # 'confluence' | 'uniform' | 'random'
    cluster_b_pattern: str,
    rng_seed: int = 0,
) -> dict:
    """Build subject_data dict with two clusters; returns the input shape that
    run_subject_rank_entropy expects (per plan §10.2 / §6.0).

    cluster_a / cluster_b pattern controls the rank generation:
      'random'     → each event a random permutation
      'confluence' → endpoints (rank 1, n_valid) jitter random, middle locked
      'uniform'    → fixed permutation (deterministic)
    """
    rng = np.random.default_rng(rng_seed)
    n_clusters = 2

    def _gen_event(pattern: str) -> np.ndarray:
        if pattern == "random":
            return (rng.permutation(n_valid) + 1).astype(int)
        if pattern == "uniform":
            return np.arange(1, n_valid + 1, dtype=int)
        if pattern == "confluence":
            # Middle stays in original order; endpoints (channels 0 and n_valid-1)
            # get assigned a random extreme rank position pair.
            base = np.arange(1, n_valid + 1, dtype=int)
            # Pick one of two extreme channels (0 or n_valid-1) at random for
            # rank=1; the other goes to rank=n_valid.
            if rng.random() < 0.5:
                base[0], base[n_valid - 1] = 1, n_valid
            else:
                base[0], base[n_valid - 1] = n_valid, 1
            return base
        raise ValueError(pattern)

    n_events_total = 2 * (n_events_per_cluster_full + n_events_per_cluster_partial)
    ranks = np.zeros((n_valid, n_events_total), dtype=int)
    bools = np.ones((n_valid, n_events_total), dtype=bool)
    labels = np.zeros(n_events_total, dtype=int)

    cursor = 0
    for cluster_idx, pattern in enumerate([cluster_a_pattern, cluster_b_pattern]):
        for _ in range(n_events_per_cluster_full):
            ranks[:, cursor] = _gen_event(pattern)
            labels[cursor] = cluster_idx
            cursor += 1
        for _ in range(n_events_per_cluster_partial):
            ranks[:, cursor] = _gen_event(pattern)
            labels[cursor] = cluster_idx
            # Drop one channel from this event: random non-endpoint channel
            drop_ch = rng.integers(1, n_valid - 1)
            bools[drop_ch, cursor] = False
            cursor += 1

    return {
        "ranks": ranks,
        "bools": bools,
        "labels": labels,
        "valid_mask": np.ones(n_valid, dtype=bool),
        "channel_names": [f"ch{i:02d}" for i in range(n_valid)],
        "n_clusters": n_clusters,
    }


def test_partial_participation_events_are_dropped():
    """1000 fully-participating + 500 partial → kept 1000, drop_rate 1/3."""
    subj = _make_subject(
        n_events_per_cluster_full=1000,
        n_events_per_cluster_partial=500,
        n_valid=10,
        cluster_a_pattern="confluence",
        cluster_b_pattern="confluence",
        rng_seed=0,
    )
    out = run_subject_rank_entropy(subj)
    cluster_0 = out["clusters"]["0"]
    assert cluster_0["n_events_total_k"] == 1500
    assert cluster_0["n_events_kept_k"] == 1000
    assert cluster_0["drop_rate_k"] == pytest.approx(500.0 / 1500)
    assert cluster_0["eligibility_flag"] in {"ok", "high_drop_rate_warning"}


def test_low_kept_events_triggers_excluded_flag():
    """30 fully-participating events < 50 threshold → excluded_low_kept_events."""
    subj = _make_subject(
        n_events_per_cluster_full=30,
        n_events_per_cluster_partial=0,
        n_valid=10,
        cluster_a_pattern="confluence",
        cluster_b_pattern="confluence",
        rng_seed=0,
    )
    out = run_subject_rank_entropy(subj)
    assert out["clusters"]["0"]["eligibility_flag"] == "excluded_low_kept_events"


def test_high_drop_rate_triggers_warning_not_exclude():
    """100 full + 200 partial → drop_rate 200/300 = 0.667 > 0.5 → warning."""
    subj = _make_subject(
        n_events_per_cluster_full=100,
        n_events_per_cluster_partial=200,
        n_valid=10,
        cluster_a_pattern="confluence",
        cluster_b_pattern="confluence",
        rng_seed=0,
    )
    out = run_subject_rank_entropy(subj)
    cluster_0 = out["clusters"]["0"]
    assert cluster_0["eligibility_flag"] == "high_drop_rate_warning"
    assert cluster_0["n_events_kept_k"] == 100
    assert cluster_0["drop_rate_k"] == pytest.approx(200.0 / 300)


def test_per_subject_pipeline_on_confluence_synthetic():
    """Cluster A confluence (high Δ), cluster B uniform (Δ ≈ 0)."""
    subj = _make_subject(
        n_events_per_cluster_full=1000,
        n_events_per_cluster_partial=0,
        n_valid=10,
        cluster_a_pattern="confluence",
        cluster_b_pattern="uniform",
        rng_seed=0,
    )
    out = run_subject_rank_entropy(subj)
    cluster_0 = out["clusters"]["0"]
    cluster_1 = out["clusters"]["1"]
    # Cluster A confluence → delta high
    assert cluster_0["delta"] > 0.3
    # Cluster B uniform (every event identical permutation) → H_p = 0 →
    # delta = 0 - 0 = 0
    assert abs(cluster_1["delta"]) < 1e-9
    # Subject-level
    assert "subject" in out
    assert "delta_obs_subject" in out["subject"]
    # Mean(0.x, 0) ≈ delta_a / 2
    assert out["subject"]["delta_obs_subject"] == pytest.approx(
        (cluster_0["delta"] + cluster_1["delta"]) / 2, abs=1e-9
    )
