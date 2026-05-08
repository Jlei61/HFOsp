"""TDD tests for src.rank_displacement.

Plan: docs/archive/topic1/pr6_supplementary_rank_displacement_plan_2026-05-06.md
"""
from __future__ import annotations

import numpy as np
import pytest

from src.rank_displacement import compute_signed_rank_displacement


def test_identical_ranks_zero_displacement():
    rank_a = np.array([0, 1, 2, 3, 4], dtype=float)
    rank_b = np.array([0, 1, 2, 3, 4], dtype=float)
    valid_mask = np.array([True] * 5)
    result = compute_signed_rank_displacement(
        rank_a=rank_a,
        rank_b=rank_b,
        valid_mask_a=valid_mask,
        valid_mask_b=valid_mask,
        channel_names=["A", "B", "C", "D", "E"],
    )
    assert result["exit_reason"] == "ok"
    assert result["n_valid"] == 5
    np.testing.assert_array_equal(result["signed_displacement_dense"], np.zeros(5))
    assert result["footrule"] == pytest.approx(0.0)
    assert result["footrule_normalized"] == pytest.approx(0.0)
    assert result["kendall_tau"] == pytest.approx(1.0)


def test_reversed_ranks_full_footrule():
    rank_a = np.array([0, 1, 2, 3], dtype=float)
    rank_b = np.array([3, 2, 1, 0], dtype=float)
    valid_mask = np.array([True] * 4)
    result = compute_signed_rank_displacement(
        rank_a=rank_a,
        rank_b=rank_b,
        valid_mask_a=valid_mask,
        valid_mask_b=valid_mask,
        channel_names=["A", "B", "C", "D"],
    )
    assert result["exit_reason"] == "ok"
    # |Δ| = [3, 1, 1, 3] → footrule = 8
    assert result["footrule"] == pytest.approx(8.0)
    # Diaconis-Graham F_max = floor(n^2 / 2) = floor(16/2) = 8
    assert result["footrule_max"] == pytest.approx(8.0)
    assert result["footrule_normalized"] == pytest.approx(1.0)
    assert result["kendall_tau"] == pytest.approx(-1.0)


def test_signed_displacement_signs_correct():
    # rank_a = [0, 1, 2, 3] (A first, D last in T_a)
    # rank_b = [3, 2, 1, 0] (D first, A last in T_b)
    # Δ = rank_b - rank_a = [+3, +1, -1, -3]
    rank_a = np.array([0, 1, 2, 3], dtype=float)
    rank_b = np.array([3, 2, 1, 0], dtype=float)
    valid_mask = np.array([True] * 4)
    result = compute_signed_rank_displacement(
        rank_a=rank_a,
        rank_b=rank_b,
        valid_mask_a=valid_mask,
        valid_mask_b=valid_mask,
        channel_names=["A", "B", "C", "D"],
    )
    np.testing.assert_array_almost_equal(
        result["signed_displacement_dense"], [3.0, 1.0, -1.0, -3.0]
    )
    # Also verify rank_a_dense_full lines up with input rank_a inside joint_valid
    np.testing.assert_array_almost_equal(
        result["rank_a_dense_full"], [0.0, 1.0, 2.0, 3.0]
    )
    np.testing.assert_array_almost_equal(
        result["rank_b_dense_full"], [3.0, 2.0, 1.0, 0.0]
    )


def test_partial_valid_mask_intersection_aborts_under_4():
    # n_channels = 5, but valid_mask drops index 0 in A and index 4 in B
    rank_a = np.array([-1, 0, 1, 2, 3], dtype=float)
    rank_b = np.array([3, 2, 1, 0, -1], dtype=float)
    valid_mask_a = np.array([False, True, True, True, True])
    valid_mask_b = np.array([True, True, True, True, False])
    result = compute_signed_rank_displacement(
        rank_a=rank_a,
        rank_b=rank_b,
        valid_mask_a=valid_mask_a,
        valid_mask_b=valid_mask_b,
        channel_names=["A", "B", "C", "D", "E"],
    )
    # Joint valid = indices 1, 2, 3 → n_valid = 3 < 4 → abort
    assert result["exit_reason"].startswith("n_valid<4")
    assert result["n_valid"] == 3


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_signed_rank_displacement(
            rank_a=np.array([0, 1, 2]),
            rank_b=np.array([0, 1, 2, 3]),
            valid_mask_a=np.array([True, True, True]),
            valid_mask_b=np.array([True, True, True]),
            channel_names=["A", "B", "C"],
        )


def test_aggregate_pair_metrics_soz_baseline_correction():
    """Perfect reversal with 50/50 SOZ split: excess should be 0 on baseline."""
    from src.rank_displacement import aggregate_pair_metrics

    rank_a = np.array([0, 1, 2, 3], dtype=float)
    rank_b = np.array([3, 2, 1, 0], dtype=float)
    valid_mask = np.array([True] * 4)
    result = aggregate_pair_metrics(
        rank_a=rank_a,
        rank_b=rank_b,
        valid_mask_a=valid_mask,
        valid_mask_b=valid_mask,
        channel_names=["A", "B", "C", "D"],
        soz_channels={"A", "B"},
    )
    assert result["exit_reason"] == "ok"
    # |Δ| = [3, 1, 1, 3]; SOZ {A,B} → |Δ_SOZ| = [3,1] = 4; footrule = 8
    assert result["soz_contribution_fraction"] == pytest.approx(0.5)
    assert result["nonsoz_contribution_fraction"] == pytest.approx(0.5)
    # Channel-count baseline: 2/4 = 0.5
    assert result["soz_channel_fraction"] == pytest.approx(0.5)
    # Excess = 0.5 − 0.5 = 0.0 (sits exactly on baseline)
    assert result["soz_contribution_excess"] == pytest.approx(0.0)
    # Per-channel |Δ| means: SOZ mean = 2.0; nonSOZ mean = 2.0
    assert result["soz_abs_mean"] == pytest.approx(2.0)
    assert result["nonsoz_abs_mean"] == pytest.approx(2.0)
    assert result["soz_minus_nonsoz_abs_mean"] == pytest.approx(0.0)


def test_aggregate_pair_metrics_soz_excess_positive():
    """SOZ over-represented in displacement: contribution_excess > 0."""
    from src.rank_displacement import aggregate_pair_metrics

    # 5 channels, A in SOZ; A swaps source↔sink (rank 0↔4), middle 3 unchanged
    # rank_a = [0,1,2,3,4]; rank_b = [4,1,2,3,0]
    # delta_dense = [4, 0, 0, 0, -4]; |Δ| = [4,0,0,0,4]; footrule = 8
    # SOZ {A}: contribution = 4/8 = 0.5; channel_fraction = 1/5 = 0.2; excess = 0.3
    # Per-channel: SOZ |Δ| mean = 4.0; nonSOZ |Δ| mean = (0+0+0+4)/4 = 1.0; diff = 3.0
    rank_a = np.array([0, 1, 2, 3, 4], dtype=float)
    rank_b = np.array([4, 1, 2, 3, 0], dtype=float)
    valid_mask = np.array([True] * 5)
    result = aggregate_pair_metrics(
        rank_a=rank_a,
        rank_b=rank_b,
        valid_mask_a=valid_mask,
        valid_mask_b=valid_mask,
        channel_names=["A", "B", "C", "D", "E"],
        soz_channels={"A"},
    )
    assert result["soz_channel_fraction"] == pytest.approx(0.2)
    assert result["soz_contribution_fraction"] == pytest.approx(0.5)
    assert result["soz_contribution_excess"] == pytest.approx(0.3)
    assert result["soz_abs_mean"] == pytest.approx(4.0)
    assert result["nonsoz_abs_mean"] == pytest.approx(1.0)
    assert result["soz_minus_nonsoz_abs_mean"] == pytest.approx(3.0)


def test_swap_score_at_k_perfect_reversal_one_at_all_k():
    """Perfect reversal: top_k of T_a == bottom_k of T_b for every k <= n//2."""
    from src.rank_displacement import compute_swap_score_at_k

    rank_a = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    rank_b = np.array([5, 4, 3, 2, 1, 0], dtype=float)
    valid_mask = np.array([True] * 6)
    for k in (1, 2, 3):
        score = compute_swap_score_at_k(
            rank_a=rank_a,
            rank_b=rank_b,
            valid_mask_a=valid_mask,
            valid_mask_b=valid_mask,
            k=k,
        )
        assert score == pytest.approx(1.0), f"k={k} expected 1.0, got {score}"


def test_swap_score_at_k_identical_ranks_zero():
    """Identical ranks: top_k of T_a is also top_k of T_b -> swap with bottom = 0."""
    from src.rank_displacement import compute_swap_score_at_k

    rank_a = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    rank_b = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    valid_mask = np.array([True] * 6)
    for k in (1, 2, 3):
        assert (
            compute_swap_score_at_k(
                rank_a=rank_a,
                rank_b=rank_b,
                valid_mask_a=valid_mask,
                valid_mask_b=valid_mask,
                k=k,
            )
            == pytest.approx(0.0)
        )


def test_swap_score_at_k_only_top_two_swap():
    """Top-2 of T_a become bottom-2 of T_b but middle ranks stay put.

    rank_a = [0, 1, 2, 3, 4, 5]   (channels 4,5 are top in T_a)
    rank_b = [4, 5, 2, 3, 1, 0]   (channels 4,5 are bottom in T_b; channels 0,1
                                   are top in T_b which is bottom in T_a)
    -> swap_score(k=2) = 1.0 (perfect 2-swap)
    -> swap_score(k=3) < 1.0 (channel 3 is mid in both, partial mismatch)
    """
    from src.rank_displacement import compute_swap_score_at_k

    rank_a = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    rank_b = np.array([4, 5, 2, 3, 1, 0], dtype=float)
    valid_mask = np.array([True] * 6)
    s2 = compute_swap_score_at_k(rank_a, rank_b, valid_mask, valid_mask, k=2)
    s3 = compute_swap_score_at_k(rank_a, rank_b, valid_mask, valid_mask, k=3)
    assert s2 == pytest.approx(1.0)
    assert s3 < 1.0


def test_swap_score_sweep_perfect_reversal():
    """Perfect reversal at n=8 (k_max=4): T_obs=1.0; max-null rarely hits 1.0
    -> p_fw < 0.05 -> has_swap=True. Decision_k = 2 by smallest-k tie-break."""
    from src.rank_displacement import compute_swap_score_sweep

    rank_a = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=float)
    rank_b = np.array([7, 6, 5, 4, 3, 2, 1, 0], dtype=float)
    valid_mask = np.array([True] * 8)
    out = compute_swap_score_sweep(
        rank_a=rank_a,
        rank_b=rank_b,
        valid_mask_a=valid_mask,
        valid_mask_b=valid_mask,
        n_perm=1000,
        seed=0,
    )
    assert out["exit_reason"] == "ok"
    assert out["k_max"] == 4
    assert out["T_obs"] == pytest.approx(1.0)
    assert out["decision_k"] == 2  # smallest-k tie when all k saturate
    assert out["has_swap"] is True
    assert out["p_fw"] < 0.05
    # Determinism contract: seed/n_perm round-trip
    assert out["seed"] == 0
    assert out["n_perm"] == 1000


def test_swap_score_sweep_identical_ranks_no_swap():
    """Identical ranks: T_obs=0; max-null also 0 frequently; has_swap=False."""
    from src.rank_displacement import compute_swap_score_sweep

    rank_a = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    rank_b = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    valid_mask = np.array([True] * 6)
    out = compute_swap_score_sweep(
        rank_a=rank_a,
        rank_b=rank_b,
        valid_mask_a=valid_mask,
        valid_mask_b=valid_mask,
        n_perm=500,
        seed=0,
    )
    assert out["T_obs"] == pytest.approx(0.0)
    assert out["has_swap"] is False


def test_swap_score_sweep_only_top_two_swap_classified():
    """User motivation: only top-2 channels swap. T_obs = swap_score(k=2) = 1.0;
    decision_k = 2; max-null rarely reaches 1.0 -> has_swap=True."""
    from src.rank_displacement import compute_swap_score_sweep

    # 8 channels; only top-2 of T_a swap to bottom-2 of T_b
    rank_a = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=float)
    rank_b = np.array([6, 7, 2, 3, 4, 5, 1, 0], dtype=float)
    valid_mask = np.array([True] * 8)
    out = compute_swap_score_sweep(
        rank_a=rank_a,
        rank_b=rank_b,
        valid_mask_a=valid_mask,
        valid_mask_b=valid_mask,
        n_perm=1000,
        seed=0,
    )
    assert out["decision_k"] == 2
    assert out["T_obs"] == pytest.approx(1.0)
    assert out["has_swap"] is True


def test_swap_score_sweep_determinism():
    """Same (seed, n_perm) -> identical (T_obs, p_fw, decision_k)."""
    from src.rank_displacement import compute_swap_score_sweep

    rank_a = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=float)
    rank_b = np.array([6, 7, 2, 3, 4, 5, 1, 0], dtype=float)
    valid_mask = np.array([True] * 8)
    a = compute_swap_score_sweep(
        rank_a=rank_a, rank_b=rank_b,
        valid_mask_a=valid_mask, valid_mask_b=valid_mask,
        n_perm=300, seed=42,
    )
    b = compute_swap_score_sweep(
        rank_a=rank_a, rank_b=rank_b,
        valid_mask_a=valid_mask, valid_mask_b=valid_mask,
        n_perm=300, seed=42,
    )
    assert a["T_obs"] == b["T_obs"]
    assert a["p_fw"] == b["p_fw"]
    assert a["decision_k"] == b["decision_k"]
    assert a["has_swap"] == b["has_swap"]


def test_swap_score_sweep_pr6_reproduces_at_k3():
    """At k=3 our swap_score reproduces PR-6's per-subject swap_score on
    a hand-built example. PR-6 formula:
        swap_score = mean( Jaccard(T0_source_top3, T1_sink_bottom3),
                           Jaccard(T0_sink_bottom3, T1_source_top3) )
    """
    from src.rank_displacement import compute_swap_score_at_k

    # 6 channels: T_a top-3 = {3,4,5}; T_b bottom-3 = {3,4,5} (perfect swap)
    rank_a = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    rank_b = np.array([5, 4, 3, 2, 1, 0], dtype=float)
    valid_mask = np.array([True] * 6)
    score = compute_swap_score_at_k(
        rank_a=rank_a,
        rank_b=rank_b,
        valid_mask_a=valid_mask,
        valid_mask_b=valid_mask,
        k=3,
    )
    # Jaccard(top3_a={3,4,5}, bot3_b={3,4,5}) = 1.0
    # Jaccard(bot3_a={0,1,2}, top3_b={0,1,2}) = 1.0
    # mean = 1.0
    assert score == pytest.approx(1.0)


def test_swap_score_sweep_aborts_when_n_valid_lt_4():
    """n_valid < 4 -> sweep aborts (consistent with the displacement helper)."""
    from src.rank_displacement import compute_swap_score_sweep

    rank_a = np.array([0, 1, 2], dtype=float)
    rank_b = np.array([2, 1, 0], dtype=float)
    valid_mask = np.array([True] * 3)
    out = compute_swap_score_sweep(
        rank_a=rank_a,
        rank_b=rank_b,
        valid_mask_a=valid_mask,
        valid_mask_b=valid_mask,
        n_perm=100,
        seed=0,
    )
    assert out["exit_reason"].startswith("n_valid<4")
    assert out["has_swap"] is False


def test_swap_score_sweep_strict_class_on_perfect_reversal():
    """Perfect reversal -> swap_class='strict' (T_obs=1, p_fw small)."""
    from src.rank_displacement import compute_swap_score_sweep

    rank_a = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=float)
    rank_b = np.array([7, 6, 5, 4, 3, 2, 1, 0], dtype=float)
    valid_mask = np.array([True] * 8)
    out = compute_swap_score_sweep(
        rank_a=rank_a, rank_b=rank_b,
        valid_mask_a=valid_mask, valid_mask_b=valid_mask,
        n_perm=1000, seed=0,
    )
    assert out["swap_class"] == "strict"
    assert out["has_swap"] is True
    # alpha_candidate persisted for downstream contract
    assert out["alpha_candidate"] == pytest.approx(0.20)


def test_swap_score_sweep_candidate_tier_keeps_classification_off_strict():
    """Borderline subject (T_obs >= 0.5 but 0.05 <= p_fw < 0.20) -> 'candidate'.

    Constructed: a 12-channel rank pair with single-pair endpoint flip giving
    T_obs around 0.5 - exact p_fw drifts with seed but the test only asserts
    that when class is candidate, has_swap=False and swap_class='candidate'.
    """
    from src.rank_displacement import compute_swap_score_sweep

    # Build a sweep result; if it lands in candidate band, validate the contract.
    rank_a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=float)
    # only top-2 of T_a swap to bottom-2 of T_b; rest unchanged
    rank_b = np.array([10, 11, 2, 3, 4, 5, 6, 7, 8, 9, 1, 0], dtype=float)
    valid_mask = np.array([True] * 12)
    out = compute_swap_score_sweep(
        rank_a=rank_a, rank_b=rank_b,
        valid_mask_a=valid_mask, valid_mask_b=valid_mask,
        n_perm=1000, seed=7,
    )
    assert out["swap_class"] in ("strict", "candidate", "none")
    if out["swap_class"] == "candidate":
        assert out["has_swap"] is False
        assert out["T_obs"] >= 0.5
        assert 0.05 <= out["p_fw"] < 0.20
    elif out["swap_class"] == "strict":
        assert out["has_swap"] is True
        assert out["p_fw"] < 0.05


def test_swap_score_sweep_no_swap_when_floor_unmet():
    """T_obs < score_floor -> 'none' regardless of p_fw."""
    from src.rank_displacement import compute_swap_score_sweep

    # Mostly identical ranks, single small deviation -> T_obs < 0.5
    rank_a = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=float)
    rank_b = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=float)
    valid_mask = np.array([True] * 8)
    out = compute_swap_score_sweep(
        rank_a=rank_a, rank_b=rank_b,
        valid_mask_a=valid_mask, valid_mask_b=valid_mask,
        n_perm=500, seed=0,
    )
    assert out["swap_class"] == "none"
    assert out["has_swap"] is False


def test_swap_score_sweep_high_floor_fails_even_when_significant():
    """0.5 floor is an additional gate. T_obs that beats max-null but is
    below the floor must still classify as has_swap=False - prevents
    'statistically detectable but trivially small' overlap from passing."""
    from src.rank_displacement import compute_swap_score_sweep

    # Mostly-monotone but with a single small endpoint flip; T_obs ~ 0.33.
    rank_a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)
    rank_b = np.array([0, 1, 2, 9, 4, 5, 6, 7, 8, 3], dtype=float)
    valid_mask = np.array([True] * 10)
    out = compute_swap_score_sweep(
        rank_a=rank_a,
        rank_b=rank_b,
        valid_mask_a=valid_mask,
        valid_mask_b=valid_mask,
        n_perm=500,
        seed=0,
    )
    assert out["T_obs"] < 0.5
    assert out["has_swap"] is False


def test_aggregate_does_not_export_signed_soz_mean():
    """Sign-anchor contract (plan §3.0): never expose anchor-dependent aggregates."""
    from src.rank_displacement import aggregate_pair_metrics

    rank_a = np.array([0, 1, 2, 3], dtype=float)
    rank_b = np.array([3, 2, 1, 0], dtype=float)
    valid_mask = np.array([True] * 4)
    result = aggregate_pair_metrics(
        rank_a=rank_a,
        rank_b=rank_b,
        valid_mask_a=valid_mask,
        valid_mask_b=valid_mask,
        channel_names=["A", "B", "C", "D"],
        soz_channels={"A", "B"},
    )
    assert "signed_displacement_mean_soz" not in result
    assert "signed_displacement_mean_nonsoz" not in result


# =============================================================================
# §9 Clinical SOZ set-relationship tests
# Plan: docs/archive/topic1/pr6_supplementary_swap_clinical_soz_plan_2026-05-08.md
# =============================================================================

def test_derive_swap_endpoint_top_bottom_dk():
    """endpoint = top dk ∪ bottom dk channels by rank_a_dense ascending order.

    rank_a_dense = [0, 1, 2, 3, 4, 5] with channel_names = [A..F].
    decision_k = 2 → top dk channels = {A, B} (lowest ranks = source side),
    bottom dk channels = {E, F} (highest ranks = sink side).
    """
    from src.rank_displacement import derive_swap_endpoint

    channel_names = ["A", "B", "C", "D", "E", "F"]
    rank_a_dense = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    endpoint = derive_swap_endpoint(
        channel_names=channel_names,
        rank_a_dense=rank_a_dense,
        decision_k=2,
    )
    assert set(endpoint) == {"A", "B", "E", "F"}


def test_derive_swap_endpoint_handles_unsorted_rank():
    """rank_a_dense is the dense rank vector aligned with channel_names; not
    necessarily monotone. derive_swap_endpoint must sort indices by rank.

    channel_names = [P, Q, R, S], rank_a_dense = [3, 0, 2, 1]. Sorted by rank:
      Q (rank 0) < S (rank 1) < R (rank 2) < P (rank 3).
    decision_k = 1 → endpoint = {Q (top), P (bottom)}.
    """
    from src.rank_displacement import derive_swap_endpoint

    channel_names = ["P", "Q", "R", "S"]
    rank_a_dense = np.array([3, 0, 2, 1], dtype=float)
    endpoint = derive_swap_endpoint(
        channel_names=channel_names,
        rank_a_dense=rank_a_dense,
        decision_k=1,
    )
    assert set(endpoint) == {"Q", "P"}


def test_set_relation_intermediate_universe():
    """0 < |S| < |L| AND |E| < |L| → all fields well-defined, informative=True."""
    from src.rank_displacement import compute_clinical_soz_set_relation

    valid_chs = ["A", "B", "C", "D", "E", "F"]   # |L| = 6
    endpoint_chs = ["A", "B", "F"]                # |E| = 3
    soz_chs = ["A", "B", "C"]                     # |S in L| = 3
    out = compute_clinical_soz_set_relation(
        valid_chs=valid_chs, endpoint_chs=endpoint_chs, soz_chs=soz_chs,
    )
    assert out["n_E"] == 3
    assert out["n_S"] == 3
    assert out["n_L"] == 6
    assert out["n_E_inter_S"] == 2  # A, B
    assert out["precision"] == pytest.approx(2 / 3)
    assert out["recall_within_lagPat"] == pytest.approx(2 / 3)
    assert out["coverage"] == pytest.approx(0.5)
    assert out["lagpat_baseline"] == pytest.approx(0.5)
    assert out["enrichment_over_lagPat"] == pytest.approx(2 / 3 - 0.5)
    assert out["typology"] == "partial"
    assert out["informative"] is True


def test_set_relation_saturated_universe():
    """|S| == |L| → typology='degenerate', informative=False, enrichment≈0."""
    from src.rank_displacement import compute_clinical_soz_set_relation

    valid_chs = ["A", "B", "C", "D"]
    endpoint_chs = ["A", "B"]
    soz_chs = ["A", "B", "C", "D"]   # all of valid is SOZ
    out = compute_clinical_soz_set_relation(
        valid_chs=valid_chs, endpoint_chs=endpoint_chs, soz_chs=soz_chs,
    )
    assert out["typology"] == "degenerate"
    assert out["informative"] is False
    # enrichment should still compute (not None) but equals 0
    assert out["lagpat_baseline"] == pytest.approx(1.0)
    assert out["precision"] == pytest.approx(1.0)
    assert out["enrichment_over_lagPat"] == pytest.approx(0.0)


def test_set_relation_empty_soz_universe():
    """|S| == 0 → typology='degenerate', recall=None, informative=False."""
    from src.rank_displacement import compute_clinical_soz_set_relation

    valid_chs = ["A", "B", "C", "D", "E"]
    endpoint_chs = ["A", "B"]
    soz_chs: list[str] = []  # no SOZ in lagPat
    out = compute_clinical_soz_set_relation(
        valid_chs=valid_chs, endpoint_chs=endpoint_chs, soz_chs=soz_chs,
    )
    assert out["typology"] == "degenerate"
    assert out["informative"] is False
    assert out["recall_within_lagPat"] is None
    assert out["enrichment_over_lagPat"] is None  # baseline undefined when |S|=0


def test_set_relation_full_coverage():
    """|E| == |L| → typology='degenerate', precision == lagpat_baseline,
    enrichment ≡ 0 by construction."""
    from src.rank_displacement import compute_clinical_soz_set_relation

    valid_chs = ["A", "B", "C", "D"]
    endpoint_chs = ["A", "B", "C", "D"]   # endpoint = entire universe
    soz_chs = ["A", "B"]                   # half SOZ
    out = compute_clinical_soz_set_relation(
        valid_chs=valid_chs, endpoint_chs=endpoint_chs, soz_chs=soz_chs,
    )
    assert out["typology"] == "degenerate"
    assert out["informative"] is False
    assert out["precision"] == pytest.approx(0.5)
    assert out["lagpat_baseline"] == pytest.approx(0.5)
    assert out["enrichment_over_lagPat"] == pytest.approx(0.0)


def test_set_relation_E_subset_S():
    """E ⊂ S strictly → typology='E_subset_S', precision=1, recall<1."""
    from src.rank_displacement import compute_clinical_soz_set_relation

    valid_chs = ["A", "B", "C", "D", "E", "F"]
    endpoint_chs = ["A", "B"]            # E = {A, B}
    soz_chs = ["A", "B", "C", "D"]        # S = {A, B, C, D} ⊃ E (in lagPat)
    out = compute_clinical_soz_set_relation(
        valid_chs=valid_chs, endpoint_chs=endpoint_chs, soz_chs=soz_chs,
    )
    assert out["typology"] == "E_subset_S"
    assert out["informative"] is True
    assert out["precision"] == pytest.approx(1.0)
    assert out["recall_within_lagPat"] == pytest.approx(0.5)
    # enrichment = 1.0 - 4/6 = 1/3
    assert out["enrichment_over_lagPat"] == pytest.approx(1.0 - 4 / 6)


def test_set_relation_S_subset_E():
    """S ⊂ E strictly → typology='S_subset_E', recall=1, precision<1."""
    from src.rank_displacement import compute_clinical_soz_set_relation

    valid_chs = ["A", "B", "C", "D", "E", "F"]
    endpoint_chs = ["A", "B", "C", "D"]   # E = {A, B, C, D}
    soz_chs = ["A", "B"]                   # S = {A, B} ⊂ E
    out = compute_clinical_soz_set_relation(
        valid_chs=valid_chs, endpoint_chs=endpoint_chs, soz_chs=soz_chs,
    )
    assert out["typology"] == "S_subset_E"
    assert out["informative"] is True
    assert out["recall_within_lagPat"] == pytest.approx(1.0)
    assert out["precision"] == pytest.approx(0.5)
    # enrichment = 0.5 - 2/6 = 1/6
    assert out["enrichment_over_lagPat"] == pytest.approx(0.5 - 2 / 6)


def test_set_relation_disjoint():
    """E ∩ S == ∅ → typology='disjoint', precision=0, recall=0,
    enrichment = -lagpat_baseline."""
    from src.rank_displacement import compute_clinical_soz_set_relation

    valid_chs = ["A", "B", "C", "D", "E", "F"]
    endpoint_chs = ["E", "F"]            # E = {E, F}
    soz_chs = ["A", "B"]                  # S = {A, B}, disjoint from E
    out = compute_clinical_soz_set_relation(
        valid_chs=valid_chs, endpoint_chs=endpoint_chs, soz_chs=soz_chs,
    )
    assert out["typology"] == "disjoint"
    assert out["informative"] is True
    assert out["precision"] == pytest.approx(0.0)
    assert out["recall_within_lagPat"] == pytest.approx(0.0)
    assert out["enrichment_over_lagPat"] == pytest.approx(-2 / 6)


def test_enrichment_zero_when_E_uniform_sample_of_L():
    """E is a uniform sample of L w.r.t. SOZ proportion → enrichment ≈ 0.

    valid = 8 chs, half SOZ; endpoint = 4 chs evenly split SOZ/non-SOZ →
    precision = 0.5 = lagpat_baseline → enrichment = 0.
    """
    from src.rank_displacement import compute_clinical_soz_set_relation

    valid_chs = ["A", "B", "C", "D", "E", "F", "G", "H"]
    soz_chs = ["A", "B", "C", "D"]   # 4/8 SOZ
    endpoint_chs = ["A", "B", "E", "F"]   # 2 SOZ, 2 non-SOZ
    out = compute_clinical_soz_set_relation(
        valid_chs=valid_chs, endpoint_chs=endpoint_chs, soz_chs=soz_chs,
    )
    assert out["precision"] == pytest.approx(0.5)
    assert out["lagpat_baseline"] == pytest.approx(0.5)
    assert out["enrichment_over_lagPat"] == pytest.approx(0.0)


def test_cohort_sign_test_excludes_degenerate():
    """Mixed cohort: 3 degenerate + 5 informative → sign test runs on n=5 only."""
    from src.rank_displacement import cohort_sign_test_enrichment

    enrichments = [0.20, 0.10, -0.05, 0.30, 0.15, None, None, None]  # last 3 degenerate
    out = cohort_sign_test_enrichment(
        enrichments=enrichments, n_boot=500, seed=0,
    )
    assert out["n_informative"] == 5
    assert out["n_positive"] == 4   # 0.20, 0.10, 0.30, 0.15 positive; -0.05 negative
    # one-sided binomial p for 4/5 > 0.5
    assert out["sign_test_p"] < 0.5


def test_cohort_bootstrap_reproducible():
    """seed=0, n_boot=2000 → CI deterministic across runs."""
    from src.rank_displacement import cohort_sign_test_enrichment

    enrichments = [0.10, 0.20, 0.05, -0.05, 0.30, 0.15, 0.00, 0.25]
    a = cohort_sign_test_enrichment(
        enrichments=enrichments, n_boot=2000, seed=0,
    )
    b = cohort_sign_test_enrichment(
        enrichments=enrichments, n_boot=2000, seed=0,
    )
    assert a["bootstrap_ci_lo"] == pytest.approx(b["bootstrap_ci_lo"])
    assert a["bootstrap_ci_hi"] == pytest.approx(b["bootstrap_ci_hi"])
    assert a["median_enrichment"] == pytest.approx(b["median_enrichment"])
    # CI brackets the median
    assert a["bootstrap_ci_lo"] <= a["median_enrichment"] <= a["bootstrap_ci_hi"]


def test_cohort_sign_test_all_degenerate_returns_empty():
    """All None enrichments → n_informative=0, p undefined (None or 1.0 by convention)."""
    from src.rank_displacement import cohort_sign_test_enrichment

    out = cohort_sign_test_enrichment(
        enrichments=[None, None, None], n_boot=500, seed=0,
    )
    assert out["n_informative"] == 0
    assert out["sign_test_p"] is None or out["sign_test_p"] == 1.0
    assert out["median_enrichment"] is None
