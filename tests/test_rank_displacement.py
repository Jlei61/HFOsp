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
