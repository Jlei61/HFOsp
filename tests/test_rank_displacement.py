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
