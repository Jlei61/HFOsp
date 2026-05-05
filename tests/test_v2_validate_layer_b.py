import numpy as np
import pytest
from scripts.v2_validate_layer_b import (
    compute_n_participating_stats,
    compute_pack_width_stats,
    compute_splithalf_rank_corr,
    compute_chunk_boundary_event_frac,
    compute_subset_rank_corr,
)


def test_n_participating_basic():
    n_part = np.array([2, 5, 7, 10, 3])
    s = compute_n_participating_stats(n_part)
    assert s["p50"] == 5.0
    assert s["p10"] == 2.4


def test_pack_width_basic():
    starts = np.array([0.0, 1.0, 2.0])
    ends = np.array([0.05, 1.10, 2.20])
    s = compute_pack_width_stats(starts, ends)
    assert s["p50"] == pytest.approx(0.10)  # 100 ms; float-precision tolerance, consistent with sibling tests


def test_subset_rank_corr_perfect():
    # both subsets have identical lagPatRank → rank corr should be 1.0
    rank = np.array([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]])
    valid = np.ones_like(rank, dtype=bool)
    a = np.array([0, 1])
    b = np.array([2, 3])
    rho = compute_subset_rank_corr(rank, valid, a, b)
    assert rho == pytest.approx(1.0)


def test_subset_rank_corr_inverted():
    # subset b is the inverse rank → corr ≈ -1.0
    rank_a = np.array([[0, 1, 2, 3], [0, 1, 2, 3]])
    rank_b = np.array([[3, 2, 1, 0], [3, 2, 1, 0]])
    rank = np.vstack([rank_a, rank_b])
    valid = np.ones_like(rank, dtype=bool)
    a = np.array([0, 1])
    b = np.array([2, 3])
    rho = compute_subset_rank_corr(rank, valid, a, b)
    assert rho == pytest.approx(-1.0)


def test_chunk_boundary_event_frac():
    # chunk_sec=200, tolerance=2 → events near 200, 400 boundaries
    starts = np.array([10.0, 198.5, 250.0, 401.0, 600.0])
    frac = compute_chunk_boundary_event_frac(starts, chunk_sec=200.0, tol_sec=2.0)
    # 198.5 within [198, 202] of 200; 401 within [398, 402] of 400; 600 is on 600 boundary
    assert frac == pytest.approx(3.0 / 5.0)


def test_subset_rank_corr_with_valid_mask_excludes_zero_cnt_channel():
    """If a channel has zero participation in either subset, it must be
    dropped from the corr (not counted as zero-rank)."""
    # 4 channels, 4 events. Event-major rank.
    rank = np.array([
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
    ])
    valid = np.array([
        [True,  True,  True,  False],   # event 0: ch3 absent
        [True,  True,  True,  False],   # event 1: ch3 absent
        [True,  True,  True,  False],   # event 2: ch3 absent
        [True,  True,  True,  False],   # event 3: ch3 absent
    ])
    rho = compute_subset_rank_corr(
        rank, valid,
        np.array([0, 1]), np.array([2, 3]),
    )
    # ch3 dropped → 3 channels left: rank [0,1,2] vs [0,1,2] → +1.0
    assert rho == pytest.approx(1.0)


def test_subset_rank_corr_returns_nan_when_too_few_channels_kept():
    """If fewer than 3 channels participate in BOTH subsets, return NaN."""
    rank = np.array([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]])
    valid = np.array([
        [True, True, False, False],
        [True, True, False, False],
        [True, True, False, False],
        [True, True, False, False],
    ])
    rho = compute_subset_rank_corr(
        rank, valid,
        np.array([0, 1]), np.array([2, 3]),
    )
    assert np.isnan(rho)
