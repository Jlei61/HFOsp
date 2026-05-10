"""TDD tests for src.ictal_seizure_plotting pure helpers."""

from __future__ import annotations

import numpy as np
import pytest


def test_subtype_sort_indices_orders_subtypes_then_outliers():
    from src.ictal_seizure_plotting import subtype_sort_indices
    # 6 seizures: subtypes [0, 1, 0, 1, -1, 0], outliers [F, F, F, F, T, F]
    order = subtype_sort_indices(
        subtype_label=[0, 1, 0, 1, -1, 0],
        outlier_flag=[False, False, False, False, True, False],
    )
    # Expected order: subtype 0 first (idx 0, 2, 5), then subtype 1 (idx 1, 3),
    # then outlier (idx 4)
    assert order == [0, 2, 5, 1, 3, 4]


def test_subtype_sort_indices_handles_singleton_outlier():
    from src.ictal_seizure_plotting import subtype_sort_indices
    # All in subtype 0 except idx 2 outlier
    order = subtype_sort_indices(
        subtype_label=[0, 0, -1, 0],
        outlier_flag=[False, False, True, False],
    )
    assert order == [0, 1, 3, 2]


def test_subtype_sort_indices_length_mismatch_raises():
    from src.ictal_seizure_plotting import subtype_sort_indices
    with pytest.raises(ValueError):
        subtype_sort_indices([0, 1], [False])


def test_compute_mds_2d_returns_correct_shape():
    from src.ictal_seizure_plotting import compute_mds_2d
    n = 10
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(n, 5))
    # Build symmetric distance matrix
    D = np.linalg.norm(pts[:, None] - pts[None, :], axis=-1)
    emb = compute_mds_2d(D, random_state=42)
    assert emb.shape == (n, 2)


def test_compute_mds_2d_handles_n_eq_1():
    from src.ictal_seizure_plotting import compute_mds_2d
    D = np.zeros((1, 1))
    emb = compute_mds_2d(D)
    assert emb.shape == (1, 2)
    assert np.allclose(emb, 0.0)


def test_subtype_color_palette_returns_distinct_colors():
    from src.ictal_seizure_plotting import subtype_color_palette
    cols, out_col = subtype_color_palette(3)
    assert len(cols) == 3
    assert len(set(cols)) == 3  # all distinct
    assert out_col != cols[0]


def test_channel_sort_by_subtype_means_orders_ascending():
    from src.ictal_seizure_plotting import channel_sort_by_subtype_means
    # 4 channels x 5 seizures, all in subtype 0
    onset = np.array([
        [50, 60, 70, 55, 65],   # ch 0: mean ~60
        [10, 20, 15, 25, 5],    # ch 1: mean ~15
        [100, 90, 110, 95, 105],  # ch 2: mean ~100
        [30, 35, 40, 25, 45],   # ch 3: mean ~35
    ], dtype=float)
    order = channel_sort_by_subtype_means(onset, [0, 0, 0, 0, 0])
    # Smallest mean first: ch 1 (15), ch 3 (35), ch 0 (60), ch 2 (100)
    assert order == [1, 3, 0, 2]


def test_channel_sort_by_subtype_means_uses_largest_subtype():
    from src.ictal_seizure_plotting import channel_sort_by_subtype_means
    # 2 channels, 5 seizures. Subtype 0 (n=4): ch 0 mean LOW, ch 1 mean HIGH.
    # Subtype 1 (n=1): ch 0 mean HIGH, ch 1 mean LOW.
    # Largest is subtype 0 → ch 0 should come first.
    onset = np.array([
        [10, 20, 15, 25,  100],
        [80, 90, 85, 95,  10 ],
    ], dtype=float)
    order = channel_sort_by_subtype_means(onset, [0, 0, 0, 0, 1])
    assert order == [0, 1]
