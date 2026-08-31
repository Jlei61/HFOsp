from __future__ import annotations

import numpy as np

from src.topic5_continuous_marked_state_h2b.v03_seed_stability import (
    cross_seed_stability,
    linear_cka,
    procrustes_similarity,
)


def _trace(value: np.ndarray) -> dict[str, np.ndarray]:
    n = len(value)
    return {
        "anchor_time": np.arange(n, dtype=np.float64) * 30.0,
        "anchor_session": np.zeros(n, dtype=np.int64),
        "persistent_decoder": value.astype(np.float32),
        "persistent_state": value[:, :2].astype(np.float32),
    }


def test_seed_stability_recovers_shared_geometry_above_permuted_null() -> None:
    rng = np.random.default_rng(8)
    base = rng.normal(size=(120, 3))
    traces = [_trace(base + rng.normal(scale=0.01, size=base.shape)) for _ in range(3)]
    observed = cross_seed_stability(
        "subject", [0, 1, 2], traces, max_anchors=120, n_permutations=20,
    )
    assert observed["median_decoder_distance_correlation"] > 0.99
    assert observed["preliminary_Q5_pass"] is True
    assert observed["pairs_above_seed_permuted_null"] == 3


def test_cka_and_procrustes_are_invariant_to_orthogonal_rotation() -> None:
    rng = np.random.default_rng(2)
    left = rng.normal(size=(80, 3))
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    right = left @ q
    assert linear_cka(left, right) > 0.999
    assert procrustes_similarity(left, right) > 0.999


def test_linear_cka_rejects_nonfinite_or_incompatible_inputs() -> None:
    left = np.arange(12, dtype=np.float64).reshape(4, 3)
    assert linear_cka(left, left[:3]) is None
    left[0, 0] = np.nan
    assert linear_cka(left, left) is None


def test_similarity_functions_do_not_mutate_inputs() -> None:
    rng = np.random.default_rng(12)
    left = rng.normal(size=(30, 3))
    right = rng.normal(size=(30, 3))
    left_before, right_before = left.copy(), right.copy()
    linear_cka(left, right)
    procrustes_similarity(left, right)
    np.testing.assert_array_equal(left, left_before)
    np.testing.assert_array_equal(right, right_before)
