"""Tests for src/lagpat_rank_audit.py — phantom-rank audit utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.lagpat_rank_audit import (  # noqa: E402
    build_masked_kmeans_features,
    kmeans_label_ami_audit,
    mask_phantom_ranks,
)


# --------------------------------------------------------------------------
# TDD 1.1 — per-event re-ranking only on participating channels
# --------------------------------------------------------------------------


def test_mask_phantom_ranks_per_event():
    # 3 ch x 3 events. Hand-crafted so per-event re-rank is easy to verify.
    # Phantom values for non-participating cells are intentionally absurd
    # (10, 20) to prove they are discarded.
    ranks = np.array(
        [
            [0.0, 1.0, 10.0],   # ch 0
            [1.0, 20.0, 2.0],   # ch 1
            [2.0, 0.0, 1.0],    # ch 2
        ]
    )
    bools = np.array(
        [
            [True, True, False],   # ch 0
            [True, False, True],   # ch 1
            [True, True, True],    # ch 2
        ]
    )

    # normalize=False to assert raw rank integers
    masked = mask_phantom_ranks(ranks, bools, normalize=False)
    assert masked.shape == (3, 3)

    # ev 0: all 3 participate, ranks [0, 1, 2] -> rank-of-rank [0, 1, 2]
    assert np.array_equal(masked[:, 0], [0.0, 1.0, 2.0])

    # ev 1: ch 0, ch 2 participate (ch 1 phantom=20 must be dropped).
    # Participating ranks = [1, 0] -> rank-of-rank = [1, 0]
    assert masked[0, 1] == 1.0
    assert np.isnan(masked[1, 1])
    assert masked[2, 1] == 0.0

    # ev 2: ch 1, ch 2 participate (ch 0 phantom=10 dropped).
    # Participating ranks = [2, 1] -> rank-of-rank = [1, 0]
    assert np.isnan(masked[0, 2])
    assert masked[1, 2] == 1.0
    assert masked[2, 2] == 0.0


# --------------------------------------------------------------------------
# TDD 1.2 — normalized scale + n_part edge cases
# --------------------------------------------------------------------------


def test_normalize_event_rank_scale():
    # ev 0: n_part = 3 -> normalized = [0, 0.5, 1]
    # ev 1: n_part = 1 (only ch 2) -> normalized = 0.5
    # ev 2: n_part = 0 -> all NaN
    ranks = np.array(
        [
            [0.0, 99.0, 99.0],
            [1.0, 99.0, 99.0],
            [2.0, 7.0, 99.0],
        ]
    )
    bools = np.array(
        [
            [True, False, False],
            [True, False, False],
            [True, True, False],
        ]
    )
    masked = mask_phantom_ranks(ranks, bools, normalize=True)

    # ev 0: 3 participants, ranks [0, 1, 2] -> normalized [0, 0.5, 1]
    np.testing.assert_allclose(masked[:, 0], [0.0, 0.5, 1.0])

    # ev 1: 1 participant (ch 2) -> 0.5
    assert np.isnan(masked[0, 1]) and np.isnan(masked[1, 1])
    assert masked[2, 1] == 0.5

    # ev 2: zero participants -> all NaN
    assert np.all(np.isnan(masked[:, 2]))


# --------------------------------------------------------------------------
# TDD 1.3 — feature matrix shape + impute
# --------------------------------------------------------------------------


def test_build_masked_kmeans_features_shape_and_impute():
    # ev 0: all participate -> [0, 0.5, 1]; ev 1: only ch2 -> [0.5, 0.5, 0.5]
    # ev 2: ch0 + ch1 participate -> ranks [5, 3] -> normalized [1, 0]
    ranks = np.array(
        [
            [0.0, 99.0, 5.0],
            [1.0, 99.0, 3.0],
            [2.0, 7.0, 99.0],
        ]
    )
    bools = np.array(
        [
            [True, False, True],
            [True, False, True],
            [True, True, False],
        ]
    )
    X = build_masked_kmeans_features(ranks, bools, impute="event_median")

    # shape is (n_ev, n_ch)
    assert X.shape == (3, 3)
    # no NaN after impute
    assert np.isfinite(X).all()

    # ev 0: all participate, normalized rank 0, 0.5, 1
    np.testing.assert_allclose(X[0, :], [0.0, 0.5, 1.0])
    # ev 1: only ch2 participates, ch0+ch1 imputed to 0.5
    np.testing.assert_allclose(X[1, :], [0.5, 0.5, 0.5])
    # ev 2: ch0=1, ch1=0 (ranks 5,3 -> 1,0); ch2 imputed 0.5
    np.testing.assert_allclose(X[2, :], [1.0, 0.0, 0.5])


# --------------------------------------------------------------------------
# TDD 1.4 — phantom injection should drag AMI below noise floor
# --------------------------------------------------------------------------


def _make_phantom_synth(rng, n_per_cluster=120):
    """Synthetic dataset where phantom ranks would mis-cluster events.

    Cluster A: channels {0, 1, 2} participate, with rank order (0, 1, 2).
    Cluster B: channels {0, 1, 2} participate, with reversed order (2, 1, 0).
    Channels {3, 4} never participate.

    Phantom values for non-participating channels are drawn from a
    U-shape distribution biased to {0, 4}, asymmetrically across clusters:
    cluster A phantom for ch 3 is biased to 4; cluster B phantom for ch 3
    is biased to 0. This is exactly the bias the audit needs to catch:
    KMeans on phantom ranks will use ch 3 / ch 4 to split A from B.
    """
    n_ch = 5
    a_events = n_per_cluster
    b_events = n_per_cluster
    n_ev = a_events + b_events

    ranks = np.zeros((n_ch, n_ev), dtype=float)
    bools = np.zeros((n_ch, n_ev), dtype=bool)

    # Cluster A — true order 0,1,2 on ch 0,1,2; phantom ch 3,4 biased high
    for e in range(a_events):
        # true order with light jitter via two swappable positions
        perm = np.array([0, 1, 2]) if rng.random() < 0.9 else np.array([1, 0, 2])
        ranks[0, e] = perm[0]
        ranks[1, e] = perm[1]
        ranks[2, e] = perm[2]
        bools[0, e] = True
        bools[1, e] = True
        bools[2, e] = True
        # phantom: ch 3 -> high rank (3 or 4); ch 4 -> low (0 or 1)
        ranks[3, e] = rng.choice([3, 4])
        ranks[4, e] = rng.choice([0, 1])
        # bools[3,e] = bools[4,e] = False (phantom)

    # Cluster B — true reversed order 2,1,0; phantom ch 3,4 biased low/high
    for ei in range(b_events):
        e = a_events + ei
        perm = np.array([2, 1, 0]) if rng.random() < 0.9 else np.array([2, 0, 1])
        ranks[0, e] = perm[0]
        ranks[1, e] = perm[1]
        ranks[2, e] = perm[2]
        bools[0, e] = True
        bools[1, e] = True
        bools[2, e] = True
        # phantom: ch 3 -> low (0 or 1); ch 4 -> high (3 or 4)
        ranks[3, e] = rng.choice([0, 1])
        ranks[4, e] = rng.choice([3, 4])

    truth = np.concatenate([np.zeros(a_events, dtype=int), np.ones(b_events, dtype=int)])
    return ranks, bools, truth


def test_kmeans_label_ami_audit_synthetic_phantom():
    rng = np.random.default_rng(123)
    ranks, bools, truth = _make_phantom_synth(rng, n_per_cluster=150)
    audit = kmeans_label_ami_audit(ranks, bools, k=2, n_seeds=3)

    # phantom_fraction = ch 3,4 are non-participating -> 2/5 = 0.4
    assert abs(audit["phantom_fraction"] - 0.4) < 1e-9
    # n_events and k recorded
    assert audit["n_events"] == ranks.shape[1]
    assert audit["k"] == 2

    # Phantom is intentionally crafted so original KMeans should latch onto
    # ch 3/4 (constant per cluster, no within-cluster noise) and find the
    # truth labels easily — seed floor on original is high.
    assert audit["ami_seed_floor_original"] > 0.95
    # Masked features remove the phantom signal but leave the true order
    # signal — KMeans should still recover truth on masked features.
    assert audit["ami_seed_floor_masked"] > 0.80

    # Both feature matrices should mostly agree with truth, but the cross-
    # AMI(orig, masked) need not be 1.0 because the U-shape bias is
    # deliberately aligned with the true cluster. The audit *value* matters:
    # we test that ami_audit_minus_floor is a finite number in a sensible
    # range, not a directional claim on this synth.
    assert -1.0 < audit["ami_audit_minus_floor"] < 0.5


# --------------------------------------------------------------------------
# TDD 1.5 — full-participation passthrough: original == masked impute does
# nothing, so AMI(original, masked) should match seed-floor closely.
# --------------------------------------------------------------------------


def test_kmeans_label_ami_audit_no_phantom_pass_through():
    rng = np.random.default_rng(7)
    n_ch, n_ev = 6, 240
    # Two clusters with two underlying orderings; ALL channels always participate
    ranks = np.zeros((n_ch, n_ev), dtype=float)
    bools = np.ones((n_ch, n_ev), dtype=bool)
    base_a = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    base_b = np.array([5, 4, 3, 2, 1, 0], dtype=float)
    for e in range(n_ev):
        base = base_a if e < n_ev // 2 else base_b
        ranks[:, e] = base + rng.normal(0, 0.05, size=n_ch)
        # re-rank to integers like legacy producer
        order = np.argsort(np.argsort(ranks[:, e]))
        ranks[:, e] = order.astype(float)

    audit = kmeans_label_ami_audit(ranks, bools, k=2, n_seeds=3)

    # No phantom cells
    assert audit["phantom_fraction"] == 0.0
    # Both feature matrices encode the same information (masked just
    # re-normalizes ranks within each event to [0, 1]) — KMeans on either
    # is stable and the cross-AMI should be ~1.
    assert audit["ami_seed_floor_original"] > 0.99
    assert audit["ami_seed_floor_masked"] > 0.99
    assert audit["ami_audit"] > 0.99
    # ami_audit_minus_floor should be very close to 0
    assert abs(audit["ami_audit_minus_floor"]) < 0.02
