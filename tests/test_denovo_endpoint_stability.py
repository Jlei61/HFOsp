"""TDD for direction-agnostic endpoint stability (the redesigned de novo layer).

Primary question: in a low-event window, can KMeans k=2 recover the full-recording
ENDPOINT channels (first + last to fire), without needing to know which end is 'source'?

Key property: forward-template source = reverse-template sink. Both appear as extremes
in their respective clusters. Taking the union of cluster extremes gives the correct
endpoint set regardless of which template dominates the window.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.low_rate_template_stability import window_endpoint_stability_denovo


def _make_full_axis(n_ch=6):
    return np.linspace(0, 1, n_ch)


def _make_forward_events(n_ch, n_ev, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    base = np.linspace(0, n_ch - 1, n_ch)[:, None]
    ranks = np.clip(np.round(base + rng.normal(0, noise * n_ch, (n_ch, n_ev))), 0, n_ch - 1)
    bools = np.ones((n_ch, n_ev), dtype=bool)
    return ranks.astype(float), bools


def _make_reverse_events(n_ch, n_ev, noise=0.05, seed=1):
    ranks, bools = _make_forward_events(n_ch, n_ev, noise=noise, seed=seed)
    return (n_ch - 1) - ranks, bools


# ---------- ES-1: pure-window recovers endpoints ----------

def test_pure_forward_window_recovers_endpoints():
    """Pure-forward window: endpoints {0,1} (source) and {4,5} (sink) should be found."""
    n_ch = 6
    ranks, bools = _make_forward_events(n_ch, 12)
    full_axis = _make_full_axis(n_ch)
    full_count = bools.sum(axis=1).astype(float)
    result = window_endpoint_stability_denovo(
        ranks, bools, full_axis, full_count, np.arange(12), k=2)
    assert result["endpoint_jaccard"] > 0.5   # should recover most endpoints
    assert np.isfinite(result["rate_topk_jaccard"])


def test_mixed_forward_reverse_union_recovers_same_endpoints():
    """6 forward + 6 reverse events: endpoints are the same channels {0,1,4,5} in both
    templates. The union of cluster extremes must still recover them."""
    n_ch = 6
    r_fwd, b_fwd = _make_forward_events(n_ch, 6, noise=0.02, seed=0)
    r_rev, b_rev = _make_reverse_events(n_ch, 6, noise=0.02, seed=2)
    ranks = np.concatenate([r_fwd, r_rev], axis=1)
    bools = np.concatenate([b_fwd, b_rev], axis=1)
    full_axis = _make_full_axis(n_ch)
    full_count = bools.sum(axis=1).astype(float)
    result = window_endpoint_stability_denovo(
        ranks, bools, full_axis, full_count, np.arange(12), k=2)
    # Mix of two templates: union of cluster extremes should recover {0,1,4,5}
    assert result["endpoint_jaccard"] > 0.5


def test_naive_mean_blur_would_fail_but_kmeans_union_does_not():
    """
    Naive mean with 50/50 mix blurs endpoint channels to rank ~0.5 (indistinguishable
    from middle channels). The KMeans union approach should NOT have this failure mode.
    """
    n_ch = 6
    r_fwd, b_fwd = _make_forward_events(n_ch, 8, noise=0.01, seed=0)
    r_rev, b_rev = _make_reverse_events(n_ch, 8, noise=0.01, seed=2)
    ranks = np.concatenate([r_fwd, r_rev], axis=1)
    bools = np.concatenate([b_fwd, b_rev], axis=1)
    full_axis = _make_full_axis(n_ch)
    full_count = bools.sum(axis=1).astype(float)
    window_ev = np.arange(16)

    # Verify naive mean DOES blur the endpoint channels toward center
    from src.lagpat_rank_audit import mask_phantom_ranks
    masked = mask_phantom_ranks(ranks, bools)
    naive_axis = np.array([np.nanmean(r) for r in masked])
    # Endpoint channels 0 and 5 should be near 0.5 with naive mean (blur)
    assert abs(naive_axis[0] - 0.5) < 0.1
    assert abs(naive_axis[5] - 0.5) < 0.1

    # KMeans union should recover endpoints despite the blur
    result = window_endpoint_stability_denovo(
        ranks, bools, full_axis, full_count, window_ev, k=2)
    assert result["endpoint_jaccard"] > 0.5   # blur problem is solved


# ---------- ES-2: insufficient cases ----------

def test_insufficient_when_too_few_common_channels():
    """< 2k+1 common channels -> NaN returned."""
    full_axis = np.array([0.0, 0.5, 1.0, np.nan])
    ranks = np.tile(np.array([0, 1, 2, 0], float)[:, None], (1, 8))
    bools = np.array([[1]*8, [1]*8, [1]*8, [0]*8], dtype=bool)
    full_count = np.array([8., 6., 4., 0.])
    result = window_endpoint_stability_denovo(
        ranks, bools, full_axis, full_count, np.arange(8), k=2)
    # only 3 common channels, need >= 2k+1 = 5 -> NaN
    assert np.isnan(result["endpoint_jaccard"])


def test_small_m_fallback_uses_naive_mean():
    """m < min_cluster_events falls back to naive mean without crashing."""
    n_ch = 6
    ranks, bools = _make_forward_events(n_ch, 3)
    full_axis = _make_full_axis(n_ch)
    full_count = bools.sum(axis=1).astype(float)
    result = window_endpoint_stability_denovo(
        ranks, bools, full_axis, full_count, np.arange(3), k=2, min_cluster_events=4)
    # should return finite values (fallback path)
    assert np.isfinite(result["endpoint_jaccard"])
