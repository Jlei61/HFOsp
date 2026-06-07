"""TDD for the v4 MAIN analysis: SOZ-internal readout stability over sampling.

Question (plan v4, post-review): within SOZ∩U, under short/noisy sampling, which readout
keeps its top-k closest to the full-recording top-k — firing participation COUNT, or a
propagation-geometry readout? Geometry readouts: source (reversal-SENSITIVE), sink,
ENDPOINT/axis (reversal-INVARIANT), and template-stratified source_fwd/source_rev.
count-matched null separates few-events sampling noise from real temporal drift.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.sef_hfo_soz_localization import (
    compute_readouts,
    topk_indices,
    jaccard,
    soz_internal_source_stability,
)


# ---------- readouts ----------

def test_compute_readouts_returns_rate_meanrank_endpoint():
    masked = np.array([[0.0, 0.0, np.nan, 0.0],    # c0 always source (rank 0)
                       [0.5, 1.0, 0.0, np.nan],
                       [1.0, np.nan, 1.0, 1.0]])
    bools = np.array([[1, 1, 0, 1],
                      [1, 1, 1, 0],
                      [1, 0, 1, 1]], dtype=bool)
    r = compute_readouts(masked, bools, ch_idx=[0, 1, 2], ev_idx=[0, 1, 2, 3])
    assert list(r["rate"]) == [3.0, 3.0, 3.0]
    assert r["mean_rank"][0] == pytest.approx(0.0)         # c0 always source
    assert r["endpoint"][0] == pytest.approx(1.0)          # rank 0 -> 2|0-0.5| = 1 (extreme = endpoint)
    assert r["endpoint"][2] == pytest.approx(1.0)          # c2 rank 1 -> also extreme (sink end)


def test_compute_readouts_template_stratified_when_labels_given():
    # c0 source in template A (rank 0), sink in template B (rank 1) -> all-events mean_rank ~0.5
    masked = np.array([[0.0, 0.0, 1.0, 1.0],   # c0: A-events rank0, B-events rank1
                       [1.0, 1.0, 0.0, 0.0]])
    bools = np.ones((2, 4), dtype=bool)
    labels = np.array([0, 0, 1, 1])            # events 0,1 = template A; 2,3 = template B
    r = compute_readouts(masked, bools, [0, 1], [0, 1, 2, 3], labels=labels, fwd_id=0, rev_id=1)
    assert r["mean_rank"][0] == pytest.approx(0.5)         # blurred when templates mixed
    assert r["mean_rank_fwd"][0] == pytest.approx(0.0)     # c0 is source in template A
    assert r["mean_rank_rev"][0] == pytest.approx(1.0)     # c0 is sink in template B


def test_topk_indices_high_for_rate_low_for_source_nan_is_worst():
    assert topk_indices([3.0, 1.0, 5.0, 2.0], k=2, largest=True) == {2, 0}
    assert topk_indices([0.1, 0.9, 0.4, np.nan], k=2, largest=False) == {0, 2}


def test_jaccard():
    assert jaccard({1, 2, 3}, {2, 3, 4}) == pytest.approx(2 / 4)
    assert jaccard(set(), set()) == 1.0
    assert jaccard({1}, {2}) == 0.0


# ---------- main: source/endpoint stable, rate drifts ----------

def _drift_stream():
    """200 events: c0 is the source (rank 0) in every event; the high-RATE channel flips
    halves (c1 leads 0-99, c2 leads 100-199) -> rate top-1 drifts, source/endpoint top-1 stable."""
    n_ev = 200
    chans = ["c0", "c1", "c2", "c3"]
    times = np.arange(n_ev, dtype=float)
    bools = np.zeros((4, n_ev), dtype=bool)
    bools[0, ::2] = True
    bools[1, :100] = True
    bools[2, 100:] = True
    bools[3, ::10] = True
    ranks = np.tile(np.arange(4, dtype=float)[:, None], (1, n_ev))   # c0 smallest -> source
    return ranks, bools, times, chans


def test_source_and_endpoint_more_stable_than_rate_with_null_separating_drift():
    ranks, bools, times, chans = _drift_stream()
    out = soz_internal_source_stability(
        ranks, bools, times, chans, soz_u_channels=chans,
        M_grid=[100], k=1, n_null=200, n_starts=2, seed=0)
    c = out["curves"][100]
    assert c["source"]["jaccard_obs"] > c["rate"]["jaccard_obs"]
    assert c["endpoint"]["jaccard_obs"] >= c["rate"]["jaccard_obs"]
    assert c["rate"]["jaccard_obs"] < c["rate"]["jaccard_null"]                 # rate drifts
    assert abs(c["source"]["jaccard_obs"] - c["source"]["jaccard_null"]) < 0.1  # source stable
    # E1: full target names are the correct SOZ-internal channels (c0 is the source)
    assert out["full_targets"]["source"] == ["c0"]


def test_window_marked_insufficient_when_fewer_than_k_finite_source_channels():
    # k=3 but only 2 channels ever participate -> source can never form a top-3 -> insufficient
    n_ev = 100
    chans = ["a", "b", "c", "d"]
    times = np.arange(n_ev, dtype=float)
    bools = np.zeros((4, n_ev), dtype=bool)
    bools[0] = True
    bools[1] = True                       # only a,b participate; c,d never -> source NaN for c,d
    ranks = np.tile(np.arange(4, dtype=float)[:, None], (1, n_ev))
    out = soz_internal_source_stability(
        ranks, bools, times, chans, chans, M_grid=[50], k=3, n_null=20, n_starts=3, seed=0)
    c = out["curves"][50]
    # source needs 3 finite channels but only 2 exist -> no valid windows, obs is NaN (not arbitrary)
    assert c["source"]["n_valid_windows"] == 0
    assert np.isnan(c["source"]["jaccard_obs"])
    # rate uses counts (c,d = 0, still finite) -> top-3 formable -> rate stays valid
    assert c["rate"]["n_valid_windows"] > 0
