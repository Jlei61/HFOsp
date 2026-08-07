"""Contract tests for the WE-SLP-RNN v0.3 topology nulls."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_we_graph_analysis import (  # noqa: E402
    clustering_coefficient,
    contiguous_random_lesion,
    distance_controlled_similarity,
    length_preserving_rewire,
    modularity_q,
    summarise,
)
from src.topic5_wiring_economy_rnn import initial_mask  # noqa: E402


def _plane(n=60, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-30, 30, size=(n, 2))
    return xy, np.linalg.norm(xy[:, None] - xy[None], axis=-1)


def test_length_preserving_rewire_holds_degree_exactly():
    xy, d = _plane()
    mask = initial_mask(60, 0.1, d, spatial=True, seed=1) > 0
    out = length_preserving_rewire(mask, d, seed=2) > 0
    assert out.sum() == mask.sum()
    assert np.array_equal(out.sum(1), mask.sum(1))
    assert np.array_equal(out.sum(0), mask.sum(0))


def test_length_preserving_rewire_holds_the_edge_length_histogram():
    xy, d = _plane()
    mask = initial_mask(60, 0.1, d, spatial=True, seed=1) > 0
    out = length_preserving_rewire(mask, d, seed=2) > 0
    bins = np.quantile(d[mask], np.linspace(0, 1, 9))
    before = np.histogram(d[mask], bins=bins)[0]
    after = np.histogram(d[out], bins=bins)[0]
    assert np.array_equal(before, after)


def test_length_preserving_rewire_actually_moves_edges():
    xy, d = _plane()
    mask = initial_mask(60, 0.1, d, spatial=True, seed=1) > 0
    out = length_preserving_rewire(mask, d, seed=2) > 0
    assert (out != mask).sum() > 0.2 * mask.sum()


def test_contiguous_lesion_is_a_patch_not_confetti():
    xy, d = _plane(n=80)
    patch = contiguous_random_lesion(xy, 12, seed=3)
    scatter = np.random.default_rng(3).choice(80, size=12, replace=False)
    assert patch.size == 12
    spread = lambda idx: float(np.linalg.norm(xy[idx] - xy[idx].mean(0), axis=1).mean())  # noqa: E731
    assert spread(patch) < 0.6 * spread(scatter)


def test_contiguous_lesion_never_repeats_a_unit():
    xy, _ = _plane(n=40)
    for seed in range(8):
        idx = contiguous_random_lesion(xy, 10, seed=seed)
        assert len(set(idx.tolist())) == 10


def test_the_growth_prior_alone_already_shortens_edges_and_raises_clustering():
    # The untrained reference is not decoration.  Distance-biased growth on a
    # plane produces edges a third shorter and a visibly more clustered graph
    # before a single gradient step, so any post-training number has to be read
    # against this, not against zero.
    xy, d = _plane(n=80)
    spatial = initial_mask(80, 0.08, d, spatial=True, seed=5) > 0
    uniform = initial_mask(80, 0.08, d, spatial=False, seed=5) > 0
    assert d[spatial].mean() < 0.75 * d[uniform].mean()
    assert clustering_coefficient(spatial) > 1.08 * clustering_coefficient(uniform)


def test_the_growth_prior_alone_barely_moves_modularity():
    # 1/d weighting almost exactly cancels the 2D growth in the number of pairs
    # at distance d, so the proposal distribution over lengths is far flatter
    # than "distance-biased" suggests and modularity shifts by only a few
    # hundredths.  Recording it here stops a later reader from assuming the
    # growth rule alone explains a large modularity effect.
    gaps = []
    for seed in range(6):
        xy, d = _plane(n=80, seed=seed)
        spatial = initial_mask(80, 0.08, d, spatial=True, seed=seed) > 0
        uniform = initial_mask(80, 0.08, d, spatial=False, seed=seed) > 0
        gaps.append(modularity_q(spatial)[0] - modularity_q(uniform)[0])
    assert 0.0 < float(np.mean(gaps)) < 0.10


def test_summarise_reports_every_field_the_analysis_consumes():
    xy, d = _plane(n=40)
    mask = initial_mask(40, 0.1, d, spatial=True, seed=1)
    out = summarise(mask, d, with_small_world=False)
    for key in ("n_edges", "modularity_q", "n_modules", "clustering",
                "mean_edge_len_mm", "long_edge_fraction", "participation_mean",
                "connector_fraction"):
        assert key in out, key
    assert out["n_edges"] == int(mask.sum())


def test_distance_binning_removes_similarity_that_is_only_proximity():
    # Similarity is a pure function of distance and the graph is distance-biased.
    # Without binning the connected pairs look more similar; with binning the
    # difference has to vanish.
    xy, d = _plane(n=70)
    mask = initial_mask(70, 0.1, d, spatial=True, seed=4) > 0
    similarity = np.exp(-d / 12.0)
    off = ~np.eye(70, dtype=bool)
    naive = similarity[off & mask].mean() - similarity[off & ~mask].mean()
    binned = distance_controlled_similarity(similarity, d, mask)
    assert naive > 0.05
    assert abs(binned["delta"]) < 0.2 * naive
    assert binned["n_bins_used"] >= 4


def test_distance_controlled_similarity_still_sees_real_homophily():
    xy, d = _plane(n=70)
    mask = initial_mask(70, 0.1, d, spatial=True, seed=4) > 0
    rng = np.random.default_rng(0)
    similarity = np.exp(-d / 12.0) + 0.5 * (np.maximum(mask, mask.T))
    similarity = (similarity + similarity.T) / 2
    assert distance_controlled_similarity(similarity, d, mask)["delta"] > 0.1
