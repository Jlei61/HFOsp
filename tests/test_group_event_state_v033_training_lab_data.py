"""Task 2: DataView (design §3, clauses D1-D7)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.topic5_group_event_state.v032_model.data import bundle_from_arrays
from src.topic5_group_event_state.v032_model.history_baseline import (
    HistoryBaseline,
    fit_provisional_history_baseline,
)
from src.topic5_group_event_state.v033_training_lab.data import (
    DEFAULT_BINS,
    bin_counts,
    build_view,
    robust_scale_apply,
    robust_scale_fit,
)
from tests.test_group_event_state_v032_model_toyutil import HORIZONS, make_toy_bundle, toy_timeline


def _bundle_with_segments(seg_len: float, seed: int = 0):
    tl, part = toy_timeline(seed, seg_len=seg_len)
    rng = np.random.default_rng(seed + 1)
    x_raw = rng.normal(size=(tl.event_times.size, 6)).astype(np.float32)
    history = fit_provisional_history_baseline(tl, part, HORIZONS)
    return bundle_from_arrays(tl, part, x_raw=x_raw, feature_names=tuple(f"x{i}" for i in range(6)),
                              history=history, eligibility=None, fingerprint={"toy": True, "seed": seed})


def test_d1_bin_counts_match_bundle_cumulative_windows():
    bundle, _ = make_toy_bundle(seed=0, planted_beta=0.0)
    view = build_view(bundle, bins=((0.0, 300.0), (0.0, 1800.0), (300.0, 900.0)))
    rows = np.concatenate([view.phase_index["train"], view.phase_index["inner_val"]])
    assert np.array_equal(view.counts[rows, 0], bundle.counts[rows, 0])
    assert np.array_equal(view.counts[rows, 1], bundle.counts[rows, 1])
    direct = bin_counts(bundle.event_times, bundle.t_anchor, ((300.0, 900.0),))
    assert np.array_equal(view.counts[rows, 2], direct[rows, 0])
    assert view.horizon == 1800.0


def test_d2_only_train_and_inner_val_are_exposed_and_dev_test_is_masked():
    bundle, _ = make_toy_bundle(seed=1, planted_beta=0.0)
    view = build_view(bundle)
    assert set(view.phase_index) == {"train", "inner_val"}
    assert np.array_equal(view.phase_index["train"], np.flatnonzero(bundle.anchor_mask("state_train", 1800.0)))
    assert np.array_equal(view.phase_index["inner_val"], np.flatnonzero(bundle.anchor_mask("dev_val", 1800.0)))
    exposed = np.zeros(bundle.n_anchors, dtype=bool)
    exposed[view.phase_index["train"]] = True
    exposed[view.phase_index["inner_val"]] = True
    assert (view.counts[~exposed] == -1).all()
    assert (view.counts[exposed] >= 0).all()
    dev_test = np.flatnonzero(bundle.anchor_mask("dev_test", 1800.0))
    with pytest.raises(ValueError):
        view.assert_no_dev_test(dev_test[:3])
    view.assert_no_dev_test(view.phase_index["train"][:3])
    assert view.n("train") > 0 and view.n("inner_val") > 0


def test_d3_split_hash_follows_partition_and_input_hash_follows_features():
    bundle, _ = make_toy_bundle(seed=2, planted_beta=0.0)
    z = build_view(bundle, scaling="zscore")
    r = build_view(bundle, scaling="robust")
    assert z.split_hash == r.split_hash
    assert z.input_hash != r.input_hash
    other_seed, _ = make_toy_bundle(seed=3, planted_beta=0.0)
    assert build_view(other_seed).split_hash == z.split_hash          # same segments -> same split
    assert build_view(other_seed).input_hash != z.input_hash
    other_partition = _bundle_with_segments(seg_len=30_000.0)
    assert build_view(other_partition).split_hash != z.split_hash


def test_d4_robust_scaling_statistics_depend_on_train_rows_only():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 4))
    x[:, 3] = 1.0                                   # degenerate column
    train = np.zeros(200, dtype=bool)
    train[:120] = True
    stats = robust_scale_fit(x, train)
    polluted = x.copy()
    polluted[~train] += 1e6
    assert robust_scale_fit(polluted, train) == stats
    z = robust_scale_apply(x, stats)
    assert z.dtype == np.float32 and np.isfinite(z).all()
    assert np.allclose(z[:, 3], 0.0)
    assert abs(float(np.median(z[train, 0]))) < 1e-6


def test_d5_event_balanced_weights_average_one_on_train_and_favour_busy_anchors():
    bundle, _ = make_toy_bundle(seed=4, planted_beta=0.0)
    view = build_view(bundle)
    ones = view.sample_weights("train", "anchor_balanced", lookback_seconds=7200.0)
    assert np.allclose(ones, 1.0)
    w = view.sample_weights("train", "event_balanced", lookback_seconds=7200.0)
    assert w.shape == (view.n("train"),) and abs(float(w.mean()) - 1.0) < 1e-9
    t = bundle.t_anchor[view.phase_index["train"]]
    seg_start = bundle.segment_bounds[bundle.anchor_segment[view.phase_index["train"]], 0]
    lo = np.searchsorted(bundle.event_times, np.maximum(t - 7200.0, seg_start), side="left")
    hi = np.searchsorted(bundle.event_times, t, side="left")
    n_lookback = hi - lo
    assert np.allclose(w, (1.0 + n_lookback) / (1.0 + n_lookback).mean())
    with pytest.raises(ValueError):
        view.sample_weights("train", "unknown", lookback_seconds=10.0)


def test_d6_registry_backed_bundle_never_gets_provisional_bins():
    bundle, _ = make_toy_bundle(seed=5, planted_beta=0.0)
    toy = build_view(bundle, bins=DEFAULT_BINS)
    assert toy.missing_h_bins == [] and toy.h_source == "provisional_local"
    rows = np.concatenate([toy.phase_index["train"], toy.phase_index["inner_val"]])
    assert np.isfinite(toy.log_mu_h[rows]).all() and toy.log_mu_h.shape == (bundle.n_anchors, 3)
    assert toy.log_r_h.shape == (3,) and np.isfinite(toy.log_r_h).all()
    registry = HistoryBaseline(log_mu=dict(bundle.history.log_mu), nb_log_dispersion=dict(bundle.history.nb_log_dispersion),
                               source="agent2_registry", meta={})
    human_like = replace(bundle, history=registry)
    view = build_view(human_like, bins=DEFAULT_BINS)
    assert view.missing_h_bins == [1, 2] and view.h_source == "agent2_registry"
    assert np.isfinite(view.log_mu_h[rows, 0]).all()
    assert np.isnan(view.log_mu_h[:, 1]).all() and np.isnan(view.log_mu_h[:, 2]).all()
    cumulative = build_view(human_like, bins=((0.0, 300.0), (0.0, 1800.0)))
    assert cumulative.missing_h_bins == []


def test_d7_blocks_never_cross_segments_and_span_at_most_block_length():
    bundle, _ = make_toy_bundle(seed=6, planted_beta=0.0)
    view = build_view(bundle)
    idx = view.phase_index["inner_val"]
    blocks = view.blocks("inner_val")
    assert blocks.shape == idx.shape
    for b in np.unique(blocks):
        member = idx[blocks == b]
        assert np.unique(bundle.anchor_segment[member]).size == 1
        assert bundle.t_anchor[member].max() - bundle.t_anchor[member].min() < 1800.0
    assert np.unique(blocks).size >= 2
    assert view.effective_independent_windows("inner_val") == bundle.effective_independent_windows("dev_val", 1800.0)
