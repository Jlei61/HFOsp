"""TDD for the de novo layer (LR-7): the HARDER complement to read-back.

Read-back (main result) let each low-event window borrow the full-recording event labels
to orient its source->sink axis. De novo FORBIDS that: each window must re-cluster and
orient using ONLY its own events. The single change is the window's axis computation; rate,
the common-channel mask, and the count-matched null structure stay identical -> global vs
de novo EXCESS is a PAIRED contrast (the cost of forbidding the peek). See advisor note
2026-06-07. Primary metric stays SIGNED (apples-to-apples with the +0.131 read-back anchor);
|.| and the polarity-free endpoint-union are the decomposition.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.low_rate_template_stability import (
    align_template_events,
    window_reproductions,
    denovo_window_axis,
    window_recovery_paired,
    count_matched_null_gap_paired,
    window_endpoint_union_denovo,
)


# ---------- LR-7a: de novo axis must NOT blur a direction-pure window ----------

def test_denovo_axis_pure_direction_window_not_blurred():
    # 5 channels, ch0 source .. ch4 sink, ALL events one direction, small noise.
    # KMeans(k=2) always returns 2 clusters; an arbitrary split of one-direction events
    # yields two POSITIVELY-correlated centroids -> the aligner must NOT flip -> axis stays
    # a clean source->sink gradient (NOT averaged to ~0.5). This failure mode is de novo-only.
    rng = np.random.default_rng(0)
    n_ev = 8
    base = np.linspace(0, n_ev_ch := 4, 5)[:, None]               # integer-ish ranks 0..4
    ranks = np.clip(np.round(base + rng.normal(0, 0.3, (5, n_ev))), 0, 4)
    bools = np.ones((5, n_ev), dtype=bool)
    axis = denovo_window_axis(ranks, bools, min_cluster_events=4)
    # ch0 clearly source, ch4 clearly sink; monotone, not flattened
    assert axis[0] < 0.3 and axis[4] > 0.7
    rho = np.corrcoef(axis, np.arange(5))[0, 1]
    assert rho > 0.9                                              # de novo recovered the gradient


def test_denovo_axis_mixed_window_reclusters_and_deblurs():
    # 6 forward (ch0 source) + 2 reverse (ch0 sink). The de-blur is that the reverse minority
    # is FLIPPED (re-clustered) onto the forward axis, so ch0 reaches a clean source value ~0 --
    # NOT averaged toward the middle (naive masked mean of ch0 = (0*6 + 1*2)/8 = 0.25). On
    # scale-free Spearman a strict majority preserves order either way, so the distinguishing
    # de-blur signature is the magnitude: source pinned to ~0, sink to ~1.
    from src.lagpat_rank_audit import mask_phantom_ranks
    fwd = np.tile(np.array([0, 1, 2, 3, 4], float)[:, None], (1, 6))
    rev = np.tile(np.array([4, 3, 2, 1, 0], float)[:, None], (1, 2))
    ranks = np.concatenate([fwd, rev], axis=1)
    bools = np.ones((5, 8), dtype=bool)
    axis = denovo_window_axis(ranks, bools, min_cluster_events=4)
    assert np.corrcoef(axis, np.arange(5))[0, 1] > 0.9          # forward direction recovered
    naive_masked = np.array([np.nanmean(r) for r in mask_phantom_ranks(ranks, bools)])
    assert axis[0] < naive_masked[0]                            # reverse minority flipped, not averaged
    assert axis[0] < 0.1 and axis[4] > 0.9                      # clean source/sink, not blurred to 0.25/0.75


def test_denovo_axis_reverse_dominated_window_anticorrelates_full_axis():
    # 6 reverse + 2 forward. De novo larger cluster = reverse -> anchors reverse -> the window
    # axis ANTI-correlates with the full forward axis. Honest polarity penalty: signed < 0,
    # but |.| (axis line) is high. No peeking can rescue the sign.
    rev = np.tile(np.array([4, 3, 2, 1, 0], float)[:, None], (1, 6))
    fwd = np.tile(np.array([0, 1, 2, 3, 4], float)[:, None], (1, 2))
    ranks = np.concatenate([rev, fwd], axis=1)
    bools = np.ones((5, 8), dtype=bool)
    full_axis = np.array([0.0, 0.25, 0.5, 0.75, 1.0])           # forward target
    axis = denovo_window_axis(ranks, bools, min_cluster_events=4)
    signed = np.corrcoef(axis, full_axis)[0, 1]
    assert signed < 0                                           # reverse-dominated -> negative
    assert abs(signed) > 0.7                                    # but the axis LINE is recovered


def test_denovo_axis_small_window_falls_back_to_naive_mean():
    # m=3 < min_cluster_events=4 -> no clustering, naive nanmean; a pure-direction tiny window
    # still returns an oriented axis, no crash.
    ranks = np.tile(np.array([0, 1, 2, 3, 4], float)[:, None], (1, 3))
    bools = np.ones((5, 3), dtype=bool)
    axis = denovo_window_axis(ranks, bools, min_cluster_events=4)
    assert np.corrcoef(axis, np.arange(5))[0, 1] > 0.9


# ---------- LR-7b: paired global-vs-de-novo scoring on ONE index set ----------

def test_window_recovery_paired_global_arm_equals_main_reproductions():
    # The global arm of the paired scorer must reproduce the main read-back result exactly
    # (same aligned axis, same common mask, same rate). Only the de novo arm is new.
    rng = np.random.default_rng(1)
    n_ev = 30
    base = np.linspace(0, 4, 5)[:, None]
    ranks = np.clip(np.round(base + rng.normal(0, 0.3, (5, n_ev))), 0, 4)
    bools = rng.random((5, n_ev)) < np.linspace(0.5, 0.95, 5)[:, None]
    masked = np.where(bools, ranks / 4.0, np.nan)
    aligned, _ = align_template_events(masked, np.zeros(n_ev, dtype=int))
    full_axis = np.array([np.nanmean(r) if np.any(~np.isnan(r)) else np.nan for r in aligned])
    full_count = bools.sum(axis=1).astype(float)
    win = np.arange(12)
    main = window_reproductions(aligned, bools, full_axis, full_count, win, min_ch=3)
    paired = window_recovery_paired(aligned, ranks, bools, full_axis, full_count, win, min_ch=3)
    assert paired["template_repro_global"] == pytest.approx(main["template_repro"])
    assert paired["rate_repro"] == pytest.approx(main["rate_repro"])
    assert paired["n_common_channels"] == main["n_common_channels"]
    # de novo arm present; abs >= |signed-as-stored| consistency
    assert np.isfinite(paired["template_repro_denovo_signed"])
    assert paired["template_repro_denovo_abs"] == pytest.approx(abs(paired["template_repro_denovo_signed"]))


def test_count_matched_null_paired_returns_both_arms_finite():
    rng = np.random.default_rng(2)
    n_ev = 200
    base = np.linspace(0, 4, 6)[:, None]
    ranks = np.clip(np.round(base + rng.normal(0, 0.4, (6, n_ev))), 0, 5)
    bools = rng.random((6, n_ev)) < np.linspace(0.5, 0.95, 6)[:, None]
    masked = np.where(bools, ranks / 5.0, np.nan)
    aligned, _ = align_template_events(masked, np.zeros(n_ev, dtype=int))
    full_axis = np.array([np.nanmean(r) for r in aligned])
    full_count = bools.sum(axis=1).astype(float)
    ng, nd, nda = count_matched_null_gap_paired(
        aligned, ranks, bools, full_axis, full_count, m=30, n_ev=n_ev,
        rng=np.random.default_rng(3), n_null=40, min_ch=3)
    assert np.isfinite(ng) and np.isfinite(nd) and np.isfinite(nda)


# ---------- LR-7c: polarity-free endpoint union secondary ----------

def test_endpoint_union_is_polarity_free_and_recovers_full_union():
    # The source-UNION-sink set is invariant to axis polarity (no source/sink labelling).
    full_axis = np.array([0.0, 0.1, 0.5, 0.5, 0.9, 1.0])
    # window events mirror the full axis exactly -> union must be recovered (Jaccard 1)
    ranks = np.tile(np.array([0, 1, 2, 3, 4, 5], float)[:, None], (1, 8))
    bools = np.ones((6, 8), dtype=bool)
    full_count = np.array([10.0, 9.0, 5.0, 5.0, 2.0, 1.0])
    rep = window_endpoint_union_denovo(None, ranks, bools, full_axis, full_count,
                                       np.arange(8), k=2, min_ch=3, min_cluster_events=4)
    assert rep["endpoint_union_jaccard"] == pytest.approx(1.0)   # {0,1} ∪ {4,5} recovered
    assert np.isfinite(rep["rate_topk_jaccard"])


def test_endpoint_union_insufficient_when_too_few_common_channels():
    full_axis = np.array([0.0, 0.5, 1.0, np.nan])
    ranks = np.tile(np.array([0, 1, 2, 0], float)[:, None], (1, 4))
    bools = np.array([[1] * 4, [1] * 4, [1] * 4, [0] * 4], dtype=bool)
    full_count = np.array([3.0, 2.0, 1.0, 5.0])
    rep = window_endpoint_union_denovo(None, ranks, bools, full_axis, full_count,
                                       np.arange(4), k=2, min_ch=3, min_cluster_events=4)
    assert np.isnan(rep["endpoint_union_jaccard"])
