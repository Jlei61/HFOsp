"""TDD for size-based cluster relabeling.

TA/TB is defined as "TA = the cluster with more events". KMeans emits labels 0/1
in arbitrary (init-order) order, so label 0 is the larger cluster only ~half the
time. ``_relabel_clusters_by_size`` renames clusters so id 0 = largest, keeping
every per-index field in sync (labels, clusters list + cluster_id,
inter_cluster_corr_matrix, candidate_forward_reverse_pairs). This is a pure
rename: all relative structure (per-cluster tau, symmetric correlations,
forward/reverse relationship) is preserved.
"""
from __future__ import annotations

import copy

from src.interictal_propagation import _relabel_clusters_by_size, _relabel_subject_clusters


def _cluster_dict(n0, n1, labels):
    tot = n0 + n1
    return {
        "n_clusters": 2,
        "n_valid_events": tot,
        "clusters": [
            {"cluster_id": 0, "n_events": n0, "fraction": n0 / tot, "raw_tau": 0.10, "template_rank": [0, 1, 2]},
            {"cluster_id": 1, "n_events": n1, "fraction": n1 / tot, "raw_tau": 0.30, "template_rank": [2, 1, 0]},
        ],
        "inter_cluster_corr": -0.7,
        "labels": list(labels),
    }


def test_no_swap_when_label0_already_largest():
    r = _cluster_dict(100, 40, [0, 0, 0, 1, 1])
    orig = copy.deepcopy(r)
    _relabel_clusters_by_size(r)
    assert r == orig  # already largest-first => untouched


def test_swap_when_label0_is_smaller():
    r = _cluster_dict(40, 100, [0, 0, 1, 1, 1])  # cluster 0 = 40 (small), cluster 1 = 100 (large)
    _relabel_clusters_by_size(r)
    assert r["clusters"][0]["n_events"] == 100  # largest now id 0 = TA
    assert r["clusters"][1]["n_events"] == 40
    assert r["clusters"][0]["cluster_id"] == 0
    assert r["clusters"][1]["cluster_id"] == 1
    assert r["labels"] == [1, 1, 0, 0, 0]  # 0<->1 flipped
    # per-cluster tau follows its cluster (the large cluster's tau was 0.30)
    assert r["clusters"][0]["raw_tau"] == 0.30
    assert r["clusters"][1]["raw_tau"] == 0.10


def test_adaptive_swaps_corr_matrix_and_forward_reverse_pairs():
    r = _cluster_dict(40, 100, [0, 1])
    r["inter_cluster_corr_matrix"] = [[1.0, -0.7], [-0.7, 1.0]]
    r["candidate_forward_reverse_pairs"] = [
        {"cluster_a": 0, "cluster_b": 1, "spearman_r": -0.7, "label": "candidate_forward_reverse"}
    ]
    _relabel_clusters_by_size(r)
    # symmetric matrix: diagonal stays 1, off-diagonal stays -0.7 after row/col swap
    assert r["inter_cluster_corr_matrix"] == [[1.0, -0.7], [-0.7, 1.0]]
    # pair indices remapped old->new (0->1, 1->0)
    assert r["candidate_forward_reverse_pairs"][0]["cluster_a"] == 1
    assert r["candidate_forward_reverse_pairs"][0]["cluster_b"] == 0
    assert r["candidate_forward_reverse_pairs"][0]["spearman_r"] == -0.7  # relationship unchanged


def test_ties_keep_original_order():
    r = _cluster_dict(50, 50, [0, 1])
    orig = copy.deepcopy(r)
    _relabel_clusters_by_size(r)
    assert r == orig  # equal sizes -> smaller id first -> no change


def test_relabel_returns_order_permutation():
    r = _cluster_dict(40, 100, [0, 1])
    assert _relabel_clusters_by_size(r) == [1, 0]  # largest (old id 1) first
    r2 = _cluster_dict(100, 40, [0, 1])
    assert _relabel_clusters_by_size(r2) is None  # already sorted -> no swap


def test_subject_level_remaps_within_cluster_centered_by_adaptive_order():
    d = {
        "cluster": _cluster_dict(40, 100, [0, 1]),
        "adaptive_cluster": _cluster_dict(40, 100, [0, 1]),
        "within_cluster_centered": {
            "per_cluster": {"0": {"raw_tau": 0.10}, "1": {"raw_tau": 0.30}},
            "mean_raw_tau": 0.20,
        },
    }
    _relabel_subject_clusters(d)
    # adaptive swapped so id 0 = larger; within_cluster_centered follows
    assert d["adaptive_cluster"]["clusters"][0]["n_events"] == 100
    assert d["within_cluster_centered"]["per_cluster"]["0"]["raw_tau"] == 0.30  # was cluster 1
    assert d["within_cluster_centered"]["per_cluster"]["1"]["raw_tau"] == 0.10
    assert d["within_cluster_centered"]["mean_raw_tau"] == 0.20  # scalar untouched
