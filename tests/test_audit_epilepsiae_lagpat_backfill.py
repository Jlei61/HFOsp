"""Unit tests for Stage C audit (plan §4 Tasks C.1, C.2)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.audit_epilepsiae_lagpat_backfill import (
    _assign_subject_bucket,
    _classify_chn_bucket,
    _classify_count_bucket,
    _decide_stage_d,
    compute_record_metrics,
)


# ---------------------------------------------------------------------------
# Bucket classification
# ---------------------------------------------------------------------------


def test_classify_chn_bucket_thresholds():
    assert _classify_chn_bucket(0.9) == "stable"
    assert _classify_chn_bucket(0.7) == "stable"
    assert _classify_chn_bucket(0.6) == "moderate_drift"
    assert _classify_chn_bucket(0.5) == "moderate_drift"
    assert _classify_chn_bucket(0.49) == "large_drift"
    assert _classify_chn_bucket(0.0) == "large_drift"
    # missing => conservative worst case
    assert _classify_chn_bucket(None) == "large_drift"
    assert _classify_chn_bucket(float("nan")) == "large_drift"


def test_classify_count_bucket_thresholds():
    # stable band [0.7, 1.4]
    assert _classify_count_bucket(1.0) == "stable"
    assert _classify_count_bucket(0.7) == "stable"
    assert _classify_count_bucket(1.4) == "stable"
    # moderate band [0.5, 0.7) ∪ (1.4, 2.0]
    assert _classify_count_bucket(0.6) == "moderate_drift"
    assert _classify_count_bucket(1.5) == "moderate_drift"
    assert _classify_count_bucket(2.0) == "moderate_drift"
    # large
    assert _classify_count_bucket(0.4) == "large_drift"
    assert _classify_count_bucket(2.1) == "large_drift"
    # NaN / inf => large
    assert _classify_count_bucket(None) == "large_drift"
    assert _classify_count_bucket(float("nan")) == "large_drift"
    assert _classify_count_bucket(float("inf")) == "large_drift"


def test_subject_bucket_conservative_minimum():
    """Plan §4 Task C.2: 取较保守的桶."""
    # both stable => stable
    assert _assign_subject_bucket("stable", "stable") == "stable"
    # stable + moderate => moderate (worse)
    assert _assign_subject_bucket("stable", "moderate_drift") == "moderate_drift"
    assert _assign_subject_bucket("moderate_drift", "stable") == "moderate_drift"
    # stable + large => large (worst)
    assert _assign_subject_bucket("stable", "large_drift") == "large_drift"
    assert _assign_subject_bucket("large_drift", "stable") == "large_drift"
    # moderate + large => large
    assert _assign_subject_bucket("moderate_drift", "large_drift") == "large_drift"


def test_decide_stage_d():
    """Plan §4 Task C.2 decision table."""
    # enter_full: stable >= 14 AND large <= 2
    assert _decide_stage_d(stable=14, moderate=4, large=2) == "enter_full"
    assert _decide_stage_d(stable=15, moderate=4, large=1) == "enter_full"
    assert _decide_stage_d(stable=20, moderate=0, large=0) == "enter_full"
    # boundary: stable=14 large=3 => not enter_full -> enter_smoke
    assert _decide_stage_d(stable=14, moderate=3, large=3) == "enter_smoke"
    # pause: large >= 8
    assert _decide_stage_d(stable=10, moderate=0, large=10) == "pause"
    assert _decide_stage_d(stable=4, moderate=4, large=12) == "pause"
    # pause: stable < 5 (regardless of large)
    assert _decide_stage_d(stable=4, moderate=12, large=4) == "pause"
    # enter_smoke
    assert _decide_stage_d(stable=10, moderate=5, large=5) == "enter_smoke"
    assert _decide_stage_d(stable=8, moderate=8, large=4) == "enter_smoke"


# ---------------------------------------------------------------------------
# Per-record metric — risk-aware behavior (zero-event + missing-side)
# ---------------------------------------------------------------------------


def _make_rec(chns, raw, rank=None, eb=None):
    raw = np.asarray(raw, dtype=float)
    if rank is None:
        # default rank = argsort(argsort(raw)) per col
        rank = np.argsort(np.argsort(raw, axis=0), axis=0).astype(np.int64)
    rank = np.asarray(rank, dtype=np.int64)
    if eb is None:
        eb = (raw != 0).astype(float)
    return {
        "chnNames": np.array(chns, dtype=object),
        "lagPatRaw": raw,
        "lagPatRank": rank,
        "eventsBool": eb,
        "start_t": np.float64(0.0),
    }


def test_record_metrics_paired_overlap_ratios():
    new = _make_rec(
        chns=["A", "B", "C", "D"],
        raw=np.array([[0.0, 0.1], [0.2, 0.0], [0.4, 0.5], [0.6, 0.7]]),
    )
    legacy = _make_rec(
        chns=["A", "B", "C"],
        raw=np.array([[0.0, 0.1, 0.05], [0.2, 0.0, 0.15], [0.4, 0.5, 0.25]]),
    )
    m = compute_record_metrics(new, legacy)
    assert m["n_chns_new"] == 4
    assert m["n_chns_legacy"] == 3
    assert m["n_chns_shared"] == 3  # A B C
    assert m["chn_overlap_jaccard"] == pytest.approx(3 / 4)
    assert m["chn_overlap_jaccard_eligible"] is True
    assert m["n_events_new"] == 2
    assert m["n_events_legacy"] == 3
    assert m["count_ratio"] == pytest.approx(2 / 3)
    assert m["count_ratio_eligible"] is True
    assert m["zero_event_new"] is False
    assert m["zero_event_legacy"] is False
    assert m["both_zero"] is False
    # KS, lag span, rank should all be eligible (3 shared chns + events both >= 1)
    assert m["participation_ks_eligible"] is True
    assert m["lag_span_eligible"] is True
    assert m["rank_template_eligible"] is True


def test_record_metrics_zero_event_new_blocks_ratio_ks_lagspan_rank():
    new = _make_rec(chns=["A", "B", "C"], raw=np.empty((3, 0)))
    legacy = _make_rec(chns=["A", "B", "C"], raw=np.array([[0.1, 0.2], [0.0, 0.1], [0.4, 0.3]]))
    m = compute_record_metrics(new, legacy)
    assert m["zero_event_new"] is True
    assert m["zero_event_legacy"] is False
    assert m["both_zero"] is False
    # Channel jaccard still computable (chns exist on both sides)
    assert m["chn_overlap_jaccard_eligible"] is True
    # Ratio / KS / lag_span / rank ALL ineligible
    assert m["count_ratio"] is None
    assert m["count_ratio_eligible"] is False
    assert m["participation_ks_p"] is None
    assert m["participation_ks_eligible"] is False
    assert m["lag_span_diff"] is None
    assert m["lag_span_eligible"] is False
    assert m["rank_template_corr"] is None
    assert m["rank_template_eligible"] is False


def test_record_metrics_both_zero_event():
    new = _make_rec(chns=["A", "B"], raw=np.empty((2, 0)))
    legacy = _make_rec(chns=["A", "B"], raw=np.empty((2, 0)))
    m = compute_record_metrics(new, legacy)
    assert m["both_zero"] is True
    assert m["count_ratio"] is None
    assert m["count_ratio_eligible"] is False
    assert m["chn_overlap_jaccard_eligible"] is True
    assert m["chn_overlap_jaccard"] == pytest.approx(1.0)


def test_record_metrics_new_only_record():
    """new exists, legacy is None (legacy missing)."""
    new = _make_rec(chns=["A"], raw=np.array([[0.5, 0.6]]))
    m = compute_record_metrics(new, None)
    assert m["n_chns_new"] == 1
    assert m["n_chns_legacy"] == 0
    assert m["n_chns_shared"] == 0
    # No legacy => not eligible for chn jaccard / count ratio / ks / lag span / rank
    assert m["chn_overlap_jaccard"] is None
    assert m["chn_overlap_jaccard_eligible"] is False
    assert m["count_ratio"] is None
    assert m["count_ratio_eligible"] is False
    assert m["participation_ks_eligible"] is False
    assert m["lag_span_eligible"] is False
    assert m["rank_template_eligible"] is False


def test_record_metrics_few_shared_chns_blocks_rank_correlation():
    """Plan §4 Task C.1: rank_template_corr requires shared >= 3."""
    # Only 2 shared (A, B); rank should be ineligible
    new = _make_rec(chns=["A", "B", "X", "Y"], raw=np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]))
    legacy = _make_rec(chns=["A", "B", "M"], raw=np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]))
    m = compute_record_metrics(new, legacy)
    assert m["n_chns_shared"] == 2
    assert m["rank_template_eligible"] is False
    # But jaccard, count ratio, KS, lag span still eligible
    assert m["chn_overlap_jaccard_eligible"] is True
    assert m["count_ratio_eligible"] is True


def test_record_metrics_jaccard_disjoint():
    new = _make_rec(chns=["A", "B"], raw=np.array([[0.1, 0.2], [0.3, 0.4]]))
    legacy = _make_rec(chns=["X", "Y"], raw=np.array([[0.1, 0.2], [0.3, 0.4]]))
    m = compute_record_metrics(new, legacy)
    assert m["n_chns_shared"] == 0
    assert m["chn_overlap_jaccard"] == pytest.approx(0.0)
    assert m["chn_overlap_jaccard_eligible"] is True
    assert m["rank_template_eligible"] is False  # need shared >= 3
