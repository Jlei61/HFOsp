"""TDD tests for PR-T3-1 ``src/data_driven_soz.py``.

Step 1 covers M1 (HFO-onset rate) three variants + ranking + aggregation:
T1–T10. Step 2 will add T11–T18 for M2 (ER log-ratio) and Nyquist /
filter padding guards.

See ``docs/archive/topic3/pr_t3_1_data_driven_soz_audit_plan_2026-04-30.md``
§10 for the full TDD list.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.data_driven_soz import (
    aggregate_consensus,
    aggregate_median_rank,
    annotate_clinical_soz,
    check_channel_schema_consistency,
    compute_hfo_onset_metrics,
    matched_clinical_contacts,
    rank_top_k_per_seizure,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _events_with_n(n_pre: int, n_post: int, t_s: float, w_pre: float, w_post: float):
    """Return absolute timestamps placing ``n_pre`` events strictly inside
    ``[t_s - w_pre, t_s)`` and ``n_post`` events strictly inside
    ``(t_s, t_s + w_post]``.
    """
    pre = np.linspace(t_s - w_pre + 0.1, t_s - 0.1, num=n_pre) if n_pre > 0 else np.array([])
    post = np.linspace(t_s + 0.1, t_s + w_post - 0.1, num=n_post) if n_post > 0 else np.array([])
    return np.concatenate([pre, post])


# ---------------------------------------------------------------------------
# T1 — M1_raw simple arithmetic
# ---------------------------------------------------------------------------


def test_t1_m1_raw_arithmetic():
    t_s = 1000.0
    w_pre, w_post = 30.0, 10.0
    # post = 5 events / 10s, pre = 1 event / 30s
    events = {"chA": _events_with_n(1, 5, t_s, w_pre, w_post)}
    out = compute_hfo_onset_metrics(events, t_s, w_pre=w_pre, w_post=w_post)
    rec = out["chA"]
    # rate_post = 0.5 ; rate_pre = 1/30 ≈ 0.0333
    np.testing.assert_allclose(rec["M1_raw"], 0.5 - 1.0 / 30.0, rtol=1e-9)
    np.testing.assert_allclose(rec["rate_post"], 0.5, rtol=1e-9)
    np.testing.assert_allclose(rec["rate_pre"], 1.0 / 30.0, rtol=1e-9)


# ---------------------------------------------------------------------------
# T2 — M1_log formula with W_post / W_pre correction
# ---------------------------------------------------------------------------


def test_t2_m1_log_formula():
    t_s = 1000.0
    w_pre, w_post = 30.0, 10.0
    # post = 5, pre = 1 → log(6) - log(2) - log(10/30) = 2*log(3) ≈ 2.1972
    events = {"chA": _events_with_n(1, 5, t_s, w_pre, w_post)}
    out = compute_hfo_onset_metrics(events, t_s, w_pre=w_pre, w_post=w_post)
    expected = math.log(6.0) - math.log(2.0) - math.log(10.0 / 30.0)
    np.testing.assert_allclose(out["chA"]["M1_log"], expected, rtol=1e-9)
    np.testing.assert_allclose(out["chA"]["M1_log"], 2.0 * math.log(3.0), rtol=1e-9)


# ---------------------------------------------------------------------------
# T3 — M1_pois Poisson z arithmetic
# ---------------------------------------------------------------------------


def test_t3_m1_pois_arithmetic():
    t_s = 1000.0
    w_pre, w_post = 30.0, 10.0
    # post=5, pre=1, μ_pre = (1/30)*10 = 1/3
    # M1_pois = (5 - 1/3) / sqrt(1/3 + 1) ≈ 4.041
    events = {"chA": _events_with_n(1, 5, t_s, w_pre, w_post)}
    out = compute_hfo_onset_metrics(events, t_s, w_pre=w_pre, w_post=w_post)
    mu_pre = (1.0 / 30.0) * 10.0
    expected = (5 - mu_pre) / math.sqrt(mu_pre + 1.0)
    np.testing.assert_allclose(out["chA"]["M1_pois"], expected, rtol=1e-9)
    np.testing.assert_allclose(out["chA"]["M1_pois"], 4.041, atol=1e-2)


# ---------------------------------------------------------------------------
# T4 — channel with zero events → all three variants 0
# ---------------------------------------------------------------------------


def test_t4_no_events_all_zero():
    t_s = 1000.0
    events = {"silent": np.array([])}
    out = compute_hfo_onset_metrics(events, t_s, w_pre=30.0, w_post=10.0)
    rec = out["silent"]
    np.testing.assert_allclose(rec["M1_raw"], 0.0, atol=1e-12)
    # M1_log: log(0+1) - log(0+1) - log(10/30) = -log(1/3) = log(3)
    # The plan T4 states "全无 events → 三 variant 全 0".
    # That requires M1_log = 0 when n_pre = n_post = 0. The cleanest way is
    # to subtract the rate-correction term only when there are events on the
    # channel; or to set all three to 0 explicitly when both windows are
    # empty. The implementation chooses the explicit short-circuit per the
    # plan §3.3.
    np.testing.assert_allclose(rec["M1_log"], 0.0, atol=1e-12)
    # M1_pois: (0 - 0) / sqrt(0 + 1) = 0
    np.testing.assert_allclose(rec["M1_pois"], 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# T5 — rank_top_k 5 channels → top 3 deterministic
# ---------------------------------------------------------------------------


def test_t5_rank_top_k_basic():
    scores = {"a": 1.0, "b": 5.0, "c": 3.0, "d": 4.0, "e": 2.0}
    top3 = rank_top_k_per_seizure(scores, k=3)
    assert top3 == ["b", "d", "c"]


# ---------------------------------------------------------------------------
# T6 — NaN channels routed to bottom
# ---------------------------------------------------------------------------


def test_t6_rank_top_k_nan_bottom():
    scores = {"a": 1.0, "b": float("nan"), "c": 3.0, "d": 4.0, "e": float("nan")}
    top3 = rank_top_k_per_seizure(scores, k=3)
    # Non-NaN sorted: d, c, a; NaN channels b/e never enter top3.
    assert top3 == ["d", "c", "a"]


def test_t6b_rank_top_k_nan_tail_when_k_exceeds_finite_count():
    """Plan §3.5 ``rank_last`` contract: NaN scores rank LAST, not dropped.

    If ``k`` exceeds the number of finite-score channels, the NaN
    channels must fill the tail in alphabetical order — otherwise the
    size-matched primary k = ``|clinical_matched|`` (plan §3.6) would
    silently truncate when many channels have zero baseline rate.
    """
    scores = {"a": 1.0, "b": float("nan"), "c": 3.0, "d": 4.0, "e": float("nan")}
    top5 = rank_top_k_per_seizure(scores, k=5)
    # Finite descending: d (4), c (3), a (1); NaN tail alphabetical: b, e
    assert top5 == ["d", "c", "a", "b", "e"]


def test_t6c_rank_top_k_only_nan_returns_alphabetical_tail():
    scores = {"zeta": float("nan"), "alpha": float("nan"), "delta": float("nan")}
    top2 = rank_top_k_per_seizure(scores, k=2)
    assert top2 == ["alpha", "delta"]


# ---------------------------------------------------------------------------
# T7 — tie-break deterministic by ascending channel name
# ---------------------------------------------------------------------------


def test_t7_rank_top_k_tie_breaks_by_name():
    # All scores tied → top 3 must be alphabetically smallest 3
    scores = {"zeta": 1.0, "alpha": 1.0, "gamma": 1.0, "beta": 1.0, "delta": 1.0}
    top3 = rank_top_k_per_seizure(scores, k=3)
    assert top3 == ["alpha", "beta", "delta"]


# ---------------------------------------------------------------------------
# T8 — aggregate_consensus 50% threshold positive case
# ---------------------------------------------------------------------------


def test_t8_aggregate_consensus_positive():
    # 4 seizures, channel A appears in 3 (75% ≥ 50%) → IN
    per_seizure_topk = [
        ["A", "B", "C"],
        ["A", "B", "D"],
        ["A", "E", "F"],
        ["B", "G", "H"],
    ]
    consensus = aggregate_consensus(per_seizure_topk, min_seizure_fraction=0.5)
    assert "A" in consensus
    assert "B" in consensus  # B in 3/4 too


# ---------------------------------------------------------------------------
# T9 — aggregate_consensus 50% threshold negative case
# ---------------------------------------------------------------------------


def test_t9_aggregate_consensus_negative():
    # 4 seizures, channel A in 1 (25% < 50%) → OUT
    per_seizure_topk = [
        ["A", "B", "C"],
        ["X", "Y", "Z"],
        ["P", "Q", "R"],
        ["M", "N", "O"],
    ]
    consensus = aggregate_consensus(per_seizure_topk, min_seizure_fraction=0.5)
    assert "A" not in consensus


# ---------------------------------------------------------------------------
# T10 — aggregate_median_rank with median rank + missing → bottom rank
# ---------------------------------------------------------------------------


def test_t10_aggregate_median_rank():
    # 4 seizures × 5 channels. A has rank=2 in all 4 → median 2.
    # B has rank 1, 1, 5, 5 → median 3. C consistently rank 3 → median 3.
    # D and E rotate at the bottom.
    per_seizure_ranks = [
        {"A": 2, "B": 1, "C": 3, "D": 4, "E": 5},
        {"A": 2, "B": 1, "C": 3, "D": 4, "E": 5},
        {"A": 2, "B": 5, "C": 3, "D": 1, "E": 4},
        {"A": 2, "B": 5, "C": 3, "D": 1, "E": 4},
    ]
    top3 = aggregate_median_rank(per_seizure_ranks, k=3)
    # Medians: A=2, B=3, C=3, D=2.5, E=4.5 → smallest 3: {A=2, D=2.5, then B=3 or C=3}
    # Tie between B & C resolved by alphabetical order → B
    assert "A" in top3
    assert "D" in top3
    assert "B" in top3
    assert top3 == {"A", "B", "D"}


# ---------------------------------------------------------------------------
# Audit helpers (Step 0 hardening)
# ---------------------------------------------------------------------------


def test_matched_clinical_contacts_bipolar_partial_overlap():
    """Plan §3.2: matched contacts are clinical SOZ entries that touch
    at least one analysis channel. Contacts with no matching pair are
    reported via the unmatched complement.
    """
    analysis = ["A1-A2", "A2-A3", "B1-B2"]
    clinical = ["A1", "A3", "C1"]  # C1 not present anywhere
    matched = matched_clinical_contacts(analysis, clinical)
    assert matched == {"A1", "A3"}


def test_matched_clinical_contacts_normalizes_eeg_prefix():
    """``_normalize_channel_name`` strips ``EEG `` / ``EEG_`` and
    upper-cases. Reusing it for the unmatched stat keeps Step 0 audit
    aligned with the canonical matcher (plan §3.2 hard requirement).
    """
    analysis = ["EEG A1-EEG A2", "eeg_a3-eeg_a4"]
    clinical = ["A1", "a4"]
    matched = matched_clinical_contacts(analysis, clinical)
    assert matched == {"A1", "A4"}


def test_matched_clinical_contacts_car_channel():
    """CAR / monopolar analysis channels: single contact per channel."""
    analysis = ["GA1", "GA2", "GA3"]
    clinical = ["GA2"]
    matched = matched_clinical_contacts(analysis, clinical)
    assert matched == {"GA2"}


def test_annotate_clinical_soz_unknown_for_malformed_bipolar():
    """Plan §3.2: ``X-`` (empty side) → unknown."""
    analysis = ["A1-A2", "B1-", "-C2", "A1"]
    clinical = ["A1"]
    labels = annotate_clinical_soz(analysis, clinical)
    assert labels["A1-A2"] == "soz"
    assert labels["B1-"] == "unknown"
    assert labels["-C2"] == "unknown"
    assert labels["A1"] == "soz"


def test_check_channel_schema_consistency_consistent():
    blocks = [["A1", "A2", "A3"], ["A1", "A2", "A3"]]
    res = check_channel_schema_consistency(blocks)
    assert res["all_consistent"] is True
    assert res["mismatched_block_indices"] == []
    assert res["n_channels_min"] == 3
    assert res["n_channels_max"] == 3


def test_check_channel_schema_consistency_order_mismatch():
    """Channel ordering mismatch is a real problem because every other
    artifact in the pipeline indexes by position. Must flag, not silent."""
    blocks = [["A1", "A2", "A3"], ["A2", "A1", "A3"]]
    res = check_channel_schema_consistency(blocks)
    assert res["all_consistent"] is False
    assert res["mismatched_block_indices"] == [1]


def test_check_channel_schema_consistency_partial_blocks():
    blocks = [["A1", "A2", "A3"], ["A1", "A2"]]
    res = check_channel_schema_consistency(blocks)
    assert res["all_consistent"] is False
    assert res["n_channels_min"] == 2
    assert res["n_channels_max"] == 3
    assert res["mismatched_block_indices"] == [1]


def test_check_channel_schema_consistency_empty():
    res = check_channel_schema_consistency([])
    assert res["all_consistent"] is True
    assert res["mismatched_block_indices"] == []
    assert res["n_channels_min"] == 0
    assert res["n_channels_max"] == 0


def test_t10b_aggregate_median_rank_missing_seizure_goes_to_bottom():
    # 4 seizures, channel set varies. n_channels = 4.
    # X is ranked 1 in only 1 seizure, missing in others → median > 1.
    per_seizure_ranks = [
        {"X": 1, "Y": 2, "Z": 3, "W": 4},
        {"Y": 1, "Z": 2, "W": 3},  # X missing → counted as rank=4
        {"Y": 1, "Z": 2, "W": 3},  # X missing
        {"Y": 1, "Z": 2, "W": 3},  # X missing
    ]
    top3 = aggregate_median_rank(per_seizure_ranks, k=3)
    # Y median 1.5, Z median 2, W median 3, X median ≈ 4 → top3 = Y/Z/W
    assert top3 == {"Y", "Z", "W"}
