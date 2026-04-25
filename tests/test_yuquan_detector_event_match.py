"""Lock the per-channel one-to-one onset matcher used by Track A.

The matcher must:
  - assign each legacy event to at most one new event (and vice versa);
  - choose globally tighter pairs over local greedy collisions;
  - bucket each match into tight (≤1.25 ms) / medium (≤6.25 ms) / loose
    (≤25 ms) at Yuquan's 800 Hz detector resample rate;
  - detect near-duplicate legacy / new events (onset within 1 sample);
  - flag unmatched events near the record boundary;
  - detect a coherent global time shift on a synthesized shifted set.

These properties drive the cohort-level verdict
(`physical_FP_only_threshold_sensitive` / `coarse_logic_divergence` /
`unexplained_residual`) so they need to be regression-pinned.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from audit_yuquan_detector_event_match import (  # noqa: E402
    DEFAULT_TOL_LOOSE_MS,
    DEFAULT_TOL_MEDIUM_MS,
    DEFAULT_TOL_TIGHT_MS,
    estimate_global_onset_shift_ms,
    match_events_one_to_one,
)


def _ev(*pairs):
    """Helper: turn (start_ms, dur_ms) tuples into legacy-style [start_s, end_s] lists."""
    return [[s_ms / 1000.0, (s_ms + d_ms) / 1000.0] for s_ms, d_ms in pairs]


def test_default_tolerances_match_yuquan_resample_rate():
    """Tolerance defaults are anchored to Yuquan's 800 Hz detector resample
    (1 sample = 1.25 ms). Anything else means the report header lies."""
    assert DEFAULT_TOL_TIGHT_MS == pytest.approx(1.25)
    assert DEFAULT_TOL_MEDIUM_MS == pytest.approx(6.25)
    assert DEFAULT_TOL_LOOSE_MS == pytest.approx(25.0)


def test_one_to_one_no_double_claim():
    """When two legacy events fight over one new event, only the closer one
    wins. The other is reported as unmatched."""
    legacy = _ev((100.0, 50.0), (101.0, 50.0))      # both starts within 1 ms of new[0]
    new = _ev((100.5, 50.0))
    res = match_events_one_to_one(legacy, new)
    assert len(res["matched"]) == 1
    m = res["matched"][0]
    assert (m["legacy_idx"], m["new_idx"]) == (0, 0)  # legacy 100.0 is closer to 100.5 (0.5 ms) than legacy 101.0 (0.5 ms tied) but legacy 100.0 wins on tie-break by lower index after equal delta
    # legacy event 1 must NOT have stolen new[0]
    assert 1 in res["unmatched_legacy_idx"]
    assert res["unmatched_new_idx"] == []


def test_globally_tighter_pair_beats_local_greedy():
    """legacy[0]→new[0] is 0.6 ms; legacy[0]→new[1] is 0.1 ms. legacy[1]→new[0]
    is 0.0 ms. A naive 'sort legacy, greedy claim' that processes legacy[0] first
    would steal new[0] for legacy[0] (0.6 ms) and orphan legacy[1]. Globally
    tighter pairing assigns legacy[1]→new[0] (0.0 ms) and legacy[0]→new[1]
    (0.1 ms). This is what `match_events_one_to_one` must do."""
    legacy = _ev((100.0, 50.0), (100.6, 50.0))
    new = _ev((100.6, 50.0), (100.1, 50.0))
    res = match_events_one_to_one(legacy, new)
    pairs = {(m["legacy_idx"], m["new_idx"]) for m in res["matched"]}
    assert pairs == {(1, 0), (0, 1)}, f"unexpected pairing {pairs}"


def test_tolerance_bucket_assignment():
    """Match buckets reflect onset_delta_ms strictly: tight ≤ 1.25, medium ≤ 6.25, loose ≤ 25."""
    legacy = _ev((100.0, 50.0), (200.0, 50.0), (300.0, 50.0), (400.0, 50.0))
    new = _ev(
        (100.5, 50.0),   # +0.5 ms → tight
        (203.0, 50.0),   # +3.0 ms → medium
        (320.0, 50.0),   # +20.0 ms → loose
        (450.0, 50.0),   # +50.0 ms → unmatched (above loose)
    )
    res = match_events_one_to_one(legacy, new)
    bucket_by_legacy = {m["legacy_idx"]: m["bucket"] for m in res["matched"]}
    assert bucket_by_legacy[0] == "tight"
    assert bucket_by_legacy[1] == "medium"
    assert bucket_by_legacy[2] == "loose"
    assert 3 not in bucket_by_legacy  # unmatched
    assert 3 in res["unmatched_legacy_idx"]
    assert 3 in res["unmatched_new_idx"]


def test_empty_channels():
    """Empty legacy or new channels return zero matches without error."""
    res = match_events_one_to_one([], [])
    assert res["matched"] == []
    assert res["unmatched_legacy_idx"] == []
    assert res["unmatched_new_idx"] == []

    res = match_events_one_to_one(_ev((100.0, 50.0)), [])
    assert res["matched"] == []
    assert res["unmatched_legacy_idx"] == [0]
    assert res["unmatched_new_idx"] == []

    res = match_events_one_to_one([], _ev((100.0, 50.0)))
    assert res["matched"] == []
    assert res["unmatched_legacy_idx"] == []
    assert res["unmatched_new_idx"] == [0]


def test_near_duplicate_legacy_events_reported():
    """Two legacy events with onsets within 1 sample (1.25 ms) are flagged as
    a near-duplicate pair. The matcher does not silently absorb one into the
    other — both are kept and at most one can match."""
    legacy = _ev((100.0, 50.0), (100.5, 50.0))   # 0.5 ms apart, sub-sample
    new = _ev((100.0, 50.0), (200.0, 50.0))
    res = match_events_one_to_one(legacy, new)
    assert (0, 1) in res["near_duplicate_legacy_idx_pairs"] or \
           (1, 0) in res["near_duplicate_legacy_idx_pairs"]
    # one of {0, 1} matched legacy → new[0]; the other unmatched
    matched_legacy = {m["legacy_idx"] for m in res["matched"]}
    assert len(matched_legacy) == 1
    assert 1 in res["unmatched_new_idx"]  # new[1] at 200 ms unrelated


def test_boundary_flag_set_for_edge_events():
    """Unmatched events within boundary_window_sec of record_last_sec carry
    `boundary_like = True` so the cohort verdict can excuse them."""
    legacy = _ev((10.0, 50.0), (4900.0, 50.0))     # 10 ms and 4.9 s
    new = _ev((11.0, 50.0))                        # only the first matches
    record_last_sec = 5.0
    res = match_events_one_to_one(
        legacy, new, record_last_sec=record_last_sec, boundary_window_sec=0.2
    )
    by_idx = {u["idx"]: u for u in res["unmatched_legacy_detail"]}
    # legacy[1] at 4.9 s is within 0.2 s of last 5.0 s → boundary_like True
    assert by_idx[1]["boundary_like"] is True
    # legacy[0] not in unmatched (it matched)
    assert 0 not in by_idx


def test_global_time_shift_detection():
    """Synthesize new = legacy + 5 samples (= +6.25 ms) on every event.
    `estimate_global_onset_shift_ms` must report ≈ +6.25 ms across matched pairs."""
    np.random.seed(0)
    legacy_starts = np.sort(np.random.uniform(1.0, 10.0, 50))
    legacy = [[s, s + 0.05] for s in legacy_starts]
    new = [[s + 6.25 / 1000.0, s + 0.05 + 6.25 / 1000.0] for s in legacy_starts]
    res = match_events_one_to_one(legacy, new)
    shift_ms = estimate_global_onset_shift_ms(res)
    assert shift_ms == pytest.approx(6.25, abs=0.01)


def test_no_global_shift_for_iid_jitter():
    """Sub-sample IID FP jitter (mean 0) must give |global shift| ≪ 1 sample.
    This is the cohort 'no_coarse_logic_shift' baseline behavior."""
    rng = np.random.default_rng(42)
    n = 200
    legacy_starts = np.sort(rng.uniform(1.0, 100.0, n))
    legacy = [[s, s + 0.05] for s in legacy_starts]
    jitter = rng.normal(0.0, 0.5e-3, n)  # 0.5 ms σ, IID, mean 0
    new = [[s + j, s + 0.05 + j] for s, j in zip(legacy_starts, jitter)]
    res = match_events_one_to_one(legacy, new)
    shift_ms = estimate_global_onset_shift_ms(res)
    assert abs(shift_ms) < 0.2, f"unexpected global shift {shift_ms} ms on IID jitter"


def test_loose_tolerance_does_not_admit_far_events():
    """A new event 50 ms away from any legacy event must remain unmatched
    even if no closer candidate exists."""
    legacy = _ev((1000.0, 50.0))
    new = _ev((1050.0, 50.0))                # 50 ms apart, > 25 ms loose
    res = match_events_one_to_one(legacy, new)
    assert res["matched"] == []
    assert res["unmatched_legacy_idx"] == [0]
    assert res["unmatched_new_idx"] == [0]
