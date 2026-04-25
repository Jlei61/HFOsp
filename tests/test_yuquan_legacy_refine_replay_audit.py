"""Lock the comparator + provenance gate used by Track B.

`compare_lagpat_pair` aligns:
  rows    by `chnNames` (set + order; legacy order canonical)
  columns by `packedTimes` (1:1 nearest within `pack_match_tol_ms`)

and only diffs `eventsBool` / `lagPatRank` / `lagPatRaw` on aligned indices.
A naive shape-equality diff is forbidden.

Provenance gate: `verdict_record` refuses to verdict any record whose
manifest reports `gpu_npz_used` under `DETECT_ROOT` — that would be a
same-source run masquerading as a legacy-refine replay.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from audit_yuquan_legacy_refine_replay import (  # noqa: E402
    DEFAULT_LAGRAW_TOL_SEC,
    DEFAULT_PACK_MATCH_TOL_MS,
    DEFAULT_PACKED_TOL_SEC,
    compare_lagpat_pair,
    provenance_violates_gate,
)


def _make_lagpat(
    chn_names, packed_times, *,
    lag_raw=None, lag_rank=None, events_bool=None, start_t=0.0,
):
    n_chn = len(chn_names)
    n_ev = len(packed_times)
    if lag_raw is None:
        lag_raw = np.zeros((n_chn, n_ev), dtype=np.float64)
    if lag_rank is None:
        lag_rank = np.zeros((n_chn, n_ev), dtype=np.int64)
    if events_bool is None:
        events_bool = np.ones((n_chn, n_ev), dtype=np.float64)
    return {
        "chnNames": np.array(chn_names),
        "packedTimes": np.asarray(packed_times, dtype=np.float64),
        "lagPatRaw": lag_raw,
        "lagPatRank": lag_rank,
        "eventsBool": events_bool,
        "start_t": np.float64(start_t),
    }


def test_equal_pair_passes_with_no_failures():
    chn = ["A1", "A2", "B1"]
    pt = np.array([[1.0, 1.3], [5.0, 5.3], [12.0, 12.3]], dtype=np.float64)
    raw = np.array([[0.0, 0.1, 0.2], [0.3, 0.4, 0.5], [0.6, 0.7, 0.8]])
    rank = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=np.int64)
    bool_ = np.ones((3, 3), dtype=np.float64)
    a = _make_lagpat(chn, pt, lag_raw=raw, lag_rank=rank, events_bool=bool_, start_t=100.0)
    b = _make_lagpat(chn, pt, lag_raw=raw, lag_rank=rank, events_bool=bool_, start_t=100.0)
    res = compare_lagpat_pair(a, b)
    assert res["chn_match"] is True
    assert res["events_bool_exact"] is True
    assert res["lag_rank_exact"] is True
    assert res["lag_raw_maxabs_sec"] == pytest.approx(0.0)
    assert res["packed_unmatched_legacy"] == 0
    assert res["packed_unmatched_new"] == 0
    assert res["failures"] == []


def test_chn_set_mismatch_fails_record_without_array_diff():
    """Chn set mismatch must short-circuit before any array op (which would
    raise a shape error). Failure label = `chn_set_mismatch`."""
    new = _make_lagpat(["A1", "A2"], np.array([[1.0, 1.3]]))
    legacy = _make_lagpat(["A1", "B1"], np.array([[1.0, 1.3]]))
    res = compare_lagpat_pair(new, legacy)
    assert res["chn_match"] is False
    assert "chn_set_mismatch" in res["failures"]


def test_chn_permutation_aligns_via_row_reorder():
    """Same chn set, different order — comparator row-permutes new to legacy
    order and the diff is exact again."""
    pt = np.array([[1.0, 1.3], [5.0, 5.3]])
    legacy_chn = ["A1", "A2"]
    new_chn = ["A2", "A1"]   # permuted
    legacy_raw = np.array([[10.0, 11.0], [20.0, 21.0]])
    # new written in new_chn order ⇒ row 0 = A2 values, row 1 = A1 values
    new_raw = np.array([[20.0, 21.0], [10.0, 11.0]])
    legacy_rank = np.array([[1, 2], [3, 4]], dtype=np.int64)
    new_rank = np.array([[3, 4], [1, 2]], dtype=np.int64)
    bool_legacy = np.array([[1.0, 1.0], [0.0, 1.0]])
    bool_new = np.array([[0.0, 1.0], [1.0, 1.0]])
    legacy = _make_lagpat(legacy_chn, pt, lag_raw=legacy_raw, lag_rank=legacy_rank,
                          events_bool=bool_legacy)
    new = _make_lagpat(new_chn, pt, lag_raw=new_raw, lag_rank=new_rank,
                       events_bool=bool_new)
    res = compare_lagpat_pair(new, legacy)
    assert res["chn_match"] is True
    assert res["events_bool_exact"] is True
    assert res["lag_rank_exact"] is True
    assert res["lag_raw_maxabs_sec"] == pytest.approx(0.0)
    assert res["failures"] == []


def test_packedtimes_count_differs_reports_unmatched_no_naive_diff():
    """When new has 4 windows but legacy has 3 (with a window deleted in the
    middle), the comparator must report 1 unmatched_new and align the rest.
    Naive shape diff would raise — this test pins that the comparator never
    does that."""
    chn = ["A1", "A2"]
    pt_legacy = np.array([[1.0, 1.3], [5.0, 5.3], [12.0, 12.3]])
    pt_new = np.array([[1.0, 1.3], [3.5, 3.8], [5.0, 5.3], [12.0, 12.3]])  # extra at 3.5
    raw_legacy = np.zeros((2, 3))
    raw_new = np.zeros((2, 4))
    rank_legacy = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.int64)
    rank_new = np.array([[1, 1, 1, 1], [2, 2, 2, 2]], dtype=np.int64)
    bool_legacy = np.ones((2, 3))
    bool_new = np.ones((2, 4))
    legacy = _make_lagpat(chn, pt_legacy, lag_raw=raw_legacy, lag_rank=rank_legacy,
                          events_bool=bool_legacy)
    new = _make_lagpat(chn, pt_new, lag_raw=raw_new, lag_rank=rank_new,
                       events_bool=bool_new)
    res = compare_lagpat_pair(new, legacy)
    assert res["chn_match"] is True
    assert res["packed_n_legacy"] == 3
    assert res["packed_n_new"] == 4
    assert res["packed_n_matched"] == 3
    assert res["packed_unmatched_new"] == 1
    assert res["packed_unmatched_legacy"] == 0
    assert "unmatched_columns" in res["failures"]
    # Aligned rank check still ran, no shape error:
    assert res["lag_rank_exact"] is True


def test_lag_raw_fp_noise_at_5e_12_passes_default_tolerance():
    chn = ["A1"]
    pt = np.array([[1.0, 1.3]])
    legacy_raw = np.array([[0.5]])
    new_raw = np.array([[0.5 + 5e-12]])  # within 1e-9 default
    legacy = _make_lagpat(chn, pt, lag_raw=legacy_raw,
                          lag_rank=np.array([[1]]),
                          events_bool=np.array([[1.0]]))
    new = _make_lagpat(chn, pt, lag_raw=new_raw,
                       lag_rank=np.array([[1]]),
                       events_bool=np.array([[1.0]]))
    res = compare_lagpat_pair(new, legacy, lag_raw_tol_sec=1e-9)
    assert res["lag_raw_maxabs_sec"] < 1e-9
    assert "raw_above_tolerance" not in res["failures"]


def test_lag_raw_noise_at_1e_3_fails_default_tolerance():
    chn = ["A1"]
    pt = np.array([[1.0, 1.3]])
    legacy_raw = np.array([[0.5]])
    new_raw = np.array([[0.5 + 1e-3]])  # 1 ms — well above any reasonable tolerance
    legacy = _make_lagpat(chn, pt, lag_raw=legacy_raw,
                          lag_rank=np.array([[1]]),
                          events_bool=np.array([[1.0]]))
    new = _make_lagpat(chn, pt, lag_raw=new_raw,
                       lag_rank=np.array([[1]]),
                       events_bool=np.array([[1.0]]))
    res = compare_lagpat_pair(new, legacy, lag_raw_tol_sec=1e-9)
    assert res["lag_raw_maxabs_sec"] >= 1e-3
    assert "raw_above_tolerance" in res["failures"]


def test_default_tolerances_are_published():
    """The default tolerances are exposed at module level so the preflight
    can override them coherently."""
    assert DEFAULT_PACK_MATCH_TOL_MS > 0
    assert DEFAULT_LAGRAW_TOL_SEC > 0
    assert DEFAULT_PACKED_TOL_SEC >= 1e-9


def test_provenance_gate_rejects_detect_root_paths():
    """A manifest record showing `gpu_npz_used` under `results/hfo_detection`
    fails the provenance gate. That's a same-source run, not a legacy-refine
    replay."""
    rec = {
        "record": "FA0013KP",
        "gpu_npz_used": "/home/honglab/leijiaxin/HFOsp/results/hfo_detection/gaolan/FA0013KP_gpu.npz",
    }
    summary = {"refine_npz_used": "/mnt/yuquan_data/yuquan_24h_edf/gaolan/_refineGpu.npz"}
    assert provenance_violates_gate(summary, rec) is True


def test_provenance_gate_rejects_detect_root_in_refine():
    rec = {
        "record": "FA0013KP",
        "gpu_npz_used": "/mnt/yuquan_data/yuquan_24h_edf/gaolan/FA0013KP_gpu.npz",
    }
    summary = {
        "refine_npz_used": "/home/honglab/leijiaxin/HFOsp/results/hfo_detection/gaolan/_refineGpu.npz"
    }
    assert provenance_violates_gate(summary, rec) is True


def test_provenance_gate_passes_legacy_paths():
    rec = {
        "record": "FA0013KP",
        "gpu_npz_used": "/mnt/yuquan_data/yuquan_24h_edf/gaolan/FA0013KP_gpu.npz",
    }
    summary = {
        "refine_npz_used": "/mnt/yuquan_data/yuquan_24h_edf/gaolan/_refineGpu.npz"
    }
    assert provenance_violates_gate(summary, rec) is False
