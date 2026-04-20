"""
Phase A: Algorithm Identity validation for Yuquan lagPat backfill.

Goal: prove that, given the *exact same legacy inputs* (legacy `_gpu.npz`,
`_refineGpu.npz`, `_packedTimes.npy` and EDF), our new packing + spectrogram
centroid pipeline reproduces the legacy `_lagPat.npz` contract closely enough
to trust batch generation.

Per ``docs/archive/yuquan_lagpat/yuquan_lagpat_backfill_validation.plan.md``:

  A1 picked channels  : exact match against ``lagPat["chnNames"]``
  A2 packed windows   : new ``build_windows_from_detections`` vs legacy
                        ``packedTimes.npy`` (n match, start diff, P/R)
  A3 eventsBool       : reproduce legacy ``get_packedEvents_bool`` semantics
                        on legacy packedTimes; compare to ``lagPat["eventsBool"]``
  A4 lagPatRaw        : new ``compute_centroid_matrix_spectrogram`` vs legacy
                        per-window centroid offset (legacy uses concat-spec;
                        we compare *within-window centroid offset* to remove
                        per-segment concatenation bookkeeping)
  A5 lagPatRank       : ``argsort(argsort(centroid_per_event))`` over all
                        picked channels (legacy contract); compare exact match
                        and participating-only match

This script is **read-only against the canonical data directory**. It writes
results to ``results/validation/phaseA/<subject>__<block>.json`` plus a
markdown summary at ``results/validation/phaseA/SUMMARY.md``.

Usage::

    python scripts/validate_pack_against_legacy.py \
        --subject gaolan --block FA0013KP

    # Or run the canonical reference set:
    python scripts/validate_pack_against_legacy.py --reference-set
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.signal

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.group_event_analysis import (  # noqa: E402
    EventWindow,
    build_stitched_window_signal,
    build_windows_from_detections,
    build_windows_from_packed_times,
    compare_window_sets,
    compute_stitched_spectrogram_centroids_legacy,
    filter_windows_for_legacy_segment_loop,
)
from src.utils.bqk_utils import band_filt, notch_filt  # noqa: E402

# ---------------------------------------------------------------------------
# Legacy global parameters (mirroring P16_packGroupEvents_per2h_showSpecs_*.py)
# ---------------------------------------------------------------------------

LEGACY_SEGMENT_TIME = 200.0  # s, per-segment concatenation window
LEGACY_RESAMPLE_TO = 800.0  # Hz
LEGACY_NOTCH_FREQS = np.arange(50.0, 251.0, 50.0)
LEGACY_HIGHPASS_BAND = (80.0, 250.0)
LEGACY_EXT_MS = 30.0
LEGACY_CHNS_THR = 0.5
LEGACY_TIME_AXIS_HZ = 500.0
LEGACY_PACKING_GAP_LIMIT_S = 2.0  # pick_noOverlap_timeRanges(..., 2)
LEGACY_SPEC_FREQ_RANGE = (50.0, 300.0)
LEGACY_SPEC_NPERSEG_S = 0.05
LEGACY_SPEC_NOVERLAP_RATIO = 0.8
LEGACY_GAUSSIAN_SIGMA = 1.5
LEGACY_CENTROID_POWER = 3.0  # spec ** 3, NOT default
LEGACY_PACKED_BOOL_FS = 500.0  # get_packedEvents_bool fs

DATA_ROOT = Path("/mnt/yuquan_data/yuquan_24h_edf")
OUTPUT_ROOT = REPO_ROOT / "results" / "validation" / "phaseA"

# Legacy per-subject parameters (from
# `ReplayIED/inter_events/yuquan_24h_perPatientAnalysis_dropRef/
#  P16_packGroupEvents_per2h_showSpecs_bipolar_refine_bool.py` __main__ block
# and the commented `sub_pickT_list`/`sub_packWL_list` for older subjects).
LEGACY_SUBJECT_PARAMS: Dict[str, Dict[str, float]] = {
    "gaolan":      {"pick_k": 1.9, "pack_win_sec": 0.300, "drop_chns": ["B'4"]},
    "dongyiming":  {"pick_k": 0.5, "pack_win_sec": 0.220, "drop_chns": []},
    "wangyiyang":  {"pick_k": 1.0, "pack_win_sec": 0.250, "drop_chns": []},
    "chengshuai":  {"pick_k": 1.0, "pack_win_sec": 0.500, "drop_chns": []},
}


# ---------------------------------------------------------------------------
# Legacy preprocessing helpers (faithful re-implementation, kept inline so the
# validator does not depend on any new-code preprocessing path).
# ---------------------------------------------------------------------------

_CHN_RE = re.compile(r"([A-Z]'?)(\d+)")


def _split_chn(name: str) -> Tuple[str, str]:
    m = _CHN_RE.search(name)
    if not m:
        raise ValueError(f"channel name '{name}' does not match prefix+digits pattern")
    return m.group(1), m.group(2)


def _check_ch(name: str) -> bool:
    return re.search(r"([a-zA-Z]+) ([a-zA-Z]'?\d+)", name) is not None


def _standard_name(name: str) -> str:
    m = re.search(r"([a-zA-Z]+) ([a-zA-Z]'?\d+)", name)
    if not m:
        raise ValueError(f"channel name '{name}' is not standardisable")
    return m.group(2)


def _legacy_valid_chan_index(raw_ch_names: Sequence[str]) -> np.ndarray:
    """Faithful port of `return_valid_chan_index` from highEvents_yuquan0910_utils.

    Drops the outermost contact of each shaft.
    """
    valid_mask = np.array([_check_ch(c) for c in raw_ch_names], dtype=bool)
    valid_idx = np.where(valid_mask)[0]
    standardised = [_standard_name(c) for c in np.asarray(raw_ch_names)[valid_mask]]
    parsed = [_CHN_RE.search(s) for s in standardised]
    pre_list = [m.group(1) for m in parsed]
    num_list = [m.group(2) for m in parsed]
    counts = Counter(pre_list)
    keep = np.ones(len(pre_list), dtype=bool)
    for i, (pre, num) in enumerate(zip(pre_list, num_list)):
        if int(num) > counts[pre] - 1:
            keep[i] = False
    return valid_idx[keep]


def _legacy_bipolar_reref_and_drop(
    data: np.ndarray, chn_names: Sequence[str], drop_chns: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Faithful port of `bipolar_rerefAndDrop_eeg`.

    Channel names returned use the *left contact alias* (e.g. ``A1`` for
    pair ``A1-A2``). drop_chns may remove one bipolar pair near a noisy contact.
    """
    parsed = [_split_chn(c) for c in chn_names]
    pre_arr = np.array([p[0] for p in parsed])
    num_arr = [p[1] for p in parsed]
    pre_set = sorted(set(pre_arr))
    reref_chunks: List[np.ndarray] = []
    reref_names: List[str] = []
    for pre in pre_set:
        idx = np.where(pre_arr == pre)[0]
        nums = np.array([int(num_arr[i]) for i in idx])
        order = np.argsort(nums)
        idx = idx[order]
        nums = nums[order]
        chunk = data[idx]
        reref_chunks.append(chunk[:-1] - chunk[1:])
        reref_names += [f"{pre}{n}" for n in nums[:-1]]
    reref_data = np.concatenate(reref_chunks, axis=0)
    reref_names_arr = np.array(reref_names)
    if len(drop_chns) == 0 or len(drop_chns) > 2:
        return reref_data, reref_names_arr
    blocked: List[str] = []
    for chn in drop_chns:
        pre, num = _split_chn(chn)
        blocked.append(f"{pre}{int(num) - 1}")
        blocked.append(chn)
    keep = np.array([n not in blocked for n in reref_names_arr])
    return reref_data[keep], reref_names_arr[keep]


def _legacy_resample_notch_band(
    data: np.ndarray, fs_in: float
) -> np.ndarray:
    """Resample to 800 Hz, notch 50/100/150/200/250, bandpass 80-250 Hz."""
    factor_down = int(round(2.0 * fs_in / LEGACY_RESAMPLE_TO))
    if factor_down <= 0:
        raise ValueError(f"input sfreq {fs_in} cannot be resampled to {LEGACY_RESAMPLE_TO}")
    rs = scipy.signal.resample_poly(data, 2, factor_down, axis=-1)
    rs = notch_filt(rs, LEGACY_RESAMPLE_TO, LEGACY_NOTCH_FREQS)
    return band_filt(rs, LEGACY_RESAMPLE_TO, LEGACY_HIGHPASS_BAND)


# ---------------------------------------------------------------------------
# Legacy concat-spectrogram bookkeeping for A4 comparison
# ---------------------------------------------------------------------------


def _per_segment_within_window_offsets(
    legacy_lag_raw: np.ndarray,
    legacy_packed_times: np.ndarray,
    fs: float,
    segment_time_s: float,
) -> np.ndarray:
    """Convert legacy lagPatRaw (concat-segment time axis) to within-window
    centroid offsets in seconds.

    Legacy pipeline concatenates all packed-window samples *that fall fully
    inside one ``segment_time_s`` segment*, then runs one spectrogram and
    reports centroid time on that concatenated axis. The within-window offset
    is therefore::

        offset[ch, ev] = lagPatRaw[ch, ev] - cum_concat_start[ev_in_segment]

    where ``cum_concat_start`` accumulates *previous* packed-window durations
    expressed in samples-then-seconds at ``fs`` (legacy resample_to=800 Hz).

    Returns
    -------
    offsets : (n_ch, n_ev) float, NaN where legacy value was NaN.
    """
    n_ch, n_ev = legacy_lag_raw.shape
    if legacy_packed_times.shape[0] != n_ev:
        raise ValueError(
            f"packedTimes len {legacy_packed_times.shape[0]} != lagPat events {n_ev}"
        )
    offsets = np.full_like(legacy_lag_raw, np.nan, dtype=np.float64)
    seg_id_per_event = np.empty(n_ev, dtype=np.int64)
    seg_event_index = np.empty(n_ev, dtype=np.int64)

    # Faithful seg assignment: legacy used `np.arange(0, edf_data.times[-1], 200)`
    # then appended last sample, and admits an event if its [start, end] is
    # fully contained in [t0, t1] of the segment. We approximate by using
    # `floor(start / segment_time_s)`. Cross-segment events are silently
    # dropped by legacy (they were not in `inSeg_index`); we mark them NaN.
    for ev_idx, (s, e) in enumerate(legacy_packed_times):
        seg_id = int(s // segment_time_s)
        seg_id_end = int(np.floor((e - 1.0 / fs) / segment_time_s))
        if seg_id_end != seg_id:
            seg_id_per_event[ev_idx] = -1  # cross-segment, legacy dropped
        else:
            seg_id_per_event[ev_idx] = seg_id

    # For events kept, compute their within-segment ordinal and cum start.
    # Walk events in original order (legacy preserves order).
    cum_start_samples_per_seg: Dict[int, int] = {}
    for ev_idx in range(n_ev):
        seg = seg_id_per_event[ev_idx]
        if seg < 0:
            seg_event_index[ev_idx] = -1
            continue
        order = cum_start_samples_per_seg.get(seg, 0)
        seg_event_index[ev_idx] = order
        # Convert window duration to integer sample count exactly like legacy
        # `len(np.where(twBool)[0])` over `batch_t = arange(N) / fs`.
        s, e = legacy_packed_times[ev_idx]
        n_samples = int(np.floor((e - s) * fs)) + 1  # batch_t inclusive both ends
        # Legacy uses `(batch_t >= s) & (batch_t <= e)` → at fs=800, that
        # gives roughly (e - s) * fs + 1 samples.
        cum_start_samples_per_seg[seg] = order + n_samples

    # Now compute the *cumulative start time* for each event, in seconds.
    # cum_start_seconds[ev] = sum of preceding window sample-counts / fs.
    sample_cursors: Dict[int, int] = {}
    for ev_idx in range(n_ev):
        seg = seg_id_per_event[ev_idx]
        if seg < 0:
            continue
        cur = sample_cursors.get(seg, 0)
        cum_start_sec = cur / fs
        s, e = legacy_packed_times[ev_idx]
        n_samples = int(np.floor((e - s) * fs)) + 1
        sample_cursors[seg] = cur + n_samples
        # offset = lagPatRaw - cum_start_sec
        col = legacy_lag_raw[:, ev_idx] - cum_start_sec
        offsets[:, ev_idx] = col

    return offsets


# ---------------------------------------------------------------------------
# Phase A check implementations
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    name: str
    passed: bool
    metrics: Dict[str, float]
    notes: str = ""


def check_a1_picked_channels(
    refine_counts: np.ndarray,
    refine_names: np.ndarray,
    legacy_chn_names: Sequence[str],
    pick_k: float,
) -> CheckResult:
    counts = refine_counts.astype(np.float64)
    thr = float(counts.mean() + pick_k * counts.std())
    pick_idx = np.where(counts > thr)[0]
    new_picks = [str(refine_names[i]) for i in pick_idx]
    legacy_picks = [str(c) for c in legacy_chn_names]
    exact = new_picks == legacy_picks
    set_match = set(new_picks) == set(legacy_picks)
    return CheckResult(
        name="A1_picked_channels",
        passed=bool(exact),
        metrics={
            "n_new": float(len(new_picks)),
            "n_legacy": float(len(legacy_picks)),
            "exact_order_match": float(exact),
            "set_match": float(set_match),
            "threshold": thr,
            "mean_count": float(counts.mean()),
            "std_count": float(counts.std()),
        },
        notes=(
            "" if exact else (
                f"new={new_picks!r}\nlegacy={legacy_picks!r}\nmissing_in_new={sorted(set(legacy_picks) - set(new_picks))!r}\nextra_in_new={sorted(set(new_picks) - set(legacy_picks))!r}"
            )
        ),
    )


def check_a2_packed_windows(
    gpu_dets: np.ndarray,
    gpu_names: np.ndarray,
    picked_names: Sequence[str],
    legacy_packed: np.ndarray,
    pack_win_sec: float,
    t_max_sec: float,
    record_last_sec: float,
    record_sfreq: float,
) -> CheckResult:
    name_to_idx = {str(n): i for i, n in enumerate(gpu_names)}
    dets: Dict[str, np.ndarray] = {}
    for nm in picked_names:
        if nm in name_to_idx:
            dets[nm] = np.asarray(gpu_dets[name_to_idx[nm]], dtype=np.float64)
        else:
            dets[nm] = np.zeros((0, 2), dtype=np.float64)

    new_windows = build_windows_from_detections(
        dets,
        window_sec=float(pack_win_sec),
        ext_ms=LEGACY_EXT_MS,
        chns_thr=LEGACY_CHNS_THR,
        time_axis_hz=LEGACY_TIME_AXIS_HZ,
        max_window_sec=LEGACY_PACKING_GAP_LIMIT_S,
        t_max_sec=t_max_sec,
    )
    new_windows = filter_windows_for_legacy_segment_loop(
        new_windows,
        segment_duration_sec=LEGACY_SEGMENT_TIME,
        record_last_sec=float(record_last_sec),
        sfreq=float(record_sfreq),
        start_sec=0.0,
    )
    legacy_windows = build_windows_from_packed_times(legacy_packed)

    n_new = len(new_windows)
    n_legacy = len(legacy_windows)

    # match by overlap with min_overlap_sec = pack_win_sec * 0.5 (legacy windows
    # are ``pack_win_sec`` long; require >= 50% overlap to match).
    cmp_metrics = compare_window_sets(
        new_windows, legacy_windows, min_overlap_sec=float(pack_win_sec) * 0.5
    )

    # Per-pair start-time diff for matched windows (greedy index alignment when
    # n match and n_new == n_legacy is most informative)
    start_diffs_ms: List[float] = []
    if n_new == n_legacy and n_new > 0:
        a_starts = np.array([w.start for w in new_windows])
        b_starts = np.array([w.start for w in legacy_windows])
        start_diffs_ms = (np.abs(a_starts - b_starts) * 1000.0).tolist()
    median_diff = float(np.median(start_diffs_ms)) if start_diffs_ms else float("nan")
    p95_diff = float(np.percentile(start_diffs_ms, 95)) if start_diffs_ms else float("nan")

    precision = float(cmp_metrics["precision"])
    recall = float(cmp_metrics["recall"])
    n_match = float(cmp_metrics["n_match"])

    passed = (
        (n_new == n_legacy)
        and precision >= 0.98
        and recall >= 0.98
        and (np.isnan(median_diff) or median_diff <= 5.0)
        and (np.isnan(p95_diff) or p95_diff <= 20.0)
    )

    return CheckResult(
        name="A2_packed_windows",
        passed=bool(passed),
        metrics={
            "n_new": float(n_new),
            "n_legacy": float(n_legacy),
            "n_match": n_match,
            "precision": precision,
            "recall": recall,
            "median_abs_start_diff_ms": median_diff,
            "p95_abs_start_diff_ms": p95_diff,
        },
        notes="indexwise diff requires n_new == n_legacy" if n_new != n_legacy else "",
    )


def _legacy_get_packed_events_bool(
    high_events_times: List[np.ndarray],
    packed_times: np.ndarray,
    fs: float,
) -> np.ndarray:
    """Faithful port of legacy `get_packedEvents_bool`."""
    n_ch = len(high_events_times)
    n_ev = len(packed_times)
    bool_matrix = np.zeros((n_ch, n_ev), dtype=np.float64)
    all_concat = []
    for arr in high_events_times:
        if arr is None:
            continue
        a = np.asarray(arr, dtype=np.float64)
        if a.size:
            all_concat.append(a)
    if not all_concat:
        return bool_matrix
    all_arr = np.vstack(all_concat)
    max_times = float(all_arr.max())
    n_samp = int((max_times + 1.0) * fs)
    index_matrix = np.zeros((n_ch, n_samp), dtype=np.int8)
    for chi, arr in enumerate(high_events_times):
        if arr is None or len(arr) == 0:
            continue
        a = np.asarray(arr, dtype=np.float64)
        for s, e in a:
            i0 = int(s * fs)
            i1 = int(e * fs)
            i0 = max(0, min(n_samp, i0))
            i1 = max(0, min(n_samp, i1))
            if i1 > i0:
                index_matrix[chi, i0:i1] = 1
    for ti, (s, e) in enumerate(packed_times):
        i0 = int(s * fs)
        i1 = int(e * fs)
        i0 = max(0, min(n_samp, i0))
        i1 = max(0, min(n_samp, i1))
        if i1 <= i0:
            continue
        for chi in range(n_ch):
            if index_matrix[chi, i0:i1].sum() > 0:
                bool_matrix[chi, ti] = 1.0
    return bool_matrix


def check_a3_events_bool(
    gpu_dets: np.ndarray,
    gpu_names: np.ndarray,
    picked_names: Sequence[str],
    legacy_packed: np.ndarray,
    legacy_events_bool: np.ndarray,
) -> CheckResult:
    name_to_idx = {str(n): i for i, n in enumerate(gpu_names)}
    dets_list: List[np.ndarray] = []
    for nm in picked_names:
        if nm in name_to_idx:
            dets_list.append(np.asarray(gpu_dets[name_to_idx[nm]], dtype=np.float64))
        else:
            dets_list.append(np.zeros((0, 2), dtype=np.float64))
    new_bool = _legacy_get_packed_events_bool(dets_list, legacy_packed, LEGACY_PACKED_BOOL_FS) > 0
    legacy_bool = legacy_events_bool > 0

    if new_bool.shape != legacy_bool.shape:
        return CheckResult(
            name="A3_events_bool",
            passed=False,
            metrics={
                "new_shape": float(new_bool.size),
                "legacy_shape": float(legacy_bool.size),
            },
            notes=f"shape mismatch: new={new_bool.shape} legacy={legacy_bool.shape}",
        )
    intersection = float(np.logical_and(new_bool, legacy_bool).sum())
    union = float(np.logical_or(new_bool, legacy_bool).sum())
    jaccard = intersection / union if union > 0 else 1.0
    per_event_new = new_bool
    per_event_legacy = legacy_bool
    per_event_match = (per_event_new == per_event_legacy).all(axis=0)
    exact_event_rate = float(per_event_match.mean()) if per_event_match.size else 0.0

    passed = jaccard >= 0.98 and exact_event_rate >= 0.95
    return CheckResult(
        name="A3_events_bool",
        passed=bool(passed),
        metrics={
            "jaccard": jaccard,
            "exact_event_match_rate": exact_event_rate,
            "n_events": float(legacy_bool.shape[1]),
            "n_picked": float(legacy_bool.shape[0]),
        },
    )


def check_a4_a5_centroid_and_rank(
    edf_path: Path,
    picked_names: Sequence[str],
    drop_chns: Sequence[str],
    legacy_packed: np.ndarray,
    legacy_lag_raw: np.ndarray,
    legacy_lag_rank: np.ndarray,
    legacy_events_bool: np.ndarray,
) -> Tuple[CheckResult, CheckResult, Dict[str, float]]:
    """Run the exact legacy segment loop and compare lagPatRaw / lagPatRank."""
    import mne  # local import; expensive
    from scipy.stats import kendalltau

    raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False, encoding="latin1")
    fs_in = float(raw.info["sfreq"])
    valid_idx = _legacy_valid_chan_index(raw.ch_names)
    if valid_idx.size == 0:
        raise RuntimeError(f"no valid channels found in {edf_path}")
    valid_names = [_standard_name(raw.ch_names[i]) for i in valid_idx]
    legacy_bool = legacy_events_bool > 0

    time_inter = np.arange(0.0, float(raw.times[-1]), LEGACY_SEGMENT_TIME, dtype=np.float64)
    time_inter = np.append(time_inter, float(raw.times[-1]))
    time_ranges = np.stack([time_inter[:-1], time_inter[1:]], axis=1)

    centroids_list: List[np.ndarray] = []
    rank_list: List[np.ndarray] = []
    kept_event_indices: List[np.ndarray] = []
    total_stitched_samples = 0

    for tr in time_ranges:
        seg_start_idx, seg_end_idx = raw.time_as_index(tuple(float(x) for x in tr))
        seg_start_idx = int(seg_start_idx)
        seg_end_idx = int(seg_end_idx)
        if seg_end_idx <= seg_start_idx:
            continue

        seg_time = raw.times[seg_start_idx:seg_end_idx]
        if seg_time.size == 0:
            continue
        if float(seg_time[-1] - seg_time[0]) < 5.0:
            continue

        in_seg_idx = np.where(
            (legacy_packed[:, 0] >= float(seg_time[0]))
            & (legacy_packed[:, 1] <= float(seg_time[-1]))
        )[0]
        if in_seg_idx.size == 0:
            continue

        seg_data = raw.get_data(
            picks=valid_idx.tolist(),
            start=seg_start_idx,
            stop=seg_end_idx,
        )
        seg_data, bipolar_names = _legacy_bipolar_reref_and_drop(seg_data, valid_names, drop_chns)
        bipolar_index = {str(n): i for i, n in enumerate(bipolar_names)}
        miss = [n for n in picked_names if n not in bipolar_index]
        if miss:
            raise RuntimeError(
                f"picked channels missing after bipolar reref/drop in segment {tr.tolist()}: {miss}"
            )
        pick_rows = np.array([bipolar_index[n] for n in picked_names], dtype=int)
        seg_pick = seg_data[pick_rows]
        seg_band = _legacy_resample_notch_band(seg_pick, fs_in)

        seg_windows = [
            EventWindow(float(legacy_packed[i, 0]), float(legacy_packed[i, 1]), int(i))
            for i in in_seg_idx
        ]
        stitched, split_border_t = build_stitched_window_signal(
            seg_band,
            seg_windows,
            sfreq=LEGACY_RESAMPLE_TO,
            start_sec=float(seg_time[0]),
        )
        if stitched.shape[1] == 0 or split_border_t.size == 0:
            continue

        seg_centroids = compute_stitched_spectrogram_centroids_legacy(
            stitched,
            split_border_t,
            sfreq=LEGACY_RESAMPLE_TO,
            spec_freq_range=LEGACY_SPEC_FREQ_RANGE,
            spec_nperseg_sec=LEGACY_SPEC_NPERSEG_S,
            spec_noverlap_ratio=LEGACY_SPEC_NOVERLAP_RATIO,
            gaussian_sigma=LEGACY_GAUSSIAN_SIGMA,
            centroid_power=LEGACY_CENTROID_POWER,
        )
        seg_rank = np.array([np.argsort(np.argsort(x)) for x in seg_centroids.T]).T.astype(np.int64)

        centroids_list.append(seg_centroids)
        rank_list.append(seg_rank)
        kept_event_indices.append(in_seg_idx.astype(np.int64))
        total_stitched_samples += int(stitched.shape[1])

    if not centroids_list:
        a4 = CheckResult(
            name="A4_lagPatRaw",
            passed=False,
            metrics={"n_compare": 0.0},
            notes="legacy segment loop yielded no comparable stitched windows",
        )
        a5 = CheckResult(
            name="A5_lagPatRank",
            passed=False,
            metrics={"shape_mismatch": 1.0},
            notes="legacy segment loop yielded no comparable stitched windows",
        )
        return a4, a5, {"n_segments_used": 0.0, "total_stitched_samples": 0.0}

    centroids = np.concatenate(centroids_list, axis=1)
    new_rank = np.concatenate(rank_list, axis=1)
    kept_idx = np.concatenate(kept_event_indices, axis=0)

    if not np.array_equal(kept_idx, np.arange(legacy_packed.shape[0], dtype=np.int64)):
        raise RuntimeError(
            "legacy segment loop did not reproduce packedTimes column order exactly; "
            f"kept {kept_idx.shape[0]} / expected {legacy_packed.shape[0]}"
        )

    # ---- A4: lagPatRaw exact stitched timeline ----
    mask = legacy_bool & np.isfinite(legacy_lag_raw) & np.isfinite(centroids)
    n_compare = int(mask.sum())
    if n_compare == 0:
        a4 = CheckResult(
            name="A4_lagPatRaw",
            passed=False,
            metrics={"n_compare": 0.0},
            notes="no comparable cells (mask empty)",
        )
    else:
        diff_ms = (centroids[mask] - legacy_lag_raw[mask]) * 1000.0
        abs_diff_ms = np.abs(diff_ms)
        median_ae = float(np.median(abs_diff_ms))
        p95_ae = float(np.percentile(abs_diff_ms, 95))
        rmse = float(np.sqrt(np.mean(diff_ms ** 2)))
        a4_passed = median_ae <= 5.0 and p95_ae <= 20.0 and rmse <= 10.0
        a4 = CheckResult(
            name="A4_lagPatRaw",
            passed=bool(a4_passed),
            metrics={
                "n_compare": float(n_compare),
                "median_abs_err_ms": median_ae,
                "p95_abs_err_ms": p95_ae,
                "rmse_ms": rmse,
                "mean_signed_err_ms": float(np.mean(diff_ms)),
            },
        )

    # ---- A5: lagPatRank exact matrix contract ----
    if new_rank.shape != legacy_lag_rank.shape:
        a5 = CheckResult(
            name="A5_lagPatRank",
            passed=False,
            metrics={"shape_mismatch": 1.0},
            notes=f"new_rank shape {new_rank.shape} vs legacy {legacy_lag_rank.shape}",
        )
    else:
        full_match_rate = float((new_rank == legacy_lag_rank).all(axis=0).mean())
        partic_rate = float((new_rank == legacy_lag_rank)[legacy_bool].mean()) if legacy_bool.any() else float("nan")
        partic_event_exact = float(
            np.mean(
                [
                    np.array_equal(
                        new_rank[:, i][legacy_bool[:, i]],
                        legacy_lag_rank[:, i][legacy_bool[:, i]],
                    )
                    for i in range(new_rank.shape[1])
                ]
            )
        ) if new_rank.shape[1] > 0 else float("nan")

        tau_vals: List[float] = []
        pair_acc_vals: List[float] = []
        rel_lag_abs_err_ms: List[float] = []
        for i in range(new_rank.shape[1]):
            mask_i = legacy_bool[:, i]
            idx = np.where(mask_i)[0]
            if idx.size == 0:
                continue
            rel_legacy = legacy_lag_raw[idx, i] - float(np.min(legacy_lag_raw[idx, i]))
            rel_new = centroids[idx, i] - float(np.min(centroids[idx, i]))
            rel_lag_abs_err_ms.extend(np.abs((rel_new - rel_legacy) * 1000.0).tolist())
            if idx.size < 2:
                continue
            tau = kendalltau(legacy_lag_rank[idx, i], new_rank[idx, i]).statistic
            tau_vals.append(1.0 if (tau is None or np.isnan(tau)) else float(tau))
            total = 0
            ok = 0
            for a in range(idx.size):
                for b in range(a + 1, idx.size):
                    total += 1
                    if np.sign(legacy_lag_rank[idx[a], i] - legacy_lag_rank[idx[b], i]) == np.sign(
                        new_rank[idx[a], i] - new_rank[idx[b], i]
                    ):
                        ok += 1
            if total > 0:
                pair_acc_vals.append(float(ok) / float(total))

        a5_passed = full_match_rate >= 0.95 and (
            np.isnan(partic_rate) or partic_rate >= 0.99
        )
        a5 = CheckResult(
            name="A5_lagPatRank",
            passed=bool(a5_passed),
            metrics={
                "full_event_match_rate": full_match_rate,
                "participating_only_match_rate": partic_rate,
                "participating_event_exact_rate": partic_event_exact,
                "participating_kendall_tau_median": float(np.median(tau_vals)) if tau_vals else float("nan"),
                "participating_kendall_tau_p05": float(np.percentile(tau_vals, 5)) if tau_vals else float("nan"),
                "participating_pairwise_order_accuracy_median": float(np.median(pair_acc_vals)) if pair_acc_vals else float("nan"),
                "participating_pairwise_order_accuracy_p05": float(np.percentile(pair_acc_vals, 5)) if pair_acc_vals else float("nan"),
                "relative_lag_abs_err_ms_median": float(np.median(rel_lag_abs_err_ms)) if rel_lag_abs_err_ms else float("nan"),
                "relative_lag_abs_err_ms_p95": float(np.percentile(rel_lag_abs_err_ms, 95)) if rel_lag_abs_err_ms else float("nan"),
                "n_events": float(legacy_lag_rank.shape[1]),
                "n_picked": float(legacy_lag_rank.shape[0]),
            },
        )

    debug = {
        "n_segments_used": float(len(centroids_list)),
        "total_stitched_samples": float(total_stitched_samples),
        "x_band_sfreq": LEGACY_RESAMPLE_TO,
        "n_picked_loaded": float(len(picked_names)),
    }
    return a4, a5, debug


# ---------------------------------------------------------------------------
# Per-block driver
# ---------------------------------------------------------------------------


def find_legacy_files(subject: str, block: str) -> Dict[str, Path]:
    base = DATA_ROOT / subject
    paths = {
        "edf": base / f"{block}.edf",
        "gpu": base / f"{block}_gpu.npz",
        "lagpat": base / f"{block}_lagPat.npz",
        "packed": base / f"{block}_packedTimes.npy",
        "refine": base / "_refineGpu.npz",
    }
    for k, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"missing legacy {k}: {p}")
    return paths


def validate_block(subject: str, block: str) -> Dict[str, object]:
    import mne  # local import

    if subject not in LEGACY_SUBJECT_PARAMS:
        raise KeyError(f"no legacy params recorded for subject '{subject}'")
    params = LEGACY_SUBJECT_PARAMS[subject]
    paths = find_legacy_files(subject, block)
    print(f"\n=== Phase A: subject={subject} block={block} ===", flush=True)
    print(f"  pick_k={params['pick_k']}  pack_win_sec={params['pack_win_sec']}  drop={params['drop_chns']}", flush=True)

    refine = np.load(paths["refine"], allow_pickle=True)
    gpu = np.load(paths["gpu"], allow_pickle=True)
    packed = np.load(paths["packed"], allow_pickle=True)
    lag = np.load(paths["lagpat"], allow_pickle=True)

    refine_counts = np.asarray(refine["events_count"], dtype=np.float64)
    refine_names = np.asarray(refine["chns_names"]).astype(str)
    gpu_dets = np.asarray(gpu["whole_dets"], dtype=object)
    gpu_names = np.asarray(gpu["chns_names"]).astype(str)
    legacy_picks = [str(c) for c in lag["chnNames"]]
    legacy_packed = np.asarray(packed, dtype=np.float64)
    legacy_lag_raw = np.asarray(lag["lagPatRaw"], dtype=np.float64)
    legacy_lag_rank = np.asarray(lag["lagPatRank"], dtype=np.int64)
    legacy_bool = np.asarray(lag["eventsBool"], dtype=np.float64)

    t0 = time.time()
    a1 = check_a1_picked_channels(
        refine_counts=refine_counts,
        refine_names=refine_names,
        legacy_chn_names=legacy_picks,
        pick_k=float(params["pick_k"]),
    )
    print(f"  [A1] {a1.passed}  metrics={a1.metrics}", flush=True)
    if not a1.passed:
        print(f"  [A1] notes:\n{a1.notes}", flush=True)

    if not a1.passed:
        # Stop early per failure-handling contract
        return {
            "subject": subject,
            "block": block,
            "params": params,
            "results": {a1.name: _result_to_dict(a1)},
            "all_passed": False,
            "stopped_at": "A1",
        }

    picked_names = legacy_picks  # use legacy as authoritative for downstream

    # max recording time for packing window cropping
    max_event_t = 0.0
    for arr in gpu_dets:
        if arr is None:
            continue
        a = np.asarray(arr, dtype=np.float64)
        if a.size:
            max_event_t = max(max_event_t, float(a[:, 1].max()))
    raw_hdr = mne.io.read_raw_edf(str(paths["edf"]), preload=False, verbose=False, encoding="latin1")
    a2 = check_a2_packed_windows(
        gpu_dets=gpu_dets,
        gpu_names=gpu_names,
        picked_names=picked_names,
        legacy_packed=legacy_packed,
        pack_win_sec=float(params["pack_win_sec"]),
        t_max_sec=max_event_t + 5.0,
        record_last_sec=float(raw_hdr.times[-1]),
        record_sfreq=float(raw_hdr.info["sfreq"]),
    )
    print(f"  [A2] {a2.passed}  metrics={a2.metrics}", flush=True)
    if a2.notes:
        print(f"  [A2] notes: {a2.notes}", flush=True)

    a3 = check_a3_events_bool(
        gpu_dets=gpu_dets,
        gpu_names=gpu_names,
        picked_names=picked_names,
        legacy_packed=legacy_packed,
        legacy_events_bool=legacy_bool,
    )
    print(f"  [A3] {a3.passed}  metrics={a3.metrics}", flush=True)

    a4, a5, debug = check_a4_a5_centroid_and_rank(
        edf_path=paths["edf"],
        picked_names=picked_names,
        drop_chns=params["drop_chns"],
        legacy_packed=legacy_packed,
        legacy_lag_raw=legacy_lag_raw,
        legacy_lag_rank=legacy_lag_rank,
        legacy_events_bool=legacy_bool,
    )
    print(f"  [A4] {a4.passed}  metrics={a4.metrics}", flush=True)
    if a4.notes:
        print(f"  [A4] notes: {a4.notes}", flush=True)
    print(f"  [A5] {a5.passed}  metrics={a5.metrics}", flush=True)
    if a5.notes:
        print(f"  [A5] notes: {a5.notes}", flush=True)

    elapsed = time.time() - t0
    print(f"  elapsed={elapsed:.1f}s", flush=True)

    all_passed = all(r.passed for r in (a1, a2, a3, a4, a5))
    return {
        "subject": subject,
        "block": block,
        "params": params,
        "elapsed_sec": elapsed,
        "results": {r.name: _result_to_dict(r) for r in (a1, a2, a3, a4, a5)},
        "debug": debug,
        "all_passed": bool(all_passed),
        "stopped_at": None,
    }


def _result_to_dict(r: CheckResult) -> Dict[str, object]:
    return {
        "passed": r.passed,
        "metrics": r.metrics,
        "notes": r.notes,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


REFERENCE_SET = [
    ("gaolan", "FA0013KP"),
    ("dongyiming", "FA134D2R"),
    ("wangyiyang", "FA0012P5"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=str, default=None)
    parser.add_argument("--block", type=str, default=None)
    parser.add_argument(
        "--reference-set",
        action="store_true",
        help="Run the canonical reference subjects (gaolan/dongyiming/wangyiyang first block).",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=OUTPUT_ROOT,
        help="Output directory (default: results/validation/phaseA)",
    )
    args = parser.parse_args()

    if args.reference_set:
        targets = REFERENCE_SET
    else:
        if not (args.subject and args.block):
            parser.error("must specify either --reference-set or both --subject and --block")
        targets = [(args.subject, args.block)]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_lines = ["# Phase A — Algorithm Identity Validation\n"]
    summary_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    summary_lines.append(
        "Compares the new packing + spec-centroid pipeline against legacy "
        "`_lagPat.npz` / `_packedTimes.npy` using the *exact same* legacy "
        "`_gpu.npz` / `_refineGpu.npz` inputs.\n"
    )

    all_results = []
    for subject, block in targets:
        try:
            res = validate_block(subject, block)
        except Exception as exc:  # surface failures into the report instead of crashing
            print(f"ERROR validating {subject}/{block}: {exc}", file=sys.stderr, flush=True)
            res = {
                "subject": subject,
                "block": block,
                "all_passed": False,
                "error": repr(exc),
            }
        all_results.append(res)
        out_path = args.out_dir / f"{subject}__{block}.json"
        out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  wrote {out_path}", flush=True)

        # Markdown row for this block
        summary_lines.append(f"\n## {subject} / {block}\n")
        if "error" in res:
            summary_lines.append(f"- ERROR: `{res['error']}`\n")
            continue
        params = res.get("params", {})
        summary_lines.append(
            f"- pick_k = `{params.get('pick_k')}`  pack_win_sec = `{params.get('pack_win_sec')}`\n"
        )
        if res.get("stopped_at"):
            summary_lines.append(f"- STOPPED at `{res['stopped_at']}` (per failure-handling contract)\n")
        summary_lines.append("\n| Check | Passed | Key Metrics |\n|---|---|---|\n")
        for name, r in res.get("results", {}).items():
            metrics_str = ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in r["metrics"].items())
            mark = "PASS" if r["passed"] else "FAIL"
            summary_lines.append(f"| `{name}` | **{mark}** | {metrics_str} |\n")
        summary_lines.append(f"\n**Overall: {'PASS' if res.get('all_passed') else 'FAIL'}**\n")

    summary_path = args.out_dir / "SUMMARY.md"
    summary_path.write_text("".join(summary_lines), encoding="utf-8")
    print(f"\nWrote summary {summary_path}", flush=True)

    # Exit nonzero if any failed (CI hook)
    if any(not r.get("all_passed", False) for r in all_results):
        print("\n[!] At least one block FAILED Phase A.", flush=True)
        return 1
    print("\n[OK] All Phase A checks passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
