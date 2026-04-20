"""
Phase E §9.3 — Yuquan lagPat / packedTimes batch backfill.

For the 11 yuquan backfill subjects (3 reference + 8 backfill-only), generate
per-record `_lagPat.npz` + `_packedTimes.npy` from the Phase D-aligned
`results/hfo_detection/<subject>/{<record>_gpu.npz, _refineGpu.npz}` and the
raw EDF, and write them back into
`/mnt/yuquan_data/yuquan_24h_edf/<subject>/`.

Why a dedicated script (and not `compute_and_save_group_analysis`)
-----------------------------------------------------------------
`compute_and_save_group_analysis` builds per-window centroids on the raw
`band_filt(edf)`. Legacy yuquan lagPatRaw uses a *stitched-per-200s-segment*
spectrogram timeline (centroid time relative to the concatenated segment
buffer, not the per-window axis). That semantic is implemented by
`compute_stitched_spectrogram_centroids_legacy` and exercised by Phase A's
`scripts/validate_pack_against_legacy.py`, but `compute_and_save_group_analysis`
has no toggle to switch into that mode. Rather than overload the public
API, this script directly reuses the same low-level helpers the Phase A
validator uses, so the batch output is byte-for-byte the same code path
that was proven against three legacy reference subjects.

Output schema (matches legacy):
  <record>_packedTimes.npy : (n_events, 2) float64, [start_sec, end_sec]
  <record>_lagPat.npz      : lagPatRaw   (n_picked, n_events) float64 stitched-time
                             lagPatRank  (n_picked, n_events) int64
                             eventsBool  (n_picked, n_events) float64 (0/1)
                             chnNames    (n_picked,) <U dtype, alias-left names
                             start_t     () float64, EDF meas_date epoch sec

Failure contract
----------------
- atomic write: <path>.tmp -> os.replace(path)
- if any *_lagPat.npz / *_packedTimes.npy already exists in the subject raw
  dir, all such files are first moved to `<subject>/.legacy_backup/<basename>`
  (one-time per subject; manifest recorded in
  `results/lagpat_backfill/<subject>/manifest.json`).
- alias collision (e.g. `A1-A2` and `A2-A3` both alias to `A2` after stripping
  the bipolar pair): keep the entry with higher subject-level events_count.
  Both originals are recorded in the manifest.
- dry-run mode: write to a separate `--dry-run-out-dir` and never touch raw.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
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
    compute_stitched_spectrogram_centroids_legacy,
    filter_windows_for_legacy_segment_loop,
)
from src.utils.bqk_utils import band_filt, notch_filt  # noqa: E402

# ---------------------------------------------------------------------------
# Legacy globals (from `P16_packGroupEvents_per2h_showSpecs_bipolar_refine_bool.py`
# and the Phase A validator). Kept in sync with `validate_pack_against_legacy.py`.
# ---------------------------------------------------------------------------

LEGACY_SEGMENT_TIME = 200.0  # s
LEGACY_RESAMPLE_TO = 800.0  # Hz
LEGACY_NOTCH_FREQS = np.arange(50.0, 251.0, 50.0)
LEGACY_HIGHPASS_BAND = (80.0, 250.0)
LEGACY_EXT_MS = 30.0
LEGACY_CHNS_THR = 0.5
LEGACY_TIME_AXIS_HZ = 500.0
LEGACY_PACKING_GAP_LIMIT_S = 2.0
LEGACY_SPEC_FREQ_RANGE = (50.0, 300.0)
LEGACY_SPEC_NPERSEG_S = 0.05
LEGACY_SPEC_NOVERLAP_RATIO = 0.8
LEGACY_GAUSSIAN_SIGMA = 1.5
LEGACY_CENTROID_POWER = 3.0
LEGACY_PACKED_BOOL_FS = 500.0

# ---------------------------------------------------------------------------
# Per-subject packing parameters (legacy P16 __main__ block, both commented and
# uncommented sub_pickT_list / sub_packWL_list entries unioned).
# ---------------------------------------------------------------------------
LEGACY_SUBJECT_PARAMS: Dict[str, Dict[str, float]] = {
    # Pre-existing-lagPat reference subjects (from sub_packWL_list active block)
    "chenziyang":   {"pick_k": 1.0, "pack_win_sec": 0.300},
    "hanyuxuan":    {"pick_k": 1.0, "pack_win_sec": 0.300},
    "huanghanwen":  {"pick_k": 1.0, "pack_win_sec": 0.200},
    "litengsheng":  {"pick_k": 1.0, "pack_win_sec": 0.300},
    "xuxinyi":      {"pick_k": 0.7, "pack_win_sec": 0.200},
    "zhangjinhan":  {"pick_k": 1.0, "pack_win_sec": 0.200},
    "sunyuanxin":   {"pick_k": 1.0, "pack_win_sec": 0.250},
    "gaolan":       {"pick_k": 1.9, "pack_win_sec": 0.300},
    "wangyiyang":   {"pick_k": 1.0, "pack_win_sec": 0.250},
    "dongyiming":   {"pick_k": 0.5, "pack_win_sec": 0.220},
    # Backfill-only subjects (from commented block)
    "zhangkexuan":  {"pick_k": 0.5, "pack_win_sec": 0.500},
    "pengzihang":   {"pick_k": 1.0, "pack_win_sec": 0.500},
    "chengshuai":   {"pick_k": 1.0, "pack_win_sec": 0.500},
    "huangwanling": {"pick_k": 3.0, "pack_win_sec": 0.300},
    "liyouran":     {"pick_k": 1.0, "pack_win_sec": 0.250},
    "songzishuo":   {"pick_k": 1.0, "pack_win_sec": 0.300},
    "zhangbichen":  {"pick_k": 0.5, "pack_win_sec": 0.300},
    "zhangjiaqi":   {"pick_k": 1.7, "pack_win_sec": 0.150},
    "zhaochenxi":   {"pick_k": 0.5, "pack_win_sec": 0.300},
    "zhaojinrui":   {"pick_k": 1.0, "pack_win_sec": 0.300},
    "zhourongxuan": {"pick_k": 1.0, "pack_win_sec": 0.200},
}

# Per-subject drop_chns. Legacy P16
# (`yuquan_24h_perPatientAnalysis_dropRef/P16_packGroupEvents_per2h_showSpecs_bipolar_refine_bool.py`)
# uses a *global empty* drop_chns (`drop_chns = np.array([])` at module top, the
# only commented-out value being `np.array(['A8'])`). There is no per-subject
# drop table for yuquan in the legacy code path. Keep it empty for all subjects
# to match legacy. Phase A on `gaolan` was incidentally validated with
# `["B'4"]` in the validator, but `B'4` was never picked anyway, so the
# validator passing is not evidence for a non-empty drop list.
SUBJECT_DROP_CHNS: Dict[str, List[str]] = {s: [] for s in LEGACY_SUBJECT_PARAMS}

DATA_ROOT = Path("/mnt/yuquan_data/yuquan_24h_edf")
DETECT_ROOT = REPO_ROOT / "results" / "hfo_detection"
RESULTS_ROOT = REPO_ROOT / "results" / "lagpat_backfill"


# ---------------------------------------------------------------------------
# Inline copies of legacy helpers. Kept identical to the Phase A validator so
# the validator's pass/fail story applies to this script verbatim.
# ---------------------------------------------------------------------------

import re

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
    from collections import Counter
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


def _legacy_resample_notch_band(data: np.ndarray, fs_in: float) -> np.ndarray:
    factor_down = int(round(2.0 * fs_in / LEGACY_RESAMPLE_TO))
    if factor_down <= 0:
        raise ValueError(f"input sfreq {fs_in} cannot be resampled to {LEGACY_RESAMPLE_TO}")
    rs = scipy.signal.resample_poly(data, 2, factor_down, axis=-1)
    rs = notch_filt(rs, LEGACY_RESAMPLE_TO, LEGACY_NOTCH_FREQS)
    return band_filt(rs, LEGACY_RESAMPLE_TO, LEGACY_HIGHPASS_BAND)


def _legacy_get_packed_events_bool(
    high_events_times: List[np.ndarray],
    packed_times: np.ndarray,
    fs: float,
) -> np.ndarray:
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
            i0 = int(s * fs); i1 = int(e * fs)
            i0 = max(0, min(n_samp, i0)); i1 = max(0, min(n_samp, i1))
            if i1 > i0:
                index_matrix[chi, i0:i1] = 1
    for ti, (s, e) in enumerate(packed_times):
        i0 = int(s * fs); i1 = int(e * fs)
        i0 = max(0, min(n_samp, i0)); i1 = max(0, min(n_samp, i1))
        if i1 <= i0:
            continue
        for chi in range(n_ch):
            if index_matrix[chi, i0:i1].sum() > 0:
                bool_matrix[chi, ti] = 1.0
    return bool_matrix


# ---------------------------------------------------------------------------
# Alias bipolar to left + collision arbitration
# ---------------------------------------------------------------------------


@dataclass
class AliasMap:
    alias: str          # e.g. 'A1'
    orig: str           # e.g. 'A1-A2' (the chosen winner if collision)
    counts: int
    losers: List[Dict[str, int]] = field(default_factory=list)  # other 'A?-A?' losing the tie


def _alias_left(name: str) -> str:
    """`'A1-A2'` -> `'A1'`, `"A'1-A'2"` -> `"A'1"`. Names without `'-'` returned as-is."""
    return name.split("-")[0] if "-" in name else name


def alias_bipolar_to_left_with_arbitration(
    chns_names: Sequence[str], counts: np.ndarray
) -> Tuple[Dict[str, AliasMap], List[Dict[str, int]]]:
    """Collapse `'A1-A2'`-style names into alias-left `'A1'`. On collision keep
    the entry with higher events_count.

    Returns
    -------
    alias_map : dict alias -> AliasMap
    collisions : list of collision records (for QC manifest)
    """
    if len(chns_names) != len(counts):
        raise ValueError("chns_names / counts length mismatch")
    aliases: Dict[str, AliasMap] = {}
    for orig, c in zip(chns_names, counts.tolist()):
        a = _alias_left(str(orig))
        if a not in aliases:
            aliases[a] = AliasMap(alias=a, orig=str(orig), counts=int(c))
        else:
            cur = aliases[a]
            new_loser = {"orig": str(orig), "counts": int(c)}
            if int(c) > cur.counts:
                cur.losers.append({"orig": cur.orig, "counts": cur.counts})
                cur.orig = str(orig)
                cur.counts = int(c)
            else:
                cur.losers.append(new_loser)
    collisions = []
    for a, am in aliases.items():
        if am.losers:
            collisions.append({
                "alias": a,
                "winner": {"orig": am.orig, "counts": am.counts},
                "losers": am.losers,
            })
    return aliases, collisions


# ---------------------------------------------------------------------------
# Stitched-segment lagPat computation (mirrors validator check_a4_a5)
# ---------------------------------------------------------------------------


def compute_stitched_lagpat(
    edf_path: Path,
    picked_alias_names: Sequence[str],
    drop_chns: Sequence[str],
    packed_times: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (lagPatRaw, lagPatRank) of shape (n_picked, n_events).

    lagPatRaw is on the *stitched-per-200s-segment* timeline (legacy semantic).
    Events whose [start, end] are not fully contained in any single 200s segment
    are silently dropped by the legacy loop and yield NaN rows in lagPatRaw and
    arbitrary rank rows. We follow the same convention.
    """
    import mne

    raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False, encoding="latin1")
    fs_in = float(raw.info["sfreq"])
    valid_idx = _legacy_valid_chan_index(raw.ch_names)
    if valid_idx.size == 0:
        raise RuntimeError(f"no valid channels in {edf_path}")
    valid_names = [_standard_name(raw.ch_names[i]) for i in valid_idx]

    n_pick = len(picked_alias_names)
    n_ev = packed_times.shape[0]
    centroids = np.full((n_pick, n_ev), np.nan, dtype=np.float64)

    time_inter = np.arange(0.0, float(raw.times[-1]), LEGACY_SEGMENT_TIME, dtype=np.float64)
    time_inter = np.append(time_inter, float(raw.times[-1]))
    time_ranges = np.stack([time_inter[:-1], time_inter[1:]], axis=1)

    for tr in time_ranges:
        seg_start_idx, seg_end_idx = raw.time_as_index(tuple(float(x) for x in tr))
        seg_start_idx = int(seg_start_idx); seg_end_idx = int(seg_end_idx)
        if seg_end_idx <= seg_start_idx:
            continue
        seg_time = raw.times[seg_start_idx:seg_end_idx]
        if seg_time.size == 0 or float(seg_time[-1] - seg_time[0]) < 5.0:
            continue
        in_seg_idx = np.where(
            (packed_times[:, 0] >= float(seg_time[0]))
            & (packed_times[:, 1] <= float(seg_time[-1]))
        )[0]
        if in_seg_idx.size == 0:
            continue
        seg_data = raw.get_data(picks=valid_idx.tolist(), start=seg_start_idx, stop=seg_end_idx)
        seg_data, bipolar_names = _legacy_bipolar_reref_and_drop(seg_data, valid_names, drop_chns)
        bipolar_index = {str(n): i for i, n in enumerate(bipolar_names)}
        miss = [n for n in picked_alias_names if n not in bipolar_index]
        if miss:
            raise RuntimeError(
                f"picked channels not found after bipolar reref/drop in seg {tr.tolist()}: {miss}"
            )
        pick_rows = np.array([bipolar_index[n] for n in picked_alias_names], dtype=int)
        seg_pick = seg_data[pick_rows]
        seg_band = _legacy_resample_notch_band(seg_pick, fs_in)
        seg_windows = [
            EventWindow(float(packed_times[i, 0]), float(packed_times[i, 1]), int(i))
            for i in in_seg_idx
        ]
        stitched, split_border_t = build_stitched_window_signal(
            seg_band, seg_windows, sfreq=LEGACY_RESAMPLE_TO, start_sec=float(seg_time[0])
        )
        if stitched.shape[1] == 0 or split_border_t.size == 0:
            continue
        seg_centroids = compute_stitched_spectrogram_centroids_legacy(
            stitched, split_border_t,
            sfreq=LEGACY_RESAMPLE_TO,
            spec_freq_range=LEGACY_SPEC_FREQ_RANGE,
            spec_nperseg_sec=LEGACY_SPEC_NPERSEG_S,
            spec_noverlap_ratio=LEGACY_SPEC_NOVERLAP_RATIO,
            gaussian_sigma=LEGACY_GAUSSIAN_SIGMA,
            centroid_power=LEGACY_CENTROID_POWER,
        )
        for col_in_seg, ev_idx in enumerate(in_seg_idx):
            centroids[:, ev_idx] = seg_centroids[:, col_in_seg]

    # Rank via argsort(argsort) per event. NaN columns -> rank still computed
    # over NaN, which numpy returns as positions. We emit zeros in such columns
    # to match legacy (legacy never wrote rank for skipped events; downstream
    # consumers always mask via eventsBool).
    rank = np.zeros((n_pick, n_ev), dtype=np.int64)
    for i in range(n_ev):
        col = centroids[:, i]
        if np.all(np.isnan(col)):
            continue
        rank[:, i] = np.argsort(np.argsort(np.where(np.isnan(col), np.inf, col)))
    return centroids, rank


# ---------------------------------------------------------------------------
# Per-record packing
# ---------------------------------------------------------------------------


def pack_one_record(
    *,
    subject: str,
    edf_path: Path,
    gpu_npz_path: Path,
    refine_npz_path: Path,
    pack_pick_k: float,
    pack_win_sec: float,
    drop_chns: Sequence[str],
    out_dir: Path,
    backup_dir: Optional[Path] = None,
) -> Dict[str, object]:
    import mne
    raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False, encoding="latin1")
    fs_in = float(raw.info["sfreq"])
    record_last_sec = float(raw.times[-1])
    meas_date = raw.info.get("meas_date")
    start_t = float(meas_date.timestamp()) if meas_date is not None else 0.0

    # 1. Load subject-level _refineGpu.npz (event_count over alias-left set)
    refine = np.load(refine_npz_path, allow_pickle=True)
    refine_counts_orig = refine["events_count"].astype(np.int64)
    refine_names_orig = [str(x) for x in refine["chns_names"]]

    aliases, collisions = alias_bipolar_to_left_with_arbitration(
        refine_names_orig, refine_counts_orig
    )
    alias_keys = sorted(aliases.keys())
    alias_counts = np.array([aliases[a].counts for a in alias_keys], dtype=np.int64)

    # 2. Pick channels by mean + pack_pick_k * std
    counts_f = alias_counts.astype(np.float64)
    thr = float(counts_f.mean() + pack_pick_k * counts_f.std())
    pick_mask = counts_f > thr
    picked_alias_names = [alias_keys[i] for i in np.where(pick_mask)[0]]
    if len(picked_alias_names) < 2:
        return {
            "status": "skipped",
            "reason": f"only {len(picked_alias_names)} picked channels under thr={thr:.4g}",
            "n_picked": len(picked_alias_names),
            "alias_collisions": collisions,
        }

    # 3. Per-record _gpu.npz: load whole_dets and look up by *original* name
    gpu = np.load(gpu_npz_path, allow_pickle=True)
    gpu_names_orig = [str(x) for x in gpu["chns_names"]]
    gpu_dets = gpu["whole_dets"]
    gpu_name_to_idx = {n: i for i, n in enumerate(gpu_names_orig)}

    dets: Dict[str, np.ndarray] = {}
    missing_in_gpu: List[str] = []
    for alias in picked_alias_names:
        orig = aliases[alias].orig
        if orig in gpu_name_to_idx:
            dets[alias] = np.asarray(gpu_dets[gpu_name_to_idx[orig]], dtype=np.float64)
        else:
            dets[alias] = np.zeros((0, 2), dtype=np.float64)
            missing_in_gpu.append(orig)

    # 4. Build packed windows from picked-channel detections
    new_windows = build_windows_from_detections(
        dets,
        window_sec=float(pack_win_sec),
        ext_ms=LEGACY_EXT_MS,
        chns_thr=LEGACY_CHNS_THR,
        time_axis_hz=LEGACY_TIME_AXIS_HZ,
        max_window_sec=LEGACY_PACKING_GAP_LIMIT_S,
        t_max_sec=record_last_sec,
    )
    new_windows = filter_windows_for_legacy_segment_loop(
        new_windows,
        segment_duration_sec=LEGACY_SEGMENT_TIME,
        record_last_sec=record_last_sec,
        sfreq=fs_in,
        start_sec=0.0,
    )
    if not new_windows:
        return {
            "status": "skipped",
            "reason": "no packed windows after segment filter",
            "n_picked": len(picked_alias_names),
            "alias_collisions": collisions,
        }
    packed_times = np.array([[w.start, w.end] for w in new_windows], dtype=np.float64)
    n_events = packed_times.shape[0]

    # 5. eventsBool over picked channels
    dets_list_in_pick_order = [dets[nm] for nm in picked_alias_names]
    events_bool = _legacy_get_packed_events_bool(
        dets_list_in_pick_order, packed_times, fs=LEGACY_PACKED_BOOL_FS
    )

    # 6. Stitched-segment lagPat
    lag_raw, lag_rank = compute_stitched_lagpat(
        edf_path=edf_path,
        picked_alias_names=picked_alias_names,
        drop_chns=drop_chns,
        packed_times=packed_times,
    )

    # 7. Backup pre-existing files (one-time per subject — caller responsibility
    # to make sure backup_dir is the target subject's `.legacy_backup`)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_lagpat = out_dir / f"{edf_path.stem}_lagPat.npz"
    out_packed = out_dir / f"{edf_path.stem}_packedTimes.npy"
    if backup_dir is not None:
        backup_dir.mkdir(parents=True, exist_ok=True)
        for src in (out_lagpat, out_packed):
            if src.exists() and not (backup_dir / src.name).exists():
                os.rename(str(src), str(backup_dir / src.name))

    # 8. Atomic write. np.savez/save auto-append .npz/.npy when the path does
    # not end with the corresponding suffix; we put .tmp *before* the suffix
    # so the final file on disk matches the path we hand to os.replace.
    tmp_lag = out_lagpat.with_name(out_lagpat.stem + ".tmp.npz")
    tmp_packed = out_packed.with_name(out_packed.stem + ".tmp.npy")
    np.savez(
        str(tmp_lag),
        lagPatRaw=lag_raw.astype(np.float64),
        lagPatRank=lag_rank.astype(np.int64),
        eventsBool=events_bool.astype(np.float64),
        chnNames=np.array(picked_alias_names),
        start_t=np.float64(start_t),
    )
    np.save(str(tmp_packed), packed_times)
    os.replace(str(tmp_lag), str(out_lagpat))
    os.replace(str(tmp_packed), str(out_packed))

    return {
        "status": "ok",
        "record": edf_path.stem,
        "n_picked": len(picked_alias_names),
        "n_events": int(n_events),
        "n_events_with_finite_lag": int(np.isfinite(lag_raw).all(axis=0).sum()),
        "alias_collisions": collisions,
        "missing_in_gpu_npz": missing_in_gpu,
        "out_lagpat": str(out_lagpat),
        "out_packed": str(out_packed),
        "pack_pick_k": pack_pick_k,
        "pack_win_sec": pack_win_sec,
        "picked_chnNames": picked_alias_names,
        "start_t_epoch": start_t,
    }


# ---------------------------------------------------------------------------
# Subject driver
# ---------------------------------------------------------------------------


def run_subject(
    subject: str,
    *,
    dry_run_out_dir: Optional[Path] = None,
    only_records: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    if subject not in LEGACY_SUBJECT_PARAMS:
        raise KeyError(f"subject {subject} missing from LEGACY_SUBJECT_PARAMS")
    params = LEGACY_SUBJECT_PARAMS[subject]
    drop_chns = SUBJECT_DROP_CHNS.get(subject, [])

    raw_dir = DATA_ROOT / subject
    detect_dir = DETECT_ROOT / subject
    refine_npz = detect_dir / "_refineGpu.npz"
    if not refine_npz.exists():
        raise FileNotFoundError(f"missing {refine_npz}")

    if dry_run_out_dir is not None:
        out_dir = dry_run_out_dir / subject
        backup_dir = None  # never touch raw in dry-run
    else:
        out_dir = raw_dir
        backup_dir = raw_dir / ".legacy_backup"

    edfs = sorted(raw_dir.glob("*.edf"))
    if only_records:
        keep = set(only_records)
        edfs = [e for e in edfs if e.stem in keep]
    if not edfs:
        raise FileNotFoundError(f"no EDF found in {raw_dir} (only_records={only_records})")

    records: List[Dict[str, object]] = []
    t_start = time.time()
    for i, edf in enumerate(edfs, 1):
        gpu_npz = detect_dir / f"{edf.stem}_gpu.npz"
        if not gpu_npz.exists():
            records.append({
                "record": edf.stem, "status": "missing_gpu_npz",
                "gpu_npz": str(gpu_npz),
            })
            print(f"  [{i}/{len(edfs)}] {edf.stem}: MISSING gpu_npz")
            continue
        try:
            t0 = time.time()
            r = pack_one_record(
                subject=subject,
                edf_path=edf,
                gpu_npz_path=gpu_npz,
                refine_npz_path=refine_npz,
                pack_pick_k=params["pick_k"],
                pack_win_sec=params["pack_win_sec"],
                drop_chns=drop_chns,
                out_dir=out_dir,
                backup_dir=backup_dir,
            )
            r["elapsed_sec"] = round(time.time() - t0, 1)
            records.append(r)
            status = r["status"]
            extra = ""
            if status == "ok":
                extra = f"  n_pick={r['n_picked']:>3} n_ev={r['n_events']:>5} elapsed={r['elapsed_sec']}s"
            elif status == "skipped":
                extra = f"  reason={r.get('reason','?')}"
            print(f"  [{i}/{len(edfs)}] {edf.stem}: {status}{extra}")
        except Exception as e:
            records.append({"record": edf.stem, "status": "error", "error": repr(e)})
            print(f"  [{i}/{len(edfs)}] {edf.stem}: ERROR {e!r}")

    summary = {
        "subject": subject,
        "params": params,
        "drop_chns": drop_chns,
        "n_records_total": len(edfs),
        "n_records_ok": sum(1 for r in records if r.get("status") == "ok"),
        "n_records_skipped": sum(1 for r in records if r.get("status") == "skipped"),
        "n_records_error": sum(1 for r in records if r.get("status") == "error"),
        "n_records_missing_gpu_npz": sum(1 for r in records if r.get("status") == "missing_gpu_npz"),
        "out_dir": str(out_dir),
        "backup_dir": str(backup_dir) if backup_dir else None,
        "elapsed_sec_total": round(time.time() - t_start, 1),
        "records": records,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase E §9.3 lagPat backfill")
    parser.add_argument("--subject", action="append", required=True,
                        help="Subject id (repeat for multiple subjects).")
    parser.add_argument("--dry-run-out-dir", type=str, default=None,
                        help="If set, write to this directory instead of raw "
                             "subject dir; legacy files in raw dir are NOT moved.")
    parser.add_argument("--records", type=str, default=None,
                        help="Comma-separated EDF stems (e.g. FA0012P5,FA0012P6) "
                             "to limit the per-subject loop. Useful for prototyping.")
    parser.add_argument("--summary-out", type=str, default=None,
                        help="Where to write per-subject summary JSON. "
                             "Default: results/lagpat_backfill/<subject>/summary.json "
                             "(or <dry-run-out-dir>/<subject>/summary.json)")
    args = parser.parse_args()

    only_records = tuple(args.records.split(",")) if args.records else None
    dry_dir = Path(args.dry_run_out_dir) if args.dry_run_out_dir else None

    rc = 0
    for subject in args.subject:
        print(f"\n=========== subject={subject} ===========")
        try:
            summary = run_subject(subject, dry_run_out_dir=dry_dir, only_records=only_records)
        except Exception as e:
            print(f"FATAL subject={subject}: {e!r}")
            rc = 1
            continue

        out_root = dry_dir if dry_dir is not None else RESULTS_ROOT
        summary_path = (
            Path(args.summary_out) if args.summary_out
            else (out_root / subject / "summary.json")
        )
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"summary: {summary_path}")
        print(f"  ok={summary['n_records_ok']}/{summary['n_records_total']}  "
              f"skipped={summary['n_records_skipped']}  "
              f"err={summary['n_records_error']}  "
              f"miss_gpu={summary['n_records_missing_gpu_npz']}  "
              f"elapsed={summary['elapsed_sec_total']}s")

    return rc


if __name__ == "__main__":
    sys.exit(main())
