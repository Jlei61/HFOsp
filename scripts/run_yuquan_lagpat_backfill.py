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
from typing import Callable, Dict, List, Optional, Sequence, Tuple

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
# Per-subject packing parameters are sourced from `config/subject_params.json`
# (yuquan section). Required pack-stage fields per subject:
#   - pick_k             pack-stage mean+k*std threshold  (legacy sub_pickT_list)
#   - pack_win_sec       packing window length            (legacy sub_packWL_list)
#   - pack_drop_channels packing-stage bipolar drop list  (legacy module-level
#                        `drop_chns`, globally empty for yuquan)
#   - pack_top_n         optional: explicit cardinality cap (must be documented
#                        in docs/archive/yuquan_lagpat/, not introduced silently)
# `_defaults` provides the inheritable values; per-subject entries only override.
# Detector-stage `drop_channels` is NOT used for packing.
# ---------------------------------------------------------------------------

DATA_ROOT = Path("/mnt/yuquan_data/yuquan_24h_edf")
DETECT_ROOT = REPO_ROOT / "results" / "hfo_detection"
RESULTS_ROOT = REPO_ROOT / "results" / "lagpat_backfill"
SUBJECT_PARAMS_PATH = REPO_ROOT / "config" / "subject_params.json"

# 24-subject same-source Yuquan cohort (10 main cohort that have legacy lagPat
# under the legacy 30-subject pipeline + 3 reference subjects with legacy lagPat
# + 8 backfill-only subjects that the legacy pipeline skipped). All must have
# an entry under `config/subject_params.json::yuquan` with pack-stage fields
# resolvable.
YUQUAN_SAME_SOURCE_SUBJECTS: Tuple[str, ...] = (
    # Reference subjects (have legacy lagPat AND legacy detection)
    "gaolan", "dongyiming", "wangyiyang",
    # Main cohort with legacy lagPat (legacy 30-subject set, yuquan side)
    "chenziyang", "hanyuxuan", "huanghanwen", "huangwanling", "litengsheng",
    "sunyuanxin", "xuxinyi", "zhangjinhan",
    # Backfill subjects (legacy detection ran but lagPat was missing/skipped)
    "chengshuai", "liyouran", "pengzihang", "songzishuo",
    "zhangbichen", "zhangjiaqi", "zhangkexuan", "zhaochenxi",
    "zhaojinrui", "zhourongxuan",
)

_PACK_REQUIRED_FIELDS = ("pick_k", "pack_win_sec", "pack_drop_channels")
_PACK_OPTIONAL_FIELDS = ("pack_top_n",)


def _load_subject_params_registry() -> Dict[str, Dict[str, object]]:
    if not SUBJECT_PARAMS_PATH.exists():
        raise FileNotFoundError(f"subject params config missing: {SUBJECT_PARAMS_PATH}")
    with SUBJECT_PARAMS_PATH.open("r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    if "yuquan" not in cfg:
        raise KeyError(f"no yuquan section in {SUBJECT_PARAMS_PATH}")
    return cfg["yuquan"]


def resolve_subject_pack_params(subject: str) -> Dict[str, object]:
    """Return resolved pack-stage params for `subject` from config.

    Raises KeyError if subject is missing from the yuquan section, and
    ValueError if any required pack-stage field is missing after merge.
    Only the pack-relevant subset is returned; the detector-stage
    `drop_channels` is intentionally excluded from pack params.
    """
    registry = _load_subject_params_registry()
    if subject not in registry:
        raise KeyError(f"subject '{subject}' missing from yuquan section of {SUBJECT_PARAMS_PATH}")
    if subject == "_defaults":
        raise KeyError("'_defaults' is not a real subject")
    defaults = dict(registry.get("_defaults", {}))
    merged = dict(defaults)
    merged.update(registry[subject])
    missing = [f for f in _PACK_REQUIRED_FIELDS if f not in merged]
    if missing:
        raise ValueError(f"subject '{subject}' pack params missing fields: {missing}")
    out: Dict[str, object] = {
        "pick_k": float(merged["pick_k"]),
        "pack_win_sec": float(merged["pack_win_sec"]),
        "pack_drop_channels": list(merged["pack_drop_channels"]),
    }
    for f in _PACK_OPTIONAL_FIELDS:
        if f in merged:
            out[f] = merged[f]
    return out


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
) -> Tuple[Dict[str, AliasMap], List[Dict[str, int]], List[Dict[str, object]]]:
    """Collapse `'A1-A2'`-style names into alias-left `'A1'`. On collision keep
    the entry with higher events_count. Then drop the *outermost* alias of each
    shaft so the resulting alias set is identical to the one produced by legacy
    `_legacy_bipolar_reref_and_drop` (whose chain
    `valid_chan_index`(drop max contact) -> `bipolar_rerefAndDrop_eeg`
    (drop new max contact) makes max alias = `max_edf_contact_num - 2`).

    The new detection pipeline pairs *every* contact, so a 14-contact shaft
    yields aliases up to `<pre>13`, but legacy only reaches `<pre>12`. Any
    `'<pre>13'` picked channel will be missing from the legacy bipolar output
    and crash `compute_stitched_lagpat`.

    Returns
    -------
    alias_map  : dict alias -> AliasMap (after outermost drop)
    collisions : list of collision records (for QC manifest)
    outer_drops: list of {alias, orig, counts, reason} for outermost-shaft drops
    """
    if len(chns_names) != len(counts):
        raise ValueError("chns_names / counts length mismatch")
    # Detect input format: legacy 2021-era refineGpu was already
    # bipolar-collapsed (single-electrode names like 'A1', 'A2'); the new
    # detector pipeline emits bipolar-pair names ('A1-A2', 'A2-A3') that
    # need both alias collapse + outermost-shaft drop. The outer drop is a
    # no-op (and can be incorrect, removing a legitimately picked channel)
    # when the input is already in single-electrode form.
    is_bipolar_pair_input = any("-" in str(n) for n in chns_names)
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

    outer_drops: List[Dict[str, object]] = []
    if is_bipolar_pair_input:
        # Bipolar-pair input ('A1-A2', ...): the alias set ends one contact
        # higher than the legacy bipolar reref output, so drop the highest
        # alias per shaft to align with `_legacy_bipolar_reref_and_drop`.
        by_pre: Dict[str, List[Tuple[int, str]]] = {}
        for a in aliases:
            try:
                pre, num = _split_chn(a)
            except ValueError:
                continue
            by_pre.setdefault(pre, []).append((int(num), a))
        for pre, items in by_pre.items():
            items.sort(key=lambda t: t[0])
            max_alias = items[-1][1]
            am = aliases.pop(max_alias)
            outer_drops.append({
                "alias": max_alias,
                "orig": am.orig,
                "counts": am.counts,
                "reason": "shaft_outermost_alias_not_in_legacy_bipolar",
            })

    return aliases, collisions, outer_drops


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


START_TIME_DELTA_THRESHOLD_SEC = 1.0


def _block_qc_metrics(events_bool: np.ndarray, lag_raw: np.ndarray) -> Dict[str, float]:
    """Per-block QC: median n_participating per event, median lag span (ms)."""
    if events_bool.size == 0 or events_bool.shape[1] == 0:
        return {"median_n_participating": float("nan"), "median_lag_span_ms": float("nan")}
    n_part = events_bool.sum(axis=0)  # (n_events,)
    median_n_part = float(np.median(n_part)) if n_part.size else float("nan")

    spans_ms: List[float] = []
    for i in range(lag_raw.shape[1]):
        col = lag_raw[:, i]
        finite = col[np.isfinite(col)]
        if finite.size >= 2:
            spans_ms.append(float((finite.max() - finite.min()) * 1000.0))
    median_lag_span_ms = float(np.median(spans_ms)) if spans_ms else float("nan")
    return {"median_n_participating": median_n_part, "median_lag_span_ms": median_lag_span_ms}


def pack_one_record(
    *,
    subject: str,
    edf_path: Path,
    gpu_npz_path: Path,
    aliases: Dict[str, AliasMap],
    pack_pick_k: float,
    pack_win_sec: float,
    pack_top_n: Optional[int],
    drop_chns: Sequence[str],
    out_dir: Path,
    backup_dir: Optional[Path] = None,
    start_time_threshold_sec: float = START_TIME_DELTA_THRESHOLD_SEC,
) -> Dict[str, object]:
    import mne
    raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False, encoding="latin1")
    fs_in = float(raw.info["sfreq"])
    record_last_sec = float(raw.times[-1])
    meas_date = raw.info.get("meas_date")
    edf_start_t = float(meas_date.timestamp()) if meas_date is not None else None

    # Per-record _gpu.npz: load whole_dets and look up by *original* name
    gpu = np.load(gpu_npz_path, allow_pickle=True)
    gpu_names_orig = [str(x) for x in gpu["chns_names"]]
    gpu_dets = gpu["whole_dets"]
    gpu_name_to_idx = {n: i for i, n in enumerate(gpu_names_orig)}
    gpu_start_t = float(gpu["start_time"]) if "start_time" in gpu.files else None

    # 0. start_time hard check: gpu.npz['start_time'] vs EDF meas_date.timestamp()
    start_time_validation: Dict[str, object] = {
        "gpu_npz_start_time": gpu_start_t,
        "edf_meas_date_epoch": edf_start_t,
        "delta_sec": (
            float(abs(gpu_start_t - edf_start_t))
            if (gpu_start_t is not None and edf_start_t is not None) else None
        ),
        "threshold_sec": float(start_time_threshold_sec),
        "passed": False,
    }
    if gpu_start_t is None or edf_start_t is None:
        start_time_validation["passed"] = False
        return {
            "record": edf_path.stem,
            "status": "skipped",
            "skip_reason": "start_time_missing",
            "start_time_validation": start_time_validation,
            "n_picked": 0,
            "wrote_lagpat": False,
            "wrote_packed": False,
        }
    if start_time_validation["delta_sec"] > start_time_threshold_sec:
        return {
            "record": edf_path.stem,
            "status": "skipped",
            "skip_reason": (
                f"start_time_mismatch delta={start_time_validation['delta_sec']:.3f}s "
                f"> {start_time_threshold_sec}s"
            ),
            "start_time_validation": start_time_validation,
            "n_picked": 0,
            "wrote_lagpat": False,
            "wrote_packed": False,
        }
    start_time_validation["passed"] = True
    start_t = edf_start_t  # canonical: legacy lagPat.start_t used EDF meas_date

    alias_keys = sorted(aliases.keys())
    alias_counts = np.array([aliases[a].counts for a in alias_keys], dtype=np.int64)

    # 1. Pick channels by mean + pack_pick_k * std
    counts_f = alias_counts.astype(np.float64)
    thr = float(counts_f.mean() + pack_pick_k * counts_f.std())
    pick_mask = counts_f > thr
    picked_alias_names = [alias_keys[i] for i in np.where(pick_mask)[0]]
    if pack_top_n is not None and len(picked_alias_names) > int(pack_top_n):
        picked_alias_names = sorted(
            picked_alias_names,
            key=lambda nm: (-aliases[nm].counts, nm),
        )[: int(pack_top_n)]
    n_collisions_in_picked = sum(
        1 for nm in picked_alias_names if aliases[nm].losers
    )
    if len(picked_alias_names) < 2:
        return {
            "record": edf_path.stem,
            "status": "skipped",
            "skip_reason": f"only {len(picked_alias_names)} picked channels under thr={thr:.4g}",
            "n_picked": len(picked_alias_names),
            "n_alias_collisions_in_picked": int(n_collisions_in_picked),
            "start_time_validation": start_time_validation,
            "wrote_lagpat": False,
            "wrote_packed": False,
        }

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
            "record": edf_path.stem,
            "status": "skipped",
            "skip_reason": "no packed windows after segment filter",
            "n_picked": len(picked_alias_names),
            "n_alias_collisions_in_picked": int(n_collisions_in_picked),
            "missing_in_gpu_npz": missing_in_gpu,
            "start_time_validation": start_time_validation,
            "wrote_lagpat": False,
            "wrote_packed": False,
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
    backup_events: List[Dict[str, str]] = []
    if backup_dir is not None:
        backup_dir.mkdir(parents=True, exist_ok=True)
        for src in (out_lagpat, out_packed):
            if src.exists() and not (backup_dir / src.name).exists():
                dst = backup_dir / src.name
                os.rename(str(src), str(dst))
                backup_events.append({"src": str(src), "dst": str(dst)})

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

    qc = _block_qc_metrics(events_bool, lag_raw)

    return {
        "record": edf_path.stem,
        "status": "ok",
        "skip_reason": None,
        "n_picked": len(picked_alias_names),
        "n_events": int(n_events),
        "n_events_with_finite_lag": int(np.isfinite(lag_raw).all(axis=0).sum()),
        "n_alias_collisions_in_picked": int(n_collisions_in_picked),
        "missing_in_gpu_npz": missing_in_gpu,
        "out_lagpat": str(out_lagpat),
        "out_packed": str(out_packed),
        "wrote_lagpat": True,
        "wrote_packed": True,
        "backup_events": backup_events,
        "pack_pick_k": float(pack_pick_k),
        "pack_win_sec": float(pack_win_sec),
        "pack_top_n": int(pack_top_n) if pack_top_n is not None else None,
        "picked_chnNames": picked_alias_names,
        "start_t_epoch": start_t,
        "start_time_validation": start_time_validation,
        "median_n_participating": qc["median_n_participating"],
        "median_lag_span_ms": qc["median_lag_span_ms"],
    }


# ---------------------------------------------------------------------------
# Subject driver
# ---------------------------------------------------------------------------


SCHEMA_VERSION = "yuquan_lagpat_backfill_v2_2026Q2"


def _file_stat(p: Path) -> Dict[str, object]:
    try:
        st = p.stat()
        return {"path": str(p), "size_bytes": int(st.st_size), "mtime": float(st.st_mtime)}
    except FileNotFoundError:
        return {"path": str(p), "size_bytes": None, "mtime": None}


def _legacy_block_presence_diff(
    raw_dir: Path, edf_stems: Sequence[str], records: Sequence[Dict[str, object]]
) -> Dict[str, object]:
    """Compare new-pipeline written blocks to pre-existing legacy lagPat presence.

    Legacy semantics — `.legacy_backup/` is the single source of truth once it
    exists. A subject's `.legacy_backup/` directory is created the first time
    `pack_one_record` writes into the raw dir, and from that moment forward
    `<raw_dir>/<stem>_lagPat.npz` is a *new-pipeline* artifact, not legacy.

    - If `.legacy_backup/` exists:
        legacy_present = stems with `<.legacy_backup>/<stem>_lagPat.npz`
    - Else (subject has never been backfilled):
        legacy_present = stems with `<raw_dir>/<stem>_lagPat.npz`

    This avoids the previous bug where the audit would read a v1 backfill
    output sitting in the raw dir as if it were legacy evidence.
    """
    backup_dir = raw_dir / ".legacy_backup"
    legacy_source: Path
    legacy_source_kind: str
    if backup_dir.exists():
        legacy_source = backup_dir
        legacy_source_kind = "legacy_backup_dir"
    else:
        legacy_source = raw_dir
        legacy_source_kind = "raw_dir_untouched"

    legacy_present: List[str] = []
    legacy_absent: List[str] = []
    for stem in edf_stems:
        if (legacy_source / f"{stem}_lagPat.npz").exists():
            legacy_present.append(stem)
        else:
            legacy_absent.append(stem)

    rec_by_stem = {r["record"]: r for r in records if "record" in r}

    def _bucket(stems: Sequence[str]) -> Dict[str, List[str]]:
        out = {"ok": [], "skipped": [], "missing_gpu_npz": [], "error": []}
        for s in stems:
            r = rec_by_stem.get(s)
            if r is None:
                out.setdefault("not_processed", []).append(s)
                continue
            st = r.get("status", "unknown")
            out.setdefault(st, []).append(s)
        return out

    legacy_present_status = _bucket(legacy_present)
    legacy_absent_status = _bucket(legacy_absent)
    return {
        "legacy_source_kind": legacy_source_kind,
        "legacy_source_path": str(legacy_source),
        "n_legacy_present": len(legacy_present),
        "n_legacy_absent": len(legacy_absent),
        "legacy_present_status": legacy_present_status,
        "legacy_absent_status": legacy_absent_status,
        # Red flag: legacy had it, new pipeline failed/skipped to write
        "regressions": [
            s for s in legacy_present
            if rec_by_stem.get(s, {}).get("status") not in ("ok",)
        ],
        # Informational: new pipeline wrote a block legacy didn't have
        "extras_written": [
            s for s in legacy_absent
            if rec_by_stem.get(s, {}).get("status") == "ok"
        ],
    }


# ---------------------------------------------------------------------------
# Path-injection layer (Track B replay reuse)
#
# `run_subject` is shared between the canonical same-source contract (refine
# + gpu both under DETECT_ROOT) and the legacy-refine replay (refine + gpu
# under <legacy_root>/<subject>/...). Path resolution is extracted into one
# helper so both call sites go through the same code and provenance fields
# end up in `summary.json` / `manifest.json` for the audit's provenance gate.
# ---------------------------------------------------------------------------

_SENTINEL: object = object()
_GpuResolver = Callable[[str, str], Path]


@dataclass(frozen=True)
class _ResolvedSubjectIO:
    raw_dir: Path
    refine_npz: Path
    gpu_resolver: _GpuResolver
    out_dir: Path
    backup_dir: Optional[Path]
    same_source_assertion: bool
    legacy_refine_root_recorded: Optional[str]
    legacy_gpu_root_recorded: Optional[str]


def _resolve_run_subject_io(
    subject: str,
    *,
    legacy_refine_root: Optional[Path] = None,
    legacy_gpu_root: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    backup_dir: object = _SENTINEL,
    same_source_assertion: bool = True,
    dry_run_out_dir: Optional[Path] = None,
) -> _ResolvedSubjectIO:
    """Resolve all I/O paths for one subject in one place.

    Behavior:

    - No overrides ⇒ same-source contract:
      * `refine_npz = DETECT_ROOT / subject / _refineGpu.npz`
      * `gpu_resolver(s, r) = DETECT_ROOT / s / f"{r}_gpu.npz"`
      * `out_dir = DATA_ROOT / subject`
      * `backup_dir = DATA_ROOT / subject / .legacy_backup`
      * `same_source_assertion = True`

    - `legacy_refine_root` and/or `legacy_gpu_root` set ⇒ paths are rerouted
      to `<root>/<subject>/...`. Same-source assertion drops to False and
      the recorded provenance fields capture the supplied roots so the
      replay-audit provenance gate can verify the run actually consumed
      legacy artifacts.

    - `dry_run_out_dir` is the pre-refactor backward-compat knob: same-source
      paths, but write to `<dry>/<subject>/` and never auto-create a
      `.legacy_backup`.

    Defensive guards:

    - When `same_source_assertion=False`, `out_dir` must not equal
      `DATA_ROOT/<subject>` — replays must not overwrite the live cohort.
    """
    raw_dir = DATA_ROOT / subject
    detect_dir = DETECT_ROOT / subject

    # refine
    if legacy_refine_root is None:
        refine_npz = detect_dir / "_refineGpu.npz"
        legacy_refine_root_recorded: Optional[str] = None
    else:
        legacy_refine_root = Path(legacy_refine_root)
        refine_npz = legacy_refine_root / subject / "_refineGpu.npz"
        legacy_refine_root_recorded = str(legacy_refine_root)

    # gpu resolver
    if legacy_gpu_root is None:
        def _default_gpu_resolver(s: str, r: str) -> Path:
            return DETECT_ROOT / s / f"{r}_gpu.npz"
        gpu_resolver: _GpuResolver = _default_gpu_resolver
        legacy_gpu_root_recorded: Optional[str] = None
    else:
        legacy_gpu_root = Path(legacy_gpu_root)

        def _legacy_gpu_resolver(s: str, r: str, _root: Path = legacy_gpu_root) -> Path:
            return _root / s / f"{r}_gpu.npz"
        gpu_resolver = _legacy_gpu_resolver
        legacy_gpu_root_recorded = str(legacy_gpu_root)

    # same_source_assertion auto-flag (any legacy root override forces False)
    if legacy_refine_root is not None or legacy_gpu_root is not None:
        same_source_assertion = False

    # out_dir
    if out_dir is None:
        if dry_run_out_dir is not None:
            resolved_out_dir = Path(dry_run_out_dir) / subject
        else:
            resolved_out_dir = raw_dir
    else:
        resolved_out_dir = Path(out_dir)

    # backup_dir
    if backup_dir is _SENTINEL:
        if dry_run_out_dir is not None or out_dir is not None:
            # explicit out_dir or dry-run ⇒ never auto-create a backup
            resolved_backup_dir: Optional[Path] = None
        else:
            resolved_backup_dir = raw_dir / ".legacy_backup"
    else:
        resolved_backup_dir = Path(backup_dir) if backup_dir is not None else None

    # defensive guard
    if not same_source_assertion and resolved_out_dir == raw_dir:
        raise ValueError(
            f"replay (same_source_assertion=False) must not write replay output "
            f"into the production raw tree at {raw_dir}"
        )

    return _ResolvedSubjectIO(
        raw_dir=raw_dir,
        refine_npz=refine_npz,
        gpu_resolver=gpu_resolver,
        out_dir=resolved_out_dir,
        backup_dir=resolved_backup_dir,
        same_source_assertion=same_source_assertion,
        legacy_refine_root_recorded=legacy_refine_root_recorded,
        legacy_gpu_root_recorded=legacy_gpu_root_recorded,
    )


def run_subject(
    subject: str,
    *,
    legacy_refine_root: Optional[Path] = None,
    legacy_gpu_root: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    backup_dir: object = _SENTINEL,
    same_source_assertion: bool = True,
    dry_run_out_dir: Optional[Path] = None,
    only_records: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Run packing for one subject and return (summary_dict, manifest_dict).

    See `_resolve_run_subject_io` for the path-injection contract. The
    same-source CLI (`main()`) calls this with no path overrides; the
    Track B replay driver passes `legacy_refine_root` + `legacy_gpu_root`
    + an explicit `out_dir`."""
    io = _resolve_run_subject_io(
        subject,
        legacy_refine_root=legacy_refine_root,
        legacy_gpu_root=legacy_gpu_root,
        out_dir=out_dir,
        backup_dir=backup_dir,
        same_source_assertion=same_source_assertion,
        dry_run_out_dir=dry_run_out_dir,
    )
    raw_dir = io.raw_dir
    refine_npz = io.refine_npz
    gpu_resolver = io.gpu_resolver
    resolved_out_dir = io.out_dir
    resolved_backup_dir = io.backup_dir

    params = resolve_subject_pack_params(subject)
    drop_chns = list(params["pack_drop_channels"])
    pack_top_n = int(params["pack_top_n"]) if "pack_top_n" in params else None

    if not refine_npz.exists():
        raise FileNotFoundError(f"missing {refine_npz}")

    edfs = sorted(raw_dir.glob("*.edf"))
    if only_records:
        keep = set(only_records)
        edfs = [e for e in edfs if e.stem in keep]
    if not edfs:
        raise FileNotFoundError(f"no EDF found in {raw_dir} (only_records={only_records})")

    # Subject-level alias resolution (refine_npz is loaded ONCE, not per block)
    refine = np.load(refine_npz, allow_pickle=True)
    refine_counts_orig = refine["events_count"].astype(np.int64)
    refine_names_orig = [str(x) for x in refine["chns_names"]]
    aliases, alias_collisions, alias_outer_drops = alias_bipolar_to_left_with_arbitration(
        refine_names_orig, refine_counts_orig
    )

    records: List[Dict[str, object]] = []
    t_start = time.time()
    for i, edf in enumerate(edfs, 1):
        gpu_npz = gpu_resolver(subject, edf.stem)
        if not gpu_npz.exists():
            records.append({
                "record": edf.stem,
                "status": "missing_gpu_npz",
                "skip_reason": "missing_gpu_npz",
                "gpu_npz": str(gpu_npz),
                "wrote_lagpat": False,
                "wrote_packed": False,
            })
            print(f"  [{i}/{len(edfs)}] {edf.stem}: MISSING gpu_npz")
            continue
        try:
            t0 = time.time()
            r = pack_one_record(
                subject=subject,
                edf_path=edf,
                gpu_npz_path=gpu_npz,
                aliases=aliases,
                pack_pick_k=params["pick_k"],
                pack_win_sec=params["pack_win_sec"],
                pack_top_n=pack_top_n,
                drop_chns=drop_chns,
                out_dir=resolved_out_dir,
                backup_dir=resolved_backup_dir,
            )
            r["gpu_npz_used"] = str(gpu_npz)
            r["refine_npz_used"] = str(refine_npz)
            r["elapsed_sec"] = round(time.time() - t0, 1)
            records.append(r)
            status = r["status"]
            extra = ""
            if status == "ok":
                extra = (
                    f"  n_pick={r['n_picked']:>3} n_ev={r['n_events']:>5} "
                    f"med_part={r['median_n_participating']:.1f} "
                    f"med_lag={r['median_lag_span_ms']:.1f}ms "
                    f"elapsed={r['elapsed_sec']}s"
                )
            elif status == "skipped":
                extra = f"  reason={r.get('skip_reason','?')}"
            print(f"  [{i}/{len(edfs)}] {edf.stem}: {status}{extra}")
        except Exception as e:
            records.append({
                "record": edf.stem,
                "status": "error",
                "skip_reason": "exception",
                "error": repr(e),
                "wrote_lagpat": False,
                "wrote_packed": False,
            })
            print(f"  [{i}/{len(edfs)}] {edf.stem}: ERROR {e!r}")

    n_total = len(edfs)
    n_ok = sum(1 for r in records if r.get("status") == "ok")
    n_skipped = sum(1 for r in records if r.get("status") == "skipped")
    n_error = sum(1 for r in records if r.get("status") == "error")
    n_missing = sum(1 for r in records if r.get("status") == "missing_gpu_npz")

    if n_total == 0:
        write_status = "no_inputs"
    elif n_ok == 0:
        write_status = "all_failed"
    elif n_ok == n_total:
        write_status = "ok"
    else:
        write_status = "partial_ok"

    # Aggregate QC across blocks that wrote successfully
    ok_records = [r for r in records if r.get("status") == "ok"]
    if ok_records:
        med_part = float(np.median([r["median_n_participating"] for r in ok_records]))
        med_lag = float(np.median([r["median_lag_span_ms"] for r in ok_records]))
        n_coll_in_picked_max = max(
            int(r.get("n_alias_collisions_in_picked", 0)) for r in ok_records
        )
        n_coll_in_picked_min = min(
            int(r.get("n_alias_collisions_in_picked", 0)) for r in ok_records
        )
    else:
        med_part = float("nan")
        med_lag = float("nan")
        n_coll_in_picked_max = 0
        n_coll_in_picked_min = 0

    # start_time validation aggregate
    st_validations = [
        r["start_time_validation"] for r in records
        if isinstance(r.get("start_time_validation"), dict)
    ]
    start_time_overall_pass = bool(st_validations) and all(
        v.get("passed") is True for v in st_validations
    )

    block_presence = _legacy_block_presence_diff(raw_dir, [e.stem for e in edfs], records)

    summary: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "subject": subject,
        "cohort": "yuquan_same_source" if io.same_source_assertion else "yuquan_legacy_refine_replay",
        "same_source_assertion": io.same_source_assertion,
        "legacy_refine_root": io.legacy_refine_root_recorded,
        "legacy_gpu_root": io.legacy_gpu_root_recorded,
        "refine_npz_used": str(refine_npz),
        "params": params,
        "drop_chns": drop_chns,
        "write_status": write_status,
        "n_blocks_total": n_total,
        "n_blocks_written": n_ok,
        "n_blocks_skipped": n_skipped,
        "n_blocks_error": n_error,
        "n_blocks_missing_gpu_npz": n_missing,
        "n_alias_collisions": len(alias_collisions),
        "n_alias_collisions_in_picked_max": n_coll_in_picked_max,
        "n_alias_collisions_in_picked_min": n_coll_in_picked_min,
        "median_n_participating": med_part,
        "median_lag_span_ms": med_lag,
        "start_time_validation_overall_pass": start_time_overall_pass,
        "out_dir": str(resolved_out_dir),
        "backup_dir": str(resolved_backup_dir) if resolved_backup_dir else None,
        "elapsed_sec_total": round(time.time() - t_start, 1),
        "legacy_block_presence_diff": block_presence,
        "records": records,
    }

    manifest: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "subject": subject,
        "cohort": "yuquan_same_source" if io.same_source_assertion else "yuquan_legacy_refine_replay",
        "same_source_assertion": io.same_source_assertion,
        "legacy_refine_root": io.legacy_refine_root_recorded,
        "legacy_gpu_root": io.legacy_gpu_root_recorded,
        "in_yuquan_same_source_24": subject in YUQUAN_SAME_SOURCE_SUBJECTS,
        "subject_params_path": str(SUBJECT_PARAMS_PATH),
        "params_resolved": params,
        "data_root": str(DATA_ROOT),
        "detect_root": str(DETECT_ROOT),
        "refine_npz": _file_stat(refine_npz),
        "refine_npz_used": str(refine_npz),
        "alias_collisions": alias_collisions,
        "alias_outer_drops": alias_outer_drops,
        "blocks": [
            {
                "record": e.stem,
                "edf": _file_stat(e),
                "gpu_npz": _file_stat(gpu_resolver(subject, e.stem)),
                "gpu_npz_used": str(gpu_resolver(subject, e.stem)),
                "status": next(
                    (r.get("status") for r in records if r.get("record") == e.stem),
                    None,
                ),
                "skip_reason": next(
                    (r.get("skip_reason") for r in records if r.get("record") == e.stem),
                    None,
                ),
                "wrote_lagpat": next(
                    (r.get("wrote_lagpat") for r in records if r.get("record") == e.stem),
                    False,
                ),
                "wrote_packed": next(
                    (r.get("wrote_packed") for r in records if r.get("record") == e.stem),
                    False,
                ),
                "out_lagpat": next(
                    (r.get("out_lagpat") for r in records if r.get("record") == e.stem),
                    None,
                ),
                "out_packed": next(
                    (r.get("out_packed") for r in records if r.get("record") == e.stem),
                    None,
                ),
                "backup_events": next(
                    (r.get("backup_events", []) for r in records if r.get("record") == e.stem),
                    [],
                ),
            }
            for e in edfs
        ],
    }
    return summary, manifest


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
            summary, manifest = run_subject(
                subject, dry_run_out_dir=dry_dir, only_records=only_records
            )
        except Exception as e:
            print(f"FATAL subject={subject}: {e!r}")
            rc = 1
            continue

        out_root = dry_dir if dry_dir is not None else RESULTS_ROOT
        subj_out_dir = (
            Path(args.summary_out).parent if args.summary_out
            else (out_root / subject)
        )
        subj_out_dir.mkdir(parents=True, exist_ok=True)
        summary_path = (
            Path(args.summary_out) if args.summary_out
            else (subj_out_dir / "summary.json")
        )
        manifest_path = subj_out_dir / "manifest.json"

        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))
        print(f"summary:  {summary_path}")
        print(f"manifest: {manifest_path}")
        print(
            f"  write_status={summary['write_status']} "
            f"ok={summary['n_blocks_written']}/{summary['n_blocks_total']}  "
            f"skipped={summary['n_blocks_skipped']}  "
            f"err={summary['n_blocks_error']}  "
            f"miss_gpu={summary['n_blocks_missing_gpu_npz']}  "
            f"alias_coll={summary['n_alias_collisions']} (in_picked_max={summary['n_alias_collisions_in_picked_max']})  "
            f"st_ok={summary['start_time_validation_overall_pass']}  "
            f"elapsed={summary['elapsed_sec_total']}s"
        )
        if summary["legacy_block_presence_diff"]["regressions"]:
            print(
                "  WARN: legacy-present blocks not written by new pipeline: "
                f"{summary['legacy_block_presence_diff']['regressions']}"
            )

    return rc


if __name__ == "__main__":
    sys.exit(main())
