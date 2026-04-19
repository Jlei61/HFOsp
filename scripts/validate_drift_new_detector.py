"""
Phase B: end-to-end drift validation for the Yuquan lagPat backfill.

Goal
----
Quantify the drift introduced by switching the detector source from the
legacy ``/mnt/yuquan_data/.../<block>_gpu.npz`` + legacy
``_refineGpu.npz`` to the new ``results/hfo_detection/<subject>/`` outputs
while keeping the **same legacy-clone packing + spectrogram pipeline**
(see ``scripts/validate_pack_against_legacy.py`` for the algorithm path).

Per ``docs/plans/yuquan_lagpat_backfill_validation.plan.md`` Phase B:

  L1 picked channels       : Jaccard, exclusive sets, alias collisions
  L2 packed window count   : ratio (new / legacy), per-block + per-subject
  L3 n_participating shift : median + KS, per-subject across all events
  L4 lag span shift        : median + p95 of (max-min centroid per event)

Stops short of L5/L6 (Topic 1 / Topic 2 summary drift) on purpose; those
require running the downstream summary code, which is a separate cost.

Outputs
-------
- ``results/validation/phaseB/<subject>__<block>.json`` per common block
- ``results/validation/phaseB/<subject>__SUBJECT.json`` per subject (cohort-level)
- ``results/validation/phaseB/SUMMARY.md``

Read-only against ``/mnt/yuquan_data/yuquan_24h_edf`` and
``results/hfo_detection``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Reuse legacy-clone helpers from the Phase A validator. They define the
# global LEGACY_* parameters and the EDF preprocessing path; this keeps
# Phase B "same pipeline, only the detector source changes".
from scripts.validate_pack_against_legacy import (  # noqa: E402
    DATA_ROOT,
    LEGACY_CHNS_THR,
    LEGACY_EXT_MS,
    LEGACY_PACKED_BOOL_FS,
    LEGACY_PACKING_GAP_LIMIT_S,
    LEGACY_RESAMPLE_TO,
    LEGACY_SEGMENT_TIME,
    LEGACY_SUBJECT_PARAMS,
    LEGACY_TIME_AXIS_HZ,
    _legacy_bipolar_reref_and_drop,
    _legacy_get_packed_events_bool,
    _legacy_resample_notch_band,
    _legacy_valid_chan_index,
)
from src.group_event_analysis import (  # noqa: E402
    build_stitched_window_signal,
    build_windows_from_detections,
    build_windows_from_packed_times,
    compute_stitched_spectrogram_centroids_legacy,
    filter_windows_for_legacy_segment_loop,
)

OUTPUT_ROOT = REPO_ROOT / "results" / "validation" / "phaseB"
NEW_DET_ROOT = REPO_ROOT / "results" / "hfo_detection"

REFERENCE_SUBJECTS = ["gaolan", "dongyiming", "wangyiyang"]


# ---------------------------------------------------------------------------
# Bipolar -> left-contact alias mapping (legacy contract) with collision QC
# ---------------------------------------------------------------------------

_BIPOLAR_RE = re.compile(r"^([A-Za-z]+'?)(\d+)\s*-\s*([A-Za-z]+'?)(\d+)$")


def alias_left_contact(bipolar_name: str) -> Optional[str]:
    """Map a bipolar pair name like ``A1-A2`` or ``B'3-B'4`` to its
    left-contact alias ``A1`` / ``B'3``. Returns None when the name does
    not match the bipolar pattern (caller should treat as a non-paired
    channel and skip)."""
    m = _BIPOLAR_RE.match(str(bipolar_name).strip())
    if not m:
        return None
    pre1, n1, pre2, n2 = m.group(1), m.group(2), m.group(3), m.group(4)
    if pre1 != pre2:
        return None
    if int(n2) <= int(n1):
        return None
    return f"{pre1}{n1}"


@dataclass
class AliasResult:
    aliased_names: List[str]
    aliased_counts: np.ndarray
    bipolar_to_alias: Dict[str, str]
    alias_to_bipolar: Dict[str, str]
    skipped_non_bipolar: List[str]
    collisions: List[Dict[str, object]]


def alias_refine_to_left_contact(
    refine_names: Sequence[str], refine_counts: Sequence[float]
) -> AliasResult:
    """Convert new-style bipolar refine to legacy left-contact alias.

    Collision rule (per plan section §5): keep the higher events_count;
    losers are recorded in ``collisions``.
    """
    counts = np.asarray(refine_counts, dtype=np.float64)
    skipped: List[str] = []
    bucket: Dict[str, List[Tuple[str, float]]] = {}
    for name, c in zip(refine_names, counts):
        alias = alias_left_contact(name)
        if alias is None:
            skipped.append(str(name))
            continue
        bucket.setdefault(alias, []).append((str(name), float(c)))

    aliased_names: List[str] = []
    aliased_counts: List[float] = []
    bipolar_to_alias: Dict[str, str] = {}
    alias_to_bipolar: Dict[str, str] = {}
    collisions: List[Dict[str, object]] = []

    for alias, items in bucket.items():
        if len(items) == 1:
            bp, c = items[0]
            aliased_names.append(alias)
            aliased_counts.append(c)
            bipolar_to_alias[bp] = alias
            alias_to_bipolar[alias] = bp
            continue
        items_sorted = sorted(items, key=lambda x: -x[1])
        winner_bp, winner_c = items_sorted[0]
        loser = items_sorted[1:]
        ratio = winner_c / max(loser[0][1], 1.0)
        collisions.append({
            "alias": alias,
            "winner_bipolar": winner_bp,
            "winner_count": winner_c,
            "losers": [{"bipolar": bp, "count": c} for bp, c in loser],
            "winner_loser_ratio": float(ratio),
        })
        aliased_names.append(alias)
        aliased_counts.append(winner_c)
        bipolar_to_alias[winner_bp] = alias
        alias_to_bipolar[alias] = winner_bp

    order = np.argsort(aliased_names)
    aliased_names = [aliased_names[i] for i in order]
    aliased_counts_arr = np.asarray([aliased_counts[i] for i in order], dtype=np.float64)
    return AliasResult(
        aliased_names=aliased_names,
        aliased_counts=aliased_counts_arr,
        bipolar_to_alias=bipolar_to_alias,
        alias_to_bipolar=alias_to_bipolar,
        skipped_non_bipolar=skipped,
        collisions=collisions,
    )


# ---------------------------------------------------------------------------
# L1 picked channel comparison
# ---------------------------------------------------------------------------


@dataclass
class L1Result:
    new_picked: List[str]
    legacy_picked: List[str]
    n_new: int
    n_legacy: int
    n_overlap: int
    only_new: List[str]
    only_legacy: List[str]
    jaccard: float
    threshold_new: float
    mean_count_new: float
    std_count_new: float
    n_alias_collisions: int
    alias_collisions_picked: List[str]
    alias_skipped_non_bipolar: int


def compute_l1(
    refine_names: Sequence[str],
    refine_counts: np.ndarray,
    legacy_picked: Sequence[str],
    pick_k: float,
) -> Tuple[L1Result, AliasResult]:
    alias = alias_refine_to_left_contact(refine_names, refine_counts)
    counts = alias.aliased_counts
    thr = float(counts.mean() + pick_k * counts.std())
    pick_mask = counts > thr
    new_picked = [alias.aliased_names[i] for i in np.where(pick_mask)[0]]
    legacy_picked_list = [str(c) for c in legacy_picked]
    set_new = set(new_picked)
    set_legacy = set(legacy_picked_list)
    overlap = sorted(set_new & set_legacy)
    only_new = sorted(set_new - set_legacy)
    only_legacy = sorted(set_legacy - set_new)
    union = set_new | set_legacy
    jaccard = float(len(overlap) / len(union)) if union else float("nan")

    # alias collisions that landed inside picked set
    coll_picked = [c["alias"] for c in alias.collisions if c["alias"] in set_new]

    return L1Result(
        new_picked=new_picked,
        legacy_picked=legacy_picked_list,
        n_new=int(len(new_picked)),
        n_legacy=int(len(legacy_picked_list)),
        n_overlap=int(len(overlap)),
        only_new=only_new,
        only_legacy=only_legacy,
        jaccard=jaccard,
        threshold_new=thr,
        mean_count_new=float(counts.mean()),
        std_count_new=float(counts.std()),
        n_alias_collisions=int(len(alias.collisions)),
        alias_collisions_picked=coll_picked,
        alias_skipped_non_bipolar=int(len(alias.skipped_non_bipolar)),
    ), alias


# ---------------------------------------------------------------------------
# L2-L4 per-block driver
# ---------------------------------------------------------------------------


@dataclass
class BlockResult:
    subject: str
    block: str
    pick_k: float
    pack_win_sec: float
    drop_chns: List[str]

    # L2
    n_new_windows: int
    n_legacy_windows: int
    n_new_picked_in_block: int
    count_ratio: float

    # L3 (over all events, both pipelines)
    median_n_participating_new: float
    median_n_participating_legacy: float
    median_n_participating_shift: float

    # L4
    median_lag_span_ms_new: float
    p95_lag_span_ms_new: float
    median_lag_span_ms_legacy: float
    p95_lag_span_ms_legacy: float
    median_lag_span_shift: float

    notes: str = ""

    # arrays kept for cohort aggregation (NOT serialised per-block)
    n_part_new_arr: np.ndarray = field(default_factory=lambda: np.zeros(0))
    n_part_leg_arr: np.ndarray = field(default_factory=lambda: np.zeros(0))
    span_new_arr: np.ndarray = field(default_factory=lambda: np.zeros(0))
    span_leg_arr: np.ndarray = field(default_factory=lambda: np.zeros(0))


def _per_event_n_participating(events_bool: np.ndarray) -> np.ndarray:
    if events_bool.size == 0:
        return np.zeros(0, dtype=np.int64)
    return np.sum(events_bool > 0.5, axis=0).astype(np.int64)


def _per_event_lag_span_ms(centroids: np.ndarray, events_bool: np.ndarray) -> np.ndarray:
    """For each event column, compute (max - min) of valid (eventsBool==True
    AND finite centroid) values; returns NaN for events with <2 valid channels.
    """
    if centroids.shape != events_bool.shape:
        raise ValueError(f"shape mismatch: centroids {centroids.shape} vs events_bool {events_bool.shape}")
    n_ev = centroids.shape[1]
    span = np.full(n_ev, np.nan, dtype=np.float64)
    for j in range(n_ev):
        mask = (events_bool[:, j] > 0.5) & np.isfinite(centroids[:, j])
        if int(mask.sum()) < 2:
            continue
        vals = centroids[mask, j]
        span[j] = float(vals.max() - vals.min())
    return span * 1000.0  # ms


def _legacy_lag_span_ms_per_event(
    legacy_lag_raw: np.ndarray,
    legacy_packed_times: np.ndarray,
    legacy_events_bool: np.ndarray,
    fs: float,
    segment_time_s: float,
) -> np.ndarray:
    """Convert legacy stitched-time lagPatRaw to within-window centroid then
    compute (max - min) per event in ms.

    Legacy stores absolute time on the per-segment stitched timeline. For
    *lag span* we only need the spread within an event, which equals the
    spread of the per-window centroid offsets — both derivations cancel
    the per-event ``cum_start_sec`` term, so we can use lagPatRaw values
    directly per event without reconstructing the offsets.
    """
    n_ev = legacy_lag_raw.shape[1]
    span = np.full(n_ev, np.nan, dtype=np.float64)
    for j in range(n_ev):
        mask = (legacy_events_bool[:, j] > 0.5) & np.isfinite(legacy_lag_raw[:, j])
        if int(mask.sum()) < 2:
            continue
        vals = legacy_lag_raw[mask, j]
        span[j] = float(vals.max() - vals.min())
    return span * 1000.0


def validate_block_phaseB(
    subject: str,
    block: str,
    new_picked_aliased: Sequence[str],
    alias_to_bipolar: Dict[str, str],
    pack_win_sec: float,
    pick_k: float,
    drop_chns: Sequence[str],
) -> Optional[BlockResult]:
    import mne

    new_gpu_path = NEW_DET_ROOT / subject / f"{block}_gpu.npz"
    leg_dir = DATA_ROOT / subject
    leg_packed_path = leg_dir / f"{block}_packedTimes.npy"
    leg_lag_path = leg_dir / f"{block}_lagPat.npz"
    edf_path = leg_dir / f"{block}.edf"
    for p in (new_gpu_path, leg_packed_path, leg_lag_path, edf_path):
        if not p.exists():
            print(f"  [skip] {subject}/{block}: missing {p.name}", flush=True)
            return None

    gpu = np.load(new_gpu_path, allow_pickle=True)
    gpu_dets_all = np.asarray(gpu["whole_dets"], dtype=object)
    gpu_names_all = np.asarray(gpu["chns_names"]).astype(str)
    gpu_name_to_idx = {str(n): i for i, n in enumerate(gpu_names_all)}

    # Resolve picked alias -> bipolar -> dets[picked_alias_position]
    dets: Dict[str, np.ndarray] = {}
    n_new_picked_in_block = 0
    for alias in new_picked_aliased:
        bp = alias_to_bipolar.get(alias)
        if bp is None or bp not in gpu_name_to_idx:
            dets[alias] = np.zeros((0, 2), dtype=np.float64)
            continue
        arr = gpu_dets_all[gpu_name_to_idx[bp]]
        if arr is None:
            dets[alias] = np.zeros((0, 2), dtype=np.float64)
        else:
            arr2 = np.asarray(arr, dtype=np.float64)
            if arr2.ndim != 2 or arr2.shape[1] != 2:
                dets[alias] = np.zeros((0, 2), dtype=np.float64)
            else:
                dets[alias] = arr2
                if arr2.size:
                    n_new_picked_in_block += 1

    raw_hdr = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False, encoding="latin1")
    record_last_sec = float(raw_hdr.times[-1])
    record_sfreq = float(raw_hdr.info["sfreq"])

    # ---- L2: packed windows ----
    max_event_t = 0.0
    for arr in dets.values():
        if arr.size:
            max_event_t = max(max_event_t, float(arr[:, 1].max()))
    new_windows = build_windows_from_detections(
        dets,
        window_sec=float(pack_win_sec),
        ext_ms=LEGACY_EXT_MS,
        chns_thr=LEGACY_CHNS_THR,
        time_axis_hz=LEGACY_TIME_AXIS_HZ,
        max_window_sec=LEGACY_PACKING_GAP_LIMIT_S,
        t_max_sec=max_event_t + 5.0,
    )
    new_windows = filter_windows_for_legacy_segment_loop(
        new_windows,
        segment_duration_sec=LEGACY_SEGMENT_TIME,
        record_last_sec=record_last_sec,
        sfreq=record_sfreq,
        start_sec=0.0,
    )

    legacy_packed = np.asarray(np.load(leg_packed_path, allow_pickle=True), dtype=np.float64)
    legacy_lag = np.load(leg_lag_path, allow_pickle=True)
    legacy_lag_raw = np.asarray(legacy_lag["lagPatRaw"], dtype=np.float64)
    legacy_events_bool = np.asarray(legacy_lag["eventsBool"], dtype=np.float64)
    legacy_picks = [str(c) for c in legacy_lag["chnNames"]]
    n_new_w = len(new_windows)
    n_leg_w = int(legacy_packed.shape[0])
    count_ratio = float(n_new_w) / float(n_leg_w) if n_leg_w > 0 else float("inf")

    # ---- L3: events_bool for new pipeline ----
    high_events_per_alias: List[np.ndarray] = []
    for alias in new_picked_aliased:
        a = dets.get(alias, np.zeros((0, 2)))
        high_events_per_alias.append(a if a.size else np.zeros((0, 2), dtype=np.float64))

    if n_new_w > 0:
        new_packed = np.array([[w.start, w.end] for w in new_windows], dtype=np.float64)
        new_events_bool = _legacy_get_packed_events_bool(
            high_events_per_alias, new_packed, fs=LEGACY_PACKED_BOOL_FS
        )
    else:
        new_events_bool = np.zeros((len(new_picked_aliased), 0), dtype=np.float64)
    n_part_new = _per_event_n_participating(new_events_bool)
    n_part_leg = _per_event_n_participating(legacy_events_bool)
    med_part_new = float(np.median(n_part_new)) if n_part_new.size else float("nan")
    med_part_leg = float(np.median(n_part_leg)) if n_part_leg.size else float("nan")
    if med_part_leg and not np.isnan(med_part_leg):
        shift_part = (med_part_new - med_part_leg) / med_part_leg
    else:
        shift_part = float("nan")

    # ---- L4: lag span ----
    legacy_span_ms = _legacy_lag_span_ms_per_event(
        legacy_lag_raw, legacy_packed, legacy_events_bool, fs=LEGACY_RESAMPLE_TO,
        segment_time_s=LEGACY_SEGMENT_TIME,
    )
    med_span_leg = float(np.nanmedian(legacy_span_ms)) if np.any(np.isfinite(legacy_span_ms)) else float("nan")
    p95_span_leg = float(np.nanpercentile(legacy_span_ms[np.isfinite(legacy_span_ms)], 95)) if np.any(np.isfinite(legacy_span_ms)) else float("nan")

    new_span_ms = np.zeros(0, dtype=np.float64)
    if n_new_w > 0 and n_new_picked_in_block > 0:
        # Reuse Phase A's stitched-spec helpers on the EDF + new windows.
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False, encoding="latin1")
        data_in = raw.get_data()
        ch_names_in = list(raw.info["ch_names"])
        valid_idx = _legacy_valid_chan_index(ch_names_in)
        data_v = data_in[valid_idx]
        ch_names_v = [ch_names_in[i] for i in valid_idx]
        bip_data, bip_names = _legacy_bipolar_reref_and_drop(data_v, ch_names_v, drop_chns)
        bip_band = _legacy_resample_notch_band(bip_data, float(raw.info["sfreq"]))

        bip_idx = {n: i for i, n in enumerate(bip_names)}
        rows = []
        keep_aliases = []
        for alias in new_picked_aliased:
            if alias in bip_idx:
                rows.append(bip_idx[alias])
                keep_aliases.append(alias)
        if not rows:
            new_span_ms = np.zeros(0, dtype=np.float64)
        else:
            x_band = bip_band[rows]
            # Re-index events_bool to the kept-alias rows so L4's span aligns
            # with L3's participation mask
            keep_to_picked = {a: i for i, a in enumerate(new_picked_aliased)}
            keep_rows_in_picked = np.array([keep_to_picked[a] for a in keep_aliases], dtype=np.int64)
            new_events_bool_kept = new_events_bool[keep_rows_in_picked]
            # Process per legacy 200s segment
            seg_id_per_window = np.array(
                [int(np.floor(float(w.start) / LEGACY_SEGMENT_TIME)) for w in new_windows],
                dtype=np.int64,
            )
            centroids_full = np.full((len(keep_aliases), n_new_w), np.nan, dtype=np.float64)
            for seg in np.unique(seg_id_per_window):
                ev_idx = np.where(seg_id_per_window == seg)[0]
                seg_windows = [new_windows[i] for i in ev_idx]
                stitched, borders = build_stitched_window_signal(
                    x_band, seg_windows, sfreq=float(LEGACY_RESAMPLE_TO), start_sec=0.0,
                )
                if borders.size == 0:
                    continue
                seg_cent = compute_stitched_spectrogram_centroids_legacy(
                    stitched, borders, sfreq=float(LEGACY_RESAMPLE_TO),
                )
                centroids_full[:, ev_idx] = seg_cent
            new_span_ms = _per_event_lag_span_ms(centroids_full, new_events_bool_kept)
    med_span_new = float(np.nanmedian(new_span_ms)) if np.any(np.isfinite(new_span_ms)) else float("nan")
    p95_span_new = float(np.nanpercentile(new_span_ms[np.isfinite(new_span_ms)], 95)) if np.any(np.isfinite(new_span_ms)) else float("nan")
    if med_span_leg and not np.isnan(med_span_leg) and med_span_leg > 0:
        shift_span = (med_span_new - med_span_leg) / med_span_leg
    else:
        shift_span = float("nan")

    notes = ""
    if set(legacy_picks) != set(new_picked_aliased):
        notes = (
            f"L3/L4 use new picked set (n={len(new_picked_aliased)}); "
            f"legacy picked set (n={len(legacy_picks)}) differs"
        )

    return BlockResult(
        subject=subject,
        block=block,
        pick_k=float(pick_k),
        pack_win_sec=float(pack_win_sec),
        drop_chns=list(drop_chns),
        n_new_windows=int(n_new_w),
        n_legacy_windows=int(n_leg_w),
        n_new_picked_in_block=int(n_new_picked_in_block),
        count_ratio=float(count_ratio),
        median_n_participating_new=med_part_new,
        median_n_participating_legacy=med_part_leg,
        median_n_participating_shift=float(shift_part),
        median_lag_span_ms_new=med_span_new,
        p95_lag_span_ms_new=p95_span_new,
        median_lag_span_ms_legacy=med_span_leg,
        p95_lag_span_ms_legacy=p95_span_leg,
        median_lag_span_shift=float(shift_span),
        notes=notes,
        n_part_new_arr=n_part_new.astype(np.int64),
        n_part_leg_arr=n_part_leg.astype(np.int64),
        span_new_arr=new_span_ms.astype(np.float64),
        span_leg_arr=legacy_span_ms.astype(np.float64),
    )


# ---------------------------------------------------------------------------
# Subject-level driver and reporting
# ---------------------------------------------------------------------------


def _block_result_to_dict(r: BlockResult) -> Dict[str, object]:
    d = r.__dict__.copy()
    # do not serialise per-event arrays in the per-block JSON
    for k in ("n_part_new_arr", "n_part_leg_arr", "span_new_arr", "span_leg_arr"):
        d.pop(k, None)
    return d


def _aggregate_subject(
    subject: str,
    l1: L1Result,
    alias: AliasResult,
    block_results: List[BlockResult],
) -> Dict[str, object]:
    if not block_results:
        return {
            "subject": subject,
            "n_blocks_compared": 0,
            "L1": l1.__dict__,
            "L2_subject": None,
            "L3_subject": None,
            "L4_subject": None,
            "drift_flags": ["no_blocks_compared"],
        }
    sum_new = int(sum(r.n_new_windows for r in block_results))
    sum_leg = int(sum(r.n_legacy_windows for r in block_results))
    overall_ratio = float(sum_new) / float(sum_leg) if sum_leg > 0 else float("inf")

    n_part_new_all = np.concatenate([r.n_part_new_arr for r in block_results]) if block_results else np.zeros(0)
    n_part_leg_all = np.concatenate([r.n_part_leg_arr for r in block_results]) if block_results else np.zeros(0)
    med_part_new = float(np.median(n_part_new_all)) if n_part_new_all.size else float("nan")
    med_part_leg = float(np.median(n_part_leg_all)) if n_part_leg_all.size else float("nan")
    shift_part = (
        (med_part_new - med_part_leg) / med_part_leg
        if med_part_leg and not np.isnan(med_part_leg) and med_part_leg > 0
        else float("nan")
    )

    span_new_all = np.concatenate([r.span_new_arr for r in block_results]) if block_results else np.zeros(0)
    span_leg_all = np.concatenate([r.span_leg_arr for r in block_results]) if block_results else np.zeros(0)
    span_new_finite = span_new_all[np.isfinite(span_new_all)]
    span_leg_finite = span_leg_all[np.isfinite(span_leg_all)]
    med_span_new = float(np.median(span_new_finite)) if span_new_finite.size else float("nan")
    p95_span_new = float(np.percentile(span_new_finite, 95)) if span_new_finite.size else float("nan")
    med_span_leg = float(np.median(span_leg_finite)) if span_leg_finite.size else float("nan")
    p95_span_leg = float(np.percentile(span_leg_finite, 95)) if span_leg_finite.size else float("nan")
    shift_span = (
        (med_span_new - med_span_leg) / med_span_leg
        if med_span_leg and not np.isnan(med_span_leg) and med_span_leg > 0
        else float("nan")
    )

    drift_flags: List[str] = []
    if not (0.67 <= overall_ratio <= 1.50):
        drift_flags.append("packed_count_ratio_out_of_band")
    if np.isfinite(shift_part) and abs(shift_part) > 0.20:
        drift_flags.append("median_n_participating_shift_gt_20pct")
    if np.isfinite(shift_span) and abs(shift_span) > 0.20:
        drift_flags.append("median_lag_span_shift_gt_20pct")
    if l1.n_alias_collisions > 0 and l1.alias_collisions_picked:
        drift_flags.append("alias_collision_in_picked")

    return {
        "subject": subject,
        "n_blocks_compared": int(len(block_results)),
        "L1": l1.__dict__,
        "L2_subject": {
            "sum_new_windows": sum_new,
            "sum_legacy_windows": sum_leg,
            "overall_count_ratio": overall_ratio,
            "per_block_ratios": [r.count_ratio for r in block_results],
        },
        "L3_subject": {
            "n_events_new": int(n_part_new_all.size),
            "n_events_legacy": int(n_part_leg_all.size),
            "median_n_participating_new": med_part_new,
            "median_n_participating_legacy": med_part_leg,
            "median_n_participating_shift": shift_part,
        },
        "L4_subject": {
            "n_events_new_finite": int(span_new_finite.size),
            "n_events_legacy_finite": int(span_leg_finite.size),
            "median_lag_span_ms_new": med_span_new,
            "p95_lag_span_ms_new": p95_span_new,
            "median_lag_span_ms_legacy": med_span_leg,
            "p95_lag_span_ms_legacy": p95_span_leg,
            "median_lag_span_shift": shift_span,
        },
        "drift_flags": drift_flags,
        "alias_collisions": alias.collisions,
        "alias_skipped_non_bipolar": alias.skipped_non_bipolar[:20],
    }


def run_subject(
    subject: str, restrict_blocks: Optional[Sequence[str]] = None
) -> Tuple[Dict[str, object], List[BlockResult]]:
    if subject not in LEGACY_SUBJECT_PARAMS:
        raise KeyError(f"no legacy params recorded for subject '{subject}'")
    params = LEGACY_SUBJECT_PARAMS[subject]
    pick_k = float(params["pick_k"])
    pack_win_sec = float(params["pack_win_sec"])
    drop_chns = list(params["drop_chns"])

    new_refine_path = NEW_DET_ROOT / subject / "_refineGpu.npz"
    if not new_refine_path.exists():
        raise FileNotFoundError(f"missing new refine: {new_refine_path}")
    new_refine = np.load(new_refine_path, allow_pickle=True)
    new_refine_names = np.asarray(new_refine["chns_names"]).astype(str)
    new_refine_counts = np.asarray(new_refine["events_count"], dtype=np.float64)

    # Use the chnNames of any existing legacy lagPat as the legacy-picked set
    # (it is identical across blocks for one subject).
    leg_dir = DATA_ROOT / subject
    legacy_lag_files = sorted(leg_dir.glob("*_lagPat.npz"))
    if not legacy_lag_files:
        raise FileNotFoundError(f"no legacy lagPat for subject {subject}")
    legacy_picked = list(np.load(legacy_lag_files[0], allow_pickle=True)["chnNames"].astype(str))

    l1, alias = compute_l1(new_refine_names, new_refine_counts, legacy_picked, pick_k=pick_k)
    print(f"\n=== Phase B subject={subject} ===", flush=True)
    print(
        f"  pick_k={pick_k}  pack_win_sec={pack_win_sec}  drop={drop_chns}",
        flush=True,
    )
    print(
        f"  [L1] n_new={l1.n_new}  n_leg={l1.n_legacy}  jaccard={l1.jaccard:.3f}  "
        f"only_new={len(l1.only_new)}  only_leg={len(l1.only_legacy)}  collisions={l1.n_alias_collisions}",
        flush=True,
    )
    if l1.only_new:
        print(f"    only_new: {l1.only_new}", flush=True)
    if l1.only_legacy:
        print(f"    only_legacy: {l1.only_legacy}", flush=True)

    # Block intersection
    new_blocks = sorted(p.name.replace("_gpu.npz", "") for p in (NEW_DET_ROOT / subject).glob("*_gpu.npz"))
    leg_blocks = sorted(p.name.replace("_lagPat.npz", "") for p in legacy_lag_files)
    common = sorted(set(new_blocks) & set(leg_blocks))
    if restrict_blocks is not None:
        common = [b for b in common if b in set(restrict_blocks)]
    print(f"  common_blocks={len(common)} new_only={sorted(set(new_blocks) - set(leg_blocks))} legacy_only={sorted(set(leg_blocks) - set(new_blocks))}", flush=True)

    block_results: List[BlockResult] = []
    for block in common:
        t0 = time.time()
        res = validate_block_phaseB(
            subject=subject,
            block=block,
            new_picked_aliased=l1.new_picked,
            alias_to_bipolar=alias.alias_to_bipolar,
            pack_win_sec=pack_win_sec,
            pick_k=pick_k,
            drop_chns=drop_chns,
        )
        if res is None:
            continue
        block_results.append(res)
        print(
            f"  [{block}] L2 ratio={res.count_ratio:.3f} (n_new={res.n_new_windows} n_leg={res.n_legacy_windows})  "
            f"L3 med_part new={res.median_n_participating_new:.1f} leg={res.median_n_participating_legacy:.1f} shift={res.median_n_participating_shift*100:.1f}%  "
            f"L4 med_span new={res.median_lag_span_ms_new:.2f}ms leg={res.median_lag_span_ms_legacy:.2f}ms shift={res.median_lag_span_shift*100:.1f}%  "
            f"elapsed={time.time()-t0:.1f}s",
            flush=True,
        )

    subject_summary = _aggregate_subject(subject, l1, alias, block_results)
    return subject_summary, block_results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subject", type=str, default=None,
        help="Single subject (must have legacy lagPat). Defaults to all 3 reference subjects.",
    )
    parser.add_argument(
        "--block", type=str, default=None,
        help="Single block to restrict to (only used with --subject).",
    )
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.subject:
        subjects = [args.subject]
    else:
        subjects = list(REFERENCE_SUBJECTS)

    summary_lines = ["# Phase B — End-to-End Drift Validation\n"]
    summary_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    summary_lines.append(
        "Compares the new packing + spec-centroid pipeline driven by the **new** "
        "`results/hfo_detection/<subject>/*_gpu.npz` + `_refineGpu.npz` against "
        "the legacy `_lagPat.npz` / `_packedTimes.npy`. Pipeline algorithm path "
        "is identical to Phase A (legacy-clone), so observed shifts are pure "
        "detector-source drift.\n"
    )

    overall_drift_flags: List[str] = []
    n_subjects_with_flags = 0
    for subject in subjects:
        restrict = [args.block] if args.block else None
        summary, blocks = run_subject(subject, restrict_blocks=restrict)
        # Per-block JSONs
        for r in blocks:
            (args.out_dir / f"{subject}__{r.block}.json").write_text(
                json.dumps(_block_result_to_dict(r), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        (args.out_dir / f"{subject}__SUBJECT.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=lambda o: list(o) if hasattr(o, "tolist") else str(o)),
            encoding="utf-8",
        )

        flags = summary.get("drift_flags", [])
        if flags:
            n_subjects_with_flags += 1
            overall_drift_flags.extend(f"{subject}:{f}" for f in flags)

        # Markdown rows
        summary_lines.append(f"\n## {subject}\n")
        l1d = summary["L1"]
        summary_lines.append(
            f"- pick_k={l1d['threshold_new']:.3g} (mean={l1d['mean_count_new']:.1f}, std={l1d['std_count_new']:.1f})  "
            f"alias_collisions={l1d['n_alias_collisions']} (in_picked={len(l1d['alias_collisions_picked'])})  "
            f"alias_skipped_non_bipolar={l1d['alias_skipped_non_bipolar']}\n"
        )
        summary_lines.append(
            f"- **L1** n_new={l1d['n_new']} n_leg={l1d['n_legacy']} overlap={l1d['n_overlap']} jaccard={l1d['jaccard']:.3f}\n"
        )
        if l1d["only_new"]:
            summary_lines.append(f"  - only_new ({len(l1d['only_new'])}): `{l1d['only_new']}`\n")
        if l1d["only_legacy"]:
            summary_lines.append(f"  - only_legacy ({len(l1d['only_legacy'])}): `{l1d['only_legacy']}`\n")
        l2 = summary["L2_subject"]
        l3 = summary["L3_subject"]
        l4 = summary["L4_subject"]
        if l2 is None:
            summary_lines.append("- (no blocks compared)\n")
        else:
            summary_lines.append(
                f"- **L2** sum_new_windows={l2['sum_new_windows']} sum_legacy_windows={l2['sum_legacy_windows']} "
                f"**overall_ratio={l2['overall_count_ratio']:.3f}** (per-block range: "
                f"[{min(l2['per_block_ratios']):.2f}, {max(l2['per_block_ratios']):.2f}])\n"
            )
            summary_lines.append(
                f"- **L3** med_n_participating new={l3['median_n_participating_new']:.2f} legacy={l3['median_n_participating_legacy']:.2f} "
                f"shift={l3['median_n_participating_shift']*100:.1f}%\n"
            )
            summary_lines.append(
                f"- **L4** med_lag_span_ms new={l4['median_lag_span_ms_new']:.2f} legacy={l4['median_lag_span_ms_legacy']:.2f} "
                f"shift={l4['median_lag_span_shift']*100:.1f}%  (p95 new={l4['p95_lag_span_ms_new']:.2f}, p95 leg={l4['p95_lag_span_ms_legacy']:.2f})\n"
            )
        if summary["drift_flags"]:
            summary_lines.append(f"- **drift_flags**: `{summary['drift_flags']}`\n")

    summary_lines.append("\n---\n\n")
    summary_lines.append(f"**Subjects with any drift flag: {n_subjects_with_flags} / {len(subjects)}**\n")
    if overall_drift_flags:
        summary_lines.append("\nFlags raised:\n")
        for f in overall_drift_flags:
            summary_lines.append(f"- `{f}`\n")
    plan_threshold_msg = (
        "\nPlan stop condition: drift considered 'too large' when any of "
        "{count_ratio outside [0.67, 1.50], |median n_participating shift| > 20%, "
        "|median lag span shift| > 20%, alias collision in picked} fires on >= 2/3 reference subjects.\n"
    )
    summary_lines.append(plan_threshold_msg)

    summary_path = args.out_dir / "SUMMARY.md"
    summary_path.write_text("".join(summary_lines), encoding="utf-8")
    print(f"\nWrote summary {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
