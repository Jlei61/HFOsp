"""Epilepsiae new-pipeline pack + lagPat backfill driver.

Reads:  results/hfo_detection/<subject>/*_gpu.npz       (whole_dets, chns_names, start_time)
        results/hfo_detection/<subject>/_refineGpu.npz   (subject-level refined channel list)
        Raw .data + .head via load_epilepsiae_block       (CAR signal, variable sfreq)
Writes: results/epilepsiae_lagpat_backfill/<subject>/<stem>_packedTimes.npy
        results/epilepsiae_lagpat_backfill/<subject>/<stem>_lagPat.npz
        results/epilepsiae_lagpat_backfill/<subject>/_backfill_log.json

NOT a Track B replay: legacy *_gpu.npz are 216-byte stubs (per artifact census 2026-04-27).
Output is a NEW pack/lag artifact for sensitivity-audit purposes only.

Plan: docs/archive/epilepsiae_lagpat/epilepsiae_lagpat_backfill_plan_2026-04-29.md
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.epilepsiae_dataset import EpilepsiaePaths, _collect_raw_blocks
from src.preprocessing import _read_epilepsiae_head_for_streaming

NEW_GPU_ROOT = Path("results/hfo_detection")
OUTPUT_ROOT = Path("results/epilepsiae_lagpat_backfill")
RIPPLE_BAND = (80.0, 250.0)
NYQUIST_GATE_HZ = 2.0 * RIPPLE_BAND[1]  # 500 Hz; 256 Hz blocks fail this
SEGMENT_SEC = 200.0  # mirrors Yuquan stitched-segment legacy semantics


def _refine_path_for_subject(subject: str) -> Path:
    """Path to the subject-level refine artifact.

    The new pipeline (scripts/run_hfo_detection.py) writes ONE _refineGpu.npz
    per subject (no per-record suffix, no `sub_` prefix). Schema:
    {chns_names, events_count}.
    """
    return NEW_GPU_ROOT / subject / "_refineGpu.npz"


@lru_cache(maxsize=32)
def load_refine_chns_for_subject(subject: str) -> Tuple[str, ...]:
    """Read subject-level refined channel names (cached per subject).

    Returns a tuple (immutable) so lru_cache is safe; convert to list at call
    site if mutation is needed.
    """
    refine_path = _refine_path_for_subject(subject)
    if not refine_path.exists():
        raise FileNotFoundError(
            f"Subject-level refine artifact missing: {refine_path}\n"
            "Expected file produced by scripts/run_hfo_detection.py."
        )
    z = np.load(refine_path, allow_pickle=True)
    return tuple(str(c) for c in z["chns_names"])


def _discover_records(subject: str) -> List[Dict]:
    """Cross-reference results/hfo_detection/<subject>/*_gpu.npz with raw .data/.head.

    Returns list of dicts: stem, sfreq, new_gpu_path, raw_data_path, raw_head_path.
    Records where raw .data/.head are missing are skipped with a warning (not error)
    because the new pipeline already filtered Nyquist-failing blocks.
    """
    paths = EpilepsiaePaths()
    raw_blocks = _collect_raw_blocks(subject, paths)  # stem -> RawBlockFiles
    new_gpu_dir = NEW_GPU_ROOT / subject
    if not new_gpu_dir.exists():
        raise FileNotFoundError(f"No new-pipeline gpu dir: {new_gpu_dir}")
    out: List[Dict] = []
    for gpu_path in sorted(new_gpu_dir.glob("*_gpu.npz")):
        # suffix-only strip; replace() would mangle e.g. "X_gpu_y_gpu"
        stem = gpu_path.stem.removesuffix("_gpu")
        if stem.endswith("_refineGpu") or stem.startswith("sub_"):
            continue
        if stem not in raw_blocks:
            print(f"  WARN: {stem} has new gpu but no raw .data/.head; skip")
            continue
        block = raw_blocks[stem]
        if block.head_path is None or block.data_path is None:
            print(f"  WARN: {stem} raw .data or .head missing on disk; skip")
            continue
        head_info = _read_epilepsiae_head_for_streaming(block.head_path)
        sfreq = float(head_info.get("sample_freq", 0))
        out.append(
            {
                "stem": stem,
                "sfreq": sfreq,
                "new_gpu_path": gpu_path,
                "raw_data_path": block.data_path,
                "raw_head_path": block.head_path,
            }
        )
    return out


def pack_record(subject: str, stem: str) -> np.ndarray:
    """Produce (n_events, 2) packed [start_sec, end_sec] times for one record.

    whole_dets contract: each element shape (n_dets, 2) in SECONDS [start, end]
    relative to record start (verified at src/hfo_detector.py:82 / :224 and
    re-verified per Stage B.2 probe 2026-04-29).

    Filters per-channel detections to the subject-level refined channel list,
    then runs build_windows_from_detections with the legacy-aligned defaults
    (window_sec=0.5, ext_ms=30, chns_thr=0.5, time_axis_hz=500). Returns
    empty (0, 2) array if there are no participating channels or no surviving
    windows.
    """
    from src.group_event_analysis import build_windows_from_detections

    gpu_path = NEW_GPU_ROOT / subject / f"{stem}_gpu.npz"
    z = np.load(gpu_path, allow_pickle=True)
    chns_names = [str(c) for c in z["chns_names"]]
    whole_dets = z["whole_dets"]  # already in seconds; do NOT divide

    refine_chns_set = set(load_refine_chns_for_subject(subject))

    detections: Dict[str, np.ndarray] = {}
    for i, ch in enumerate(chns_names):
        if ch not in refine_chns_set:
            continue
        arr = np.atleast_2d(np.asarray(whole_dets[i], dtype=float))
        if arr.size == 0:
            continue
        detections[ch] = arr

    if not detections:
        return np.empty((0, 2), dtype=float)

    windows = build_windows_from_detections(
        detections,
        window_sec=0.5,
        chns_thr=0.5,
        ext_ms=30.0,
        time_axis_hz=500.0,
    )
    if not windows:
        return np.empty((0, 2), dtype=float)
    return np.array([(w.start, w.end) for w in windows], dtype=float)


def _empty_lagpat_record(start_t_epoch: float) -> Dict[str, np.ndarray]:
    return {
        "lagPatRaw": np.empty((0, 0), dtype=np.float64),
        "lagPatRank": np.empty((0, 0), dtype=np.int64),
        "eventsBool": np.empty((0, 0), dtype=np.float64),
        "chnNames": np.array([], dtype=object),
        "start_t": np.float64(start_t_epoch),
        "packedTimes": np.empty((0, 2), dtype=np.float64),
    }


def compute_lagpat_record(
    subject: str,
    stem: str,
    *,
    packed_times: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """End-to-end per-record pack + lagPat for the Epilepsiae new pipeline.

    Schema matches legacy {chnNames, lagPatRaw, lagPatRank, eventsBool, start_t}.
    NO Yuquan-style assumptions: CAR is already applied by load_epilepsiae_block,
    sfreq is variable (use as-is, no resample), one .data/.head per record, no
    .legacy_backup writes, no /mnt root.
    """
    from scipy.signal import butter, filtfilt

    from src.group_event_analysis import (
        EventWindow,
        compute_centroid_matrix_spectrogram,
        lag_rank_from_centroids,
    )
    from src.preprocessing import load_epilepsiae_block

    paths = EpilepsiaePaths()
    raw_blocks = _collect_raw_blocks(subject, paths)
    if stem not in raw_blocks:
        raise FileNotFoundError(f"No raw block for {subject}/{stem}")
    blk = raw_blocks[stem]
    if blk.head_path is None or blk.data_path is None:
        raise FileNotFoundError(f"Raw .data/.head missing on disk for {subject}/{stem}")

    # 1. CAR signal + notch already applied by loader
    pre = load_epilepsiae_block(
        data_path=blk.data_path,
        head_path=blk.head_path,
        reference="car",
    )
    sfreq = float(pre.sfreq)
    start_t_epoch = float(pre.start_time)
    sig_data = pre.data
    ch_names_full = list(pre.ch_names)

    if sfreq < NYQUIST_GATE_HZ:
        raise ValueError(
            f"sfreq {sfreq} < {NYQUIST_GATE_HZ} (Nyquist for ripple band) "
            "— this record should have been filtered upstream"
        )

    # 2. Restrict to refined channels (subject-level cache)
    refine_chns_set = set(load_refine_chns_for_subject(subject))
    keep_idx = [i for i, c in enumerate(ch_names_full) if c in refine_chns_set]
    if len(keep_idx) < 3:
        return _empty_lagpat_record(start_t_epoch)
    sig_pick = sig_data[keep_idx]
    pick_names = [ch_names_full[i] for i in keep_idx]

    # 3. Bandpass to ripple band [80, 250] Hz
    nyq = 0.5 * sfreq
    b, a = butter(4, [RIPPLE_BAND[0] / nyq, RIPPLE_BAND[1] / nyq], btype="band")
    sig_band = filtfilt(b, a, sig_pick, axis=-1)

    # 4. Pack windows (already in seconds); reuse a precomputed array when available
    if packed_times is None:
        packed_times = pack_record(subject, stem)
    if packed_times.size == 0:
        return _empty_lagpat_record(start_t_epoch)

    # 5. Per-channel detections dict (refine-filtered, in seconds)
    z_gpu = np.load(NEW_GPU_ROOT / subject / f"{stem}_gpu.npz", allow_pickle=True)
    gpu_chns = [str(c) for c in z_gpu["chns_names"]]
    detections: Dict[str, np.ndarray] = {}
    for i, ch in enumerate(gpu_chns):
        if ch not in refine_chns_set:
            continue
        arr = np.atleast_2d(np.asarray(z_gpu["whole_dets"][i], dtype=float))
        if arr.size > 0:
            detections[ch] = arr

    # 6. Per-segment (200s) centroid matrix (legacy stitched-segment semantic)
    n_pick = len(pick_names)
    n_ev = packed_times.shape[0]
    centroids = np.full((n_pick, n_ev), np.nan, dtype=np.float64)
    events_bool = np.zeros((n_pick, n_ev), dtype=np.float64)

    duration_sec = sig_band.shape[1] / sfreq
    seg_starts = np.arange(0.0, duration_sec, SEGMENT_SEC)
    seg_starts = np.append(seg_starts, duration_sec)
    min_seg_samples = int(0.05 * sfreq)

    for s0, s1 in zip(seg_starts[:-1], seg_starts[1:]):
        in_seg = np.where(
            (packed_times[:, 0] >= s0) & (packed_times[:, 1] <= s1)
        )[0]
        if in_seg.size == 0:
            continue
        i0, i1 = int(s0 * sfreq), int(s1 * sfreq)
        seg_band = sig_band[:, i0:i1]
        if seg_band.shape[1] < min_seg_samples:
            continue
        seg_windows = [
            EventWindow(
                start=float(packed_times[i, 0]),
                end=float(packed_times[i, 1]),
                event_id=int(i),
            )
            for i in in_seg
        ]
        seg_centroids, seg_evbool = compute_centroid_matrix_spectrogram(
            windows=seg_windows,
            detections=detections,
            ch_names=pick_names,
            x_band=seg_band,
            sfreq=sfreq,
            start_sec=float(s0),
        )
        for col_in_seg, ev_idx in enumerate(in_seg):
            centroids[:, ev_idx] = seg_centroids[:, col_in_seg]
            events_bool[:, ev_idx] = float(1.0) * seg_evbool[:, col_in_seg]

    # Drop any event column with n_participating == 0. Sources of empty
    # columns: window with start < 0, window crossing a 200s segment boundary,
    # or segment too short for the spectrogram. Keeping such columns would
    # contaminate downstream event counts, structural similarity (Stage C),
    # and rate-coupling stats (Stage D).
    keep_mask = events_bool.sum(axis=0) > 0
    if not keep_mask.any():
        return _empty_lagpat_record(start_t_epoch)
    centroids_kept = centroids[:, keep_mask]
    events_bool_kept = events_bool[:, keep_mask]
    packed_times_kept = packed_times[keep_mask]

    lag_raw, lag_rank = lag_rank_from_centroids(
        centroids_kept, events_bool_kept.astype(bool), align="first_centroid"
    )

    return {
        "lagPatRaw": lag_raw.astype(np.float64),
        "lagPatRank": lag_rank.astype(np.int64),
        "eventsBool": events_bool_kept.astype(np.float64),
        "chnNames": np.array(pick_names, dtype=object),
        "start_t": np.float64(start_t_epoch),
        "packedTimes": packed_times_kept.astype(np.float64),
    }


def _outputs_exist_and_loadable(pt_path: Path, lag_path: Path) -> bool:
    """Both files exist on disk AND can be loaded without raising.

    Catches a partially-written record left behind by a SIGKILL: the file
    is present but truncated, so np.load raises. In that case the caller
    treats the record as not-yet-done and re-runs.
    """
    if not (pt_path.exists() and lag_path.exists()):
        return False
    try:
        np.load(pt_path)
        with np.load(lag_path, allow_pickle=True) as z:
            _ = list(z.files)
        return True
    except Exception:
        return False


def process_one_record(
    subject: str,
    stem: str,
    *,
    force: bool = False,
) -> Dict[str, Any]:
    """Pack + lagPat for one (subject, stem); write outputs to OUTPUT_ROOT.

    Returns a per-record summary suitable for inclusion in _backfill_log.json.
    When force=False (default), skips records whose outputs already exist
    AND load cleanly; when force=True, overwrites unconditionally.
    """
    out_dir = OUTPUT_ROOT / subject
    pt_path = out_dir / f"{stem}_packedTimes.npy"
    lag_path = out_dir / f"{stem}_lagPat.npz"

    if not force and _outputs_exist_and_loadable(pt_path, lag_path):
        return {
            "stem": stem,
            "skipped": True,
            "reason": "outputs exist (loadable)",
            "pt_path": str(pt_path),
            "lag_path": str(lag_path),
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    lag = compute_lagpat_record(subject, stem)
    runtime_sec = time.time() - t0

    np.save(pt_path, lag["packedTimes"])
    np.savez_compressed(
        lag_path,
        chnNames=lag["chnNames"],
        lagPatRaw=lag["lagPatRaw"],
        lagPatRank=lag["lagPatRank"],
        eventsBool=lag["eventsBool"],
        start_t=lag["start_t"],
    )

    return {
        "stem": stem,
        "skipped": False,
        "n_events": int(lag["lagPatRaw"].shape[1]),
        "n_channels": int(len(lag["chnNames"])),
        "n_packed": int(lag["packedTimes"].shape[0]),
        "start_t": float(lag["start_t"]),
        "runtime_sec": float(runtime_sec),
        "pt_path": str(pt_path),
        "lag_path": str(lag_path),
    }


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    """Write JSON via tmp + rename so a SIGKILL mid-write does not corrupt the file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=2, default=str)
    tmp.replace(path)


def process_subject(
    subject: str,
    *,
    force: bool = False,
) -> Dict[str, Any]:
    """Run pack + lagPat for every record of a subject, with per-record logging.

    Failure-tolerant: if one record raises, log the failure and continue. The
    log is flushed atomically after every record so a crash leaves a usable
    state for resume on the next invocation.
    """
    out_dir = OUTPUT_ROOT / subject
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "_backfill_log.json"

    records = _discover_records(subject)
    log: Dict[str, Any] = {
        "subject": subject,
        "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "completed_at": None,
        "n_records_total": len(records),
        "n_records_done": 0,
        "n_skipped_existing": 0,
        "n_failed": 0,
        "failures": [],
        "per_record_seconds": {},
        "median_record_seconds": None,
    }
    _atomic_write_json(log_path, log)

    for rec in records:
        stem = rec["stem"]
        try:
            result = process_one_record(subject, stem, force=force)
            if result.get("skipped"):
                log["n_skipped_existing"] += 1
            else:
                log["n_records_done"] += 1
                log["per_record_seconds"][stem] = float(result["runtime_sec"])
        except Exception as exc:
            log["n_failed"] += 1
            log["failures"].append(
                {"stem": stem, "type": type(exc).__name__, "error": str(exc)}
            )
            print(f"  FAIL {stem}: {type(exc).__name__}: {exc}")
        _atomic_write_json(log_path, log)

    if log["per_record_seconds"]:
        log["median_record_seconds"] = float(
            np.median(list(log["per_record_seconds"].values()))
        )
    log["completed_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    _atomic_write_json(log_path, log)
    return log


def _smoke_print(subject: str) -> None:
    """Dry-print first record's metadata; do not write any files.

    Stage A scope: gpu/raw discovery only. Refine-list lookup belongs to
    Stage B.1 and is not invoked here, so subjects without _refineGpu.npz
    can still complete Stage A smoke without error.
    """
    recs = _discover_records(subject)
    if not recs:
        print(f"[smoke] subject={subject} no records discovered")
        return
    first = recs[0]
    z = np.load(first["new_gpu_path"], allow_pickle=True)
    chns_names = [str(c) for c in z["chns_names"]]
    whole_dets = z["whole_dets"]
    n_dets = sum(
        int(np.atleast_2d(np.asarray(d)).shape[0])
        for d in whole_dets
        if np.asarray(d).size
    )
    print(f"[smoke] subject={subject}  total_records={len(recs)}")
    print(f"[smoke] first stem={first['stem']}  sfreq={first['sfreq']}")
    print(f"[smoke]   gpu_path={first['new_gpu_path']}")
    print(f"[smoke]   raw_data={first['raw_data_path']}")
    print(f"[smoke]   raw_head={first['raw_head_path']}")
    print(f"[smoke]   n_chns_full={len(chns_names)}  n_dets_total={n_dets}")
    print("[smoke] (no files written)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Epilepsiae new-pipeline pack + lagPat backfill (Stage A skeleton).",
    )
    parser.add_argument("--subject", type=str, default=None)
    parser.add_argument("--stem", type=str, default=None,
                        help="Process a single record stem (B.4.a single-record mode).")
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Process first record of given subject only; dry-print, no writes.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if output files already exist (overwrite).",
    )
    args = parser.parse_args()

    if args.smoke:
        if not args.subject:
            parser.error("--smoke requires --subject")
        _smoke_print(args.subject)
        return

    if args.stem:
        if not args.subject:
            parser.error("--stem requires --subject")
        result = process_one_record(args.subject, args.stem, force=args.force)
        if result.get("skipped"):
            print(
                f"[B.4.a] subject={args.subject} stem={args.stem} SKIPPED "
                f"({result.get('reason')}); use --force to overwrite"
            )
            print(f"  pt_path={result['pt_path']}")
            print(f"  lag_path={result['lag_path']}")
        else:
            print(
                f"[B.4.a] subject={args.subject} stem={args.stem}  "
                f"n_events={result['n_events']}  n_channels={result['n_channels']}  "
                f"n_packed={result['n_packed']}  runtime={result['runtime_sec']:.1f}s"
            )
            print(f"  start_t={result['start_t']:.0f} (Unix epoch)")
            print(f"  pt_path={result['pt_path']}")
            print(f"  lag_path={result['lag_path']}")
        return

    if args.subject:
        log = process_subject(args.subject, force=args.force)
        median = log.get("median_record_seconds")
        median_str = f"{median:.1f}s" if median is not None else "n/a"
        print(
            f"[B.4.b] subject={args.subject}  "
            f"done={log['n_records_done']}/{log['n_records_total']}  "
            f"skipped={log['n_skipped_existing']}  failed={log['n_failed']}  "
            f"median_runtime={median_str}"
        )
        if log["failures"]:
            print(f"  first {min(5, len(log['failures']))} failures:")
            for f in log["failures"][:5]:
                print(f"    {f['stem']}: {f['type']}: {f['error']}")
        return

    if args.all:
        raise NotImplementedError(
            "Cohort batch driver (Stage B.5) not implemented yet. "
            "Run --subject <X> per subject; cross-subject parallel comes in B.5."
        )

    parser.error("provide one of --subject, --subject --stem, or --smoke")


if __name__ == "__main__":
    main()
