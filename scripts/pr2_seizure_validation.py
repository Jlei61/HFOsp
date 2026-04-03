"""
PR2.5 Spatial-Extent Seizure Detection Validation

Produces four artifacts required by the plan:
  1. Per-EDF overlay plot (participation fraction + manual/detected onsets)
  2. 24h concatenated timeline (manual vs algorithm)
  3. Per-event onset/offset error scatter (FP/FN marked)
  4. Audit CSV  (per-EDF TP/FP/FN, median errors)

Usage:
    python scripts/pr2_seizure_validation.py [--dataset yuquan --subject litengsheng] [--n-jobs 4]
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from src.preprocessing import (
    detect_seizure_by_spatial_extent,
    detect_seizure_by_spatial_extent_epilepsiae,
    match_seizure_intervals,
    _flag_to_runs,
    _merge_close_runs,
    parse_seizure_annotation_events,
    read_edf_start_time,
)
from src.epilepsiae_dataset import survey_epilepsiae_dataset
from src.visualization import plot_bipolar_onset_context_from_edf

SEIZURE_LABELS = [
    "EEG SZ", "SZ", "SZ1", "SZ2", "SZ3", "SZ4", "SZ5", "SZ6", "SZ7",
    "SZ8", "SZ9", "SZ10",
    "EEG onset", "seizure", "Seizure", "SEIZURE",
    "onset", "Onset", "ictal", "Ictal",
    "sz onset", "seizure onset", "clinical seizure",
    "subclinical seizure", "electrographic seizure",
]

DEFAULT_ROOT = Path("/mnt/yuquan_data/yuquan_24h_edf")


# ── plotting helpers ─────────────────────────────────────────────────


def plot_single_edf(det, manual_rels, manual_onset_only, record, out_path):
    """Artifact 1: single-EDF overlay (participation fraction + onset lines)."""
    fig, ax = plt.subplots(1, 1, figsize=(18, 4.5))

    ax.plot(
        det["participation_t"],
        det["participation"],
        linewidth=0.7,
        color="#1f77b4",
        label="Active-channel fraction",
    )
    ax.axhline(
        det["min_active_frac"],
        color="#d62728",
        ls="--",
        lw=1.0,
        label=f"Participation thresh ({det['min_active_frac']:.2f})",
    )

    for on, off in manual_rels:
        ax.axvspan(on, off, color="lime", alpha=0.18, zorder=0)
        ax.axvline(on, color="green", ls="-", lw=1.2, alpha=0.7)
        ax.axvline(off, color="green", ls=":", lw=1.0, alpha=0.5)
    for on in manual_onset_only:
        ax.axvline(on, color="green", ls="-", lw=1.2, alpha=0.7)
    for on, off in zip(det["onsets_sec"], det["offsets_sec"]):
        ax.axvline(on, color="red", ls="-", lw=1.2, alpha=0.7)
        ax.axvline(off, color="red", ls=":", lw=1.0, alpha=0.5)

    ax.set_xlim(0, det["duration_sec"])
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Participation fraction")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(
        f"{record}  |  manual(green) vs detected(red) | k={det['per_channel_k']:.2f}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_24h_timeline(edf_results, subject, out_path):
    """Artifact 2: 24h concatenated overview — manual vs algorithm intervals."""
    fig, axes = plt.subplots(2, 1, figsize=(20, 5), sharex=True)
    ax_man, ax_det = axes
    ax_man.set_ylabel("Manual")
    ax_det.set_ylabel("Detected")

    t_min = min(r["start_epoch"] for r in edf_results)
    t_max = max(r["start_epoch"] + r["det"]["duration_sec"] for r in edf_results)

    for r in edf_results:
        base = r["start_epoch"] - t_min
        dur = r["det"]["duration_sec"]
        ax_man.axvspan(base, base + dur, color="#f0f0f0", zorder=0)
        ax_det.axvspan(base, base + dur, color="#f0f0f0", zorder=0)

        for on, off in r["manual_rel"]:
            ax_man.axvspan(base + on, base + off, color="green", alpha=0.5)
        for on, off in zip(r["det"]["onsets_sec"], r["det"]["offsets_sec"]):
            ax_det.axvspan(base + on, base + off, color="red", alpha=0.5)

    for ax in axes:
        ax.set_xlim(0, t_max - t_min)
        ax.set_yticks([])

    hours = (t_max - t_min) / 3600
    ax_det.set_xlabel(f"Time from first EDF start (s)  [{hours:.1f}h total]")
    fig.suptitle(f"{subject} — 24h seizure timeline: manual(green) vs detected(red)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_error_scatter(all_tp, all_fp_count, all_fn_count, subject, out_path):
    """Artifact 3: per-event onset/offset error scatter + FP/FN counts."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    onset_errs = [t["onset_err"] for t in all_tp]
    offset_errs = [t["offset_err"] for t in all_tp]

    ax = axes[0]
    if onset_errs:
        ax.scatter(range(len(onset_errs)), onset_errs, s=30, c="#1f77b4", label="onset err (s)")
        ax.axhline(0, color="gray", ls="--", lw=0.8)
        med = float(np.median(onset_errs))
        ax.axhline(med, color="red", ls="--", lw=1, label=f"median={med:.1f}s")
    ax.set_xlabel("TP event index")
    ax.set_ylabel("Onset error (detected − manual) [s]")
    ax.set_title(f"Onset error  (n_TP={len(all_tp)}, FP={all_fp_count}, FN={all_fn_count})")
    ax.legend(fontsize=8)

    ax = axes[1]
    if offset_errs:
        ax.scatter(range(len(offset_errs)), offset_errs, s=30, c="#ff7f0e", label="offset err (s)")
        ax.axhline(0, color="gray", ls="--", lw=0.8)
        med = float(np.median(offset_errs))
        ax.axhline(med, color="red", ls="--", lw=1, label=f"median={med:.1f}s")
    ax.set_xlabel("TP event index")
    ax.set_ylabel("Offset error (detected − manual) [s]")
    ax.set_title("Offset error")
    ax.legend(fontsize=8)

    fig.suptitle(f"{subject} — per-event error scatter", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_audit_csv(rows, out_path):
    """Artifact 4: audit table CSV."""
    fieldnames = [
        "dataset", "subject", "record", "n_manual", "n_detected",
        "TP", "FP", "FN",
        "recall", "precision", "f1",
        "median_onset_err_s", "median_offset_err_s",
        "peak_mem_est_mb", "elapsed_sec",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _epilepsiae_block_cache_path(cache_dir: Path, block_stem: str) -> Path:
    return cache_dir / f"{block_stem}_seizureFeatures.npz"


def _load_or_build_feature_cache_epilepsiae(
    data_path: Path,
    head_path: Path,
    cache_dir: Path,
) -> dict:
    """Cache per-block Epilepsiae LL z-traces just like the EDF path."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _epilepsiae_block_cache_path(cache_dir, data_path.stem)
    if cache_path.exists():
        meta = np.load(str(cache_path), allow_pickle=False)
        if "ch_ll_z" in meta.files:
            return {
                "participation_t": meta["participation_t"],
                "ch_ll_z": meta["ch_ll_z"],
                "sfreq": float(meta["sfreq"][0]),
                "n_channels": int(meta["n_channels"][0]),
                "n_records": int(meta["n_records"][0]),
                "record_duration_sec": float(meta["record_duration_sec"][0]),
                "duration_sec": float(meta["duration_sec"][0]),
                "peak_mem_est_mb": float(meta["peak_mem_est_mb"][0]),
            }

    det = detect_seizure_by_spatial_extent_epilepsiae(
        data_path,
        head_path,
        per_channel_k=99.0,
        min_active_frac=1.0,
        min_duration_sec=9999.0,
        merge_gap_sec=1.0,
    )
    np.savez_compressed(
        cache_path,
        participation_t=det["participation_t"],
        ch_ll_z=det["ch_ll_z"],
        sfreq=np.array([det["sfreq"]], dtype=np.float64),
        n_channels=np.array([det["n_channels"]], dtype=np.int64),
        n_records=np.array([det["n_records"]], dtype=np.int64),
        record_duration_sec=np.array([det["record_duration_sec"]], dtype=np.float64),
        duration_sec=np.array([det["duration_sec"]], dtype=np.float64),
        peak_mem_est_mb=np.array([det["peak_mem_est_mb"]], dtype=np.float64),
    )
    return {
        "participation_t": det["participation_t"],
        "ch_ll_z": det["ch_ll_z"],
        "sfreq": float(det["sfreq"]),
        "n_channels": int(det["n_channels"]),
        "n_records": int(det["n_records"]),
        "record_duration_sec": float(det["record_duration_sec"]),
        "duration_sec": float(det["duration_sec"]),
        "peak_mem_est_mb": float(det["peak_mem_est_mb"]),
    }


def _epilepsiae_manual_rel(block_row: dict, seizure_rows: list[dict]) -> list[tuple[float, float]]:
    """Clip recording-level EEG seizure intervals to the current block."""
    return [
        (seg["rel_onset_sec"], seg["rel_offset_sec"])
        for seg in _epilepsiae_manual_segments(block_row, seizure_rows)
    ]


def _epilepsiae_manual_segments(block_row: dict, seizure_rows: list[dict]) -> list[dict]:
    """Return block-clipped seizure segments with stable seizure ids."""
    block_start = float(block_row["block_start_epoch"])
    block_end = float(block_row["block_end_epoch"])
    manual_segments = []
    for seizure in seizure_rows:
        onset = seizure.get("eeg_onset_epoch")
        offset = seizure.get("eeg_offset_epoch")
        if onset is None or offset is None:
            continue
        onset = float(onset)
        offset = float(offset)
        overlap_on = max(onset, block_start)
        overlap_off = min(offset, block_end)
        if overlap_off > overlap_on:
            manual_segments.append(
                {
                    "seizure_id": str(seizure["seizure_id"]),
                    "recording_id": str(seizure["recording_id"]),
                    "abs_onset_epoch": overlap_on,
                    "abs_offset_epoch": overlap_off,
                    "rel_onset_sec": overlap_on - block_start,
                    "rel_offset_sec": overlap_off - block_start,
                }
            )
    manual_segments.sort(key=lambda x: (x["abs_onset_epoch"], x["abs_offset_epoch"]))
    return manual_segments


def _merge_interval_pairs(intervals, gap_sec=0.0):
    """Merge absolute intervals if they overlap or nearly touch."""
    pairs = sorted((float(on), float(off)) for on, off in intervals if float(off) > float(on))
    if not pairs:
        return []
    merged = [pairs[0]]
    for on, off in pairs[1:]:
        prev_on, prev_off = merged[-1]
        if on <= (prev_off + float(gap_sec)):
            merged[-1] = (prev_on, max(prev_off, off))
        else:
            merged.append((on, off))
    return merged


def _subject_level_match(results, dataset):
    """Compute subject-level event metrics on a single absolute timeline."""
    dataset = str(dataset).strip().lower()
    detected_abs = []
    manual_abs = []
    manual_onset_only_abs = []
    for r in results:
        detected_abs.extend(r.get("detected_abs_intervals", []))
        manual_abs.extend(r.get("manual_abs_intervals", []))
        manual_onset_only_abs.extend(r.get("manual_abs_onset_only", []))

    detected_pairs = _merge_interval_pairs(
        detected_abs,
        gap_sec=1.0 if dataset == "epilepsiae" else 0.0,
    )
    if dataset == "epilepsiae":
        by_seizure = {}
        for seg in manual_abs:
            seizure_id = str(seg["seizure_id"])
            prev = by_seizure.get(seizure_id)
            if prev is None:
                by_seizure[seizure_id] = (
                    float(seg["abs_onset_epoch"]),
                    float(seg["abs_offset_epoch"]),
                )
            else:
                by_seizure[seizure_id] = (
                    min(prev[0], float(seg["abs_onset_epoch"])),
                    max(prev[1], float(seg["abs_offset_epoch"])),
                )
        manual_pairs = sorted(by_seizure.values())
        onset_only_abs = []
    else:
        manual_pairs = sorted(
            (float(seg["abs_onset_epoch"]), float(seg["abs_offset_epoch"]))
            for seg in manual_abs
        )
        onset_only_abs = sorted(float(x) for x in manual_onset_only_abs)

    interval_match = match_seizure_intervals(manual_pairs, detected_pairs)
    matched_detected = {tp["detected_idx"] for tp in interval_match["tp"]}
    onset_only_tp = _match_onset_only_annotations(onset_only_abs, detected_pairs, matched_detected)
    tp = interval_match["tp"] + onset_only_tp
    fp = [di for di in range(len(detected_pairs)) if di not in matched_detected]
    fn = interval_match["fn"] + list(range(len(onset_only_tp), len(onset_only_abs)))
    n_manual_total = len(manual_pairs) + len(onset_only_abs)
    n_tp = len(tp)
    n_det = len(detected_pairs)
    recall = n_tp / n_manual_total if n_manual_total else 1.0
    precision = n_tp / n_det if n_det else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "manual_pairs": manual_pairs,
        "detected_pairs": detected_pairs,
        "manual_onset_only_abs": onset_only_abs,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }


# ── per-EDF worker (runs in subprocess via joblib) ───────────────────


def _feature_cache_path(cache_dir: Path, edf_path: Path) -> Path:
    return cache_dir / f"{edf_path.stem}_seizureFeatures.npz"


def _load_or_build_feature_cache(edf_path: Path, cache_dir: Path) -> dict:
    """
    The expensive part is reading EDF. Cache feature traces once and reuse them.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _feature_cache_path(cache_dir, edf_path)
    if cache_path.exists():
        meta = np.load(str(cache_path), allow_pickle=False)
        if "ch_ll_z" in meta.files:
            return {
                "participation_t": meta["participation_t"],
                "ch_ll_z": meta["ch_ll_z"],
                "sfreq": float(meta["sfreq"][0]),
                "n_channels": int(meta["n_channels"][0]),
                "n_records": int(meta["n_records"][0]),
                "record_duration_sec": float(meta["record_duration_sec"][0]),
                "duration_sec": float(meta["duration_sec"][0]),
                "peak_mem_est_mb": float(meta["peak_mem_est_mb"][0]),
            }

    det = detect_seizure_by_spatial_extent(
        edf_path,
        per_channel_k=99.0,
        min_active_frac=1.0,
        min_duration_sec=9999.0,
        merge_gap_sec=1.0,
    )
    np.savez_compressed(
        cache_path,
        participation_t=det["participation_t"],
        ch_ll_z=det["ch_ll_z"],
        sfreq=np.array([det["sfreq"]], dtype=np.float64),
        n_channels=np.array([det["n_channels"]], dtype=np.int64),
        n_records=np.array([det["n_records"]], dtype=np.int64),
        record_duration_sec=np.array([det["record_duration_sec"]], dtype=np.float64),
        duration_sec=np.array([det["duration_sec"]], dtype=np.float64),
        peak_mem_est_mb=np.array([det["peak_mem_est_mb"]], dtype=np.float64),
    )
    return {
        "participation_t": det["participation_t"],
        "ch_ll_z": det["ch_ll_z"],
        "sfreq": float(det["sfreq"]),
        "n_channels": int(det["n_channels"]),
        "n_records": int(det["n_records"]),
        "record_duration_sec": float(det["record_duration_sec"]),
        "duration_sec": float(det["duration_sec"]),
        "peak_mem_est_mb": float(det["peak_mem_est_mb"]),
    }


def _apply_thresholds_to_feature_traces(
    feat: dict,
    *,
    per_channel_k: float,
    min_active_frac: float,
    min_duration_sec: float,
    merge_gap_sec: float,
    min_channel_consec_sec: float = 0.0,
) -> dict:
    from src.preprocessing import _sustained_active_mask
    participation_t = feat["participation_t"]
    ch_ll_z = feat["ch_ll_z"]
    duration_sec = float(feat["duration_sec"])
    record_duration_sec = float(feat["record_duration_sec"])
    active_mask = ch_ll_z >= float(per_channel_k)
    min_consec = max(1, int(round(float(min_channel_consec_sec) / float(record_duration_sec))))
    if min_consec > 1:
        active_mask = _sustained_active_mask(active_mask, min_consec)
    participation = active_mask.mean(axis=0)
    ictal_flag = participation >= float(min_active_frac)

    runs = _flag_to_runs(ictal_flag)
    merged = _merge_close_runs(runs, participation_t, merge_gap_sec)

    onsets = []
    offsets = []
    ictal_mask = np.zeros((len(participation_t),), dtype=bool)
    for s_idx, e_idx in merged:
        t0 = float(participation_t[s_idx])
        t1 = float(participation_t[min(e_idx - 1, len(participation_t) - 1)] + record_duration_sec)
        if (t1 - t0) < float(min_duration_sec):
            continue
        onsets.append(t0)
        offsets.append(t1)
        ictal_mask[s_idx:e_idx] = True

    return {
        "onsets_sec": np.asarray(onsets, dtype=np.float64),
        "offsets_sec": np.asarray(offsets, dtype=np.float64),
        "ictal_mask": ictal_mask,
        "participation_t": participation_t,
        "participation": participation,
        "active_counts": active_mask.sum(axis=0),
        "per_channel_k": float(per_channel_k),
        "min_active_frac": float(min_active_frac),
        "sfreq": float(feat["sfreq"]),
        "n_channels": int(feat["n_channels"]),
        "duration_sec": duration_sec,
        "record_duration_sec": record_duration_sec,
        "peak_mem_est_mb": float(feat["peak_mem_est_mb"]),
    }


def _match_onset_only_annotations(onset_only, detected_pairs, matched_detected, tol_sec=60.0):
    """Match onset-only labels by containment or onset proximity."""
    tp = []
    for onset_sec in onset_only:
        best_di = -1
        best_dist = float("inf")
        for di, (d_on, d_off) in enumerate(detected_pairs):
            if di in matched_detected:
                continue
            contains = d_on <= onset_sec <= d_off
            dist = abs(d_on - onset_sec)
            if contains or dist <= float(tol_sec):
                if dist < best_dist:
                    best_dist = dist
                    best_di = di
        if best_di >= 0:
            tp.append({
                "manual_idx": None,
                "detected_idx": best_di,
                "onset_err": float(detected_pairs[best_di][0] - onset_sec),
                "offset_err": float("nan"),
            })
            matched_detected.add(best_di)
    return tp


def _process_one_edf(
    edf_path: Path,
    per_channel_k: float,
    min_active_frac: float,
    min_duration_sec: float,
    merge_gap_sec: float,
    cache_dir: Path,
    min_channel_consec_sec: float = 0.0,
) -> dict:
    """Pure-compute worker: detect + parse annotations.  No matplotlib."""
    record = edf_path.stem
    t0 = time.time()
    start_epoch = read_edf_start_time(edf_path)
    feat = _load_or_build_feature_cache(edf_path, cache_dir)
    det = _apply_thresholds_to_feature_traces(
        feat,
        per_channel_k=per_channel_k,
        min_active_frac=min_active_frac,
        min_duration_sec=min_duration_sec,
        merge_gap_sec=merge_gap_sec,
        min_channel_consec_sec=min_channel_consec_sec,
    )
    elapsed = time.time() - t0

    manual = parse_seizure_annotation_events(
        edf_path, SEIZURE_LABELS, start_epoch,
    )
    manual_rel = [
        (float(on - start_epoch), float(off - start_epoch))
        for on, off in manual["intervals"]
    ]
    manual_onset_only = [float(on - start_epoch) for on in manual["orphan_onsets"]]

    detected_pairs = list(zip(
        det["onsets_sec"].tolist(), det["offsets_sec"].tolist(),
    ))
    interval_match = match_seizure_intervals(manual_rel, detected_pairs)
    matched_detected = {tp["detected_idx"] for tp in interval_match["tp"]}
    onset_only_tp = _match_onset_only_annotations(
        manual_onset_only, detected_pairs, matched_detected
    )
    match = {
        "tp": interval_match["tp"] + onset_only_tp,
        "fp": [di for di in range(len(detected_pairs)) if di not in matched_detected],
        "fn": interval_match["fn"] + list(range(len(onset_only_tp), len(manual_onset_only))),
    }
    n_manual_total = len(manual_rel) + len(manual_onset_only)
    n_tp = len(match["tp"])
    n_det = len(detected_pairs)
    match["recall"] = n_tp / n_manual_total if n_manual_total else 1.0
    match["precision"] = n_tp / n_det if n_det else 0.0
    match["f1"] = (
        2 * match["precision"] * match["recall"] / (match["precision"] + match["recall"])
        if (match["precision"] + match["recall"]) > 0
        else 0.0
    )

    print(f"  [{record}] {len(det['onsets_sec'])} det, "
          f"{len(manual_rel)} interval + {len(manual_onset_only)} onset-only manual, "
          f"TP={len(match['tp'])} FP={len(match['fp'])} FN={len(match['fn'])}, "
          f"mem≈{det['peak_mem_est_mb']}MB, {elapsed:.1f}s", flush=True)

    return {
        "record": record,
        "dataset": "yuquan",
        "edf_path": str(edf_path),
        "start_epoch": start_epoch,
        "det": det,
        "manual": manual,
        "manual_rel": manual_rel,
        "manual_onset_only": manual_onset_only,
        "manual_abs_intervals": [
            {"abs_onset_epoch": float(on), "abs_offset_epoch": float(off)}
            for on, off in manual["intervals"]
        ],
        "manual_abs_onset_only": [float(on) for on in manual["orphan_onsets"]],
        "detected_abs_intervals": [
            (float(start_epoch + on), float(start_epoch + off)) for on, off in detected_pairs
        ],
        "match": match,
        "elapsed": elapsed,
    }


def _process_one_epilepsiae_block(
    block_row: dict,
    seizure_rows: list[dict],
    per_channel_k: float,
    min_active_frac: float,
    min_duration_sec: float,
    merge_gap_sec: float,
    cache_dir: Path,
    min_channel_consec_sec: float = 0.0,
) -> dict:
    """Pure-compute worker for one Epilepsiae raw block."""
    data_path = Path(block_row["data_path"])
    head_path = Path(block_row["head_path"])
    record = str(block_row["block_stem"])
    t0 = time.time()
    feat = _load_or_build_feature_cache_epilepsiae(data_path, head_path, cache_dir)
    det = _apply_thresholds_to_feature_traces(
        feat,
        per_channel_k=per_channel_k,
        min_active_frac=min_active_frac,
        min_duration_sec=min_duration_sec,
        merge_gap_sec=merge_gap_sec,
        min_channel_consec_sec=min_channel_consec_sec,
    )
    elapsed = time.time() - t0
    manual_segments = _epilepsiae_manual_segments(block_row, seizure_rows)
    manual_rel = [
        (seg["rel_onset_sec"], seg["rel_offset_sec"])
        for seg in manual_segments
    ]
    detected_pairs = list(zip(det["onsets_sec"].tolist(), det["offsets_sec"].tolist()))
    match = match_seizure_intervals(manual_rel, detected_pairs)

    print(
        f"  [{record}] {len(det['onsets_sec'])} det, "
        f"{len(manual_rel)} interval manual, "
        f"TP={len(match['tp'])} FP={len(match['fp'])} FN={len(match['fn'])}, "
        f"mem≈{det['peak_mem_est_mb']}MB, {elapsed:.1f}s",
        flush=True,
    )
    return {
        "record": record,
        "dataset": "epilepsiae",
        "data_path": str(data_path),
        "head_path": str(head_path),
        "start_epoch": float(block_row["block_start_epoch"]),
        "det": det,
        "manual": {"raw_interval_details": [], "orphan_onsets": []},
        "manual_rel": manual_rel,
        "manual_onset_only": [],
        "manual_abs_intervals": manual_segments,
        "manual_abs_onset_only": [],
        "detected_abs_intervals": [
            (
                float(block_row["block_start_epoch"] + on),
                float(block_row["block_start_epoch"] + off),
            )
            for on, off in detected_pairs
        ],
        "match": match,
        "elapsed": elapsed,
    }


# ── main ─────────────────────────────────────────────────────────────


def run_validation(
    subject: str,
    dataset: str,
    data_root: Path,
    output_dir: Path,
    per_channel_k: float = 5.0,
    min_active_frac: float = 0.30,
    min_duration_sec: float = 30.0,
    merge_gap_sec: float = 5.0,
    min_channel_consec_sec: float = 0.0,
    cache_dir: Path | None = None,
    n_jobs: int = 4,
    make_plots: bool = True,
    write_audit: bool = True,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = cache_dir if cache_dir is not None else (output_dir / "feature_cache")
    dataset = str(dataset).strip().lower()
    workers = []
    subject_desc = ""
    if dataset == "yuquan":
        subj_dir = data_root / subject
        edfs = sorted(subj_dir.glob("*.edf"))
        if not edfs:
            print(f"No EDF files found for {subject} in {subj_dir}")
            return None
        workers = [
            delayed(_process_one_edf)(
                edf,
                per_channel_k,
                min_active_frac,
                min_duration_sec,
                merge_gap_sec,
                cache_dir,
                min_channel_consec_sec,
            )
            for edf in edfs
        ]
        subject_desc = f"{len(edfs)} EDFs"
    elif dataset == "epilepsiae":
        inventory = survey_epilepsiae_dataset()
        block_rows = [
            row
            for row in inventory.block_rows
            if str(row["subject"]) == str(subject)
            and bool(row["head_exists"])
            and bool(row["data_exists"])
            and int(row["intracranial_channels"] or 0) >= 2
        ]
        seizure_rows = [
            row
            for row in inventory.seizure_rows
            if str(row["subject"]) == str(subject)
            and bool(row["has_complete_eeg_interval"])
        ]
        if not block_rows:
            print(f"No Epilepsiae raw blocks found for subject {subject}")
            return None
        workers = [
            delayed(_process_one_epilepsiae_block)(
                block_row,
                seizure_rows,
                per_channel_k,
                min_active_frac,
                min_duration_sec,
                merge_gap_sec,
                cache_dir,
                min_channel_consec_sec,
            )
            for block_row in block_rows
        ]
        subject_desc = f"{len(block_rows)} raw blocks"
    else:
        raise ValueError("dataset must be 'yuquan' or 'epilepsiae'")

    print(f"=== PR2.5 Validation: {dataset}/{subject} ({subject_desc}, n_jobs={n_jobs}) ===",
          flush=True)
    print(
        f"    per_channel_k={per_channel_k}, min_active_frac={min_active_frac}, "
        f"min_dur={min_duration_sec}s, merge_gap={merge_gap_sec}s, "
        f"consec={min_channel_consec_sec}s",
        flush=True,
    )

    # ── parallel detect ──
    t_wall = time.time()
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(workers)
    wall = time.time() - t_wall
    print(f"\nAll {len(results)} items done in {wall:.0f}s "
          f"(≈{wall/max(len(results), 1):.0f}s/item effective)", flush=True)

    # ── serial aggregate + plot ──
    results.sort(key=lambda r: (r["start_epoch"], r["record"]))
    audit_rows = []

    for r in results:
        det = r["det"]
        match = r["match"]
        manual_rel = r["manual_rel"]
        record = r["record"]

        n_tp = len(match["tp"])
        n_fp = len(match["fp"])
        n_fn = len(match["fn"])

        onset_errs = [t["onset_err"] for t in match["tp"]]
        offset_errs = [t["offset_err"] for t in match["tp"] if np.isfinite(t["offset_err"])]
        med_onset = float(np.median(onset_errs)) if onset_errs else float("nan")
        med_offset = float(np.median(offset_errs)) if offset_errs else float("nan")

        audit_rows.append({
            "dataset": dataset,
            "subject": subject,
            "record": record,
            "n_manual": len(manual_rel) + len(r["manual_onset_only"]),
            "n_detected": len(list(zip(det["onsets_sec"], det["offsets_sec"]))),
            "TP": n_tp, "FP": n_fp, "FN": n_fn,
            "recall": f"{match['recall']:.3f}",
            "precision": f"{match['precision']:.3f}",
            "f1": f"{match['f1']:.3f}",
            "median_onset_err_s": f"{med_onset:.2f}",
            "median_offset_err_s": f"{med_offset:.2f}",
            "peak_mem_est_mb": det["peak_mem_est_mb"],
            "elapsed_sec": f"{r['elapsed']:.1f}",
        })

        if make_plots and (manual_rel or r["manual_onset_only"] or len(det["onsets_sec"]) > 0):
            plot_single_edf(
                det,
                manual_rel,
                r["manual_onset_only"],
                record,
                output_dir / f"pr25_{record}_overlay.png",
            )

        if make_plots and dataset == "yuquan" and (r["manual"]["raw_interval_details"] or r["manual_onset_only"]):
            onset_dir = output_dir / "manual_onset_context"
            onset_dir.mkdir(parents=True, exist_ok=True)
            for idx, detail in enumerate(r["manual"]["raw_interval_details"]):
                rel_on = float(detail["onset_epoch"] - r["start_epoch"])
                rel_off = (
                    None
                    if detail["offset_epoch"] is None
                    else float(detail["offset_epoch"] - r["start_epoch"])
                )
                fig = plot_bipolar_onset_context_from_edf(
                    edf_path=r["edf_path"],
                    onset_sec=rel_on,
                    offset_sec=rel_off,
                    output_dir=output_dir,
                    output_prefix=f"{record}_manual_{idx:02d}",
                    pre_sec=15.0,
                    post_sec=30.0,
                    channels="all",
                    title=(
                        f"{record} | manual onset {idx:02d} | "
                        f"label={detail['label']} | offset_source={detail['offset_source']}"
                    ),
                )
                fig.savefig(
                    onset_dir / f"pr25_{record}_manual_onset_{idx:02d}_bipolar_raw.png",
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close(fig)
            for idx, onset_sec in enumerate(r["manual_onset_only"]):
                fig = plot_bipolar_onset_context_from_edf(
                    edf_path=r["edf_path"],
                    onset_sec=float(onset_sec),
                    offset_sec=None,
                    output_dir=output_dir,
                    output_prefix=f"{record}_orphan_{idx:02d}",
                    pre_sec=15.0,
                    post_sec=30.0,
                    channels="all",
                    title=f"{record} | orphan onset {idx:02d} | no reliable offset",
                )
                fig.savefig(
                    onset_dir / f"pr25_{record}_orphan_onset_{idx:02d}_bipolar_raw.png",
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close(fig)

    # ── summary ──
    subject_match = _subject_level_match(results, dataset)
    all_tp = subject_match["tp"]
    all_fp_count = len(subject_match["fp"])
    all_fn_count = len(subject_match["fn"])
    total_man = len(subject_match["manual_pairs"]) + len(subject_match["manual_onset_only_abs"])
    total_det = len(subject_match["detected_pairs"])
    total_tp = len(all_tp)
    recall = float(subject_match["recall"])
    precision = float(subject_match["precision"])

    print(f"\n{'='*60}")
    print(f"  SUMMARY: {dataset}/{subject}")
    print(f"  Manual events:     {total_man}")
    print(f"  Detected intervals:{total_det}")
    print(f"  TP={total_tp}  FP={all_fp_count}  FN={all_fn_count}")
    print(f"  Recall={recall:.3f}  Precision={precision:.3f}")
    if all_tp:
        onset_med = float(np.median([t["onset_err"] for t in all_tp]))
        finite_offset = [t["offset_err"] for t in all_tp if np.isfinite(t["offset_err"])]
        offset_med = float(np.median(finite_offset)) if finite_offset else float("nan")
        print(f"  Median onset err:  {onset_med:.1f}s")
        print(f"  Median offset err: {offset_med:.1f}s")
    print(f"{'='*60}")

    timeline_path = output_dir / f"pr25_{dataset}_{subject}_timeline.png"
    error_scatter_path = output_dir / f"pr25_{dataset}_{subject}_error_scatter.png"
    audit_path = output_dir / f"pr25_{dataset}_{subject}_audit.csv"
    if make_plots:
        plot_24h_timeline(results, f"{dataset}/{subject}", timeline_path)
        plot_error_scatter(
            all_tp,
            all_fp_count,
            all_fn_count,
            f"{dataset}/{subject}",
            error_scatter_path,
        )
    if write_audit:
        write_audit_csv(audit_rows, audit_path)
    print(f"\nArtifacts saved to {output_dir}/")
    onset_med = float(np.median([t["onset_err"] for t in all_tp])) if all_tp else float("nan")
    finite_offset = [t["offset_err"] for t in all_tp if np.isfinite(t["offset_err"])]
    offset_med = float(np.median(finite_offset)) if finite_offset else float("nan")
    fp_per_manual = (all_fp_count / total_man) if total_man else float("inf")
    return {
        "dataset": dataset,
        "subject": subject,
        "n_records": len(results),
        "n_manual": total_man,
        "n_detected": total_det,
        "TP": total_tp,
        "FP": all_fp_count,
        "FN": all_fn_count,
        "recall": float(recall),
        "precision": float(precision),
        "f1": float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0,
        "median_onset_err_s": onset_med,
        "median_offset_err_s": offset_med,
        "fp_per_manual": float(fp_per_manual),
        "per_channel_k": float(per_channel_k),
        "min_active_frac": float(min_active_frac),
        "min_duration_sec": float(min_duration_sec),
        "merge_gap_sec": float(merge_gap_sec),
        "min_channel_consec_sec": float(min_channel_consec_sec),
        "output_dir": str(output_dir),
        "audit_csv": str(audit_path) if write_audit else None,
        "timeline_png": str(timeline_path) if make_plots else None,
        "error_scatter_png": str(error_scatter_path) if make_plots else None,
    }


def main():
    ap = argparse.ArgumentParser(description="PR2.5 spatial-extent seizure validation")
    ap.add_argument("--dataset", choices=["yuquan", "epilepsiae"], default="yuquan")
    ap.add_argument("--subject", default="litengsheng")
    ap.add_argument("--data-root", default=str(DEFAULT_ROOT))
    ap.add_argument("--output-dir", default="results/pr2_seizure")
    ap.add_argument("--per-channel-k", type=float, default=5.0)
    ap.add_argument("--min-active-frac", type=float, default=0.30)
    ap.add_argument("--min-dur", type=float, default=30.0,
                    help="Min seizure duration (s)")
    ap.add_argument("--merge-gap", type=float, default=5.0,
                    help="Merge detections closer than this (s)")
    ap.add_argument("--min-channel-consec", type=float, default=0.0,
                    help="Channel must sustain z > k for this many consecutive seconds")
    ap.add_argument("--cache-dir", default=None,
                    help="Optional feature cache directory; defaults to <output-dir>/feature_cache")
    ap.add_argument("--n-jobs", type=int, default=4,
                    help="Parallel EDF workers (4 is sweet spot for NFS)")
    args = ap.parse_args()
    run_validation(
        subject=args.subject,
        dataset=args.dataset,
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
        per_channel_k=args.per_channel_k,
        min_active_frac=args.min_active_frac,
        min_duration_sec=args.min_dur,
        merge_gap_sec=args.merge_gap,
        min_channel_consec_sec=args.min_channel_consec,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
