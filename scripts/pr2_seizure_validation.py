"""
PR2.5 Spatial-Extent Seizure Detection Validation

Produces four artifacts required by the plan:
  1. Per-EDF overlay plot (participation fraction + manual/detected onsets)
  2. 24h concatenated timeline (manual vs algorithm)
  3. Per-event onset/offset error scatter (FP/FN marked)
  4. Audit CSV  (per-EDF TP/FP/FN, median errors)

Usage:
    python scripts/pr2_seizure_validation.py [--subject litengsheng] [--n-jobs 4]
"""
import argparse
import csv
import json
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
    match_seizure_intervals,
    _flag_to_runs,
    _merge_close_runs,
    parse_seizure_annotation_events,
    read_edf_start_time,
)
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
        "record", "n_manual", "n_detected",
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
) -> dict:
    participation_t = feat["participation_t"]
    ch_ll_z = feat["ch_ll_z"]
    duration_sec = float(feat["duration_sec"])
    record_duration_sec = float(feat["record_duration_sec"])
    active_mask = ch_ll_z >= float(per_channel_k)
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
        "edf_path": str(edf_path),
        "start_epoch": start_epoch,
        "det": det,
        "manual": manual,
        "manual_rel": manual_rel,
        "manual_onset_only": manual_onset_only,
        "match": match,
        "elapsed": elapsed,
    }


# ── main ─────────────────────────────────────────────────────────────


def run_validation(
    subject: str,
    data_root: Path,
    output_dir: Path,
    per_channel_k: float = 3.5,
    min_active_frac: float = 0.25,
    min_duration_sec: float = 30.0,
    merge_gap_sec: float = 15.0,
    cache_dir: Path | None = None,
    n_jobs: int = 4,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = cache_dir if cache_dir is not None else (output_dir / "feature_cache")
    subj_dir = data_root / subject
    edfs = sorted(subj_dir.glob("*.edf"))
    if not edfs:
        print(f"No EDF files found for {subject} in {subj_dir}")
        return

    print(f"=== PR2.5 Validation: {subject} ({len(edfs)} EDFs, n_jobs={n_jobs}) ===",
          flush=True)
    print(
        f"    per_channel_k={per_channel_k}, min_active_frac={min_active_frac}, "
        f"min_dur={min_duration_sec}s, merge_gap={merge_gap_sec}s",
        flush=True,
    )

    # ── parallel detect ──
    t_wall = time.time()
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(_process_one_edf)(
            edf,
            per_channel_k,
            min_active_frac,
            min_duration_sec,
            merge_gap_sec,
            cache_dir,
        )
        for edf in edfs
    )
    wall = time.time() - t_wall
    print(f"\nAll {len(edfs)} EDFs done in {wall:.0f}s "
          f"(≈{wall/len(edfs):.0f}s/EDF effective)", flush=True)

    # ── serial aggregate + plot ──
    results.sort(key=lambda r: r["record"])
    audit_rows = []
    all_tp = []
    all_fp_count = 0
    all_fn_count = 0

    for r in results:
        det = r["det"]
        match = r["match"]
        manual_rel = r["manual_rel"]
        record = r["record"]

        n_tp = len(match["tp"])
        n_fp = len(match["fp"])
        n_fn = len(match["fn"])
        all_tp.extend(match["tp"])
        all_fp_count += n_fp
        all_fn_count += n_fn

        onset_errs = [t["onset_err"] for t in match["tp"]]
        offset_errs = [t["offset_err"] for t in match["tp"]]
        med_onset = float(np.median(onset_errs)) if onset_errs else float("nan")
        med_offset = float(np.median(offset_errs)) if offset_errs else float("nan")

        audit_rows.append({
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

        if manual_rel or r["manual_onset_only"] or len(det["onsets_sec"]) > 0:
            plot_single_edf(
                det,
                manual_rel,
                r["manual_onset_only"],
                record,
                output_dir / f"pr25_{record}_overlay.png",
            )

        if r["manual"]["raw_interval_details"] or r["manual_onset_only"]:
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
    total_man = sum(len(r["manual_rel"]) + len(r["manual_onset_only"]) for r in results)
    total_det = sum(len(r["det"]["onsets_sec"]) for r in results)
    total_tp = len(all_tp)
    recall = total_tp / total_man if total_man else 0.0
    precision = total_tp / total_det if total_det else 0.0

    print(f"\n{'='*60}")
    print(f"  SUMMARY: {subject}")
    print(f"  Manual intervals:  {total_man}")
    print(f"  Detected intervals:{total_det}")
    print(f"  TP={total_tp}  FP={all_fp_count}  FN={all_fn_count}")
    print(f"  Recall={recall:.3f}  Precision={precision:.3f}")
    if all_tp:
        onset_med = float(np.median([t["onset_err"] for t in all_tp]))
        offset_med = float(np.median([t["offset_err"] for t in all_tp]))
        print(f"  Median onset err:  {onset_med:.1f}s")
        print(f"  Median offset err: {offset_med:.1f}s")
    print(f"{'='*60}")

    plot_24h_timeline(results, subject, output_dir / f"pr25_{subject}_24h_timeline.png")
    plot_error_scatter(
        all_tp,
        all_fp_count,
        all_fn_count,
        subject,
        output_dir / f"pr25_{subject}_error_scatter.png",
    )
    write_audit_csv(audit_rows, output_dir / f"pr25_{subject}_audit.csv")
    print(f"\nArtifacts saved to {output_dir}/")


def main():
    ap = argparse.ArgumentParser(description="PR2.5 spatial-extent seizure validation")
    ap.add_argument("--subject", default="litengsheng")
    ap.add_argument("--data-root", default=str(DEFAULT_ROOT))
    ap.add_argument("--output-dir", default="results/pr2_seizure")
    ap.add_argument("--per-channel-k", type=float, default=3.5)
    ap.add_argument("--min-active-frac", type=float, default=0.25)
    ap.add_argument("--min-dur", type=float, default=30.0,
                    help="Min seizure duration (s)")
    ap.add_argument("--merge-gap", type=float, default=15.0,
                    help="Merge detections closer than this (s)")
    ap.add_argument("--cache-dir", default=None,
                    help="Optional feature cache directory; defaults to <output-dir>/feature_cache")
    ap.add_argument("--n-jobs", type=int, default=4,
                    help="Parallel EDF workers (4 is sweet spot for NFS)")
    args = ap.parse_args()
    run_validation(
        subject=args.subject,
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
        per_channel_k=args.per_channel_k,
        min_active_frac=args.min_active_frac,
        min_duration_sec=args.min_dur,
        merge_gap_sec=args.merge_gap,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
