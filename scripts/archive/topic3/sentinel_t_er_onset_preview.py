"""Sentinel preview for per-channel ER onset timing before formal Step 3.

This script answers one narrow question on the already-selected Step 2
sentinel seizures: can ER produce a stable channel recruitment order
*before* clinical onset when we move the search window earlier?

It intentionally does not implement the full Step 3 contract
(per-subject permutation-calibrated lambda, final Page-Hinkley
statistics, rank gating). Instead it exports preview-only per-channel
``t_ER_onset`` tables for the same sentinel seizure set currently used in
Step 2 figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.sentinel_pr6a_step2 import (
    _focal_channels,
    _load_focus_rel,
    _load_lagpat,
    _pick_display_cluster,
)
from src.ictal_onset_extraction import (
    BROAD_ER_BANDS,
    GAMMA_ER_BANDS,
    baseline_zscore_er,
    compute_er,
    detect_er_onset_preview,
    extract_seizure_window,
    preview_threshold_from_baseline,
    resolve_baseline_window,
    resolve_detection_window,
)


STEP2_SUMMARY_PATH = (
    _PROJECT_ROOT
    / "results"
    / "interictal_propagation"
    / "ictal_alignment"
    / "_sentinel_step2"
    / "sentinel_step2_summary.json"
)
OUT_DIR = _PROJECT_ROOT / "results" / "seizure_onset" / "er_onset_preview"


def _channel_role(ch_name: str, *, focal_upper: set[str], high_hi_upper: set[str]) -> str:
    ch_upper = ch_name.upper()
    if ch_upper in high_hi_upper and ch_upper in focal_upper:
        return "high_hi_ictal"
    if ch_upper in high_hi_upper:
        return "high_hi_index"
    if ch_upper in focal_upper:
        return "ictal"
    return "other"


def _float_or_none(x):
    if x is None:
        return None
    x = float(x)
    if not np.isfinite(x):
        return None
    return x


def _post_window_peak(z_1d: np.ndarray, t_axis_er: np.ndarray, *, start_sec: float = 0.0, end_sec: float = 30.0) -> float | None:
    mask = (t_axis_er >= float(start_sec)) & (t_axis_er <= float(end_sec))
    if not mask.any():
        return None
    with np.errstate(invalid="ignore"):
        val = np.nanmax(z_1d[mask])
    return _float_or_none(val)


def _sorted_detected_rows(rows: List[dict]) -> List[dict]:
    return sorted(
        [r for r in rows if r["onset_detected"]],
        key=lambda r: (r["t_er_onset_sec"], r["channel"]),
    )


def _order_list(rows: List[dict], *, preclinical_only: bool = False) -> List[dict]:
    detected = _sorted_detected_rows(rows)
    if preclinical_only:
        detected = [r for r in detected if r["t_er_onset_sec"] is not None and r["t_er_onset_sec"] < 0.0]
    return [
        {
            "channel": r["channel"],
            "t_er_onset_sec": r["t_er_onset_sec"],
            "role": r["role"],
            "high_hi_rank": r["high_hi_rank"],
        }
        for r in detected
    ]


def _rank_correlation(order_a: List[dict], order_b: List[dict]) -> dict:
    map_a = {row["channel"]: idx for idx, row in enumerate(order_a)}
    map_b = {row["channel"]: idx for idx, row in enumerate(order_b)}
    common = [ch for ch in map_a if ch in map_b]
    if len(common) < 2:
        return {
            "n_common_detected": len(common),
            "common_detected_channels": sorted(common),
            "spearman_rho": None,
        }
    ra = np.asarray([map_a[ch] for ch in common], dtype=float)
    rb = np.asarray([map_b[ch] for ch in common], dtype=float)
    rho = None
    if np.std(ra) > 0.0 and np.std(rb) > 0.0:
        rho = float(np.corrcoef(ra, rb)[0, 1])
    return {
        "n_common_detected": len(common),
        "common_detected_channels": sorted(common),
        "spearman_rho": rho,
    }


def _load_selected_sentinel_seizures(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    selected = []
    for sentinel in d["sentinels"]:
        selected.append(
            {
                "key": sentinel["key"],
                "subject": sentinel["subject"],
                "rationale": sentinel.get("rationale", ""),
                "seizure_indices": [int(sz["seizure_idx"]) for sz in sentinel["seizures"]],
            }
        )
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step2-summary", type=Path, default=STEP2_SUMMARY_PATH)
    parser.add_argument("--start-floor-sec", type=float, default=-120.0)
    parser.add_argument("--end-sec", type=float, default=30.0)
    parser.add_argument("--bias", type=float, default=0.5)
    parser.add_argument("--threshold-margin", type=float, default=1.0)
    parser.add_argument("--pre-sec", type=float, default=300.0)
    parser.add_argument("--post-sec", type=float, default=30.0)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    focus_rel = _load_focus_rel()
    selected = _load_selected_sentinel_seizures(args.step2_summary)

    rows: List[dict] = []
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "step2_summary_path": str(args.step2_summary.relative_to(_PROJECT_ROOT)),
        "settings": {
            "start_floor_sec": float(args.start_floor_sec),
            "end_sec": float(args.end_sec),
            "bias": float(args.bias),
            "threshold_margin": float(args.threshold_margin),
            "pre_sec": float(args.pre_sec),
            "post_sec": float(args.post_sec),
        },
        "sentinels": [],
    }

    bands = (GAMMA_ER_BANDS, BROAD_ER_BANDS)
    for sentinel in selected:
        subj = sentinel["subject"]
        focal_set = set(_focal_channels(subj, focus_rel))
        high_hi_channels, clusters = _load_lagpat(subj)
        display_cluster = _pick_display_cluster(clusters)
        high_hi_upper = {ch.upper() for ch in high_hi_channels}
        rank_by_channel = {
            ch.upper(): rank for ch, rank in display_cluster["rank_by_channel"].items()
        }

        sentinel_log = {
            "key": sentinel["key"],
            "subject": subj,
            "rationale": sentinel["rationale"],
            "display_cluster_id": int(display_cluster["cluster_id"]),
            "seizures": [],
        }
        print(f"\n=== {sentinel['key']}  {subj}  seizures={sentinel['seizure_indices']} ===")

        for seizure_idx in sentinel["seizure_indices"]:
            window = extract_seizure_window(
                subj,
                seizure_idx,
                pre_sec=args.pre_sec,
                post_sec=args.post_sec,
                results_root=_PROJECT_ROOT / "results",
                reference="car",
            )
            eeg_onset_rel_sec = None
            if window.eeg_onset_epoch is not None and window.clin_onset_epoch is not None:
                eeg_onset_rel_sec = float(window.eeg_onset_epoch - window.clin_onset_epoch)

            seizure_log = {
                "seizure_idx": int(seizure_idx),
                "seizure_id": window.seizure_id,
                "block_stem": window.block_stem,
                "eeg_onset_rel_sec": _float_or_none(eeg_onset_rel_sec),
                "bands": {},
            }
            print(f"  seizure {seizure_idx}: block={window.block_stem} seizure_id={window.seizure_id}")

            per_band_orders: Dict[str, List[dict]] = {}
            for band in bands:
                hop_sec = 0.1
                win_sec = 1.0
                er = compute_er(
                    window.signal,
                    fs=window.fs,
                    fast_band=band["fast"],
                    slow_band=band["slow"],
                    win_sec=win_sec,
                    hop_sec=hop_sec,
                )
                bl_win = resolve_baseline_window(
                    er.shape[1],
                    hop_sec=hop_sec,
                    pre_sec=window.pre_sec,
                    buffer_sec=60.0,
                    eeg_onset_rel_sec=eeg_onset_rel_sec,
                )
                if not bl_win.valid:
                    seizure_log["bands"][band["key"]] = {
                        "skipped": True,
                        "skip_reason": "baseline_invalid",
                        "baseline_end_rel_sec": _float_or_none(bl_win.end_sec),
                    }
                    continue

                z = baseline_zscore_er(
                    er,
                    baseline_idx_window=(bl_win.start_idx, bl_win.end_idx),
                    hop_sec=hop_sec,
                )
                t_axis_er = (np.arange(er.shape[1]) * hop_sec + win_sec / 2.0) - window.pre_sec
                det_win = resolve_detection_window(
                    er.shape[1],
                    hop_sec=hop_sec,
                    pre_sec=window.pre_sec,
                    baseline_end_sec=bl_win.end_sec,
                    start_floor_sec=args.start_floor_sec,
                    end_sec=args.end_sec,
                )
                valid_mask = ~np.isnan(z).any(axis=1)

                band_rows: List[dict] = []
                for ch_idx, ch_name in enumerate(window.ch_names):
                    role = _channel_role(
                        ch_name,
                        focal_upper={c.upper() for c in focal_set},
                        high_hi_upper=high_hi_upper,
                    )
                    preview = None
                    preview_threshold = None
                    if valid_mask[ch_idx] and det_win.valid:
                        preview_threshold = preview_threshold_from_baseline(
                            z[ch_idx],
                            baseline_idx_window=(bl_win.start_idx, bl_win.end_idx),
                            bias=args.bias,
                            threshold_margin=args.threshold_margin,
                        )
                        preview = detect_er_onset_preview(
                            z[ch_idx],
                            t_axis_er,
                            detection_idx_window=(det_win.start_idx, det_win.end_idx),
                            bias=args.bias,
                            threshold=preview_threshold,
                        )
                    row = {
                        "subject": subj,
                        "seizure_idx": int(seizure_idx),
                        "seizure_id": window.seizure_id,
                        "block_stem": window.block_stem,
                        "band": band["key"],
                        "channel": ch_name,
                        "channel_valid": bool(valid_mask[ch_idx]),
                        "role": role,
                        "is_ictal": role in {"ictal", "high_hi_ictal"},
                        "is_high_hi": role in {"high_hi_index", "high_hi_ictal"},
                        "high_hi_rank": rank_by_channel.get(ch_name.upper()),
                        "eeg_onset_rel_sec": _float_or_none(eeg_onset_rel_sec),
                        "baseline_end_rel_sec": _float_or_none(bl_win.end_sec),
                        "detection_start_rel_sec": _float_or_none(det_win.start_sec),
                        "detection_end_rel_sec": _float_or_none(det_win.end_sec),
                        "onset_detected": bool(preview.detected) if preview is not None else False,
                        "t_er_onset_sec": _float_or_none(preview.onset_sec) if preview is not None else None,
                        "t_er_onset_preclinical": (
                            bool(preview.detected and preview.onset_sec is not None and preview.onset_sec < 0.0)
                            if preview is not None
                            else False
                        ),
                        "peak_cusum_stat": _float_or_none(preview.peak_stat) if preview is not None else None,
                        "preview_threshold": _float_or_none(preview_threshold),
                        "peak_post30_zER": _post_window_peak(z[ch_idx], t_axis_er, start_sec=0.0, end_sec=args.end_sec)
                        if valid_mask[ch_idx]
                        else None,
                    }
                    rows.append(row)
                    band_rows.append(row)

                detected_order = _order_list(band_rows, preclinical_only=False)
                preclinical_order = _order_list(band_rows, preclinical_only=True)
                per_band_orders[band["key"]] = detected_order
                seizure_log["bands"][band["key"]] = {
                    "band": band["key"],
                    "baseline_end_rel_sec": _float_or_none(bl_win.end_sec),
                    "detection_start_rel_sec": _float_or_none(det_win.start_sec),
                    "detection_end_rel_sec": _float_or_none(det_win.end_sec),
                    "n_channels_total": int(z.shape[0]),
                    "n_channels_valid": int(valid_mask.sum()),
                    "n_detected": len(detected_order),
                    "n_detected_preclinical": len(preclinical_order),
                    "order_detected": detected_order,
                    "order_detected_preclinical": preclinical_order,
                }
                print(
                    f"    {band['key']}: detected={len(detected_order)} "
                    f"preclinical={len(preclinical_order)} "
                    f"window=[{det_win.start_sec:.1f},{det_win.end_sec:.1f}]s"
                )

            if all(band_key in seizure_log["bands"] and not seizure_log["bands"][band_key].get("skipped")
                   for band_key in ("gamma_ER", "broad_ER")):
                seizure_log["gamma_vs_broad"] = _rank_correlation(
                    seizure_log["bands"]["gamma_ER"]["order_detected_preclinical"],
                    seizure_log["bands"]["broad_ER"]["order_detected_preclinical"],
                )

            sentinel_log["seizures"].append(seizure_log)
        summary["sentinels"].append(sentinel_log)

    csv_path = OUT_DIR / "sentinel_t_er_onset_preview.csv"
    fieldnames = [
        "subject",
        "seizure_idx",
        "seizure_id",
        "block_stem",
        "band",
        "channel",
        "channel_valid",
        "role",
        "is_ictal",
        "is_high_hi",
        "high_hi_rank",
        "eeg_onset_rel_sec",
        "baseline_end_rel_sec",
        "detection_start_rel_sec",
        "detection_end_rel_sec",
        "onset_detected",
        "t_er_onset_sec",
        "t_er_onset_preclinical",
        "peak_cusum_stat",
        "preview_threshold",
        "peak_post30_zER",
    ]
    rows_sorted = sorted(
        rows,
        key=lambda r: (
            r["subject"],
            r["seizure_idx"],
            r["band"],
            not r["onset_detected"],
            float("inf") if r["t_er_onset_sec"] is None else r["t_er_onset_sec"],
            r["channel"],
        ),
    )
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_sorted)

    json_path = OUT_DIR / "sentinel_t_er_onset_preview_summary.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print(f"\nCSV -> {csv_path}")
    print(f"JSON -> {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
