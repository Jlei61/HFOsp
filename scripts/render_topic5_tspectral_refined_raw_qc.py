#!/usr/bin/env python3
"""Render accepted per-seizure T_spectral markers with the raw_qc_low_eeg layout."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper_figures.plot_fig3_raw_spectral_context import _alias_index  # noqa: E402
from scripts.render_topic5_onset_energy_raw_qc import (  # noqa: E402
    DISTAL_BASELINE,
    RAW_POST_SEC,
    _plot_one,
)
from scripts.run_topic5_subject_spectral_onset import (  # noqa: E402
    DEFAULT_MANIFEST,
    _build_subject_contexts,
    _manifest_lookup,
)
from scripts.run_topic5_t0_eligibility import ICTAL_REFERENCE  # noqa: E402
from src.ictal_onset_extraction import extract_seizure_window  # noqa: E402
from src.topic5_spectral_onset import SpectralOnsetConfig  # noqa: E402


DEFAULT_TIMING = (
    ROOT
    / "results/topic5_ictal_recruitment/peri_onset_energy_timing/epilepsiae/refinement_v1p2/per_seizure_subject_refined_onset.csv"
)
DEFAULT_OUT = (
    ROOT
    / "results/topic5_ictal_recruitment/peri_onset_energy_timing/epilepsiae/refinement_v1p2/raw_qc_accepted_tspectral"
)


def _truth(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _finite_float(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _accepted_rows(path: Path) -> pd.DataFrame:
    rows = pd.read_csv(path)
    accepted = rows[rows["has_accepted_t_best"].map(_truth)].copy()
    accepted["seizure_idx"] = accepted["seizure_idx"].astype(int)
    return accepted


def _render_subject(
    subject: str,
    accepted: pd.DataFrame,
    manifest: dict[tuple[str, int], dict],
    out_root: Path,
    *,
    force: bool,
    max_seizures: int | None,
) -> tuple[list[dict], list[dict]]:
    config = SpectralOnsetConfig(n_boot=0)
    contexts, band_order, _ = _build_subject_contexts(subject, manifest, config)
    accepted_by_idx = {
        int(row["seizure_idx"]): row for _, row in accepted.iterrows()
    }
    contexts = [
        context
        for context in contexts
        if int(context["event"]["seizure_idx"]) in accepted_by_idx
    ]
    if max_seizures is not None:
        contexts = contexts[: int(max_seizures)]

    output_rows: list[dict] = []
    errors: list[dict] = []
    subject_dir = out_root / "figures" / subject
    for context in contexts:
        event = context["event"]
        diagnostics = context["diagnostics"]
        idx = int(event["seizure_idx"])
        timing = accepted_by_idx[idx]
        t_best_eeg = float(timing["t_spectral_best_rel_eeg_sec"])
        t_best_cache = None
        if "t_spectral_best_rel_cache_zero_sec" in timing.index:
            t_best_cache = _finite_float(
                timing["t_spectral_best_rel_cache_zero_sec"]
            )
        if t_best_cache is None:
            t_best_cache = float(timing["t_spectral_best_rel_clinical_sec"])
        q05 = float(timing["bootstrap_q05_rel_eeg_sec"])
        q95 = float(timing["bootstrap_q95_rel_eeg_sec"])
        eeg_rel = float(event["eeg_rel_clinical_sec"])
        clinical_available = bool(event.get("clinical_onset_available", True))
        out_path = (
            subject_dir
            / f"{subject}_seizure_{idx:02d}_accepted_tspectral_raw_qc.png"
        )
        try:
            if force or not out_path.exists():
                x0_eeg = max(float(diagnostics.rel_t[0]), DISTAL_BASELINE[0])
                desired_right_cache = max(RAW_POST_SEC, t_best_cache + 8.0)
                x1_eeg = min(
                    float(diagnostics.rel_t[-1]), desired_right_cache - eeg_rel
                )
                x0_cache = x0_eeg + eeg_rel
                x1_cache = x1_eeg + eeg_rel
                raw_sw = extract_seizure_window(
                    f"{event['dataset']}/{event['sid']}",
                    idx,
                    pre_sec=max(10.0, -x0_cache),
                    post_sec=max(10.0, x1_cache),
                    results_root=ROOT / "results",
                    reference=ICTAL_REFERENCE[event["dataset"]],
                )
                raw_lookup = _alias_index(raw_sw.ch_names)
                absent = [
                    name
                    for name in event["timing_channels"]
                    if name not in raw_lookup
                ]
                if absent:
                    raise ValueError(f"raw timing contacts missing: {absent}")
                raw_idx = [
                    int(raw_lookup[name]) for name in event["timing_channels"]
                ]
                traces = {
                    band: diagnostics.band_trace[band_idx]
                    for band_idx, band in enumerate(band_order)
                }
                marker = SimpleNamespace(onset_sec=t_best_eeg, detected=True)
                marker_status = (
                    "frozen-type refined"
                    if str(timing["timing_status"])
                    == "accepted_frozen_type_refined"
                    else "accepted existing"
                )
                _plot_one(
                    ds_sid=subject,
                    idx=idx,
                    seizure_id=str(event["seizure_id"]),
                    eeg_rel=eeg_rel,
                    timing_channels=list(event["timing_channels"]),
                    rel_eeg=diagnostics.rel_t,
                    traces=traces,
                    recruitment=marker,
                    eeg_window_hit=False,
                    n_band_eeg_hits=0,
                    raw_sw=raw_sw,
                    raw_channel_idx=raw_idx,
                    out_path=out_path,
                    marker_label="accepted T_spectral",
                    marker_status=marker_status,
                    marker_q05_rel_eeg=q05,
                    marker_q95_rel_eeg=q95,
                    marker_color="#1F77B4",
                    show_eeg_hit_context=False,
                    clinical_onset_available=clinical_available,
                )
                del raw_sw
            output_rows.append(
                {
                    "subject": subject,
                    "seizure_idx": idx,
                    "seizure_id": event["seizure_id"],
                    "timing_status": timing["timing_status"],
                    "annotation_mode": event.get("annotation_mode", "eeg_and_clinical"),
                    "cache_zero_reference": event.get(
                        "cache_zero_reference", "clinical_onset"
                    ),
                    "cache_tier": event.get("cache_tier", "primary_existing"),
                    "t_spectral_best_rel_eeg_sec": t_best_eeg,
                    "t_spectral_best_rel_cache_zero_sec": t_best_cache,
                    "t_spectral_best_rel_clinical_sec": (
                        t_best_cache if clinical_available else ""
                    ),
                    "bootstrap_q05_rel_eeg_sec": q05,
                    "bootstrap_q95_rel_eeg_sec": q95,
                    "n_timing_contacts": len(event["timing_channels"]),
                    "figure": str(out_path.relative_to(ROOT)),
                }
            )
            print(f"[accepted-raw-qc] {subject} seizure {idx}", flush=True)
        except Exception as exc:  # noqa: BLE001 - retain event-level provenance
            errors.append(
                {
                    "subject": subject,
                    "seizure_idx": idx,
                    "error": f"{type(exc).__name__}:{exc}",
                }
            )
            print(f"[ERROR] {errors[-1]}", flush=True)

    subject_dir.mkdir(parents=True, exist_ok=True)
    (subject_dir / "README.md").write_text(
        f"# {subject} accepted T_spectral raw QC\n\n"
        f"### {subject}_seizure_*_accepted_tspectral_raw_qc.png\n\n"
        "每张图原样复用 raw_qc_low_eeg 的原始波形、五频带和 consensus 版式，收录已有 accepted 事件及在既有三类频谱标签内仅精修时间的事件。蓝线是 accepted T_spectral，浅蓝区是90%重采样区间。\n\n"
        "**关注点**：检查蓝线是否落在该事件既有频谱类型的早期能量变化边缘，而不是后续更大的发作状态；Yuquan 横轴以 EEG onset 为零，不显示伪造的 clinical marker。\n",
        encoding="utf-8",
    )
    return output_rows, errors


def run(args: argparse.Namespace) -> Path:
    accepted = _accepted_rows(args.timing_csv)
    subjects = list(args.subjects) if args.subjects else sorted(accepted["subject"].unique())
    accepted = accepted[accepted["subject"].isin(subjects)]
    manifest = _manifest_lookup(args.manifest)
    rows: list[dict] = []
    errors: list[dict] = []
    for subject in subjects:
        subject_rows, subject_errors = _render_subject(
            subject,
            accepted[accepted["subject"].eq(subject)],
            manifest,
            args.out_root,
            force=bool(args.force),
            max_seizures=args.max_seizures,
        )
        rows.extend(subject_rows)
        errors.extend(subject_errors)

    index_path = args.out_root / "accepted_tspectral_raw_qc_index.csv"
    _write_csv(index_path, rows)
    (args.out_root / "processing_errors.json").write_text(
        json.dumps(errors, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    summary = {
        "analysis_version": "topic5_tspectral_subject_v1p2_raw_qc",
        "plot_function_reused": "scripts.render_topic5_onset_energy_raw_qc._plot_one",
        "timing_source": str(args.timing_csv),
        "n_accepted_in_scope": int(len(accepted)),
        "n_rendered": int(len(rows)),
        "n_errors": int(len(errors)),
        "n_subjects": int(len(set(row["subject"] for row in rows))),
        "subjects": subjects,
    }
    (args.out_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    figures = args.out_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    (figures / "README.md").write_text(
        "# Accepted T_spectral raw QC figures\n\n"
        "### <dataset>_<subject>/<dataset>_<subject>_seizure_*_accepted_tspectral_raw_qc.png\n\n"
        "逐 seizure 图原样复用 raw_qc_low_eeg 版式：最上方为固定 timing contacts 原始波形，随后是五频带/consensus 热图和独立能量轨迹。图中既包括原 accepted 事件，也包括在既有三类频谱标签内仅精修 T_spectral 的 Yuquan 事件；Epilepsiae 保留 clinical 参考，Yuquan 使用 EEG-only 横轴。\n\n"
        "**关注点**：蓝线及浅蓝90%区间是否对应既有频谱类型的早期能量变化；Yuquan 不显示不存在的 clinical marker。\n",
        encoding="utf-8",
    )
    if errors:
        raise RuntimeError(f"{len(errors)} accepted event(s) failed raw-QC rendering")
    return index_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-csv", type=Path, default=DEFAULT_TIMING)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--max-seizures", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.timing_csv = args.timing_csv.resolve()
    args.manifest = args.manifest.resolve()
    args.out_root = args.out_root.resolve()
    print(run(args))


if __name__ == "__main__":
    main()
