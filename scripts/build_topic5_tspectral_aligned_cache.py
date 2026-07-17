#!/usr/bin/env python3
"""Build a versioned ictal cache aligned to accepted per-seizure T_spectral.

The original v2 band cache is never modified.  This cache contains only seizures
with an accepted timing row and the five common 1--80 Hz bands used to define
T_spectral.  Time arrays keep the canonical ``__relt__`` key pattern but are
re-zeroed to accepted T_spectral rather than clinical onset.  Acceptance may
come from the original subject-recurrence contract or a time-only refinement
inside an already frozen frequency label; this builder never assigns a type.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper_figures.plot_fig3_raw_spectral_context import (  # noqa: E402
    _load_lagpat_channels,
)


SOURCE_CACHE = ROOT / "results/topic5_ictal_recruitment/v2_band_scan/cache"
TIMING_CSV = (
    ROOT
    / "results/topic5_ictal_recruitment/peri_onset_energy_timing/epilepsiae/refinement_v1p2/per_seizure_subject_refined_onset.csv"
)
OUT_ROOT = (
    ROOT
    / "results/topic5_ictal_recruitment/v2_band_scan/cache_tspectral_v1p2_common_1_80hz"
)
BANDS = (
    "delta_HYP_slow",
    "theta_preictal_PAC",
    "alpha_sharp_leq13",
    "beta_LVFA_low",
    "gamma_LVFA",
)


def align_rel_time(rel_clinical: np.ndarray, t_rel_clinical: float) -> np.ndarray:
    """Return the same grid relative to T_spectral instead of clinical onset."""
    return np.asarray(rel_clinical, dtype=float) - float(t_rel_clinical)


def _truth(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _cache_zero_time(row: pd.Series) -> float:
    value = row.get("t_spectral_best_rel_cache_zero_sec", np.nan)
    try:
        out = float(value)
    except (TypeError, ValueError):
        out = float("nan")
    if np.isfinite(out):
        return out
    return float(row["t_spectral_best_rel_clinical_sec"])


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _fallback_paths(subject: str) -> list[Path]:
    base = (
        ROOT
        / "results/topic5_ictal_recruitment/peri_onset_energy_timing/epilepsiae/source_cache"
    )
    return [
        base
        / "primary_common_1_80hz"
        / "per_subject"
        / subject
        / "cache"
        / f"{subject}_common_1_80hz_eeg_relative_missing_bands.npz",
        base
        / "per_subject"
        / subject
        / "cache"
        / f"{subject}_eeg_relative_missing_bands.npz",
    ]


def _fallback_channels(path: Path) -> list[str]:
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    return [str(value) for value in meta["contract"]["channels"]]


def _has_event(obj, idx: int) -> bool:
    return all(f"{band}__zt__{idx}" in obj.files for band in BANDS)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_subject(
    subject: str,
    timing: pd.DataFrame,
    *,
    source_root: Path,
    out_root: Path,
    timing_source: Path,
) -> list[dict]:
    inventory: list[dict] = []
    source_npz = source_root / f"{subject}.npz"
    source_json = source_root / f"{subject}.json"
    if not source_npz.exists() or not source_json.exists():
        return [
            {
                "subject": subject,
                "seizure_idx": "",
                "cache_status": "missing_source_subject_cache",
                "source": "",
                "n_channels": 0,
                "n_bands": 0,
                "cache_zero_reference": "",
                "t_spectral_rel_cache_zero_sec": "",
                "t_spectral_rel_clinical_sec": "",
            }
        ]

    base_obj = np.load(source_npz, allow_pickle=False)
    base_meta = json.loads(source_json.read_text(encoding="utf-8"))
    base_channels = [str(value) for value in base_obj["channels"]]
    lagpat_channels, lagpat_source = _load_lagpat_channels(subject)
    timing_channels = [name for name in lagpat_channels if name in set(base_channels)]
    if not timing_channels:
        base_obj.close()
        raise RuntimeError(f"{subject}: no timing contacts overlap source cache")

    fallback_objects: list[tuple[Path, object, list[str]]] = []
    for path in _fallback_paths(subject):
        if path.exists():
            fallback_objects.append(
                (path, np.load(path, allow_pickle=False), _fallback_channels(path))
            )

    arrays: dict[str, np.ndarray] = {"channels": np.asarray(timing_channels)}
    aligned_idxs: list[int] = []
    per_event: dict[str, dict] = {}
    for _, row in timing.sort_values("seizure_idx").iterrows():
        idx = int(row["seizure_idx"])
        if not _truth(row["has_accepted_t_best"]):
            inventory.append(
                {
                    "subject": subject,
                    "seizure_idx": idx,
                    "cache_status": f"excluded_{row['timing_status']}",
                    "source": "",
                    "n_channels": len(timing_channels),
                    "n_bands": 0,
                    "cache_zero_reference": row.get(
                        "cache_zero_reference", "clinical_onset"
                    ),
                    "t_spectral_rel_cache_zero_sec": "",
                    "t_spectral_rel_clinical_sec": "",
                }
            )
            continue

        source_obj = None
        source_channels: list[str] = []
        source_label = ""
        if _has_event(base_obj, idx):
            source_obj = base_obj
            source_channels = base_channels
            source_label = "v2_band_scan/cache"
        else:
            for path, obj, channels in fallback_objects:
                if _has_event(obj, idx):
                    source_obj = obj
                    source_channels = channels
                    source_label = str(path.relative_to(ROOT))
                    break
        if source_obj is None:
            inventory.append(
                {
                    "subject": subject,
                    "seizure_idx": idx,
                    "cache_status": "accepted_but_source_arrays_missing",
                    "source": "",
                    "n_channels": len(timing_channels),
                    "n_bands": 0,
                    "cache_zero_reference": row.get(
                        "cache_zero_reference", "clinical_onset"
                    ),
                    "t_spectral_rel_cache_zero_sec": _cache_zero_time(row),
                    "t_spectral_rel_clinical_sec": row[
                        "t_spectral_best_rel_clinical_sec"
                    ] if _truth(row.get("clinical_onset_available", True)) else "",
                }
            )
            continue
        if any(name not in source_channels for name in timing_channels):
            inventory.append(
                {
                    "subject": subject,
                    "seizure_idx": idx,
                    "cache_status": "accepted_but_timing_channels_missing",
                    "source": source_label,
                    "n_channels": len(timing_channels),
                    "n_bands": 0,
                    "cache_zero_reference": row.get(
                        "cache_zero_reference", "clinical_onset"
                    ),
                    "t_spectral_rel_cache_zero_sec": _cache_zero_time(row),
                    "t_spectral_rel_clinical_sec": row[
                        "t_spectral_best_rel_clinical_sec"
                    ] if _truth(row.get("clinical_onset_available", True)) else "",
                }
            )
            continue
        source_index = np.asarray(
            [source_channels.index(name) for name in timing_channels], dtype=int
        )
        t_rel_cache_zero = _cache_zero_time(row)
        rel_reference = None
        for band in BANDS:
            z = np.asarray(source_obj[f"{band}__zt__{idx}"], dtype=np.float32)
            rel = np.asarray(source_obj[f"{band}__relt__{idx}"], dtype=float)
            aligned = align_rel_time(rel, t_rel_cache_zero).astype(np.float32)
            if rel_reference is None:
                rel_reference = aligned
            elif not np.allclose(aligned, rel_reference, atol=1e-5):
                raise ValueError(f"{subject} seizure {idx}: band time grids differ")
            arrays[f"{band}__zt__{idx}"] = z[source_index]
            arrays[f"{band}__relt__{idx}"] = aligned
        aligned_idxs.append(idx)
        per_event[str(idx)] = {
            "alignment_status": row["timing_status"],
            "t_spectral_rel_eeg_sec": float(row["t_spectral_best_rel_eeg_sec"]),
            "cache_zero_reference": row.get(
                "cache_zero_reference", "clinical_onset"
            ),
            "t_spectral_rel_cache_zero_sec": t_rel_cache_zero,
            "t_spectral_rel_clinical_sec": (
                t_rel_cache_zero
                if _truth(row.get("clinical_onset_available", True))
                else None
            ),
            "clinical_onset_rel_tspectral_sec": (
                -t_rel_cache_zero
                if _truth(row.get("clinical_onset_available", True))
                else None
            ),
            "eeg_onset_rel_tspectral_sec": -float(
                row["t_spectral_best_rel_eeg_sec"]
            ),
            "bootstrap_q05_rel_eeg_sec": float(row["bootstrap_q05_rel_eeg_sec"]),
            "bootstrap_q95_rel_eeg_sec": float(row["bootstrap_q95_rel_eeg_sec"]),
            "source": source_label,
            "nearest_grid_to_zero_sec": float(rel_reference[np.argmin(np.abs(rel_reference))]),
        }
        inventory.append(
            {
                "subject": subject,
                "seizure_idx": idx,
                    "cache_status": "aligned_accepted_tspectral",
                "source": source_label,
                "n_channels": len(timing_channels),
                "n_bands": len(BANDS),
                "cache_zero_reference": row.get(
                    "cache_zero_reference", "clinical_onset"
                ),
                "t_spectral_rel_cache_zero_sec": t_rel_cache_zero,
                "t_spectral_rel_clinical_sec": (
                    t_rel_cache_zero
                    if _truth(row.get("clinical_onset_available", True))
                    else ""
                ),
            }
        )

    out_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_root / f"{subject}.npz", **arrays)
    meta = {
        "analysis_version": "topic5_tspectral_cache_v1p2",
        "subject": subject,
        "status": "accepted_tspectral_events_only",
        "time_zero": "patient-specific accepted T_spectral_best",
        "source_time_reference": (
            "EEG onset" if base_meta.get("dataset") == "yuquan" else "clinical onset"
        ),
        "source_cache": _display_path(source_npz),
        "timing_source": _display_path(timing_source),
        "lagpat_channel_source": lagpat_source,
        "channels": timing_channels,
        "analysis_channels": timing_channels,
        "analysis_channels_basis": "fixed_lagpat_timing_contacts_intersect_source_cache",
        "seizure_idxs": aligned_idxs,
        "bands": list(BANDS),
        "excluded_event_policy": (
            "only timing rows with has_accepted_t_best=true are included; "
            "frequency labels are attached separately and are never inferred here"
        ),
        "seizure": per_event,
        "source_subject_meta": {
            "dataset": base_meta.get("dataset"),
            "subject": base_meta.get("subject"),
            "fs": base_meta.get("fs"),
            "spec_win_sec": base_meta.get("spec_win_sec"),
            "spec_hop_sec": base_meta.get("spec_hop_sec"),
        },
    }
    (out_root / f"{subject}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    base_obj.close()
    for _, obj, _ in fallback_objects:
        obj.close()
    return inventory


def run(timing_csv: Path, source_root: Path, out_root: Path) -> Path:
    timing = pd.read_csv(timing_csv)
    rows: list[dict] = []
    for subject, use in timing.groupby("subject", sort=True):
        rows.extend(
            build_subject(
                subject,
                use,
                source_root=source_root,
                out_root=out_root,
                timing_source=timing_csv,
            )
        )
    _write_csv(out_root / "cache_alignment_inventory.csv", rows)
    summary = {
        "analysis_version": "topic5_tspectral_cache_v1p2",
        "source_cache": str(source_root),
        "timing_csv": str(timing_csv),
        "n_subjects_in_timing": int(timing["subject"].nunique()),
        "n_events_in_timing": int(len(timing)),
        "cache_status_counts": pd.Series(
            [row["cache_status"] for row in rows]
        ).value_counts().to_dict(),
        "n_subject_cache_pairs_written": int(
            sum(
                (out_root / f"{subject}.npz").exists()
                and (out_root / f"{subject}.json").exists()
                for subject in timing["subject"].unique()
            )
        ),
        "n_subjects_with_aligned_event": int(
            len(
                {
                    row["subject"]
                    for row in rows
                    if row["cache_status"]
                    == "aligned_accepted_tspectral"
                }
            )
        ),
    }
    (out_root / "cache_alignment_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_root / "README.md").write_text(
        "# T_spectral-aligned ictal cache v1.2\n\n"
        "该目录不覆盖原始 `v2_band_scan/cache`。只收录 `has_accepted_t_best=true` 的 seizure；五个 1–80 Hz band 的 `__relt__` 数组已重新置零到 accepted `T_spectral_best`。频谱类型不在本脚本中推断，而由既有三类频率合同另行写入 JSON sidecar。\n\n"
        "未接受的时间行不会写入 aligned arrays；详见 `cache_alignment_inventory.csv`。缺失于原始长窗 cache但已由 onset pipeline 的短窗 fallback 补出的事件，会明确记录 source。\n",
        encoding="utf-8",
    )
    return out_root / "cache_alignment_summary.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-csv", type=Path, default=TIMING_CSV)
    parser.add_argument("--source-cache", type=Path, default=SOURCE_CACHE)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    args = parser.parse_args()
    print(run(args.timing_csv.resolve(), args.source_cache.resolve(), args.out_root.resolve()))


if __name__ == "__main__":
    main()
