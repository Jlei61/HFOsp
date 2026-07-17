#!/usr/bin/env python3
"""Audit type-matched ``T_spectral`` times without reclassifying seizures.

The event labels are read verbatim from the committed early-spectral table.
Existing accepted cache events stay in the audit even when their label is
``other`` or unavailable; those events retain their current time.  The four
Yuquan events that already have one of the three formal labels but were absent
from the old broadband-only aligned cache are also audited, without inventing
labels for the remaining Yuquan seizures.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_tspectral_type_refinement import (  # noqa: E402
    BANDS,
    FROZEN_TYPES,
    TypeRefinementConfig,
    bootstrap_frozen_type_onset,
    refine_frozen_type_onset,
)


SOURCE_CACHE = ROOT / "results/topic5_ictal_recruitment/v2_band_scan/cache"
EPI_ALIGNED = (
    ROOT
    / "results/topic5_ictal_recruitment/v2_band_scan/cache_tspectral_v1p2_common_1_80hz"
)
YUQ_ALIGNED = (
    ROOT
    / "results/topic5_ictal_recruitment/v2_band_scan/cache_tspectral_v1p2_yuquan_common_1_80hz"
)
PHENOTYPE_ROOT = (
    ROOT
    / "results/topic5_ictal_recruitment/peri_onset_energy_timing/early_spectral_phenotype"
)
PHENOTYPE_TABLE = PHENOTYPE_ROOT / "per_seizure_spectral_overlap_state.csv"
EPI_EEG_OFFSETS = (
    ROOT
    / "results/topic5_ictal_recruitment/peri_onset_energy_timing/epilepsiae/source_cache/sensitivity_extended_1_250hz/band_seizure_level_onset_energy.csv"
)
DEFAULT_OUT = PHENOTYPE_ROOT / "tspectral_time_refinement_audit.csv"


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _seed(subject: str, seizure_idx: int, base: int) -> int:
    token = f"{subject}:{int(seizure_idx)}"
    return int((zlib.crc32(token.encode("utf-8")) + int(base)) % (2**32 - 1))


def _current_cache_records() -> tuple[dict[tuple[str, int], dict], dict[str, dict]]:
    events: dict[tuple[str, int], dict] = {}
    subjects: dict[str, dict] = {}
    for aligned_root in (EPI_ALIGNED, YUQ_ALIGNED):
        for path in sorted(aligned_root.glob("*.json")):
            if path.name in {
                "cache_alignment_summary.json",
                "cohort_summary.json",
                "contract.json",
                "summary.json",
            }:
                continue
            meta = json.loads(path.read_text(encoding="utf-8"))
            subject = str(meta.get("subject", path.stem))
            if "seizure_idxs" not in meta:
                continue
            subjects[subject] = {
                "aligned_root": aligned_root,
                "analysis_channels": [str(value) for value in meta["analysis_channels"]],
                "meta_path": path,
            }
            for idx, event in (meta.get("seizure") or {}).items():
                events[(subject, int(idx))] = dict(event)
    return events, subjects


def _eeg_offsets() -> dict[tuple[str, int], float]:
    table = pd.read_csv(EPI_EEG_OFFSETS).drop_duplicates(
        ["subject", "seizure_idx"]
    )
    return {
        (str(row.subject), int(row.seizure_idx)): float(
            row.eeg_onset_rel_clinical_sec
        )
        for row in table.itertuples()
    }


def _phenotype_rows() -> dict[tuple[str, int], dict]:
    table = pd.read_csv(PHENOTYPE_TABLE)
    return {
        (str(row["subject"]), int(row["seizure_idx"])): row.to_dict()
        for _, row in table.iterrows()
    }


def _scope(
    current: dict[tuple[str, int], dict],
    phenotype: dict[tuple[str, int], dict],
) -> list[tuple[str, int]]:
    keys = set(current)
    for key, row in phenotype.items():
        if (
            str(row.get("dataset")) == "yuquan"
            and str(row.get("simple_phenotype")) in FROZEN_TYPES
        ):
            keys.add(key)
    return sorted(keys)


def run(
    *,
    out_csv: Path,
    n_boot: int,
    seed: int,
    subjects: set[str] | None = None,
) -> Path:
    current, subject_contracts = _current_cache_records()
    phenotype = _phenotype_rows()
    eeg_offsets = _eeg_offsets()
    by_subject: dict[str, list[int]] = defaultdict(list)
    for subject, idx in _scope(current, phenotype):
        if subjects is not None and subject not in subjects:
            continue
        by_subject[subject].append(int(idx))

    rows: list[dict] = []
    errors: list[dict] = []
    config = TypeRefinementConfig()
    for subject in sorted(by_subject):
        if subject not in subject_contracts:
            errors.append(
                {"subject": subject, "seizure_idx": "", "error": "missing_aligned_subject_contract"}
            )
            continue
        source_path = SOURCE_CACHE / f"{subject}.npz"
        source_meta_path = SOURCE_CACHE / f"{subject}.json"
        if not source_path.exists() or not source_meta_path.exists():
            errors.append(
                {"subject": subject, "seizure_idx": "", "error": "missing_source_cache"}
            )
            continue
        obj = np.load(source_path, allow_pickle=False)
        source_channels = [str(value) for value in obj["channels"]]
        analysis_channels = subject_contracts[subject]["analysis_channels"]
        missing_channels = [name for name in analysis_channels if name not in source_channels]
        if missing_channels:
            obj.close()
            errors.append(
                {
                    "subject": subject,
                    "seizure_idx": "",
                    "error": f"analysis_channels_missing:{missing_channels}",
                }
            )
            continue
        channel_idx = np.asarray(
            [source_channels.index(name) for name in analysis_channels], dtype=int
        )
        for idx in by_subject[subject]:
            key = (subject, idx)
            old = current.get(key)
            label = phenotype.get(key)
            simple = None if label is None else str(label.get("simple_phenotype"))
            base_row = {
                "analysis_version": "topic5_tspectral_frozen_type_refinement_v1",
                "dataset": subject.split("_", 1)[0],
                "subject": subject,
                "seizure_idx": int(idx),
                "seizure_id": "" if label is None else label.get("seizure_id", ""),
                "in_existing_aligned_cache": bool(old is not None),
                "label_version": "" if label is None else label.get("analysis_version", ""),
                "simple_phenotype": "not_classified" if label is None else simple,
                "strict_broadband_5of6": False if label is None else bool(label.get("strict_broadband_5of6")),
                "gamma_band_30_80_support": False if label is None else bool(label.get("gamma_band_30_80_support")),
                "low_frequency_1_13_support": False if label is None else bool(label.get("low_frequency_1_13_support")),
                "old_t_spectral_rel_eeg_sec": "" if old is None else old.get("t_spectral_rel_eeg_sec", ""),
                "refined_candidate_detected": False,
                "refined_candidate_rel_eeg_sec": "",
                "candidate_minus_old_sec": "",
                "quiet_baseline_start_rel_eeg_sec": "",
                "quiet_baseline_end_rel_eeg_sec": "",
                "n_band_post_sustained": "",
                "n_required_bands": "",
                "bootstrap_n": int(n_boot),
                "bootstrap_support_fraction": "",
                "bootstrap_q05_rel_eeg_sec": "",
                "bootstrap_q95_rel_eeg_sec": "",
                "bootstrap_width_sec": "",
                "bootstrap_consistency_1s": "",
                "refinement_status": "",
            }
            if simple not in FROZEN_TYPES:
                base_row["refinement_status"] = (
                    "existing_time_retained_not_classified"
                    if label is None
                    else "existing_time_retained_outside_three_frozen_types"
                )
                rows.append(base_row)
                continue
            try:
                if not all(f"{band}__zt__{idx}" in obj.files for band in BANDS):
                    raise KeyError("missing_one_or_more_1_150hz_bands")
                rel_eeg = np.asarray(obj[f"{BANDS[0]}__relt__{idx}"], dtype=float)
                if subject.startswith("epilepsiae_"):
                    rel_eeg = rel_eeg - float(eeg_offsets[key])
                z = np.stack(
                    [
                        np.asarray(obj[f"{band}__zt__{idx}"], dtype=float)[channel_idx]
                        for band in BANDS
                    ]
                )
                result = refine_frozen_type_onset(
                    z,
                    rel_eeg,
                    simple,
                    float(label["anchor_rel_eeg_sec"]),
                    config=config,
                )
                base_row.update(
                    {
                        "refined_candidate_detected": bool(result.detected),
                        "refined_candidate_rel_eeg_sec": float(result.onset_sec),
                        "candidate_minus_old_sec": (
                            ""
                            if old is None
                            else float(result.onset_sec)
                            - float(old["t_spectral_rel_eeg_sec"])
                        ),
                        "quiet_baseline_start_rel_eeg_sec": float(
                            result.baseline_start_sec
                        ),
                        "quiet_baseline_end_rel_eeg_sec": float(
                            result.baseline_end_sec
                        ),
                        "n_band_post_sustained": int(result.n_band_post_sustained),
                        "n_required_bands": int(result.n_required_bands),
                    }
                )
                if int(n_boot) > 0:
                    bootstrap = bootstrap_frozen_type_onset(
                        z,
                        rel_eeg,
                        simple,
                        float(label["anchor_rel_eeg_sec"]),
                        n_boot=int(n_boot),
                        seed=_seed(subject, idx, seed),
                        config=config,
                    )
                    base_row.update(
                        {
                            "bootstrap_support_fraction": bootstrap[
                                "support_fraction"
                            ],
                            "bootstrap_q05_rel_eeg_sec": bootstrap["q05_sec"],
                            "bootstrap_q95_rel_eeg_sec": bootstrap["q95_sec"],
                            "bootstrap_width_sec": bootstrap["width_sec"],
                            "bootstrap_consistency_1s": bootstrap[
                                "consistency_1s"
                            ],
                        }
                    )
                if result.detected:
                    base_row["refinement_status"] = (
                        "candidate_for_existing_event"
                        if old is not None
                        else "candidate_for_formal_yuquan_extension"
                    )
                else:
                    base_row["refinement_status"] = (
                        "existing_time_retained_no_type_matched_candidate"
                        if old is not None
                        else "not_added_no_type_matched_candidate"
                    )
                rows.append(base_row)
            except Exception as exc:  # noqa: BLE001 - event-level provenance
                base_row["refinement_status"] = (
                    "existing_time_retained_processing_error"
                    if old is not None
                    else "not_added_processing_error"
                )
                rows.append(base_row)
                errors.append(
                    {
                        "subject": subject,
                        "seizure_idx": int(idx),
                        "error": f"{type(exc).__name__}:{exc}",
                    }
                )
        obj.close()
        print(f"[frozen-type T_spectral] {subject}: {len(by_subject[subject])} events", flush=True)

    _write_csv(out_csv, rows)
    is_default_out = out_csv.resolve() == DEFAULT_OUT.resolve()
    error_path = out_csv.with_name(
        "tspectral_time_refinement_errors.json"
        if is_default_out
        else f"{out_csv.stem}_errors.json"
    )
    error_path.write_text(
        json.dumps(errors, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    frame = pd.DataFrame(rows)
    summary = {
        "analysis_version": "topic5_tspectral_frozen_type_refinement_v1",
        "scientific_scope": "time refinement only; frozen labels are never recomputed",
        "label_source": str(PHENOTYPE_TABLE.relative_to(ROOT)),
        "frozen_types": list(FROZEN_TYPES),
        "n_events_audited": int(len(frame)),
        "n_existing_cache_events": int(frame.in_existing_aligned_cache.sum()),
        "n_formal_yuquan_extension_events": int((~frame.in_existing_aligned_cache).sum()),
        "n_refined_candidates_detected": int(frame.refined_candidate_detected.sum()),
        "n_processing_errors": int(len(errors)),
        "simple_phenotype_counts": frame.simple_phenotype.value_counts().to_dict(),
        "refinement_status_counts": frame.refinement_status.value_counts().to_dict(),
        "config": {
            "quiet_pool_sec": list(config.quiet_pool_sec),
            "quiet_window_sec": config.quiet_window_sec,
            "quiet_step_sec": config.quiet_step_sec,
            "early_domain_sec": list(config.early_domain_sec),
            "local_pre_anchor_sec": config.local_pre_anchor_sec,
            "local_post_anchor_sec": config.local_post_anchor_sec,
            "smooth_sec": config.smooth_sec,
            "flank_sec": config.flank_sec,
            "post_sec": config.post_sec,
            "sustain_sec": config.sustain_sec,
            "baseline_quantile": config.baseline_quantile,
            "spatial_quantile": config.spatial_quantile,
        },
        "n_boot": int(n_boot),
        "seed": int(seed),
        "subject_filter": None if subjects is None else sorted(subjects),
    }
    summary_path = out_csv.with_name(
        "tspectral_time_refinement_summary.json"
        if is_default_out
        else f"{out_csv.stem}_summary.json"
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return out_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-boot", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--subjects", nargs="+")
    args = parser.parse_args()
    print(
        run(
            out_csv=args.out.resolve(),
            n_boot=int(args.n_boot),
            seed=int(args.seed),
            subjects=None if not args.subjects else set(args.subjects),
        )
    )


if __name__ == "__main__":
    main()
