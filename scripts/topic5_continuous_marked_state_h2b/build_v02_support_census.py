#!/usr/bin/env python3
"""Build the patient/lead H2b v0.2 support and runtime-availability census."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np


REPO = Path(__file__).resolve().parents[2]
SOURCE_REPO = Path("/home/honglab/leijiaxin/HFOsp")
RESULT_ROOT = REPO / (
    "results/epi_prssm/continuous_marked_state/h2b_cross_task/v0_2"
)
R1_ROOT = SOURCE_REPO / "results/epi_prssm/continuous_marked_state/r1"
LEADS = (5, 15, 30, 60, 120)
PRIMARY_LEAD = 30


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def support_tier(n: int) -> str:
    if n >= 10:
        return "primary_chronological"
    if n >= 5:
        return "sensitivity_loso"
    if n >= 2:
        return "descriptive_case_series"
    return "not_estimable"


def seizure_rows(subject: str) -> tuple[list[dict], Path, str]:
    dataset, short = subject.split("_", 1)
    if dataset == "epilepsiae":
        path = SOURCE_REPO / "results/epilepsiae_seizure_inventory.csv"
        rows = [row for row in csv.DictReader(path.open()) if str(row["subject"]) == short]
        output = []
        for row in rows:
            onset_key = "clin_onset_epoch" if finite(row.get("clin_onset_epoch")) else "eeg_onset_epoch"
            if not finite(row.get(onset_key)):
                continue
            offset_key = "clin_offset_epoch" if onset_key == "clin_onset_epoch" else "eeg_offset_epoch"
            output.append({
                "subject": subject,
                "seizure_id": str(row["seizure_id"]),
                "recording_code": str(row["recording_id"]),
                "onset_epoch": float(row[onset_key]),
                "offset_epoch": float(row[offset_key]) if finite(row.get(offset_key)) else float(row[onset_key]),
                "onset_kind": "clinical" if onset_key.startswith("clin") else "eeg",
                "classification": row.get("classification"),
                "pattern": row.get("pattern"),
                "match_route": "canonical_frozen_inventory",
                "onset_difference_seconds": 0.0,
                "matched": True,
                "ambiguous": False,
            })
        return output, path, "Epilepsiae frozen SQL-derived seizure inventory"
    path = SOURCE_REPO / "results/dataset_inventory/yuquan_seizure_inventory.csv"
    rows = [row for row in csv.DictReader(path.open()) if str(row["subject"]) == short]
    output = []
    for row in rows:
        if not finite(row.get("eeg_onset_epoch")):
            continue
        output.append({
            "subject": subject,
            "seizure_id": str(row["seizure_id"]),
            "recording_code": str(row["record"]),
            "onset_epoch": float(row["eeg_onset_epoch"]),
            "offset_epoch": float(row["eeg_offset_epoch"]) if finite(row.get("eeg_offset_epoch")) else float(row["eeg_onset_epoch"]),
            "onset_kind": "eeg",
            "classification": None,
            "pattern": None,
            "match_route": "canonical_record_code_inventory",
            "onset_difference_seconds": 0.0,
            "matched": True,
            "ambiguous": False,
        })
    return output, path, "Yuquan frozen recording-code seizure inventory"


def raw_cache_dir(subject: str) -> Path:
    dataset = subject.split("_", 1)[0]
    base = {
        "epilepsiae": Path("/mnt/yuquan_data/hfosp_cache/raw_seeg_state_r0_1"),
        "yuquan": Path("/mnt/epilepsia_data/hfosp_cache/raw_seeg_state_r0_1"),
    }[dataset]
    return base / subject


def raw_cache_status(subject: str) -> tuple[bool, list[str], Path]:
    root = raw_cache_dir(subject)
    required = [
        root / "raw_256hz.zarr/zarr.json",
        root / "artifact_mask.zarr/zarr.json",
        root / "train_stats.json",
        root / "window_index_refined.parquet",
        root / "cache_index.parquet",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    return not missing, missing, root


def upstream_design(subject: str) -> tuple[Path | None, Path | None]:
    candidates = [
        R1_ROOT / "r1_7b_cohort_extension/upstream_r1_2",
        R1_ROOT / "r1_7a/upstream_r1_2",
    ]
    for root in candidates:
        design = root / "cache" / subject / "full_design.npz"
        baseline = root / "baselines" / subject / "seed_0/models.pt"
        if design.is_file() and baseline.is_file():
            return design, baseline
    return None, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=RESULT_ROOT / "manifests/r1_7_checkpoint_inventory.json")
    parser.add_argument("--output-root", type=Path, default=RESULT_ROOT)
    args = parser.parse_args()
    inventory = read_json(args.inventory)
    if inventory.get("status") != "COMPLETE":
        raise ValueError("checkpoint inventory is not COMPLETE")
    by_subject: dict[str, list[dict]] = {}
    for entry in inventory["entries"]:
        by_subject.setdefault(str(entry["subject"]), []).append(entry)

    from src.topic5_continuous_marked_state_r1.coverage import CoverageTable

    patient_rows: list[dict] = []
    support_rows: list[dict] = []
    crosswalk_rows: list[dict] = []
    for subject in inventory["subjects"]:
        cells = by_subject[subject]
        checkpoints = [row for row in cells if row["checkpoint_available"]]
        seizures, seizure_path, seizure_truth = seizure_rows(subject)
        crosswalk_rows.extend(seizures)
        coverage_path = R1_ROOT / "r1_2/coverage" / f"{subject}.npz"
        coverage_available = coverage_path.is_file()
        design, baseline = upstream_design(subject)
        raw_available, raw_missing, raw_root = raw_cache_status(subject)
        primary_complete = 0
        if coverage_available:
            coverage = CoverageTable.load(coverage_path)
            for lead in LEADS:
                complete = 0
                development = 0
                per_seizure = []
                for seizure in seizures:
                    onset = float(seizure["onset_epoch"])
                    in_development = onset < float(coverage.dev_end_epoch)
                    if in_development:
                        development += 1
                    cutoff = onset - lead * 60.0
                    segment = np.flatnonzero(
                        (coverage.start <= cutoff) & (onset <= coverage.stop)
                    ) if in_development else np.empty(0, dtype=int)
                    covered = len(segment) == 1
                    complete += int(covered)
                    per_seizure.append({
                        "subject": subject,
                        "seizure_id": seizure["seizure_id"],
                        "lead_minutes": lead,
                        "onset_epoch": onset,
                        "cutoff_epoch": cutoff,
                        "in_development_partition": in_development,
                        "complete_recorded_lead_window": covered,
                        "coverage_segment": int(segment[0]) if covered else None,
                    })
                if lead == PRIMARY_LEAD:
                    primary_complete = complete
                support_rows.append({
                    "subject": subject,
                    "lead_minutes": lead,
                    "primary_lead": lead == PRIMARY_LEAD,
                    "n_seizures_total": len(seizures),
                    "n_seizures_development": development,
                    "n_complete_recorded_lead_window": complete,
                    "n_checkpoint_available_seeds": len(checkpoints),
                    "raw_inference_cache_available": raw_available,
                    "upstream_design_available": design is not None,
                    "final_eligible_pending_raw_reader": bool(
                        checkpoints and complete and coverage_available
                    ),
                    "final_n_eligible_seizures": None,
                    "provisional_support_tier_from_coverage_only": support_tier(complete),
                    "support_tier": "PENDING_RAW_INFERENCE_CENSUS" if complete and checkpoints else "not_estimable",
                })
        patient_rows.append({
            "subject": subject,
            "dataset": subject.split("_", 1)[0],
            "n_r1_7_cells": len(cells),
            "n_checkpoint_available_seeds": len(checkpoints),
            "n_stable_checkpoint_seeds": sum(bool(row["stable_checkpoint"]) for row in cells),
            "h1_stable_subject": any(bool(row["h1_stable_subject"]) for row in cells),
            "h1_is_stratification_not_h2b_gate": True,
            "n_seizures_in_frozen_inventory": len(seizures),
            "primary_complete_coverage_seizures": primary_complete,
            "coverage_path": str(coverage_path),
            "coverage_available": coverage_available,
            "upstream_design_path": str(design) if design else None,
            "upstream_baseline_path": str(baseline) if baseline else None,
            "upstream_design_available": design is not None,
            "raw_cache_root": str(raw_root),
            "raw_inference_cache_available": raw_available,
            "raw_missing_count": len(raw_missing),
            "raw_missing_paths": "|".join(raw_missing),
            "seizure_inventory_path": str(seizure_path),
            "seizure_inventory_sha256": sha256_file(seizure_path),
            "seizure_metadata_truth": seizure_truth,
            "runnable_now": bool(checkpoints and seizures and coverage_available and design and raw_available),
            "exclusion_or_deferred_reason": (
                "no_checkpoint" if not checkpoints else
                "no_frozen_seizures" if not seizures else
                "missing_coverage" if not coverage_available else
                "missing_upstream_design_sync" if design is None else
                "raw_cache_mount_unavailable" if not raw_available else
                "ready"
            ),
        })

    output_root = args.output_root.resolve()
    manifest_root = output_root / "manifests"
    atomic_csv(manifest_root / "patient_support_census.csv", patient_rows)
    atomic_csv(manifest_root / "support_by_lead_provisional.csv", support_rows)
    atomic_csv(manifest_root / "seizure_crosswalk.csv", crosswalk_rows)
    payload = {
        "status": "COMPLETE",
        "revision": "h2b_v0_2_support_census_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_subjects": len(patient_rows),
        "n_checkpoint_available_subjects": sum(row["n_checkpoint_available_seeds"] > 0 for row in patient_rows),
        "n_subjects_with_frozen_seizures": sum(row["n_seizures_in_frozen_inventory"] > 0 for row in patient_rows),
        "n_subjects_with_primary_complete_coverage": sum(row["primary_complete_coverage_seizures"] > 0 for row in patient_rows),
        "n_subjects_with_upstream_design_synced": sum(row["upstream_design_available"] for row in patient_rows),
        "n_subjects_with_raw_inference_cache_mounted": sum(row["raw_inference_cache_available"] for row in patient_rows),
        "n_subjects_runnable_now": sum(row["runnable_now"] for row in patient_rows),
        "raw_mounts_present": {
            "/mnt/yuquan_data": any(Path("/mnt/yuquan_data").iterdir()),
            "/mnt/epilepsia_data": any(Path("/mnt/epilepsia_data").iterdir()),
        },
        "formal_test_partition_opened": False,
        "sealed_opened": False,
        "h3_or_t2_run": False,
        "patient_rows": patient_rows,
    }
    atomic_json(manifest_root / "support_census.json", payload)
    print(json.dumps({key: payload[key] for key in (
        "status", "n_subjects", "n_checkpoint_available_subjects",
        "n_subjects_with_frozen_seizures", "n_subjects_with_primary_complete_coverage",
        "n_subjects_with_upstream_design_synced",
        "n_subjects_with_raw_inference_cache_mounted", "n_subjects_runnable_now",
        "raw_mounts_present",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
