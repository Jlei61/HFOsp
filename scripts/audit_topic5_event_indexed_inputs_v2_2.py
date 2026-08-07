#!/usr/bin/env python3
"""Audit the event-indexed time/block contract before any v2.2 model exists."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
import sys
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.interictal_propagation import load_subject_propagation_events  # noqa: E402
from src.topic5_interictal_operator import (  # noqa: E402
    encode_recruitment_matrix,
)


DEFAULT_CONFIG = ROOT / "config/topic5_event_indexed_evolving_rank_field_v2_2.yaml"
DEFAULT_OUTPUT = (
    ROOT
    / "results/topic5_event_indexed_evolving_rank_field/development/input_audit"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def raw_subject_dir(subject: str) -> Path:
    dataset, short = subject.split("_", 1)
    if dataset == "epilepsiae":
        return (
            Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns")
            / short
            / "all_recs"
        )
    if dataset == "yuquan":
        return Path("/mnt/yuquan_data/yuquan_24h_edf") / short
    raise ValueError(f"unknown dataset: {dataset}")


def _time_of_day_counts(times: np.ndarray, timezone: str) -> tuple[int, int, int]:
    zone = ZoneInfo(timezone)
    day = 0
    night = 0
    dates: set[str] = set()
    for value in np.asarray(times, float):
        stamp = datetime.fromtimestamp(float(value), zone)
        dates.add(stamp.date().isoformat())
        if 8 <= stamp.hour < 20:
            day += 1
        else:
            night += 1
    return day, night, len(dates)


def audit_subject(
    subject: str,
    dataset_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    npz_path = dataset_root / "per_subject" / f"{subject}.npz"
    metadata_path = npz_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(npz_path, allow_pickle=False) as frozen:
        required = {
            "event_local_rank",
            "event_group_ids",
            "event_group_count",
            "event_participation",
            "event_lag_raw",
            "event_abs_time",
            "event_source_index",
            "event_split",
            "contact_names",
            "selected_block_ids",
        }
        missing = sorted(required - set(frozen.files))
        if missing:
            raise RuntimeError(f"{subject}: missing frozen fields {missing}")
        local_rank = np.asarray(frozen["event_local_rank"], np.float32)
        group_ids = np.asarray(frozen["event_group_ids"], np.int16)
        group_count = np.asarray(frozen["event_group_count"], np.int16)
        participation = np.asarray(frozen["event_participation"], bool)
        lag_raw = np.asarray(frozen["event_lag_raw"], np.float32)
        event_time = np.asarray(frozen["event_abs_time"], np.float64)
        source_index = np.asarray(frozen["event_source_index"], np.int64)
        event_split = np.asarray(frozen["event_split"], np.uint8)
        contact_names = [str(value) for value in frozen["contact_names"]]
        selected_blocks = np.asarray(frozen["selected_block_ids"], np.int32)

    raw = load_subject_propagation_events(raw_subject_dir(subject))
    raw_names = [str(value) for value in raw["channel_names"]]
    raw_count = int(len(raw["event_abs_times"]))
    source_index_valid = bool(
        source_index.shape == event_time.shape
        and np.all(source_index >= 0)
        and np.all(source_index < raw_count)
        and len(np.unique(source_index)) == len(source_index)
    )
    if not source_index_valid:
        raise RuntimeError(f"{subject}: source indices are invalid")
    raw_participation = np.asarray(raw["bools"], bool)[:, source_index].T
    raw_lag = np.asarray(raw["lag_raw"], float)[:, source_index].T
    raw_rank = np.asarray(raw["ranks"], float)[:, source_index]
    raw_time = np.asarray(raw["event_abs_times"], float)[source_index]
    source_block = np.asarray(raw["block_ids"], np.int32)[source_index]
    tie_tolerance = float(metadata.get("tie_tolerance_seconds_primary", 0.0))
    rebuilt_local, rebuilt_groups, rebuilt_counts = encode_recruitment_matrix(
        raw_rank,
        np.asarray(raw["bools"], bool)[:, source_index],
        np.asarray(raw["lag_raw"], float)[:, source_index],
        tie_tolerance_seconds=tie_tolerance,
    )

    time_difference = np.abs(event_time - raw_time)
    finite_part = participation & np.isfinite(lag_raw) & np.isfinite(raw_lag)
    raw_lag_float32 = raw_lag.astype(np.float32)
    mapping_checks = {
        "contact_order_exact": contact_names == raw_names,
        "source_index_valid_unique": source_index_valid,
        "absolute_time_exact": bool(np.max(time_difference, initial=0.0) <= 1e-9),
        "participation_exact": bool(np.array_equal(participation, raw_participation)),
        "local_rank_equal_nanaware": bool(
            np.allclose(local_rank, rebuilt_local, equal_nan=True, atol=1e-7)
        ),
        "group_ids_exact": bool(np.array_equal(group_ids, rebuilt_groups)),
        "group_count_exact": bool(np.array_equal(group_count, rebuilt_counts)),
        "lag_raw_float32_roundtrip": bool(
            np.array_equal(lag_raw[finite_part], raw_lag_float32[finite_part])
        ),
        "selected_blocks_exact_cover": bool(
            set(np.unique(source_block)) == set(map(int, selected_blocks))
        ),
        "chronological": bool(np.all(np.diff(event_time) >= 0.0)),
        "split_is_single_train_to_heldout_cut": bool(
            set(np.unique(event_split)) == {0, 1}
            and np.sum(np.diff(event_split.astype(int)) != 0) == 1
            and event_split[0] == 0
            and event_split[-1] == 1
        ),
        "metadata_hash_matches": (
            str(metadata.get("dataset_npz_sha256", "")) == sha256(npz_path)
        ),
    }
    gaps = np.diff(event_time)
    same_block = np.diff(source_block) == 0
    valid_iei = gaps[same_block & (gaps > 0)]
    cross_block = gaps[~same_block]
    dataset = subject.split("_", 1)[0]
    timezone = "Europe/Berlin" if dataset == "epilepsiae" else "Asia/Shanghai"
    day, night, recording_days = _time_of_day_counts(event_time, timezone)
    record_names = np.asarray(raw["record_names"], dtype=str)
    event_record_name = record_names[source_block]
    derived_path = output_root / "per_subject" / f"{subject}.npz"
    derived_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        derived_path,
        event_source_block_id=source_block,
        event_source_record_name=event_record_name,
        source_dataset_npz_sha256=np.asarray(sha256(npz_path)),
        source_raw_manifest_sha256=np.asarray(
            str(metadata.get("source_file_manifest_sha256", ""))
        ),
    )
    status = "PASS" if all(mapping_checks.values()) else "FAIL"
    row = {
        "contract": "topic5_event_indexed_evolving_rank_field_v2_2_input_audit",
        "status": status,
        "dataset": dataset,
        "subject": subject,
        "n_events": len(event_time),
        "n_contacts": len(contact_names),
        "n_source_blocks": len(np.unique(source_block)),
        "n_train80": int(np.sum(event_split == 0)),
        "n_old_heldout20": int(np.sum(event_split == 1)),
        "absolute_time_start": float(event_time[0]),
        "absolute_time_end": float(event_time[-1]),
        "recording_days_covered": recording_days,
        "day_events": day,
        "night_events": night,
        "within_block_iei_seconds": {
            "n": len(valid_iei),
            "q10": float(np.quantile(valid_iei, 0.1)) if len(valid_iei) else None,
            "q50": float(np.quantile(valid_iei, 0.5)) if len(valid_iei) else None,
            "q90": float(np.quantile(valid_iei, 0.9)) if len(valid_iei) else None,
            "q99": float(np.quantile(valid_iei, 0.99)) if len(valid_iei) else None,
        },
        "cross_block_gap_seconds": {
            "n": len(cross_block),
            "minimum": float(np.min(cross_block)) if len(cross_block) else None,
            "maximum": float(np.max(cross_block)) if len(cross_block) else None,
        },
        "ties": {
            "tie_tolerance_seconds_primary": tie_tolerance,
            "fraction_events_with_tied_group": float(
                np.mean(group_count < participation.sum(axis=1))
            ),
            "near_tie_audit": metadata.get("tie_audit", {}),
        },
        "timing_semantics": {
            "event_abs_time": "packedTimes window start plus lagPat start_t",
            "within_event_lag": "spectrogram centroid time",
            "precise_contact_peak_time_available": False,
            "iei_primary_model_input": False,
            "cross_source_block_iei_biological": False,
        },
        "mapping_checks": mapping_checks,
        "derived_block_mapping_path": str(derived_path),
        "derived_block_mapping_sha256": sha256(derived_path),
        "source_dataset_npz": str(npz_path),
        "source_dataset_npz_sha256": sha256(npz_path),
        "old_heldout20_entered_into_analysis": False,
        "snn_inputs_read": False,
        "forbidden_labels_read": False,
    }
    atomic_json(output_root / "per_subject" / f"{subject}.json", row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--cohort",
        choices=("pilot", "all"),
        default="pilot",
        help="all is inventory only and does not enter a scientific gate",
    )
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset_root = ROOT / config["data"]["dataset_dir"]
    manifest_path = dataset_root / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    subjects = (
        list(map(str, config["pilot"]["subjects"]))
        if args.cohort == "pilot"
        else list(map(str, manifest["cohort_subjects"]))
    )
    rows = []
    for subject in subjects:
        row = audit_subject(subject, dataset_root, output)
        rows.append(row)
        print(subject, row["status"])
    frame = pd.DataFrame(
        [
            {
                "dataset": row["dataset"],
                "subject": row["subject"],
                "status": row["status"],
                "n_events": row["n_events"],
                "n_contacts": row["n_contacts"],
                "n_source_blocks": row["n_source_blocks"],
                "n_train80": row["n_train80"],
                "n_old_heldout20": row["n_old_heldout20"],
                "recording_days_covered": row["recording_days_covered"],
                "within_block_iei_q50_seconds": row["within_block_iei_seconds"]["q50"],
                "within_block_iei_q99_seconds": row["within_block_iei_seconds"]["q99"],
                "mapping_checks_all": all(row["mapping_checks"].values()),
            }
            for row in rows
        ]
    )
    suffix = "pilot" if args.cohort == "pilot" else "all_inventory"
    frame.to_csv(output / f"event_indexed_input_audit_{suffix}.csv", index=False)
    payload = {
        "contract": "topic5_event_indexed_evolving_rank_field_v2_2_input_audit",
        "status": "COMPLETE" if all(row["status"] == "PASS" for row in rows) else "FAIL",
        "cohort_role": (
            "six_patient_development_pilot"
            if args.cohort == "pilot"
            else "full_cohort_field_inventory_only_not_scientific_gate"
        ),
        "n_subjects": len(rows),
        "n_pass": sum(row["status"] == "PASS" for row in rows),
        "patients": rows,
        "dataset_manifest_path": str(manifest_path),
        "dataset_manifest_sha256": sha256(manifest_path),
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__)),
        "old_heldout20_entered_into_analysis": False,
        "snn_inputs_read": False,
        "forbidden_labels_read": False,
    }
    atomic_json(output / f"EVENT_INDEXED_INPUT_AUDIT_{suffix.upper()}.json", payload)
    print(json.dumps({"status": payload["status"], "n_pass": payload["n_pass"]}))


if __name__ == "__main__":
    main()
