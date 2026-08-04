#!/usr/bin/env python3
"""Freeze the real-data chronology and anchor contract for Topic 5 v3.0.

This is an index/provenance audit only.  One position is one complete
interictal event.  It does not fit an observer, an innovation response, or any
within-event next-rank model.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

# Pin native pools before NumPy/pandas are imported.  Phase 0 is I/O-bound and
# must not steal cores from the bounded v2.7 patient workers.
for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["MALLOC_ARENA_MAX"] = "2"

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_event_innovation_data import (  # noqa: E402
    AnchorSplits,
    ContinuityDecision,
    ContinuitySequence,
    assign_continuity_units,
    audit_phase0_contract,
    build_blocked_chronological_crossfit_folds,
    build_continuity_sequences,
    build_cumulative_anchor_splits,
    build_single_event_anchor_splits,
)
from src.topic5_resource_guard import atomic_write_json, pin_thread_environment  # noqa: E402
from src.topic5_source_intervals import (  # noqa: E402,F401
    SourceSegment,
    build_source_segments,
)


DEFAULT_CONFIG = ROOT / "config/topic5_event_innovation_v3_0.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def chronological_split_indices(
    eligible_indices: Sequence[int],
    fractions: Sequence[float],
    *,
    minimum_events: int,
) -> dict[str, np.ndarray]:
    """Return one frozen chronological split inside the canonical train80 pool."""

    eligible = np.asarray(eligible_indices, dtype=np.int64)
    weights = np.asarray(fractions, dtype=float)
    if eligible.ndim != 1 or len(np.unique(eligible)) != len(eligible):
        raise ValueError("eligible indices must be unique and one-dimensional")
    if np.any(np.diff(eligible) <= 0):
        raise ValueError("eligible indices must preserve canonical chronology")
    if weights.shape != (3,) or np.any(weights <= 0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("split fractions must be three positive values summing to one")
    minimum = int(minimum_events)
    if minimum < 1 or len(eligible) < 3 * minimum:
        raise ValueError("insufficient train80 events for the frozen chronological split")
    first = int(np.floor(len(eligible) * weights[0]))
    second = int(np.floor(len(eligible) * (weights[0] + weights[1])))
    first = min(max(first, minimum), len(eligible) - 2 * minimum)
    second = min(max(second, first + minimum), len(eligible) - minimum)
    output = {
        "train": eligible[:first],
        "validation": eligible[first:second],
        "test": eligible[second:],
    }
    if min(map(len, output.values())) < minimum:
        raise RuntimeError("chronological split violates the minimum event count")
    return output


def _load_subject(subject: str, config: Mapping[str, Any]) -> dict[str, Any]:
    dataset_path = ROOT / str(config["dataset_root"]) / "per_subject" / f"{subject}.npz"
    mapping_path = ROOT / str(config["source_mapping_root"]) / f"{subject}.npz"
    with np.load(dataset_path, allow_pickle=False) as data:
        required = {
            "event_local_rank",
            "event_participation",
            "event_abs_time",
            "event_split",
            "contact_names",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise RuntimeError(f"{subject}: dataset fields missing: {missing}")
        values = {
            "rank": np.asarray(data["event_local_rank"], np.float32),
            "participation": np.asarray(data["event_participation"], bool),
            "event_time": np.asarray(data["event_abs_time"], np.float64),
            "event_split": np.asarray(data["event_split"], np.uint8),
            "contact_names": np.asarray(data["contact_names"]).astype(str),
        }
    with np.load(mapping_path, allow_pickle=False) as mapping:
        values["source_id"] = np.asarray(mapping["event_source_block_id"]).astype(str)
        values["record_name"] = np.asarray(mapping["event_source_record_name"]).astype(str)
    length = len(values["event_time"])
    if any(len(values[key]) != length for key in ("rank", "participation", "event_split", "source_id", "record_name")):
        raise RuntimeError(f"{subject}: event arrays are not aligned")
    if values["rank"].shape != values["participation"].shape:
        raise RuntimeError(f"{subject}: rank and participation shapes differ")
    if not np.all(np.diff(values["event_time"]) >= 0):
        raise RuntimeError(f"{subject}: event chronology is not monotonic")
    pairs = set(zip(values["source_id"], values["record_name"]))
    if len(pairs) != len(np.unique(values["source_id"])):
        raise RuntimeError(f"{subject}: source ID does not map one-to-one to record name")
    values["dataset_path"] = dataset_path
    values["mapping_path"] = mapping_path
    return values


def _decision_rows(
    subject: str,
    records: Sequence[Mapping[str, Any]],
    decisions: Sequence[ContinuityDecision],
) -> list[dict[str, Any]]:
    by_source = {str(row["source_id"]): dict(row) for row in records}
    rows: list[dict[str, Any]] = []
    for decision in decisions:
        row = dict(by_source[decision.source_id])
        row.update(asdict(decision))
        rows.append(row)
    return rows


def _sequences_for_splits(
    raw: Mapping[str, Any],
    decisions: Sequence[ContinuityDecision],
    split_indices: Mapping[str, np.ndarray],
) -> dict[str, tuple[ContinuitySequence, ...]]:
    return {
        split: build_continuity_sequences(
            np.asarray(raw["event_time"]),
            np.asarray(raw["source_id"]),
            decisions,
            eligible_indices=indices,
        )
        for split, indices in split_indices.items()
    }


def _anchor_counts(anchors: AnchorSplits) -> dict[str, int]:
    return {split: len(getattr(anchors, split)) for split in ("train", "validation", "test")}


def audit_subject(subject: str, config: Mapping[str, Any], output: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = _load_subject(subject, config)
    train80 = np.flatnonzero(np.asarray(raw["event_split"]) == 0)
    if np.any(np.asarray(raw["event_split"])[train80] != 0):
        raise RuntimeError(f"{subject}: old heldout20 entered eligibility")
    segments, source_records = build_source_segments(
        subject, np.asarray(raw["source_id"]), np.asarray(raw["record_name"]), config
    )
    continuity = config["continuity"]
    decisions = assign_continuity_units(
        segments,
        maximum_gap_seconds=float(continuity["maximum_gap_seconds"]),
        maximum_overlap_seconds=float(continuity["maximum_overlap_seconds"]),
    )
    primary_horizon = int(config["primary_horizon"])
    primary_pre = int(config["primary_pre_events"])
    primary_exposure = min(map(int, config["cumulative_events"]))
    minimum = primary_pre + primary_exposure + primary_horizon
    split_indices = chronological_split_indices(
        train80, config["split_fractions"], minimum_events=minimum
    )
    split_sequences = _sequences_for_splits(raw, decisions, split_indices)
    single = build_single_event_anchor_splits(
        split_sequences,
        pre_events=primary_pre,
        horizon=primary_horizon,
    )
    cumulative = build_cumulative_anchor_splits(
        split_sequences,
        pre_events=primary_pre,
        exposure_events=primary_exposure,
        horizon=primary_horizon,
    )
    folds = build_blocked_chronological_crossfit_folds(
        split_sequences["train"],
        n_splits=int(config["crossfit_splits"]),
        embargo_events=int(config["crossfit_embargo_events"]),
        minimum_train_events=int(config["crossfit_minimum_train_events"]),
        minimum_validation_events=int(config["crossfit_minimum_validation_events"]),
    )
    contract = audit_phase0_contract(split_sequences, single, cumulative, folds)
    if contract["status"] != "PASS":
        raise RuntimeError(f"{subject}: Phase0 anchor contract failed: {contract}")

    horizon_counts: dict[str, Any] = {}
    for pre_events in map(int, config["pre_event_windows"]):
        horizon_counts[str(pre_events)] = {}
        for horizon in map(int, config["horizons"]):
            horizon_counts[str(pre_events)][str(horizon)] = _anchor_counts(
                build_single_event_anchor_splits(
                    split_sequences,
                    pre_events=pre_events,
                    horizon=horizon,
                )
            )
    cumulative_counts: dict[str, Any] = {}
    for pre_events in map(int, config["pre_event_windows"]):
        cumulative_counts[str(pre_events)] = {}
        for exposure in map(int, config["cumulative_events"]):
            cumulative_counts[str(pre_events)][str(exposure)] = _anchor_counts(
                build_cumulative_anchor_splits(
                    split_sequences,
                    pre_events=pre_events,
                    exposure_events=exposure,
                    horizon=primary_horizon,
                )
            )
    event_time = np.asarray(raw["event_time"])
    row = {
        "contract": str(config["contract"]),
        "status": "PHASE0_PATIENT_PASS",
        "subject": subject,
        "one_step_is_one_complete_event": True,
        "n_events": len(event_time),
        "n_train80": len(train80),
        "n_old_heldout20": int(np.sum(np.asarray(raw["event_split"]) == 1)),
        "n_contacts": int(np.asarray(raw["rank"]).shape[1]),
        "n_sources": len(segments),
        "n_continuity_units": len({item.continuity_unit_id for item in decisions}),
        "n_joined_source_boundaries": sum(item.decision == "join_previous" for item in decisions),
        "split_counts": {key: len(value) for key, value in split_indices.items()},
        "split_index_sha256": {key: array_sha256(value) for key, value in split_indices.items()},
        "horizon_anchor_counts": horizon_counts,
        "cumulative_anchor_counts": cumulative_counts,
        "n_crossfit_folds": len(folds),
        "phase0_contract": contract,
        "dataset_path": str(raw["dataset_path"]),
        "dataset_sha256": sha256(Path(raw["dataset_path"])),
        "mapping_path": str(raw["mapping_path"]),
        "mapping_sha256": sha256(Path(raw["mapping_path"])),
        "old_heldout20_entered_into_analysis": False,
        "within_event_next_rank_model_fit": False,
        "observer_or_transition_model_fit": False,
        "forbidden_inputs_read": False,
    }
    patient_root = output / "per_subject"
    atomic_write_json(patient_root / f"{subject}.json", _jsonable(row))
    return row, _decision_rows(subject, source_records, decisions)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--subjects", nargs="*")
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pin_thread_environment(1, disable_cuda=True)
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = (
        args.output_dir
        if args.output_dir is not None and args.output_dir.is_absolute()
        else ROOT / (args.output_dir or Path(str(config["output_root"])))
    )
    manifest_path = ROOT / str(config["dataset_root"]) / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cohort = list(map(str, manifest["cohort_subjects"]))
    subjects = cohort if not args.subjects else list(map(str, args.subjects))
    full_cohort = subjects == cohort
    unknown = sorted(set(subjects) - set(cohort))
    if unknown:
        raise SystemExit(f"subjects outside canonical cohort: {unknown}")

    rows: list[dict[str, Any]] = []
    continuity_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for subject in subjects:
        try:
            row, source_rows = audit_subject(subject, config, output)
        except Exception as exc:  # fail-closed cohort artifact retains exact patient reason
            failures.append({"subject": subject, "error": f"{type(exc).__name__}: {exc}"})
            print(subject, "FAIL", exc, flush=True)
            continue
        rows.append(row)
        continuity_rows.extend(source_rows)
        print(subject, "PASS", flush=True)

    if continuity_rows:
        _atomic_csv(output / "source_continuity_manifest.csv", pd.DataFrame(continuity_rows))
    if failures:
        _atomic_csv(output / "failures.csv", pd.DataFrame(failures))
    feature_schema = {
        "contract": str(config["contract"]),
        "primary": ["masked_normalized_contact_rank", "co_participating_non_tied_pairwise_precedence"],
        "secondary": ["train_only_template_occupancy"],
        "tertiary": ["contact_participation"],
        "nonparticipating_rank_is_masked": True,
        "ties_are_not_forced_into_order": True,
        "one_step_is_one_complete_event": True,
    }
    atomic_write_json(output / "feature_schema.json", feature_schema)
    state_status = (
        "PHASE0_FAIL_CLOSED"
        if failures
        else "PHASE0_COMPLETE"
        if full_cohort
        else "PHASE0_PARTIAL_AUDIT"
    )
    state = {
        "contract": str(config["contract"]),
        "status": state_status,
        "cohort_scope": "full_34" if full_cohort else "explicit_partial_audit",
        "n_requested": len(subjects),
        "n_pass": len(rows),
        "n_failed": len(failures),
        "subjects": subjects,
        "patient_summary": rows,
        "failures": failures,
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__)),
        "data_module_sha256": sha256(ROOT / "src/topic5_event_innovation_data.py"),
        "dataset_manifest_path": str(manifest_path),
        "dataset_manifest_sha256": sha256(manifest_path),
        "old_heldout20_entered_into_analysis": False,
        "within_event_next_rank_model_fit": False,
        "observer_or_transition_model_fit": False,
        "forbidden_inputs_read": False,
    }
    atomic_write_json(output / "anchor_contract.json", _jsonable(state))
    event_inventory = {
        "contract": str(config["contract"]),
        "status": state["status"],
        "n_subjects": len(rows),
        "n_events": int(sum(row["n_events"] for row in rows)),
        "n_train80": int(sum(row["n_train80"] for row in rows)),
        "n_old_heldout20_excluded": int(sum(row["n_old_heldout20"] for row in rows)),
        "subjects": [
            {
                key: row[key]
                for key in (
                    "subject",
                    "n_events",
                    "n_train80",
                    "n_old_heldout20",
                    "n_contacts",
                    "n_sources",
                    "n_continuity_units",
                    "n_joined_source_boundaries",
                    "split_counts",
                )
            }
            for row in rows
        ],
    }
    atomic_write_json(output / "event_inventory.json", _jsonable(event_inventory))
    print(json.dumps({"status": state["status"], "n_pass": len(rows), "n_failed": len(failures)}))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
