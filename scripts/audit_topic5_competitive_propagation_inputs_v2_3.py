#!/usr/bin/env python3
"""Freeze v2.3 cohorts, tied-rank denominator, axes, and target seal."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_transition_decomposition_v0_1 import (  # noqa: E402
    estimate_pair_residual,
    geometry_features,
    select_axis_residual,
)


DATASET = (
    ROOT / "results/topic5_interictal_rank_distribution/dataset_v0_4"
)
V22 = ROOT / "results/topic5_symmetric_axis_propagation_state_v2_2"
DECOMPOSITION = (
    ROOT / "results/topic5_interictal_transition_decomposition_v0_1"
)
OUT = (
    ROOT
    / "results/topic5_symmetric_axis_competitive_propagation_v2_3"
    / "input_audit"
)
DEVELOPMENT = (
    "epilepsiae_1077",
    "epilepsiae_1146",
    "yuquan_chengshuai",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_npz(subject: str) -> dict[str, np.ndarray]:
    path = DATASET / "per_subject" / f"{subject}.npz"
    with np.load(path, allow_pickle=False) as data:
        record = {key: np.asarray(data[key]) for key in data.files}
    groups = np.asarray(record["event_group_ids"], dtype=np.int64)
    split = np.asarray(record["event_split"], dtype=np.uint8)
    if groups.ndim != 2 or split.shape != (len(groups),):
        raise ValueError(f"{subject}: invalid event arrays")
    return record


def non_source_tie_ranks(event: np.ndarray) -> list[int]:
    ranks = np.asarray(event, dtype=np.int64)
    values, counts = np.unique(ranks[ranks > 0], return_counts=True)
    return [int(value) for value, count in zip(values, counts) if count > 1]


def development_axis(
    groups: np.ndarray,
    split: np.ndarray,
    names: np.ndarray,
    coords: np.ndarray,
) -> dict[str, Any]:
    train80 = np.flatnonzero(split == 0)
    pair = estimate_pair_residual(groups, train80)
    features = geometry_features(names.astype(str), coords)
    selected = select_axis_residual(
        pair,
        coords,
        [features["same_shaft"], features["local_distance"]],
        n_directions=32,
    )
    axis = np.asarray(selected["axis"], dtype=np.float64)
    return {
        "selected_axis_index": int(selected["axis_index"]),
        "axis_x": float(axis[0]),
        "axis_y": float(axis[1]),
        "axis_z": float(axis[2]),
        "local_axis_frobenius_cosine": float(
            selected["local_axis_frobenius_cosine"]
        ),
        "axis_excess_coefficient": float(
            np.asarray(selected["coefficients"])[-1]
        ),
        "axis_train_pair_mse": float(selected["train_pair_mse"]),
        "axis_selection_events": int(len(train80)),
        "axis_selection_split": "chronological_train80_only",
        "n_candidate_directions": 32,
    }


def main() -> None:
    manifest_path = DATASET / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    subjects = tuple(map(str, manifest["cohort_subjects"]))
    if len(subjects) != 34 or manifest.get("n_subjects_ok") != 34:
        raise SystemExit("dataset v0.4 cohort is not the frozen 34-patient cohort")
    if tuple(subject for subject in subjects if subject in DEVELOPMENT) != DEVELOPMENT:
        raise SystemExit("development patient order drifted")

    sequence_lock_path = V22 / "formal/ALL_SUBJECT_SEQUENCE_LOCK.json"
    physical_lock_path = V22 / "formal/PHYSICAL_AXIS_FORMAL_LOCK.json"
    sequence_lock = json.loads(sequence_lock_path.read_text(encoding="utf-8"))
    physical_lock = json.loads(physical_lock_path.read_text(encoding="utf-8"))
    sequence = tuple(map(str, sequence_lock["subjects"]))
    physical = tuple(map(str, physical_lock["subjects"]))
    expected_sequence = tuple(
        subject for subject in subjects if subject not in DEVELOPMENT
    )
    if sequence != expected_sequence or len(sequence) != 31:
        raise SystemExit("31-patient sequence lock drifted")
    if len(physical) != 22 or any(subject in DEVELOPMENT for subject in physical):
        raise SystemExit("22-patient physical-axis lock drifted")

    target_gate_path = V22 / "target_audit/TARGET_METADATA_GATE.json"
    target_gate = json.loads(target_gate_path.read_text(encoding="utf-8"))
    forbidden_target_flags = (
        "energy_values_read",
        "recruitment_values_read",
        "target_values_read",
    )
    if any(bool(target_gate.get(key, False)) for key in forbidden_target_flags):
        raise SystemExit("early-ictal target seal is not intact")

    operator_path = DECOMPOSITION / "operator_component_metrics.csv"
    operator = pd.read_csv(operator_path)
    if set(operator.subject.astype(str)) != set(physical):
        raise SystemExit("decomposition axis inventory does not match n=22 lock")

    OUT.mkdir(parents=True, exist_ok=True)
    subject_rows: list[dict[str, Any]] = []
    tied_rows: list[dict[str, Any]] = []
    development_axis_rows: list[dict[str, Any]] = []
    total_events = 0
    total_tied = 0
    for subject in subjects:
        record = load_npz(subject)
        groups = np.asarray(record["event_group_ids"], dtype=np.int64)
        split = np.asarray(record["event_split"], dtype=np.uint8)
        train80 = np.flatnonzero(split == 0)
        heldout20 = np.flatnonzero(split == 1)
        train60_count = int(np.floor(0.75 * len(train80)))
        train60 = train80[:train60_count]
        validation20 = train80[train60_count:]
        tied_indices = [
            index
            for index, event in enumerate(groups)
            if non_source_tie_ranks(event)
        ]
        total_events += len(groups)
        total_tied += len(tied_indices)
        for event_index in tied_indices:
            tied_rows.append(
                {
                    "subject": subject,
                    "event_index": int(event_index),
                    "event_split": (
                        "train80" if split[event_index] == 0 else "heldout20"
                    ),
                    "tied_non_source_ranks": ";".join(
                        map(str, non_source_tie_ranks(groups[event_index]))
                    ),
                    "event_group_count": int(
                        np.max(groups[event_index][groups[event_index] >= 0])
                        + 1
                    ),
                    "excluded_from_primary": True,
                }
            )
        subject_rows.append(
            {
                "subject": subject,
                "n_contacts": int(groups.shape[1]),
                "n_events_total": int(len(groups)),
                "n_events_train60": int(len(train60)),
                "n_events_validation20": int(len(validation20)),
                "n_events_train80": int(len(train80)),
                "n_events_heldout20": int(len(heldout20)),
                "n_non_source_tied_events": int(len(tied_indices)),
                "n_primary_events": int(len(groups) - len(tied_indices)),
                "development": subject in DEVELOPMENT,
                "sequence_formal": subject in sequence,
                "physical_axis_formal": subject in physical,
                "target_values_read": False,
                "input_sha256": sha256(
                    DATASET / "per_subject" / f"{subject}.npz"
                ),
            }
        )
        if subject in DEVELOPMENT:
            row = {
                "subject": subject,
                **development_axis(
                    groups,
                    split,
                    np.asarray(record["contact_names"]),
                    np.asarray(record["contact_coords"], dtype=np.float64),
                ),
                "target_values_read": False,
            }
            development_axis_rows.append(row)

    if total_events != 864_163 or total_tied != 25:
        raise SystemExit(
            f"frozen denominator drifted: events={total_events}, tied={total_tied}"
        )
    if len(tied_rows) != 25 or len(development_axis_rows) != 3:
        raise SystemExit("tied-event or development-axis inventory is incomplete")

    subject_frame = pd.DataFrame(subject_rows)
    tied_frame = pd.DataFrame(tied_rows)
    development_axis_frame = pd.DataFrame(development_axis_rows)
    formal_axis_frame = operator[
        [
            "subject",
            "selected_axis_index",
            "axis_x",
            "axis_y",
            "axis_z",
            "local_axis_frobenius_cosine",
            "axis_excess_coefficient",
            "axis_train_pair_mse",
        ]
    ].copy()
    formal_axis_frame["axis_selection_split"] = (
        "frozen_transition_decomposition_train80_only"
    )
    formal_axis_frame["n_candidate_directions"] = 32
    formal_axis_frame["target_values_read"] = False

    subject_frame.to_csv(OUT / "subject_denominator_inventory.csv", index=False)
    tied_frame.to_csv(OUT / "excluded_tied_event_inventory.csv", index=False)
    development_axis_frame.to_csv(
        OUT / "development_axis_inventory.csv", index=False
    )
    formal_axis_frame.to_csv(OUT / "formal_axis_inventory.csv", index=False)

    status = {
        "contract": "topic5_symmetric_axis_competitive_propagation_v2_3",
        "status": "PASS",
        "n_subjects": len(subjects),
        "n_events_total": total_events,
        "n_non_source_tied_events_excluded": total_tied,
        "n_primary_events": total_events - total_tied,
        "n_development_patients": len(DEVELOPMENT),
        "n_sequence_formal_patients": len(sequence),
        "n_physical_axis_formal_patients": len(physical),
        "development_patients": list(DEVELOPMENT),
        "sequence_formal_patients": list(sequence),
        "physical_axis_formal_patients": list(physical),
        "axis_direction_count": 32,
        "development_axes_recomputed_from_train80": True,
        "formal_axes_consumed_from_frozen_decomposition": True,
        "chronological_split_intact": True,
        "heldout_used_for_axis_selection": False,
        "heldout_used_for_node_bias": False,
        "target_values_read": False,
        "early_ictal_transfer_status": (
            "BLOCKED_INTERICTAL_GATE_AND_MISSING_SOURCE_METADATA"
        ),
        "checksums": {
            "dataset_manifest": sha256(manifest_path),
            "sequence_lock": sha256(sequence_lock_path),
            "physical_axis_lock": sha256(physical_lock_path),
            "decomposition_operator_table": sha256(operator_path),
            "target_metadata_gate": sha256(target_gate_path),
        },
    }
    atomic_json(OUT / "INPUT_AUDIT_STATUS.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
