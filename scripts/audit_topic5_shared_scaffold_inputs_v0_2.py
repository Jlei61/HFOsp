#!/usr/bin/env python3
"""Freeze the 34-patient interictal input and chronological split inventory."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.propagation_skeleton_geometry import parse_shaft  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def split_descriptor(indices: np.ndarray) -> dict[str, Any]:
    index = np.asarray(indices, dtype=np.int64)
    if not index.size:
        raise ValueError("split is empty")
    contiguous = bool(np.array_equal(index, np.arange(index[0], index[-1] + 1)))
    descriptor: dict[str, Any] = {
        "n": int(index.size),
        "first": int(index[0]),
        "last": int(index[-1]),
        "stop_exclusive": int(index[-1] + 1),
        "contiguous": contiguous,
        "indices_sha256": sha256_array(index.astype("<i8")),
    }
    if not contiguous:
        descriptor["indices"] = index.tolist()
    return descriptor


def validate_group_encoding(groups: np.ndarray, counts: np.ndarray, subject: str) -> None:
    if groups.ndim != 2 or counts.shape != groups.shape[:1]:
        raise RuntimeError(f"{subject}: group arrays are misaligned")
    if np.any(groups < -1):
        raise RuntimeError(f"{subject}: group IDs below the masked value -1")
    maximum = np.max(groups, axis=1) + 1
    if not np.array_equal(maximum.astype(counts.dtype), counts):
        raise RuntimeError(f"{subject}: group_count differs from max group ID + 1")
    if np.any(counts < 1):
        raise RuntimeError(f"{subject}: empty event")
    # Exhaustive contiguity check.  This is intentionally an audit-time cost,
    # not part of the trainer's hot path.
    for event_index, (row, count) in enumerate(zip(groups, counts)):
        observed = np.unique(row[row >= 0])
        if not np.array_equal(observed, np.arange(int(count))):
            raise RuntimeError(
                f"{subject}: event {event_index} has non-contiguous rank groups"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/topic5_shared_scaffold_propagation_rnn_v0_2.yaml",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    dataset_artifact_root = Path(config["dataset_artifact_root"]).resolve()
    dataset_root = dataset_artifact_root / config["dataset_root"]
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else ROOT / config["output_root"] / "input_audit"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_path = dataset_root / "dataset_manifest.json"
    source_manifest = json.loads(manifest_path.read_text())
    if bool(source_manifest.get("target_values_read", True)):
        raise RuntimeError("source rank dataset does not certify a sealed target")
    if int(source_manifest.get("n_subjects_ok", -1)) != 34:
        raise RuntimeError("source rank dataset is not the frozen 34-patient cohort")
    audit = pd.read_csv(dataset_root / "subject_audit.csv")
    expected = sorted(audit.loc[audit.status.astype(str) == "ok", "subject"].astype(str))
    if len(expected) != 34:
        raise RuntimeError(f"expected 34 audited subjects, found {len(expected)}")

    rows: list[dict[str, Any]] = []
    splits: dict[str, Any] = {}
    input_hashes: dict[str, str] = {}
    for subject in expected:
        path = dataset_root / "per_subject" / f"{subject}.npz"
        sidecar = path.with_suffix(".json")
        metadata = json.loads(sidecar.read_text())
        actual_hash = sha256_file(path)
        if actual_hash != str(metadata["dataset_npz_sha256"]):
            raise RuntimeError(f"{subject}: source NPZ hash mismatch")
        with np.load(path, allow_pickle=False) as data:
            required = {
                "event_group_ids",
                "event_group_count",
                "event_abs_time",
                "event_split",
                "contact_names",
                "event_participation",
            }
            missing = required.difference(data.files)
            if missing:
                raise RuntimeError(f"{subject}: missing arrays {sorted(missing)}")
            groups = np.asarray(data["event_group_ids"], dtype=np.int16)
            counts = np.asarray(data["event_group_count"], dtype=np.int16)
            timestamps = np.asarray(data["event_abs_time"], dtype=np.float64)
            event_split = np.asarray(data["event_split"], dtype=np.uint8)
            contact_names = np.asarray(data["contact_names"]).astype(str)
            participation = np.asarray(data["event_participation"], dtype=bool)
        validate_group_encoding(groups, counts, subject)
        if participation.shape != groups.shape or not np.array_equal(
            participation, groups >= 0
        ):
            raise RuntimeError(f"{subject}: masked participation does not match group IDs")
        if timestamps.shape != counts.shape or not np.isfinite(timestamps).all():
            raise RuntimeError(f"{subject}: invalid event timestamps")
        if np.any(np.diff(timestamps) < 0):
            raise RuntimeError(f"{subject}: events are not chronological")
        if event_split.shape != counts.shape or not set(np.unique(event_split)).issubset({0, 1}):
            raise RuntimeError(f"{subject}: invalid outer event split")
        if len(np.unique(contact_names)) != len(contact_names):
            raise RuntimeError(f"{subject}: duplicate contact name")

        train80 = np.flatnonzero(event_split == 0)
        test20 = np.flatnonzero(event_split == 1)
        fit_n = int(np.floor(0.75 * len(train80)))
        fit60 = train80[:fit_n]
        validation20 = train80[fit_n:]
        if not len(fit60) or not len(validation20) or not len(test20):
            raise RuntimeError(f"{subject}: empty chronological split")
        if not (
            timestamps[fit60[-1]] <= timestamps[validation20[0]]
            <= timestamps[test20[0]]
        ):
            raise RuntimeError(f"{subject}: split boundaries are not chronological")
        if np.intersect1d(fit60, validation20).size or np.intersect1d(
            np.concatenate([fit60, validation20]), test20
        ).size:
            raise RuntimeError(f"{subject}: split overlap")

        parsed = [parse_shaft(name) for name in contact_names]
        shaft_names = [str(shaft) if shaft is not None else "" for shaft, _ in parsed]
        shaft_ordinals = [int(ordinal) if ordinal is not None else None for _, ordinal in parsed]
        n_parsed = sum(bool(name) for name in shaft_names)
        by_shaft: dict[str, int] = {}
        for shaft in shaft_names:
            if shaft:
                by_shaft[shaft] = by_shaft.get(shaft, 0) + 1
        n_shaft_edges = sum(max(0, count - 1) for count in by_shaft.values())
        if not n_shaft_edges:
            raise RuntimeError(f"{subject}: no fixed shaft adjacency edge")

        split_payload = {
            "fit60": split_descriptor(fit60),
            "validation20": split_descriptor(validation20),
            "test20": split_descriptor(test20),
            "contact_names": contact_names.tolist(),
            "shaft_names": shaft_names,
            "shaft_ordinals": shaft_ordinals,
            "contact_mapping_shared_across_splits": True,
            "event_timestamp_sha256": sha256_array(timestamps.astype("<f8")),
        }
        splits[subject] = split_payload
        input_hashes[subject] = actual_hash
        rows.append(
            {
                "subject": subject,
                "dataset": str(
                    audit.loc[audit.subject.astype(str) == subject, "dataset"].iloc[0]
                ),
                "n_contacts": int(len(contact_names)),
                "n_shafts": int(len(by_shaft)),
                "n_parsed_shaft_contacts": int(n_parsed),
                "n_fixed_shaft_edges": int(n_shaft_edges),
                "n_events": int(len(groups)),
                "n_fit60": int(len(fit60)),
                "n_validation20": int(len(validation20)),
                "n_test20": int(len(test20)),
                "max_rank_sets": int(np.max(counts)),
                "masked_nonparticipation_valid": True,
                "chronology_valid": True,
                "split_nonempty": True,
                "contact_mapping_shared": True,
                "dataset_npz_sha256": actual_hash,
            }
        )

    inventory = pd.DataFrame(rows).sort_values("subject")
    inventory.to_csv(output_root / "subject_inventory.csv", index=False)
    atomic_json(
        output_root / "split_manifest.json",
        {
            "contract": config["contract"],
            "n_subjects": len(splits),
            "split": "chronological fit60/validation20/test20",
            "target_values_read": False,
            "subjects": splits,
        },
    )
    fingerprints = {
        "contract": config["contract"],
        "n_subjects": len(input_hashes),
        "target_values_read": False,
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "dataset_subject_audit_sha256": sha256_file(dataset_root / "subject_audit.csv"),
        "config_sha256": sha256_file(config_path),
        "core_code_sha256": sha256_file(ROOT / "src/topic5_shared_scaffold_rnn.py"),
        "runner_code_sha256": sha256_file(
            ROOT / "scripts/run_topic5_shared_scaffold_rnn_unit_v0_2.py"
        ),
        "per_subject_npz_sha256": input_hashes,
    }
    atomic_json(output_root / "input_fingerprints.json", fingerprints)
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "n_subjects": len(rows),
                "n_events": int(inventory.n_events.sum()),
                "n_contacts_min": int(inventory.n_contacts.min()),
                "n_contacts_max": int(inventory.n_contacts.max()),
                "target_values_read": False,
                "output_root": str(output_root),
            }
        )
    )


if __name__ == "__main__":
    main()
