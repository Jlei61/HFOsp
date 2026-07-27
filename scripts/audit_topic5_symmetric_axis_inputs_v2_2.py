#!/usr/bin/env python3
"""Audit v2.2 rank, geometry, split, provenance, and A/B read-back inputs.

This script is strictly interictal. It does not open any ictal target table,
cache, value, or label.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _audit_subject(
    row: Any,
    dataset_root: Path,
    development: set[str],
) -> dict[str, Any]:
    subject = str(row.subject)
    npz_path = dataset_root / "per_subject" / f"{subject}.npz"
    json_path = dataset_root / "per_subject" / f"{subject}.json"
    if not npz_path.is_file() or not json_path.is_file():
        raise FileNotFoundError(f"{subject}: missing dataset NPZ/JSON")

    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    with np.load(npz_path, allow_pickle=False) as data:
        required = {
            "event_group_ids",
            "event_local_rank",
            "event_participation",
            "event_abs_time",
            "event_split",
            "contact_names",
            "contact_coords",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise RuntimeError(f"{subject}: missing arrays {missing}")
        group_ids = np.asarray(data["event_group_ids"], dtype=np.int64)
        local_rank = np.asarray(data["event_local_rank"], dtype=float)
        participation = np.asarray(data["event_participation"], dtype=bool)
        event_times = np.asarray(data["event_abs_time"], dtype=float)
        event_split = np.asarray(data["event_split"], dtype=np.uint8)
        names = [str(value) for value in data["contact_names"]]
        coords = np.asarray(data["contact_coords"], dtype=float)

    n_events, n_contacts = participation.shape
    shape_ok = bool(
        group_ids.shape == (n_events, n_contacts)
        and local_rank.shape == (n_events, n_contacts)
        and event_times.shape == (n_events,)
        and event_split.shape == (n_events,)
        and coords.shape == (n_contacts, 3)
        and len(names) == n_contacts
    )
    chronological = bool(
        np.all(np.isfinite(event_times))
        and np.all(np.diff(event_times) >= 0.0)
    )
    split_transition_count = int(np.sum(np.diff(event_split.astype(int)) != 0))
    formal_split_valid = bool(
        set(np.unique(event_split)).issubset({0, 1})
        and event_split[0] == 0
        and event_split[-1] == 1
        and split_transition_count == 1
    )
    mask_valid = bool(
        np.all(group_ids[~participation] == -1)
        and np.all(~np.isfinite(local_rank[~participation]))
        and np.all(np.isfinite(local_rank[participation]))
        and np.all(group_ids[participation] >= 0)
    )
    mapped = np.all(np.isfinite(coords), axis=1)
    geometry_mapped = int(mapped.sum())
    geometry_complete = bool(geometry_mapped == n_contacts)
    metadata_hash_match = bool(
        str(metadata.get("dataset_npz_sha256", "")) == sha256(npz_path)
    )
    forbidden = metadata.get("forbidden_inputs_present")
    forbidden_clear = bool(
        forbidden in (False, None, [], {})
        or (
            isinstance(forbidden, dict)
            and not any(bool(value) for value in forbidden.values())
        )
    )
    dev60 = int(np.floor(0.60 * n_events))
    dev80 = int(np.floor(0.80 * n_events))
    return {
        "dataset": str(row.dataset),
        "subject": subject,
        "development": subject in development,
        "status": str(row.status),
        "n_events": n_events,
        "n_contacts": n_contacts,
        "geometry_mapped": geometry_mapped,
        "geometry_complete": geometry_complete,
        "coord_space": str((metadata.get("geometry") or {}).get("coord_space", "")),
        "shape_ok": shape_ok,
        "chronological": chronological,
        "formal_split_valid": formal_split_valid,
        "n_train80": int(np.sum(event_split == 0)),
        "n_heldout20": int(np.sum(event_split == 1)),
        "n_dev_fit60": dev60,
        "n_dev_validation20": max(dev80 - dev60, 0),
        "n_dev_confirmation20": max(n_events - dev80, 0),
        "masked_rank_valid": mask_valid,
        "forbidden_inputs_clear": forbidden_clear,
        "npz_sha256": sha256(npz_path),
        "json_sha256": sha256(json_path),
        "metadata_npz_hash_match": metadata_hash_match,
        "source_manifest_sha256": str(metadata.get("source_file_manifest_sha256", "")),
        "candidate_target_patient_ignored": True,
        "subject_gate_pass": bool(
            str(row.status) == "ok"
            and shape_ok
            and chronological
            and formal_split_valid
            and mask_valid
            and forbidden_clear
            and metadata_hash_match
        ),
    }


def _ab_readback_inventory(
    subjects: pd.DataFrame,
    dataset_root: Path,
    ab_root: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for subject in subjects["subject"].astype(str):
        dataset_npz = dataset_root / "per_subject" / f"{subject}.npz"
        with np.load(dataset_npz, allow_pickle=False) as data:
            dataset_names = [str(value) for value in data["contact_names"]]
        path = ab_root / f"{subject}.json"
        record: dict[str, Any] = {}
        if path.is_file():
            record = json.loads(path.read_text(encoding="utf-8"))
        axis_pair = record.get("axis_pair") or {}
        shared = axis_pair.get("shared_axis") or {}
        axis_names = [str(value) for value in record.get("names") or []]
        joined = sorted(set(dataset_names) & set(axis_names))
        u = np.asarray(shared.get("u") or [], dtype=float)
        rows.append(
            {
                "dataset": subject.split("_", 1)[0],
                "subject": subject,
                "artifact_exists": path.is_file(),
                "artifact_path": _relative(path),
                "artifact_sha256": sha256(path) if path.is_file() else "",
                "axis_definition": str(record.get("axis_definition", "")),
                "shared_axis_status": str(shared.get("status", "missing")),
                "shared_axis_vector_finite_unit": bool(
                    u.shape == (3,)
                    and np.all(np.isfinite(u))
                    and np.isclose(np.linalg.norm(u), 1.0, atol=1e-5)
                ),
                "n_dataset_contacts": len(dataset_names),
                "n_axis_contacts": len(axis_names),
                "n_exact_joined_contacts": len(joined),
                "all_dataset_contacts_joined": set(dataset_names).issubset(
                    set(axis_names)
                ),
                "readback_estimable": bool(
                    str(record.get("axis_definition", ""))
                    == "template_propagation_axis_v2"
                    and str(shared.get("status", "")) == "ok"
                    and u.shape == (3,)
                    and np.all(np.isfinite(u))
                    and len(joined) >= 3
                ),
                "used_for_training": False,
                "used_for_target_unlock": False,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/topic5_symmetric_axis_propagation_state_v2_2.yaml",
    )
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset_root = ROOT / cfg["inputs"]["rank_dataset"]
    output_root = ROOT / cfg["outputs"]["root"]
    audit_dir = output_root / "input_audit"
    provenance_dir = output_root / "provenance"
    audit_dir.mkdir(parents=True, exist_ok=True)
    provenance_dir.mkdir(parents=True, exist_ok=True)

    source_audit = pd.read_csv(dataset_root / "subject_audit.csv")
    source_audit = source_audit.sort_values(["dataset", "subject"]).reset_index(drop=True)
    development = set(map(str, cfg["cohort"]["development"]))
    rows = [
        _audit_subject(row, dataset_root, development)
        for row in source_audit.itertuples(index=False)
    ]
    inventory = pd.DataFrame(rows)
    inventory.to_csv(audit_dir / "subject_inventory.csv", index=False)

    geometry_complete = set(
        inventory.loc[inventory.geometry_complete, "subject"].astype(str)
    )
    all_subjects = set(inventory.subject.astype(str))
    sequence_formal = sorted(all_subjects - development)
    physical_formal = sorted(geometry_complete - development)
    geometry_incomplete_sequence = sorted(
        set(sequence_formal) - set(physical_formal)
    )

    atomic_json(
        audit_dir / "development_cohort.json",
        {
            "subjects": sorted(development),
            "all_geometry_complete": development.issubset(geometry_complete),
            "selection_used_target_values": False,
        },
    )
    atomic_json(
        audit_dir / "physical_axis_formal_cohort.json",
        {
            "subjects": physical_formal,
            "n_subjects": len(physical_formal),
            "development_excluded": True,
            "geometry_complete_required": True,
        },
    )
    atomic_json(
        audit_dir / "all_subject_sequence_cohort.json",
        {
            "subjects": sequence_formal,
            "n_subjects": len(sequence_formal),
            "physical_axis_subjects": physical_formal,
            "geometry_incomplete_subjects": geometry_incomplete_sequence,
            "geometry_incomplete_axis_fallback": False,
        },
    )

    expected = cfg["cohort"]
    assertions = {
        "all_34": len(inventory) == int(expected["expected_all"]),
        "all_subject_gates": bool(inventory.subject_gate_pass.all()),
        "geometry_complete_25": int(inventory.geometry_complete.sum())
        == int(expected["expected_geometry_complete"]),
        "development_exact": set(inventory.loc[inventory.development, "subject"])
        == development,
        "development_all_geometry_complete": development.issubset(
            geometry_complete
        ),
        "sequence_formal_31": len(sequence_formal)
        == int(expected["expected_sequence_formal"]),
        "physical_axis_formal_22": len(physical_formal)
        == int(expected["expected_physical_axis_formal"]),
        "geometry_incomplete_sequence_9": len(geometry_incomplete_sequence)
        == int(expected["expected_geometry_incomplete_sequence"]),
        "candidate_target_not_used_for_routing": True,
        "target_values_not_read": True,
    }
    gate = {
        "contract": cfg["contract"]["name"],
        "version": cfg["contract"]["version"],
        "status": "pass" if all(assertions.values()) else "fail",
        "assertions": assertions,
        "counts": {
            "all": len(inventory),
            "geometry_complete": int(inventory.geometry_complete.sum()),
            "development": int(inventory.development.sum()),
            "sequence_formal": len(sequence_formal),
            "physical_axis_formal": len(physical_formal),
            "geometry_incomplete_sequence": len(geometry_incomplete_sequence),
        },
        "ictal_target_values_read": False,
    }
    atomic_json(audit_dir / "INPUT_AUDIT_GATE.json", gate)

    ab_inventory = _ab_readback_inventory(
        inventory, dataset_root, ROOT / cfg["inputs"]["ab_axis_root"]
    )
    ab_inventory.to_csv(audit_dir / "ab_axis_readback_inventory.csv", index=False)

    provenance_paths = {
        "config": config_path,
        "spec": ROOT / cfg["contract"]["spec"],
        "plan": ROOT / cfg["contract"]["plan"],
        "old_closeout": ROOT / cfg["inputs"]["old_closeout"],
        "old_report": ROOT / cfg["inputs"]["old_report"],
        "rank_dataset_manifest": dataset_root / "dataset_manifest.json",
        "rank_subject_audit": dataset_root / "subject_audit.csv",
    }
    missing = [name for name, path in provenance_paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing provenance files: {missing}")
    atomic_json(
        provenance_dir / "upstream_manifest.json",
        {
            "contract": cfg["contract"]["name"],
            "version": cfg["contract"]["version"],
            "upstream_status": "v1_complete_bounded_negative",
            "ictal_target_read": False,
            "files": {
                name: {"path": _relative(path), "sha256": sha256(path)}
                for name, path in provenance_paths.items()
            },
        },
    )
    print(json.dumps(gate, indent=2, ensure_ascii=False))
    if gate["status"] != "pass":
        raise SystemExit("v2.2 input audit failed")


if __name__ == "__main__":
    main()
