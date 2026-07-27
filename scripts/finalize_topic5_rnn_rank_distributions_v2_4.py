#!/usr/bin/env python3
"""Median-aggregate and freeze target-sealed v2.4 interictal representations."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/topic5_rnn_axis_positive_static_transfer_v2_4"
AUDIT = BASE / "input_audit/INPUT_AUDIT_STATUS.json"
REP = BASE / "representations"
SEEDS = (17, 29, 43)
ARRAYS = (
    "full_fixed_axis",
    "no_history",
    "local_isotropic",
    "node_only",
    "empirical_train80",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if audit.get("target_values_read") or audit.get(
        "target_arrays_deserialized"
    ):
        raise SystemExit("target seal failed")
    subjects = list(map(str, audit["target_metadata_eligible_patients"]))
    out_root = REP / "per_subject"
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for subject in subjects:
        records = []
        sidecars = []
        for seed in SEEDS:
            path = REP / "per_seed" / f"{subject}_seed{seed}.npz"
            sidecar = REP / "per_seed" / f"{subject}_seed{seed}.json"
            if not path.exists() or not sidecar.exists():
                raise SystemExit(f"missing representation artifact: {path}")
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            if metadata.get("target_values_read"):
                raise SystemExit(f"{subject}: seed representation broke target seal")
            with np.load(path, allow_pickle=False) as data:
                records.append({name: np.asarray(data[name]) for name in data.files})
            sidecars.append(metadata)
        names = records[0]["contact_names"].astype(str)
        if any(
            not np.array_equal(names, record["contact_names"].astype(str))
            for record in records[1:]
        ):
            raise SystemExit(f"{subject}: seed contact order mismatch")
        arrays: dict[str, np.ndarray] = {"contact_names": names}
        for name in ARRAYS:
            values = np.stack([record[name] for record in records])
            aggregate = np.median(values, axis=0)
            row_sum = aggregate.sum(axis=1, keepdims=True)
            if np.any(row_sum <= 0):
                raise SystemExit(f"{subject}/{name}: zero probability row")
            aggregate = (aggregate / row_sum).astype(np.float32)
            if not np.allclose(aggregate.sum(axis=1), 1.0, atol=1.0e-6):
                raise SystemExit(f"{subject}/{name}: invalid probability rows")
            arrays[name] = aggregate
        out_path = out_root / f"{subject}.npz"
        np.savez_compressed(out_path, **arrays)
        rows.append(
            {
                "subject": subject,
                "n_contacts": len(names),
                "n_seeds": len(SEEDS),
                "n_rollouts_per_seed": sidecars[0]["n_rollouts"],
                "output": str(out_path.relative_to(ROOT)),
                "output_sha256": sha256(out_path),
                "seed_outputs_sha256": {
                    str(seed): sidecar["output_sha256"]
                    for seed, sidecar in zip(SEEDS, sidecars)
                },
                "target_values_read": False,
            }
        )
    manifest = {
        "contract": "topic5_rnn_rank_distribution_freeze_v2_4",
        "status": "FROZEN_INTERICTAL_REPRESENTATIONS",
        "n_subjects": len(subjects),
        "subjects": subjects,
        "seeds": list(SEEDS),
        "n_rollouts_per_seed": 5000,
        "representation_arrays": list(ARRAYS),
        "aggregation": (
            "patient_contact_feature_median_across_model_seeds_then_simplex_closure"
        ),
        "rows": rows,
        "target_arrays_deserialized": False,
        "target_values_read": False,
    }
    atomic_json(REP / "REPRESENTATION_FREEZE_MANIFEST.json", manifest)
    unlock = {
        "contract": "topic5_source_free_static_target_unlock_v2_4",
        "status": "FROZEN_INTERICTAL_REPRESENTATIONS",
        "representation_manifest": str(
            (REP / "REPRESENTATION_FREEZE_MANIFEST.json").relative_to(ROOT)
        ),
        "representation_manifest_sha256": sha256(
            REP / "REPRESENTATION_FREEZE_MANIFEST.json"
        ),
        "allowed_target": (
            "clinical-onset [0,10] s 1-150 Hz static contact energy only"
        ),
        "dynamic_source_conditioned_rollout": (
            "BLOCKED_MISSING_EXACT_CLINICAL_ONSET_SOURCE_METADATA"
        ),
        "target_arrays_deserialized_before_freeze": False,
        "target_values_read": False,
    }
    atomic_json(BASE / "TARGET_UNLOCK.json", unlock)
    print(json.dumps(unlock, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
