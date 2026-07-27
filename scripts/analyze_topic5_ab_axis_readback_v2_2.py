#!/usr/bin/env python3
"""Frozen post-hoc RNN-to-A/B physical-axis read-back for Topic-5 v2.2."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_axis_readback_v2_2 import (  # noqa: E402
    empirical_upper_percentile,
    frozen_random_axes_by_subject,
    line_axis_consensus,
    sign_invariant_cosine,
    sign_invariant_projection_spearman,
)


BASE = ROOT / "results/topic5_symmetric_axis_propagation_state_v2_2"
DATASET = (
    ROOT / "results/topic5_interictal_rank_distribution/dataset_v0_4/per_subject"
)


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
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def _require_frozen_claim2() -> dict[str, Any]:
    path = BASE / "formal/analysis/CLAIM2_STATUS.json"
    if not path.is_file():
        raise SystemExit("Claim-2 status is absent; A/B read-back remains locked")
    status = json.loads(path.read_text(encoding="utf-8"))
    if status.get("status") != "complete":
        raise SystemExit("all 66 Claim-2 scores must be frozen before read-back")
    if status.get("target_values_read") is not False:
        raise RuntimeError("Claim-2 target seal violation")
    return status


def _learned_axes(subject: str, seeds: list[int]) -> tuple[np.ndarray, list[dict]]:
    axes = []
    rows = []
    for seed in seeds:
        run = BASE / "formal/claim2_runs" / subject / f"seed_{seed}"
        if not (run / "COMPLETE").is_file():
            raise RuntimeError(f"{subject}/seed_{seed} is not complete")
        record = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        if record.get("target_values_read") is not False:
            raise RuntimeError(f"{subject}/seed_{seed} target seal violation")
        axis = np.asarray(
            record["models"]["full"]["heldout_fit"]["parameters"]["axis"],
            dtype=np.float64,
        )
        axes.append(axis)
        rows.append(
            {
                "subject": subject,
                "seed": seed,
                "axis_x": float(axis[0]),
                "axis_y": float(axis[1]),
                "axis_z": float(axis[2]),
            }
        )
    return np.stack(axes), rows


def _load_joined_geometry(subject: str, artifact: dict[str, Any]) -> np.ndarray:
    names = list(map(str, artifact["names"]))
    coords = np.asarray(artifact["coords"], dtype=np.float64)
    with np.load(DATASET / f"{subject}.npz", allow_pickle=False) as data:
        dataset_names = list(map(str, data["contact_names"]))
        dataset_coords = np.asarray(data["contact_coords"], dtype=np.float64)
    if len(names) != len(set(names)) or len(dataset_names) != len(set(dataset_names)):
        raise RuntimeError(f"{subject}: duplicated contact names")
    if set(names) != set(dataset_names):
        raise RuntimeError(f"{subject}: A/B read-back contact join is not exact")
    index = {name: i for i, name in enumerate(names)}
    aligned = coords[[index[name] for name in dataset_names]]
    if not np.allclose(aligned, dataset_coords, rtol=0, atol=1.0e-3):
        raise RuntimeError(f"{subject}: A/B and rank-dataset coordinates drifted")
    return dataset_coords


def main() -> None:
    claim2 = _require_frozen_claim2()
    lock_path = BASE / "formal/PHYSICAL_AXIS_FORMAL_LOCK.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    subjects = list(map(str, lock["subjects"]))
    seeds = list(map(int, lock["seeds"]))
    if len(subjects) != 22 or seeds != [17, 29, 43]:
        raise RuntimeError("formal cohort or seed contract drifted")
    inventory_path = BASE / "input_audit/ab_axis_readback_inventory.csv"
    inventory = pd.read_csv(inventory_path).set_index("subject", drop=False)
    random_axes = frozen_random_axes_by_subject(
        subjects,
        seed=int(lock["null_seed"]),
        n_directions=256,
    )
    random_lock = BASE / "formal/claim3_random_axis_nulls/RANDOM_AXIS_NULL_LOCK.json"
    if random_lock.is_file():
        manifest = json.loads(random_lock.read_text(encoding="utf-8"))
        for subject in subjects:
            saved = np.load(ROOT / manifest["files"][subject]["path"])
            if not np.array_equal(saved, random_axes[subject]):
                raise RuntimeError(f"{subject}: read-back and Claim-3 null axes differ")

    seed_rows: list[dict[str, Any]] = []
    patient_rows: list[dict[str, Any]] = []
    for subject in subjects:
        axes, axis_rows = _learned_axes(subject, seeds)
        if subject not in inventory.index:
            raise RuntimeError(f"{subject}: A/B inventory missing")
        inv = inventory.loc[subject]
        if not bool(inv["readback_estimable"]):
            patient_rows.append(
                {
                    "subject": subject,
                    "status": "not_estimable",
                    "reason": f"shared_axis_status={inv['shared_axis_status']}",
                }
            )
            continue
        artifact_path = ROOT / str(inv["artifact_path"])
        if sha256(artifact_path) != str(inv["artifact_sha256"]):
            raise RuntimeError(f"{subject}: A/B artifact changed after inventory")
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        if (
            artifact.get("axis_definition") != "template_propagation_axis_v2"
            or artifact["axis_pair"]["shared_axis"]["status"] != "ok"
        ):
            raise RuntimeError(f"{subject}: A/B axis contract drifted")
        ab_axis = np.asarray(
            artifact["axis_pair"]["shared_axis"]["u"], dtype=np.float64
        )
        coords = _load_joined_geometry(subject, artifact)
        for row, learned in zip(axis_rows, axes):
            row.update(
                {
                    "abs_axis_cosine": sign_invariant_cosine(learned, ab_axis),
                    "abs_projection_spearman": (
                        sign_invariant_projection_spearman(
                            coords, learned, ab_axis
                        )
                    ),
                }
            )
            seed_rows.append(row)
        consensus = line_axis_consensus(axes)
        observed_cosine = sign_invariant_cosine(consensus, ab_axis)
        observed_spearman = sign_invariant_projection_spearman(
            coords, consensus, ab_axis
        )
        null_cosine = np.asarray(
            [
                sign_invariant_cosine(axis, ab_axis)
                for axis in random_axes[subject]
            ]
        )
        null_spearman = np.asarray(
            [
                sign_invariant_projection_spearman(coords, axis, ab_axis)
                for axis in random_axes[subject]
            ]
        )
        patient_rows.append(
            {
                "subject": subject,
                "status": "estimable",
                "reason": "",
                "n_contacts": len(coords),
                "consensus_axis_x": float(consensus[0]),
                "consensus_axis_y": float(consensus[1]),
                "consensus_axis_z": float(consensus[2]),
                "abs_axis_cosine": observed_cosine,
                "abs_axis_cosine_null_percentile": empirical_upper_percentile(
                    observed_cosine, null_cosine
                ),
                "abs_projection_spearman": observed_spearman,
                "abs_projection_spearman_null_percentile": (
                    empirical_upper_percentile(observed_spearman, null_spearman)
                ),
                "random_axis_n": 256,
            }
        )

    analysis = BASE / "formal/analysis"
    seed_frame = pd.DataFrame(seed_rows)
    patient_frame = pd.DataFrame(patient_rows)
    seed_frame.to_csv(analysis / "ab_axis_readback_seed_metrics.csv", index=False)
    patient_frame.to_csv(analysis / "ab_axis_readback.csv", index=False)
    estimable = patient_frame[patient_frame.status == "estimable"]
    status = {
        "contract": "topic5_symmetric_axis_propagation_state_rnn",
        "version": "2.2",
        "status": "complete",
        "role": "posthoc_secondary_nonblocking",
        "formal_patients": 22,
        "estimable_patients": int(len(estimable)),
        "median_abs_axis_cosine": (
            float(estimable.abs_axis_cosine.median())
            if len(estimable)
            else None
        ),
        "median_abs_projection_spearman": (
            float(estimable.abs_projection_spearman.median())
            if len(estimable)
            else None
        ),
        "median_axis_null_percentile": (
            float(estimable.abs_axis_cosine_null_percentile.median())
            if len(estimable)
            else None
        ),
        "claim2_status_sha256": sha256(
            BASE / "formal/analysis/CLAIM2_STATUS.json"
        ),
        "physical_lock_sha256": sha256(lock_path),
        "ab_inventory_sha256": sha256(inventory_path),
        "used_for_training": False,
        "used_for_target_unlock": False,
        "target_values_read": False,
        "claim2_gate_result": {
            "next": claim2["claim2_next"],
            "future": claim2["claim2_future"],
        },
    }
    atomic_json(analysis / "AB_AXIS_READBACK_STATUS.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
