#!/usr/bin/env python3
"""Freeze v2.4 axis-positive cohorts and BB150 target metadata without values."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any
import zipfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


V23 = ROOT / "results/topic5_symmetric_axis_competitive_propagation_v2_3"
DATASET = ROOT / "results/topic5_interictal_rank_distribution/dataset_v0_4"
AXIS = ROOT / "results/topic5_ictal_recruitment/template_axis_field"
TARGET = (
    ROOT
    / "results/topic5_ictal_recruitment/t0_feature_cache_bb150_1_150"
)
OUT = (
    ROOT
    / "results/topic5_rnn_axis_positive_static_transfer_v2_4/input_audit"
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


def full_subject(dataset: str, subject: str) -> str:
    value = str(subject)
    return value if value.startswith(f"{dataset}_") else f"{dataset}_{value}"


def npz_member_names(path: Path) -> set[str]:
    """Inspect ZIP member names only; target arrays are never deserialized."""
    with zipfile.ZipFile(path) as archive:
        return {
            Path(name).stem
            for name in archive.namelist()
            if name.endswith(".npy")
        }


def load_dataset_names(subject: str) -> list[str]:
    path = DATASET / "per_subject" / f"{subject}.npz"
    with np.load(path, allow_pickle=False) as data:
        return [str(value) for value in data["contact_names"]]


def main() -> None:
    v23_status = json.loads(
        (V23 / "input_audit/INPUT_AUDIT_STATUS.json").read_text(
            encoding="utf-8"
        )
    )
    formal = tuple(map(str, v23_status["physical_axis_formal_patients"]))
    if len(formal) != 22:
        raise SystemExit("v2.3 physical-axis formal cohort is not n=22")

    axis_frame = pd.read_csv(AXIS / "axis_cohort.csv")
    axis_frame["full_subject"] = [
        full_subject(dataset, subject)
        for dataset, subject in zip(axis_frame.dataset, axis_frame.subject)
    ]
    primary_frame = axis_frame.loc[
        axis_frame.full_subject.isin(formal)
        & axis_frame.axis_pair_estimable.fillna(False).astype(bool)
        & axis_frame.geometry_2d_supported.fillna(False).astype(bool)
        & axis_frame.collinear_60deg.fillna(False).astype(bool)
    ].copy()
    reversed_frame = primary_frame.loc[
        primary_frame.relation.astype(str) == "reversed"
    ].copy()
    strict_frame = reversed_frame.loc[
        reversed_frame.strict_stability_pass.fillna(False).astype(bool)
    ].copy()
    if (len(primary_frame), len(reversed_frame), len(strict_frame)) != (9, 6, 5):
        raise SystemExit(
            "axis-positive denominator drifted: "
            f"{len(primary_frame)}/{len(reversed_frame)}/{len(strict_frame)}"
        )

    axis_rows: list[dict[str, Any]] = []
    for _, row in primary_frame.sort_values("full_subject").iterrows():
        subject = str(row.full_subject)
        record_path = AXIS / "per_subject" / f"{subject}.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        shared = record["axis_pair"]["shared_axis"]
        if shared.get("status") != "ok":
            raise SystemExit(f"{subject}: shared axis is unavailable")
        vector = np.asarray(shared["u"], dtype=np.float64)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise SystemExit(f"{subject}: invalid shared axis")
        axis_rows.append(
            {
                "subject": subject,
                "relation": str(row.relation),
                "abs_cos_uA_uB": float(row.abs_cos_uA_uB),
                "strict_stability_pass": bool(row.strict_stability_pass),
                "robust_collinear": bool(row.robust_collinear),
                "shared_axis_x": float(vector[0]),
                "shared_axis_y": float(vector[1]),
                "shared_axis_z": float(vector[2]),
                "axis_artifact": str(record_path.relative_to(ROOT)),
                "axis_artifact_sha256": sha256(record_path),
                "selected_without_v23_metrics": True,
                "target_values_read": False,
            }
        )

    target_rows: list[dict[str, Any]] = []
    for subject in formal:
        sidecar_path = TARGET / f"{subject}.json"
        npz_path = TARGET / f"{subject}.npz"
        row: dict[str, Any] = {
            "subject": subject,
            "sidecar_exists": sidecar_path.exists(),
            "npz_exists": npz_path.exists(),
            "metadata_eligible": False,
            "target_values_read": False,
        }
        if sidecar_path.exists() and npz_path.exists():
            metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
            members = npz_member_names(npz_path)
            expected = [int(value) for value in metadata.get("eligible_idxs", [])]
            target_members = {f"bb150_auc__{index}" for index in expected}
            dataset_names = load_dataset_names(subject)
            target_names = [str(value) for value in metadata.get("channels", [])]
            common = sorted(set(dataset_names).intersection(target_names))
            contract_ok = (
                metadata.get("band_broad_1_150") == [1.0, 150.0]
                and metadata.get("t_window") == [0.0, 10.0]
                and bool(metadata.get("line_noise_masked_1_150"))
                and bool(expected)
                and target_members.issubset(members)
                and "channels" in members
            )
            row.update(
                {
                    "n_expected_seizures": len(expected),
                    "n_cached_seizures": len(target_members.intersection(members)),
                    "n_model_contacts": len(dataset_names),
                    "n_target_contacts": len(target_names),
                    "n_exact_joined_contacts": len(common),
                    "minimum_contact_gate": len(common) >= 6,
                    "target_contract_ok": contract_ok,
                    "metadata_eligible": bool(contract_ok and len(common) >= 6),
                    "sidecar_sha256": sha256(sidecar_path),
                    "npz_sha256": sha256(npz_path),
                }
            )
        target_rows.append(row)

    OUT.mkdir(parents=True, exist_ok=True)
    axis_out = pd.DataFrame(axis_rows)
    target_out = pd.DataFrame(target_rows)
    axis_out.to_csv(OUT / "axis_positive_cohort.csv", index=False)
    target_out.to_csv(OUT / "target_metadata_inventory.csv", index=False)
    eligible = target_out.loc[target_out.metadata_eligible, "subject"].tolist()
    if len(eligible) < 8:
        raise SystemExit(f"fewer than 8 target-metadata eligible patients: {eligible}")

    status = {
        "contract": "topic5_axis_positive_static_transfer_v2_4",
        "status": "PASS",
        "physical_axis_formal_n": len(formal),
        "axis_positive_primary_n": len(primary_frame),
        "axis_reversed_n": len(reversed_frame),
        "axis_strict_reversed_n": len(strict_frame),
        "axis_positive_primary_patients": axis_out.subject.tolist(),
        "axis_reversed_patients": sorted(
            reversed_frame.full_subject.astype(str).tolist()
        ),
        "axis_strict_reversed_patients": sorted(
            strict_frame.full_subject.astype(str).tolist()
        ),
        "target_metadata_eligible_n": len(eligible),
        "target_metadata_eligible_patients": eligible,
        "axis_positive_target_metadata_intersection": sorted(
            set(axis_out.subject).intersection(eligible)
        ),
        "target_arrays_deserialized": False,
        "target_values_read": False,
        "checksums": {
            "v23_input_audit": sha256(
                V23 / "input_audit/INPUT_AUDIT_STATUS.json"
            ),
            "v23_formal_gate": sha256(
                V23 / "formal/FORMAL_GATE_STATUS.json"
            ),
            "axis_cohort": sha256(AXIS / "axis_cohort.csv"),
            "dataset_manifest": sha256(DATASET / "dataset_manifest.json"),
        },
    }
    atomic_json(OUT / "INPUT_AUDIT_STATUS.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
