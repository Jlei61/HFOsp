#!/usr/bin/env python3
"""Prepare a blinded per-seizure clinical-onset source annotation registry.

The script reads routing metadata only.  It never opens early-ictal energy
arrays and never substitutes SOZ, A/B endpoints, patient-level focus, or
energy-ranked contacts for a per-seizure clinical-onset source set.
"""
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
V22 = ROOT / "results/topic5_symmetric_axis_propagation_state_v2_2"
SOURCE = V22 / "target_audit/seizure_inventory.csv"
DATASET = ROOT / "results/topic5_interictal_rank_distribution/dataset_v0_4/per_subject"
OUT = ROOT / "results/topic5_clinical_onset_source_annotation_v0_1"
REGISTRY = OUT / "annotation_registry.csv"


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


def expected_registry(source: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in source.itertuples(index=False):
        rows.append(
            {
                "patient_id": str(record.subject),
                "seizure_id": str(record.seizure_id),
                "dataset": str(record.dataset),
                "clinical_onset_time": float(record.clinical_onset_epoch_metadata),
                "clinical_onset_contacts": "",
                "montage": "",
                "reference": "",
                "annotation_source": "",
                "reviewer_1": "",
                "reviewer_1_contacts": "",
                "reviewer_2": "",
                "reviewer_2_contacts": "",
                "consensus_status": "PENDING_BLINDED_REVIEW",
                "consensus_contacts": "",
                "confidence": "",
                "exact_contact_join_status": "NOT_ATTEMPTED",
                "exclusion_reason": "awaiting_blinded_manual_annotation",
                "model_contact_count": int(record.n_model_contacts),
                "annotation_blinded_to_model_scores": True,
                "annotation_blinded_to_energy_values": True,
                "target_values_read": False,
            }
        )
    return pd.DataFrame(rows)


def validate_registry(frame: pd.DataFrame, source: pd.DataFrame) -> None:
    required = set(expected_registry(source).columns)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"annotation registry missing fields: {sorted(missing)}")
    expected_keys = set(
        zip(source.subject.astype(str), source.seizure_id.astype(str))
    )
    observed_keys = set(
        zip(frame.patient_id.astype(str), frame.seizure_id.astype(str))
    )
    if expected_keys != observed_keys or len(frame) != len(source):
        raise ValueError("annotation registry seizure denominator drifted")
    forbidden = {
        "soz_contacts",
        "energy_top_contacts",
        "ab_source",
        "patient_level_focus",
    }
    if forbidden.intersection(frame.columns):
        raise ValueError("forbidden source-substitution field in registry")
    if not frame.annotation_blinded_to_model_scores.astype(bool).all():
        raise ValueError("model-score blinding contract was changed")
    if not frame.annotation_blinded_to_energy_values.astype(bool).all():
        raise ValueError("energy-value blinding contract was changed")
    if frame.target_values_read.astype(bool).any():
        raise ValueError("target-value seal was violated")


def main() -> None:
    target = json.loads(
        (V22 / "target_audit/TARGET_METADATA_GATE.json").read_text()
    )
    if target.get("energy_values_read") or target.get("recruitment_values_read"):
        raise SystemExit("target values were read before clinical annotation")
    source = pd.read_csv(SOURCE, dtype={"seizure_id": str})
    if len(source) != 71 or source.subject.nunique() != 13:
        raise SystemExit("clinical metadata denominator drifted from 13/71")
    OUT.mkdir(parents=True, exist_ok=True)
    if REGISTRY.exists():
        registry = pd.read_csv(
            REGISTRY,
            dtype={"patient_id": str, "seizure_id": str},
            keep_default_na=False,
        )
    else:
        registry = expected_registry(source)
        registry.to_csv(REGISTRY, index=False)
    validate_registry(registry, source)

    contact_rows = []
    for patient in sorted(source.subject.astype(str).unique()):
        path = DATASET / f"{patient}.npz"
        with np.load(path, allow_pickle=False) as data:
            names = np.asarray(data["contact_names"]).astype(str)
        for index, contact in enumerate(names):
            contact_rows.append(
                {
                    "patient_id": patient,
                    "contact_index": index,
                    "model_contact": contact,
                }
            )
    pd.DataFrame(contact_rows).to_csv(
        OUT / "model_contact_inventory.csv", index=False
    )

    consensus = registry.consensus_status.astype(str)
    exact = registry.exact_contact_join_status.astype(str)
    eligible = (consensus == "CONSENSUS_EXACT") & (exact == "EXACT_JOINED")
    patients_eligible = registry.loc[eligible, "patient_id"].nunique()
    status = {
        "contract": "topic5_clinical_onset_source_annotation",
        "version": "0.1",
        "status": (
            "READY_FOR_TRANSFER"
            if bool(eligible.any())
            else "AWAITING_BLINDED_MANUAL_ANNOTATION"
        ),
        "patients_in_registry": int(registry.patient_id.nunique()),
        "seizures_in_registry": int(len(registry)),
        "consensus_exact_seizures": int(eligible.sum()),
        "consensus_exact_patients": int(patients_eligible),
        "early_ictal_transfer_metadata_ready": bool(eligible.any()),
        "source_substitutions_allowed": False,
        "allowed_primary_source": (
            "double-reviewed or expert-adjudicated exact per-seizure "
            "clinical-onset contacts"
        ),
        "target_values_read": False,
        "source_inventory_sha256": sha256(SOURCE),
        "registry_sha256": sha256(REGISTRY),
    }
    atomic_json(OUT / "READINESS_STATUS.json", status)
    atomic_json(
        OUT / "BLINDING_CONTRACT.json",
        {
            "status": "LOCKED",
            "reviewers_must_not_view": [
                "RNN scores",
                "Markov scores",
                "transition decomposition scores",
                "early-ictal energy values",
            ],
            "forbidden_substitutions": [
                "SOZ",
                "patient-level focus",
                "A/B template source",
                "energy-top contacts",
            ],
            "primary_acceptance": (
                "reviewer agreement or expert adjudication plus exact contact join"
            ),
            "ambiguous_cases": "exclude with explicit reason",
            "target_values_read": False,
        },
    )
    (OUT / "README.md").write_text(
        "# Clinical-onset source annotation v0.1\n\n"
        "本目录只准备逐发作 clinical-onset contact 的盲法人工标注表，不包含也不读取"
        " early-ictal energy 数值。`annotation_registry.csv` 已预填 13 人 71 次发作"
        "的 patient/seizure/time metadata，source contacts、两位 reviewer 和"
        " consensus 仍为空。\n\n"
        "只有 `consensus_status=CONSENSUS_EXACT` 且"
        " `exact_contact_join_status=EXACT_JOINED` 的 seizure 才能进入 primary "
        "transfer。SOZ、患者级 focus、A/B source 和 energy-top contacts 均不能"
        "补位。\n",
        encoding="utf-8",
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

