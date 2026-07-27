#!/usr/bin/env python3
"""Audit v2.2 early-ictal metadata while keeping all target values sealed.

Only routing columns, seizure timestamps/focus metadata, cache JSON sidecars,
and contact names are read. No NPZ target array or observed/null score column
is opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable

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


def _normalize_contact(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def exact_focus_contacts(focus: Any, contact_names: Iterable[str]) -> list[str]:
    """Return only contact names literally present in the per-seizure focus field."""
    if focus is None or (isinstance(focus, float) and np.isnan(focus)):
        return []
    text = str(focus).strip()
    if not text:
        return []
    tokens = {
        _normalize_contact(token)
        for token in re.split(r"[,;/+|\\s]+", text)
        if token.strip()
    }
    return [
        name
        for name in contact_names
        if _normalize_contact(name) in tokens
    ]


def _load_routing(path: Path, cfg: dict[str, Any]) -> pd.DataFrame:
    allowed = [
        "dataset",
        "subject",
        "seizure_idx",
        "group_id",
        "time_reference",
        "window_start_sec",
        "window_end_sec",
        "band",
        "field_plane",
        "n_finite_contacts",
    ]
    frame = pd.read_csv(path, usecols=allowed)
    target = cfg["target_metadata"]
    frame = frame[
        frame["group_id"].astype(str).eq(str(target["group_id"]))
        & frame["time_reference"].astype(str).eq(str(target["time_reference"]))
        & frame["band"].astype(str).eq(str(target["band"]))
    ].copy()
    return frame.sort_values(["subject", "seizure_idx"]).reset_index(drop=True)


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
    output_root = ROOT / cfg["outputs"]["root"]
    audit_dir = output_root / "target_audit"
    input_dir = output_root / "input_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    input_gate = json.loads(
        (input_dir / "INPUT_AUDIT_GATE.json").read_text(encoding="utf-8")
    )
    if input_gate.get("status") != "pass":
        raise RuntimeError("input audit must pass before target metadata audit")
    physical = set(
        json.loads(
            (input_dir / "physical_axis_formal_cohort.json").read_text(
                encoding="utf-8"
            )
        )["subjects"]
    )
    development = set(map(str, cfg["cohort"]["development"]))
    rank_root = ROOT / cfg["inputs"]["rank_dataset"]
    target_meta_root = ROOT / cfg["inputs"]["target_metadata_root"]
    routing_path = ROOT / cfg["inputs"]["target_routing_table"]
    routing = _load_routing(routing_path, cfg)
    inventory = pd.read_csv(ROOT / cfg["inputs"]["epilepsiae_seizure_inventory"])
    inventory["subject"] = inventory["subject"].astype(str)

    cache_records: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for event in routing.itertuples(index=False):
        subject = str(event.subject)
        short = subject.split("_", 1)[1]
        rank_npz = rank_root / "per_subject" / f"{subject}.npz"
        with np.load(rank_npz, allow_pickle=False) as data:
            model_contacts = [str(value) for value in data["contact_names"]]
        cache_json = target_meta_root / f"{subject}.json"
        cache = cache_records.get(subject)
        if cache is None:
            cache = (
                json.loads(cache_json.read_text(encoding="utf-8"))
                if cache_json.is_file()
                else {}
            )
            cache_records[subject] = cache
        target_contacts = [str(value) for value in cache.get("channels") or []]
        eligible_idxs = {int(value) for value in cache.get("eligible_idxs") or []}
        joined = sorted(set(model_contacts) & set(target_contacts))

        subject_inventory = inventory[inventory.subject.eq(short)].reset_index(
            drop=True
        )
        seizure_idx = int(event.seizure_idx)
        seizure_row = (
            subject_inventory.iloc[seizure_idx]
            if 0 <= seizure_idx < len(subject_inventory)
            else None
        )
        clinical_onset = (
            float(seizure_row["clin_onset_epoch"])
            if seizure_row is not None
            and pd.notna(seizure_row.get("clin_onset_epoch"))
            else np.nan
        )
        focus = seizure_row.get("focus") if seizure_row is not None else None
        source_contacts = exact_focus_contacts(focus, model_contacts)
        source_available = bool(source_contacts)
        non_source_joined = len(set(joined) - set(source_contacts))
        feature = str(cache.get("feature", ""))
        window = list(map(float, cfg["target_metadata"]["window_seconds"]))
        energy_metadata_available = bool(
            cache_json.is_file()
            and seizure_idx in eligible_idxs
            and "1-150" in feature
            and "0_10" in feature
        )
        structural_eligible = bool(
            subject in physical
            and subject not in development
            and np.isfinite(clinical_onset)
            and source_available
            and energy_metadata_available
            and non_source_joined
            >= int(cfg["target_metadata"]["min_non_source_contacts"])
        )
        reasons: list[str] = []
        if subject not in physical:
            reasons.append("not_development_excluded_physical_axis")
        if subject in development:
            reasons.append("development_subject_supportive_only")
        if not np.isfinite(clinical_onset):
            reasons.append("missing_clinical_onset_anchor")
        if not source_available:
            reasons.append("missing_exact_per_seizure_clinical_onset_contact_set")
        if not energy_metadata_available:
            reasons.append("missing_frozen_1_150hz_0_10s_energy_metadata")
        if non_source_joined < int(cfg["target_metadata"]["min_non_source_contacts"]):
            reasons.append("too_few_non_source_joined_contacts")
        rows.append(
            {
                "dataset": str(event.dataset),
                "subject": subject,
                "seizure_idx": seizure_idx,
                "seizure_id": (
                    str(seizure_row.get("seizure_id"))
                    if seizure_row is not None
                    else ""
                ),
                "clinical_onset_anchor_available": bool(
                    np.isfinite(clinical_onset)
                ),
                "clinical_onset_epoch_metadata": (
                    clinical_onset if np.isfinite(clinical_onset) else np.nan
                ),
                "clinical_onset_source_set_available": source_available,
                "clinical_onset_source_contacts": "|".join(source_contacts),
                "source_contact_policy": str(
                    cfg["target_metadata"]["source_policy"]
                ),
                "energy_metadata_available": energy_metadata_available,
                "energy_metadata_json": str(cache_json.relative_to(ROOT))
                if cache_json.is_file()
                else "",
                "energy_metadata_sha256": sha256(cache_json)
                if cache_json.is_file()
                else "",
                "window_start_sec": float(event.window_start_sec),
                "window_end_sec": float(event.window_end_sec),
                "expected_window_start_sec": window[0],
                "expected_window_end_sec": window[1],
                "band": str(event.band),
                "time_reference": str(event.time_reference),
                "n_model_contacts": len(model_contacts),
                "n_target_metadata_contacts": len(target_contacts),
                "n_exact_joined_contacts": len(joined),
                "all_model_contacts_joined": set(model_contacts).issubset(
                    set(target_contacts)
                ),
                "n_source_contacts": len(source_contacts),
                "n_non_source_joined_contacts": non_source_joined,
                "development_subject": subject in development,
                "physical_axis_formal_subject": subject in physical,
                "structural_transfer_eligible": structural_eligible,
                "structural_exclusion_reasons": "|".join(reasons),
                "dynamic_recruitment_rank_metadata_available": False,
                "target_values_read": False,
            }
        )

    seizures = pd.DataFrame(rows)
    seizures.to_csv(audit_dir / "seizure_inventory.csv", index=False)
    patient_rows: list[dict[str, Any]] = []
    for subject, group in seizures.groupby("subject", sort=True):
        patient_rows.append(
            {
                "dataset": str(group.dataset.iloc[0]),
                "subject": subject,
                "n_routed_seizures": len(group),
                "n_energy_metadata_available": int(
                    group.energy_metadata_available.sum()
                ),
                "n_source_set_available": int(
                    group.clinical_onset_source_set_available.sum()
                ),
                "n_structural_transfer_eligible": int(
                    group.structural_transfer_eligible.sum()
                ),
                "development_subject": bool(group.development_subject.iloc[0]),
                "physical_axis_formal_subject": bool(
                    group.physical_axis_formal_subject.iloc[0]
                ),
                "primary_structural_patient_eligible": bool(
                    group.structural_transfer_eligible.sum()
                    >= int(cfg["target_metadata"]["min_seizures_primary"])
                ),
                "target_values_read": False,
            }
        )
    patients = pd.DataFrame(patient_rows)
    patients.to_csv(audit_dir / "patient_inventory.csv", index=False)

    endpoint_denominators = {
        "routing": {
            "patients": int(seizures.subject.nunique()),
            "seizures": int(len(seizures)),
        },
        "energy_metadata": {
            "patients": int(
                seizures.loc[
                    seizures.energy_metadata_available, "subject"
                ].nunique()
            ),
            "seizures": int(seizures.energy_metadata_available.sum()),
        },
        "exact_clinical_onset_source_set": {
            "patients": int(
                seizures.loc[
                    seizures.clinical_onset_source_set_available, "subject"
                ].nunique()
            ),
            "seizures": int(
                seizures.clinical_onset_source_set_available.sum()
            ),
        },
        "primary_structural_transfer": {
            "patients": int(
                patients.primary_structural_patient_eligible.sum()
            ),
            "seizures": int(seizures.structural_transfer_eligible.sum()),
        },
        "dynamic_recruitment_rank": {"patients": 0, "seizures": 0},
        "analysis_denominator": "SEALED_UNTIL_INTERICTAL_UNLOCK",
        "target_values_read": False,
    }
    atomic_json(audit_dir / "endpoint_denominators.json", endpoint_denominators)

    source_ready = (
        endpoint_denominators["exact_clinical_onset_source_set"]["seizures"] > 0
    )
    primary_ready = (
        endpoint_denominators["primary_structural_transfer"]["patients"] > 0
    )
    gate = {
        "contract": cfg["contract"]["name"],
        "version": cfg["contract"]["version"],
        "status": "pass" if primary_ready else "complete_with_blocker",
        "routing_metadata_audited": True,
        "energy_values_read": False,
        "recruitment_values_read": False,
        "source_contact_metadata_ready": source_ready,
        "primary_transfer_metadata_ready": primary_ready,
        "blocker": (
            ""
            if primary_ready
            else "no exact per-seizure clinical-onset contact set is available; "
            "SOZ, patient-level focus, A/B source, and energy-top contacts were "
            "not substituted"
        ),
        "interictal_model_execution_allowed": True,
        "early_ictal_transfer_allowed": False,
        "endpoint_denominators": endpoint_denominators,
    }
    atomic_json(audit_dir / "TARGET_METADATA_GATE.json", gate)
    atomic_json(
        audit_dir / "TARGET_VALUES_SEALED.json",
        {
            "energy_values_read": False,
            "recruitment_values_read": False,
            "sealed": True,
            "unlock_requires": [
                "claim2_next_PASS",
                "claim2_future_PASS",
                "claim3_random_axis_PASS",
                "claim4_shared_scaffold_PASS",
                "primary_transfer_metadata_ready",
            ],
        },
    )
    print(json.dumps(gate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
