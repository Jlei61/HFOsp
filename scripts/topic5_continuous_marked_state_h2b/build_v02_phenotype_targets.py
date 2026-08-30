#!/usr/bin/env python3
"""Join pre-existing frozen seizure subtypes to H2b v0.2 case states.

No subtype is recomputed.  Existing ictal recruitment caches are inventoried
but not converted into a new scalar target after seeing H2b states.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch as _torch  # noqa: F401
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.topic5_continuous_marked_state_h2b.contract import (  # noqa: E402
    V0_2_RESULT_ROOT,
    atomic_csv,
    atomic_json,
    sha256_file,
    utc_now,
)


SOURCE = Path("/home/honglab/leijiaxin/HFOsp")
SUBTYPE_ROOT = SOURCE / (
    "results/data_driven_soz/layer_a_ictal_er_rank/seizure_clusters/per_subject"
)
RECRUITMENT_ROOT = SOURCE / (
    "results/topic5_ictal_recruitment/ictal_field_long_cache"
)
TARGETS = (
    ("frozen_subtype_broad_ER", "broad_ER"),
    ("frozen_subtype_gamma_ER", "gamma_ER"),
)


def _subtype_map(path: Path, band: str) -> tuple[dict[str, int], dict]:
    if not path.is_file():
        return {}, {"status": "MISSING_PREEXISTING_TARGET"}
    payload = json.loads(path.read_text())
    value = (payload.get("per_band") or {}).get(band) or {}
    if value.get("status") != "ok":
        return {}, {"status": str(value.get("status", "UNAVAILABLE"))}
    ids = list(map(str, value.get("seizure_ids_kept") or []))
    labels = value.get("subtype_label") or []
    outliers = value.get("outlier_flag") or [False] * len(ids)
    if not (len(ids) == len(labels) == len(outliers)):
        raise ValueError(f"{path}: {band} subtype arrays disagree")
    mapping = {
        seizure_id: int(label)
        for seizure_id, label, outlier in zip(ids, labels, outliers)
        if not bool(outlier) and int(label) >= 0
    }
    return mapping, {
        "status": "AVAILABLE",
        "n_ids_kept": len(ids),
        "n_nonoutlier_labels": len(mapping),
        "n_classes": len(set(mapping.values())),
    }


def run(root: Path) -> dict:
    root = root.resolve()
    rows = []
    availability = []
    subject_tables = {}
    subject_roots = sorted(
        path.parent for path in (root / "risk_sets").glob(
            "*/primary_risk_sets.csv"
        )
    )
    for subject_root in subject_roots:
        subject = subject_root.name
        subject_row_start = len(rows)
        # Frozen subtype transfer asks whether the primary state adds beyond
        # the current observation.  It must not require a usable wrong-time
        # donor, which is a separate H1 time-specificity diagnostic.
        risk_path = subject_root / "primary_risk_sets.csv"
        risk = pd.read_csv(risk_path, dtype={
            "patient_id": str, "seizure_id": str, "split": str,
            "evaluation_tier": str,
        })
        cases = risk[
            risk["is_case"].astype(str).str.lower().eq("true")
            & (risk["lead_minutes"].astype(int) == 30)
        ].copy()
        subtype_path = SUBTYPE_ROOT / f"{subject}__zer_binned.json"
        recruitment_path = RECRUITMENT_ROOT / f"{subject}.json"
        maps = {}
        for target_name, band in TARGETS:
            mapping, status = _subtype_map(subtype_path, band)
            maps[target_name] = mapping
            availability.append({
                "subject": subject,
                "target_name": target_name,
                "target_kind": "classification",
                "target_source_path": str(subtype_path) if subtype_path.is_file() else None,
                "target_source_sha256": (
                    sha256_file(subtype_path) if subtype_path.is_file() else None
                ),
                "n_primary_case_seizures": int(cases.seizure_id.nunique()),
                "n_primary_case_seizures_with_target": int(
                    cases[cases.seizure_id.astype(str).isin(mapping)].seizure_id.nunique()
                ),
                **status,
                "target_reclustered": False,
            })
        availability.append({
            "subject": subject,
            "target_name": "blind_early_recruitment_extent",
            "target_kind": "continuous",
            "target_source_path": (
                str(recruitment_path) if recruitment_path.is_file() else None
            ),
            "target_source_sha256": (
                sha256_file(recruitment_path) if recruitment_path.is_file() else None
            ),
            "n_primary_case_seizures": int(cases.seizure_id.nunique()),
            "n_primary_case_seizures_with_target": 0,
            "status": (
                "NOT_ESTIMABLE_EXISTING_CACHE_IS_NOT_A_FROZEN_SCALAR_TARGET"
                if recruitment_path.is_file() else "MISSING_PREEXISTING_TARGET"
            ),
            "target_reclustered": False,
        })

        feature_columns = [
            name for name in cases.columns
            if name.startswith(("history__", "observation__", "state__", "wrong_time__"))
        ]
        for case in cases.itertuples(index=False):
            base = {
                "patient_id": subject,
                "seed": int(case.seed),
                "seizure_id": str(case.seizure_id),
                "split": str(case.split),
                "evaluation_tier": str(case.evaluation_tier),
                "target_frozen": True,
            }
            # itertuples(index=False) has no stable original index; select the
            # unique patient/seed/seizure case row explicitly.
            selected = cases[
                (cases.seed.astype(int) == int(case.seed))
                & (cases.seizure_id.astype(str) == str(case.seizure_id))
            ]
            if len(selected) != 1:
                raise ValueError(f"{subject}: duplicate 30-min case row")
            selected = selected.iloc[0]
            for name in feature_columns:
                output_name = (
                    f"baseline__{name.removeprefix('history__')}"
                    if name.startswith("history__") else name
                )
                base[output_name] = selected[name]
            for target_name, band in TARGETS:
                mapping = maps[target_name]
                available = str(case.seizure_id) in mapping
                rows.append({
                    **base,
                    "target_name": target_name,
                    "target_kind": "classification",
                    "target_value": (
                        float(mapping[str(case.seizure_id)]) if available else None
                    ),
                    "target_provenance": (
                        f"preexisting_topic5_zER_{band}_subtype_nonoutlier"
                        if available else "preexisting_target_unavailable_for_seizure"
                    ),
                    "target_source_sha256": (
                        sha256_file(subtype_path) if subtype_path.is_file() else None
                    ),
                })
            rows.append({
                **base,
                "target_name": "blind_early_recruitment_extent",
                "target_kind": "continuous",
                "target_value": None,
                "target_provenance": (
                    "no_preexisting_blind_seizure_level_scalar_target"
                ),
                "target_source_sha256": (
                    sha256_file(recruitment_path) if recruitment_path.is_file() else None
                ),
            })

        subject_rows = rows[subject_row_start:]
        if subject_rows:
            preferred = [
                "patient_id", "seed", "seizure_id", "split", "evaluation_tier",
                "target_name", "target_kind", "target_value", "target_frozen",
                "target_provenance", "target_source_sha256",
            ]
            feature_names = sorted({
                name for row in subject_rows for name in row if name not in preferred
            })
            table = subject_root / "frozen_phenotype_targets.csv"
            atomic_csv(
                table, subject_rows,
                fieldnames=[*preferred, *feature_names],
            )
            subject_tables[subject] = {
                "path": str(table),
                "sha256": sha256_file(table),
                "n_rows": len(subject_rows),
                "n_available_target_rows": sum(
                    row["target_value"] is not None for row in subject_rows
                ),
            }

    availability_path = root / "reports/phenotype_target_availability.csv"
    availability_fields = sorted({
        name for row in availability for name in row
    }) if availability else ["subject", "target_name", "status"]
    atomic_csv(
        availability_path, availability,
        fieldnames=availability_fields,
    )
    payload = {
        "status": "COMPLETE",
        "revision": "h2b_v0_2_preexisting_phenotype_target_join_v1",
        "created_utc": utc_now(),
        "n_subjects_with_primary_risk_sets": len(subject_roots),
        "matched_wrong_time_is_not_a_phenotype_gate": True,
        "n_target_rows": len(rows),
        "n_available_target_rows": sum(row["target_value"] is not None for row in rows),
        "subject_tables": subject_tables,
        "heterogeneous_patient_feature_dimensions_never_pooled": True,
        "target_reclustered": False,
        "early_recruitment_scalar_derived_here": False,
        "replacement_target_invented": False,
        "availability_path": str(availability_path),
        "availability_sha256": sha256_file(availability_path),
        "formal_test_partition_opened": False,
        "sealed_opened": False,
    }
    atomic_json(root / "reports/phenotype_target_availability.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=V0_2_RESULT_ROOT)
    args = parser.parse_args()
    print(json.dumps(run(args.result_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
