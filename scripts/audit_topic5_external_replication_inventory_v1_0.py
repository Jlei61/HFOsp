#!/usr/bin/env python3
"""Metadata-only inventory for an independent contact-topography replication cohort."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TARGET_CACHE = ROOT / "results/topic5_ictal_recruitment/t0_feature_cache_bb150_1_150"
EVENT_TABLE = (
    ROOT
    / "results/topic5_state_conditioned_predictor/fit12_clinical_bb150/"
    "fit1/fig6_fit1_clinical_onset_scaffold_event.csv"
)
OUT = ROOT / "results/topic5_static_scaffold_fixed_readout_validation"


def main() -> None:
    frame = pd.read_csv(
        EVENT_TABLE,
        usecols=[
            "dataset",
            "subject",
            "seizure_idx",
            "group_id",
            "time_reference",
        ],
    )
    cached = sorted(path.stem for path in TARGET_CACHE.glob("*.npz"))
    strict = frame.loc[
        (frame["group_id"] == "strict_broadband")
        & (frame["time_reference"] == "clinical_onset")
    ].copy()
    used_subjects = sorted(strict["subject"].astype(str).unique().tolist())
    rows = []
    for subject in cached:
        subject_rows = frame.loc[frame["subject"].astype(str) == subject]
        strict_rows = strict.loc[strict["subject"].astype(str) == subject]
        references = sorted(
            subject_rows["time_reference"].dropna().astype(str).unique().tolist()
        )
        if subject in used_subjects:
            status = "CURRENT_TARGET_ALREADY_READ"
            reason = "member of the frozen 16-patient clinical-onset cohort"
        elif not strict_rows.empty:
            status = "POTENTIAL_INDEPENDENT"
            reason = "strict clinical-onset rows exist outside the frozen cohort"
        elif references:
            status = "INELIGIBLE_CURRENT_CONTRACT"
            reason = "no strict clinical-onset rows; available reference=" + ",".join(
                references
            )
        else:
            status = "INELIGIBLE_CURRENT_CONTRACT"
            reason = "target cache exists but no eligible event-table rows"
        rows.append(
            {
                "subject": subject,
                "dataset": subject.split("_", 1)[0],
                "n_strict_clinical_onset_seizures": int(
                    strict_rows["seizure_idx"].nunique()
                ),
                "time_references_in_event_table": ";".join(references),
                "replication_status": status,
                "reason": reason,
            }
        )

    inventory = pd.DataFrame(rows).sort_values("subject")
    potential = inventory.loc[
        inventory["replication_status"] == "POTENTIAL_INDEPENDENT"
    ]
    summary = {
        "contract": "topic5_external_clinical_onset_replication_protocol_v1_0",
        "status": (
            "READY_TO_REPLICATE"
            if len(potential)
            else "READY_BUT_BLOCKED_NO_INDEPENDENT_PATIENT_COHORT"
        ),
        "metadata_only": True,
        "target_values_read": False,
        "n_cached_subjects": int(len(inventory)),
        "n_current_target_subjects": int(
            (inventory.replication_status == "CURRENT_TARGET_ALREADY_READ").sum()
        ),
        "n_potential_independent_subjects": int(len(potential)),
        "n_potential_independent_seizures": int(
            potential.n_strict_clinical_onset_seizures.sum()
        ),
        "current_target": {
            "n_patients": int(strict["subject"].nunique()),
            "n_seizures": int(strict["seizure_idx"].nunique())
            if strict["subject"].nunique() == 1
            else int(len(strict)),
            "time_reference": "clinical_onset",
            "window_sec": [0, 10],
            "band_hz": [1, 150],
        },
        "independent_unit": "previously target-unread patient",
        "current_patient_new_seizures": (
            "within-patient stability only; not patient-level external replication"
        ),
        "frozen_pipeline_not_old_patient_weights": {
            "frozen": [
                "model architecture and hidden size",
                "loss and seeds",
                "chronological split",
                "field definition",
                "regularizer candidates and validation selection rule",
                "target window and band",
                "nulls, statistics, exclusions and endpoint hierarchy",
            ],
            "allowed_on_new_patient_interictal_data_only": [
                "raw participation estimation",
                "target-free regularized field selection and refit",
                "patient-specific GRU and contact calibration",
            ],
        },
        "replication_endpoints": [
            "orientation-free within-shaft morphology margin",
            "signed direction and sign heterogeneity",
            "GRU increment over best target-free static and rank-shuffle controls",
        ],
        "equivalence_margin_static": 0.05,
        "acquisition_routes": [
            {
                "priority": 1,
                "route": (
                    "complete exact clinical-onset annotation, homogeneous "
                    "target and contact join for currently unused cached patients"
                ),
            },
            {
                "priority": 2,
                "route": (
                    "add previously unanalyzed Epilepsiae or hospital "
                    "clinical-onset patients"
                ),
            },
            {
                "priority": 3,
                "route": (
                    "prospectively establish seizure-level onset annotation, "
                    "target production and interictal event processing"
                ),
            },
        ],
        "decision": (
            "Do not reuse the current target as replication. Wait for a new "
            "exact clinical-onset patient cohort and apply the frozen workflow "
            "without ictal-target-guided fitting."
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(OUT / "replication_inventory.csv", index=False)
    (OUT / "REPLICATION_INVENTORY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
