#!/usr/bin/env python3
"""Target-blind temporal audit for ordered-history RNN controls.

This audit distinguishes two different recurrent axes:

1. within-event recruitment rank steps (the frozen primary RNN contract);
2. across-event histories before a clinical-onset seizure (an exploratory
   extension that is only admissible when distinct causal histories exist).

The script reads target metadata and seizure times, but never deserializes an
early-ictal target array.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "results/topic5_interictal_rank_distribution/dataset_v0_4"
FIT1 = (
    ROOT
    / "results/topic5_state_conditioned_predictor/fit12_clinical_bb150/"
    "fit1/fig6_fit1_clinical_onset_scaffold_event.csv"
)
SEIZURES = ROOT / "results/epilepsiae_seizure_inventory.csv"
OUT = ROOT / "results/topic5_ordered_history_architecture_audit/input_audit"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_inventory() -> pd.DataFrame:
    frame = pd.read_csv(FIT1)
    strict = (
        frame.loc[
            (frame.group_id == "strict_broadband")
            & (frame.time_reference == "clinical_onset"),
            ["subject", "seizure_idx"],
        ]
        .drop_duplicates()
        .sort_values(["subject", "seizure_idx"])
        .reset_index(drop=True)
    )
    if strict.subject.nunique() != 16 or len(strict) != 106:
        raise RuntimeError("strict clinical-onset denominator drifted")
    return strict


def onset_table() -> dict[str, pd.DataFrame]:
    frame = pd.read_csv(SEIZURES, dtype={"subject": str})
    frame = frame.loc[frame.clin_onset_epoch.notna()].copy()
    return {
        f"epilepsiae_{subject}": group.sort_values(
            "clin_onset_epoch"
        ).reset_index(drop=True)
        for subject, group in frame.groupby("subject")
    }


def main() -> None:
    strict = strict_inventory()
    by_subject_onset = onset_table()
    rows: list[dict] = []
    subject_audit_path = DATASET / "subject_audit.csv"
    dataset_manifest_path = DATASET / "dataset_manifest.json"
    subject_audit = pd.read_csv(subject_audit_path)
    frozen_subjects = sorted(
        subject_audit.loc[
            subject_audit.status.astype(str).eq("ok"), "subject"
        ].astype(str)
    )
    if len(frozen_subjects) != 34:
        raise RuntimeError(
            f"frozen interictal cohort denominator drifted: {len(frozen_subjects)}/34"
        )
    dataset_hashes = {
        subject: sha256(DATASET / "per_subject" / f"{subject}.npz")
        for subject in frozen_subjects
    }
    for subject, subject_targets in strict.groupby("subject", sort=True):
        path = DATASET / "per_subject" / f"{subject}.npz"
        with np.load(path, allow_pickle=False) as data:
            event_time = np.asarray(data["event_abs_time"], dtype=np.float64)
            event_split = np.asarray(data["event_split"], dtype=np.uint8)
        if np.any(~np.isfinite(event_time)):
            raise RuntimeError(f"{subject}: non-finite event timestamp")
        if np.any(np.diff(event_time) < 0):
            raise RuntimeError(f"{subject}: event timestamps are not chronological")
        onset = by_subject_onset[subject]
        for target in subject_targets.itertuples(index=False):
            seizure_index = int(target.seizure_idx)
            if not 0 <= seizure_index < len(onset):
                raise RuntimeError(
                    f"{subject}: seizure_idx={seizure_index} is outside "
                    f"the clinical-onset inventory (n={len(onset)})"
                )
            seizure = onset.iloc[seizure_index]
            onset_epoch = float(seizure.clin_onset_epoch)
            preceding = np.flatnonzero(event_time < onset_epoch)
            last_index = int(preceding[-1]) if preceding.size else -1
            last_gap_hours = (
                float((onset_epoch - event_time[last_index]) / 3600.0)
                if last_index >= 0
                else np.nan
            )
            row = {
                "subject": subject,
                "seizure_idx": seizure_index,
                "seizure_id": str(seizure.seizure_id),
                "clinical_onset_epoch": onset_epoch,
                "n_interictal_events_total": int(event_time.size),
                "n_train80_events_total": int(np.count_nonzero(event_split == 0)),
                "n_heldout20_events_total": int(
                    np.count_nonzero(event_split == 1)
                ),
                "n_causal_events_available": int(preceding.size),
                "last_causal_event_index": last_index,
                "last_event_gap_hours": last_gap_hours,
            }
            for hours in (1, 3, 6, 12, 24, 48, 72):
                row[f"n_events_prior_{hours}h"] = int(
                    np.count_nonzero(
                        (event_time < onset_epoch)
                        & (event_time >= onset_epoch - hours * 3600.0)
                    )
                )
            for length in (32, 64, 128, 256):
                selected = preceding[-length:]
                row[f"history_{length}_available"] = bool(
                    selected.size == length
                )
                row[f"history_{length}_span_hours"] = (
                    float((event_time[selected[-1]] - event_time[selected[0]]) / 3600)
                    if selected.size >= 2
                    else np.nan
                )
            rows.append(row)

    event_frame = pd.DataFrame(rows)
    if len(event_frame) != 106:
        raise RuntimeError("temporal audit output denominator drifted")
    subject_rows = []
    for subject, group in event_frame.groupby("subject", sort=True):
        n_unique = int(group.last_causal_event_index.nunique())
        subject_rows.append(
            {
                "subject": subject,
                "n_strict_seizures": int(len(group)),
                "n_distinct_causal_histories": n_unique,
                "distinct_history_fraction": float(n_unique / len(group)),
                "median_last_event_gap_hours": float(
                    group.last_event_gap_hours.median()
                ),
                "n_last_event_within_6h": int(
                    np.count_nonzero(group.last_event_gap_hours <= 6.0)
                ),
                "n_with_at_least_32_events_prior_6h": int(
                    np.count_nonzero(group.n_events_prior_6h >= 32)
                ),
                "circular_shift_min3_distinct_histories": bool(n_unique >= 3),
            }
        )
    subject_frame = pd.DataFrame(subject_rows)

    OUT.mkdir(parents=True, exist_ok=True)
    event_frame.to_csv(OUT / "seizure_history_availability.csv", index=False)
    subject_frame.to_csv(OUT / "subject_history_independence.csv", index=False)
    summary = {
        "contract": "topic5_ordered_history_architecture_audit_v0_1",
        "status": "TEMPORAL_SEMANTICS_AND_PAIRING_AUDITED",
        "target_values_read": False,
        "target_arrays_deserialized": False,
        "primary_recurrent_axis": {
            "unit": "rank step within one interictal group event",
            "state_reset": "at every group-event boundary",
            "real_time_constant_claim_allowed": False,
        },
        "strict_clinical_onset_metadata": {
            "n_patients": int(event_frame.subject.nunique()),
            "n_seizures": int(len(event_frame)),
            "n_distinct_causal_histories": int(
                event_frame.groupby("subject").last_causal_event_index.nunique().sum()
            ),
            "n_patients_with_at_least_3_distinct_histories": int(
                subject_frame.circular_shift_min3_distinct_histories.sum()
            ),
            "n_seizures_last_event_within_6h": int(
                np.count_nonzero(event_frame.last_event_gap_hours <= 6.0)
            ),
            "n_seizures_with_at_least_32_events_prior_6h": int(
                np.count_nonzero(event_frame.n_events_prior_6h >= 32)
            ),
            "median_last_event_gap_hours": float(
                event_frame.last_event_gap_hours.median()
            ),
            "max_last_event_gap_hours": float(
                event_frame.last_event_gap_hours.max()
            ),
        },
        "decision": {
            "within_event_architecture_and_information_controls": "PRIMARY_RUN",
            "across_event_history_to_seizure_target": "EXPLORATORY_ONLY",
            "reason": (
                "seizures sharing the same last available interictal event have "
                "identical causal histories; seizure rows are therefore not "
                "independent history samples"
            ),
            "across_event_statistical_unit": (
                "distinct patient-specific causal history, never raw seizure row"
            ),
            "iei_as_predictor": "FORBIDDEN",
            "timestamps_use": "eligibility and causal pairing only",
        },
        "input_hashes": {
            "fit1_metadata_csv": sha256(FIT1),
            "seizure_inventory_csv": sha256(SEIZURES),
            "rank_dataset_manifest": sha256(dataset_manifest_path),
            "rank_subject_audit": sha256(subject_audit_path),
            "rank_dataset_npz": dataset_hashes,
        },
    }
    (OUT / "PAIRING_AUDIT.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
