#!/usr/bin/env python3
"""Metadata-only input audit for fixed-readout static scaffold validation."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "results/topic5_interictal_rank_distribution/dataset_v0_4"
RUNS = (
    ROOT
    / "results/topic5_interictal_rank_distribution/runs/"
    "formal_multiseed_20260725_v1"
)
TARGET = ROOT / "results/topic5_ictal_recruitment/t0_feature_cache_bb150_1_150"
FIT1 = (
    ROOT
    / "results/topic5_state_conditioned_predictor/fit12_clinical_bb150/"
    "fit1/fig6_fit1_clinical_onset_scaffold_event.csv"
)
OUT = ROOT / "results/topic5_static_scaffold_fixed_readout_validation"
SEED_DIRS = ("seed_20260725", "seed_20260726", "seed_20260727")
MODELS = (
    "empirical_rank_distribution",
    "full_history_gru",
    "static_contact_hazard",
    "unordered_prefix",
    "last_set_first_order",
    "rank_shuffle_gru",
)


def shaft(name: str) -> str:
    value = re.sub(r"\d+$", "", str(name)).strip("_- ")
    return value or str(name)


def strict_inventory() -> dict[str, list[int]]:
    frame = pd.read_csv(FIT1)
    strict = frame.loc[
        (frame.group_id == "strict_broadband")
        & (frame.time_reference == "clinical_onset")
    ]
    result = {
        str(subject): sorted(group.seizure_idx.astype(int).unique().tolist())
        for subject, group in strict.groupby("subject")
    }
    if len(result) != 16 or sum(map(len, result.values())) != 106:
        raise RuntimeError("strict clinical-onset denominator drifted")
    return result


def main() -> None:
    inventory = strict_inventory()
    soz_path = ROOT / "results/epilepsiae_soz_core_channels.json"
    soz = json.loads(soz_path.read_text()) if soz_path.exists() else {}
    rows = []
    for subject, seizures in inventory.items():
        dataset_path = DATASET / "per_subject" / f"{subject}.npz"
        with np.load(dataset_path, allow_pickle=False) as data:
            names = np.asarray(data["contact_names"]).astype(str)
            coords = np.asarray(data["contact_coords"], dtype=np.float64)
            n_train = int(np.count_nonzero(data["event_split"] == 0))
            n_heldout = int(np.count_nonzero(data["event_split"] == 1))
            feature_names = set(
                np.asarray(data["contact_feature_names"]).astype(str).tolist()
            )
        with np.load(TARGET / f"{subject}.npz", allow_pickle=False) as data:
            target_names = np.asarray(data["channels"]).astype(str)
            available_keys = set(data.files)
        target_set = set(target_names.tolist())
        joined = np.asarray([name in target_set for name in names], dtype=bool)
        groups: dict[str, list[int]] = {}
        for index, name in enumerate(names[joined]):
            groups.setdefault(shaft(name), []).append(index)
        shufflable = sum(len(group) for group in groups.values() if len(group) >= 2)
        size_counts: dict[int, int] = {}
        for group in groups.values():
            size_counts[len(group)] = size_counts.get(len(group), 0) + 1
        exchangeable = sum(
            size * count
            for size, count in size_counts.items()
            if count >= 2
        )
        geometry = np.all(np.isfinite(coords[joined]), axis=1)
        missing_model_cells = []
        for seed_dir in SEED_DIRS:
            path = RUNS / seed_dir / subject / "contact_rank_distributions.csv"
            if not path.exists():
                missing_model_cells.append(f"{seed_dir}:missing_csv")
                continue
            present = set(pd.read_csv(path, usecols=["control"]).control.unique())
            for model in MODELS:
                if model not in present:
                    missing_model_cells.append(f"{seed_dir}:{model}")
        subject_key = subject.replace("epilepsiae_", "")
        rows.append(
            {
                "subject": subject,
                "n_seizures": len(seizures),
                "n_model_contacts": len(names),
                "n_exact_joined_contacts": int(np.count_nonzero(joined)),
                "n_train80_events": n_train,
                "n_heldout20_events": n_heldout,
                "n_shafts_joined": len(groups),
                "within_shaft_circular_eligible": bool(
                    shufflable >= 4
                    and shufflable / max(np.count_nonzero(joined), 1) >= 0.5
                ),
                "within_shaft_reversal_eligible": bool(shufflable >= 4),
                "shaft_profile_exchangeable_fraction": float(
                    exchangeable / max(np.count_nonzero(joined), 1)
                ),
                "shaft_label_permutation_eligible": bool(
                    exchangeable >= 4
                    and exchangeable / max(np.count_nonzero(joined), 1) >= 0.5
                ),
                "geometry_complete_joined": bool(np.all(geometry)),
                "n_geometry_mapped_joined": int(np.count_nonzero(geometry)),
                "geometry_smooth_eligible": bool(
                    np.all(geometry) and np.count_nonzero(joined) >= 6
                ),
                "prefix_participation_available": (
                    "prefix_participation_support" in feature_names
                ),
                "shaft_position_available": (
                    "within_shaft_position" in feature_names
                ),
                "contact_spacing_constructible": bool(np.count_nonzero(geometry) >= 6),
                "soz_labels_available": bool(subject_key in soz),
                "baseline_band_power_cached": False,
                "gm_wm_label_available": False,
                "artifact_rejection_rate_available": False,
                "all_model_fields_available": not missing_model_cells,
                "missing_model_cells": ";".join(missing_model_cells),
                "all_strict_target_keys_present": all(
                    f"bb150_auc__{index}" in available_keys for index in seizures
                ),
            }
        )
    frame = pd.DataFrame(rows).sort_values("subject")
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "input_availability.csv", index=False)
    summary = {
        "contract": "topic5_static_scaffold_fixed_readout_validation_v0_1",
        "status": "INPUTS_AUDITED",
        "target_values_read": False,
        "early_ictal_arrays_deserialized": False,
        "strict_clinical_onset": {
            "n_patients": len(frame),
            "n_seizures": int(frame.n_seizures.sum()),
            "window_sec": [0, 10],
            "band_hz": [1, 150],
            "dataset": "epilepsiae",
        },
        "availability": {
            column: int(frame[column].sum())
            for column in (
                "all_model_fields_available",
                "within_shaft_circular_eligible",
                "within_shaft_reversal_eligible",
                "shaft_label_permutation_eligible",
                "geometry_complete_joined",
                "geometry_smooth_eligible",
                "prefix_participation_available",
                "shaft_position_available",
                "contact_spacing_constructible",
                "soz_labels_available",
                "baseline_band_power_cached",
                "gm_wm_label_available",
                "artifact_rejection_rate_available",
                "all_strict_target_keys_present",
            )
        },
        "constructible_without_new_raw_data": {
            "beta_binomial_participation": True,
            "laplacian_smoothed_participation": True,
            "dirichlet_contact_rank_histogram": True,
            "low_rank_nonrecurrent_contact_rank_estimator": True,
            "teacher_forced_one_step_field": True,
        },
        "not_currently_available_and_not_to_be_imputed": [
            "GM/WM labels",
            "artifact/rejection rate",
        ],
        "baseline_power": (
            "not cached; reconstructible from raw seizure windows with the "
            "existing build_topic5_v2_confound_maps.py producer, but must be "
            "tracked as a separate long-running data task"
        ),
    }
    (OUT / "INPUT_AUDIT.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
