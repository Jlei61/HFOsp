#!/usr/bin/env python3
"""Contact-level confound audit for the fixed static-scaffold readout."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_topic5_rnn_bidirectional_cross_model_v2_5 import (  # noqa: E402
    load_target,
    ordinary_model_fields,
    strict_clinical_inventory,
)
from scripts.analyze_topic5_static_scaffold_fixed_readout_phase1 import (  # noqa: E402
    bootstrap_summary,
    bh_fdr,
    load_coords,
)
from src.topic5_static_scaffold_validation import (  # noqa: E402
    partial_rank_score,
    shaft_groups,
)


OUT = ROOT / "results/topic5_static_scaffold_fixed_readout_validation"
BASELINE_ROOT = OUT / "target_free_baselines/per_subject"
TF_ROOT = OUT / "teacher_forced_fields/per_seed"
SEEDS = (20260725, 20260726, 20260727)
SOZ_PATH = ROOT / "results/epilepsiae_soz_core_channels.json"
CONFOUND_MAP_PATH = OUT / "confound_maps/phase1_confound_maps.json"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def contact_number(name: str) -> int:
    match = re.search(r"(\d+)$", str(name))
    return int(match.group(1)) if match else 1_000_000


def shaft_position(names: np.ndarray) -> np.ndarray:
    result = np.zeros(len(names), dtype=np.float64)
    for indices in shaft_groups(names).values():
        ordered = indices[
            np.argsort(
                [contact_number(names[index]) for index in indices]
            )
        ]
        if len(ordered) == 1:
            result[ordered] = 0.0
        else:
            result[ordered] = np.linspace(-1.0, 1.0, len(ordered))
    return result


def geometry_covariates(coords: np.ndarray) -> dict[str, np.ndarray]:
    coords = np.asarray(coords, dtype=np.float64)
    if not np.all(np.isfinite(coords)):
        return {}
    centered = coords - coords.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    pc1 = centered @ vh[0]
    distance = squareform(pdist(coords))
    positive = np.where(distance > 0, distance, np.inf)
    nearest = positive.min(axis=1)
    count = min(3, max(len(coords) - 1, 1))
    local_scale = np.sort(positive, axis=1)[:, :count].mean(axis=1)
    density = 1.0 / np.maximum(local_scale, 1.0e-8)
    return {
        "geometry_pc1": pc1,
        "nearest_contact_spacing": nearest,
        "local_contact_density": density,
    }


def mapped_covariate(
    mapping: dict[str, Any], names: np.ndarray
) -> np.ndarray:
    return np.asarray(
        [mapping.get(str(name), np.nan) for name in names],
        dtype=np.float64,
    )


def load_teacher_full(subject: str, names: np.ndarray) -> np.ndarray:
    fields = []
    for seed in SEEDS:
        path = TF_ROOT / f"{subject}_seed{seed}_full_history_gru.npz"
        with np.load(path, allow_pickle=False) as data:
            current = np.asarray(data["contact_names"]).astype(str)
            if not np.array_equal(current, names):
                raise RuntimeError(f"{subject}: teacher contact ordering drifted")
            fields.append(
                np.asarray(data["union_participation"], dtype=np.float64)
            )
    return np.median(np.row_stack(fields), axis=0)


def main() -> None:
    soz_all = json.loads(SOZ_PATH.read_text())
    external = (
        json.loads(CONFOUND_MAP_PATH.read_text())
        if CONFOUND_MAP_PATH.exists()
        else {}
    )
    rows: list[dict[str, Any]] = []
    association_rows: list[dict[str, Any]] = []
    inventory = strict_clinical_inventory()
    for patient_index, (subject, seizures) in enumerate(inventory.items()):
        names, ordinary = ordinary_model_fields(subject)
        keep, target, used = load_target(subject, seizures, names)
        joined_names = names[keep]
        coords = load_coords(subject, names)[keep]
        with np.load(
            BASELINE_ROOT / f"{subject}.npz", allow_pickle=False
        ) as data:
            if not np.array_equal(
                np.asarray(data["contact_names"]).astype(str), names
            ):
                raise RuntimeError(f"{subject}: baseline ordering drifted")
            raw = np.asarray(
                data["raw_train80_participation"], dtype=np.float64
            )
            best = np.asarray(
                data["best_validation_regularized_participation"],
                dtype=np.float64,
            )
        fields = {
            "raw_train80_participation": raw[keep],
            "best_validation_regularized_participation": best[keep],
            "full_history_gru": np.median(
                np.row_stack(
                    [
                        np.asarray(seed["participation"], dtype=np.float64)
                        for seed in ordinary["full_history_gru"]
                    ]
                ),
                axis=0,
            )[keep],
            "rank_shuffle_gru": np.median(
                np.row_stack(
                    [
                        np.asarray(seed["participation"], dtype=np.float64)
                        for seed in ordinary["rank_shuffle_gru"]
                    ]
                ),
                axis=0,
            )[keep],
            "teacher_forced_full_gru": load_teacher_full(
                subject, names
            )[keep],
        }
        covariates: dict[str, np.ndarray] = {
            "within_shaft_position": shaft_position(joined_names)
        }
        covariates.update(geometry_covariates(coords))
        subject_key = subject.replace("epilepsiae_", "")
        if subject_key in soz_all:
            soz = set(map(str, soz_all[subject_key]))
            covariates["soz_indicator"] = np.asarray(
                [name in soz for name in joined_names], dtype=np.float64
            )
        external_record = external.get(subject, {})
        for key in ("baseline_band_power", "broadband_1_250"):
            mapping = external_record.get(key, {})
            if mapping:
                covariates[key] = mapped_covariate(mapping, joined_names)
        adjustment_blocks = dict(covariates)
        adjustment_blocks["raw_participation"] = fields[
            "raw_train80_participation"
        ]
        for model_index, (model, field) in enumerate(fields.items()):
            for block_index, (block, covariate) in enumerate(
                adjustment_blocks.items()
            ):
                if (
                    block == "raw_participation"
                    and model == "raw_train80_participation"
                ):
                    continue
                result = partial_rank_score(
                    field,
                    target,
                    covariate,
                    n_null_draws=2_000,
                    null_seed=(
                        2026080400
                        + patient_index * 1000
                        + model_index * 100
                        + block_index
                    ),
                )
                row = {
                    "subject": subject,
                    "model": model,
                    "confound_block": block,
                    "eligible": bool(result["eligible"]),
                    "n_seizures": len(used),
                    "n_contacts_total": len(joined_names),
                    "interpretation": (
                        "increment_beyond_raw_participation"
                        if block == "raw_participation"
                        else "single_block_partial_rank_sensitivity"
                    ),
                    **{
                        key: value
                        for key, value in result.items()
                        if key != "per_seizure_signed_rho"
                    },
                }
                rows.append(row)
                finite = np.isfinite(field) & np.isfinite(covariate)
                if (
                    np.count_nonzero(finite) >= 4
                    and np.std(field[finite]) > 1.0e-12
                    and np.std(covariate[finite]) > 1.0e-12
                ):
                    field_assoc = float(
                        spearmanr(
                            field[finite], covariate[finite]
                        ).statistic
                    )
                    target_assoc = np.median(
                        [
                            spearmanr(
                                seizure[finite], covariate[finite]
                            ).statistic
                            for seizure in target
                        ]
                    )
                else:
                    field_assoc = np.nan
                    target_assoc = np.nan
                association_rows.append(
                    {
                        "subject": subject,
                        "model": model,
                        "confound_block": block,
                        "n_contacts": int(np.count_nonzero(finite)),
                        "field_confound_spearman": field_assoc,
                        "target_confound_spearman_seizure_median": (
                            float(target_assoc)
                        ),
                    }
                )
        print(f"phase4 {patient_index + 1}/16 {subject}", flush=True)

    patient = pd.DataFrame(rows).sort_values(
        ["confound_block", "model", "subject"]
    )
    patient.to_csv(
        OUT / "phase4_contact_confound_partial_scores.csv", index=False
    )
    association = pd.DataFrame(association_rows).sort_values(
        ["confound_block", "model", "subject"]
    )
    association.to_csv(
        OUT / "phase4_contact_confound_associations.csv", index=False
    )
    summaries: dict[str, Any] = {}
    eligible = patient.loc[patient.eligible].copy()
    for (block, model), group in eligible.groupby(
        ["confound_block", "model"]
    ):
        for metric in (
            "signed_rho",
            "absolute_rho",
            "signed_margin",
            "absolute_margin",
        ):
            summaries[f"{block}__{model}__{metric}"] = bootstrap_summary(
                group[metric].to_numpy(float),
                2026078400 + len(summaries),
            )
    summary_rows = []
    for (block, metric), group in eligible.melt(
        id_vars=["subject", "model", "confound_block"],
        value_vars=["signed_margin", "absolute_margin"],
        var_name="metric",
        value_name="value",
    ).groupby(["confound_block", "metric"]):
        family_start = len(summary_rows)
        for model, model_group in group.groupby("model"):
            value = model_group.value.to_numpy(float)
            summary_rows.append(
                {
                    "confound_block": block,
                    "metric": metric,
                    "model": model,
                    **bootstrap_summary(
                        value, 2026079400 + len(summary_rows)
                    ),
                }
            )
        q = bh_fdr(
            [
                summary_rows[index]["wilcoxon_greater_p"]
                for index in range(family_start, len(summary_rows))
            ]
        )
        for index, q_value in zip(
            range(family_start, len(summary_rows)), q
        ):
            summary_rows[index]["family_bh_fdr_q"] = q_value
    pd.DataFrame(summary_rows).to_csv(
        OUT / "phase4_contact_confound_cohort_summary.csv", index=False
    )
    paired_rows = []
    paired_specs = (
        ("full_history_gru", "rank_shuffle_gru"),
        (
            "full_history_gru",
            "best_validation_regularized_participation",
        ),
        ("full_history_gru", "teacher_forced_full_gru"),
    )
    for block, group in eligible.groupby("confound_block"):
        for metric in ("signed_margin", "absolute_margin"):
            family_start = len(paired_rows)
            wide = group.pivot(
                index="subject", columns="model", values=metric
            )
            for left, right in paired_specs:
                if left not in wide or right not in wide:
                    continue
                difference = (wide[left] - wide[right]).dropna().to_numpy(
                    float
                )
                paired_rows.append(
                    {
                        "confound_block": block,
                        "metric": metric,
                        "left": left,
                        "right": right,
                        **bootstrap_summary(
                            difference,
                            2026081400 + len(paired_rows),
                        ),
                    }
                )
            q = bh_fdr(
                [
                    paired_rows[index]["wilcoxon_greater_p"]
                    for index in range(family_start, len(paired_rows))
                ]
            )
            for index, q_value in zip(
                range(family_start, len(paired_rows)), q
            ):
                paired_rows[index]["family_bh_fdr_q"] = q_value
    pd.DataFrame(paired_rows).to_csv(
        OUT / "phase4_contact_confound_paired_comparisons.csv", index=False
    )
    availability = (
        patient.groupby("confound_block")
        .agg(
            n_patients=("subject", "nunique"),
            n_eligible_rows=("eligible", "sum"),
        )
        .reset_index()
        .to_dict(orient="records")
    )
    result = {
        "contract": "topic5_static_scaffold_fixed_readout_validation_v0_1",
        "phase": "contact_confound_audit",
        "status": "COMPLETE",
        "n_patients": 16,
        "confound_blocks": sorted(patient.confound_block.unique()),
        "availability": availability,
        "cohort_metrics": summaries,
        "paired_comparisons": paired_rows,
        "shaft_identity_handling": (
            "primary all-contact null plus within-shaft circular/dihedral "
            "nulls; not fitted as a high-dimensional contact regression"
        ),
        "baseline_power_maps_loaded": bool(external),
        "unavailable_not_imputed": [
            "GM/WM label",
            "artifact/rejection rate",
        ],
    }
    atomic_json(OUT / "PHASE4_CONTACT_CONFOUND_SUMMARY.json", result)
    atomic_json(
        OUT / "RUN_STATUS.json",
        {
            "status": (
                "PHASE4_COMPLETE_BASELINE_POWER_INCLUDED"
                if external
                else "PHASE4_TIER1_COMPLETE_BASELINE_POWER_PENDING"
            ),
            "phase1_summary": "PHASE1_EXISTING_FIELDS_SUMMARY.json",
            "phase2_summary": "PHASE2_REGULARIZED_BASELINE_SUMMARY.json",
            "phase3_summary": "PHASE3_TEACHER_FORCED_SUMMARY.json",
            "phase4_summary": "PHASE4_CONTACT_CONFOUND_SUMMARY.json",
        },
    )
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "n_patient_rows": len(patient),
                "baseline_power_maps_loaded": bool(external),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
