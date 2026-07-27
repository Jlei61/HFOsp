#!/usr/bin/env python3
"""Run frozen interictal-distribution -> clinical-onset BB150 static LOSO readout."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_axis_positive_static_transfer_v2_4 import (  # noqa: E402
    robust_patient_standardize,
    weighted_ridge_predict,
)
from src.topic5_transition_decomposition_v0_1 import contact_shaft  # noqa: E402


BASE = ROOT / "results/topic5_rnn_axis_positive_static_transfer_v2_4"
AUDIT = BASE / "input_audit/INPUT_AUDIT_STATUS.json"
UNLOCK = BASE / "TARGET_UNLOCK.json"
REP = BASE / "representations"
TARGET = (
    ROOT
    / "results/topic5_ictal_recruitment/t0_feature_cache_bb150_1_150"
)
OUT = BASE / "static_readout"
VARIANTS = (
    "full_fixed_axis",
    "no_history",
    "local_isotropic",
    "node_only",
    "empirical_train80",
)
N_PERM = 5000


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


def bh_fdr(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result = np.empty_like(ranked)
    result[order] = np.minimum(ranked, 1.0)
    return result.tolist()


def bootstrap_ci(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    sample = rng.choice(values, size=(20_000, len(values)), replace=True)
    return np.quantile(np.median(sample, axis=1), [0.025, 0.975]).tolist()


def summarize(values: np.ndarray, seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    p = (
        1.0
        if np.allclose(values, 0.0)
        else float(wilcoxon(values, alternative="greater").pvalue)
    )
    return {
        "n": len(values),
        "median": float(np.median(values)),
        "bootstrap_ci95": bootstrap_ci(values, seed),
        "n_positive": int(np.count_nonzero(values > 0)),
        "wilcoxon_greater_p": p,
    }


def permutation_correlations(
    prediction: np.ndarray,
    target: np.ndarray,
    permutations: np.ndarray,
) -> np.ndarray:
    pred_rank = rankdata(prediction).astype(np.float64)
    target_rank = rankdata(target).astype(np.float64)
    pred_rank -= pred_rank.mean()
    target_rank -= target_rank.mean()
    denominator = np.linalg.norm(pred_rank) * np.linalg.norm(target_rank)
    if denominator <= 0:
        raise ValueError("constant prediction or target")
    return (target_rank[permutations] @ pred_rank) / denominator


def all_contact_permutations(n: int, rng: np.random.Generator) -> np.ndarray:
    return np.stack([rng.permutation(n) for _ in range(N_PERM)])


def within_shaft_permutations(
    names: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    base = np.arange(len(names))
    shafts: dict[str, np.ndarray] = {}
    for shaft in sorted({contact_shaft(name) for name in names}):
        shafts[shaft] = np.asarray(
            [
                index
                for index, name in enumerate(names)
                if contact_shaft(name) == shaft
            ],
            dtype=np.int64,
        )
    rows = []
    for _ in range(N_PERM):
        permutation = base.copy()
        for indices in shafts.values():
            permutation[indices] = rng.permutation(indices)
        rows.append(permutation)
    return np.stack(rows)


def load_patient(subject: str) -> dict[str, Any]:
    representation_path = REP / "per_subject" / f"{subject}.npz"
    sidecar_path = TARGET / f"{subject}.json"
    target_path = TARGET / f"{subject}.npz"
    metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
    with np.load(representation_path, allow_pickle=False) as data:
        representation_names = data["contact_names"].astype(str)
        representations = {
            variant: np.asarray(data[variant], dtype=np.float64)
            for variant in VARIANTS
        }
    # Target deserialization occurs only after TARGET_READ_STATE is written.
    with np.load(target_path, allow_pickle=True) as data:
        target_names = np.asarray(data["channels"]).astype(str)
        seizures = []
        used_indices = []
        for seizure_index in metadata["eligible_idxs"]:
            key = f"bb150_auc__{int(seizure_index)}"
            if key not in data.files:
                continue
            seizures.append(np.asarray(data[key], dtype=np.float64))
            used_indices.append(int(seizure_index))
    if not seizures:
        raise ValueError(f"{subject}: no BB150 seizure arrays")
    seizure_matrix = np.stack(seizures)
    target_index = {name: index for index, name in enumerate(target_names)}
    keep_model = [
        index
        for index, name in enumerate(representation_names)
        if name in target_index
    ]
    keep_target = [target_index[representation_names[index]] for index in keep_model]
    names = representation_names[keep_model]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        target_field = np.nanmedian(seizure_matrix[:, keep_target], axis=0)
    finite = np.isfinite(target_field)
    names = names[finite]
    target_field = target_field[finite]
    if len(names) < 6:
        raise ValueError(f"{subject}: fewer than six finite exact-joined contacts")
    standardized_target = robust_patient_standardize(target_field)
    return {
        "subject": subject,
        "names": names,
        "target_raw": target_field,
        "target": standardized_target,
        "representations": {
            variant: values[np.asarray(keep_model)[finite]]
            for variant, values in representations.items()
        },
        "n_seizures": len(used_indices),
        "seizure_indices": used_indices,
        "representation_sha256": sha256(representation_path),
        "target_sha256": sha256(target_path),
        "sidecar_sha256": sha256(sidecar_path),
    }


def main() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    unlock = json.loads(UNLOCK.read_text(encoding="utf-8"))
    manifest_path = ROOT / unlock["representation_manifest"]
    if unlock.get("status") != "FROZEN_INTERICTAL_REPRESENTATIONS":
        raise SystemExit("interictal representations are not frozen")
    if unlock.get("target_values_read"):
        raise SystemExit("target unlock chronology is invalid")
    if sha256(manifest_path) != unlock["representation_manifest_sha256"]:
        raise SystemExit("representation manifest hash drifted")
    if unlock.get("allowed_target") != (
        "clinical-onset [0,10] s 1-150 Hz static contact energy only"
    ):
        raise SystemExit("target contract drifted")

    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(
        OUT / "TARGET_READ_STATE.json",
        {
            "status": "READING_TARGET_AFTER_REPRESENTATION_FREEZE",
            "representation_manifest_sha256": sha256(manifest_path),
            "target_contract": (
                "clinical-onset [0,10] s 1-150 Hz baseline-robust-z static field"
            ),
            "dynamic_source_conditioned_rollout": "BLOCKED",
            "target_values_read": True,
        },
    )
    patients = [
        load_patient(subject)
        for subject in audit["target_metadata_eligible_patients"]
    ]
    if len(patients) < 8:
        raise SystemExit("fewer than eight target-ready patients after value audit")

    predictions: dict[str, dict[str, np.ndarray]] = {
        variant: {} for variant in VARIANTS
    }
    for heldout in patients:
        training = [patient for patient in patients if patient is not heldout]
        for variant in VARIANTS:
            train_x = np.concatenate(
                [patient["representations"][variant] for patient in training]
            )
            train_y = np.concatenate(
                [patient["target"] for patient in training]
            )
            train_weight = np.concatenate(
                [
                    np.full(
                        len(patient["target"]),
                        1.0 / len(patient["target"]),
                        dtype=np.float64,
                    )
                    for patient in training
                ]
            )
            predictions[variant][heldout["subject"]] = weighted_ridge_predict(
                train_x,
                train_y,
                train_weight,
                heldout["representations"][variant],
                alpha=1.0,
            )

    metric_rows = []
    per_subject_root = OUT / "per_subject"
    per_subject_root.mkdir(parents=True, exist_ok=True)
    for patient_index, patient in enumerate(patients):
        subject = patient["subject"]
        rng = np.random.default_rng(20260727 + patient_index)
        all_permutations = all_contact_permutations(len(patient["names"]), rng)
        shaft_permutations = within_shaft_permutations(
            patient["names"], rng
        )
        subject_predictions = {}
        for variant in VARIANTS:
            prediction = predictions[variant][subject]
            observed = float(spearmanr(prediction, patient["target"]).statistic)
            all_null = permutation_correlations(
                prediction, patient["target"], all_permutations
            )
            shaft_null = permutation_correlations(
                prediction, patient["target"], shaft_permutations
            )
            metric_rows.append(
                {
                    "subject": subject,
                    "model": variant,
                    "n_contacts": len(patient["names"]),
                    "n_seizures": patient["n_seizures"],
                    "spearman_rho": observed,
                    "all_contact_null_median": float(np.median(all_null)),
                    "all_contact_margin": float(
                        observed - np.median(all_null)
                    ),
                    "all_contact_empirical_p": float(
                        (1 + np.count_nonzero(all_null >= observed))
                        / (N_PERM + 1)
                    ),
                    "within_shaft_null_median": float(np.median(shaft_null)),
                    "within_shaft_margin": float(
                        observed - np.median(shaft_null)
                    ),
                    "within_shaft_empirical_p": float(
                        (1 + np.count_nonzero(shaft_null >= observed))
                        / (N_PERM + 1)
                    ),
                    "target_values_read": True,
                }
            )
            subject_predictions[variant] = prediction.tolist()
        atomic_json(
            per_subject_root / f"{subject}.json",
            {
                "subject": subject,
                "contact_names": patient["names"].tolist(),
                "target_raw": patient["target_raw"].tolist(),
                "target_standardized": patient["target"].tolist(),
                "predictions": subject_predictions,
                "n_seizures": patient["n_seizures"],
                "seizure_indices": patient["seizure_indices"],
                "representation_sha256": patient["representation_sha256"],
                "target_sha256": patient["target_sha256"],
                "target_values_read": True,
            },
        )

    frame = pd.DataFrame(metric_rows)
    frame.to_csv(OUT / "patient_model_metrics.csv", index=False)
    wide_rho = frame.pivot(
        index="subject", columns="model", values="spearman_rho"
    )
    wide_margin = frame.pivot(
        index="subject", columns="model", values="all_contact_margin"
    )
    full_margin = wide_margin["full_fixed_axis"].to_numpy(float)
    history = (
        wide_rho["full_fixed_axis"] - wide_rho["no_history"]
    ).to_numpy(float)
    axis = (
        wide_rho["full_fixed_axis"] - wide_rho["local_isotropic"]
    ).to_numpy(float)
    full_over_node = (
        wide_rho["full_fixed_axis"] - wide_rho["node_only"]
    ).to_numpy(float)
    empirical_over_full = (
        wide_rho["empirical_train80"] - wide_rho["full_fixed_axis"]
    ).to_numpy(float)
    summaries = {
        "gate_s_full_all_contact_margin": summarize(full_margin, 20260731),
        "gate_h_full_over_no_history_rho": summarize(history, 20260801),
        "gate_x_full_over_isotropic_rho": summarize(axis, 20260802),
        "full_over_node_only_rho": summarize(full_over_node, 20260803),
        "empirical_over_full_rho": summarize(empirical_over_full, 20260804),
    }
    family_keys = (
        "gate_s_full_all_contact_margin",
        "gate_h_full_over_no_history_rho",
        "gate_x_full_over_isotropic_rho",
    )
    q_values = bh_fdr(
        [summaries[key]["wilcoxon_greater_p"] for key in family_keys]
    )
    for key, q_value in zip(family_keys, q_values):
        summaries[key]["bh_fdr_q"] = q_value

    def passes(summary: dict[str, Any]) -> bool:
        return bool(
            summary["median"] > 0
            and summary["bootstrap_ci95"][0] > 0
            and summary["n_positive"] > summary["n"] / 2
            and summary["bh_fdr_q"] < 0.05
        )

    gate_s = passes(summaries["gate_s_full_all_contact_margin"])
    gate_h = passes(summaries["gate_h_full_over_no_history_rho"])
    gate_x = passes(summaries["gate_x_full_over_isotropic_rho"])
    axis_target_subjects = set(
        audit["axis_positive_target_metadata_intersection"]
    )
    axis_target = frame.loc[
        (frame.model == "full_fixed_axis")
        & frame.subject.isin(axis_target_subjects)
    ]
    comparison_rows = []
    for subject in wide_rho.index:
        comparison_rows.append(
            {
                "subject": subject,
                "full_all_contact_margin": float(
                    wide_margin.loc[subject, "full_fixed_axis"]
                ),
                "full_over_no_history_rho": float(
                    wide_rho.loc[subject, "full_fixed_axis"]
                    - wide_rho.loc[subject, "no_history"]
                ),
                "full_over_isotropic_rho": float(
                    wide_rho.loc[subject, "full_fixed_axis"]
                    - wide_rho.loc[subject, "local_isotropic"]
                ),
                "full_over_node_only_rho": float(
                    wide_rho.loc[subject, "full_fixed_axis"]
                    - wide_rho.loc[subject, "node_only"]
                ),
                "empirical_over_full_rho": float(
                    wide_rho.loc[subject, "empirical_train80"]
                    - wide_rho.loc[subject, "full_fixed_axis"]
                ),
                "axis_positive_target_sensitivity": subject
                in axis_target_subjects,
            }
        )
    pd.DataFrame(comparison_rows).to_csv(
        OUT / "patient_model_comparisons.csv", index=False
    )
    result = {
        "contract": "topic5_source_free_static_readout_v2_4",
        "status": "COMPLETE",
        "n_patients": len(patients),
        "patients": [patient["subject"] for patient in patients],
        "target": (
            "clinical-onset [0,10] s 1-150 Hz baseline-robust-z static field"
        ),
        "readout": "patient-LOSO ridge alpha=1.0",
        "n_perm": N_PERM,
        "metrics": summaries,
        "gate_s_source_free_static_readout": "PASS" if gate_s else "FAIL",
        "gate_h_history_contribution": "PASS" if gate_h else "FAIL",
        "gate_x_axis_contribution": "PASS" if gate_x else "FAIL",
        "axis_positive_target_sensitivity": {
            "n": len(axis_target),
            "patients": sorted(axis_target_subjects),
            "full_rho_median": float(axis_target.spearman_rho.median()),
            "full_all_contact_margin_median": float(
                axis_target.all_contact_margin.median()
            ),
            "inference": "descriptive_only_n5",
        },
        "dynamic_source_conditioned_rollout": (
            "BLOCKED_MISSING_EXACT_CLINICAL_ONSET_SOURCE_METADATA"
        ),
        "target_values_read": True,
    }
    atomic_json(OUT / "STATIC_READOUT_GATE_STATUS.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
