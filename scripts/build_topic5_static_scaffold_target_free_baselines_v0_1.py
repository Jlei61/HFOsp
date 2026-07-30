#!/usr/bin/env python3
"""Build and freeze target-free regularized static scaffold baselines.

All tuning uses only the chronological train60/validation20 subdivision of the
already frozen interictal train80. Early-ictal arrays are never opened here.
After selecting each hyperparameter, the estimator is refit on the complete
interictal train80.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_topic5_rnn_bidirectional_cross_model_v2_5 import (  # noqa: E402
    strict_clinical_inventory,
)
from src.topic5_rnn_internal_state import split_train80  # noqa: E402
from src.topic5_static_scaffold_validation import (  # noqa: E402
    beta_binomial_participation,
    categorical_event_nll,
    contact_graph,
    contact_rank_categories,
    dirichlet_contact_rank_distribution,
    event_brier,
    laplacian_smooth,
    participation_rate,
)


DATASET = ROOT / "results/topic5_interictal_rank_distribution/dataset_v0_4"
OUT = ROOT / "results/topic5_static_scaffold_fixed_readout_validation"
PER_SUBJECT = OUT / "target_free_baselines/per_subject"
CONCENTRATIONS = (
    0.0,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
    50.0,
    100.0,
    200.0,
    500.0,
    1_000.0,
    2_000.0,
)
LAPLACIAN_PENALTIES = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
LOW_RANKS = (1, 2, 3, 4)
N_RANK_BINS = 10


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_low_rank_logit_distribution(
    categories: np.ndarray, rank: int
) -> np.ndarray:
    """Exact truncated-SVD estimator on smoothed contact-category logits."""
    values = np.asarray(categories, dtype=np.int64)
    counts = np.column_stack(
        [
            np.sum(values == category, axis=0)
            for category in range(N_RANK_BINS + 1)
        ]
    ).astype(np.float64)
    matrix = counts + 0.5
    matrix /= matrix.sum(axis=1, keepdims=True)
    logit = np.log(matrix)
    logit -= logit.mean(axis=1, keepdims=True)
    rank = min(int(rank), min(matrix.shape))
    u, singular, vh = np.linalg.svd(logit, full_matrices=False)
    reconstructed = (u[:, :rank] * singular[:rank]) @ vh[:rank]
    reconstructed -= reconstructed.max(axis=1, keepdims=True)
    probability = np.exp(reconstructed)
    probability /= probability.sum(axis=1, keepdims=True)
    return probability


def choose(
    candidates: list[tuple[float | int, np.ndarray]],
    validation_categories: np.ndarray,
    *,
    score: str,
) -> tuple[float | int, float]:
    rows = []
    for parameter, estimate in candidates:
        if score == "brier":
            value = event_brier(
                np.asarray(estimate, dtype=np.float64),
                np.where(validation_categories > 0, 0, -1),
            )
        elif score == "categorical_nll":
            value = categorical_event_nll(estimate, validation_categories)
        else:
            raise ValueError(f"unknown selection score: {score}")
        rows.append((float(value), parameter))
    best_value, best_parameter = min(rows, key=lambda item: (item[0], item[1]))
    return best_parameter, float(best_value)


def main() -> None:
    audit = json.loads((OUT / "INPUT_AUDIT.json").read_text(encoding="utf-8"))
    if audit["target_values_read"] or audit["early_ictal_arrays_deserialized"]:
        raise RuntimeError("input audit did not preserve target sealing")
    subjects = sorted(strict_clinical_inventory())
    PER_SUBJECT.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for subject_index, subject in enumerate(subjects):
        source = DATASET / "per_subject" / f"{subject}.npz"
        with np.load(source, allow_pickle=False) as data:
            names = np.asarray(data["contact_names"]).astype(str)
            coords = np.asarray(data["contact_coords"], dtype=np.float64)
            groups = np.asarray(data["event_group_ids"], dtype=np.int16)
            split = np.asarray(data["event_split"], dtype=np.uint8)
        train80 = np.flatnonzero(split == 0)
        train60, validation20 = split_train80(train80)
        groups60 = groups[train60]
        groups20 = groups[validation20]
        groups80 = groups[train80]
        categories60 = contact_rank_categories(groups60, N_RANK_BINS)
        categories20 = contact_rank_categories(groups20, N_RANK_BINS)
        categories80 = contact_rank_categories(groups80, N_RANK_BINS)

        fields: dict[str, np.ndarray] = {
            "raw_train80_participation": participation_rate(groups80)
        }
        validation_fields: dict[str, np.ndarray] = {}
        metadata: dict[str, Any] = {
            "subject": subject,
            "target_values_read": False,
            "source_npz": str(source.relative_to(ROOT)),
            "source_npz_sha256": sha256(source),
            "n_contacts": int(len(names)),
            "n_train60_events": int(len(train60)),
            "n_validation20_events": int(len(validation20)),
            "n_refit_train80_events": int(len(train80)),
            "selection_contract": (
                "hyperparameters selected on interictal validation20; "
                "selected estimator refit on full interictal train80"
            ),
            "estimators": {},
        }

        beta_candidates = [
            (
                concentration,
                beta_binomial_participation(groups60, concentration),
            )
            for concentration in CONCENTRATIONS
        ]
        beta, beta_score = choose(
            beta_candidates, categories20, score="brier"
        )
        fields["beta_binomial_participation"] = beta_binomial_participation(
            groups80, float(beta)
        )
        validation_fields["beta_binomial_participation"] = next(
            estimate
            for parameter, estimate in beta_candidates
            if float(parameter) == float(beta)
        )
        metadata["estimators"]["beta_binomial_participation"] = {
            "selected_concentration": float(beta),
            "validation_event_brier": beta_score,
            "participation_effective_df": float(
                len(names) * len(train80) / (len(train80) + float(beta))
            ),
        }

        for mode in ("shaft", "geometry"):
            name = f"{mode}_laplacian_participation"
            eligible = mode == "shaft" or bool(np.all(np.isfinite(coords)))
            if not eligible:
                fields[name] = np.full(len(names), np.nan)
                metadata["estimators"][name] = {
                    "eligible": False,
                    "reason": "nonfinite_contact_geometry",
                }
                continue
            graph = contact_graph(
                names, coords=coords if mode == "geometry" else None, mode=mode
            )
            raw60 = participation_rate(groups60)
            candidates = [
                (penalty, laplacian_smooth(raw60, graph, penalty))
                for penalty in LAPLACIAN_PENALTIES
            ]
            penalty, score = choose(candidates, categories20, score="brier")
            fields[name] = laplacian_smooth(
                participation_rate(groups80), graph, float(penalty)
            )
            validation_fields[name] = next(
                estimate
                for parameter, estimate in candidates
                if float(parameter) == float(penalty)
            )
            smoother = np.linalg.inv(
                np.eye(len(names))
                + float(penalty)
                * (np.diag(graph.sum(axis=1)) - graph)
            )
            metadata["estimators"][name] = {
                "eligible": True,
                "selected_penalty": float(penalty),
                "validation_event_brier": score,
                "participation_effective_df": float(np.trace(smoother)),
                "graph_edge_count": int(np.count_nonzero(graph) // 2),
            }

        dirichlet_candidates = [
            (
                concentration,
                dirichlet_contact_rank_distribution(
                    categories60,
                    concentration,
                    n_rank_bins=N_RANK_BINS,
                ),
            )
            for concentration in CONCENTRATIONS
        ]
        dirichlet, dirichlet_score = choose(
            dirichlet_candidates,
            categories20,
            score="categorical_nll",
        )
        dirichlet_distribution = dirichlet_contact_rank_distribution(
            categories80,
            float(dirichlet),
            n_rank_bins=N_RANK_BINS,
        )
        fields["dirichlet_rank_participation"] = (
            1.0 - dirichlet_distribution[:, 0]
        )
        validation_fields["dirichlet_rank_participation"] = 1.0 - next(
            estimate
            for parameter, estimate in dirichlet_candidates
            if float(parameter) == float(dirichlet)
        )[:, 0]
        metadata["estimators"]["dirichlet_rank_participation"] = {
            "selected_concentration": float(dirichlet),
            "validation_event_categorical_nll": dirichlet_score,
            "participation_effective_df": float(
                len(names)
                * len(train80)
                / (len(train80) + float(dirichlet))
            ),
        }

        ranks = [rank for rank in LOW_RANKS if rank <= min(len(names), 11)]
        low_rank_fits60 = {
            rank: fit_low_rank_logit_distribution(categories60, rank)
            for rank in ranks
        }
        low_rank_candidates = [
            (rank, low_rank_fits60[rank]) for rank in ranks
        ]
        low_rank, low_rank_score = choose(
            low_rank_candidates, categories20, score="categorical_nll"
        )
        low_rank_distribution = fit_low_rank_logit_distribution(
            categories80, int(low_rank)
        )
        fields["low_rank_logit_participation"] = (
            1.0 - low_rank_distribution[:, 0]
        )
        validation_fields["low_rank_logit_participation"] = (
            1.0 - low_rank_fits60[int(low_rank)][:, 0]
        )
        metadata["estimators"]["low_rank_logit_participation"] = {
            "selected_rank": int(low_rank),
            "validation_event_categorical_nll": low_rank_score,
            "solver": "exact_truncated_svd_of_jeffreys_smoothed_logits",
            "converged": True,
            "nominal_factor_df": int(
                int(low_rank) * (len(names) + 11) - int(low_rank) ** 2
            ),
        }

        validation_brier = {
            estimator: event_brier(field, groups20)
            for estimator, field in validation_fields.items()
            if np.all(np.isfinite(field))
        }
        best_regularized = min(
            validation_brier,
            key=lambda estimator: (validation_brier[estimator], estimator),
        )
        fields["best_validation_regularized_participation"] = fields[
            best_regularized
        ].copy()
        metadata["estimators"][
            "best_validation_regularized_participation"
        ] = {
            "selected_estimator": best_regularized,
            "validation_event_brier": float(
                validation_brier[best_regularized]
            ),
            "candidate_validation_event_brier": validation_brier,
            "target_values_read": False,
        }

        output_npz = PER_SUBJECT / f"{subject}.npz"
        np.savez_compressed(
            output_npz,
            contact_names=names,
            **{key: np.asarray(value, dtype=np.float32) for key, value in fields.items()},
        )
        metadata["output_npz_sha256"] = sha256(output_npz)
        atomic_json(output_npz.with_suffix(".json"), metadata)
        for estimator, detail in metadata["estimators"].items():
            selection_rows.append(
                {"subject": subject, "estimator": estimator, **detail}
            )
        manifest_rows.append(
            {
                "subject": subject,
                "source_npz_sha256": metadata["source_npz_sha256"],
                "output_npz_sha256": metadata["output_npz_sha256"],
                "n_contacts": len(names),
                "n_train60_events": len(train60),
                "n_validation20_events": len(validation20),
                "status": "ok",
            }
        )
        print(
            f"target-free baseline {subject_index + 1}/{len(subjects)} {subject}",
            flush=True,
        )

    manifest = pd.DataFrame(manifest_rows).sort_values("subject")
    selection = pd.DataFrame(selection_rows).sort_values(
        ["estimator", "subject"]
    )
    manifest.to_csv(OUT / "target_free_baseline_manifest.csv", index=False)
    selection.to_csv(OUT / "target_free_baseline_selection.csv", index=False)
    freeze = {
        "contract": "topic5_static_scaffold_fixed_readout_validation_v0_1",
        "phase": "target_free_regularized_baseline_freeze",
        "status": "COMPLETE",
        "target_values_read": False,
        "early_ictal_arrays_deserialized": False,
        "n_patients": len(manifest),
        "n_estimators": 7,
        "primary_field": "participation",
        "selection_partition": "chronological train60/validation20 within train80",
        "refit_partition": "chronological train80",
        "candidate_concentrations": list(CONCENTRATIONS),
        "candidate_laplacian_penalties": list(LAPLACIAN_PENALTIES),
        "candidate_low_ranks": list(LOW_RANKS),
        "manifest_sha256": sha256(OUT / "target_free_baseline_manifest.csv"),
        "selection_sha256": sha256(OUT / "target_free_baseline_selection.csv"),
    }
    atomic_json(OUT / "BASELINE_FREEZE.json", freeze)
    atomic_json(
        OUT / "RUN_STATUS.json",
        {
            "status": "BASELINES_FROZEN_TARGET_EVALUATION_PENDING",
            "input_audit": "INPUT_AUDIT.json",
            "phase1_summary": "PHASE1_EXISTING_FIELDS_SUMMARY.json",
            "baseline_freeze": "BASELINE_FREEZE.json",
        },
    )
    print(json.dumps(freeze, indent=2))


if __name__ == "__main__":
    main()
