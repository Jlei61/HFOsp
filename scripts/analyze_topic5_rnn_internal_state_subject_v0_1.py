#!/usr/bin/env python3
"""Analyze one subject across all three frozen GRU seeds without ictal targets."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_rnn_internal_state import (  # noqa: E402
    event_first_mean,
    fit_pca,
    linear_cka,
    observable_design,
    prefix_observables,
    subspace_overlap,
)


DATASET = ROOT / "results/topic5_interictal_rank_distribution/dataset_v0_4"
BASE = ROOT / "results/topic5_rnn_internal_state_reduction"
SEED_DIRS = ("seed_20260725", "seed_20260726", "seed_20260727")
CONTROLS = ("full_history_gru", "rank_shuffle_gru")
K_VALUES = (2, 4, 8, 16, 32)
ALPHAS = (1.0e-5, 1.0e-4, 1.0e-3)
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
TEMPERATURES = (0.25, 0.5, 1.0, 2.0, 4.0)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def event_first_nll(
    probabilities: np.ndarray,
    target: np.ndarray,
    event_index: np.ndarray,
) -> float:
    values = probabilities[np.arange(len(target)), target]
    return event_first_mean(-np.log(np.clip(values, 1.0e-9, 1.0)), event_index)


def fit_action_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    validation_event: np.ndarray,
    heldout_x: np.ndarray,
    heldout_y: np.ndarray,
    heldout_event: np.ndarray,
    *,
    n_classes: int,
    random_state: int,
) -> tuple[float, float, float, float]:
    train_onehot = np.eye(n_classes, dtype=np.float32)[train_y]
    validation_onehot = np.eye(n_classes, dtype=np.float32)[validation_y]
    scored = []
    for alpha in RIDGE_ALPHAS:
        model = make_pipeline(
            StandardScaler(),
            Ridge(alpha=float(alpha), solver="cholesky"),
        )
        model.fit(train_x, train_onehot)
        validation_score = model.predict(validation_x)
        for temperature in TEMPERATURES:
            probability = softmax(
                validation_score / float(temperature), axis=1
            )
            scored.append(
                (
                    event_first_nll(
                        probability, validation_y, validation_event
                    ),
                    alpha,
                    temperature,
                )
            )
    _, selected, temperature = min(
        scored, key=lambda item: (item[0], item[1], item[2])
    )
    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=float(selected), solver="cholesky"),
    )
    model.fit(
        np.row_stack([train_x, validation_x]),
        np.row_stack([train_onehot, validation_onehot]),
    )
    probability = softmax(
        model.predict(heldout_x) / float(temperature), axis=1
    )
    nll = event_first_nll(probability, heldout_y, heldout_event)
    accuracy = float(
        np.mean(np.argmax(probability, axis=1) == heldout_y)
    )
    return nll, accuracy, float(selected), float(temperature)


def candidate_mse(
    prediction: np.ndarray,
    target: np.ndarray,
    candidate: np.ndarray,
    event_index: np.ndarray,
) -> float:
    squared = (np.asarray(prediction) - np.asarray(target)) ** 2
    per_prefix = np.sum(squared * candidate, axis=1) / np.maximum(
        candidate.sum(1), 1
    )
    return event_first_mean(per_prefix, event_index)


def fit_ridge_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    validation_candidate: np.ndarray,
    validation_event: np.ndarray,
    heldout_x: np.ndarray,
    heldout_y: np.ndarray,
    heldout_candidate: np.ndarray,
    heldout_event: np.ndarray,
) -> tuple[float, float]:
    scores = []
    for alpha in RIDGE_ALPHAS:
        model = make_pipeline(
            StandardScaler(),
            Ridge(alpha=float(alpha), solver="cholesky"),
        )
        model.fit(train_x, train_y)
        prediction = np.clip(model.predict(validation_x), 0.0, 1.0)
        scores.append(
            (
                candidate_mse(
                    prediction,
                    validation_y,
                    validation_candidate,
                    validation_event,
                ),
                alpha,
            )
        )
    _, selected = min(scores, key=lambda item: (item[0], item[1]))
    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=float(selected), solver="cholesky"),
    )
    model.fit(
        np.row_stack([train_x, validation_x]),
        np.row_stack([train_y, validation_y]),
    )
    prediction = np.clip(model.predict(heldout_x), 0.0, 1.0)
    return (
        candidate_mse(
            prediction, heldout_y, heldout_candidate, heldout_event
        ),
        float(selected),
    )


def residualize_hidden(
    train_x: np.ndarray,
    train_hidden: np.ndarray,
    validation_x: np.ndarray,
    validation_hidden: np.ndarray,
    heldout_x: np.ndarray,
    heldout_hidden: np.ndarray,
) -> tuple[np.ndarray, float]:
    scored = []
    for alpha in RIDGE_ALPHAS:
        model = make_pipeline(
            StandardScaler(),
            Ridge(alpha=float(alpha), solver="cholesky"),
        )
        model.fit(train_x, train_hidden)
        prediction = model.predict(validation_x)
        scored.append((float(np.mean((prediction - validation_hidden) ** 2)), alpha))
    _, selected = min(scored, key=lambda item: (item[0], item[1]))
    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=float(selected), solver="cholesky"),
    )
    model.fit(
        np.row_stack([train_x, validation_x]),
        np.row_stack([train_hidden, validation_hidden]),
    )
    return heldout_hidden - model.predict(heldout_x), float(selected)


def load_record(subject: str) -> dict[str, np.ndarray]:
    with np.load(
        DATASET / "per_subject" / f"{subject}.npz", allow_pickle=False
    ) as data:
        return {
            "contact_names": np.asarray(data["contact_names"]).astype(str),
            "groups": np.asarray(data["event_group_ids"], np.int16),
            "counts": np.asarray(data["event_group_count"], np.int16),
        }


def load_cells(subject: str) -> dict[str, dict[str, np.ndarray]]:
    cells = {}
    for seed_dir in SEED_DIRS:
        path = BASE / "interictal/cells" / seed_dir / subject / "hidden_states.npz"
        status = json.loads(
            path.with_name("CELL_STATUS.json").read_text(encoding="utf-8")
        )
        if status.get("status") != "COMPLETE":
            raise RuntimeError(f"{subject}/{seed_dir}: cell incomplete")
        with np.load(path, allow_pickle=False) as data:
            cells[seed_dir] = {key: np.asarray(data[key]) for key in data.files}
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    args = parser.parse_args()
    started = time.time()
    out = BASE / "interictal/per_subject" / args.subject
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / "ANALYSIS_STATUS.json"
    atomic_json(
        status_path,
        {
            "status": "RUNNING",
            "subject": args.subject,
            "target_values_read": False,
        },
    )
    record = load_record(args.subject)
    cells = load_cells(args.subject)
    first = cells[SEED_DIRS[0]]
    for seed_dir, cell in cells.items():
        if not np.array_equal(first["contact_names"], cell["contact_names"]):
            raise RuntimeError(f"{args.subject}: contact ordering changed across seeds")
        for split in ("train60", "validation20", "heldout20"):
            for key in ("event_index", "step"):
                name = f"{split}_{key}"
                if not np.array_equal(first[name], cell[name]):
                    raise RuntimeError(
                        f"{args.subject}: prefix identities changed across seeds"
                    )

    split_observables = {}
    split_design = {}
    for split in ("train60", "validation20", "heldout20"):
        split_observables[split] = prefix_observables(
            record["groups"],
            record["counts"],
            first[f"{split}_event_index"],
            first[f"{split}_step"],
        )
        split_design[split] = {
            mode: observable_design(split_observables[split], mode)
            for mode in ("progress", "last_set", "unordered")
        }

    stability_rows = []
    residuals: dict[str, dict[str, np.ndarray]] = {
        control: {} for control in CONTROLS
    }
    for control in CONTROLS:
        for seed_dir in SEED_DIRS:
            cell = cells[seed_dir]
            train_hidden = cell[f"{control}_train60_hidden"].astype(np.float64)
            validation_hidden = cell[
                f"{control}_validation20_hidden"
            ].astype(np.float64)
            heldout_hidden = cell[f"{control}_heldout20_hidden"].astype(np.float64)
            residual, alpha = residualize_hidden(
                split_design["train60"]["unordered"],
                train_hidden,
                split_design["validation20"]["unordered"],
                validation_hidden,
                split_design["heldout20"]["unordered"],
                heldout_hidden,
            )
            residuals[control][seed_dir] = residual
            train_events = cell["train60_event_index"]
            unique = np.unique(train_events)
            cut = len(unique) // 2
            first_mask = np.isin(train_events, unique[:cut])
            second_mask = np.isin(train_events, unique[cut:])
            first_pca = fit_pca(train_hidden[first_mask])
            second_pca = fit_pca(train_hidden[second_mask])
            for k in (2, 4, 8):
                stability_rows.append(
                    {
                        "subject": args.subject,
                        "control": control,
                        "seed_dir": seed_dir,
                        "comparison": "chronological_split_half_subspace",
                        "k": k,
                        "value": subspace_overlap(
                            first_pca.components[:k],
                            second_pca.components[:k],
                        ),
                        "residual_alpha": alpha,
                    }
                )
        for left_index, left in enumerate(SEED_DIRS):
            for right in SEED_DIRS[left_index + 1 :]:
                raw = linear_cka(
                    cells[left][f"{control}_heldout20_hidden"],
                    cells[right][f"{control}_heldout20_hidden"],
                )
                residual = linear_cka(
                    residuals[control][left], residuals[control][right]
                )
                stability_rows.extend(
                    [
                        {
                            "subject": args.subject,
                            "control": control,
                            "seed_dir": f"{left}__{right}",
                            "comparison": "cross_seed_raw_cka",
                            "k": -1,
                            "value": raw,
                        },
                        {
                            "subject": args.subject,
                            "control": control,
                            "seed_dir": f"{left}__{right}",
                            "comparison": "cross_seed_residual_cka",
                            "k": -1,
                            "value": residual,
                        },
                    ]
                )

    probe_rows = []
    n_classes = len(record["contact_names"]) + 1
    for seed_index, seed_dir in enumerate(SEED_DIRS):
        cell = cells[seed_dir]
        pca = {
            control: {
                "mean": cell[f"{control}_pca_mean"].astype(np.float64),
                "components": cell[
                    f"{control}_pca_components"
                ].astype(np.float64),
            }
            for control in CONTROLS
        }
        scores = {}
        for control in CONTROLS:
            scores[control] = {}
            for split in ("train60", "validation20", "heldout20"):
                hidden = cell[f"{control}_{split}_hidden"].astype(np.float64)
                scores[control][split] = (
                    hidden - pca[control]["mean"]
                ) @ pca[control]["components"].T

        for baseline_mode in ("progress", "last_set", "unordered"):
            nll, accuracy, alpha, temperature = fit_action_probe(
                split_design["train60"][baseline_mode],
                split_observables["train60"]["next_action"],
                split_design["validation20"][baseline_mode],
                split_observables["validation20"]["next_action"],
                first["validation20_event_index"],
                split_design["heldout20"][baseline_mode],
                split_observables["heldout20"]["next_action"],
                first["heldout20_event_index"],
                n_classes=n_classes,
                random_state=20260728 + seed_index,
            )
            probe_rows.append(
                {
                    "subject": args.subject,
                    "seed_dir": seed_dir,
                    "task": "next_action",
                    "feature": baseline_mode,
                    "k": 0,
                    "metric": "event_first_nll",
                    "value": nll,
                    "secondary_accuracy": accuracy,
                    "selected_alpha": alpha,
                    "selected_temperature": temperature,
                }
            )

        for control in CONTROLS:
            label = "full_hidden" if control == "full_history_gru" else "rank_shuffle_hidden"
            for k in K_VALUES:
                used = min(k, scores[control]["train60"].shape[1])
                matrices = {
                    split: np.column_stack(
                        [
                            split_design[split]["unordered"],
                            scores[control][split][:, :used],
                        ]
                    )
                    for split in ("train60", "validation20", "heldout20")
                }
                nll, accuracy, alpha, temperature = fit_action_probe(
                    matrices["train60"],
                    split_observables["train60"]["next_action"],
                    matrices["validation20"],
                    split_observables["validation20"]["next_action"],
                    first["validation20_event_index"],
                    matrices["heldout20"],
                    split_observables["heldout20"]["next_action"],
                    first["heldout20_event_index"],
                    n_classes=n_classes,
                    random_state=20260728 + seed_index + k * 17,
                )
                probe_rows.append(
                    {
                        "subject": args.subject,
                        "seed_dir": seed_dir,
                        "task": "next_action",
                        "feature": f"unordered_plus_{label}",
                        "k": used,
                        "metric": "event_first_nll",
                        "value": nll,
                        "secondary_accuracy": accuracy,
                        "selected_alpha": alpha,
                        "selected_temperature": temperature,
                    }
                )
                if k != 8:
                    continue
                for task, target_name in (
                    ("future_participation", "future_participation"),
                    ("remaining_rank_score", "remaining_score"),
                ):
                    value, ridge_alpha = fit_ridge_probe(
                        matrices["train60"],
                        split_observables["train60"][target_name],
                        matrices["validation20"],
                        split_observables["validation20"][target_name],
                        split_observables["validation20"]["candidate"],
                        first["validation20_event_index"],
                        matrices["heldout20"],
                        split_observables["heldout20"][target_name],
                        split_observables["heldout20"]["candidate"],
                        first["heldout20_event_index"],
                    )
                    probe_rows.append(
                        {
                            "subject": args.subject,
                            "seed_dir": seed_dir,
                            "task": task,
                            "feature": f"unordered_plus_{label}",
                            "k": used,
                            "metric": (
                                "event_first_candidate_brier"
                                if task == "future_participation"
                                else "event_first_candidate_mse"
                            ),
                            "value": value,
                            "selected_alpha": ridge_alpha,
                        }
                    )

        for task, target_name in (
            ("future_participation", "future_participation"),
            ("remaining_rank_score", "remaining_score"),
        ):
            value, ridge_alpha = fit_ridge_probe(
                split_design["train60"]["unordered"],
                split_observables["train60"][target_name],
                split_design["validation20"]["unordered"],
                split_observables["validation20"][target_name],
                split_observables["validation20"]["candidate"],
                first["validation20_event_index"],
                split_design["heldout20"]["unordered"],
                split_observables["heldout20"][target_name],
                split_observables["heldout20"]["candidate"],
                first["heldout20_event_index"],
            )
            probe_rows.append(
                {
                    "subject": args.subject,
                    "seed_dir": seed_dir,
                    "task": task,
                    "feature": "unordered",
                    "k": 0,
                    "metric": (
                        "event_first_candidate_brier"
                        if task == "future_participation"
                        else "event_first_candidate_mse"
                    ),
                    "value": value,
                    "selected_alpha": ridge_alpha,
                }
            )

    pd.DataFrame(stability_rows).to_csv(out / "stability_metrics.csv", index=False)
    pd.DataFrame(probe_rows).to_csv(out / "probe_metrics.csv", index=False)
    atomic_json(
        status_path,
        {
            "contract": "topic5_rnn_internal_state_reduction_v0_1",
            "status": "COMPLETE",
            "subject": args.subject,
            "n_contacts": int(len(record["contact_names"])),
            "n_stability_rows": len(stability_rows),
            "n_probe_rows": len(probe_rows),
            "runtime_seconds": float(time.time() - started),
            "target_values_read": False,
            "early_ictal_arrays_deserialized": False,
        },
    )


if __name__ == "__main__":
    main()
