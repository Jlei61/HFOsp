"""Low-capacity transfer from frozen interictal state to frozen seizure targets.

No target is created here.  The input must already contain a seizure-level target,
its provenance, and a SHA256.  Continuous targets use ridge regression;
classification targets use ridge logistic regression.  Optimizer seeds are
collapsed inside each patient before any cohort summary.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.preprocessing import label_binarize


ARMS: Mapping[str, tuple[str, ...]] = {
    "baseline": ("baseline__",),
    "observation": ("baseline__", "observation__"),
    "state": ("baseline__", "observation__", "state__"),
    "wrong_time": ("baseline__", "observation__", "wrong_time__"),
}
TIERS = {
    "primary_chronological",
    "sensitivity_loso",
    "descriptive_case_series",
    "not_estimable",
}
TARGET_KINDS = {"continuous", "classification", "binary"}
BASE_COLUMNS = (
    "patient_id", "seed", "seizure_id", "split", "evaluation_tier",
    "target_name", "target_kind", "target_value", "target_frozen",
    "target_provenance", "target_source_sha256",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class PhenotypeRunResult:
    per_seed: pd.DataFrame
    patient_medians: pd.DataFrame
    audit: dict[str, Any]


def _missing(frame: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    return sorted(set(columns).difference(frame.columns))


def arm_columns(frame: pd.DataFrame, arm: str) -> list[str]:
    if arm not in ARMS:
        raise ValueError(f"unknown phenotype arm {arm!r}")
    output = []
    for prefix in ARMS[arm]:
        matched = sorted(column for column in frame if column.startswith(prefix))
        if not matched:
            raise ValueError(f"arm {arm} has no columns with prefix {prefix!r}")
        output.extend(matched)
    return output


def target_table_hash(frame: pd.DataFrame) -> str:
    columns = [
        "patient_id", "seed", "seizure_id", "split", "evaluation_tier",
        "target_name", "target_kind", "target_value", "target_provenance",
        "target_source_sha256",
    ]
    missing = _missing(frame, columns)
    if missing:
        raise ValueError(f"cannot hash target table; missing {missing}")
    rows = frame[columns].sort_values(
        ["patient_id", "target_name", "seed", "seizure_id"], kind="mergesort",
    ).where(pd.notna(frame[columns]), None).to_dict(orient="records")
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_phenotype_table(frame: pd.DataFrame) -> dict[str, Any]:
    missing = _missing(frame, BASE_COLUMNS)
    if missing:
        return {
            "status": "NOT_ESTIMABLE_MISSING_TARGET_COLUMNS",
            "missing_columns": missing,
            "target_reclustered": False,
        }
    if frame.empty:
        return {"status": "NOT_ESTIMABLE_EMPTY_TABLE", "target_reclustered": False}
    if not pd.api.types.is_bool_dtype(frame["target_frozen"]):
        raise ValueError("target_frozen must have boolean dtype")
    if not frame["target_frozen"].all():
        raise ValueError("phenotype target was not frozen before seizure probe fitting")
    if frame.duplicated(["patient_id", "seed", "seizure_id", "target_name"]).any():
        raise ValueError("duplicate patient/seed/seizure/target row")
    observed_kinds = set(frame["target_kind"].dropna().astype(str))
    if not observed_kinds.issubset(TARGET_KINDS):
        raise ValueError(f"unsupported target kinds: {sorted(observed_kinds - TARGET_KINDS)}")
    observed_tiers = set(frame["evaluation_tier"].dropna().astype(str))
    if not observed_tiers.issubset(TIERS):
        raise ValueError(f"unsupported evidence tiers: {sorted(observed_tiers - TIERS)}")

    available = frame[frame["target_value"].notna()].copy()
    if not available.empty:
        for arm in ARMS:
            columns = arm_columns(available, arm)
            if not np.isfinite(available[columns].to_numpy(dtype=float)).all():
                raise ValueError(f"non-finite features in {arm}")
        binary = available[available["target_kind"] == "binary"]
        if not set(binary["target_value"].astype(float).unique()).issubset({0.0, 1.0}):
            raise ValueError("binary targets must be encoded as 0/1")
        classified = available[available["target_kind"].isin(["binary", "classification"])]
        classified_values = classified["target_value"].to_numpy(dtype=float)
        if not np.all(classified_values == classified_values.astype(int)):
            raise ValueError("classification targets must use integer class labels")
    provenance_ready = frame["target_provenance"].notna() & frame[
        "target_source_sha256"
    ].astype(str).map(lambda value: bool(SHA256_RE.fullmatch(value)))
    if not provenance_ready[frame["target_value"].notna()].all():
        raise ValueError("every available frozen target needs provenance and a SHA256")

    for _, group in available.groupby(
        ["patient_id", "seizure_id", "target_name"], sort=False,
    ):
        if group["split"].nunique() != 1:
            raise ValueError("one seizure has different splits across seeds")
        if group["target_kind"].nunique() != 1:
            raise ValueError("target kind changes across optimizer seeds")
        values = group["target_value"].to_numpy(dtype=float)
        if not np.allclose(values, values[0], rtol=0.0, atol=0.0):
            raise ValueError("frozen target value changes across optimizer seeds")
    for _, group in available.groupby(["patient_id", "target_name"], sort=False):
        if group["target_source_sha256"].nunique() != 1:
            raise ValueError("target source hash changes within patient/target")
        if group["target_provenance"].nunique() != 1:
            raise ValueError("target provenance changes within patient/target")

    split_ids: dict[str, dict[str, dict[str, list[str]]]] = {}
    for (patient, target), group in available.groupby(
        ["patient_id", "target_name"], sort=True,
    ):
        split_ids.setdefault(str(patient), {})[str(target)] = {
            str(split): sorted(values["seizure_id"].astype(str).unique().tolist())
            for split, values in group.groupby("split", sort=True)
        }
    return {
        "status": "PASS",
        "target_reclustered": False,
        "target_frozen_before_probe": True,
        "target_values_seed_invariant": True,
        "split_seizure_ids": split_ids,
        "target_table_hash": target_table_hash(frame),
        "n_rows": int(len(frame)),
        "n_available_target_rows": int(frame["target_value"].notna().sum()),
    }


def _standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.mean(train, axis=0)
    scale = np.std(train, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (train - center) / scale, (test - center) / scale


def _fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: Sequence[str],
    kind: str,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    x_train, x_test = _standardize(
        train[list(columns)].to_numpy(dtype=float),
        test[list(columns)].to_numpy(dtype=float),
    )
    y = train["target_value"].to_numpy(dtype=float)
    if kind == "continuous":
        return Ridge(alpha=float(alpha)).fit(x_train, y).predict(x_test), None
    if len(np.unique(y)) < 2:
        raise ValueError("classification training fold has one target class")
    model = LogisticRegression(
        C=1.0 / float(alpha), solver="lbfgs", max_iter=2000,
    ).fit(x_train, y.astype(int))
    return model.predict_proba(x_test), np.asarray(model.classes_, dtype=int)


def _loss(
    kind: str,
    truth: np.ndarray,
    prediction: np.ndarray,
    classes: np.ndarray | None,
) -> float:
    if kind == "continuous":
        return float(np.mean((truth - prediction) ** 2))
    if classes is None or not set(truth.astype(int)).issubset(set(classes)):
        raise ValueError("evaluation contains a class absent from probe training")
    return float(log_loss(truth.astype(int), prediction, labels=classes))


def _metrics(
    kind: str,
    truth: np.ndarray,
    prediction: np.ndarray,
    classes: np.ndarray | None,
) -> dict[str, float]:
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if kind == "continuous":
        mse = float(np.mean((truth - prediction) ** 2))
        denominator = float(np.sum((truth - np.mean(truth)) ** 2))
        r2 = float(1.0 - np.sum((truth - prediction) ** 2) / denominator) \
            if denominator > 0 else float("nan")
        return {
            "loss": mse,
            "mse": mse,
            "mae": float(np.mean(np.abs(truth - prediction))),
            "r2": r2,
        }
    labels = truth.astype(int)
    if classes is None:
        raise ValueError("classification metrics require fitted class labels")
    encoded = label_binarize(labels, classes=classes)
    if len(classes) == 2:
        encoded = np.column_stack([1 - encoded[:, 0], encoded[:, 0]])
    output = {
        "loss": float(log_loss(labels, prediction, labels=classes)),
        "log_loss": float(log_loss(labels, prediction, labels=classes)),
        "calibration_brier": float(np.mean(np.sum((encoded - prediction) ** 2, axis=1))),
    }
    try:
        if len(classes) == 2:
            positive = (labels == classes[1]).astype(int)
            output["auroc"] = float(roc_auc_score(positive, prediction[:, 1]))
            output["auprc"] = float(average_precision_score(positive, prediction[:, 1]))
        else:
            output["auroc"] = float(roc_auc_score(
                labels, prediction, labels=classes, multi_class="ovr", average="macro",
            ))
            output["auprc"] = float(average_precision_score(
                encoded, prediction, average="macro",
            ))
    except ValueError:
        output["auroc"] = float("nan")
        output["auprc"] = float("nan")
    return output


def _select_alpha(
    train: pd.DataFrame,
    select: pd.DataFrame,
    columns: Sequence[str],
    kind: str,
    grid: Sequence[float],
) -> float:
    values = []
    for alpha in grid:
        try:
            prediction, classes = _fit_predict(train, select, columns, kind, alpha)
            candidate_loss = _loss(
                kind, select["target_value"].to_numpy(dtype=float), prediction, classes,
            )
        except ValueError:
            continue
        values.append((candidate_loss, -float(alpha), float(alpha)))
    if not values:
        raise ValueError("no estimable regularization candidate")
    return min(values)[2]


def _nested_loso_alpha(
    train: pd.DataFrame,
    columns: Sequence[str],
    kind: str,
    grid: Sequence[float],
) -> float:
    values = []
    for alpha in grid:
        losses = []
        for seizure_id in train["seizure_id"].astype(str).unique():
            select = train[train["seizure_id"].astype(str) == seizure_id]
            inner = train[train["seizure_id"].astype(str) != seizure_id]
            try:
                prediction, classes = _fit_predict(inner, select, columns, kind, alpha)
                candidate_loss = _loss(
                    kind, select["target_value"].to_numpy(dtype=float), prediction, classes,
                )
            except ValueError:
                continue
            losses.append(candidate_loss)
        if losses:
            values.append((float(np.mean(losses)), -float(alpha), float(alpha)))
    if not values:
        raise ValueError("nested LOSO cannot estimate regularization")
    return min(values)[2]


def _evaluate_arm(
    frame: pd.DataFrame,
    arm: str,
    kind: str,
    tier: str,
    grid: Sequence[float],
) -> dict[str, Any]:
    columns = arm_columns(frame, arm)
    predictions, truths, alphas, class_orders = [], [], [], []
    try:
        if tier == "primary_chronological":
            if not {"TRAIN", "SELECT", "TEST"}.issubset(set(frame["split"])):
                return {"status": "NOT_ESTIMABLE_AT_SPLIT"}
            train = frame[frame["split"] == "TRAIN"]
            select = frame[frame["split"] == "SELECT"]
            test = frame[frame["split"] == "TEST"]
            alpha = _select_alpha(train, select, columns, kind, grid)
            refit = pd.concat([train, select], ignore_index=True)
            prediction, classes = _fit_predict(refit, test, columns, kind, alpha)
            _loss(kind, test["target_value"].to_numpy(dtype=float), prediction, classes)
            predictions.append(prediction)
            truths.append(test["target_value"].to_numpy(dtype=float))
            alphas.append(alpha)
            class_orders.append(classes)
            scope = "TRAIN_fit_SELECT_choose_then_TRAIN_plus_SELECT_refit"
        elif tier in {"sensitivity_loso", "descriptive_case_series"}:
            for heldout in frame["seizure_id"].astype(str).unique():
                test = frame[frame["seizure_id"].astype(str) == heldout]
                train = frame[frame["seizure_id"].astype(str) != heldout]
                alpha = (
                    _nested_loso_alpha(train, columns, kind, grid)
                    if tier == "sensitivity_loso" else float(max(grid))
                )
                try:
                    prediction, classes = _fit_predict(train, test, columns, kind, alpha)
                    _loss(kind, test["target_value"].to_numpy(dtype=float), prediction, classes)
                except ValueError:
                    continue
                predictions.append(prediction)
                truths.append(test["target_value"].to_numpy(dtype=float))
                alphas.append(alpha)
                class_orders.append(classes)
            scope = (
                "nested_LOSO_within_probe_training_seizures"
                if tier == "sensitivity_loso"
                else "prespecified_strongest_ridge_descriptive_only"
            )
        else:
            return {"status": "NOT_ESTIMABLE_FEWER_THAN_TWO_SEIZURES"}
    except ValueError as error:
        return {"status": "NOT_ESTIMABLE_TARGET_CLASS_SUPPORT", "reason": str(error)}
    if not predictions:
        return {"status": "NOT_ESTIMABLE_TARGET_CLASS_SUPPORT"}
    classes = class_orders[0]
    if kind != "continuous" and any(
        not np.array_equal(classes, value) for value in class_orders[1:]
    ):
        return {"status": "NOT_ESTIMABLE_TARGET_CLASS_SUPPORT"}
    truth = np.concatenate(truths)
    prediction = np.concatenate(predictions)
    return {
        "status": "ok",
        "n_features": int(len(columns)),
        "chosen_alpha": float(np.median(alphas)),
        "selection_scope": scope,
        "n_evaluation_seizures": int(len(truth)),
        **_metrics(kind, truth, prediction, classes),
    }


def _patient_medians(per_seed: pd.DataFrame) -> pd.DataFrame:
    if per_seed.empty:
        return per_seed.copy()
    keys = ["patient_id", "target_name", "target_kind", "evaluation_tier"]
    numeric = [
        column for column in per_seed
        if column not in keys + ["seed"]
        and pd.api.types.is_numeric_dtype(per_seed[column])
        and not pd.api.types.is_bool_dtype(per_seed[column])
    ]
    medians = per_seed.groupby(keys, as_index=False)[numeric].median(numeric_only=True)
    counts = per_seed.groupby(keys, as_index=False)["seed"].nunique().rename(
        columns={"seed": "n_optimizer_seeds"}
    )
    output = medians.merge(counts, on=keys, how="left")
    output["seed_aggregation"] = "median_within_patient"
    output["seed_is_patient_replicate"] = False
    return output


def run_phenotype_table(
    frame: pd.DataFrame,
    *,
    regularization_grid: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
) -> PhenotypeRunResult:
    audit = validate_phenotype_table(frame)
    if audit["status"] != "PASS":
        return PhenotypeRunResult(pd.DataFrame(), pd.DataFrame(), audit)
    grid = tuple(float(value) for value in regularization_grid)
    if not grid or any(value <= 0 for value in grid):
        raise ValueError("regularization_grid must contain positive values")
    rows = []
    group_keys = [
        "patient_id", "seed", "target_name", "target_kind", "evaluation_tier",
    ]
    for keys, group in frame.groupby(group_keys, sort=True, dropna=False):
        patient, seed, target_name, kind, tier = keys
        row: dict[str, Any] = {
            "patient_id": str(patient),
            "seed": int(seed),
            "target_name": str(target_name),
            "target_kind": str(kind),
            "evaluation_tier": str(tier),
            "target_reclustered": False,
            "regularization_selected_only_on_train_select": True,
        }
        if group["target_value"].isna().all():
            row["status"] = "NOT_ESTIMABLE_MISSING_TARGET"
            rows.append(row)
            continue
        available = group[group["target_value"].notna()].copy()
        row["n_target_seizures"] = int(available["seizure_id"].nunique())
        if str(tier) == "not_estimable" or row["n_target_seizures"] < 2:
            row["status"] = "NOT_ESTIMABLE_FEWER_THAN_TWO_SEIZURES"
            rows.append(row)
            continue
        if str(kind) == "continuous" and available["target_value"].nunique() < 2:
            row["status"] = "NOT_ESTIMABLE_CONSTANT_TARGET"
            rows.append(row)
            continue
        arm_results = {
            arm: _evaluate_arm(available, arm, str(kind), str(tier), grid)
            for arm in ARMS
        }
        row["status"] = (
            "ok" if all(value["status"] == "ok" for value in arm_results.values())
            else "NOT_ESTIMABLE_ARM_SUPPORT"
        )
        for arm, result in arm_results.items():
            for key, value in result.items():
                if isinstance(value, (str, bool, int, float, np.integer, np.floating)):
                    row[f"{arm}__{key}"] = value
        if row["status"] == "ok":
            row["state_minus_observation_loss"] = (
                arm_results["state"]["loss"] - arm_results["observation"]["loss"]
            )
            row["state_minus_baseline_loss"] = (
                arm_results["state"]["loss"] - arm_results["baseline"]["loss"]
            )
            row["correct_minus_wrong_time_loss"] = (
                arm_results["state"]["loss"] - arm_results["wrong_time"]["loss"]
            )
        rows.append(row)
    per_seed = pd.DataFrame(rows)
    patient_medians = _patient_medians(per_seed)
    any_estimable = bool(
        not per_seed.empty and (per_seed["status"].astype(str) == "ok").any()
    )
    audit.update({
        "status": (
            "COMPLETE" if any_estimable
            else "NOT_ESTIMABLE_NO_USABLE_FROZEN_TARGET"
        ),
        "target_reclustered": False,
        "identical_seizure_target_rows_across_arms": True,
        "regularization_selected_only_on_train_select": True,
        "seed_is_patient_replicate": False,
        "seed_aggregation": "median_within_patient_before_cohort_inference",
        "continuous_estimator": "ridge_regression",
        "classification_estimator": "ridge_logistic_regression",
        "primary_secondary_effect": (
            "held-out state-minus-observation loss; negative favours phenotype transfer"
        ),
        "regularization_grid": list(grid),
    })
    return PhenotypeRunResult(per_seed, patient_medians, audit)


def make_synthetic_phenotype_table(
    *,
    n_seizures: int = 60,
    n_seeds: int = 2,
    random_seed: int = 4815,
    missing_target: bool = False,
) -> pd.DataFrame:
    """Frozen continuous and three-class targets driven by persistent state."""
    rng = np.random.default_rng(int(random_seed))
    rows = []
    source_hash = hashlib.sha256(b"pre_frozen_synthetic_target").hexdigest()
    latent = rng.normal(size=int(n_seizures))
    continuous = 2.0 * latent + rng.normal(scale=0.25, size=int(n_seizures))
    subtype = np.digitize(latent, [-0.5, 0.5]).astype(int)
    for seed in range(int(n_seeds)):
        for index in range(int(n_seizures)):
            split = (
                "TRAIN" if index < int(0.6 * n_seizures) else
                "SELECT" if index < int(0.8 * n_seizures) else "TEST"
            )
            common = {
                "patient_id": "synthetic_patient",
                "seed": seed,
                "seizure_id": f"sz{index:03d}",
                "split": split,
                "evaluation_tier": "primary_chronological",
                "target_frozen": True,
                "target_provenance": "pre_frozen_synthetic_v1",
                "target_source_sha256": source_hash,
                "baseline__time_of_day": rng.normal(),
                "observation__spectral": rng.normal(),
                "state__persistent_0": latent[index] + rng.normal(scale=0.03),
                "wrong_time__state_0": rng.normal(),
            }
            for name, kind, value in (
                ("early_recruitment_extent", "continuous", continuous[index]),
                ("frozen_subtype", "classification", subtype[index]),
            ):
                row = dict(common)
                row.update({
                    "target_name": name,
                    "target_kind": kind,
                    "target_value": np.nan if missing_target else float(value),
                })
                rows.append(row)
    return pd.DataFrame(rows)
