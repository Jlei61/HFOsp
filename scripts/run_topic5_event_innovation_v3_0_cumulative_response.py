#!/usr/bin/env python3
"""Validation-only repeated-innovation accumulation for Topic 5 v3.0."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["MALLOC_ARENA_MAX"] = "2"

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_topic5_event_innovation_v3_0_phase0 import sha256  # noqa: E402
from scripts.run_topic5_event_innovation_v3_0_local_response import (  # noqa: E402
    fit_final_observer_innovations,
    group_balanced_weights,
    innovation_lookup,
)
from scripts.run_topic5_event_innovation_v3_0_observer import (  # noqa: E402
    sequence_metadata,
)
from scripts.run_topic5_event_innovation_v3_0_phase1_measurement import (  # noqa: E402
    _prepare,
    unit_balanced_dense_fields,
)
from src.topic5_event_innovation_data import (  # noqa: E402
    ContinuitySequence,
    build_cumulative_anchor_splits,
    resolve_cumulative_anchor,
)
from src.topic5_event_innovation_response_v3_0 import (  # noqa: E402
    fit_weighted_local_projection,
    future_precedence_brier,
    masked_innovation_projection,
    masked_rank_field_mse,
    masked_state_projection,
    observable_propagation_gain,
)
from src.topic5_event_innovation_v3_0 import (  # noqa: E402
    RankStateBasis,
    fit_rank_state_basis,
    innovation_alignment,
    masked_window_rank_field,
)
from src.topic5_resource_guard import atomic_write_json, pin_thread_environment  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/topic5_event_innovation_v3_0.yaml"


@dataclass(frozen=True)
class CumulativeRows:
    anchor_event: np.ndarray
    group: np.ndarray
    pre_state: np.ndarray
    future_state: np.ndarray
    cumulative_innovation: np.ndarray
    dose: np.ndarray
    alignment: np.ndarray
    nuisance: np.ndarray
    observed_future_field: np.ndarray
    future_support: np.ndarray
    future_windows: list[np.ndarray]


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def project_exposure_innovations(
    exposure: np.ndarray,
    innovations: Mapping[int, tuple[np.ndarray, np.ndarray]],
    basis: RankStateBasis,
    *,
    event_times: np.ndarray | None = None,
    tau_seconds: float | None = None,
) -> tuple[np.ndarray, float, float] | None:
    values = []
    masks = []
    for event in np.asarray(exposure, dtype=np.int64):
        if int(event) not in innovations:
            return None
        residual, valid = innovations[int(event)]
        values.append(residual)
        masks.append(valid)
    projected, estimable = masked_innovation_projection(
        np.vstack(values), np.vstack(masks), basis
    )
    if not np.all(estimable):
        return None
    weighted = projected
    if tau_seconds is not None:
        tau = float(tau_seconds)
        if tau <= 0 or event_times is None:
            raise ValueError("positive tau and event_times are required together")
        times = np.asarray(event_times, dtype=float)[np.asarray(exposure, dtype=np.int64)]
        weights = np.exp(-(times[-1] - times) / tau)
        weighted = projected * weights[:, None]
    cumulative = weighted.sum(axis=0)
    dose = float(np.linalg.norm(weighted, axis=1).sum())
    return cumulative, dose, innovation_alignment(weighted)


def project_innovation_lookup(
    innovations: Mapping[int, tuple[np.ndarray, np.ndarray]],
    basis: RankStateBasis,
) -> dict[int, np.ndarray]:
    """Project each event once, then reuse it across all m/h/tau windows."""

    events = np.asarray(sorted(innovations), dtype=np.int64)
    residual = np.vstack([innovations[int(event)][0] for event in events])
    valid = np.vstack([innovations[int(event)][1] for event in events])
    projected, estimable = masked_innovation_projection(residual, valid, basis)
    return {
        int(event): projected[row]
        for row, event in enumerate(events)
        if estimable[row]
    }


def cumulative_from_projected_lookup(
    exposure: np.ndarray,
    projected: Mapping[int, np.ndarray],
    *,
    event_times: np.ndarray | None = None,
    tau_seconds: float | None = None,
) -> tuple[np.ndarray, float, float] | None:
    events = np.asarray(exposure, dtype=np.int64)
    if any(int(event) not in projected for event in events):
        return None
    values = np.vstack([projected[int(event)] for event in events])
    weighted = values
    if tau_seconds is not None:
        tau = float(tau_seconds)
        if tau <= 0 or event_times is None:
            raise ValueError("positive tau and event_times are required together")
        times = np.asarray(event_times, dtype=float)[events]
        weighted = values * np.exp(-(times[-1] - times) / tau)[:, None]
    return (
        weighted.sum(axis=0),
        float(np.linalg.norm(weighted, axis=1).sum()),
        innovation_alignment(weighted),
    )


def build_cumulative_rows(
    raw: Mapping[str, Any],
    anchors,
    sequences: Sequence[ContinuitySequence],
    basis: RankStateBasis,
    innovations: Mapping[int, tuple[np.ndarray, np.ndarray]],
    nuisance_by_event: np.ndarray,
    *,
    tau_seconds: float | None = None,
    projected_innovations: Mapping[int, np.ndarray] | None = None,
) -> CumulativeRows:
    anchor_event = []
    group = []
    pre_fields = []
    pre_valid = []
    future_fields = []
    future_support = []
    cumulative = []
    dose = []
    alignment = []
    nuisance = []
    future_windows = []
    for row in range(len(anchors)):
        sequence_index = int(anchors.sequence_index[row])
        pre, exposure, future = resolve_cumulative_anchor(anchors, row, sequences)
        projected = (
            project_exposure_innovations(
                exposure,
                innovations,
                basis,
                event_times=raw["event_time"],
                tau_seconds=tau_seconds,
            )
            if projected_innovations is None
            else cumulative_from_projected_lookup(
                exposure,
                projected_innovations,
                event_times=raw["event_time"],
                tau_seconds=tau_seconds,
            )
        )
        if projected is None:
            continue
        pre_field, pre_support = masked_window_rank_field(
            raw["rank"], raw["participation"], pre
        )
        future_field, support = masked_window_rank_field(
            raw["rank"], raw["participation"], future
        )
        cumulative_value, dose_value, alignment_value = projected
        anchor_event.append(int(exposure[-1]))
        group.append(sequence_index)
        pre_fields.append(pre_field)
        pre_valid.append(pre_support > 0)
        future_fields.append(future_field)
        future_support.append(support)
        cumulative.append(cumulative_value)
        dose.append(dose_value)
        alignment.append(alignment_value)
        nuisance.append(nuisance_by_event[int(exposure[-1])])
        future_windows.append(future)
    if not anchor_event:
        raise ValueError("no cumulative anchor has complete innovations")
    pre_state, pre_ok = masked_state_projection(
        np.vstack(pre_fields), np.vstack(pre_valid), basis
    )
    future_state, future_ok = masked_state_projection(
        np.vstack(future_fields), np.vstack(future_support) > 0, basis
    )
    keep = pre_ok & future_ok
    if np.sum(keep) < max(10, 3 * basis.dimension):
        raise ValueError("insufficient estimable cumulative anchors")
    selected = np.flatnonzero(keep)
    return CumulativeRows(
        anchor_event=np.asarray(anchor_event, dtype=np.int64)[keep],
        group=np.asarray(group, dtype=np.int32)[keep],
        pre_state=pre_state[keep],
        future_state=future_state[keep],
        cumulative_innovation=np.vstack(cumulative)[keep],
        dose=np.asarray(dose, dtype=float)[keep],
        alignment=np.asarray(alignment, dtype=float)[keep],
        nuisance=np.vstack(nuisance)[keep],
        observed_future_field=np.vstack(future_fields)[keep],
        future_support=np.vstack(future_support)[keep],
        future_windows=[future_windows[index] for index in selected],
    )


def matched_cumulative_donor(
    rows: CumulativeRows,
    *,
    seed: int,
    top_k: int = 5,
) -> tuple[np.ndarray, dict[str, float]]:
    """Reassign complete exposure vectors while matching state/dose/alignment."""

    pools, pool_distance = matched_cumulative_donor_pool(rows, top_k=top_k)
    rng = np.random.default_rng(int(seed))
    chosen = np.asarray([rng.choice(pool) for pool in pools], dtype=np.int64)
    counts = np.bincount(chosen, minlength=len(rows.pre_state))
    return rows.cumulative_innovation[chosen], {
        "mean_matched_distance": float(np.mean(pool_distance)),
        "maximum_donor_reuse": int(np.max(counts)),
        "eligible_anchor_fraction": 1.0,
    }


def matched_cumulative_donor_pool(
    rows: CumulativeRows,
    *,
    top_k: int = 5,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Precompute matched complete-exposure donors once for all null draws."""

    pre = rows.pre_state
    progress = rows.nuisance[:, 0]
    state_scale = np.where(np.std(pre, axis=0) > 1e-8, np.std(pre, axis=0), 1.0)
    dose_scale = max(float(np.std(rows.dose)), 1e-8)
    alignment_scale = max(float(np.std(rows.alignment)), 1e-8)
    progress_scale = max(float(np.std(progress)), 1e-8)
    pools = []
    distance = []
    for row in range(len(pre)):
        candidates = np.flatnonzero(rows.group == rows.group[row])
        candidates = candidates[candidates != row]
        if not len(candidates):
            candidates = np.flatnonzero(np.arange(len(pre)) != row)
        if not len(candidates):
            raise ValueError("cumulative donor needs at least two anchors")
        metric = np.sum(((pre[candidates] - pre[row]) / state_scale) ** 2, axis=1)
        metric += ((rows.dose[candidates] - rows.dose[row]) / dose_scale) ** 2
        metric += ((rows.alignment[candidates] - rows.alignment[row]) / alignment_scale) ** 2
        metric += ((progress[candidates] - progress[row]) / progress_scale) ** 2
        order = candidates[np.argsort(metric, kind="stable")[: max(1, min(int(top_k), len(candidates)))]]
        pools.append(order)
        distance.append(float(np.sqrt(np.min(metric))))
    return pools, np.asarray(distance, dtype=float)


def _state_mse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean((np.asarray(observed) - np.asarray(predicted)) ** 2))


def dose_alignment_effect(
    rows: CumulativeRows,
    autonomous_prediction: np.ndarray,
) -> dict[str, float]:
    displacement = np.linalg.norm(rows.future_state - autonomous_prediction, axis=1)
    design = np.column_stack([rows.dose, rows.alignment])
    mean = design.mean(axis=0)
    scale = np.where(design.std(axis=0) > 1e-8, design.std(axis=0), 1.0)
    model = Ridge(alpha=1.0).fit((design - mean) / scale, displacement)
    lower = rows.alignment <= np.quantile(rows.alignment, 0.25)
    upper = rows.alignment >= np.quantile(rows.alignment, 0.75)
    return {
        "dose_coefficient": float(model.coef_[0]),
        "alignment_coefficient": float(model.coef_[1]),
        "high_minus_low_alignment_displacement": float(
            np.mean(displacement[upper]) - np.mean(displacement[lower])
        ),
    }


def evaluate_cumulative(
    raw: Mapping[str, Any],
    basis: RankStateBasis,
    train: CumulativeRows,
    validation: CumulativeRows,
    alphas: Sequence[float],
    *,
    null_draws: int,
    null_seed: int,
) -> dict[str, Any]:
    candidates = []
    weight = group_balanced_weights(train.group)
    for alpha in map(float, alphas):
        fit = fit_weighted_local_projection(
            train.pre_state,
            train.future_state,
            train.cumulative_innovation,
            nuisance=train.nuisance,
            alpha=alpha,
            sample_weight=weight,
        )
        prediction = fit.predict(
            validation.pre_state,
            validation.cumulative_innovation,
            validation.nuisance,
        )
        candidates.append(
            {"alpha": alpha, "validation_state_mse": _state_mse(validation.future_state, prediction), "fit": fit}
        )
    selected = min(candidates, key=lambda row: (row["validation_state_mse"], row["alpha"]))
    fit = selected["fit"]
    true_prediction = fit.predict(
        validation.pre_state,
        validation.cumulative_innovation,
        validation.nuisance,
    )
    autonomous = fit.predict(
        validation.pre_state,
        np.zeros_like(validation.cumulative_innovation),
        validation.nuisance,
    )
    observable = observable_propagation_gain(
        basis,
        validation.observed_future_field,
        validation.future_support,
        validation.future_windows,
        raw["rank"],
        raw["participation"],
        raw["rank"],
        autonomous,
        true_prediction,
    )
    train_backbone = np.broadcast_to(basis.backbone, train.observed_future_field.shape)
    rank_scale = masked_rank_field_mse(
        train_backbone, train.observed_future_field, train.future_support
    )
    pair_scale = future_precedence_brier(
        train_backbone,
        train.future_windows,
        raw["rank"],
        raw["participation"],
        raw["rank"],
    )
    observable["rank_gain_standardized"] = observable["rank_gain"] / max(rank_scale, 1e-12)
    observable["precedence_gain_standardized"] = observable["precedence_gain"] / max(pair_scale, 1e-12)
    observable["propagation_gain_standardized"] = 0.5 * (
        observable["rank_gain_standardized"] + observable["precedence_gain_standardized"]
    )
    matched = []
    audits = []
    true_state_error = _state_mse(validation.future_state, true_prediction)
    donor_pools, donor_distance = matched_cumulative_donor_pool(
        validation, top_k=5
    )
    for draw in range(int(null_draws)):
        rng = np.random.default_rng(int(null_seed) + draw)
        chosen = np.asarray([rng.choice(pool) for pool in donor_pools], dtype=np.int64)
        donor = validation.cumulative_innovation[chosen]
        counts = np.bincount(chosen, minlength=len(validation.pre_state))
        audit = {
            "mean_matched_distance": float(np.mean(donor_distance)),
            "maximum_donor_reuse": int(np.max(counts)),
            "eligible_anchor_fraction": 1.0,
        }
        donor_prediction = fit.predict(
            validation.pre_state, donor, validation.nuisance
        )
        # Euclidean state error is invariant to the orthogonal basis rotation.
        # Observable rank/precedence scoring is retained for the primary
        # true-versus-autonomous contrast and need not be recomputed 100 times.
        matched.append(
            _state_mse(validation.future_state, donor_prediction)
            - true_state_error
        )
        audits.append(audit)
    return {
        "selected_alpha": float(selected["alpha"]),
        "n_train_anchors": int(len(train.anchor_event)),
        "n_validation_anchors": int(len(validation.anchor_event)),
        "state_gain_over_autonomous": _state_mse(validation.future_state, autonomous)
        - _state_mse(validation.future_state, true_prediction),
        "observable": observable,
        "true_minus_matched_cumulative_null_gain": float(np.median(matched)),
        "matched_null_score_coordinate": "orthogonal_rotation_invariant_state_mse",
        "matched_null_q05_q95": np.quantile(matched, [0.05, 0.95]).tolist(),
        "matched_null_mean_distance": float(np.mean([row["mean_matched_distance"] for row in audits])),
        "matched_null_maximum_reuse": int(max(row["maximum_donor_reuse"] for row in audits)),
        "dose_alignment": dose_alignment_effect(validation, autonomous),
        "uniform_weight_within_window_order_invariant": True,
        "within_window_order_shuffle_used_for_primary": False,
        "candidate_table": [
            {"alpha": row["alpha"], "validation_state_mse": row["validation_state_mse"]}
            for row in candidates
        ],
    }


def run_subject(
    subject: str,
    config: Mapping[str, Any],
    phase0_root: Path,
    innovation_root: Path,
) -> dict[str, Any]:
    observer_path = innovation_root / "per_subject" / f"{subject}.json"
    observer = json.loads(observer_path.read_text(encoding="utf-8"))
    if observer.get("status") != "INNOVATION_VALID":
        return {"subject": subject, "status": str(observer.get("status")), "eligible": False}
    raw, split_indices, sequences, phase0_path = _prepare(subject, config, phase0_root)
    selected = observer["observer_selection"]
    dimension = int(selected["dimension"])
    dense_fields, _, dense_weight = unit_balanced_dense_fields(
        raw["rank"], raw["participation"], sequences["train"], window=int(config["primary_horizon"])
    )
    basis = fit_rank_state_basis(dense_fields, dimension, sample_weight=dense_weight)
    with np.load(Path(observer["crossfit_artifact"]), allow_pickle=False) as data:
        train_innovations = innovation_lookup(
            data["event_index"], data["rank_residual"], data["rank_valid"]
        )
    validation_innovations = fit_final_observer_innovations(
        raw, split_indices, sequences, basis, selected, config
    )
    train_projected_innovations = project_innovation_lookup(
        train_innovations, basis
    )
    validation_projected_innovations = project_innovation_lookup(
        validation_innovations, basis
    )
    _, _, train_nuisance = sequence_metadata(sequences["train"], len(raw["rank"]))
    _, _, validation_nuisance = sequence_metadata(sequences["validation"], len(raw["rank"]))
    combinations = {}
    for exposure in map(int, config["cumulative_events"]):
        combinations[str(exposure)] = {}
        for horizon in map(int, config["horizons"]):
            anchors = build_cumulative_anchor_splits(
                sequences,
                pre_events=int(config["primary_pre_events"]),
                exposure_events=exposure,
                horizon=horizon,
            )
            try:
                train_rows = build_cumulative_rows(
                    raw,
                    anchors.train,
                    sequences["train"],
                    basis,
                    train_innovations,
                    train_nuisance,
                    projected_innovations=train_projected_innovations,
                )
                validation_rows = build_cumulative_rows(
                    raw,
                    anchors.validation,
                    sequences["validation"],
                    basis,
                    validation_innovations,
                    validation_nuisance,
                    projected_innovations=validation_projected_innovations,
                )
                combinations[str(exposure)][str(horizon)] = evaluate_cumulative(
                    raw,
                    basis,
                    train_rows,
                    validation_rows,
                    config.get("local_projection_alphas", [0.1, 1.0, 10.0, 100.0]),
                    null_draws=int(config.get("cumulative_null_draws", 100)),
                    null_seed=int(config.get("cumulative_null_seed", 7501)) + 100 * exposure + horizon,
                )
            except ValueError as exc:
                combinations[str(exposure)][str(horizon)] = {
                    "status": "INSUFFICIENT_SUPPORT",
                    "reason": str(exc),
                }
    primary_exposure = int(config.get("primary_cumulative_events", 20))
    primary_horizon = int(config["primary_horizon"])
    primary = combinations[str(primary_exposure)][str(primary_horizon)]
    eligible = "observable" in primary
    iei_sensitivity = {}
    primary_anchors = build_cumulative_anchor_splits(
        sequences,
        pre_events=int(config["primary_pre_events"]),
        exposure_events=primary_exposure,
        horizon=primary_horizon,
    )
    for tau in map(float, config.get("iei_decay_tau_seconds", [])):
        try:
            train_tau = build_cumulative_rows(
                raw,
                primary_anchors.train,
                sequences["train"],
                basis,
                train_innovations,
                train_nuisance,
                tau_seconds=tau,
                projected_innovations=train_projected_innovations,
            )
            validation_tau = build_cumulative_rows(
                raw,
                primary_anchors.validation,
                sequences["validation"],
                basis,
                validation_innovations,
                validation_nuisance,
                tau_seconds=tau,
                projected_innovations=validation_projected_innovations,
            )
            iei_sensitivity[str(tau)] = evaluate_cumulative(
                raw,
                basis,
                train_tau,
                validation_tau,
                config.get("local_projection_alphas", [0.1, 1.0, 10.0, 100.0]),
                null_draws=int(config.get("cumulative_null_draws", 100)),
                null_seed=int(config.get("cumulative_null_seed", 7501)) + int(tau),
            )
        except ValueError as exc:
            iei_sensitivity[str(tau)] = {
                "status": "INSUFFICIENT_SUPPORT",
                "reason": str(exc),
            }
    return {
        "contract": str(config["contract"]),
        "subject": subject,
        "status": "CUMULATIVE_RESPONSE_VALIDATION_COMPLETE" if eligible else "CUMULATIVE_PRIMARY_UNAVAILABLE",
        "eligible": bool(eligible),
        "dimension": dimension,
        "primary_exposure_events": primary_exposure,
        "primary_horizon": primary_horizon,
        "combinations": combinations,
        "iei_decay_sensitivity": iei_sensitivity,
        "phase0_sha256": sha256(phase0_path),
        "observer_record_sha256": sha256(observer_path),
        "human_test_outcomes_read": False,
        "one_step_is_one_complete_event": True,
        "within_event_next_rank_model_fit": False,
    }


def goal3_handoff(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if row.get("eligible")]
    primary = [
        row["combinations"][str(row["primary_exposure_events"])][str(row["primary_horizon"])]
        for row in eligible
    ]
    gain = np.asarray([row["observable"]["propagation_gain_standardized"] for row in primary])
    matched = np.asarray([row["true_minus_matched_cumulative_null_gain"] for row in primary])
    alignment = np.asarray([row["dose_alignment"]["alignment_coefficient"] for row in primary])
    opened = bool(
        len(primary)
        and np.median(gain) > 0
        and np.median(matched) > 0
        and np.median(alignment) > 0
    )
    return {
        "route": "goal3_cumulative_response",
        "status": "OPEN" if opened else "NOT_OPEN",
        "n_eligible": len(primary),
        "median_cumulative_propagation_gain": float(np.median(gain)) if len(gain) else float("nan"),
        "median_true_minus_matched_null_gain": float(np.median(matched)) if len(matched) else float("nan"),
        "median_alignment_coefficient": float(np.median(alignment)) if len(alignment) else float("nan"),
        "favorable_gain": int(np.sum(gain > 0)),
        "favorable_matched": int(np.sum(matched > 0)),
        "favorable_alignment": int(np.sum(alignment > 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--subjects", nargs="*")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    pin_thread_environment(1, disable_cuda=True)
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    phase0_root = ROOT / str(config["output_root"])
    innovation_root = ROOT / str(config["innovation_output_root"])
    innovation_state_path = innovation_root / "innovation_validity.json"
    innovation_state = json.loads(innovation_state_path.read_text(encoding="utf-8"))
    if innovation_state.get("status") != "INNOVATION_VALIDITY_COMPLETE" or innovation_state.get("n_pass") != 34 or innovation_state.get("human_test_outcomes_read") is not False:
        raise SystemExit("34-patient validation-only innovation state is not complete")
    cohort = [row["subject"] for row in innovation_state["patients"]]
    subjects = cohort if not args.subjects else list(map(str, args.subjects))
    output = (
        args.output_dir
        if args.output_dir is not None and args.output_dir.is_absolute()
        else ROOT / (args.output_dir or Path(str(config.get(
            "cumulative_output_root",
            "results/topic5_event_innovation_impulse_response/v3_0/cumulative_response_validation_only",
        ))))
    )
    rows = []
    failures = []
    for subject in subjects:
        try:
            row = run_subject(subject, config, phase0_root, innovation_root)
            rows.append(row)
            atomic_write_json(output / "per_subject" / f"{subject}.json", _jsonable(row))
            print(subject, row["status"], flush=True)
        except Exception as exc:
            failures.append({"subject": subject, "error": f"{type(exc).__name__}: {exc}"})
            print(subject, "FAIL", exc, flush=True)
    handoff = goal3_handoff(rows)
    summary = []
    for row in rows:
        primary = row.get("combinations", {}).get(str(row.get("primary_exposure_events")), {}).get(str(row.get("primary_horizon")), {})
        summary.append({
            "subject": row["subject"],
            "status": row["status"],
            "eligible": row.get("eligible", False),
            "propagation_gain": primary.get("observable", {}).get("propagation_gain_standardized", np.nan),
            "true_minus_matched": primary.get("true_minus_matched_cumulative_null_gain", np.nan),
            "dose_coefficient": primary.get("dose_alignment", {}).get("dose_coefficient", np.nan),
            "alignment_coefficient": primary.get("dose_alignment", {}).get("alignment_coefficient", np.nan),
            "n_validation_anchors": primary.get("n_validation_anchors", 0),
        })
    if summary:
        _atomic_csv(output / "dose_response.csv", pd.DataFrame(summary))
    if failures:
        _atomic_csv(output / "failures.csv", pd.DataFrame(failures))
    state = {
        "contract": str(config["contract"]),
        "status": "CUMULATIVE_RESPONSE_VALIDATION_COMPLETE" if not failures else "CUMULATIVE_RESPONSE_VALIDATION_FAIL_CLOSED",
        "n_requested": len(subjects),
        "n_completed": len(rows),
        "n_failed": len(failures),
        "goal3_handoff": handoff,
        "patients": rows,
        "failures": failures,
        "uniform_weight_order_null_revision": {
            "within_window_order_is_mathematically_invariant": True,
            "primary_null": "matched_complete_exposure_reassignment",
            "within_window_order_reserved_for_iei_decay_sensitivity": True,
        },
        "innovation_state_sha256": sha256(innovation_state_path),
        "config_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__)),
        "human_test_outcomes_read": False,
        "one_step_is_one_complete_event": True,
        "within_event_next_rank_model_fit": False,
    }
    atomic_write_json(output / "cumulative_response_state.json", _jsonable(state))
    atomic_write_json(output / "GOAL3_HANDOFF_STATE.json", _jsonable(handoff))
    atomic_write_json(
        output / "iei_decay_sensitivity.json",
        _jsonable(
            {
                "status": "VALIDATION_ONLY_IEI_SENSITIVITY_COMPLETE",
                "tau_seconds": list(map(float, config.get("iei_decay_tau_seconds", []))),
                "patients": [
                    {
                        "subject": row["subject"],
                        "status": row["status"],
                        "sensitivity": row.get("iei_decay_sensitivity", {}),
                    }
                    for row in rows
                ],
                "human_test_outcomes_read": False,
                "biological_time_constant_claimed": False,
            }
        ),
    )
    print(json.dumps({"status": state["status"], "handoff": handoff}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
