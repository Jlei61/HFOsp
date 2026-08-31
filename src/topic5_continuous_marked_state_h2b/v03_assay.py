"""Outcome-blind semi-synthetic assay for the H2b v0.3 estimators.

The template is a frozen interictal decoder trajectory on its real recorded
coverage. Synthetic seizure times are the only generated outcome. T (state
increment), M (persistent-history residual), lag degradation, and geometry
recovery are deliberately separate quantities.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


WORLDS = (
    "null", "observation_only", "persistent_state", "clock_confounded",
    "basin_gating", "directed_approach", "abrupt_transition",
)
ALPHA_GRID = (0.1, 1.0, 10.0, 100.0)


@dataclass(frozen=True)
class AssayTemplate:
    time_epoch: np.ndarray
    segment: np.ndarray
    base: np.ndarray
    observation_axis: np.ndarray
    persistent_decoder: np.ndarray
    memoryless_decoder: np.ndarray
    lagged_persistent_decoder: np.ndarray
    lag_available: np.ndarray
    persistent_axis: np.ndarray
    basin_score: np.ndarray
    approach_score: np.ndarray
    abrupt_score: np.ndarray
    onset_tod_log_weight: np.ndarray
    n_seizures: int
    horizon_minutes: float = 30.0
    lag_minutes: float = 60.0

    def validate(self) -> None:
        n = len(self.time_epoch)
        arrays = (
            self.segment, self.base, self.observation_axis,
            self.persistent_decoder, self.memoryless_decoder,
            self.lagged_persistent_decoder, self.lag_available,
            self.persistent_axis, self.basin_score, self.approach_score,
            self.abrupt_score, self.onset_tod_log_weight,
        )
        if any(len(value) != n for value in arrays):
            raise ValueError("assay template arrays disagree")
        if n < 50 or int(self.n_seizures) < 6:
            raise ValueError("assay template has insufficient support")
        if np.any(np.diff(self.time_epoch) < 0):
            raise ValueError("assay template must be chronological")
        if not all(np.isfinite(value).all() for value in arrays):
            raise ValueError("assay template contains non-finite values")
        if self.persistent_decoder.shape != self.memoryless_decoder.shape:
            raise ValueError("persistent and memoryless decoder shapes disagree")


def _z(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    centre = np.mean(array, axis=0, keepdims=True)
    scale = np.std(array, axis=0, keepdims=True)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return (array - centre) / scale


def _drop_constant(value: np.ndarray) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    keep = np.std(matrix, axis=0) > 1e-8
    return matrix[:, keep] if np.any(keep) else np.zeros((len(matrix), 1))


def _pca_scores(
    value: np.ndarray, *, maximum_components: int = 8,
    variance_fraction: float = 0.95,
) -> np.ndarray:
    matrix = _z(_drop_constant(value))
    _, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    if not len(singular):
        return np.zeros((len(matrix), 1), dtype=np.float64)
    variance = singular ** 2
    cumulative = np.cumsum(variance) / max(float(np.sum(variance)), 1e-12)
    count = int(np.searchsorted(cumulative, float(variance_fraction)) + 1)
    count = max(1, min(count, int(maximum_components), vt.shape[0]))
    scores = matrix @ vt[:count].T
    for column in range(scores.shape[1]):
        largest = int(np.argmax(np.abs(vt[column])))
        if vt[column, largest] < 0:
            scores[:, column] *= -1.0
    return _z(scores)


def _pc1(value: np.ndarray) -> np.ndarray:
    return _pca_scores(value, maximum_components=1)[:, 0]


def _tod_log_weight(time_epoch: np.ndarray, observed_onsets: np.ndarray) -> np.ndarray:
    """Circular kernel density that preserves empirical seizure clock support."""
    onset = np.asarray(observed_onsets, dtype=np.float64)
    if not len(onset):
        return np.zeros(len(time_epoch), dtype=np.float64)
    phase = 2.0 * np.pi * ((np.asarray(time_epoch) % 86400.0) / 86400.0)
    observed = 2.0 * np.pi * ((onset % 86400.0) / 86400.0)
    density = np.mean(
        np.exp(2.0 * np.cos(phase[:, None] - observed[None, :])), axis=1,
    )
    density = np.maximum(density, 0.05 * float(np.mean(density)))
    return np.log(density) - float(np.mean(np.log(density)))


def _lagged_rows(
    time: np.ndarray, segment: np.ndarray, lag_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    index = np.zeros(len(time), dtype=np.int64)
    available = np.zeros(len(time), dtype=bool)
    tolerance = 7.5 * 60.0
    for label in np.unique(segment):
        rows = np.flatnonzero(segment == label)
        rows = rows[np.argsort(time[rows], kind="stable")]
        target = time[rows] - float(lag_seconds)
        position = np.searchsorted(time[rows], target, side="right") - 1
        valid = position >= 0
        clipped = np.maximum(position, 0)
        donor = rows[clipped]
        valid &= np.abs(time[donor] - target) <= tolerance
        index[rows] = donor
        available[rows] = valid
    return index, available


def build_template(
    *,
    time_epoch: np.ndarray,
    segment: np.ndarray,
    deterministic_history: np.ndarray,
    current_observation: np.ndarray,
    persistent_decoder: np.ndarray,
    memoryless_decoder: np.ndarray,
    n_seizures: int,
    observed_seizure_onsets: np.ndarray | None = None,
    minimum_spacing_seconds: float = 300.0,
    horizon_minutes: float = 30.0,
    lag_minutes: float = 60.0,
) -> AssayTemplate:
    """Build a five-minute decoder-metric template without bridging gaps."""
    time = np.asarray(time_epoch, dtype=np.float64)
    group = np.asarray(segment, dtype=np.int64)
    history = np.asarray(deterministic_history, dtype=np.float64)
    observation = np.asarray(current_observation, dtype=np.float64)
    persistent = np.asarray(persistent_decoder, dtype=np.float64)
    memoryless = np.asarray(memoryless_decoder, dtype=np.float64)
    if not all(len(value) == len(time) for value in (
        group, history, observation, persistent, memoryless,
    )):
        raise ValueError("raw assay arrays disagree")
    keep: list[int] = []
    for label in np.unique(group):
        rows = np.flatnonzero(group == label)
        rows = rows[np.argsort(time[rows], kind="stable")]
        cursor = -np.inf
        for row in rows:
            if float(time[row]) >= cursor - 1e-9:
                keep.append(int(row))
                cursor = float(time[row]) + float(minimum_spacing_seconds)
    selected = np.asarray(sorted(keep, key=lambda row: time[row]), dtype=np.int64)
    time, group = time[selected], group[selected]
    history, observation = history[selected], observation[selected]
    persistent, memoryless = persistent[selected], memoryless[selected]

    observation_code = _pca_scores(observation, maximum_components=8)
    base = _z(_drop_constant(np.column_stack([
        history[:, :min(11, history.shape[1])], observation_code,
    ])))
    active = np.std(persistent, axis=0) > 1e-8
    if not np.any(active):
        active = np.ones(persistent.shape[1], dtype=bool)
    centre = np.mean(persistent[:, active], axis=0)
    scale = np.std(persistent[:, active], axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    persistent = (persistent[:, active] - centre) / scale
    memoryless = (memoryless[:, active] - centre) / scale
    residual_axis = _pc1(persistent - memoryless)

    centred = persistent - np.mean(persistent, axis=0, keepdims=True)
    radius = np.linalg.norm(centred, axis=1)
    basin = -_z(radius)
    approach = np.zeros(len(time), dtype=np.float64)
    abrupt = np.zeros(len(time), dtype=np.float64)
    for label in np.unique(group):
        rows = np.flatnonzero(group == label)
        rows = rows[np.argsort(time[rows], kind="stable")]
        if len(rows) < 2:
            continue
        dt = np.maximum(np.diff(time[rows]) / 60.0, 1e-6)
        approach[rows[1:]] = -(radius[rows[1:]] - radius[rows[:-1]]) / dt
        abrupt[rows[1:]] = np.linalg.norm(
            centred[rows[1:]] - centred[rows[:-1]], axis=1,
        ) / np.sqrt(dt)
    lag_index, lag_available = _lagged_rows(
        time, group, float(lag_minutes) * 60.0,
    )
    value = AssayTemplate(
        time_epoch=time, segment=group, base=base,
        observation_axis=_pc1(observation_code),
        persistent_decoder=persistent, memoryless_decoder=memoryless,
        lagged_persistent_decoder=persistent[lag_index],
        lag_available=lag_available, persistent_axis=_z(residual_axis),
        basin_score=_z(basin), approach_score=_z(approach),
        abrupt_score=_z(abrupt), onset_tod_log_weight=_tod_log_weight(
            time, np.asarray(
                observed_seizure_onsets if observed_seizure_onsets is not None
                else [], dtype=np.float64,
            ),
        ), n_seizures=int(n_seizures),
        horizon_minutes=float(horizon_minutes), lag_minutes=float(lag_minutes),
    )
    value.validate()
    return value


def _cluster_noise(template: AssayTemplate, rng: np.random.Generator) -> np.ndarray:
    result = np.zeros(len(template.time_epoch), dtype=np.float64)
    for label in np.unique(template.segment):
        rows = np.flatnonzero(template.segment == label)
        rows = rows[np.argsort(template.time_epoch[rows], kind="stable")]
        for position, row in enumerate(rows):
            previous = result[rows[position - 1]] if position else 0.0
            result[row] = 0.85 * previous + rng.normal(scale=np.sqrt(1 - 0.85 ** 2))
    return _z(result)


def _lead_rows_for_onsets(template: AssayTemplate) -> tuple[np.ndarray, np.ndarray]:
    onset_rows: list[int] = []
    lead_rows: list[int] = []
    horizon = float(template.horizon_minutes) * 60.0
    tolerance = 7.5 * 60.0
    for label in np.unique(template.segment):
        rows = np.flatnonzero(template.segment == label)
        rows = rows[np.argsort(template.time_epoch[rows], kind="stable")]
        target = template.time_epoch[rows] - horizon
        position = np.searchsorted(template.time_epoch[rows], target, side="right") - 1
        for onset, donor_position, expected in zip(rows, position, target):
            if donor_position < 0:
                continue
            lead = int(rows[int(donor_position)])
            if abs(float(template.time_epoch[lead]) - float(expected)) <= tolerance:
                onset_rows.append(int(onset))
                lead_rows.append(lead)
    return np.asarray(onset_rows, dtype=np.int64), np.asarray(lead_rows, dtype=np.int64)


def _weighted_seizure_sample(
    score: np.ndarray, template: AssayTemplate, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample fixed-count onsets from valid within-segment lead anchors."""
    candidates, lead = _lead_rows_for_onsets(template)
    if len(candidates) < int(template.n_seizures):
        raise ValueError("insufficient within-segment synthetic onset support")
    gumbel = -np.log(-np.log(np.clip(
        rng.uniform(size=len(candidates)), 1e-12, 1 - 1e-12,
    )))
    priority = (
        np.asarray(score, dtype=np.float64)[lead]
        + template.onset_tod_log_weight[candidates] + gumbel
    )
    order = np.argsort(priority)[::-1]
    selected: list[int] = []
    minimum_gap = 5.0 * 60.0
    for candidate_position in order:
        candidate = int(candidates[candidate_position])
        if all(
            template.segment[candidate] != template.segment[other]
            or abs(template.time_epoch[candidate] - template.time_epoch[other])
            >= minimum_gap for other in selected
        ):
            selected.append(candidate)
            if len(selected) == int(template.n_seizures):
                break
    selected = sorted(selected)
    if len(selected) != int(template.n_seizures):
        raise ValueError("fixed seizure count could not be placed")
    lead_lookup = {int(onset): int(anchor) for onset, anchor in zip(candidates, lead)}
    return (
        np.asarray(selected, dtype=np.int64),
        np.asarray([lead_lookup[row] for row in selected], dtype=np.int64),
    )


def simulate_world(
    template: AssayTemplate, world: str, rng: np.random.Generator, *,
    effect_scale: float = 1.75,
) -> dict[str, np.ndarray]:
    if world not in WORLDS:
        raise ValueError(f"unknown assay world: {world}")
    score = 0.55 * _cluster_noise(template, rng)
    if world == "observation_only":
        score += effect_scale * template.observation_axis
    elif world == "persistent_state":
        score += effect_scale * template.persistent_axis
    elif world == "clock_confounded":
        clock = template.base[:, 8] if template.base.shape[1] > 8 else template.base[:, 0]
        score += effect_scale * clock
    elif world == "basin_gating":
        score += effect_scale * template.basin_score
    elif world == "directed_approach":
        score += effect_scale * template.approach_score
    elif world == "abrupt_transition":
        score += effect_scale * template.abrupt_score
    onset, lead = _weighted_seizure_sample(score, template, rng)
    outcome = np.zeros(len(template.time_epoch), dtype=np.int8)
    horizon = float(template.horizon_minutes) * 60.0
    for index in onset:
        same = template.segment == template.segment[index]
        outcome[
            same & (template.time_epoch < template.time_epoch[index])
            & (template.time_epoch >= template.time_epoch[index] - horizon)
        ] = 1
    return {
        "outcome": outcome, "onset_index": onset, "lead_index": lead,
        "generator_score": score,
    }


def _logistic_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    matrix = np.column_stack([np.ones(len(x)), np.asarray(x, dtype=np.float64)])
    target = np.asarray(y, dtype=np.float64)
    beta = np.zeros(matrix.shape[1], dtype=np.float64)
    prevalence = np.clip(np.mean(target), 1e-5, 1 - 1e-5)
    beta[0] = np.log(prevalence / (1.0 - prevalence))
    penalty = np.eye(matrix.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    for _ in range(40):
        linear = np.clip(matrix @ beta, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-linear))
        weight = np.maximum(probability * (1.0 - probability), 1e-5)
        gradient = matrix.T @ (target - probability) - penalty @ beta
        hessian = (matrix.T * weight) @ matrix + penalty
        step = np.linalg.solve(hessian + np.eye(len(beta)) * 1e-8, gradient)
        beta += step
        if float(np.linalg.norm(step)) < 1e-6:
            break
    return beta


def _logistic_predict(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    matrix = np.column_stack([np.ones(len(x)), np.asarray(x, dtype=np.float64)])
    linear = np.clip(matrix @ beta, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-linear))


def _logloss(y: np.ndarray, probability: np.ndarray) -> float:
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-8, 1 - 1e-8)
    target = np.asarray(y, dtype=np.float64)
    return float(-np.mean(target * np.log(p) + (1 - target) * np.log(1 - p)))


def _fit_with_inner_alpha(
    x: np.ndarray, y: np.ndarray, alpha_grid: Sequence[float],
) -> tuple[np.ndarray, float]:
    n = len(x)
    split = max(20, int(np.floor(0.80 * n)))
    if split >= n - 10 or len(np.unique(y[:split])) < 2:
        alpha = float(alpha_grid[len(alpha_grid) // 2])
        return _logistic_fit(x, y, alpha), alpha
    rows = []
    for alpha in alpha_grid:
        beta = _logistic_fit(x[:split], y[:split], float(alpha))
        rows.append((_logloss(y[split:], _logistic_predict(x[split:], beta)), float(alpha)))
    alpha = min(rows)[1]
    return _logistic_fit(x, y, alpha), alpha


def _standardise_train_test(
    train: np.ndarray, test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    centre = np.mean(train, axis=0, dtype=np.float64)
    scale = np.std(train, axis=0, dtype=np.float64)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return (train - centre) / scale, (test - centre) / scale


def residualise_train_test(
    persistent_train: np.ndarray, persistent_test: np.ndarray,
    covariate_train: np.ndarray, covariate_test: np.ndarray, *,
    alpha: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the persistent-history residual map on outer training only."""
    x_train, x_test = _standardise_train_test(covariate_train, covariate_test)
    y_train = np.asarray(persistent_train, dtype=np.float64)
    y_test = np.asarray(persistent_test, dtype=np.float64)
    design = np.column_stack([np.ones(len(x_train)), x_train])
    penalty = np.eye(design.shape[1]) * float(alpha)
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ y_train)
    train_residual = y_train - design @ coef
    test_residual = y_test - np.column_stack([np.ones(len(x_test)), x_test]) @ coef
    return train_residual, test_residual


def _fit_loss(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray,
    y_test: np.ndarray, alpha_grid: Sequence[float],
) -> tuple[float, float]:
    train, test = _standardise_train_test(x_train, x_test)
    beta, alpha = _fit_with_inner_alpha(train, y_train, alpha_grid)
    return _logloss(y_test, _logistic_predict(test, beta)), alpha


def evaluate_prequential_transfer(
    template: AssayTemplate, outcome: np.ndarray, onset_index: np.ndarray, *,
    initial_k: int, alpha_grid: Sequence[float] = ALPHA_GRID,
) -> dict[str, Any]:
    """Predict later seizure intervals using only strictly earlier events."""
    y = np.asarray(outcome, dtype=np.int8)
    onset = np.asarray(onset_index, dtype=np.int64)
    base = np.asarray(template.base, dtype=np.float64)
    persistent = np.asarray(template.persistent_decoder, dtype=np.float64)
    memoryless = np.asarray(template.memoryless_decoder, dtype=np.float64)
    fold_rows = []
    for event_position in range(int(initial_k), len(onset)):
        cutoff = int(onset[event_position - 1])
        stop = int(onset[event_position])
        train = np.arange(cutoff + 1)
        test = np.arange(cutoff + 1, stop + 1)
        minimum_train = max(30, base.shape[1] + persistent.shape[1] + 10)
        if len(train) < minimum_train or len(test) < 1 or len(np.unique(y[train])) < 2:
            continue
        m1_loss, m1_alpha = _fit_loss(
            base[train], y[train], base[test], y[test], alpha_grid,
        )
        m2_loss, m2_alpha = _fit_loss(
            np.column_stack([base[train], persistent[train]]), y[train],
            np.column_stack([base[test], persistent[test]]), y[test], alpha_grid,
        )
        m3_cov_train = np.column_stack([base[train], memoryless[train]])
        m3_cov_test = np.column_stack([base[test], memoryless[test]])
        m3_loss, m3_alpha = _fit_loss(
            m3_cov_train, y[train], m3_cov_test, y[test], alpha_grid,
        )
        residual_train, residual_test = residualise_train_test(
            persistent[train], persistent[test], m3_cov_train, m3_cov_test,
        )
        m4_loss, m4_alpha = _fit_loss(
            np.column_stack([m3_cov_train, residual_train]), y[train],
            np.column_stack([m3_cov_test, residual_test]), y[test], alpha_grid,
        )
        lag_train = train[template.lag_available[train]]
        lag_test = test[template.lag_available[test]]
        current_same_loss = lag_loss = None
        if (
            len(lag_train) >= minimum_train and len(lag_test) >= 1
            and len(np.unique(y[lag_train])) >= 2
        ):
            current_same_loss, _ = _fit_loss(
                np.column_stack([base[lag_train], persistent[lag_train]]),
                y[lag_train],
                np.column_stack([base[lag_test], persistent[lag_test]]),
                y[lag_test], alpha_grid,
            )
            lag_loss, _ = _fit_loss(
                np.column_stack([
                    base[lag_train], template.lagged_persistent_decoder[lag_train],
                ]), y[lag_train],
                np.column_stack([
                    base[lag_test], template.lagged_persistent_decoder[lag_test],
                ]), y[lag_test], alpha_grid,
            )
        fold_rows.append({
            "heldout_onset_rank": int(event_position + 1),
            "train_rows": int(len(train)), "test_rows": int(len(test)),
            "M1_base_logloss": m1_loss, "M2_persistent_logloss": m2_loss,
            "M3_memoryless_logloss": m3_loss, "M4_residual_logloss": m4_loss,
            "current_same_rows_logloss": current_same_loss,
            "lagged_same_rows_logloss": lag_loss,
            "M1_alpha": m1_alpha, "M2_alpha": m2_alpha,
            "M3_alpha": m3_alpha, "M4_alpha": m4_alpha,
        })
    if not fold_rows:
        return {"status": "NOT_ESTIMABLE", "folds": []}
    weights = np.asarray([row["test_rows"] for row in fold_rows], dtype=np.float64)
    aggregate = {}
    for key in (
        "M1_base_logloss", "M2_persistent_logloss", "M3_memoryless_logloss",
        "M4_residual_logloss",
    ):
        aggregate[key] = float(np.average([row[key] for row in fold_rows], weights=weights))
    lag_rows = [row for row in fold_rows if row["lagged_same_rows_logloss"] is not None]
    if lag_rows:
        lag_weights = np.asarray([row["test_rows"] for row in lag_rows], dtype=np.float64)
        current_same = float(np.average(
            [row["current_same_rows_logloss"] for row in lag_rows], weights=lag_weights,
        ))
        lagged_same = float(np.average(
            [row["lagged_same_rows_logloss"] for row in lag_rows], weights=lag_weights,
        ))
        lag_degradation = (
            (lagged_same - current_same) / current_same if current_same > 1e-12 else None
        )
    else:
        current_same = lagged_same = lag_degradation = None
    t_value = (
        (aggregate["M1_base_logloss"] - aggregate["M2_persistent_logloss"])
        / aggregate["M1_base_logloss"] if aggregate["M1_base_logloss"] > 1e-12 else None
    )
    m_value = (
        (aggregate["M3_memoryless_logloss"] - aggregate["M4_residual_logloss"])
        / aggregate["M3_memoryless_logloss"]
        if aggregate["M3_memoryless_logloss"] > 1e-12 else None
    )
    direct_value = (
        (aggregate["M3_memoryless_logloss"] - aggregate["M2_persistent_logloss"])
        / aggregate["M3_memoryless_logloss"]
        if aggregate["M3_memoryless_logloss"] > 1e-12 else None
    )
    return {
        "status": "COMPLETE", "n_oof_seizures": len(fold_rows),
        "folds": fold_rows, **aggregate,
        "T_relative_logloss_improvement": t_value,
        "M_relative_logloss_improvement": m_value,
        "persistent_vs_memoryless_relative_improvement": direct_value,
        "lag_minutes": float(template.lag_minutes),
        "current_same_rows_logloss": current_same,
        "lagged_same_rows_logloss": lagged_same,
        "lag_degradation": lag_degradation,
    }


def geometry_recovery(
    template: AssayTemplate, outcome: np.ndarray, lead_index: np.ndarray,
) -> dict[str, Any]:
    case = np.asarray(lead_index, dtype=np.int64)
    control = np.flatnonzero(np.asarray(outcome) == 0)
    scores = {}
    for name, value in (
        ("basin_gating", template.basin_score),
        ("directed_approach", template.approach_score),
        ("abrupt_transition", template.abrupt_score),
    ):
        if not len(case) or not len(control):
            effect = None
        else:
            pooled = float(np.std(value[control]))
            effect = (
                float((np.mean(value[case]) - np.mean(value[control])) / pooled)
                if pooled > 1e-8 else None
            )
        scores[name] = effect
    finite = {name: value for name, value in scores.items() if value is not None}
    winner = max(finite, key=lambda name: finite[name]) if finite else None
    return {"n_lead_states": int(len(case)), "scores": scores, "winning_family": winner}


def run_replicate(
    template: AssayTemplate, world: str, seed: int, *, initial_k: int,
    effect_scale: float = 1.75,
) -> dict[str, Any]:
    generated = simulate_world(
        template, world, np.random.default_rng(int(seed)), effect_scale=effect_scale,
    )
    transfer = evaluate_prequential_transfer(
        template, generated["outcome"], generated["onset_index"],
        initial_k=int(initial_k),
    )
    geometry = geometry_recovery(
        template, generated["outcome"], generated["lead_index"],
    )
    return {
        "world": world, "seed": int(seed), "initial_k": int(initial_k),
        "n_simulated_seizures": int(len(generated["onset_index"])),
        "n_positive_grid_rows": int(np.sum(generated["outcome"])),
        "transfer": transfer, "geometry": geometry,
    }


def wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054,
) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    p = float(successes) / float(total)
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    half = z * np.sqrt(
        p * (1 - p) / total + z * z / (4.0 * total ** 2)
    ) / denominator
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))
