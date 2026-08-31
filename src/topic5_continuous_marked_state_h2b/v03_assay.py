"""Semi-synthetic assay for H2b v0.3 transfer and geometry diagnostics.

The assay reuses a real patient's recorded grid, coverage segments, clocks and
frozen state trajectory.  It changes only the synthetic seizure generator.
No real seizure outcome is used to fit or score the synthetic probes.
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
    persistent_increment: np.ndarray
    observation_axis: np.ndarray
    persistent_axis: np.ndarray
    basin_score: np.ndarray
    approach_score: np.ndarray
    abrupt_score: np.ndarray
    n_seizures: int
    horizon_minutes: float = 30.0

    def validate(self) -> None:
        n = len(self.time_epoch)
        arrays = (
            self.segment, self.base, self.persistent_increment,
            self.observation_axis, self.persistent_axis, self.basin_score,
            self.approach_score, self.abrupt_score,
        )
        if any(len(value) != n for value in arrays):
            raise ValueError("assay template arrays disagree")
        if n < 50 or int(self.n_seizures) < 6:
            raise ValueError("assay template has insufficient support")
        if not np.all(np.diff(self.time_epoch) >= 0):
            raise ValueError("assay template must be chronological")
        if not all(np.isfinite(value).all() for value in arrays):
            raise ValueError("assay template contains non-finite values")


def _z(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    centre = np.mean(array, axis=0, keepdims=True)
    scale = np.std(array, axis=0, keepdims=True)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return (array - centre) / scale


def _pc1(value: np.ndarray) -> np.ndarray:
    matrix = _z(np.asarray(value, dtype=np.float64))
    if matrix.ndim == 1:
        return matrix
    if not matrix.shape[1]:
        return np.zeros(len(matrix), dtype=np.float64)
    _, _, vt = np.linalg.svd(matrix, full_matrices=False)
    axis = matrix @ vt[0]
    # Deterministic orientation.
    largest = int(np.argmax(np.abs(vt[0])))
    return axis if vt[0, largest] >= 0 else -axis


def build_template(
    *,
    time_epoch: np.ndarray,
    segment: np.ndarray,
    deterministic_history: np.ndarray,
    persistent_state: np.ndarray,
    memoryless_state: np.ndarray,
    n_seizures: int,
    minimum_spacing_seconds: float = 300.0,
) -> AssayTemplate:
    """Downsample a real state cache without bridging coverage segments."""
    time = np.asarray(time_epoch, dtype=np.float64)
    group = np.asarray(segment, dtype=np.int64)
    history = np.asarray(deterministic_history, dtype=np.float64)
    persistent = np.asarray(persistent_state, dtype=np.float64)
    memoryless = np.asarray(memoryless_state, dtype=np.float64)
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
    history, persistent, memoryless = (
        history[selected], persistent[selected], memoryless[selected]
    )
    # The first eleven frozen history fields are scalar timing/load/clock
    # summaries.  State and observation axes are added explicitly below.
    base = np.column_stack([history[:, :min(11, history.shape[1])], memoryless])
    increment = persistent - memoryless
    observation_axis = _pc1(memoryless)
    persistent_axis = _pc1(increment)
    centred = _z(persistent)
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
    value = AssayTemplate(
        time_epoch=time, segment=group, base=_z(base),
        persistent_increment=_z(increment),
        observation_axis=_z(observation_axis),
        persistent_axis=_z(persistent_axis), basin_score=_z(basin),
        approach_score=_z(approach), abrupt_score=_z(abrupt),
        n_seizures=int(n_seizures),
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


def _weighted_seizure_sample(
    score: np.ndarray,
    template: AssayTemplate,
    rng: np.random.Generator,
    *,
    eligible: np.ndarray | None = None,
) -> np.ndarray:
    """Sample a fixed seizure count while retaining empirical coverage."""
    gumbel = -np.log(-np.log(np.clip(rng.uniform(size=len(score)), 1e-12, 1 - 1e-12)))
    priority = np.asarray(score, dtype=np.float64) + gumbel
    if eligible is not None:
        priority = np.where(np.asarray(eligible, dtype=bool), priority, -np.inf)
    order = np.argsort(priority)[::-1]
    selected: list[int] = []
    minimum_gap = 5.0 * 60.0
    for candidate in order:
        if not np.isfinite(priority[candidate]):
            continue
        if all(
            template.segment[candidate] != template.segment[other]
            or abs(template.time_epoch[candidate] - template.time_epoch[other]) >= minimum_gap
            for other in selected
        ):
            selected.append(int(candidate))
            if len(selected) == min(int(template.n_seizures), len(template.time_epoch)):
                break
    return np.asarray(sorted(selected), dtype=np.int64)


def lead_source_index(template: AssayTemplate) -> np.ndarray:
    """Latest same-segment state available at the frozen pre-seizure lead."""
    result = np.full(len(template.time_epoch), -1, dtype=np.int64)
    horizon = float(template.horizon_minutes) * 60.0
    for label in np.unique(template.segment):
        rows = np.flatnonzero(template.segment == label)
        rows = rows[np.argsort(template.time_epoch[rows], kind="stable")]
        position = np.searchsorted(
            template.time_epoch[rows], template.time_epoch[rows] - horizon,
            side="right",
        ) - 1
        valid = position >= 0
        result[rows[valid]] = rows[position[valid]]
    return result


def simulate_world(
    template: AssayTemplate,
    world: str,
    rng: np.random.Generator,
    *,
    effect_scale: float = 1.75,
) -> dict[str, np.ndarray]:
    if world not in WORLDS:
        raise ValueError(f"unknown assay world: {world}")
    cluster = _cluster_noise(template, rng)
    lead = lead_source_index(template)
    eligible = lead >= 0
    safe = np.clip(lead, 0, len(lead) - 1)
    score = 0.55 * cluster[safe]
    if world == "observation_only":
        score += effect_scale * template.observation_axis[safe]
    elif world == "persistent_state":
        score += effect_scale * template.persistent_axis[safe]
    elif world == "clock_confounded":
        # Frozen history columns 8/9 are local time-of-day sin/cos.
        clock = template.base[:, 8] if template.base.shape[1] > 8 else template.base[:, 0]
        score += effect_scale * clock[safe]
    elif world == "basin_gating":
        score += effect_scale * template.basin_score[safe]
    elif world == "directed_approach":
        score += effect_scale * template.approach_score[safe]
    elif world == "abrupt_transition":
        score += effect_scale * template.abrupt_score[safe]
    onset = _weighted_seizure_sample(
        score, template, rng, eligible=eligible,
    )
    outcome = np.zeros(len(template.time_epoch), dtype=np.int8)
    horizon = float(template.horizon_minutes) * 60.0
    for index in onset:
        same = template.segment == template.segment[index]
        outcome[
            same & (template.time_epoch < template.time_epoch[index])
            & (template.time_epoch >= template.time_epoch[index] - horizon)
        ] = 1
    return {"outcome": outcome, "onset_index": onset, "generator_score": score}


def _logistic_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    matrix = np.column_stack([np.ones(len(x)), np.asarray(x, dtype=np.float64)])
    target = np.asarray(y, dtype=np.float64)
    beta = np.zeros(matrix.shape[1], dtype=np.float64)
    prevalence = np.clip(np.mean(target), 1e-5, 1 - 1e-5)
    beta[0] = np.log(prevalence / (1.0 - prevalence))
    penalty = np.eye(matrix.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    for _ in range(30):
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


def evaluate_prequential_transfer(
    template: AssayTemplate,
    outcome: np.ndarray,
    onset_index: np.ndarray,
    *,
    initial_k: int,
    alpha_grid: Sequence[float] = ALPHA_GRID,
) -> dict[str, Any]:
    """Predict each later seizure interval using only earlier intervals."""
    y = np.asarray(outcome, dtype=np.int8)
    onset = np.asarray(onset_index, dtype=np.int64)
    base = np.asarray(template.base, dtype=np.float64)
    full = np.column_stack([base, template.persistent_increment])
    fold_rows = []
    for event_position in range(int(initial_k), len(onset)):
        cutoff = int(onset[event_position - 1])
        stop = int(onset[event_position])
        train = np.arange(cutoff + 1)
        test = np.arange(cutoff + 1, stop + 1)
        if len(train) < 20 or len(test) < 1 or len(np.unique(y[train])) < 2:
            continue
        base_train, base_test = _standardise_train_test(base[train], base[test])
        full_train, full_test = _standardise_train_test(full[train], full[test])
        beta_base, alpha_base = _fit_with_inner_alpha(base_train, y[train], alpha_grid)
        beta_full, alpha_full = _fit_with_inner_alpha(full_train, y[train], alpha_grid)
        fold_rows.append({
            "heldout_onset_rank": int(event_position + 1),
            "train_rows": int(len(train)), "test_rows": int(len(test)),
            "base_logloss": _logloss(y[test], _logistic_predict(base_test, beta_base)),
            "full_logloss": _logloss(y[test], _logistic_predict(full_test, beta_full)),
            "base_alpha": alpha_base, "full_alpha": alpha_full,
        })
    if not fold_rows:
        return {"status": "NOT_ESTIMABLE", "folds": [], "detected": False}
    weights = np.asarray([row["test_rows"] for row in fold_rows], dtype=np.float64)
    base_loss = float(np.average([row["base_logloss"] for row in fold_rows], weights=weights))
    full_loss = float(np.average([row["full_logloss"] for row in fold_rows], weights=weights))
    improvement = (base_loss - full_loss) / base_loss if base_loss > 1e-12 else None
    return {
        "status": "COMPLETE", "n_oof_seizures": len(fold_rows), "folds": fold_rows,
        "base_logloss": base_loss, "full_logloss": full_loss,
        "relative_logloss_improvement": improvement,
        "T_detected_at_5_percent": bool(improvement is not None and improvement >= 0.05),
        "M_detected_at_5_percent": bool(improvement is not None and improvement >= 0.05),
        "detected": bool(improvement is not None and improvement >= 0.05),
    }


def _standardise_train_test(
    train: np.ndarray, test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    centre = np.mean(train, axis=0, dtype=np.float64)
    scale = np.std(train, axis=0, dtype=np.float64)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return (train - centre) / scale, (test - centre) / scale


def geometry_recovery(
    template: AssayTemplate, outcome: np.ndarray, onset_index: np.ndarray,
) -> dict[str, Any]:
    horizon = float(template.horizon_minutes) * 60.0
    lead_rows = []
    for onset in np.asarray(onset_index, dtype=np.int64):
        candidate = np.flatnonzero(
            (template.segment == template.segment[onset])
            & (template.time_epoch <= template.time_epoch[onset] - horizon)
        )
        if len(candidate):
            lead_rows.append(int(candidate[-1]))
    case = np.asarray(lead_rows, dtype=np.int64)
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
            pooled = np.std(value[control])
            effect = (
                float((np.mean(value[case]) - np.mean(value[control])) / pooled)
                if pooled > 1e-8 else None
            )
        scores[name] = effect
    finite = {name: value for name, value in scores.items() if value is not None}
    winner = max(finite, key=lambda name: finite[name]) if finite else None
    return {
        "n_lead_states": int(len(case)), "scores": scores,
        "winning_family": winner,
    }


def run_replicate(
    template: AssayTemplate,
    world: str,
    seed: int,
    *,
    initial_k: int,
    effect_scale: float = 1.75,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    generated = simulate_world(template, world, rng, effect_scale=effect_scale)
    transfer = evaluate_prequential_transfer(
        template, generated["outcome"], generated["onset_index"],
        initial_k=int(initial_k),
    )
    geometry = geometry_recovery(
        template, generated["outcome"], generated["onset_index"],
    )
    return {
        "world": world, "seed": int(seed), "initial_k": int(initial_k),
        "n_simulated_seizures": int(len(generated["onset_index"])),
        "n_positive_grid_rows": int(np.sum(generated["outcome"])),
        "transfer": transfer, "geometry": geometry,
    }


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054
                    ) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    p = float(successes) / float(total)
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4.0 * total ** 2)) / denominator
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))
