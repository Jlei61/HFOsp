"""T2-R2.0: cross-fitted N=100 event innovation and one-shot state edge.

This module deliberately does not reuse the retired long-boxcar estimator.
The primary estimand is a post-event jump learned on next-event likelihood;
H5/H10 only ask whether that single jump persists through the frozen T1
generator when all later observation corrections and T2 jumps are disabled.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np
import torch
from torch import nn

from .t2_s1 import OneStepDesign, OneStepMetrics, _evaluate_rows, _score


T2_R2_REVISION = "t2_r2_n100_crossfit_one_shot_v1"


def _feature_matrix(pre_event_state: np.ndarray, history: np.ndarray,
                    observation: np.ndarray | None) -> np.ndarray:
    state = np.asarray(pre_event_state, dtype=np.float64)
    history = np.asarray(history, dtype=np.float64)
    if state.ndim != 2 or history.ndim != 2 or len(state) != len(history):
        raise ValueError("event-innovation covariates disagree")
    values = [state, history[:, :11]]
    if observation is not None:
        observation = np.asarray(observation, dtype=np.float64)
        if observation.ndim != 2 or len(observation) != len(state):
            raise ValueError("event observation embedding disagrees")
        values.append(observation)
    feature = np.column_stack(values)
    if not np.isfinite(feature).all():
        raise ValueError("event-innovation covariates are non-finite")
    return feature


def _ridge_predict(train_x: np.ndarray, train_y: np.ndarray,
                   target_x: np.ndarray, ridge: float) -> tuple[np.ndarray, dict]:
    mean = train_x.mean(0)
    scale = train_x.std(0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    x_train = (train_x - mean) / scale
    x_target = (target_x - mean) / scale
    design = np.column_stack([np.ones(len(x_train)), x_train])
    target_design = np.column_stack([np.ones(len(x_target)), x_target])
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(
        design.T @ design + penalty, design.T @ train_y
    )
    return target_design @ beta, {
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "beta": np.asarray(beta).tolist(),
    }


def crossfit_expected_mark(feature: np.ndarray, outcome: np.ndarray,
                           train_mask: np.ndarray, *, folds: int = 5,
                           ridge: float = 1e-2) -> tuple[np.ndarray, dict]:
    """Out-of-fold TRAIN predictions and one full-TRAIN validation prediction."""
    feature = np.asarray(feature, dtype=np.float64)
    outcome = np.asarray(outcome, dtype=np.float64)
    train_mask = np.asarray(train_mask, dtype=bool)
    if (feature.ndim != 2 or len(feature) != len(outcome)
            or train_mask.shape != (len(feature),)):
        raise ValueError("cross-fit arrays disagree")
    if outcome.ndim not in (1, 2):
        raise ValueError("cross-fit outcome must be scalar or matrix valued")
    train_index = np.flatnonzero(train_mask)
    if len(train_index) < max(20, 2 * int(folds)):
        raise ValueError("too few TRAIN events for cross-fitting")
    fold_index = [value for value in np.array_split(train_index, int(folds)) if len(value)]
    prediction = np.full(outcome.shape, np.nan, dtype=np.float64)
    fold_rows = []
    for fold, held_out in enumerate(fold_index):
        fit_index = np.setdiff1d(train_index, held_out, assume_unique=True)
        value, _ = _ridge_predict(
            feature[fit_index], outcome[fit_index], feature[held_out], ridge
        )
        prediction[held_out] = value
        fold_rows.append({
            "fold": int(fold), "n_fit": int(len(fit_index)),
            "n_held_out": int(len(held_out)),
            "held_out_start": int(held_out[0]),
            "held_out_stop": int(held_out[-1]),
        })
    validation_index = np.flatnonzero(~train_mask)
    validation_model = None
    if len(validation_index):
        prediction[validation_index], validation_model = _ridge_predict(
            feature[train_index], outcome[train_index],
            feature[validation_index], ridge,
        )
    if not np.isfinite(prediction).all():
        raise RuntimeError("cross-fit expectation left unpredicted rows")
    return prediction, {
        "folds": int(len(fold_index)),
        "fold_rows": fold_rows,
        "ridge": float(ridge),
        "train_predictions_are_out_of_fold": True,
        "validation_model_fit_on_train_only": True,
        "uses_validation_outcome": False,
        "validation_model": validation_model,
    }


def fit_load_innovation_crossfit(
    pre_event_state: np.ndarray,
    history: np.ndarray,
    observation: np.ndarray,
    load: np.ndarray,
    train_mask: np.ndarray,
    *,
    folds: int = 5,
    ridge: float = 1e-2,
) -> tuple[np.ndarray, dict]:
    feature = _feature_matrix(pre_event_state, history, observation)
    load = np.asarray(load, dtype=np.float64)
    prediction, audit = crossfit_expected_mark(
        feature, load, train_mask, folds=folds, ridge=ridge
    )
    innovation = load - prediction
    train_mask = np.asarray(train_mask, dtype=bool)
    centre = float(np.mean(innovation[train_mask]))
    scale = float(np.std(innovation[train_mask]))
    if not np.isfinite(scale) or scale <= 1e-8:
        raise ValueError("load innovation is degenerate on TRAIN")
    innovation = (innovation - centre) / scale
    audit.update({
        "source": "scalar_load",
        "conditioned_on_pre_event_state": True,
        "conditioned_on_fixed_history": True,
        "conditioned_on_observation_embedding": True,
        "train_residual_centre": centre,
        "train_residual_sd": scale,
    })
    return innovation.astype(np.float32), audit


def fit_participation_innovation_crossfit(
    pre_event_state: np.ndarray,
    history: np.ndarray,
    observation: np.ndarray,
    participation: np.ndarray,
    train_mask: np.ndarray,
    *,
    components: int = 2,
    folds: int = 5,
    ridge: float = 1e-2,
) -> tuple[np.ndarray, dict]:
    participation = np.asarray(participation, dtype=np.float64)
    if participation.ndim != 2:
        raise ValueError("participation must have shape (event,contact)")
    load = participation.sum(1, keepdims=True)
    if np.any(load <= 0):
        raise ValueError("participation contains an empty event")
    composition = participation / load
    feature = _feature_matrix(pre_event_state, history, observation)
    expected, audit = crossfit_expected_mark(
        feature, composition, train_mask, folds=folds, ridge=ridge
    )
    residual = composition - expected
    train_mask = np.asarray(train_mask, dtype=bool)
    _, singular, right = np.linalg.svd(residual[train_mask], full_matrices=False)
    basis = right[:min(int(components), right.shape[0])].copy()
    for row in basis:
        anchor = int(np.argmax(np.abs(row)))
        if row[anchor] < 0:
            row *= -1
    score = residual @ basis.T
    centre = score[train_mask].mean(0)
    scale = score[train_mask].std(0)
    keep = scale > 1e-8
    if not np.any(keep):
        raise ValueError("composition innovation is degenerate on TRAIN")
    score = (score[:, keep] - centre[keep]) / scale[keep]
    variance = np.square(singular)
    audit.update({
        "source": "participation_composition_after_total_load_removal",
        "conditioned_on_pre_event_state": True,
        "conditioned_on_fixed_history": True,
        "conditioned_on_observation_embedding": True,
        "composition_removes_total_load": True,
        "requested_components": int(components),
        "retained_components": int(np.sum(keep)),
        "basis": basis[keep].tolist(),
        "explained_variance_ratio": (
            variance[:len(basis)][keep] / max(float(variance.sum()), 1e-12)
        ).tolist(),
        "train_score_centre": centre[keep].tolist(),
        "train_score_sd": scale[keep].tolist(),
    })
    return score.astype(np.float32), audit


def exponential_event_exposure(innovation: np.ndarray, segment: np.ndarray,
                               scale_events: int = 100,
                               *, burn_in_events: int | None = None
                               ) -> tuple[np.ndarray, np.ndarray, dict]:
    """x_e=exp(-1/N)x_(e-1)+eta_e, reset at recorded gaps."""
    value = np.asarray(innovation, dtype=np.float64)
    scalar = value.ndim == 1
    if scalar:
        value = value[:, None]
    segment = np.asarray(segment, dtype=np.int64)
    n = int(scale_events)
    burn = int(n if burn_in_events is None else burn_in_events)
    if value.ndim != 2 or len(value) != len(segment) or n < 1 or burn < 1:
        raise ValueError("invalid exponential exposure input")
    alpha = math.exp(-1.0 / n)
    exposure = np.zeros_like(value, dtype=np.float64)
    eligible = np.zeros(len(value), dtype=bool)
    segment_rows = []
    for label in np.unique(segment):
        index = np.flatnonzero(segment == label)
        running = np.zeros(value.shape[1], dtype=np.float64)
        for position, row in enumerate(index):
            running = alpha * running + value[row]
            exposure[row] = running
            if position + 1 >= burn:
                eligible[row] = True
        segment_rows.append({
            "segment": int(label), "events": int(len(index)),
            "eligible": int(max(0, len(index) - burn + 1)),
        })
    output = exposure[:, 0] if scalar else exposure
    return output.astype(np.float32), eligible, {
        "scale_events": n,
        "alpha": float(alpha),
        "burn_in_events": burn,
        "resets_at_recorded_segment": True,
        "segment_rows": segment_rows,
    }


def state_matched_nonoverlap_placebo(
    exposure: np.ndarray,
    pre_event_state: np.ndarray,
    history: np.ndarray,
    observation: np.ndarray,
    train_mask: np.ndarray,
    eligible: np.ndarray,
    segment: np.ndarray,
    *,
    scale_events: int = 100,
    history_multiples: int = 5,
    neighbours: int = 256,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Nearest TRAIN donor whose effective exposure history cannot overlap."""
    from scipy.spatial import cKDTree

    exposure = np.asarray(exposure, dtype=np.float64)
    state = np.asarray(pre_event_state, dtype=np.float64)
    history = np.asarray(history, dtype=np.float64)
    observation = np.asarray(observation, dtype=np.float64)
    train_mask = np.asarray(train_mask, dtype=bool)
    eligible = np.asarray(eligible, dtype=bool)
    segment = np.asarray(segment, dtype=np.int64)
    if any(len(x) != len(state) for x in (
        exposure, history, observation, train_mask, eligible, segment
    )):
        raise ValueError("placebo arrays disagree")
    feature = _feature_matrix(state, history, observation)
    donor_index = np.flatnonzero(train_mask & eligible)
    if len(donor_index) < 2:
        raise ValueError("state-matched placebo has too few TRAIN donors")
    mean = feature[donor_index].mean(0)
    scale = feature[donor_index].std(0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    z = (feature - mean) / scale
    tree = cKDTree(z[donor_index])
    query_k = min(max(1, int(neighbours)), len(donor_index))
    history_length = int(scale_events) * int(history_multiples)
    target = np.flatnonzero(eligible)
    result = np.zeros_like(exposure, dtype=np.float32)
    matched = np.zeros(len(exposure), dtype=bool)
    distances = []
    donor_rows = []
    matched_feature_differences = []
    for row in target:
        distance, position = tree.query(z[row], k=query_k)
        candidates = donor_index[np.atleast_1d(position).astype(np.int64)]
        candidate_distance = np.atleast_1d(distance)
        same = segment[candidates] == segment[row]
        disjoint = (~same) | (np.abs(candidates - row) >= history_length)
        if not np.any(disjoint) and query_k < len(donor_index):
            fallback_k = min(len(donor_index), max(2048, 4 * query_k))
            distance, position = tree.query(z[row], k=fallback_k)
            candidates = donor_index[np.atleast_1d(position).astype(np.int64)]
            candidate_distance = np.atleast_1d(distance)
            same = segment[candidates] == segment[row]
            disjoint = (~same) | (np.abs(candidates - row) >= history_length)
        if not np.any(disjoint):
            continue
        chosen_position = int(np.flatnonzero(disjoint)[0])
        donor = int(candidates[chosen_position])
        result[row] = exposure[donor]
        matched[row] = True
        distances.append(float(candidate_distance[chosen_position]))
        donor_rows.append(donor)
        matched_feature_differences.append(np.abs(z[row] - z[donor]))
    donor_array = np.asarray(donor_rows, dtype=np.int64)
    if len(donor_array):
        _, donor_counts = np.unique(donor_array, return_counts=True)
        effective_donors = float(
            np.square(donor_counts.sum()) / np.square(donor_counts).sum()
        )
        maximum_reuse = float(donor_counts.max() / donor_counts.sum())
        feature_difference = np.asarray(
            matched_feature_differences, dtype=np.float64
        )
    else:
        donor_counts = np.asarray([], dtype=np.int64)
        effective_donors = 0.0
        maximum_reuse = 1.0
        feature_difference = np.empty((0, feature.shape[1]), dtype=np.float64)
    return result, matched, {
        "train_donor_pool": int(len(donor_index)),
        "targets": int(len(target)),
        "matched": int(matched[target].sum()),
        "missing": int((~matched[target]).sum()),
        "history_nonoverlap_events": int(history_length),
        "history_nonoverlap_residual_weight_upper_bound": float(
            math.exp(-history_length / int(scale_events))
        ),
        "validation_donors_from_train_only": True,
        "all_matched": bool(matched[target].all()),
        "median_match_distance": (
            float(np.median(distances)) if distances else None
        ),
        "match_distance_q95": (
            float(np.quantile(distances, .95)) if distances else None
        ),
        "match_distance_max": (
            float(np.max(distances)) if distances else None
        ),
        "unique_donors": int(len(donor_counts)),
        "effective_donors": effective_donors,
        "maximum_donor_reuse_fraction": maximum_reuse,
        "matched_feature_abs_z_difference_median": (
            float(np.median(feature_difference)) if len(feature_difference)
            else None
        ),
        "matched_feature_abs_z_difference_q95": (
            float(np.quantile(feature_difference, .95))
            if len(feature_difference) else None
        ),
        "donor_rows_sha256": hashlib.sha256(
            donor_array.tobytes()
        ).hexdigest(),
    }


class ExposureEdge(nn.Module):
    """Signed scalar or low-dimensional exposure mapped into frozen T1 state."""

    def __init__(self, state_dim: int, exposure_dim: int = 1):
        super().__init__()
        self.matrix = nn.Parameter(torch.zeros(int(exposure_dim), int(state_dim)))

    def forward(self, state: torch.Tensor, exposure: torch.Tensor) -> torch.Tensor:
        if exposure.ndim == 1:
            exposure = exposure.unsqueeze(-1)
        return state + exposure @ self.matrix


def edge_estimability_audit(model, design: OneStepDesign, *,
                            device: torch.device | str,
                            batch_size: int = 4096) -> dict:
    train = np.flatnonzero(design.split == 0)
    if not len(train):
        raise ValueError("edge audit has no TRAIN pairs")
    exposure = np.asarray(design.exposure[train], dtype=np.float64)
    matrix = exposure[:, None] if exposure.ndim == 1 else exposure
    edge = ExposureEdge(
        design.current_state.shape[1], matrix.shape[1]
    ).to(device)
    total = len(train)
    for lo in range(0, total, int(batch_size)):
        rows = train[lo:lo + int(batch_size)]
        loss, _ = _score(
            model, edge, design, rows, device=device, require_grad=True
        )
        (loss * (len(rows) / total)).backward()
    gradient = edge.matrix.grad.detach().cpu().numpy()
    return {
        "train_pairs": int(total),
        "exposure_dim": int(matrix.shape[1]),
        "exposure_rank": int(np.linalg.matrix_rank(matrix)),
        "exposure_mean": matrix.mean(0).tolist(),
        "exposure_sd": matrix.std(0).tolist(),
        "exposure_zero_fraction": float(np.mean(np.abs(matrix) <= 1e-8)),
        "gradient_at_zero_norm": float(np.linalg.norm(gradient)),
        "gradient_at_zero_max_abs": float(np.max(np.abs(gradient))),
        "gradient_finite": bool(np.isfinite(gradient).all()),
    }


def fit_r2_edge(model, design: OneStepDesign, *, device: torch.device | str,
                seed: int = 0, epochs: int = 30,
                learning_rate: float = 2e-2,
                batch_size: int = 4096) -> tuple[ExposureEdge, dict]:
    """Select epochs on chronological inner TRAIN, then refit all TRAIN."""
    torch.manual_seed(int(seed))
    train = np.flatnonzero(design.split == 0)
    if len(train) < 100:
        raise ValueError("T2-R2.0 needs at least 100 TRAIN pairs")
    cut = int(np.clip(math.floor(0.8 * len(train)), 1, len(train) - 1))
    inner_train, inner_validation = train[:cut], train[cut:]
    exposure_dim = 1 if design.exposure.ndim == 1 else design.exposure.shape[1]

    def fresh() -> ExposureEdge:
        return ExposureEdge(design.current_state.shape[1], exposure_dim).to(device)

    base_edge = fresh()
    base = _evaluate_rows(
        model, base_edge, design, inner_validation,
        device=device, batch_size=batch_size,
    )
    best_epoch = 0
    best_value = base.joint_nll_per_event
    edge = fresh()
    optimizer = torch.optim.AdamW(
        edge.parameters(), lr=float(learning_rate), weight_decay=1e-3
    )
    trajectory = [{"epoch": 0, "joint_nll": float(best_value)}]
    rng = np.random.default_rng(int(seed))
    for epoch in range(1, int(epochs) + 1):
        order = rng.permutation(inner_train)
        for lo in range(0, len(order), int(batch_size)):
            rows = order[lo:lo + int(batch_size)]
            loss, _ = _score(
                model, edge, design, rows, device=device, require_grad=True
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(edge.parameters(), 1.0)
            optimizer.step()
        if epoch == 1 or epoch % 2 == 0:
            value = _evaluate_rows(
                model, edge, design, inner_validation,
                device=device, batch_size=batch_size,
            ).joint_nll_per_event
            trajectory.append({"epoch": int(epoch), "joint_nll": float(value)})
            if value < best_value:
                best_value = float(value)
                best_epoch = int(epoch)
    edge = fresh()
    if best_epoch:
        optimizer = torch.optim.AdamW(
            edge.parameters(), lr=float(learning_rate), weight_decay=1e-3
        )
        rng = np.random.default_rng(int(seed))
        for _ in range(best_epoch):
            order = rng.permutation(train)
            for lo in range(0, len(order), int(batch_size)):
                rows = order[lo:lo + int(batch_size)]
                loss, _ = _score(
                    model, edge, design, rows,
                    device=device, require_grad=True,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(edge.parameters(), 1.0)
                optimizer.step()
    return edge.eval(), {
        "selected_epoch": int(best_epoch),
        "inner_validation_joint_nll": float(best_value),
        "trajectory": trajectory,
        "edge_norm": float(torch.linalg.vector_norm(edge.matrix).detach().cpu()),
        "edge_left_zero_initialisation": bool(
            torch.linalg.vector_norm(edge.matrix).detach().cpu() > 1e-8
        ),
    }


@dataclass(frozen=True)
class HorizonMarkDesign:
    current_state: np.ndarray
    target_state: np.ndarray
    current_index: np.ndarray
    target_history: np.ndarray
    target_group_ids: np.ndarray
    target_group_count: np.ndarray
    delta_minutes: np.ndarray
    exposure: np.ndarray
    split: np.ndarray
    horizon_events: int


@dataclass(frozen=True)
class HorizonMarkMetrics:
    mark_nll_per_event: float
    group_size_nll_per_event: float
    subset_nll_per_event: float
    stop_nll_per_event: float
    first_group_subset_nll_per_event: float
    continuation_subset_nll_per_event: float
    n_events: int
    n_continuation_events: int
    mean_state_displacement_from_no_edge: float
    state_mse_to_filtered_target: float


def build_horizon_mark_design(full_design, pre_event_state: np.ndarray,
                              event_segment: np.ndarray,
                              exposure: np.ndarray, eligible: np.ndarray,
                              horizon_events: int) -> HorizonMarkDesign:
    h = int(horizon_events)
    if h < 1:
        raise ValueError("horizon must be positive")
    state = np.asarray(pre_event_state, dtype=np.float32)
    segment = np.asarray(event_segment, dtype=np.int64)
    exposure = np.asarray(exposure, dtype=np.float32)
    eligible = np.asarray(eligible, dtype=bool)
    n = len(full_design.event_time)
    current = np.arange(max(n - h, 0), dtype=np.int64)
    target = current + h
    keep = (
        eligible[current]
        & (segment[current] >= 0)
        & (segment[current] == segment[target])
        & (full_design.event_split[current] == full_design.event_split[target])
        & (full_design.event_time[target] > full_design.event_time[current])
    )
    current, target = current[keep], target[keep]
    if not len(current):
        raise ValueError(f"T2-R2.0 H{h} has no within-segment pairs")
    return HorizonMarkDesign(
        current_state=state[current],
        target_state=state[target],
        current_index=current,
        target_history=np.asarray(full_design.event_history[target], dtype=np.float32),
        target_group_ids=np.asarray(full_design.event_group_ids[target], dtype=np.int64),
        target_group_count=np.asarray(full_design.event_group_count[target], dtype=np.int64),
        delta_minutes=np.asarray(
            (full_design.event_time[target] - full_design.event_time[current]) / 60.0,
            dtype=np.float32,
        ),
        exposure=exposure[current],
        split=np.asarray(full_design.event_split[current], dtype=np.int8),
        horizon_events=h,
    )


def evaluate_horizon_mark(model, edge: ExposureEdge,
                          design: HorizonMarkDesign, *, split: str,
                          device: torch.device | str,
                          batch_size: int = 4096) -> HorizonMarkMetrics:
    code = {"train": 0, "validation": 1}[split]
    rows = np.flatnonzero(design.split == code)
    if not len(rows):
        raise ValueError("cannot score empty horizon split")
    totals = {
        "mark": 0.0, "size": 0.0, "subset": 0.0, "stop": 0.0,
        "first": 0.0, "continuation": 0.0, "displacement": 0.0,
        "state_square": 0.0,
    }
    continuation_events = 0
    with torch.no_grad():
        matrix = model.state.generator.matrix().float()
        mu = model.state.generator.mu
        for lo in range(0, len(rows), int(batch_size)):
            take = rows[lo:lo + int(batch_size)]
            current = torch.as_tensor(design.current_state[take], device=device)
            exposure = torch.as_tensor(design.exposure[take], device=device)
            shifted = edge(current, exposure)
            delta = torch.as_tensor(
                design.delta_minutes[take], dtype=torch.float32, device=device
            )
            transition = torch.matrix_exp(matrix.unsqueeze(0) * delta[:, None, None])
            state = mu.unsqueeze(0) + torch.matmul(
                transition, (shifted - mu).unsqueeze(-1)
            ).squeeze(-1)
            base_state = mu.unsqueeze(0) + torch.matmul(
                transition, (current - mu).unsqueeze(-1)
            ).squeeze(-1)
            history = torch.as_tensor(design.target_history[take], device=device)
            mark = model.mark_terms(
                history, state,
                torch.as_tensor(design.target_group_ids[take], dtype=torch.long, device=device),
                torch.as_tensor(design.target_group_count[take], dtype=torch.long, device=device),
            )
            steps = torch.arange(mark.group_size_step_log_prob.shape[1], device=device)[None]
            terminal = mark.active_step & ~mark.select_step
            first = mark.select_step & (steps == 0)
            continuation = mark.select_step & (steps >= 1)
            n_chunk = len(take)
            n_cont = int((design.target_group_count[take] >= 2).sum())
            totals["mark"] += float((-mark.event_log_prob.sum()).cpu())
            totals["size"] += float((-mark.group_size_log_prob.sum()).cpu())
            totals["subset"] += float((-mark.subset_log_prob.sum()).cpu())
            totals["stop"] += float((-mark.group_size_step_log_prob[terminal].sum()).cpu())
            totals["first"] += float((-mark.subset_step_log_prob[first].sum()).cpu())
            totals["continuation"] += float((-mark.subset_step_log_prob[continuation].sum()).cpu())
            totals["displacement"] += float(
                torch.linalg.vector_norm(state - base_state, dim=1).sum().cpu()
            )
            target_state = torch.as_tensor(
                design.target_state[take], dtype=state.dtype, device=device
            )
            totals["state_square"] += float(
                torch.square(state - target_state).mean(1).sum().cpu()
            )
            continuation_events += n_cont
    return HorizonMarkMetrics(
        mark_nll_per_event=totals["mark"] / len(rows),
        group_size_nll_per_event=totals["size"] / len(rows),
        subset_nll_per_event=totals["subset"] / len(rows),
        stop_nll_per_event=totals["stop"] / len(rows),
        first_group_subset_nll_per_event=totals["first"] / len(rows),
        continuation_subset_nll_per_event=(
            totals["continuation"] / max(continuation_events, 1)
        ),
        n_events=int(len(rows)),
        n_continuation_events=int(continuation_events),
        mean_state_displacement_from_no_edge=totals["displacement"] / len(rows),
        state_mse_to_filtered_target=totals["state_square"] / len(rows),
    )


def evaluate_r2_edge(model, edge: ExposureEdge, design: OneStepDesign, *,
                     split: str, device: torch.device | str,
                     batch_size: int = 4096) -> OneStepMetrics:
    code = {"train": 0, "validation": 1}[split]
    rows = np.flatnonzero(design.split == code)
    return _evaluate_rows(
        model, edge, design, rows, device=device, batch_size=batch_size
    )


def classify_one_shot_persistence(
    comparison: dict,
    real_metrics: dict,
    *,
    real_edge_estimable: bool,
) -> dict:
    """Classify persistence only when the real edge actually moved state.

    A zero-selected real edge can look better than a nonzero placebo edge, but
    that contrast is not evidence that the real exposure produced a persistent
    update.  Keep the component signs for diagnosis while requiring a fitted,
    estimable real edge and nonzero propagated displacement for the combined
    label.
    """
    mark_increment = bool(comparison["mark_nll_per_event"] < 0)
    state_increment = bool(comparison["state_mse_to_filtered_target"] < 0)
    nonzero_displacement = bool(
        real_metrics["mean_state_displacement_from_no_edge"] > 1e-8
    )
    return {
        "mark_prediction_increment": mark_increment,
        "state_prediction_increment": state_increment,
        "state_and_mark_persist": bool(
            real_edge_estimable
            and nonzero_displacement
            and mark_increment
            and state_increment
        ),
        "nonzero_propagated_displacement": nonzero_displacement,
        "requires_estimable_nonzero_real_edge": True,
    }
