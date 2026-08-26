"""Minimal one-step event-exposure to persistent-state edge for H3/T2-S1."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn


T2_S1_REVISION = "t2_s1_one_step_signed_load_innovation_v1"


def fit_load_innovation(pre_event_state: np.ndarray, history: np.ndarray,
                        load: np.ndarray, train_mask: np.ndarray,
                        *, ridge: float = 1e-2
                        ) -> tuple[np.ndarray, dict]:
    """TRAIN-only ridge expectation and signed per-event load innovation."""
    state = np.asarray(pre_event_state, dtype=np.float64)
    history = np.asarray(history, dtype=np.float64)
    load = np.asarray(load, dtype=np.float64)
    train_mask = np.asarray(train_mask, dtype=bool)
    if state.shape[0] != len(load) or history.shape[0] != len(load):
        raise ValueError("load-innovation arrays disagree")
    feature = np.column_stack([state, history[:, :11]])
    mean = feature[train_mask].mean(0)
    scale = feature[train_mask].std(0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    x = (feature - mean) / scale
    design = np.column_stack([np.ones(len(x)), x])
    xt = design[train_mask]
    penalty = np.eye(xt.shape[1]) * float(ridge)
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(xt.T @ xt + penalty, xt.T @ load[train_mask])
    prediction = design @ beta
    innovation = load - prediction
    train_sd = float(np.std(innovation[train_mask]))
    train_sd = train_sd if train_sd > 1e-8 else 1.0
    innovation = innovation / train_sd
    return innovation.astype(np.float32), {
        "ridge": float(ridge),
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "beta": beta.tolist(),
        "train_residual_sd": train_sd,
        "uses_validation_outcome": False,
    }


def fit_participation_innovation(
    pre_event_state: np.ndarray,
    history: np.ndarray,
    participation: np.ndarray,
    train_mask: np.ndarray,
    *,
    n_components: int = 2,
    ridge: float = 1e-2,
) -> tuple[np.ndarray, dict]:
    """TRAIN-only low-rank innovation in contact-participation composition.

    Each event is first converted from a binary contact mask to a composition
    summing to one.  This removes total event load before the expected
    composition is residualised against pre-event state and deterministic
    history.  PCA is then fit on TRAIN residuals only.  The returned scores are
    therefore a compact repertoire-composition exposure, not another copy of
    participation count.
    """
    state = np.asarray(pre_event_state, dtype=np.float64)
    history = np.asarray(history, dtype=np.float64)
    participation = np.asarray(participation, dtype=np.float64)
    train_mask = np.asarray(train_mask, dtype=bool)
    if (participation.ndim != 2 or state.shape[0] != len(participation)
            or history.shape[0] != len(participation)
            or train_mask.shape != (len(participation),)):
        raise ValueError("participation-innovation arrays disagree")
    load = participation.sum(1, keepdims=True)
    if np.any(load <= 0.0):
        raise ValueError("participation composition contains an empty event")
    composition = participation / load
    feature = np.column_stack([state, history[:, :11]])
    mean = feature[train_mask].mean(0)
    scale = feature[train_mask].std(0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    x = (feature - mean) / scale
    design = np.column_stack([np.ones(len(x)), x])
    xt = design[train_mask]
    penalty = np.eye(xt.shape[1]) * float(ridge)
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(
        xt.T @ xt + penalty, xt.T @ composition[train_mask]
    )
    residual = composition - design @ beta
    train_residual = residual[train_mask]
    _, singular, right = np.linalg.svd(train_residual, full_matrices=False)
    components = right[:min(int(n_components), right.shape[0])].copy()
    if not len(components):
        raise ValueError("participation innovation has no PCA component")
    # Fix the arbitrary SVD sign so repeated runs and seed comparisons use the
    # same repertoire direction.
    for row in components:
        anchor = int(np.argmax(np.abs(row)))
        if row[anchor] < 0.0:
            row *= -1.0
    score = residual @ components.T
    score_sd = score[train_mask].std(0)
    keep = score_sd > 1e-8
    if not np.any(keep):
        raise ValueError("participation innovation is degenerate on TRAIN")
    components = components[keep]
    score_sd = score_sd[keep]
    score = score[:, keep] / score_sd
    variance = np.square(singular)
    explained = variance / max(float(variance.sum()), 1e-12)
    return score.astype(np.float32), {
        "ridge": float(ridge),
        "n_contacts": int(participation.shape[1]),
        "requested_components": int(n_components),
        "retained_components": int(score.shape[1]),
        "components": components.tolist(),
        "component_train_sd_before_scaling": score_sd.tolist(),
        "explained_variance_ratio": explained[:len(components)].tolist(),
        "composition_removes_total_load": True,
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "uses_validation_outcome": False,
    }


def rolling_event_exposure(innovation: np.ndarray, segment: np.ndarray,
                           scale_events: int) -> tuple[np.ndarray, np.ndarray]:
    """Finite rolling signed exposure ending at the current event.

    Division by sqrt(N) puts different N on a comparable innovation scale; the
    learned edge still estimates direction and magnitude.  Windows reset at
    every recorded-coverage segment and never cross an unobserved gap.
    """
    innovation = np.asarray(innovation, dtype=np.float64)
    segment = np.asarray(segment, dtype=np.int64)
    n = int(scale_events)
    if n < 1 or innovation.shape != segment.shape:
        raise ValueError("invalid rolling exposure input")
    exposure = np.zeros(len(innovation), dtype=np.float32)
    eligible = np.zeros(len(innovation), dtype=bool)
    for label in np.unique(segment):
        index = np.flatnonzero(segment == label)
        cumulative = np.concatenate([[0.0], np.cumsum(innovation[index])])
        if len(index) < n:
            continue
        position = np.arange(n - 1, len(index))
        total = cumulative[position + 1] - cumulative[position + 1 - n]
        selected = index[position]
        exposure[selected] = (total / math.sqrt(n)).astype(np.float32)
        eligible[selected] = True
    return exposure, eligible


def state_matched_placebo(exposure: np.ndarray, pre_event_state: np.ndarray,
                          history: np.ndarray, train_mask: np.ndarray,
                          eligible: np.ndarray, *, exclusion_events: int,
                          neighbours: int = 128
                          ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Use a TRAIN-pool exposure from a similar pre-event state/history.

    Every target - TRAIN and validation alike - excludes donors inside its own
    +/-N event neighbourhood so the placebo cannot copy the real cumulative
    window.  Restricting that exclusion to TRAIN targets would let a validation
    row near the split boundary draw a TRAIN donor whose rolling window overlaps
    its own by up to N-1 events, which biases the real-minus-placebo contrast
    toward zero exactly where the donor pool is densest.
    """
    from scipy.spatial import cKDTree

    exposure = np.asarray(exposure, dtype=np.float64)
    state = np.asarray(pre_event_state, dtype=np.float64)
    history = np.asarray(history, dtype=np.float64)
    train_mask = np.asarray(train_mask, dtype=bool)
    eligible = np.asarray(eligible, dtype=bool)
    feature = np.column_stack([state, history[:, 1:11]])
    base = train_mask & eligible
    mean = feature[base].mean(0)
    scale = feature[base].std(0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    z = (feature - mean) / scale
    donor_index = np.flatnonzero(base)
    if len(donor_index) < 2:
        raise ValueError("state-matched placebo has too few TRAIN donors")
    tree = cKDTree(z[donor_index])
    k = min(int(neighbours), len(donor_index))
    target = np.flatnonzero(eligible)
    distance, neighbour = tree.query(z[target], k=k)
    if k == 1:
        neighbour = neighbour[:, None]; distance = distance[:, None]
    result = np.zeros(len(exposure), dtype=np.float32)
    matched = np.zeros(len(exposure), dtype=bool)
    used_distance = []
    missing = 0
    for row, candidate_position, candidate_distance in zip(
        target, neighbour, distance
    ):
        candidate = donor_index[np.asarray(candidate_position, dtype=np.int64)]
        keep = np.abs(candidate - row) >= int(exclusion_events)
        candidate = candidate[keep]
        candidate_distance = np.asarray(candidate_distance)[keep]
        if not len(candidate) and k < len(donor_index):
            fallback_k = min(
                len(donor_index),
                max(4 * k, 2 * int(exclusion_events) + 3),
            )
            fallback_distance, fallback_position = tree.query(
                z[row], k=fallback_k
            )
            fallback_position = np.atleast_1d(fallback_position).astype(np.int64)
            fallback_distance = np.atleast_1d(fallback_distance)
            fallback_candidate = donor_index[fallback_position]
            fallback_keep = (
                np.abs(fallback_candidate - row) >= int(exclusion_events)
            )
            candidate = fallback_candidate[fallback_keep]
            candidate_distance = fallback_distance[fallback_keep]
        if not len(candidate):
            missing += 1
            continue
        result[row] = float(exposure[candidate[0]])
        matched[row] = True
        used_distance.append(float(np.asarray(candidate_distance)[0]))
    return result, matched, {
        "train_donor_pool": int(len(donor_index)),
        "targets": int(len(target)),
        "missing_donor": int(missing),
        "neighbours_queried": int(k),
        "median_match_distance": (
            float(np.median(used_distance)) if used_distance else None
        ),
        "validation_donors_from_train_only": True,
        "exclusion_events_all_targets": int(exclusion_events),
        "all_targets_have_donor": bool(missing == 0),
    }


@dataclass(frozen=True)
class OneStepDesign:
    current_state: np.ndarray
    current_index: np.ndarray
    next_history: np.ndarray
    next_group_ids: np.ndarray
    next_group_count: np.ndarray
    delta_minutes: np.ndarray
    quadrature_delta_minutes: np.ndarray
    quadrature_history: np.ndarray
    quadrature_weight_seconds: np.ndarray
    exposure: np.ndarray
    split: np.ndarray

    def validate(self) -> None:
        n = len(self.current_state)
        for value in (
            self.current_index, self.next_history, self.next_group_ids,
            self.next_group_count, self.delta_minutes,
            self.quadrature_delta_minutes, self.quadrature_history,
            self.quadrature_weight_seconds, self.exposure, self.split,
        ):
            if len(value) != n:
                raise ValueError("one-step T2 arrays disagree")
        if self.quadrature_delta_minutes.ndim != 2:
            raise ValueError("one-step quadrature offsets must be rectangular")
        if self.quadrature_history.shape[:2] != self.quadrature_delta_minutes.shape:
            raise ValueError("one-step quadrature history shape mismatch")
        if np.any(self.delta_minutes <= 0) or np.any(
            self.quadrature_weight_seconds <= 0
        ):
            raise ValueError("one-step interval is non-positive")


def build_one_step_design(full_design, pre_event_state: np.ndarray,
                          event_segment: np.ndarray, exposure: np.ndarray,
                          eligible: np.ndarray) -> OneStepDesign:
    """Build exact current-event to next-event pairs inside one recorded segment.

    The parent full design uses order-four Gauss-Legendre quadrature after
    splitting recorded coverage at every event.  A valid adjacent event pair
    therefore has exactly four integration nodes.  Requiring the same explicit
    coverage segment prevents both the exposure and the survival integral from
    crossing an unrecorded gap.
    """
    state = np.asarray(pre_event_state, dtype=np.float32)
    segment = np.asarray(event_segment, dtype=np.int64)
    exposure = np.asarray(exposure, dtype=np.float32)
    eligible = np.asarray(eligible, dtype=bool)
    n_event = len(full_design.event_time)
    if any(len(value) != n_event for value in (state, segment, exposure, eligible)):
        raise ValueError("T2-S1 event arrays disagree")
    if state.ndim != 2:
        raise ValueError("pre-event state must have shape (event,state)")
    current = np.arange(max(n_event - 1, 0), dtype=np.int64)
    keep = (
        eligible[current]
        & (segment[current] >= 0)
        & (segment[current] == segment[current + 1])
        & (full_design.event_split[current] == full_design.event_split[current + 1])
        & (full_design.event_time[current + 1] > full_design.event_time[current])
    )
    current = current[keep]
    q_delta: list[np.ndarray] = []
    q_history: list[np.ndarray] = []
    q_weight: list[np.ndarray] = []
    accepted: list[int] = []
    q_time = np.asarray(full_design.quadrature_time, dtype=np.float64)
    for row in current:
        left = float(full_design.event_time[row])
        right = float(full_design.event_time[row + 1])
        lo = int(np.searchsorted(q_time, left, side="right"))
        hi = int(np.searchsorted(q_time, right, side="left"))
        candidate = np.arange(lo, hi, dtype=np.int64)
        candidate = candidate[
            (full_design.quadrature_split[candidate] == full_design.event_split[row])
            & (full_design.quadrature_session[candidate] == full_design.event_session[row])
        ]
        if len(candidate) != 4:
            continue
        accepted.append(int(row))
        q_delta.append((q_time[candidate] - left) / 60.0)
        q_history.append(full_design.quadrature_history[candidate])
        q_weight.append(full_design.quadrature_weight_seconds[candidate])
    if not accepted:
        raise ValueError("T2-S1 has no exact one-step pairs")
    current = np.asarray(accepted, dtype=np.int64)
    result = OneStepDesign(
        current_state=state[current],
        current_index=current,
        next_history=np.asarray(full_design.event_history[current + 1], dtype=np.float32),
        next_group_ids=np.asarray(full_design.event_group_ids[current + 1], dtype=np.int64),
        next_group_count=np.asarray(full_design.event_group_count[current + 1], dtype=np.int64),
        delta_minutes=np.asarray(
            (full_design.event_time[current + 1] - full_design.event_time[current]) / 60.0,
            dtype=np.float32,
        ),
        quadrature_delta_minutes=np.asarray(q_delta, dtype=np.float32),
        quadrature_history=np.asarray(q_history, dtype=np.float32),
        quadrature_weight_seconds=np.asarray(q_weight, dtype=np.float32),
        exposure=exposure[current],
        split=np.asarray(full_design.event_split[current], dtype=np.int8),
    )
    result.validate()
    return result


class SignedExposureEdge(nn.Module):
    """One signed scalar exposure mapped to the frozen T1 state coordinates."""

    def __init__(self, state_dim: int):
        super().__init__()
        self.vector = nn.Parameter(torch.zeros(int(state_dim)))

    def forward(self, state: torch.Tensor, exposure: torch.Tensor) -> torch.Tensor:
        return state + exposure.unsqueeze(-1) * self.vector.unsqueeze(0)


@dataclass(frozen=True)
class OneStepMetrics:
    joint_nll_per_event: float
    timing_nll_per_event: float
    mark_nll_per_event: float
    group_size_nll_per_event: float
    subset_nll_per_event: float
    stop_nll_per_event: float
    first_group_subset_nll_per_event: float
    continuation_subset_nll_per_event: float
    n_events: int
    n_continuation_events: int


def _score(model, edge: SignedExposureEdge, design: OneStepDesign,
           rows: np.ndarray, *, device: torch.device | str,
           require_grad: bool) -> tuple[torch.Tensor, OneStepMetrics]:
    rows = np.asarray(rows, dtype=np.int64)
    context = torch.enable_grad() if require_grad else torch.no_grad()
    with context:
        current = torch.as_tensor(design.current_state[rows], device=device)
        exposure = torch.as_tensor(design.exposure[rows], device=device)
        shifted = edge(current, exposure)
        matrix = model.state.generator.matrix().float()
        mu = model.state.generator.mu
        delta = torch.as_tensor(
            design.delta_minutes[rows], dtype=torch.float32, device=device
        )
        transition = torch.matrix_exp(matrix.unsqueeze(0) * delta[:, None, None])
        next_state = mu.unsqueeze(0) + torch.matmul(
            transition, (shifted - mu).unsqueeze(-1)
        ).squeeze(-1)
        history = torch.as_tensor(design.next_history[rows], device=device)
        event_log = model.timing_log_rate(history, next_state)
        mark = model.mark_terms(
            history, next_state,
            torch.as_tensor(design.next_group_ids[rows], dtype=torch.long, device=device),
            torch.as_tensor(design.next_group_count[rows], dtype=torch.long, device=device),
        )
        q_delta = torch.as_tensor(
            design.quadrature_delta_minutes[rows], dtype=torch.float32, device=device
        )
        q_transition = torch.matrix_exp(
            matrix.view(1, 1, *matrix.shape) * q_delta[..., None, None]
        )
        q_state = mu.view(1, 1, -1) + torch.matmul(
            q_transition, (shifted - mu).view(len(rows), 1, -1, 1)
        ).squeeze(-1)
        q_history = torch.as_tensor(design.quadrature_history[rows], device=device)
        q_log = model.timing_log_rate(
            q_history.reshape(-1, q_history.shape[-1]),
            q_state.reshape(-1, q_state.shape[-1]),
        ).reshape(q_delta.shape)
        q_weight = torch.as_tensor(
            design.quadrature_weight_seconds[rows], dtype=q_log.dtype, device=device
        )
        survival = torch.sum(
            q_weight * torch.exp(torch.clamp(q_log, max=20.0)), dim=1
        )
        timing = survival - event_log
        loss = (timing - mark.event_log_prob).mean()
        steps = torch.arange(
            mark.group_size_step_log_prob.shape[1], device=device
        ).unsqueeze(0)
        terminal = mark.active_step & ~mark.select_step
        first = mark.select_step & (steps == 0)
        continuation = mark.select_step & (steps >= 1)
        n_continuation = int((design.next_group_count[rows] >= 2).sum())
        metrics = OneStepMetrics(
            joint_nll_per_event=float(loss.detach()),
            timing_nll_per_event=float(timing.mean().detach()),
            mark_nll_per_event=float((-mark.event_log_prob.mean()).detach()),
            group_size_nll_per_event=float(
                (-mark.group_size_log_prob.mean()).detach()
            ),
            subset_nll_per_event=float((-mark.subset_log_prob.mean()).detach()),
            stop_nll_per_event=float(
                (-mark.group_size_step_log_prob[terminal].sum() / max(len(rows), 1)).detach()
            ),
            first_group_subset_nll_per_event=float(
                (-mark.subset_step_log_prob[first].sum() / max(len(rows), 1)).detach()
            ),
            continuation_subset_nll_per_event=float(
                (-mark.subset_step_log_prob[continuation].sum()
                 / max(n_continuation, 1)).detach()
            ),
            n_events=int(len(rows)),
            n_continuation_events=int(n_continuation),
        )
    return loss, metrics


def _evaluate_rows(model, edge: SignedExposureEdge, design: OneStepDesign,
                   rows: np.ndarray, *, device: torch.device | str,
                   batch_size: int = 4096) -> OneStepMetrics:
    rows = np.asarray(rows, dtype=np.int64)
    if not len(rows):
        raise ValueError("cannot score an empty T2-S1 split")
    chunks = []
    for lo in range(0, len(rows), int(batch_size)):
        _, value = _score(
            model, edge, design, rows[lo:lo + int(batch_size)],
            device=device, require_grad=False,
        )
        chunks.append(value)
    event_fields = (
        "joint_nll_per_event", "timing_nll_per_event", "mark_nll_per_event",
        "group_size_nll_per_event", "subset_nll_per_event",
        "stop_nll_per_event", "first_group_subset_nll_per_event",
    )
    output = {
        name: float(sum(getattr(value, name) * value.n_events for value in chunks) / len(rows))
        for name in event_fields
    }
    n_continuation = int(sum(value.n_continuation_events for value in chunks))
    output["continuation_subset_nll_per_event"] = float(
        sum(
            value.continuation_subset_nll_per_event * value.n_continuation_events
            for value in chunks
        ) / max(n_continuation, 1)
    )
    return OneStepMetrics(
        **output, n_events=int(len(rows)),
        n_continuation_events=n_continuation,
    )


def fit_edge(model, design: OneStepDesign, *, device: torch.device | str,
             seed: int = 0, epochs: int = 100,
             learning_rate: float = 3e-2,
             batch_size: int = 4096) -> tuple[SignedExposureEdge, dict]:
    """Inner-TRAIN select edge epochs, then refit on all TRAIN pairs."""
    torch.manual_seed(int(seed))
    train = np.flatnonzero(design.split == 0)
    if len(train) < 20:
        raise ValueError("T2-S1 needs at least 20 TRAIN one-step pairs")
    cut = int(np.clip(math.floor(0.8 * len(train)), 1, len(train) - 1))
    inner_train, inner_validation = train[:cut], train[cut:]
    initial = SignedExposureEdge(design.current_state.shape[1]).to(device)
    base = _evaluate_rows(
        model, initial, design, inner_validation, device=device,
        batch_size=batch_size,
    )
    best_epoch = 0
    best_value = base.joint_nll_per_event
    edge = SignedExposureEdge(design.current_state.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        edge.parameters(), lr=float(learning_rate), weight_decay=1e-3
    )
    trajectory = [{"epoch": 0, "joint_nll": best_value}]
    rng = np.random.default_rng(int(seed))
    for epoch in range(1, int(epochs) + 1):
        order = rng.permutation(inner_train)
        for lo in range(0, len(order), int(batch_size)):
            loss, _ = _score(
                model, edge, design, order[lo:lo + int(batch_size)],
                device=device, require_grad=True,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(edge.parameters(), 1.0)
            optimizer.step()
        if epoch == 1 or epoch % 5 == 0:
            metrics = _evaluate_rows(
                model, edge, design, inner_validation,
                device=device, batch_size=batch_size,
            )
            trajectory.append({"epoch": epoch, "joint_nll": metrics.joint_nll_per_event})
            if metrics.joint_nll_per_event < best_value:
                best_value = metrics.joint_nll_per_event
                best_epoch = epoch
    edge = SignedExposureEdge(design.current_state.shape[1]).to(device)
    if best_epoch:
        optimizer = torch.optim.AdamW(
            edge.parameters(), lr=float(learning_rate), weight_decay=1e-3
        )
        refit_rng = np.random.default_rng(int(seed))
        for _ in range(best_epoch):
            order = refit_rng.permutation(train)
            for lo in range(0, len(order), int(batch_size)):
                loss, _ = _score(
                    model, edge, design, order[lo:lo + int(batch_size)],
                    device=device, require_grad=True,
                )
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(edge.parameters(), 1.0)
                optimizer.step()
    return edge.eval(), {
        "selected_epoch": int(best_epoch),
        "inner_validation_joint_nll": float(best_value),
        "trajectory": trajectory,
    }


def evaluate_edge(model, edge: SignedExposureEdge, design: OneStepDesign,
                  *, split: str, device: torch.device | str,
                  batch_size: int = 4096) -> OneStepMetrics:
    code = {"train": 0, "validation": 1}[split]
    rows = np.flatnonzero(design.split == code)
    return _evaluate_rows(
        model, edge, design, rows, device=device, batch_size=batch_size
    )
