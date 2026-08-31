"""Interictal-only state-instrument diagnostics for H2b v0.3.

All metrics in this module are computed from the frozen R1.7 state model and
its interictal TRAIN/D_state design.  No seizure-risk label or H2b probe value
is accepted as an input.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from src.topic5_continuous_marked_state_r1.r1_2 import FullAnchorDesign


LAG_MINUTES = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0, 240.0)


@dataclass(frozen=True)
class InterictalStateTrace:
    anchor_time: np.ndarray
    anchor_session: np.ndarray
    anchor_split: np.ndarray
    persistent_state: np.ndarray
    state_minus: np.ndarray
    memoryless_state: np.ndarray
    persistent_decoder: np.ndarray
    state_minus_decoder: np.ndarray
    memoryless_decoder: np.ndarray
    generator_state_step_norm: np.ndarray
    correction_state_step_norm: np.ndarray
    generator_decoder_step_norm: np.ndarray
    correction_decoder_step_norm: np.ndarray

    def validate(self) -> None:
        n = len(self.anchor_time)
        arrays = (
            self.anchor_session, self.anchor_split, self.persistent_state,
            self.state_minus, self.memoryless_state, self.persistent_decoder,
            self.state_minus_decoder, self.memoryless_decoder,
            self.generator_state_step_norm, self.correction_state_step_norm,
            self.generator_decoder_step_norm, self.correction_decoder_step_norm,
        )
        if any(len(value) != n for value in arrays):
            raise ValueError("interictal trace arrays disagree")
        if self.anchor_time.dtype != np.float64:
            raise ValueError("interictal trace absolute time must be float64")
        if self.anchor_session.dtype != np.int64:
            raise ValueError("interictal trace session must be int64")
        for value in arrays[2:]:
            if not np.isfinite(value).all():
                raise ValueError("interictal trace contains non-finite values")


def decoder_output(model: torch.nn.Module, state: torch.Tensor) -> torch.Tensor:
    """Frozen state contribution to timing and exact-mark decoder logits."""
    timing = model.state_timing(state)
    contact = model.state_contact(state)
    size = model.state_size(state)
    return torch.cat([timing, contact, size], dim=-1)


def scan_interictal_state(
    model: torch.nn.Module,
    design: FullAnchorDesign,
    embedding: np.ndarray,
    *,
    device: str | torch.device = "cpu",
) -> InterictalStateTrace:
    """Run the exact frozen filter and expose generator/correction components."""
    model.eval()
    for parameter in model.parameters():
        if parameter.requires_grad:
            raise ValueError("instrument scan received a trainable state model")
    observation = np.asarray(embedding, dtype=np.float32)
    if observation.shape[0] != len(design.anchor_time):
        raise ValueError("embedding/design anchor count mismatch")
    n = len(design.anchor_time)
    state_dim = int(model.state.dim)
    with torch.inference_mode():
        probe = torch.zeros((1, state_dim), dtype=next(model.parameters()).dtype,
                            device=device)
        decoder_dim = int(decoder_output(model, probe).shape[-1])
    plus = np.zeros((n, state_dim), dtype=np.float32)
    minus = np.zeros_like(plus)
    memoryless = np.zeros_like(plus)
    plus_decoder = np.zeros((n, decoder_dim), dtype=np.float32)
    minus_decoder = np.zeros_like(plus_decoder)
    memoryless_decoder = np.zeros_like(plus_decoder)
    generator_state_norm = np.zeros(n, dtype=np.float32)
    correction_state_norm = np.zeros(n, dtype=np.float32)
    generator_decoder_norm = np.zeros(n, dtype=np.float32)
    correction_decoder_norm = np.zeros(n, dtype=np.float32)
    dtype = next(model.parameters()).dtype
    matrix = model.state.generator.matrix().to(device=device, dtype=dtype)
    mu = model.state.generator.mu.to(device=device, dtype=dtype)

    with torch.inference_mode():
        for label in design.session_label:
            anchors = np.flatnonzero(design.anchor_session == label)
            if not len(anchors):
                continue
            anchors = anchors[np.argsort(design.anchor_time[anchors], kind="stable")]
            state = torch.zeros(state_dim, dtype=dtype, device=device)
            cursor = float(design.session_start_for(np.asarray([label]))[0])
            transition_cache: dict[float, torch.Tensor] = {}
            previous_decoder = decoder_output(model, state.unsqueeze(0))[0]
            for anchor in anchors:
                time = float(design.anchor_time[anchor])
                delta = round(max((time - cursor) / 60.0, 0.0), 9)
                transition = transition_cache.get(delta)
                if transition is None:
                    transition = torch.matrix_exp(matrix * delta)
                    transition_cache[delta] = transition
                state_minus = mu + torch.matmul(state - mu, transition.T)
                current_embedding = torch.as_tensor(
                    np.array(observation[anchor], copy=True), dtype=dtype, device=device,
                )
                state_plus = model.state.correction(
                    state_minus, current_embedding, enabled=True,
                )
                state_memoryless = model.state.correction(
                    mu, current_embedding, enabled=True,
                )
                dec_minus = decoder_output(model, state_minus.unsqueeze(0))[0]
                dec_plus = decoder_output(model, state_plus.unsqueeze(0))[0]
                dec_memoryless = decoder_output(
                    model, state_memoryless.unsqueeze(0)
                )[0]
                plus[anchor] = state_plus.detach().cpu().numpy()
                minus[anchor] = state_minus.detach().cpu().numpy()
                memoryless[anchor] = state_memoryless.detach().cpu().numpy()
                plus_decoder[anchor] = dec_plus.detach().cpu().numpy()
                minus_decoder[anchor] = dec_minus.detach().cpu().numpy()
                memoryless_decoder[anchor] = dec_memoryless.detach().cpu().numpy()
                generator_state_norm[anchor] = float(torch.linalg.vector_norm(
                    state_minus - state
                ))
                correction_state_norm[anchor] = float(torch.linalg.vector_norm(
                    state_plus - state_minus
                ))
                generator_decoder_norm[anchor] = float(torch.linalg.vector_norm(
                    dec_minus - previous_decoder
                ))
                correction_decoder_norm[anchor] = float(torch.linalg.vector_norm(
                    dec_plus - dec_minus
                ))
                state = state_plus
                previous_decoder = dec_plus
                cursor = time
    trace = InterictalStateTrace(
        anchor_time=np.asarray(design.anchor_time, dtype=np.float64),
        anchor_session=np.asarray(design.anchor_session, dtype=np.int64),
        anchor_split=np.asarray(design.anchor_split, dtype=np.int64),
        persistent_state=plus,
        state_minus=minus,
        memoryless_state=memoryless,
        persistent_decoder=plus_decoder,
        state_minus_decoder=minus_decoder,
        memoryless_decoder=memoryless_decoder,
        generator_state_step_norm=generator_state_norm,
        correction_state_step_norm=correction_state_norm,
        generator_decoder_step_norm=generator_decoder_norm,
        correction_decoder_step_norm=correction_decoder_norm,
    )
    trace.validate()
    return trace


def standardise_decoder(
    train: np.ndarray, target: np.ndarray, *, minimum_scale: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_value = np.asarray(train, dtype=np.float64)
    target_value = np.asarray(target, dtype=np.float64)
    centre = np.mean(train_value, axis=0)
    scale = np.std(train_value, axis=0, ddof=0)
    active = np.isfinite(scale) & (scale > float(minimum_scale))
    if not bool(active.any()):
        return (
            np.empty((len(target_value), 0), dtype=np.float64),
            centre, scale, active,
        )
    return (target_value[:, active] - centre[active]) / scale[active], centre, scale, active


def effective_rank(values: np.ndarray) -> dict[str, float | int | None]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("effective-rank input must be two dimensional")
    if len(matrix) < 2 or matrix.shape[1] == 0:
        return {"effective_rank": 0.0, "top_pc_share": None, "matrix_rank": 0}
    matrix = matrix - np.mean(matrix, axis=0, keepdims=True)
    singular = np.linalg.svd(matrix, full_matrices=False, compute_uv=False)
    variance = singular ** 2
    total = float(np.sum(variance))
    if total <= 1e-20:
        return {"effective_rank": 0.0, "top_pc_share": None, "matrix_rank": 0}
    eig = variance / total
    return {
        "effective_rank": float(1.0 / np.sum(eig ** 2)),
        "top_pc_share": float(eig[0]),
        "matrix_rank": int(np.sum(singular > singular[0] * 1e-8)),
    }


def lagged_decoder_autocorrelation(
    time_epoch: np.ndarray,
    session: np.ndarray,
    values: np.ndarray,
    *,
    lag_minutes: Sequence[float] = LAG_MINUTES,
) -> dict[str, Any]:
    time = np.asarray(time_epoch, dtype=np.float64)
    group = np.asarray(session, dtype=np.int64)
    matrix = np.asarray(values, dtype=np.float64)
    if len(time) != len(group) or len(time) != len(matrix):
        raise ValueError("autocorrelation arrays disagree")
    # Autocorrelation concerns within-segment fluctuations.  A global centre
    # leaves between-segment offsets in every within-segment pair and can make
    # a piecewise-constant trace look arbitrarily slow.
    matrix = np.array(matrix, copy=True)
    for label in np.unique(group):
        rows = np.flatnonzero(group == label)
        matrix[rows] -= np.mean(matrix[rows], axis=0, keepdims=True)
    result = []
    for lag in lag_minutes:
        left_values = []
        right_values = []
        target_delta = float(lag) * 60.0
        tolerance = max(30.0, min(120.0, 0.2 * target_delta))
        for label in np.unique(group):
            rows = np.flatnonzero(group == label)
            if len(rows) < 2:
                continue
            rows = rows[np.argsort(time[rows], kind="stable")]
            local_time = time[rows]
            targets = local_time + target_delta
            position = np.searchsorted(local_time, targets, side="left")
            for source_index, candidate in enumerate(position):
                choices = [value for value in (candidate - 1, candidate)
                           if 0 <= value < len(rows) and value > source_index]
                if not choices:
                    continue
                selected = min(choices, key=lambda value: abs(
                    local_time[value] - targets[source_index]
                ))
                if abs(local_time[selected] - targets[source_index]) > tolerance:
                    continue
                left_values.append(matrix[rows[source_index]])
                right_values.append(matrix[rows[selected]])
        if not left_values or matrix.shape[1] == 0:
            correlation = None
            n_pairs = 0
        else:
            left = np.asarray(left_values, dtype=np.float64)
            right = np.asarray(right_values, dtype=np.float64)
            numerator = float(np.sum(left * right))
            denominator = math.sqrt(float(np.sum(left ** 2) * np.sum(right ** 2)))
            correlation = numerator / denominator if denominator > 1e-20 else None
            n_pairs = int(len(left))
        result.append({
            "lag_minutes": float(lag),
            "correlation": correlation,
            "n_pairs": n_pairs,
        })
    threshold = math.exp(-1.0)
    empirical_tau = None
    previous_lag, previous_corr = 0.0, 1.0
    for row in result:
        corr = row["correlation"]
        if corr is None or not np.isfinite(corr):
            continue
        lag = float(row["lag_minutes"])
        if corr <= threshold:
            if previous_corr <= threshold or previous_corr == corr:
                empirical_tau = lag
            else:
                fraction = (previous_corr - threshold) / (previous_corr - corr)
                empirical_tau = previous_lag + fraction * (lag - previous_lag)
            break
        previous_lag, previous_corr = lag, float(corr)
    return {
        "lags": result,
        "e_folding_threshold": threshold,
        "empirical_tau_minutes": empirical_tau,
        "right_censored": empirical_tau is None,
    }


def _adjacent_decoder_distances(
    time_epoch: np.ndarray,
    session: np.ndarray,
    values: np.ndarray,
    *,
    maximum_gap_seconds: float = 90.0,
) -> np.ndarray:
    time = np.asarray(time_epoch, dtype=np.float64)
    group = np.asarray(session, dtype=np.int64)
    matrix = np.asarray(values, dtype=np.float64)
    distances: list[np.ndarray] = []
    for label in np.unique(group):
        rows = np.flatnonzero(group == label)
        rows = rows[np.argsort(time[rows], kind="stable")]
        if len(rows) < 2:
            continue
        keep = np.diff(time[rows]) <= float(maximum_gap_seconds)
        if bool(keep.any()):
            distances.append(np.linalg.norm(
                matrix[rows[1:][keep]] - matrix[rows[:-1][keep]], axis=1,
            ))
    return np.concatenate(distances) if distances else np.empty(0, dtype=np.float64)


def shuffled_temporal_structure_null(
    time_epoch: np.ndarray,
    session: np.ndarray,
    values: np.ndarray,
    *,
    n_permutations: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Test local decoder continuity against within-segment time shuffles."""
    time = np.asarray(time_epoch, dtype=np.float64)
    group = np.asarray(session, dtype=np.int64)
    matrix = np.asarray(values, dtype=np.float64)
    observed_values = _adjacent_decoder_distances(time, group, matrix)
    observed = _median(observed_values)
    null: list[float] = []
    for _ in range(int(n_permutations)):
        permuted = np.array(matrix, copy=True)
        for label in np.unique(group):
            rows = np.flatnonzero(group == label)
            permuted[rows] = matrix[rng.permutation(rows)]
        value = _median(_adjacent_decoder_distances(time, group, permuted))
        if value is not None and np.isfinite(value):
            null.append(float(value))
    p_lower = (
        float((1 + sum(value <= float(observed) for value in null)) / (1 + len(null)))
        if observed is not None and null else None
    )
    return {
        "observed_median_adjacent_decoder_distance": observed,
        "n_observed_adjacent_pairs": int(len(observed_values)),
        "n_finite_permutations": int(len(null)),
        "null_median": float(np.median(null)) if null else None,
        "null_q05": float(np.quantile(null, 0.05)) if null else None,
        "lower_tail_monte_carlo_p": p_lower,
        "temporally_smoother_than_shuffled": bool(
            p_lower is not None and p_lower <= 0.05
        ),
    }


def reset_phase_explained_variance(
    time_epoch: np.ndarray,
    session: np.ndarray,
    values: np.ndarray,
) -> float | None:
    """Fraction of decoder variance explained by proximity to a segment reset."""
    time = np.asarray(time_epoch, dtype=np.float64)
    group = np.asarray(session, dtype=np.int64)
    matrix = np.asarray(values, dtype=np.float64)
    labels = np.unique(group)
    # With one segment, chronological drift and time-since-reset are the same
    # regressor.  Calling either one a reset artefact would be unidentifiable.
    if len(matrix) < 5 or matrix.shape[1] == 0 or len(labels) < 2:
        return None
    elapsed = np.zeros(len(time), dtype=np.float64)
    for label in np.unique(group):
        rows = np.flatnonzero(group == label)
        elapsed[rows] = np.maximum(time[rows] - np.min(time[rows]), 0.0) / 60.0
    segment_design = np.column_stack([(group == label).astype(np.float64)
                                      for label in labels])
    phase_design = np.column_stack([
        np.log1p(elapsed),
        (elapsed <= 1.0).astype(np.float64),
        (elapsed <= 5.0).astype(np.float64),
    ])
    base_fitted = segment_design @ np.linalg.lstsq(
        segment_design, matrix, rcond=None,
    )[0]
    full_design = np.column_stack([segment_design, phase_design])
    full_fitted = full_design @ np.linalg.lstsq(full_design, matrix, rcond=None)[0]
    base_sse = float(np.sum((matrix - base_fitted) ** 2))
    if base_sse <= 1e-20:
        return None
    full_sse = float(np.sum((matrix - full_fitted) ** 2))
    return float(np.clip((base_sse - full_sse) / base_sse, 0.0, 1.0))


def open_loop_and_reset_diagnostics(
    model: torch.nn.Module,
    trace: InterictalStateTrace,
    embedding: np.ndarray,
    *,
    train: np.ndarray,
    validation: np.ndarray,
    decoder_centre: np.ndarray,
    decoder_scale: np.ndarray,
    decoder_active: np.ndarray,
    horizons_minutes: Sequence[float] = (0.5, 1.0, 2.0, 5.0, 15.0, 30.0, 60.0),
    maximum_reset_trials: int = 8,
) -> dict[str, Any]:
    """Measure autonomous retention and recovery from artificial state resets."""
    del train  # Scaling was already fit on TRAIN and is passed explicitly.
    active = np.asarray(decoder_active, dtype=bool)
    if not bool(active.any()):
        return {
            "status": "NOT_ESTIMABLE_COLLAPSED_DECODER",
            "open_loop": [],
            "reset_trials": [],
            "preliminary_open_loop_reset_pass": False,
        }
    scale = np.asarray(decoder_scale, dtype=np.float64)[active]
    centre = np.asarray(decoder_centre, dtype=np.float64)[active]

    def standard_decoder(state: torch.Tensor) -> np.ndarray:
        with torch.inference_mode():
            value = decoder_output(model, state).detach().cpu().numpy().astype(np.float64)
        return (value[:, active] - centre) / scale

    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    valid_state = torch.as_tensor(
        trace.persistent_state[validation], dtype=dtype, device=device,
    )
    mu = model.state.generator.mu.detach()
    mu_decoder = standard_decoder(mu.unsqueeze(0))[0]
    start_decoder = standard_decoder(valid_state)
    start_distance = np.linalg.norm(start_decoder - mu_decoder, axis=1)
    usable = start_distance > 1e-6
    matrix = model.state.generator.matrix().to(dtype)
    open_loop_rows = []
    for horizon in horizons_minutes:
        transition = torch.matrix_exp(matrix * float(horizon))
        flowed = mu.unsqueeze(0) + torch.matmul(
            valid_state - mu.unsqueeze(0), transition.T,
        )
        moved = standard_decoder(flowed)
        distance = np.linalg.norm(moved - mu_decoder, axis=1)
        ratio = distance[usable] / start_distance[usable] if bool(usable.any()) else []
        open_loop_rows.append({
            "horizon_minutes": float(horizon),
            "median_decoder_retention_ratio": _median(np.asarray(ratio)),
            "n_states": int(np.sum(usable)),
        })
    threshold = math.exp(-1.0)
    e_folding = None
    previous_horizon, previous_ratio = 0.0, 1.0
    for row in open_loop_rows:
        ratio = row["median_decoder_retention_ratio"]
        if ratio is None:
            continue
        horizon = float(row["horizon_minutes"])
        if float(ratio) <= threshold:
            if previous_ratio <= threshold or previous_ratio == float(ratio):
                e_folding = horizon
            else:
                fraction = (previous_ratio - threshold) / (previous_ratio - float(ratio))
                e_folding = previous_horizon + fraction * (horizon - previous_horizon)
            break
        previous_horizon, previous_ratio = horizon, float(ratio)

    observation = np.asarray(embedding, dtype=np.float32)
    candidate: list[int] = []
    for label in np.unique(trace.anchor_session[validation]):
        rows = np.flatnonzero(validation & (trace.anchor_session == label))
        rows = rows[np.argsort(trace.anchor_time[rows], kind="stable")]
        if len(rows) < 22:
            continue
        eligible = rows[:-20]
        take = np.linspace(0, len(eligible) - 1,
                           min(maximum_reset_trials, len(eligible))).round().astype(int)
        candidate.extend(int(eligible[index]) for index in np.unique(take))
    if len(candidate) > int(maximum_reset_trials):
        take = np.linspace(0, len(candidate) - 1, int(maximum_reset_trials)).round().astype(int)
        candidate = [candidate[index] for index in np.unique(take)]
    reset_rows = []
    with torch.inference_mode():
        for reset_anchor in candidate:
            label = int(trace.anchor_session[reset_anchor])
            future = np.flatnonzero(
                validation & (trace.anchor_session == label)
                & (trace.anchor_time >= trace.anchor_time[reset_anchor])
            )
            future = future[np.argsort(trace.anchor_time[future], kind="stable")]
            alternative = model.state.correction(
                mu,
                torch.as_tensor(
                    np.array(observation[reset_anchor], copy=True),
                    dtype=dtype, device=device,
                ),
                enabled=True,
            )
            cursor = float(trace.anchor_time[reset_anchor])
            distances = []
            elapsed = []
            for position, anchor in enumerate(future):
                if position:
                    delta = max((float(trace.anchor_time[anchor]) - cursor) / 60.0, 0.0)
                    transition = torch.matrix_exp(matrix * delta)
                    state_minus = mu + torch.matmul(alternative - mu, transition.T)
                    alternative = model.state.correction(
                        state_minus,
                        torch.as_tensor(
                            np.array(observation[anchor], copy=True),
                            dtype=dtype, device=device,
                        ),
                        enabled=True,
                    )
                    cursor = float(trace.anchor_time[anchor])
                observed = standard_decoder(alternative.unsqueeze(0))[0]
                reference = (
                    trace.persistent_decoder[anchor, active].astype(np.float64) - centre
                ) / scale
                distances.append(float(np.linalg.norm(observed - reference)))
                elapsed.append(float(
                    (trace.anchor_time[anchor] - trace.anchor_time[reset_anchor]) / 60.0
                ))
            initial = distances[0] if distances else 0.0
            if initial <= 1e-6:
                continue
            recovery = next((minute for minute, value in zip(elapsed[1:], distances[1:])
                             if value <= initial * threshold), None)
            reset_rows.append({
                "reset_anchor_time": float(trace.anchor_time[reset_anchor]),
                "session": label,
                "initial_decoder_distance": float(initial),
                "recovery_minutes": recovery,
                "right_censored": recovery is None,
                "observed_followup_minutes": float(elapsed[-1]) if elapsed else 0.0,
            })
    recovery = [row["recovery_minutes"] for row in reset_rows
                if row["recovery_minutes"] is not None]
    censored_fraction = (
        float(sum(row["right_censored"] for row in reset_rows) / len(reset_rows))
        if reset_rows else None
    )
    retention_beyond_window = bool(
        (e_folding is not None and e_folding > 0.5)
        or (e_folding is None and any(row["n_states"] for row in open_loop_rows))
    )
    reset_memory_beyond_window = bool(
        reset_rows and (
            (recovery and float(np.median(recovery)) > 0.5)
            or (censored_fraction is not None and censored_fraction >= 0.5)
        )
    )
    return {
        "status": "COMPLETE",
        "open_loop": open_loop_rows,
        "open_loop_e_folding_minutes": e_folding,
        "open_loop_right_censored": e_folding is None,
        "reset_trials": reset_rows,
        "n_informative_reset_trials": len(reset_rows),
        "median_reset_recovery_minutes": (
            float(np.median(recovery)) if recovery else None
        ),
        "reset_recovery_right_censored_fraction": censored_fraction,
        "preliminary_open_loop_reset_pass": bool(
            retention_beyond_window and reset_memory_beyond_window
            and len(reset_rows) >= 3
        ),
    }


def _flow_from_cut_states(
    model: torch.nn.Module,
    cut_state: np.ndarray,
    cut_time: np.ndarray,
    query_time: np.ndarray,
) -> torch.Tensor:
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    state = torch.as_tensor(cut_state, dtype=dtype, device=device)
    delta = torch.as_tensor(
        np.maximum(np.asarray(query_time) - np.asarray(cut_time), 0.0) / 60.0,
        dtype=dtype, device=device,
    )
    matrix = model.state.generator.matrix().to(dtype)
    transition = torch.matrix_exp(matrix.unsqueeze(0) * delta[:, None, None])
    mu = model.state.generator.mu
    return mu.unsqueeze(0) + torch.matmul(
        transition, (state - mu.unsqueeze(0)).unsqueeze(-1),
    ).squeeze(-1)


def _score_open_loop_rows(
    model: torch.nn.Module,
    design: FullAnchorDesign,
    *,
    anchor_state: np.ndarray,
    event_rows: np.ndarray,
    event_cut: np.ndarray,
    quadrature_rows: np.ndarray,
    quadrature_cut: np.ndarray,
) -> dict[str, float | int | None]:
    event_log = 0.0
    mark_log = 0.0
    survival = 0.0
    with torch.inference_mode():
        for lo in range(0, len(event_rows), 4096):
            rows = event_rows[lo:lo + 4096]
            cuts = event_cut[lo:lo + len(rows)]
            state = _flow_from_cut_states(
                model, anchor_state[cuts], design.anchor_time[cuts],
                design.event_time[rows],
            )
            history = torch.as_tensor(
                design.event_history[rows], device=state.device,
            )
            event_log += float(model.timing_log_rate(history, state).sum())
            mark = model.mark_terms(
                history, state,
                torch.as_tensor(
                    design.event_group_ids[rows], dtype=torch.long, device=state.device,
                ),
                torch.as_tensor(
                    design.event_group_count[rows], dtype=torch.long, device=state.device,
                ),
            )
            mark_log += float(mark.event_log_prob.sum())
        for lo in range(0, len(quadrature_rows), 65536):
            rows = quadrature_rows[lo:lo + 65536]
            cuts = quadrature_cut[lo:lo + len(rows)]
            state = _flow_from_cut_states(
                model, anchor_state[cuts], design.anchor_time[cuts],
                design.quadrature_time[rows],
            )
            history = torch.as_tensor(
                design.quadrature_history[rows], device=state.device,
            )
            log_rate = model.timing_log_rate(history, state)
            weight = torch.as_tensor(
                design.quadrature_weight_seconds[rows],
                dtype=log_rate.dtype, device=state.device,
            )
            survival += float(torch.sum(weight * torch.exp(torch.clamp(log_rate, max=20.0))))
    if not len(event_rows):
        return {
            "joint_nll_per_event": None,
            "timing_nll_per_event": None,
            "mark_nll_per_event": None,
            "n_events": 0,
            "recorded_seconds": float(np.sum(
                design.quadrature_weight_seconds[quadrature_rows]
            )),
        }
    timing = (survival - event_log) / len(event_rows)
    mark = -mark_log / len(event_rows)
    return {
        "joint_nll_per_event": float(timing + mark),
        "timing_nll_per_event": float(timing),
        "mark_nll_per_event": float(mark),
        "n_events": int(len(event_rows)),
        "recorded_seconds": float(np.sum(
            design.quadrature_weight_seconds[quadrature_rows]
        )),
    }


def open_loop_interictal_prediction(
    model: torch.nn.Module,
    design: FullAnchorDesign,
    trace: InterictalStateTrace,
    validation: np.ndarray,
    *,
    horizons_minutes: Sequence[float] = (5.0, 15.0, 30.0, 60.0),
) -> dict[str, Any]:
    """Score future IED timing/marks after closing the observer at sparse cuts."""
    maximum_horizon_seconds = float(max(horizons_minutes)) * 60.0
    cuts: list[int] = []
    for label in np.unique(trace.anchor_session[validation]):
        rows = np.flatnonzero(validation & (trace.anchor_session == label))
        rows = rows[np.argsort(trace.anchor_time[rows], kind="stable")]
        if not len(rows):
            continue
        last_time = float(trace.anchor_time[rows[-1]])
        cursor = float(trace.anchor_time[rows[0]])
        while cursor + maximum_horizon_seconds <= last_time + 1e-9:
            position = int(np.searchsorted(trace.anchor_time[rows], cursor, side="left"))
            if position >= len(rows):
                break
            cut = int(rows[position])
            if float(trace.anchor_time[cut]) + maximum_horizon_seconds > last_time + 1e-9:
                break
            cuts.append(cut)
            cursor = float(trace.anchor_time[cut]) + maximum_horizon_seconds
    rows_by_horizon = []
    for horizon in horizons_minutes:
        event_pieces: list[np.ndarray] = []
        event_cuts: list[np.ndarray] = []
        q_pieces: list[np.ndarray] = []
        q_cuts: list[np.ndarray] = []
        for cut in cuts:
            start = float(trace.anchor_time[cut])
            stop = start + float(horizon) * 60.0
            label = int(trace.anchor_session[cut])
            event = np.flatnonzero(
                (design.event_split == 1)
                & (design.event_session == label)
                & (design.event_time > start)
                & (design.event_time <= stop)
            )
            quadrature = np.flatnonzero(
                (design.quadrature_split == 1)
                & (design.quadrature_session == label)
                & (design.quadrature_time > start)
                & (design.quadrature_time <= stop)
            )
            if len(event):
                event_pieces.append(event)
                event_cuts.append(np.full(len(event), cut, dtype=np.int64))
            if len(quadrature):
                q_pieces.append(quadrature)
                q_cuts.append(np.full(len(quadrature), cut, dtype=np.int64))
        event_rows = np.concatenate(event_pieces) if event_pieces else np.empty(0, dtype=np.int64)
        event_cut = np.concatenate(event_cuts) if event_cuts else np.empty(0, dtype=np.int64)
        q_rows = np.concatenate(q_pieces) if q_pieces else np.empty(0, dtype=np.int64)
        q_cut = np.concatenate(q_cuts) if q_cuts else np.empty(0, dtype=np.int64)
        persistent = _score_open_loop_rows(
            model, design, anchor_state=trace.persistent_state,
            event_rows=event_rows, event_cut=event_cut,
            quadrature_rows=q_rows, quadrature_cut=q_cut,
        )
        memoryless = _score_open_loop_rows(
            model, design, anchor_state=trace.memoryless_state,
            event_rows=event_rows, event_cut=event_cut,
            quadrature_rows=q_rows, quadrature_cut=q_cut,
        )
        persistent_value = persistent["joint_nll_per_event"]
        memoryless_value = memoryless["joint_nll_per_event"]
        difference = (
            float(persistent_value) - float(memoryless_value)
            if persistent_value is not None and memoryless_value is not None else None
        )
        rows_by_horizon.append({
            "horizon_minutes": float(horizon),
            "n_nonoverlap_cut_windows": len(cuts),
            "persistent": persistent,
            "memoryless": memoryless,
            "persistent_minus_memoryless_joint_nll_per_event": difference,
            "direction_favourable": bool(difference is not None and difference < 0),
        })
    predictive_horizon = None
    for row in rows_by_horizon:
        if row["direction_favourable"]:
            predictive_horizon = float(row["horizon_minutes"])
        else:
            break
    return {
        "status": "COMPLETE" if cuts else "NOT_ESTIMABLE_NO_COMPLETE_WINDOWS",
        "observer_mode": "closed_after_each_cut",
        "event_history_mode": "teacher_forced_recorded_history",
        "interpretation": (
            "state-generator retention conditional on recorded future event history; "
            "not a fully autonomous event rollout"
        ),
        "horizons": rows_by_horizon,
        "predictive_horizon_minutes": predictive_horizon,
        "preliminary_predictive_pass": bool(
            predictive_horizon is not None and predictive_horizon >= 5.0
        ),
    }


def _median(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if len(finite) else None


def summarise_instrument_trace(
    model: torch.nn.Module,
    design: FullAnchorDesign,
    trace: InterictalStateTrace,
    *,
    state_start: float,
    state_stop: float,
    interictal_persistent_minus_memoryless_joint: float,
    embedding: np.ndarray,
    rng_seed: int,
    n_null_permutations: int = 100,
) -> dict[str, Any]:
    train = trace.anchor_split == 0
    validation = (
        (trace.anchor_split == 1)
        & (trace.anchor_time >= float(state_start))
        & (trace.anchor_time < float(state_stop))
    )
    if int(np.sum(train)) < 2 or int(np.sum(validation)) < 2:
        raise ValueError("interictal TRAIN/D_state support is insufficient")
    standard, centre, scale, active = standardise_decoder(
        trace.persistent_decoder[train], trace.persistent_decoder[validation]
    )
    memory_standard = (
        (trace.memoryless_decoder[validation][:, active] - centre[active]) / scale[active]
        if bool(active.any()) else np.empty((int(np.sum(validation)), 0))
    )
    rank = effective_rank(standard)
    persistence_distance = (
        np.linalg.norm(standard - memory_standard, axis=1)
        if standard.shape[1] else np.zeros(len(standard))
    )
    autocorrelation = lagged_decoder_autocorrelation(
        trace.anchor_time[validation], trace.anchor_session[validation], standard,
    )
    temporal_null = shuffled_temporal_structure_null(
        trace.anchor_time[validation], trace.anchor_session[validation], standard,
        n_permutations=int(n_null_permutations),
        rng=np.random.default_rng(int(rng_seed)),
    )
    reset_r2 = reset_phase_explained_variance(
        trace.anchor_time[validation], trace.anchor_session[validation], standard,
    )
    matrix = model.state.generator.matrix().detach().cpu().numpy().astype(np.float64)
    eigenvalue = np.linalg.eigvals(matrix)
    slowest_real = float(np.max(np.real(eigenvalue)))
    analytic_tau = float(-1.0 / slowest_real) if slowest_real < 0 else None
    durations = []
    for label in np.unique(trace.anchor_session[validation]):
        local = trace.anchor_time[validation & (trace.anchor_session == label)]
        if len(local) >= 2:
            durations.append(float((np.max(local) - np.min(local)) / 60.0))
    median_segment = float(np.median(durations)) if durations else None
    empirical_tau = autocorrelation["empirical_tau_minutes"]
    q1_absolute_pass = bool(
        int(np.sum(active)) >= 2
        and float(rank["effective_rank"] or 0.0) >= 2.0
        and rank["top_pc_share"] is not None
        and float(rank["top_pc_share"]) <= 0.95
        and float(np.median(persistence_distance)) > 1e-6
    )
    reset_not_dominant = bool(reset_r2 is None or reset_r2 < 0.50)
    q1_pass = bool(
        q1_absolute_pass
        and temporal_null["temporally_smoother_than_shuffled"]
        and reset_not_dominant
    )
    generator_decoder = trace.generator_decoder_step_norm[validation]
    correction_decoder = trace.correction_decoder_step_norm[validation]
    median_generator = _median(generator_decoder)
    median_correction = _median(correction_decoder)
    total = float((median_generator or 0.0) + (median_correction or 0.0))
    generator_fraction = (float(median_generator or 0.0) / total) if total > 0 else 0.0
    q3_absolute_pass = bool(
        (median_generator or 0.0) > 1e-6 and generator_fraction >= 0.10
    )
    open_loop_reset = open_loop_and_reset_diagnostics(
        model, trace, embedding,
        train=train, validation=validation,
        decoder_centre=centre, decoder_scale=scale, decoder_active=active,
    )
    open_loop_prediction = open_loop_interictal_prediction(
        model, design, trace, validation,
    )
    q3_pass = bool(
        q3_absolute_pass
        and open_loop_reset["preliminary_open_loop_reset_pass"]
        and open_loop_prediction["preliminary_predictive_pass"]
    )
    q4_pass = bool(
        empirical_tau is not None
        and float(empirical_tau) > 0.5
        and median_segment is not None
        and float(empirical_tau) < float(median_segment)
    )
    return {
        "status": "COMPLETE_DIAGNOSTIC_PENDING_Q5_Q6",
        "n_train_anchors": int(np.sum(train)),
        "n_d_state_anchors": int(np.sum(validation)),
        "decoder_dimension": int(trace.persistent_decoder.shape[1]),
        "active_decoder_dimensions": int(np.sum(active)),
        "Q1_noncollapse": {
            **rank,
            "median_persistent_memoryless_decoder_distance": float(
                np.median(persistence_distance)
            ),
            "preliminary_absolute_threshold_pass": q1_absolute_pass,
            "collapsed_null_effective_rank": 0.0,
            "temporal_shuffled_null": temporal_null,
            "reset_phase_explained_variance": reset_r2,
            "reset_phase_estimable": reset_r2 is not None,
            "reset_not_dominant_or_not_estimable": reset_not_dominant,
            "null_calibrated_pass": q1_pass,
        },
        "Q2_cross_window_information": {
            "persistent_minus_memoryless_joint_nll_per_event": float(
                interictal_persistent_minus_memoryless_joint
            ),
            "direction_favourable": bool(
                float(interictal_persistent_minus_memoryless_joint) < 0
            ),
            "source": "frozen R1.7B D_state interictal validation",
        },
        "Q3_generator_contribution": {
            "median_generator_state_step_norm": _median(
                trace.generator_state_step_norm[validation]
            ),
            "median_observation_correction_state_step_norm": _median(
                trace.correction_state_step_norm[validation]
            ),
            "median_generator_decoder_step_norm": median_generator,
            "median_observation_correction_decoder_step_norm": median_correction,
            "generator_fraction_of_decoder_motion": generator_fraction,
            "preliminary_absolute_threshold_pass": q3_absolute_pass,
            "open_loop_and_reset": open_loop_reset,
            "open_loop_interictal_prediction": open_loop_prediction,
            "open_loop_reset_pass": q3_pass,
        },
        "Q4_time_constant": {
            "analytic_generator_slowest_mode_minutes": analytic_tau,
            "empirical_decoder_tau_minutes": empirical_tau,
            "empirical_tau_right_censored": autocorrelation["right_censored"],
            "median_d_state_continuous_segment_minutes": median_segment,
            "preliminary_absolute_threshold_pass": q4_pass,
            "interpretation_if_right_censored": (
                "TIME_CONSTANT_NOT_IDENTIFIABLE_WITHIN_AVAILABLE_SEGMENTS; "
                "not a biological failure"
            ),
            "autocorrelation": autocorrelation["lags"],
        },
        "Q5_seed_stability": {"status": "PENDING_PATIENT_AGGREGATION"},
        "Q6_not_only_clock": {"status": "PENDING_NUISANCE_AUDIT"},
        "state_qualified": False,
        "state_qualified_reason": "Q5 patient aggregation and Q6 are not complete",
    }


def trace_npz_payload(trace: InterictalStateTrace, *, state_start: float,
                      state_stop: float) -> Mapping[str, np.ndarray]:
    keep = (
        (trace.anchor_split == 1)
        & (trace.anchor_time >= float(state_start))
        & (trace.anchor_time < float(state_stop))
    )
    return {
        "anchor_time": trace.anchor_time[keep].astype(np.float64, copy=False),
        "anchor_session": trace.anchor_session[keep].astype(np.int64, copy=False),
        "persistent_state": trace.persistent_state[keep].astype(np.float32, copy=False),
        "state_minus": trace.state_minus[keep].astype(np.float32, copy=False),
        "memoryless_state": trace.memoryless_state[keep].astype(np.float32, copy=False),
        "persistent_decoder": trace.persistent_decoder[keep].astype(np.float32, copy=False),
        "state_minus_decoder": trace.state_minus_decoder[keep].astype(np.float32, copy=False),
        "memoryless_decoder": trace.memoryless_decoder[keep].astype(np.float32, copy=False),
        "generator_state_step_norm": trace.generator_state_step_norm[keep].astype(
            np.float32, copy=False
        ),
        "correction_state_step_norm": trace.correction_state_step_norm[keep].astype(
            np.float32, copy=False
        ),
        "generator_decoder_step_norm": trace.generator_decoder_step_norm[keep].astype(
            np.float32, copy=False
        ),
        "correction_decoder_step_norm": trace.correction_decoder_step_norm[keep].astype(
            np.float32, copy=False
        ),
    }
