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


def _median(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if len(finite) else None


def summarise_instrument_trace(
    model: torch.nn.Module,
    trace: InterictalStateTrace,
    *,
    state_start: float,
    state_stop: float,
    interictal_persistent_minus_memoryless_joint: float,
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
    q1_pass = bool(
        int(np.sum(active)) >= 2
        and float(rank["effective_rank"] or 0.0) >= 2.0
        and rank["top_pc_share"] is not None
        and float(rank["top_pc_share"]) <= 0.95
        and float(np.median(persistence_distance)) > 1e-6
    )
    generator_decoder = trace.generator_decoder_step_norm[validation]
    correction_decoder = trace.correction_decoder_step_norm[validation]
    median_generator = _median(generator_decoder)
    median_correction = _median(correction_decoder)
    total = float((median_generator or 0.0) + (median_correction or 0.0))
    generator_fraction = (float(median_generator or 0.0) / total) if total > 0 else 0.0
    q3_pass = bool((median_generator or 0.0) > 1e-6 and generator_fraction >= 0.10)
    q4_pass = bool(
        empirical_tau is not None
        and float(empirical_tau) > 1.0
        and median_segment is not None
        and float(empirical_tau) < 0.25 * float(median_segment)
    )
    return {
        "status": "COMPLETE_DIAGNOSTIC_PENDING_NULL_Q5_Q6",
        "n_train_anchors": int(np.sum(train)),
        "n_d_state_anchors": int(np.sum(validation)),
        "decoder_dimension": int(trace.persistent_decoder.shape[1]),
        "active_decoder_dimensions": int(np.sum(active)),
        "Q1_noncollapse": {
            **rank,
            "median_persistent_memoryless_decoder_distance": float(
                np.median(persistence_distance)
            ),
            "preliminary_absolute_threshold_pass": q1_pass,
            "final_null_calibrated_status": "PENDING",
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
            "preliminary_absolute_threshold_pass": q3_pass,
            "final_reset_memoryless_null_status": "PENDING",
        },
        "Q4_time_constant": {
            "analytic_generator_slowest_mode_minutes": analytic_tau,
            "empirical_decoder_tau_minutes": empirical_tau,
            "empirical_tau_right_censored": autocorrelation["right_censored"],
            "median_d_state_continuous_segment_minutes": median_segment,
            "preliminary_absolute_threshold_pass": q4_pass,
            "autocorrelation": autocorrelation["lags"],
        },
        "Q5_seed_stability": {"status": "PENDING_PATIENT_AGGREGATION"},
        "Q6_not_only_clock": {"status": "PENDING_NUISANCE_AUDIT"},
        "state_qualified": False,
        "state_qualified_reason": "Q1 null calibration, Q5 and Q6 are not complete",
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
