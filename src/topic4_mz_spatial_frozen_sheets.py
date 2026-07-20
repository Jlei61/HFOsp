"""Operational frozen-sheet tools for the mass-balanced MZ patch model.

These helpers keep ``z/p/m`` frozen and therefore test only the fast spatial
scaffold.  They do not implement the later persistence latch or claim a full
seizure lifecycle.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from src.sef_hfo_lif import TREF_E, TREF_I
from src.topic4_mz_spatial_patch import (
    LOCAL_FIELDS,
    PatchKernels,
    PatchParameters,
    PreparedPatchRHS,
    pack_patch_state,
    patch_rhs_fast_and_moments,
    state_size,
    unpack_patch_state,
)
from src.topic4_spatial_slowfast_stage0c import S_MAX


def lift_product_history(
    stage_templates: np.ndarray,
    kernels: PatchKernels,
    *,
    z: Sequence[float],
    parameters: PatchParameters,
    persistence: Sequence[float] | None = None,
    additive_fraction: Sequence[float] | None = None,
) -> np.ndarray:
    """Lift homogeneous Stage-0C histories into one coupled patch state.

    The synaptic coordinates are mixed by the fixed spatial operators, while
    the unique shared pool is the area-weighted combination of template pool
    histories.  The latter identity is exact only for the registered
    ``pool_p=1`` frozen-sheet gate.
    """

    checked_kernels = kernels.validate()
    checked_parameters = parameters.validate()
    templates = np.asarray(stage_templates, dtype=float)
    n_patches = checked_kernels.n_patches
    if templates.shape != (n_patches, 9) or not np.all(np.isfinite(templates)):
        raise ValueError("one finite Stage-0C template is required per patch")
    if checked_parameters.pool_p != 1.0:
        raise ValueError("product-history lift is registered only for pool_p=1")
    z_array = np.asarray(z, dtype=float)
    p_array = np.zeros(n_patches) if persistence is None else np.asarray(persistence, dtype=float)
    m_array = np.zeros(n_patches) if additive_fraction is None else np.asarray(additive_fraction, dtype=float)
    if z_array.shape != (n_patches,) or np.any((z_array <= 0.0) | (z_array > 1.0)):
        raise ValueError("z must align with patches and lie in (0,1]")
    if p_array.shape != (n_patches,) or np.any((p_array < 0.0) | (p_array > 1.0)):
        raise ValueError("persistence must align and lie in [0,1]")
    if m_array.shape != (n_patches,) or np.any((m_array < 0.0) | (m_array > 1.0)):
        raise ValueError("additive fractions must align and lie in [0,1]")

    local = {
        "rE": templates[:, 0].copy(),
        "rI": templates[:, 1].copy(),
        "sEE": checked_kernels.K_EE @ templates[:, 2],
        "sEI": checked_kernels.K_I @ templates[:, 3],
        "sIE": checked_kernels.K_I @ templates[:, 4],
        "sII": checked_kernels.K_I @ templates[:, 5],
        "rE_fast": templates[:, 6].copy(),
        "z": z_array,
        "p": p_array,
        "m": m_array,
    }
    weights = checked_kernels.weights()
    return pack_patch_state(
        local,
        mu_g=float(weights @ templates[:, 7]),
        s_g=float(weights @ templates[:, 8]),
    )


def integrate_frozen_patch_batch(
    initial_states: np.ndarray,
    prepared: PreparedPatchRHS,
    transfer: Any,
    *,
    dt_ms: float,
    duration_ms: float,
    save_dt_ms: float,
    section_level_khz: float = 0.020,
    rearm_level_khz: float = 0.015,
) -> dict[str, Any]:
    """Vectorized Euler forks with streamed local directed-return events."""

    state = np.asarray(initial_states, dtype=float).copy()
    n_patches = prepared.n_patches
    if state.ndim != 2 or state.shape[1] != state_size(n_patches) or not np.all(np.isfinite(state)):
        raise ValueError("initial states must be a finite aligned batch")
    if dt_ms <= 0.0 or duration_ms <= dt_ms or save_dt_ms < dt_ms:
        raise ValueError("invalid integration time contract")
    n_steps = int(round(float(duration_ms) / float(dt_ms)))
    save_stride = int(round(float(save_dt_ms) / float(dt_ms)))
    if not np.isclose(n_steps * dt_ms, duration_ms) or not np.isclose(save_stride * dt_ms, save_dt_ms):
        raise ValueError("duration and save interval must be integer multiples of dt")
    sample_steps = np.arange(0, n_steps + 1, save_stride, dtype=int)
    if sample_steps[-1] != n_steps:
        sample_steps = np.r_[sample_steps, n_steps]
    n_forks = state.shape[0]
    trace_shape = (sample_steps.size, n_forks, n_patches)
    trace_e = np.full(trace_shape, np.nan, dtype=np.float32)
    trace_i = np.full(trace_shape, np.nan, dtype=np.float32)
    trace_fast = np.full(trace_shape, np.nan, dtype=np.float32)
    trace_shared = np.full((sample_steps.size, n_forks, 2), np.nan, dtype=np.float32)
    support_violations = np.zeros((n_forks, n_patches), dtype=np.int64)
    bound_violations = np.zeros((n_forks, n_patches), dtype=np.int64)
    finite = np.ones(n_forks, dtype=bool)
    return_times: list[list[list[float]]] = [
        [[] for _ in range(n_patches)] for _ in range(n_forks)
    ]

    local0, _, _ = _unpack_batch(state, n_patches)
    previous_fast = local0["rE_fast"].copy()
    armed = previous_fast <= float(rearm_level_khz)
    sample_index = 0
    for step in range(n_steps + 1):
        rhs, moments = patch_rhs_fast_and_moments(state, prepared, transfer)
        mu_e, sigma_e, mu_i, sigma_i, _ = moments
        supported = transfer.support_mask(mu_e, sigma_e) & transfer.support_mask(mu_i, sigma_i)
        support_violations += ~supported
        local, mu_g, s_g = _unpack_batch(state, n_patches)
        finite_now = np.all(np.isfinite(state), axis=1) & np.all(np.isfinite(rhs), axis=1)
        finite &= finite_now
        bad = (
            (local["rE"] < -1e-9)
            | (local["rE"] > 1.0 / TREF_E + 1e-9)
            | (local["rI"] < -1e-9)
            | (local["rI"] > 1.0 / TREF_I + 1e-9)
            | (local["sEE"] < -1e-9)
            | (local["sEE"] > 1.0 / TREF_E + 1e-9)
            | (local["sEI"] < -1e-9)
            | (local["sEI"] > 1.0 / TREF_I + 1e-9)
            | (local["sIE"] < -1e-9)
            | (local["sIE"] > 1.0 / TREF_E + 1e-9)
            | (local["sII"] < -1e-9)
            | (local["sII"] > 1.0 / TREF_I + 1e-9)
            | (local["rE_fast"] < -1e-9)
            | (local["rE_fast"] > 1.0 / TREF_E + 1e-9)
            | (mu_g[:, None] < -1e-9)
            | (mu_g[:, None] > 1.0 + 1e-9)
            | (s_g[:, None] < -1e-9)
            | (s_g[:, None] > S_MAX + 1e-9)
        )
        bound_violations += bad
        if sample_index < sample_steps.size and step == sample_steps[sample_index]:
            trace_e[sample_index] = local["rE"]
            trace_i[sample_index] = local["rI"]
            trace_fast[sample_index] = local["rE_fast"]
            trace_shared[sample_index, :, 0] = mu_g
            trace_shared[sample_index, :, 1] = s_g
            sample_index += 1
        if step == n_steps:
            break
        next_state = state + float(dt_ms) * rhs
        next_local, _, _ = _unpack_batch(next_state, n_patches)
        next_fast = next_local["rE_fast"]
        armed |= previous_fast <= float(rearm_level_khz)
        crossed = armed & (previous_fast < float(section_level_khz)) & (next_fast >= float(section_level_khz))
        for fork_index, patch_index in np.argwhere(crossed):
            denominator = float(next_fast[fork_index, patch_index] - previous_fast[fork_index, patch_index])
            fraction = (
                float(section_level_khz - previous_fast[fork_index, patch_index]) / denominator
                if denominator > 0.0 else 0.0
            )
            return_times[int(fork_index)][int(patch_index)].append(
                float(step) * float(dt_ms) + fraction * float(dt_ms)
            )
        armed[crossed] = False
        state = next_state
        previous_fast = next_fast.copy()

    return {
        "time_ms": sample_steps.astype(float) * float(dt_ms),
        "rE_khz": trace_e,
        "rI_khz": trace_i,
        "rE_fast_khz": trace_fast,
        "shared_state": trace_shared,
        "final_state": state,
        "finite": finite,
        "support_violation_count": support_violations,
        "state_bound_violation_count": bound_violations,
        "return_times_ms": return_times,
    }


def summarize_local_state(
    time_ms: Sequence[float],
    rate_e_khz: Sequence[float],
    rate_fast_khz: Sequence[float],
    return_times_ms: Sequence[float],
    *,
    support_violation_count: int,
    state_bound_violation_count: int,
    finite: bool,
    discard_returns: int = 2,
) -> dict[str, Any]:
    """Classify one local trace without forcing it into L/C."""

    time = np.asarray(time_ms, dtype=float)
    rate = 1000.0 * np.asarray(rate_e_khz, dtype=float)
    fast = 1000.0 * np.asarray(rate_fast_khz, dtype=float)
    returns = np.asarray(return_times_ms, dtype=float)
    if time.ndim != 1 or rate.shape != time.shape or fast.shape != time.shape:
        raise ValueError("time and local traces must be aligned")
    peak = float(np.nanmax(rate))
    high_fraction = float(np.mean(rate > 100.0))
    if (
        not finite
        or int(support_violation_count) > 0
        or int(state_bound_violation_count) > 0
        or not np.all(np.isfinite(rate))
    ):
        status = "physical_or_numerical_failure"
    elif peak >= 120.0 or high_fraction >= 0.05:
        status = "unbounded_or_saturation"
    else:
        retained = returns[int(discard_returns):]
        intervals = np.diff(retained)
        valid_intervals = intervals[(intervals >= 300.0) & (intervals <= 12000.0)]
        recent = valid_intervals[-3:]
        period_cv = (
            float(np.std(recent) / np.mean(recent))
            if recent.size == 3 and float(np.mean(recent)) > 0.0 else None
        )
        cycle = bool(
            retained.size >= 4
            and recent.size == 3
            and period_cv is not None
            and period_cv <= 0.10
            and peak >= 20.0
        )
        previous_period = float(np.median(valid_intervals[-3:])) if valid_intervals.size else 0.0
        low_window_ms = max(1000.0, 2.0 * previous_period)
        tail = time >= max(float(time[-1]) - low_window_ms, float(time[0]))
        low = bool(
            np.all(rate[tail] < 5.0)
            and np.all(fast[tail] < 5.0)
            and not np.any(returns >= float(time[tail][0]))
        )
        if cycle:
            status = "C"
        elif low:
            status = "L"
        elif float(np.mean(rate[tail])) >= 20.0 and float(np.ptp(rate[tail])) <= 10.0:
            status = "tonic_plateau"
        else:
            status = "O_unresolved"
    retained = returns[int(discard_returns):]
    valid = np.diff(retained)
    valid = valid[(valid >= 300.0) & (valid <= 12000.0)]
    recent = valid[-3:]
    tail_values = rate[time >= max(time[-1] - 1000.0, time[0])]
    tail_mean = (
        float(np.mean(tail_values[np.isfinite(tail_values)]))
        if np.any(np.isfinite(tail_values)) else None
    )
    return {
        "status": status,
        "n_returns": int(returns.size),
        "n_retained_returns": int(retained.size),
        "recent_period_ms": float(np.mean(recent)) if recent.size == 3 else None,
        "recent_period_cv": float(np.std(recent) / np.mean(recent)) if recent.size == 3 else None,
        "peak_rE_hz": peak,
        "fraction_over_100hz": high_fraction,
        "tail_mean_rE_hz": tail_mean,
        "support_violation_count": int(support_violation_count),
        "state_bound_violation_count": int(state_bound_violation_count),
        "finite": bool(finite),
        "return_times_ms": [float(value) for value in returns],
    }


def sheet_label(core_status: str, surround_status: str) -> str:
    """Return LL/CL/LC/CC only when both local states are resolved."""

    if core_status in {"L", "C"} and surround_status in {"L", "C"}:
        return f"{core_status}{surround_status}"
    if "physical_or_numerical_failure" in {core_status, surround_status}:
        return "physical_or_numerical_failure"
    if "unbounded_or_saturation" in {core_status, surround_status}:
        return "unbounded_or_saturation"
    return "O_unresolved"


def _unpack_batch(
    state: np.ndarray,
    n_patches: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Internal unchecked batch view matching the field-major state contract."""

    batch = np.asarray(state, dtype=float)
    local = {
        name: batch[:, index * n_patches:(index + 1) * n_patches]
        for index, name in enumerate(LOCAL_FIELDS)
    }
    return local, batch[:, -2], batch[:, -1]
