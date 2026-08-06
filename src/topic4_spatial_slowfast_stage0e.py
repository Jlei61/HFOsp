"""Topology-native periodic-orbit audit for the frozen Stage-0C fast system.

Stage 0E adds no model term.  It replaces fixed-window rate classification with
an event-located Poincare map, fixed-point shooting, transverse finite-difference
Jacobians, and deterministic return-to-orbit perturbation tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from src.topic4_spatial_slowfast_stage0c import (
    E_CEILING_KHZ,
    FINITE_HIGH_MAX_KHZ,
    I_CEILING_KHZ,
    PoolParameters,
    S_MAX,
)
from src.topic4_spatial_slowfast_stage0c_transfer import (
    ExtendedSiegertTransfer,
    _rhs_and_moments,
    prepare_pool_parameters,
)


STATE_NAMES: tuple[str, ...] = (
    "rE",
    "rI",
    "sEE",
    "sEI",
    "sIE",
    "sII",
    "rE_fast",
    "mu_G",
    "S_G",
)
SECTION_INDEX = 8
SECTION_COORDINATES = np.asarray([0, 1, 2, 3, 4, 5, 6, 7], dtype=int)
LOCKED_MAX_REFRACTORY_OCCUPANCY = 0.05


@dataclass(frozen=True)
class SectionDefinition:
    """A fixed, directed coordinate Poincare section."""

    index: int = SECTION_INDEX
    level: float = 0.15
    direction: str = "upward"
    min_return_ms: float = 300.0
    max_return_ms: float = 1200.0

    def validate(self) -> "SectionDefinition":
        if self.index != SECTION_INDEX or self.direction != "upward":
            raise ValueError("Stage0E requires the locked upward S_G section")
        if not np.isclose(self.level, 0.15):
            raise ValueError("Stage0E section level drifted")
        if not 0.0 < self.min_return_ms < self.max_return_ms:
            raise ValueError("invalid return-time interval")
        return self


def _natural_state_bad(state: np.ndarray) -> np.ndarray:
    """Return one fail-closed flag per state; no coordinate is clipped."""

    state = np.asarray(state, dtype=float)
    one = state.ndim == 1
    batch = state[None, :] if one else state
    if batch.ndim != 2 or batch.shape[1] != 9:
        raise ValueError("state must have shape (9,) or (n,9)")
    bad = ~np.all(np.isfinite(batch), axis=1)
    bad |= np.any(batch < -1e-9, axis=1)
    bad |= (batch[:, 0] > E_CEILING_KHZ + 1e-9) | (batch[:, 1] > I_CEILING_KHZ + 1e-9)
    bad |= np.any(batch[:, [2, 4, 6]] > E_CEILING_KHZ + 1e-9, axis=1)
    bad |= np.any(batch[:, [3, 5]] > I_CEILING_KHZ + 1e-9, axis=1)
    bad |= (batch[:, 7] > 1.0 + 1e-9) | (batch[:, 8] > S_MAX + 1e-9)
    return bad[0] if one else bad


def scaled_inf_distance(left: np.ndarray, right: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Full-state infinity distance after a locked diagonal normalization."""

    left, right = np.broadcast_arrays(np.asarray(left, dtype=float), np.asarray(right, dtype=float))
    scales = np.asarray(scales, dtype=float)
    if left.shape[-1] != 9 or scales.shape != (9,) or np.any(scales <= 0.0):
        raise ValueError("scaled distance requires (...,9) states and nine positive scales")
    return np.max(np.abs(left - right) / scales, axis=-1)


def interpolate_upward_crossing(
    state_before: np.ndarray,
    state_after: np.ndarray,
    time_before_ms: float,
    dt_ms: float,
    section: SectionDefinition,
) -> tuple[float, np.ndarray]:
    """Linearly locate one bracketing upward crossing in time and full state."""

    section.validate()
    before = np.asarray(state_before, dtype=float)
    after = np.asarray(state_after, dtype=float)
    if before.shape != (9,) or after.shape != (9,):
        raise ValueError("crossing states must have shape (9,)")
    h0 = float(before[section.index] - section.level)
    h1 = float(after[section.index] - section.level)
    if not (h0 < 0.0 <= h1) or not np.isfinite(h0 + h1):
        raise ValueError("states do not bracket an upward crossing")
    fraction = -h0 / (h1 - h0)
    crossing = before + fraction * (after - before)
    crossing[section.index] = section.level
    return float(time_before_ms + fraction * dt_ms), crossing


def audit_crossing_state(
    crossing: np.ndarray,
    params: PoolParameters,
    transfer: ExtendedSiegertTransfer,
    section: SectionDefinition,
) -> dict[str, Any]:
    """Fail-closed audit of an interpolated event state and its full RHS."""

    state = np.asarray(crossing, dtype=float)
    if state.shape != (9,):
        raise ValueError("crossing state must have shape (9,)")
    prepared = prepare_pool_parameters([params.validate()])
    rhs, moments = _rhs_and_moments(
        state[None, :],
        prepared,
        transfer,
        mechanism="dynamic",
        clamp_s=None,
        subtractive_beta_mv=None,
    )
    mu_e, sigma_e, mu_i, sigma_i, _ = moments
    moment_values = np.asarray(
        [mu_e[0], sigma_e[0], mu_i[0], sigma_i[0]], dtype=float
    )
    finite = bool(
        np.all(np.isfinite(state))
        and np.all(np.isfinite(moment_values))
        and np.all(np.isfinite(rhs[0]))
    )
    support = bool(
        transfer.support_mask(mu_e, sigma_e)[0]
        and transfer.support_mask(mu_i, sigma_i)[0]
    )
    natural_bounds = bool(not _natural_state_bad(state))
    below_100hz = bool(state[0] < FINITE_HIGH_MAX_KHZ)
    on_section = bool(np.isclose(state[section.index], section.level, atol=1e-12))
    transversality = float(rhs[0, section.index]) if finite else np.nan
    upward = bool(np.isfinite(transversality) and transversality > 0.0)
    return {
        "clean": bool(finite and support and natural_bounds and below_100hz and on_section and upward),
        "finite_full_rhs": finite,
        "transfer_support": support,
        "natural_bounds": natural_bounds,
        "below_100hz": below_100hz,
        "on_section": on_section,
        "upward_transversality": upward,
        "transversality_per_ms": transversality if np.isfinite(transversality) else None,
        "moments": moment_values,
    }


def phase_resample(
    time_ms: np.ndarray,
    state: np.ndarray,
    start_ms: float,
    stop_ms: float,
    n_phase: int,
) -> np.ndarray:
    """Resample one complete full-state cycle on normalized phase."""

    time = np.asarray(time_ms, dtype=float)
    state = np.asarray(state, dtype=float)
    if time.ndim != 1 or state.shape != (time.size, 9) or n_phase < 16:
        raise ValueError("invalid phase-resampling input")
    if not (time[0] <= start_ms < stop_ms <= time[-1]):
        raise ValueError("cycle lies outside supplied trace")
    target = np.linspace(start_ms, stop_ms, int(n_phase), dtype=float)
    return np.column_stack([np.interp(target, time, state[:, index]) for index in range(9)])


def aligned_waveform_residual(left: np.ndarray, right: np.ndarray, scales: np.ndarray) -> float:
    """Phase-aligned full-state maximum residual."""

    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    scales = np.asarray(scales, dtype=float)
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 9:
        raise ValueError("waveforms must align as (phase,9)")
    return float(np.max(np.abs(left - right) / scales[None, :]))


def _empty_audit(n: int) -> dict[str, np.ndarray]:
    return {
        "finite": np.ones(n, dtype=bool),
        "support_violation_count": np.zeros(n, dtype=np.int64),
        "state_bound_violation_count": np.zeros(n, dtype=np.int64),
        "over_100hz_count": np.zeros(n, dtype=np.int64),
        "e_refractory_count": np.zeros(n, dtype=np.int64),
        "i_refractory_count": np.zeros(n, dtype=np.int64),
        "above_80hz_count": np.zeros(n, dtype=np.int64),
        "n_euler_states": np.zeros(n, dtype=np.int64),
        "moment_min": np.full((n, 4), np.inf, dtype=float),
        "moment_max": np.full((n, 4), -np.inf, dtype=float),
        "peak_rE_hz": np.full(n, -np.inf, dtype=float),
    }


def _audit_state(
    state: np.ndarray,
    moments: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    transfer: ExtendedSiegertTransfer,
    audit: dict[str, np.ndarray],
    active: np.ndarray,
) -> np.ndarray:
    """Update every-Euler audits and return fatal flags for active trajectories."""

    mu_e, sigma_e, mu_i, sigma_i, _ = moments
    moment_matrix = np.column_stack((mu_e, sigma_e, mu_i, sigma_i))
    support_bad = ~(transfer.support_mask(mu_e, sigma_e) & transfer.support_mask(mu_i, sigma_i))
    state_bad = _natural_state_bad(state)
    finite_bad = ~np.all(np.isfinite(state), axis=1) | ~np.all(np.isfinite(moment_matrix), axis=1)
    selected = np.flatnonzero(active)
    audit["n_euler_states"][selected] += 1
    audit["support_violation_count"][selected] += support_bad[selected]
    audit["state_bound_violation_count"][selected] += state_bad[selected]
    audit["over_100hz_count"][selected] += state[selected, 0] >= FINITE_HIGH_MAX_KHZ
    audit["e_refractory_count"][selected] += state[selected, 0] >= 0.95 * E_CEILING_KHZ
    audit["i_refractory_count"][selected] += state[selected, 1] >= 0.95 * I_CEILING_KHZ
    audit["above_80hz_count"][selected] += state[selected, 0] >= 0.080
    audit["peak_rE_hz"][selected] = np.fmax(audit["peak_rE_hz"][selected], 1000.0 * state[selected, 0])
    audit["moment_min"][selected] = np.fmin(audit["moment_min"][selected], moment_matrix[selected])
    audit["moment_max"][selected] = np.fmax(audit["moment_max"][selected], moment_matrix[selected])
    audit["finite"][selected] &= ~finite_bad[selected]
    return finite_bad | support_bad | state_bad


def _audit_clean(audit: Mapping[str, np.ndarray], index: int) -> bool:
    n_states = int(np.asarray(audit["n_euler_states"])[index])
    if n_states <= 0:
        return False
    e_occupancy = int(np.asarray(audit["e_refractory_count"])[index]) / n_states
    i_occupancy = int(np.asarray(audit["i_refractory_count"])[index]) / n_states
    return bool(
        np.asarray(audit["finite"])[index]
        and int(np.asarray(audit["support_violation_count"])[index]) == 0
        and int(np.asarray(audit["state_bound_violation_count"])[index]) == 0
        and int(np.asarray(audit["over_100hz_count"])[index]) == 0
        and e_occupancy <= LOCKED_MAX_REFRACTORY_OCCUPANCY
        and i_occupancy <= LOCKED_MAX_REFRACTORY_OCCUPANCY
    )


def audit_row(audit: Mapping[str, np.ndarray], index: int) -> dict[str, Any]:
    """Serialize one trajectory's physical/numerical audit."""

    n_states = int(np.asarray(audit["n_euler_states"])[index])
    above_80 = int(np.asarray(audit["above_80hz_count"])[index])
    return {
        "clean": _audit_clean(audit, index),
        "finite": bool(np.asarray(audit["finite"])[index]),
        "n_euler_states": n_states,
        "support_violation_count": int(np.asarray(audit["support_violation_count"])[index]),
        "state_bound_violation_count": int(np.asarray(audit["state_bound_violation_count"])[index]),
        "over_100hz_count": int(np.asarray(audit["over_100hz_count"])[index]),
        "e_refractory_count": int(np.asarray(audit["e_refractory_count"])[index]),
        "i_refractory_count": int(np.asarray(audit["i_refractory_count"])[index]),
        "e_refractory_occupancy": float(
            int(np.asarray(audit["e_refractory_count"])[index]) / n_states
        )
        if n_states
        else None,
        "i_refractory_occupancy": float(
            int(np.asarray(audit["i_refractory_count"])[index]) / n_states
        )
        if n_states
        else None,
        "above_80hz_count": above_80,
        "above_80hz_occupancy": float(above_80 / n_states) if n_states else None,
        "peak_rE_hz": float(np.asarray(audit["peak_rE_hz"])[index]),
        "moment_min": np.asarray(audit["moment_min"])[index].tolist(),
        "moment_max": np.asarray(audit["moment_max"])[index].tolist(),
    }


def integrate_full_trace(
    initial_state: np.ndarray,
    params: PoolParameters,
    transfer: ExtendedSiegertTransfer,
    *,
    dt_ms: float,
    duration_ms: float,
    section: SectionDefinition,
) -> dict[str, Any]:
    """Integrate one full-state scout, saving and auditing every Euler state."""

    section.validate()
    state = np.asarray(initial_state, dtype=float).copy()
    if state.shape != (9,) or bool(_natural_state_bad(state)):
        raise ValueError("invalid full-trace initial state")
    if dt_ms <= 0.0 or duration_ms <= section.max_return_ms:
        raise ValueError("invalid full-trace schedule")
    n_steps = int(round(duration_ms / dt_ms))
    if not np.isclose(n_steps * dt_ms, duration_ms):
        raise ValueError("duration must be an integer multiple of dt")
    time = np.arange(n_steps + 1, dtype=float) * dt_ms
    states = np.full((n_steps + 1, 9), np.nan, dtype=np.float64)
    moments_trace = np.full((n_steps + 1, 4), np.nan, dtype=np.float64)
    prepared = prepare_pool_parameters([params.validate()])
    audit = _empty_audit(1)
    crossing_times: list[float] = []
    crossing_states: list[np.ndarray] = []
    crossing_transversality: list[float] = []
    crossing_audits: list[dict[str, Any]] = []
    below_seen = bool(state[section.index] < section.level)
    fatal = False

    for step in range(n_steps + 1):
        states[step] = state
        rhs, moments = _rhs_and_moments(
            state[None, :],
            prepared,
            transfer,
            mechanism="dynamic",
            clamp_s=None,
            subtractive_beta_mv=None,
        )
        moments_trace[step] = np.asarray([value[0] for value in moments[:4]])
        fatal_flags = _audit_state(state[None, :], moments, transfer, audit, np.asarray([True]))
        if bool(fatal_flags[0]) or not np.all(np.isfinite(rhs[0])):
            fatal = True
            break
        if step == n_steps:
            break
        next_state = state + dt_ms * rhs[0]
        h0 = float(state[section.index] - section.level)
        h1 = float(next_state[section.index] - section.level)
        if h0 < 0.0:
            below_seen = True
        if below_seen and h0 < 0.0 <= h1:
            crossing_time, crossing = interpolate_upward_crossing(state, next_state, time[step], dt_ms, section)
            crossing_audit = audit_crossing_state(crossing, params, transfer, section)
            crossing_audits.append(crossing_audit)
            if not crossing_audit["clean"]:
                fatal = True
                break
            crossing_times.append(crossing_time)
            crossing_states.append(crossing)
            crossing_transversality.append(float(crossing_audit["transversality_per_ms"]))
            below_seen = False
        state = next_state

    used = step + 1
    crossing_times_array = np.asarray(crossing_times, dtype=float)
    periods = np.diff(crossing_times_array)
    return {
        "time_ms": time[:used],
        "state": states[:used],
        "moments": moments_trace[:used],
        "crossing_time_ms": crossing_times_array,
        "crossing_state": np.asarray(crossing_states, dtype=float).reshape((-1, 9)),
        "crossing_transversality_per_ms": np.asarray(crossing_transversality, dtype=float),
        "crossing_audit": crossing_audits,
        "period_ms": periods,
        "audit": audit_row(audit, 0),
        "fatal": fatal,
    }


def integrate_to_returns_batch(
    initial_states: np.ndarray,
    params: Sequence[PoolParameters],
    transfer: ExtendedSiegertTransfer,
    *,
    dt_ms: float,
    n_returns: int,
    section: SectionDefinition,
) -> dict[str, Any]:
    """Integrate a batch until each member has the requested upward returns.

    The first return may be a partial cycle for an off-section phase restart.
    All later inter-return intervals must satisfy the locked 300--1200 ms band.
    """

    section.validate()
    state = np.asarray(initial_states, dtype=float).copy()
    if state.ndim != 2 or state.shape[1] != 9 or state.shape[0] != len(params):
        raise ValueError("initial_states and params must align")
    if dt_ms <= 0.0 or n_returns < 1 or np.any(_natural_state_bad(state)):
        raise ValueError("invalid return-map input")
    n = state.shape[0]
    prepared = prepare_pool_parameters(params)
    audit = _empty_audit(n)
    return_state = np.full((n_returns, n, 9), np.nan, dtype=float)
    return_time = np.full((n_returns, n), np.nan, dtype=float)
    transversality = np.full((n_returns, n), np.nan, dtype=float)
    crossing_audit: list[list[dict[str, Any] | None]] = [
        [None for _ in range(n)] for _ in range(n_returns)
    ]
    return_count = np.zeros(n, dtype=int)
    below_seen = state[:, section.index] < section.level
    active = np.ones(n, dtype=bool)
    fatal = np.zeros(n, dtype=bool)
    elapsed_ms = np.zeros(n, dtype=float)
    # One partial initial cycle plus all requested full cycles.
    max_duration_ms = (n_returns + 1) * section.max_return_ms
    n_steps = int(np.ceil(max_duration_ms / dt_ms))

    for step in range(n_steps + 1):
        if not np.any(active):
            break
        rhs, moments = _rhs_and_moments(
            state,
            prepared,
            transfer,
            mechanism="dynamic",
            clamp_s=None,
            subtractive_beta_mv=None,
        )
        fatal_flags = _audit_state(state, moments, transfer, audit, active)
        fatal_flags |= ~np.all(np.isfinite(rhs), axis=1)
        newly_fatal = active & fatal_flags
        fatal |= newly_fatal
        active[newly_fatal] = False
        if not np.any(active):
            break
        next_state = state.copy()
        next_state[active] = state[active] + dt_ms * rhs[active]
        h0 = state[:, section.index] - section.level
        h1 = next_state[:, section.index] - section.level
        below_seen |= active & (h0 < 0.0)
        crossing_members = np.flatnonzero(active & below_seen & (h0 < 0.0) & (h1 >= 0.0))
        restarted = np.zeros(n, dtype=bool)
        for member in crossing_members:
            crossing_time, crossing = interpolate_upward_crossing(
                state[member], next_state[member], elapsed_ms[member], dt_ms, section
            )
            slot = int(return_count[member])
            if slot >= n_returns:
                continue
            crossing_check = audit_crossing_state(
                crossing, params[member], transfer, section
            )
            return_state[slot, member] = crossing
            return_time[slot, member] = crossing_time
            crossing_audit[slot][member] = crossing_check
            transversality[slot, member] = (
                float(crossing_check["transversality_per_ms"])
                if crossing_check["transversality_per_ms"] is not None
                else np.nan
            )
            if not crossing_check["clean"]:
                fatal[member] = True
                active[member] = False
                continue
            return_count[member] += 1
            below_seen[member] = False
            if return_count[member] >= n_returns:
                active[member] = False
            else:
                # Reapply exactly the same event-located numerical Poincare map
                # on every return; do not continue from the overshooting grid
                # state, which would define a different P^k object.
                next_state[member] = crossing
                elapsed_ms[member] = crossing_time
                restarted[member] = True
        advanced = active & ~restarted
        elapsed_ms[advanced] += dt_ms
        state = next_state

    valid = np.zeros(n, dtype=bool)
    for member in range(n):
        times = return_time[:, member]
        periods = np.diff(times)
        starts_on_section = bool(np.isclose(initial_states[member, section.index], section.level, atol=1e-12))
        first_ok = True
        if starts_on_section and np.isfinite(times[0]):
            first_ok = section.min_return_ms <= times[0] <= section.max_return_ms
        elif np.isfinite(times[0]):
            first_ok = 0.0 < times[0] <= section.max_return_ms
        periods_ok = bool(
            periods.size == 0
            or (
                np.all(np.isfinite(periods))
                and np.all(periods >= section.min_return_ms)
                and np.all(periods <= section.max_return_ms)
            )
        )
        valid[member] = bool(
            return_count[member] == n_returns
            and not fatal[member]
            and _audit_clean(audit, member)
            and first_ok
            and periods_ok
            and np.all(transversality[:, member] > 0.0)
        )
    return {
        "return_state": return_state,
        "return_time_ms": return_time,
        "transversality_per_ms": transversality,
        "crossing_audit": crossing_audit,
        "return_count": return_count,
        "valid": valid,
        "audit": audit,
    }


def scout_scales(trace: Mapping[str, Any], *, scale_floor: float, n_phase: int) -> tuple[np.ndarray, float]:
    """Freeze state scales from the last two complete scout cycles."""

    crossings = np.asarray(trace["crossing_time_ms"], dtype=float)
    if crossings.size < 3:
        raise ValueError("at least three crossings are required to define two cycles")
    time = np.asarray(trace["time_ms"], dtype=float)
    state = np.asarray(trace["state"], dtype=float)
    left = phase_resample(time, state, crossings[-3], crossings[-2], n_phase)
    right = phase_resample(time, state, crossings[-2], crossings[-1], n_phase)
    scales = np.maximum(np.ptp(np.vstack((left, right)), axis=0), float(scale_floor))
    residual = aligned_waveform_residual(left, right, scales)
    return scales, residual


def poincare_fixed_point_shooting(
    seed_state: np.ndarray,
    params: PoolParameters,
    transfer: ExtendedSiegertTransfer,
    *,
    dt_ms: float,
    section: SectionDefinition,
    scales: np.ndarray,
    max_iterations: int,
    residual_tolerance: float,
    minimum_iterations: int = 4,
) -> dict[str, Any]:
    """Fixed-point shooting by repeated application of the directed return map."""

    current = np.asarray(seed_state, dtype=float).copy()
    if current.shape != (9,):
        raise ValueError("shooting seed must have shape (9,)")
    current[section.index] = section.level
    residuals: list[float] = []
    periods: list[float] = []
    transversality: list[float] = []
    audits: list[dict[str, Any]] = []
    crossing_audits: list[dict[str, Any] | None] = []
    states = [current.copy()]
    valid = True
    for iteration in range(int(max_iterations)):
        mapped = integrate_to_returns_batch(
            current[None, :], [params], transfer, dt_ms=dt_ms, n_returns=1, section=section
        )
        audits.append(audit_row(mapped["audit"], 0))
        crossing_audits.append(mapped["crossing_audit"][0][0])
        if not bool(mapped["valid"][0]):
            valid = False
            break
        returned = mapped["return_state"][0, 0]
        residuals.append(float(scaled_inf_distance(returned, current, scales)))
        periods.append(float(mapped["return_time_ms"][0, 0]))
        transversality.append(float(mapped["transversality_per_ms"][0, 0]))
        current = returned
        states.append(current.copy())
        recent_monotone = bool(
            len(residuals) >= 3
            and np.all(np.diff(np.asarray(residuals[-3:], dtype=float)) <= 1e-14)
        )
        if (
            iteration + 1 >= minimum_iterations
            and residuals[-1] <= residual_tolerance
            and recent_monotone
        ):
            break
    residual_array = np.asarray(residuals, dtype=float)
    period_array = np.asarray(periods, dtype=float)
    monotone = bool(
        residual_array.size >= 3
        and np.all(np.diff(residual_array[-3:]) <= 1e-14)
    )
    converged = bool(
        valid
        and residual_array.size >= minimum_iterations
        and residual_array[-1] <= residual_tolerance
        and monotone
    )
    return {
        "converged": converged,
        "valid": valid,
        "fixed_state": current,
        "iterate_state": np.asarray(states, dtype=float),
        "residual": residual_array,
        "period_ms": period_array,
        "transversality_per_ms": np.asarray(transversality, dtype=float),
        "per_iteration_audit": audits,
        "per_iteration_crossing_audit": crossing_audits,
    }


def shooting_cycle_validation(
    fixed_state: np.ndarray,
    params: PoolParameters,
    transfer: ExtendedSiegertTransfer,
    *,
    dt_ms: float,
    section: SectionDefinition,
    scales: np.ndarray,
    n_phase: int,
) -> dict[str, Any]:
    """Independently integrate two cycles from a shooting fixed point."""

    duration = 3.0 * section.max_return_ms
    trace = integrate_full_trace(
        fixed_state, params, transfer, dt_ms=dt_ms, duration_ms=duration, section=section
    )
    crossings = np.asarray(trace["crossing_time_ms"], dtype=float)
    if trace["fatal"] or not trace["audit"]["clean"] or crossings.size < 2:
        return {"valid": False, "trace": trace}
    # Initial state lies on the section; the first two detected crossings close
    # the first and second complete return cycles.
    first = phase_resample(trace["time_ms"], trace["state"], 0.0, crossings[0], n_phase)
    second = phase_resample(trace["time_ms"], trace["state"], crossings[0], crossings[1], n_phase)
    closure = float(scaled_inf_distance(trace["crossing_state"][0], fixed_state, scales))
    second_closure = float(
        scaled_inf_distance(trace["crossing_state"][1], trace["crossing_state"][0], scales)
    )
    waveform_residual = aligned_waveform_residual(first, second, scales)
    periods = np.asarray([crossings[0], crossings[1] - crossings[0]], dtype=float)
    valid = bool(
        np.all(periods >= section.min_return_ms)
        and np.all(periods <= section.max_return_ms)
        and np.all(np.asarray(trace["crossing_transversality_per_ms"][:2]) > 0.0)
    )
    return {
        "valid": valid,
        "trace": trace,
        "period_ms": periods,
        "closure_residual": closure,
        "second_closure_residual": second_closure,
        "aligned_cycle_residual": waveform_residual,
        "waveform_first": first,
        "waveform_second": second,
    }


def finite_difference_poincare_jacobian(
    fixed_state: np.ndarray,
    params: PoolParameters,
    transfer: ExtendedSiegertTransfer,
    *,
    dt_ms: float,
    epsilon_relative: float,
    section: SectionDefinition,
    scales: np.ndarray,
) -> dict[str, Any]:
    """Central finite-difference transverse Poincare Jacobian.

    The returned 8x8 matrix is expressed in scale-normalized section
    coordinates.  Consequently its eigenvalues are invariant to the diagonal
    state-unit normalization and are the eight non-trivial Floquet multipliers.
    """

    fixed = np.asarray(fixed_state, dtype=float)
    scales = np.asarray(scales, dtype=float)
    if fixed.shape != (9,) or scales.shape != (9,) or epsilon_relative <= 0.0:
        raise ValueError("invalid finite-difference input")
    states: list[np.ndarray] = []
    for coordinate in SECTION_COORDINATES:
        for sign in (1.0, -1.0):
            changed = fixed.copy()
            changed[coordinate] += sign * epsilon_relative * scales[coordinate]
            changed[section.index] = section.level
            states.append(changed)
    initial = np.asarray(states, dtype=float)
    if np.any(_natural_state_bad(initial)):
        return {
            "valid": False,
            "reason": "finite_difference_state_outside_natural_bounds",
            "epsilon_relative": float(epsilon_relative),
        }
    batch = integrate_to_returns_batch(
        initial,
        [params] * initial.shape[0],
        transfer,
        dt_ms=dt_ms,
        n_returns=1,
        section=section,
    )
    if not np.all(batch["valid"]):
        return {
            "valid": False,
            "reason": "one_or_more_perturbed_returns_failed",
            "epsilon_relative": float(epsilon_relative),
            "valid_returns": int(np.sum(batch["valid"])),
            "n_returns": int(initial.shape[0]),
            "per_return_audit": [audit_row(batch["audit"], index) for index in range(initial.shape[0])],
            "per_return_crossing_audit": [
                batch["crossing_audit"][0][index] for index in range(initial.shape[0])
            ],
        }
    returned = np.asarray(batch["return_state"][0], dtype=float)
    jacobian = central_jacobian_from_returns(returned, scales, epsilon_relative)
    eigenvalues = np.linalg.eigvals(jacobian)
    return {
        "valid": bool(np.all(np.isfinite(jacobian)) and np.all(np.isfinite(eigenvalues))),
        "epsilon_relative": float(epsilon_relative),
        "jacobian": jacobian,
        "multipliers": eigenvalues,
        "spectral_radius": float(np.max(np.abs(eigenvalues))),
        "return_period_ms": np.asarray(batch["return_time_ms"][0], dtype=float),
        "minimum_transversality_per_ms": float(np.min(batch["transversality_per_ms"][0])),
        "per_return_audit": [audit_row(batch["audit"], index) for index in range(initial.shape[0])],
        "per_return_crossing_audit": [
            batch["crossing_audit"][0][index] for index in range(initial.shape[0])
        ],
    }


def central_jacobian_from_returns(
    returned_states: np.ndarray,
    scales: np.ndarray,
    epsilon_relative: float,
) -> np.ndarray:
    """Construct the normalized 8D central-difference Jacobian.

    This pure helper is deliberately separate from integration so that the
    column ordering and unit normalization can be regression tested exactly.
    Rows must alternate ``(+epsilon, -epsilon)`` for each locked section
    coordinate.
    """

    returned = np.asarray(returned_states, dtype=float)
    scales = np.asarray(scales, dtype=float)
    if returned.shape != (16, 9) or scales.shape != (9,) or np.any(scales <= 0.0):
        raise ValueError("central Jacobian requires 16 full states and nine positive scales")
    if not np.isfinite(epsilon_relative) or epsilon_relative <= 0.0:
        raise ValueError("epsilon_relative must be positive and finite")
    jacobian = np.empty((8, 8), dtype=float)
    transverse_scales = scales[SECTION_COORDINATES]
    for column in range(8):
        plus = returned[2 * column, SECTION_COORDINATES]
        minus = returned[2 * column + 1, SECTION_COORDINATES]
        jacobian[:, column] = (plus - minus) / transverse_scales / (2.0 * epsilon_relative)
    return jacobian


def jacobian_ladder_summary(rows: Sequence[Mapping[str, Any]], cfg: Mapping[str, float]) -> dict[str, Any]:
    """Apply locked epsilon-platform checks to three transverse Jacobians."""

    rows = list(rows)
    epsilon_order = np.asarray(
        [row.get("epsilon_relative", np.nan) for row in rows], dtype=float
    )
    if (
        len(rows) != 3
        or not np.array_equal(epsilon_order, np.asarray([1e-3, 3e-4, 1e-4]))
        or not all(bool(row.get("valid", False)) for row in rows)
    ):
        return {"pass": False, "reason": "invalid_epsilon_level"}
    matrices = [np.asarray(row["jacobian"], dtype=float) for row in rows]
    radii = np.asarray([float(row["spectral_radius"]) for row in rows], dtype=float)
    differences = []
    for index in range(2):
        left_norm = float(np.linalg.norm(matrices[index], ord="fro"))
        right_norm = float(np.linalg.norm(matrices[index + 1], ord="fro"))
        denominator = max(left_norm, right_norm, np.finfo(float).eps)
        differences.append(
            float(np.linalg.norm(matrices[index] - matrices[index + 1], ord="fro") / denominator)
        )
    rho_range = float(np.ptp(radii))
    platform = bool(
        rho_range <= float(cfg["epsilon_rho_range_max"])
        and max(differences) <= float(cfg["jacobian_relative_difference_max"])
        and differences[1]
        <= float(cfg["gradient_ratio_max"]) * differences[0]
        + float(cfg["gradient_additive_tolerance"])
    )
    return {
        "pass": platform,
        "spectral_radii": radii,
        "spectral_radius_range": rho_range,
        "jacobian_relative_differences": np.asarray(differences, dtype=float),
        "gradient_pass": bool(
            differences[1]
            <= float(cfg["gradient_ratio_max"]) * differences[0]
            + float(cfg["gradient_additive_tolerance"])
        ),
    }


def interpolate_cycle_phases(cycle: Mapping[str, Any], phases: Sequence[float]) -> np.ndarray:
    """Interpolate full states at locked fractions of the first shooting cycle."""

    trace = cycle["trace"]
    period = float(np.asarray(cycle["period_ms"])[0])
    phase_array = np.asarray(phases, dtype=float)
    if np.any(phase_array < 0.0) or np.any(phase_array >= 1.0):
        raise ValueError("restart phases must lie in [0,1)")
    target = phase_array * period
    time = np.asarray(trace["time_ms"], dtype=float)
    state = np.asarray(trace["state"], dtype=float)
    output = np.column_stack([np.interp(target, time, state[:, index]) for index in range(9)])
    output[0] = np.asarray(cycle["waveform_first"])[0]
    return output


def build_return_battery(
    phase_states: np.ndarray,
    *,
    phases: Sequence[float],
    perturbation_fraction: float,
    fast_directions: Sequence[Sequence[float]],
    pool_directions: Sequence[Sequence[float]],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Build four phase anchors plus fixed non-collinear fast/pool histories."""

    phase_states = np.asarray(phase_states, dtype=float)
    phases = tuple(float(value) for value in phases)
    fast = np.asarray(fast_directions, dtype=float)
    pool = np.asarray(pool_directions, dtype=float)
    if phase_states.shape != (4, 9) or phases != (0.0, 0.25, 0.5, 0.75):
        raise ValueError("return phase battery drifted")
    if not np.isclose(perturbation_fraction, 0.03) or fast.shape != (2, 6) or pool.shape != (2, 3):
        raise ValueError("return perturbation contract drifted")
    if np.linalg.matrix_rank(fast) != 2 or np.linalg.matrix_rank(pool) != 2:
        raise ValueError("return perturbations must be non-collinear within family")
    locked_fast = np.asarray([[1, -1, 1, -1, -1, 1], [1, 1, -1, -1, 1, -1]], dtype=float)
    locked_pool = np.asarray([[1, -1, 0], [-1, 0, 1]], dtype=float)
    if not np.array_equal(fast, locked_fast) or not np.array_equal(pool, locked_pool):
        raise ValueError("return perturbation directions drifted from locked contract")
    metadata: list[dict[str, Any]] = []
    states: list[np.ndarray] = []
    for phase_index, phase in enumerate(phases):
        anchor = phase_states[phase_index]
        variants: list[tuple[str, str, np.ndarray]] = [("phase_anchor", "anchor", anchor.copy())]
        for direction_index, direction in enumerate(fast):
            changed = anchor.copy()
            changed[:6] *= 1.0 + perturbation_fraction * direction
            variants.append((f"fast_direction_{direction_index + 1}", "fast", changed))
        for direction_index, direction in enumerate(pool):
            changed = anchor.copy()
            changed[6:9] *= 1.0 + perturbation_fraction * direction
            variants.append((f"pool_direction_{direction_index + 1}", "pool", changed))
        for label, family, state in variants:
            if bool(_natural_state_bad(state)):
                raise ValueError("return perturbation left natural state bounds")
            metadata.append(
                {
                    "phase_fraction": phase,
                    "phase_id": f"phase_{int(round(100 * phase)):03d}",
                    "history": label,
                    "family": family,
                    "perturbed": family != "anchor",
                }
            )
            states.append(state)
    return metadata, np.asarray(states, dtype=float)


def summarize_return_battery(
    metadata: Sequence[Mapping[str, Any]],
    batch: Mapping[str, Any],
    fixed_state: np.ndarray,
    scales: np.ndarray,
    cfg: Mapping[str, float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Measure cycle-wise return distances and apply family convergence gates."""

    return_states = np.asarray(batch["return_state"], dtype=float)
    if return_states.shape[0] != 8:
        raise ValueError("return battery must contain exactly eight Poincare returns")
    rows: list[dict[str, Any]] = []
    for index, meta in enumerate(metadata):
        distances = scaled_inf_distance(return_states[:, index], fixed_state, scales)
        finite_distances = bool(np.all(np.isfinite(distances)))
        if finite_distances and distances.size >= 2:
            floor = 1e-14
            log_slope = float(np.polyfit(np.arange(distances.size), np.log(np.maximum(distances, floor)), 1)[0])
            ratio = float(distances[-1] / max(distances[0], floor))
        else:
            log_slope = np.nan
            ratio = np.nan
        transversality_values = np.asarray(batch["transversality_per_ms"][:, index], dtype=float)
        minimum_transversality = (
            float(np.min(transversality_values))
            if np.all(np.isfinite(transversality_values))
            else None
        )
        rows.append(
            {
                **dict(meta),
                "valid": bool(np.asarray(batch["valid"])[index] and finite_distances),
                "return_distance": distances.tolist(),
                "first_distance": float(distances[0]) if finite_distances else None,
                "final_distance": float(distances[-1]) if finite_distances else None,
                "final_to_first_ratio": ratio if np.isfinite(ratio) else None,
                "log_distance_slope_per_return": log_slope if np.isfinite(log_slope) else None,
                "minimum_transversality_per_ms": minimum_transversality,
                "audit": audit_row(batch["audit"], index),
                "crossing_audit": [
                    batch["crossing_audit"][return_index][index]
                    for return_index in range(return_states.shape[0])
                ],
            }
        )
    anchors = [row for row in rows if row["family"] == "anchor"]
    anchor_pass = bool(
        len(anchors) == 4
        and all(row["valid"] for row in anchors)
        and max(float(row["final_distance"]) for row in anchors) <= float(cfg["anchor_final_distance_max"])
    )
    family_results: dict[str, Any] = {}
    for family in ("fast", "pool"):
        members = [row for row in rows if row["family"] == family]
        numeric_ratios = [
            float(row["final_to_first_ratio"])
            for row in members
            if row["final_to_first_ratio"] is not None
        ]
        numeric_finals = [
            float(row["final_distance"])
            for row in members
            if row["final_distance"] is not None
        ]
        ratios = np.asarray(numeric_ratios, dtype=float)
        family_pass = bool(
            len(members) == 8
            and all(row["valid"] for row in members)
            and ratios.size == 8
            and len(numeric_finals) == 8
            and all(float(row["log_distance_slope_per_return"]) < float(cfg["log_slope_max"]) for row in members)
            and all(float(row["final_distance"]) < float(row["first_distance"]) for row in members)
            and float(np.median(ratios)) <= float(cfg["family_median_ratio_max"])
            and max(numeric_finals) <= float(cfg["family_max_final_distance"])
        )
        family_results[family] = {
            "pass": family_pass,
            "n_histories": len(members),
            "median_final_to_first_ratio": float(np.median(ratios)) if ratios.size else None,
            "maximum_final_distance": max(numeric_finals) if len(numeric_finals) == 8 else None,
            "directions": sorted({str(row["history"]) for row in members}),
            "phases": sorted({str(row["phase_id"]) for row in members}),
        }
    summary = {
        "pass": bool(anchor_pass and family_results["fast"]["pass"] and family_results["pool"]["pass"]),
        "phase_restart_pass": anchor_pass,
        "families": family_results,
    }
    return rows, summary


def shooting_gate_summary(
    shooting: Mapping[str, Any],
    cycle: Mapping[str, Any] | None,
    cfg: Mapping[str, float],
) -> dict[str, Any]:
    """Apply the locked shooting, period-CV, and two-cycle closure gates."""

    residual = np.asarray(shooting.get("residual", []), dtype=float)
    periods = np.asarray(shooting.get("period_ms", []), dtype=float)
    period_tail = periods[-4:]
    period_cv = (
        float(np.std(period_tail, ddof=0) / np.mean(period_tail))
        if period_tail.size == 4 and np.mean(period_tail) > 0.0
        else np.inf
    )
    cycle_valid = bool(cycle is not None and cycle.get("valid", False))
    closure = float(cycle.get("closure_residual", np.inf)) if cycle is not None else np.inf
    second_closure = (
        float(cycle.get("second_closure_residual", np.inf)) if cycle is not None else np.inf
    )
    aligned = float(cycle.get("aligned_cycle_residual", np.inf)) if cycle is not None else np.inf
    residual_tolerance = float(cfg["residual_tolerance"])
    checks = {
        "shooting_converged": bool(shooting.get("converged", False)),
        "final_residual": bool(
            residual.size > 0 and np.isfinite(residual[-1]) and residual[-1] <= residual_tolerance
        ),
        "period_cv": bool(
            np.isfinite(period_cv) and period_cv <= float(cfg["period_cv_tolerance"])
        ),
        "cycle_valid": cycle_valid,
        "first_cycle_closure": bool(closure <= residual_tolerance),
        "second_cycle_closure": bool(second_closure <= residual_tolerance),
        "aligned_cycle_residual": bool(
            aligned <= float(cfg["aligned_cycle_residual_tolerance"])
        ),
    }
    return {
        "pass": bool(all(checks.values())),
        "checks": checks,
        "n_iterations": int(residual.size),
        "residual_series": residual,
        "period_series_ms": periods,
        "last_four_period_cv": period_cv if np.isfinite(period_cv) else None,
        "closure_residual": closure if np.isfinite(closure) else None,
        "second_closure_residual": second_closure if np.isfinite(second_closure) else None,
        "aligned_cycle_residual": aligned if np.isfinite(aligned) else None,
        "minimum_shooting_transversality_per_ms": float(
            np.min(np.asarray(shooting.get("transversality_per_ms", []), dtype=float))
        )
        if len(shooting.get("transversality_per_ms", []))
        else None,
    }


def dt_cycle_consistency_summary(
    base_cycle: Mapping[str, Any],
    half_cycle: Mapping[str, Any],
    scales: np.ndarray,
    cfg: Mapping[str, float],
) -> dict[str, Any]:
    """Compare independently shot base- and half-step cycles on normalized phase."""

    if not bool(base_cycle.get("valid", False)) or not bool(half_cycle.get("valid", False)):
        return {"pass": False, "reason": "invalid_cycle_at_one_or_both_time_steps"}
    base_period = float(np.mean(np.asarray(base_cycle["period_ms"], dtype=float)))
    half_period = float(np.mean(np.asarray(half_cycle["period_ms"], dtype=float)))
    period_difference = abs(base_period - half_period)
    period_tolerance = max(
        float(cfg["period_abs_ms"]), float(cfg["period_relative"]) * abs(base_period)
    )
    waveform_residual = aligned_waveform_residual(
        np.asarray(base_cycle["waveform_first"], dtype=float),
        np.asarray(half_cycle["waveform_first"], dtype=float),
        np.asarray(scales, dtype=float),
    )
    return {
        "pass": bool(
            period_difference <= period_tolerance
            and waveform_residual <= float(cfg["aligned_waveform_residual"])
        ),
        "base_period_ms": base_period,
        "half_period_ms": half_period,
        "period_difference_ms": period_difference,
        "period_tolerance_ms": period_tolerance,
        "aligned_waveform_residual": waveform_residual,
        "aligned_waveform_tolerance": float(cfg["aligned_waveform_residual"]),
    }


def floquet_stability_summary(
    base_rows: Sequence[Mapping[str, Any]],
    half_rows: Sequence[Mapping[str, Any]],
    base_ladder: Mapping[str, Any],
    half_ladder: Mapping[str, Any],
    cfg: Mapping[str, float],
) -> dict[str, Any]:
    """Resolve stability only outside the locked epsilon/dt uncertainty band."""

    base_rows = list(base_rows)
    half_rows = list(half_rows)
    locked_epsilons = np.asarray([1.0e-3, 3.0e-4, 1.0e-4], dtype=float)
    base_epsilons = np.asarray(
        [row.get("epsilon_relative", np.nan) for row in base_rows], dtype=float
    )
    half_epsilons = np.asarray(
        [row.get("epsilon_relative", np.nan) for row in half_rows], dtype=float
    )
    if (
        len(base_rows) != 3
        or len(half_rows) != 3
        or not np.array_equal(base_epsilons, locked_epsilons)
        or not np.array_equal(half_epsilons, locked_epsilons)
        or not bool(base_ladder.get("pass", False))
        or not bool(half_ladder.get("pass", False))
    ):
        return {
            "pass": False,
            "reason": "epsilon_ladder_or_order_unresolved",
            "base_ladder_pass": bool(base_ladder.get("pass", False)),
            "half_ladder_pass": bool(half_ladder.get("pass", False)),
        }
    base_rho = np.asarray([float(row["spectral_radius"]) for row in base_rows], dtype=float)
    half_rho = np.asarray([float(row["spectral_radius"]) for row in half_rows], dtype=float)
    if not np.all(np.isfinite(base_rho)) or not np.all(np.isfinite(half_rho)):
        return {"pass": False, "reason": "nonfinite_spectral_radius"}
    epsilon_spread = max(float(np.ptp(base_rho)), float(np.ptp(half_rho)))
    dt_differences = np.abs(base_rho - half_rho)
    dt_spread = float(np.max(dt_differences))
    rho_max = float(max(np.max(base_rho), np.max(half_rho)))
    margin = float(1.0 - rho_max)
    required_margin = float(
        max(
            float(cfg["minimum_unit_circle_margin"]),
            float(cfg["uncertainty_multiplier"]) * epsilon_spread,
            float(cfg["uncertainty_multiplier"]) * dt_spread,
        )
    )
    dt_pass = bool(dt_spread <= float(cfg["dt_rho_difference_max"]))
    all_inside = bool(rho_max < 1.0)
    return {
        "pass": bool(dt_pass and all_inside and margin >= required_margin),
        "base_spectral_radii": base_rho,
        "half_spectral_radii": half_rho,
        "same_epsilon_dt_differences": dt_differences,
        "epsilon_spectral_radius_spread": epsilon_spread,
        "dt_spectral_radius_spread": dt_spread,
        "dt_sensitivity_pass": dt_pass,
        "rho_max": rho_max,
        "all_nontrivial_multipliers_inside_unit_circle": all_inside,
        "unit_circle_margin": margin,
        "required_margin": required_margin,
        "robust_margin_pass": bool(margin >= required_margin),
    }


def orbit_physical_summary(cycle: Mapping[str, Any]) -> dict[str, Any]:
    """Report physical ranges on exactly the first complete shooting cycle."""

    if not bool(cycle.get("valid", False)):
        return {"valid": False}
    trace = cycle["trace"]
    period = float(np.asarray(cycle["period_ms"], dtype=float)[0])
    time = np.asarray(trace["time_ms"], dtype=float)
    state = np.asarray(trace["state"], dtype=float)
    moments = np.asarray(trace["moments"], dtype=float)
    mask = time <= period + 1e-12
    if np.sum(mask) < 2:
        return {"valid": False}
    rates_hz = 1000.0 * state[mask, 0]
    moment_names = ("muE_mv", "sigmaE_mv", "muI_mv", "sigmaI_mv")
    moment_ranges = {
        name: {
            "minimum": float(np.min(moments[mask, index])),
            "maximum": float(np.max(moments[mask, index])),
        }
        for index, name in enumerate(moment_names)
    }
    return {
        "valid": bool(np.all(np.isfinite(state[mask])) and np.all(np.isfinite(moments[mask]))),
        "n_euler_states": int(np.sum(mask)),
        "period_ms": period,
        "peak_rE_hz": float(np.max(rates_hz)),
        "above_80hz_occupancy": float(np.mean(rates_hz >= 80.0)),
        "above_100hz_occupancy": float(np.mean(rates_hz >= 100.0)),
        "state_minimum": np.min(state[mask], axis=0),
        "state_maximum": np.max(state[mask], axis=0),
        "moment_ranges": moment_ranges,
        "full_validation_trace_audit": dict(trace["audit"]),
    }


def floquet_row_report(row: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one raw Jacobian result into an explicit JSON-facing report."""

    report: dict[str, Any] = {
        key: value
        for key, value in row.items()
        if key not in {"jacobian", "multipliers", "per_return_audit"}
    }
    if "jacobian" in row:
        report["jacobian"] = np.asarray(row["jacobian"], dtype=float)
    if "multipliers" in row:
        multipliers = np.asarray(row["multipliers"], dtype=complex)
        report["multipliers"] = [
            {
                "index": int(index),
                "real": float(value.real),
                "imag": float(value.imag),
                "modulus": float(abs(value)),
            }
            for index, value in enumerate(multipliers)
        ]
    if "per_return_audit" in row:
        report["per_return_audit"] = list(row["per_return_audit"])
    return report


def _point_result_shell(params: PoolParameters) -> dict[str, Any]:
    return {
        "z": float(params.z),
        "alpha_G": float(params.alpha_g),
        "w_ee_mult": float(params.w_ee_mult),
        "ratio": float(params.ratio),
        "outcome": None,
        "stable_periodic_orbit": False,
        "failed_gates": [],
        "stage1_open": False,
        "space_open": False,
    }


def audit_parameter_point(
    seed_state: np.ndarray,
    params: PoolParameters,
    transfer: ExtendedSiegertTransfer,
    cfg: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the locked cheap-first Stage0E audit at one fixed parameter point."""

    params.validate()
    if not (
        np.isclose(params.z, 0.85)
        and params.alpha_g in (15.0, 16.0)
        and np.isclose(params.w_ee_mult, 1.1)
        and np.isclose(params.ratio, 1.0)
    ):
        raise ValueError("parameter point drifted from locked Stage0E contract")
    if tuple(float(value) for value in cfg["floquet"]["epsilon_relative"]) != (
        1.0e-3,
        3.0e-4,
        1.0e-4,
    ):
        raise ValueError("Stage0E epsilon order drifted")
    if int(cfg["return_battery"]["n_returns"]) != 8:
        raise ValueError("Stage0E return count drifted")
    result = _point_result_shell(params)
    artifacts: dict[str, Any] = {}
    section_cfg = cfg["section"]
    section = SectionDefinition(
        index=int(section_cfg["state_index"]),
        level=float(section_cfg["level"]),
        direction=str(section_cfg["direction"]),
        min_return_ms=float(section_cfg["min_return_ms"]),
        max_return_ms=float(section_cfg["max_return_ms"]),
    ).validate()
    scout_cfg = cfg["scout"]
    shooting_cfg = cfg["shooting"]
    base_dt = float(scout_cfg["dt_ms"])
    half_dt = float(cfg["dt_half"]["dt_ms"])

    scout = integrate_full_trace(
        seed_state,
        params,
        transfer,
        dt_ms=base_dt,
        duration_ms=float(scout_cfg["duration_ms"]),
        section=section,
    )
    artifacts["scout"] = scout
    crossing_count = int(np.asarray(scout["crossing_time_ms"]).size)
    periods = np.asarray(scout["period_ms"], dtype=float)
    scout_physical_clean = bool(not scout["fatal"] and scout["audit"]["clean"])
    period_band_pass = bool(
        periods.size > 0
        and np.all(periods >= section.min_return_ms)
        and np.all(periods <= section.max_return_ms)
    )
    transverse_pass = bool(
        crossing_count > 0
        and np.all(np.asarray(scout["crossing_transversality_per_ms"], dtype=float) > 0.0)
    )
    result["scout"] = {
        "pass": bool(
            scout_physical_clean
            and crossing_count >= int(scout_cfg["minimum_returns"])
            and period_band_pass
            and transverse_pass
        ),
        "crossing_count": crossing_count,
        "period_ms": periods,
        "period_mean_ms": float(np.mean(periods)) if periods.size else None,
        "period_cv": float(np.std(periods) / np.mean(periods))
        if periods.size and np.mean(periods) > 0.0
        else None,
        "minimum_transversality_per_ms": float(
            np.min(np.asarray(scout["crossing_transversality_per_ms"], dtype=float))
        )
        if crossing_count
        else None,
        "physical_audit": scout["audit"],
    }
    if not scout_physical_clean:
        result["outcome"] = "numerical_failure"
        result["failed_gates"] = ["scout_physical_or_numerical_audit"]
        return result, artifacts
    if not result["scout"]["pass"]:
        result["outcome"] = "periodic_orbit_numerically_unresolved"
        failed = []
        if crossing_count < int(scout_cfg["minimum_returns"]):
            failed.append("scout_minimum_returns")
        if not period_band_pass:
            failed.append("scout_period_band")
        if not transverse_pass:
            failed.append("scout_transversality")
        result["failed_gates"] = failed
        return result, artifacts

    scales, scout_aligned = scout_scales(
        scout,
        scale_floor=float(shooting_cfg["scale_floor"]),
        n_phase=int(scout_cfg["waveform_phase_bins"]),
    )
    result["state_scales"] = scales
    result["scout"]["last_two_cycle_aligned_residual"] = scout_aligned

    base_shooting = poincare_fixed_point_shooting(
        np.asarray(scout["crossing_state"])[-1],
        params,
        transfer,
        dt_ms=base_dt,
        section=section,
        scales=scales,
        max_iterations=int(shooting_cfg["max_iterations"]),
        residual_tolerance=float(shooting_cfg["residual_tolerance"]),
        minimum_iterations=max(4, int(shooting_cfg["final_monotone_count"]) + 1),
    )
    artifacts["base_shooting"] = base_shooting
    base_cycle = None
    if bool(base_shooting["converged"]):
        base_cycle = shooting_cycle_validation(
            base_shooting["fixed_state"],
            params,
            transfer,
            dt_ms=base_dt,
            section=section,
            scales=scales,
            n_phase=int(scout_cfg["waveform_phase_bins"]),
        )
        artifacts["base_cycle"] = base_cycle
    base_gate = shooting_gate_summary(base_shooting, base_cycle, shooting_cfg)
    result["base_dt_shooting"] = base_gate
    if not base_gate["pass"]:
        result["outcome"] = "periodic_orbit_numerically_unresolved"
        result["failed_gates"] = ["base_dt_shooting_or_cycle"]
        return result, artifacts

    half_shooting = poincare_fixed_point_shooting(
        np.asarray(base_shooting["fixed_state"], dtype=float),
        params,
        transfer,
        dt_ms=half_dt,
        section=section,
        scales=scales,
        max_iterations=int(shooting_cfg["max_iterations"]),
        residual_tolerance=float(shooting_cfg["residual_tolerance"]),
        minimum_iterations=max(4, int(shooting_cfg["final_monotone_count"]) + 1),
    )
    artifacts["half_shooting"] = half_shooting
    half_cycle = None
    if bool(half_shooting["converged"]):
        half_cycle = shooting_cycle_validation(
            half_shooting["fixed_state"],
            params,
            transfer,
            dt_ms=half_dt,
            section=section,
            scales=scales,
            n_phase=int(scout_cfg["waveform_phase_bins"]),
        )
        artifacts["half_cycle"] = half_cycle
    half_gate = shooting_gate_summary(half_shooting, half_cycle, shooting_cfg)
    result["half_dt_shooting"] = half_gate
    if not half_gate["pass"]:
        result["outcome"] = "periodic_orbit_numerically_unresolved"
        result["failed_gates"] = ["half_dt_shooting_or_cycle"]
        return result, artifacts

    dt_cycle = dt_cycle_consistency_summary(base_cycle, half_cycle, scales, cfg["dt_half"])
    result["dt_cycle_consistency"] = dt_cycle
    if not dt_cycle["pass"]:
        result["outcome"] = "periodic_orbit_numerically_unresolved"
        result["failed_gates"] = ["dt_cycle_consistency"]
        return result, artifacts

    # Physical reporting, Floquet, and return batteries are parallel child
    # audits of an accepted shooting cycle.  None may short-circuit another.
    result["base_orbit_physical"] = orbit_physical_summary(base_cycle)
    result["half_orbit_physical"] = orbit_physical_summary(half_cycle)
    physical_pass = bool(
        result["base_orbit_physical"].get("valid", False)
        and result["half_orbit_physical"].get("valid", False)
        and result["base_orbit_physical"]["above_100hz_occupancy"] == 0.0
        and result["half_orbit_physical"]["above_100hz_occupancy"] == 0.0
        and base_cycle["trace"]["audit"]["clean"]
        and half_cycle["trace"]["audit"]["clean"]
    )
    result["physical_acceptance_pass"] = physical_pass

    floquet_cfg = cfg["floquet"]
    epsilons = tuple(float(value) for value in floquet_cfg["epsilon_relative"])
    base_floquet = [
        finite_difference_poincare_jacobian(
            base_shooting["fixed_state"],
            params,
            transfer,
            dt_ms=base_dt,
            epsilon_relative=epsilon,
            section=section,
            scales=scales,
        )
        for epsilon in epsilons
    ]
    half_floquet = [
        finite_difference_poincare_jacobian(
            half_shooting["fixed_state"],
            params,
            transfer,
            dt_ms=half_dt,
            epsilon_relative=epsilon,
            section=section,
            scales=scales,
        )
        for epsilon in epsilons
    ]
    artifacts["base_floquet"] = base_floquet
    artifacts["half_floquet"] = half_floquet
    base_ladder = jacobian_ladder_summary(base_floquet, floquet_cfg)
    half_ladder = jacobian_ladder_summary(half_floquet, floquet_cfg)
    stability = floquet_stability_summary(
        base_floquet, half_floquet, base_ladder, half_ladder, floquet_cfg
    )
    result["floquet"] = {
        "base_epsilon_ladder": base_ladder,
        "half_epsilon_ladder": half_ladder,
        "stability": stability,
    }
    battery_cfg = cfg["return_battery"]
    phase_states = interpolate_cycle_phases(base_cycle, battery_cfg["phases"])
    metadata, initial_states = build_return_battery(
        phase_states,
        phases=battery_cfg["phases"],
        perturbation_fraction=float(battery_cfg["perturbation_fraction"]),
        fast_directions=battery_cfg["fast_directions"],
        pool_directions=battery_cfg["pool_directions"],
    )
    battery = integrate_to_returns_batch(
        initial_states,
        [params] * initial_states.shape[0],
        transfer,
        dt_ms=base_dt,
        n_returns=int(battery_cfg["n_returns"]),
        section=section,
    )
    battery_rows, battery_summary = summarize_return_battery(
        metadata, battery, base_shooting["fixed_state"], scales, battery_cfg
    )
    artifacts["return_battery"] = battery
    artifacts["return_battery_initial_state"] = initial_states
    artifacts["return_battery_rows"] = battery_rows
    result["return_battery"] = battery_summary
    failed = []
    if not stability["pass"]:
        failed.append("floquet_epsilon_dt_or_margin")
    if not battery_summary["pass"]:
        failed.append("phase_restart_or_perturbation_return")
    if not physical_pass:
        failed.append("final_orbit_physical_audit")
    if failed:
        result["outcome"] = "periodic_orbit_numerically_unresolved"
        result["failed_gates"] = failed
        return result, artifacts

    result["outcome"] = "stable_periodic_orbit"
    result["stable_periodic_orbit"] = True
    return result, artifacts
