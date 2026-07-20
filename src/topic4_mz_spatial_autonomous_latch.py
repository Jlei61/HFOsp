"""Minimal autonomous regional Z/M latch on the fixed P3 current scaffold.

The mechanism is deliberately orthogonal to the parallel recurrent-conductance
line.  E-to-E coupling is untouched.  Repeated focal events consume the
postsynaptic inhibitory resource through the already spatially mixed ``sEI``
field.  Additive recovery builds only when core *and* annulus are jointly
recruited, so ordinary focal interictal events do not pre-activate the exit
current.  The slow coordinates remain local in the packed P-patch state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from src.sef_hfo_lif import TREF_E, TREF_I
from src.topic4_mz_spatial_patch import (
    LOCAL_FIELDS,
    PreparedPatchRHS,
    patch_rhs_fast_and_moments,
    state_size,
)
from src.topic4_spatial_slowfast_stage0c import S_MAX


@dataclass(frozen=True)
class RegionalSlowParameters:
    """One slow arm; all times are ms and synaptic-rate thresholds are kHz."""

    z_rest: float = 0.90
    tau_z_recovery_ms: float = 20000.0
    tau_z_depletion_ms: float = 4000.0
    inhibitory_use_threshold_khz: float = 0.004
    inhibitory_use_width_khz: float = 0.004
    tau_p_ms: float = 750.0
    occupancy_threshold_khz: float = 0.020
    occupancy_width_khz: float = 0.010
    persistence_on: float = 0.115
    persistence_off: float = 0.03
    recruitment_on: float = 0.60
    low_reset_threshold_khz: float = 0.005
    z_safe: float = 0.885
    tau_m_up_ms: float = 3000.0
    tau_m_down_ms: float = 12000.0
    depletion_mask: tuple[float, float, float] = (1.0, 1.0, 0.0)
    pool_core_annulus_resource: bool = True
    pool_core_annulus_effector: bool = True
    enable_z: bool = True
    enable_m: bool = True

    def validate(self) -> "RegionalSlowParameters":
        values = (
            self.z_rest, self.tau_z_recovery_ms, self.tau_z_depletion_ms,
            self.inhibitory_use_threshold_khz, self.inhibitory_use_width_khz,
            self.tau_p_ms, self.occupancy_threshold_khz,
            self.occupancy_width_khz, self.persistence_on,
            self.persistence_off, self.recruitment_on,
            self.low_reset_threshold_khz, self.z_safe,
            self.tau_m_up_ms, self.tau_m_down_ms, *self.depletion_mask,
        )
        if not all(np.isfinite(values)):
            raise ValueError("regional slow parameters must be finite")
        if not 0.0 < self.z_rest <= 1.0:
            raise ValueError("z_rest must lie in (0,1]")
        if min(
            self.tau_z_recovery_ms, self.tau_z_depletion_ms,
            self.tau_p_ms, self.tau_m_up_ms, self.tau_m_down_ms,
            self.inhibitory_use_width_khz, self.occupancy_width_khz,
        ) <= 0.0:
            raise ValueError("slow times and sensor widths must be positive")
        if self.inhibitory_use_threshold_khz < 0.0 or self.occupancy_threshold_khz < 0.0:
            raise ValueError("sensor thresholds must be non-negative")
        if not 0.0 <= self.persistence_off < self.persistence_on <= 1.0:
            raise ValueError("persistence thresholds must satisfy 0<=off<on<=1")
        if not 0.0 < self.recruitment_on <= 1.0:
            raise ValueError("recruitment_on must lie in (0,1]")
        if not 0.0 < self.low_reset_threshold_khz < self.occupancy_threshold_khz:
            raise ValueError("low reset threshold must lie below occupancy threshold")
        if not 0.0 < self.z_safe <= self.z_rest:
            raise ValueError("z_safe must lie in (0,z_rest]")
        mask = np.asarray(self.depletion_mask, dtype=float)
        if mask.shape != (3,) or np.any((mask < 0.0) | (mask > 1.0)):
            raise ValueError("depletion_mask must be a three-patch [0,1] mask")
        return self


@dataclass(frozen=True)
class Pulse:
    """One finite external E-current pulse with a patch-aligned spatial profile."""

    onset_ms: float
    duration_ms: float
    amplitude_mv: float
    profile: tuple[float, float, float] = (1.0, 0.0, 0.0)

    def validate(self) -> "Pulse":
        profile = np.asarray(self.profile, dtype=float)
        if profile.shape != (3,):
            raise ValueError("pulse profile must contain exactly three patch weights")
        values = (self.onset_ms, self.duration_ms, self.amplitude_mv, *profile)
        if not all(np.isfinite(values)) or self.onset_ms < 0.0 or self.duration_ms <= 0.0:
            raise ValueError("pulse times and values must be finite with positive duration")
        if self.amplitude_mv < 0.0 or np.any(profile < 0.0):
            raise ValueError("pulse amplitude/profile must be non-negative")
        return self


def smooth_gate(value: np.ndarray, threshold: float, width: float) -> np.ndarray:
    """C1 compact gate: zero below threshold and one above threshold+width."""

    x = np.clip((np.asarray(value, dtype=float) - float(threshold)) / float(width), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def regional_slow_rhs(
    state: np.ndarray,
    arms: Sequence[RegionalSlowParameters],
    *,
    inhibitory_baseline_khz: Sequence[float],
    recruitment_kernel: np.ndarray,
    patch_weights: Sequence[float],
    latch_state: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return z/p/m derivatives and the registered spatial slow sensors."""

    batch = np.asarray(state, dtype=float)
    if batch.ndim != 2:
        raise ValueError("state must be a batch")
    n_forks = batch.shape[0]
    if len(arms) != n_forks:
        raise ValueError("one slow parameter arm is required per fork")
    p = (batch.shape[1] - 2) // len(LOCAL_FIELDS)
    if p != 3 or batch.shape[1] != state_size(p):
        raise ValueError("autonomous latch is registered for P=3")
    baseline = np.asarray(inhibitory_baseline_khz, dtype=float)
    if baseline.shape != (p,) or np.any(baseline < 0.0) or not np.all(np.isfinite(baseline)):
        raise ValueError("inhibitory baseline must be finite, non-negative, and patch aligned")
    kernel = np.asarray(recruitment_kernel, dtype=float)
    if kernel.shape != (p, p) or np.any(kernel < 0.0) or not np.all(np.isfinite(kernel)):
        raise ValueError("recruitment kernel must be a finite non-negative P-by-P matrix")
    if not np.allclose(kernel.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("recruitment kernel must preserve a constant field")
    weights = np.asarray(patch_weights, dtype=float)
    if (
        weights.shape != (p,) or np.any(weights <= 0.0)
        or not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=1.0e-12)
    ):
        raise ValueError("patch weights must be positive, aligned, and normalized")
    latch = np.asarray(latch_state, dtype=bool)
    if latch.shape != (n_forks, p) or np.any(latch[:, 2]):
        raise ValueError("latch state must be fork-by-patch with bath latch off")

    def local(name: str) -> np.ndarray:
        index = LOCAL_FIELDS.index(name)
        return batch[:, index * p:(index + 1) * p]

    s_ei = local("sEI")
    r_e_fast = local("rE_fast")
    z = local("z")
    persistence = local("p")
    m = local("m")
    output = np.zeros_like(batch)
    z_sensor = np.empty((n_forks, p), dtype=float)
    occupancy = np.empty((n_forks, p), dtype=float)
    recruitment = np.empty((n_forks, p), dtype=float)
    for fork, raw in enumerate(arms):
        arm = raw.validate()
        local_z_use = np.asarray(arm.depletion_mask, dtype=float) * smooth_gate(
            s_ei[fork] - baseline,
            arm.inhibitory_use_threshold_khz,
            arm.inhibitory_use_width_khz,
        )
        if arm.pool_core_annulus_resource:
            regional_weights = weights[:2] / np.sum(weights[:2])
            regional_use = float(regional_weights @ local_z_use[:2])
            local_z_use[:2] = regional_use
        z_sensor[fork] = local_z_use
        occupancy[fork] = smooth_gate(
            r_e_fast[fork], arm.occupancy_threshold_khz, arm.occupancy_width_khz
        )
        recruitment[fork] = kernel @ occupancy[fork]
        z_slice = slice(7 * p, 8 * p)
        p_slice = slice(8 * p, 9 * p)
        m_slice = slice(9 * p, 10 * p)
        output[fork, p_slice] = (occupancy[fork] - persistence[fork]) / arm.tau_p_ms
        if arm.enable_z:
            output[fork, z_slice] = (
                (arm.z_rest - z[fork]) / arm.tau_z_recovery_ms
                - z_sensor[fork] * z[fork] / arm.tau_z_depletion_ms
            )
        if arm.enable_m:
            regional_m = m[fork, :2]
            active = latch[fork, :2].astype(float)
            if arm.pool_core_annulus_effector:
                regional_weights = weights[:2] / np.sum(weights[:2])
                regional_m_value = float(regional_weights @ regional_m)
                joint_occupancy = float(np.prod(occupancy[fork, :2]))
                regional_active = float(np.any(active))
                dm_value = (
                    regional_active * joint_occupancy * (1.0 - regional_m_value)
                    / arm.tau_m_up_ms
                    - (1.0 - regional_active) * regional_m_value / arm.tau_m_down_ms
                )
                dm = np.asarray([dm_value, dm_value], dtype=float)
            else:
                dm = active * occupancy[fork, :2] * (
                    1.0 - regional_m
                ) / arm.tau_m_up_ms - (
                    1.0 - active
                ) * regional_m / arm.tau_m_down_ms
            output[fork, m_slice.start:m_slice.start + 2] = dm
            output[fork, m_slice.start + 2] = -m[fork, 2] / arm.tau_m_down_ms
    return output, {
        "z_use": z_sensor,
        "occupancy": occupancy,
        "neighborhood_recruitment": recruitment,
    }


def update_regional_latch(
    state: np.ndarray,
    arms: Sequence[RegionalSlowParameters],
    sensors: dict[str, np.ndarray],
    latch_state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply regional set/reset logic without a time-scripted seizure switch."""

    batch = np.asarray(state, dtype=float)
    latch = np.asarray(latch_state, dtype=bool).copy()
    p = (batch.shape[1] - 2) // len(LOCAL_FIELDS)
    if p != 3 or latch.shape != (batch.shape[0], p):
        raise ValueError("regional latch requires aligned P=3 state and bits")
    persistence = batch[:, 8 * p:9 * p]
    r_e_fast = batch[:, 6 * p:7 * p]
    z = batch[:, 7 * p:8 * p]
    recruitment = np.asarray(sensors["neighborhood_recruitment"], dtype=float)
    if recruitment.shape != (batch.shape[0], p):
        raise ValueError("neighborhood recruitment sensor is not aligned")
    set_now = np.zeros(batch.shape[0], dtype=bool)
    reset_now = np.zeros(batch.shape[0], dtype=bool)
    for fork, raw in enumerate(arms):
        arm = raw.validate()
        if not bool(np.any(latch[fork, :2])):
            set_now[fork] = bool(
                np.all(persistence[fork, :2] >= arm.persistence_on)
                and np.all(recruitment[fork, :2] >= arm.recruitment_on)
            )
            if set_now[fork]:
                latch[fork, :2] = True
        else:
            reset_now[fork] = bool(
                np.all(r_e_fast[fork, :2] <= arm.low_reset_threshold_khz)
                and np.all(z[fork, :2] >= arm.z_safe)
                and np.all(persistence[fork, :2] <= arm.persistence_off)
            )
            if reset_now[fork]:
                latch[fork, :2] = False
        latch[fork, 2] = False
    return latch, set_now, reset_now


def pulse_drive(
    time_ms: float,
    pulses: Sequence[Pulse],
    n_forks: int,
) -> np.ndarray:
    """Evaluate a common deterministic pulse schedule as batch-by-patch mV."""

    drive = np.zeros((int(n_forks), 3), dtype=float)
    for raw in pulses:
        pulse = raw.validate()
        if pulse.onset_ms <= time_ms < pulse.onset_ms + pulse.duration_ms:
            drive += pulse.amplitude_mv * np.asarray(pulse.profile, dtype=float)[None, :]
    return drive


def integrate_autonomous_latch_batch(
    initial_states: np.ndarray,
    prepared: PreparedPatchRHS,
    transfer: Any,
    arms: Sequence[RegionalSlowParameters],
    pulses: Sequence[Pulse],
    *,
    inhibitory_baseline_khz: Sequence[float],
    dt_ms: float,
    duration_ms: float,
    save_dt_ms: float,
    section_level_khz: float = 0.020,
    rearm_level_khz: float = 0.015,
    max_trace_bytes: int = 512 * 1024 * 1024,
) -> dict[str, Any]:
    """Vectorized Euler integration with autonomous regional Z/p/M and fixed pulses."""

    state = np.asarray(initial_states, dtype=float).copy()
    safe_reference = state.copy()
    p = prepared.n_patches
    if p != 3 or state.ndim != 2 or state.shape[1] != state_size(p):
        raise ValueError("initial states must be an aligned P=3 batch")
    if len(arms) != state.shape[0] or not np.all(np.isfinite(state)):
        raise ValueError("slow arms and finite initial states must align")
    if (
        dt_ms <= 0.0 or duration_ms <= dt_ms or save_dt_ms < dt_ms
        or section_level_khz <= rearm_level_khz or rearm_level_khz < 0.0
    ):
        raise ValueError("invalid integration contract")
    n_steps = int(round(duration_ms / dt_ms))
    save_stride = int(round(save_dt_ms / dt_ms))
    if not np.isclose(n_steps * dt_ms, duration_ms) or not np.isclose(save_stride * dt_ms, save_dt_ms):
        raise ValueError("duration/save interval must be integer multiples of dt")
    sample_steps = np.arange(0, n_steps + 1, save_stride, dtype=int)
    if sample_steps[-1] != n_steps:
        sample_steps = np.r_[sample_steps, n_steps]
    n_forks = state.shape[0]
    estimated_trace_bytes = (
        sample_steps.size
        * (
            n_forks * p * 9 * np.dtype(np.float32).itemsize
            + n_forks * p * np.dtype(np.uint8).itemsize
            + p * np.dtype(np.float32).itemsize
            + n_forks * 2 * np.dtype(np.float32).itemsize
        )
    )
    if estimated_trace_bytes > int(max_trace_bytes):
        raise MemoryError(
            f"requested traces need about {estimated_trace_bytes} bytes, above max_trace_bytes"
        )
    trace = {
        name: np.full((sample_steps.size, n_forks, p), np.nan, dtype=np.float32)
        for name in (
            "rE", "rI", "rE_fast", "z", "p", "m", "z_use",
            "occupancy", "neighborhood_recruitment",
        )
    }
    trace["latch"] = np.zeros((sample_steps.size, n_forks, p), dtype=np.uint8)
    trace["external_e_mv"] = np.full((sample_steps.size, p), np.nan, dtype=np.float32)
    trace["shared"] = np.full((sample_steps.size, n_forks, 2), np.nan, dtype=np.float32)
    support_violations = np.zeros((n_forks, p), dtype=np.int64)
    bound_violations = np.zeros((n_forks, p), dtype=np.int64)
    finite = np.ones(n_forks, dtype=bool)
    active = np.ones(n_forks, dtype=bool)
    first_support_failure_ms = np.full(n_forks, np.nan, dtype=float)
    first_bound_failure_ms = np.full(n_forks, np.nan, dtype=float)
    first_nonfinite_ms = np.full(n_forks, np.nan, dtype=float)
    return_times: list[list[list[float]]] = [[[] for _ in range(p)] for _ in range(n_forks)]
    return_states: list[list[list[np.ndarray]]] = [
        [[] for _ in range(p)] for _ in range(n_forks)
    ]
    latch_set_times: list[list[float]] = [[] for _ in range(n_forks)]
    latch_reset_times: list[list[float]] = [[] for _ in range(n_forks)]
    previous_fast = state[:, 6 * p:7 * p].copy()
    armed = previous_fast <= rearm_level_khz
    sample_index = 0
    latch = np.zeros((n_forks, p), dtype=bool)
    last_sensors = {
        "z_use": np.zeros((n_forks, p)),
        "occupancy": np.zeros((n_forks, p)),
        "neighborhood_recruitment": np.zeros((n_forks, p)),
    }
    external_schedule = np.zeros((n_steps + 1, p), dtype=np.float32)
    for raw in pulses:
        pulse = raw.validate()
        if not (
            np.isclose(pulse.onset_ms / dt_ms, round(pulse.onset_ms / dt_ms))
            and np.isclose(pulse.duration_ms / dt_ms, round(pulse.duration_ms / dt_ms))
        ):
            raise ValueError("pulse onset and duration must align with dt")
        start = int(round(pulse.onset_ms / dt_ms))
        stop = start + int(round(pulse.duration_ms / dt_ms))
        if start <= n_steps and stop > 0:
            left = max(0, start)
            right = min(n_steps + 1, stop)
            external_schedule[left:right] += (
                pulse.amplitude_mv * np.asarray(pulse.profile, dtype=np.float32)[None, :]
            )
    for step in range(n_steps + 1):
        time_ms = float(step) * float(dt_ms)
        external = external_schedule[step]
        evaluation_state = state
        if not np.all(active):
            evaluation_state = state.copy()
            evaluation_state[~active] = safe_reference[~active]
        fast_rhs, moments = patch_rhs_fast_and_moments(
            evaluation_state, prepared, transfer, external_e_mv=external
        )
        slow_rhs, last_sensors = regional_slow_rhs(
            evaluation_state, arms, inhibitory_baseline_khz=inhibitory_baseline_khz,
            recruitment_kernel=prepared.K_EE, patch_weights=prepared.patch_weights,
            latch_state=latch,
        )
        updated_latch, set_now, reset_now = update_regional_latch(
            evaluation_state, arms, last_sensors, latch
        )
        updated_latch[~active] = latch[~active]
        set_now[~active] = False
        reset_now[~active] = False
        if np.any(updated_latch != latch):
            latch = updated_latch
            slow_rhs, last_sensors = regional_slow_rhs(
                evaluation_state, arms, inhibitory_baseline_khz=inhibitory_baseline_khz,
                recruitment_kernel=prepared.K_EE, patch_weights=prepared.patch_weights,
                latch_state=latch,
            )
        for fork in np.flatnonzero(set_now):
            latch_set_times[int(fork)].append(time_ms)
        for fork in np.flatnonzero(reset_now):
            latch_reset_times[int(fork)].append(time_ms)
        rhs = fast_rhs + slow_rhs
        mu_e, sigma_e, mu_i, sigma_i, _ = moments
        supported = transfer.support_mask(mu_e, sigma_e) & transfer.support_mask(mu_i, sigma_i)
        active_before = active.copy()
        support_violations += (~supported) & active_before[:, None]
        r_e = state[:, 0:p]
        r_i = state[:, p:2 * p]
        z = state[:, 7 * p:8 * p]
        persistence = state[:, 8 * p:9 * p]
        m = state[:, 9 * p:10 * p]
        finite_now = np.all(np.isfinite(state), axis=1) & np.all(np.isfinite(rhs), axis=1)
        finite &= finite_now
        bad = (
            (r_e < -1.0e-9) | (r_e > 1.0 / TREF_E + 1.0e-9)
            | (r_i < -1.0e-9) | (r_i > 1.0 / TREF_I + 1.0e-9)
            | (state[:, 2 * p:3 * p] < -1.0e-9)
            | (state[:, 2 * p:3 * p] > 1.0 / TREF_E + 1.0e-9)
            | (state[:, 3 * p:4 * p] < -1.0e-9)
            | (state[:, 3 * p:4 * p] > 1.0 / TREF_I + 1.0e-9)
            | (state[:, 4 * p:5 * p] < -1.0e-9)
            | (state[:, 4 * p:5 * p] > 1.0 / TREF_E + 1.0e-9)
            | (state[:, 5 * p:6 * p] < -1.0e-9)
            | (state[:, 5 * p:6 * p] > 1.0 / TREF_I + 1.0e-9)
            | (state[:, 6 * p:7 * p] < -1.0e-9)
            | (state[:, 6 * p:7 * p] > 1.0 / TREF_E + 1.0e-9)
            | (z <= 0.0) | (z > 1.0 + 1.0e-9)
            | (persistence < -1.0e-9) | (persistence > 1.0 + 1.0e-9)
            | (m < -1.0e-9) | (m > 1.0 + 1.0e-9)
            | (state[:, -2, None] < -1.0e-9) | (state[:, -2, None] > 1.0 + 1.0e-9)
            | (state[:, -1, None] < -1.0e-9) | (state[:, -1, None] > S_MAX + 1.0e-9)
        )
        bound_violations += bad & active_before[:, None]
        support_failed = active_before & ~np.all(supported, axis=1)
        bound_failed = active_before & np.any(bad, axis=1)
        finite_failed = active_before & ~finite_now
        first_support_failure_ms[support_failed] = time_ms
        first_bound_failure_ms[bound_failed] = time_ms
        first_nonfinite_ms[finite_failed] = time_ms
        active[support_failed | bound_failed | finite_failed] = False
        rhs[~active] = 0.0
        if sample_index < sample_steps.size and step == sample_steps[sample_index]:
            trace["rE"][sample_index] = r_e
            trace["rI"][sample_index] = r_i
            trace["rE_fast"][sample_index] = state[:, 6 * p:7 * p]
            trace["z"][sample_index] = z
            trace["p"][sample_index] = persistence
            trace["m"][sample_index] = m
            trace["z_use"][sample_index] = last_sensors["z_use"]
            trace["occupancy"][sample_index] = last_sensors["occupancy"]
            trace["neighborhood_recruitment"][sample_index] = last_sensors[
                "neighborhood_recruitment"
            ]
            trace["latch"][sample_index] = latch.astype(np.uint8)
            trace["external_e_mv"][sample_index] = external
            trace["shared"][sample_index, :, 0] = state[:, -2]
            trace["shared"][sample_index, :, 1] = state[:, -1]
            sample_index += 1
        if step == n_steps:
            break
        next_state = state + dt_ms * rhs
        next_fast = next_state[:, 6 * p:7 * p]
        armed |= previous_fast <= rearm_level_khz
        crossed = (
            armed & active[:, None]
            & (previous_fast < section_level_khz) & (next_fast >= section_level_khz)
        )
        for fork, patch in np.argwhere(crossed):
            denominator = float(next_fast[fork, patch] - previous_fast[fork, patch])
            fraction = (
                float(section_level_khz - previous_fast[fork, patch]) / denominator
                if denominator > 0.0 else 0.0
            )
            return_times[int(fork)][int(patch)].append(time_ms + fraction * dt_ms)
            return_states[int(fork)][int(patch)].append(
                state[int(fork)].copy()
                + fraction * (next_state[int(fork)] - state[int(fork)])
            )
        armed[crossed] = False
        state = next_state
        previous_fast = next_fast.copy()
    return {
        "time_ms": sample_steps.astype(float) * dt_ms,
        **trace,
        "final_state": state,
        "finite": finite,
        "active_at_end": active,
        "support_violation_count": support_violations,
        "state_bound_violation_count": bound_violations,
        "first_support_failure_ms": first_support_failure_ms,
        "first_bound_failure_ms": first_bound_failure_ms,
        "first_nonfinite_ms": first_nonfinite_ms,
        "return_times_ms": return_times,
        "return_states": return_states,
        "latch_set_times_ms": latch_set_times,
        "latch_reset_times_ms": latch_reset_times,
    }
