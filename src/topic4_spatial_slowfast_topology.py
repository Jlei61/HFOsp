"""Stage 0A topology oracle for the Topic 4 slow--fast field route.

This module deliberately uses a canonical normal form, not the HFOsp model.  Its only
scientific role is to test the branch/state-fork analysis chain before that chain is
applied to a reduced E--I field.  The quintic subcritical-Hopf normal form is useful
because its topology is known analytically::

    dx/dt = (mu + beta*rho**2 - rho**4) * x - omega * y
    dy/dt = omega * x + (mu + beta*rho**2 - rho**4) * y

For ``beta > 0`` the origin is the low fixed point, the entry boundary is ``mu=0``,
and the stable/unstable cycles meet at ``mu=-beta**2/4``.  Thus a state-fork routine
must recover a bistable interval without confusing a ceiling or a long transient for
the finite stable cycle.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class NormalFormParameters:
    """Parameters of the canonical quintic Hopf normal form."""

    beta: float = 1.0
    omega_hz: float = 1.0

    def validate(self) -> "NormalFormParameters":
        if not np.isfinite(self.beta) or self.beta <= 0:
            raise ValueError("beta must be finite and positive")
        if not np.isfinite(self.omega_hz) or self.omega_hz <= 0:
            raise ValueError("omega_hz must be finite and positive")
        return self

    @property
    def entry_mu(self) -> float:
        return 0.0

    @property
    def exit_mu(self) -> float:
        return -(self.beta**2) / 4.0

    def cycle_radii(self, mu: float) -> dict[str, float | None]:
        """Return analytic unstable/stable cycle radii at frozen ``mu``."""

        disc = self.beta**2 + 4.0 * float(mu)
        if disc < 0:
            return {"unstable": None, "stable": None}
        root = float(np.sqrt(max(disc, 0.0)))
        s_inner = 0.5 * (self.beta - root)
        s_outer = 0.5 * (self.beta + root)
        unstable = float(np.sqrt(s_inner)) if s_inner > 0 else None
        stable = float(np.sqrt(s_outer)) if s_outer > 0 else None
        return {"unstable": unstable, "stable": stable}


@dataclass(frozen=True)
class OrbitClassifierThresholds:
    """Numerical contract for asymptotic-orbit classification."""

    tail_fraction: float = 0.40
    low_radius: float = 0.04
    min_cycle_radius: float = 0.12
    max_radial_cv: float = 0.06
    max_relative_drift: float = 0.06
    min_tail_cycles: float = 5.0
    min_spectral_power_ratio: float = 0.50
    ceiling_radius: float = 2.0
    ceiling_margin_fraction: float = 0.02
    max_ceiling_occupancy: float = 0.05

    def validate(self) -> "OrbitClassifierThresholds":
        if not 0.1 <= self.tail_fraction <= 0.9:
            raise ValueError("tail_fraction must be in [0.1, 0.9]")
        if not 0 < self.low_radius < self.min_cycle_radius < self.ceiling_radius:
            raise ValueError("radius thresholds must satisfy low < cycle < ceiling")
        if not 0 < self.max_radial_cv < 1:
            raise ValueError("max_radial_cv must be in (0, 1)")
        if not 0 < self.max_relative_drift < 1:
            raise ValueError("max_relative_drift must be in (0, 1)")
        if self.min_tail_cycles < 2:
            raise ValueError("min_tail_cycles must be >= 2")
        if not 0 < self.min_spectral_power_ratio <= 1:
            raise ValueError("min_spectral_power_ratio must be in (0, 1]")
        if not 0 <= self.ceiling_margin_fraction < 0.5:
            raise ValueError("ceiling_margin_fraction must be in [0, 0.5)")
        if not 0 <= self.max_ceiling_occupancy < 1:
            raise ValueError("max_ceiling_occupancy must be in [0, 1)")
        return self


@dataclass(frozen=True)
class SlowLoopParameters:
    """Toy, closed permissivity/recovery loop used only for analyzer sanity."""

    mu_base: float = -0.18
    permissivity_gain: float = 0.35
    recovery_gain: float = 0.55
    permissivity_target: float = 1.0
    tau_permissivity_s: float = 40.0
    recruited_permissivity_depletion_per_s: float = 0.05
    tau_recovery_s: float = 10.0
    recovery_gate_radius: float = 0.30
    recovery_gate_width: float = 0.03
    initial_radius: float = 0.02
    initial_permissivity: float = 0.0
    initial_recovery: float = 0.0
    recruited_radius: float = 0.30
    minimum_recruited_s: float = 5.0
    return_radius: float = 0.05
    minimum_return_s: float = 3.0
    early_retrigger_delay_s: float = 1.0
    late_retrigger_delay_s: float = 25.0
    retrigger_radius: float = 0.90

    def validate(self) -> "SlowLoopParameters":
        positive = {
            "permissivity_gain": self.permissivity_gain,
            "recovery_gain": self.recovery_gain,
            "permissivity_target": self.permissivity_target,
            "tau_permissivity_s": self.tau_permissivity_s,
            "recruited_permissivity_depletion_per_s": self.recruited_permissivity_depletion_per_s,
            "tau_recovery_s": self.tau_recovery_s,
            "recovery_gate_radius": self.recovery_gate_radius,
            "recovery_gate_width": self.recovery_gate_width,
            "initial_radius": self.initial_radius,
            "recruited_radius": self.recruited_radius,
            "minimum_recruited_s": self.minimum_recruited_s,
            "return_radius": self.return_radius,
            "minimum_return_s": self.minimum_return_s,
            "early_retrigger_delay_s": self.early_retrigger_delay_s,
            "late_retrigger_delay_s": self.late_retrigger_delay_s,
            "retrigger_radius": self.retrigger_radius,
        }
        bad = [name for name, value in positive.items() if not np.isfinite(value) or value <= 0]
        if bad:
            raise ValueError(f"slow-loop parameters must be finite and positive: {bad}")
        if self.late_retrigger_delay_s <= self.early_retrigger_delay_s:
            raise ValueError("late retrigger must follow early retrigger")
        if not 0 <= self.initial_permissivity <= 1 or not 0 <= self.initial_recovery <= 1:
            raise ValueError("initial slow states must be in [0, 1]")
        return self


def _normal_form_rhs(
    state: np.ndarray, mu: np.ndarray | float, params: NormalFormParameters
) -> np.ndarray:
    state = np.asarray(state, dtype=float)
    x = state[..., 0]
    y = state[..., 1]
    radius_sq = x * x + y * y
    growth = np.asarray(mu, dtype=float) + params.beta * radius_sq - radius_sq * radius_sq
    omega = 2.0 * np.pi * params.omega_hz
    return np.stack((growth * x - omega * y, omega * x + growth * y), axis=-1)


def _rk4_normal_form_step(
    state: np.ndarray, mu: np.ndarray | float, dt_s: float, params: NormalFormParameters
) -> np.ndarray:
    k1 = _normal_form_rhs(state, mu, params)
    k2 = _normal_form_rhs(state + 0.5 * dt_s * k1, mu, params)
    k3 = _normal_form_rhs(state + 0.5 * dt_s * k2, mu, params)
    k4 = _normal_form_rhs(state + dt_s * k3, mu, params)
    return state + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate_frozen(
    mu: float | Sequence[float] | np.ndarray,
    initial_states: Sequence[Sequence[float]] | np.ndarray,
    *,
    params: NormalFormParameters | None = None,
    dt_s: float = 0.01,
    duration_s: float = 200.0,
    save_stride: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate one or many frozen-control state forks with deterministic RK4."""

    params = (params or NormalFormParameters()).validate()
    if not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("dt_s must be finite and positive")
    if not np.isfinite(duration_s) or duration_s <= dt_s:
        raise ValueError("duration_s must exceed dt_s")
    if int(save_stride) != save_stride or save_stride < 1:
        raise ValueError("save_stride must be a positive integer")
    state = np.asarray(initial_states, dtype=float)
    if state.ndim == 1:
        state = state[None, :]
    if state.ndim != 2 or state.shape[1] != 2 or not np.all(np.isfinite(state)):
        raise ValueError("initial_states must be finite with shape (n_state, 2)")
    controls = np.asarray(mu, dtype=float)
    if controls.ndim == 0:
        controls = np.full(state.shape[0], float(controls))
    if controls.shape != (state.shape[0],) or not np.all(np.isfinite(controls)):
        raise ValueError("mu must be scalar or have one finite value per state")

    n_steps = int(np.ceil(duration_s / dt_s))
    sample_steps = np.arange(0, n_steps + 1, int(save_stride), dtype=int)
    if sample_steps[-1] != n_steps:
        sample_steps = np.r_[sample_steps, n_steps]
    states = np.empty((sample_steps.size, state.shape[0], 2), dtype=float)
    states[0] = state
    sample_index = 1
    for step in range(1, n_steps + 1):
        step_dt = min(dt_s, duration_s - (step - 1) * dt_s)
        state = _rk4_normal_form_step(state, controls, step_dt, params)
        if not np.all(np.isfinite(state)):
            raise FloatingPointError("normal-form integration produced a non-finite state")
        if sample_index < sample_steps.size and step == sample_steps[sample_index]:
            states[sample_index] = state
            sample_index += 1
    time_s = np.minimum(sample_steps.astype(float) * dt_s, duration_s)
    return time_s, states


def classify_orbit(
    time_s: Sequence[float] | np.ndarray,
    xy: Sequence[Sequence[float]] | np.ndarray,
    thresholds: OrbitClassifierThresholds | None = None,
) -> dict[str, Any]:
    """Classify a frozen trajectory without calling ceilings/transients cycles."""

    thresholds = (thresholds or OrbitClassifierThresholds()).validate()
    time_s = np.asarray(time_s, dtype=float)
    xy = np.asarray(xy, dtype=float)
    if time_s.ndim != 1 or xy.shape != (time_s.size, 2) or time_s.size < 20:
        raise ValueError("time_s and xy must have shapes (time,) and (time, 2), with >=20 samples")
    if not np.all(np.isfinite(time_s)) or np.any(np.diff(time_s) <= 0):
        raise ValueError("time_s must be finite and strictly increasing")
    if not np.all(np.isfinite(xy)):
        return {"classification": "numerical_divergence", "finite": False}

    tail_start = max(1, int(np.floor((1.0 - thresholds.tail_fraction) * time_s.size)))
    tail_xy = xy[tail_start:]
    tail_time = time_s[tail_start:]
    radius = np.linalg.norm(tail_xy, axis=1)
    radius_mean = float(np.mean(radius))
    radius_sd = float(np.std(radius))
    radial_cv = float(radius_sd / max(radius_mean, np.finfo(float).eps))
    split = max(2, radius.size // 2)
    first_mean = float(np.mean(radius[:split]))
    second_mean = float(np.mean(radius[split:]))
    relative_drift = float(abs(second_mean - first_mean) / max(radius_mean, thresholds.low_radius))
    ceiling_cut = thresholds.ceiling_radius * (1.0 - thresholds.ceiling_margin_fraction)
    ceiling_occupancy = float(np.mean(radius >= ceiling_cut))

    phase = np.unwrap(np.arctan2(tail_xy[:, 1], tail_xy[:, 0]))
    tail_cycles = float(abs(phase[-1] - phase[0]) / (2.0 * np.pi))
    dt_s = float(np.median(np.diff(tail_time)))
    signal = tail_xy[:, 0] - float(np.mean(tail_xy[:, 0]))
    power = np.abs(np.fft.rfft(signal)) ** 2
    freqs = np.fft.rfftfreq(signal.size, d=dt_s)
    if power.size > 1 and float(np.sum(power[1:])) > 0:
        peak_index = int(np.argmax(power[1:]) + 1)
        dominant_frequency_hz = float(freqs[peak_index])
        spectral_power_ratio = float(power[peak_index] / np.sum(power[1:]))
    else:
        dominant_frequency_hz = 0.0
        spectral_power_ratio = 0.0

    if ceiling_occupancy > thresholds.max_ceiling_occupancy or float(np.max(radius)) > (
        thresholds.ceiling_radius * 1.25
    ):
        label = "saturation_or_ceiling"
    elif radius_mean <= thresholds.low_radius and float(np.max(radius[-split:])) <= (
        2.0 * thresholds.low_radius
    ):
        label = "low_fixed_point"
    elif (
        radius_mean >= thresholds.min_cycle_radius
        and radial_cv <= thresholds.max_radial_cv
        and relative_drift <= thresholds.max_relative_drift
        and tail_cycles >= thresholds.min_tail_cycles
        and spectral_power_ratio >= thresholds.min_spectral_power_ratio
    ):
        label = "finite_limit_cycle"
    elif relative_drift > thresholds.max_relative_drift or radial_cv > thresholds.max_radial_cv:
        label = "indeterminate_long_transient"
    else:
        label = "bounded_nonoscillatory"

    return {
        "classification": label,
        "finite": True,
        "tail_radius_mean": radius_mean,
        "tail_radius_sd": radius_sd,
        "tail_radial_cv": radial_cv,
        "tail_relative_drift": relative_drift,
        "tail_cycles": tail_cycles,
        "dominant_frequency_hz": dominant_frequency_hz,
        "spectral_power_ratio": spectral_power_ratio,
        "ceiling_occupancy": ceiling_occupancy,
        "tail_start_s": float(tail_time[0]),
        "tail_duration_s": float(tail_time[-1] - tail_time[0]),
    }


def run_state_fork_map(
    mu_values: Iterable[float],
    initial_radii: Iterable[float],
    *,
    params: NormalFormParameters | None = None,
    thresholds: OrbitClassifierThresholds | None = None,
    dt_s: float = 0.01,
    duration_s: float = 200.0,
    save_stride: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Run a frozen-control initial-condition map in one vectorized integration."""

    params = (params or NormalFormParameters()).validate()
    thresholds = (thresholds or OrbitClassifierThresholds()).validate()
    mus = np.asarray(list(mu_values), dtype=float)
    radii = np.asarray(list(initial_radii), dtype=float)
    if mus.ndim != 1 or not mus.size or not np.all(np.isfinite(mus)):
        raise ValueError("mu_values must be a non-empty finite 1D sequence")
    if radii.ndim != 1 or not radii.size or np.any(radii <= 0) or not np.all(np.isfinite(radii)):
        raise ValueError("initial_radii must be a non-empty positive finite 1D sequence")
    grid_mu, grid_radius = np.meshgrid(mus, radii, indexing="ij")
    controls = grid_mu.ravel()
    start_radii = grid_radius.ravel()
    initial = np.column_stack((start_radii, np.zeros_like(start_radii)))
    time_s, states = simulate_frozen(
        controls,
        initial,
        params=params,
        dt_s=dt_s,
        duration_s=duration_s,
        save_stride=save_stride,
    )
    rows: list[dict[str, Any]] = []
    labels: list[str] = []
    final_radius: list[float] = []
    for index, (mu, radius) in enumerate(zip(controls, start_radii)):
        metrics = classify_orbit(time_s, states[:, index, :], thresholds)
        rows.append(
            {
                "mu": float(mu),
                "initial_radius": float(radius),
                **metrics,
            }
        )
        labels.append(str(metrics["classification"]))
        final_radius.append(float(np.linalg.norm(states[-1, index])))
    arrays = {
        "mu_values": mus,
        "initial_radii": radii,
        "classification": np.asarray(labels, dtype="U40").reshape(mus.size, radii.size),
        "final_radius": np.asarray(final_radius, dtype=float).reshape(mus.size, radii.size),
        "time_s": time_s,
        "states": states,
    }
    return rows, arrays


def _transition_bracket(
    rows: Sequence[Mapping[str, Any]], initial_radius: float
) -> list[float] | None:
    selected = sorted(
        (
            row
            for row in rows
            if np.isclose(float(row["initial_radius"]), float(initial_radius), rtol=0, atol=1e-12)
        ),
        key=lambda row: float(row["mu"]),
    )
    labels = [row["classification"] == "finite_limit_cycle" for row in selected]
    first_cycle = next((index for index, value in enumerate(labels) if value), None)
    if first_cycle is None:
        return None
    if first_cycle == 0:
        value = float(selected[0]["mu"])
        return [value, value]
    return [float(selected[first_cycle - 1]["mu"]), float(selected[first_cycle]["mu"])]


def detect_entry_exit_boundaries(
    rows: Sequence[Mapping[str, Any]], *, low_initial_radius: float, high_initial_radius: float
) -> dict[str, list[float] | None]:
    """Detect entry/exit brackets from low- and high-initial-condition forks."""

    return {
        "entry_bracket_mu": _transition_bracket(rows, low_initial_radius),
        "exit_bracket_mu": _transition_bracket(rows, high_initial_radius),
    }


def bracket_contains(value: float, bracket: Sequence[float] | None, tolerance: float = 0.0) -> bool:
    if bracket is None or len(bracket) != 2:
        return False
    lo, hi = sorted(float(item) for item in bracket)
    return bool((lo - tolerance) <= float(value) <= (hi + tolerance))


def _slow_loop_rhs(
    state: np.ndarray, normal: NormalFormParameters, slow: SlowLoopParameters
) -> tuple[np.ndarray, dict[str, float]]:
    x, y, permissivity, recovery = (float(value) for value in state)
    radius = float(np.hypot(x, y))
    gate_arg = np.clip(
        (radius - slow.recovery_gate_radius) / slow.recovery_gate_width, -60.0, 60.0
    )
    gate = float(1.0 / (1.0 + np.exp(-gate_arg)))
    mu = float(
        slow.mu_base
        + slow.permissivity_gain * permissivity
        - slow.recovery_gain * recovery
    )
    fast = _normal_form_rhs(np.asarray([x, y]), mu, normal)
    dpermissivity = (
        (slow.permissivity_target - permissivity) / slow.tau_permissivity_s
        - slow.recruited_permissivity_depletion_per_s * gate * permissivity
    )
    drecovery = (gate - recovery) / slow.tau_recovery_s
    return np.asarray([fast[0], fast[1], dpermissivity, drecovery]), {
        "radius": radius,
        "mu": mu,
        "recovery_gate": gate,
    }


def simulate_closed_slow_loop(
    *,
    normal: NormalFormParameters | None = None,
    slow: SlowLoopParameters | None = None,
    dt_s: float = 0.02,
    duration_s: float = 135.0,
    save_stride: int = 5,
) -> dict[str, np.ndarray]:
    """Integrate the toy closed loop with no reset and no external state switching."""

    normal = (normal or NormalFormParameters()).validate()
    slow = (slow or SlowLoopParameters()).validate()
    if dt_s <= 0 or duration_s <= dt_s or save_stride < 1:
        raise ValueError("invalid slow-loop integration settings")
    state = np.asarray(
        [
            slow.initial_radius,
            0.0,
            slow.initial_permissivity,
            slow.initial_recovery,
        ],
        dtype=float,
    )
    n_steps = int(np.ceil(duration_s / dt_s))
    sample_steps = np.arange(0, n_steps + 1, int(save_stride), dtype=int)
    if sample_steps[-1] != n_steps:
        sample_steps = np.r_[sample_steps, n_steps]
    saved = np.empty((sample_steps.size, 4), dtype=float)
    radius = np.empty(sample_steps.size, dtype=float)
    mu = np.empty(sample_steps.size, dtype=float)
    gate = np.empty(sample_steps.size, dtype=float)
    saved[0] = state
    _, obs = _slow_loop_rhs(state, normal, slow)
    radius[0], mu[0], gate[0] = obs["radius"], obs["mu"], obs["recovery_gate"]
    sample_index = 1
    for step in range(1, n_steps + 1):
        step_dt = min(dt_s, duration_s - (step - 1) * dt_s)
        k1, _ = _slow_loop_rhs(state, normal, slow)
        k2, _ = _slow_loop_rhs(state + 0.5 * step_dt * k1, normal, slow)
        k3, _ = _slow_loop_rhs(state + 0.5 * step_dt * k2, normal, slow)
        k4, _ = _slow_loop_rhs(state + step_dt * k3, normal, slow)
        state = state + (step_dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        state[2:] = np.clip(state[2:], 0.0, 1.0)
        if not np.all(np.isfinite(state)):
            raise FloatingPointError("slow-loop integration produced a non-finite state")
        if sample_index < sample_steps.size and step == sample_steps[sample_index]:
            saved[sample_index] = state
            _, obs = _slow_loop_rhs(state, normal, slow)
            radius[sample_index] = obs["radius"]
            mu[sample_index] = obs["mu"]
            gate[sample_index] = obs["recovery_gate"]
            sample_index += 1
    return {
        "time_s": np.minimum(sample_steps.astype(float) * dt_s, duration_s),
        "x": saved[:, 0],
        "y": saved[:, 1],
        "permissivity": saved[:, 2],
        "recovery": saved[:, 3],
        "radius": radius,
        "mu": mu,
        "recovery_gate": gate,
    }


def _true_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.r_[False, mask.astype(bool), False].astype(np.int8)))
    return [(int(start), int(stop)) for start, stop in zip(edges[::2], edges[1::2])]


def analyze_closed_slow_loop(
    traces: Mapping[str, np.ndarray],
    *,
    normal: NormalFormParameters | None = None,
    slow: SlowLoopParameters | None = None,
    classifier: OrbitClassifierThresholds | None = None,
    retrigger_duration_s: float = 120.0,
    retrigger_dt_s: float = 0.01,
) -> dict[str, Any]:
    """Check entry/exit order, basin return, and early/late frozen retrigger forks."""

    normal = (normal or NormalFormParameters()).validate()
    slow = (slow or SlowLoopParameters()).validate()
    classifier = (classifier or OrbitClassifierThresholds()).validate()
    time_s = np.asarray(traces["time_s"], dtype=float)
    radius = np.asarray(traces["radius"], dtype=float)
    mu = np.asarray(traces["mu"], dtype=float)
    x = np.asarray(traces["x"], dtype=float)
    y = np.asarray(traces["y"], dtype=float)
    if not (time_s.shape == radius.shape == mu.shape == x.shape == y.shape):
        raise ValueError("slow-loop trace arrays must have identical shape")

    dt_save = float(np.median(np.diff(time_s)))
    min_episode_samples = int(np.ceil(slow.minimum_recruited_s / dt_save))
    episodes = [
        (start, stop)
        for start, stop in _true_segments(radius >= slow.recruited_radius)
        if (stop - start) >= min_episode_samples
    ]
    if not episodes:
        return {
            "pass": False,
            "reason": "no_established_recruited_episode",
            "episodes": [],
        }
    onset, offset = episodes[0]
    onset_time = float(time_s[onset])
    offset_time = float(time_s[min(offset, time_s.size - 1)])
    phase = np.unwrap(np.arctan2(y[onset:offset], x[onset:offset]))
    episode_cycles = float(abs(phase[-1] - phase[0]) / (2.0 * np.pi)) if phase.size > 1 else 0.0

    entry_crossings = np.flatnonzero((mu[:-1] < normal.entry_mu) & (mu[1:] >= normal.entry_mu))
    entry_before_onset = entry_crossings[entry_crossings < onset]
    entry_index = int(entry_before_onset[-1]) if entry_before_onset.size else None
    exit_crossings = np.flatnonzero((mu[:-1] > normal.exit_mu) & (mu[1:] <= normal.exit_mu))
    exit_during = exit_crossings[(exit_crossings >= onset) & (exit_crossings <= offset)]
    exit_index = int(exit_during[0]) if exit_during.size else None

    return_samples = int(np.ceil(slow.minimum_return_s / dt_save))
    post_mask = radius[offset:] <= slow.return_radius
    return_segments = _true_segments(post_mask)
    accepted_returns = [segment for segment in return_segments if segment[1] - segment[0] >= return_samples]
    return_index = (offset + accepted_returns[0][0]) if accepted_returns else None

    early_time = offset_time + slow.early_retrigger_delay_s
    late_time = offset_time + slow.late_retrigger_delay_s
    early_index = int(np.searchsorted(time_s, early_time, side="left"))
    late_index = int(np.searchsorted(time_s, late_time, side="left"))
    retrigger_available = early_index < time_s.size and late_index < time_s.size
    retrigger: dict[str, Any] = {"available": retrigger_available}
    if retrigger_available:
        fork_time, fork_states = simulate_frozen(
            [float(mu[early_index]), float(mu[late_index])],
            [[slow.retrigger_radius, 0.0], [slow.retrigger_radius, 0.0]],
            params=normal,
            dt_s=retrigger_dt_s,
            duration_s=retrigger_duration_s,
            save_stride=max(1, int(round(0.05 / retrigger_dt_s))),
        )
        early_metrics = classify_orbit(fork_time, fork_states[:, 0], classifier)
        late_metrics = classify_orbit(fork_time, fork_states[:, 1], classifier)
        retrigger.update(
            {
                "early_time_s": float(time_s[early_index]),
                "early_mu": float(mu[early_index]),
                "early": early_metrics,
                "late_time_s": float(time_s[late_index]),
                "late_mu": float(mu[late_index]),
                "late": late_metrics,
                "early_suppressed": early_metrics["classification"] != "finite_limit_cycle",
                "late_recovered": late_metrics["classification"] == "finite_limit_cycle",
            }
        )

    gates = {
        "entry_crossed_before_recruitment": entry_index is not None,
        "finite_recruited_episode": (offset_time - onset_time) >= slow.minimum_recruited_s
        and episode_cycles >= slow.minimum_recruited_s * normal.omega_hz * 0.75,
        "exit_crossed_during_recruitment": exit_index is not None,
        "returned_to_low_basin_without_reset": return_index is not None,
        "early_retrigger_suppressed": bool(retrigger.get("early_suppressed", False)),
        "late_retrigger_recovered": bool(retrigger.get("late_recovered", False)),
    }
    return {
        "pass": bool(all(gates.values())),
        "reason": "all_slow_loop_sanity_gates_pass" if all(gates.values()) else "slow_loop_gate_failure",
        "gates": gates,
        "episodes": [
            {
                "onset_s": float(time_s[start]),
                "offset_s": float(time_s[min(stop, time_s.size - 1)]),
                "duration_s": float(time_s[min(stop, time_s.size - 1)] - time_s[start]),
            }
            for start, stop in episodes
        ],
        "first_episode_cycles": episode_cycles,
        "entry_crossing_s": float(time_s[entry_index]) if entry_index is not None else None,
        "exit_crossing_s": float(time_s[exit_index]) if exit_index is not None else None,
        "return_s": float(time_s[return_index]) if return_index is not None else None,
        "retrigger": retrigger,
        "manual_reset_used": False,
    }


def dataclass_dict(instance: Any) -> dict[str, Any]:
    """Small public helper used by the runner's machine-readable provenance."""

    return asdict(instance)
