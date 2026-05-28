"""Trajectory simulation + burst detection + regime classification.

simulate_trajectory uses the numba JIT kernels from hr_core + ou_noise.
detect_bursts and classify_regime stay numpy / pure Python (not hot loops).

Spec: docs/superpowers/specs/2026-05-27-sef-itp-phase4-v1-design.md §3 Stage 1
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from .hr_config import BurstConfig, RegimeConfig
from .hr_core import HRParams, rk4_step_jit
from .ou_noise import generate_ou_noise


@njit(cache=True, fastmath=True)
def _trajectory_jit(
    n_steps: int, dt: float,
    x0: float, y0: float, z0: float,
    a: float, b: float, c: float, d: float, r: float, s: float, x_R: float,
    I: float,
    eta_trace: np.ndarray,
) -> np.ndarray:
    """numba hot loop: run RK4 trajectory using pre-computed eta trace."""
    traj = np.empty((n_steps, 3))
    traj[0, 0] = x0
    traj[0, 1] = y0
    traj[0, 2] = z0
    x, y, z = x0, y0, z0
    for i in range(1, n_steps):
        x, y, z = rk4_step_jit(x, y, z, a, b, c, d, r, s, x_R,
                                I, eta_trace[i], dt)
        traj[i, 0] = x
        traj[i, 1] = y
        traj[i, 2] = z
    return traj


def simulate_trajectory(
    params: HRParams,
    I: float, T: float, dt: float,
    sigma_ou: float, tau_ou: float, seed: int,
    x0: float = -1.6, y0: float = -10.0, z0: float = 2.0,
    burn_in: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run HR single-node trajectory for duration T, return (t, traj).

    Uses numba JIT kernels under the hood (~50ms per simulation for
    T=500, dt=0.05 = 10000 steps after first compile).

    Args:
        burn_in: discard the first ``burn_in`` HR time units of trajectory
            (relaxation from initial conditions; OU noise samples during
            burn-in are consumed but the trajectory slice is dropped).
            Returned t starts at 0; traj has shape (int(T/dt), 3).
            Default 0.0 = no burn-in (back-compat). Downstream consumers
            (evaluate_cell in Task 4) pass a non-zero default so the
            baseline picker isn't fooled by initial-condition transients.
    """
    n_steps = int(T / dt)
    burn_n = int(burn_in / dt)
    total_steps = burn_n + n_steps
    t = np.arange(n_steps) * dt
    eta = generate_ou_noise(total_steps, dt, tau_ou, sigma_ou, seed)
    p = params
    full_traj = _trajectory_jit(
        total_steps, dt, x0, y0, z0,
        p.a, p.b, p.c, p.d, p.r, p.s, p.x_R,
        I, eta,
    )
    return t, full_traj[burn_n:]


def detect_bursts(
    x: np.ndarray, t: np.ndarray, cfg: BurstConfig,
) -> list[tuple[float, float]]:
    """Detect bursts in x trace via hysteresis + min-duration filter.

    Algorithm:
        1. Find above-threshold contiguous segments
        2. Bridge segments separated by gaps shorter than cfg.bridge_gap
        3. Filter by minimum duration cfg.min_burst_duration
    """
    above = x > cfg.x_threshold
    if not above.any():
        return []
    trans = np.diff(above.astype(np.int8))
    rises = np.where(trans == 1)[0] + 1
    falls = np.where(trans == -1)[0] + 1
    if above[0]:
        rises = np.concatenate([[0], rises])
    if above[-1]:
        falls = np.concatenate([falls, [len(x)]])
    segments = list(zip(rises, falls))
    # Bridge close-together segments
    bridged: list[tuple[int, int]] = []
    for seg in segments:
        if bridged and (t[seg[0]] - t[bridged[-1][1] - 1]) < cfg.bridge_gap:
            bridged[-1] = (bridged[-1][0], seg[1])
        else:
            bridged.append(seg)
    # Filter by min duration
    out: list[tuple[float, float]] = []
    for r, f in bridged:
        t_start = float(t[r])
        t_end = float(t[f - 1])
        if (t_end - t_start) >= cfg.min_burst_duration:
            out.append((t_start, t_end))
    return out


@dataclass(frozen=True)
class BurstEnvelope:
    """A node burst-envelope event: a cluster of merged spike-level excursions.

    Stage 1b user-ratified observation unit (2026-05-28): the primary event is
    the envelope, not the single spike. ``onset`` is the propagation timestamp
    Stage 2-3 will diff between nodes — it is the FIRST spike's start, not the
    peak time or centroid.
    """
    onset: float    # first spike's start time (HR units) — propagation timestamp
    offset: float   # last spike's end time
    n_spikes: int   # number of spike-level excursions merged into this envelope
    peak_x: float   # max x over [onset, offset]

    @property
    def duration(self) -> float:
        return self.offset - self.onset


def detect_burst_envelopes(
    x: np.ndarray, t: np.ndarray, cfg: BurstConfig,
) -> list[BurstEnvelope]:
    """Group spike-level excursions into burst-envelope events.

    Contract (Stage 1b plan §contract clauses):
        #1 re-use: spike atoms come from ``detect_bursts`` (not reinvented).
        #2 merge: spikes whose inter-spike gap (prev offset → next onset) is
           strictly < ``cfg.envelope_gap`` collapse into one envelope; a gap
           ≥ envelope_gap starts a new envelope.
        #3 onset: envelope onset = the first spike's START within the cluster.
        #4 secondary: each envelope carries offset, n_spikes, peak_x (and
           duration via property).
        #5 config: the merge threshold is ``cfg.envelope_gap`` (not a literal).
    """
    spikes = detect_bursts(x, t, cfg)  # clause #1: re-use spike-level detector
    if not spikes:
        return []
    # clause #2: greedily merge consecutive spikes whose gap < envelope_gap.
    clusters: list[list[tuple[float, float]]] = [[spikes[0]]]
    for onset, offset in spikes[1:]:
        prev_offset = clusters[-1][-1][1]
        if (onset - prev_offset) < cfg.envelope_gap:
            clusters[-1].append((onset, offset))
        else:
            clusters.append([(onset, offset)])
    out: list[BurstEnvelope] = []
    for cluster in clusters:
        env_onset = cluster[0][0]    # clause #3: first spike start
        env_offset = cluster[-1][1]
        window = (t >= env_onset) & (t <= env_offset)
        out.append(BurstEnvelope(
            onset=env_onset,
            offset=env_offset,
            n_spikes=len(cluster),                  # clause #4
            peak_x=float(x[window].max()),          # clause #4
        ))
    return out


def classify_regime(
    bursts: list[tuple[float, float]],
    T: float,
    cfg: RegimeConfig,
) -> str:
    """Classify regime from burst list: silent / excitable / repetitive-burst / unstable.

    Operational definitions (spec §3 stage 1 + plan task 3):
        - silent: 0 bursts
        - unstable: any burst longer than cfg.max_burst_duration (takes precedence)
        - excitable: all bursts ≤ cfg.excitable_max_burst AND mean IBI ≥ cfg.excitable_min_ibi
        - repetitive-burst: otherwise (short IBI = spontaneous regular firing)
    """
    if not bursts:
        return "silent"
    durations = [end - start for start, end in bursts]
    if any(d > cfg.max_burst_duration for d in durations):
        return "unstable"
    if len(bursts) >= 2:
        ibis = [bursts[i + 1][0] - bursts[i][1] for i in range(len(bursts) - 1)]
        mean_ibi = float(np.mean(ibis))
    else:
        mean_ibi = float("inf")
    if (
        all(d <= cfg.excitable_max_burst for d in durations)
        and mean_ibi >= cfg.excitable_min_ibi
    ):
        return "excitable"
    return "repetitive-burst"
