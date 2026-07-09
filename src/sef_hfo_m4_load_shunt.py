"""M4-3A load->shunt recovery variable (n -> a).

Continuous, activity-driven, baseline-centered slow variable. `n` is an abstract
activity load; `a = a_max * Pi(n)` is the shunt strength. This elementwise ODE is
shared by the offline P0 calibration AND SpatialSlowField.step (network) so both
paths use one implementation (DRY).

Sign contract (spec D2): `a` NEVER divides a signed net current. This module only
produces `a`; the divisive (conductance) / subtractive coupling lives in the engine.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class LoadShuntParams:
    tau_n: float          # ms, load recovery toward n_base (SLOW; > tau_q)
    k_n: float            # load build rate on baseline-centered drive
    rho_n: float          # load consumption via Pi(n)
    n_base: float         # baseline n0 (Hill offset)
    n50: float            # Hill half-point (on n - n_base)
    hill_h: float         # Hill exponent
    a_max: float          # shunt ceiling
    u_n0: float = 0.0     # baseline drive set-point (homeostatic constant, from Arm0)
    n_min: float = 0.0    # clamp
    n_max: float = 10.0   # clamp

    def validate(self) -> None:
        if self.tau_n <= 0:
            raise ValueError("tau_n must be > 0")
        if self.n50 <= 0:
            raise ValueError("n50 must be > 0")
        if self.hill_h <= 0:
            raise ValueError("hill_h must be > 0")
        if self.a_max < 0:
            raise ValueError("a_max must be >= 0")
        if self.n_min > self.n_max:
            raise ValueError("n_min must be <= n_max")


def hill_pi(n, p: LoadShuntParams):
    """Continuous pump/conductance activation Pi(n) in [0,1). NOT a seizure sensor."""
    x = np.maximum(np.asarray(n, float) - p.n_base, 0.0) ** p.hill_h
    return x / (p.n50 ** p.hill_h + x)


def load_shunt_step(n, u_n, dt: float, p: LoadShuntParams):
    """One forward-Euler step of the load ODE + shunt readout.

    dn/dt = -(n - n_base)/tau_n + k_n * [u_n - u_n0]_+ - rho_n * Pi(n)
    a     = a_max * Pi(n_new)

    Elementwise: works on scalars, 1D traces (P0), or 2D grids (SpatialSlowField).
    Returns (n_new, a), both clamped.
    """
    n = np.asarray(n, float)
    u_tilde = np.maximum(np.asarray(u_n, float) - p.u_n0, 0.0)   # baseline-centered, rectified
    dn = -(n - p.n_base) / p.tau_n + p.k_n * u_tilde - p.rho_n * hill_pi(n, p)
    n_new = np.clip(n + dt * dn, p.n_min, p.n_max)
    a = np.clip(p.a_max * hill_pi(n_new, p), 0.0, p.a_max)
    return n_new, a


def event_triggered_a_response(a_trace, event_idx, dt, *, pre_ms, post0_ms, post1_ms):
    """Mean a in [t+post0, t+post1] minus mean a in [t-pre, t], averaged over events.
    a_trace is 1D (per-step). Events without a full pre/post window are skipped.
    Raises if no event has a full window."""
    a = np.asarray(a_trace, float)
    pre = int(round(pre_ms / dt)); p0 = int(round(post0_ms / dt)); p1 = int(round(post1_ms / dt))
    deltas = []
    for i in event_idx:
        i = int(i)
        if i - pre < 0 or i + p1 > a.size:
            continue
        deltas.append(a[i + p0:i + p1].mean() - a[i - pre:i].mean())
    if not deltas:
        raise ValueError("no events with a full pre/post window")
    return float(np.mean(deltas))


def compute_R_A(delta_a_ictal, delta_a_ied):
    """Duty-cycle ratio: sustained-ictal a-accumulation over single-IED a-bump.
    R_A >> 1 is the sensor-free signature (spec 6.1). Returns inf when the IED did
    not move a at all (delta_a_ied <= 0) -> caller flags a soft ictal gate."""
    if delta_a_ied <= 0:
        return float("inf")
    return float(delta_a_ictal / delta_a_ied)
