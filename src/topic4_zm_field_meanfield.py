# src/topic4_zm_field_meanfield.py
"""Phase-0 uniform mean-field (r,mu,S) for the reduced S_L(x)+S_G field.
Fix A dual pool: alpha*S divides the recurrent term (matches SNN S_G on I_E_rec) and beta*S subtracts on the
membrane (NEW on this line -- the Z/M sg arm ran beta_SG=0). Gates the 2-D field: no orbit -> STOP.
Spec docs/superpowers/specs/2026-07-24-topic4-zm-reduced-field-Sl-Sg-design.md rev3 §6.0."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

def F(u, u_half=0.5):
    x = max(float(u), 0.0)
    return x / (u_half + x)

def psi(r, r50=0.4, n=2.0):
    x = max(float(r), 0.0) ** n
    return x / (r50 ** n + x)

def psi_prime(r, r50=0.4, n=2.0):
    r = max(float(r), 0.0)
    if r <= 0.0:
        return 0.0
    a = r50 ** n
    return n * r ** (n - 1) * a / (a + r ** n) ** 2

@dataclass
class MFParams:
    W0: float; alpha: float; beta: float; theta: float; I0: float
    tau_a: float = 10.0; tau_mu: float = 30.0; tau_S: float = 80.0; S_max: float = 1.0

def simulate_meanfield(p: MFParams, T=6000.0, dt=0.25, r0=0.15):
    n = int(round(T / dt)); r, mu, S = float(r0), 0.0, 0.0
    tr = np.empty((n, 3))
    for t in range(n):
        u = p.I0 + p.W0 * r / (1.0 + p.alpha * S) - p.beta * S - p.theta
        r = max(r + dt * (-r + F(u)) / p.tau_a, 0.0)
        mu = mu + dt * (-mu + psi(r)) / p.tau_mu
        S = S + dt * (-S + p.S_max * mu) / p.tau_S
        tr[t] = (r, mu, S)
    return tr

def detect_orbit(traj, dt, settle=0.5):
    r = np.asarray(traj)[int(len(traj) * settle):, 0]
    peak, trough, mean = float(r.max()), float(r.min()), float(r.mean())
    depth = (peak - trough) / (mean + 1e-9)
    mid = 0.5 * (peak + trough)
    cr = np.flatnonzero((r[:-1] < mid) & (r[1:] >= mid))
    period_ms = float(np.mean(np.diff(cr)) * dt) if cr.size >= 2 else float("nan")
    return dict(oscillates=bool(cr.size >= 4 and depth > 0.5 and trough < 0.25 * peak and peak > 0.1),
                depth=depth, trough=trough, peak=peak, period_ms=period_ms, ncyc=int(cr.size))
