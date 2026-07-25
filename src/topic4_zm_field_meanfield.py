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

# append to src/topic4_zm_field_meanfield.py
import itertools

_DEFAULT_GRID = dict(W0=[2, 3, 4, 6], alpha=[1, 2, 4], beta=[0, 1, 2, 4, 8], theta=[0.4, 0.5, 0.6])

def contiguous_runs(flags):
    """Half-open (i0,i1) index runs of consecutive True in `flags`."""
    runs, start = [], None
    for i, f in enumerate(list(flags) + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            runs.append((start, i)); start = None
    return runs

def _selection_key(cfg):
    W0, alpha, beta, theta = cfg
    return (beta, abs(np.log2(alpha / 16.0)), alpha, abs(W0 - 2.0), W0, abs(theta - 0.5), theta)

def meanfield_continuation(grid=None, I0s=None, dt=0.25, min_seg=5):
    """Continuation over the pre-registered grid; MINIMAL-INTERVENTION selection (smallest beta, then closest
    to the SNN anchor alpha=16, then W0, then theta; deterministic lexicographic tie-break). Only a SINGLE
    contiguous oscillatory I0 run of >= min_seg points is usable; the 5 levels come from its interior."""
    grid = dict(_DEFAULT_GRID if grid is None else grid)
    I0s = np.arange(0.5, 2.01, 0.1) if I0s is None else np.asarray(I0s, float)
    cands = []
    for W0, alpha, beta, theta in itertools.product(grid["W0"], grid["alpha"], grid["beta"], grid["theta"]):
        flags, pers = [], []
        for I0 in I0s:
            o = detect_orbit(simulate_meanfield(MFParams(W0, alpha, beta, theta, float(I0)), dt=dt), dt)
            flags.append(o["oscillates"]); pers.append(o["period_ms"])
        runs = [(a, b) for a, b in contiguous_runs(flags) if b - a >= min_seg]
        if len(runs) != 1:                      # 0 -> no usable segment; >1 -> ambiguous, skip (spec §6.0)
            continue
        a, b = runs[0]
        cands.append(((W0, alpha, beta, theta), a, b, float(np.nanmedian(pers[a:b]))))
    if not cands:
        return dict(has_orbit=False, operating_point=None, segment=None, n_configs_with_segment=0)
    cfg, a, b, per = sorted(cands, key=lambda c: _selection_key(c[0]))[0]
    seg_I0 = I0s[a:b]
    interior = seg_I0[1:-1]
    mid = float(interior[len(interior) // 2])
    return dict(has_orbit=True, n_configs_with_segment=len(cands),
                operating_point=dict(W0=cfg[0], alpha=cfg[1], beta=cfg[2], theta=cfg[3], I0=mid),
                segment=dict(I0_lo=float(seg_I0[0]), I0_hi=float(seg_I0[-1]),
                             interior_I0s=[float(x) for x in interior], period_ms=per))
