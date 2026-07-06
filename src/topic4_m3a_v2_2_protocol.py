"""M3A-v2.2 sustained drive protocol builders (runner-level nu_signal_fn; engine untouched).

simulate_kick(p, net, KICK_BOOST, nu_signal_fn=f, ...) evaluates nu_now = f(t_ms) + xi each step
(kick_probe.py:229). f returns the absolute external drive rate nu_ext(t) = nu_theta * r(t).
Canonical math: docs/snn_core_model_equations.md §B6 (event protocol) + the v2.2 design spec §4.1.

HOLD (ramp_hold_drive) is the PRIMARY recovery gate (drive NOT withdrawn -> return must be
endogenous, spec C2). ramp_release_drive is the exogenous-recovery CONTROL only.
"""
from __future__ import annotations


def _ramp_r(t, r0, r_hold, t0, t_ramp):
    """Linear ramp r0->r_hold over [t0, t0+t_ramp], clamped outside."""
    if t <= t0:
        return r0
    frac = min(max((t - t0) / t_ramp, 0.0), 1.0)
    return r0 + (r_hold - r0) * frac


def ramp_hold_drive(nu_theta, r0, r_hold, t0, t_ramp):
    """Ramp r0->r_hold over [t0, t0+t_ramp], then HOLD r_hold forever (primary protocol, C2)."""
    def f(t_ms):
        return nu_theta * _ramp_r(t_ms, r0, r_hold, t0, t_ramp)
    return f


def ramp_release_drive(nu_theta, r0, r_hold, t0, t_ramp, t_release):
    """ramp_hold but drop back to r0 at t_release (exogenous-recovery CONTROL only, C2)."""
    def f(t_ms):
        if t_ms >= t_release:
            return nu_theta * r0
        return nu_theta * _ramp_r(t_ms, r0, r_hold, t0, t_ramp)
    return f
