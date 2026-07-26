"""FCXR pump lifecycle — dimensionless activity-dependent load u_i and its electrogenic pump current.

NAMING CONTRACT (spec §2.1): ``u_i`` is an **activity-dependent intracellular load
(Na/pump-inspired)**. It is NOT an intracellular sodium concentration, an ATP model, or a complete
ionic-homeostasis model, and must never be reported as one.

    phi(u) = u^h / (1 + u^h)                      h = 3 fixed for the primary tier
    du/dt  = a_load * S_i(t) - phi(u_i)/tau_N     S_i = spike train
    I_pump_excess = Imax * [phi(u_i) - p0_i]      distributionally baseline-centered, NO positive part

Three clauses that are science contract, not implementation detail (spec §2.2/§2.3):

  1. the spike jump is PER SPIKE (never scaled by dt); the clearance IS scaled by dt/tau_N;
  2. the SAME phi drives the clearance and the membrane current;
  3. the membrane effect is Imax*(phi-p0) with NO positive part -- ``+Imax*p0`` compensates the
     steady state already implicit in the FCXR baseline, so a negative excess only means "pump
     activation below the baseline reference", never "pump running backwards".

Contract enumerated 1:1 in tests/test_topic4_mz_fcxr_pump.py.
Design: docs/superpowers/specs/2026-07-26-topic4-mz-fcxr-pump-lifecycle-design.md
"""
from __future__ import annotations

import numpy as np

PRIMARY_H = 3


def require_primary_h(h):
    """Primary tier fixes h=3 (spec §2.1); h in {2,4} is a DEFERRED sensitivity, not a sweep axis."""
    if int(h) != PRIMARY_H:
        raise ValueError(f"primary tier requires h={PRIMARY_H} (got {h}); h in {{2,4}} is a "
                         "deferred Tier-A sensitivity, not a primary sweep axis")


def pump_activation(u, h=PRIMARY_H):
    """phi(u) = u^h/(1+u^h): monotone, smooth, in [0,1), phi(0)=0. Same phi for clearance + membrane."""
    uh = np.asarray(u, float) ** h
    return uh / (1.0 + uh)


def step_spike_load(u, spikes, *, a_load, tau_N, dt, h=PRIMARY_H):
    """One discrete load step (spec §2.2), evaluated at the PRE-step load u(t^-):

        u(t+dt) = max[0, u(t) + a_load*N_spike - (dt/tau_N)*phi(u(t))]

    ``spikes`` is a per-cell spike COUNT for this step (a bool mask is the count 0/1 case). The jump
    carries no dt; the clearance carries dt/tau_N. Non-finite load fails fast -- a candidate that
    blows up is a failed candidate, not something to clamp (spec §2.2 "safety cap = fail-fast").
    """
    u = np.asarray(u, float)
    if not np.all(np.isfinite(u)):
        raise FloatingPointError("non-finite activity-dependent load u")
    jump = a_load * np.asarray(spikes, float)                 # per-spike, NOT scaled by dt
    clearance = (dt / tau_N) * pump_activation(u, h)          # scaled by dt/tau_N, at u(t^-)
    return np.maximum(u + jump - clearance, 0.0)


def excess_pump_current(u, p0, *, Imax, h=PRIMARY_H):
    """Baseline-centered electrogenic pump current Imax*[phi(u)-p0] subtracted from the E drive.

    NO positive part: phi<p0 gives a negative excess (pump activation below the baseline reference),
    which cancels the mean bias that a rectifier would inject from baseline fluctuations alone.
    """
    return Imax * (pump_activation(u, h) - np.asarray(p0, float))
