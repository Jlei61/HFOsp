"""Causal IED-exposure integration ``x_{p,tau}(t)``.

Analytic, closed-form and strictly causal:

    x_tau(e)^-  = x_tau(e-1)^+ * exp(-dt_e / tau_x)
    x_tau(e)^+  = x_tau(e)^-  + L_e_tilde

A clock kernel decays with real elapsed time; an event-count kernel decays one
step per event regardless of elapsed time and is the control that says whether an
effect is about time or about how many events happened.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExposureTrace:
    kind: str          # "clock" or "event_count"
    scale: float       # seconds for clock, events for event_count
    pre: np.ndarray    # (E,) exposure entering event e
    post: np.ndarray   # (E,) exposure after absorbing event e


def integrate_exposure(load: np.ndarray, delta_t: np.ndarray, *, tau_seconds: float,
                       session_opening: np.ndarray | None = None,
                       reset_on_session: bool = False) -> ExposureTrace:
    load = np.asarray(load, dtype=np.float64)
    dt = np.asarray(delta_t, dtype=np.float64)
    pre = np.zeros(len(load))
    post = np.zeros(len(load))
    carry = 0.0
    for e in range(len(load)):
        gap = dt[e]
        if not np.isfinite(gap):
            # a session opening: the unobserved wall time still decays exposure,
            # but we do not know it, so the state is either carried with a full
            # decay or explicitly reset -- both are labelled, never silent.
            carry = 0.0 if reset_on_session else carry
        else:
            carry = carry * float(np.exp(-gap / tau_seconds))
        pre[e] = carry
        carry = carry + float(load[e])
        post[e] = carry
    return ExposureTrace("clock", float(tau_seconds), pre, post)


def integrate_event_count_exposure(load: np.ndarray, *, n_events: float) -> ExposureTrace:
    """Event-count control: decays per event, blind to elapsed time."""
    load = np.asarray(load, dtype=np.float64)
    decay = float(np.exp(-1.0 / max(n_events, 1e-6)))
    pre = np.zeros(len(load))
    post = np.zeros(len(load))
    carry = 0.0
    for e in range(len(load)):
        carry = carry * decay
        pre[e] = carry
        carry = carry + float(load[e])
        post[e] = carry
    return ExposureTrace("event_count", float(n_events), pre, post)
