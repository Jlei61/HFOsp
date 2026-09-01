"""Attach a frozen per-event state trajectory to fixed physical-time anchors.

The v0.2 common contract §5.2 propagates the state from the last event to the
grid instant. Two rules make that safe and are enforced here:

* **causal prefix** -- an anchor reads the last event *strictly before* it.
  An event landing exactly on the anchor is the thing we are predicting, not an
  input to the prediction.
* **no unbounded staleness** -- a state older than ``max_age_seconds`` is
  refused rather than stretched across a silence it never observed. The age is
  returned so downstream models can carry it as a covariate instead of pretending
  a six-hour-old state is current.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AttachedState:
    state: np.ndarray        # (n_anchors, d), NaN where unavailable
    age_seconds: np.ndarray  # (n_anchors,), NaN where unavailable
    available: np.ndarray    # (n_anchors,) bool
    source_event_index: np.ndarray  # (n_anchors,) int, -1 where unavailable


def attach_state_to_anchors(
    anchor_epochs: np.ndarray,
    event_epochs: np.ndarray,
    event_states: np.ndarray,
    max_age_seconds: float,
) -> AttachedState:
    """Carry each anchor the state of the last event strictly preceding it."""

    a = np.asarray(anchor_epochs, float)
    t = np.asarray(event_epochs, float)
    s = np.asarray(event_states, float)
    if t.size != s.shape[0]:
        raise ValueError("event_epochs and event_states must describe the same events")

    order = np.argsort(t, kind="stable")
    t_sorted, s_sorted = t[order], s[order]

    # strictly-before: 'left' puts an exact tie on the anchor *after* the cut
    idx = np.searchsorted(t_sorted, a, side="left") - 1

    state = np.full((a.size, s.shape[1]), np.nan)
    age = np.full(a.size, np.nan)
    avail = np.zeros(a.size, dtype=bool)
    src = np.full(a.size, -1, dtype=int)

    ok = idx >= 0
    if ok.any():
        ages = a[ok] - t_sorted[idx[ok]]
        fresh = ages <= float(max_age_seconds)
        rows = np.flatnonzero(ok)[fresh]
        picked = idx[ok][fresh]
        state[rows] = s_sorted[picked]
        age[rows] = ages[fresh]
        avail[rows] = True
        src[rows] = order[picked]
    return AttachedState(state=state, age_seconds=age, available=avail,
                         source_event_index=src)
