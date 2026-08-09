"""Whether a per-cell, spike-driven load can tell the discharge from the interictal train.

Every terminator tried on this substrate has failed on one of two sides.  A load that each cell
builds from its own firing was on during the interictal train and suppressed it (the pump line's
baseline gate, and the adaptation arm that blocked entry outright).  A brake that read whole-array
recruitment left the interictal train untouched and did not terminate anything.  The accepted
contract requires the *state* to be per-cell; the entry ledger's separation analysis says the only
measured variable that tells the states apart is how much of the array joins in.  Those two pull in
opposite directions only if a per-cell load cannot separate them -- which is a measurable question,
not a design preference.

So this module asks it directly: given the same cell's firing in the two states, is there a load
level above everything the interictal train produces and below where the discharge sits?  If there
is, an actuator whose half-activation is placed there is seizure-selective without reading anything
but the cell's own spikes.  If there is not, no per-cell spike-driven mechanism can be, whatever
its time constant or its actuator shape.

Two readouts, because a threshold placed on a transient is not the threshold placed on a level:

* **peak** -- the highest load a cell reaches.  Decides whether the interictal train ever crosses.
* **settled** -- where the load sits while a state persists.  Decides whether the discharge stays
  across.

``src.topic4_mz_slowvars.replay_adaptation_peak`` already answers the first and is left alone; this
integrates the same ODE once and returns both, and is tested against it so the two cannot drift.
"""
from __future__ import annotations

import numpy as np


def replay_load(E_spk_bool, dt_ms, tau_ms, init=None, settle_from_ms=0.0):
    """Integrate the per-cell adaptation load over a spike raster, exactly as the engine steps it.

    The engine decays before it increments (``mz_slow_vars.py``: ``m -= (m/tau)*dt`` then
    ``m[spk] += 1``); replaying in the other order shifts every trajectory by one step's decay.

    ``init`` seeds the load per cell.  A replay that starts at zero spends its first few time
    constants charging, so over a window shorter than that it reports the charge ramp rather than
    the state -- and for the interictal side that would understate the load and make the separation
    look better than it is.  The stationary load of a cell firing at rate r is exactly r*tau, so
    passing it removes the ramp instead of waiting it out.
    """
    E = np.asarray(E_spk_bool)
    nsteps, ncell = E.shape
    m = np.zeros(ncell) if init is None else np.array(init, dtype=float)
    if m.shape != (ncell,):
        raise ValueError(f"init has shape {m.shape}, expected ({ncell},)")
    peak = m.copy()
    settle_i = int(round(settle_from_ms / dt_ms))
    acc, n_acc = np.zeros(ncell), 0
    for t in range(nsteps):
        m -= (m / tau_ms) * dt_ms
        sp = E[t]
        if sp.any():
            m[sp] += 1.0
        np.maximum(peak, m, out=peak)
        if t >= settle_i:
            acc += m
            n_acc += 1
    return dict(peak=peak, settled=(acc / n_acc if n_acc else m.copy()), final=m.copy())


def stationary_load(per_cell_hz, tau_ms):
    """The load a cell settles at, from its rate alone.

    The ODE is linear, so the long-run mean load is r*tau exactly, whatever the firing pattern.
    That is also why the time constant on its own cannot buy selectivity: it scales both states by
    the same factor and leaves their ratio at the ratio of the rates.
    """
    return np.asarray(per_cell_hz, float) * (tau_ms / 1000.0)


def separation(interictal_peak, ictal_settled, quiet_q=99.9, need_frac=0.20):
    """How far apart the two states' loads sit, reported as a gap rather than a verdict.

    An earlier version asked "is there a level above *every* interictal load and below the
    discharge", and answered with a boolean.  Both halves were wrong.  A handful of cells touching
    a level for a few milliseconds during a normal event does not necessarily disturb the
    interictal state -- what matters is how much activation the population carries, which is what
    ``aggregate_activation`` measures.  And placing the threshold on a percentile while reporting
    the ratio to it reads as a gap when it is not one: on this substrate the ratio to the 99.9th
    percentile was 1.5-3.2x while the ratio to the interictal *maximum* was 0.98-1.96x.

    So this returns the numbers and leaves the criterion to the caller.  ``gap`` is the honest one:
    the discharge's lowest load over the interictal highest.  Below 1 the distributions overlap.
    """
    q = np.asarray(interictal_peak, float)
    ict = np.asarray(ictal_settled, float)
    K = float(np.percentile(q, quiet_q))
    quiet_max = float(q.max())
    return dict(
        K=K, quiet_q=quiet_q, quiet_max=quiet_max,
        quiet_frac_above_K=float(np.mean(q > K)),
        ictal_frac_above_K=float(np.mean(ict > K)),
        ictal_median=float(np.median(ict)), ictal_min=float(np.min(ict)),
        headroom=float(np.median(ict) / K) if K > 0 else float("inf"),
        gap=float(np.min(ict) / quiet_max) if quiet_max > 0 else float("inf"),
        overlap_frac=float(np.mean(q > np.min(ict))),
        need_frac=need_frac,
        boundary=("a gap here is between a load a cell reaches and a load it sits at; whether an "
                  "actuator sees the first depends on how fast it opens, which this does not "
                  "measure.  A small gap rules out a memoryless monotone threshold on this one "
                  "linear-filtered load -- not mechanisms using duration, two timescales, joint "
                  "voltage-load gating, within-burst accumulation, or state-dependent activation."),
    )


def aggregate_activation(load, K, n):
    """What fraction of the population a cooperative actuator would switch on at these loads.

    The gap between the extremes decides whether one threshold can be slipped between the two
    states; this decides whether the mechanism is *quiet* in one and *engaged* in the other, which
    is the question the interictal side actually failed on every previous attempt.  A distribution
    whose bulk sits far below K carries almost no activation even when its own tail crosses.
    """
    x = (np.asarray(load, float) / float(K)) ** float(n)
    a = x / (1.0 + x)
    return dict(mean=float(a.mean()), frac_half_on=float(np.mean(a > 0.5)),
                median=float(np.median(a)), K=float(K), n=float(n))
