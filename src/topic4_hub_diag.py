"""Hub-relay recruitment diagnostics for the M3 hub-gated SNN screen.

A spiking network has a "corridor" region, a "hub" subset (a few cells at the
corridor's far end), and a "global" region beyond it. An event either (a) stays
in the corridor and dies at the hub (interictal), or (b) crosses the hub and
broadcasts to the global region (seizure-like).

The discriminating signal is TIMING: in a relay the hub fires AFTER the corridor
onset and BEFORE the global region, and in the interictal case the global region
barely fires. This module turns a boolean spike record into those timing /
recruitment scalars.

Pure numpy; no engine dependency.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def _region_fraction(any_spiked: np.ndarray, region_idx: np.ndarray) -> float:
    """Fraction of cells in `region_idx` that spiked at least once.

    `any_spiked` is a per-cell boolean (True if the cell spiked anywhere in the
    record). Empty region => 0.0 (no cells to recruit).
    """
    region_idx = np.asarray(region_idx, dtype=int)
    if region_idx.size == 0:
        return 0.0
    return float(np.count_nonzero(any_spiked[region_idx]) / region_idx.size)


def _region_onset_ms(E_spk_bool: np.ndarray, region_idx: np.ndarray, dt: float) -> float:
    """First timestep at which ANY cell in `region_idx` spikes, times dt.

    Returns np.nan if the region is empty or never spikes.
    """
    region_idx = np.asarray(region_idx, dtype=int)
    if region_idx.size == 0:
        return float("nan")
    # (n_timesteps,) boolean: did any region cell spike at this timestep?
    region_any_t = np.any(E_spk_bool[:, region_idx], axis=1)
    fired = np.flatnonzero(region_any_t)
    if fired.size == 0:
        return float("nan")
    return float(fired[0] * dt)


def _diff_ms(later_ms: float, earlier_ms: float) -> float:
    """later_ms - earlier_ms, propagating nan if either operand is nan."""
    if np.isnan(later_ms) or np.isnan(earlier_ms):
        return float("nan")
    return float(later_ms - earlier_ms)


def hub_global_recruitment(
    E_spk_bool: np.ndarray,
    hub_idx: np.ndarray,
    global_idx: np.ndarray,
    corridor_idx: np.ndarray,
    dt: float,
) -> Dict[str, float]:
    """Hub-relay recruitment + timing diagnostics for one spike record.

    Parameters
    ----------
    E_spk_bool : (n_timesteps, n_E) bool array
        True where an E cell spiked at that timestep.
    hub_idx, global_idx, corridor_idx : int index arrays
        E-local columns of `E_spk_bool` for each region. `hub_idx` is a subset
        of `corridor_idx`.
    dt : float
        ms per timestep.

    Returns
    -------
    dict of float (np.nan where a region is empty / silent)
        See module docstring for the relay-timing semantics. Positive
        `global_first_spike_after_hub_ms` => global fires after hub (broadcast
        direction); positive `hub_after_corridor_ms` => hub is a relay, not the
        primary igniter.
    """
    E_spk_bool = np.asarray(E_spk_bool, dtype=bool)

    # Per-cell "ever spiked" reduction over the whole record.
    if E_spk_bool.ndim != 2:
        raise ValueError("E_spk_bool must be 2-D (n_timesteps, n_E)")
    any_spiked = np.any(E_spk_bool, axis=0)  # (n_E,)

    hub_recruited_fraction = _region_fraction(any_spiked, hub_idx)
    global_E_spike_fraction = _region_fraction(any_spiked, global_idx)
    corridor_spike_fraction = _region_fraction(any_spiked, corridor_idx)

    corridor_onset_ms = _region_onset_ms(E_spk_bool, corridor_idx, dt)
    hub_onset_ms = _region_onset_ms(E_spk_bool, hub_idx, dt)
    global_onset_ms = _region_onset_ms(E_spk_bool, global_idx, dt)

    global_first_spike_after_hub_ms = _diff_ms(global_onset_ms, hub_onset_ms)
    hub_after_corridor_ms = _diff_ms(hub_onset_ms, corridor_onset_ms)

    return {
        "hub_recruited_fraction": hub_recruited_fraction,
        "global_E_spike_fraction": global_E_spike_fraction,
        "corridor_spike_fraction": corridor_spike_fraction,
        "corridor_onset_ms": corridor_onset_ms,
        "hub_onset_ms": hub_onset_ms,
        "global_onset_ms": global_onset_ms,
        "global_first_spike_after_hub_ms": global_first_spike_after_hub_ms,
        "hub_after_corridor_ms": hub_after_corridor_ms,
    }
