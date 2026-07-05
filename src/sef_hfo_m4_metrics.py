"""M4 Pass-1 per-cell metric EXTRACTION from a simulate_kick result (spec rev4 §9.1, §10.1).

Reuses src.sef_hfo_snn_metrics (onset_times / onset_axis / peak_active_fraction / self_limit /
event_peak_time / pre_kick_ignition); adds the M4-specific spatial + branching metrics. Every function
here is PURE (operates on already-produced arrays); NO simulation is run in this module.

DEFINITIONAL CHOICES (load-bearing science contract — flagged for review, rev4 §9.1):
- s_grad   = R^2 of the onset~position linear fit (fraction of onset-time variance explained by a spatial
             gradient) in [0,1]. 0 = simultaneous whole-field ignition (no spatial sequence).
- f_off    = fraction of ACTIVE E neurons whose perpendicular distance from the onset-axis line (through the
             event center) exceeds `band_half` mm.
- core_overlap = (post-kick spikes emitted by CORE neurons) / (total post-kick E spikes). Power on the core.
- globality    = participation ratio of the per-neuron post-kick spike-count vector, sum^2/(NE*sumsq), in
             [1/NE, 1]. 1 = uniform (distributed amplitude); ->0 = concentrated (low-amplitude skirt on core).
- self_limited = SPATIAL retreat: late-window recruited extent << peak extent (the wavefront advanced then
             pulled back to core). DISTINCT from tail_returns (temporal rate fall). This distinction keeps a
             sustained-then-terminating bounded core (tail_returns True, self_limited False) OUT of TRIVIAL-B.
- b_delta_avg  = geometric-mean step-to-step population-rate ratio over supra-threshold bins.
- monotonic_saturation = rate stays near its peak through the tail (never falls) AND peak near ceiling.
- finite_energy = peak rate < sat_ceiling (not pinned at a runaway ceiling).
"""
from __future__ import annotations

import numpy as np

from src.sef_hfo_m4_phaseplane import CellMetrics
from src.sef_hfo_snn_metrics import (
    onset_times, onset_axis, peak_active_fraction, self_limit, event_peak_time,
)


# ---------------------------------------------------------------- spatial metrics
def onset_gradient_r2(posE, onset, min_n=20) -> float:
    """s_grad: fraction of onset-time variance explained by a linear spatial gradient (R^2 of t ~ a+g.x).
    0 -> simultaneous ignition (no spatial onset sequence). Reuses the sef_hfo_snn_metrics onset fit."""
    onset = np.asarray(onset, float)
    fin = np.isfinite(onset)
    if fin.sum() < min_n:
        return 0.0
    X = np.asarray(posE, float)[fin]
    tc = onset[fin] - onset[fin].mean()
    Xc = X - X.mean(0)
    g, *_ = np.linalg.lstsq(Xc, tc, rcond=None)
    ss_tot = float((tc * tc).sum())
    if ss_tot < 1e-12:
        return 0.0
    ss_res = float(((tc - Xc @ g) ** 2).sum())
    return max(0.0, 1.0 - ss_res / ss_tot)


def off_axis_fraction(posE, active_mask, axis_unit, center, band_half) -> float:
    """f_off: fraction of ACTIVE E neurons farther than `band_half` mm (perpendicular) from the onset-axis
    line through `center`. axis_unit=None (no gradient) -> nan (undefined)."""
    if axis_unit is None:
        return float("nan")
    X = np.asarray(posE, float)[np.asarray(active_mask, bool)]
    if X.shape[0] == 0:
        return 0.0
    rel = X - np.asarray(center, float)
    perp = rel - np.outer(rel @ np.asarray(axis_unit, float), axis_unit)
    d = np.linalg.norm(perp, axis=1)
    return float((d > band_half).mean())


def core_overlap_spikes(E_spk_bool, dt, t_kick, core_neuron_mask) -> float:
    """core_overlap: post-kick spikes from CORE neurons / total post-kick E spikes. nan if no spikes."""
    i_k = int(round(t_kick / dt))
    counts = np.asarray(E_spk_bool)[i_k:].sum(axis=0).astype(float)
    tot = float(counts.sum())
    if tot <= 0.0:
        return float("nan")
    return float(counts[np.asarray(core_neuron_mask, bool)].sum() / tot)


def globality_pr(E_spk_bool, dt, t_kick) -> float:
    """globality: participation ratio of the per-neuron post-kick spike-count vector = s^2/(NE*ss),
    in [1/NE, 1]. 1 = uniform (distributed amplitude); ->0 = concentrated (skirt on core)."""
    i_k = int(round(t_kick / dt))
    c = np.asarray(E_spk_bool)[i_k:].sum(axis=0).astype(float)
    ss = float((c * c).sum())
    if ss <= 0.0:
        return 0.0
    return float((c.sum() ** 2) / (c.size * ss))


def active_mask_post_kick(E_spk_bool, dt, t_kick) -> np.ndarray:
    """Per-E-neuron 'fired at least once after the kick' boolean."""
    i_k = int(round(t_kick / dt))
    return np.asarray(E_spk_bool)[i_k:].any(axis=0)


# ---------------------------------------------------------------- temporal metrics
def branching_ratio(rate, dt, t_kick, thresh_hz) -> float:
    """b_delta_avg: geometric-mean step-to-step rate ratio over supra-threshold post-kick bins (rev4 §9.1
    relaxation: window-averaged, so a burst rising phase's instantaneous B>1 does not read as runaway).
    Returns 0.0 if there are no supra-threshold transitions."""
    r = np.asarray(rate, float)
    i_k = int(round(t_kick / dt))
    seg = r[i_k:]
    a = seg[:-1]
    b = seg[1:]
    m = a > thresh_hz
    if not m.any():
        return 0.0
    ratios = b[m] / np.maximum(a[m], 1e-9)
    ratios = np.clip(ratios, 1e-6, None)
    return float(np.exp(np.mean(np.log(ratios))))


def monotonic_saturation(rate, dt, t_kick, sat_ceiling, tail_frac=0.85, ceiling_frac=0.8) -> bool:
    """Runaway signature: the tail-window mean rate stays >= tail_frac*peak (never falls) AND the peak is
    >= ceiling_frac*sat_ceiling (pinned near the saturation ceiling)."""
    r = np.asarray(rate, float)
    i_k = int(round(t_kick / dt))
    post = r[i_k:]
    if post.size == 0:
        return False
    peak = float(post.max())
    tail = post[int(0.7 * post.size):]
    tail_mean = float(tail.mean()) if tail.size else 0.0
    return (peak >= ceiling_frac * sat_ceiling) and (tail_mean >= tail_frac * peak)


def finite_energy_ok(rate, dt, t_kick, sat_ceiling) -> bool:
    """finite_energy: peak post-kick rate below the saturation ceiling (not a pinned runaway)."""
    r = np.asarray(rate, float)
    i_k = int(round(t_kick / dt))
    post = r[i_k:]
    return bool(post.size and post.max() < sat_ceiling)


def spatial_self_limited(E_spk_bool, dt, t_kick, peak_t, late_after, win, retreat_factor) -> bool:
    """SPATIAL retreat (TRIVIAL-B 'retreats to core'): recruited extent in a LATE window (peak_t+late_after,
    +win) is < retreat_factor * the peak extent (measured around peak_t). Distinct from a temporal rate
    fall — a stable core that just terminates in time is NOT a spatial retreat."""
    peak_ext = peak_active_fraction(E_spk_bool, dt, max(t_kick, peak_t - win / 2), peak_t + win / 2)
    late_ext = peak_active_fraction(E_spk_bool, dt, peak_t + late_after, peak_t + late_after + win)
    if peak_ext <= 0.0:
        return False
    return late_ext < retreat_factor * peak_ext


# ---------------------------------------------------------------- assembler
def extract_cell_metrics(res, posE, dt, t_kick, *, core_neuron_mask, center, T_min,
                         band_half, sat_ceiling, thresh_hz, retreat_factor,
                         axis_unit=None, event_lo=None, event_hi=None) -> CellMetrics:
    """Assemble a CellMetrics (src.sef_hfo_m4_phaseplane) from a simulate_kick result `res`
    (needs res['E_spk_bool'], res['rate_E']) + E positions. `center` = the axis line's anchor point.
    `axis_unit` = the propagation axis for f_off; if None it is DERIVED from the onset gradient, else the
    caller's KNOWN axis is used (e.g. a subject's registered source->sink axis). `event_lo/hi` default to
    [t_kick, end]. All threshold/window params come from calibration (§9.1). UNITS: `sat_ceiling` and
    `thresh_hz` are per-neuron mean Hz; res['rate_E'] (a per-step spike COUNT) is converted to Hz internally
    before the saturation/energy/branching comparisons."""
    E_spk = np.asarray(res["E_spk_bool"])
    rate = np.asarray(res["rate_E"], float)
    nsteps = E_spk.shape[0]
    if event_lo is None:
        event_lo = t_kick
    if event_hi is None:
        event_hi = nsteps * dt

    onset = onset_times(E_spk, dt, t_kick)
    axis_for_foff = axis_unit if axis_unit is not None else onset_axis(posE, onset)
    active = active_mask_post_kick(E_spk, dt, t_kick)
    sl = self_limit(rate, dt, t_kick)
    peak_t = float(sl["peak_t"])

    act_frac = peak_active_fraction(E_spk, dt, event_lo, event_hi)
    s_grad = onset_gradient_r2(posE, onset)
    f_off = off_axis_fraction(posE, active, axis_for_foff, center, band_half)
    core_ov = core_overlap_spikes(E_spk, dt, t_kick, core_neuron_mask)
    glob = globality_pr(E_spk, dt, t_kick)
    # rate_E is a per-step SPIKE COUNT (kick_probe.py:305), NOT Hz. Convert to per-neuron mean Hz using
    # the engine's own readout formula rate_E/NE/dt*1e3 (kick_probe.py:360) so `sat_ceiling` (Hz) and
    # `thresh_hz` (Hz) are meaningful. self_limit above stays on the raw rate (it is ratio-based).
    NE = E_spk.shape[1]
    rate_hz = rate / NE / dt * 1e3
    b_avg = branching_ratio(rate_hz, dt, t_kick, thresh_hz)             # thresh_hz = per-neuron Hz
    mono_sat = monotonic_saturation(rate_hz, dt, t_kick, sat_ceiling)   # sat_ceiling = per-neuron Hz
    fin_e = finite_energy_ok(rate_hz, dt, t_kick, sat_ceiling)
    self_lim = spatial_self_limited(E_spk, dt, t_kick, peak_t, late_after=80.0, win=40.0,
                                    retreat_factor=retreat_factor)

    persist = bool(sl["burst_duration_ms"] >= T_min)
    tail_returns = bool(sl["returned"] and sl["tail_complete"])

    return CellMetrics(
        persist=persist,
        act_frac=float(act_frac),
        s_grad=float(s_grad),
        f_off=float(f_off) if np.isfinite(f_off) else 0.0,
        core_overlap=float(core_ov) if np.isfinite(core_ov) else 0.0,
        globality=float(glob),
        self_limited=self_lim,
        b_delta_avg=float(b_avg),
        monotonic_saturation=mono_sat,
        tail_returns=tail_returns,
        finite_energy=fin_e,
    )
