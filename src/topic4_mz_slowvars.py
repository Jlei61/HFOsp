"""M4-MZ phenotype classifier (7 labels) + slow-off calibration helpers.

Design: docs/superpowers/specs/2026-07-18-topic4-mz-per-neuron-slowvars-design.md §6 (calibration)
and §8 (phenotype contract).

Phenotype contract (branch-scoped): the slow-off spontaneous interictal-event distribution is the
ONLY baseline. This branch does NOT require axis-breaking (design §8 override) -- axis/off-axis/
globality are DESCRIPTIVE only and are not gates here. Labels:

    interictal_like | expanded_bounded | expanded_returned | fragment | suppress | runaway | insufficient

Gate STRUCTURE is the science contract; the numeric factors are calibration and are exercised on
SYNTHETIC fixtures (tests/test_topic4_mz_slowvars.py), never tuned on the z+m traces being classified
(anti-circularity, mirrors src/sef_hfo_m4_termination.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Baseline reference (from slow-off, same seed) + gates.
# ---------------------------------------------------------------------------
@dataclass
class MZBaseline:
    n_events: int          # # returning interictal events in slow-off (baseline-anchor gate)
    dur_med: float         # median baseline event duration (ms)
    dur_hi: float          # P90 baseline event duration (ms)
    part_lo: float         # P10 baseline event participation (peak active fraction)
    part_hi: float         # P90 baseline event participation
    act_lo: float          # P10 baseline event peak rate (Hz)
    act_hi: float          # P90 baseline event peak rate (Hz)
    floor_af: float        # baseline active-fraction floor (P95 of af in quiet window)
    baseline_rate: float   # mean E rate in quiet window (Hz) -- recovery band anchor
    sigma_rate: float      # std E rate in quiet window (Hz)


@dataclass
class MZPhenotypeGates:
    min_base_events: int = 3     # slow-off < this returning events -> insufficient (per seed)
    f_dur: float = 1.0           # expanded requires peak_dur > f_dur * dur_hi
    f_part: float = 1.0          # AND peak_participation > f_part * part_hi
    f_act: float = 1.0           # AND peak_rate > f_act * act_hi
    f_suppress: float = 1.0      # peak_participation < f_suppress * part_lo -> suppress (activity killed)
    f_frag: float = 2.0          # n_events >= f_frag * base_n AND short -> fragment
    recovery_m: float = 1.5      # returned: post-event mean rate <= baseline + recovery_m*sigma (runner use)
    t_return: float = 120.0      # ms recovery window (runner use)
    recover_offset: float = 30.0 # ms settle after t_off before the recovery window (avoid trailing-edge leak)


def classify_mz_run(rm: dict, baseline: MZBaseline, runaway_ms, gates: MZPhenotypeGates | None = None) -> str:
    """Deterministic 7-label phenotype from a run-metrics dict + slow-off baseline reference.

    rm keys: n_events, peak_dur, peak_participation, peak_rate, peak_returned(bool), max_dur, peak_af.
    runaway_ms: from run_m4_dynamic_qi._first_sustained (or engine early-stop); not None -> runaway.

    Decision tree (design §8; gate STRUCTURE is the contract, factors are calibration):
      1. runaway_ms is not None                                   -> 'runaway'   (injected, shape-independent)
      2. baseline.n_events < min_base_events                      -> 'insufficient' (no baseline phase, per seed)
      3. expanded == (peak_dur AND peak_participation AND peak_rate all above baseline P90):
           returned  -> 'expanded_returned'   else -> 'expanded_bounded'
      4. not expanded:
           n_events==0 or peak_participation < f_suppress*part_lo -> 'suppress' (activity killed)
           n_events >= f_frag*base_n and max_dur < dur_med        -> 'fragment' (many short bursts)
           otherwise                                              -> 'interictal_like'
    """
    g = gates or MZPhenotypeGates()
    if runaway_ms is not None:
        return "runaway"
    if baseline.n_events < g.min_base_events:
        return "insufficient"
    expanded = (rm["peak_dur"] > g.f_dur * baseline.dur_hi
                and rm["peak_participation"] > g.f_part * baseline.part_hi
                and rm["peak_rate"] > g.f_act * baseline.act_hi)
    if expanded:
        return "expanded_returned" if rm["peak_returned"] else "expanded_bounded"
    if rm["n_events"] == 0 or rm["peak_participation"] < g.f_suppress * baseline.part_lo:
        return "suppress"
    if rm["n_events"] >= g.f_frag * baseline.n_events and rm["max_dur"] < baseline.dur_med:
        return "fragment"
    return "interictal_like"


# ---------------------------------------------------------------------------
# Calibration helpers (slow-off baseline ONLY; never from z+m results -- design §6).
# ---------------------------------------------------------------------------
def pooled_quantiles_from_hist(hist, edges, qs):
    """Quantiles of a distribution summarized by a fixed-bin histogram (bin count `hist`, bin `edges`).
    Returns {q: value_at_bin_center}. NaN for every q if the histogram is empty. Used to pool the
    observer's per-step E-cell I_I / I_E histograms over event steps (design §6.3)."""
    hist = np.asarray(hist, float)
    edges = np.asarray(edges, float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    total = hist.sum()
    if total <= 0:
        return {q: float("nan") for q in qs}
    cum = np.cumsum(hist) / total
    out = {}
    for q in qs:
        i = int(np.searchsorted(cum, q, side="left"))
        out[q] = float(centers[min(i, len(centers) - 1)])
    return out


def replay_adaptation_peak(E_spk_bool, dt, tau_adp, event_step_mask=None):
    """Offline replay of the m ODE over a slow-off E spike raster (design §6.6). Integrates
    m_i (decay -m/tau_adp, +1 per E spike) exactly as MZSlowVars.step, and returns per-E-cell PEAK m
    within `event_step_mask` (or over all steps). m accumulation uses the FULL history (correct
    inter-event decay); only the peak is restricted to event steps."""
    E = np.asarray(E_spk_bool)
    nsteps, NE = E.shape
    m = np.zeros(NE)
    peak = np.zeros(NE)
    for t in range(nsteps):
        m -= (m / tau_adp) * dt          # decay first (matches MZSlowVars.step)
        sp = E[t]
        if sp.any():
            m[sp] += 1.0                  # E spike increments
        if event_step_mask is None or event_step_mask[t]:
            np.maximum(peak, m, out=peak)
    return peak


def eta_m_from_frac(frac, I_EE_scale, peak_m):
    """eta_m so the baseline event adaptation-current peak (eta_m * peak_m) is `frac` of the
    excitatory current scale I_EE_scale (design §6.6). frac in {0.05, 0.10, 0.20} -> low/mid/high."""
    return float(frac) * float(I_EE_scale) / float(peak_m)


def select_by_targets(values, targets):
    """Indices of the `values` closest to each target level (design §7 arm-C pre-registered
    weak/mid/strong selection). E.g. realized z-depletion per cell vs targets [0.8,0.5,0.2]."""
    vals = np.asarray(values, float)
    return [int(np.argmin(np.abs(vals - t))) for t in targets]
