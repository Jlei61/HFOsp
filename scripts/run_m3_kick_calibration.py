#!/usr/bin/env python3
"""M3 Task 1.5: kick amplitude + one-generation window calibration.

SKELETON — infrastructure only. DO NOT run the calibration sweep until
src/topic4_propagation_operator.py (Task 1) is available.

Purpose
-------
Sweep KICK_BOOST × win_ms over 3–5 representative spatial bins to select:
  (a) quasi-linear local regime: response grows ~ linearly with KICK_BOOST,
      small enough that the kick does NOT directly trigger a self-sustained wave.
  (b) first-recruitment one-generation window [Δ1, Δ2]: the response window
      that captures the first downstream generation (beyond the direct-stimulation
      artifact) without including a self-sustained second wave.

The calibrated (kick_boost, [Δ1, Δ2]) are written to
  results/topic4_sef_hfo/m3_local_w/kick_calibration/kick_calibration.json
and consumed by Task 4 (preregistration freeze).

Usage (once Task 1 is available)
---------------------------------
  python3 scripts/run_m3_kick_calibration.py \\
      --L 20 --density 100 --seed 1 \\
      --n-bins-per-axis 4 --n-rep-bins 4 --seeds 3 \\
      --kick-boosts 0.5 1.0 2.0 4.0 \\
      --win-ms 2,6 4,10 8,16 12,24 \\
      --out-dir results/topic4_sef_hfo/m3_local_w/kick_calibration
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

# Reuse the locked self-termination constants from src/sef_hfo_events.py (the
# active-fraction event detector). RETURN_FRAC / RUNAWAY_FRAC / FRAC_TIME_ON_MAX /
# SETTLE_MS are the SAME IDEA here: judge whether the KICK RUN'S OWN activity
# self-terminates (returned) vs runs away / stays "mostly on" — NOT a kick-minus-
# sham late difference (which in a chaotic net measures two trajectories drifting
# apart, backwards in kick strength). We import the constants only (no runner code).
from src.sef_hfo_events import RETURN_FRAC, RUNAWAY_FRAC, FRAC_TIME_ON_MAX, SETTLE_MS

# Active-fraction trace bin width (ms) — fine enough to resolve a single-generation
# response within DUR_KICK..T_end, mirrors active_fraction(bin_ms=..) concept in
# scripts/run_sef_hfo_snn_cm_spontaneous_readout.py.
TRACE_BIN_MS: float = 2.0

# --- Selector floors / caps (module-level so the selector test imports them) ---
# DRAFT — to be tuned on the L=20 explore pilot. The LOGIC (min-kick + LOCAL +
# RETURNED + non-global) is the load-bearing part; these magnitudes are placeholders.
SOURCE_FLOOR: float = 2.0       # DRAFT — to be tuned on the L=20 explore pilot. kick must ignite locally
DOWNSTREAM_FLOOR: float = 2.0   # DRAFT — to be tuned on the L=20 explore pilot. there IS a first-gen response

# --- core_only confound gate: RELATIVE to the bare-sheet background ----------
# The bare sheet itself spits out spontaneous spikes per window (e.g. ~25 downstream
# at drive=0.6), so an ABSOLUTE "core_only < 2 spikes" floor is wrong: every candidate
# (even a barely-a-core mean=18.0) trips it and dies at pass_core_quiet before reaching
# the local/return gates. The confound we actually want to catch is the core producing
# MATERIALLY MORE activity than the bare sheet (extra self-ignition), so the gate
# compares core_only against its OWN paired bare-sheet run (no_core_no_kick): a quiet
# core is one whose core_only response is NOT much above the bare-sheet background.
# Real data: narrow cores have core_only_downstream≈bare (ratio~1.0 -> quiet PASS); the
# WIDE core has core_only_downstream≫bare (ratio 3–18 -> quiet FAIL, genuinely igniting).
CORE_BG_RATIO: float = 1.5      # DRAFT — to be tuned. core_only <= RATIO*bare + MARGIN to be "quiet"
CORE_BG_MARGIN: float = 5.0     # DRAFT — to be tuned. additive slack on the bare-background bar
BINS_CAP_FRAC: float = 0.5      # DRAFT — to be tuned on the L=20 explore pilot. n_activated_bins <= BINS_CAP_FRAC*n_bins (LOCAL, not global wave)
FAR_FRAC_CAP: float = 0.5       # DRAFT — to be tuned on the L=20 explore pilot. far_field_frac <= this (response not dumped far away)
R95_CAP_FRAC: float = 0.30      # DRAFT — to be tuned on the L=20 explore pilot. r95_mm <= R95_CAP_FRAC * L_mm (a local response stays within ~0.3*L of the kick)
R95_CAP_MM_FALLBACK: float = 6.0  # fallback r95 cap (mm) when L is not threaded (selector unit tests omit L)

# Per-seed / per-rep-bin aggregation fractions (DRAFT). An instrument parameter must
# not be pulled over the line by one position or half the seeds, so the gate is
# fraction-of-seeds-local AND fraction-of-bins-local, not mean-over-everything.
SEED_PASS_FRAC: float = 2.0 / 3.0  # DRAFT — fraction of seeds that must be locally-returned per rep bin
BIN_PASS_FRAC: float = 0.5         # DRAFT — fraction of rep bins that must pass for a (boost, win) to qualify

# --- EVENT-ALIGNED response window (B-branch finite-event recruitment operator) ---
# The FIXED window (e.g. [22,32] ms after t_kick) measures the response at a fixed delay.
# A finite event's onset latency varies with kick / core / seed, so the fixed window
# mixes events whose first-generation recruitment starts at different times. The
# EVENT-ALIGNED window measures the SAME metrics relative to the EVENT ONSET t0 (the
# moment the differenced downstream response first crosses an onset bar), so W_event is
# an event RECRUITMENT operator, not a fixed-delay response. This is PURELY ADDITIVE:
# the fixed-window metrics + the selector gate are unchanged; the *_ea fields are
# reported ALONGSIDE for comparison (agreement of fixed vs event-aligned spatial-local
# structure => high confidence). These constants are DRAFT.
ONSET_Z: float = 3.0          # DRAFT — to be tuned on the L=20 explore pilot. onset bar = baseline_mean + ONSET_Z*baseline_std
ONSET_MIN_MASS: float = 3.0   # DRAFT — to be tuned on the L=20 explore pilot. onset also requires down_diff > this absolute mass
EA_DELTA1: float = 0.0        # DRAFT — to be tuned on the L=20 explore pilot. event-aligned first-gen window start = t0 + EA_DELTA1
EA_DELTA2: float = 10.0       # DRAFT — to be tuned on the L=20 explore pilot. event-aligned first-gen window end = t0 + EA_DELTA2 (≈ fixed window width)


def _parse_win_ms(raw: list[str]) -> list[tuple[float, float]]:
    """Parse '2,6' → (2.0, 6.0)."""
    result = []
    for s in raw:
        parts = s.split(",")
        if len(parts) != 2:
            raise ValueError(f"win_ms must be 'lo,hi', got: {s!r}")
        result.append((float(parts[0]), float(parts[1])))
    return result


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="M3 Task 1.5: kick amplitude + one-generation window calibration"
    )
    # Network parameters
    p.add_argument("--L", type=float, default=20.0, help="Network side length (mm)")
    p.add_argument("--density", type=float, default=100.0, help="Neuron density")
    p.add_argument("--seed", type=int, default=1, help="Network construction seed")
    p.add_argument("--theta-ee", type=float, default=45.0, help="E→E elongation angle (deg)")
    p.add_argument("--AR", type=float, default=2.0, help="E→E anisotropy ratio")
    p.add_argument("--T", type=float, default=500.0, help="Simulation duration (ms)")
    p.add_argument("--dt", type=float, default=0.1, help="Time step (ms)")
    p.add_argument("--nu-ext-ratio", type=float, default=0.6, help="External drive ratio")
    p.add_argument("--vth0", type=float, default=18.0, help="Baseline threshold (mV)")
    # Calibration sweep parameters
    p.add_argument("--n-bins-per-axis", type=int, default=4,
                   help="Spatial bins per axis for W_resp")
    p.add_argument("--n-rep-bins", type=int, default=4,
                   help="Number of representative bins to probe (center + periphery)")
    p.add_argument("--seeds", type=int, default=3,
                   help="Number of SNN seeds per (bin, kick_boost, win_ms) combo")
    p.add_argument("--kick-boosts", type=float, nargs="+",
                   default=[0.5, 1.0, 2.0, 4.0],
                   help="KICK_BOOST values to sweep (extra external Poisson rate, 1/ms; "
                        "quasi-linear range ~1–3)")
    p.add_argument("--win-ms", type=str, nargs="+",
                   default=["18,24", "18,28", "20,30", "22,34",
                            "2,6", "4,10"],
                   help=(
                       "Response window candidates as 'lo,hi' pairs (ms, relative to t_kick). "
                       "Defaults include post-DUR_KICK windows (lo>=18=DUR_KICK) for "
                       "artifact-free first-gen measurement, plus two during-kick windows "
                       "for contrast (flagged window_after_dur_kick=False). "
                       "Selection requires window_after_dur_kick=True."
                   ))
    p.add_argument("--r-kick", type=float, default=0.5,
                   help="Kick radius (mm; local injection)")
    p.add_argument("--t-kick", type=float, default=100.0,
                   help="Kick time within simulation (ms)")
    # Heterogeneous-core mode (M3-1.5 core path). Core mode is ON iff --core-mean given.
    p.add_argument("--core-mean", type=float, default=None,
                   help="Mean E threshold (mV) of a SMALL PATHOLOGICAL CORE (low-threshold "
                        "disk). None (default) = BARE SHEET / no core (legacy kick-vs-sham). "
                        "Setting it turns on the 4-condition core path: the kick lands at the "
                        "core center, the differenced core_kick - core_only measures the LOCAL "
                        "returned first-generation response.")
    p.add_argument("--core-std", type=float, default=0.5,
                   help="Std of the core E-threshold distribution (mV; core mode only)")
    # Static-μ permissivity (spec m3_static_mu_pilot). μ=0 (default) = bit-parity.
    p.add_argument("--mu", type=float, default=0.0,
                   help="Static slow-permissivity μ in [0,1]: V_th_eff = vth_core - dvth_at_mu1*μ*h. "
                        "0 (default) = no change = byte-identical to the pre-μ runner.")
    p.add_argument("--dvth-at-mu1", type=float, default=1.333,
                   help="Max extra threshold depression (mV) at μ=1; ΔVth(μ)=dvth_at_mu1*μ. "
                        "Default 1.333 so the μ grid {0..0.9} maps to {0..1.2} mV.")
    p.add_argument("--h-mode", choices=["core_susceptibility", "uniform", "shuffled"],
                   default="core_susceptibility",
                   help="Susceptibility field μ couples to: core_susceptibility (the埋下 pathology, "
                        "default), uniform (global-μ control), shuffled (location-broken control).")
    p.add_argument("--core-r", type=float, default=1.5,
                   help="Core disk radius (mm; core mode only)")
    p.add_argument("--core-center-xy", type=float, nargs=2, default=None,
                   help="Core center (x y) in mm. Default = geometric center of the sheet. "
                        "Core mode kicks at this center (kick_center = core_center).")
    p.add_argument("--kick-xy", type=float, nargs=2, default=None,
                   help="mini-W_event multi-source: place the kick at this (x y) in mm "
                        "while the core field STAYS at core_center. None (default) = kick at "
                        "core_center (core mode) / rep bin (bare) — bit-identical to before. "
                        "When set, the run probes ONE source bin (the bin nearest (x,y)).")
    p.add_argument("--emit-ea-bins", action="store_true",
                   help="mini-W_event: also write ea_net_bins.npz (per (kick,seed) the "
                        "per-bin event-aligned differenced response — the W_shape ingredient "
                        "the scalar artifacts drop). Default OFF: existing artifacts are "
                        "byte-identical. Requires --kick-xy (single source per invocation).")
    p.add_argument("--emit-ea-aux", action="store_true",
                   help="mini-W_event B1c-rescue: ALSO write ea_aux_bins.npz next to "
                        "ea_net_bins.npz — per (kick,seed,bin) the core_only (sham) EA-window "
                        "count (a 'local rate' predictor) AND the first-spike onset time in ms "
                        "(true recruitment order, replacing the early-response-RANK proxy). "
                        "Default OFF: existing artifacts (incl. ea_net_bins.npz) byte-identical. "
                        "Requires --emit-ea-bins. Does NOT run the dense per-bin K_min "
                        "susceptibility sweep (a separate, costlier source sweep — next step).")
    p.add_argument("--no-event-figure", action="store_true",
                   help="skip the per-run event-diagnostic figure (raster + early per-bin "
                        "heatmap + return-to-quiet). Default: the figure IS produced for every "
                        "run (re-sims ONE representative seed with the built network, ~cheap).")
    p.add_argument("--event-figure-only", action="store_true",
                   help="BACKFILL: skip the sweep; rebuild the network, read the representative "
                        "kick from the existing --out-dir/per_seed_metrics.csv, and (re)write "
                        "only figures/event_diagnostic.png. Non-destructive (does not touch the "
                        "sweep artifacts). Use to add the event figure to a completed run.")
    p.add_argument("--far-radius-mm", type=float, default=None,
                   help="Far-field radius (mm) for far_field_frac (LOCAL vs global wave). "
                        "Default = 0.35*L. Response mass beyond this radius from the kick "
                        "center counts as far-field (a global-wave signature).")
    # Output
    p.add_argument("--out-dir", type=str,
                   default="results/topic4_sef_hfo/m3_local_w/kick_calibration",
                   help="Output directory for calibration JSON + figures")
    p.add_argument("--run", action="store_true",
                   help="Actually execute the calibration sweep. OFF by default: the "
                        "sweep is a measurement that feeds the frozen pre-registration "
                        "(Task 1.5, pilot-gated) and must not run accidentally.")
    p.add_argument("--mode", choices=["explore", "strict"], default="explore",
                   help="explore (default, current L20-explore phase): on a no-candidate "
                        "sweep NEVER RuntimeError — write the full diagnostic artifact and "
                        "exit 0 with a printed GO/NO_GO summary. strict (post-prereg): keep "
                        "the GO/NO-GO RuntimeError when no candidate qualifies.")
    return p


def _select_rep_bins(bin_centers: np.ndarray, n: int, rng: np.random.Generator) -> list[int]:
    """Select n representative bin indices: center bin + periphery sample."""
    n_bins = len(bin_centers)
    # Always include the spatial center
    center = bin_centers.mean(axis=0)
    dists = np.linalg.norm(bin_centers - center, axis=1)
    center_idx = int(np.argmin(dists))
    selected = {center_idx}
    # Fill remaining from periphery (farthest bins)
    sorted_by_dist = np.argsort(dists)[::-1]
    for idx in sorted_by_dist:
        if len(selected) >= n:
            break
        selected.add(int(idx))
    return sorted(selected)


def _bin_spike_counts_in_window(res, bin_of_cell, n_bins, t_lo, t_hi, dt):
    """Total E spikes per bin in [t_lo, t_hi) ms. Mirrors build_w_resp._bin_spike_counts."""
    lo_step = int(round(t_lo / dt))
    hi_step = int(round(t_hi / dt))
    per_cell = res["E_spk_bool"][lo_step:hi_step].sum(axis=0).astype(float)
    return np.bincount(bin_of_cell, weights=per_cell, minlength=n_bins)


def _bin_first_onset_in_window(res, bin_of_cell, n_bins, t_lo, t_hi, dt):
    """Per-bin first-spike onset (ms relative to t_lo) of one cached run in [t_lo, t_hi).
    For each bin, the earliest spike over its E cells; NaN for bins with no spike in the
    window. Unlike ``_bin_spike_counts_in_window`` this does NOT sum over time — it keeps
    onset timing, so event order can use TRUE recruitment timing instead of the
    early-response-RANK proxy (B1c DATA_MISSING item 3). Pure read of cached spikes; no
    SNN re-run. Used only under --emit-ea-aux."""
    lo_step = int(round(t_lo / dt))
    hi_step = int(round(t_hi / dt))
    seg = res["E_spk_bool"][lo_step:hi_step]                 # (n_step_window, NE) bool
    if seg.shape[0] == 0:
        return np.full(n_bins, np.nan)
    fired = seg.any(axis=0)                                  # cells with >=1 spike in window
    if not fired.any():
        return np.full(n_bins, np.nan)
    first_ms = seg.argmax(axis=0).astype(float) * dt        # first-spike ms rel window start
    onset = np.full(n_bins, np.inf)
    np.minimum.at(onset, bin_of_cell[fired], first_ms[fired])
    onset[np.isinf(onset)] = np.nan
    return onset


def _active_fraction_trace(E_spk_bool: np.ndarray, dt: float, bin_ms: float) -> np.ndarray:
    """Fraction of E cells with >=1 spike per ~bin_ms bin (mirrors the concept of
    active_fraction() in run_sef_hfo_snn_cm_spontaneous_readout.py). Returns a 1-D
    trace, one value per bin: trace[k] = mean over E cells of (any spike in bin k)."""
    bs = max(1, int(round(bin_ms / dt)))
    nb = E_spk_bool.shape[0] // bs
    if nb == 0:
        return np.zeros(0)
    binned = E_spk_bool[:nb * bs].reshape(nb, bs, -1).any(axis=1)  # (nb, NE) bool
    return binned.mean(axis=1)


def _trace_window_mean(trace: np.ndarray, dt: float, bin_ms: float,
                       t_lo: float, t_hi: float):
    """Mean (and std) of the active-fraction trace over absolute time [t_lo, t_hi) ms."""
    lo_b = int(np.floor(t_lo / bin_ms))
    hi_b = int(np.ceil(t_hi / bin_ms))
    lo_b = max(0, min(lo_b, len(trace)))
    hi_b = max(lo_b, min(hi_b, len(trace)))
    seg = trace[lo_b:hi_b]
    if seg.size == 0:
        return 0.0, 0.0
    return float(seg.mean()), float(seg.std())


def _trace_run_properties(E_spk_bool: np.ndarray, dt: float, t_kick: float,
                          T_end: float, dur_kick: float) -> dict:
    """Run-property metrics from ONE run's OWN active-fraction trace (returned /
    runaway / frac_time_on_post + the relative self-baseline bar). Factored out of
    the old inline per-seed body so it can be applied to the RAW core_kick run
    (the returned/runaway gate) AND to the RAW core_only run (the confound gate) —
    each judged on its OWN activity (NOT the difference; differencing two
    persistently-active runs can fall to 0 and FAKE 'returned')."""
    trace = _active_fraction_trace(E_spk_bool, dt, TRACE_BIN_MS)
    baseline, baseline_std = _trace_window_mean(
        trace, dt, TRACE_BIN_MS, t_kick - 50.0, t_kick)
    post_lo_b = int(np.floor(t_kick / TRACE_BIN_MS))
    post = trace[post_lo_b:] if trace.size else np.zeros(0)
    peak = float(post.max()) if post.size else 0.0

    rel_bar = baseline + 3.0 * baseline_std  # relative self-baseline "on" bar
    settle_mean, _ = _trace_window_mean(
        trace, dt, TRACE_BIN_MS, max(t_kick, T_end - SETTLE_MS), T_end)
    return_thresh = max(RETURN_FRAC * peak, rel_bar)
    returned = bool(settle_mean <= return_thresh)
    end_val = float(trace[-1]) if trace.size else 0.0
    runaway = bool((not returned) and (end_val >= RUNAWAY_FRAC * peak))

    # frac of [t_kick+dur_kick, T_end] where trace > relative self-baseline bar
    on_lo_b = int(np.floor((t_kick + dur_kick) / TRACE_BIN_MS))
    on_lo_b = max(0, min(on_lo_b, len(trace)))
    post_seg = trace[on_lo_b:]
    frac_on_post = float((post_seg > rel_bar).mean()) if post_seg.size else 0.0

    return {
        "trace": trace,
        "rel_bar": rel_bar,
        "returned": returned,
        "runaway": runaway,
        "frac_time_on_post": frac_on_post,
    }


def _trace_has_event_in(trace: np.ndarray, rel_bar: float,
                        t_lo: float, t_hi: float) -> bool:
    """True iff ANY bin of the active-fraction ``trace`` inside [t_lo, t_hi) ms is
    above the run's own relative self-baseline bar (``baseline + 3*std``) — i.e. an
    active-fraction event is present in that interval. Reuses the SAME relative-bar
    logic as the returned/runaway gate. For the core_only confound: an event
    overlapping [t_kick, t_kick + win_hi] means the core's OWN spontaneous activity
    contaminates the measurement window -> NOT quiet."""
    lo_b = int(np.floor(t_lo / TRACE_BIN_MS))
    hi_b = int(np.ceil(t_hi / TRACE_BIN_MS))
    lo_b = max(0, min(lo_b, len(trace)))
    hi_b = max(lo_b, min(hi_b, len(trace)))
    seg = trace[lo_b:hi_b]
    if seg.size == 0:
        return False
    return bool((seg > rel_bar).any())


def _core_only_quiet(co_src: float, co_downstream: float, co_frac_on_post: float,
                     co_event_in_win: bool, nc_src: float, nc_downstream: float) -> bool:
    """RELATIVE-to-bare-background confound verdict for one core_only run.

    A core is "quiet" iff its core_only window activity is NOT materially above the
    PAIRED bare sheet (no_core_no_kick) — i.e. it does not self-ignite extra spikes:
      - core_only_downstream <= CORE_BG_RATIO * bare_downstream + CORE_BG_MARGIN
      - core_only_source     <= CORE_BG_RATIO * bare_source     + CORE_BG_MARGIN
      - frac_time_on_post <= FRAC_TIME_ON_MAX  (relative "mostly on" check, kept)
      - NOT core_only_event_in_win             (discrete-event-in-window check, kept)

    The two ratio checks REPLACE the old absolute floors (< 2 spikes), which sat below
    the bare sheet's own spontaneous background (~25 downstream spikes/window) and so
    falsely flagged EVERY core as confounded. Single source of truth shared by the
    runner (_measure_response) and the offline reclassify script."""
    return bool(
        (co_downstream <= CORE_BG_RATIO * nc_downstream + CORE_BG_MARGIN) and
        (co_src <= CORE_BG_RATIO * nc_src + CORE_BG_MARGIN) and
        (co_frac_on_post <= FRAC_TIME_ON_MAX) and
        (not co_event_in_win)
    )


def _spatial_extent(net_bins: np.ndarray, bin_centers: np.ndarray, src_bin_idx: int,
                    far_radius_mm: float):
    """Self-contained reimplementation of the extent concept (mirrors the r95/reach
    idea of event_field_geometry()/`_extent` in the spontaneous readout script):
    from the early-window per-bin net response (kick - sham, clipped >=0), measure
    how LOCAL the response is relative to the kicked source bin.

    The source bin is EXCLUDED from every metric: under strong direct stimulation
    its mass dwarfs the genuine downstream propagation, so including it inflates the
    activation floor and dilutes the far-field denominator — both make a non-local
    response masquerade as local. All three metrics are computed over NON-source bins.

    Returns (n_activated_bins, r95_mm, far_field_frac):
      n_activated_bins : count of NON-source bins with net_bins > activation_floor
                         (floor = max(2, 0.05*max NON-source mass))
      r95_mm           : response-weighted 95th-pct of |bin_center - kick_center|
                         over activated (NON-source) bins (0 if none)
      far_field_frac   : sum(NON-source mass beyond far_radius_mm)
                         / sum(NON-source mass) (0 if total 0)
    """
    kick_center = bin_centers[src_bin_idx]
    radii = np.linalg.norm(bin_centers - kick_center, axis=1)  # per-bin radius (mm)

    non_source = np.ones(len(net_bins), dtype=bool)
    non_source[src_bin_idx] = False

    # Floor from NON-SOURCE bins only. Using net_bins.max() includes the kicked
    # source bin (huge under strong direct stimulation), which inflates the floor so
    # that every downstream bin falls below it -> the response falsely looks local.
    nonsrc = net_bins[non_source]
    activation_floor = max(2.0, 0.05 * float(nonsrc.max() if nonsrc.size else 0.0))
    activated = non_source & (net_bins > activation_floor)
    n_activated_bins = int(activated.sum())

    if n_activated_bins > 0:
        act_radii = radii[activated]
        act_weights = net_bins[activated]
        # response-weighted 95th percentile of activated-bin radii
        order = np.argsort(act_radii)
        r_sorted = act_radii[order]
        w_sorted = act_weights[order]
        cw = np.cumsum(w_sorted)
        if cw[-1] > 0:
            r95_mm = float(np.interp(0.95 * cw[-1], cw, r_sorted))
        else:
            r95_mm = float(np.percentile(act_radii, 95))
    else:
        r95_mm = 0.0

    # Far-field fraction over NON-SOURCE bins only. Including the source bin in both
    # numerator and denominator lets the huge direct-stimulation mass dominate the
    # denominator, diluting the far-field fraction so the response falsely looks local.
    far_mask = non_source & (radii > far_radius_mm)
    total_ns = float(net_bins[non_source].sum())
    far_ns = float(net_bins[far_mask].sum())
    far_field_frac = far_ns / total_ns if total_ns > 0 else 0.0

    return n_activated_bins, r95_mm, far_field_frac


# --- EVENT-ALIGNED window helpers (pure, engine-free, unit-testable) ------------
# These factor the onset-detection + event-aligned slicing logic out of the per-seed
# body so they can be tested WITHOUT the engine. The trace they consume (down_diff)
# is built by _downstream_diff_trace below (which DOES touch cached spikes).

def _event_onset(down_diff: np.ndarray, bin_ms: float, t_kick: float,
                 dur_kick: float, baseline_lo: float, baseline_hi: float):
    """Find the EVENT ONSET t0 from a per-time-bin DIFFERENCED downstream trace.

    ``down_diff[k]`` is the outward differenced response (core_kick − core_only, clipped
    >=0, summed over NON-source spatial bins) in the k-th time bin of width ``bin_ms``,
    indexed from t=0 of the run. t0 is the first time-bin AT OR AFTER ``t_kick+dur_kick``
    (no direct-stim contamination) where ``down_diff`` exceeds BOTH:
      - a RELATIVE bar: baseline_mean + ONSET_Z * baseline_std, where baseline is
        ``down_diff`` over the QUIET pre-kick window [baseline_lo, baseline_hi) ms, AND
      - an ABSOLUTE floor: ONSET_MIN_MASS.

    Returns (t0_ms, detected): t0_ms is the absolute time (ms) of the onset bin's left
    edge; detected is True iff a crossing was found. If ``down_diff`` never crosses (no
    event), returns (nan, False) — the caller marks event-aligned metrics absent.
    """
    n = len(down_diff)
    if n == 0:
        return float("nan"), False
    base_lo_b = max(0, min(int(np.floor(baseline_lo / bin_ms)), n))
    base_hi_b = max(base_lo_b, min(int(np.ceil(baseline_hi / bin_ms)), n))
    base_seg = down_diff[base_lo_b:base_hi_b]
    if base_seg.size:
        base_mean = float(base_seg.mean())
        base_std = float(base_seg.std())
    else:
        base_mean = 0.0
        base_std = 0.0
    onset_bar = max(base_mean + ONSET_Z * base_std, ONSET_MIN_MASS)
    # First candidate bin AT OR AFTER t_kick + dur_kick (artifact-free).
    search_lo_b = max(0, int(np.floor((t_kick + dur_kick) / bin_ms)))
    for k in range(search_lo_b, n):
        if down_diff[k] > onset_bar:
            return float(k * bin_ms), True
    return float("nan"), False


def _event_aligned_window(t0_ms: float, delta1: float, delta2: float):
    """Absolute [lo, hi) ms window for the event-aligned first-gen response, relative
    to the event onset t0: [t0 + delta1, t0 + delta2). Pure helper so the slice the
    event-aligned metrics are computed over is unit-testable from a known t0."""
    return float(t0_ms + delta1), float(t0_ms + delta2)


def _downstream_diff_trace(res_kick, res_only, bin_of_cell, n_bins, src_bin_idx,
                           dt, bin_ms, T_end) -> np.ndarray:
    """Per-time-bin DIFFERENCED source-excluded downstream response trace over the WHOLE
    run [0, T_end). For each fine time bin of width ~``bin_ms`` (TRACE_BIN_MS), compute
    per-spatial-bin (core_kick − core_only) spike counts, clip >=0, then SUM over the
    NON-source spatial bins. Reuses _bin_spike_counts_in_window per fine time bin (same
    differencing + source-exclusion contract as the fixed-window primary). The source
    bin is excluded so the huge direct-stimulation footprint does not swamp the onset
    of the genuine outward propagation.

    The trace starts at t=0 (NOT t_kick) so its bin index k maps to absolute time
    k*bin_ms — _event_onset reads its QUIET pre-kick baseline from the leading bins and
    indexes candidate onset bins by absolute time directly."""
    if T_end <= 0:
        return np.zeros(0)
    nb = int(np.floor(T_end / bin_ms))
    trace = np.zeros(nb)
    for k in range(nb):
        t_lo = k * bin_ms
        t_hi = t_lo + bin_ms
        bins_ck = _bin_spike_counts_in_window(res_kick, bin_of_cell, n_bins,
                                              t_lo, t_hi, dt)
        bins_co = _bin_spike_counts_in_window(res_only, bin_of_cell, n_bins,
                                              t_lo, t_hi, dt)
        net = np.clip(bins_ck - bins_co, 0.0, np.inf)
        trace[k] = float(net.sum() - net[src_bin_idx])
    return trace


# --- The four (or two) paired conditions (CORE mode contract) -------------------
# Per (kick_boost, seed) we run FOUR conditions, ALL on the SAME rng seed so they are
# paired (a difference between two is the kick / core contribution, not seed drift):
#   core_kick       = core threshold field  + KICK_BOOST=kick
#   core_only       = core threshold field  + KICK_BOOST=0
#   kick_only       = uniform-vth0 field     + KICK_BOOST=kick
#   no_core_no_kick = uniform-vth0 field     + KICK_BOOST=0
# In BARE-SHEET mode (core_mean is None) the field is uniform-vth0 everywhere, so
# core_kick == kick_only (the kick run) and core_only == no_core_no_kick (the sham
# run) by construction — the legacy "kick vs sham" path, numbers UNCHANGED.
_CONDITIONS = ("core_kick", "core_only", "kick_only", "no_core_no_kick")


def _resolve_kick_center_and_src(kick_xy, core_center, core_mode: bool,
                                 bin_centers: np.ndarray, bin_idx: int):
    """Resolve (kick_center, src_bin_idx) for one rep bin (mini-W_event multi-source).

    kick_xy=None reproduces the EXISTING behavior exactly (bit-parity): core mode
    kicks at core_center with source = the rep bin_idx; bare mode kicks at this rep
    bin's center. When kick_xy is set, the kick center AND the excluded source bin
    move to (x,y) together (source bin = nearest bin to the off-grid kick); the core
    field — computed elsewhere at core_center — is unaffected.
    """
    if kick_xy is not None:
        kc = np.asarray(kick_xy, dtype=float)
        src = int(np.argmin(np.linalg.norm(bin_centers - kc[None, :], axis=1)))
        return kc, src
    kc = core_center if core_mode else bin_centers[bin_idx]
    return np.asarray(kc, dtype=float), int(bin_idx)


def _run_conditions(p, net, NE, NI, kick_boost: float, bin_center: np.ndarray,
                    vth_core: np.ndarray, vth_uniform: np.ndarray,
                    seeds: int, r_kick: float, t_kick: float,
                    simulate_kick, seed_indices=None) -> list[dict]:
    """Run the FOUR paired conditions ONCE per seed and CACHE each run's E_spk_bool.
    Every candidate win_ms then SLICES these cached spikes (no re-simulation per
    window — efficiency restructure, M3-1.5 contract). Returns a list (length=seeds)
    of dict(condition -> res) where res is the raw simulate_kick output (carries
    E_spk_bool). The 4 conditions in each seed share np.random.default_rng(s+200) so
    they are paired (paired-seed contract). ``seed_indices`` (default range(seeds)) lets a
    caller re-simulate ONE specific seed index j (rng seed j+200) — used by the event figure
    to reproduce a chosen representative seed without re-running the whole batch."""
    runs = []
    for s in (range(seeds) if seed_indices is None else seed_indices):
        seed = s + 200
        conds = {}
        # (field, KICK_BOOST) per condition. Same rng seed across all four => paired.
        spec = {
            "core_kick": (vth_core, kick_boost),
            "core_only": (vth_core, 0.0),
            "kick_only": (vth_uniform, kick_boost),
            "no_core_no_kick": (vth_uniform, 0.0),
        }
        for name in _CONDITIONS:
            vth_field, boost = spec[name]
            net_c = dict(net)
            net_c["rng"] = np.random.default_rng(seed)
            conds[name] = simulate_kick(
                p, net_c, KICK_BOOST=boost,
                t_kick=t_kick, r_kick=r_kick,
                V_th_per_neuron=vth_field,
                kick_center=bin_center,
            )
        runs.append(conds)
    return runs


def _rep_kick_from_csv(csv_path: str, fallback: float) -> float:
    """Representative kick for the event figure read from an existing per_seed_metrics.csv:
    the kick with the most locally-returned seeds (max over windows, tie -> lowest kick)."""
    import csv as _csv
    if not os.path.exists(csv_path):
        return fallback

    def _truthy(v):
        s = str(v).strip().lower()
        if s in ("1", "true"):
            return True
        try:
            return float(s) > 0.5
        except ValueError:
            return False

    by: dict = {}
    with open(csv_path) as fh:
        for r in _csv.DictReader(fh):
            key = (float(r["kick_boost"]), r["win_lo"], r["win_hi"])
            by.setdefault(key, []).append(_truthy(r.get("seed_local_returned", 0)))
    best_pf: dict = {}
    for (k, _, _), vals in by.items():
        pf = sum(vals) / len(vals) if vals else 0.0
        best_pf[k] = max(best_pf.get(k, 0.0), pf)
    return max(best_pf, key=lambda k: (best_pf[k], -k)) if best_pf else fallback


def _emit_event_figure(p, net, NE, NI, args, core_mode, core_center, vth_core,
                       vth_uniform, bin_centers, bin_of_cell, n_bins, posE,
                       rep_bin_indices, rep_kick, out_dir, diag_status, simulate_kick,
                       dur_kick):
    """Re-simulate ONE representative (kick, seed) with the already-built network and render
    the 3-panel event-diagnostic figure. Shared by the normal run path and --event-figure-only.
    ``dur_kick`` is passed in (it is imported locally in run_calibration, not module-level).
    """
    from src.sef_hfo_event_figure import plot_event_diagnostic, median_representative
    from src.sef_hfo_mini_w_event import (_per_seed_ea, _per_seed_core_only,
                                          spontaneous_ignition_flag, success_seeds_at_kick)
    bidx = int(rep_bin_indices[0])
    kick_center, src_bin_idx = _resolve_kick_center_and_src(
        args.kick_xy, core_center, core_mode, bin_centers, bidx)

    # Representative SEED (P1-3): not fixed seed 0. From the existing per_seed_metrics.csv,
    # pick the seed at rep_kick that is EA-local-returned AND not spontaneous, with r95_ea
    # closest to the median (a TYPICAL event, no cherry-pick). Fall back to returned-only,
    # then seed 0, recording the gate in the title so the figure can't be over-read.
    by_kick = _per_seed_ea(out_dir)
    rep_seed, gate = 0, "no-event-fallback"
    if by_kick:
        rk = min(by_kick, key=lambda k: abs(k - rep_kick))
        recs = by_kick[rk]
        co_by_seed, bg_med = _per_seed_core_only(out_dir)
        spont = {int(s) for s, v in co_by_seed.items()
                 if spontaneous_ignition_flag(v, bg_med)}
        succ = success_seeds_at_kick(recs, spont)
        succ_recs = [r for r in recs if int(r["seed"]) in succ]
        if succ_recs:
            rep_seed = int(median_representative(
                [int(r["seed"]) for r in succ_recs], [r["r95_ea"] for r in succ_recs]))
            gate = "EA-local-returned"
        else:
            det = [r for r in recs if r.get("returned", 0) >= 1 and int(r["seed"]) not in spont]
            if det:
                rep_seed = int(median_representative(
                    [int(r["seed"]) for r in det], [r["r95_ea"] for r in det]))
                gate = "returned-only (NOT EA-local)"

    conds = _run_conditions(
        p, net, NE, NI, kick_boost=rep_kick, bin_center=kick_center,
        vth_core=vth_core, vth_uniform=vth_uniform, seeds=1,
        r_kick=args.r_kick, t_kick=args.t_kick, simulate_kick=simulate_kick,
        seed_indices=[rep_seed])[0]
    T_end = float(args.T)
    down_diff = _downstream_diff_trace(conds["core_kick"], conds["core_only"],
                                       bin_of_cell, n_bins, src_bin_idx, p.dt,
                                       TRACE_BIN_MS, T_end)
    t0_ms, ev = _event_onset(down_diff, TRACE_BIN_MS, args.t_kick, dur_kick,
                             baseline_lo=args.t_kick - 50.0, baseline_hi=args.t_kick)
    a_lo = (t0_ms + EA_DELTA1) if ev else (args.t_kick + dur_kick)
    a_hi = (t0_ms + EA_DELTA2) if ev else (args.t_kick + dur_kick + EA_DELTA2)
    ea_net_bins_fig = np.clip(
        _window_bins(conds["core_kick"], bin_of_cell, n_bins, a_lo, a_hi, p.dt)
        - _window_bins(conds["core_only"], bin_of_cell, n_bins, a_lo, a_hi, p.dt),
        0.0, np.inf)
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    sub = f"core{args.core_mean:g}" if core_mode else "bare"
    title = (f"event diagnostic — {sub}, kick={rep_kick:g}, seed={rep_seed} (gate: {gate}), "
             f"src@({kick_center[0]:.1f},{kick_center[1]:.1f})  [{diag_status}]")
    plot_event_diagnostic(
        conds["core_kick"]["E_spk_bool"], conds["core_only"]["E_spk_bool"],
        posE, ea_net_bins_fig, src_bin_idx, bin_centers, args.n_bins_per_axis, p.dt,
        args.t_kick, dur_kick, (t0_ms if ev else float("nan")), EA_DELTA1, EA_DELTA2,
        kick_center, os.path.join(fig_dir, "event_diagnostic.png"), title)
    print(f"Wrote event_diagnostic.png → {fig_dir}/event_diagnostic.png (kick={rep_kick:g})")


def _window_bins(res, bin_of_cell, n_bins, t_lo, t_hi, dt):
    """Per-bin E spike counts of one cached run in [t_lo, t_hi) ms (thin alias)."""
    return _bin_spike_counts_in_window(res, bin_of_cell, n_bins, t_lo, t_hi, dt)


def _measure_response(p, net, NE, NI, bin_center: np.ndarray, src_bin_idx: int,
                      bin_of_cell: np.ndarray, n_bins: int,
                      bin_centers: np.ndarray, far_radius_mm: float,
                      kick_boost: float,
                      win_ms: tuple[float, float], seeds: int,
                      vth0: float, r_kick: float, t_kick: float,
                      dur_kick: float, bins_cap: float, r95_cap: float,
                      runs: list[dict], core_mode: bool,
                      collect_aux: bool = False) -> dict:
    """Compute one (kick_boost, win_ms) record by SLICING the CACHED ``runs`` (the
    four paired conditions, simulated once per seed by _run_conditions). The metric
    sourcing is split EXACTLY as the M3-1.5 core contract dictates — do NOT collapse:

      DIFFERENCED  net_bins = clip(core_kick_bins(win) - core_only_bins(win), 0)
        -> source_resp / downstream_resp / n_activated_bins / r95_mm / far_field_frac.
        Differencing removes the core's OWN spontaneous activity; the source-bin
        exclusion in _spatial_extent still applies.

      RAW core_kick active-fraction trace
        -> returned / runaway / frac_time_on_post. NOT the difference: differencing
        two persistently-active runs can fall to 0 and FAKE "returned".

      RAW core_only (confound gate, CORE mode only)
        -> core_only_quiet = ALL of: core_only_downstream_resp <=
        CORE_BG_RATIO*no_core_no_kick_downstream + CORE_BG_MARGIN (the core does not
        produce materially MORE downstream activity than its paired bare sheet — no
        extra self-ignition) AND core_only_source_resp <=
        CORE_BG_RATIO*no_core_no_kick_source + CORE_BG_MARGIN (same, source bin) AND
        core_only_frac_time_on_post <= FRAC_TIME_ON_MAX (relative "mostly on" check) AND
        core_only has NO active-fraction event overlapping [t_kick, t_kick+win_hi].
        The two RELATIVE-to-bare-background ratio checks replace the old ABSOLUTE
        floors (< 2 spikes), which were below the bare sheet's own spontaneous
        background and so falsely killed every candidate. A confounded
        (core_mean, kick, win) can NEVER qualify (gated in _bin_pass).

      kick_only / no_core_no_kick are CONTROLS recorded in the record (their window
        response + the bare-sheet differenced kick_only - no_core_no_kick) but they
        do NOT enter the primary returned/local gate.

    Numeric fields are seed-means; booleans (returned/runaway) are seed-majority.
    In BARE-SHEET mode core_only_quiet defaults True (no confound gate).
    """
    lo, hi = win_ms
    T_end = float(getattr(p, "T", t_kick))

    # Differenced (primary) accumulators
    src_resps = []
    downstream_resps = []
    net_bins_seeds = []
    # Raw core_kick run-property accumulators
    returned_seeds = []
    runaway_seeds = []
    frac_on_post_seeds = []
    seed_local_returned_seeds = []
    # Raw core_only confound accumulators
    co_src_resps = []
    co_downstream_resps = []
    co_frac_on_post_seeds = []
    co_event_in_win_seeds = []
    co_quiet_seeds = []
    # kick_only / no_core_no_kick CONTROL accumulators (window response + bare diff)
    ko_downstream_resps = []
    nc_src_resps = []         # bare-sheet source-bin background (relative confound gate)
    nc_downstream_resps = []
    ko_minus_nc_seeds = []   # bare-sheet differenced kick_only - no_core_no_kick (downstream)
    # EVENT-ALIGNED accumulators (B-branch, reported ALONGSIDE the fixed window). Same
    # DIFFERENCED spatial metrics but measured in [t0+EA_DELTA1, t0+EA_DELTA2] relative
    # to the per-seed event onset t0 (NOT the fixed delay). returned/runaway are NOT
    # re-derived here — they are whole-run RAW core_kick properties, shared with the
    # fixed window. NaN where no event was detected (so seed-means ignore absent events).
    t0_ms_seeds = []
    event_detected_seeds = []
    downstream_resp_ea_seeds = []
    n_activated_bins_ea_seeds = []
    r95_mm_ea_seeds = []
    far_field_frac_ea_seeds = []
    # Per-bin event-aligned differenced response per seed (the W_shape ingredient; NaN
    # row when no event). Always collected — popped before serialization, emitted only
    # under --emit-ea-bins. Window-independent (event-aligned), so identical across win_ms.
    ea_net_bins_seeds = []
    # B1c-rescue sidecar rows (per-seed per-bin): sham = core_only EA-window count (a 'local
    # rate' predictor); onset = first-spike ms rel window start (true recruitment order, not
    # the response-RANK proxy). ONLY populated under collect_aux (--emit-ea-aux); empty
    # otherwise so the default path is byte-identical.
    ea_sham_bins_seeds: list = []
    ea_onset_bins_seeds: list = []
    # Per-seed raw/differenced rows (so seed variability is visible in per_seed_metrics.csv).
    per_seed_rows = []

    for seed_i, conds in enumerate(runs):
        # --- DIFFERENCED primary: core_kick - core_only in the early window ---------
        bins_ck = _window_bins(conds["core_kick"], bin_of_cell, n_bins,
                               t_kick + lo, t_kick + hi, p.dt)
        bins_co = _window_bins(conds["core_only"], bin_of_cell, n_bins,
                               t_kick + lo, t_kick + hi, p.dt)
        net_bins = np.clip(bins_ck - bins_co, 0.0, np.inf)
        src_resps.append(float(net_bins[src_bin_idx]))
        downstream_resps.append(float(net_bins.sum() - net_bins[src_bin_idx]))
        net_bins_seeds.append(net_bins)

        # --- RAW core_kick run properties (returned / runaway / frac_time_on_post) ---
        rp = _trace_run_properties(conds["core_kick"]["E_spk_bool"], p.dt,
                                   t_kick, T_end, dur_kick)
        returned = rp["returned"]; runaway = rp["runaway"]
        frac_on_post = rp["frac_time_on_post"]
        returned_seeds.append(returned)
        runaway_seeds.append(runaway)
        frac_on_post_seeds.append(frac_on_post)

        # --- Per-seed spatial extent on THIS seed's DIFFERENCED net_bins ------------
        seed_n_act, seed_r95, seed_far = _spatial_extent(
            net_bins, bin_centers, src_bin_idx, far_radius_mm)
        seed_local_returned = bool(
            returned and (not runaway) and
            (frac_on_post <= FRAC_TIME_ON_MAX) and
            (seed_n_act <= bins_cap) and
            (seed_r95 <= r95_cap) and
            (seed_far <= FAR_FRAC_CAP)
        )
        seed_local_returned_seeds.append(seed_local_returned)

        # --- EVENT-ALIGNED window (B-branch, reported ALONGSIDE the fixed window) ----
        # Build this seed's DIFFERENCED source-excluded downstream trace over the whole
        # run, find the event onset t0 (first crossing at/after t_kick+dur_kick of the
        # onset bar, relative to the QUIET pre-kick baseline), then compute the SAME
        # DIFFERENCED spatial metrics in [t0+EA_DELTA1, t0+EA_DELTA2]. If no event is
        # detected, the event-aligned metrics are absent (NaN) and event_detected=False.
        down_diff = _downstream_diff_trace(
            conds["core_kick"], conds["core_only"], bin_of_cell, n_bins, src_bin_idx,
            p.dt, TRACE_BIN_MS, T_end)
        t0_ms, ev_detected = _event_onset(
            down_diff, TRACE_BIN_MS, t_kick, dur_kick,
            baseline_lo=t_kick - 50.0, baseline_hi=t_kick)
        if ev_detected:
            ea_lo, ea_hi = _event_aligned_window(t0_ms, EA_DELTA1, EA_DELTA2)
            ea_bins_ck = _window_bins(conds["core_kick"], bin_of_cell, n_bins,
                                      ea_lo, ea_hi, p.dt)
            ea_bins_co = _window_bins(conds["core_only"], bin_of_cell, n_bins,
                                      ea_lo, ea_hi, p.dt)
            ea_net_bins = np.clip(ea_bins_ck - ea_bins_co, 0.0, np.inf)
            ea_downstream = float(ea_net_bins.sum() - ea_net_bins[src_bin_idx])
            ea_n_act, ea_r95, ea_far = _spatial_extent(
                ea_net_bins, bin_centers, src_bin_idx, far_radius_mm)
            if collect_aux:
                ea_sham_row = np.asarray(ea_bins_co, dtype=float)
                ea_onset_row = _bin_first_onset_in_window(
                    conds["core_kick"], bin_of_cell, n_bins, ea_lo, ea_hi, p.dt)
        else:
            t0_ms = float("nan")
            ea_downstream = float("nan")
            ea_n_act = float("nan")
            ea_r95 = float("nan")
            ea_far = float("nan")
            # No event => no per-bin shape; NaN row (module excludes it, mirroring *_ea=NaN).
            ea_net_bins = np.full(n_bins, np.nan)
            if collect_aux:
                ea_sham_row = np.full(n_bins, np.nan)
                ea_onset_row = np.full(n_bins, np.nan)
        ea_net_bins_seeds.append(np.asarray(ea_net_bins, dtype=float))
        if collect_aux:
            ea_sham_bins_seeds.append(ea_sham_row)
            ea_onset_bins_seeds.append(ea_onset_row)
        t0_ms_seeds.append(t0_ms)
        event_detected_seeds.append(bool(ev_detected))
        downstream_resp_ea_seeds.append(ea_downstream)
        n_activated_bins_ea_seeds.append(ea_n_act)
        r95_mm_ea_seeds.append(ea_r95)
        far_field_frac_ea_seeds.append(ea_far)

        # --- kick_only / no_core_no_kick CONTROLS (NOT in the primary gate) ---------
        # Computed BEFORE the confound gate because no_core_no_kick is the bare-sheet
        # reference the RELATIVE core_only_quiet gate compares against.
        ko_bins = _window_bins(conds["kick_only"], bin_of_cell, n_bins,
                               t_kick + lo, t_kick + hi, p.dt)
        nc_bins = _window_bins(conds["no_core_no_kick"], bin_of_cell, n_bins,
                               t_kick + lo, t_kick + hi, p.dt)
        nc_src = float(nc_bins[src_bin_idx])
        nc_downstream = float(nc_bins.sum() - nc_bins[src_bin_idx])
        ko_downstream_resps.append(float(ko_bins.sum() - ko_bins[src_bin_idx]))
        nc_src_resps.append(nc_src)
        nc_downstream_resps.append(nc_downstream)
        ko_minus_nc = np.clip(ko_bins - nc_bins, 0.0, np.inf)
        ko_minus_nc_seeds.append(float(ko_minus_nc.sum() - ko_minus_nc[src_bin_idx]))

        # --- RAW core_only confound gate (CORE mode) --------------------------------
        # Raw core_only window spikes (source / non-source) + its OWN run properties +
        # whether any active-fraction event overlaps [t_kick, t_kick + win_hi].
        # The quiet test is RELATIVE to this seed's PAIRED bare sheet (no_core_no_kick):
        # a quiet core does NOT produce materially MORE window activity (source or
        # downstream) than the bare sheet does spontaneously. The old absolute floors
        # (< 2 spikes) sat below the bare-sheet background and falsely killed every core.
        co_bins = _window_bins(conds["core_only"], bin_of_cell, n_bins,
                               t_kick + lo, t_kick + hi, p.dt)
        co_src = float(co_bins[src_bin_idx])
        co_downstream = float(co_bins.sum() - co_bins[src_bin_idx])
        co_rp = _trace_run_properties(conds["core_only"]["E_spk_bool"], p.dt,
                                      t_kick, T_end, dur_kick)
        co_event_in_win = _trace_has_event_in(
            co_rp["trace"], co_rp["rel_bar"], t_kick, t_kick + hi)
        co_quiet = _core_only_quiet(
            co_src, co_downstream, co_rp["frac_time_on_post"],
            co_event_in_win, nc_src, nc_downstream)
        co_src_resps.append(co_src)
        co_downstream_resps.append(co_downstream)
        co_frac_on_post_seeds.append(co_rp["frac_time_on_post"])
        co_event_in_win_seeds.append(bool(co_event_in_win))
        co_quiet_seeds.append(co_quiet)

        # Per-seed row (raw + differenced) so seed variability is auditable.
        per_seed_rows.append({
            "seed": seed_i,
            "source_resp": float(net_bins[src_bin_idx]),
            "downstream_resp": float(net_bins.sum() - net_bins[src_bin_idx]),
            "n_activated_bins": int(seed_n_act),
            "r95_mm": float(seed_r95),
            "far_field_frac": float(seed_far),
            "returned": bool(returned),
            "runaway": bool(runaway),
            "frac_time_on_post": float(frac_on_post),
            "seed_local_returned": bool(seed_local_returned),
            "core_only_source_resp": float(co_src),
            "core_only_downstream_resp": float(co_downstream),
            "core_only_frac_time_on_post": float(co_rp["frac_time_on_post"]),
            "core_only_event_in_win": bool(co_event_in_win),
            "core_only_quiet": bool(co_quiet),
            "kick_only_downstream": float(ko_bins.sum() - ko_bins[src_bin_idx]),
            "no_core_no_kick_source": nc_src,
            "no_core_no_kick_downstream": nc_downstream,
            # EVENT-ALIGNED (B-branch, reported alongside; NaN if no event detected)
            "t0_ms": float(t0_ms),
            "event_detected": bool(ev_detected),
            "downstream_resp_ea": float(ea_downstream),
            "n_activated_bins_ea": float(ea_n_act),
            "r95_mm_ea": float(ea_r95),
            "far_field_frac_ea": float(ea_far),
        })

    mean_src = float(np.mean(src_resps))
    mean_downstream = float(np.mean(downstream_resps))
    mean_net_bins = np.mean(np.stack(net_bins_seeds, axis=0), axis=0)

    # Spatial extent from the seed-averaged early-window net response (summary/figure)
    n_activated_bins, r95_mm, far_field_frac = _spatial_extent(
        mean_net_bins, bin_centers, src_bin_idx, far_radius_mm)

    # Aggregate run-property flags across seeds: majority for booleans, mean for frac.
    returned = bool(np.mean(returned_seeds) >= 0.5)
    runaway = bool(np.mean(runaway_seeds) >= 0.5)
    frac_time_on_post = float(np.mean(frac_on_post_seeds))
    # Fraction of seeds that were locally-returned (the robust per-seed gate input).
    pass_frac_seeds = float(np.mean(seed_local_returned_seeds))

    # core_only confound aggregation. In BARE-SHEET mode there is no confound gate, so
    # core_only_quiet is forced True (and the bare-sheet path keeps its old behavior).
    core_only_source_resp = float(np.mean(co_src_resps))
    core_only_downstream_resp = float(np.mean(co_downstream_resps))
    core_only_frac_time_on_post = float(np.mean(co_frac_on_post_seeds))
    core_only_event_in_win = bool(np.mean(co_event_in_win_seeds) >= 0.5)
    core_only_quiet = (bool(np.mean(co_quiet_seeds) >= 0.5) if core_mode else True)
    # bare-sheet source-bin background (relative confound gate reference).
    no_core_no_kick_source = float(np.mean(nc_src_resps))

    window_after_dur_kick = bool(lo >= dur_kick)
    # Overlap of [lo, hi) with [0, dur_kick) — how much of the window is inside kick drive
    kick_dur_overlap_ms = float(max(0.0, min(hi, dur_kick) - max(lo, 0.0)))

    # EVENT-ALIGNED aggregation (B-branch, reported alongside the fixed window). Numerics
    # are means over ONLY the seeds that detected an event (absent events are NaN and
    # excluded); event_detected_frac is the fraction of seeds that detected an event. When
    # no seed detected an event every *_ea numeric is NaN (no events to average).
    def _nanmean_or_nan(vals):
        arr = np.asarray(vals, dtype=float)
        return float(np.nanmean(arr)) if np.any(np.isfinite(arr)) else float("nan")

    event_detected_frac = float(np.mean(event_detected_seeds))
    t0_ms = _nanmean_or_nan(t0_ms_seeds)
    downstream_resp_ea = _nanmean_or_nan(downstream_resp_ea_seeds)
    n_activated_bins_ea = _nanmean_or_nan(n_activated_bins_ea_seeds)
    r95_mm_ea = _nanmean_or_nan(r95_mm_ea_seeds)
    far_field_frac_ea = _nanmean_or_nan(far_field_frac_ea_seeds)

    return {
        "kick_boost": kick_boost,
        "win_ms": list(win_ms),
        # DIFFERENCED primary (core_kick - core_only)
        "source_resp": mean_src,
        "downstream_resp": mean_downstream,
        # RAW core_kick run properties
        "returned": returned,
        "runaway": runaway,
        "frac_time_on_post": frac_time_on_post,
        "pass_frac_seeds": pass_frac_seeds,
        # DIFFERENCED spatial extent
        "n_activated_bins": int(n_activated_bins),
        "r95_mm": float(r95_mm),
        "far_field_frac": float(far_field_frac),
        "window_after_dur_kick": window_after_dur_kick,
        "kick_dur_overlap_ms": kick_dur_overlap_ms,
        # RAW core_only confound gate (bool + underlying numbers)
        "core_only_quiet": core_only_quiet,
        "core_only_source_resp": core_only_source_resp,
        "core_only_downstream_resp": core_only_downstream_resp,
        "core_only_frac_time_on_post": core_only_frac_time_on_post,
        "core_only_event_in_win": core_only_event_in_win,
        # kick_only / no_core_no_kick CONTROLS (NOT in the primary gate)
        "kick_only_downstream": float(np.mean(ko_downstream_resps)),
        "no_core_no_kick_source": no_core_no_kick_source,
        "no_core_no_kick_downstream": float(np.mean(nc_downstream_resps)),
        "kick_only_minus_no_core_downstream": float(np.mean(ko_minus_nc_seeds)),
        # EVENT-ALIGNED (B-branch, reported ALONGSIDE the fixed window; NOT gate inputs).
        # Same DIFFERENCED spatial metrics measured in [t0+EA_DELTA1, t0+EA_DELTA2]
        # relative to the per-seed event onset t0. returned/runaway are whole-run RAW
        # core_kick properties (above) — shared with the fixed window, not re-derived.
        "t0_ms": t0_ms,
        "event_detected_frac": event_detected_frac,
        "downstream_resp_ea": downstream_resp_ea,
        "n_activated_bins_ea": n_activated_bins_ea,
        "r95_mm_ea": r95_mm_ea,
        "far_field_frac_ea": far_field_frac_ea,
        # legacy field kept for figure code compatibility
        "net_response": mean_src + mean_downstream,
        # Per-seed raw/differenced rows (for per_seed_metrics.csv; not consumed by the selector).
        "per_seed": per_seed_rows,
        # PRIVATE (leading underscore): per-seed per-bin event-aligned matrix, the W_shape
        # ingredient. Popped in the main loop before serialization; emitted to ea_net_bins.npz
        # only under --emit-ea-bins. Shape (n_seeds, n_bins). NOT a gate/selector input.
        "_ea_net_bins_seeds": np.stack(ea_net_bins_seeds, axis=0),
        # PRIVATE B1c-rescue rows; present ONLY under collect_aux (np.stack not even called
        # otherwise). Popped in the main loop, emitted to ea_aux_bins.npz under --emit-ea-aux.
        **({
            "_ea_sham_bins_seeds": np.stack(ea_sham_bins_seeds, axis=0),
            "_ea_onset_bins_seeds": np.stack(ea_onset_bins_seeds, axis=0),
        } if collect_aux else {}),
    }


# Waterfall gate order — the order in which gates are checked, and the order the
# gate_waterfall.csv reports cumulative survivors. first_failed_gate is the FIRST gate
# in this list that fails. Keep this list and _candidate_gates() in lockstep.
_GATE_ORDER = (
    "pass_window_after",
    "pass_core_quiet",
    "pass_source",
    "pass_early",
    "pass_local",
    "pass_return",
    "pass_not_runaway",
    "pass_seed_frac",
    "pass_bin_frac",
)


def _candidate_gates(agg: dict, bins_cap: float, r95_cap: float) -> dict:
    """Evaluate every gate (waterfall order) as an explicit boolean from the
    per-candidate AGGREGATED values (mean for numerics, majority/pass-frac for
    booleans). Returns an ordered dict gate_name -> bool. Each gate is exposed
    separately so STATUS / candidate_table can show exactly where a candidate dies.

    The gate definitions mirror the per-rep-bin _bin_pass logic but operate on the
    candidate's rep-bin-aggregated fields (so the candidate record summarises the
    whole (kick_boost, win) point)."""
    return {
        "pass_window_after": bool(agg["window_after_dur_kick"]),
        "pass_core_quiet": bool(agg["core_only_quiet"]),
        "pass_source": agg["source_resp"] >= SOURCE_FLOOR,
        "pass_early": agg["downstream_resp"] >= DOWNSTREAM_FLOOR,
        "pass_local": (
            (agg["n_activated_bins"] <= bins_cap) and
            (agg["r95_mm"] <= r95_cap) and
            (agg["far_field_frac"] <= FAR_FRAC_CAP)
        ),
        "pass_return": bool(agg["returned"]),
        "pass_not_runaway": (
            (not bool(agg["runaway"])) and
            (agg["frac_time_on_post"] <= FRAC_TIME_ON_MAX)
        ),
        "pass_seed_frac": agg["pass_frac_seeds"] >= SEED_PASS_FRAC,
        "pass_bin_frac": agg["pass_frac_bins"] >= BIN_PASS_FRAC,
    }


def _first_failed_gate(gates: dict) -> str | None:
    """First gate (in _GATE_ORDER) that is False, or None if all pass."""
    for g in _GATE_ORDER:
        if not gates[g]:
            return g
    return None


def _candidate_class(agg: dict, gates: dict) -> str:
    """3-class label for a candidate (waterfall of mutually-exclusive cases):

      confounded   : NOT core_only_quiet (core self-ignites — can't trust the difference).
      silent       : else downstream_resp (source-excluded early) < DOWNSTREAM_FLOOR (≈0).
      escape       : else NOT returned (runaway / sustained — for basin K_min, not W_0).
      linear_probe : else returned AND local (pass_local) AND not runaway (the small-kick W_small candidate).
      finite_event : else returned AND not runaway BUT NOT local (a self-limited finite event
                     with early propagation bigger/farther than the linear caps — the W_event candidate).
    """
    if not bool(agg["core_only_quiet"]):
        return "confounded"
    if agg["downstream_resp"] < DOWNSTREAM_FLOOR:
        return "silent"
    if not bool(agg["returned"]):
        return "escape"
    not_runaway = (not bool(agg["runaway"])) and (agg["frac_time_on_post"] <= FRAC_TIME_ON_MAX)
    if bool(agg["returned"]) and gates["pass_local"] and not_runaway:
        return "linear_probe"
    return "finite_event"


def _diagnose(results_by_bin: dict, n_bins: int | None = None,
              L: float | None = None, mode: str = "explore") -> dict:
    """PURE diagnostic + selection core. Builds one aggregated candidate record per
    (kick_boost, win) with every gate as an explicit boolean + the underlying number +
    first_failed_gate + candidate_class, plus the gate waterfall, the best-failed
    candidate, and (if any qualifies) the selection.

    mode == 'explore' : NEVER raise on "no candidate qualifies" — return status=='NO_GO'
                        with the full candidate table (the L20-explore phase needs the
                        diagnostics, not a bare RuntimeError that loses the data).
    mode == 'strict'  : keep the GO/NO-GO RuntimeError when no candidate qualifies.

    The no-post-DUR_KICK-window RuntimeError is a USAGE error (all windows fall during
    the kick drive) and is raised in BOTH modes — it is not a science no-go.
    """
    from collections import defaultdict

    bins_by_key: dict = defaultdict(list)
    agg_window_after: dict = {}
    for bin_records in results_by_bin.values():
        for rec in bin_records:
            key = (rec["kick_boost"], tuple(rec["win_ms"]))
            bins_by_key[key].append(rec)
            agg_window_after[key] = rec["window_after_dur_kick"]

    grid_n_bins = n_bins if n_bins is not None else _infer_n_bins(results_by_bin)
    bins_cap = BINS_CAP_FRAC * grid_n_bins
    r95_cap = R95_CAP_FRAC * L if L is not None else R95_CAP_MM_FALLBACK

    keys = sorted(bins_by_key.keys())

    # Usage error (raised in BOTH modes): no artifact-free post-DUR_KICK window.
    valid_windows = sorted({wm for (kb, wm) in keys if agg_window_after.get((kb, wm), False)})
    if not valid_windows:
        msg = (
            "[CALIBRATION FAIL] No candidate window starts at or after DUR_KICK. "
            "All swept windows fall during the kick drive, meaning the first-gen "
            "response is contaminated by direct stimulation. Add window candidates "
            "with lo >= DUR_KICK (e.g. [DUR_KICK, DUR_KICK+6], [DUR_KICK, DUR_KICK+10]) "
            "before re-running calibration."
        )
        print(msg)
        raise RuntimeError(msg)

    def _bin_pass(rec) -> bool:
        """Per-rep-bin local+returned verdict (mirrors the legacy gate). Every condition
        stays individually visible. Used to compute pass_frac_bins per candidate."""
        return (
            bool(rec["window_after_dur_kick"]) and
            bool(rec.get("core_only_quiet", True)) and
            rec["source_resp"] >= SOURCE_FLOOR and
            rec["downstream_resp"] >= DOWNSTREAM_FLOOR and
            rec.get("pass_frac_seeds", 1.0) >= SEED_PASS_FRAC and
            rec["n_activated_bins"] <= bins_cap and
            rec.get("r95_mm", 0.0) <= r95_cap and
            rec["far_field_frac"] <= FAR_FRAC_CAP and
            bool(rec["returned"]) and
            (not bool(rec["runaway"])) and
            rec["frac_time_on_post"] <= FRAC_TIME_ON_MAX
        )

    def _mean(key, f, default=0.0):
        # Diagnostic-only extras (confound numbers, controls) may be absent on the
        # synthetic unit-test records — default benign so the gate-bearing core fields
        # (which are always present) still drive the verdict.
        return float(np.mean([r.get(f, default) for r in bins_by_key[key]]))

    def _maj(key, f, default=False):
        return bool(np.mean([bool(r.get(f, default)) for r in bins_by_key[key]]) >= 0.5)

    def _nanmean(key, f):
        # Nan-aware mean over rep bins for the EVENT-ALIGNED fields: a rep bin with no
        # detected event carries NaN and is excluded; all-NaN -> NaN (no events). Default
        # NaN (not 0.0) for records that omit the field (e.g. synthetic unit-test rows).
        vals = np.asarray([float(r.get(f, float("nan"))) for r in bins_by_key[key]],
                          dtype=float)
        return float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else float("nan")

    # ---- Build one aggregated candidate record per (kick_boost, win) ----
    candidates: list[dict] = []
    for key in keys:
        kb, wm = key
        recs = bins_by_key[key]
        pass_frac_bins = float(np.mean([_bin_pass(r) for r in recs])) if recs else 0.0
        agg = {
            "kick_boost": kb,
            "win_ms": list(wm),
            "window_after_dur_kick": bool(agg_window_after.get(key, False)),
            # DIFFERENCED primary (mean over rep bins)
            "source_resp": _mean(key, "source_resp"),
            "downstream_resp": _mean(key, "downstream_resp"),
            "n_activated_bins": _mean(key, "n_activated_bins"),
            "r95_mm": _mean(key, "r95_mm"),
            "far_field_frac": _mean(key, "far_field_frac"),
            # RAW core_kick run properties (majority for booleans, mean for frac)
            "returned": _maj(key, "returned"),
            "runaway": _maj(key, "runaway"),
            "frac_time_on_post": _mean(key, "frac_time_on_post"),
            "pass_frac_seeds": _mean(key, "pass_frac_seeds", default=1.0),
            # RAW core_only confound (majority bool + mean numbers).
            # Bare-sheet records carry core_only_quiet=True by construction (no confound).
            "core_only_quiet": _maj(key, "core_only_quiet", default=True),
            "core_only_source_resp": _mean(key, "core_only_source_resp"),
            "core_only_downstream_resp": _mean(key, "core_only_downstream_resp"),
            "core_only_frac_time_on_post": _mean(key, "core_only_frac_time_on_post"),
            "core_only_event_in_win": _maj(key, "core_only_event_in_win", default=False),
            # CONTROLS (mean over rep bins)
            "kick_only_downstream": _mean(key, "kick_only_downstream"),
            "no_core_no_kick_source": _mean(key, "no_core_no_kick_source"),
            "no_core_no_kick_downstream": _mean(key, "no_core_no_kick_downstream"),
            "kick_only_minus_no_core_downstream": _mean(key, "kick_only_minus_no_core_downstream"),
            # EVENT-ALIGNED (B-branch, reported alongside; NOT a gate input). event_detected_frac
            # is a plain mean (0 when absent); the numeric *_ea use nan-aware mean over detected
            # events. These never enter _candidate_gates / _bin_pass / the selection logic.
            "event_detected_frac": _mean(key, "event_detected_frac"),
            "t0_ms": _nanmean(key, "t0_ms"),
            "downstream_resp_ea": _nanmean(key, "downstream_resp_ea"),
            "n_activated_bins_ea": _nanmean(key, "n_activated_bins_ea"),
            "r95_mm_ea": _nanmean(key, "r95_mm_ea"),
            "far_field_frac_ea": _nanmean(key, "far_field_frac_ea"),
            # per-candidate robustness fraction
            "pass_frac_bins": pass_frac_bins,
        }
        gates = _candidate_gates(agg, bins_cap, r95_cap)
        agg["gates"] = gates
        agg["first_failed_gate"] = _first_failed_gate(gates)
        agg["candidate_class"] = _candidate_class(agg, gates)
        agg["qualifies"] = (agg["first_failed_gate"] is None)
        candidates.append(agg)

    # ---- Gate waterfall (cumulative survivors in _GATE_ORDER) ----
    waterfall = [("total", len(candidates))]
    survivors = list(candidates)
    for g in _GATE_ORDER:
        survivors = [c for c in survivors if c["gates"][g]]
        waterfall.append((g, len(survivors)))
    selected_candidates = [c for c in candidates if c["qualifies"]]
    waterfall.append(("SELECTED", len(selected_candidates)))

    # ---- Best-failed candidate: passed the MOST gates; ties -> lowest
    #      first_failed_gate index, then highest downstream ----
    def _n_gates_passed(c):
        return sum(1 for g in _GATE_ORDER if c["gates"][g])

    def _ffg_index(c):
        ffg = c["first_failed_gate"]
        return len(_GATE_ORDER) if ffg is None else _GATE_ORDER.index(ffg)

    best_failed = None
    failed = [c for c in candidates if not c["qualifies"]]
    if failed:
        best_failed = max(
            failed,
            key=lambda c: (_n_gates_passed(c), _ffg_index(c), c["downstream_resp"]),
        )

    # ---- Selection (minimum kick, then earliest window) ----
    status = "GO" if selected_candidates else "NO_GO"
    selected = None
    rationale = None
    if selected_candidates:
        ordered = sorted(selected_candidates,
                         key=lambda c: (c["kick_boost"], c["win_ms"][0], c["win_ms"][1]))
        sel = ordered[0]
        selected = sel
        rationale = (
            f"selected MINIMUM qualifying kick_boost={sel['kick_boost']}, win_ms={sel['win_ms']}: "
            f"pass_frac_bins={sel['pass_frac_bins']:.2f}>=BIN_PASS_FRAC({BIN_PASS_FRAC}), "
            f"mean pass_frac_seeds={sel['pass_frac_seeds']:.2f}>=SEED_PASS_FRAC({SEED_PASS_FRAC:.2f}), "
            f"source_resp={sel['source_resp']:.2f}>=SOURCE_FLOOR({SOURCE_FLOOR}), "
            f"downstream_resp={sel['downstream_resp']:.2f}>=DOWNSTREAM_FLOOR({DOWNSTREAM_FLOOR}), "
            f"returned (majority), runaway=False, "
            f"frac_time_on_post={sel['frac_time_on_post']:.3f}<=FRAC_TIME_ON_MAX({FRAC_TIME_ON_MAX}), "
            f"n_activated_bins={sel['n_activated_bins']:.1f}<=cap({bins_cap:.1f}), "
            f"r95_mm={sel['r95_mm']:.2f}<=r95_cap({r95_cap:.2f}), "
            f"far_field_frac={sel['far_field_frac']:.3f}<=FAR_FRAC_CAP({FAR_FRAC_CAP}). "
            f"LOCAL + RETURNED + non-global first-generation response (NOT the biggest wave)."
        )
        # selector_rank: order among qualifying candidates (0 = selected).
        rank_of = {id(c): i for i, c in enumerate(ordered)}
        for c in candidates:
            c["selector_rank"] = rank_of.get(id(c))
    else:
        for c in candidates:
            c["selector_rank"] = None

    # strict mode: keep the GO/NO-GO RuntimeError when nothing qualifies.
    if status == "NO_GO" and mode == "strict":
        raise RuntimeError(
            "[CALIBRATION GO/NO-GO FAIL] no minimal kick produces a LOCAL, RETURNED, "
            "non-global first-generation response at this config — the small-kick "
            "local-W premise does not hold here; consider a finite-amplitude / "
            "event-conditioned operator instead of a linear W. "
            "(No (kick_boost, window) had pass_frac_bins>=BIN_PASS_FRAC over rep bins, "
            "where each bin needs source>=floor, downstream>=floor, "
            "pass_frac_seeds>=SEED_PASS_FRAC, returned, not-runaway, "
            "frac_time_on_post<=max, local extent within r95/bins/far caps.)"
        )

    return {
        "status": status,
        "mode": mode,
        "selected": selected,
        "rationale": rationale,
        "candidates": candidates,
        "waterfall": waterfall,
        "best_failed": best_failed,
        "bins_cap": float(bins_cap),
        "r95_cap": float(r95_cap),
    }


def _select_calibration(results_by_bin: dict, n_bins: int | None = None,
                        L: float | None = None) -> dict:
    """Pick the MINIMUM kick_boost whose response is LOCAL, RETURNED, and non-global.

    Domain fact (Stage3): a LOCAL interictal event is STRONG ignition followed by
    propagation-relay FAILURE — so "maximize downstream / biggest wave" is exactly
    the wrong end (it picks the global runaway). The selector instead picks the
    SMALLEST kick (and among ties the earliest qualifying window).

    For an INSTRUMENT parameter a mean over rep bins is too lenient — one position can
    pull a kick over the line. So qualification is hierarchical and robust:

      Per REP BIN (`bin_pass`):
        - window_after_dur_kick == True            (no direct-stim contamination)
        - mean source_resp     >= SOURCE_FLOOR     (kick ignited locally)
        - mean downstream_resp >= DOWNSTREAM_FLOOR (there IS a first-gen response)
        - pass_frac_seeds >= SEED_PASS_FRAC        (most seeds local+returned, not half)
        - mean n_activated_bins <= bins_cap        (LOCAL bin count)
        - mean r95_mm           <= r95_cap         (response stays near the kick)
        - mean far_field_frac   <= FAR_FRAC_CAP    (not dumped far away)
        - returned (majority) AND not runaway (majority)
        - mean frac_time_on_post <= FRAC_TIME_ON_MAX
      Per (boost, win): `pass_frac_bins` = fraction of rep bins with bin_pass;
      QUALIFIES iff pass_frac_bins >= BIN_PASS_FRAC.

    The individual per-condition gates stay visible inside bin_pass so each is testable.
    If NO (boost, window) qualifies -> raise RuntimeError as the GO/NO-GO. Never a default.

    bins_cap = BINS_CAP_FRAC * n_bins (grid inferred when n_bins omitted, e.g. unit tests).
    r95_cap  = R95_CAP_FRAC * L when L is given, else R95_CAP_MM_FALLBACK (deterministic
    tests).

    Thin wrapper over the pure ``_diagnose`` core in STRICT mode: it RAISES the GO/NO-GO
    RuntimeError when nothing qualifies (legacy behavior) and repackages the selected
    candidate into the legacy return shape the figure code + JSON writer expect.
    """
    diag = _diagnose(results_by_bin, n_bins=n_bins, L=L, mode="strict")
    sel = diag["selected"]  # strict mode raised already if None

    sweep_summary = {}
    for c in diag["candidates"]:
        key = (c["kick_boost"], tuple(c["win_ms"]))
        sweep_summary[str(key)] = {
            "source_resp": c["source_resp"],
            "downstream_resp": c["downstream_resp"],
            "returned_majority": bool(c["returned"]),
            "runaway_majority": bool(c["runaway"]),
            "frac_time_on_post": c["frac_time_on_post"],
            "pass_frac_seeds": c["pass_frac_seeds"],
            "n_activated_bins": c["n_activated_bins"],
            "r95_mm": c["r95_mm"],
            "far_field_frac": c["far_field_frac"],
            "window_after_dur_kick": bool(c["window_after_dur_kick"]),
            "pass_frac_bins": c["pass_frac_bins"],
            "qualifies": bool(c["qualifies"]),
        }

    return {
        "kick_boost": sel["kick_boost"],
        "win_ms": list(sel["win_ms"]),
        "window_after_dur_kick": bool(sel["window_after_dur_kick"]),
        "pass_frac_bins": sel["pass_frac_bins"],
        "pass_frac_seeds": sel["pass_frac_seeds"],
        "rationale": diag["rationale"],
        "sweep_summary": sweep_summary,
    }


def _infer_n_bins(results_by_bin: dict) -> float:
    """Infer the spatial bin-grid size for the BINS_CAP. n_activated_bins counts
    NON-source bins, so the grid is at least (max observed n_activated_bins + 1).
    Falls back to the rep-bin count when no activation was observed."""
    max_act = 0
    for recs in results_by_bin.values():
        for rec in recs:
            max_act = max(max_act, int(rec.get("n_activated_bins", 0)))
    rep = max((len(recs) for recs in results_by_bin.values()), default=1)
    return float(max(max_act + 1, rep))


def _git_sha() -> str:
    """`git rev-parse HEAD`, tolerate failure (returns 'unknown')."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"


# Per-candidate diagnostic fields written to candidate_table.csv (gates expanded inline).
_CANDIDATE_NUMERIC_FIELDS = (
    "source_resp", "downstream_resp", "n_activated_bins", "r95_mm", "far_field_frac",
    "frac_time_on_post", "pass_frac_seeds", "pass_frac_bins",
    "core_only_source_resp", "core_only_downstream_resp", "core_only_frac_time_on_post",
    "kick_only_downstream", "no_core_no_kick_source", "no_core_no_kick_downstream",
    "kick_only_minus_no_core_downstream",
    # EVENT-ALIGNED (B-branch, reported alongside the fixed window)
    "event_detected_frac", "t0_ms", "downstream_resp_ea", "n_activated_bins_ea",
    "r95_mm_ea", "far_field_frac_ea",
)
_CANDIDATE_BOOL_FIELDS = (
    "window_after_dur_kick", "returned", "runaway", "core_only_event_in_win",
    "core_only_quiet", "qualifies",
)
# Per-seed fields written to per_seed_metrics.csv (raw + differenced).
_PER_SEED_FIELDS = (
    "seed", "source_resp", "downstream_resp", "n_activated_bins", "r95_mm",
    "far_field_frac", "returned", "runaway", "frac_time_on_post",
    "seed_local_returned", "core_only_source_resp", "core_only_downstream_resp",
    "core_only_frac_time_on_post", "core_only_event_in_win", "core_only_quiet",
    "kick_only_downstream", "no_core_no_kick_source", "no_core_no_kick_downstream",
    # EVENT-ALIGNED per-seed (B-branch, reported alongside the fixed window)
    "t0_ms", "event_detected", "downstream_resp_ea", "n_activated_bins_ea",
    "r95_mm_ea", "far_field_frac_ea",
)


def _failure_mode_read(first_failed_gate: str | None, class_hist: dict) -> str:
    """One-line plain-language read of the dominant failure mode (Chinese, per
    CLAUDE.md §8). Maps the dominant first_failed_gate (or the class histogram) to
    what that failure means physically. NOT over-claimed — one core config."""
    n_escape = class_hist.get("escape", 0)
    n_linear = class_hist.get("linear_probe", 0)
    n_finite = class_hist.get("finite_event", 0)
    if n_escape > 0 and (n_linear + n_finite) == 0:
        return ("大量候选是 escape（一戳就持续招募/跑飞），且没有一个 linear_probe / "
                "finite_event —— 看起来是 all-or-none，小 kick 的 local-W 前提有风险。")
    mapping = {
        "pass_core_quiet": "多数候选死在『核安静门』—— 核在测量窗里自己就在点火（核自燃），差分不可信。",
        "pass_early": "多数候选死在『早期响应门』—— 源外几乎没有第一代响应（kick 太弱或核太钝）。",
        "pass_local": "多数候选死在『局部门』—— 响应不局部（源污染 / 直接大范围蔓延）。",
        "pass_return": "多数候选死在『回静息门』—— 一戳就持续招募，活动不回到基线。",
        "pass_not_runaway": "多数候选死在『非跑飞门』—— 活动跑飞 / 大部分时间都在 on。",
        "pass_source": "多数候选死在『源点火门』—— kick 在核中心都没点着（kick 太弱）。",
        "pass_window_after": "多数候选死在『窗在 DUR_KICK 之后门』—— 窗落在 kick 驱动期内（用法问题）。",
        "pass_seed_frac": "多数候选死在『多数 seed 局部门』—— 只有少数 seed 局部+回静息，不稳健。",
        "pass_bin_frac": "多数候选死在『多数 rep-bin 门』—— 跨代表位置不一致。",
    }
    if first_failed_gate is None:
        return "全部候选都过了所有门（GO）。"
    return mapping.get(first_failed_gate,
                       f"多数候选最先死在 {first_failed_gate}。")


def _write_diagnostics(diag: dict, results_by_bin: dict, *, thresholds: dict,
                       config: dict, out_dir: str) -> None:
    """Write the full diagnostic artifact (config / thresholds / git_sha /
    candidate_table.csv / per_seed_metrics.csv / gate_waterfall.csv /
    best_failed_candidate.json / STATUS.md) — in BOTH explore and strict, even on
    no-go. STATUS.md is Chinese plain-language (CLAUDE.md §8)."""
    os.makedirs(out_dir, exist_ok=True)
    candidates = diag["candidates"]

    # --- config / thresholds / git_sha ---
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(out_dir, "thresholds.json"), "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)
    with open(os.path.join(out_dir, "git_sha.txt"), "w", encoding="utf-8") as f:
        f.write(_git_sha() + "\n")

    # --- candidate_table.csv: one row per (kick_boost, win) with ALL fields ---
    cand_path = os.path.join(out_dir, "candidate_table.csv")
    cand_cols = (["kick_boost", "win_lo", "win_hi"]
                 + list(_CANDIDATE_NUMERIC_FIELDS)
                 + list(_CANDIDATE_BOOL_FIELDS)
                 + [g for g in _GATE_ORDER]
                 + ["candidate_class", "first_failed_gate", "selector_rank"])
    with open(cand_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cand_cols)
        for c in candidates:
            row = [c["kick_boost"], c["win_ms"][0], c["win_ms"][1]]
            row += [c.get(k, "") for k in _CANDIDATE_NUMERIC_FIELDS]
            row += [int(bool(c.get(k, False))) for k in _CANDIDATE_BOOL_FIELDS]
            row += [int(bool(c["gates"][g])) for g in _GATE_ORDER]
            row += [c["candidate_class"], c["first_failed_gate"],
                    c.get("selector_rank")]
            w.writerow(row)

    # --- per_seed_metrics.csv: one row per (kick_boost, win, rep_bin, seed) ---
    seed_path = os.path.join(out_dir, "per_seed_metrics.csv")
    seed_cols = ["kick_boost", "win_lo", "win_hi", "rep_bin"] + list(_PER_SEED_FIELDS)
    with open(seed_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(seed_cols)
        for rep_bin, recs in results_by_bin.items():
            for rec in recs:
                kb = rec["kick_boost"]
                lo, hi = rec["win_ms"]
                for srow in rec.get("per_seed", []):
                    out = [kb, lo, hi, rep_bin]
                    for k in _PER_SEED_FIELDS:
                        v = srow.get(k, "")
                        out.append(int(v) if isinstance(v, bool) else v)
                    w.writerow(out)

    # --- gate_waterfall.csv: cumulative survivors per gate (one glance = where they die) ---
    wf_path = os.path.join(out_dir, "gate_waterfall.csv")
    with open(wf_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "n_surviving"])
        for stage, n in diag["waterfall"]:
            w.writerow([stage, n])

    # --- best_failed_candidate.json: the candidate that passed the MOST gates ---
    with open(os.path.join(out_dir, "best_failed_candidate.json"), "w",
              encoding="utf-8") as f:
        json.dump(diag["best_failed"], f, indent=2)

    # --- STATUS.md (Chinese plain-language, §8) ---
    _write_status_md(diag, out_dir)


def _write_status_md(diag: dict, out_dir: str) -> None:
    """STATUS.md — verdict + waterfall-in-words + dominant first_failed_gate + class
    histogram + a one-line read of the failure mode. Chinese plain-language (§8).
    Does NOT over-claim — one core config."""
    from collections import Counter
    candidates = diag["candidates"]
    status = diag["status"]
    waterfall = diag["waterfall"]
    wf = dict(waterfall)

    class_hist = Counter(c["candidate_class"] for c in candidates)
    ffg_hist = Counter(c["first_failed_gate"] for c in candidates
                       if c["first_failed_gate"] is not None)
    dominant_ffg = ffg_hist.most_common(1)[0][0] if ffg_hist else None

    n_total = wf.get("total", len(candidates))
    n_core_quiet = wf.get("pass_core_quiet", 0)
    n_early = wf.get("pass_early", 0)
    n_selected = wf.get("SELECTED", 0)

    lines = []
    lines.append("# kick 标定诊断 STATUS")
    lines.append("")
    if status == "GO":
        sel = diag["selected"]
        lines.append(f"## 结论：GO")
        lines.append("")
        lines.append(f"选定 kick_boost = **{sel['kick_boost']}**，"
                     f"win_ms = **{sel['win_ms']}**。")
        lines.append("")
        lines.append(f"理由：{diag['rationale']}")
    else:
        lines.append("## 结论：NO_GO")
        lines.append("")
        lines.append("这个工作点上没有任何 (kick, 窗) 候选满足『小 kick 触发局部、"
                     "活动回到基线、没有蔓延成全局波』的全部条件。")
    lines.append("")

    # 漏斗（用话讲）
    lines.append("## 漏斗（候选在哪一关死掉）")
    lines.append("")
    lines.append(f"一共 {n_total} 个候选：其中 {n_core_quiet} 个过了核安静门，"
                 f"{n_early} 个有早期响应，最终 {n_selected} 个被选中。")
    lines.append("")
    lines.append("逐关存活数（沿门的顺序累计递减）：")
    lines.append("")
    for stage, n in waterfall:
        lines.append(f"- {stage}: {n}")
    lines.append("")

    # 最先失败的门
    lines.append("## 最先失败的门（dominant first_failed_gate）")
    lines.append("")
    if dominant_ffg is None:
        lines.append("无（所有候选都通过）。" if status == "GO"
                     else "无候选有 first_failed_gate（异常空表）。")
    else:
        lines.append(f"**{dominant_ffg}**（最多候选最先死在这一关）。")
        lines.append("")
        lines.append(f"各门 first_failed 计数：")
        for g, n in ffg_hist.most_common():
            lines.append(f"- {g}: {n}")
    lines.append("")

    # 类别直方图
    lines.append("## 候选类别直方图")
    lines.append("")
    lines.append("- confounded（核自燃，差分不可信）："
                 f"{class_hist.get('confounded', 0)}")
    lines.append("- silent（源外几乎没有早期响应）："
                 f"{class_hist.get('silent', 0)}")
    lines.append("- escape（一戳就持续招募/跑飞，属于 basin K_min 不属于 W_0）："
                 f"{class_hist.get('escape', 0)}")
    lines.append("- finite_event（回静息且非跑飞，但比线性 caps 更大/更远的自限有限事件，W_event 候选）："
                 f"{class_hist.get('finite_event', 0)}")
    lines.append("- linear_probe（回静息+局部+非跑飞的小 kick W_small 候选）："
                 f"{class_hist.get('linear_probe', 0)}")
    lines.append("")

    # 一句话解读
    lines.append("## 一句话解读")
    lines.append("")
    lines.append(_failure_mode_read(dominant_ffg, class_hist))
    lines.append("")
    lines.append("> 注意：这只是一个 core 配置上的诊断，不要过度外推。")
    lines.append("")

    with open(os.path.join(out_dir, "STATUS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _make_figures(results_by_bin: dict, calibration: dict, out_dir: str) -> None:
    """Generate calibration diagnostic figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    kick_boosts_all = sorted(set(rec["kick_boost"]
                                 for bin_recs in results_by_bin.values()
                                 for rec in bin_recs))
    win_ms_all = sorted(set(tuple(rec["win_ms"])
                             for bin_recs in results_by_bin.values()
                             for rec in bin_recs))

    # Aggregate downstream_resp per (boost, win) across rep bins (mean), plus the
    # per-(boost, win) qualification (LOCAL+RETURNED+non-global) from the selector's
    # sweep_summary so we can color-code qualifying vs disqualified points.
    from collections import defaultdict
    agg_ds: dict = defaultdict(list)
    win_after_dur: dict = {}
    for bin_recs in results_by_bin.values():
        for rec in bin_recs:
            key = (rec["kick_boost"], tuple(rec["win_ms"]))
            agg_ds[key].append(rec["downstream_resp"])
            win_after_dur[tuple(rec["win_ms"])] = rec["window_after_dur_kick"]
    mean_agg = {k: float(np.mean(v)) for k, v in agg_ds.items()}

    sweep_summary = calibration.get("sweep_summary", {})

    def _qual(kb, wm):
        return bool(sweep_summary.get(str((kb, wm)), {}).get("qualifies", False))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel 1 (kept): downstream_resp vs KICK_BOOST per window candidate.
    ax = axes[0]
    for wm in win_ms_all:
        ys = [mean_agg.get((kb, wm), 0.0) for kb in kick_boosts_all]
        after = win_after_dur.get(wm, False)
        ls = "-" if after else "--"
        label = f"win {wm[0]}–{wm[1]}ms" + ("" if after else " [during-kick]")
        ax.plot(kick_boosts_all, ys, linestyle=ls, color="0.6", zorder=1, label=label)
    ax.axvline(calibration["kick_boost"], color="black", linestyle=":",
               label=f"selected kick={calibration['kick_boost']}")
    ax.set_xlabel("KICK_BOOST (extra ext. rate, 1/ms)")
    ax.set_ylabel("Mean downstream_resp (E spikes in non-source bins, kick − sham)")
    ax.set_title("First-gen propagation vs KICK_BOOST per window\n"
                 "(solid=post-kick artifact-free; dashed=during-kick)")
    ax.legend(fontsize=7)

    # Panel 2 (NEW question): LOCAL+RETURNED vs global/not-returned, per (boost, win).
    # Each point = one (boost, win); GREEN = qualifies (local, returned, non-global),
    # RED = disqualified (not-returned OR runaway OR too far / too many bins / mostly-on).
    # The selected point is ringed. This answers "which kick is the smallest LOCAL one",
    # which the left panel (downstream magnitude) cannot — they are different questions.
    ax2 = axes[1]
    sel = (calibration["kick_boost"], tuple(calibration["win_ms"]))
    plotted_q = plotted_d = False
    for kb in kick_boosts_all:
        for wm in win_ms_all:
            if (kb, wm) not in mean_agg:
                continue
            y = mean_agg[(kb, wm)]
            q = _qual(kb, wm)
            color = "tab:green" if q else "tab:red"
            lbl = None
            if q and not plotted_q:
                lbl, plotted_q = "LOCAL + RETURNED (qualifies)", True
            elif (not q) and not plotted_d:
                lbl, plotted_d = "global / not-returned (disqualified)", True
            ax2.scatter([kb], [y], c=color, s=45, edgecolors="k", linewidths=0.4,
                        zorder=2, label=lbl)
    if sel in mean_agg:
        ax2.scatter([sel[0]], [mean_agg[sel]], s=240, facecolors="none",
                    edgecolors="black", linewidths=1.8, zorder=3, label="selected")
    ax2.set_xlabel("KICK_BOOST (extra ext. rate, 1/ms)")
    ax2.set_ylabel("Mean downstream_resp (E spikes)")
    ax2.set_title("LOCAL+RETURNED vs global per (kick, window)\n"
                  "(selector picks the MINIMUM-kick green point)")
    ax2.legend(fontsize=7)

    plt.tight_layout()
    fig_path = os.path.join(fig_dir, "kick_calibration_sweep.png")
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Write README (Chinese, per AGENTS.md convention)
    readme_path = os.path.join(fig_dir, "README.md")
    readme_content = """\
# kick_calibration figures — 说明

### kick_calibration_sweep.png

左图：对代表 bin 取平均，展示第一代下游响应（kick − sham，非源 bin 的 E spike 数）
随 KICK_BOOST 幅度变化的曲线，每条线对应一个候选窗 [Δ1, Δ2]；实线=起点≥DUR_KICK 的
无直接刺激窗，虚线=during-kick 对照窗；黑色点线 = 选定的 kick_boost。
右图：每个 (kick, 窗) 点按"是否局部+回静息+非全局"着色——绿色=合格（局部点火、
活动回到基线、没有蔓延成全局波），红色=不合格（不回静息 / runaway / 蔓延太远或激活 bin 太多 /
大部分时间都在 on）。黑色空心圈 = 选定点。

**关注点**：选择器挑的是绿色点里 **KICK_BOOST 最小** 的那个（不是下游响应最大的那个 —
最大的往往就是错误的那一端，即全局波）。判断方式不再是 kick−sham 的晚期差（在混沌网络里
那只是两条轨迹漂开、随 kick 变强反而变小），而是直接看这一次 kick 自己的活动是否局部、
是否回到基线、是否没有变成全局波。若右图没有任何绿色点 → 选择器大声失败（go/no-go：
这个工作点上不存在"小 kick 触发局部、自停、非全局的第一代响应"）。
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)


def run_calibration(args: argparse.Namespace) -> None:
    """Full calibration sweep — only callable when Task 1 module is available."""
    from params import Params
    from connectivity import place_neurons
    from connectivity_rot import build_connectivity_rot
    from kick_probe import simulate_kick, DUR_KICK
    from src.topic4_propagation_operator import spatial_bins
    from src.sef_hfo_heterogeneity import sample_core_field

    core_mode = args.core_mean is not None
    # Core-mode forced semantics (hard rule): kicking a SMALL pathological core means
    # ONE probe site (the core), so multiple rep bins are meaningless. Hard-fail rather
    # than silently override — a stale --n-rep-bins should fail loudly.
    if core_mode and args.n_rep_bins != 1:
        raise RuntimeError(
            f"[CALIBRATION FAIL] core mode (--core-mean={args.core_mean}) requires "
            f"--n-rep-bins 1 (the kick lands at the single core center), got "
            f"--n-rep-bins {args.n_rep_bins}. Set --n-rep-bins 1.")

    out_dir = os.path.join(ROOT, args.out_dir) if not os.path.isabs(args.out_dir) else args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Build network
    p = Params(
        L=args.L, density=args.density, T=args.T, dt=args.dt,
        nu_ext_ratio=args.nu_ext_ratio, seed=args.seed,
    )
    rng = np.random.default_rng(args.seed)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(
        p, pos, labels, NE, NI, rng,
        theta_EE=np.radians(args.theta_ee), AR=args.AR,
    )

    posE = pos[:NE]
    bins_info = spatial_bins(posE, args.n_bins_per_axis)
    bin_centers = bins_info["bin_centers"]
    bin_of_cell = bins_info["bin_of_cell"]
    n_bins = bin_centers.shape[0]

    win_ms_list = _parse_win_ms(args.win_ms)

    # E-population labels (first NE are E; the engine orders E before I) — for the
    # heterogeneous-core threshold field.
    is_E = (labels == 0)

    # Uniform threshold field (length NE+NI) — the bare-sheet / kick_only / no_core
    # field. vth0 everywhere.
    vth_uniform = np.full(NE + NI, args.vth0)

    if core_mode:
        # Core center = --core-center-xy, else the geometric center of the sheet.
        if args.core_center_xy is not None:
            core_center = np.asarray(args.core_center_xy, dtype=float)
        else:
            core_center = bin_centers.mean(axis=0)
        # src_bin_idx = the bin whose center is nearest the core center.
        src_bin_idx = int(np.argmin(
            np.linalg.norm(bin_centers - core_center[None, :], axis=1)))
        rep_bin_indices = [src_bin_idx]
        # Core threshold field: in-core E thresholds ~ N(core_mean, core_std) truncated;
        # everyone else = vth0 (base_mean). Dedicated rng=default_rng(seed+7).
        vth_core = sample_core_field(
            pos, is_E, core_center, args.core_r,
            np.random.default_rng(args.seed + 7),
            core_mean=args.core_mean, core_std=args.core_std,
            base_mean=args.vth0)["vth"]
    else:
        # BARE SHEET: legacy behavior — kick lands at each rep bin's center; the
        # "core" field is the uniform field (so core_kick==kick_only, core_only==sham).
        rep_bin_indices = _select_rep_bins(bin_centers, args.n_rep_bins, rng)
        core_center = None
        vth_core = vth_uniform

    # Static-μ permissivity (spec m3_static_mu_pilot_2026-06-24): deepen the susceptibility
    # field by ΔVth(μ)=dvth_at_mu1·μ times the h field. h=core_susceptibility (default) acts only
    # where the core is; uniform = global-μ control; shuffled = location-broken control. μ=0
    # (default) returns vth_core UNCHANGED -> byte-identical to the pre-μ runner. Engine untouched.
    from src.sef_hfo_mu_basin import apply_mu
    core_mean_for_mu = args.core_mean if args.core_mean is not None else args.vth0
    vth_core = apply_mu(vth_core, args.vth0, core_mean_for_mu, args.mu, args.dvth_at_mu1,
                        args.h_mode, np.random.default_rng(args.seed + 13))

    # mini-W_event: a named kick source means ONE source bin per invocation (the bin
    # nearest the kick), in BOTH core and bare mode. Without --kick-xy the legacy rep-bin
    # selection above is unchanged (kick_xy=None => bit-identical).
    if args.kick_xy is not None:
        kx = np.asarray(args.kick_xy, dtype=float)
        rep_bin_indices = [int(np.argmin(np.linalg.norm(bin_centers - kx[None, :], axis=1)))]

    # far_radius for far_field_frac: explicit if passed, else a fraction of L.
    far_radius_mm = (args.far_radius_mm if args.far_radius_mm is not None
                     else 0.35 * float(args.L))

    # Per-seed locality caps threaded into the detector (so locality is judged per seed).
    bins_cap = BINS_CAP_FRAC * n_bins
    r95_cap = R95_CAP_FRAC * float(args.L)

    print(f"DUR_KICK={DUR_KICK} ms (kick drive duration from kick_probe.py)")
    print(f"core_mode={core_mode}"
          + (f" (core_mean={args.core_mean}, core_std={args.core_std}, "
             f"core_r={args.core_r}, src_bin_idx={rep_bin_indices[0]})"
             if core_mode else " (bare sheet)"))
    # --event-figure-only BACKFILL: skip the sweep, read the representative kick from the
    # existing per_seed_metrics.csv, and (re)write only figures/event_diagnostic.png.
    if args.event_figure_only:
        rep_kick = _rep_kick_from_csv(os.path.join(out_dir, "per_seed_metrics.csv"),
                                      fallback=float(args.kick_boosts[-1]))
        _emit_event_figure(p, net, NE, NI, args, core_mode, core_center, vth_core,
                           vth_uniform, bin_centers, bin_of_cell, n_bins, posE,
                           rep_bin_indices, rep_kick, out_dir, "figure-only", simulate_kick,
                           DUR_KICK)
        return

    print(f"Sweeping {len(args.kick_boosts)} kick_boosts × "
          f"{len(win_ms_list)} windows × "
          f"{len(rep_bin_indices)} rep bins × "
          f"{args.seeds} seeds …")
    for wm in win_ms_list:
        after = wm[0] >= DUR_KICK
        overlap = max(0.0, min(wm[1], DUR_KICK) - max(wm[0], 0.0))
        print(f"  win {wm}: window_after_dur_kick={after}, "
              f"kick_dur_overlap={overlap:.1f}ms")

    results_by_bin: dict = {}
    ea_emit_mats: dict = {}   # (bin_idx, kb) -> (n_seed, n_bins) EA matrix (window-independent)
    ea_emit_src: dict = {}    # bin_idx -> src_bin_idx
    ea_sham_emit_mats: dict = {}   # (bin_idx, kb) -> (n_seed, n_bins) sham EA counts (--emit-ea-aux)
    ea_onset_emit_mats: dict = {}  # (bin_idx, kb) -> (n_seed, n_bins) first-onset ms (--emit-ea-aux)
    for bin_idx in rep_bin_indices:
        # Resolve where the kick lands + which bin is the source. kick_xy=None keeps
        # the legacy behavior (core mode: core center; bare: rep bin) bit-for-bit;
        # --kick-xy moves the kick + source bin to (x,y) while the core stays put.
        bin_center, src_bin_idx = _resolve_kick_center_and_src(
            args.kick_xy, core_center, core_mode, bin_centers, bin_idx)
        ea_emit_src[int(bin_idx)] = int(src_bin_idx)
        bin_recs = []
        for kb in args.kick_boosts:
            # Run the four paired conditions ONCE for this (bin, kick) and cache the
            # spikes; every win_ms below slices these cached runs (no re-simulation).
            runs = _run_conditions(
                p, net, NE, NI,
                kick_boost=kb,
                bin_center=bin_center,
                vth_core=vth_core,
                vth_uniform=vth_uniform,
                seeds=args.seeds,
                r_kick=args.r_kick,
                t_kick=args.t_kick,
                simulate_kick=simulate_kick,
            )
            for wm in win_ms_list:
                rec = _measure_response(
                    p, net, NE, NI,
                    bin_center=bin_center,
                    src_bin_idx=src_bin_idx,
                    bin_of_cell=bin_of_cell,
                    n_bins=n_bins,
                    bin_centers=bin_centers,
                    far_radius_mm=far_radius_mm,
                    kick_boost=kb,
                    win_ms=wm,
                    seeds=args.seeds,
                    vth0=args.vth0,
                    r_kick=args.r_kick,
                    t_kick=args.t_kick,
                    dur_kick=DUR_KICK,
                    bins_cap=bins_cap,
                    r95_cap=r95_cap,
                    runs=runs,
                    core_mode=core_mode,
                    collect_aux=args.emit_ea_aux,
                )
                # Always pop the private per-bin EA matrix so the diagnostic writer never
                # serializes a numpy array. It is window-independent, so any win_ms gives
                # the same matrix for this (bin, kick) — last write wins (identical).
                ea_emit_mats[(int(bin_idx), float(kb))] = rec.pop("_ea_net_bins_seeds")
                if args.emit_ea_aux:
                    ea_sham_emit_mats[(int(bin_idx), float(kb))] = rec.pop("_ea_sham_bins_seeds")
                    ea_onset_emit_mats[(int(bin_idx), float(kb))] = rec.pop("_ea_onset_bins_seeds")
                bin_recs.append(rec)
        results_by_bin[int(bin_idx)] = bin_recs

    # Config + thresholds (written into the diagnostic artifact in BOTH GO and NO_GO).
    config = {
        "mode": args.mode,
        "sweep_parameters": {
            "kick_boosts_swept": args.kick_boosts,
            "win_ms_swept": [list(wm) for wm in win_ms_list],
            "n_rep_bins": args.n_rep_bins,
            "seeds_per_combo": args.seeds,
            "r_kick": args.r_kick,
            "t_kick": args.t_kick,
            "L": args.L,
            "density": args.density,
            "network_seed": args.seed,
            "theta_ee_deg": args.theta_ee,
            "AR": args.AR,
            "vth0": args.vth0,
            "far_radius_mm": float(far_radius_mm),
            "DUR_KICK_ms": float(DUR_KICK),
        },
        "core_config": {
            "core_mode": bool(core_mode),
            "core_mean": (float(args.core_mean) if core_mode else None),
            "core_std": (float(args.core_std) if core_mode else None),
            "core_r": (float(args.core_r) if core_mode else None),
            "core_center_xy": ([float(c) for c in core_center]
                               if core_mode and core_center is not None else None),
            "kick_xy": ([float(c) for c in args.kick_xy]
                        if args.kick_xy is not None else None),
            "mu": float(args.mu),
            "dvth_at_mu1": float(args.dvth_at_mu1),
            "h_mode": args.h_mode,
        },
        "rep_bin_indices": rep_bin_indices,
    }
    thresholds = {
        "SOURCE_FLOOR": SOURCE_FLOOR,
        "DOWNSTREAM_FLOOR": DOWNSTREAM_FLOOR,
        "CORE_BG_RATIO": CORE_BG_RATIO,
        "CORE_BG_MARGIN": CORE_BG_MARGIN,
        "BINS_CAP_FRAC": BINS_CAP_FRAC,
        "FAR_FRAC_CAP": FAR_FRAC_CAP,
        "R95_CAP_FRAC": R95_CAP_FRAC,
        "SEED_PASS_FRAC": SEED_PASS_FRAC,
        "BIN_PASS_FRAC": BIN_PASS_FRAC,
        "FRAC_TIME_ON_MAX": FRAC_TIME_ON_MAX,
        "RETURN_FRAC": RETURN_FRAC,
        "RUNAWAY_FRAC": RUNAWAY_FRAC,
        # derived caps (n_bins / L are config-specific)
        "bins_cap": float(bins_cap),
        "r95_cap": float(r95_cap),
        "n_bins": int(n_bins),
        "L": float(args.L),
    }

    # PURE diagnostic + selection. Built in explore mode so it NEVER raises on no-go —
    # the artifact is written FIRST and the strict GO/NO-GO RuntimeError is raised below,
    # AFTER the per-candidate data is on disk (it can no longer be lost). The only
    # exception _diagnose can raise here is the no-post-DUR_KICK usage error, which
    # propagates before any artifact is written (it is a usage error, not a science no-go).
    diag = _diagnose(results_by_bin, n_bins=n_bins, L=float(args.L), mode="explore")

    # Always dump the full diagnostic artifact (config / thresholds / git_sha /
    # candidate_table.csv / per_seed_metrics.csv / gate_waterfall.csv /
    # best_failed_candidate.json / STATUS.md) — even on no-go.
    _write_diagnostics(diag, results_by_bin, thresholds=thresholds,
                       config=config, out_dir=out_dir)
    print(f"Wrote diagnostics → {out_dir}")

    # mini-W_event: the per-bin event-aligned matrix the scalar artifacts drop. Emitted in
    # BOTH GO and NO_GO (ceiling is all-returned-no-runaway = NO_GO, but its W_shape is what
    # we want). --emit-ea-bins requires --kick-xy => exactly one source bin (guarded above).
    if args.emit_ea_bins:
        kicks = list(args.kick_boosts)
        bidx = int(rep_bin_indices[0])
        ea_stack = np.stack([ea_emit_mats[(bidx, float(kb))] for kb in kicks], axis=0)
        ea_path = os.path.join(out_dir, "ea_net_bins.npz")
        np.savez(
            ea_path,
            kicks=np.asarray(kicks, dtype=float),
            seeds=np.arange(ea_stack.shape[1]),
            ea_net_bins=ea_stack,                       # (n_kick, n_seed, n_bins); NaN row = no event
            bin_idx=bidx,
            src_bin_idx=int(ea_emit_src[bidx]),
            bin_centers=np.asarray(bin_centers, dtype=float),
            n_bins=int(n_bins),
            core_mean=(float(args.core_mean) if core_mode else float("nan")),
            kick_xy=np.asarray(args.kick_xy, dtype=float),
        )
        print(f"Wrote ea_net_bins.npz → {ea_path}  shape={ea_stack.shape} (n_kick,n_seed,n_bins)")

    # mini-W_event B1c-rescue sidecar (DATA_MISSING items 1+3). Separate file so ea_net_bins.npz
    # is byte-identical: per (kick,seed,bin) the core_only (sham) EA-window count (a 'local rate'
    # predictor) + the first-spike onset ms (true recruitment order vs the response-RANK proxy).
    # Does NOT add the dense per-bin K_min susceptibility map (item 2 = a separate source sweep).
    if args.emit_ea_aux:
        kicks = list(args.kick_boosts)
        bidx = int(rep_bin_indices[0])
        sham_stack = np.stack([ea_sham_emit_mats[(bidx, float(kb))] for kb in kicks], axis=0)
        onset_stack = np.stack([ea_onset_emit_mats[(bidx, float(kb))] for kb in kicks], axis=0)
        aux_path = os.path.join(out_dir, "ea_aux_bins.npz")
        np.savez(
            aux_path,
            kicks=np.asarray(kicks, dtype=float),
            seeds=np.arange(sham_stack.shape[1]),
            ea_sham_bins=sham_stack,        # (n_kick,n_seed,n_bins) per-bin core_only EA count
            ea_first_onset_ms=onset_stack,  # (n_kick,n_seed,n_bins) first-spike ms rel window; NaN=no spike
            bin_idx=bidx,
            src_bin_idx=int(ea_emit_src[bidx]),
            bin_centers=np.asarray(bin_centers, dtype=float),
            n_bins=int(n_bins),
            core_mean=(float(args.core_mean) if core_mode else float("nan")),
            kick_xy=np.asarray(args.kick_xy, dtype=float),
        )
        print(f"Wrote ea_aux_bins.npz → {aux_path}  shape={sham_stack.shape} "
              f"(sham-rate + first-onset; NOT the dense K_min map)")

    # Per-SNN-run event-diagnostic figure (user request 2026-06-23): re-simulate ONE
    # representative (kick, seed) with the already-built network (no rebuild) and render
    # raster + early per-bin heatmap + return-to-quiet. Default ON; runs in BOTH GO and NO_GO
    # so every run gets a figure. The spikes are not persisted elsewhere, so this is in-run.
    if not args.no_event_figure:
        bidx = int(rep_bin_indices[0])
        # representative kick = the one with the most locally-returned seeds (tie -> lowest)
        best_pf: dict = {}
        for rec in results_by_bin[bidx]:
            k = float(rec["kick_boost"])
            best_pf[k] = max(best_pf.get(k, 0.0), rec["pass_frac_seeds"])
        rep_kick = (max(best_pf, key=lambda k: (best_pf[k], -k)) if best_pf
                    else float(args.kick_boosts[-1]))
        _emit_event_figure(p, net, NE, NI, args, core_mode, core_center, vth_core,
                           vth_uniform, bin_centers, bin_of_cell, n_bins, posE,
                           rep_bin_indices, rep_kick, out_dir, diag["status"], simulate_kick,
                           DUR_KICK)

    if diag["status"] == "NO_GO":
        print(f"\n[CALIBRATION {diag['status']}] no (kick_boost, window) qualifies "
              f"at this config — see {os.path.join(out_dir, 'STATUS.md')} for the "
              f"per-candidate waterfall + class histogram.")
        if args.mode == "strict":
            # strict: keep the GO/NO-GO RuntimeError (after the artifact is on disk).
            raise RuntimeError(
                "[CALIBRATION GO/NO-GO FAIL] no minimal kick produces a LOCAL, RETURNED, "
                "non-global first-generation response at this config (strict mode). "
                "Diagnostics were written; see STATUS.md.")
        return  # explore: exit 0 (caller prints summary)

    # --- GO: write kick_calibration.json (selected kick/win) + figures, as before ---
    calibration = _select_calibration(results_by_bin, n_bins=n_bins, L=float(args.L))
    output = {
        "calibrated_kick_boost": calibration["kick_boost"],
        "calibrated_win_ms": calibration["win_ms"],
        "calibrated_window_after_dur_kick": calibration["window_after_dur_kick"],
        "calibrated_pass_frac_bins": calibration["pass_frac_bins"],
        "calibrated_pass_frac_seeds": calibration["pass_frac_seeds"],
        "DUR_KICK_ms": float(DUR_KICK),
        "rationale": calibration["rationale"],
        "sweep_parameters": config["sweep_parameters"],
        "core_config": config["core_config"],
        "selector_constants": thresholds,
        "sweep_summary": calibration["sweep_summary"],
        "rep_bin_indices": rep_bin_indices,
        "per_bin_results": {str(k): v for k, v in results_by_bin.items()},
        # per_bin_results records carry the 4-control raw summaries + the split sources:
        #   DIFFERENCED (core_kick - core_only): source_resp, downstream_resp,
        #     n_activated_bins, r95_mm, far_field_frac
        #   RAW core_kick: returned, runaway, frac_time_on_post, pass_frac_seeds
        #   RAW core_only confound: core_only_quiet, core_only_source_resp,
        #     core_only_downstream_resp, core_only_frac_time_on_post
        #   CONTROLS (NOT in gate): kick_only_downstream, no_core_no_kick_downstream,
        #     kick_only_minus_no_core_downstream
        #   plus kick_boost, win_ms, window_after_dur_kick, kick_dur_overlap_ms, net_response
    }

    json_path = os.path.join(out_dir, "kick_calibration.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote calibration JSON → {json_path}")

    _make_figures(results_by_bin, calibration, out_dir)
    print(f"Wrote figures → {os.path.join(out_dir, 'figures/')}")
    print(f"\nCalibration result (GO):")
    print(f"  kick_boost = {calibration['kick_boost']}")
    print(f"  win_ms     = {calibration['win_ms']}")
    print(f"  rationale  : {calibration['rationale']}")


def main() -> None:
    parser = _build_argparser()
    args = parser.parse_args()

    # Fail fast (before network build): --emit-ea-bins needs a single named source so the
    # ea_net_bins.npz is one (n_kick, n_seed, n_bins) matrix, not a silently-truncated
    # multi-bin bare-sheet sweep. --kick-xy guarantees the single-source collapse.
    if args.emit_ea_bins and args.kick_xy is None:
        raise SystemExit("--emit-ea-bins requires --kick-xy (one source bin per invocation)")
    if args.emit_ea_aux and not args.emit_ea_bins:
        raise SystemExit("--emit-ea-aux requires --emit-ea-bins "
                         "(sham-rate + first-onset sidecar to ea_net_bins.npz)")

    # Guard: Task 1 module must exist before the calibration sweep can run.
    try:
        import importlib
        importlib.import_module("src.topic4_propagation_operator")
    except ImportError:
        print(
            "[run_m3_kick_calibration] Task 1 module not yet available — "
            "run after src/topic4_propagation_operator.py lands.\n"
            "This skeleton is syntactically valid and structurally complete; "
            "re-run once Task 1 is committed."
        )
        sys.exit(0)

    if not args.run:
        print(
            "[run_m3_kick_calibration] PILOT-FIRST gate: the calibration sweep is a "
            "measurement that feeds the (to-be-frozen) pre-registration and is NOT run "
            "by default. Pass --run to execute it (intended for the Task 1.5 round, "
            "after pilot discussion). Nothing was run."
        )
        sys.exit(0)

    run_calibration(args)


if __name__ == "__main__":
    main()
