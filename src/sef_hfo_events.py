"""Step 1 SEF-HFO: noise-driven spontaneous-event detection + OU noise drive.

Detector thresholds are LOCKED here per the frozen contract
``docs/archive/topic4/sef_itp_phase4_v2/step1_noise_contract_2026-06-03.md`` §1.
Changing any threshold requires editing that doc's changelog too.

The detector consumes the ``(ext, front[, rE_final])`` returned by the canonical
``src.sef_hfo_lif.integrate_lif_field`` — it does NOT re-implement any dynamics
(single-track discipline; dual-track was removed in commit e95af61).
"""
from __future__ import annotations

import numpy as np

from src.sef_hfo_field import isotropic_gaussian, convolve_periodic

# ---------------------------------------------------------------------------
# LOCKED detector thresholds (contract §1, v1.1 coherence amendment) — fraction, ms, kHz
# ---------------------------------------------------------------------------
# The activity series fed to the detector is the COHERENCE active-fraction
# (smoothed-field threshold crossing, coh_len ~ ELL_PAR), NOT the raw per-pixel
# active-fraction — per-pixel OU noise speckle inflates the raw measure (smoke
# 2026-06-03: σ=2.0 speckle raw_ext 0.048 / coh_ext 0.034 vs coherent pulse
# coh_ext 0.142). EVENT_ON_FRAC is set ABOVE the σ=2.0 speckle coh-floor (0.034)
# and below the validated coherent-pulse coh peak (0.142): a noise-floor+margin
# threshold, NOT tuned to yield discrete events.
EVENT_ON_FRAC: float = 0.05    # coherence-active-fraction "event in progress" threshold
MIN_DUR_MS: float = 8.0        # minimum ON duration to count as an event
MERGE_GAP_MS: float = 12.0     # ON intervals closer than this are one event
RETURN_FRAC: float = 0.2       # self-terminated if activity falls to <= RETURN_FRAC*peak
CAPTURE_FRAC: float = 0.5      # (window B) settled mean past lo+0.5*(hi-lo) -> captured
RUNAWAY_FRAC: float = 0.5      # max activity >= this and not returned -> runaway
SUSTAINED_MS: float = 400.0    # single ON interval longer than this (not returned) -> sustained
SETTLE_MS: float = 50.0        # post-event window used to judge return-to-low
# Temporal separation (advisor 2026-06-03): "discrete_events" requires events to
# be temporally SEPARATED (mostly quiescent), not continuous flicker. If activity
# is ON more than this fraction of the run, the run is sustained, not discrete.
FRAC_TIME_ON_MAX: float = 0.30


# ---------------------------------------------------------------------------
# OU spatiotemporal noise drive
# ---------------------------------------------------------------------------

def make_ou_noise(n: int, L: float, dt: float, sigma_noise: float,
                  tau_noise: float = 100.0, ell_noise: float = 0.5, seed: int = 0):
    """Return a stateful ``stim_fn(t_ms)`` producing an OU noise field added to muE.

    NOTE on ``tau_noise`` default (2026-06-03): the OU term on mu represents the
    SLOW afferent-input component (tens-hundreds of ms). It must NOT be set near
    TAU_AMPA (~3.5-5 ms): fast synaptic fluctuations are already inside Phi_LIF's
    sigma, so a fast tau_noise double-counts them and drives continuous ignition
    (sustained, never discrete -- see step1_noise_contract §9.1). Default is now
    100 ms (slow); the validated discrete band used 50-200 ms.

    Per-pixel Ornstein-Uhlenbeck in time (correlation time ``tau_noise`` ms,
    steady-state std ``sigma_noise`` mV) with white innovations smoothed by a
    Gaussian of correlation length ``ell_noise`` mm (renormalized so the smoothed
    per-pixel innovation has unit std, keeping the OU steady-state std == sigma_noise).

    Contract: ``integrate_lif_field`` calls ``stim_fn`` exactly once per dt step in
    increasing time order, so the closure advances one OU step per call.

    ``sigma_noise == 0`` returns the scalar ``0.0`` every call (no rng draw) →
    the field run is byte-identical to the deterministic run. Reproducible by seed.
    """
    if sigma_noise == 0:
        return lambda t_ms: 0.0

    rng = np.random.default_rng(seed)
    a = dt / tau_noise
    b = np.sqrt(2.0 * dt / tau_noise)
    use_smooth = ell_noise > 0
    K = isotropic_gaussian(n, L, ell_noise) if use_smooth else None

    # Calibrate the smoothing gain so smoothed unit-white -> unit std.
    if use_smooth:
        probe = rng.standard_normal((n, n))
        gain = convolve_periodic(probe, K).std() / max(probe.std(), 1e-12)
        norm = 1.0 / max(gain, 1e-12)
    else:
        norm = 1.0

    state = {"x": np.zeros((n, n))}

    def stim_fn(t_ms):
        white = rng.standard_normal((n, n))
        xi = convolve_periodic(white, K) * norm if use_smooth else white
        state["x"] = state["x"] - a * state["x"] + b * sigma_noise * xi
        return state["x"]

    return stim_fn


# ---------------------------------------------------------------------------
# Event detection
# ---------------------------------------------------------------------------

def _on_intervals(ext: np.ndarray, dt: float):
    """Contiguous ON runs (ext > EVENT_ON_FRAC), merged across gaps < MERGE_GAP_MS.

    Returns a list of (start_step, end_step) inclusive index pairs.
    """
    idx = np.where(ext > EVENT_ON_FRAC)[0]
    if idx.size == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0]
    groups = np.split(idx, splits + 1)
    ivs = [(int(g[0]), int(g[-1])) for g in groups]
    merge_steps = MERGE_GAP_MS / dt
    merged = [ivs[0]]
    for s, e in ivs[1:]:
        if s - merged[-1][1] <= merge_steps:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


def detect_events(ext: np.ndarray, dt: float):
    """Detect discrete events from the active-fraction series.

    Returns a list of dicts: ``t_on, t_off, dur_ms, peak_ext, returned`` —
    ``returned`` flags self-termination (ext drops to <= RETURN_FRAC*peak within
    SETTLE_MS after the interval ends, or the run ends already below that).
    Only intervals with dur >= MIN_DUR_MS are returned.
    """
    events = []
    n = len(ext)
    for s, e in _on_intervals(ext, dt):
        dur = (e - s + 1) * dt
        if dur < MIN_DUR_MS:
            continue
        peak = float(ext[s:e + 1].max())
        settle_end = min(n, e + 1 + int(SETTLE_MS / dt))
        post = ext[e + 1:settle_end]
        if post.size > 0:
            returned = float(post.min()) <= RETURN_FRAC * max(peak, 1e-12)
        else:
            returned = float(ext[e]) <= RETURN_FRAC * max(peak, 1e-12)
        events.append(dict(t_on=float(s * dt), t_off=float(e * dt), dur_ms=float(dur),
                           peak_ext=peak, returned=bool(returned)))
    return events


def classify_run(ext: np.ndarray, dt: float, op: dict,
                 rE_final: np.ndarray | None = None, window: str = "A") -> dict:
    """Classify a single noise run into one regime (contract §1).

    Priority: sustained > runaway > captured_high (window B) > extinction_only >
    discrete_events. ``rE_final`` (from ``integrate_lif_field(..., return_field=True)``)
    is required for the window-B capture check (uses ``op["roots"]`` lo/hi).
    """
    events = detect_events(ext, dt)
    n_events_total = len(events)
    n_self_term = sum(1 for ev in events if ev["returned"])
    # discrete requires EVERY ON segment to self-terminate -- one normal event plus
    # a trailing non-returning segment is NOT discrete (it didn't fully self-limit).
    all_returned = n_events_total > 0 and n_self_term == n_events_total
    max_ext = float(ext.max())
    final_ext = float(ext[-1])
    returned_global = final_ext <= RETURN_FRAC * max(max_ext, 1e-12)

    ivs = _on_intervals(ext, dt)
    longest_on = max(((e - s + 1) * dt for s, e in ivs), default=0.0)
    frac_time_on = float((ext > EVENT_ON_FRAC).mean())

    captured = False
    if window == "B" and rE_final is not None and len(op.get("roots", [])) >= 2:
        lo = op["roots"][0]["nuE"]
        hi = op["roots"][-1]["nuE"]
        captured = float(rE_final.mean()) > lo + CAPTURE_FRAC * (hi - lo)

    if longest_on > SUSTAINED_MS and not returned_global:
        label = "sustained"
    elif max_ext >= RUNAWAY_FRAC and not returned_global:
        label = "runaway"
    elif captured:
        label = "captured_high"
    elif n_events_total == 0:
        label = "extinction_only"
    elif frac_time_on >= FRAC_TIME_ON_MAX:
        label = "sustained"   # continuously active / flickering -> not temporally discrete
    elif not all_returned:
        label = "sustained"   # >=1 ON segment did not self-terminate -> not discrete
    else:
        label = "discrete_events"   # >=1 event AND every ON segment self-terminated

    return dict(label=label, n_events=int(n_self_term), n_events_total=int(n_events_total),
                all_returned=bool(all_returned), max_ext=max_ext,
                final_ext=final_ext, longest_on_ms=float(longest_on),
                frac_time_on=frac_time_on,
                returned=bool(returned_global), captured_high=bool(captured),
                events=events)
