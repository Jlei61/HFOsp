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
# Per-operating-point amplitude-bar recalibration (contract §10.2)
# ---------------------------------------------------------------------------
# EVENT_ON_FRAC above was calibrated ONCE at drive=1.0 (rest νE~0.22 Hz). At a
# near-zero-rest operating point (e.g. drive 0.6, rest ~0.02 Hz) slow OU noise
# can hover ABOVE that fixed bar BETWEEN events -> frac_time_on inflates -> false
# "sustained" (§9.7 confound). The (drive×σ) joint analysis recalibrates the bar
# PER operating point between the σ=0/σ_ref floor and the deterministic-kick
# event peak. Only this AMPLITUDE bar recalibrates; the shape/time constants
# above (MIN_DUR/MERGE_GAP/RETURN_FRAC/SUSTAINED/SETTLE/FRAC_TIME_ON_MAX) are
# operating-point invariant and stay fixed.

class UndetectableOperatingPoint(ValueError):
    """The σ=0/σ_ref noise floor is not strictly below the deterministic-kick
    event peak — noise reaches the event amplitude, so no separable detection bar
    exists at this operating point (contract §10.2 A loud-fail; also marks a
    drive as non-excitable / non-interictal per §10.3)."""


def event_on_frac_from_refs(floor: float, peak: float, frac: float = 0.5) -> float:
    """Per-operating-point coherence-active-fraction bar (contract §10.2 A).

    Lock the amplitude bar at ``floor + frac*(peak-floor)`` — the pre-registered
    midpoint between the σ=0/σ_ref noise floor and the deterministic-kick event
    peak, both measured at THIS operating point. Anti-circular: ``floor`` and
    ``peak`` come only from no-noise / sub-threshold references and the
    deterministic kick, NEVER from the noise-driven event grid.

    Raises ``UndetectableOperatingPoint`` when ``floor`` is not strictly below
    ``peak``.
    """
    if not floor < peak:
        raise UndetectableOperatingPoint(
            f"noise floor {floor:.4g} not below event peak {peak:.4g}")
    return float(floor + frac * (peak - floor))


def calibrate_detector(ref_series, kick_series, frac: float = 0.5) -> dict:
    """Per-operating-point detector bar from coherence reference + kick series (§10.2/§10.5).

    Parameters
    ----------
    ref_series : list of 1-D arrays
        Coherence active-fraction (``ext_coh``) series from the no-event reference
        runs at THIS operating point: the σ=0 deterministic run and the σ_ref
        sub-threshold-noise run. The floor is the worst case = max over all of them
        (so the bar sits above even the highest sub-threshold noise hovering — the
        §9.7 mechanism).
    kick_series : 1-D array
        ``ext_coh`` from the deterministic finite-pulse run = the genuine event peak.
    frac : float
        Midpoint fraction (§10.2 A, default 0.5).

    Returns ``dict(floor, peak, event_on_frac)``. Raises ``UndetectableOperatingPoint``
    if the floor is not below the peak. Anti-circular: consumes ONLY the reference
    and kick series, never the noise-driven event grid.
    """
    floor = max(float(np.max(np.asarray(s))) for s in ref_series)
    peak = float(np.max(np.asarray(kick_series)))
    bar = event_on_frac_from_refs(floor, peak, frac)
    return dict(floor=floor, peak=peak, event_on_frac=bar)


# Acceptance band (contract §10.4): a (drive, σ) cell counts toward the discrete
# REGION iff a robust majority of seeds are discrete AND their rate is in the data
# magnitude band. Machined here (not eyeballed off a heatmap) so the verdict can't
# drift — per the saved lesson "acceptance gates must encode the conclusion".
ACCEPT_FRAC_MIN: float = 0.60       # >=60% of seeds discrete
ACCEPT_RATE_LO: float = 0.01        # events/s — data magnitude band [0.01, 1]
ACCEPT_RATE_HI: float = 1.00


def accepted_cell(frac_discrete: float, mean_rate_discrete,
                  frac_min: float = ACCEPT_FRAC_MIN,
                  rate_lo: float = ACCEPT_RATE_LO, rate_hi: float = ACCEPT_RATE_HI) -> bool:
    """True iff this (drive, σ) cell meets the §10.4 region criterion: >= frac_min
    of seeds discrete AND the discrete-seed mean rate within [rate_lo, rate_hi]/s.
    ``mean_rate_discrete is None`` (no discrete seeds) -> not accepted."""
    if mean_rate_discrete is None:
        return False
    return frac_discrete >= frac_min and rate_lo <= mean_rate_discrete <= rate_hi


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

def _on_intervals(ext: np.ndarray, dt: float, event_on_frac: float = EVENT_ON_FRAC):
    """Contiguous ON runs (ext > event_on_frac), merged across gaps < MERGE_GAP_MS.

    ``event_on_frac`` defaults to the module constant (validated drive=1.0 bar);
    the (drive×σ) joint runner passes the per-operating-point recalibrated bar
    (``event_on_frac_from_refs``, contract §10.2).

    Returns a list of (start_step, end_step) inclusive index pairs.
    """
    idx = np.where(ext > event_on_frac)[0]
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


def detect_events(ext: np.ndarray, dt: float, event_on_frac: float = EVENT_ON_FRAC):
    """Detect discrete events from the active-fraction series.

    ``event_on_frac``: per-operating-point bar (contract §10.2); defaults to the
    module constant.

    Returns a list of dicts: ``t_on, t_off, dur_ms, peak_ext, returned`` —
    ``returned`` flags self-termination (ext drops to <= RETURN_FRAC*peak within
    SETTLE_MS after the interval ends, or the run ends already below that).
    Only intervals with dur >= MIN_DUR_MS are returned.
    """
    events = []
    n = len(ext)
    for s, e in _on_intervals(ext, dt, event_on_frac):
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
                 rE_final: np.ndarray | None = None, window: str = "A",
                 event_on_frac: float = EVENT_ON_FRAC) -> dict:
    """Classify a single noise run into one regime (contract §1).

    Priority: sustained > runaway > captured_high (window B) > extinction_only >
    discrete_events. ``rE_final`` (from ``integrate_lif_field(..., return_field=True)``)
    is required for the window-B capture check (uses ``op["roots"]`` lo/hi).

    ``event_on_frac``: per-operating-point amplitude bar (contract §10.2);
    defaults to the module constant (validated drive=1.0 bar). The (drive×σ)
    joint runner MUST pass the recalibrated per-op bar — a forgotten arg falling
    back to the fixed bar is the §9.7 confound at off-1.0 operating points.
    """
    events = detect_events(ext, dt, event_on_frac)
    n_events_total = len(events)
    n_self_term = sum(1 for ev in events if ev["returned"])
    # discrete requires EVERY ON segment to self-terminate -- one normal event plus
    # a trailing non-returning segment is NOT discrete (it didn't fully self-limit).
    all_returned = n_events_total > 0 and n_self_term == n_events_total
    max_ext = float(ext.max())
    final_ext = float(ext[-1])
    returned_global = final_ext <= RETURN_FRAC * max(max_ext, 1e-12)

    ivs = _on_intervals(ext, dt, event_on_frac)
    longest_on = max(((e - s + 1) * dt for s, e in ivs), default=0.0)
    frac_time_on = float((ext > event_on_frac).mean())

    captured = False
    if window == "B" and rE_final is not None and len(op.get("roots", [])) >= 2:
        lo = op["roots"][0]["nuE"]
        hi = op["roots"][-1]["nuE"]
        # LIMITATION (window B is SENSITIVITY-tier, does not gate the window-A main
        # line): this uses the SINGLE final-frame field mean, not a post-event /
        # settled-window mean. A final frame caught mid-fluctuation can mis-call
        # capture either way. Acceptable for the current screen; before any SERIOUS
        # window-B report, upgrade to a settled-window mean (average rE over the last
        # ~SETTLE_MS, which needs the integrator to expose a final-window field avg).
        captured = float(rE_final.mean()) > lo + CAPTURE_FRAC * (hi - lo)

    # `reason` records WHICH rule produced the label — the `sustained` label has
    # three distinct mechanisms (long_plateau / too_frequent / non_returning) that
    # imply different science (§1; the §11 disambiguation depends on this).
    if longest_on > SUSTAINED_MS and not returned_global:
        label, reason = "sustained", "long_plateau"
    elif max_ext >= RUNAWAY_FRAC and not returned_global:
        label, reason = "runaway", "runaway"
    elif captured:
        label, reason = "captured_high", "captured_high"
    elif n_events_total == 0:
        label, reason = "extinction_only", "extinction"
    elif frac_time_on >= FRAC_TIME_ON_MAX:
        label, reason = "sustained", "too_frequent"   # ON >30% of the run -> not temporally discrete
    elif not all_returned:
        label, reason = "sustained", "non_returning"  # >=1 ON segment did not self-terminate
    else:
        label, reason = "discrete_events", "discrete"  # event(s) AND every ON segment self-terminated

    return dict(label=label, reason=reason,
                n_events=int(n_self_term), n_events_total=int(n_events_total),
                all_returned=bool(all_returned), max_ext=max_ext,
                final_ext=final_ext, longest_on_ms=float(longest_on),
                frac_time_on=frac_time_on,
                returned=bool(returned_global), captured_high=bool(captured),
                events=events)
