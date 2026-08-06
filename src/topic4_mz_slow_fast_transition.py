"""Topic 4 MZ slow–fast dynamical transition — pure, import-safe, testable functions.

Design contract (BINDING):
  docs/superpowers/specs/2026-07-20-topic4-mz-slow-fast-transition-design.md

Tier = model-side mechanism analysis. Every phenotype is a *detection label*; we test whether a frozen fast
system crosses a repeatable OPERATIONAL-runaway boundary (120 Hz / 100 ms). NOT seizure validation.

This module is SIDE-EFFECT-FREE (no sims, no file writes — those live in
scripts/run_topic4_mz_slow_fast_transition.py). It holds only pure helpers:

  branch_rng_state   — deterministic independent PCG64 future-noise branch state (P_runaway, design §3.1)
  wilson_ci          — Wilson score interval for the P_runaway proportion
  recovery_time      — fast-rate return-to-band time after a subthreshold pulse (design §3.3)
  state_step_schedule / matched_d_times — checkpoint step indices (design §2)
  classify_transition — result-neutral 5-label transition classifier (design §5)

Simulation primitives (MZOnsetProbe, run_loop checkpoint/resume, score_runaway, epsilon_c_from_ladder) are
REUSED from src.topic4_mz_onset_dynamics by the runner — NOT reimplemented here, NO engine edits.
"""
from __future__ import annotations

import hashlib

import numpy as np

SCHEMA_VERSION = "mz-slow-fast-transition-1.0"


# ============================================================ P_runaway replay branches (design §3.1)
def branch_rng_state(seed, cond, state, idx):
    """A PCG64 ``bit_generator.state`` dict for one independent future-noise replay branch.

    Deterministic in ``(seed, cond, state, idx)`` and reproducible ACROSS processes (stable SHA-256 key,
    never the salted builtin ``hash``). Distinct ``idx`` -> distinct stream. Swappable directly into a
    ``LoopState.rng_state`` (run_loop restores ``rng.bit_generator.state``), so a frozen checkpoint can be
    replayed under different future noise while V / currents / z / m stay identical."""
    key = f"{int(seed)}|{cond}|{state}|{int(idx)}".encode()
    digest = hashlib.sha256(key).digest()
    entropy = [int.from_bytes(digest[i:i + 4], "little") for i in range(0, 16, 4)]   # 4 x uint32
    ss = np.random.SeedSequence(entropy)
    return np.random.default_rng(np.random.PCG64(ss)).bit_generator.state


def wilson_ci(k, n, z=1.96):
    """Wilson score interval (lo, hi) for a binomial proportion k/n, clipped to [0,1]. n=0 -> (nan, nan)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


# ============================================================ fast-rate recovery time (design §3.3)
def recovery_time(rate_hz, dt, pulse_off_idx, band_lo, band_hi, *, smooth_ms=20.0, min_hold_ms=50.0):
    """Time (ms after pulse offset) for the smoothed E-rate to return to the pre-pulse band
    [band_lo, band_hi] and STAY inside it for ``min_hold_ms``. None (censored) if it never re-enters within
    the trace — i.e. the perturbed frozen state ran away or stayed elevated. Smoothing is a 20-ms EMA (same
    constant as score_runaway). The band is supplied by the caller (pre_mean +/- k*pre_std of the frozen
    state's pre-pulse window), keeping this function pure/testable."""
    r = np.asarray(rate_hz, float)
    n = r.size
    if n == 0:
        return None
    alpha = 1.0 - np.exp(-dt / smooth_ms)
    ema = np.empty(n)
    acc = float(r[0])                                   # seed at first sample (no spurious ramp from 0)
    for i in range(n):
        acc += alpha * (r[i] - acc)
        ema[i] = acc
    hold = max(1, int(round(min_hold_ms / dt)))
    start = int(pulse_off_idx)
    in_band = (ema >= band_lo) & (ema <= band_hi)
    for i in range(start, n - hold + 1):
        if in_band[i:i + hold].all():
            return float((i - start) * dt)
    return None


# ============================================================ checkpoint schedules (design §2)
def state_step_schedule(onset_ms, dt):
    """Matched-time registered-checkpoint step indices anchored to this seed's z-only onset ``onset_ms``
    (design §2.1). ``first_crossing`` is per-condition and appended later by the runner. Returned dict is in
    chronological order."""
    o = float(onset_ms)
    times = {
        "baseline_1000ms": 1000.0,
        "mid_fraction": 0.50 * o,
        "pre_onset_2000ms": o - 2000.0,
        "pre_onset_1000ms": o - 1000.0,
        "pre_onset_500ms": o - 500.0,
        "pre_onset_200ms": o - 200.0,
        "pre_onset_100ms": o - 100.0,
    }
    return {k: int(round(v / float(dt))) for k, v in times.items()}


def matched_d_times(D_trace, t_ms, targets):
    """First time (ms) each D target is reached on the natural trajectory (design §2.2). Never reached ->
    None (censored: e.g. plateau never crosses the higher targets)."""
    D = np.asarray(D_trace, float)
    t = np.asarray(t_ms, float)
    out = {}
    for tg in targets:
        hit = np.where(D >= float(tg))[0]
        out[float(tg)] = float(t[hit[0]]) if hit.size else None
    return out


# ============================================================ result-neutral transition classifier (design §5)
# First-principles thresholds (NOT tuned on any result): a P_runaway "step" > half the [0,1] range; epsilon_c
# "near zero" below the smallest real ignition rung (0.025); a "drop"/"rise" of at least one rung / 20 ms.
_P_STEP = 0.5          # adjacent-by-D P_runaway jump that counts as a sharp threshold
_P_SMOOTH_RANGE = 0.4  # total P_runaway range that counts as a real rise
_P_FLAT_LOW = 0.2      # P_runaway max below this = frozen states essentially never self-ignite under noise
_EPS_ZERO = 0.03       # epsilon_c at/below this ~ spontaneous ignition (below smallest rung 0.025)
_EPS_DROP = 0.03       # epsilon_c fall (low-D -> high-D) that counts as "easier to ignite as D grows"
_TAU_RISE = 20.0       # tau_rec increase (ms) toward high D that counts as critical slowing


def classify_transition(per_state, *, natural_crosses, plateau_outside):
    """Result-NEUTRAL label for how the frozen fast system behaves as the natural slow state advances
    (design §5). No category is a gate; returns {"label", "features"} with the features used, so the verdict
    is auditable. ``per_state``: list of dicts ordered baseline->onset, keys D, p_runaway, epsilon_c (None=
    censored), tau_rec (None=censored). ``natural_crosses`` / ``plateau_outside`` are the cross-condition
    context (does this condition's natural run cross; does plateau stay below)."""
    resolved = [s for s in per_state if s.get("p_runaway") is not None and np.isfinite(s.get("p_runaway"))]
    n_res = len(resolved)
    feats = dict(n_resolved=n_res)
    if n_res < 3:
        feats["reason"] = "too_few_resolved"
        return dict(label="unresolved", features=feats)

    order = sorted(resolved, key=lambda s: s["D"])
    pv = [float(s["p_runaway"]) for s in order]
    p_max = max(pv)
    p_range = p_max - min(pv)
    max_jump = max((pv[i + 1] - pv[i]) for i in range(len(pv) - 1))

    eps = [float(s["epsilon_c"]) for s in order if s.get("epsilon_c") is not None]
    eps_min = min(eps) if eps else None
    eps_near_zero = eps_min is not None and eps_min <= _EPS_ZERO
    eps_decreasing = len(eps) >= 2 and (eps[-1] - eps[0]) <= -_EPS_DROP

    tau = [float(s["tau_rec"]) for s in order if s.get("tau_rec") is not None]
    tau_increasing = len(tau) >= 2 and (tau[-1] - tau[0]) >= _TAU_RISE

    p_steep = max_jump >= _P_STEP
    p_smooth = (p_range >= _P_SMOOTH_RANGE) and (max_jump < _P_STEP)
    p_flat_low = p_max < _P_FLAT_LOW

    feats.update(p_max=round(p_max, 4), p_range=round(p_range, 4), max_adjacent_p_jump=round(max_jump, 4),
                 eps_min=eps_min, eps_near_zero=eps_near_zero, eps_decreasing=eps_decreasing,
                 tau_increasing=tau_increasing, p_steep=p_steep, p_smooth=p_smooth, p_flat_low=p_flat_low,
                 natural_crosses=bool(natural_crosses), plateau_outside=bool(plateau_outside))

    if p_steep and eps_near_zero and tau_increasing and natural_crosses and plateau_outside:
        label = "dynamical_tipping"
    elif p_flat_low and eps_decreasing:
        label = "finite_amplitude_escape"
    elif p_smooth and not eps_near_zero:
        label = "noise_driven_escape"
    elif p_range < _P_FLAT_LOW and not eps_decreasing and not eps_near_zero:
        label = "smooth_crossover"
    else:
        label = "unresolved"
    return dict(label=label, features=feats)
