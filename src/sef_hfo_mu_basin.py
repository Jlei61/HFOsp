"""R0–R4 regime classifier + R_event recruitment gain for the static-μ finite-event basin
pilot (Topic 4 M3). Spec: docs/archive/topic4/sef_hfo/m3_static_mu_pilot_2026-06-24.md §4-§5.

Pure functions on one event's scalar metrics. The R4a (W-aligned sustained, a bridge candidate)
vs R4b (tonic runaway, NEVER a bridge) split turns ONLY on whether the sustained phase keeps a
propagation front — that is the load-bearing discriminator. Thresholds are draft (audit-tunable);
they live in DEFAULT_CAPS so a future re-derivation is one edit.
"""
import numpy as np

__all__ = ["DEFAULT_CAPS", "classify_event", "r_event",
           "susceptibility_field", "apply_mu",
           "detect_events", "event_props", "aggregate_spontaneous"]


def detect_events(trace, thresh, min_gap_bins=1):
    """Spontaneous events in a 1-D population-activity trace = contiguous runs above `thresh`,
    merging runs separated by fewer than `min_gap_bins` quiet bins. Returns [(start, end), ...]
    (inclusive bin indices). No-kick long records feed this (the events are self-ignited)."""
    above = np.asarray(trace, dtype=float) > thresh
    runs, i, n = [], 0, len(above)
    while i < n:
        if above[i]:
            j = i
            while j + 1 < n and above[j + 1]:
                j += 1
            runs.append([i, j]); i = j + 1
        else:
            i += 1
    merged = []
    for r in runs:
        if merged and (r[0] - merged[-1][1] - 1) < min_gap_bins:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [tuple(r) for r in merged]


def event_props(trace, interval, bin_ms, n_record_bins):
    """Per-event scalar props from the trace: duration, peak active fraction, and returned (the
    event ended before the record ended) vs sustained (still active at the record's last bin)."""
    start, end = interval
    tr = np.asarray(trace, dtype=float)
    returned = bool(end < n_record_bins - 1)
    return {"duration_ms": float((end - start + 1) * bin_ms),
            "peak_active": float(np.max(tr[start:end + 1])),
            "returned": returned, "sustained": not returned}


def aggregate_spontaneous(n_events, record_ms, classes):
    """Aggregate spontaneous statistics over one record: event rate (Hz) + R-class fractions."""
    from collections import Counter
    c = Counter(classes)
    total = max(len(classes), 1)
    return {"event_rate_hz": (n_events / (record_ms / 1000.0)) if record_ms > 0 else float("nan"),
            "n_events": int(n_events),
            "frac": {k: c.get(k, 0) / total for k in ("R0", "R1", "R2", "R3", "R4a", "R4b")}}


def susceptibility_field(vth_core, vth0, core_mean, h_mode, rng):
    """Per-neuron h field in units of the mean core depression: h ≈ 1 inside the core, 0 outside.
    - core_susceptibility: h = clip(vth0 - vth_core, 0)/(vth0 - core_mean)  (the埋下的 pathology).
    - uniform: h = 1 everywhere (global-μ control).
    - shuffled: core_susceptibility h permuted in space (same depression multiset, broken location)."""
    base = np.clip(vth0 - np.asarray(vth_core, dtype=float), 0.0, np.inf) / max(vth0 - core_mean, 1e-9)
    if h_mode == "core_susceptibility":
        return base
    if h_mode == "uniform":
        return np.ones_like(base)
    if h_mode == "shuffled":
        return base[rng.permutation(len(base))]
    raise ValueError(f"unknown h_mode {h_mode!r}")


def apply_mu(vth_core, vth0, core_mean, mu, dvth_at_mu1, h_mode, rng):
    """Static-μ threshold permissivity: V_th_eff = vth_core - ΔVth(μ)·h, ΔVth(μ)=dvth_at_mu1·μ.

    μ=0 returns vth_core UNCHANGED (exact bit-parity — the short-circuit guarantees byte identity
    with the current runner). Engine untouched: the caller feeds V_th_eff to the core conditions.
    """
    vth_core = np.asarray(vth_core, dtype=float)
    if mu == 0:
        return vth_core.copy()
    h = susceptibility_field(vth_core, vth0, core_mean, h_mode, rng)
    return vth_core - dvth_at_mu1 * mu * h

DEFAULT_CAPS = {
    "R95_CAP": 6.0,        # EA r95 local cap (mirrors reclassify R95_LOCAL_CAP_MM)
    "FAR_CAP": 0.5,        # EA far-field cap
    "ACT_FLOOR": 1e-3,     # peak active fraction below this = failed/negligible ignition
    "FRONT_THRESH": 0.5,   # sustained_front_score >= this = retains a front (R4a) vs tonic (R4b)
}


def classify_event(m, caps=DEFAULT_CAPS):
    """Map one event's metrics dict to a regime label R0/R1/R2/R3/R4a/R4b.

    Decision tree (spec §4):
      R0  = no onset.
      R4a = not returned (sustained) AND sustained_front_score >= FRONT_THRESH (has front).
      R4b = not returned (sustained) AND sustained_front_score <  FRONT_THRESH (uniform tonic).
      R1  = onset but peak active fraction < ACT_FLOOR (failed/negligible ignition).
      R2  = returned, real, LOCAL (r95_ea <= R95_CAP AND far_ea <= FAR_CAP).
      R3  = returned, real, NOT local (large / near-critical, still returns).
    """
    if not m.get("event_detected", False):
        return "R0"
    sustained = bool(m.get("runaway", False)) or not bool(m.get("returned", True))
    if sustained:
        return "R4a" if m.get("sustained_front_score", 0.0) >= caps["FRONT_THRESH"] else "R4b"
    if m.get("active_peak", 0.0) < caps["ACT_FLOOR"]:
        return "R1"
    local = (m.get("r95_ea", np.inf) <= caps["R95_CAP"]
             and m.get("far_ea", np.inf) <= caps["FAR_CAP"])
    return "R2" if local else "R3"


def r_event(front_bins, next_active):
    """Finite-event recruitment gain = #newly-recruited bins (next-gen active not in the current
    front) / #front bins. NaN if the front is empty (undefined gain)."""
    front = set(front_bins)
    if not front:
        return float("nan")
    newly = set(next_active) - front
    return len(newly) / len(front)
