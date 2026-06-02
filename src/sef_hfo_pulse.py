# src/sef_hfo_pulse.py
"""SEF-HFO Step-0b finite-pulse response: wavefront-aware classifier, adaptive thresholds."""
import numpy as np
from src.sef_hfo_field import build_kernels, make_Feff_lookup, integrate_field, _grid

def _centroid_x(frame, rest, detect):
    above = frame > (rest + detect)
    if not above.any(): return np.nan
    xs = np.where(above)[1]; return float(xs.mean())

def classify_response(activity, stim_mask, rest_level, runaway_frac=0.5, return_frac=0.2,
                      detect=0.05, travel_cells=1.0):
    above = activity > (rest_level + detect)
    extent = above.reshape(above.shape[0], -1).mean(axis=1)
    stim_extent = float(stim_mask.mean()); max_extent = float(extent.max()); final_extent = float(extent[-1])
    cx = np.array([_centroid_x(f, rest_level, detect) for f in activity])
    cx0 = next((c for c in cx if not np.isnan(c)), np.nan)
    moved = np.nanmax(np.abs(cx - cx0)) if not np.isnan(cx0) else 0.0
    grew = max_extent > 1.5 * stim_extent
    returned = final_extent <= return_frac * max(max_extent, 1e-12)
    if max_extent >= runaway_frac:
        return "runaway"
    if grew and moved < travel_cells:
        return "global_synchronous"          # extent grew but front did NOT travel -> not propagation
    if grew and moved >= travel_cells and not returned:
        return "runaway"
    if grew and moved >= travel_cells and returned:
        return "self_limited_propagation"
    if returned:
        return "extinction"
    return "local_bump"


DETECTABLE = ("local_bump", "self_limited_propagation", "global_synchronous", "runaway")
PULSE_FAMILY = {"radii": (3.0, 6.0), "durations": (1.0, 3.0)}   # single location (patch-approx)

def _disk_mask(p, r):
    X, Y = _grid(p.n, p.L); return ((X**2 + Y**2) <= r**2).astype(float)

def run_pulse(p, op, I_E, I_I, r, T, A, dt, t_max, return_activity=False, **cls_kw):
    mask = _disk_mask(p, r)
    def stim_fn(t): return (A * mask) if t < T else 0.0 * mask
    act = integrate_field(p, op, I_E, I_I, stim_fn, dt, t_max)
    label = classify_response(act, mask, op["r_E0"], **cls_kw)
    return (label, act) if return_activity else label

def _first_with(label_fn, a_grid, predicate):
    for A in sorted(a_grid):
        if predicate(label_fn(A)):
            return float(A)
    return np.inf

def amplitude_thresholds(label_fn, a_lo, a_hi, n_coarse=12, n_refine=6):
    """REAL coarse->refine: label_fn(A) MUST run a fresh pulse at A (no nearest-label
    lookup). Separately resolve A_event (first detectable), A_self_limited (first
    'self_limited_propagation' — the gate quantity), A_runaway (first 'runaway').
    safety_margin = A_runaway - A_self_limited (NOT A_event): local_bump / global
    synchronous must not count toward the margin (review #3)."""
    coarse = list(np.linspace(a_lo, a_hi, n_coarse))
    def resolve(predicate):
        a_coarse = _first_with(label_fn, coarse, predicate)
        if not np.isfinite(a_coarse):
            return np.inf
        lo = max(a_lo, a_coarse - (a_hi - a_lo) / n_coarse)
        fine = np.linspace(lo, a_coarse, n_refine)        # endpoint a_coarse already satisfies
        return _first_with(label_fn, fine, predicate)     # real pulses at fine amplitudes
    A_event = resolve(lambda L: L in DETECTABLE)
    A_self = resolve(lambda L: L == "self_limited_propagation")
    A_run = resolve(lambda L: L == "runaway")
    return {"A_event": A_event, "A_self_limited": A_self, "A_runaway": A_run,
            "safety_margin": A_run - A_self}
