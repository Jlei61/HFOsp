"""Helpers for the Stage 3 event-triggered axial intervention probe.

Scientific question (causal sufficiency, NOT a mechanism claim): after a large interictal-like
event has already started spreading along the Stage 3 propagation axis, can an event-triggered
intervention on the propagation corridor stop further axial spread? v1 abstracts the intervention
as an idealized E-only threshold shutoff over a time window. See
docs/superpowers/specs/2026-06-25-stage3-deadzone-barrier-probe-design.md.

Module layout:
  - Pure geometry / source helpers (Task 1): no engine, no I/O -> unit-tested standalone.
  - Target masks + dynamic clamp schedule (Task 2).
  - Baseline eligibility + replay schedule (Task 3).
  - simulate_dynamic_vth (Task 4): a verbatim copy of the engine's simulate_kick loop with one
    addition (time-dependent per-neuron E-threshold). Engine imports are lazy (inside the function)
    so the pure helpers above import without the engine on path.
"""
import math
import numpy as np

CLAMP_LEVEL = 1e6   # mV; effectively non-excitable (V_th comparison only; V stays finite)


# ===================== Task 1: pure geometry + source helpers =====================

def band_mask(coords, normal_unit, center_point, thickness):
    """Bool mask of points within +/- thickness/2 of the plane through `center_point` with unit
    normal `normal_unit` -- a straight strip of width `thickness`. `coords` (M,2)."""
    coords = np.asarray(coords, float)
    n = np.asarray(normal_unit, float)
    n = n / np.linalg.norm(n)
    proj = (coords - np.asarray(center_point, float)) @ n
    return np.abs(proj) <= thickness / 2.0


def split_near_target_far(coords, axis_unit, center, source_focus, target_thickness):
    """Split points into near/target/far relative to the axial target band (at `center`, width
    `target_thickness`) oriented by the igniting `source_focus`. near = source side beyond the
    band, far = opposite side beyond the band, target = inside the band. near & far exclude the band."""
    coords = np.asarray(coords, float)
    a = np.asarray(axis_unit, float); a = a / np.linalg.norm(a)
    proj = (coords - np.asarray(center, float)) @ a
    src_proj = float((np.asarray(source_focus, float) - np.asarray(center, float)) @ a)
    src_sign = 1.0 if src_proj >= 0 else -1.0
    s = proj * src_sign
    half = target_thickness / 2.0
    return dict(near=s > half, target=np.abs(proj) <= half, far=s < -half)


def core_source_raw(on_neg, on_pos, delta_onset):
    """Core-level source label INDEPENDENT of read-out readability: looks ONLY at the two
    focus-core onset times. {neg, pos, collision, none}. None onset = that core never ignited
    (treated as +inf). Drops the stage3.label_event `readable` gate and returns 'none' (not
    'ambiguous') when neither core fires -- a successful stop must not vanish into 'ambiguous'."""
    neg = math.inf if on_neg is None else float(on_neg)
    pos = math.inf if on_pos is None else float(on_pos)
    if neg == math.inf and pos == math.inf:
        return "none"
    if abs(neg - pos) <= delta_onset:
        return "collision"
    return "neg" if neg < pos else "pos"


def participation_ratio(participate, region_mask, valid=None):
    """Fraction of (valid) region cells that participated. denom = sum(region & valid); numerator =
    sum(participate & region & valid). NaN if denom is 0. PASS `valid=free` (non-clamped cells) so an
    intervention that clamps cells inside the far region does not deflate its own far-ratio."""
    region = np.asarray(region_mask, bool)
    if valid is not None:
        region = region & np.asarray(valid, bool)
    denom = int(region.sum())
    if denom == 0:
        return float("nan")
    return float((np.asarray(participate, bool) & region).sum()) / denom


def exclude_target_contacts(valid, target_mask):
    """Valid contacts with the intervention-target-band contacts removed (read-out-exclusion
    control). Returns a new array; does not mutate `valid`."""
    return np.asarray(valid, bool) & ~np.asarray(target_mask, bool)
