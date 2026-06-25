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


# ===================== Task 2: target masks + dynamic clamp schedule =====================

def intervention_vth_at_time(base_vth, target_mask, is_E, t_ms, on_ms, off_ms, clamp_level=CLAMP_LEVEL):
    """Per-neuron V_th field at time `t_ms`. Inside [on_ms, off_ms): a COPY of base_vth with the
    target's E cells clamped to `clamp_level`; I cells never clamped. Outside the window (or when
    on_ms is None / target_mask is None): returns base_vth unchanged (same array -> zero-copy in the
    hot loop, which keeps the no-/pre-intervention path bit-identical to the engine)."""
    if on_ms is None or target_mask is None or not (on_ms <= t_ms < off_ms):
        return base_vth
    vth = np.asarray(base_vth, float).copy()
    vth[np.asarray(target_mask, bool) & np.asarray(is_E, bool)] = clamp_level
    return vth


def make_on_axis_target(pos, is_E, axis_unit, center, thickness):
    """On-axis intervention target: a band perpendicular to the propagation axis through `center`.
    Returns a full-network bool mask (E+I in the band; the E-only restriction is applied at clamp
    time by intervention_vth_at_time)."""
    return band_mask(pos, axis_unit, center, thickness)


def make_off_axis_target(pos, is_E, axis_unit, center, thickness, n_match, core_masks, rng, L,
                         mode="lateral"):
    """Count-matched off-axis control target: a mask of EXACTLY `n_match` E cells away from the
    propagation corridor. mode='lateral' -> strip parallel to the axis, offset to the side (clean:
    off corridor / off cores); mode='translate' -> perpendicular strip moved beyond a focus. Widens
    the band until it has >= n_match E candidates, then subsamples to exactly n_match. Raises if the
    chosen cells overlap either focus core."""
    pos = np.asarray(pos, float); is_E = np.asarray(is_E, bool)
    a = np.asarray(axis_unit, float); a = a / np.linalg.norm(a)
    center = np.asarray(center, float)
    perp = np.array([-a[1], a[0]])
    if mode == "lateral":
        normal, cpt = perp, center + 0.35 * L * perp
    elif mode == "translate":
        normal, cpt = a, center + 0.425 * L * a
    else:
        raise ValueError(f"mode must be 'lateral'|'translate', got {mode!r}")
    t = thickness
    band = band_mask(pos, normal, cpt, t)
    while int((band & is_E).sum()) < n_match and t < 4.0 * thickness:
        t *= 1.25
        band = band_mask(pos, normal, cpt, t)
    cand = np.flatnonzero(band & is_E)
    if cand.size < n_match:
        raise ValueError(f"off-axis band has {cand.size} E cells < n_match {n_match}")
    chosen = np.sort(rng.choice(cand, size=n_match, replace=False)) if cand.size > n_match else cand
    core_any = np.zeros(pos.shape[0], bool)
    for cm in core_masks:
        core_any |= np.asarray(cm, bool)
    if core_any[chosen].any():
        raise ValueError("off-axis target overlaps a focus core; increase the offset")
    mask = np.zeros(pos.shape[0], bool); mask[chosen] = True
    return mask


def make_static_deadzone_schedule():
    """Timing for the always-on static dead-zone control: clamp from t=0 to +inf (the placement
    upper bound -- if even a permanent band cannot block spread, no triggered strategy at that
    target is worth pursuing)."""
    return dict(on_ms=0.0, off_ms=float("inf"))
