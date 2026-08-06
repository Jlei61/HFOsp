"""Topic 5 — TA/TB field-reversal §6a Option-B axis-level robustness supplement.

Spec: docs/superpowers/specs/2026-07-06-topic5-tatb-field-reversal-design.md §6a

Reframed claim (NOT "field denoises"): reading propagation direction by electrode/shaft
order (coordinate-blind) can badly mislead on some subjects; using real coordinates
(contact-level) avoids it; smoothing (field) adds nothing beyond plain coordinate LS.

Lifted (functions only, cleaned up) from the pilot: scripts/pilot_topic5_axis_robustness.py.
Pilot report (n=5, exploratory): .superpowers/sdd/pilot_axis_report.md.

Three axes, same shared 2D plane (P0), same weighted-least-squares gradient estimator
(value ~ 1 + x + y, weighted), differing ONLY in what per-contact "value" is fit:
  - raw_contact_axis: the contact's own class-aggregate value (coordinate-aware, no smoothing).
  - field_axis:       the smoothed field's value at supported grid pixels (smoothing added).
  - sequence_axis:    every contact collapsed to its own SHAFT's support-weighted mean value
                      (coordinate-BLIND -- only "which shaft" survives, not within-shaft
                      position). This is the stand-in for a naive electrode-order-only read.
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr

from src.propagation_contact_plane_readout import S_THRESH
from src.propagation_skeleton_geometry import parse_shaft

MIN_POINTS = 3   # same floor as MIN_PLANE_CONTACTS elsewhere in this pipeline
N_RANDOM_AXES = 200   # random-direction draws per split for the chance-level null (spec: "MANY, e.g. 200")


# --------------------------------------------------------------------------- shared estimator
def weighted_ls_gradient_2d(value, x, y, weight) -> dict:
    """value_i ~ 1 + x_i + y_i, weighted LS (weight_i). Returns the unit vector pointing
    toward increasing value (early->late by construction: moving along +unit increases
    predicted value). ok=False if <MIN_POINTS usable rows or the fitted gradient is ~0
    (no resolvable direction)."""
    value = np.asarray(value, float); x = np.asarray(x, float)
    y = np.asarray(y, float); weight = np.asarray(weight, float)
    finite = np.isfinite(value) & np.isfinite(x) & np.isfinite(y) & np.isfinite(weight) & (weight > 0)
    n = int(finite.sum())
    if n < MIN_POINTS:
        return {"beta": (float("nan"), float("nan")), "unit": (float("nan"), float("nan")),
                "n": n, "ok": False}
    v = value[finite]; xx = x[finite]; yy = y[finite]; w = weight[finite]
    A = np.column_stack([np.ones_like(xx), xx, yy])
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(A * sw[:, None], v * sw, rcond=None)
    _b0, bx, by = coef
    mag = float(np.hypot(bx, by))
    if mag < 1e-12:
        return {"beta": (float(bx), float(by)), "unit": (float("nan"), float("nan")),
                "n": n, "ok": False}
    return {"beta": (float(bx), float(by)), "unit": (float(bx / mag), float(by / mag)),
            "n": n, "ok": True}


# --------------------------------------------------------------------------- the three axes
def raw_contact_axis(cav: dict, plane_xy: dict) -> dict:
    """Coordinate LS, NO smoothing: contact's own class-aggregate value ~ 1+x+y, weight=support."""
    names = [n for n in cav if n in plane_xy and np.isfinite(cav[n]["value"])]
    val = np.array([cav[n]["value"] for n in names], float)
    w = np.array([cav[n]["support"] for n in names], float)
    x = np.array([plane_xy[n][0] for n in names], float)
    y = np.array([plane_xy[n][1] for n in names], float)
    return weighted_ls_gradient_2d(val, x, y, w)


def field_axis_from_field(field: dict, X: np.ndarray, Y: np.ndarray, s_thresh: float = S_THRESH) -> dict:
    """Smooth-first LS: field value ~ 1+x+y over supported grid pixels, weight=field support S.
    Same estimator as raw_contact_axis -- differs ONLY by the prior spatial-smoothing step."""
    if field is None:
        return {"beta": (float("nan"), float("nan")), "unit": (float("nan"), float("nan")),
                "n": 0, "ok": False}
    mask = field["S"] >= s_thresh
    return weighted_ls_gradient_2d(field["T"][mask], X[mask], Y[mask], field["S"][mask])


def sequence_axis(cav: dict, plane_xy: dict) -> dict:
    """COORDINATE-BLIND direction: group contacts by shaft (parse_shaft, electrode identity
    only), collapse every contact's value to its OWN SHAFT's support-weighted mean value (so
    within-shaft spatial position carries zero information -- two contacts on the same shaft
    get the identical value regardless of where on the shaft they sit), then feed that
    shaft-collapsed value into the SAME weighted_ls_gradient_2d estimator raw_contact_axis
    uses. If early contacts happen to sit on one shaft and late contacts on another (the
    1146 failure mode), this reads as the inter-shaft direction rather than the true
    within-shaft gradient."""
    names = [n for n in cav if n in plane_xy and np.isfinite(cav[n]["value"])]
    shaft_of = {n: parse_shaft(n)[0] for n in names}
    by_shaft: Dict[object, list] = {}
    for n in names:
        by_shaft.setdefault(shaft_of[n], []).append(n)
    shaft_mean = {}
    for s, ns in by_shaft.items():
        vals = np.array([cav[n]["value"] for n in ns], float)
        ws = np.array([cav[n]["support"] for n in ns], float)
        wsum = float(ws.sum())
        shaft_mean[s] = float(np.sum(vals * ws) / wsum) if wsum > 0 else float(np.mean(vals))
    seq_val = np.array([shaft_mean[shaft_of[n]] for n in names], float)
    w = np.array([cav[n]["support"] for n in names], float)
    x = np.array([plane_xy[n][0] for n in names], float)
    y = np.array([plane_xy[n][1] for n in names], float)
    out = weighted_ls_gradient_2d(seq_val, x, y, w)
    out["shaft_mean"] = shaft_mean
    out["n_shafts"] = len(by_shaft)
    return out


# --------------------------------------------------------------------------- angle / cosine
def cos_unit(u1, u2) -> float:
    """Signed cosine between two unit vectors. NaN if either is non-finite."""
    a = np.asarray(u1, float); b = np.asarray(u2, float)
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return float("nan")
    return float(np.dot(a, b))


def axis_angle(a, b) -> float:
    """Sign-aware angle in degrees in [0, 180] between two unit vectors (NOT folded via
    abs(cos) / min(angle, 180-angle) down to <=90 -- an anti-parallel pair must read ~180,
    not ~0). NaN if either vector is non-finite."""
    c = cos_unit(a, b)
    if not np.isfinite(c):
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


# --------------------------------------------------------------------------- held-out score
def held_out_axis_score(unit: Tuple[float, float], plane_xy: Dict[str, tuple],
                        held_values: Dict[str, float], *, min_points: int = MIN_POINTS) -> dict:
    """score = Spearman(projection of each contact's REAL (x, y) position onto `unit`, that
    contact's held_values entry). `unit` is typically a TRAIN-half axis (any of the three
    axis types above); `held_values` is typically a HELD-OUT-half per-contact mean value --
    the projection always uses the contact's real coordinates (even when `unit` came from
    sequence_axis, which was FIT on shaft-collapsed values: this is exactly what tests
    whether the shaft-collapsed direction generalizes to real per-contact structure).
    Contacts = names present in BOTH plane_xy and held_values with a finite held value.
    NaN rho (n_common still reported) if `unit` is non-finite or <min_points common contacts."""
    ux, uy = float(unit[0]), float(unit[1])
    common = [n for n in plane_xy if n in held_values and np.isfinite(held_values[n])]
    if not (np.isfinite(ux) and np.isfinite(uy)) or len(common) < min_points:
        return {"rho": float("nan"), "n_common": len(common)}
    xs = np.array([plane_xy[n][0] for n in common], float)
    ys = np.array([plane_xy[n][1] for n in common], float)
    proj = xs * ux + ys * uy
    held = np.array([held_values[n] for n in common], float)
    rho = spearmanr(proj, held).correlation
    return {"rho": (float(rho) if np.isfinite(rho) else float("nan")), "n_common": len(common)}


# --------------------------------------------------------------------------- random-axis null
def random_axis_null_score(plane_xy: Dict[str, tuple], held_values: Dict[str, float], *,
                           n_random: int = N_RANDOM_AXES, rng: np.random.Generator,
                           min_points: int = MIN_POINTS) -> dict:
    """The chance-level NULL for held_out_axis_score: does a real axis (raw_contact / field /
    sequence) predict held-out propagation order better than a RANDOM direction would, given
    THIS subject's own contact geometry? Draws `n_random` unit directions uniformly on the same
    shared 2D plane (angle ~ Uniform[0, 2*pi)) and scores EACH one with the exact same
    held_out_axis_score used for the three real axes -- same real-coordinate projection, same
    Spearman-vs-held-out-mean, same min_points/finite handling -- so every comparison is
    apples-to-apples (no bespoke re-implementation that could silently drift from the real-axis
    scorer's contract).

    Returns {"rhos": [...finite scores...], "median": float, "n_random": int, "n_common": int}.
    `median` is this subject's (or this split's) OWN chance level given its particular contact
    layout -- NOT an assumed textbook zero: a lopsided or clustered contact geometry can shift
    the chance level away from 0 (this is exactly why the null must be drawn per-subject/
    per-split from the SAME contacts, not asserted analytically). `n_common` mirrors
    held_out_axis_score's own field (identical on every draw -- it depends only on
    plane_xy/held_values, never on the direction) so callers can apply the same eligibility
    floor. `median` is NaN if no draw resolves (mirrors held_out_axis_score's own NaN
    convention when common contacts < min_points)."""
    thetas = rng.uniform(0.0, 2.0 * np.pi, size=n_random)
    rhos = []
    n_common = 0
    for theta in thetas:
        unit = (float(np.cos(theta)), float(np.sin(theta)))
        scored = held_out_axis_score(unit, plane_xy, held_values, min_points=min_points)
        n_common = scored["n_common"]     # same every draw: depends only on plane_xy/held_values
        if np.isfinite(scored["rho"]):
            rhos.append(scored["rho"])
    median = float(np.median(rhos)) if rhos else float("nan")
    return {"rhos": rhos, "median": median, "n_random": int(n_random), "n_common": int(n_common)}


# --------------------------------------------------------------------------- split harness
def _aggregate_over_events(masked: np.ndarray, names: Sequence[str], cols: np.ndarray) -> dict:
    """Per-contact masked-rank mean over the given event columns -> {name:{value,support}}
    (same construction as topic5_field_reversal._aggregate_over_events; kept local per the
    pilot's own precedent rather than importing a leading-underscore cross-module name)."""
    sub = masked[:, cols]
    with np.errstate(invalid="ignore"):
        val = np.where(np.all(np.isnan(sub), axis=1), np.nan, np.nanmean(sub, axis=1))
    sup = np.isfinite(sub).mean(axis=1)
    return {n: {"value": float(val[i]), "support": float(sup[i])} for i, n in enumerate(names)}


def axis_robustness_splits(bundle: dict, plane_ref: dict, label: int, *, X, Y, sigma,
                           n_split: int, rng: np.random.Generator,
                           s_thresh: float = S_THRESH, n_random_axes: int = N_RANDOM_AXES) -> list:
    """For ONE class label (TA or TB): n_split random halves of THAT class's events.
    train-half -> raw_contact_axis + field_axis + sequence_axis (same sigma/grid/s_thresh as
    the primary stat path, spec §3.1 P0). held-out-half -> per-contact mean value. Each split
    contributes ONE row {"raw_rho","field_rho","sequence_rho","null_rho","n_common"}, scored by
    held_out_axis_score against the SAME held-out half and the SAME common-contact set
    (n_common depends only on plane_xy ∩ held_values, not on any axis's unit vector, so it is
    identical across the three real-axis scores whenever they are computed).

    `null_rho` = random_axis_null_score(...)["median"]: the SAME held-out half, the SAME
    plane_xy, scored against `n_random_axes` random directions instead of a fitted axis -- this
    split's own chance-level baseline. Drawn from the SAME `rng` passed in (not a fresh
    generator), so the whole cohort run stays deterministic given one top-level seed.

    Each REAL axis type is scored INDEPENDENTLY per split (NaN for whichever axis fails to
    resolve): sequence_axis is structurally degenerate whenever a class plane has a single
    shaft (shaft-mean collapse -> one constant value -> zero gradient, on EVERY split, by
    construction) -- that must not discard perfectly good raw_contact/field scores for the
    SAME split just because sequence_axis happened to fail. A split is dropped entirely only
    if NONE of the three real axes resolve, or fewer than min_points held-out contacts are
    common (null_rho is computed only for KEPT splits, same drop rule as before -- unchanged)."""
    from src.topic5_event_resolved_alignment import field_from_contact_values, build_plane_xy

    masked = bundle["masked"]; names = list(bundle["channel_names"])
    labels = np.asarray(bundle["labels"])
    cols = np.where(labels == label)[0]
    plane_xy = build_plane_xy(plane_ref)
    out = []
    for _ in range(n_split):
        perm = rng.permutation(cols)
        half = perm.size // 2
        if half < 1:
            continue
        train_cav = _aggregate_over_events(masked, names, perm[:half])
        held_cav = _aggregate_over_events(masked, names, perm[half:])
        held_values = {n: d["value"] for n, d in held_cav.items() if np.isfinite(d["value"])}
        n_common = len([n for n in plane_xy if n in held_values])
        if n_common < MIN_POINTS:
            continue

        rc = raw_contact_axis(train_cav, plane_xy)
        seq = sequence_axis(train_cav, plane_xy)
        v = {n: d["value"] for n, d in train_cav.items()}
        s = {n: d["support"] for n, d in train_cav.items()}
        field = field_from_contact_values(plane_ref, v, support_by_name=s, sigma=sigma,
                                          X=X, Y=Y, s_thresh=s_thresh)
        fa = field_axis_from_field(field, X, Y, s_thresh)

        raw_rho = held_out_axis_score(rc["unit"], plane_xy, held_values)["rho"] if rc["ok"] else float("nan")
        field_rho = held_out_axis_score(fa["unit"], plane_xy, held_values)["rho"] if fa["ok"] else float("nan")
        seq_rho = held_out_axis_score(seq["unit"], plane_xy, held_values)["rho"] if seq["ok"] else float("nan")
        if not (np.isfinite(raw_rho) or np.isfinite(field_rho) or np.isfinite(seq_rho)):
            continue
        null_rho = random_axis_null_score(plane_xy, held_values, n_random=n_random_axes, rng=rng)["median"]
        out.append({"raw_rho": raw_rho, "field_rho": field_rho, "sequence_rho": seq_rho,
                    "null_rho": null_rho, "n_common": n_common})
    return out
