"""Topic 5 A-line direction-rose helpers (pure, TDD).

The rose figure normalizes each subject's MEAN seizure early-activation gradient direction to
0 degrees (the black reference), then shows the per-interictal-EVENT propagation directions split
by template A / B as two full-circle hollow histograms. Two templates that are opposite ends of
ONE bidirectional axis appear as two lobes ~180 deg apart, both collinear with the seizure axis.

Everything here is sign-AWARE (full [0, 2pi)): direction = where the scalar (rank or activation)
INCREASES across the 2D contact plane. The A-line cohort statistic is the sign-FREE |corr|; this
rose is its directional, per-event visual companion (lobe at 180 deg = reverse-collinear, still
'aligned' under the sign-free statistic).
"""
from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi


def gradient_angle(x, y, values):
    """Direction (radians, [0, 2pi)) in which `values` INCREASE across the 2D plane, via a
    least-squares plane fit values ~ a*x + b*y. NaN if <3 finite points or no gradient."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    v = np.asarray(values, float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
    if ok.sum() < 3 or np.nanstd(v[ok]) < 1e-12:
        return np.nan
    X = np.column_stack([x[ok] - x[ok].mean(), y[ok] - y[ok].mean()])
    vv = v[ok] - v[ok].mean()
    beta, *_ = np.linalg.lstsq(X, vv, rcond=None)
    if np.linalg.norm(beta) < 1e-12:
        return np.nan
    return float(np.mod(np.arctan2(beta[1], beta[0]), TWO_PI))


def event_angles_by_template(event_vals, x, y, labels):
    """Per-event gradient angles, grouped by template label.

    event_vals : (n_ch, n_ev) per-event scalar already aligned to the (x, y) contact points
                 (NaN for non-participating channels in that event).
    labels     : (n_ev,) ints; 0 = template A, 1 = template B, -1 = unassigned (dropped).
    Returns {0: np.array([...]), 1: np.array([...])} of finite angles (degenerate events dropped).
    """
    event_vals = np.asarray(event_vals, float)
    labels = np.asarray(labels)
    if event_vals.ndim != 2:
        raise ValueError("event_vals must be 2D (n_ch, n_ev)")
    if labels.shape[0] != event_vals.shape[1]:
        raise ValueError("labels length must equal n_ev")
    out = {0: [], 1: []}
    for e in range(event_vals.shape[1]):
        lbl = int(labels[e])
        if lbl not in (0, 1):
            continue
        ang = gradient_angle(x, y, event_vals[:, e])
        if np.isfinite(ang):
            out[lbl].append(ang)
    return {k: np.asarray(v, float) for k, v in out.items()}


def circular_mean(angles):
    """Mean direction (radians, [0, 2pi)) of a set of angles; NaN if empty/all-NaN/zero resultant."""
    a = np.asarray(angles, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan
    c, s = np.cos(a).mean(), np.sin(a).mean()
    if np.hypot(c, s) < 1e-12:
        return np.nan
    return float(np.mod(np.arctan2(s, c), TWO_PI))


def resultant_length(angles):
    """Circular concentration R in [0, 1] (1 = all identical, 0 = uniform). NaN if empty."""
    a = np.asarray(angles, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan
    return float(np.hypot(np.cos(a).mean(), np.sin(a).mean()))


def rotate_to_reference(angles, ref):
    """Rotate angles so that `ref` maps to 0; result wrapped to [0, 2pi). NaN-safe (NaN stays NaN)."""
    a = np.asarray(angles, float)
    return np.mod(a - ref, TWO_PI)


def axial_mean(angles):
    """AXIAL (mod-pi) mean direction in [0, pi): treats theta and theta+pi as the SAME axis by
    doubling the angle (u = e^{i*2*theta}) before averaging, then halving. Unlike `circular_mean`,
    a bidirectional set {theta, theta+pi} does NOT cancel — it gives the axis. NaN if empty / the
    doubled-angle resultant is ~0 (e.g. two orthogonal axes with no net axis). This is the stable
    reference for a SIGN-FREE statistic (the A-line / rose seizure axis)."""
    a = np.asarray(angles, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan
    c, s = np.cos(2 * a).mean(), np.sin(2 * a).mean()
    if np.hypot(c, s) < 1e-12:
        return np.nan
    return float(0.5 * np.mod(np.arctan2(s, c), TWO_PI))


def axial_resultant_length(angles):
    """Axial concentration R_axial in [0,1] = |mean(e^{i*2*theta})|. 1 = all on one axis (incl. a
    perfectly bidirectional set), 0 = uniform / no axis. NaN if empty."""
    a = np.asarray(angles, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan
    return float(np.hypot(np.cos(2 * a).mean(), np.sin(2 * a).mean()))
