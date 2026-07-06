"""Tests for src/topic5_axis_robustness.py (Topic 5 TA/TB field-reversal §6a Option-B
axis-level robustness supplement -- reframed claim: reading propagation direction by
electrode/shaft order (coordinate-blind) can badly mislead on some subjects; real
coordinates avoid it; smoothing (field) adds nothing beyond plain coordinate LS).

Spec: docs/superpowers/specs/2026-07-06-topic5-tatb-field-reversal-design.md §6a
Pilot (throwaway, functions lifted from here): scripts/pilot_topic5_axis_robustness.py
Pilot report: .superpowers/sdd/pilot_axis_report.md
"""
from __future__ import annotations

import numpy as np
import pytest

from src.propagation_contact_plane_readout import make_plane_grid, S_THRESH
from src.topic5_event_resolved_alignment import field_from_contact_values
from src.topic5_axis_robustness import (
    raw_contact_axis,
    field_axis_from_field,
    sequence_axis,
    axis_angle,
    cos_unit,
    held_out_axis_score,
    axis_robustness_splits,
    random_axis_null_score,
)


def _two_shaft_synthetic(offset=0.1, slope=1.0):
    """Two shafts (x=0 "A", x=1 "B"), 4 contacts each at the SAME y = [-0.9,-0.3,0.3,0.9]
    -- a balanced factorial design in (x, y). value = offset*I(shaft==B) + slope*y: a small,
    consistent BETWEEN-shaft offset riding on top of a large WITHIN-shaft y-slope (the
    1146-style failure mechanism, spec §1: naive shaft-identity reading picks up the small
    cross-shaft offset while missing the dominant along-shaft trend). Collapsing every
    contact to its shaft's mean value (sequence_axis) destroys the y-slope entirely and keeps
    only the small offset -> reads "mostly x"; fitting the RAW per-contact values recovers
    the true "mostly y" gradient, since offset=0.1 << slope*range=1.8. The balanced factorial
    (every y value appears on BOTH shafts) also makes x and y perfectly uncorrelated, which
    the held_out_axis_score test below reuses for an exact-zero orthogonal-axis score."""
    ys = [-0.9, -0.3, 0.3, 0.9]
    names = [f"A{i}" for i in range(4)] + [f"B{i}" for i in range(4)]
    plane_xy = {}
    cav = {}
    for i, y in enumerate(ys):
        plane_xy[f"A{i}"] = (0.0, y)
        plane_xy[f"B{i}"] = (1.0, y)
        cav[f"A{i}"] = {"value": slope * y, "support": 1.0}
        cav[f"B{i}"] = {"value": offset + slope * y, "support": 1.0}
    return names, plane_xy, cav


def test_sequence_axis_diverges_from_raw_contact_while_field_agrees():
    """(a) Synthetic 1146-style geometry: sequence_axis (shaft-mean, coordinate-blind) reads
    "mostly cross-shaft" (only the small 0.1 offset survives collapsing); raw_contact_axis
    (real coordinates, no smoothing) reads "mostly along-shaft" (the dominant slope=1.0
    y-trend) -- a large angular divergence. field_axis (smoothing THEN the same LS fit)
    should closely AGREE with raw_contact_axis: smoothing a near-planar signal does not
    change its gradient direction, so "having real coordinates" (raw OR field, either) is
    what avoids the misread -- smoothing itself is not the differentiator (spec §6a)."""
    names, plane_xy, cav = _two_shaft_synthetic()
    plane_ref = {"channels": [{"name": n, "x_norm": plane_xy[n][0], "y_norm": plane_xy[n][1],
                              "support": 1.0} for n in names]}
    X, Y = make_plane_grid()

    rc = raw_contact_axis(cav, plane_xy)
    seq = sequence_axis(cav, plane_xy)
    assert rc["ok"] and seq["ok"]

    values = {n: cav[n]["value"] for n in names}
    support = {n: cav[n]["support"] for n in names}
    field = field_from_contact_values(plane_ref, values, support_by_name=support,
                                      sigma=0.5, X=X, Y=Y, s_thresh=S_THRESH)
    fa = field_axis_from_field(field, X, Y, S_THRESH)
    assert fa["ok"]

    divergence = axis_angle(seq["unit"], rc["unit"])
    agreement = cos_unit(rc["unit"], fa["unit"])
    assert divergence > 45.0        # large angle: coordinate-blind reading misled
    assert agreement > 0.9          # raw contact and field closely agree (no smoothing needed)


def test_held_out_axis_score_high_for_matching_low_for_orthogonal():
    """(b) held_out_axis_score = Spearman(projection of each contact's REAL (x,y) position
    onto a train-half axis, that contact's held-out per-contact mean value). Held-out values
    here = y itself (pure y-trend); by the balanced-factorial layout in _two_shaft_synthetic,
    x and y are exactly uncorrelated. An axis pointing along the true gradient (0,1) must
    score near-perfect; an axis orthogonal to it (1,0) must score near-zero."""
    names, plane_xy, _ = _two_shaft_synthetic()
    held_values = {n: plane_xy[n][1] for n in names}

    matching = held_out_axis_score((0.0, 1.0), plane_xy, held_values)
    orthogonal = held_out_axis_score((1.0, 0.0), plane_xy, held_values)

    assert matching["n_common"] == len(names)
    assert orthogonal["n_common"] == len(names)
    assert matching["rho"] > 0.99
    assert abs(orthogonal["rho"]) < 0.05

    # a non-finite train axis (unresolved split) must not crash -- NaN score, not an exception
    unresolved = held_out_axis_score((float("nan"), float("nan")), plane_xy, held_values)
    assert np.isnan(unresolved["rho"])


def _balanced_grid_synthetic():
    """8 x-levels x 5 y-levels = 40 points; EVERY y value appears at EVERY x value -> x and y
    are exactly uncorrelated by construction (same "balanced factorial" idea as
    _two_shaft_synthetic, just more points). held value = y itself (pure y-trend). More points
    than _two_shaft_synthetic (n=8) on purpose: at n=8, Spearman rho from a random direction is
    quantized into only a handful of discrete rungs (confirmed by hand: the median of 200
    random-direction draws jumps between roughly -0.49/0/+0.49 depending on seed) -- not a fair
    testbed for "a random direction sits near the null's own median." At n=40 the null
    distribution is smooth enough for that check to be seed-stable."""
    xs_levels = range(8)
    ys_levels = (-2, -1, 0, 1, 2)
    plane_xy = {}
    held_values = {}
    i = 0
    for xv in xs_levels:
        for yv in ys_levels:
            name = f"C{i}"; i += 1
            plane_xy[name] = (float(xv), float(yv))
            held_values[name] = float(yv)
    return plane_xy, held_values


def test_random_axis_null_score_real_axis_clears_null_random_direction_sits_near_it():
    """(c) random_axis_null_score is the chance-level null for held_out_axis_score: "does a
    real axis predict held-out order better than a random direction would, given THIS subject's
    OWN contact geometry?" -- not an assumed textbook rho=0, but the median of MANY (n_random)
    actual random-direction draws on the SAME plane. On the balanced grid (x, y exactly
    uncorrelated by construction) with held value = y itself: the REAL matching axis (0,1) must
    clear the null median by a wide margin (it captures the y-trend perfectly); a direction
    ORTHOGONAL to the trend (1,0) -- itself simply one instance of "a random direction" -- must
    land close to the null's own median, not conspicuously above or below it."""
    plane_xy, held_values = _balanced_grid_synthetic()
    rng = np.random.default_rng(7)

    null = random_axis_null_score(plane_xy, held_values, n_random=200, rng=rng)
    assert null["n_random"] == 200
    assert null["n_common"] == len(plane_xy)
    assert np.isfinite(null["median"])

    matching = held_out_axis_score((0.0, 1.0), plane_xy, held_values)["rho"]
    orthogonal = held_out_axis_score((1.0, 0.0), plane_xy, held_values)["rho"]

    assert matching - null["median"] > 0.5          # real axis clears the null by a wide margin
    assert abs(orthogonal - null["median"]) < 0.2    # an uninformative direction sits ~at null


def test_axis_robustness_splits_independent_axis_failure_does_not_block_others():
    """axis_robustness_splits must score each axis type INDEPENDENTLY per split: a
    single-shaft montage makes sequence_axis structurally degenerate on EVERY split (the
    shaft-mean collapse gives every contact on the one shaft the same constant value -> zero
    gradient), but raw_contact_axis and field_axis are perfectly resolvable on a single shaft
    that carries a real y-gradient. sequence_axis's failure must not silently discard
    raw/field's valid splits (an earlier all-three-or-nothing design did exactly that on 2
    real narrow-substrate subjects with single-shaft montages -- epilepsiae_139 and
    yuquan_zhangjiaqi -- discovered by running the cohort on real data)."""
    n_ch = 6
    names = [f"S{i}" for i in range(1, n_ch + 1)]     # ALL on shaft "S" -- single shaft
    ys = np.linspace(-0.9, 0.9, n_ch)
    # x_norm constant (a genuinely straight single shaft) -- NOT jittered as a function of the
    # same index driving y: doing so would make x and y exactly collinear (6 points, 2 free
    # parameters), an ill-conditioned design matrix (cond ~1e17) that makes the "should be
    # exactly zero" collapsed-target gradient come out as floating-point noise ABOVE the 1e-12
    # threshold instead of below it -- a test-fixture artifact, not a real single-shaft property.
    plane_ref = {"channels": [{"name": n, "x_norm": 0.0, "y_norm": float(y), "support": 1.0}
                              for n, y in zip(names, ys)]}
    rng0 = np.random.default_rng(11)
    n_ev = 60
    true = np.linspace(0.0, 1.0, n_ch)
    masked = true[:, None] + rng0.normal(0, 0.05, (n_ch, n_ev))
    bundle = {"masked": masked, "labels": np.zeros(n_ev, dtype=int),
              "channel_names": names, "bools": np.isfinite(masked)}
    X, Y = make_plane_grid()
    rng = np.random.default_rng(3)
    splits = axis_robustness_splits(bundle, plane_ref, 0, X=X, Y=Y, sigma=0.4,
                                    n_split=20, rng=rng)

    assert len(splits) > 0
    assert all(np.isnan(d["sequence_rho"]) for d in splits)      # single shaft: always degenerate
    n_raw_ok = sum(1 for d in splits if np.isfinite(d["raw_rho"]))
    n_field_ok = sum(1 for d in splits if np.isfinite(d["field_rho"]))
    assert n_raw_ok > 0 and n_field_ok > 0     # raw/field stay usable despite sequence failing


def test_axis_angle_is_sign_aware_not_folded():
    """axis_angle is the SIGNED angle in degrees in [0, 180] (spec §6a divergence distribution
    reports e.g. 148.8° for a near-opposite pair) -- it must NOT fold via abs(cos) or
    min(angle, 180-angle) down to the <=90 range. Opposite unit vectors must read 180, not 0."""
    assert axis_angle((1.0, 0.0), (1.0, 0.0)) == pytest.approx(0.0, abs=1e-6)
    assert axis_angle((1.0, 0.0), (0.0, 1.0)) == pytest.approx(90.0, abs=1e-6)
    assert axis_angle((1.0, 0.0), (-1.0, 0.0)) == pytest.approx(180.0, abs=1e-6)
    assert np.isnan(axis_angle((float("nan"), 0.0), (1.0, 0.0)))
