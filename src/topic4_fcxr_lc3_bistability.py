"""Reading a frozen-state map: where the network has two answers, and where it jumps.

A finite stochastic spiking network cannot be continued or have its fixed points proved, and the
plan says the main figure does not need that.  What it can show is the operational evidence:

* **bistability** -- the same frozen state settles low from a low start and high from a high start;
* **hysteresis** -- the boundary sits at a different place going up than coming down;
* **an abrupt jump** -- activity changes discontinuously across the boundary rather than sliding;

which together license the phrase "saddle-node-like transition in a finite stochastic network" and
do not license "a saddle-node bifurcation".

The one thing this must not do is call a point bistable because the two runs got different labels
for a reason other than the state: with one probe per point per start, a label that flips because
a burst landed on one side of the window is noise, so the criterion is on where the activity sits,
not on which of five regime names came back.
"""
from __future__ import annotations

import numpy as np

HIGH_REGIMES = ("R1_runaway", "R2_bounded_high", "R3_carrier", "R4_burst_train")
LOW_REGIMES = ("R0_interictal_only",)


def is_high(row, af_ratio_min=3.0, interictal_ceiling=None):
    """Whether a probe ended up on a high branch, decided by activity rather than by its label.

    A regime name can flip between neighbouring points for reasons that are not the state.  The
    recruited fraction cannot: a high branch sits several times above the interictal spread.
    """
    if row.get("regime") in LOW_REGIMES:
        return False
    ceil = interictal_ceiling if interictal_ceiling is not None else row.get(
        "interictal_ceiling_af")
    if ceil is None or not np.isfinite(ceil) or ceil <= 0:
        return row.get("regime") in HIGH_REGIMES
    return bool(row.get("mean_af", 0.0) >= af_ratio_min * ceil)


def bistable_points(rows, **kw):
    """Points whose two starting states disagree, with the numbers behind each disagreement."""
    by_point = {}
    for r in rows:
        by_point.setdefault((r["alpha_d"], r["alpha_x"]), {})[r["ic"]] = r
    out = []
    for (ad, ax), pair in sorted(by_point.items()):
        if set(pair) != {"interictal", "ictal"}:
            continue
        lo, hi = is_high(pair["interictal"], **kw), is_high(pair["ictal"], **kw)
        out.append(dict(alpha_d=ad, alpha_x=ax, bistable=bool(hi and not lo),
                        from_interictal="high" if lo else "low",
                        from_ictal="high" if hi else "low",
                        mean_af_interictal=pair["interictal"].get("mean_af"),
                        mean_af_ictal=pair["ictal"].get("mean_af"),
                        regime_interictal=pair["interictal"].get("regime"),
                        regime_ictal=pair["ictal"].get("regime")))
    return out


def boundary_along(points, axis, ic, favours_high=None):
    """Where the high branch *begins* along one axis, for one starting state.

    Hysteresis lives in where the branch starts, not where it ends: taking the extreme high
    coordinate in the direction the grid runs out would return the edge of the grid for both
    starting states and show no difference however large the real one is.

    Which end is the beginning depends on the axis.  Raising disinhibition favours the high branch,
    so its boundary is the *lowest* D still high; raising the relay load opposes it, so its boundary
    is the *highest* X still high.  ``favours_high`` overrides that default for other axes.
    """
    other = "alpha_x" if axis == "alpha_d" else "alpha_d"
    key = "from_interictal" if ic == "interictal" else "from_ictal"
    if favours_high is None:
        favours_high = (axis == "alpha_d")
    pick = min if favours_high else max
    out = {}
    for p in points:
        if p[key] != "high":
            continue
        o = p[other]
        out[o] = p[axis] if o not in out else pick(out[o], p[axis])
    return {k: v for k, v in sorted(out.items())}


def jump_size(rows, af_ratio_min=3.0, interictal_ceiling=None):
    """The gap between the low and the high group, in units of the interictal spread.

    A transition that slides through intermediate values leaves this near 1; one that jumps leaves
    a gap, and reporting the number is what keeps "abrupt" from being an impression.
    """
    hi = [r["mean_af"] for r in rows
          if is_high(r, af_ratio_min=af_ratio_min, interictal_ceiling=interictal_ceiling)]
    lo = [r["mean_af"] for r in rows
          if not is_high(r, af_ratio_min=af_ratio_min, interictal_ceiling=interictal_ceiling)]
    if not hi or not lo:
        return dict(separated=False, reason="only one branch is present in the map")
    lo_hi, hi_lo = float(np.max(lo)), float(np.min(hi))
    return dict(separated=bool(hi_lo > lo_hi), gap_ratio=float(hi_lo / max(lo_hi, 1e-12)),
                highest_low=lo_hi, lowest_high=hi_lo,
                n_low=len(lo), n_high=len(hi))


def evidence_summary(rows, **kw):
    """The three operational statements the map can make, each with its own verdict."""
    pts = bistable_points(rows, **kw)
    n_bi = sum(1 for p in pts if p["bistable"])
    up = boundary_along(pts, "alpha_d", "interictal")
    down = boundary_along(pts, "alpha_d", "ictal")
    shared = sorted(set(up) & set(down))
    hyst = {k: dict(from_interictal=up[k], from_ictal=down[k]) for k in shared
            if up[k] != down[k]}
    return dict(
        n_points=len(pts), n_bistable=n_bi,
        bistability=("supported" if n_bi else "not seen"),
        hysteresis=("supported" if hyst else "not seen"), hysteresis_detail=hyst,
        jump=jump_size(rows, **kw),
        claim_allowed=("saddle-node-like transition in a finite stochastic network"
                       if n_bi and hyst else
                       "not enough for a saddle-node-like statement on this map"),
        claim_forbidden="a mathematically proven saddle-node bifurcation")
