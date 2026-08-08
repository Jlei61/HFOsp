"""Per-event profile shape: the observable Stage 3 rev5 fits (spec section 9.3).

The previous objective reduced each event to a direction sign and averaged
within sign. That step is where the information went: a source in the middle of
the sheet and a source at one end give profiles of completely different shape --
0% versus 94% of events monotone on the calibration sweep -- and averaging
within a sign label destroys exactly that difference.

So the observable here is the shape of each event's profile, computed by one
function on both the model and the patient side. It deliberately does not invert
to an ignition coordinate: a shape statistic only has to be *sensitive* to where
the source is, which is measured and true, while inverting requires the readout
to be *injective*, which is measured and false (see NOT_A_POSITION).
"""
from __future__ import annotations

import numpy as np

MIN_PARTICIPANTS = 6

OBJECTIVE_FEATURES = ("slope", "r2")
REPORT_ONLY = ("curvature", "n_part", "argmin_axial")

# Two different questions, two different gates -- conflating them is how a
# statistic gets banned on grounds it actually passes.
#
#   discrimination: do the values differ across known source positions?
#                   Needed to enter the objective. Measured on the 196-run
#                   sweep: slope 5.94, r2 4.34, curvature 1.38, argmin 4.38 --
#                   ALL pass, including argmin.
#   recovery:       does the value track the true position? Needed before any
#                   sentence of the form "the source is at x mm". Measured for
#                   argmin: regression slope 0.25, correlation 0.51, with five
#                   sources spanning 18 mm all reading +0.4 mm -- it FAILS.
#
# So argmin is not disqualified from the objective by calibration; it is kept
# out of the default feature set by a judgement, stated here so it can be
# argued with: on the patient side two contacts account for 31% of events and
# four for 50%, so the statistic is dominated by which contact is easiest to
# recruit. If the model's contact recruitability differs from the patient's for
# reasons unrelated to the field, including it would let the field absorb an
# instrumentation mismatch.
NOT_A_POSITION = ("argmin_axial",)

# Frozen so two calls are comparable. A distance that rescaled its bins to each
# sample would report "closer" merely because a sample got narrower.
SLOPE_EDGES = np.linspace(-1.5, 1.5, 16)
R2_EDGES = np.linspace(0.0, 1.0, 11)


def _pairs(ranks, axial, participating=None):
    out = []
    for name, rank in (ranks or {}).items():
        if rank is None or name not in axial:
            continue
        if participating is not None and name not in participating:
            continue
        out.append((float(axial[name]), float(rank)))
    return out


def event_shape(ranks, axial, participating=None, part_min=MIN_PARTICIPANTS):
    """Shape of one event's rank profile along the axis, or None if unusable.

    `participating` is the patient side's mask. The patient's rank matrix gives
    every channel a finite value whether or not it took part, so passing the
    mask is what keeps phantom ranks out; the model side leaves absent contacts
    as None and needs no mask.
    """
    pts = _pairs(ranks, axial, participating)
    if len(pts) < int(part_min):
        return None
    x = np.array([p[0] for p in pts], float)
    y = np.array([p[1] for p in pts], float)
    if x.std() < 1e-9 or y.std() < 1e-9:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    r2 = 1.0 - float((resid ** 2).sum() / ((y - y.mean()) ** 2).sum())
    curvature = float(np.polyfit(x, y, 2)[0]) if len(pts) >= 4 else float("nan")
    return dict(slope=float(slope), r2=r2, curvature=curvature,
                n_part=len(pts))


def argmin_axial_position(ranks, axial, participating=None,
                          part_min=MIN_PARTICIPANTS):
    """Axial position of the earliest contact.

    Kept for reporting. It may not be read as a location: see NOT_A_POSITION
    for the recovery-gate measurement that disqualifies that reading.
    """
    pts = _pairs(ranks, axial, participating)
    if len(pts) < int(part_min):
        return None
    return float(min(pts, key=lambda p: p[1])[0])


def shape_table(events, axial, participating=None, part_min=MIN_PARTICIPANTS):
    """Shapes for a list of events. Accepts raw rank dicts or event records."""
    rows = []
    for ev in events:
        ranks = ev.get("ranks") if isinstance(ev, dict) and "ranks" in ev else ev
        mask = participating(ev) if callable(participating) else participating
        s = event_shape(ranks, axial, mask, part_min)
        if s is not None:
            rows.append(s)
    return rows


def objective_features(shapes, features=OBJECTIVE_FEATURES):
    """Feature matrix for the objective.

    Every feature must have passed the discrimination gate; that is checked at
    calibration time, not here. This function only assembles.
    """
    missing = [f for f in features if shapes and f not in shapes[0]]
    if missing:
        raise ValueError(f"no such shape statistic: {missing}")
    return np.array([[float(s[f]) for f in features] for s in shapes], float)


def assert_not_interpreted_as_position(name):
    """Guard the sentence "the source is at x mm", not the feature matrix."""
    if name in NOT_A_POSITION:
        raise ValueError(
            f"{name!r} failed the recovery gate (regression slope 0.25 against "
            f"known source position) and must not be read as a location")


def recovery_score(estimates, truths, threshold=0.5):
    """Does an estimator in position units actually track the true position?

    Discrimination is not enough for a locational claim: an estimator can differ
    across positions while mapping several of them onto the same value, which is
    what the earliest-contact statistic does.
    """
    e, x = np.asarray(estimates, float), np.asarray(truths, float)
    ok = np.isfinite(e) & np.isfinite(x)
    if ok.sum() < 4 or x[ok].std() < 1e-9 or e[ok].std() < 1e-9:
        return dict(passed=False, slope=float("nan"), corr=float("nan"),
                    n=int(ok.sum()), threshold=float(threshold),
                    reason="not enough spread to judge")
    slope = float(np.polyfit(x[ok], e[ok], 1)[0])
    corr = float(np.corrcoef(x[ok], e[ok])[0, 1])
    return dict(passed=bool(slope >= threshold), slope=slope, corr=corr,
                n=int(ok.sum()), threshold=float(threshold))


def passes_sensitivity(groups, threshold=1.0):
    """Does this statistic separate known source positions?

    `groups` holds the statistic's values at each ground-truth source position.
    The gate compares spread between positions against spread within a position
    across network seeds: a statistic whose seed noise swamps its position
    signal cannot carry a spatial objective no matter how interpretable it looks.
    """
    groups = [np.asarray(g, float) for g in groups if len(np.asarray(g)) > 1]
    if len(groups) < 2:
        return dict(passed=False, between_over_within=float("nan"),
                    n_groups=len(groups), threshold=float(threshold),
                    reason="fewer than two usable positions")
    means = np.array([g.mean() for g in groups])
    between = float(means.std(ddof=1))
    within = float(np.sqrt(np.mean([g.var(ddof=1) for g in groups])))
    ratio = between / within if within > 0 else float("inf")
    return dict(passed=bool(ratio >= threshold), between_over_within=ratio,
                between=between, within=within, n_groups=len(groups),
                threshold=float(threshold))


def binned_distance(a, b, edges=(SLOPE_EDGES, R2_EDGES)):
    """Total-variation distance between two shape clouds on frozen bins.

    Frozen edges are the point: a distance that fitted its bins to each sample
    would call two clouds closer simply because one of them got narrower.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("both clouds must be (n, d) with the same d")
    e = [np.asarray(x, float) for x in edges][:a.shape[1]]
    ha, _ = np.histogramdd(a, bins=e)
    hb, _ = np.histogramdd(b, bins=e)
    pa = ha / ha.sum() if ha.sum() else ha
    pb = hb / hb.sum() if hb.sum() else hb
    return 0.5 * float(np.abs(pa - pb).sum())


def split_by_block(block_ids, frac=0.3, seed=0):
    """Hold out whole recordings, never individual events.

    Events inside one recording share a night, a brain state and an electrode
    impedance, so they are not independent; splitting by event would badly
    overstate how well a fit generalises.
    """
    block_ids = np.asarray(block_ids)
    blocks = np.unique(block_ids)
    rng = np.random.default_rng(seed)
    held = set(rng.permutation(blocks)[:max(1, int(round(len(blocks) * float(frac))))])
    mask = np.array([b in held for b in block_ids])
    return np.flatnonzero(~mask), np.flatnonzero(mask)


def assert_block_disjoint(block_ids, train_idx, test_idx):
    """Fail loudly if any recording appears on both sides."""
    block_ids = np.asarray(block_ids)
    shared = set(block_ids[np.asarray(train_idx)]) & set(block_ids[np.asarray(test_idx)])
    if shared:
        raise ValueError(f"train/test shares recording block(s) {sorted(shared)}")
