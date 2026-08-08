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
from scipy.stats import wasserstein_distance

MIN_PARTICIPANTS = 6

# Stage 3 rev6 candidate observable. These are deliberately fixed rather than
# inferred separately for every model candidate. The patient training split is
# the only place where the embedding is fitted; held-out recordings and every
# model arm are transformed without refitting.
PROFILE_GRID_N = 31
PROFILE_N_COMPONENTS = 8
PROFILE_N_PROJECTIONS = 64
PROFILE_REFERENCE_N = 4096
PROFILE_REFERENCE_SEED = 20260809
OBJECTIVE_N_EVENTS = 20

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


def profile_grid(axial, n_grid=PROFILE_GRID_N):
    """Frozen axial grid shared by patient and model event profiles."""
    x = np.asarray(list(axial.values()), float)
    if x.size < 2 or not np.isfinite(x).all() or np.ptp(x) < 1e-9:
        raise ValueError("axial support must contain at least two distinct positions")
    return np.linspace(float(x.min()), float(x.max()), int(n_grid))


def normalized_rank_curve(ranks, axial, participating=None,
                          part_min=MIN_PARTICIPANTS, grid=None):
    """One event as a scale-free rank profile on a common axial grid.

    Rank values are standardized within the event before interpolation, so a
    model event with seven participating contacts is not penalized merely because
    the patient often recruits twelve. Constant endpoint extension is explicit:
    it records the observed ordering without inventing a new extremum outside the
    participating span. The final unit norm removes residual amplitude scale.
    """
    pts = _pairs(ranks, axial, participating)
    if len(pts) < int(part_min):
        return None
    pts.sort(key=lambda p: p[0])
    x = np.asarray([p[0] for p in pts], float)
    y = np.asarray([p[1] for p in pts], float)
    if np.ptp(x) < 1e-9 or y.std() < 1e-9:
        return None
    y = (y - y.mean()) / y.std()
    q = np.interp(profile_grid(axial) if grid is None else np.asarray(grid, float),
                  x, y)
    q = q - q.mean()
    norm = float(np.linalg.norm(q))
    return None if norm < 1e-12 else q / norm


def rank_curve_table(events, axial, participating=None,
                     part_min=MIN_PARTICIPANTS, grid=None):
    """Normalized rank curves for raw rank dictionaries or event records."""
    rows = []
    for ev in events:
        ranks = ev.get("ranks") if isinstance(ev, dict) and "ranks" in ev else ev
        mask = participating(ev) if callable(participating) else participating
        q = normalized_rank_curve(ranks, axial, mask, part_min, grid)
        if q is not None:
            rows.append(q)
    n_grid = len(profile_grid(axial) if grid is None else np.asarray(grid))
    return np.asarray(rows, float).reshape((-1, n_grid))


def fit_rank_curve_reference(
        patient_train_curves,
        n_components=PROFILE_N_COMPONENTS,
        n_reference=PROFILE_REFERENCE_N,
        n_projections=PROFILE_N_PROJECTIONS,
        seed=PROFILE_REFERENCE_SEED):
    """Fit the unlabeled patient-training embedding used by the joint distance.

    PCA is only a deterministic compression of the full normalized profile. The
    distance is evaluated over many fixed projections of the retained joint
    cloud; no direction labels, template assignments, or final acceptance-gate
    quantities enter this fit.
    """
    x = np.asarray(patient_train_curves, float)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("patient_train_curves must be a finite (n>=2, grid) matrix")
    k = min(int(n_components), x.shape[1], len(x) - 1)
    if k < 2:
        raise ValueError("joint profile embedding needs at least two components")
    center = x.mean(axis=0)
    _, singular, vt = np.linalg.svd(x - center, full_matrices=False)
    components = vt[:k]
    scores = (x - center) @ components.T
    score_center = scores.mean(axis=0)
    score_scale = scores.std(axis=0)
    score_scale[score_scale < 1e-12] = 1.0

    rng = np.random.default_rng(int(seed))
    take = min(int(n_reference), len(x))
    reference_index = rng.choice(len(x), size=take, replace=False)
    reference_z = (scores[reference_index] - score_center) / score_scale
    n_proj = int(n_projections)
    if n_proj < k:
        raise ValueError("n_projections must include every retained PCA axis")
    random_directions = rng.normal(size=(n_proj - k, k))
    directions = np.vstack((np.eye(k), random_directions))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    variance = singular ** 2
    explained = variance[:k] / variance.sum() if variance.sum() else np.zeros(k)
    return dict(
        center=center,
        components=components,
        score_center=score_center,
        score_scale=score_scale,
        reference_index=reference_index,
        reference_z=reference_z,
        directions=directions,
        explained_variance_ratio=explained,
        n_train=int(len(x)),
        n_reference=int(take),
        n_components=int(k),
        n_projections=int(n_proj),
        seed=int(seed),
    )


def transform_rank_curves(curves, reference):
    """Apply a frozen patient-training embedding without refitting it."""
    x = np.asarray(curves, float)
    if x.ndim != 2 or x.shape[1] != len(reference["center"]):
        raise ValueError("curves do not match the frozen profile grid")
    scores = (x - reference["center"]) @ reference["components"].T
    return (scores - reference["score_center"]) / reference["score_scale"]


def sliced_rank_curve_distance(curves, reference):
    """Sliced Wasserstein distance to the frozen patient training cloud."""
    z = transform_rank_curves(curves, reference)
    return sliced_embedding_distance(
        z, reference["reference_z"], reference["directions"])


def sliced_embedding_distance(z, target_z, directions):
    """Sliced Wasserstein between two clouds in one frozen embedding."""
    z = np.asarray(z, float)
    target = np.asarray(target_z, float)
    directions = np.asarray(directions, float)
    if z.ndim != 2 or target.ndim != 2 or z.shape[1:] != target.shape[1:]:
        raise ValueError("source and target embedding clouds must have matching 2-D shapes")
    if directions.ndim != 2 or directions.shape[1] != z.shape[1]:
        raise ValueError("directions do not match the embedding dimension")
    if len(z) < 2 or len(target) < 2:
        return float("nan")
    return float(np.mean([
        wasserstein_distance(z @ direction, target @ direction)
        for direction in directions
    ]))


def fixed_count_indices(n_available, n_events=OBJECTIVE_N_EVENTS):
    """Deterministic, order-preserving coverage of exactly ``n_events`` rows."""
    n_available, n_events = int(n_available), int(n_events)
    if n_events < 2:
        raise ValueError("fixed-count distance needs at least two events")
    if n_available < n_events:
        return None
    # Midpoints of equal-width bins cover the complete ordered event stream and
    # never duplicate an index when n_available >= n_events.
    return np.floor(
        (np.arange(n_events) + 0.5) * n_available / n_events).astype(int)


def fixed_count_sliced_distance(curves, reference,
                                n_events=OBJECTIVE_N_EVENTS):
    """Sample-size-matched distance used by optimization candidates."""
    x = np.asarray(curves, float)
    index = fixed_count_indices(len(x), n_events)
    return (float("nan") if index is None
            else sliced_rank_curve_distance(x[index], reference))


def rank_curve_reference_summary(reference):
    """JSON-sized contract fields; excludes the large reference point cloud."""
    return dict(
        observable="within-event normalized rank curve on a fixed axial grid",
        distance="sliced Wasserstein in a patient-training-only PCA embedding",
        uses_direction_labels=False,
        endpoint_extension="constant",
        n_train=int(reference["n_train"]),
        n_reference=int(reference["n_reference"]),
        n_components=int(reference["n_components"]),
        n_projections=int(reference["n_projections"]),
        seed=int(reference["seed"]),
        explained_variance_ratio=np.asarray(
            reference["explained_variance_ratio"], float).tolist(),
    )


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
