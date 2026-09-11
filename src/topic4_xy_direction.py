"""Signed, observation-matched propagation objective for the XY-only search.

An EE kernel axis is an input; these onset gradients are measured outputs.
Neither a two-cluster classifier nor OOD exclusion enters this distribution.
"""
import numpy as np

N_BINS = 24
MIN_CONTACTS = 4
MAX_CONDITION = 20.0
KERNEL_WIDTH = 0.5


def onset_directions(onsets, xy):
    """Fit t(x,y)=intercept+g.x; +g points from earlier to later onset.

    Fit each participation mask with the same physical coordinate metric.
    Adjusted R2 weights coherent planar direction; its complement stays in
    the unresolved/nonplanar mass. Collinear or nearly collinear samples
    never acquire an invented 2D direction. Nothing is dropped.
    """
    times = np.asarray(onsets, float)
    xy = np.asarray(xy, float)
    if times.ndim != 2 or xy.shape != (times.shape[1], 2):
        raise ValueError('onsets must be events x contacts, with matching XY')
    if np.isinf(times).any() or not np.isfinite(xy).all():
        raise ValueError('infinite onsets or invalid coordinates')
    n = len(times)
    angle = np.full(n, np.nan)
    coherence = np.zeros(n)
    condition = np.full(n, np.nan)
    estimable = np.zeros(n, bool)
    masks, inverse = np.unique(np.isfinite(times), axis=0, return_inverse=True)
    for mi, mask in enumerate(masks):
        rows = np.flatnonzero(inverse == mi)
        count = int(mask.sum())
        if count < MIN_CONTACTS:
            continue
        coords = xy[mask] - xy[mask].mean(axis=0)
        singular = np.linalg.svd(coords, compute_uv=False)
        if singular[-1] <= 1e-10:
            continue
        cond = singular[0] / singular[-1]
        condition[rows] = cond
        if cond > MAX_CONDITION:
            continue
        values = times[rows][:, mask]
        values = values - values.mean(axis=1, keepdims=True)
        gradient = values @ np.linalg.pinv(coords).T
        residual = values - gradient @ coords.T
        variance = np.sum(values**2, axis=1)
        good = (variance > 1e-12) & (np.linalg.norm(gradient, axis=1) > 1e-12)
        r2 = np.zeros(len(rows))
        r2[good] = 1 - np.sum(residual[good]**2, axis=1) / variance[good]
        adj = np.clip(1 - (1 - r2) * (count - 1) / (count - 3), 0, 1)
        selected = rows[good]
        angle[selected] = np.arctan2(gradient[good, 1], gradient[good, 0])
        coherence[selected] = adj[good]
        estimable[selected] = True
    return {'angle_rad': angle, 'coherence': coherence, 'estimable': estimable,
            'condition': condition, 'n_contacts': np.isfinite(times).sum(axis=1)}


def direction_histogram(view):
    """Circular linear binning plus one unresolved mass, divided by ALL events."""
    angle = np.asarray(view['angle_rad'])
    weights = np.asarray(view['coherence'])
    hist = np.zeros(N_BINS + 1)
    if not len(angle):
        return None
    good = np.isfinite(angle)
    coord = np.mod(angle[good], 2 * np.pi) * N_BINS / (2 * np.pi)
    lo = np.floor(coord).astype(int) % N_BINS
    frac = coord - np.floor(coord)
    np.add.at(hist, lo, weights[good] * (1 - frac))
    np.add.at(hist, (lo + 1) % N_BINS, weights[good] * frac)
    hist[-1] = len(angle) - weights[good].sum()
    return hist / len(angle)


def direction_distance(left, right):
    """Bounded Gaussian-kernel MMD on the signed circle and unresolved state."""
    if left is None or right is None:
        return None
    angle = np.arange(N_BINS) * 2 * np.pi / N_BINS
    points = np.column_stack([np.cos(angle), np.sin(angle), np.zeros(N_BINS)])
    points = np.vstack([points, [0, 0, 2]])
    square = np.sum((points[:, None] - points[None, :])**2, axis=-1)
    kernel = np.exp(-square / (2 * KERNEL_WIDTH**2))
    delta = np.asarray(left) - np.asarray(right)
    return float(np.sqrt(max(0., delta @ kernel @ delta / 2)))


def direction_summary(view):
    angle = np.asarray(view['angle_rad']); weights = np.asarray(view['coherence'])
    good = np.isfinite(angle); total = float(weights[good].sum())
    axial = np.sum(weights[good] * np.exp(2j * angle[good]))
    directed = np.sum(weights[good] * np.exp(1j * angle[good]))
    defined = total > 0 and abs(axial) / total >= 0.1
    return {'n_events': len(angle), 'n_geometry_estimable': int(view['estimable'].sum()),
            'coherent_mass': total / len(angle) if len(angle) else None,
            'unresolved_or_nonplanar_mass': 1 - total / len(angle) if len(angle) else None,
            'axial_angle_deg': float(np.degrees(np.angle(axial) / 2)) if defined else None,
            'axial_concentration': float(abs(axial) / total) if total else None,
            'signed_resultant': float(abs(directed) / total) if total else None,
            'axis_is_observed_behavior_not_structural_kernel': True}


def core_alignment_penalty(centers, axis_deg):
    """Label-invariant soft preference only; no midpoint/end-position target."""
    delta = np.asarray(centers, float)[1] - np.asarray(centers, float)[0]
    if np.linalg.norm(delta) <= 0:
        raise ValueError('cores coincide')
    angle = np.arctan2(delta[1], delta[0])
    return float(np.sin(angle - np.radians(axis_deg))**2)


def admission_slots(available_gib, active_rss_gib, peak_budget_gib,
                    maximum_workers=24, reserve_gib=40.):
    """Reserve not-yet-materialized peaks of ALL active workers before admission."""
    outstanding = sum(max(0., peak_budget_gib - rss) for rss in active_rss_gib)
    return max(0, min(maximum_workers - len(active_rss_gib),
                      int(np.floor((available_gib - reserve_gib - outstanding) / peak_budget_gib))))
