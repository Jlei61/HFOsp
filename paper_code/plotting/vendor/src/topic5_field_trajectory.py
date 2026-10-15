"""Descriptive early-to-late streamlines of a supported Gaussian rank field.

These curves visualize the local gradient of an interpolated rank field. They
are not time-resolved tracks, velocity estimates, or uniquely identified paths.
"""
from __future__ import annotations
from paper_paths import CODE_ROOT, INPUT_ROOT, OUTPUT_ROOT
import numpy as np
from scipy.spatial import ConvexHull

def gaussian_rank_value_gradient(query, points, ranks, support, sigma_mm):
    """Evaluate the display smoother and its analytic gradient in physical mm."""
    points = np.asarray(points, dtype=float)
    ranks = np.asarray(ranks, dtype=float)
    support = np.asarray(support, dtype=float)
    query = np.asarray(query, dtype=float)
    if sigma_mm <= 0 or not np.isfinite(sigma_mm):
        raise ValueError('sigma_mm must be positive and finite')
    weights = support * np.exp(-np.sum((points - query) ** 2, axis=1) / (2 * sigma_mm ** 2))
    mass = float(weights.sum())
    if mass <= 1e-14:
        return (float('nan'), np.full(2, np.nan), mass)
    weights /= mass
    value = float(weights @ ranks)
    gradient = weights * (ranks - value) @ (points - query) / sigma_mm ** 2
    return (value, gradient, mass)

def sampling_hull(points):
    """Return a geometry-only area centroid and the sampling convex hull."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError('At least three two-dimensional contact positions are required')
    if not np.isfinite(points).all() or np.linalg.matrix_rank(points - points.mean(0)) < 2:
        raise ValueError('Two-dimensional contact geometry is not estimable')
    hull = ConvexHull(points)
    vertices = points[hull.vertices]
    shifted = np.roll(vertices, -1, axis=0)
    cross = vertices[:, 0] * shifted[:, 1] - shifted[:, 0] * vertices[:, 1]
    centroid = ((vertices + shifted) * cross[:, None]).sum(0) / (3 * cross.sum())
    return (centroid, hull)

def rank_gradient_trajectory(points, ranks, support, *, sigma_mm=6.0, support_floor=0.0, step_mm=0.08, seed=None, gradient_floor=1e-05, max_length_mm=150.0):
    """Integrate ± normalized local rank gradient from a geometry-only seed.

    By default the same sampling-area centroid is used for both templates.
    Both directions stop at the sampling hull, display-support boundary, flat
    gradient, or finite length cap. The returned path always runs early→late.
    No seed or endpoint is chosen by resemblance to an expected direction.
    """
    points = np.asarray(points, dtype=float)
    ranks = np.asarray(ranks, dtype=float)
    support = np.asarray(support, dtype=float)
    if ranks.shape != (len(points),) or support.shape != ranks.shape:
        raise ValueError('Contact positions, ranks, and support must align')
    if not np.isfinite(ranks).all() or not np.isfinite(support).all() or np.any(support <= 0):
        raise ValueError('Every input contact must have finite rank and positive support')
    if np.ptp(ranks) <= 1e-12:
        raise ValueError('Constant ranks have no directed trajectory')
    if step_mm <= 0 or max_length_mm <= 0 or gradient_floor <= 0:
        raise ValueError('Integration scales must be positive')
    centroid, hull = sampling_hull(points)
    seed = centroid if seed is None else np.asarray(seed, dtype=float)

    def evaluate(q):
        return gaussian_rank_value_gradient(q, points, ranks, support, sigma_mm)

    def valid(q):
        if np.max(hull.equations[:, :2] @ q + hull.equations[:, 2]) > 1e-08:
            return 'sampling_hull_boundary'
        value, gradient, mass = evaluate(q)
        if mass < support_floor:
            return 'display_support_boundary'
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            return 'undefined_gradient'
        if np.linalg.norm(gradient) < gradient_floor:
            return 'flat_gradient'
        return None
    if valid(seed):
        raise ValueError(f'Geometry seed not usable: {valid(seed)}')

    def unit_gradient(q):
        gradient = evaluate(q)[1]
        return gradient / max(float(np.linalg.norm(gradient)), 1e-15)
    branches = []
    stops = []
    for sign in (-1, 1):
        path = [seed.copy()]
        reason = 'length_cap'
        for _ in range(int(max_length_mm / step_mm)):
            q = path[-1]
            h = sign * step_mm
            k1 = unit_gradient(q)
            k2 = unit_gradient(q + h * k1 / 2)
            k3 = unit_gradient(q + h * k2 / 2)
            k4 = unit_gradient(q + h * k3)
            new = q + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            reason = valid(new)
            if reason:
                break
            if sign * (evaluate(new)[0] - evaluate(q)[0]) <= 0:
                reason = 'rank_nonmonotonic_step'
                break
            path.append(new)
        branches.append(np.asarray(path))
        stops.append(reason or 'length_cap')
    path = np.concatenate([branches[0][::-1], branches[1][1:]])
    if len(path) < 3:
        raise ValueError('No resolved trajectory')
    values = np.array([evaluate(q)[0] for q in path])
    masses = np.array([evaluate(q)[2] for q in path])
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    chord = path[-1] - path[0]
    perpendicular = np.array([-chord[1], chord[0]]) / np.linalg.norm(chord)
    return {'points_mm': path, 'rank01': values, 'support_mass': masses, 'arc_length_mm': arc, 'seed_mm': seed, 'seed_rule': 'sampling_convex_hull_area_centroid' if np.allclose(seed, centroid) else 'explicit_sensitivity_seed', 'stop_early': stops[0], 'stop_late': stops[1], 'step_mm': float(step_mm), 'sigma_mm': float(sigma_mm), 'max_departure_from_endpoint_chord_mm': float(np.max(np.abs((path - path[0]) @ perpendicular)))}
