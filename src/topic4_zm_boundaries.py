"""Probability-boundary helpers for the Z/M branch-decision workflow.

This module is deliberately agnostic to how the SNN observations were
generated.  It only turns replicate-level binary outcomes into conservative
Jeffreys posterior curves and reports a 0.5 boundary when the sampled
coordinate range actually brackets one.
"""

from __future__ import annotations

import copy
from collections import defaultdict

import numpy as np
from scipy.stats import beta


BOUNDARY_VERSION = "zm_probability_boundaries_v1_2026-07-27"
HYSTERESIS_MIN_NORMALIZED_SEPARATION = 0.1
_STATE_KEY = {"z": "slow.z", "m": "slow.m", "sg": "slow.S_G"}


def _as_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    raise ValueError(f"binary outcome required, got {value!r}")


def _coordinate_vector(state, coordinate, nE):
    key = _STATE_KEY[coordinate]
    value = np.asarray(state[key], dtype=float)
    if coordinate in {"z", "m"}:
        if value.ndim != 1 or value.size < int(nE):
            raise ValueError(f"{key} does not contain nE entries")
        return value[:int(nE)].copy()
    if value.size != 1:
        raise ValueError("slow.S_G must be scalar")
    return np.asarray([float(value)])


def interpolate_slow_state(
    early_state,
    late_state,
    lam,
    *,
    coordinates,
    nE,
    allow_extrapolation=False,
):
    """Interpolate actual slow fields while preserving every fast-state array.

    Only the named coordinates are changed.  Physical bounds are checked after
    interpolation and never enforced by clipping, because clipping silently
    changes the tested trajectory direction.
    """

    lam = float(lam)
    nE = int(nE)
    coordinates = tuple(coordinates)
    if not coordinates or any(name not in _STATE_KEY for name in coordinates):
        raise ValueError("coordinates must be a non-empty subset of z, m, sg")
    if len(set(coordinates)) != len(coordinates):
        raise ValueError("coordinates must be unique")
    if not np.isfinite(lam):
        raise ValueError("lambda must be finite")
    if not allow_extrapolation and not 0.0 <= lam <= 1.0:
        raise ValueError("lambda outside [0, 1] requires allow_extrapolation")

    out = copy.deepcopy(early_state)
    for coordinate in coordinates:
        a = _coordinate_vector(early_state, coordinate, nE)
        b = _coordinate_vector(late_state, coordinate, nE)
        candidate = a + lam * (b - a)
        if not np.isfinite(candidate).all():
            raise ValueError(f"{coordinate}: interpolated field is non-finite")
        if coordinate == "z" and np.any((candidate < 0.0) | (candidate > 1.0)):
            raise ValueError("z interpolation leaves [0, 1]; refusing to clip")
        if coordinate in {"m", "sg"} and np.any(candidate < 0.0):
            raise ValueError(f"{coordinate} interpolation becomes negative")
        key = _STATE_KEY[coordinate]
        if coordinate in {"z", "m"}:
            full = np.asarray(out[key], dtype=float).copy()
            full[:nE] = candidate
            out[key] = full
        else:
            out[key] = np.asarray(float(candidate[0]))
    return out


def slow_state_coordinate_values(
    states,
    early_state,
    late_state,
    *,
    coordinates,
    nE,
):
    """Project states onto equal-weight actual early-to-late slow directions."""

    coordinates = tuple(coordinates)
    if not states:
        raise ValueError("at least one state is required")
    per_coordinate = {}
    for coordinate in coordinates:
        a = _coordinate_vector(early_state, coordinate, nE)
        b = _coordinate_vector(late_state, coordinate, nE)
        direction = b - a
        norm2 = float(np.dot(direction, direction))
        if norm2 <= 1e-18:
            raise ValueError(f"{coordinate}: early-to-late direction is degenerate")
        values = []
        for state in states:
            value = _coordinate_vector(state, coordinate, nE)
            values.append(float(np.dot(value - a, direction) / norm2))
        per_coordinate[coordinate] = values
    matrix = np.column_stack([per_coordinate[name] for name in coordinates])
    return {
        "joint_lambda": np.mean(matrix, axis=1).tolist(),
        "per_coordinate_lambda": per_coordinate,
        "coordinate_disagreement_sd": np.std(matrix, axis=1).tolist(),
        "coordinates": list(coordinates),
        "boundary_version": BOUNDARY_VERSION,
    }


def jeffreys_probability_curve(rows, coordinate_key, outcome_key, alpha=0.05):
    """Aggregate replicate outcomes at each coordinate level.

    The returned probability is the posterior median under a Jeffreys
    Beta(1/2, 1/2) prior.  Equal-tail intervals are kept separate from the
    point estimate so a downstream decision cannot silently substitute a raw
    success fraction for the registered posterior.
    """

    grouped = defaultdict(list)
    for row in rows:
        q = float(row[coordinate_key])
        if not np.isfinite(q):
            raise ValueError("coordinate values must be finite")
        grouped[q].append(_as_bool(row[outcome_key]))
    if not grouped:
        raise ValueError("at least one replicate row is required")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    curve = []
    for q in sorted(grouped):
        outcomes = np.asarray(grouped[q], dtype=bool)
        n = int(outcomes.size)
        k = int(outcomes.sum())
        a_post = k + 0.5
        b_post = n - k + 0.5
        curve.append(
            {
                "q": float(q),
                "n": n,
                "k": k,
                "success_fraction": float(k / n),
                "posterior_median": float(beta.ppf(0.5, a_post, b_post)),
                "posterior_mean": float(a_post / (a_post + b_post)),
                "posterior_ci": [
                    float(beta.ppf(alpha / 2.0, a_post, b_post)),
                    float(beta.ppf(1.0 - alpha / 2.0, a_post, b_post)),
                ],
            }
        )
    return curve


def _validate_direction(expected_direction):
    if expected_direction not in {"increasing", "decreasing"}:
        raise ValueError("expected_direction must be increasing or decreasing")


def half_boundary(curve, expected_direction, probability_key="posterior_median"):
    """Find a bracketed P=0.5 crossing on a monotonic sampled curve.

    Nonmonotonic evidence is not repaired with isotonic regression here:
    doing so could manufacture a boundary that the sampled simulations do not
    support.  Such a curve is returned as ``nonmonotonic`` for explicit
    scientific adjudication.
    """

    _validate_direction(expected_direction)
    if len(curve) < 2:
        return {
            "status": "unbracketed",
            "q_half": None,
            "direction": expected_direction,
            "boundary_version": BOUNDARY_VERSION,
        }
    ordered = sorted(curve, key=lambda row: float(row["q"]))
    q = np.asarray([row["q"] for row in ordered], dtype=float)
    p = np.asarray([row[probability_key] for row in ordered], dtype=float)
    if not np.isfinite(q).all() or not np.isfinite(p).all():
        raise ValueError("curve contains non-finite values")
    if np.any(np.diff(q) <= 0):
        raise ValueError("curve coordinates must be unique")

    signed_diff = np.diff(p) if expected_direction == "increasing" else -np.diff(p)
    if np.any(signed_diff < -1e-12):
        return {
            "status": "nonmonotonic",
            "q_half": None,
            "direction": expected_direction,
            "boundary_version": BOUNDARY_VERSION,
        }

    centered = p - 0.5
    exact = np.flatnonzero(np.isclose(centered, 0.0, atol=1e-12))
    if exact.size:
        q_half = float(q[int(exact[0])])
        bracket = [q_half, q_half]
    else:
        crossing = np.flatnonzero(centered[:-1] * centered[1:] < 0.0)
        if crossing.size == 0:
            return {
                "status": "unbracketed",
                "q_half": None,
                "direction": expected_direction,
                "sampled_range": [float(q[0]), float(q[-1])],
                "boundary_version": BOUNDARY_VERSION,
            }
        i = int(crossing[0])
        fraction = (0.5 - p[i]) / (p[i + 1] - p[i])
        q_half = float(q[i] + fraction * (q[i + 1] - q[i]))
        bracket = [float(q[i]), float(q[i + 1])]

    return {
        "status": "bracketed",
        "q_half": q_half,
        "q_bracket": bracket,
        "direction": expected_direction,
        "boundary_version": BOUNDARY_VERSION,
    }


def bootstrap_half_boundary(
    rows,
    coordinate_key,
    outcome_key,
    expected_direction,
    n_boot=2000,
    seed=0,
    alpha=0.05,
):
    """Bootstrap replicate outcomes within each sampled coordinate level."""

    if int(n_boot) <= 0:
        raise ValueError("n_boot must be positive")
    point_curve = jeffreys_probability_curve(
        rows, coordinate_key, outcome_key, alpha=alpha
    )
    point = half_boundary(point_curve, expected_direction)
    out = {
        **point,
        "curve": point_curve,
        "n_bootstrap": int(n_boot),
        "n_valid_bootstrap": 0,
        "q_half_ci": None,
    }
    if point["status"] != "bracketed":
        return out

    grouped = defaultdict(list)
    for row in rows:
        grouped[float(row[coordinate_key])].append(_as_bool(row[outcome_key]))
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(int(n_boot)):
        boot_rows = []
        for q in sorted(grouped):
            values = np.asarray(grouped[q], dtype=bool)
            draw = rng.choice(values, size=values.size, replace=True)
            boot_rows.extend(
                {coordinate_key: q, outcome_key: bool(value)} for value in draw
            )
        candidate = half_boundary(
            jeffreys_probability_curve(
                boot_rows, coordinate_key, outcome_key, alpha=alpha
            ),
            expected_direction,
        )
        if candidate["status"] == "bracketed":
            samples.append(float(candidate["q_half"]))

    out["n_valid_bootstrap"] = len(samples)
    # Fail closed when fewer than half the resamples support a bracket.
    if len(samples) >= max(2, int(np.ceil(0.5 * int(n_boot)))):
        lo, hi = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])
        out["q_half_ci"] = [float(lo), float(hi)]
    else:
        out["status"] = "bootstrap_indeterminate"
        out["q_half"] = None
    return out


def trajectory_crossing(coordinate_trajectory, boundary, expected_direction):
    """Report whether and in which direction a trajectory crosses a boundary."""

    _validate_direction(expected_direction)
    q = np.asarray(coordinate_trajectory, dtype=float)
    if q.ndim != 1 or q.size < 2 or not np.isfinite(q).all():
        raise ValueError("coordinate_trajectory must be a finite 1D sequence")
    boundary = float(boundary)
    if not np.isfinite(boundary):
        raise ValueError("boundary must be finite")

    delta = q - boundary
    crossings = []
    for i in range(q.size - 1):
        if delta[i] == 0.0 and delta[i + 1] == 0.0:
            continue
        if delta[i] == 0.0 or delta[i + 1] == 0.0 or delta[i] * delta[i + 1] < 0.0:
            direction = "increasing" if q[i + 1] > q[i] else "decreasing"
            crossings.append({"index": i, "direction": direction})
    return {
        "crossed": bool(crossings),
        "direction_ok": any(
            crossing["direction"] == expected_direction for crossing in crossings
        ),
        "crossings": crossings,
        "expected_direction": expected_direction,
        "boundary": boundary,
        "boundary_version": BOUNDARY_VERSION,
    }


def hysteresis_summary(
    onset_boundary,
    offset_boundary,
    scale,
    *,
    onset_ci=None,
    offset_ci=None,
):
    """Compare onset/offset surfaces without mistaking point noise for hysteresis."""

    onset_boundary = float(onset_boundary)
    offset_boundary = float(offset_boundary)
    scale = float(scale)
    if not np.isfinite([onset_boundary, offset_boundary, scale]).all() or scale <= 0:
        raise ValueError("finite boundaries and a positive scale are required")
    signed = offset_boundary - onset_boundary
    normalized = signed / scale
    intervals_available = onset_ci is not None and offset_ci is not None
    intervals_nonoverlap = False
    onset_interval = offset_interval = None
    if intervals_available:
        onset_interval = np.asarray(onset_ci, dtype=float)
        offset_interval = np.asarray(offset_ci, dtype=float)
        if (
            onset_interval.shape != (2,)
            or offset_interval.shape != (2,)
            or not np.isfinite(onset_interval).all()
            or not np.isfinite(offset_interval).all()
            or onset_interval[0] > onset_interval[1]
            or offset_interval[0] > offset_interval[1]
            or not onset_interval[0] <= onset_boundary <= onset_interval[1]
            or not offset_interval[0] <= offset_boundary <= offset_interval[1]
        ):
            raise ValueError("boundary intervals must be ordered and contain point estimates")
        intervals_nonoverlap = bool(
            onset_interval[1] < offset_interval[0]
            or offset_interval[1] < onset_interval[0]
        )
    point_separation_large = bool(
        abs(normalized) >= HYSTERESIS_MIN_NORMALIZED_SEPARATION
    )
    distinct = bool(
        intervals_available and intervals_nonoverlap and point_separation_large
    )
    return {
        "onset_boundary": onset_boundary,
        "offset_boundary": offset_boundary,
        "onset_ci": onset_interval.tolist() if onset_interval is not None else None,
        "offset_ci": offset_interval.tolist() if offset_interval is not None else None,
        "signed_separation": float(signed),
        "normalized_separation": float(normalized),
        "point_separation_large": point_separation_large,
        "intervals_nonoverlap": intervals_nonoverlap,
        "distinct_surfaces": distinct,
        "status": (
            "distinct"
            if distinct
            else "uncertainty_missing"
            if not intervals_available
            else "not_resolved"
        ),
        "minimum_normalized_separation": HYSTERESIS_MIN_NORMALIZED_SEPARATION,
        "boundary_version": BOUNDARY_VERSION,
    }
