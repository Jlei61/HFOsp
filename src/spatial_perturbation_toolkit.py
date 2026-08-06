"""Model-agnostic helpers for finite-dimensional spatial perturbation analyses.

The module deliberately separates four objects that are often conflated:

1. ``J``: the tangent operator at a specified state;
2. a fixed input ``b``;
3. the finite-time output ``C exp(J t) b``;
4. the operator envelope ``sigma_1(C exp(J t) B)``.

It contains no Topic-4 state construction, no MZ assumptions, no simulations and no file I/O.
Callers are responsible for supplying a scientifically justified state, operator and readout.
"""
from __future__ import annotations

import numpy as np


def finite_time_operator_svd(J, T, input_matrix, *, readout_matrix=None, output_indices=None):
    """SVD of ``C exp(JT) B`` for caller-supplied input and readout spaces.

    Exactly one of ``readout_matrix`` and ``output_indices`` may be given. The result distinguishes
    the optimal input coordinates (V1), optimal output field/vector (U1), and gain ``sigma1``.
    """
    from scipy.sparse.linalg import expm_multiply

    if readout_matrix is not None and output_indices is not None:
        raise ValueError("choose readout_matrix or output_indices, not both")
    propagated = expm_multiply(np.asarray(J, float) * float(T), np.asarray(input_matrix, float))
    if readout_matrix is not None:
        M = np.asarray(readout_matrix, float) @ propagated
    elif output_indices is not None:
        M = propagated[output_indices]
    else:
        M = propagated
    U, s, Vh = np.linalg.svd(M, full_matrices=False)
    return dict(propagator=M, sigma1=(float(s[0]) if s.size else 0.0), singular_values=s,
                optimal_output=(U[:, 0].copy() if s.size else np.zeros(M.shape[0])),
                optimal_input_coordinates=(Vh[0].copy() if s.size else np.zeros(M.shape[1])))


def operator_gain_envelope(J, times, input_matrix, *, readout_matrix=None, output_indices=None):
    """Largest singular value of the same caller-declared operator block over time."""
    gains = []
    for t in times:
        out = finite_time_operator_svd(J, t, input_matrix, readout_matrix=readout_matrix,
                                       output_indices=output_indices)
        gains.append(out["sigma1"])
    return np.asarray(times, float), np.asarray(gains, float)


def linear_response_timecourse(J, b, times, *, output_indices=None):
    """Return ``C exp(J t) b`` for each requested time.

    ``output_indices`` is the readout ``C`` expressed as NumPy indexing.  With ``None`` the full
    state is returned.  The input is never re-optimized between states or times.
    """
    from scipy.sparse.linalg import expm_multiply

    J = np.asarray(J, float)
    b = np.asarray(b, float)
    out = {}
    for t in times:
        y = b.copy() if float(t) <= 0 else expm_multiply(J * float(t), b)
        out[float(t)] = y if output_indices is None else y[output_indices]
    return out


def response_gain_curve(evolution):
    """Times and Euclidean output gains for a unit-norm fixed input response timecourse."""
    ts = np.array(sorted(evolution), float)
    gains = np.array([np.linalg.norm(np.asarray(evolution[t], float)) for t in ts], float)
    return ts, gains


def region_response_curve(evolution, mask, *, statistic="rms"):
    """Summarize the absolute response in a fixed spatial region at each time.

    ``rms`` is the default because it does not cancel signed oscillatory responses. ``mean_abs`` is
    also available. The same mask must be reused for every compared state.
    """
    mask = np.asarray(mask, bool)
    if not mask.any():
        raise ValueError("region mask is empty")
    ts = np.array(sorted(evolution), float)
    vals = []
    for t in ts:
        x = np.asarray(evolution[t], float)[mask]
        if statistic == "rms":
            vals.append(float(np.sqrt(np.mean(x ** 2))))
        elif statistic == "mean_abs":
            vals.append(float(np.mean(np.abs(x))))
        else:
            raise ValueError(f"unknown statistic {statistic!r}")
    return ts, np.asarray(vals, float)


def cumulative_response_ratio(numerator, denominator, times):
    """Cumulative RMS-energy ratio, stable when the instantaneous denominator crosses zero."""
    num = np.asarray(numerator, float)
    den = np.asarray(denominator, float)
    ts = np.asarray(times, float)
    if num.shape != den.shape or num.shape != ts.shape:
        raise ValueError("numerator, denominator and times must have the same shape")
    e_num = np.zeros_like(num)
    e_den = np.zeros_like(den)
    if ts.size > 1:
        dt = np.diff(ts)
        e_num[1:] = np.cumsum(0.5 * (num[1:] ** 2 + num[:-1] ** 2) * dt)
        e_den[1:] = np.cumsum(0.5 * (den[1:] ** 2 + den[:-1] ** 2) * dt)
    ratio = np.sqrt(np.divide(e_num, e_den, out=np.full_like(e_num, np.nan), where=e_den > 1e-30))
    if ts.size and abs(den[0]) > 1e-15:
        ratio[0] = abs(num[0] / den[0])
    return ratio


def axis_kymograph(evolution, x_coordinates, y_coordinates, *, axis_y, band):
    """Return ``(x, t, K[t,x])`` where K is band-averaged ``|response|`` along an axis."""
    X = np.asarray(x_coordinates, float)
    Y = np.asarray(y_coordinates, float)
    yline = Y[0, :]
    ymask = np.abs(yline - float(axis_y)) <= float(band)
    if not ymask.any():
        ymask[np.argmin(np.abs(yline - float(axis_y)))] = True
    ts = np.array(sorted(evolution), float)
    kymo = np.array([np.abs(np.asarray(evolution[t], float))[:, ymask].mean(axis=1) for t in ts])
    return X[:, 0].copy(), ts, kymo


def first_arrival_times(kymograph, times, *, threshold_fraction=0.10, absolute_floor=1e-12):
    """Threshold-defined first-arrival time at each position.

    The threshold is ``threshold_fraction`` of the state-specific global kymograph maximum. This is
    an operational latency, not a proof of a causal wavefront. Positions that never cross remain
    NaN. The caller should report the threshold and inspect sensitivity to it.
    """
    K = np.asarray(kymograph, float)
    ts = np.asarray(times, float)
    if K.ndim != 2 or K.shape[0] != ts.size:
        raise ValueError("kymograph must have shape (n_time, n_position)")
    peak = float(np.nanmax(K)) if K.size else 0.0
    threshold = max(float(absolute_floor), float(threshold_fraction) * peak)
    arrivals = np.full(K.shape[1], np.nan)
    for j in range(K.shape[1]):
        hit = np.flatnonzero(K[:, j] >= threshold)
        if hit.size:
            arrivals[j] = ts[int(hit[0])]
    return arrivals, threshold


def fit_arrival_time_distance(positions, arrivals, *, source_position, sink_position, min_points=4):
    """Fit arrival time against directed source-to-sink distance.

    A positive slope with a useful R2 supports sequential recruitment. A flat slope or near-zero
    first-arrival times is compatible with direct remote recruitment by the connection kernel.
    Returned velocity uses caller coordinates per millisecond; it is not physical until the caller
    supplies a coordinate conversion.
    """
    x = np.asarray(positions, float)
    t = np.asarray(arrivals, float)
    direction = 1.0 if float(sink_position) >= float(source_position) else -1.0
    distance = direction * (x - float(source_position))
    span = abs(float(sink_position) - float(source_position))
    keep = np.isfinite(t) & (distance >= -1e-12) & (distance <= span + 1e-12)
    if int(keep.sum()) < int(min_points):
        return dict(eligible=False, n_points=int(keep.sum()), slope_ms_per_unit=None,
                    velocity_unit_per_ms=None, intercept_ms=None, r2=None)
    d = distance[keep]
    y = t[keep]
    A = np.column_stack([d, np.ones_like(d)])
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    pred = slope * d + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    velocity = 1.0 / slope if slope > 0 else None
    return dict(eligible=True, n_points=int(keep.sum()), slope_ms_per_unit=float(slope),
                velocity_unit_per_ms=(None if velocity is None else float(velocity)),
                intercept_ms=float(intercept), r2=float(r2))


def normalized_field_overlap(field_a, field_b):
    """Sign-insensitive cosine overlap in [0,1] for non-negative mode-loading fields."""
    a = np.asarray(field_a, float).ravel()
    b = np.asarray(field_b, float).ravel()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(abs(np.dot(a, b)) / den) if den > 0 else np.nan
