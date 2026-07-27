"""Unit-invariant slow-coordinate functional-rank diagnostics (Rev3.1 Task 9A).

The scientific object is a local response matrix from existing slow
coordinates to source/readout observables.  Raw derivatives are not comparable
because Z, M, S_G, rate and energy have different units.  This module therefore
standardizes every derivative by locked trajectory scales before computing a
singular spectrum.

The rank-1 label is intentionally conservative and local.  It means that the
second singular direction remains small under bootstrap resampling; it does
not claim that the full slow manifold is globally one-dimensional.
"""
from __future__ import annotations

import numpy as np


EFFECTIVE_RANK_VERSION = "zm_effective_rank_v1_2026-07-27"
RANK1_S2_RATIO_MAX = 0.20
RANK1_ENERGY_MIN = 0.90


def assemble_paired_sensitivity(rows, coordinate_order):
    """Assemble dy/dq from central-difference rows under matched future noise."""
    columns = []
    for coordinate in coordinate_order:
        pair = [r for r in rows if r.get("coordinate") == coordinate]
        if len(pair) != 2 or {int(r.get("sign", 0)) for r in pair} != {-1, 1}:
            raise ValueError(f"{coordinate}: require exactly one plus and one minus row")
        plus = next(r for r in pair if int(r["sign"]) == 1)
        minus = next(r for r in pair if int(r["sign"]) == -1)
        if plus.get("bank_sha") != minus.get("bank_sha"):
            raise ValueError(f"{coordinate}: central pair uses unmatched future noise")
        dp = float(plus.get("delta", np.nan))
        dm = float(minus.get("delta", np.nan))
        if not np.isfinite(dp) or dp <= 0 or not np.isclose(dp, dm):
            raise ValueError(f"{coordinate}: plus/minus delta mismatch")
        yp = np.asarray(plus["y"], float)
        ym = np.asarray(minus["y"], float)
        if yp.shape != ym.shape or yp.ndim != 1:
            raise ValueError(f"{coordinate}: response vectors must be aligned 1D arrays")
        columns.append((yp - ym) / (2.0 * dp))
    if not columns:
        raise ValueError("no coordinates")
    return np.column_stack(columns)


def standardize_sensitivity(S, q_scales, y_scales):
    """Return S_tilde[i,j] = q_scale[j] / y_scale[i] * dy_i/dq_j."""
    S = np.asarray(S, float)
    q = np.asarray(q_scales, float)
    y = np.asarray(y_scales, float)
    if S.ndim != 2 or q.shape != (S.shape[1],) or y.shape != (S.shape[0],):
        raise ValueError("scale dimensions do not match sensitivity matrix")
    if not np.all(np.isfinite(S)) or not np.all(np.isfinite(q)) or not np.all(np.isfinite(y)):
        raise ValueError("sensitivity and scales must be finite")
    if np.any(q <= 0) or np.any(y <= 0):
        raise ValueError("trajectory scales must be positive")
    return S * q[None, :] / y[:, None]


def rank_summary(matrix):
    """Continuous singular-spectrum summary plus a descriptive rank-1 flag."""
    A = np.asarray(matrix, float)
    if A.ndim != 2 or min(A.shape) < 1 or not np.all(np.isfinite(A)):
        raise ValueError("matrix must be finite and two-dimensional")
    s = np.linalg.svd(A, compute_uv=False)
    power = s ** 2
    total = float(power.sum())
    if total <= 0:
        energy_first = 0.0
        participation = 0.0
        entropy_rank = 0.0
    else:
        p = power / total
        energy_first = float(p[0])
        participation = float(total ** 2 / np.sum(power ** 2))
        pp = p[p > 0]
        entropy_rank = float(np.exp(-np.sum(pp * np.log(pp))))
    s2_ratio = float(s[1] / s[0]) if s.size >= 2 and s[0] > 0 else 0.0
    return {
        "effective_rank_version": EFFECTIVE_RANK_VERSION,
        "singular_values": s.tolist(),
        "first_direction_energy_fraction": energy_first,
        "s2_over_s1": s2_ratio,
        "effective_rank_participation": participation,
        "effective_rank_entropy": entropy_rank,
        "near_rank1_descriptive": bool(
            s2_ratio < RANK1_S2_RATIO_MAX and energy_first > RANK1_ENERGY_MIN
        ),
        "claim_boundary": (
            "local standardized functional collinearity only; not global "
            "slow-manifold dimensionality"
        ),
    }


def bootstrap_rank(sample_matrices, *, n_boot=2000, seed=0, ci=(2.5, 97.5)):
    """Bootstrap the mean standardized response matrix over seeds/microstates."""
    X = np.asarray(sample_matrices, float)
    if X.ndim != 3 or X.shape[0] < 2 or not np.all(np.isfinite(X)):
        raise ValueError("sample_matrices must be finite sample x output x coordinate")
    if int(n_boot) < 20:
        raise ValueError("n_boot must be at least 20")
    rng = np.random.default_rng(seed)
    ratios, energies, participation = [], [], []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, X.shape[0], size=X.shape[0])
        summary = rank_summary(np.mean(X[idx], axis=0))
        ratios.append(summary["s2_over_s1"])
        energies.append(summary["first_direction_energy_fraction"])
        participation.append(summary["effective_rank_participation"])
    ratio_ci = np.percentile(ratios, ci)
    energy_ci = np.percentile(energies, ci)
    participation_ci = np.percentile(participation, ci)
    point = rank_summary(np.mean(X, axis=0))
    return {
        "effective_rank_version": EFFECTIVE_RANK_VERSION,
        "n_samples": int(X.shape[0]),
        "n_boot": int(n_boot),
        "point": point,
        "s2_over_s1_ci": ratio_ci.tolist(),
        "first_direction_energy_fraction_ci": energy_ci.tolist(),
        "effective_rank_participation_ci": participation_ci.tolist(),
        "rank1_supported": bool(
            ratio_ci[1] < RANK1_S2_RATIO_MAX
            and energy_ci[0] > RANK1_ENERGY_MIN
        ),
        "rank1_thresholds": {
            "s2_over_s1_upper_max": RANK1_S2_RATIO_MAX,
            "first_energy_lower_min": RANK1_ENERGY_MIN,
        },
    }
