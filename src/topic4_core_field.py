"""Topic 4 axis-constrained data-driven pathology field (spec rev3).

Pure computation: no simulation, no engine import.
"""
from __future__ import annotations

import numpy as np

M_DEFAULT = 9
EPS = 1e-3
TAU_H = 0.25
A0 = 1.5
B0 = 1.5
AXIAL_MARGIN = 2.0
SIGMA_S_FACTOR = 1.2
SHIFT_MM = 3.0


def axis_coords(pos, center, u_axis):
    """Axial (s) and transverse (r) coordinates. u_axis is undirected: flipping
    its sign negates both, which every score must be invariant to."""
    pos = np.asarray(pos, float)
    u = np.asarray(u_axis, float)
    u = u / np.linalg.norm(u)
    u_perp = np.array([-u[1], u[0]])
    d = pos - np.asarray(center, float)[None, :]
    return d @ u, d @ u_perp


def axial_basis_centers(s_support, M=M_DEFAULT):
    return np.linspace(float(s_support[0]), float(s_support[1]), int(M))


def partition_of_unity(s, kappa, sigma_s):
    """Normalised Gaussian bases: rows sum to exactly 1 (spec 4.1)."""
    s = np.asarray(s, float)
    kappa = np.asarray(kappa, float)
    logw = -((s[:, None] - kappa[None, :]) ** 2) / (2.0 * float(sigma_s) ** 2)
    logw -= logw.max(axis=1, keepdims=True)
    w = np.exp(logw)
    return w / w.sum(axis=1, keepdims=True)


from scipy.special import expit
from scipy.stats import truncnorm

V_BASE = 18.0
V_RESET = 11.0
CORE_MEAN = 17.5
CORE_STD = 1.0


def sample_core_quantiles(n_E, seed):
    """One uniform quantile per E neuron, drawn once and frozen. Position- and
    field-independent, so every arm shares the same latent draw."""
    return np.random.default_rng(int(seed)).uniform(0.0, 1.0, size=int(n_E))


def core_thresholds(u, core_mean=CORE_MEAN, core_std=CORE_STD, v_reset=V_RESET):
    """Truncated-normal inverse transform: same distribution as the engine's
    rejection sampler, but deterministic per neuron. Bitwise reproduction of the
    legacy draw is impossible -- rejection makes its stream position data-dependent."""
    a = (float(v_reset) - float(core_mean)) / float(core_std)
    return truncnorm.ppf(np.asarray(u, float), a=a, b=np.inf,
                         loc=float(core_mean), scale=float(core_std))


def signed_depth(v_core, v_base=V_BASE):
    return float(v_base) - np.asarray(v_core, float)


def project_to_budget(q, target_count, tau_h=TAU_H, eps=EPS, max_iter=200):
    """Bisect lambda so that sum_i h_i == target_count.

    h is strictly decreasing in lambda, so the root is unique. This is a
    LEVEL-SET operation: the region's size is pinned by the budget and q only
    sets its shape (spec 4.4).
    """
    q = np.asarray(q, float)
    if not np.isfinite(q).all():
        raise ValueError("project_to_budget: q contains non-finite values")
    if (q + eps <= 0).any():
        raise ValueError("project_to_budget: q + eps must be positive")
    target = float(target_count)
    if not np.isfinite(target) or not (0.0 < target < q.size):
        raise ValueError(
            f"project_to_budget: target_count must lie in (0, {q.size}), got {target}")

    lq = np.log(q + eps)
    lo, hi = lq.min() - 20.0, lq.max() + 20.0
    for _ in range(max_iter):
        lam = 0.5 * (lo + hi)
        if expit((lq - lam) / tau_h).sum() > target:
            lo = lam
        else:
            hi = lam
    lam = 0.5 * (lo + hi)
    return expit((lq - lam) / tau_h), lam


def build_vth(h, d, n_total, n_E, v_base=V_BASE):
    """Per-neuron threshold vector for the engine. I neurons keep baseline."""
    vth = np.full(int(n_total), float(v_base))
    vth[:int(n_E)] = float(v_base) - np.asarray(h, float) * np.asarray(d, float)
    return vth
