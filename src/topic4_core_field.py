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
