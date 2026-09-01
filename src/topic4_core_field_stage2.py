"""Stage 2: the learned axial pathology field (spec section 4.1 / section 8).

The only learned object is a low-dimensional spatial field. Everything else --
the shared axis, the E->E anisotropy, the LIF/synapse/slow-variable equations,
the virtual contacts and the read-out -- stays frozen.

Parameter vector: alpha in R^M (softmax weights over a partition-of-unity axial
basis) followed by log rho, the FIXED-AREA aspect ratio. Softmax is shift
invariant, so alpha is centred internally and only M of the M+1 numbers are free.
"""
from __future__ import annotations

import numpy as np

from src.topic4_core_field import (
    EPS, SIGMA_S_FACTOR, axial_basis_centers, partition_of_unity,
    project_to_budget, shape_metrics)

# rho beyond this collapses the corridor to a line or spreads it to a disc, at
# which point the budget projection is projecting a degenerate field.
LOG_RHO_LIMIT = float(np.log(4.0))


def N_PARAMS(M):
    """alpha (M) + log rho (1). Softmax removes one degree of freedom, so M are free."""
    return int(M) + 1


def uniform_theta(M):
    """alpha = 0, log rho = 0 -- exactly the Stage 1 uniform_axial arm."""
    return np.zeros(int(M) + 1, float)


def _unpack(theta, M):
    theta = np.asarray(theta, float)
    alpha = theta[:M] - theta[:M].mean()          # centre: softmax is shift invariant
    log_rho = float(np.clip(theta[M], -LOG_RHO_LIMIT, LOG_RHO_LIMIT))
    return alpha, float(np.exp(log_rho))


def params_to_q(theta, s, r, geom):
    """Raw field before the budget projection."""
    M = int(geom["M"])
    alpha, rho = _unpack(theta, M)
    kappa = axial_basis_centers(geom["s_support"], M)
    sigma_s = SIGMA_S_FACTOR * (kappa[1] - kappa[0]) * rho     # axial scale x rho
    sigma_perp = float(geom["sigma_perp"]) / rho               # transverse scale / rho
    a = alpha - alpha.max()
    pi = np.exp(a) / np.exp(a).sum()
    profile = partition_of_unity(np.asarray(s, float), kappa, sigma_s) @ pi
    return profile * np.exp(-np.asarray(r, float) ** 2 / (2 * sigma_perp ** 2)) + EPS


def params_to_h(theta, s, r, geom, target_count):
    h, _ = project_to_budget(params_to_q(theta, s, r, geom), target_count)
    return h


def shape_of(theta, s, r, geom, target_count):
    """h-weighted geometry of a candidate, for the equivalent-optimum family."""
    h = params_to_h(theta, s, r, geom, target_count)
    centers = (-float(geom["sep"]) / 2.0, +float(geom["sep"]) / 2.0)
    return shape_metrics(h, s, r, core_centers_s=centers)
