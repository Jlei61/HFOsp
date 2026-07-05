"""M3A-v2.2 global recovery sensors (pure leaf; engine + runner import it).

M(t)=mean activity, B(t)=soft recruited area, Pi(t)=participation/globality,
chi_G=smooth-AND Hill trigger. Canonical math: docs/snn_core_model_equations.md §B6.
Kept dependency-free (NumPy only) so src/snn_engine/slow_field.py can import it
online without pulling the readout module topic4_m3a_v2_phenotype.
"""
from __future__ import annotations

import numpy as np


def hill(z, z50, n):
    """H_n(z; z50) = z^n / (z^n + z50^n), z clipped at 0. Elementwise."""
    z = np.maximum(np.asarray(z, float), 0.0)
    zn = z ** n
    return zn / (zn + z50 ** n)


def global_M(rE_fast):
    """M(t) = mean_x r_tilde_E."""
    return float(np.mean(np.asarray(rE_fast, float)))


def global_B(rE_fast, r_A, Delta_A):
    """B(t) = mean_x sigma((r_tilde_E - r_A)/Delta_A): soft recruited area."""
    z = (np.asarray(rE_fast, float) - r_A) / Delta_A
    return float(np.mean(1.0 / (1.0 + np.exp(-z))))


def global_participation(rE_fast, eps=1e-12):
    """Pi(t) = (sum r)^2 / (N * sum r^2): ~1/N single hot cell, ~1 uniform."""
    r = np.asarray(rE_fast, float)
    s1 = r.sum(); s2 = (r * r).sum()
    return float(s1 * s1 / (r.size * s2 + eps))


def chi_G(M, B, Pi, M50, B50, Pi50, n_M, n_B, n_Pi):
    """Smooth-AND trigger chi_G = H(M)*H(B)*H(Pi) in [0,1)."""
    return float(hill(M, M50, n_M) * hill(B, B50, n_B) * hill(Pi, Pi50, n_Pi))
