"""E-threshold heterogeneity layer for the SEF-HFO LIF rate field (spec §5.2).

Effective transfer of a patch whose excitatory thresholds are spread as
N(vth_mean, vth_std^2):  Φ_eff(μ,σ) = ∫ Φ_LIF(μ,σ; v) N(v; vth_mean, vth_std) dv.

DISCIPLINE: narrowing vth_std changes Φ_eff's slope/curvature AND (by Jensen)
its mean level. Every experiment must pair the raw narrowing with a
mean-matched control (mean_match_vth) — spec §5.0 Contract 2. Direction of the
effect is COMPUTED at the operating point, never assumed (spec §7).
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial.hermite_e import hermegauss

from src.sef_hfo_lif import lif_rate, V_TH

_GH_NODES, _GH_WEIGHTS = hermegauss(24)          # probabilists' Hermite (weight e^{-x^2/2})
_GH_WSUM = _GH_WEIGHTS.sum()                     # = sqrt(2 pi)


def phi_eff_vth(mu: float, sigma: float, tau_m: float, tau_ref: float,
                vth_mean: float = V_TH, vth_std: float = 0.0) -> float:
    """Φ_LIF averaged over v_th ~ N(vth_mean, vth_std^2), via Gauss–Hermite.

    vth_std=0 returns exactly lif_rate(mu,sigma,tau_m,tau_ref,v_th=vth_mean).
    Returns rate in kHz.
    """
    if vth_std <= 0.0:
        return lif_rate(mu, sigma, tau_m, tau_ref, v_th=vth_mean)
    vths = vth_mean + vth_std * _GH_NODES
    rates = np.array([lif_rate(mu, sigma, tau_m, tau_ref, v_th=float(v)) for v in vths])
    return float(np.sum(_GH_WEIGHTS * rates) / _GH_WSUM)
