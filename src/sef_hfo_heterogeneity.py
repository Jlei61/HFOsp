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
from scipy.optimize import brentq

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


def eff_gain_curvature(mu: float, sigma: float, tau_m: float, tau_ref: float,
                       vth_mean: float = V_TH, vth_std: float = 0.0,
                       h: float = 1e-2) -> dict:
    """Slope (∂Φ_eff/∂μ) and curvature (∂²Φ_eff/∂μ²) at input mu, by central
    finite difference. Returns {"slope","curvature","rate"} (kHz, kHz/mV, kHz/mV²).
    The SIGN of these is the computed output of the forced chain — callers report
    it, they do not assume it (spec §7 non-circularity)."""
    f0 = phi_eff_vth(mu, sigma, tau_m, tau_ref, vth_mean, vth_std)
    fp = phi_eff_vth(mu + h, sigma, tau_m, tau_ref, vth_mean, vth_std)
    fm = phi_eff_vth(mu - h, sigma, tau_m, tau_ref, vth_mean, vth_std)
    return {"rate": f0, "slope": (fp - fm) / (2 * h), "curvature": (fp - 2 * f0 + fm) / (h * h)}


def mean_match_vth(target_rate: float, mu: float, sigma: float,
                   tau_m: float, tau_ref: float, vth_std: float,
                   bracket: tuple = (V_TH - 8.0, V_TH + 8.0)) -> float:
    """Solve for vth_mean such that phi_eff_vth(...; vth_mean, vth_std) == target_rate
    at the operating input mu. Restores the baseline mean rate after a variance
    change (spec §5.0 Contract 2). Raises ValueError if target unreachable in bracket."""
    def g(vm):
        return phi_eff_vth(mu, sigma, tau_m, tau_ref, vth_mean=vm, vth_std=vth_std) - target_rate
    lo, hi = bracket
    if g(lo) * g(hi) > 0:
        raise ValueError(
            f"target_rate={target_rate:.5g} not bracketed by vth_mean∈{bracket} "
            f"at mu={mu}, sigma={sigma}, vth_std={vth_std}; widen the bracket.")
    return float(brentq(g, lo, hi, xtol=1e-8))
