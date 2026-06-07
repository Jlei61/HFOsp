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
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq

from src.sef_hfo_lif import lif_rate, V_TH, V_RESET

# Fixed (deterministic) Gauss–Legendre nodes on [-1, 1].  Deterministic nodes make
# phi_eff_vth SMOOTH in mu (the nodes do not move when mu changes during a finite
# difference), which a quad-based adaptive integrator would not guarantee — needed
# for eff_gain_curvature's second derivative.
_GL_NODES, _GL_WEIGHTS = leggauss(96)


def phi_eff_vth(mu: float, sigma: float, tau_m: float, tau_ref: float,
                vth_mean: float = V_TH, vth_std: float = 0.0) -> float:
    """Φ_LIF averaged over a TRUNCATED Gaussian threshold v_th ~ N(vth_mean, vth_std^2)
    restricted to the physical support v_th >= V_RESET, renormalized by the retained
    mass.  Returns rate in kHz.

    WHY TRUNCATED (2026-06-07 fix):  an UNbounded Gaussian samples v_th < V_RESET,
    where the Siegert formula returns a NEGATIVE firing rate (a threshold below reset
    is unphysical).  The previous implementation clamped those with ``max(0, .)``,
    which (a) silently masked illegal samples and (b) made Φ_eff NON-MONOTONIC in
    vth_mean — a higher mean threshold could give a *higher* mean rate.  Truncating
    at V_RESET and renormalizing removes both pathologies; lif_rate now also RAISES
    below reset so any future leak fails loudly.

    vth_std=0 returns exactly lif_rate(mu,sigma,tau_m,tau_ref,v_th=vth_mean).

    GAP-LIMIT CAVEAT:  with V_TH - V_RESET = 7 mV at the canonical sub-reset operating
    point, lif_rate has a steep reset knee just above V_RESET.  vth_std large enough to
    place real mass in that knee makes Φ_eff dominated by near-saturated cells (reset-
    floor, not collective gain).  The usable heterogeneity range is therefore gap-
    limited (locked: wide vth_std=1.5, narrow=0.5); see spec §5.2 and the knee-gate
    test ``test_locked_std_params_stay_out_of_reset_knee``.
    """
    if vth_std <= 0.0:
        return lif_rate(mu, sigma, tau_m, tau_ref, v_th=vth_mean)
    hi = vth_mean + 8.0 * vth_std            # upper tail; lower bound is the V_RESET floor
    half = 0.5 * (hi - V_RESET)
    mid = 0.5 * (hi + V_RESET)
    vths = mid + half * _GL_NODES            # GL nodes mapped onto [V_RESET, hi]
    w_gauss = np.exp(-0.5 * ((vths - vth_mean) / vth_std) ** 2)   # unnormalized Gaussian weight
    rates = np.array([lif_rate(mu, sigma, tau_m, tau_ref, v_th=float(v)) for v in vths])
    num = float(np.sum(_GL_WEIGHTS * w_gauss * rates))   # the GL ``half`` Jacobian cancels in num/den
    den = float(np.sum(_GL_WEIGHTS * w_gauss))           # = renormalization by retained truncated mass
    return num / den


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
                   bracket: tuple = (V_RESET + 0.5, V_TH + 17.0)) -> float:
    """Solve for vth_mean such that phi_eff_vth(...; vth_mean, vth_std) == target_rate
    at the operating input mu. Restores the baseline mean rate after a variance
    change (spec §5.0 Contract 2). Raises ValueError if target unreachable in bracket.

    The default bracket starts just ABOVE V_RESET (the old default V_TH-8=10 reached
    into the sub-reset region) and extends well above V_TH so a wide-variance baseline
    — which needs a higher mean to compensate the truncated-Jensen lift — stays inside.
    phi_eff_vth is strictly decreasing in vth_mean (truncated, monotone), so the root
    is unique."""
    def g(vm):
        return phi_eff_vth(mu, sigma, tau_m, tau_ref, vth_mean=vm, vth_std=vth_std) - target_rate
    lo, hi = bracket
    if g(lo) * g(hi) > 0:
        raise ValueError(
            f"target_rate={target_rate:.5g} not bracketed by vth_mean∈{bracket} "
            f"at mu={mu}, sigma={sigma}, vth_std={vth_std}; widen the bracket.")
    return float(brentq(g, lo, hi, xtol=1e-8))
