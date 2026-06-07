"""E-threshold heterogeneity layer for the SEF-HFO LIF rate field (spec §5.2).

Effective transfer of a patch whose excitatory thresholds are spread as a
**TRUNCATED, renormalized** Gaussian — N(vth_mean, vth_std^2) restricted to the
physical support v_th >= V_RESET (a threshold below reset is unphysical and gives a
negative Siegert rate):

    Φ_eff(μ,σ) = ∫_{V_RESET}^∞ Φ_LIF(μ,σ; v) N(v; vth_mean, vth_std) dv
                 / ∫_{V_RESET}^∞ N(v; vth_mean, vth_std) dv.

DISCIPLINE: narrowing vth_std changes Φ_eff's slope/curvature AND (by Jensen)
its mean level. Every experiment must pair the raw narrowing with a
mean-matched control (mean_match_vth) — spec §5.0 Contract 2. Direction of the
effect is COMPUTED at the operating point, never assumed (spec §7). The usable
vth_std is GAP-LIMITED by V_TH-V_RESET at the sub-reset operating point (locked
wide=1.5 / narrow=0.5); see phi_eff_vth and spec §5.2.
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq

from src.sef_hfo_lif import lif_rate, V_TH, V_RESET
# Field-loop reuse for the spatial patch integrator (Task 8). Aliased to keep the
# integrator body terse; the spatial functions live at the bottom of the module.
from src.sef_hfo_field import anisotropic_gaussian, isotropic_gaussian, convolve_periodic, _grid
from src.sef_hfo_lif import (
    W_EE as _W_EE, W_EI as _W_EI, W_IE as _W_IE, W_II as _W_II,
    C_EE as _C_EE, C_EI as _C_EI, C_IE as _C_IE, C_II as _C_II,
    TAU_ME as _TME, TAU_MI as _TMI, TREF_E as _TRE, TREF_I as _TRI,
    JX_E as _JXE, JX_I as _JXI, TAU_AMPA as _TA, TAU_GABA as _TG,
    ELL_PAR as _EP, ELL_PERP as _EPP, L_INH as _LI, DETECT as _DET,
    _DEFAULT_N as _N, _DEFAULT_L as _L,
)

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
    result = num / den if den > 0.0 else float("nan")
    if not np.isfinite(result):
        # Degenerate input: the truncated mass underflowed (den→0, e.g. the whole
        # distribution sits below V_RESET) or a sampled rate was non-finite (lif_rate
        # quad overflow at an extreme threshold).  Raise loudly rather than emit a silent
        # NaN — same loud-failure philosophy as lif_rate's sub-reset guard (CLAUDE.md §6).
        raise ValueError(
            f"phi_eff_vth: non-finite effective rate (num={num:.3g}, den={den:.3g}) at "
            f"mu={mu:.4g}, sigma={sigma:.4g}, vth_mean={vth_mean:.4g}, vth_std={vth_std:.4g} "
            f"— degenerate threshold distribution relative to V_RESET={V_RESET}.")
    return result


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


# ---------------------------------------------------------------------------
# Spatial core/surround patch integrator (Task 8) — applies the truncated
# threshold heterogeneity to a finite-pulse field, core disk vs surround.
# ---------------------------------------------------------------------------

def hetero_lut(sigma: float, vth_mean: float, vth_std: float,
               lo: float = -12.0, hi: float = 45.0, npts: int = 5000):
    """(mus, rates) LUT for the E effective transfer Φ_eff(μ; vth_mean,vth_std) at
    fixed sigma (the TRUNCATED phi_eff_vth — inherits the 2026-06-07 fix). np.interp
    over this is the per-region E transfer."""
    mus = np.linspace(lo, hi, npts)
    rates = np.array([phi_eff_vth(m, sigma, _TME, _TRE, vth_mean, vth_std) for m in mus])
    return mus, rates


def integrate_hetero_field(op, stim_fn, *, x_patch: float, r_patch: float,
                           vth_std_core: float, vth_std_surround: float,
                           vth_mean_core: float, vth_mean_surround: float,
                           dt: float = 0.25, t_max: float = 300.0,
                           theta_EE: float = 0.0, n: int = _N, L: float = _L,
                           ell_par: float = _EP, ell_perp: float = _EPP, l_inh: float = _LI,
                           return_peak_field: bool = False):
    """MINIMAL finite-pulse heterogeneity integrator (NOT a full mirror of
    integrate_lif_field). Same core Euler loop and synaptic filters, but DROPS
    recovery (b_a/tau_a), coherence (coh_len) and axis_accum — all out of this
    round's scope (finite-pulse only, no recovery, no noise). The substantive
    addition is a spatially heterogeneous E threshold distribution:
    core disk (center (x_patch,0), radius r_patch) uses (vth_mean_core, vth_std_core);
    surround uses (vth_mean_surround, vth_std_surround); I population unchanged
    (homogeneous Φ_LIF). Per-region E transfer = two LUTs blended by the patch mask.

    Contract: surround params are REQUIRED kwargs (no defaults) so a caller cannot
    silently tune the core without defining the background (spec §5.2)."""
    wee = float(op.get("w_ee_mult", 1.0)) * _W_EE
    KEE = anisotropic_gaussian(n, L, ell_par, ell_perp, theta_EE)
    KI = isotropic_gaussian(n, L, l_inh)
    X, Y = _grid(n, L)
    core = ((X - x_patch) ** 2 + Y ** 2) <= r_patch ** 2

    musC, rC = hetero_lut(op["sE"], vth_mean_core, vth_std_core)
    musS, rS = hetero_lut(op["sE"], vth_mean_surround, vth_std_surround)
    musI = np.linspace(-12.0, 45.0, 5000)
    rI_lut = np.array([lif_rate(m, op["sI"], _TMI, _TRI) for m in musI])

    def fE(mu):
        out = np.interp(mu, musS, rS)
        out[core] = np.interp(mu[core], musC, rC)
        return out
    fI = lambda mu: np.interp(mu, musI, rI_lut)

    muxE = _TME * _JXE * op["nuext"]; muxI = _TMI * _JXI * op["nuext"]
    rE = np.full((n, n), op["nuE"]); rI = np.full((n, n), op["nuI"])
    sEE = convolve_periodic(rE, KEE).copy(); sEI = convolve_periodic(rI, KI).copy()
    sIE = convolve_periodic(rE, KI).copy(); sII = convolve_periodic(rI, KI).copy()

    nsteps = int(t_max / dt)
    ext = np.empty(nsteps); front = np.empty(nsteps); thr = op["nuE"] + _DET
    peak_field = rE.copy(); peak_ext = -1.0
    for t in range(nsteps):
        stim = stim_fn(t * dt)
        sEE += dt / _TA * (convolve_periodic(rE, KEE) - sEE)
        sEI += dt / _TG * (convolve_periodic(rI, KI) - sEI)
        sIE += dt / _TA * (convolve_periodic(rE, KI) - sIE)
        sII += dt / _TG * (convolve_periodic(rI, KI) - sII)
        muE = _TME * (_C_EE * wee * sEE - _C_EI * _W_EI * sEI) + muxE + stim
        muI = _TMI * (_C_IE * _W_IE * sIE - _C_II * _W_II * sII) + muxI
        rE = rE + dt / _TME * (-rE + fE(muE))
        rI = rI + dt / _TMI * (-rI + fI(muI))
        m = rE > thr
        ext[t] = m.mean()
        front[t] = float(X[m].max()) if m.any() else np.nan
        if ext[t] > peak_ext:
            peak_ext = ext[t]; peak_field = rE.copy()
    if return_peak_field:
        return ext, front, peak_field
    return ext, front
