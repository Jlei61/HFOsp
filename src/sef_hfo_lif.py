"""Canonical LIF-derived rate field for Topic-4 SEF-HFO modeling.

Transfer function: Φ_LIF(μ, σ) — Siegert formula via erfcx (numerically stable).
Connectivity: Brunel (2000) balanced-network Table-1 constants (physical mV / kHz).
Field: 2-D spatial rate field with anisotropic E→E kernel (rotatable axis), isotropic
inhibition, AMPA/GABA synaptic low-pass, and optional spike-frequency adaptation (recovery).

LINEAGE NOTE
------------
F_eff (sigmoid) in src/sef_hfo_field.py is the DEMOTED prior coarse-graining (audit trail),
retained per docs/archive/topic4/sef_itp_phase4_v2/lif_rate_field_theory_2026-06-03.md.
This module (src/sef_hfo_lif.py) is the canonical replacement used from Step 0b onward.

Validated provenance
--------------------
* ``lif_rate``, ``nu_theta_pop``, ``_ms``, ``mean_field``, ``_lut``, ``classify_response``
  transcribed from ``scripts/sef_hfo_step0b_lif.py`` (2026-06-03 run, all 4 ratio points
  reproduced self_limited_propagation).
* ``integrate_lif_field`` merges step0b's ``integrate`` (recovery b_a/tau_a) with
  step0d's ``integrate_dir`` (rotatable anisotropy axis theta_EE via angle argument).
* ``lif_gains`` is a thin finite-difference wrapper around ``lif_rate`` (not in source
  scripts; derived directly from the validated transfer function).
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.special import erfcx

from src.sef_hfo_field import anisotropic_gaussian, isotropic_gaussian, convolve_periodic, _grid

# ---------------------------------------------------------------------------
# Brunel Table-1 constants (physical units: mV, ms, kHz)
# ---------------------------------------------------------------------------
V_TH: float = 18.0       # threshold (mV above reset baseline)
V_RESET: float = 11.0    # reset potential (mV)
TAU_ME: float = 20.0     # E membrane time constant (ms)
TAU_MI: float = 10.0     # I membrane time constant (ms)
TREF_E: float = 2.0      # E refractory period (ms)
TREF_I: float = 1.0      # I refractory period (ms)
G_INH: float = 3.6       # inhibitory weight ratio

C_EE: int = 800          # E→E in-degree
W_EE: float = 0.1575     # E→E synaptic weight (mV)
C_IE: int = 800          # E→I in-degree
W_IE: float = 0.2625     # E→I synaptic weight (mV)
C_EI: int = 200          # I→E in-degree
W_EI: float = 1.07 * G_INH * 0.1575   # I→E synaptic weight (mV)
C_II: int = 200          # I→I in-degree
W_II: float = G_INH * 0.2625          # I→I synaptic weight (mV)

JX_E: float = 0.455      # external drive weight E (mV)
JX_I: float = 0.85       # external drive weight I (mV)
TAU_SYN: float = 4.2     # synaptic rise-time constant (ms)
ALPHA: float = 2.065     # synaptic amplitude factor

TAU_AMPA: float = 3.5    # AMPA (E-presynaptic) low-pass time constant (ms)
TAU_GABA: float = 18.0   # GABA (I-presynaptic) low-pass time constant (ms)

# Spatial kernel widths (mm)
ELL_PAR: float = 0.54    # E→E anisotropic kernel, parallel axis
ELL_PERP: float = 0.27   # E→E anisotropic kernel, perpendicular axis
L_INH: float = 0.25      # inhibitory kernel width (isotropic)

# Field grid defaults (match step0b validated run)
_DEFAULT_N: int = 96
_DEFAULT_L: float = 12.0

# Detection threshold above rest (kHz)
DETECT: float = 0.005    # step0b DETECT constant

# Wavefront classifier thresholds (step0b validated)
RUNAWAY_FRAC: float = 0.5   # active fraction → runaway
RETURN_FRAC: float = 0.2    # final/max fraction → returned
TRAVEL_MM: float = 3.0      # front advance → propagation

# Step0b off-center disk stimulus defaults
_STIM_X0: float = -3.0   # disk center x (mm)
_STIM_R: float = 2.0     # disk radius (mm)
_STIM_T: float = 30.0    # stimulus duration (ms)


# ---------------------------------------------------------------------------
# Transfer function
# ---------------------------------------------------------------------------

def lif_rate(mu: float, sigma: float, tau_m: float, tau_ref: float) -> float:
    """Siegert formula for the LIF firing rate (kHz).

    Numerically stable via ``scipy.special.erfcx``.  Physical units:
    mu, sigma in mV; tau_m, tau_ref in ms; return value in kHz.
    """
    y_th = (V_TH - mu) / sigma
    y_r = (V_RESET - mu) / sigma
    integ, _ = quad(lambda x: erfcx(-x), y_r, y_th, limit=200)
    return 1.0 / (tau_ref + tau_m * np.sqrt(np.pi) * integ)


# ---------------------------------------------------------------------------
# Mean-field self-consistency
# ---------------------------------------------------------------------------

def nu_theta_pop() -> float:
    """External Poisson drive scale ν_θ (kHz) — combined E+I population estimate."""
    def f(Jx, tm):
        return (
            (0.5 * ALPHA * Jx * np.sqrt(TAU_SYN)
             + np.sqrt((0.5 * ALPHA * Jx * np.sqrt(TAU_SYN)) ** 2 + 4 * tm * Jx * V_TH))
            / (2 * tm * Jx)
        ) ** 2
    return 0.8 * f(JX_E, TAU_ME) + 0.2 * f(JX_I, TAU_MI)


def _ms(nuE: float, nuI: float, nuext: float, w_ee_mult: float = 1.0):
    """Compute mean (μ) and s.d. (σ) of the synaptic input for E and I populations.

    w_ee_mult: multiplicative gain on W_EE (default 1.0; >1 probes pathological gain regime).
    """
    wee = w_ee_mult * W_EE
    muE = TAU_ME * (C_EE * wee * nuE - C_EI * W_EI * nuI) + TAU_ME * JX_E * nuext
    muI = TAU_MI * (C_IE * W_IE * nuE - C_II * W_II * nuI) + TAU_MI * JX_I * nuext
    sE = np.sqrt(max(TAU_ME * (C_EE * wee ** 2 * nuE + C_EI * W_EI ** 2 * nuI)
                     + TAU_ME * JX_E ** 2 * nuext, 1e-9))
    sI = np.sqrt(max(TAU_MI * (C_IE * W_IE ** 2 * nuE + C_II * W_II ** 2 * nuI)
                     + TAU_MI * JX_I ** 2 * nuext, 1e-9))
    return muE, muI, sE, sI


# Multi-start seeds (low-rate; the model's operating point is the stable
# sub-threshold rest = lowest-nuE clean root) and convergence tolerances.
# _RESID_TOL is set ~10^8x above the empirical residual floor of the known-good
# ratio=1.0 root (~3e-14 across all seeds) and well below test 1's 1e-4
# self-consistency bar.  _ROOT_RATE_RELTOL clusters seeds onto distinct roots so
# a genuine second (e.g. saturated high-rate) branch is surfaced, not averaged away.
_MEAN_FIELD_SEEDS = ([0.0005, 0.002], [0.001, 0.005], [0.002, 0.008], [0.005, 0.012])
_RESID_TOL: float = 1e-6
_ROOT_RATE_RELTOL: float = 0.05


def mean_field(ratio: float, w_ee_mult: float = 1.0, strict: bool = True) -> dict:
    """Solve the self-consistent mean-field equations (multi-start fsolve).

    The model's operating point is the **stable sub-threshold rest** = the
    **lowest-nuE clean root**.  Multi-start guards against silently locking onto
    the wrong branch in the pathological-gain regime (w_ee_mult > 1), where a
    high-rate saturated root can coexist with the rest state and would otherwise
    be accepted just because a seed happened to converge to it first.

    Parameters
    ----------
    ratio:
        External drive as a fraction of ν_θ (1.0 = threshold-balanced regime).
    w_ee_mult:
        Multiplicative gain on W_EE.  Values > 1 probe the pathological-gain
        regime.  Stored in the returned op so ``integrate_lif_field`` uses the
        SAME recurrent gain as the operating point it was handed.
    strict:
        If True (default), raise ``RuntimeError`` when no seed converges to a
        clean root (loud failure beats a silent wrong root).

    Returns
    -------
    dict with keys: nuE, nuI, muE, sE, muI, sI, nuext, w_ee_mult,
        resid_norm, converged, fsolve_ier, n_clean_roots
        (rates in kHz, potentials in mV).
    """
    nuext = ratio * nu_theta_pop()

    def resid(nu):
        muE, muI, sE, sI = _ms(max(nu[0], 1e-9), max(nu[1], 1e-9), nuext, w_ee_mult)
        return [
            nu[0] - lif_rate(muE, sE, TAU_ME, TREF_E),
            nu[1] - lif_rate(muI, sI, TAU_MI, TREF_I),
        ]

    clean = []
    for x0 in _MEAN_FIELD_SEEDS:
        sol, _info, ier, _msg = fsolve(resid, x0, full_output=True)
        r = resid(sol)
        rnorm = float(np.hypot(r[0], r[1]))
        nuE, nuI = float(sol[0]), float(sol[1])
        if ier == 1 and rnorm < _RESID_TOL and nuE > 0 and nuI > 0:
            clean.append(dict(nuE=nuE, nuI=nuI, resid_norm=rnorm, ier=int(ier)))

    if not clean:
        if strict:
            raise RuntimeError(
                f"mean_field did not converge to a clean root "
                f"(ratio={ratio}, w_ee_mult={w_ee_mult}); tried "
                f"{len(_MEAN_FIELD_SEEDS)} seeds, none met ier==1 & resid<{_RESID_TOL:.0e} "
                f"& positive rates."
            )
        sol, _info, ier, _msg = fsolve(resid, _MEAN_FIELD_SEEDS[1], full_output=True)
        r = resid(sol)
        best = dict(nuE=float(sol[0]), nuI=float(sol[1]),
                    resid_norm=float(np.hypot(r[0], r[1])), ier=int(ier))
        n_roots = 0
        converged = False
        roots_out = [dict(nuE=best["nuE"], nuI=best["nuI"])]
    else:
        # Dedup seeds onto distinct roots; operating point = lowest-nuE root.
        distinct = []
        for d in sorted(clean, key=lambda d: d["nuE"]):
            if not any(abs(d["nuE"] - e["nuE"]) <= _ROOT_RATE_RELTOL * max(e["nuE"], 1e-12)
                       for e in distinct):
                distinct.append(d)
        best = distinct[0]
        n_roots = len(distinct)
        converged = True
        roots_out = [dict(nuE=d["nuE"], nuI=d["nuI"]) for d in distinct]

    nuE, nuI = best["nuE"], best["nuI"]
    muE, muI, sE, sI = _ms(nuE, nuI, nuext, w_ee_mult)
    return dict(nuE=nuE, nuI=nuI, muE=muE, sE=sE, muI=muI, sI=sI, nuext=nuext,
                w_ee_mult=w_ee_mult, resid_norm=best["resid_norm"],
                converged=converged, fsolve_ier=best["ier"], n_clean_roots=n_roots,
                roots=roots_out)


# ---------------------------------------------------------------------------
# Lookup table for the LIF transfer function
# ---------------------------------------------------------------------------

def _lut(sigma: float, tm: float, tref: float,
         lo: float = -12.0, hi: float = 45.0, npts: int = 5000):
    """Build a (mus, rates) lookup table for lif_rate at fixed sigma.

    Returns (mus_array, rates_array) — use np.interp for fast evaluation.
    """
    mus = np.linspace(lo, hi, npts)
    return mus, np.array([lif_rate(m, sigma, tm, tref) for m in mus])


# ---------------------------------------------------------------------------
# Linearized gains at the operating point
# ---------------------------------------------------------------------------

def lif_gains(op: dict) -> dict:
    """Compute G_μ = (∂ν/∂μ) · τ_m for E and I at the operating point.

    Uses a symmetric finite difference with h = 1e-3 mV.

    Parameters
    ----------
    op : dict
        Operating point dict as returned by ``mean_field``.

    Returns
    -------
    dict with keys ``"E"`` and ``"I"``, each a float (dimensionless).
    """
    h = 1e-3
    dnu_E = (lif_rate(op["muE"] + h, op["sE"], TAU_ME, TREF_E)
             - lif_rate(op["muE"] - h, op["sE"], TAU_ME, TREF_E)) / (2 * h)
    dnu_I = (lif_rate(op["muI"] + h, op["sI"], TAU_MI, TREF_I)
             - lif_rate(op["muI"] - h, op["sI"], TAU_MI, TREF_I)) / (2 * h)
    return {"E": dnu_E * TAU_ME, "I": dnu_I * TAU_MI}


# ---------------------------------------------------------------------------
# LIF rate-field integrator
# ---------------------------------------------------------------------------

def integrate_lif_field(
    op: dict,
    stim_fn,
    dt: float = 0.25,
    t_max: float = 300.0,
    b_a: float = 0.0,
    tau_a: float = 80.0,
    theta_EE: float = 0.0,
    n: int = _DEFAULT_N,
    L: float = _DEFAULT_L,
    ell_par: float = ELL_PAR,
    ell_perp: float = ELL_PERP,
    l_inh: float = L_INH,
    return_field: bool = False,
    return_peak_field: bool = False,
    coh_len: float | None = None,
    axis_accum: bool = False,
):
    """Integrate the 2-D LIF rate field.

    The E→E spatial kernel is anisotropic with orientation angle ``theta_EE``
    (radians); rotating theta_EE rotates the propagation template axis.
    The inhibitory kernel is isotropic.  Transfer function is Φ_LIF (Siegert)
    via lookup table.  Recovery (spike-frequency adaptation) is switchable via b_a.

    Parameters
    ----------
    op : dict
        Operating point from ``mean_field``.
    stim_fn : callable
        ``stim_fn(t)`` → 2-D ndarray of shape (n, n) or scalar 0.
        ``t`` is the current simulation time in ms.
    dt : float
        Time step (ms).
    t_max : float
        Total integration time (ms).
    b_a : float
        Recovery (adaptation) strength (mV·kHz⁻¹).  0 = OFF.
    tau_a : float
        Recovery time constant (ms).
    theta_EE : float
        E→E kernel orientation (radians).  0 = x-axis.
    n, L : int, float
        Grid points and physical size (mm).
    ell_par, ell_perp : float
        E→E kernel widths along / across the axis (mm).
    l_inh : float
        Inhibitory kernel width (mm).

    return_field : bool
        If True, also return the final rE field (shape (n, n)) as a third element.
        Default False — callers that only need (ext, front) are unaffected.
    return_peak_field : bool
        If True, also return the rE field at the timestep of MAXIMUM active
        fraction (peak extent) as a third element.  This is the snapshot needed
        to measure the spatial principal axis of a self-limited pulse (which has
        returned near rest by the final frame).  Takes precedence over
        ``return_field`` if both are set.
    coh_len : float or None
        If set, also compute a COHERENCE-based active fraction: the field is
        spatially smoothed with a Gaussian of length ``coh_len`` (mm) before
        thresholding, so per-pixel noise speckle is suppressed and only spatially
        coherent activity (the model's own event scale, ~ell_par) is counted.
        Returned as the LAST element of the tuple.  Used by the Step-1 noise
        detector (per contract step1_noise_contract §1, v1.1 coherence amendment).

    Returns
    -------
    ext : ndarray, shape (nsteps,)
        Per-timestep fraction of active pixels (rE > op["nuE"] + DETECT).
    front : ndarray, shape (nsteps,)
        Per-timestep maximum x-coordinate of active pixels (nan if none active).
    rE_snapshot : ndarray, shape (n, n) — only when ``return_peak_field`` or
        ``return_field`` is set.  Peak-extent field if ``return_peak_field``,
        else final field.
    ext_coh : ndarray, shape (nsteps,) — only when ``coh_len`` is set; appended
        LAST (before axis_accum).  Coherence-based active fraction.
    axis : dict — only when ``axis_accum`` is set (requires ``coh_len``); appended
        VERY LAST.  Aggregated centered second-moment of the smoothed active
        region over all "event-on" frames: ``{"Sxx","Sxy","Syy","n_onframes"}``.
        The principal eigenvector of [[Sxx,Sxy],[Sxy,Syy]] is the systematic
        event elongation axis across many events (per-event triggering asymmetry
        averages out) — the contract §5 aggregated direction measure.
    """
    wee = float(op.get("w_ee_mult", 1.0)) * W_EE   # recurrent E→E gain matches the op
    KEE = anisotropic_gaussian(n, L, ell_par, ell_perp, theta_EE)
    KI = isotropic_gaussian(n, L, l_inh)
    K_coh = isotropic_gaussian(n, L, coh_len) if coh_len is not None else None
    if axis_accum and K_coh is None:
        raise ValueError("axis_accum=True requires coh_len to be set")
    X_grid, Y_grid = _grid(n, L)

    lE = _lut(op["sE"], TAU_ME, TREF_E)
    lI = _lut(op["sI"], TAU_MI, TREF_I)
    fE = lambda mu: np.interp(mu, lE[0], lE[1])
    fI = lambda mu: np.interp(mu, lI[0], lI[1])

    muxE = TAU_ME * JX_E * op["nuext"]
    muxI = TAU_MI * JX_I * op["nuext"]

    rE = np.full((n, n), op["nuE"])
    rI = np.full((n, n), op["nuI"])
    a = np.full((n, n), op["nuE"])

    sEE = convolve_periodic(rE, KEE).copy()
    sEI = convolve_periodic(rI, KI).copy()
    sIE = convolve_periodic(rE, KI).copy()
    sII = convolve_periodic(rI, KI).copy()

    nsteps = int(t_max / dt)
    ext = np.empty(nsteps)
    front = np.empty(nsteps)
    ext_coh = np.empty(nsteps) if K_coh is not None else None
    thr = op["nuE"] + DETECT
    peak_field = rE.copy()
    peak_ext = -1.0
    Sxx = Sxy = Syy = 0.0
    n_onframes = 0

    for t in range(nsteps):
        stim = stim_fn(t * dt)
        cEE = convolve_periodic(rE, KEE)
        cEI = convolve_periodic(rI, KI)
        cIE = convolve_periodic(rE, KI)
        cII = convolve_periodic(rI, KI)
        sEE += dt / TAU_AMPA * (cEE - sEE)
        sEI += dt / TAU_GABA * (cEI - sEI)
        sIE += dt / TAU_AMPA * (cIE - sIE)
        sII += dt / TAU_GABA * (cII - sII)
        muE = TAU_ME * (C_EE * wee * sEE - C_EI * W_EI * sEI) + muxE + stim - b_a * a
        muI = TAU_MI * (C_IE * W_IE * sIE - C_II * W_II * sII) + muxI
        rE = rE + dt / TAU_ME * (-rE + fE(muE))
        rI = rI + dt / TAU_MI * (-rI + fI(muI))
        a = a + dt / tau_a * (-a + rE)
        m = rE > thr
        ext[t] = m.mean()
        front[t] = float(X_grid[m].max()) if m.any() else np.nan
        if ext_coh is not None:
            cf = convolve_periodic(rE, K_coh)
            m_coh = cf > thr
            ext_coh[t] = m_coh.mean()
            if axis_accum and m_coh.sum() >= 10:
                w = cf[m_coh] - op["nuE"]
                xs = X_grid[m_coh]
                ys = Y_grid[m_coh]
                xm = np.average(xs, weights=w)
                ym = np.average(ys, weights=w)
                Sxx += float(np.sum(w * (xs - xm) ** 2))
                Syy += float(np.sum(w * (ys - ym) ** 2))
                Sxy += float(np.sum(w * (xs - xm) * (ys - ym)))
                n_onframes += 1
        if ext[t] > peak_ext:
            peak_ext = ext[t]
            peak_field = rE.copy()

    out = [ext, front]
    if return_peak_field:
        out.append(peak_field)
    elif return_field:
        out.append(rE)
    if ext_coh is not None:
        out.append(ext_coh)
    if axis_accum:
        out.append(dict(Sxx=Sxx, Sxy=Sxy, Syy=Syy, n_onframes=n_onframes))
    return tuple(out) if len(out) > 2 else (ext, front)


# ---------------------------------------------------------------------------
# Response classifier
# ---------------------------------------------------------------------------

def classify_response(ext: np.ndarray, front: np.ndarray,
                      stim_x0: float = _STIM_X0, stim_r: float = _STIM_R,
                      dt: float = 0.25) -> tuple:
    """Classify the field response into one of five categories.

    Uses the wavefront metric (front advance along x) from the off-center disk
    stimulus (same as step0b validated run: stim_x0=-3.0, stim_r=2.0).

    Parameters
    ----------
    ext : ndarray
        Active fraction time series (from ``integrate_lif_field``).
    front : ndarray
        Front-position time series (from ``integrate_lif_field``).
    stim_x0, stim_r : float
        Stimulus disk center and radius (mm); the front baseline is x0 + r.

    Returns
    -------
    (label, info_dict)
        label : str — one of ``"extinction"``, ``"local_bump"``,
            ``"self_limited_propagation"``, ``"global_synchronous"``, ``"runaway"``.
        info_dict : dict — max_ext, adv_mm, returned (bool), dur_ms (requires dt).
    """
    max_ext = float(ext.max())
    final = float(ext[-1])
    adv = (float(np.nanmax(front) - (stim_x0 + stim_r))
           if np.any(np.isfinite(front)) else 0.0)
    returned = final <= RETURN_FRAC * max(max_ext, 1e-12)
    on = ext > 0.02
    dur = (float((np.where(on)[0][-1] - np.where(on)[0][0] + 1) * dt)
           if on.any() else 0.0)
    if max_ext >= RUNAWAY_FRAC and not returned:
        lbl = "runaway"
    elif adv >= TRAVEL_MM and not returned:
        lbl = "runaway"
    elif adv >= TRAVEL_MM and returned:
        lbl = "self_limited_propagation"
    elif max_ext >= RUNAWAY_FRAC and returned:
        lbl = "global_synchronous"
    elif max_ext > 0.03:
        lbl = "local_bump"
    else:
        lbl = "extinction"
    return lbl, dict(max_ext=max_ext, adv_mm=adv, returned=bool(returned), dur_ms=dur)
