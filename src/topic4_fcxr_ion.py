"""FCXR-ION -- constitutive Na/K homeostasis on the E1146-informed E/I sheet: pure analysis.

Scope: Phase B0-B2 ONLY (spec §12).  This module holds everything that needs no engine:
the parameter provenance table, the pure ion math (spec §3/§4, rev4 deviation form), the
heterogeneous analytic pre-equilibrium (spec §4.2c), the engine voltage-unit audit
(spec §4.3, plan T1) and the initiation-site direction readout (spec §9 B-real, plan T2).

Naming contract (spec §1): this is a *reduced ion-homeostatic SNN*.  ``Na_i``/``K_o`` carry mM
because the mechanism prior does; they are NOT concentration estimates for E1146, and ``q_ion``
is the effective increment of one *model* spike, not one real action potential.

eta_pump is LOCKED to 0 in B0-B2 (spec §4.3 rev3): only the potassium-mediated pump pathway
(pump -> K recovery -> E_K -> excitability) is under test.  The electrogenic pathway is deferred
to B4 and must not be described as tested.

Design: docs/superpowers/specs/2026-07-27-topic4-fcxr-constitutive-na-k-homeostasis-design.md (rev4)
Plan:   docs/superpowers/plans/2026-07-27-topic4-fcxr-constitutive-na-k-homeostasis-B0-B2.md (rev3)
"""
from __future__ import annotations

import ast
import os
import re

import numpy as np
from scipy.optimize import brentq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ------------------------------------------------------------------ inherited constants (spec §3.1)
RHO = 1.25            # mM/s   maximum pump flux                      (Cressman 2009, normal state)
NA_HALF = 25.0        # mM                                            (Cressman 2009)
S_NA = 3.0            # mM                                            (Cressman 2009)
K_HALF = 5.5          # mM                                            (Cressman 2009)
S_K = 1.0             # mM                                            (Cressman 2009, implicit slope)
NA_I0 = 18.0          # mM     resting intracellular Na
K_I0 = 140.0          # mM     resting intracellular K
K_O0 = 4.0            # mM     resting extracellular K = k_o_inf
BETA = 7.0            # --     intracellular / extracellular volume ratio (tissue constant)
EPS = 1.2             # 1/s    reservoir / bath clearance
K_O_INF = 4.0         # mM
G_GLIA = 66.0         # mM/s
GLIA_HALF = 18.0      # mM
GLIA_SLOPE = 2.5      # mM
D_K = 2.5e-4          # mm^2/s (= Ullah 2009's 2.5e-6 cm^2/s); PHYSICAL value, not renamed
RTF = 26.64           # mV     Nernst factor at 310 K

# ------------------------------------------------------------------ locked working point (spec §7.1)
R0_HZ = 4.15805625    # Hz   arm-C pump-off pooled mean_rate_hz (primary; plan §2)
R0_HEO1_HZ = 3.838    # Hz   HEO1 slow-off reference, consistency cross-check only
V_TH_MV = 18.0        # engine threshold, used only to express dE_K as a % of threshold
ETA_PUMP_B0_B2 = 0.0  # spec §4.3 rev3 -- electrogenic pathway deferred to B4
G_K_ION_REFERENCE = 1.0  # effective reference normalization, NOT a unit-audit conclusion
F_PRIME_CANDIDATES = (0.5, 1.0, 2.0)
F_PRIME_PRIMARY = 1.0
F_PRIME_REFERENCE_ROWS = (0.25, 4.0)   # spec §3.2 reference rows, explicitly out of the candidate set

ALLOWED_SOURCES = ("Cressman 2009", "Ullah 2009", "standard value (310 K)",
                   "this model (B0 lock)", "engine params.py", "arm-C pump-off trajectory")


# =====================================================================================
#  Pure ion math -- spec §3.1 verbatim forms, spec §4 deviation form
# =====================================================================================
def pump_flux(Na_i, K_o, *, rho=RHO, Na_half=NA_HALF, s_Na=S_NA, K_half=K_HALF, s_K=S_K):
    """I_pump = rho / (1+exp((Na_half-Na_i)/s_Na)) / (1+exp((K_half-K_o)/s_K))   [mM/s]."""
    return rho / (1.0 + np.exp((Na_half - np.asarray(Na_i, float)) / s_Na)) \
               / (1.0 + np.exp((K_half - np.asarray(K_o, float)) / s_K))


def glia_uptake(K_o, *, G_glia=G_GLIA, half=GLIA_HALF, slope=GLIA_SLOPE):
    """I_glia = G_glia / (1+exp((half-K_o)/slope))   [mM/s].  SATURATING on purpose (spec §4.2)."""
    return G_glia / (1.0 + np.exp((half - np.asarray(K_o, float)) / slope))


def bath_clearance(K_o, *, eps=EPS, k_o_inf=K_O_INF):
    """I_diff = eps * (K_o - k_o_inf)   [mM/s]."""
    return eps * (np.asarray(K_o, float) - k_o_inf)


def K_i_from_Na_i(Na_i, *, K_i0=K_I0, Na_i0=NA_I0):
    """Cressman's algebraic closure: K_i = K_i0 + (Na_i0 - Na_i)."""
    return K_i0 + (Na_i0 - np.asarray(Na_i, float))


def E_K(K_o, K_i, *, RTF=RTF):
    return RTF * np.log(np.asarray(K_o, float) / np.asarray(K_i, float))


I_PUMP_0 = float(pump_flux(NA_I0, K_O0))
E_K_0 = float(E_K(K_O0, K_I0))
I_GLIA_0 = float(glia_uptake(K_O0))


def background_fluxes():
    """(J_Na_0, J_K_0) -- the background transmembrane fluxes that make the no-spike state an
    EXACT fixed point.  Derived, no degrees of freedom, independent of f' (spec §3.2 rev3)."""
    return 3.0 * I_PUMP_0, 2.0 * BETA * I_PUMP_0


def q_ion_from_fprime(f_prime, *, r0=R0_HZ):
    """q_ion = J_Na_0 * f' / r0   [mM per model spike].

    f' > 0 is the ONLY dial: spike-driven Na influx as a MULTIPLE of the background influx at r0.
    f' > 1 is mathematically legal (spec §3.2); its admissibility is decided by the B1 gates only.
    """
    f_prime = float(f_prime)
    if not np.isfinite(f_prime) or f_prime <= 0.0:
        raise ValueError(f"f' must be finite and > 0 (got {f_prime!r}); silent clipping is forbidden")
    J_Na_0, _ = background_fluxes()
    return J_Na_0 * f_prime / float(r0)


def dNa_dt(Na_i, K_o, spikes_hz, *, q_ion):
    """d[Na]_i/dt = q_ion*S_i - 3*(I_pump_i - I_pump_0)   [mM/s]  (spec §4.1 deviation form)."""
    return q_ion * np.asarray(spikes_hz, float) - 3.0 * (pump_flux(Na_i, K_o) - I_PUMP_0)


def diffusion_term(K_o, *, dx_mm, D=D_K):
    """(D/dx^2) * (sum_nb K_nb - n_nb*K_g) with ZERO-FLUX (reflective) boundaries.

    Edge padding makes the outside neighbour equal the centre, so its contribution vanishes and
    n_nb is automatically the true neighbour count (2 at corners, 3 on edges).  Summing over the
    grid cancels pairwise -> net diffusive flux is exactly 0.
    """
    K = np.asarray(K_o, float)
    P = np.pad(K, 1, mode="edge")
    lap = P[:-2, 1:-1] + P[2:, 1:-1] + P[1:-1, :-2] + P[1:-1, 2:] - 4.0 * K
    return (D / (float(dx_mm) ** 2)) * lap


def dKo_dt(K_o, r_bar, I_pump_bar, n_cells, *, q_ion, dx_mm,
           _broken_no_background=False, _broken_empty_voxel_no_tissue=False,
           _pump_term_only=False):
    """Finite-volume d[K]_o,g/dt on the whole grid   [mM/s]  (spec §4.2 rev4 deviation form).

    Every term is a DEVIATION from rest, so (Na_i, K_o) = (18, 4) with no spikes is a fixed point
    structurally -- not because two constants happen to cancel.

    Empty-voxel contract (spec §4.2): n_g == 0 is a SAMPLING GAP, not a tissue-free region.  Only
    the spike excess is zeroed; the pump term takes the unresolved tissue's resting value I_pump_0
    (deviation 0).  Clearance, glia and diffusion act normally.

    The two ``_broken_*`` switches exist ONLY for the reverse-regression tests that lock the two
    corrections in place (rev2's missing background flux; rev3's empty-voxel pump masking).
    """
    K = np.asarray(K_o, float)
    n = np.asarray(n_cells)
    occupied = n > 0
    r_eff = np.where(occupied, np.nan_to_num(np.asarray(r_bar, float), nan=0.0), 0.0)
    I_eff = np.where(occupied, np.nan_to_num(np.asarray(I_pump_bar, float), nan=I_PUMP_0), I_PUMP_0)

    if _broken_no_background:                      # rev2: absolute pump term, background flux dropped
        pump = -2.0 * BETA * I_eff
    elif _broken_empty_voxel_no_tissue:            # rev3: constant J_K_0 kept, pump zeroed on empty voxels
        _, J_K_0 = background_fluxes()
        pump = J_K_0 - 2.0 * BETA * np.where(occupied, I_eff, 0.0)
    else:
        pump = -2.0 * BETA * (I_eff - I_PUMP_0)

    if _pump_term_only:
        return pump

    source = BETA * q_ion * r_eff
    clear = -bath_clearance(K)
    glia = -(glia_uptake(K) - I_GLIA_0)
    return source + pump + clear + glia + diffusion_term(K, dx_mm=dx_mm)


def ion_increment_terms(*, Na_i, K_o, spike_count, dt_ion_ms, q_ion):
    """Split one discrete Na update into its per-spike and continuous parts (spec §4.2b).

    The spike term is a per-spike CONCENTRATION increment (never multiplied by time); the
    continuous flux term carries dt_ion_ms * 1e-3.  Guards the ms<->s 1000x error.
    """
    return dict(spike=q_ion * float(spike_count),
                continuous=(float(dt_ion_ms) * 1e-3) * (-3.0 * (float(pump_flux(Na_i, K_o)) - I_PUMP_0)))


def interictal_steady_state(q_ion, r_hz):
    """Homogeneous (Na*, K_o*) at a sustained per-cell rate.  Analytic table only -- the real runs
    use the HETEROGENEOUS initializer (spec §4.2c).

    dNa/dt = 0 gives I_pump - I_pump_0 = q_ion*r/3 exactly, so the K balance collapses to a scalar
    root in K_o alone:  (beta*q_ion*r)/3 = eps*(K_o-4) + [I_glia(K_o) - I_glia(4)].
    """
    rhs = BETA * q_ion * float(r_hz) / 3.0

    def f(K):
        return EPS * (K - K_O_INF) + (float(glia_uptake(K)) - I_GLIA_0) - rhs

    K_star = brentq(f, K_O_INF - 1e-12, 500.0, xtol=1e-14, rtol=8.9e-16)
    Na_star = _invert_pump_for_Na(I_PUMP_0 + q_ion * float(r_hz) / 3.0, K_star)
    return Na_star, K_star


def _invert_pump_for_Na(target_I_pump, K_o):
    """Solve I_pump(Na, K_o) = target for Na.  Raises when the target exceeds what the Na gate can
    deliver at this K_o (the cell would have no steady state) -- loud failure, never clipped."""
    sig_K = 1.0 / (1.0 + np.exp((K_HALF - np.asarray(K_o, float)) / S_K))
    sig_needed = np.asarray(target_I_pump, float) / (RHO * sig_K)
    if np.any(sig_needed >= 1.0) or np.any(sig_needed <= 0.0):
        n_bad = int(np.sum((sig_needed >= 1.0) | (sig_needed <= 0.0)))
        raise ValueError(f"no bounded Na steady state for {n_bad} cell(s): required pump activation "
                         f"outside (0,1); rate or q_ion too large for rho={RHO}")
    return NA_HALF - S_NA * np.log(1.0 / sig_needed - 1.0)


# =====================================================================================
#  Heterogeneous analytic pre-equilibrium -- spec §4.2c
# =====================================================================================
def _voxel_aggregates(rate_E, rate_I, voxel_E, voxel_I, n_grid):
    r_all = np.concatenate([np.asarray(rate_E, float), np.asarray(rate_I, float)])
    v_all = np.concatenate([np.asarray(voxel_E, int), np.asarray(voxel_I, int)])
    nv = n_grid * n_grid
    n_cells = np.bincount(v_all, minlength=nv).astype(float)
    r_sum = np.bincount(v_all, weights=r_all, minlength=nv)
    with np.errstate(invalid="ignore", divide="ignore"):
        r_bar = np.where(n_cells > 0, r_sum / np.where(n_cells > 0, n_cells, 1.0), 0.0)
    return r_all, v_all, n_cells.reshape(n_grid, n_grid), r_bar.reshape(n_grid, n_grid)


def _residual_report(r_all, v_all, Na_star, K_grid, n_cells, r_bar, q_ion, dx_mm, n_grid):
    K_of_cell = K_grid.ravel()[v_all]
    res_na = np.abs(dNa_dt(Na_star, K_of_cell, r_all, q_ion=q_ion))
    Ip_cell = pump_flux(Na_star, K_of_cell)
    nv = n_grid * n_grid
    n_flat = n_cells.ravel()
    Ip_sum = np.bincount(v_all, weights=Ip_cell, minlength=nv)
    with np.errstate(invalid="ignore", divide="ignore"):
        Ip_bar = np.where(n_flat > 0, Ip_sum / np.where(n_flat > 0, n_flat, 1.0), I_PUMP_0)
    res_ko = np.abs(dKo_dt(K_grid, r_bar, Ip_bar.reshape(n_grid, n_grid), n_cells,
                           q_ion=q_ion, dx_mm=dx_mm))
    return dict(
        max_abs_dNa_dt=float(res_na.max()), q99_abs_dNa_dt=float(np.quantile(res_na, 0.99)),
        q95_abs_dNa_dt=float(np.quantile(res_na, 0.95)),
        max_abs_dKo_dt=float(res_ko.max()), q99_abs_dKo_dt=float(np.quantile(res_ko, 0.99)),
        q95_abs_dKo_dt=float(np.quantile(res_ko, 0.95)),
        I_pump_bar=Ip_bar.reshape(n_grid, n_grid),
    )


def heterogeneous_steady_state(rate_E, rate_I, voxel_E, voxel_I, *, n_grid, q_ion, dx_mm,
                               tol=1e-6, max_iter=50):
    """Per-cell Na* and per-voxel K_o* at the frozen per-cell baseline rate field (spec §4.2c).

    A single global-rate scalar steady state is NOT an acceptable substitute: it leaves a slow
    spatial re-arrangement that 11 s (0.2 tau_Na) cannot expose.  Empty voxels take the unresolved
    tissue's resting value and solve to K_o = 4 under zero flux -- the deviation-form equation is
    well posed there, so no neighbour interpolation is needed (count reported as 0).
    """
    r_all, v_all, n_cells, r_bar = _voxel_aggregates(rate_E, rate_I, voxel_E, voxel_I, n_grid)
    K = np.full((n_grid, n_grid), K_O0, float)
    rhs = BETA * q_ion * r_bar / 3.0                     # exact once Na sits at its own root
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        diff = diffusion_term(K, dx_mm=dx_mm)
        K_new = np.empty_like(K)
        for idx in np.ndindex(K.shape):
            target = float(rhs[idx]) + float(diff[idx])

            def f(x, _t=target):
                return EPS * (x - K_O_INF) + (float(glia_uptake(x)) - I_GLIA_0) - _t

            K_new[idx] = brentq(f, 0.05, 500.0, xtol=1e-14, rtol=8.9e-16)
        dK = float(np.max(np.abs(K_new - K)))
        K = K_new
        if dK < tol:
            break
    Na_star = _invert_pump_for_Na(I_PUMP_0 + q_ion * r_all / 3.0, K.ravel()[v_all])
    rep = _residual_report(r_all, v_all, Na_star, K, n_cells, r_bar, q_ion, dx_mm, n_grid)
    rep.update(Na_star=Na_star, K_o_star=K, n_iter=int(n_iter), converged=bool(dK < tol),
               n_empty_voxels=int(np.sum(n_cells == 0)), n_voxels_interpolated=0,
               n_cells_per_voxel=n_cells, r_bar=r_bar, rate_all=r_all, voxel_all=v_all)
    return rep


def scalar_steady_state_init(rate_E, rate_I, voxel_E, voxel_I, *, n_grid, q_ion, dx_mm):
    """REVERSE REGRESSION ONLY: initialise every cell/voxel from the single global-rate scalar
    steady state, then report the residual it leaves (spec §4.2c says this is not acceptable)."""
    r_all, v_all, n_cells, r_bar = _voxel_aggregates(rate_E, rate_I, voxel_E, voxel_I, n_grid)
    Na_s, K_s = interictal_steady_state(q_ion, float(r_all.mean()))
    Na_star = np.full(r_all.shape, Na_s)
    K = np.full((n_grid, n_grid), K_s, float)
    rep = _residual_report(r_all, v_all, Na_star, K, n_cells, r_bar, q_ion, dx_mm, n_grid)
    rep.update(Na_star=Na_star, K_o_star=K, n_empty_voxels=int(np.sum(n_cells == 0)))
    return rep


def k_budget_closure(rate_E, rate_I, voxel_E, voxel_I, *, n_grid, q_ion, dx_mm,
                     dt_ion_ms, n_steps, K_init=None):
    """Gate H item 2: integrate the K field and check that
    sum(source) - sum(pump recovery + clearance + glia) - sum(diffusion) equals the change in
    total extracellular K.  Diffusion must contribute exactly zero under zero flux."""
    r_all, v_all, n_cells, r_bar = _voxel_aggregates(rate_E, rate_I, voxel_E, voxel_I, n_grid)
    K = np.full((n_grid, n_grid), K_O0, float) if K_init is None else np.array(K_init, float)
    Na = _invert_pump_for_Na(I_PUMP_0 + q_ion * r_all / 3.0, K.ravel()[v_all])
    nv = n_grid * n_grid
    n_flat = n_cells.ravel()
    dt_s = float(dt_ion_ms) * 1e-3
    total0 = float(K.sum())
    acc = dict(source=0.0, pump=0.0, clearance=0.0, glia=0.0, diffusion=0.0)
    for _ in range(int(n_steps)):
        Ip_cell = pump_flux(Na, K.ravel()[v_all])
        Ip_sum = np.bincount(v_all, weights=Ip_cell, minlength=nv)
        with np.errstate(invalid="ignore", divide="ignore"):
            Ip_bar = np.where(n_flat > 0, Ip_sum / np.where(n_flat > 0, n_flat, 1.0), I_PUMP_0)
        Ip_bar = Ip_bar.reshape(n_grid, n_grid)
        occupied = n_cells > 0
        src = BETA * q_ion * np.where(occupied, r_bar, 0.0)
        pmp = -2.0 * BETA * (np.where(occupied, Ip_bar, I_PUMP_0) - I_PUMP_0)
        clr = -bath_clearance(K)
        gli = -(glia_uptake(K) - I_GLIA_0)
        dif = diffusion_term(K, dx_mm=dx_mm)
        acc["source"] += dt_s * float(src.sum())
        acc["pump"] += dt_s * float(pmp.sum())
        acc["clearance"] += dt_s * float(clr.sum())
        acc["glia"] += dt_s * float(gli.sum())
        acc["diffusion"] += dt_s * float(dif.sum())
        K = K + dt_s * (src + pmp + clr + gli + dif)
        Na = Na + dt_s * (q_ion * r_all - 3.0 * (Ip_cell - I_PUMP_0))
    delta = float(K.sum()) - total0
    budget = sum(acc.values())
    return dict(delta_total_K=delta, budget=budget, terms=acc,
                diffusion_net_flux=acc["diffusion"],
                relative_error=abs(budget - delta) / max(abs(delta), 1e-12),
                K_final=K, n_negative=int(np.sum(K <= 0.0)))


# =====================================================================================
#  Provenance table + analytic feasibility -- plan §6
# =====================================================================================
def _row(value, unit, equation, source, kind):
    return dict(value=value, unit=unit, equation=equation, source=source, kind=kind)


_J_NA_0, _J_K_0 = background_fluxes()

PARAM_TABLE = {
    "rho": _row(RHO, "mM/s", "I_pump", "Cressman 2009", "inherited"),
    "Na_half": _row(NA_HALF, "mM", "I_pump", "Cressman 2009", "inherited"),
    "s_Na": _row(S_NA, "mM", "I_pump", "Cressman 2009", "inherited"),
    "K_half": _row(K_HALF, "mM", "I_pump", "Cressman 2009", "inherited"),
    "s_K": _row(S_K, "mM", "I_pump", "Cressman 2009", "inherited"),
    "Na_i0": _row(NA_I0, "mM", "rest", "Cressman 2009", "inherited"),
    "K_i0": _row(K_I0, "mM", "rest", "Cressman 2009", "inherited"),
    "K_o0": _row(K_O0, "mM", "rest = k_o_inf", "Cressman 2009", "inherited"),
    "beta": _row(BETA, "-", "K_o equation", "Cressman 2009", "inherited"),
    "eps": _row(EPS, "1/s", "I_diff = eps*(K_o - k_o_inf)", "Cressman 2009", "inherited"),
    "k_o_inf": _row(K_O_INF, "mM", "I_diff", "Cressman 2009", "inherited"),
    "G_glia": _row(G_GLIA, "mM/s", "I_glia", "Cressman 2009", "inherited"),
    "glia_half": _row(GLIA_HALF, "mM", "I_glia", "Cressman 2009", "inherited"),
    "glia_slope": _row(GLIA_SLOPE, "mM", "I_glia", "Cressman 2009", "inherited"),
    "D_K": _row(D_K, "mm^2/s", "discrete Laplacian, Ullah Eq.(5)", "Ullah 2009", "inherited"),
    "RT_over_F": _row(RTF, "mV", "E_K", "standard value (310 K)", "inherited"),
    "r0": _row(R0_HZ, "Hz", "working point", "arm-C pump-off trajectory", "inherited"),
    "I_pump_0": _row(I_PUMP_0, "mM/s", "I_pump(18, 4)", "Cressman 2009", "derived"),
    "E_K_0": _row(E_K_0, "mV", "E_K(4, 140)", "Cressman 2009", "derived"),
    "J_Na_0": _row(_J_NA_0, "mM/s", "3*I_pump_0", "Cressman 2009", "derived"),
    "J_K_0": _row(_J_K_0, "mM/s", "2*beta*I_pump_0", "Cressman 2009", "derived"),
    "q_ion": _row(q_ion_from_fprime(F_PRIME_PRIMARY), "mM/spike", "J_Na_0*f'/r0",
                  "this model (B0 lock)", "derived"),
    "tau_Na": _row(None, "s", "1/(3 dI_pump/dNa|rest)", "Cressman 2009", "derived"),
    "tau_Ko": _row(None, "s", "1/(eps + dI_glia/dK + 2 beta dI_pump/dK)|rest",
                   "Cressman 2009", "derived"),
    "q_K_per_spike": _row(None, "mM/spike", "beta*q_ion", "this model (B0 lock)", "effective"),
    "f_prime": _row(F_PRIME_PRIMARY, "-", "spike Na influx / background influx at r0",
                    "this model (B0 lock)", "effective"),
    "g_K_ion": _row(G_K_ION_REFERENCE, "engine drive unit per mV", "drive += g_K_ion*(E_K - E_K_0)",
                    "this model (B0 lock)", "effective"),
    "eta_pump": _row(ETA_PUMP_B0_B2, "engine drive unit per (mM/s)",
                     "drive -= eta_pump*(I_pump - I_pump_0)", "this model (B0 lock)", "effective"),
    "n_grid": _row(32, "-", "K_o finite-volume grid on L=20 mm", "this model (B0 lock)", "effective"),
    "dt_ion": _row(0.5, "ms", "ion sub-step", "this model (B0 lock)", "effective"),
}


def relaxation_times():
    """Linearised relaxation of the two variables at rest (spec §3.2b).  They differ by ~83x, which
    is why one common judging window must not be used for both."""
    h = 1e-6
    dIp_dNa = (pump_flux(NA_I0 + h, K_O0) - pump_flux(NA_I0 - h, K_O0)) / (2 * h)
    dIp_dK = (pump_flux(NA_I0, K_O0 + h) - pump_flux(NA_I0, K_O0 - h)) / (2 * h)
    dIg_dK = (glia_uptake(K_O0 + h) - glia_uptake(K_O0 - h)) / (2 * h)
    tau_Na = 1.0 / (3.0 * float(dIp_dNa))
    tau_Ko = 1.0 / (EPS + float(dIg_dK) + 2.0 * BETA * float(dIp_dK))
    return tau_Na, tau_Ko


def analytic_feasibility(*, r0=R0_HZ, _break=None):
    """Recompute spec §3.2 from the locked constants and check the hard feasibility gates.

    Gates: J_Na_0 > 0 (true by construction), the no-spike state is an EXACT fixed point including
    on an EMPTY voxel, and all concentrations stay positive.
    """
    rows = []
    for fp in sorted(set(F_PRIME_CANDIDATES) | set(F_PRIME_REFERENCE_ROWS)):
        q = q_ion_from_fprime(fp, r0=r0)
        Na, Ko = interictal_steady_state(q, r0)
        dE = float(E_K(Ko, K_i_from_Na_i(Na))) - E_K_0
        row = dict(f_prime=fp, q_ion=q, Na_star=Na, K_o_star=Ko, dE_K_interictal_mV=dE,
                   dE_K_interictal_pct_Vth=100.0 * dE / V_TH_MV,
                   in_candidate_set=fp in F_PRIME_CANDIDATES,
                   is_primary=(fp == F_PRIME_PRIMARY))
        for r_hz, tag in ((20.0, "20hz"), (50.0, "50hz")):
            Na_h, Ko_h = interictal_steady_state(q, r_hz)
            dE_h = float(E_K(Ko_h, K_i_from_Na_i(Na_h))) - E_K_0
            row[f"K_o_star_{tag}"] = Ko_h
            row[f"dE_K_{tag}_mV"] = dE_h
            row[f"dE_K_{tag}_pct_Vth"] = 100.0 * dE_h / V_TH_MV
        rows.append(row)

    K = np.full((3, 3), K_O0)
    n_full = np.full((3, 3), 40)
    n_empty = np.zeros((3, 3), int)
    Ip = np.full((3, 3), I_PUMP_0)
    kw = {}
    if _break == "no_background":
        kw["_broken_no_background"] = True
    elif _break == "empty_voxel_no_tissue":
        kw["_broken_empty_voxel_no_tissue"] = True
    rest_ok = (abs(float(dNa_dt(NA_I0, K_O0, 0.0, q_ion=rows[0]["q_ion"]))) < 1e-15
               and float(np.max(np.abs(dKo_dt(K, np.zeros((3, 3)), Ip, n_full,
                                              q_ion=rows[0]["q_ion"], dx_mm=0.625, **kw)))) < 1e-15)
    empty_ok = float(np.max(np.abs(dKo_dt(K, np.zeros((3, 3)), np.full((3, 3), np.nan), n_empty,
                                          q_ion=rows[0]["q_ion"], dx_mm=0.625, **kw)))) < 1e-15
    conc_ok = all(r["Na_star"] > 0 and r["K_o_star"] > 0 and
                  float(K_i_from_Na_i(r["Na_star"])) > 0 for r in rows)
    J_Na_0, J_K_0 = background_fluxes()
    tau_Na, tau_Ko = relaxation_times()
    gates = dict(J_Na_0_positive=bool(J_Na_0 > 0), rest_fixed_point=bool(rest_ok),
                 empty_voxel_fixed_point=bool(empty_ok), all_concentrations_positive=bool(conc_ok))
    return dict(rows=rows, gates=gates, status="PASS" if all(gates.values()) else "FAIL",
                I_pump_0=I_PUMP_0, E_K_0=E_K_0, J_Na_0=J_Na_0, J_K_0=J_K_0,
                tau_Na_s=tau_Na, tau_Ko_s=tau_Ko, tau_ratio=tau_Na / tau_Ko,
                r0_hz=float(r0), r0_heo1_hz=R0_HEO1_HZ,
                eta_pump=ETA_PUMP_B0_B2, g_K_ion=G_K_ION_REFERENCE,
                candidate_set=list(F_PRIME_CANDIDATES), primary=F_PRIME_PRIMARY)


# =====================================================================================
#  T1 -- engine voltage-unit audit (spec §4.3, plan §3).  DIMENSION only.
# =====================================================================================
def v_inf(drive, g_rel, g_rev):
    """The engine's conductance membrane fixed point (kick_probe.py): (drive+g_rev)/(1+g_rel)."""
    return (np.asarray(drive, float) + np.asarray(g_rev, float)) / (1.0 + np.asarray(g_rel, float))


def ion_membrane_current(*, K_o, Na_i, g_K_ion=G_K_ION_REFERENCE, eta_pump=ETA_PUMP_B0_B2):
    """The additive membrane term the ion layer contributes, in engine drive units (spec §4.3).

    Both parts are CURRENTS and apply to E and I alike -- writing the potassium part as a
    conductance would be silently discarded for I cells by the engine (spec §5).
    """
    if float(eta_pump) != 0.0:
        raise ValueError("eta_pump is locked to 0 in B0-B2 (spec §4.3 rev3): only the "
                         "potassium-mediated pump pathway is under test; the electrogenic "
                         "pathway is deferred to B4")
    dE = E_K(K_o, K_i_from_Na_i(Na_i)) - E_K_0
    return g_K_ion * dE - float(eta_pump) * (pump_flux(Na_i, K_o) - I_PUMP_0)


def _params_unit_comments(path=None):
    """Read the unit annotation of each Params field straight out of the blessed params.py."""
    path = path or os.path.join(ROOT, "src", "snn_engine", "params.py")
    pat = re.compile(r"^\s*(\w+)\s*:\s*float\s*=\s*([-\deE.+]+)\s*#\s*(\S+)")
    out = {}
    with open(path) as fh:
        for line in fh:
            m = pat.match(line)
            if m:
                out[m.group(1)] = m.group(3)
    return out


def _params_values(path=None):
    path = path or os.path.join(ROOT, "src", "snn_engine", "params.py")
    pat = re.compile(r"^\s*(\w+)\s*:\s*float\s*=\s*([-\deE.+]+)")
    out = {}
    with open(path) as fh:
        for line in fh:
            m = pat.match(line)
            if m:
                out[m.group(1)] = float(m.group(2))
    return out


def fcxr_arm_c_config_literals(path=None):
    """Extract the accepted arm-C FCXR config literals from run_topic4_mz_fcxr.py::_fc_cfg by
    parsing the source (no import, no side effects)."""
    path = path or os.path.join(ROOT, "scripts", "run_topic4_mz_fcxr.py")
    with open(path) as fh:
        tree = ast.parse(fh.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_fc_cfg":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call) and getattr(sub.func, "id", None) == "dict":
                    return {kw.arg: ast.literal_eval(kw.value) for kw in sub.keywords
                            if isinstance(kw.value, ast.Constant)}
    raise RuntimeError(f"could not locate _fc_cfg's dict(...) literal in {path}")


def audit_voltage_units():
    """Close spec §4.3's open item: is the engine's voltage coordinate mV, so that injecting a
    mV-dimensioned quantity into `drive` is self-consistent?

    This fixes the DIMENSION only.  It does NOT and cannot fix the value of g_K_ion -- that is an
    explicit effective reference normalization (spec §4.3 rev3) to be calibrated in B3.
    """
    units = _params_unit_comments()
    vals = _params_values()
    cfg = fcxr_arm_c_config_literals()

    d, gr, gv, delta = 12.0, 3.0, 25.0, 1.7
    linear_ok = abs(float(v_inf(d + delta, gr, gv) - v_inf(d, gr, gv)) - delta / (1.0 + gr)) < 1e-12

    chain = [
        dict(step="V zero point is the leak reversal",
             evidence=f"params.V_L = {vals['V_L']} ({units.get('V_L')})", ok=vals["V_L"] == 0.0),
        dict(step="threshold and reset are annotated mV",
             evidence=f"V_th = {vals['V_th']} ({units.get('V_th')}), "
                      f"V_reset = {vals['V_reset']} ({units.get('V_reset')})",
             ok=units.get("V_th") == "mV" and units.get("V_reset") == "mV"),
        dict(step="synaptic weights are annotated mV (drive inherits the same unit)",
             evidence=f"J_ext_E = {vals['J_ext_E']} ({units.get('J_ext_E')}), "
                      f"w_EE = {vals['w_EE']} ({units.get('w_EE')})",
             ok=units.get("J_ext_E") == "mV"),
        dict(step="drive and g_rev share units in V_inf = (drive+g_rev)/(1+g_rel)",
             evidence="g_rel is dimensionless (sum g_k/g_leak); g_rev is sum g_k/g_leak*E_k, "
                      "so g_rev carries the reversal-potential unit and drive must match it",
             ok=linear_ok),
        dict(step="force-matching anchor and AMPA reversal live in the same coordinate",
             evidence=f"v_match = {cfg['v_match']} == V_th = {vals['V_th']}; E_E = {cfg['E_E']} "
                      f"(rest -58 mV -> 0 mV AMPA reversal)",
             ok=cfg["v_match"] == vals["V_th"]),
        dict(step="adding a mV-dimensioned term to drive moves V_inf by delta/(1+g_rel)",
             evidence=f"checked numerically: delta={delta}, g_rel={gr}", ok=linear_ok),
        dict(step="the ion layer leaves the substrate's existing reversals untouched",
             evidence=f"arm-C config e_gaba = {cfg['e_gaba']}, e_k = {cfg['e_k']} "
                      f"(both at the leak reversal); dE_K enters as an ADDITIVE CURRENT, so the "
                      f"existing GABA / sAHP conductance paths are not re-pointed",
             ok=cfg["e_gaba"] == 0.0 and cfg["e_k"] == 0.0),
    ]
    ok = all(c["ok"] for c in chain)
    return dict(
        status="CONFIRMED" if ok else "NOT_CONFIRMED",
        chain=chain,
        V_th_mV=vals["V_th"], V_reset_mV=vals["V_reset"], V_L_mV=vals["V_L"],
        params_unit_comments=units,
        substrate_e_gaba=cfg["e_gaba"], substrate_e_k=cfg["e_k"],
        substrate_v_match=cfg["v_match"], substrate_E_E=cfg["E_E"],
        substrate_membrane_mode=cfg["membrane_mode"],
        ion_layer_modifies_e_gaba_or_e_k=False,
        drive_and_g_rev_share_units=bool(linear_ok),
        delta_E_K_injection_unit="mV",
        engine_voltage_unit="mV",
        g_K_ion_is_a_unit_audit_conclusion=False,
        g_K_ion_reference_value=G_K_ION_REFERENCE,
        g_K_ion_kind="effective reference normalization",
        g_K_ion_calibrated_in="B3 (not authorised in this sprint)",
        eta_pump_locked_to=ETA_PUMP_B0_B2,
        caveat=("Confirms the DIMENSION only. g_K_ion = 1 is a declared normalization "
                "('resting K conductance is of the same order as the leak'), not a result of "
                "this audit; every B0-B2 conclusion carries that normalization as a premise."),
    )


# =====================================================================================
#  T2 -- initiation-site direction readout (spec §9 B-real, plan §4)
# =====================================================================================
def _event_slice(spk_bool, ev, dt, t0_ms):
    s = int(round((ev["t_on"] - t0_ms) / dt))
    e = int(round((ev["t_off"] - t0_ms) / dt)) + 1
    s = max(s, 0)
    e = min(e, spk_bool.shape[0])
    return (s, e) if e > s else None


def two_sided_forward_fraction(spk_bool, pos_E, core_A_xy, core_B_xy, events, *, dt,
                               core_r, t0_ms=0.0):
    """The rev1 readout, kept ONLY as the contrast case.

    Faithful re-statement of run_topic4_mz_fcxr_pump._forward_fraction: an event is scored only
    when BOTH cores participate, otherwise it is skipped (never counted as forward).  On the
    accepted arm-C pump-off trajectory that left 2 of 22 events scoreable, which is why spec §9
    replaced it.
    """
    A = np.linalg.norm(np.asarray(pos_E, float) - np.asarray(core_A_xy, float), axis=1) <= core_r
    B = np.linalg.norm(np.asarray(pos_E, float) - np.asarray(core_B_xy, float), axis=1) <= core_r
    fwd = tot = 0
    for ev in events:
        sl = _event_slice(spk_bool, ev, dt, t0_ms)
        if sl is None:
            continue
        seg = spk_bool[sl[0]:sl[1]]
        first = np.argmax(seg, axis=0).astype(float)
        fired = seg.any(axis=0)
        a, b = fired & A, fired & B
        if not (a.any() and b.any()):
            continue
        tot += 1
        fwd += int(first[a].mean() < first[b].mean())
    return dict(forward_event_fraction=(float(fwd) / tot if tot else float("nan")),
                n_direction_events=int(tot))


def initiation_site_readout(spk_bool, pos_E, core_A_xy, core_B_xy, events, *, dt, core_r,
                            frac_earliest=0.05, ambiguous_frac=0.20, t0_ms=0.0):
    """Per-event initiation site (spec §9 B-real replacement readout).

    For each event take the earliest-firing `frac_earliest` of the participating cells, take their
    centroid, and attribute it to core_A or core_B by distance.  If the two distances differ by
    less than `ambiguous_frac` of the core radius the event is `ambiguous` and is NOT scored.

    Unlike the rev1 readout this scores EVERY event -- it does not require both cores to
    participate -- which is the whole point of the replacement.

    Naming (spec §9 rev3): core_A / core_B, never source / sink.  E1146's directionality is not
    established, and source/sink would smuggle in an unevidenced directional claim.
    """
    pos = np.asarray(pos_E, float)
    A = np.asarray(core_A_xy, float)
    B = np.asarray(core_B_xy, float)
    per = []
    for k, ev in enumerate(events):
        sl = _event_slice(spk_bool, ev, dt, t0_ms)
        if sl is None:
            per.append(dict(index=k, core="unscoreable", reason="event window outside the trace"))
            continue
        seg = spk_bool[sl[0]:sl[1]]
        fired = seg.any(axis=0)
        n_part = int(fired.sum())
        if n_part == 0:
            per.append(dict(index=k, core="unscoreable", reason="no participating cell"))
            continue
        idx = np.where(fired)[0]
        first = np.argmax(seg[:, idx], axis=0)
        n_take = max(1, int(np.ceil(frac_earliest * n_part)))
        earliest = idx[np.argsort(first, kind="stable")[:n_take]]
        cen = pos[earliest].mean(axis=0)
        dA = float(np.linalg.norm(cen - A))
        dB = float(np.linalg.norm(cen - B))
        core = ("ambiguous" if abs(dA - dB) < ambiguous_frac * core_r
                else ("A" if dA < dB else "B"))
        per.append(dict(index=k, core=core, n_participating=n_part, n_earliest=n_take,
                        centroid=[float(cen[0]), float(cen[1])], d_core_A=dA, d_core_B=dB))
    nA = sum(1 for p in per if p["core"] == "A")
    nB = sum(1 for p in per if p["core"] == "B")
    namb = sum(1 for p in per if p["core"] == "ambiguous")
    n_scoreable = nA + nB
    denom = max(n_scoreable, 1)
    return dict(n_events=len(events), n_scoreable=int(n_scoreable),
                n_A=int(nA), n_B=int(nB), n_ambiguous=int(namb),
                frac_A=nA / denom if n_scoreable else 0.0,
                frac_B=nB / denom if n_scoreable else 0.0,
                frac_ambiguous=(namb / max(nA + nB + namb, 1)),
                frac_earliest=frac_earliest, ambiguous_frac=ambiguous_frac, core_r=core_r,
                per_event=per)


# =====================================================================================
#  Gate H -- homeostasis and numerical contract (spec §9, plan §8)
# =====================================================================================
# Pre-locked BEFORE any measurement.  Rationale for the residual bound:
#   * over the full 11 s window 1e-6 mM/s integrates to 1.1e-5 mM, four orders of magnitude
#     below the interictal excursion the layer is supposed to produce (dK_o ~ 0.11 mM,
#     dNa ~ 2.07 mM at f'=1), so it cannot masquerade as dynamics;
#   * it also sits four orders below what the single-global-rate scalar initializer leaves
#     (q99 ~ 1e-2 mM/s), so the gate genuinely discriminates the two initializers.
GATE_H_RESIDUAL_MAX_MM_S = 1e-6
GATE_H_BUDGET_REL_TOL = 1e-10
GATE_H_DIFFUSION_ABS_TOL = 1e-11
GATE_H_FIXED_POINT_TOL = 1e-12
GATE_H_RECOVERY_REL_TOL = 0.01     # local K perturbation must return within 1% in 3 s
GATE_H_PUMP_MIN_FRAC = 0.5         # baseline pump must be constitutive, not silently off

_GATE_H_ORDER = [
    ("resting_fixed_point", "FAIL_EQUILIBRIUM"),
    ("empty_voxel_fixed_point", "FAIL_EMPTY_VOXEL"),
    ("heterogeneous_init_residual", "FAIL_INIT_RESIDUAL"),
    ("k_budget_closure", "FAIL_BUDGET"),
    ("zero_flux_boundary", "FAIL_BUDGET"),
    ("pump_stoichiometry", "FAIL_STOICHIOMETRY"),
    ("ions_off_byte_parity", "FAIL_PARITY"),
    ("baseline_pump_nonzero", "FAIL_NUMERICAL"),
    ("local_perturbation_recovers", "FAIL_NUMERICAL"),
    ("grid_consistency", "FAIL_NUMERICAL"),
    ("dt_ion_convergence", "FAIL_NUMERICAL"),
    ("checkpoint_restart_identity", "FAIL_NUMERICAL"),
    ("no_negative_or_bound_collision", "FAIL_NUMERICAL"),
    ("blessed_engine_unmodified", "FAIL_PARITY"),
]


def adjudicate_gate_H(checks):
    """Map the measured Gate H items to the pre-registered status enum (plan §8).

    Population-mean stationarity NEVER counts: the initialization residual item must carry
    per-cell / per-voxel q95, q99 and max, because on a heterogeneous substrate the mean can be
    flat while a slow spatial re-arrangement (tau_Na = 54.4 s) runs underneath it unseen in 11 s.
    """
    missing = [name for name, _ in _GATE_H_ORDER if name not in checks]
    if missing:
        return dict(status="UNRESOLVED", reason=f"items not measured: {missing}",
                    checks=checks, missing=missing)
    res = checks["heterogeneous_init_residual"]
    for stat in ("q95", "q99", "max"):
        for var in ("dNa_dt", "dKo_dt"):
            if f"{stat}_abs_{var}" not in res:
                return dict(status="UNRESOLVED",
                            reason=f"init residual is missing {stat}_abs_{var}; a population mean "
                                   f"is not an acceptable substitute (spec §4.2c)",
                            checks=checks)
    failures = [(name, code) for name, code in _GATE_H_ORDER if not checks[name].get("ok")]
    if not failures:
        return dict(status="PASS", checks=checks, failures=[],
                    thresholds=dict(residual_max_mM_s=GATE_H_RESIDUAL_MAX_MM_S,
                                    budget_rel_tol=GATE_H_BUDGET_REL_TOL,
                                    diffusion_abs_tol=GATE_H_DIFFUSION_ABS_TOL,
                                    fixed_point_tol=GATE_H_FIXED_POINT_TOL))
    return dict(status=failures[0][1], checks=checks,
                failures=[dict(item=n, code=c, detail=checks[n]) for n, c in failures])


# =====================================================================================
#  T7 -- f' selection on the small network (plan §9.3, rev3 five gates)
# =====================================================================================
# Every bound is pre-registered.  The rev2 gates were all rewritten because each of them failed
# by construction once the two relaxation times were computed: tau_Ko = 0.655 s and
# tau_Na = 54.42 s differ by 83x, so no single 20 s window can judge both.
F_GATES = dict(
    measurable_sigma_mult=5.0,
    measurable_abs_floor_mM=0.15,     # = 1 mV of E_K = 5.6% of V_th: "the membrane must see it"
    safe_ceiling_mM=0.90,             # = 5.4 mV = 30% of V_th from ONE interictal event: too much
    k_recovery_window_s=3.0,          # 4.6 tau_Ko
    k_recovery_sigma=1.0,
    na_recovery_window_s=20.0,        # 0.37 tau_Na -> analytic decay 30.8%
    na_decay_band=(0.154, 0.462),     # [0.5x, 1.5x] of the analytic 30.8%
    na_monotone_sigma_slack=1.0,
    integration_min_ratio=2.38,       # 0.8 x the 2.97 pure-linear superposition at 200 ms spacing
    integration_linear_prediction=2.97,
)


def evaluate_f_prime_gates(m):
    """Five gates on one f' candidate.  `m` carries the measured quantities; all five must pass."""
    g = F_GATES
    floor = max(g["measurable_sigma_mult"] * float(m["sigma_rest_K_mM"]), g["measurable_abs_floor_mM"])
    ratio = float(m["integration_ratio_5th_over_1st"])
    gates = {
        "measurable": dict(ok=bool(float(m["dK_peak_single_mM"]) >= floor),
                           value=float(m["dK_peak_single_mM"]), threshold=floor,
                           sigma_rest=float(m["sigma_rest_K_mM"]),
                           rule="single-event peak dK_o >= max(5*sigma_rest, 0.15 mM)"),
        "safe": dict(ok=bool(float(m["dK_peak_single_mM"]) <= g["safe_ceiling_mM"]),
                     value=float(m["dK_peak_single_mM"]), threshold=g["safe_ceiling_mM"],
                     rule="a SINGLE interictal event must not push the membrane 30% of V_th"),
        "recovery_K": dict(ok=bool(m["k_returns_within_1sigma_3s"]),
                           value=float(m["k_residual_after_3s_in_sigma"]),
                           threshold=g["k_recovery_sigma"],
                           rule="back inside 1 sigma of this voxel's resting mean within 3 s"),
        "recovery_Na": dict(
            ok=bool(m["na_excess_monotone_nonincreasing"]
                    and g["na_decay_band"][0] <= float(m["na_excess_decay_frac_20s"])
                    <= g["na_decay_band"][1]),
            value=float(m["na_excess_decay_frac_20s"]), band=list(g["na_decay_band"]),
            monotone=bool(m["na_excess_monotone_nonincreasing"]),
            rule="event-INDUCED excess (vs each cell's own pre-kick baseline) decays 15.4-46.2% "
                 "in 20 s and never rises after the peak"),
        "integration": dict(ok=bool(ratio >= g["integration_min_ratio"]),
                            value=ratio, threshold=g["integration_min_ratio"],
                            linear_prediction=g["integration_linear_prediction"],
                            ratio_vs_linear=ratio / g["integration_linear_prediction"],
                            supralinear=bool(ratio > g["integration_linear_prediction"]),
                            rule="5th/1st peak dK_o >= 2.38; only a ratio ABOVE 2.97 is "
                                 "supralinear -- passing 2.38 alone is NOT evidence of it"),
    }
    return dict(admissible=all(v["ok"] for v in gates.values()), gates=gates, measured=m)


def select_f_prime(rows):
    """Tie-break: among admissible candidates take the one CLOSEST to f' = 1.0.

    rev2's 'take the largest' would systematically bias toward stronger potassium positive
    feedback -- exactly the direction that makes B3/B4 run away -- i.e. it would pre-bias the
    mechanism conclusion through a tie-break rule.  1.0 is primary; 0.5 and 2.0 bracket it.
    """
    adm = [r for r in rows if r["admissible"]]
    if not adm:
        return dict(status="NO_GO_ION_SCALE", selected=None, rows=rows,
                    reason="none of {0.5, 1.0, 2.0} passed all five gates; the candidate set may "
                           "not be widened, the tie-break may not be changed and no gate may be "
                           "relaxed (plan §15)")
    best = min(adm, key=lambda r: (abs(r["f_prime"] - F_PRIME_PRIMARY), r["f_prime"]))
    return dict(status="SELECTED", selected=best["f_prime"], n_admissible=len(adm), rows=rows,
                tie_break="closest to f' = 1.0")


# =====================================================================================
#  Gate B -- the new interictal substrate (spec §9, plan §11).  All bounds pre-registered.
# =====================================================================================
GATE_B_MIN_SCOREABLE = 20          # per trajectory (plan §11)
GATE_B_MIN_FRAC = 0.15             # min(frac_A, frac_B) per trajectory
GATE_B_MIN_PASSING = 5             # of the 6 confirmatory trajectories -- a majority is not enough
GATE_B_K_WAVE_FAR_RATIO = 0.10     # far-field dK_o must stay under 10% of the event voxel's
GATE_B_PUMP_SAT_MAX = 0.50         # mean pump must stay well below rho
# "no slow countdown": 0.05 mM/s integrates to 0.55 mM over the 11 s window, about a quarter of the
# interictal Na excursion at f'=1 -- a drift that large IS a countdown.  The net drift bound is
# tighter still (0.02 mM/s -> 0.22 mM, ~10% of the excursion).
GATE_B_ION_BLOCK_DRIFT_MAX = 0.05
GATE_B_ION_NET_DRIFT_MAX = 0.02


def adjudicate_gate_B(runs, tolerances, *, template_layer):
    """Gate B verdict from the six confirmatory trajectories.

    Layer discipline (spec §9 rev3, CLAUDE.md §6.3): the real artifact supports only "TWO STABLE
    TEMPLATES EXIST"; the model side supports only "events initiate at both registered cores".
    Those are different layers and must never be collapsed -- in particular this gate never
    licenses "reproduced bidirectional propagation".
    """
    per = []
    for r in runs:
        p = r["pooled"]
        ion = r["ion"]
        minf = min(p["frac_A"], p["frac_B"])
        direction_ok = bool(p["n_scoreable"] >= GATE_B_MIN_SCOREABLE and minf >= GATE_B_MIN_FRAC)
        tol = {}
        for name, t in tolerances.items():
            if name not in p:
                continue
            delta = p[name] - t["off"]
            tol[name] = dict(value=p[name], accepted=t["off"], delta=delta, margin=t["margin"],
                             within=bool(abs(delta) <= t["margin"]),
                             underpowered=bool(t["underpowered"]))
        binding = {k: v for k, v in tol.items() if not v["underpowered"]}
        drift_ok = bool(ion["q99_abs_dNa_dt"] < GATE_B_ION_BLOCK_DRIFT_MAX
                        and ion["q99_abs_dKo_dt"] < GATE_B_ION_BLOCK_DRIFT_MAX
                        and abs(ion["Na_mean_last"] - ion["Na_mean_first"])
                        / 11.0 < GATE_B_ION_NET_DRIFT_MAX)
        wave_ok = bool(ion["k_wave_far_over_event"] < GATE_B_K_WAVE_FAR_RATIO)
        pump_ok = bool(ion["pump_saturation_frac_of_rho"] < GATE_B_PUMP_SAT_MAX)
        per.append(dict(
            tag=r["job"].get("tag"), conn_seed=r["job"]["conn_seed"],
            noise_seed=r["job"]["noise_seed"],
            direction=dict(ok=direction_ok, n_scoreable=p["n_scoreable"],
                           frac_A=p["frac_A"], frac_B=p["frac_B"], min_frac=minf),
            tolerance=tol,
            n_binding_outside=sum(1 for v in binding.values() if not v["within"]),
            binding_outside=[k for k, v in binding.items() if not v["within"]],
            ion=dict(no_slow_countdown=drift_ok, no_whole_sheet_K_wave=wave_ok,
                     pump_not_saturated=pump_ok,
                     q99_abs_dNa_dt=ion["q99_abs_dNa_dt"], q99_abs_dKo_dt=ion["q99_abs_dKo_dt"],
                     net_Na_drift_mM_s=abs(ion["Na_mean_last"] - ion["Na_mean_first"]) / 11.0,
                     k_wave_far_over_event=ion["k_wave_far_over_event"],
                     pump_saturation_frac_of_rho=ion["pump_saturation_frac_of_rho"])))

    n_dir = sum(1 for p in per if p["direction"]["ok"])
    b_real = dict(
        template_layer=template_layer,
        n_trajectories=len(per), n_direction_passing=n_dir,
        required=GATE_B_MIN_PASSING,
        ok=bool(n_dir >= GATE_B_MIN_PASSING),
        rule=f"per trajectory n_scoreable >= {GATE_B_MIN_SCOREABLE} and "
             f"min(frac_A, frac_B) >= {GATE_B_MIN_FRAC}; at least {GATE_B_MIN_PASSING} of "
             f"{len(per)} must satisfy it")
    b_model = dict(
        no_slow_countdown=all(p["ion"]["no_slow_countdown"] for p in per),
        no_whole_sheet_K_wave=all(p["ion"]["no_whole_sheet_K_wave"] for p in per),
        pump_not_saturated=all(p["ion"]["pump_not_saturated"] for p in per),
        n_runs_with_binding_metric_outside=sum(1 for p in per if p["n_binding_outside"] > 0),
        binding_metrics_outside=sorted({k for p in per for k in p["binding_outside"]}),
        note="UNDERPOWERED metrics are excluded from the binding set: 'within tolerance' there "
             "means 'indistinguishable at this window length', not equivalence")
    b_model["ok"] = bool(b_model["no_slow_countdown"] and b_model["no_whole_sheet_K_wave"]
                         and b_model["pump_not_saturated"]
                         and b_model["n_runs_with_binding_metric_outside"] == 0)

    status = "ACCEPTED" if (b_real["ok"] and b_model["ok"]) else "REJECTED"
    allowed = (
        "The reduced ion-homeostatic substrate recovered the accepted interictal working point "
        "and retained initiation at both registered cores under the locked effective "
        "normalization (g_K_ion = 1, eta_pump = 0)."
        if status == "ACCEPTED" else
        "Constitutive Na/K on this substrate did not recover the accepted interictal working point.")
    return dict(status=status, b_real=b_real, b_model=b_model, per_trajectory=per,
                allowed_statement=allowed,
                forbidden_statements=[
                    "reproduced bidirectional propagation (not established for E1146)",
                    "the ion mechanism is refuted (a Gate B failure is about THIS substrate)",
                    "the electrogenic pump pathway was tested (eta_pump was locked to 0)",
                    "a seizure lifecycle was obtained (B3/B4 were not authorised)",
                    "patient ion concentrations were reconstructed"],
                thresholds=dict(min_scoreable=GATE_B_MIN_SCOREABLE, min_frac=GATE_B_MIN_FRAC,
                                min_passing=GATE_B_MIN_PASSING,
                                k_wave_far_ratio=GATE_B_K_WAVE_FAR_RATIO,
                                pump_sat_max=GATE_B_PUMP_SAT_MAX,
                                ion_block_drift_max=GATE_B_ION_BLOCK_DRIFT_MAX,
                                ion_net_drift_max=GATE_B_ION_NET_DRIFT_MAX))


# =====================================================================================
#  T7.1 -- adjudication repair (user rulings, 2026-07-28).  See
#  docs/superpowers/specs/2026-07-28-topic4-fcxr-ion-T7_1-lock.md
# =====================================================================================
# Ruling 1(a): the relative `5*sigma_rest` term is REMOVED.  sigma_rest is not instrument noise on
#              this substrate -- it is the K background other real events leave behind -- and
#              picking a smaller multiplier would only introduce a new arbitrary parameter.  The
#              absolute floor carries the whole "the membrane must see it" intent.  Background
#              distribution / matched-background SNR stay as DIAGNOSTICS, never as a hard gate.
# Ruling 3:    per-sample monotonicity and smoothed-envelope monotonicity are both dropped.  Two
#              conditions replace them: a clear NET decay from peak to 20 s, and a tail-window
#              trend that is not persistently rising (transient up-jumps from background events
#              are allowed by construction, since a slope over the window averages them out).
# Ruling 4:    the integration ratio is a NON-BLOCKING diagnostic and a B2 risk register entry.
F_GATES_V2 = dict(
    measurable_abs_floor_mM=0.15,        # = 1 mV of E_K = 5.6% of V_th
    safe_ceiling_mM=0.90,                # = 5.4 mV = 30% of V_th from ONE interictal event
    k_recovery_window_s=3.0,             # 4.6 tau_Ko
    k_recovery_sigma=1.0,
    na_window_s=20.0,
    na_net_decay_min_frac=0.10,          # "clear net decay": at least 10% of the peak is gone
    na_tail_window_s=5.0,                # the last pre-registered tail window of the 20 s
    na_tail_slope_t_max=2.0,             # not persistently rising: slope <= 0 or t-stat < 2
)

# Retained from the small-network contract (ruling 5): the NUMERICAL tests only.  Abolished: using
# n1000/n4000 dynamic response to choose f', and any future "faithful reproduction" of that
# small-network T7 -- a fixed E->E in-degree makes small-net dynamics differ from 40k, which
# re-running cannot fix.
SMALL_NET_CONTRACT_RETAINED = ("Gate H numerical tests", "occupancy", "empty-voxel fixed point",
                               "finite-volume budget", "checkpoint/restart",
                               "initialization residual", "historical failure evidence")
SMALL_NET_CONTRACT_ABOLISHED = ("dynamic f' selection from n1000/n4000 response",
                                "future faithful reproduction of the small-network T7")


def coupled_working_point_jacobian(f_prime, *, r0=R0_HZ):
    """2x2 Jacobian of the homogeneous (Na_i, K_o) system at this f''s own interictal working point.

    Ruling 2: the Na reference must be computed on the COUPLED system.  Freezing K_o (as the first
    audit did) understates the clearance, because an event raises K_o in the participating voxels
    and I_pump rises with K_o as well as with Na.
    """
    q = q_ion_from_fprime(f_prime, r0=r0)
    Na, Ko = interictal_steady_state(q, r0)
    h = 1e-6
    dIp_dNa = float((pump_flux(Na + h, Ko) - pump_flux(Na - h, Ko)) / (2 * h))
    dIp_dK = float((pump_flux(Na, Ko + h) - pump_flux(Na, Ko - h)) / (2 * h))
    dIg_dK = float((glia_uptake(Ko + h) - glia_uptake(Ko - h)) / (2 * h))
    J = np.array([[-3.0 * dIp_dNa, -3.0 * dIp_dK],
                  [-2.0 * BETA * dIp_dNa, -2.0 * BETA * dIp_dK - EPS - dIg_dK]])
    return J, Na, Ko


def coupled_na_decay_prediction(f_prime, dNa0, dK0, *, t_s=20.0, r0=R0_HZ):
    """Predicted fraction of an event-induced Na excess cleared in `t_s`, propagating the measured
    initial (dNa, dK) perturbation through the coupled Jacobian."""
    from scipy.linalg import expm
    J, Na, Ko = coupled_working_point_jacobian(f_prime, r0=r0)
    v = np.array([float(dNa0), float(dK0)])
    if abs(v[0]) < 1e-15:
        return float("nan"), J, Na, Ko
    out = expm(J * float(t_s)) @ v
    return float(1.0 - out[0] / v[0]), J, Na, Ko


def _tail_slope(y, dt_s, window_s):
    """Least-squares slope of the final `window_s` of a trace, with its t statistic."""
    n = max(3, int(round(window_s / dt_s)))
    y = np.asarray(y, float)[-n:]
    x = np.arange(y.size) * dt_s
    A = np.vstack([x, np.ones_like(x)]).T
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    slope = float(coef[0])
    resid = y - A @ coef
    dof = max(1, y.size - 2)
    se = float(np.sqrt((resid @ resid) / dof / max(np.sum((x - x.mean()) ** 2), 1e-30)))
    return slope, se, (slope / se if se > 0 else 0.0)


def evaluate_f_prime_gates_v2(m):
    """T7.1: FOUR hard gates (measurable / safe / K recovery / numerical validity) plus
    non-blocking diagnostics.  Ruling 1(a), 3 and 4."""
    g = F_GATES_V2
    dK = float(m["dK_peak_single_mM"])
    net = float(m["na_net_decay_frac"])
    slope, se, tstat = m["na_tail_slope"], m["na_tail_slope_se"], m["na_tail_slope_t"]
    hard = {
        "measurable": dict(ok=bool(dK >= g["measurable_abs_floor_mM"]), value=dK,
                           threshold=g["measurable_abs_floor_mM"],
                           rule="single-event peak dK_o >= 0.15 mM (absolute floor only; the "
                                "5*sigma_rest term was removed -- sigma_rest is the background "
                                "other real events leave, not instrument noise)"),
        "safe": dict(ok=bool(dK <= g["safe_ceiling_mM"]), value=dK,
                     threshold=g["safe_ceiling_mM"],
                     rule="a SINGLE interictal event must not push the membrane 30% of V_th"),
        "recovery_K": dict(ok=bool(m["k_returns_within_1sigma_3s"]),
                           value=float(m["k_residual_after_3s_in_sigma"]),
                           threshold=g["k_recovery_sigma"],
                           rule="back inside 1 sigma of this voxel's resting mean within 3 s"),
        "numerical_validity": dict(ok=bool(m["numerically_valid"]),
                                   detail=m.get("numerical_detail", ""),
                                   rule="all concentrations positive and finite, no guard-band "
                                        "collision anywhere in the sensor run or the replays"),
    }
    na_ok = bool(net >= g["na_net_decay_min_frac"] and (slope <= 0.0 or tstat < g["na_tail_slope_t_max"]))
    diagnostics = {
        "na_recovery": dict(
            ok=na_ok, net_decay_frac=net, net_decay_min=g["na_net_decay_min_frac"],
            tail_window_s=g["na_tail_window_s"], tail_slope=slope, tail_slope_se=se,
            tail_slope_t=tstat, tail_not_persistently_rising=bool(slope <= 0.0 or tstat < 2.0),
            coupled_prediction=m.get("na_decay_pred_coupled"),
            k_clamped_measured=m.get("na_decay_frac_k_clamped"),
            rule="clear net decay from peak to 20 s AND a tail-window trend that is not "
                 "persistently rising; per-sample and smoothed-envelope monotonicity are both "
                 "dropped (ruling 3). Magnitude vs the coupled-Jacobian prediction is reported as "
                 "a diagnostic, not gated."),
        "integration": dict(
            value=float(m["integration_ratio_5th_over_1st"]),
            linear_prediction_at_workpoint=m.get("integration_linear_at_workpoint"),
            supralinear=bool(float(m["integration_ratio_5th_over_1st"])
                             > float(m.get("integration_linear_at_workpoint") or np.inf)),
            blocking=False,
            rule="NON-BLOCKING (ruling 4). Measured open loop, so it characterises the clearance "
                 "side only and may not gate B2. It is registered as a B2 pre-condition RISK: B2 "
                 "must report whether closed-loop K -> E_K -> firing overcomes this "
                 "load-dependent clearance."),
        "background": dict(sigma_rest_mM=m.get("sigma_rest_K_mM"),
                           matched_background_snr=m.get("matched_background_snr"),
                           blocking=False,
                           rule="reported for context only (ruling 1a)"),
    }
    return dict(admissible=all(v["ok"] for v in hard.values()), gates=hard,
                diagnostics=diagnostics, measured=m)


def select_f_prime_v2(rows):
    """T7.1 selection.  The 40k sensor replay is now the REGISTERED scale-diagnostic protocol
    (ruling 5), so it may produce a verdict -- but a PROVISIONAL candidate, not a canonical
    mechanism decision."""
    adm = [r for r in rows if r["admissible"]]
    if not adm:
        return dict(status="NO_ADMISSIBLE_SCALE", selected=None, rows=rows,
                    reason="no candidate passed all four hard gates")
    best = min(adm, key=lambda r: (abs(r["f_prime"] - F_PRIME_PRIMARY), r["f_prime"]))
    return dict(status="PROVISIONAL_CANDIDATE", selected=best["f_prime"],
                n_admissible=len(adm), rows=rows, tie_break="closest to f' = 1.0",
                semantics=("PROVISIONAL: chosen on an open-loop scale diagnostic. It is the value "
                           "B2 runs with; it is NOT a claim that the closed loop behaves well."))


# =====================================================================================
#  B2.1 -- calibration-instrument repair
#  docs/superpowers/specs/2026-07-28-topic4-fcxr-ion-B2_1-lock.md
# =====================================================================================
def _b2_1_bound(excursion, frac=0.10, window_s=10.0):
    """spec 2.1: the secular drift may move a variable by at most `frac` of its OWN interictal
    excursion over the measurement window."""
    return frac * excursion / window_s


_Q1 = q_ion_from_fprime(F_PRIME_PRIMARY)
_NA_STAR, _KO_STAR = interictal_steady_state(_Q1, R0_HZ)
B2_1_SLOPE_BOUND_NA = _b2_1_bound(_NA_STAR - NA_I0)      # 2.07e-2 mM/s
B2_1_SLOPE_BOUND_K = _b2_1_bound(_KO_STAR - K_O0)        # 1.09e-3 mM/s
B2_1_RATE_REL_TOL = 0.05
B2_1_ALPHA = 0.5
B2_1_MAX_UPDATES = 3
B2_1_SHRINK_N0 = 20.0        # spikes at which a cell's own count and its voxel weigh equally
B2_1_DRIVE_TOL = 0.15        # arms are comparable if spikes / participants agree within 15%


def signed_secular_slope(series, dt_s):
    """Least-squares SIGNED slope of every column of `series` (T, N) against time, in unit/s.

    This is the estimator the slow-countdown gate needs.  The q99 of |first differences| it
    replaces is a FLUCTUATION MAGNITUDE: on a substrate firing 2.2 interictal events per second
    every event contributes to it, so it stays large in a statistically stationary state.  A
    least-squares slope cancels those back-and-forth excursions and keeps only the one-way part.
    """
    y = np.asarray(series, float)
    if y.ndim == 1:
        y = y[:, None]
    t = np.arange(y.shape[0]) * float(dt_s)
    tc = t - t.mean()
    denom = float(tc @ tc)
    if denom <= 0:
        raise ValueError("need at least two distinct time points to fit a slope")
    return (tc @ (y - y.mean(axis=0))) / denom


def slope_stats(slopes):
    a = np.abs(np.asarray(slopes, float))
    return dict(q95_abs=float(np.quantile(a, 0.95)), q99_abs=float(np.quantile(a, 0.99)),
                max_abs=float(a.max()), mean_signed=float(np.mean(slopes)),
                median_signed=float(np.median(slopes)), n=int(a.size))


def shrink_rate_field(counts, voxel, *, window_s, n_voxels, n0=B2_1_SHRINK_N0):
    """Shrink each cell's measured rate toward its voxel mean (spec 2.2).

    An 11 s window gives a 4 Hz cell about 40 spikes -- a 16% relative standard error -- so raw
    per-cell rates would pour sampling noise into the initial ion state.  Weight
    w = n / (n + n0) means a cell with n0 spikes trusts itself and its neighbourhood equally.
    Call SEPARATELY for E and I: their baselines differ, and pooling would drag I toward E.
    """
    c = np.asarray(counts, float)
    v = np.asarray(voxel, int)
    rate = c / float(window_s)
    tot = np.bincount(v, weights=rate, minlength=n_voxels)
    num = np.bincount(v, minlength=n_voxels).astype(float)
    vox_mean = np.divide(tot, num, out=np.zeros_like(tot), where=num > 0)
    w = c / (c + float(n0))
    return w * rate + (1.0 - w) * vox_mean[v]


def damped_rate_update(current, measured, alpha=B2_1_ALPHA):
    """r^(k+1) = r^(k) + alpha * (measured - r^(k))  (spec 2.2)."""
    return np.asarray(current, float) + float(alpha) * (np.asarray(measured, float)
                                                        - np.asarray(current, float))


def damped_field_change(current, measured, alpha=B2_1_ALPHA):
    """The spec 2.2 convergence quantity: `max_i |r^(k+1)_i - r^(k)_i| / mean(r^(k))`.

    `r^(k+1)` is the DAMPED update, so the step is alpha * (measured - current), not the raw
    measurement difference.  The first implementation compared the undamped measurement, which
    reports exactly 1/alpha times too much; `undamped_max_rel` is kept so the two are comparable.

    The gate statistic stays `max` because spec 2.2 locks it and spec 4 forbids relaxing a
    threshold after seeing the data.  q95/q99 are reported ALONGSIDE, because a max over ~40k
    cells normalised by a population mean can be driven by a sparse tail, and the reader needs to
    see whether it is.
    """
    cur = np.asarray(current, float)
    step = np.abs(damped_rate_update(cur, measured, alpha) - cur)
    raw = np.abs(np.asarray(measured, float) - cur)
    denom = float(np.mean(cur))
    return dict(alpha=float(alpha), mean_current=denom,
                max_rel=float(step.max()) / denom,
                q99_rel=float(np.quantile(step, 0.99)) / denom,
                q95_rel=float(np.quantile(step, 0.95)) / denom,
                undamped_max_rel=float(raw.max()) / denom)


def active_voxel_count(dk_map, *, frac):
    """How many voxels rose to at least `frac` of THIS event's own peak excess potassium.

    Relative to the event's own peak, so the count measures spatial extent rather than amplitude:
    an event twice as strong everywhere has the same count.
    """
    d = np.asarray(dk_map, float)
    peak = float(d.max())
    return 0 if peak <= 0 else int(np.count_nonzero(d >= frac * peak))


def recruitment_radius_mm(pos, participating, *, center):
    """RMS distance of the participating cells from the kick centre.  NaN if nobody participated."""
    sel = np.asarray(pos, float)[np.asarray(participating, bool)]
    if sel.shape[0] == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.sum((sel - np.asarray(center, float)) ** 2, axis=1))))


def detect_k_rises(t_ms, y, *, min_climb):
    """Every monotone climb in a potassium trace whose total rise exceeds `min_climb`.

    Needed because the two 200 ms analysis windows in spec 2.3 turned out NOT to contain one
    kick response each: the network also fires its own bursts, so a window maximum is not an
    evoked amplitude.  Reporting the burst structure is what keeps that distinction visible.
    """
    t, v = np.asarray(t_ms, float), np.asarray(y, float)
    out, start = [], None
    for i, d in enumerate(np.diff(v)):
        if d > 0 and start is None:
            start = i
        elif d <= 0 and start is not None:
            if v[i] - v[start] > min_climb:
                out.append(dict(t_start_ms=float(t[start]), t_peak_ms=float(t[i]),
                                climb_mM=float(v[i] - v[start]), peak_mM=float(v[i])))
            start = None
    if start is not None and v[-1] - v[start] > min_climb:
        out.append(dict(t_start_ms=float(t[start]), t_peak_ms=float(t[-1]),
                        climb_mM=float(v[-1] - v[start]), peak_mM=float(v[-1])))
    return out


def adjudicate_b2_1_selfconsistency(m):
    """Both clauses of spec 2.2, and the slope must come from an INDEPENDENT window."""
    checks = dict(
        rate_converged=dict(ok=bool(m["rate_rel_change"] < B2_1_RATE_REL_TOL),
                            value=m["rate_rel_change"], threshold=B2_1_RATE_REL_TOL),
        slope_Na=dict(ok=bool(m["slope_q99_Na"] < B2_1_SLOPE_BOUND_NA),
                      value=m["slope_q99_Na"], threshold=B2_1_SLOPE_BOUND_NA),
        slope_K=dict(ok=bool(m["slope_q99_K"] < B2_1_SLOPE_BOUND_K),
                     value=m["slope_q99_K"], threshold=B2_1_SLOPE_BOUND_K),
        independent_window=dict(ok=bool(m["independent_window"]),
                                rule="the slope must be measured on a trajectory NOT used to "
                                     "derive the rate field"),
    )
    ok = all(v["ok"] for v in checks.values())
    return dict(status="CONVERGED" if ok else "NOT_CONVERGED", checks=checks,
                n_updates=m.get("n_updates"), max_updates=B2_1_MAX_UPDATES,
                semantics=("CONVERGED means the calibration INSTRUMENT is sound. It licenses "
                           "re-running the B2 closure; it is not a statement about the mechanism."))


def adjudicate_matched_control(closed, open_, *, structurally_identical_until_freeze):
    """spec 2.3, corrected 2026-07-28 after the first execution exposed a mis-specification.

    Validity here is STRUCTURAL, not response-based: the two arms are bit-identical simulations up
    to the freeze block (same q_ion, bias, initial ion field, seed, kick times / strength / radius;
    the only change is that the open arm pins `membrane_current()` from a block strictly before the
    first kick).  Every later difference is therefore attributable to the one factor that differs.

    The FIRST kick's response is a SANITY check that nothing pathological happened.  Later kicks
    are where the between-event feedback is allowed to act, so a divergence there is the EFFECT.

    The original rule pooled all kicks into one max and voided the comparison on it.  That is not a
    validity criterion: it declares the control broken exactly when the feedback matters most,
    which cannot distinguish 'the control is broken' from 'the effect is present'.
    """
    def _by_kick(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        return list(np.abs(a - b) / np.maximum(np.abs(b), 1e-12))

    spk = _by_kick(closed["spikes"], open_["spikes"])
    par = _by_kick(closed["participants"], open_["participants"])
    sane = bool(spk[0] <= B2_1_DRIVE_TOL and par[0] <= B2_1_DRIVE_TOL)
    ok = bool(structurally_identical_until_freeze and sane)
    out = dict(status="COMPARABLE" if ok else "UNRESOLVED_MATCHED_CONTROL",
               structurally_identical_until_freeze=bool(structurally_identical_until_freeze),
               sanity=dict(kick1_spikes_rel_diff=spk[0], kick1_participants_rel_diff=par[0],
                           tolerance=B2_1_DRIVE_TOL, ok=sane,
                           rule="the first kick's response must agree; the arms are identical up "
                                "to the freeze, so a large disagreement already at kick 1 would "
                                "mean something pathological, not feedback"),
               effect=dict(spikes_rel_diff_by_kick=spk, participants_rel_diff_by_kick=par,
                           rule="differences from the SECOND kick on are the feedback effect and "
                                "must NOT void the comparison"),
               closed=closed, open=open_)
    if not ok:
        out["reason"] = (("the arms were not structurally identical up to the freeze block"
                          if not structurally_identical_until_freeze else
                          "the FIRST kick's response already disagreed, so the arms did not start "
                          "comparably") + "; no reading of the feedback may be taken from the "
                         "peak ratio")
        return out
    cp, op = np.asarray(closed["peaks"], float), np.asarray(open_["peaks"], float)
    out["ratio"] = dict(closed_2nd_over_1st=float(cp[1] / cp[0]),
                        open_2nd_over_1st=float(op[1] / op[0]),
                        closed_exceeds_open=bool(cp[1] / cp[0] > op[1] / op[0]))
    return out


def withhold_canonical_verdict(sel, *, protocol_deviation, blocking_gates_are_open_loop):
    """A result obtained on a DIFFERENT experimental object may not inherit the canonical verdict
    of the contract it replaced (CLAUDE.md §5: the pre-registered tier is fixed at planning time).

    `select_f_prime` implements the pre-registered small-network contract faithfully and is left
    untouched, so a future faithful run still gets `NO_GO_ION_SCALE`.  This wrapper is what the
    runner emits when the object changed: the per-gate table stands as a diagnostic, the canonical
    label is withheld, and the reason is carried in the artifact rather than in prose only.
    """
    out = dict(sel)
    out["contract_verdict_if_protocol_had_matched"] = sel["status"]
    out["status"] = "UNRESOLVED_T7_PROTOCOL"
    out["selected"] = None
    out["verdict_withheld_because"] = protocol_deviation
    out["blocking_gates_measured_open_loop"] = blocking_gates_are_open_loop
    out["semantics"] = (
        "UNRESOLVED_T7_PROTOCOL means: the five per-gate measurements below are valid AS A "
        "DIAGNOSTIC on the object that was actually run, but they do not constitute the "
        "pre-registered scale-selection decision, because that decision was registered against a "
        "different object. It is NOT a mechanism NO-GO and must never be reported as one.")
    out["b2_entry"] = (
        "B2 stays closed because no f' was selected -- not because a mechanism was refuted. "
        "Re-opening it requires a re-locked T7 contract, not a relaxed gate.")
    return out


MIN_SCOREABLE_EVENTS = 20      # spec §9 B-real hard precondition


def direction_power_gate(stats):
    """B0-2 power precondition (plan §4).  Deliberately returns NO usable threshold: the 0.15
    minimum-fraction threshold is pre-locked later, on the development seed, and only after this
    precondition passes."""
    n = int(stats["n_scoreable"])
    fa, fb = float(stats["frac_A"]), float(stats["frac_B"])
    ok = n >= MIN_SCOREABLE_EVENTS and fa > 0.0 and fb > 0.0
    out = dict(status="PASS" if ok else "INSUFFICIENT_POWER",
               n_scoreable=n, frac_A=fa, frac_B=fb,
               required_n_scoreable=MIN_SCOREABLE_EVENTS,
               required="n_scoreable >= 20 and frac_A > 0 and frac_B > 0")
    if not ok:
        out["remedy"] = ("lengthen the window or replace the readout; do NOT relax the gate and "
                         "do NOT enter B2 with a criterion that cannot be measured on the "
                         "accepted baseline")
    return out
