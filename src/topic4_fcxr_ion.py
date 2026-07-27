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
