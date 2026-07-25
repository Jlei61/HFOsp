# src/topic4_zm_field_screen.py
"""Reduced 2-D S_L(x)+S_G rate field (Fix A dual pool) + anisotropic K_E + per-mode Floquet + streaming
metrics + 4 arms. Spec 2026-07-24 rev3. Reduced rate field only -- no SNN, no H, no E->E."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

def _offset_grid(n, L):
    """Periodic signed offsets (mm); offset (0,0) sits at index (0,0). DX varies along axis 0."""
    idx = (np.arange(n) + n // 2) % n - n // 2
    d = idx * (L / n)
    return np.meshgrid(d, d, indexing="ij")

def elliptical_exp_kernel(n, L, l_par, l_perp, theta):
    DX, DY = _offset_grid(n, L)
    u = DX * np.cos(theta) + DY * np.sin(theta)
    v = -DX * np.sin(theta) + DY * np.cos(theta)
    K = np.exp(-np.sqrt((u / l_par) ** 2 + (v / l_perp) ** 2))
    K[0, 0] = 0.0
    return K / K.sum()

def gaussian_kernel(n, L, sigma):
    DX, DY = _offset_grid(n, L)
    K = np.exp(-0.5 * (DX ** 2 + DY ** 2) / sigma ** 2)
    return K / K.sum()

def cell_mass_fraction(L, n, l_par=0.537, l_perp=0.269, theta=0.0, sub=64, span_cells=12):
    """q_cell = fraction of the CONTINUOUS K_E mass inside one lattice cell (fine quadrature). Sets the
    local/non-local recurrent split w_rec=W0*q_cell, w_c=W0*(1-q_cell) -- derived, never hand-set."""
    h = L / n
    xs = (np.arange(-span_cells * sub, span_cells * sub) + 0.5) * (h / sub)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    u = X * np.cos(theta) + Y * np.sin(theta); v = -X * np.sin(theta) + Y * np.cos(theta)
    K = np.exp(-np.sqrt((u / l_par) ** 2 + (v / l_perp) ** 2))
    inside = (np.abs(X) <= h / 2) & (np.abs(Y) <= h / 2)
    return float(K[inside].sum() / K.sum())

def kernel_axis_and_ar(K, L):
    """Principal axis (rad) + aspect ratio of a kernel, via its mass-weighted spatial covariance."""
    n = K.shape[0]
    DX, DY = _offset_grid(n, L)
    w = K / K.sum()
    cxx = float((w * DX * DX).sum()); cyy = float((w * DY * DY).sum()); cxy = float((w * DX * DY).sum())
    C = np.array([[cxx, cxy], [cxy, cyy]])
    vals, vecs = np.linalg.eigh(C)
    v = vecs[:, np.argmax(vals)]
    return float(np.arctan2(v[1], v[0])), float(np.sqrt(max(vals) / max(min(vals), 1e-30)))

# append to src/topic4_zm_field_screen.py
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src", "snn_engine"))
from slow_field import psi_recruit, pnorm_pool   # noqa: E402  (reuse the SNN pooling nonlinearities)

ARMS = ("div_global", "dual_global", "dual_local", "dual_mixed")

def _Fsat(U, u_half=0.5):
    X = np.maximum(U, 0.0)
    return X / (u_half + X)

@dataclass
class FieldParams:
    W0: float; alpha: float; beta: float; theta: float; I0: float
    n: int = 32; L: float = 20.0
    tau_a: float = 10.0; tau_mu: float = 30.0; tau_S: float = 80.0; S_max: float = 1.0
    r50: float = 0.4; n_psi: float = 2.0; p_pool: float = 3.0
    sigma_S: float = 2.0; l_par: float = 0.537; l_perp: float = 0.269; theta_EE: float = 0.0
    eps_G: float = 0.2
    w_frac: float | None = None          # None -> DERIVED from cell_mass_fraction (never hand-set 0.5)

def resolve_w_frac(p: FieldParams):
    return float(p.w_frac) if p.w_frac is not None else cell_mass_fraction(p.L, p.n, p.l_par, p.l_perp, p.theta_EE)

def arm_beta(p: FieldParams, arm):
    return 0.0 if arm == "div_global" else p.beta

def _S_eff(arm, SL, SG, eps_G):
    if arm in ("div_global", "dual_global"):
        return SG
    if arm == "dual_local":
        return SL
    return (1.0 - eps_G) * SL + eps_G * SG

def simulate_field(p: FieldParams, arm, T=6000.0, dt=0.25, seed=0, r_init=None, state_init=None,
                   record_stride=20):
    assert arm in ARMS, arm
    rng = np.random.default_rng(seed); n = p.n
    KE = np.fft.rfft2(elliptical_exp_kernel(n, p.L, p.l_par, p.l_perp, p.theta_EE))
    KS = np.fft.rfft2(gaussian_kernel(n, p.L, p.sigma_S))
    q = resolve_w_frac(p); w_rec, w_c = p.W0 * q, p.W0 * (1.0 - q)
    beta = arm_beta(p, arm)
    if state_init is not None:
        r = np.array(state_init["r"], float); muL = np.array(state_init["muL"], float)
        SL = np.array(state_init["SL"], float); muG = float(state_init["muG"]); SG = float(state_init["SG"])
    else:
        r = np.array(r_init, float) if r_init is not None else np.full((n, n), 0.15)
        muL = np.zeros((n, n)); SL = np.zeros((n, n)); muG = 0.0; SG = 0.0
    r = np.maximum(r, 0.0); rec = []
    for t in range(int(round(T / dt))):
        Se = _S_eff(arm, SL, SG, p.eps_G)
        rec_E = w_rec * r + w_c * np.fft.irfft2(np.fft.rfft2(r) * KE, s=(n, n))
        u = p.I0 + rec_E / (1.0 + p.alpha * Se) - beta * Se - p.theta
        r = np.maximum(r + dt * (-r + _Fsat(u, 0.5)) / p.tau_a, 0.0)
        z = psi_recruit(r, 0.0, p.r50, p.n_psi)                       # nonlinearity FIRST (per location)
        conv = np.fft.irfft2(np.fft.rfft2(z ** p.p_pool) * KS, s=(n, n))
        A_L = np.maximum(conv, 0.0) ** (1.0 / p.p_pool)               # clamp: FFT roundoff -> NaN under ^(1/p)
        A_G = pnorm_pool(z, p.p_pool)
        muL += dt * (-muL + A_L) / p.tau_mu; SL += dt * (-SL + p.S_max * muL) / p.tau_S
        muG += dt * (-muG + A_G) / p.tau_mu; SG += dt * (-SG + p.S_max * muG) / p.tau_S
        if t % record_stride == 0:
            rec.append(r.astype(np.float32).copy())
    return dict(r_trace=np.asarray(rec), t_ms=np.arange(len(rec)) * record_stride * dt,
                final_state=dict(r=r, muL=muL, SL=SL, muG=muG, SG=SG))

# append to src/topic4_zm_field_screen.py
def _cycle_crossings(x):
    """Upward mid-line crossing indices (relaxation-oscillation cycle markers)."""
    mid = 0.5 * (float(np.max(x)) + float(np.min(x)))
    x = np.asarray(x)
    return np.flatnonzero((x[:-1] < mid) & (x[1:] >= mid))

def field_metrics(r_trace, dt_rec_ms, a_max=1.0, settle=0.25):
    R = np.asarray(r_trace, float)[int(len(r_trace) * settle):]
    nt = R.shape[0]; flat = R.reshape(nt, -1); ncell = flat.shape[1]
    P = flat.mean(axis=1)
    P95 = float(np.percentile(P, 95))
    occ = float((P >= 0.2 * P95).mean()) if P95 > 1e-12 else 0.0
    amp = flat.max(axis=0) - flat.min(axis=0)
    active = amp >= 0.1 * a_max
    crossings = [(_cycle_crossings(flat[:, c]) if active[c] else np.array([], int)) for c in range(ncell)]
    ncyc = np.array([c.size for c in crossings])
    osc_cells = (ncyc >= 10) & (amp / a_max >= 0.20)
    # per-cell (LOCAL) period -- the gate metric; population period is diagnostic only
    locp = [float(np.mean(np.diff(crossings[c])) * dt_rec_ms) for c in np.flatnonzero(osc_cells)
            if crossings[c].size >= 2]
    median_local_period = float(np.median(locp)) if locp else float("nan")
    # phase per oscillatory cell, then R(t) only where coverage >= 50%
    phases = np.full((nt, ncell), np.nan)
    for c in np.flatnonzero(osc_cells):
        cr = crossings[c]
        for a, b in zip(cr[:-1], cr[1:]):
            phases[a:b, c] = 2 * np.pi * (np.arange(a, b) - a) / (b - a)
    n_osc = int(osc_cells.sum())
    Rt, cov = [], []
    for t in range(nt):
        ph = phases[t][~np.isnan(phases[t])]
        c = (ph.size / n_osc) if n_osc else 0.0
        cov.append(c)
        if n_osc and c >= 0.5 and ph.size >= 2:
            Rt.append(abs(np.mean(np.exp(1j * ph))))
    median_R = float(np.median(Rt)) if Rt else 1.0            # fail closed: no valid coverage -> "synchronised"
    act = flat[:, active]
    if act.shape[1] >= 2 and np.all(act.std(axis=0) > 0):
        C = np.corrcoef(act.T); iu = np.triu_indices(act.shape[1], 1); mpc = float(np.nanmean(C[iu]))
    else:
        mpc = 1.0
    crP = _cycle_crossings(P)
    pop_period = float(np.mean(np.diff(crP)) * dt_rec_ms) if crP.size >= 2 else float("nan")
    return dict(occupancy=occ, P95=P95, mean_P=float(P.mean()), active_area_frac=float(active.mean()),
                osc_frac=float(osc_cells.mean()), median_R_phase=median_R,
                phase_coverage_frac=float(np.mean(cov)), mean_pair_corr=mpc,
                median_local_period_ms=median_local_period, population_period_ms=pop_period)

# append to src/topic4_zm_field_screen.py
from src.topic4_zm_field_meanfield import (simulate_meanfield, MFParams, detect_orbit, psi_prime)

def uniform_orbit(p: FieldParams, dt, T=6000.0, settle=0.5):
    mf = MFParams(p.W0, p.alpha, p.beta, p.theta, p.I0, p.tau_a, p.tau_mu, p.tau_S, p.S_max)
    tr = simulate_meanfield(mf, T=T, dt=dt)
    o = detect_orbit(tr, dt, settle)
    if not o["oscillates"]:
        raise ValueError("no uniform orbit at this operating point (Phase-0 STOP condition)")
    per = max(2, int(round(o["period_ms"] / dt)))
    tail = tr[int(len(tr) * settle):]
    return tail[:per].copy(), o["period_ms"]

def mode_responses(p: FieldParams, mx, my):
    """(W_k, Khat_sigmaS(k)) at the INTEGER FFT lattice mode (mx,my)."""
    n = p.n
    KE = np.fft.fft2(elliptical_exp_kernel(n, p.L, p.l_par, p.l_perp, p.theta_EE))
    KS = np.fft.fft2(gaussian_kernel(n, p.L, p.sigma_S))
    q = resolve_w_frac(p); w_rec, w_c = p.W0 * q, p.W0 * (1.0 - q)
    return w_rec + w_c * float(KE[mx % n, my % n].real), float(KS[mx % n, my % n].real)

def variational_jacobian(p: FieldParams, arm, Wk, Kk, r0, mu0, S0, is_dc=False):
    beta = arm_beta(p, arm)
    D = 1.0 + p.alpha * S0
    u0 = p.I0 + p.W0 * r0 / D - beta * S0 - p.theta            # BASE state uses W0 (uniform), NOT Wk
    Fp = 0.0 if u0 <= 0 else 0.5 / (0.5 + u0) ** 2
    a_rr = (-1.0 + Fp * Wk / D) / p.tau_a
    if arm in ("div_global", "dual_global") and not is_dc:
        return np.array([[a_rr]])                              # global pool has no d.o.f. off DC -> 1-D
    c_S = 1.0 if arm == "dual_local" else (1.0 - p.eps_G) if arm == "dual_mixed" else 1.0
    a_rS = Fp * (-p.alpha * p.W0 * r0 / D ** 2 - beta) * c_S / p.tau_a
    a_mr = Kk * psi_prime(r0, p.r50, p.n_psi) / p.tau_mu
    return np.array([[a_rr, 0.0, a_rS],
                     [a_mr, -1.0 / p.tau_mu, 0.0],
                     [0.0, p.S_max / p.tau_S, -1.0 / p.tau_S]])

def transverse_floquet(p: FieldParams, arm, mx, my, orbit, dt):
    """lambda_perp at integer mode (mx,my) via the monodromy over one orbit period. DC is not transverse."""
    if (mx % p.n, my % p.n) == (0, 0):
        raise ValueError("(0,0) is the DC mode; it is not a transverse mode (its multiplier is neutral)")
    Wk, Kk = mode_responses(p, mx, my)
    dim = 1 if arm in ("div_global", "dual_global") else 3
    M = np.eye(dim)
    for r0, mu0, S0 in orbit:
        J = variational_jacobian(p, arm, Wk, Kk, r0, mu0, S0)
        M = (np.eye(dim) + dt * J) @ M
    rho = float(np.max(np.abs(np.linalg.eigvals(M))))
    return float(np.log(max(rho, 1e-300)) / (len(orbit) * dt))

def floquet_map(p: FieldParams, arm, orbit, dt, m_max=6):
    modes, lam = [], []
    for mx in range(-m_max, m_max + 1):
        for my in range(-m_max, m_max + 1):
            if (mx, my) == (0, 0):
                continue                                        # DC excluded from the transverse map
            modes.append((mx, my)); lam.append(transverse_floquet(p, arm, mx, my, orbit, dt))
    lam = np.asarray(lam); i = int(np.argmax(lam))
    return dict(modes=modes, lam=lam, lam_max=float(lam[i]), k_star=modes[i])
