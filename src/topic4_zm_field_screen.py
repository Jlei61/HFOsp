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
