# src/sef_hfo_stability.py
"""SEF-HFO Step-0a delayed linear stability. See 2026-06-02 amendment."""
import numpy as np
from dataclasses import replace, fields
from scipy.optimize import fsolve

def _f(x, beta): return 1.0 / (1.0 + np.exp(-beta * x))
def _fprime(x, beta):
    s = _f(x, beta); return beta * s * (1.0 - s)

_Z = np.linspace(-8.0, 8.0, 2001); _PZ = np.exp(-0.5 * _Z**2) / np.sqrt(2 * np.pi)

def F_eff(h, phi_bar, sigma_phi, beta):
    return np.trapz(_f(h - (phi_bar + sigma_phi * _Z), beta) * _PZ, _Z)

def gain(h, phi_bar, sigma_phi, beta):
    return np.trapz(_fprime(h - (phi_bar + sigma_phi * _Z), beta) * _PZ, _Z)

def self_consistent_operating_point(p, I_E, I_I, guesses=None, tol=1e-8):
    """Solve r0 = F_eff(W_hat(0) r0 + I) from multiple initial guesses; report
    convergence, residual, and root multiplicity (near-critical bistability)."""
    def resid(r):
        rE, rI = r
        hE = p.J_EE * rE - p.J_EI * rI + I_E - p.b_a * rE
        hI = p.J_IE * rE - p.J_II * rI + I_I
        return [rE - F_eff(hE, p.phi_bar, p.sigma_phi, p.beta),
                rI - F_eff(hI, p.phi_bar, p.sigma_phi, p.beta)]
    if guesses is None:
        guesses = [(a, b) for a in (0.02, 0.2, 0.5, 0.9) for b in (0.02, 0.2, 0.5)]
    roots = []
    for g in guesses:
        sol, info, ier, _ = fsolve(resid, g, full_output=True)
        if ier == 1 and max(abs(np.array(resid(sol)))) < tol and (sol >= -1e-6).all():
            if not any(np.allclose(sol, r, atol=1e-4) for r in roots):
                roots.append(sol)
    if not roots:
        return {"converged": False, "n_distinct_roots": 0}
    rE0, rI0 = max(roots, key=lambda r: r[0])     # pick high-activity root deterministically
    hE0 = p.J_EE * rE0 - p.J_EI * rI0 + I_E - p.b_a * rE0
    hI0 = p.J_IE * rE0 - p.J_II * rI0 + I_I
    return {"r_E0": float(rE0), "r_I0": float(rI0), "h_E0": float(hE0), "h_I0": float(hI0),
            "converged": True, "n_distinct_roots": len(roots), "bistable": len(roots) > 1}
