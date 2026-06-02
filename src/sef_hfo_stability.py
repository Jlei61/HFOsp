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

def gaussian_hat(kpar, kperp, ell_par, ell_perp):
    return np.exp(-0.5 * (ell_par**2 * kpar**2 + ell_perp**2 * kperp**2))

def _inhib_hat(p, kpar, kperp):
    wide = (1.0 - p.gamma_global) * gaussian_hat(kpar, kperp, p.sigma_I, p.sigma_I)
    return wide + (p.gamma_global if (kpar == 0.0 and kperp == 0.0) else 0.0)

def _connection_specs(p):
    # (post, pre, sign, J, kernel_hat, tau_syn[by presynaptic transmitter])
    return [
        ("E", "E", +1, p.J_EE, gaussian_hat(0, 0, p.ell_par, p.ell_perp), p.tau_AMPA, "EE"),
        ("E", "I", -1, p.J_EI, None, p.tau_GABA, "EI"),
        ("I", "E", +1, p.J_IE, None, p.tau_AMPA, "IE"),
        ("I", "I", -1, p.J_II, None, p.tau_GABA, "II"),
    ]

def _kernel_hat(p, tag, kpar, kperp):
    if tag == "EE": return gaussian_hat(kpar, kperp, p.ell_par, p.ell_perp)
    if tag == "EI": return _inhib_hat(p, kpar, kperp)
    if tag == "IE": return gaussian_hat(kpar, kperp, p.sigma_IE, p.sigma_IE)
    if tag == "II": return gaussian_hat(kpar, kperp, p.sigma_II, p.sigma_II)

def _chain_rates(p, tau_syn):
    """Erlang-n conduction-delay stages (rate n/d) then one synaptic stage (1/tau_syn)."""
    rates = []
    if p.erlang_n > 0 and p.delay_d > 0:
        rates += [p.erlang_n / p.delay_d] * p.erlang_n
    if tau_syn > 0:
        rates += [1.0 / tau_syn]
    return rates

def build_dispersion_matrix(p, op, kpar, kperp):
    """Constant linear matrix whose eigenvalues are the growth rates at mode (kpar,kperp).
    State = [r_E, r_I, (per-connection delay+synaptic chain stages...), (recovery a)]."""
    G = {"E": gain(op["h_E0"], p.phi_bar, p.sigma_phi, p.beta),
         "I": gain(op["h_I0"], p.phi_bar, p.sigma_phi, p.beta)}
    tau = {"E": p.tau_E, "I": p.tau_I}
    idx = {"E": 0, "I": 1}; nxt = 2; chains = []
    for (post, pre, sign, J, _, tau_syn, tag) in _connection_specs(p):
        weight = J * _kernel_hat(p, tag, kpar, kperp)
        rates = _chain_rates(p, tau_syn)
        sidx = list(range(nxt, nxt + len(rates))); nxt += len(rates)
        chains.append((post, pre, sign, weight, sidx, rates))
    rec_idx = None
    if p.b_a > 0: rec_idx = nxt; nxt += 1
    M = np.zeros((nxt, nxt))
    M[0, 0] = -1.0 / p.tau_E; M[1, 1] = -1.0 / p.tau_I
    for (post, pre, sign, weight, sidx, rates) in chains:
        out = sidx[-1] if sidx else idx[pre]
        M[idx[post], out] += G[post] * sign * weight / tau[post]
        src = idx[pre]
        for j, (si, rj) in enumerate(zip(sidx, rates)):
            M[si, si] += -rj; M[si, src] += rj; src = si
    if rec_idx is not None:
        M[rec_idx, rec_idx] = -1.0 / p.tau_a; M[rec_idx, 0] = 1.0 / p.tau_a
        M[0, rec_idx] += G["E"] * (-p.b_a) / p.tau_E
    return M

def max_growth_rate(p, op, k_grid):
    best = -np.inf
    for kpar in k_grid:
        for kperp in k_grid:
            ev = np.linalg.eigvals(build_dispersion_matrix(p, op, float(kpar), float(kperp)))
            best = max(best, float(ev.real.max()))
    return best

def eta_lin(p, op, k_grid):
    return -max_growth_rate(p, op, k_grid)

def leading_mode(p, op, k_grid):
    """Return (k*, omega*, max Re lambda) for the dominant mode (for k*/frequency reporting)."""
    best = (-np.inf, 0.0, 0.0)
    for kpar in k_grid:
        ev = np.linalg.eigvals(build_dispersion_matrix(p, op, float(kpar), 0.0))
        j = int(np.argmax(ev.real))
        if ev.real[j] > best[0]:
            best = (float(ev.real[j]), float(kpar), float(abs(ev.imag[j])))
    return {"max_re": best[0], "k_star": best[1], "omega_star": best[2]}
