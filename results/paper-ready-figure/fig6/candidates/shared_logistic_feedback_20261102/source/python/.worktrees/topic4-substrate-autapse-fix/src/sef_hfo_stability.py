"""Topic 4 SEF-HFO — Step 0a delayed linear stability and dispersion map.

Finds self-consistent steady-state operating points, linearises the delayed rate
field around each point to build the spatial-frequency dispersion relation
λ(k), and screens the operating-point family for low-heterogeneity gain-shift
and finite-k instability windows (candidate Step 0b targets).

CLI: ``scripts/run_sef_hfo_step0a_stability.py``,
     ``scripts/run_sef_hfo_step0a_capability_probe.py``.
"""
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

def erlang_n_convergence(p, op, k_grid, n_values=(1, 2, 4, 8), tol=1e-3):
    """Leading-mode growth rate vs Erlang n. Returns smallest n past which it stops moving."""
    vals = [(n, max_growth_rate(replace(p, erlang_n=n), op, k_grid)) for n in n_values]
    deltas = [abs(vals[i][1] - vals[i - 1][1]) for i in range(1, len(vals))]
    converged = bool(deltas and deltas[-1] < tol)
    rec = next((n for (n, _), d in zip(vals[1:], deltas) if d < tol), n_values[-1])
    return {"values": vals, "deltas": deltas, "converged": converged, "recommended_n": rec}

def _char_det(lam, p, op, kpar, kperp):
    """EXACT delayed dispersion determinant D(lambda,k) with e^{-lambda d}/(1+lambda tau_syn)."""
    G = {"E": gain(op["h_E0"], p.phi_bar, p.sigma_phi, p.beta),
         "I": gain(op["h_I0"], p.phi_bar, p.sigma_phi, p.beta)}
    def H(tau_syn):
        return np.exp(-lam * p.delay_d) / (1.0 + lam * tau_syn)
    WEE = p.J_EE * gaussian_hat(kpar, kperp, p.ell_par, p.ell_perp)
    WEI = p.J_EI * _inhib_hat(p, kpar, kperp)
    WIE = p.J_IE * gaussian_hat(kpar, kperp, p.sigma_IE, p.sigma_IE)
    WII = p.J_II * gaussian_hat(kpar, kperp, p.sigma_II, p.sigma_II)
    a = (1 + p.tau_E * lam) - G["E"] * WEE * H(p.tau_AMPA)
    b = G["E"] * WEI * H(p.tau_GABA)
    c = -G["I"] * WIE * H(p.tau_AMPA)
    d = (1 + p.tau_I * lam) + G["I"] * WII * H(p.tau_GABA)
    return a * d - b * c

def transcendental_max_re(p, op, k0, re_lo, re_hi, im_hi, n_re=24, n_im=24):
    """Bounded rightmost-root estimate of the EXACT delayed dispersion D(lambda,k0)=0.
    Multi-start complex Newton (fsolve) seeded on a grid over [re_lo,re_hi] x [0,im_hi];
    keep only converged in-box roots with small residual; return the largest Re(lambda).
    Roots are conjugate-symmetric (real params => D(conj)=conj(D)), so scanning Im>=0
    captures the rightmost root. NOTE: this replaces the plan's corner-sign-change box
    scan, which had a real-axis blind spot -- its root-in-cell test min(Im)<0<max(Im)
    can never fire on the bottom row (Im(D)=0 on the real axis), so it missed roots
    within one Im-grid step of the axis and returned -inf. See step0_results writeup."""
    res = np.linspace(re_lo, re_hi, n_re); ims = np.linspace(0.0, im_hi, n_im)
    def _fr(v):
        d = _char_det(complex(v[0], v[1]), p, op, k0, 0.0)
        return [d.real, d.imag]
    best = -np.inf
    for r0 in res:
        for i0 in ims:
            sol, info, ier, _ = fsolve(_fr, [r0, i0], full_output=True)
            if ier != 1:
                continue
            re, im = float(sol[0]), float(sol[1])
            if not (re_lo - 1e-6 <= re <= re_hi + 1e-6 and abs(im) <= im_hi + 1e-6):
                continue
            if abs(complex(*_fr(sol))) > 1e-7:
                continue
            best = max(best, re)
    return best

def leading_mode(p, op, k_grid):
    """Return (k*, omega*, max Re lambda) for the dominant mode (for k*/frequency reporting)."""
    best = (-np.inf, 0.0, 0.0)
    for kpar in k_grid:
        ev = np.linalg.eigvals(build_dispersion_matrix(p, op, float(kpar), 0.0))
        j = int(np.argmax(ev.real))
        if ev.real[j] > best[0]:
            best = (float(ev.real[j]), float(kpar), float(abs(ev.imag[j])))
    return {"max_re": best[0], "k_star": best[1], "omega_star": best[2]}

def _assert_only_sigma_phi_differs(p_base, p_patch):
    for fld in fields(p_base):
        if fld.name == "sigma_phi": continue
        if getattr(p_base, fld.name) != getattr(p_patch, fld.name):
            raise ValueError(f"forbidden rescue: '{fld.name}' changed alongside sigma_phi")

def screen_low_heterogeneity_effect(p_base, admissible_operating_points, sigma_phi_patch, k_grid):
    if sigma_phi_patch >= p_base.sigma_phi:
        raise ValueError("patch must LOWER sigma_phi")
    p_patch = replace(p_base, sigma_phi=sigma_phi_patch)
    _assert_only_sigma_phi_differs(p_base, p_patch)
    per = []
    for I_E, I_I in admissible_operating_points:
        ob = self_consistent_operating_point(p_base, I_E, I_I)
        opp = self_consistent_operating_point(p_patch, I_E, I_I)
        eb = eta_lin(p_base, ob, k_grid); ep = eta_lin(p_patch, opp, k_grid)
        per.append({"I_E": I_E, "I_I": I_I, "eta_baseline": eb, "eta_patch": ep,
                    "closer_to_critical": bool(ep < eb)})
    return {"fraction_closer": float(np.mean([d["closer_to_critical"] for d in per])),
            "n_admissible": len(per), "per_point": per}
