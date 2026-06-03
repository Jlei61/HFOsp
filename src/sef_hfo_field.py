# src/sef_hfo_field.py
"""SEF-HFO Step-0 rate field. ALL numeric defaults are TEST-ONLY SCAFFOLDS
(Step 0a/0b screen them); formal results require data-anchored units (see gate).
See docs/archive/topic4/sef_hfo_topic4_v2_plan_2026-06-01.md 2026-06-02 amendment."""
import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class SEFParams:
    n: int = 64
    L: float = 64.0
    tau_E: float = 1.0
    tau_I: float = 2.0
    tau_a: float = 20.0           # recovery timescale (anchor to event duration at formal run)
    tau_AMPA: float = 0.5         # E-presynaptic synaptic filter (fast)
    tau_GABA: float = 2.0         # I-presynaptic synaptic filter (slow)
    delay_d: float = 1.0          # conduction delay (Erlang chain mean)
    erlang_n: int = 2             # Erlang stages approximating the delay (n->inf = pure delay)
    J_EE: float = 1.0
    J_EI: float = 1.0
    J_IE: float = 1.0
    J_II: float = 0.5
    ell_par: float = 6.0
    ell_perp: float = 2.0
    axis_angle: float = 0.0
    sigma_I: float = 10.0
    sigma_IE: float = 4.0
    sigma_II: float = 4.0
    gamma_global: float = 0.2
    b_a: float = 0.0              # recovery strength; 0 = OFF
    beta: float = 4.0
    phi_bar: float = 0.0
    sigma_phi: float = 1.0

from src.sef_hfo_stability import F_eff, _connection_specs, _kernel_hat, _chain_rates

def _grid(n, L):
    x = (np.arange(n) - n // 2) * (L / n); return np.meshgrid(x, x, indexing="ij")

def anisotropic_gaussian(n, L, ell_par, ell_perp, angle):
    X, Y = _grid(n, L); u = np.cos(angle)*X + np.sin(angle)*Y; v = -np.sin(angle)*X + np.cos(angle)*Y
    g = np.exp(-(u**2)/(2*ell_par**2) - (v**2)/(2*ell_perp**2)); return g/g.sum()

def isotropic_gaussian(n, L, sigma):
    X, Y = _grid(n, L); g = np.exp(-(X**2 + Y**2)/(2*sigma**2)); return g/g.sum()

def uniform_kernel(n):
    k = np.ones((n, n)); return k/k.sum()

def build_kernels(p):
    return {"EE": anisotropic_gaussian(p.n, p.L, p.ell_par, p.ell_perp, p.axis_angle),
            "EI": ((1-p.gamma_global)*isotropic_gaussian(p.n, p.L, p.sigma_I) + p.gamma_global*uniform_kernel(p.n)),
            "IE": isotropic_gaussian(p.n, p.L, p.sigma_IE),
            "II": isotropic_gaussian(p.n, p.L, p.sigma_II)}

def convolve_periodic(field, kernel):
    return np.real(np.fft.ifft2(np.fft.fft2(field) * np.fft.fft2(np.fft.ifftshift(kernel))))

def make_Feff_lookup(p, h_min=-30.0, h_max=30.0, npts=6001):
    hs = np.linspace(h_min, h_max, npts)
    return hs, np.array([F_eff(h, p.phi_bar, p.sigma_phi, p.beta) for h in hs])

def F_eff_grid(h, lookup):
    return np.interp(h, lookup[0], lookup[1])

def integrate_field(p, op, I_E, I_I, stim_fn, dt, t_max, dE0=None):
    """2-D E-I field. Each connection: spatial conv (kernel) -> Erlang-delay+synaptic
    temporal chain (extra state grids) -> postsynaptic drive. Mirrors build_dispersion_matrix."""
    n = p.n; rE = np.full((n, n), op["r_E0"]); rI = np.full((n, n), op["r_I0"])
    a = np.full((n, n), op["r_E0"]); K = build_kernels(p); lut = make_Feff_lookup(p)
    rate_of = {"E": rE, "I": rI}
    if dE0 is not None: rE = rE + dE0
    specs = _connection_specs(p)
    chain = {tag: [np.full((n, n), 0.0) for _ in _chain_rates(p, ts)]   # init at 0 perturbation level
             for (_, _, _, _, _, ts, tag) in specs}
    # seed chains at steady convolved input so the fixed point holds
    for (post, pre, sign, J, _, ts, tag) in specs:
        steady_in = convolve_periodic(rate_of[pre], K[tag])
        for s in chain[tag]: s[:] = steady_in
    nsteps = int(round(t_max / dt)); rec = np.empty((nsteps, n, n))
    for t in range(nsteps):
        stim = stim_fn(t * dt)
        drive_E = I_E + (stim if np.ndim(stim) else stim) - p.b_a * a
        drive_I = I_I * np.ones((n, n))
        for (post, pre, sign, J, _, ts, tag) in specs:
            inp = convolve_periodic(rate_of[pre], K[tag]); rates = _chain_rates(p, ts)
            src = inp
            for j, rj in enumerate(rates):
                chain[tag][j] = chain[tag][j] + dt * rj * (src - chain[tag][j]); src = chain[tag][j]
            out = chain[tag][-1] if rates else inp
            (drive_E if post == "E" else drive_I)[...] += sign * J * out
        rE = rE + dt/p.tau_E * (-rE + F_eff_grid(drive_E, lut))
        rI = rI + dt/p.tau_I * (-rI + F_eff_grid(drive_I, lut))
        a = a + dt/p.tau_a * (-a + rE); rate_of["E"], rate_of["I"] = rE, rI
        rec[t] = rE
    return rec

def integrate_field_with_ic(p, op, I_E, I_I, stim_fn, dt, t_max, dE0):
    return integrate_field(p, op, I_E, I_I, stim_fn, dt, t_max, dE0=dE0)
