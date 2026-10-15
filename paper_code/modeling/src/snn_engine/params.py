"""
Parameters for the spatially-structured E-I LIF network.

PAPER-EXACT values are taken from Table 1 of

    Bachschmid-Romano, Hatsopoulos & Brunel (2026),
    "A Spatially Structured Spiking Network Model of Beta Traveling Waves
     and Their Attenuation in Motor Cortex", bioRxiv 2026.03.18.712701.

Every biophysical parameter below (membrane, synaptic kinetics, delays,
weights, connectivity in-degrees, kernel widths, anisotropy, OU drive, LFP
proxy) is copied verbatim from Table 1 / Methods.

The ONLY fields that deviate from the paper are the geometry/size fields used
to make the smoke run fast (`L`, `density`, `T`). Their paper-scale values are
documented inline and available via `Params.paper()`. Reducing `density` only
changes the spatial *sampling* resolution; it does not change the mean-field
dynamics, which are set by the in-degrees C_ab (kept exact).

Units: time = ms, length = mm (Table 1 lists µm; converted here), voltage = mV,
rate = 1/ms (Hz = 1e-3 / ms).
"""
from __future__ import annotations
from paper_paths import CODE_ROOT, INPUT_ROOT, OUTPUT_ROOT
from dataclasses import dataclass
ALPHA = 2.065

@dataclass
class Params:
    V_th: float = 18.0
    V_reset: float = 11.0
    E_gaba: float = 11.0
    V_L: float = 0.0
    tau_m_E: float = 20.0
    tau_m_I: float = 10.0
    tau_ref_E: float = 2.0
    tau_ref_I: float = 1.0
    tau_r_AMPA: float = 0.7
    tau_d_AMPA: float = 3.5
    tau_r_GABA: float = 1.0
    tau_d_GABA: float = 18.0
    tau0: float = 0.1
    v_axon: float = 0.3
    g: float = 3.6
    w_EE: float = 0.1575
    w_IE: float = 0.2625
    J_ext_E: float = 0.455
    J_ext_I: float = 0.85
    C_EE: int = 800
    C_IE: int = 800
    C_EI: int = 200
    C_II: int = 200
    l_EE: float = 0.38
    l_IE: float = 0.25
    l_EI: float = 0.25
    l_II: float = 0.25
    rho_EE: float = 0.6
    rho_IE: float = 0.0
    rho_EI: float = 0.0
    rho_II: float = 0.0
    sigma_n: float = 3.3
    tau_n: float = 150.0
    nu_ext_ratio: float = 1.0
    rx: float = 0.13
    Rr: float = 0.278
    grid_spacing: float = 0.4
    grid_margin: float = 0.2
    L: float = 1.0
    density: float = 4000.0
    f_E: float = 0.8
    dt: float = 0.1
    T: float = 400.0
    delay_dt: float = 0.1
    seed: int = 1

    def weights(self):
        """Return (w_EE, w_IE, w_EI, w_II) in mV, Table-1 relations."""
        w_EI = 1.07 * self.g * self.w_EE
        w_II = self.g * self.w_IE
        return (self.w_EE, self.w_IE, w_EI, w_II)

    @classmethod
    def paper(cls, **overrides):
        """Paper-scale geometry (L=2 mm, density 28000 /mm^2, N~1.12e5).

        WARNING: ~1.1e5 neurons with ~1e8 synapses. Needs a spatial-bin
        connectivity sampler (see connectivity.py TODO) and is meant to be run
        offline, not in the smoke harness.
        """
        p = cls(L=2.0, density=28000.0, T=2000.0, **overrides)
        return p

def compute_nu_theta(p: Params):
    """External rate (1/ms) needed to reach threshold WITHOUT recurrent input,
    under colored synaptic noise. Eq 20-21 with τ_syn = τ_r^AMPA + τ_d^AMPA
    (only the external AMPA channel contributes when recurrence is absent).

    Returns (nu_theta_pop, nu_theta_E, nu_theta_I) in 1/ms.
    """
    import math
    tau_syn = p.tau_r_AMPA + p.tau_d_AMPA
    theta = p.V_th

    def nu_theta_a(J_ext, tau_m):
        A = 0.5 * ALPHA * J_ext * math.sqrt(tau_syn)
        num = A + math.sqrt(A * A + 4.0 * tau_m * J_ext * theta)
        den = 2.0 * tau_m * J_ext
        return (num / den) ** 2
    nE = nu_theta_a(p.J_ext_E, p.tau_m_E)
    nI = nu_theta_a(p.J_ext_I, p.tau_m_I)
    npop = 0.8 * nE + 0.2 * nI
    return (npop, nE, nI)
