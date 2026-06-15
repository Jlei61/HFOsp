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
from dataclasses import dataclass

# sqrt(2) * |zeta(1/2)|  -- colored-noise threshold-shift constant (Eq. 17, 20)
ALPHA = 2.0650


@dataclass
class Params:
    # ---------------- Membrane & refractory (Table 1) ----------------
    V_th: float = 18.0       # mV   spike threshold (Vθ)
    V_reset: float = 11.0    # mV   reset (Vreset)
    V_L: float = 0.0         # mV   leak / rest (current-based Brunel form, Eq 3)
    tau_m_E: float = 20.0    # ms   excitatory membrane time constant
    tau_m_I: float = 10.0    # ms   inhibitory membrane time constant
    tau_ref_E: float = 2.0   # ms   excitatory refractory
    tau_ref_I: float = 1.0   # ms   inhibitory refractory

    # ------------- Synaptic kinetics (Table 1) -------------
    # delayed difference-of-exponentials (Eq 4-5); AMPA identical onto E and I
    tau_r_AMPA: float = 0.7  # ms   rise  (τ^AE_r = τ^AI_r)
    tau_d_AMPA: float = 3.5  # ms   decay (τ^AE_d = τ^AI_d)
    tau_r_GABA: float = 1.0  # ms   rise  (τ^G_r)
    tau_d_GABA: float = 18.0 # ms   decay (τ^G_d)

    # ---------------- Delays (Table 1, Eq 7) ----------------
    tau0: float = 0.1        # ms   distance-independent delay
    v_axon: float = 0.3      # mm/ms conduction speed (isotropic)

    # ------------- Synaptic weights (Table 1), mV -------------
    g: float = 3.6           # inhibitory gain (Fig 1-2 use Table default; Fig 4 uses 4.1)
    w_EE: float = 0.1575     # mV
    w_IE: float = 0.2625     # mV
    # w_EI = 1.07 * g * w_EE ; w_II = g * w_IE   (see weights())
    J_ext_E: float = 0.455   # mV   external -> E
    J_ext_I: float = 0.85    # mV   external -> I

    # ---------------- Connectivity (Table 1) ----------------
    # C_ab = average number of connections from population b onto a (in-degree)
    C_EE: int = 800
    C_IE: int = 800          # E -> I in-degree (I cells receive 800 E)
    C_EI: int = 200          # I -> E in-degree (E cells receive 200 I)
    C_II: int = 200
    l_EE: float = 0.380      # mm  (380 µm) spatial spread E->E
    l_IE: float = 0.250      # mm  E->I
    l_EI: float = 0.250      # mm  I->E
    l_II: float = 0.250      # mm  I->I
    rho_EE: float = 0.6      # E->E anisotropy (Table 1); long axis along (1,1) diagonal
    rho_IE: float = 0.0
    rho_EI: float = 0.0
    rho_II: float = 0.0

    # ------------- External input: OU process (Table 1, Figs 1-2; Eq 6) -------------
    sigma_n: float = 3.3     # Hz/sqrt(ms)  OU intensity of ν_ext fluctuation
    tau_n: float = 150.0     # ms           OU correlation time
    nu_ext_ratio: float = 1.0   # ν_signal / ν_θ  (operating point; SI/oscillatory band ~0.9-1.1)

    # ---------------- LFP proxy (Methods 4.2, Eq 9-11) ----------------
    rx: float = 0.130        # mm  near/far crossover of shape function
    Rr: float = 0.278        # mm  spatial summation cutoff (~1.85 rx)
    grid_spacing: float = 0.4   # mm  recording grid pitch (paper 400 µm)
    grid_margin: float = 0.2    # mm  margin from edges (paper 200 µm)

    # =========================================================
    #  Geometry / simulation  --  SMOKE defaults (NOT paper scale)
    #  paper: L=2.0 mm, density=28000 /mm^2  ->  N ~ 1.12e5
    # =========================================================
    L: float = 1.0           # mm   sheet side          [smoke; paper 2.0]
    density: float = 4000.0  # /mm^2 neuron density      [smoke; paper 28000]
    f_E: float = 0.8         # excitatory fraction (paper 0.8)
    dt: float = 0.1          # ms   integration step
    T: float = 400.0         # ms   simulated time       [smoke; lengthen for PSD]
    delay_dt: float = 0.1    # ms   delay quantization (>= dt). 0.1 = faithful
    seed: int = 1

    # ---------------- derived weights ----------------
    def weights(self):
        """Return (w_EE, w_IE, w_EI, w_II) in mV, Table-1 relations."""
        w_EI = 1.07 * self.g * self.w_EE
        w_II = self.g * self.w_IE
        return self.w_EE, self.w_IE, w_EI, w_II

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
    tau_syn = p.tau_r_AMPA + p.tau_d_AMPA          # 4.2 ms (Eq 18, external only)
    theta = p.V_th

    def nu_theta_a(J_ext, tau_m):
        A = 0.5 * ALPHA * J_ext * math.sqrt(tau_syn)
        num = A + math.sqrt(A * A + 4.0 * tau_m * J_ext * theta)
        den = 2.0 * tau_m * J_ext
        return (num / den) ** 2                      # 1/ms

    nE = nu_theta_a(p.J_ext_E, p.tau_m_E)
    nI = nu_theta_a(p.J_ext_I, p.tau_m_I)
    npop = 0.8 * nE + 0.2 * nI                       # Eq 21 population average
    return npop, nE, nI
