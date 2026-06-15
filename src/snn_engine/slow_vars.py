"""
Page-4 epilepsy slow-variable layer  (Zou & Lei deck, 2026-06-01).

This is the EXTENSION CONTRACT for turning the Brunel wave engine into the
seizure model on slide 4. It is OFF by default (`simulate(..., slow=None)`).
The three slow variables, with the deck's equations:

  disinhibition z (Cl- / STP, ~5 s):
      tau_z dz/dt = z_inf - z ,   z_inf = H(g_th - g^I)
      i.e. when inhibitory drive g^I exceeds g_th, z -> 0  => inhibition weakens.
      Enters the membrane as  -(z g^I / f_max)(V - E_I).
      Here (current-based engine) we use I_I as the proxy for g^I and write
          I_net = I_E - z * I_I .

  adaptive threshold phi (adaptation, ~100 ms):
      dphi/dt = -(phi - phi0)/tau_phi + dphi * S
      replaces the fixed spike threshold V_th per neuron.

  sAHP g_K  (K+-mediated, ~5 s):
      dg_K/dt = -g_K/tau_K + g_Kmax * S
      outward current -(g_K/f_max)(V - E_K); here subtracted as an outward term
          I_net = I_E - z * I_I - g_K .

!!!  PARAMETER VALUES BELOW ARE PLACEHOLDERS  !!!
The deck gives no Table for these. Defaults are order-of-magnitude only (the
timescales 100 ms / 5 s are from the deck text). Calibrate before drawing any
conclusion. The wave-engine smoke run does NOT use this file.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class SlowVarsConfig:
    use_z: bool = True
    use_phi: bool = True
    use_gK: bool = True
    # disinhibition z  (PLACEHOLDER params)
    tau_z: float = 5000.0     # ms (~5 s, deck)
    g_th: float = 6.0         # mV-equivalent threshold on I_I  (PLACEHOLDER)
    # adaptive threshold phi  (PLACEHOLDER)
    tau_phi: float = 100.0    # ms (~100 ms, deck)
    dphi: float = 1.0         # mV per spike (PLACEHOLDER)
    # sAHP g_K  (PLACEHOLDER)
    tau_K: float = 5000.0     # ms (~5 s, deck)
    gK_max: float = 0.2       # mV per spike (PLACEHOLDER)


class SlowVars:
    """Per-neuron slow state. Instantiate and pass to simulate(slow=...)."""

    def __init__(self, N, V_th0, cfg: SlowVarsConfig | None = None):
        self.cfg = cfg or SlowVarsConfig()
        self.z = np.ones(N)                       # full inhibition initially
        self.phi = np.full(N, float(V_th0))       # phi0 = base threshold
        self.phi0 = float(V_th0)
        self.gK = np.zeros(N)
        self._I_I_last = np.zeros(N)

    def apply_currents(self, I_E, I_I, labels):
        self._I_I_last = I_I
        I_net = I_E.copy()
        I_net -= (self.z * I_I) if self.cfg.use_z else I_I
        if self.cfg.use_gK:
            I_net -= self.gK
        return I_net

    def threshold(self, V_th_base):
        return self.phi if self.cfg.use_phi else V_th_base

    def step(self, spk, labels, dt):
        c = self.cfg
        if c.use_z:
            z_inf = (c.g_th - self._I_I_last > 0.0).astype(np.float64)
            self.z += (dt / c.tau_z) * (z_inf - self.z)
        if c.use_phi:
            self.phi += (-(self.phi - self.phi0) / c.tau_phi) * dt
            self.phi[spk] += c.dphi
        if c.use_gK:
            self.gK += (-self.gK / c.tau_K) * dt
            self.gK[spk] += c.gK_max
