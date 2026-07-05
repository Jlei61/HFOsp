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


# ---------------------------------------------------------------------------
# M3A-A2 Abbott local+global regional inhibitory resource (off-by-default).
# A per-region generalization of z: q scales the inhibition E cells receive,
# drains with that region's firing, recovers toward 1. q_global scales ALL E
# (A1b global_ei_scale axis); q_core/q_L/q_R is the core-extra factor.
# See docs/archive/topic4/sef_hfo/m3a_a2_abbott_lg_dynamic_slowvars_spec_2026-06-25.md.
# ---------------------------------------------------------------------------
@dataclass
class RegionalResourceConfig:
    mode: str = "two_tank"        # 'core_only' | 'two_tank' | 'per_core'
    k_use: float = 0.0            # depletion rate (0 + init=1 -> frozen-full -> byte parity)
    tau_rec: float = 5000.0       # ms recovery toward full   (PLACEHOLDER)
    tau_a: float = 100.0          # ms activity EMA           (PLACEHOLDER)
    q_min: float = 0.25           # floor                     (PLACEHOLDER)
    frozen: bool = False          # Task-0b: hold q at init, no depletion (EMAs still tracked)
    q_core_init: float = 1.0
    q_global_init: float = 1.0
    # optional sAHP (g_K) recovery term: outward E-cell current that builds per spike + decays. The
    # OPPOSING slow process the symmetric-depletion screen lacked (§8.4). gk_max=0 -> off -> parity.
    gk_max: float = 0.0           # mV-equiv added to g_K per E spike  (PLACEHOLDER)
    tau_k: float = 5000.0         # ms g_K decay                       (PLACEHOLDER)


class RegionalResource:
    """Per-region inhibitory 'fuel tank' (A2 Abbott-LG). off-by-default: q=1 + k_use=0 == slow=None."""

    def __init__(self, N, V_th0, core_mask_E, cfg: RegionalResourceConfig | None = None,
                 NE=None, left_core_E=None, right_core_E=None):
        self.cfg = cfg or RegionalResourceConfig()
        self.N = int(N)
        core = np.asarray(core_mask_E, bool)
        self.NE = int(NE) if NE is not None else int(core.size)   # # E cells (E occupy [:NE])
        self.is_E = np.arange(self.N) < self.NE                   # E mask (no `labels` dependency)
        self.core_E_idx = np.flatnonzero(core)                    # core E indices (all < NE)
        self.left_idx = np.flatnonzero(np.asarray(left_core_E, bool)) if left_core_E is not None else None
        self.right_idx = np.flatnonzero(np.asarray(right_core_E, bool)) if right_core_E is not None else None
        self.q_core = float(self.cfg.q_core_init); self.q_global = float(self.cfg.q_global_init)
        self.q_L = float(self.cfg.q_core_init); self.q_R = float(self.cfg.q_core_init)
        self._ema_core = self._ema_global = self._ema_L = self._ema_R = 0.0
        self._I_I_last = np.zeros(N); self._alpha_a = None
        self.trace_core = []; self.trace_global = []
        self.trace_a_core = []; self.trace_a_global = []          # EMA activity (auditable a_bar)
        self.trace_L = []; self.trace_R = []                      # per_core q (stay 1.0 otherwise)
        self.gK = np.zeros(N)                                     # sAHP recovery (per E cell); 0 if gk_max=0
        self.trace_gk = []

    def apply_currents(self, I_E, I_I, labels=None):
        self._I_I_last = I_I
        scale = np.ones(self.N, dtype=float)
        scale[self.is_E] = self.q_global                          # global multiplier on ALL E
        if self.cfg.mode == "per_core" and self.left_idx is not None:
            scale[self.left_idx] *= self.q_L; scale[self.right_idx] *= self.q_R
        else:
            scale[self.core_E_idx] *= self.q_core                 # core-extra factor
        return I_E - scale * I_I - self.gK                        # I cells: scale=1, gK=0 -> I_E - I_I

    def threshold(self, V_th_base):
        return V_th_base                                          # A2 keeps the heterogeneous core field

    def _ode_step(self, q, a_ema, dt):
        q = q + dt * ((1.0 - q) / self.cfg.tau_rec - self.cfg.k_use * a_ema * q)
        return float(min(1.0, max(self.cfg.q_min, q)))

    def step(self, spk, labels, dt):
        if self._alpha_a is None:
            self._alpha_a = 1.0 - np.exp(-dt / self.cfg.tau_a)
        spk = np.asarray(spk, bool); e_mask = self.is_E; a = self._alpha_a
        def reg_frac(idx):
            return float(spk[idx].mean()) if idx.size else 0.0
        # activity EMAs ALWAYS updated (drive q + a_bar + auditable trace), independent of frozen
        self._ema_core += a * (reg_frac(self.core_E_idx) - self._ema_core)
        self._ema_global += a * (float(spk[e_mask].mean()) - self._ema_global)
        if self.left_idx is not None:
            self._ema_L += a * (reg_frac(self.left_idx) - self._ema_L)
            self._ema_R += a * (reg_frac(self.right_idx) - self._ema_R)
        if not self.cfg.frozen:                                   # frozen HOLDS q; dynamic updates it
            if self.cfg.mode == "per_core" and self.left_idx is not None:
                self.q_L = self._ode_step(self.q_L, self._ema_L, dt)
                self.q_R = self._ode_step(self.q_R, self._ema_R, dt)
                self.q_global = self._ode_step(self.q_global, self._ema_global, dt)
            else:
                self.q_core = self._ode_step(self.q_core, self._ema_core, dt)
                if self.cfg.mode == "two_tank":
                    self.q_global = self._ode_step(self.q_global, self._ema_global, dt)
        if self.cfg.gk_max > 0.0:                                 # sAHP recovery (always on, independent of frozen)
            self.gK += (-self.gK / self.cfg.tau_k) * dt
            self.gK[spk & e_mask] += self.cfg.gk_max
        self.trace_core.append(self.q_core); self.trace_global.append(self.q_global)
        self.trace_a_core.append(self._ema_core); self.trace_a_global.append(self._ema_global)
        self.trace_L.append(self.q_L); self.trace_R.append(self.q_R)
        self.trace_gk.append(float(self.gK[e_mask].mean()))
