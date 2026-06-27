"""M3A-v2 spatial slow-variable field (off-by-default; plugs into simulate_kick).

Spatial generalization of `slow_vars.RegionalResource` (§B2): the two SCALAR tanks
q_core/q_global become two spatial FIELDS on an n_grid x n_grid lattice over the
L x L sheet --

    q_I(x,t)  inhibitory-resource field   (depletes with local activity, refills to 1)
    g_K(x,t)  fatigue / sAHP field         (builds with local E activity, decays to 0)

so "axis fatigues while off-axis permissiveness rises" becomes representable, which a
two-scalar model cannot carry (§B5.0). Each neuron is coupled at its own position:

    I_net,i = I_E,i - q_I(x_i,t) * I_I,i - eta_K * g_K(x_i,t)        (i in E)
    I_net,i = I_E,i - I_I,i                                          (i in I)

OFF-BY-DEFAULT: k_q=0, k_K=0, q_init=1  =>  q_I==1, g_K==0  =>  apply_currents returns
I_E - I_I  =>  BYTE-PARITY with slow=None (the BASELINE_SHA regression).

Canonical math: docs/snn_core_model_equations.md §B5.
Plan / TDD:     docs/superpowers/plans/2026-06-28-sef-hfo-m3a-v2-spatial-slowvar-field-plan.md

STATUS: IMPLEMENTED (2026-06-28; Tasks 1-5 green). The default values in
SpatialSlowFieldConfig ARE the locked spec defaults (§B5.2-B5.3). Off-by-default
(k_q=0, k_K=0, q_init=1) leaves q_I==1, g_K==0 -> byte-identical to slow=None.
Mechanism SCREEN only: the four-state label is a detector, not a seizure claim;
calibration / ablation deferred (plan "Deferred").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.sef_hfo_field import isotropic_gaussian, convolve_periodic


# ---------------------------------------------------------------------------
# Config. Data only -- defaults are the §B5.2/§B5.3 locked spec values. The
# structural invariant sigma_q > sigma_K (wide disinhibition footprint, narrow
# fatigue footprint, §B5.3) and eta_I >= eta_E are enforced by validate().
# ---------------------------------------------------------------------------
@dataclass
class SpatialSlowFieldConfig:
    # ---- field grid ----
    n_grid: int = 32          # field lattice resolution per axis (n_grid x n_grid)
    # ---- firing-rate field K_r (spikes -> r_E, r_I), §B5.1 ----
    sigma_r: float = 0.5      # mm, spatial smoothing of spikes into the rate field
    tau_a: float = 100.0      # ms, temporal EMA of the rate field
    # ---- q_I(x,t) inhibitory-resource field, §B5.2 ----
    use_qI: bool = True
    tau_q: float = 5000.0     # ms, recovery toward q_I=1
    k_q: float = 0.0          # depletion rate; 0 -> OFF -> byte-parity
    q_min: float = 0.25       # floor
    q_init: float = 1.0       # initial / full-tank value
    sigma_q: float = 1.5      # mm, K_q width (WIDE; must be > sigma_K)
    eta_E: float = 0.3        # r_E weight in a_q
    eta_I: float = 1.0        # r_I weight in a_q (>= eta_E: resource tracks inhib usage)
    a0_q: float = 0.0         # saturation onset
    a50_q: float = 1.0        # saturation half-saturation
    # ---- g_K(x,t) fatigue / sAHP field, §B5.3 ----
    use_gK: bool = True
    tau_K: float = 5000.0     # ms, g_K decay
    k_K: float = 0.0          # build-rate STRENGTH knob; 0 -> OFF -> byte-parity
    gK_max: float = 1.0       # ceiling
    sigma_K: float = 0.5      # mm, K_K width (NARROW; must be < sigma_q)
    eta_K: float = 1.0        # coupling strength of g_K into the membrane
    a0_K: float = 0.0
    a50_K: float = 1.0

    def validate(self) -> None:
        """Raise ValueError on any breached structural invariant (§B5.2-B5.3):
        sigma_q > sigma_K, eta_I >= eta_E, q_min in [0,1], gK_max >= 0, n_grid >= 2.
        (q_min=0 is valid: full depletion; §B5.2 only requires q_min <= q_I <= 1.)"""
        if not (self.sigma_q > self.sigma_K):
            raise ValueError(f"sigma_q ({self.sigma_q}) must be > sigma_K ({self.sigma_K}) "
                             "(wide disinhibition footprint, narrow fatigue footprint; §B5.3)")
        if self.eta_I < self.eta_E:
            raise ValueError(f"eta_I ({self.eta_I}) must be >= eta_E ({self.eta_E}) (§B5.2)")
        if not (0.0 <= self.q_min <= 1.0):
            raise ValueError(f"q_min must be in [0, 1], got {self.q_min}")
        if self.gK_max < 0.0:
            raise ValueError(f"gK_max must be >= 0, got {self.gK_max}")
        if self.n_grid < 2:
            raise ValueError(f"n_grid must be >= 2, got {self.n_grid}")


# ---------------------------------------------------------------------------
# Stateless helpers (unit-testable in isolation), §B5.1-B5.2.
# ---------------------------------------------------------------------------
def saturation(a, a0, a50):
    """f(a) = [a-a0]_+ / (a50 + [a-a0]_+).  Hill-like; f(a0)=0, f(a0+a50)=0.5,
    f -> 1 as a -> inf. Elementwise on arrays. Implemented per the plan (Task 2)."""
    x = np.maximum(np.asarray(a, dtype=float) - a0, 0.0)   # [a - a0]_+
    return x / (a50 + x)


def aq_drive(rE, rI, eta_E, eta_I):
    """Pre-convolution q_I depletion drive: eta_E*r_E + eta_I*r_I (§B5.2). The inhibitory
    resource depletes mainly with inhibitory USE, so eta_I >= eta_E (config.validate enforces
    it). Factored out + unit-pinned so `step` cannot silently drop the r_I term and degrade
    into pure-E depletion. Implemented per the plan (Task 5)."""
    return eta_E * np.asarray(rE, float) + eta_I * np.asarray(rI, float)


def _grid_index(pos, L, n_grid):
    ix = np.clip((np.asarray(pos)[:, 0] / L * n_grid).astype(int), 0, n_grid - 1)
    iy = np.clip((np.asarray(pos)[:, 1] / L * n_grid).astype(int), 0, n_grid - 1)
    return ix, iy


def firing_rate_field(spk_bool, pos, L, n_grid, sigma):
    """One timestep's INSTANTANEOUS rate field (pre-EMA): bin the spikes `spk_bool`
    (1-D bool over a subpopulation) at continuous positions `pos` (n,2 mm) onto an
    n_grid x n_grid lattice over [0,L]^2, then convolve with an isotropic Gaussian of
    width `sigma` mm (periodic, via src.sef_hfo_field). Returns (n_grid, n_grid).
    Implemented per the plan (Task 3)."""
    counts = np.zeros((n_grid, n_grid))
    spk_bool = np.asarray(spk_bool, bool)
    if spk_bool.any():
        ix, iy = _grid_index(pos[spk_bool], L, n_grid)
        np.add.at(counts, (iy, ix), 1.0)                       # field[iy, ix]
    return convolve_periodic(counts, isotropic_gaussian(n_grid, L, sigma))


def sample_field_at(field, pos, L, n_grid):
    """Sample a (n_grid, n_grid) field at continuous positions `pos` (n,2 mm) on the
    L x L sheet (nearest-lattice). Returns (n,). Inverse of firing_rate_field's
    binning. Implemented per the plan (Task 3)."""
    ix, iy = _grid_index(pos, L, n_grid)
    return np.asarray(field)[iy, ix]


# ---------------------------------------------------------------------------
# The field object. Implements the simulate_kick `slow` protocol:
#   apply_currents(I_E, I_I, labels) -> I_net   (called each step, pre-membrane)
#   threshold(V_th_base)            -> V_th_eff (keeps the heterogeneous core)
#   step(spk, labels, dt)           -> None     (advances the fields, called post-spike)
# ---------------------------------------------------------------------------
class SpatialSlowField:
    """Spatial slow-variable field; off-by-default == slow=None (byte-parity)."""

    def __init__(self, N, V_th0, posE, posI, L, core_mask_E=None,
                 cfg: SpatialSlowFieldConfig | None = None):
        """Allocate q_I (== q_init) and g_K (== 0) on the n_grid lattice, store the E/I
        positions for binning + sampling, validate cfg. Implemented per the plan (Task 4)."""
        self.cfg = cfg or SpatialSlowFieldConfig()
        self.cfg.validate()
        self.N = int(N); self.nE = int(np.asarray(posE).shape[0]); self.L = float(L)
        self.posE = np.asarray(posE, float); self.posI = np.asarray(posI, float)
        n = self.cfg.n_grid
        self.q_I = np.full((n, n), self.cfg.q_init, dtype=float)
        self.g_K = np.zeros((n, n), dtype=float)
        self.rE = np.zeros((n, n)); self.rI = np.zeros((n, n))         # EMA rate fields
        self._Kq = isotropic_gaussian(n, L, self.cfg.sigma_q)
        self._Kk = isotropic_gaussian(n, L, self.cfg.sigma_K)
        self._ixE, self._iyE = _grid_index(self.posE, L, n)            # fixed E->grid map
        self._alpha_a = None
        self.trace_qI_mean = []; self.trace_gK_mean = []

    def apply_currents(self, I_E, I_I, labels=None):
        """I_net = I_E - q_I(x_i,t)*I_I - eta_K*g_K(x_i,t) for E cells; I_E - I_I for I
        cells. q_I==1, g_K==0 -> returns I_E - I_I exactly (parity). Task 4."""
        qI_E = self.q_I[self._iyE, self._ixE]                          # (nE,)
        gK_E = self.g_K[self._iyE, self._ixE]
        out = np.asarray(I_E, float) - np.asarray(I_I, float)          # I cells: I_E - I_I
        nE = self.nE
        out[:nE] = I_E[:nE] - qI_E * I_I[:nE] - self.cfg.eta_K * gK_E  # E cells
        return out

    def threshold(self, V_th_base):
        """Protocol passthrough: v2 rides the heterogeneous core threshold unchanged
        (like RegionalResource.threshold)."""
        return V_th_base

    def step(self, spk, labels, dt):
        """Advance the fields one dt: (1) EMA-update r_E,r_I from this step's spikes,
        (2) form a_q = K_q*aq_drive(r_E,r_I,eta_E,eta_I), a_K = K_K*r_E, (3) integrate the
        q_I ODE (depletion ~ k_q*f*q_I) and the BOUNDED-build g_K ODE
        (build ~ k_K*f*(gK_max-g_K)) on the lattice (bounds [q_min,1] and [0,gK_max]).
        q_I/g_K are read directly in apply_currents (no per-neuron cache). Task 5."""
        cfg = self.cfg
        spk = np.asarray(spk, bool)
        rE_inst = firing_rate_field(spk[:self.nE], self.posE, self.L, cfg.n_grid, cfg.sigma_r)
        rI_inst = firing_rate_field(spk[self.nE:], self.posI, self.L, cfg.n_grid, cfg.sigma_r)
        if self._alpha_a is None:
            self._alpha_a = 1.0 - np.exp(-dt / cfg.tau_a)
        a = self._alpha_a
        self.rE += a * (rE_inst - self.rE)                            # EMA (§B5.1)
        self.rI += a * (rI_inst - self.rI)
        if cfg.use_qI and cfg.k_q != 0.0:                            # §B5.2 (depletion ~ k_q*f*q_I)
            a_q = convolve_periodic(aq_drive(self.rE, self.rI, cfg.eta_E, cfg.eta_I), self._Kq)
            fq = saturation(a_q, cfg.a0_q, cfg.a50_q)
            self.q_I += dt * ((1.0 - self.q_I) / cfg.tau_q - cfg.k_q * fq * self.q_I)
            np.clip(self.q_I, cfg.q_min, 1.0, out=self.q_I)
        if cfg.use_gK and cfg.k_K != 0.0:                            # §B5.3 BOUNDED build (k_K is the knob)
            a_K = convolve_periodic(self.rE, self._Kk)
            fk = saturation(a_K, cfg.a0_K, cfg.a50_K)
            self.g_K += dt * (-self.g_K / cfg.tau_K + cfg.k_K * fk * (cfg.gK_max - self.g_K))
            np.clip(self.g_K, 0.0, cfg.gK_max, out=self.g_K)
        self.trace_qI_mean.append(float(self.q_I.mean()))
        self.trace_gK_mean.append(float(self.g_K.mean()))
