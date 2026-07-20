"""Composite current-based Z/M slow variables plus the M4 recurrent-E divisive pool.

The adapter deliberately composes the two already-tested implementations instead of changing the
integration engine. ``cfg.use_SG`` activates the engine's existing recurrent-only current recorder.
Conductance membrane dynamics, global-GABA topology, qI/gK/hG, STD, shunting, and dynamic threshold
are outside this branch.

For E cells::

    I_net = I_E - z*I_I - eta_m*m
            - I_E_rec * (alpha_G*S_G)/(1 + alpha_G*S_G)

which is algebraically ``I_ff + I_rec/(1+alpha_G*S_G) - z*I_I - eta_m*m``.
I cells remain exactly ``I_E-I_I``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mz_slow_vars import MZSlowVars, MZSlowVarsConfig
from slow_field import SpatialSlowField, SpatialSlowFieldConfig


def slow_gate_drive(A_G: float, *, A0: float, A50: float, exponent: float) -> float:
    """High-state-only Hill drive for the independent slow recurrent-gain gate.

    The hard ``A0`` floor is load-bearing: ordinary IED activity below it must produce literal zero
    drive rather than a small tonic adaptation. ``A50`` is defined on the excess above ``A0``.
    """
    if A50 <= 0.0 or exponent <= 0.0:
        raise ValueError("A50 and exponent must be > 0")
    excess = max(float(A_G) - float(A0), 0.0)
    if excess == 0.0:
        return 0.0
    x = excess ** float(exponent)
    return float(x / (float(A50) ** float(exponent) + x))


@dataclass
class MZDivisivePoolConfig:
    # Existing per-E-cell MZ mechanism.
    use_z: bool = False
    use_m: bool = False
    tau_z: float = 5000.0
    I_th_EI: float = 0.0
    tau_adp: float = 2000.0
    eta_m: float = 0.0

    # Existing M4 dynamic pool. beta_SG is intentionally fixed at zero in this adapter.
    use_SG: bool = False
    alpha_G: float = 0.0
    r0_psi: float = 0.0
    r50_psi: float = 0.4
    n_psi: float = 2.0
    p_pool: float = 3.0
    tau_mu: float = 30.0
    tau_S: float = 80.0
    S_max: float = 1.0
    clamp_SG: float | None = None

    # Independent high-state-gated slow recurrent-E divisor (v2; OFF by default).
    use_TG: bool = False
    alpha_TG: float = 0.0
    AG0_TG: float = 0.15
    AG50_TG: float = 0.10
    n_TG: float = 4.0
    tau_TG: float = 750.0
    TG_max: float = 1.0
    clamp_TG: float | None = None

    # Sensor lattice only; qI/gK/hG/shunt remain hard-off below.
    n_grid: int = 32
    sigma_r: float = 0.5
    tau_sensor: float = 20.0
    tau_fast: float = 15.0

    def validate(self) -> None:
        for name, value in (
            ("tau_z", self.tau_z),
            ("tau_adp", self.tau_adp),
            ("r50_psi", self.r50_psi),
            ("n_psi", self.n_psi),
            ("tau_mu", self.tau_mu),
            ("tau_S", self.tau_S),
            ("S_max", self.S_max),
            ("AG50_TG", self.AG50_TG),
            ("n_TG", self.n_TG),
            ("tau_TG", self.tau_TG),
            ("TG_max", self.TG_max),
            ("tau_sensor", self.tau_sensor),
            ("tau_fast", self.tau_fast),
        ):
            if value <= 0.0:
                raise ValueError(f"{name} must be > 0, got {value}")
        if self.p_pool < 1.0:
            raise ValueError(f"p_pool must be >= 1, got {self.p_pool}")
        if self.alpha_G < 0.0:
            raise ValueError(f"alpha_G must be >= 0, got {self.alpha_G}")
        if self.alpha_TG < 0.0:
            raise ValueError(f"alpha_TG must be >= 0, got {self.alpha_TG}")
        if self.AG0_TG < 0.0:
            raise ValueError(f"AG0_TG must be >= 0, got {self.AG0_TG}")
        if self.eta_m < 0.0:
            raise ValueError(f"eta_m must be >= 0, got {self.eta_m}")
        if self.n_grid < 2:
            raise ValueError(f"n_grid must be >= 2, got {self.n_grid}")
        if self.clamp_SG is not None and not (0.0 <= self.clamp_SG <= self.S_max):
            raise ValueError(
                f"clamp_SG must be in [0,S_max], got {self.clamp_SG} with S_max={self.S_max}"
            )
        if self.use_TG and not self.use_SG:
            raise ValueError("use_TG requires use_SG so A_G is defined")
        if self.clamp_TG is not None and not (0.0 <= self.clamp_TG <= self.TG_max):
            raise ValueError(
                f"clamp_TG must be in [0,TG_max], got {self.clamp_TG} with TG_max={self.TG_max}"
            )


class MZDivisivePoolSlowVars:
    """Adapter implementing the ``simulate_kick`` slow protocol without engine edits."""

    def __init__(
        self,
        N,
        V_th0,
        posE,
        posI,
        L,
        cfg: MZDivisivePoolConfig | None = None,
        *,
        NE,
        core_mask_E=None,
        snapshot_steps=None,
    ):
        self.cfg = cfg or MZDivisivePoolConfig()
        self.cfg.validate()
        self.N = int(N)
        self.NE = int(NE)
        if self.NE != int(np.asarray(posE).shape[0]):
            raise ValueError("NE must equal len(posE)")
        if self.N - self.NE != int(np.asarray(posI).shape[0]):
            raise ValueError("N-NE must equal len(posI)")

        mz_cfg = MZSlowVarsConfig(
            use_z=self.cfg.use_z,
            use_m=self.cfg.use_m,
            tau_z=self.cfg.tau_z,
            I_th_EI=self.cfg.I_th_EI,
            tau_adp=self.cfg.tau_adp,
            eta_m=self.cfg.eta_m,
        )
        self.mz = MZSlowVars(
            self.N,
            V_th0,
            mz_cfg,
            NE=self.NE,
            core_mask_E=core_mask_E,
            snapshot_steps=snapshot_steps,
        )

        self.pool = None
        if self.cfg.use_SG:
            pool_cfg = SpatialSlowFieldConfig(
                n_grid=self.cfg.n_grid,
                sigma_r=self.cfg.sigma_r,
                tau_a=self.cfg.tau_sensor,
                use_qI=False,
                k_q=0.0,
                q_init=1.0,
                use_gK=False,
                k_K=0.0,
                use_hG=False,
                use_A=False,
                use_SG=True,
                alpha_G=self.cfg.alpha_G,
                beta_SG=0.0,
                r0_psi=self.cfg.r0_psi,
                r50_psi=self.cfg.r50_psi,
                n_psi=self.cfg.n_psi,
                p_pool=self.cfg.p_pool,
                tau_mu=self.cfg.tau_mu,
                tau_S=self.cfg.tau_S,
                S_max=self.cfg.S_max,
                clamp_SG=self.cfg.clamp_SG,
                tau_s=self.cfg.tau_fast,
            )
            self.pool = SpatialSlowField(
                self.N,
                V_th0,
                np.asarray(posE, float),
                np.asarray(posI, float),
                float(L),
                core_mask_E=core_mask_E,
                cfg=pool_cfg,
            )
        self.T_G = float(self.cfg.clamp_TG or 0.0)
        self.trace_TG = []
        self.trace_UTG = []

    def apply_currents(self, I_E, I_I, labels=None, I_E_rec=None):
        """Apply current-based Z/M, then remove only the divided fraction of recurrent E current."""
        out = self.mz.apply_currents(I_E, I_I, labels)
        if not self.cfg.use_SG:
            return out
        active_fast = self.cfg.alpha_G != 0.0
        active_slow = self.cfg.use_TG and self.cfg.alpha_TG != 0.0
        if not active_fast and not active_slow:
            # Literal neutral path: the pool remains an observer and the returned MZ array is untouched.
            if I_E_rec is not None:
                self.pool.trace_Irec_mean.append(float(np.asarray(I_E_rec)[: self.NE].mean()))
            return out
        if I_E_rec is None:
            raise RuntimeError(
                "active divisive pool requires I_E_rec; drive this adapter through simulate_kick"
            )
        aS = self.cfg.alpha_G * self.pool.S_G
        if active_slow:
            aS += self.cfg.alpha_TG * self.T_G
        frac = aS / (1.0 + aS)
        out[: self.NE] -= np.asarray(I_E_rec, float)[: self.NE] * frac
        self.pool.trace_Irec_mean.append(float(np.asarray(I_E_rec, float)[: self.NE].mean()))
        return out

    def threshold(self, V_th_base):
        return self.mz.threshold(V_th_base)

    def step(self, spk, labels, dt):
        self.mz.step(spk, labels, dt)
        if self.pool is not None:
            self.pool.step(spk, labels, dt)
        if self.cfg.use_TG:
            A_G = float(self.pool.trace_AG[-1])
            U_TG = slow_gate_drive(
                A_G,
                A0=self.cfg.AG0_TG,
                A50=self.cfg.AG50_TG,
                exponent=self.cfg.n_TG,
            )
            if self.cfg.clamp_TG is None:
                self.T_G += float(dt) * (-self.T_G + U_TG) / self.cfg.tau_TG
                self.T_G = float(np.clip(self.T_G, 0.0, self.cfg.TG_max))
            else:
                self.T_G = float(self.cfg.clamp_TG)
            self.trace_UTG.append(U_TG)
            self.trace_TG.append(self.T_G)

    # Explicit state/trace aliases keep runner code readable and make provenance unambiguous.
    @property
    def z(self):
        return self.mz.z

    @property
    def m(self):
        return self.mz.m

    @property
    def S_G(self):
        return 0.0 if self.pool is None else self.pool.S_G

    @property
    def mu_G(self):
        return 0.0 if self.pool is None else self.pool.mu_G

    @property
    def trace_z_mean(self):
        return self.mz.trace_z_mean

    @property
    def trace_z_min(self):
        return self.mz.trace_z_min

    @property
    def trace_m_mean(self):
        return self.mz.trace_m_mean

    @property
    def trace_adap_current(self):
        return self.mz.trace_adap_current

    @property
    def trace_AG(self):
        return [] if self.pool is None else self.pool.trace_AG

    @property
    def trace_SG(self):
        return [] if self.pool is None else self.pool.trace_SG

    @property
    def trace_muG(self):
        return [] if self.pool is None else self.pool.trace_muG

    @property
    def trace_rEfast_max(self):
        return [] if self.pool is None else self.pool.trace_rEfast_max
