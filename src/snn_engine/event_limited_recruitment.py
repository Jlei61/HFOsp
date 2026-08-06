"""Event-limited recruitment layer (FCXR-HYB2).

**This is NOT extracellular potassium, NOT ion homeostasis, NOT a patient ion mechanism.**
`R_evt` is a phenomenological ACTUATOR abstracted from the B2.1 dynamic-K recruitment effect: it
carries one job from the HYB2 responsibility split -- within a single high-load event, widen who
gets recruited -- and it carries no concentration interpretation.  Its state has no mM units; the
B2.1 amplitude anchor is borrowed as a FORCE anchor only.

Plan of record: docs/superpowers/plans/2026-07-31-topic4-fcxr-hyb2.md section 2 (fully locked).

Why this replaces HYB1's activity-excess K: that layer carried recruitment on a 0.65 s
concentration memory, which is the same order as the interictal inter-event GAP, so potassium
never cleared between events and the floor ratcheted 4-6x over 8 s.  Here the memory is the EVENT
timescale (tens of ms), the amplitude is bounded by a tanh, and the state decays autonomously when
activity stops.  There is NO offline event label and NO scripted reset -- a reset would turn
recovery into a script action and destroy onset/termination attribution.

    e_v(t) = R_eps_s(s_v(t) - b_v)                        strictly 0 at or below background
    tau_R dq_v/dt = -q_v + e_v                            exact exponential update
    u_v      = R_eps_q(q_v - Q_on)                        strictly 0 at or below the interictal cap
    R_evt,v  = I_R_max * tanh(u_v / Q_scale)              bounded for ANY sustained input

Reaching the membrane, for E AND I cells alike, as an additive CURRENT (never a conductance: the
engine discards a conductance for I cells).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class ELRSafetyError(FloatingPointError):
    """q_v left its guard band.  A tripwire, never a clamp: a clamp is an undeclared saturating
    nonlinearity and could manufacture a bounded high state the model does not produce."""


@dataclass
class ELRConfig:
    """Every field is locked by plan section 2 / 4.  `tau_R_ms`, `Q_on`, `Q_scale` and `I_R_max`
    are NOT swept in this sprint; a runner that changes one is out of contract."""
    b_v: np.ndarray            # (n_voxel,) registered background upper envelope [Hz]
    eps_s: float               # deadband softness on the load    [Hz]  = 0.1 * median_v(b_v)
    tau_R_ms: float            # event-scale envelope memory      [ms]
    Q_on: float                # interictal cap on q_v            [Hz]  = 1.10 * calibration max
    Q_scale: float             # = Q_on (plan 4.3)                [Hz]
    eps_q: float               # deadband softness on q - Q_on    [Hz]  = 0.1 * Q_on
    I_R_max: float = 4.134151260609386      # B2.1 force anchor, engine-drive units
    n_grid: int = 32
    dt_R_ms: float = 0.5
    enabled: bool = True                    # False = sensor arm: state evolves, current is zeroed
    record_load: bool = False
    record_q_trace: bool = True
    q_bounds: tuple = (-1e-12, 1e6)


def deadband_positive(u, eps):
    """R_eps(u) = u^2/(u+eps) for u > 0, exactly 0 otherwise.  C1 at the origin.

    softplus is not acceptable: it is positive everywhere, so it would drive the envelope in every
    interictal voxel at every step and destroy the strict below-background zero.
    """
    u = np.asarray(u, float)
    if not (eps > 0):
        raise ValueError("eps must be > 0")
    out = np.zeros_like(u)
    pos = u > 0.0
    up = u[pos]
    out[pos] = up * up / (up + eps)
    return out


def envelope_step(q, e, *, dt_ms, tau_ms):
    """Exact solution of tau dq/dt = -q + e over one block with piecewise-constant e.

    Exact, not Euler: the block is 0.5 ms against a ~27 ms tau, and an exact update is
    unconditionally stable and removes dt as a hidden parameter.
    """
    if not (tau_ms > 0):
        raise ValueError("tau_R_ms must be > 0")
    a = float(np.exp(-float(dt_ms) / float(tau_ms)))
    return np.asarray(q, float) * a + np.asarray(e, float) * (1.0 - a)


def recruit_current(q, cfg: ELRConfig):
    """I_R_max * tanh(R_eps_q(q - Q_on) / Q_scale).  Zero at or below Q_on, bounded by I_R_max."""
    u = deadband_positive(np.asarray(q, float) - cfg.Q_on, cfg.eps_q)
    return cfg.I_R_max * np.tanh(u / cfg.Q_scale)


def cell_to_voxel(pos, L, n_grid):
    p = np.asarray(pos, float)
    ix = np.clip((p[:, 0] / L * n_grid).astype(int), 0, n_grid - 1)
    iy = np.clip((p[:, 1] / L * n_grid).astype(int), 0, n_grid - 1)
    return (iy * n_grid + ix).astype(np.int32)


class EventLimitedRecruitment:
    """Per-voxel event envelope q_v on the same 32x32 grid, with NO diffusion term.

    Without diffusion the actuator can only amplify tissue that is ALREADY active -- it cannot
    light up a silent voxel.  Recruitment may still widen, but through the recurrent scaffold (one
    synapse), not through a field.  That is what Gate A0 tests, and it is stated here so the code
    and the contract agree.
    """

    def __init__(self, N, cell_voxel, cfg: ELRConfig):
        self.cfg = cfg
        self.N = int(N)
        self.cell_voxel = np.asarray(cell_voxel, np.int32)
        if self.cell_voxel.shape != (self.N,):
            raise ValueError(f"cell_voxel must be ({self.N},), got {self.cell_voxel.shape}")
        nv = cfg.n_grid * cfg.n_grid
        if np.asarray(cfg.b_v).shape != (nv,):
            raise ValueError(f"b_v must be ({nv},), got {np.asarray(cfg.b_v).shape}")
        self.n_per_voxel = np.bincount(self.cell_voxel, minlength=nv).astype(float)
        self.occupied = self.n_per_voxel > 0
        self.q = np.zeros(nv, float)
        self._counts = np.zeros(nv, float)
        self._acc_ms = 0.0
        self.n_updates = 0
        self._cur = np.zeros(self.N, float)
        self.load_trace = [] if cfg.record_load else None
        self.q_trace = [] if cfg.record_q_trace else None
        self.active_num = 0.0                 # occupancy of R_evt > 0 over (t, occupied voxel)
        self.active_den = 0.0
        self.q_running_max = 0.0
        self.t_gate_block = None              # first block where max_v q_v > Q_on

    # ---------------------------------------------------------------- engine-facing
    def membrane_current(self):
        return self._cur

    def accumulate(self, spk):
        s = np.asarray(spk)
        if s.dtype != bool:
            s = s.astype(bool)
        np.add.at(self._counts, self.cell_voxel[s], 1.0)

    def maybe_update(self, dt_ms):
        self._acc_ms += float(dt_ms)
        if self._acc_ms + 1e-12 >= self.cfg.dt_R_ms:
            self.update(self._acc_ms)
            self._acc_ms = 0.0

    def update(self, block_ms=None):
        cfg = self.cfg
        blk = float(cfg.dt_R_ms if block_ms is None else block_ms)
        load = np.zeros_like(self._counts)
        np.divide(self._counts, self.n_per_voxel * (blk * 1e-3), out=load, where=self.occupied)
        self._counts[:] = 0.0
        if self.load_trace is not None:
            self.load_trace.append(load.astype(np.float32))

        e = deadband_positive(load - np.asarray(cfg.b_v, float), cfg.eps_s)
        e[~self.occupied] = 0.0                    # sampling gap: never a source
        self.q = envelope_step(self.q, e, dt_ms=blk, tau_ms=cfg.tau_R_ms)

        lo, hi = cfg.q_bounds
        if not np.all(np.isfinite(self.q)) or self.q.min() < lo or self.q.max() > hi:
            raise ELRSafetyError(
                f"q_v left [{lo}, {hi}] at block {self.n_updates}: "
                f"min={float(np.nanmin(self.q)):.4g} max={float(np.nanmax(self.q)):.4g}")

        occ = self.occupied
        qmax = float(self.q[occ].max()) if occ.any() else 0.0
        self.q_running_max = max(self.q_running_max, qmax)
        if self.t_gate_block is None and qmax > cfg.Q_on:
            self.t_gate_block = int(self.n_updates)

        r = recruit_current(self.q, cfg)
        self.active_num += float(np.count_nonzero(r[occ] > 0.0))
        self.active_den += float(np.count_nonzero(occ))
        # `enabled=False` is the OPEN arm: q_v keeps evolving (so t_gate is a counterfactual sensor
        # tracked identically in both arms), only the membrane current is zeroed.
        self._cur = r[self.cell_voxel] if cfg.enabled else np.zeros(self.N, float)

        if self.q_trace is not None:
            self.q_trace.append((self.n_updates, qmax, float(self.q[occ].mean()) if occ.any()
                                 else 0.0, int(np.count_nonzero(r[occ] > 0.0))))
        self.n_updates += 1

    # ---------------------------------------------------------------- readout
    def active_occupancy(self):
        return (self.active_num / self.active_den) if self.active_den > 0 else 0.0

    def t_gate_ms(self):
        return None if self.t_gate_block is None else self.t_gate_block * self.cfg.dt_R_ms

    def state_dict(self):
        return dict(q=self.q.copy(), n_updates=self.n_updates, counts=self._counts.copy(),
                    acc_ms=self._acc_ms, active_num=self.active_num, active_den=self.active_den,
                    q_running_max=self.q_running_max, t_gate_block=self.t_gate_block)

    def load_state_dict(self, s):
        self.q[:] = s["q"]
        self.n_updates = int(s["n_updates"])
        self._counts[:] = s["counts"]
        self._acc_ms = float(s["acc_ms"])
        self.active_num = float(s["active_num"])
        self.active_den = float(s["active_den"])
        self.q_running_max = float(s["q_running_max"])
        self.t_gate_block = s["t_gate_block"]
        r = recruit_current(self.q, self.cfg)
        self._cur = r[self.cell_voxel] if self.cfg.enabled else np.zeros(self.N, float)


class ELRMZAdapter:
    """Add the recruitment current on top of an existing MZSlowVars without touching Z / X / M.

    Same `__getattr__` delegation as the accepted B2.1 adapter: absent attributes must STAY absent,
    because the engine guards branches with hasattr(slow, 'nE' / 'q_I' / 'uses_shunt') and
    synthesising them would silently move the run onto a different execution path.
    """

    def __init__(self, mz, elr: EventLimitedRecruitment | None):
        object.__setattr__(self, "mz", mz)
        object.__setattr__(self, "elr", elr)

    def __getattr__(self, name):
        if name in ("mz", "elr"):
            raise AttributeError(name)
        return getattr(self.mz, name)

    def _active(self):
        return self.elr is not None and self.elr.cfg.enabled

    def membrane_terms(self, *a, **k):
        drive, g_rel, g_rev = self.mz.membrane_terms(*a, **k)
        if not self._active():
            return drive, g_rel, g_rev
        return drive + self.elr.membrane_current(), g_rel, g_rev

    def apply_currents(self, *a, **k):
        I_net = self.mz.apply_currents(*a, **k)
        if not self._active():
            return I_net
        return I_net + self.elr.membrane_current()

    def step(self, spk, labels, dt):
        self.mz.step(spk, labels, dt)
        if self.elr is not None:
            self.elr.accumulate(spk)
            self.elr.maybe_update(dt)
