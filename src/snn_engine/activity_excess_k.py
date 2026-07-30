"""Activity-excess K recruitment layer (FCXR-HYB1).

**This is NOT a full patient ion-concentration model.**  It carries one job from the HYB1
responsibility split: activity that exceeds the tissue's own registered interictal load raises
extracellular potassium, which raises recruitment.  There is no sodium, no pump (`eta_pump = 0`),
and no claim about absolute concentrations.

Plan of record: docs/superpowers/plans/2026-07-29-topic4-fcxr-hyb1.md section 2.

Why an EXCESS field rather than a re-centred K_o: the B2.1 sprint tried to reach a self-consistent
ionic baseline by ITERATION and did not converge at frozen bias.  Here `dK = 0` is a fixed point
because the source is exactly zero below the registered background envelope -- but the plan does
not take that on trust, and a measured baseline-preservation gate (plan 2.5) decides it.

Field (units mM, mM/s; identical source form to the accepted B2.1 K balance):

    d dK/dt = BETA * q_K * R_eps(s_v - b_v)  -  dK / tau_K  +  D_K * laplacian(dK)

with R_eps the smooth deadband positive part (exactly zero at or below background).  Reaching the
membrane, for E AND I cells alike:

    I_dK,i = g_dK * ( E_K(K_o0 + dK_v(i)) - E_K(K_o0) )
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.topic4_fcxr_ion import (                                            # noqa: E402
    BETA, D_K, E_K, K_I0, K_O0, E_K_0, diffusion_term,
)


class ExcessKSafetyError(FloatingPointError):
    """dK left its guard band.  A tripwire, never a clamp: a clamp would act as an undeclared
    saturating nonlinearity and could manufacture a bounded high state that the model does not
    actually produce."""


@dataclass
class ActivityExcessKConfig:
    """Everything is locked by the plan.  `g_dK`, `q_K`, `tau_K_s`, `D_K` and `Q_BG` are NOT swept
    in this sprint; a runner that changes one is out of contract."""
    b_v: np.ndarray                       # (n_voxel,) registered background upper envelope [Hz]
    eps: float                            # deadband softness [Hz] = 0.1 * median(b_v)
    q_K: float                            # mM per model spike (T7.1 f'=1.0 anchor)
    n_grid: int = 32
    dx_mm: float = 0.625
    dt_ion_ms: float = 0.5
    tau_K_s: float = 0.6546               # B2.1 measured tau_Ko at the working point
    D_K: float = D_K
    g_dK: float = 1.0                     # B2.1 anchor
    enabled: bool = True
    dk_bounds: tuple = (-1e-9, 12.0)      # mM; the low edge is a numerical tripwire, not a floor
    trace_stride: int = 2                 # ion blocks between recorded trace points
    snapshot_blocks: tuple = ()           # ion-block indices at which to keep a full dK map


def deadband_positive(u, eps):
    """R_eps(u) = u^2/(u+eps) for u > 0, exactly 0 otherwise.

    C1 at the origin (value and derivative both 0) so the source has no kink, and STRICTLY zero at
    or below background.  softplus is not acceptable here: it is positive everywhere, so it would
    pour a source into every interictal voxel at every step and destroy the fixed point.
    """
    u = np.asarray(u, float)
    if not (eps > 0):
        raise ValueError("eps must be > 0")
    return np.where(u > 0.0, u * u / (u + eps), 0.0)


def excess_source(load_hz, b_v, eps, q_K):
    """BETA * q_K * R_eps(load - background)   [mM/s], the same source form as the B2.1 K balance."""
    return BETA * float(q_K) * deadband_positive(np.asarray(load_hz, float)
                                                 - np.asarray(b_v, float), eps)


def d_dK_dt(dK, load_hz, cfg: ActivityExcessKConfig):
    """The whole right-hand side.  dK == 0 with load <= b_v returns EXACTLY 0.0, elementwise."""
    dK = np.asarray(dK, float)
    # b_v is stored flat (one entry per voxel); the field is a grid.  Ravel both so a caller may
    # pass the load in either layout without silently broadcasting the background across rows.
    src = excess_source(np.asarray(load_hz, float).ravel(), np.asarray(cfg.b_v, float).ravel(),
                        cfg.eps, cfg.q_K).reshape(dK.shape)
    return src - dK / float(cfg.tau_K_s) + diffusion_term(dK, dx_mm=cfg.dx_mm, D=cfg.D_K)


def membrane_current_from_dK(dK_per_cell, g_dK):
    """g_dK * (E_K(K_o0 + dK) - E_K(K_o0)), in engine drive units, per cell.  Zero when dK is 0."""
    return float(g_dK) * (E_K(K_O0 + np.asarray(dK_per_cell, float), K_I0) - E_K_0)


def cell_to_voxel(pos, L, n_grid):
    p = np.asarray(pos, float)
    ix = np.clip((p[:, 0] / L * n_grid).astype(int), 0, n_grid - 1)
    iy = np.clip((p[:, 1] / L * n_grid).astype(int), 0, n_grid - 1)
    return (iy * n_grid + ix).astype(np.int32)


def background_envelope(load_tv, q):
    """Per-voxel registered upper envelope from a sensor-only interictal trajectory (plan 2.1)."""
    a = np.asarray(load_tv, float)
    if a.ndim != 2:
        raise ValueError("load_tv must be (n_time, n_voxel)")
    return np.quantile(a, float(q), axis=0)


class ActivityExcessK:
    """Per-voxel dK on the same 32x32 grid as B2.1, integrated on its own sub-step."""

    def __init__(self, N, cell_voxel, cfg: ActivityExcessKConfig):
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
        self.dK_grid = np.zeros((cfg.n_grid, cfg.n_grid), float)
        self._counts = np.zeros(nv, float)
        self._acc_ms = 0.0
        self.n_updates = 0
        self.trace = []
        self.snapshots = {}
        self._cur = np.zeros(self.N, float)          # membrane current, refreshed per ion block
        self.duty_num = 0.0                          # running mean of 1[load > b_v] over (t, voxel)
        self.duty_den = 0.0
        self.dK_running_max = 0.0

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
        if self._acc_ms + 1e-12 >= self.cfg.dt_ion_ms:
            self.update(self._acc_ms)
            self._acc_ms = 0.0

    def update(self, block_ms=None):
        cfg = self.cfg
        blk_s = float(cfg.dt_ion_ms if block_ms is None else block_ms) * 1e-3
        load = np.zeros_like(self._counts)
        np.divide(self._counts, self.n_per_voxel * blk_s, out=load, where=self.occupied)
        self._counts[:] = 0.0

        act = load > np.asarray(cfg.b_v)
        self.duty_num += float(np.count_nonzero(act & self.occupied))
        self.duty_den += float(np.count_nonzero(self.occupied))

        flat = self.dK_grid.ravel()
        rhs = d_dK_dt(self.dK_grid, load.reshape(cfg.n_grid, cfg.n_grid), cfg).ravel()
        flat += blk_s * rhs
        lo, hi = cfg.dk_bounds
        if not np.all(np.isfinite(flat)) or flat.min() < lo or flat.max() > hi:
            raise ExcessKSafetyError(
                f"dK left [{lo}, {hi}] at block {self.n_updates}: "
                f"min={float(np.nanmin(flat)):.4g} max={float(np.nanmax(flat)):.4g}")
        self.dK_running_max = max(self.dK_running_max, float(flat.max()))

        self._cur = membrane_current_from_dK(flat[self.cell_voxel], cfg.g_dK) if cfg.enabled \
            else np.zeros(self.N, float)

        if cfg.trace_stride and self.n_updates % cfg.trace_stride == 0:
            self.trace.append((self.n_updates, float(flat.mean()), float(flat.max()),
                               int(np.count_nonzero(flat > 0.05))))
        if self.n_updates in cfg.snapshot_blocks:
            self.snapshots[self.n_updates] = flat.copy()
        self.n_updates += 1

    # ---------------------------------------------------------------- readout
    def duty_cycle(self):
        return (self.duty_num / self.duty_den) if self.duty_den > 0 else 0.0

    def state_dict(self):
        return dict(dK_grid=self.dK_grid.copy(), n_updates=self.n_updates,
                    counts=self._counts.copy(), acc_ms=self._acc_ms,
                    duty_num=self.duty_num, duty_den=self.duty_den,
                    dK_running_max=self.dK_running_max)

    def load_state_dict(self, s):
        self.dK_grid[:] = s["dK_grid"]
        self.n_updates = int(s["n_updates"])
        self._counts[:] = s["counts"]
        self._acc_ms = float(s["acc_ms"])
        self.duty_num = float(s["duty_num"])
        self.duty_den = float(s["duty_den"])
        self.dK_running_max = float(s["dK_running_max"])
        self._cur = membrane_current_from_dK(self.dK_grid.ravel()[self.cell_voxel],
                                             self.cfg.g_dK) if self.cfg.enabled \
            else np.zeros(self.N, float)


class ExcessKMZAdapter:
    """Add the excess-K current on top of an existing MZSlowVars without touching Z / X / M.

    Same `__getattr__` delegation as the accepted B2.1 adapter: absent attributes must STAY absent,
    because the engine guards branches with hasattr(slow, 'nE' / 'q_I' / 'uses_shunt') and
    synthesising them would silently move the run onto a different execution path.
    """

    def __init__(self, mz, dk: ActivityExcessK | None):
        object.__setattr__(self, "mz", mz)
        object.__setattr__(self, "dk", dk)

    def __getattr__(self, name):
        if name in ("mz", "dk"):
            raise AttributeError(name)
        return getattr(self.mz, name)

    def _active(self):
        return self.dk is not None and self.dk.cfg.enabled

    def membrane_terms(self, *a, **k):
        drive, g_rel, g_rev = self.mz.membrane_terms(*a, **k)
        if not self._active():
            return drive, g_rel, g_rev
        # a CURRENT, never a conductance: the engine discards a conductance for I cells, and the
        # plan requires E and I to both receive the excess-K term.
        return drive + self.dk.membrane_current(), g_rel, g_rev

    def apply_currents(self, *a, **k):
        I_net = self.mz.apply_currents(*a, **k)
        if not self._active():
            return I_net
        return I_net + self.dk.membrane_current()

    def step(self, spk, labels, dt):
        self.mz.step(spk, labels, dt)
        if self.dk is not None:
            self.dk.accumulate(spk)
            self.dk.maybe_update(dt)
