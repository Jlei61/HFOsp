"""Constitutive Na/K ion homeostasis for the FCXR substrate (spec §4, §6, §13; rev4).

Two state variables are added on top of the accepted FCXR arm-C substrate:

    Na_i     per cell   (E AND I)          slow negative feedback via the pump
    K_o      per voxel  (32x32 by default) fast local positive feedback via E_K

and the pump that couples them.  Every flux is written in DEVIATION form, so the no-spike state
(Na_i, K_o) = (18, 4) is an exact fixed point structurally -- including on an EMPTY voxel, which
is a sampling gap in a sub-sampled sheet, not a tissue-free region (spec §4.2).

The layer reaches the membrane ONLY as an additive current, for E and I alike: the engine silently
discards g_rel/g_rev for I cells (spec §5), so a conductance-shaped potassium term would produce an
E-only mechanism while being reported as E/I.

eta_pump is LOCKED to 0 for B0-B2: only the potassium-mediated pathway (pump -> K recovery -> E_K
-> excitability) is under test.  The electrogenic pump current is deferred to B4.

The six blessed engine files are NOT touched; the layer attaches by WRAPPING MZSlowVars.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.dirname(_SRC)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.topic4_fcxr_ion import (  # noqa: E402
    BETA, EPS, I_GLIA_0, I_PUMP_0, K_O0, K_O_INF, NA_I0, E_K_0,
    E_K, K_i_from_Na_i, bath_clearance, diffusion_term, glia_uptake, pump_flux,
    heterogeneous_steady_state,
)


@dataclass
class IonHomeostasisConfig:
    """All spatial / temporal / coupling constants of the ion layer.

    `enabled=False` is the parity switch: the adapter then touches no membrane term and integrates
    no ion state, so the whole engine is byte-identical to bare MZSlowVars.
    """
    q_ion: float                       # mM per model spike (derived from f', spec §3.2)
    n_grid: int = 32
    dx_mm: float = 0.625
    dt_ion_ms: float = 0.5
    g_K_ion: float = 1.0               # effective reference normalization (spec §4.3), NOT a unit result
    eta_pump: float = 0.0              # LOCKED 0 in B0-B2
    I_bias_E: float = 0.0              # the only two nuisance parameters (spec §7.2)
    I_bias_I: float = 0.0
    enabled: bool = True
    na_bounds: tuple = (1.0, 80.0)     # fail-fast guards, NEVER saturators
    ko_bounds: tuple = (0.2, 80.0)
    record_trace: bool = False
    k_trace_stride: int = 0            # >0: keep the K_o grid every N ion blocks (T7 measurement)
    na_snapshot_blocks: tuple = ()     # ion-block indices at which to keep the full Na_i field
    capture_counts_range: tuple = ()   # (b0, b1): keep per-cell spike counts for these ion blocks

    def __post_init__(self):
        if self.q_ion <= 0.0:
            raise ValueError("q_ion must be > 0")
        if float(self.eta_pump) != 0.0:
            raise ValueError("eta_pump is locked to 0 in B0-B2 (spec §4.3 rev3)")


def cell_to_voxel(pos, L, n_grid):
    p = np.asarray(pos, float)
    ix = np.clip((p[:, 0] / L * n_grid).astype(int), 0, n_grid - 1)
    iy = np.clip((p[:, 1] / L * n_grid).astype(int), 0, n_grid - 1)
    return (iy * n_grid + ix).astype(np.int32)


class IonSafetyError(FloatingPointError):
    """Raised when a concentration leaves its guard band.  The band is a FAIL-FAST tripwire; it is
    never applied as a clamp, because a clamp would act as an undeclared saturating nonlinearity."""


class IonHomeostasis:
    """Per-cell Na_i + per-voxel K_o with a constitutive pump, integrated on its own sub-step."""

    def __init__(self, N, NE, cell_voxel, cfg: IonHomeostasisConfig, *, Na_init, K_o_init):
        self.N, self.NE = int(N), int(NE)
        self.cfg = cfg
        self.nv = cfg.n_grid * cfg.n_grid
        self.cell_voxel = np.asarray(cell_voxel, np.int32)
        if self.cell_voxel.shape != (self.N,):
            raise ValueError(f"cell_voxel must have shape ({self.N},)")
        self.n_per_grid = np.bincount(self.cell_voxel, minlength=self.nv).astype(np.int64)
        self.Na_i_all = np.array(Na_init, float)
        if self.Na_i_all.shape != (self.N,):
            raise ValueError(f"Na_init must have shape ({self.N},)")
        self.K_o_grid = np.array(K_o_init, float).reshape(cfg.n_grid, cfg.n_grid)
        self._cell_spikes = np.zeros(self.N, np.int32)
        self._step_i = 0
        self.n_updates = 0
        self.grid_spikes_E = np.zeros(self.nv, np.int64)   # cumulative, diagnostics only
        self.grid_spikes_I = np.zeros(self.nv, np.int64)
        self.trace = [] if cfg.record_trace else None
        self.k_trace = [] if cfg.k_trace_stride else None
        self.k_trace_blocks = []
        self.na_snapshots = {}
        self._na_snapshot_at = set(int(b) for b in cfg.na_snapshot_blocks)
        self.captured_counts = [] if cfg.capture_counts_range else None
        self._refresh_membrane_state()

    # ------------------------------------------------------------------ derived membrane state
    def _refresh_membrane_state(self):
        """E_K / I_pump seen by the membrane for the WHOLE of the next block (causal: a block's
        spikes only reach the membrane through the state produced at the end of that block)."""
        K_cell = self.K_o_grid.ravel()[self.cell_voxel]
        self.pump_flux_all = pump_flux(self.Na_i_all, K_cell)
        self.E_K_all = E_K(K_cell, K_i_from_Na_i(self.Na_i_all))

    def membrane_current(self):
        """g_K_ion*(E_K - E_K_0) - eta_pump*(I_pump - I_pump_0), in engine drive units, per cell."""
        cur = self.cfg.g_K_ion * (self.E_K_all - E_K_0)
        if self.cfg.eta_pump:
            cur = cur - self.cfg.eta_pump * (self.pump_flux_all - I_PUMP_0)
        return cur

    # ------------------------------------------------------------------ integration
    def accumulate(self, spk):
        self._cell_spikes += np.asarray(spk, bool)

    def steps_per_block(self, dt_ms):
        n = int(round(self.cfg.dt_ion_ms / float(dt_ms)))
        if n < 1:
            raise ValueError(f"dt_ion_ms={self.cfg.dt_ion_ms} is smaller than the engine dt={dt_ms}")
        return n

    def maybe_update(self, dt_ms):
        self._step_i += 1
        if self._step_i % self.steps_per_block(dt_ms) == 0:
            self.update()

    def update(self):
        """One ion block, explicit deviation-form Euler on the block-start pump values."""
        cfg = self.cfg
        dt_s = cfg.dt_ion_ms * 1e-3
        counts = self._cell_spikes.astype(float)
        Ip = self.pump_flux_all
        if self.captured_counts is not None:
            b0, b1 = cfg.capture_counts_range
            if b0 <= self.n_updates < b1:
                self.captured_counts.append(self._cell_spikes.astype(np.int8).copy())

        occ = self.n_per_grid > 0
        n_safe = np.where(occ, self.n_per_grid, 1).astype(float)
        spk_vox = np.bincount(self.cell_voxel, weights=counts, minlength=self.nv)
        Ip_sum = np.bincount(self.cell_voxel, weights=Ip, minlength=self.nv)
        Ip_bar = np.where(occ, Ip_sum / n_safe, I_PUMP_0)            # empty voxel -> resting tissue
        src = BETA * cfg.q_ion * np.where(occ, spk_vox / n_safe, 0.0)

        K = self.K_o_grid
        dK = (-2.0 * BETA * (Ip_bar.reshape(K.shape) - I_PUMP_0)
              - bath_clearance(K) - (glia_uptake(K) - I_GLIA_0)
              + diffusion_term(K, dx_mm=cfg.dx_mm))
        # per-spike increments are NOT multiplied by time; continuous fluxes are (spec §4.2b)
        self.Na_i_all = self.Na_i_all + cfg.q_ion * counts - dt_s * 3.0 * (Ip - I_PUMP_0)
        self.K_o_grid = K + src.reshape(K.shape) + dt_s * dK

        self.grid_spikes_E += np.bincount(self.cell_voxel[:self.NE],
                                          weights=counts[:self.NE], minlength=self.nv).astype(np.int64)
        self.grid_spikes_I += np.bincount(self.cell_voxel[self.NE:],
                                          weights=counts[self.NE:], minlength=self.nv).astype(np.int64)
        self._cell_spikes[:] = 0
        self.n_updates += 1
        self._check_bounds()
        self._refresh_membrane_state()
        if self.trace is not None:
            self.trace.append(dict(n=self.n_updates, Na_mean=float(self.Na_i_all.mean()),
                                   K_mean=float(self.K_o_grid.mean()),
                                   K_max=float(self.K_o_grid.max()),
                                   pump_mean=float(self.pump_flux_all.mean())))
        if self.k_trace is not None and self.n_updates % self.cfg.k_trace_stride == 0:
            self.k_trace.append(self.K_o_grid.astype(np.float32).copy())
            self.k_trace_blocks.append(self.n_updates)
        if self.n_updates in self._na_snapshot_at:
            self.na_snapshots[self.n_updates] = self.Na_i_all.astype(np.float32).copy()

    def replay_block(self, counts):
        """Advance one ion block from a STORED per-cell spike count vector.

        In sensor mode (g_K_ion = 0) the ion state is a pure function of the raster, so a recorded
        event can be replayed / superposed offline.  This is the same device the accepted pump
        sprint used for its sensor-only load calibration."""
        self._cell_spikes[:] = np.asarray(counts, np.int32)
        self.update()

    def _check_bounds(self):
        lo, hi = self.cfg.na_bounds
        klo, khi = self.cfg.ko_bounds
        if not (np.all(np.isfinite(self.Na_i_all)) and np.all(np.isfinite(self.K_o_grid))):
            raise IonSafetyError("non-finite ion state")
        if self.Na_i_all.min() < lo or self.Na_i_all.max() > hi:
            raise IonSafetyError(f"Na_i left the guard band [{lo}, {hi}]: "
                                 f"min={self.Na_i_all.min():.4g} max={self.Na_i_all.max():.4g}")
        if self.K_o_grid.min() < klo or self.K_o_grid.max() > khi:
            raise IonSafetyError(f"K_o left the guard band [{klo}, {khi}]: "
                                 f"min={self.K_o_grid.min():.4g} max={self.K_o_grid.max():.4g}")

    # ------------------------------------------------------------------ diagnostics / checkpoint
    def derivatives(self, rates_hz):
        """Instantaneous (dNa/dt per cell, dK_o/dt per voxel) at the current state for a frozen
        per-cell rate field -- the residual Gate H judges (spec §4.2c: q95/q99/max, never the mean)."""
        K_cell = self.K_o_grid.ravel()[self.cell_voxel]
        Ip = pump_flux(self.Na_i_all, K_cell)
        dNa = self.cfg.q_ion * np.asarray(rates_hz, float) - 3.0 * (Ip - I_PUMP_0)
        occ = self.n_per_grid > 0
        n_safe = np.where(occ, self.n_per_grid, 1).astype(float)
        r_bar = np.bincount(self.cell_voxel, weights=np.asarray(rates_hz, float),
                            minlength=self.nv) / n_safe
        Ip_bar = np.where(occ, np.bincount(self.cell_voxel, weights=Ip, minlength=self.nv) / n_safe,
                          I_PUMP_0)
        K = self.K_o_grid
        dK = (BETA * self.cfg.q_ion * np.where(occ, r_bar, 0.0).reshape(K.shape)
              - 2.0 * BETA * (Ip_bar.reshape(K.shape) - I_PUMP_0)
              - bath_clearance(K) - (glia_uptake(K) - I_GLIA_0)
              + diffusion_term(K, dx_mm=self.cfg.dx_mm))
        return dNa, dK

    def total_extracellular_K(self):
        """Total extracellular K content, grid-invariant: sum(K_o,g) * voxel area."""
        return float(self.K_o_grid.sum()) * (self.cfg.dx_mm ** 2)

    def state_dict(self):
        return dict(Na_i_all=self.Na_i_all.copy(), K_o_grid=self.K_o_grid.copy(),
                    cell_spikes=self._cell_spikes.copy(), step_i=self._step_i,
                    n_updates=self.n_updates, grid_spikes_E=self.grid_spikes_E.copy(),
                    grid_spikes_I=self.grid_spikes_I.copy())

    def load_state_dict(self, sd):
        self.Na_i_all = np.array(sd["Na_i_all"], float)
        self.K_o_grid = np.array(sd["K_o_grid"], float)
        self._cell_spikes = np.array(sd["cell_spikes"], np.int32)
        self._step_i = int(sd["step_i"])
        self.n_updates = int(sd["n_updates"])
        self.grid_spikes_E = np.array(sd["grid_spikes_E"], np.int64)
        self.grid_spikes_I = np.array(sd["grid_spikes_I"], np.int64)
        self._refresh_membrane_state()


def build_from_rate_field(N, NE, cell_voxel, cfg, rate_E, rate_I, *, return_report=False):
    """Initial state = the HETEROGENEOUS analytic pre-equilibrium of this network's own frozen
    per-cell rate field (spec §4.2c).  A single global-rate scalar steady state is NOT acceptable:
    tau_Na = 54.4 s, so an 11 s run cannot expose the slow spatial re-arrangement it would leave."""
    cell_voxel = np.asarray(cell_voxel, np.int32)
    rep = heterogeneous_steady_state(rate_E, rate_I, cell_voxel[:NE], cell_voxel[NE:],
                                     n_grid=cfg.n_grid, q_ion=cfg.q_ion, dx_mm=cfg.dx_mm)
    ions = IonHomeostasis(N, NE, cell_voxel, cfg,
                          Na_init=rep["Na_star"], K_o_init=rep["K_o_star"])
    return (ions, rep) if return_report else ions


def resting_state(N, NE, cell_voxel, cfg):
    """Uniform resting initial state -- for the fixed-point / parity tests only."""
    return IonHomeostasis(N, NE, cell_voxel, cfg,
                          Na_init=np.full(N, NA_I0),
                          K_o_init=np.full((cfg.n_grid, cfg.n_grid), K_O0))


class IonHomeostaticMZAdapter:
    """Wrap an existing MZSlowVars: keep Z and the FCXR conductance path byte-identical, and only
    ADD the ion current on top of its result (spec §6).

    Delegation uses ``__getattr__`` rather than a whitelist so that ABSENT attributes stay absent:
    the engine guards several branches with hasattr(slow, 'nE' / 'q_I' / 'uses_shunt'), and
    synthesising them would silently flip the engine onto a different execution path.
    """

    def __init__(self, mz, ions: IonHomeostasis | None):
        object.__setattr__(self, "mz", mz)
        object.__setattr__(self, "ions", ions)
        object.__setattr__(self, "_bias", None)

    def __getattr__(self, name):
        if name in ("mz", "ions", "_bias"):
            raise AttributeError(name)                     # pre-init guard, never delegate these
        return getattr(self.mz, name)

    # ------------------------------------------------------------------ internals
    def _active(self):
        return self.ions is not None and self.ions.cfg.enabled

    def _bias_vector(self, n):
        if self._bias is None or self._bias.shape != (n,):
            c = self.ions.cfg
            b = np.empty(n, float)
            b[:self.mz.NE] = c.I_bias_E
            b[self.mz.NE:] = c.I_bias_I
            object.__setattr__(self, "_bias", b)
        return self._bias

    def _ion_drive(self, n):
        """The additive term for E AND I: the two nuisance biases plus the ion membrane current,
        evaluated on the PREVIOUS ion block's state."""
        return self._bias_vector(n) + self.ions.membrane_current()

    # ------------------------------------------------------------------ engine protocol
    def membrane_terms(self, *a, **k):
        drive, g_rel, g_rev = self.mz.membrane_terms(*a, **k)
        if not self._active():
            return drive, g_rel, g_rev
        # g_rel / g_rev are not touched by a single byte: the ion layer is a CURRENT, never a
        # conductance (the engine would discard a conductance for I cells -- spec §5).
        return drive + self._ion_drive(drive.shape[0]), g_rel, g_rev

    def apply_currents(self, *a, **k):
        I_net = self.mz.apply_currents(*a, **k)
        if not self._active():
            return I_net
        # symmetric with membrane_terms: same current, so a current-membrane config cannot
        # silently lose the ion layer.
        return I_net + self._ion_drive(np.asarray(I_net).shape[0])

    def step(self, spk, labels, dt):
        self.mz.step(spk, labels, dt)                      # existing Z/M/X order and values intact
        if self._active():
            self.ions.accumulate(spk)
            self.ions.maybe_update(dt)
