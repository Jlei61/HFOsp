"""Observation-invariant dynamic accessibility for Topic 4 rev10-D."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.snn_engine.slow_field import (
    _grid_index,
    aq_drive,
    convolve_periodic,
    firing_rate_field,
    isotropic_gaussian,
    saturation,
)


@dataclass(frozen=True)
class AdaptationConfig:
    mode: str
    tau_ms: float
    increment_mV: float
    trace_dt_ms: float = 10.0

    def validate(self, dt_ms: float) -> None:
        if self.mode not in {"local", "global"}:
            raise ValueError("adaptation mode must be local or global")
        if not np.isfinite(self.tau_ms) or self.tau_ms <= 0.0:
            raise ValueError("tau_ms must be positive")
        if not np.isfinite(self.increment_mV) or self.increment_mV < 0.0:
            raise ValueError("increment_mV must be non-negative")
        if not np.isfinite(self.trace_dt_ms) or self.trace_dt_ms < dt_ms:
            raise ValueError("trace_dt_ms must be at least one engine step")


class SpikeTriggeredAdaptation:
    """E-only subtractive adaptation with local or mean-matched global state."""

    def __init__(self, n_total: int, n_e: int, dt_ms: float,
                 cfg: AdaptationConfig):
        cfg.validate(dt_ms)
        if not 0 < int(n_e) <= int(n_total):
            raise ValueError("n_e must define a non-empty prefix of all neurons")
        self.cfg = cfg
        self.n_total = int(n_total)
        self.n_e = int(n_e)
        self.dt_ms = float(dt_ms)
        self.decay = float(np.exp(-self.dt_ms / cfg.tau_ms))
        self.local_state = np.zeros(self.n_e, dtype=np.float64)
        self.global_state = 0.0
        self._step_index = 0
        self._trace_every = max(1, int(round(cfg.trace_dt_ms / self.dt_ms)))
        self.trace_time_ms = []
        self.trace_mean_mV = []
        self.trace_sd_mV = []
        self.trace_max_mV = []

    def threshold(self, v_th_base):
        """Preserve the frozen data-driven threshold vector exactly."""
        return v_th_base

    def _state_e(self):
        if self.cfg.mode == "local":
            return self.local_state
        return self.global_state

    def apply_currents(self, i_e, i_i, labels=None):
        i_e = np.asarray(i_e, dtype=np.float64)
        i_i = np.asarray(i_i, dtype=np.float64)
        if i_e.shape != (self.n_total,) or i_i.shape != (self.n_total,):
            raise ValueError("synaptic current arrays must align to all neurons")
        output = i_e - i_i
        output[:self.n_e] -= self._state_e()
        return output

    def step(self, spikes, labels, dt_ms):
        if not np.isclose(float(dt_ms), self.dt_ms, rtol=0.0, atol=1e-12):
            raise ValueError("adaptation and engine dt differ")
        spikes = np.asarray(spikes, dtype=bool)
        if spikes.shape != (self.n_total,):
            raise ValueError("spike vector must align to all neurons")
        e_spikes = spikes[:self.n_e]
        if self.cfg.mode == "local":
            self.local_state *= self.decay
            self.local_state[e_spikes] += self.cfg.increment_mV
        else:
            self.global_state *= self.decay
            self.global_state += (
                self.cfg.increment_mV * float(np.count_nonzero(e_spikes))
                / self.n_e
            )
        self._step_index += 1
        if self._step_index % self._trace_every == 0:
            if self.cfg.mode == "local":
                mean = float(np.mean(self.local_state))
                sd = float(np.std(self.local_state))
                maximum = float(np.max(self.local_state, initial=0.0))
            else:
                mean = maximum = float(self.global_state)
                sd = 0.0
            self.trace_time_ms.append(self._step_index * self.dt_ms)
            self.trace_mean_mV.append(mean)
            self.trace_sd_mV.append(sd)
            self.trace_max_mV.append(maximum)

    def trace_arrays(self):
        return {
            "time_ms": np.asarray(self.trace_time_ms, dtype=np.float32),
            "mean_mV": np.asarray(self.trace_mean_mV, dtype=np.float32),
            "sd_mV": np.asarray(self.trace_sd_mV, dtype=np.float32),
            "max_mV": np.asarray(self.trace_max_mV, dtype=np.float32),
        }


@dataclass(frozen=True)
class InhibitoryResourceConfig:
    mode: str
    tau_q_ms: float
    k_q_per_ms: float
    q_min: float = 0.5
    n_grid: int = 32
    sigma_rate_mm: float = 0.5
    tau_rate_ms: float = 100.0
    sigma_q_mm: float = 1.5
    eta_e: float = 0.3
    eta_i: float = 1.0
    a0: float = 0.0
    a50: float = 1.0
    trace_dt_ms: float = 10.0

    def validate(self, dt_ms: float) -> None:
        if self.mode not in {"local", "global"}:
            raise ValueError("resource mode must be local or global")
        if self.tau_q_ms <= 0.0 or self.tau_rate_ms <= 0.0:
            raise ValueError("resource time constants must be positive")
        if self.k_q_per_ms < 0.0:
            raise ValueError("resource depletion must be non-negative")
        if not 0.0 <= self.q_min <= 1.0:
            raise ValueError("q_min must lie in [0,1]")
        if self.n_grid < 2 or self.sigma_rate_mm <= 0.0 or self.sigma_q_mm <= 0.0:
            raise ValueError("resource grid and widths must be positive")
        if self.eta_i < self.eta_e:
            raise ValueError("inhibitory drive weight must not be smaller than E")
        if self.trace_dt_ms < dt_ms:
            raise ValueError("trace interval must be at least one engine step")


class ActivityDependentInhibitoryResource:
    """Continuous local disinhibition and its mean-drive global control."""

    def __init__(self, positions_e, positions_i, sheet_l_mm: float,
                 dt_ms: float, cfg: InhibitoryResourceConfig):
        cfg.validate(dt_ms)
        self.cfg = cfg
        self.positions_e = np.asarray(positions_e, float)
        self.positions_i = np.asarray(positions_i, float)
        self.n_e = len(self.positions_e)
        self.n_i = len(self.positions_i)
        self.n_total = self.n_e + self.n_i
        if self.n_e == 0 or self.n_i == 0:
            raise ValueError("resource requires non-empty E and I populations")
        self.sheet_l_mm = float(sheet_l_mm)
        self.dt_ms = float(dt_ms)
        shape = (cfg.n_grid, cfg.n_grid)
        self.rate_e = np.zeros(shape)
        self.rate_i = np.zeros(shape)
        self.q_field = np.ones(shape)
        self.q_global = 1.0
        self.kernel_q = isotropic_gaussian(
            cfg.n_grid, self.sheet_l_mm, cfg.sigma_q_mm,
        )
        self.index_e_x, self.index_e_y = _grid_index(
            self.positions_e, self.sheet_l_mm, cfg.n_grid,
        )
        self.alpha_rate = float(1.0 - np.exp(-dt_ms / cfg.tau_rate_ms))
        self.trace_every = max(1, int(round(cfg.trace_dt_ms / dt_ms)))
        self.step_index = 0
        self.last_mean_drive = 0.0
        self.trace_time_ms = []
        self.trace_q_mean = []
        self.trace_q_sd = []
        self.trace_q_min = []
        self.trace_mean_drive = []

    def threshold(self, v_th_base):
        return v_th_base

    def _q_at_e(self):
        if self.cfg.mode == "local":
            return self.q_field[self.index_e_y, self.index_e_x]
        return self.q_global

    def apply_currents(self, i_e, i_i, labels=None):
        i_e = np.asarray(i_e, float)
        i_i = np.asarray(i_i, float)
        if i_e.shape != (self.n_total,) or i_i.shape != (self.n_total,):
            raise ValueError("synaptic current arrays must align to all neurons")
        output = i_e - i_i
        output[:self.n_e] = i_e[:self.n_e] - self._q_at_e() * i_i[:self.n_e]
        return output

    def step(self, spikes, labels, dt_ms):
        if not np.isclose(float(dt_ms), self.dt_ms, rtol=0.0, atol=1e-12):
            raise ValueError("resource and engine dt differ")
        spikes = np.asarray(spikes, bool)
        if spikes.shape != (self.n_total,):
            raise ValueError("spike vector must align to all neurons")
        cfg = self.cfg
        instantaneous_e = firing_rate_field(
            spikes[:self.n_e], self.positions_e, self.sheet_l_mm,
            cfg.n_grid, cfg.sigma_rate_mm,
        )
        instantaneous_i = firing_rate_field(
            spikes[self.n_e:], self.positions_i, self.sheet_l_mm,
            cfg.n_grid, cfg.sigma_rate_mm,
        )
        self.rate_e += self.alpha_rate * (instantaneous_e - self.rate_e)
        self.rate_i += self.alpha_rate * (instantaneous_i - self.rate_i)
        drive = saturation(
            convolve_periodic(
                aq_drive(self.rate_e, self.rate_i, cfg.eta_e, cfg.eta_i),
                self.kernel_q,
            ),
            cfg.a0, cfg.a50,
        )
        self.last_mean_drive = float(np.mean(drive))
        if cfg.mode == "local":
            self.q_field += self.dt_ms * (
                (1.0 - self.q_field) / cfg.tau_q_ms
                - cfg.k_q_per_ms * drive * self.q_field
            )
            np.clip(self.q_field, cfg.q_min, 1.0, out=self.q_field)
            q_mean = float(np.mean(self.q_field))
            q_sd = float(np.std(self.q_field))
            q_minimum = float(np.min(self.q_field))
        else:
            self.q_global += self.dt_ms * (
                (1.0 - self.q_global) / cfg.tau_q_ms
                - cfg.k_q_per_ms * self.last_mean_drive * self.q_global
            )
            self.q_global = float(np.clip(self.q_global, cfg.q_min, 1.0))
            q_mean = q_minimum = self.q_global
            q_sd = 0.0
        self.step_index += 1
        if self.step_index % self.trace_every == 0:
            self.trace_time_ms.append(self.step_index * self.dt_ms)
            self.trace_q_mean.append(q_mean)
            self.trace_q_sd.append(q_sd)
            self.trace_q_min.append(q_minimum)
            self.trace_mean_drive.append(self.last_mean_drive)

    def trace_arrays(self):
        return {
            "time_ms": np.asarray(self.trace_time_ms, np.float32),
            "q_mean": np.asarray(self.trace_q_mean, np.float32),
            "q_sd": np.asarray(self.trace_q_sd, np.float32),
            "q_min": np.asarray(self.trace_q_min, np.float32),
            "mean_drive": np.asarray(self.trace_mean_drive, np.float32),
        }
