"""Observation-invariant dynamic accessibility for Topic 4 rev10-D."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
