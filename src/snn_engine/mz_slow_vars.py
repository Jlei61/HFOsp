"""Per-neuron Z/M slow variables for the current-based spatial LIF engine.

For excitatory cells only,

    tau_z dz_i/dt = H(I_th_EI - I_i^EI) - z_i
    dm_i/dt = -m_i/tau_adp + sum_k delta(t - t_i^k)
    I_net,i = I_i^E - z_i I_i^I - eta_m m_i

Inhibitory cells retain ``I_E - I_I``.  With both mechanisms disabled,
``apply_currents`` takes the exact legacy expression so the engine output is
byte-identical to ``slow=None`` under common random numbers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MZSlowVarsConfig:
    use_z: bool = False
    use_m: bool = False
    tau_z: float = 5000.0
    I_th_EI: float = 0.0
    tau_adp: float = 2000.0
    eta_m: float = 0.0
    trace_stride_steps: int = 1

    def validate(self) -> None:
        if self.tau_z <= 0.0:
            raise ValueError("tau_z must be positive")
        if self.tau_adp <= 0.0:
            raise ValueError("tau_adp must be positive")
        if self.eta_m < 0.0:
            raise ValueError("eta_m must be non-negative")
        if int(self.trace_stride_steps) != self.trace_stride_steps:
            raise ValueError("trace_stride_steps must be an integer")
        if self.trace_stride_steps < 1:
            raise ValueError("trace_stride_steps must be at least one")


class MZSlowVars:
    """Slow protocol consumed by ``kick_probe.simulate_kick``.

    E cells must occupy ``[:NE]``. ``core_mask_E`` changes audit summaries
    only; it never changes the equations or state update.
    """

    TRACE_NAMES = (
        "time_ms",
        "z_mean",
        "z_min",
        "z_core_mean",
        "z_surround_mean",
        "m_mean",
        "m_max",
        "m_core_mean",
        "m_surround_mean",
        "adaptation_current_mean",
        "inhibitory_current_mean",
        "fraction_inhibitory_current_above_threshold",
        "spike_count_E",
        "spike_count_I",
    )

    def __init__(self, N, V_th0, cfg=None, *, NE, core_mask_E=None):
        self.cfg = cfg or MZSlowVarsConfig()
        self.cfg.validate()
        self.N = int(N)
        self.NE = int(NE)
        self.V_th0 = float(V_th0)
        if not 0 < self.NE <= self.N:
            raise ValueError("NE must lie in [1, N]")
        self.is_E = np.arange(self.N) < self.NE
        if core_mask_E is None:
            core = np.zeros(self.NE, dtype=bool)
        else:
            core = np.asarray(core_mask_E, dtype=bool)
            if core.shape != (self.NE,):
                raise ValueError(f"core_mask_E must have shape ({self.NE},)")
        self.core_e_idx = np.flatnonzero(core)
        self.surround_e_idx = np.flatnonzero(~core)
        self.z = np.ones(self.N, dtype=float)
        self.m = np.zeros(self.N, dtype=float)
        self._I_I_last = np.zeros(self.N, dtype=float)
        self._step_index = 0
        # ZM-ITX slow-current accumulator (off by default -> no float touched).
        self._acc_n = 0
        self._acc_seen = 0
        self._acc_D = None
        self._acc_A = None
        self._trace = {name: [] for name in self.TRACE_NAMES}

    def apply_currents(self, I_E, I_I, labels=None, I_E_rec=None):
        """Return membrane current; extra arguments preserve engine protocol."""
        del labels, I_E_rec
        # Accumulate the PRODUCT the membrane equation is about to use, at the
        # instant it uses it. mean_t[(1-z)*I_I] != (1-mean_t[z])*mean_t[I_I]
        # whenever z and I_I co-vary, which they do; averaging the factors
        # separately would bias the reported disinhibition field.
        if self._acc_D is not None and self._acc_seen < self._acc_n:
            inhibitory = np.asarray(I_I, dtype=float)
            self._acc_D += (1.0 - self.z[:self.NE]) * inhibitory[:self.NE]
            self._acc_A += self.cfg.eta_m * self.m[:self.NE]
            self._acc_seen += 1
        self._I_I_last = I_I
        if not self.cfg.use_z and not self.cfg.use_m:
            return I_E - I_I
        inhibition = self.z * I_I if self.cfg.use_z else I_I
        current = I_E - inhibition
        if self.cfg.use_m:
            current = current - self.cfg.eta_m * self.m
        return current

    @staticmethod
    def threshold(V_th_base):
        """Preserve the heterogeneous data-driven threshold field exactly."""
        return V_th_base

    def enable_field_accumulator(self, n_steps):
        n_steps = int(n_steps)
        if n_steps < 1:
            raise ValueError("field accumulator needs at least one step")
        self._acc_n = n_steps
        self._acc_seen = 0
        self._acc_D = np.zeros(self.NE, dtype=float)
        self._acc_A = np.zeros(self.NE, dtype=float)

    def field_accumulator_result(self):
        if self._acc_D is None or self._acc_seen == 0:
            return None
        scale = 1.0 / float(self._acc_seen)
        disinhibition = self._acc_D * scale
        adaptation = self._acc_A * scale
        return {
            "n_steps": int(self._acc_seen),
            "disinhibition_D": disinhibition,
            "adaptation_A": adaptation,
            "net_slow_current": disinhibition - adaptation,
        }

    def step(self, spk, labels, dt):
        del labels
        dt = float(dt)
        spikes = np.asarray(spk, dtype=bool)
        if spikes.shape != (self.N,):
            raise ValueError(f"spk must have shape ({self.N},)")
        if self.cfg.use_z:
            z_inf = (
                self._I_I_last[:self.NE] < self.cfg.I_th_EI
            ).astype(float)
            z_e = self.z[:self.NE]
            z_e += (dt / self.cfg.tau_z) * (z_inf - z_e)
            np.clip(z_e, 0.0, 1.0, out=z_e)
        if self.cfg.use_m:
            m_e = self.m[:self.NE]
            m_e -= (dt / self.cfg.tau_adp) * m_e
            np.maximum(m_e, 0.0, out=m_e)
            m_e[spikes[:self.NE]] += 1.0
        if self._step_index % self.cfg.trace_stride_steps == 0:
            self._record_trace(spikes, dt)
        self._step_index += 1

    def _record_trace(self, spikes, dt):
        z_e = self.z[:self.NE]
        m_e = self.m[:self.NE]
        core = self.core_e_idx
        surround = self.surround_e_idx
        values = {
            "time_ms": (self._step_index + 1) * dt,
            "z_mean": np.mean(z_e),
            "z_min": np.min(z_e),
            "z_core_mean": np.mean(z_e[core]) if core.size else np.nan,
            "z_surround_mean": (
                np.mean(z_e[surround]) if surround.size else np.nan
            ),
            "m_mean": np.mean(m_e),
            "m_max": np.max(m_e),
            "m_core_mean": np.mean(m_e[core]) if core.size else np.nan,
            "m_surround_mean": (
                np.mean(m_e[surround]) if surround.size else np.nan
            ),
            "adaptation_current_mean": self.cfg.eta_m * np.mean(m_e),
            "inhibitory_current_mean": np.mean(self._I_I_last[:self.NE]),
            "fraction_inhibitory_current_above_threshold": np.mean(
                self._I_I_last[:self.NE] >= self.cfg.I_th_EI
            ),
            "spike_count_E": np.sum(spikes[:self.NE]),
            "spike_count_I": np.sum(spikes[self.NE:]),
        }
        for name, value in values.items():
            self._trace[name].append(float(value))

    def trace_arrays(self):
        return {
            name: np.asarray(values, dtype=np.float32)
            for name, values in self._trace.items()
        }

    def summary(self):
        traces = self.trace_arrays()
        return {
            "trace_samples": int(len(traces["time_ms"])),
            "final_z_mean": float(np.mean(self.z[:self.NE])),
            "minimum_z": float(np.min(self.z[:self.NE])),
            "final_m_mean": float(np.mean(self.m[:self.NE])),
            "maximum_m": float(np.max(self.m[:self.NE])),
            "peak_mean_adaptation_current": float(np.max(
                traces["adaptation_current_mean"], initial=0.0,
            )),
            "mean_fraction_above_z_threshold": float(np.mean(
                traces["fraction_inhibitory_current_above_threshold"],
            )) if len(traces["time_ms"]) else 0.0,
        }
