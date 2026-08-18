"""Z/M slow protocol with this round's extra recorders, as a SUBCLASS.

``src/snn_engine/mz_slow_vars.py`` is hash-locked by the frozen Z/M baseline
(``config/topic4_data_driven_snn_baseline_zm_v1.json`` -> ``inputs.mz_engine``),
so editing it in place would break that lock and, worse, silently redefine the
frozen mechanism reference this round is supposed to sit on top of. Everything
this round adds therefore lives here:

* the per-neuron slow-current PRODUCT accumulator, and
* the h-weighted trajectory, which has to be recorded on-line because a 20 s
  run never persists per-neuron z/m and the node field is only 3.53 % of the E
  population, so an unweighted population mean mostly reports background.

Both are off unless explicitly enabled, and the inherited equations, state
update and default trace are untouched.
"""
from __future__ import annotations

import numpy as np

from src.snn_engine.mz_slow_vars import MZSlowVars, MZSlowVarsConfig  # noqa: F401

WEIGHTED_TRACE_NAMES = (
    "time_ms",
    "z_weighted_mean",
    "m_weighted_mean",
    "disinhibition_weighted_mean",
    "adaptation_weighted_mean",
    "net_slow_current_weighted_mean",
)


class ZMTracedSlowVars(MZSlowVars):
    """MZSlowVars plus this round's recorders. Equations inherited verbatim."""

    WEIGHTED_TRACE_NAMES = WEIGHTED_TRACE_NAMES

    def __init__(self, *args, trace_weights_E=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._acc_n = 0
        self._acc_seen = 0
        self._acc_D = None
        self._acc_A = None
        if trace_weights_E is None:
            self._trace_weights = None
            self._weighted_trace = None
        else:
            weights = np.asarray(trace_weights_E, dtype=float)
            if weights.shape != (self.NE,):
                raise ValueError(f"trace_weights_E must have shape ({self.NE},)")
            total = float(weights.sum())
            if not np.isfinite(total) or total <= 0.0:
                raise ValueError("trace_weights_E must have positive finite mass")
            self._trace_weights = weights / total
            self._weighted_trace = {name: [] for name in WEIGHTED_TRACE_NAMES}

    # ---- slow-current product accumulator ----
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
        return {"n_steps": int(self._acc_seen),
                "disinhibition_D": disinhibition,
                "adaptation_A": adaptation,
                "net_slow_current": disinhibition - adaptation}

    def apply_currents(self, I_E, I_I, labels=None, I_E_rec=None):
        # Accumulate the PRODUCT the membrane equation is about to use, at the
        # instant it uses it: mean_t[(1-z)*I_I] != (1-mean_t[z])*mean_t[I_I]
        # whenever z and I_I co-vary, which they do.
        if self._acc_D is not None and self._acc_seen < self._acc_n:
            inhibitory = np.asarray(I_I, dtype=float)
            self._acc_D += (1.0 - self.z[:self.NE]) * inhibitory[:self.NE]
            self._acc_A += self.cfg.eta_m * self.m[:self.NE]
            self._acc_seen += 1
        return super().apply_currents(I_E, I_I, labels, I_E_rec)

    # ---- h-weighted trajectory ----
    def _record_trace(self, spikes, dt):
        super()._record_trace(spikes, dt)
        if self._weighted_trace is None:
            return
        weights = self._trace_weights
        z_e, m_e = self.z[:self.NE], self.m[:self.NE]
        inhibitory = self._I_I_last[:self.NE]
        disinhibition = float(np.dot(weights, (1.0 - z_e) * inhibitory))
        adaptation = float(self.cfg.eta_m * np.dot(weights, m_e))
        values = {
            "time_ms": float((self._step_index + 1) * dt),
            "z_weighted_mean": float(np.dot(weights, z_e)),
            "m_weighted_mean": float(np.dot(weights, m_e)),
            "disinhibition_weighted_mean": disinhibition,
            "adaptation_weighted_mean": adaptation,
            "net_slow_current_weighted_mean": disinhibition - adaptation,
        }
        for name, value in values.items():
            self._weighted_trace[name].append(value)

    def weighted_trace_arrays(self):
        if self._weighted_trace is None:
            return None
        return {name: np.asarray(values, dtype=np.float32)
                for name, values in self._weighted_trace.items()}
