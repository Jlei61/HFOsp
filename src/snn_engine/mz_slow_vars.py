"""
M4-MZ per-neuron slow variables: inhibitory efficacy z_i + spike-frequency adaptation m_i.

Peer-proposed minimal push-pull (both act on E CELLS ONLY; I cells are unmodulated):

  z_i in [0,1]  -- inhibitory efficacy (phenomenological Cl-/GABA_A depletion):
      tau_z dz_i/dt = z_inf,i - z_i ,   z_inf,i = H(I_th_EI - I_i^{E,I})
      I_I >= I_th_EI -> z_inf=0 -> z decays (disinhibition);  I_I < I_th_EI -> z_inf=1 -> z recovers.
      Effective E inhibition = z_i * I_i^{E,I}.

  m_i >= 0  -- adaptation count:
      dm_i/dt = -m_i/tau_adp + sum_k delta(t - t_i^k) ;  each E spike: m_i += 1
      Adaptation CURRENT = eta_m * m_i, SUBTRACTED from I_net (NOT a threshold shift).

  Membrane (E):  tau_m dV/dt = -V + I_E - z_i I_I - eta_m m_i
  Membrane (I):  tau_m dV/dt = -V + I_E - I_I                     (unmodulated)

Off-by-default: use_z=False AND use_m=False -> apply_currents returns I_E - I_I EXACTLY,
so a full simulate_kick run is byte-identical to slow=None (design §4). This module plugs into
src/snn_engine/kick_probe.py::simulate_kick via the slow protocol (apply_currents/threshold/step)
with ZERO edits to the 6 guarded engine files -> no engine re-bless.

Parameter values are CALIBRATION placeholders (the peer draft gives no numeric table); they are
set from the slow-off baseline distribution only (design §6), never from the z+m result.

Contract (multi-clause invariants) enumerated 1:1 in tests/test_mz_slow_vars.py.
Design: docs/superpowers/specs/2026-07-18-topic4-mz-per-neuron-slowvars-design.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MZSlowVarsConfig:
    use_z: bool = False            # OFF by default -> byte parity with slow=None
    use_m: bool = False            # OFF by default -> byte parity with slow=None
    tau_z: float = 5000.0          # ms   inhibitory-efficacy recovery/depletion time constant (CALIBRATION)
    I_th_EI: float = 0.0           # E-cell GABA current depletion threshold (CALIBRATION)
    tau_adp: float = 2000.0        # ms   adaptation decay time constant (CALIBRATION)
    eta_m: float = 0.0             # adaptation current per unit m (CALIBRATION)
    record_calib: bool = False     # slow-off OBSERVER: also bin I_I[E]/I_E[E] each step (pure side-effect)
    calib_hist_edges: "np.ndarray | None" = None


class MZSlowVars:
    """Per-E-neuron z_i (inhibitory efficacy) + m_i (adaptation). Pass to simulate_kick(slow=...).

    E cells occupy indices [:NE]; I cells [NE:]. z/m are full-N arrays whose I-cell entries stay
    pinned (z==1, m==0) and are never updated -> I cells always see I_E - I_I (E-only clause).
    core_mask_E is E-indexed (length NE); union of the two low-V_th cores (for core/surround traces).
    """

    def __init__(self, N, V_th0, cfg=None, *, NE, core_mask_E=None, snapshot_steps=None):
        self.cfg = cfg or MZSlowVarsConfig()
        self.N = int(N)
        self.NE = int(NE)
        self.V_th0 = float(V_th0)
        self.is_E = np.arange(self.N) < self.NE                 # E occupy [:NE]
        if core_mask_E is None:
            self.core_e_idx = np.array([], dtype=int)
            self.surr_e_idx = np.arange(self.NE)
        else:
            cm = np.asarray(core_mask_E, bool)
            self.core_e_idx = np.flatnonzero(cm)                # E-indexed == full-N index (E in [:NE])
            self.surr_e_idx = np.flatnonzero(~cm)
        # state: full-N. I-cell entries pinned (z=1, m=0), never touched by step() -> E-only.
        self.z = np.ones(self.N)
        self.m = np.zeros(self.N)
        self._I_I_last = np.zeros(self.N)
        # off-by-default slow-state snapshot observer (design §4.3): copy z_E/m_E at registered
        # INTEGER steps only, AFTER the slow update; None -> no capture, exact simulation parity.
        self._snap_steps = self._normalize_snapshot_steps(snapshot_steps)
        self._step_i = 0
        self.snapshots = {}                                     # label -> {z_E, m_E, step, captured_after_update}
        # audit traces (streaming scalars only -- NO N x T matrices)
        self.trace_z_mean = []; self.trace_z_min = []
        self.trace_z_core_mean = []; self.trace_z_surround_mean = []
        self.trace_m_mean = []; self.trace_m_max = []
        self.trace_m_core_mean = []; self.trace_m_surround_mean = []
        self.trace_adap_current = []                            # eta_m * mean(m[E])
        self.trace_I_EI_E_mean = []                             # E-cell inhibitory current summary
        self.trace_rate_E = []; self.trace_rate_I = []
        # calibration observer (slow-off only): per-step histograms of E-cell I_I / I_E
        self.calib_hist_I_EI = []; self.calib_hist_I_EE = []

    # ------------------------------------------------------------------ hooks
    def apply_currents(self, I_E, I_I, labels=None, I_E_rec=None):
        """I_net for the membrane update. I_E_rec accepted for engine-protocol compatibility
        (only passed when cfg.use_SG, not our case) and unused here."""
        self._I_I_last = I_I
        if self.cfg.record_calib:
            self._record_calib(I_E, I_I)                        # pure side-effect (does not alter return)
        if not self.cfg.use_z and not self.cfg.use_m:
            return I_E - I_I                                    # EXACT byte-parity path (== membrane_step)
        inh = self.z * I_I if self.cfg.use_z else I_I          # z==1 on I cells -> unscaled there
        I_net = I_E - inh
        if self.cfg.use_m:
            I_net = I_net - self.cfg.eta_m * self.m            # m==0 on I cells -> E-only adaptation current
        return I_net

    def threshold(self, V_th_base):
        return V_th_base                                        # pass through -> heterogeneous double core preserved

    def step(self, spk, labels, dt):
        c = self.cfg
        spk = np.asarray(spk, bool)
        if c.use_z:
            # z_inf = H(I_th_EI - I_I): 1 iff I_I < I_th_EI (strict); I_I >= I_th_EI -> 0 (deplete)
            z_inf_E = (self._I_I_last[self.is_E] < c.I_th_EI).astype(float)
            zE = self.z[self.is_E]
            zE = zE + (dt / c.tau_z) * (z_inf_E - zE)
            self.z[self.is_E] = np.clip(zE, 0.0, 1.0)          # z in [0,1]
        if c.use_m:
            mE = self.m[self.is_E]
            mE = mE - (mE / c.tau_adp) * dt                    # decay
            self.m[self.is_E] = np.maximum(mE, 0.0)            # m >= 0
            self.m[spk & self.is_E] += 1.0                     # E spike -> +1 ; I spikes ignored (E-only)
        self._record_traces(spk)
        # snapshot AFTER the slow update + trace record -> snapshots[label].z_E.mean() == trace_z_mean[step_i]
        if self._snap_steps is not None and self._step_i in self._snap_steps:
            self._capture(self._snap_steps[self._step_i])
        self._step_i += 1

    # ------------------------------------------------------------------ traces
    def _record_traces(self, spk):
        zE = self.z[self.is_E]; mE = self.m[self.is_E]
        self.trace_z_mean.append(float(zE.mean()))
        self.trace_z_min.append(float(zE.min()))
        self.trace_m_mean.append(float(mE.mean()))
        self.trace_m_max.append(float(mE.max()))
        self.trace_adap_current.append(float(self.cfg.eta_m * mE.mean()))
        self.trace_I_EI_E_mean.append(float(self._I_I_last[self.is_E].mean()))
        ci, si = self.core_e_idx, self.surr_e_idx
        self.trace_z_core_mean.append(float(self.z[ci].mean()) if ci.size else float("nan"))
        self.trace_z_surround_mean.append(float(self.z[si].mean()) if si.size else float("nan"))
        self.trace_m_core_mean.append(float(self.m[ci].mean()) if ci.size else float("nan"))
        self.trace_m_surround_mean.append(float(self.m[si].mean()) if si.size else float("nan"))
        self.trace_rate_E.append(int(spk[self.is_E].sum()))
        self.trace_rate_I.append(int(spk[~self.is_E].sum()))

    def _record_calib(self, I_E, I_I):
        edges = self.cfg.calib_hist_edges
        if edges is None:
            return
        hI, _ = np.histogram(I_I[self.is_E], bins=edges)
        hE, _ = np.histogram(I_E[self.is_E], bins=edges)
        self.calib_hist_I_EI.append(hI.astype(np.int64))
        self.calib_hist_I_EE.append(hE.astype(np.int64))

    # ------------------------------------------------------------------ snapshot observer (design §4.3)
    @staticmethod
    def _normalize_snapshot_steps(snapshot_steps):
        """Validate/normalize {step: label} -> {int_step: str}; None -> None (off). Raises on
        negative / non-integer-valued step or duplicate label (Gate B: invalid steps fail clearly)."""
        if snapshot_steps is None:
            return None
        norm = {}
        for k, v in dict(snapshot_steps).items():
            step = int(k)
            if step != k or step < 0:                           # k==round(t_ms/dt) must be a non-neg integer
                raise ValueError(f"snapshot step {k!r} must be a non-negative integer (== round(t_ms/dt))")
            norm[step] = str(v)                                 # dict keys unique -> no duplicate step
        if len(set(norm.values())) != len(norm):
            raise ValueError("snapshot labels must be unique")
        return norm

    def _capture(self, label):
        """Copy ONLY z_E/m_E (E cells [:NE]) at the current step -> n_snapshots x NE (never n_steps x NE)."""
        self.snapshots[label] = dict(
            z_E=self.z[:self.NE].copy(), m_E=self.m[:self.NE].copy(),
            step=int(self._step_i), captured_after_update=True,
        )

    @property
    def n_steps_run(self):
        """Number of step() calls executed (== simulate_kick iterations run, honoring early-stop)."""
        return self._step_i
