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

  Current membrane (E):  tau_m dV/dt = -V + I_E - z_i I_I - eta_m m_i
  Membrane (I):  tau_m dV/dt = -V + I_E - I_I                     (unmodulated)

Optional conductance membrane (E; OFF by default):
      tau_m dV/dt = -V + I_E + g_I(E_GABA - V) + g_M(E_K - V)
  ``I_I`` and ``eta_m*m`` are current proxies, not conductances.  They are mapped to
  leak-relative conductances by matching their force at ``v_match``.  A fraction
  ``global_gaba_fraction`` either replaces part of local received GABA by its
  E-population mean or adds that mean as a protected brake.  Both are explicitly
  labelled rank-1 received-current surrogates, not an extra slow M4 pool.

Off-by-default: use_z=False AND use_m=False -> apply_currents returns I_E - I_I EXACTLY,
so a full simulate_kick run is byte-identical to slow=None (design §4). This module plugs into
src/snn_engine/kick_probe.py::simulate_kick via an opt-in conductance protocol plus the historical
apply_currents/threshold/step protocol.  The guarded engine change therefore requires explicit
regression and re-blessing before any scientific run.

Parameter values are CALIBRATION placeholders (the peer draft gives no numeric table); they are
set from the slow-off baseline distribution only (design §6), never from the z+m result.

Contract (multi-clause invariants) enumerated 1:1 in tests/test_mz_slow_vars.py.
Design: docs/superpowers/specs/2026-07-18-topic4-mz-per-neuron-slowvars-design.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def cooperative_u_tilde(u, A_c, u_c, K_c, n):
    """FCXR-HEO1 cooperative recurrent-conductance gate (design lock 2026-07-24 §mechanism).

    Applied to the RAW recurrent conductance u = gErec_raw BEFORE the g_sat*tanh saturation.
    OFF when A_c<=0 -> returns u unchanged (byte parity). For u<=u_c the excess is 0 -> H=0 ->
    u_tilde = u*(1+0) = u exactly. Monotone non-decreasing, non-negative and finite in u>=0;
    the boost is bounded by (1+A_c) since H<1. n is the Hill exponent (fixed n=4 this sprint).

        H = relu(u-u_c)^n / (K_c^n + relu(u-u_c)^n) ;  u_tilde = u * (1 + A_c * H)
    """
    if A_c <= 0.0:
        return u
    excess = np.maximum(u - u_c, 0.0)
    exc_n = excess ** n
    H = exc_n / (K_c ** n + exc_n)
    return u * (1.0 + A_c * H)


def gerec_baseline_quantiles(counts, edges, qs):
    """Quantiles of gErec_raw from the slow-off cumulative fixed-edge histogram (FCXR-HEO1 calibration).

    Linear interpolation within the crossing bin. A quantile that falls in a trailing overflow bin
    (edges[-1]=inf) returns +inf, so the runner widens the edge grid + re-runs F0 rather than locking
    a bogus u_c. counts/edges come straight off MZSlowVars.gerec_hist_* + cfg.gerec_hist_edges."""
    counts = np.asarray(counts, float)
    edges = np.asarray(edges, float)
    total = counts.sum()
    if total <= 0:
        raise ValueError("empty gErec histogram (no baseline samples)")
    cum = np.cumsum(counts) / total
    out = {}
    for q in qs:
        k = min(int(np.searchsorted(cum, q, side="left")), len(counts) - 1)
        lo, hi = edges[k], edges[k + 1]
        c_lo = cum[k - 1] if k > 0 else 0.0
        c_hi = cum[k]
        frac = 0.0 if c_hi <= c_lo else (q - c_lo) / (c_hi - c_lo)
        out[float(q)] = float(lo + frac * (hi - lo))
    return out


@dataclass
class MZSlowVarsConfig:
    use_z: bool = False            # OFF by default -> byte parity with slow=None
    use_m: bool = False            # OFF by default -> byte parity with slow=None
    tau_z: float = 5000.0          # ms   inhibitory-efficacy recovery/depletion time constant (CALIBRATION)
    I_th_EI: float = 0.0           # E-cell GABA current depletion threshold (CALIBRATION)
    tau_adp: float = 2000.0        # ms   adaptation decay time constant (CALIBRATION)
    eta_m: float = 0.0             # adaptation current per unit m (CALIBRATION)
    m_enable_ms: "float | None" = None  # FCXR-HEO2: delayed adaptation onset — m stays 0 until step*dt>=this (None = from step 0)
    m_frozen_E: "np.ndarray | None" = None  # FCXR-HEO2: static-K control — hold m frozen at this per-E field (requires use_m=False, full_conductance)
    m_frozen_enable_ms: "float | None" = None  # FCXR-HEO2.1: delayed static-K — inject m_frozen_E only at step*dt>=this (None = t=0; requires m_frozen_E)
    tau_adp_E: "np.ndarray | None" = None  # FCXR-HEO3: per-E-cell adaptation RECOVERY time (patchy field; None = scalar tau_adp)
    eta_m_E: "np.ndarray | None" = None    # FCXR-HEO3: per-E-cell adaptation strength (None = scalar eta_m); pair with tau_adp_E as eta_i=eta0*tau0/tau_i to hold each cell's steady-state K load fixed
    m_mean_field: bool = False             # FCXR-HEO3 control: replace per-cell m by the POPULATION MEAN each step (pure temporal modulation, no inter-cell differences)
    use_phi: bool = False          # optional Abbott-style spike-triggered threshold adaptation
    tau_phi: float = 100.0         # ms
    delta_phi: float = 0.0         # mV threshold increment per E spike
    membrane_mode: str = "current"  # current | conductance | full_conductance (both conductance opt-in)
    gaba_gain: float = 1.0         # multiplier after V_match force matching
    m_conductance_gain: float = 1.0
    # ---- FCXR: full-conductance E-cell AMPA + persistence-gated E->E relay (all OFF by default) ----
    E_E: float = 58.0              # AMPA reversal (full_conductance only), engine V_L=0 coords
    c_E: float = 1.0               # excitatory force-match coefficient (full_conductance only)
    ff_conductance: bool = True    # full_conductance: feedforward(external) AMPA as conductance, else additive c_E*I
    rec_conductance: bool = True   # full_conductance: recurrent E->E AMPA as conductance, else additive c_E*I
    rec_sat_g: float = 0.0         # FCXR-RC1 Stage C: >0 -> recurrent conductance smooth-saturates g_sat*tanh(g_raw/g_sat)
    # ---- FCXR-HEO1: cooperative recurrent-conductance gate (all OFF by default -> RC1 byte parity) ----
    coop_A: float = 0.0            # A_c: cooperative gain amplitude (0 -> gate off); acts on gErec_raw only
    coop_uc: float = 0.0           # u_c: gErec_raw activation threshold (locked from slow-off baseline quantile)
    coop_Kc: float = 0.0           # K_c: Hill half-activation above u_c (0.25*u_c)
    coop_n: int = 4                # Hill exponent (fixed n=4)
    record_gerec_hist: bool = False  # slow-off OBSERVER: cumulative fixed-edge gErec_raw histogram (overall/core/surround)
    gerec_hist_edges: "np.ndarray | None" = None
    use_x: bool = False            # persistence-gated presynaptic E->E relay availability x_j
    tau_y: float = 120.0           # ms  persistence sensor time constant
    tau_x: float = 1000.0          # ms  relay availability time constant (symmetric; used when tau_x_down/up are None)
    tau_x_down: "float | None" = None  # FCXR-LC1: asymmetric relay depletion tau (x_inf<x); None+None -> symmetric tau_x
    tau_x_up: "float | None" = None    # FCXR-LC1: asymmetric relay recovery tau (x_inf>=x)
    x_min: float = 0.0             # relay availability floor
    y_gate: float = 0.0            # Hz  sensor gate (locked from slow-off Q99.9)
    K_y: float = 5.0               # Hz  Hill half-activation above the gate
    hill_n: int = 4                # Hill exponent
    global_gaba_fraction: float = 0.0  # gamma: local received GABA -> E-population mean replacement
    global_gaba_mode: str = "replace"  # replace | additive (local + gamma*population mean)
    z_scope: str = "total"        # total | local_only
    v_match: float = 18.0          # mV-equivalent reference voltage for current-force matching
    e_gaba: float = 11.0           # GABA reversal in the engine's V_L=0 coordinates
    e_k: float = 0.0               # sAHP reversal in the engine's V_L=0 coordinates
    max_total_conductance: float = np.inf  # leak-relative cap; runner sets a finite safety limit
    fail_on_clip: bool = False      # scientific runner sets True; clipping then fails the cell immediately
    record_calib: bool = False     # slow-off OBSERVER: also bin I_I[E]/I_E[E] each step (pure side-effect)
    calib_hist_edges: "np.ndarray | None" = None
    record_clip_identity: bool = False  # FCXR-RC1 clip audit: per-cell clip_count + max raw gErec/total (pure side-effect)
    z_frozen_E: "np.ndarray | None" = None  # FCXR Stage D: hold z_i frozen at this per-E field (requires use_z=False)
    # ---- FCXR pump lifecycle: per-E activity-dependent load u_i + electrogenic pump (all OFF by default) ----
    # u_i is an ACTIVITY-DEPENDENT INTRACELLULAR LOAD (Na/pump-inspired) -- never a Na concentration.
    use_pump: bool = False         # master switch; False -> no state allocated, no float touched (byte parity)
    pump_sensor_only: bool = False # u evolves but the membrane is byte-identical to pump-off (Imax must be 0)
    pump_a_load: float = 0.0       # per-E-spike load jump (NOT scaled by dt)
    pump_tau_ms: float = 0.0       # ms  pump-mediated load release time (clearance scaled by dt/tau_N)
    pump_Imax: float = 0.0         # max electrogenic membrane effect; >0 requires pump_p0_E
    pump_h: int = 3                # Hill exponent of phi(u)=u^h/(1+u^h); primary tier fixes 3
    pump_p0_E: "np.ndarray | None" = None    # per-E baseline pump activation E_baseline[phi(u_i)] (shrunken)
    pump_u_init_E: "np.ndarray | None" = None  # start the load at an equilibrated field (None -> zeros)
    pump_record_calibration: bool = False    # OBSERVER: cumulative per-cell sum phi(u) + spike counts
    pump_interventions: "list | None" = None  # scheduled INTEGER-step interventions (see _normalize_pump_interventions)


class MZSlowVars:
    """Per-E-neuron z_i (inhibitory efficacy) + m_i (adaptation). Pass to simulate_kick(slow=...).

    E cells occupy indices [:NE]; I cells [NE:]. z/m are full-N arrays whose I-cell entries stay
    pinned (z==1, m==0) and are never updated -> I cells always see I_E - I_I (E-only clause).
    core_mask_E is E-indexed (length NE); union of the two low-V_th cores (for core/surround traces).
    """

    def __init__(self, N, V_th0, cfg=None, *, NE, core_mask_E=None, snapshot_steps=None):
        self.cfg = cfg or MZSlowVarsConfig()
        self._validate_config()
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
        if self.cfg.z_frozen_E is not None:                       # FCXR Stage D: hold E-cell z frozen at a preset field
            zf = np.asarray(self.cfg.z_frozen_E, float)
            if zf.shape != (self.NE,):
                raise ValueError(f"z_frozen_E must have length NE={self.NE}, got {zf.shape}")
            self.z[:self.NE] = zf                                 # I-cell z stays 1 (E-only clause)
        self.m = np.zeros(self.N)
        self._m_frozen_cached = None
        if self.cfg.m_frozen_E is not None:                       # FCXR-HEO2: hold E-cell m frozen (static-K)
            mf = np.asarray(self.cfg.m_frozen_E, float)
            if mf.shape != (self.NE,):
                raise ValueError(f"m_frozen_E must have length NE={self.NE}, got {mf.shape}")
            self._m_frozen_cached = mf                            # FCXR-HEO2.1: cache for (possibly delayed) inject
            if self.cfg.m_frozen_enable_ms is None:
                self.m[:self.NE] = mf                             # immediate static-K (I-cell m stays 0)
            # else: leave m=0 until m_frozen_enable_ms, inject in step()
        # FCXR-HEO3: per-E-cell adaptation fields (default = the scalars -> byte-parity)
        self._eta_E = (np.full(self.NE, float(self.cfg.eta_m)) if self.cfg.eta_m_E is None
                       else np.asarray(self.cfg.eta_m_E, float).copy())
        self._tau_E = (np.full(self.NE, float(self.cfg.tau_adp)) if self.cfg.tau_adp_E is None
                       else np.asarray(self.cfg.tau_adp_E, float).copy())
        self._eta_full = np.full(self.N, float(self.cfg.eta_m))
        self._eta_full[:self.NE] = self._eta_E
        self.phi = np.zeros(self.N)
        # ---- FCXR pump lifecycle state (allocated ONLY when use_pump; off -> None -> byte parity) ----
        self.u_pump_E = np.zeros(self.NE) if self.cfg.use_pump else None
        self.pump_phi_sum_E = None
        self.pump_spike_count_E = None
        self.pump_phi_count = 0
        self._pump_knockout_step = None                         # set by a scheduled current knockout
        self._pump_set_load = {}                                # step -> field (one-shot load reset/injection)
        self._pump_p0_E = None
        self._pump_phi_last = None
        self._pump_excess_last = None
        # Non-blessed virtual-SEEG component observer (src/topic4_mz_fcxr_pump.py). Assigned by the
        # runner AFTER construction; None -> never called -> byte parity. Pure side-effect: it only
        # reduces the already-computed E-cell components onto the electrode weights.
        self.seeg_observer = None
        if self.cfg.use_pump:
            if self.cfg.pump_u_init_E is not None:              # start equilibrated (no startup transient)
                u0 = np.asarray(self.cfg.pump_u_init_E, float)
                if u0.shape != (self.NE,) or not np.all(np.isfinite(u0)) or u0.min() < 0.0:
                    raise ValueError(f"pump_u_init_E must be a finite field of shape ({self.NE},) with values >= 0")
                self.u_pump_E[:] = u0
            if self.cfg.pump_Imax > 0.0:                        # a live membrane effect NEEDS its baseline
                if self.cfg.pump_p0_E is None:
                    raise ValueError("pump_Imax>0 requires pump_p0_E (per-E baseline pump activation)")
                p0 = np.asarray(self.cfg.pump_p0_E, float)
                if p0.shape != (self.NE,) or not np.all(np.isfinite(p0)):
                    raise ValueError(f"pump_p0_E must be a finite field of shape ({self.NE},), got {p0.shape}")
                self._pump_p0_E = p0
            if self.cfg.pump_record_calibration:
                self.pump_phi_sum_E = np.zeros(self.NE)
                self.pump_spike_count_E = np.zeros(self.NE, dtype=np.int64)
            self._pump_knockout_step, self._pump_set_load = \
                self._normalize_pump_interventions(self.cfg.pump_interventions)
            for fld in self._pump_set_load.values():
                if fld.shape != (self.NE,):
                    raise ValueError(f"set_load field must have shape ({self.NE},), got {fld.shape}")
        # FCXR persistence sensor y_j (Hz) + presynaptic E->E relay availability x_j in [0,1] (E cells only).
        # ee_relay_send is the x_j(t-) snapshot the engine scatter reads BEFORE step() updates y/x this frame
        # (causal send scale). All three stay untouched unless cfg.use_x -> no effect on non-relay runs.
        self.y = np.zeros(self.NE)
        self.x_relay = np.ones(self.NE)
        self.ee_relay_send = np.ones(self.NE)
        self._I_I_last = np.zeros(self.N)
        self._z_sensor_last_E = np.zeros(self.NE)
        self._gI_last_E = np.zeros(self.NE)
        self._gM_last_E = np.zeros(self.NE)
        self._gbar_last = 0.0
        self._g_global_pre_z_last = 0.0
        self._gI_mean_last = self._gI_max_last = 0.0
        self._gM_mean_last = self._gM_max_last = 0.0
        self._gEff_mean_last = self._gErec_mean_last = 0.0   # FCXR AMPA conductance split (full_conductance)
        self._tau_ratio_mean_last = self._tau_ratio_min_last = 1.0
        self._clip_frac_last = 0.0
        self._coop_engaged_frac_last = 0.0        # FCXR-HEO1: fraction of E cells with gErec_raw > u_c
        self._coop_H_mean_last = 0.0              # FCXR-HEO1: mean cooperative Hill activation
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
        self.trace_phi_mean = []; self.trace_phi_max = []
        self.trace_adap_current = []                            # eta_m * mean(m[E])
        self.trace_I_EI_E_mean = []                             # E-cell inhibitory current summary
        self.trace_rate_E = []; self.trace_rate_I = []
        self.trace_gaba_received_mean = []
        self.trace_global_pre_z = []; self.trace_z_sensor_mean = []
        self.trace_gI_mean = []; self.trace_gI_max = []
        self.trace_gM_mean = []; self.trace_gM_max = []
        self.trace_tau_eff_ratio_mean = []; self.trace_tau_eff_ratio_min = []
        self.trace_conductance_clip_frac = []
        # FCXR relay traces (appended only when cfg.use_x): sensor y_j (Hz) + relay availability x_j
        self.trace_gEff_mean = []; self.trace_gErec_mean = []
        self.trace_y_mean = []; self.trace_y_max = []
        self.trace_x_relay_mean = []; self.trace_x_relay_min = []
        self.trace_coop_engaged_frac = []; self.trace_coop_H_mean = []   # FCXR-HEO1 (appended when coop_A>0)
        # FCXR pump traces (appended only when cfg.use_pump); excess traces only when it reaches the membrane
        self.trace_u_mean = []; self.trace_u_max = []
        self.trace_phi_pump_mean = []; self.trace_phi_pump_max = []
        self.trace_pump_excess_mean = []; self.trace_pump_excess_max = []; self.trace_pump_excess_min = []
        # calibration observer (slow-off only): per-step histograms of E-cell I_I / I_E
        self.calib_hist_I_EI = []; self.calib_hist_I_EE = []
        # FCXR-RC1 clip-identity observer (pure side-effect; only allocated when record_clip_identity).
        # Answers "WHICH E cells hit the conductance cap, how often, how hard" -> is the clip a localized mode?
        if self.cfg.record_clip_identity:
            self.clip_count = np.zeros(self.NE, dtype=np.int64)   # # steps this cell's total conductance > cap
            self.max_raw_gErec = np.zeros(self.NE, dtype=float)   # peak pre-clip recurrent AMPA conductance
            self.max_raw_total = np.zeros(self.NE, dtype=float)   # peak pre-clip total conductance
            self.first_clip_step = np.full(self.NE, -1, dtype=np.int64)
            self.last_clip_step = np.full(self.NE, -1, dtype=np.int64)
        if self.cfg.record_gerec_hist:                             # FCXR-HEO1 baseline calibration observer
            if self.cfg.gerec_hist_edges is None:
                raise ValueError("record_gerec_hist requires gerec_hist_edges")
            nb = len(np.asarray(self.cfg.gerec_hist_edges)) - 1    # cumulative fixed-edge gErec_raw histograms
            self.gerec_hist_overall = np.zeros(nb, dtype=np.int64)
            self.gerec_hist_core = np.zeros(nb, dtype=np.int64)
            self.gerec_hist_surround = np.zeros(nb, dtype=np.int64)

    # ------------------------------------------------------------------ hooks
    def apply_currents(self, I_E, I_I, labels=None, I_E_rec=None):
        """I_net for the membrane update. I_E_rec accepted for engine-protocol compatibility
        (only passed when cfg.use_SG, not our case) and unused here."""
        self._I_I_last = I_I
        self._z_sensor_last_E = np.asarray(I_I[:self.NE], float)
        if self.cfg.record_calib:
            self._record_calib(I_E, I_I)                        # pure side-effect (does not alter return)
        ex = self._pump_excess_E()                              # None unless the pump reaches the membrane
        if ex is None and not self.cfg.use_z and not self.cfg.use_m and self.cfg.z_frozen_E is None:
            return I_E - I_I                                    # EXACT byte-parity path (== membrane_step)
        inh = self.z * I_I if (self.cfg.use_z or self.cfg.z_frozen_E is not None) else I_I  # frozen z is applied
        I_net = I_E - inh
        if self.cfg.use_m:
            I_net = I_net - self._eta_full * self.m            # m==0 on I cells -> E-only adaptation current
        if ex is not None:
            I_net = I_net.copy()
            I_net[:self.NE] -= ex                              # E-only electrogenic pump (I cells untouched)
        return I_net

    def uses_conductance_membrane(self):
        """True for either conductance branch (partial GABA-only or full AMPA+GABA)."""
        return self.cfg.membrane_mode in ("conductance", "full_conductance")

    def uses_split_excitation(self):
        """True only for full_conductance: the engine must supply the recurrent AMPA component I_E_rec."""
        return self.cfg.membrane_mode == "full_conductance"

    def uses_ee_relay(self):
        """True only when the persistence-gated presynaptic E->E relay is active (full_conductance)."""
        return bool(self.cfg.use_x)

    def membrane_terms(self, I_E, I_I, labels=None, I_E_rec=None):
        """Return ``(drive, g_rel, g_rev)`` for an exact conductance membrane update.

        The engine consumes these as

            V_inf = (drive + g_rev) / (1 + g_rel)
            V(t+dt) = V_inf + (V(t)-V_inf) * decay_V ** (1+g_rel)

        I cells deliberately remain on the literal current path: drive=I_E-I_I,
        g_rel=g_rev=0.  Only O(N) scratch is allocated; no neuron-by-time trace is kept.
        """
        if not self.uses_conductance_membrane():
            raise RuntimeError("membrane_terms requires membrane_mode='conductance'")
        I_E = np.asarray(I_E, float)
        I_I = np.asarray(I_I, float)
        if I_E.shape != (self.N,) or I_I.shape != (self.N,):
            raise ValueError(f"I_E/I_I must have shape ({self.N},)")
        full = self.cfg.membrane_mode == "full_conductance"
        if full and I_E_rec is None:
            raise ValueError("full_conductance membrane_terms requires the recurrent AMPA component I_E_rec")
        if not full and I_E_rec is not None:
            raise ValueError("partial conductance membrane_terms does not accept I_E_rec")
        if full:
            I_E_rec = np.asarray(I_E_rec, float)
            if I_E_rec.shape != (self.N,):
                raise ValueError(f"I_E_rec must have shape ({self.N},)")
        self._I_I_last = I_I

        drive = I_E - I_I                         # literal current path for I cells
        drive = drive.copy()
        g_rel = np.zeros(self.N, dtype=float)
        g_rev = np.zeros(self.N, dtype=float)

        c = self.cfg
        I_received = np.maximum(I_I[:self.NE], 0.0)
        I_bar = float(I_received.mean()) if I_received.size else 0.0
        gamma = float(c.global_gaba_fraction)
        if c.global_gaba_mode == "replace":
            local = (1.0 - gamma) * I_received
            global_part = gamma * I_bar
        else:  # additive: retain the local restraint and add a rank-1 population-mean brake
            local = I_received
            global_part = gamma * I_bar
        pre_z_total = local + global_part
        # The Z depletion sensor must see the same pre-z received GABA that the membrane sees.
        # The protected-global sensitivity intentionally lets Z sense local use only.
        self._z_sensor_last_E = pre_z_total if c.z_scope == "total" else local
        if c.record_calib:
            sensor = I_I.copy()
            sensor[:self.NE] = self._z_sensor_last_E
            self._record_calib(I_E, sensor)
        if c.use_z or c.z_frozen_E is not None:                   # FCXR Stage D: a frozen field is APPLIED (not evolved)
            zE = self.z[:self.NE]
            if c.z_scope == "total":
                I_inh_eff = zE * pre_z_total
            else:  # local_only: protected spatially non-specific restraint sensitivity
                I_inh_eff = zE * local + global_part
        else:
            I_inh_eff = local + global_part

        gI = c.gaba_gain * I_inh_eff / (c.v_match - c.e_gaba)
        if c.use_m or c.m_frozen_E is not None:                   # FCXR-HEO2: frozen m also drives a static gM
            I_adap = self._eta_E * self.m[:self.NE]
            gM = c.m_conductance_gain * I_adap / (c.v_match - c.e_k)
        else:
            gM = np.zeros(self.NE, dtype=float)

        # FCXR full conductance: AMPA becomes a reversal-aware conductance toward E_E, force-matched at
        # V_match by c_E.  Split feedforward (external) vs recurrent (E->E) is exposed for diagnostics;
        # the x-modulation is applied at the presynaptic SCATTER (source-level) so I_E_rec already carries
        # it -> g_E_ff+g_E_rec == c_E*I_E/(E_E-V_match).  Partial conductance keeps AMPA additive (gE=0).
        # Pathway split (feedforward vs recurrent AMPA): each side is independently a conductance (toward
        # E_E) or an additive current (c_E*I).  All combinations share the SAME V_match force anchor
        # (ampa_drive + gE*(E_E-V_match) == c_E*I_E), so arms differ ONLY in state-dependence off V_match.
        # Default ff_conductance=rec_conductance=True == the original full_conductance (arm D).
        ampa_drive = np.zeros(self.NE, dtype=float)
        if full:
            denomE = c.E_E - c.v_match
            I_ffE = np.maximum(I_E[:self.NE] - I_E_rec[:self.NE], 0.0)
            I_recE = np.maximum(I_E_rec[:self.NE], 0.0)
            if c.ff_conductance:
                gEff = c.c_E * I_ffE / denomE
            else:
                gEff = np.zeros(self.NE, dtype=float); ampa_drive = ampa_drive + c.c_E * I_ffE
            if c.rec_conductance:
                gErec_raw = c.c_E * I_recE / denomE
                # FCXR-HEO1: cooperative gate boosts the RAW recurrent conductance in a mid-activity band
                # BEFORE saturation. gErec_raw is kept raw (clip/histogram audit reads it below); coop_A=0
                # -> u_tilde IS gErec_raw -> the saturation line stays byte-identical to FCXR-RC1.
                u_tilde = cooperative_u_tilde(gErec_raw, c.coop_A, c.coop_uc, c.coop_Kc, c.coop_n)
                # FCXR-RC1 Stage C: smooth-saturate the recurrent conductance (slope 1 at small input ->
                # interictal workpoint preserved; saturates toward g_sat at high input -> no hard clip).
                gErec = (c.rec_sat_g * np.tanh(u_tilde / c.rec_sat_g)) if c.rec_sat_g > 0.0 else u_tilde
            else:
                gErec_raw = np.zeros(self.NE, dtype=float)
                gErec = np.zeros(self.NE, dtype=float); ampa_drive = ampa_drive + c.c_E * I_recE
            gE = gEff + gErec
        else:
            gEff = np.zeros(self.NE, dtype=float)
            gErec_raw = np.zeros(self.NE, dtype=float)
            gErec = np.zeros(self.NE, dtype=float)
            gE = np.zeros(self.NE, dtype=float)

        total = gE + gI + gM                      # partial: gE==0 -> total==gI+gM (byte-identical)
        clip = total > c.max_total_conductance
        if c.record_clip_identity:                # pure read of RAW (pre-clip, pre-saturation) gErec/total + clip mask
            self.max_raw_gErec = np.maximum(self.max_raw_gErec, gErec_raw)
            self.max_raw_total = np.maximum(self.max_raw_total, total)
            if np.any(clip):
                self.clip_count[clip] += 1
                t = int(self._step_i)
                new = clip & (self.first_clip_step < 0)
                self.first_clip_step[new] = t
                self.last_clip_step[clip] = t
        if c.record_gerec_hist:                   # FCXR-HEO1: pool slow-off gErec_raw distribution for u_c
            self._record_gerec_hist(gErec_raw)
        if c.coop_A > 0.0:                         # FCXR-HEO1 cooperative engagement diagnostics (pure side-effect)
            _excess = np.maximum(gErec_raw - c.coop_uc, 0.0)
            self._coop_engaged_frac_last = float(np.mean(_excess > 0.0)) if _excess.size else 0.0
            _en = _excess ** c.coop_n
            self._coop_H_mean_last = float(np.mean(_en / (c.coop_Kc ** c.coop_n + _en))) if _en.size else 0.0
        if np.any(clip):
            if c.fail_on_clip:
                raise FloatingPointError(
                    f"MZ conductance exceeded cap {c.max_total_conductance:g} "
                    f"in {float(np.mean(clip)):.3%} of E cells"
                )
            scale = c.max_total_conductance / total[clip]
            gI = gI.copy(); gM = gM.copy(); gE = gE.copy()
            gI[clip] *= scale
            gM[clip] *= scale
            gE[clip] *= scale                      # partial: gE all-zero, scaling is a no-op
            total = gE + gI + gM
        if not (np.all(np.isfinite(total)) and np.all(total >= 0.0)):
            raise FloatingPointError("non-finite or negative MZ conductance")

        if full:
            drive[:self.NE] = ampa_drive                      # additive AMPA parts (0 when both sides conductance)
            g_rev[:self.NE] = gE * c.E_E + gI * c.e_gaba + gM * c.e_k
        else:
            drive[:self.NE] = I_E[:self.NE]                   # partial: AMPA stays additive (byte-identical)
            g_rev[:self.NE] = gI * c.e_gaba + gM * c.e_k
        ex_pump = self._pump_excess_E()                       # None -> no float touched (pump-off parity)
        if self.seeg_observer is not None:                    # pure read of the final (post-clip) components
            self.seeg_observer.sample(I_E, I_I, gE, gI, gM, ex_pump)
        if ex_pump is not None:
            # Electrogenic pump: an E-only CURRENT in the numerator of V_inf=(drive+g_rev)/(1+g_rel),
            # i.e. tau_m dV/dt = ... - Imax*phi(u) + Imax*p0.  It is NOT a conductance: g_rel/g_rev
            # (and hence tau_eff) are untouched, so the pump cannot shunt the membrane.
            drive[:self.NE] -= ex_pump
        g_rel[:self.NE] = total
        if not (np.all(np.isfinite(drive)) and np.all(np.isfinite(g_rev))):
            raise FloatingPointError("non-finite MZ membrane term")
        self._gI_last_E = gI
        self._gM_last_E = gM
        self._gEff_mean_last = float(gEff.mean()); self._gErec_mean_last = float(gErec.mean())
        self._gbar_last = I_bar
        self._g_global_pre_z_last = float(global_part)
        self._gI_mean_last = float(gI.mean()); self._gI_max_last = float(gI.max())
        self._gM_mean_last = float(gM.mean()); self._gM_max_last = float(gM.max())
        ratio = 1.0 / (1.0 + total)
        self._tau_ratio_mean_last = float(ratio.mean())
        self._tau_ratio_min_last = float(ratio.min())
        self._clip_frac_last = float(np.mean(clip)) if clip.size else 0.0
        return drive, g_rel, g_rev

    def threshold(self, V_th_base):
        if not self.cfg.use_phi:
            return V_th_base                                    # exact pass-through -> heterogeneous core preserved
        base = np.asarray(V_th_base, float)
        if base.ndim == 0:
            base = np.full(self.N, float(base))
        out = base.copy()
        out[:self.NE] += self.phi[:self.NE]
        return out

    def step(self, spk, labels, dt):
        c = self.cfg
        spk = np.asarray(spk, bool)
        if c.use_x:
            # FCXR relay. CAUSAL: snapshot x_j(t-) BEFORE this frame's y/x update, so the engine scatter
            # sends the current spikes with the pre-spike relay availability (a single spike never weakens
            # its own send; only sustained firing leaves an outgoing-relay wake).
            np.copyto(self.ee_relay_send, self.x_relay)
            spkE = spk[:self.NE]
            # y_j: exact decay + per-E-spike jump of 1000/tau_y -> a Hz-unit local persistence sensor.
            self.y *= np.exp(-dt / c.tau_y)
            self.y[spkE] += 1000.0 / c.tau_y
            # x_j: relax toward x_inf(y) = 1 - (1-x_min)*Hill([y-y_gate]_+; K_y, n).  x stays in [0,1].
            u = np.maximum(self.y - c.y_gate, 0.0)
            un = u ** c.hill_n
            hill = un / (c.K_y ** c.hill_n + un)
            x_inf = 1.0 - (1.0 - c.x_min) * hill
            if c.tau_x_down is None and c.tau_x_up is None:
                self.x_relay += (x_inf - self.x_relay) * (1.0 - np.exp(-dt / c.tau_x))   # symmetric (byte-parity)
            else:
                # FCXR-LC1 asymmetric: deplete (x_inf<x) on tau_x_down, recover (x_inf>=x) on tau_x_up.
                # Equal taus reduce EXACTLY to the symmetric relaxation above (per-element factor identical).
                tau_sel = np.where(x_inf < self.x_relay, c.tau_x_down, c.tau_x_up)
                self.x_relay += (x_inf - self.x_relay) * (1.0 - np.exp(-dt / tau_sel))
        if c.use_z:
            # z_inf = H(I_th_EI - I_I): 1 iff I_I < I_th_EI (strict); I_I >= I_th_EI -> 0 (deplete)
            z_inf_E = (self._z_sensor_last_E < c.I_th_EI).astype(float)
            zE = self.z[self.is_E]
            zE = zE + (dt / c.tau_z) * (z_inf_E - zE)
            self.z[self.is_E] = np.clip(zE, 0.0, 1.0)          # z in [0,1]
        if c.use_m and (c.m_enable_ms is None or self._step_i * dt >= c.m_enable_ms):
            # FCXR-HEO2: before m_enable_ms both decay AND accumulation are skipped -> m stays 0 (its init
            # value), so apply_currents/membrane_terms see no adaptation in the pre-enable window.
            mE = self.m[self.is_E]
            mE = mE - (mE / self._tau_E) * dt                  # decay (per-cell recovery time)
            self.m[self.is_E] = np.maximum(mE, 0.0)            # m >= 0
            self.m[spk & self.is_E] += 1.0                     # E spike -> +1 ; I spikes ignored (E-only)
            if c.m_mean_field:                                 # FCXR-HEO3 control: no inter-cell differences
                self.m[self.is_E] = self.m[self.is_E].mean()
        if c.m_frozen_E is not None and c.m_frozen_enable_ms is not None and self._step_i * dt >= c.m_frozen_enable_ms:
            self.m[:self.NE] = self._m_frozen_cached          # FCXR-HEO2.1: delayed static-K inject (idempotent)
        if c.use_phi:
            phiE = self.phi[self.is_E]
            phiE = phiE - (phiE / c.tau_phi) * dt
            self.phi[self.is_E] = np.maximum(phiE, 0.0)
            self.phi[spk & self.is_E] += c.delta_phi
        if c.use_pump:
            self._pump_step(spk, dt)
        self._record_traces(spk)
        # snapshot AFTER the slow update + trace record -> snapshots[label].z_E.mean() == trace_z_mean[step_i]
        if self._snap_steps is not None and self._step_i in self._snap_steps:
            self._capture(self._snap_steps[self._step_i])
        self._step_i += 1

    # ------------------------------------------------------------------ pump plugin (off by default)
    def _pump_excess_E(self):
        """Baseline-centered electrogenic pump current on E cells at the CURRENT load u(t^-), or None
        when the pump must not reach the membrane (off / sensor-only / after a scheduled knockout).

        Formula pinned to src/topic4_mz_fcxr_pump.excess_pump_current by
        tests/test_mz_slow_vars.py::test_membrane_uses_pre_step_load_and_step_applies_the_jump_after.
        NO positive part: phi<p0 gives a negative excess (activation below the baseline reference).
        """
        c = self.cfg
        if not c.use_pump or c.pump_sensor_only or c.pump_Imax <= 0.0:
            self._pump_excess_last = None
            return None
        if self._pump_knockout_step is not None and self._step_i >= self._pump_knockout_step:
            self._pump_excess_last = None                       # scheduled current knockout (u keeps evolving)
            return None
        uh = self.u_pump_E ** c.pump_h
        ex = c.pump_Imax * (uh / (1.0 + uh) - self._pump_p0_E)
        self._pump_excess_last = ex
        return ex

    def _pump_step(self, spk, dt):
        """Load mass balance at the LOCKED causal order (spec §2.2): the membrane above already used
        u(t^-); the clearance is evaluated at u(t^-) and the per-spike jump is added on top, so a
        spike generated this step only reaches the pump current from the NEXT step.

            u(t+dt) = max[0, u(t) + a_load*N_spike - (dt/tau_N)*phi(u(t))]

        The jump carries no dt; the clearance carries dt/tau_N. A one-shot ``set_load`` intervention
        registered for this step is applied LAST (first affected membrane call = step+1).
        """
        c = self.cfg
        u = self.u_pump_E
        if not np.all(np.isfinite(u)):                          # fail-fast: a blown-up candidate is failed
            raise FloatingPointError("non-finite activity-dependent pump load u")
        uh = u ** c.pump_h
        phi = uh / (1.0 + uh)                                   # phi(u(t^-)) == what the membrane used
        self._pump_phi_last = phi
        spkE = spk[:self.NE]
        if self.pump_phi_sum_E is not None:                     # calibration observer (pure side-effect)
            self.pump_phi_sum_E += phi
            self.pump_spike_count_E += spkE
            self.pump_phi_count += 1
        np.maximum(u + c.pump_a_load * spkE - (dt / c.pump_tau_ms) * phi, 0.0, out=u)
        fld = self._pump_set_load.get(self._step_i)
        if fld is not None:
            np.copyto(u, fld)                                   # scheduled load reset / sufficiency injection

    @staticmethod
    def _normalize_pump_interventions(items):
        """Validate scheduled interventions -> (knockout_step, {step: load_field}).

        INTEGER steps only (no float-time equality). Two primitives, both prefix-preserving:
          * ``pump_current_knockout``  membrane pump current = 0 from this step on; u keeps evolving.
                                       First affected membrane call = this step.
          * ``set_load``               one-shot u <- field at the END of this step (load reset,
                                       sufficiency injection, uniform/shuffle matched controls).
                                       First affected membrane call = this step + 1.
        """
        knockout, set_load = None, {}
        for item in (items or []):
            kind = item.get("kind")
            step = item.get("step")
            if not isinstance(step, (int, np.integer)) or bool(step != int(step)) or int(step) < 0:
                raise ValueError(f"pump intervention step must be a non-negative integer, got {step!r}")
            step = int(step)
            if kind == "pump_current_knockout":
                if knockout is not None:
                    raise ValueError("at most one pump_current_knockout may be scheduled")
                knockout = step
            elif kind == "set_load":
                fld = np.asarray(item["field"], float)
                if fld.ndim != 1 or not np.all(np.isfinite(fld)) or fld.min() < 0.0:
                    raise ValueError("set_load field must be a finite 1-D load field with values >= 0")
                if step in set_load:
                    raise ValueError(f"duplicate set_load intervention at step {step}")
                set_load[step] = fld
            else:
                raise ValueError(f"unknown pump intervention kind {kind!r}")
        return knockout, set_load

    # ------------------------------------------------------------------ traces
    def _record_traces(self, spk):
        zE = self.z[self.is_E]; mE = self.m[self.is_E]
        self.trace_z_mean.append(float(zE.mean()))
        self.trace_z_min.append(float(zE.min()))
        self.trace_m_mean.append(float(mE.mean()))
        self.trace_m_max.append(float(mE.max()))
        self.trace_phi_mean.append(float(self.phi[self.is_E].mean()))
        self.trace_phi_max.append(float(self.phi[self.is_E].max()))
        self.trace_adap_current.append(float((self._eta_E * mE).mean()))
        self.trace_I_EI_E_mean.append(float(self._I_I_last[self.is_E].mean()))
        ci, si = self.core_e_idx, self.surr_e_idx
        self.trace_z_core_mean.append(float(self.z[ci].mean()) if ci.size else float("nan"))
        self.trace_z_surround_mean.append(float(self.z[si].mean()) if si.size else float("nan"))
        self.trace_m_core_mean.append(float(self.m[ci].mean()) if ci.size else float("nan"))
        self.trace_m_surround_mean.append(float(self.m[si].mean()) if si.size else float("nan"))
        self.trace_rate_E.append(int(spk[self.is_E].sum()))
        self.trace_rate_I.append(int(spk[~self.is_E].sum()))
        self.trace_gaba_received_mean.append(float(self._gbar_last))
        self.trace_global_pre_z.append(float(self._g_global_pre_z_last))
        self.trace_z_sensor_mean.append(float(self._z_sensor_last_E.mean()))
        self.trace_gI_mean.append(float(self._gI_mean_last))
        self.trace_gI_max.append(float(self._gI_max_last))
        self.trace_gM_mean.append(float(self._gM_mean_last))
        self.trace_gM_max.append(float(self._gM_max_last))
        self.trace_tau_eff_ratio_mean.append(float(self._tau_ratio_mean_last))
        self.trace_tau_eff_ratio_min.append(float(self._tau_ratio_min_last))
        self.trace_conductance_clip_frac.append(float(self._clip_frac_last))
        # AMPA ff/rec conductance split is a full_conductance property (independent of the relay), so record
        # it whenever the membrane is full_conductance -> Stage 0 (use_x=False) can attribute ff vs rec drift.
        if self.cfg.membrane_mode == "full_conductance":
            self.trace_gEff_mean.append(float(self._gEff_mean_last))
            self.trace_gErec_mean.append(float(self._gErec_mean_last))
        if self.cfg.use_x:                                    # relay sensor/availability traces (relay only)
            self.trace_y_mean.append(float(self.y.mean()))
            self.trace_y_max.append(float(self.y.max()))
            self.trace_x_relay_mean.append(float(self.x_relay.mean()))
            self.trace_x_relay_min.append(float(self.x_relay.min()))
        if self.cfg.coop_A > 0.0:                             # FCXR-HEO1 cooperative engagement (coop on only)
            self.trace_coop_engaged_frac.append(float(self._coop_engaged_frac_last))
            self.trace_coop_H_mean.append(float(self._coop_H_mean_last))
        if self.cfg.use_pump:
            # u is POST-update (same convention as z/m); phi/excess are the values the membrane
            # actually USED at this step, i.e. evaluated at u(t^-) -- documented +-1 step offset.
            self.trace_u_mean.append(float(self.u_pump_E.mean()))
            self.trace_u_max.append(float(self.u_pump_E.max()))
            self.trace_phi_pump_mean.append(float(self._pump_phi_last.mean()))
            self.trace_phi_pump_max.append(float(self._pump_phi_last.max()))
            ex = self._pump_excess_last
            if ex is not None:                                # only when the pump reaches the membrane
                self.trace_pump_excess_mean.append(float(ex.mean()))
                self.trace_pump_excess_max.append(float(ex.max()))
                self.trace_pump_excess_min.append(float(ex.min()))

    def _record_calib(self, I_E, I_I):
        edges = self.cfg.calib_hist_edges
        if edges is None:
            return
        hI, _ = np.histogram(I_I[self.is_E], bins=edges)
        hE, _ = np.histogram(I_E[self.is_E], bins=edges)
        self.calib_hist_I_EI.append(hI.astype(np.int64))
        self.calib_hist_I_EE.append(hE.astype(np.int64))

    def _record_gerec_hist(self, gErec_raw):
        """FCXR-HEO1: accumulate the pooled (cell x step) gErec_raw distribution into fixed edges for
        overall / core / surround E cells. Pure side-effect (never alters the trajectory)."""
        edges = self.cfg.gerec_hist_edges
        self.gerec_hist_overall += np.histogram(gErec_raw, bins=edges)[0].astype(np.int64)
        ci, si = self.core_e_idx, self.surr_e_idx
        if ci.size:
            self.gerec_hist_core += np.histogram(gErec_raw[ci], bins=edges)[0].astype(np.int64)
        if si.size:
            self.gerec_hist_surround += np.histogram(gErec_raw[si], bins=edges)[0].astype(np.int64)

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
        """Copy ONLY z_E/m_E (E cells [:NE]) at the current step -> n_snapshots x NE (never n_steps x NE).
        FCXR-LC1: also copy the E->E relay x_E and its persistence sensor y_E when the relay is active, so
        D_X and regional (core/axis/off) x/y recruitment can be computed post-hoc from a few snapshots."""
        snap = dict(z_E=self.z[:self.NE].copy(), m_E=self.m[:self.NE].copy(),
                    step=int(self._step_i), captured_after_update=True)
        if self.cfg.use_x:
            snap["x_E"] = self.x_relay.copy()        # relay availability (length NE)
            snap["y_E"] = self.y.copy()              # persistence sensor y_j (length NE)
        if self.cfg.use_pump:
            snap["u_E"] = self.u_pump_E.copy()       # activity-dependent load (length NE)
            if self.pump_phi_sum_E is not None:      # CUMULATIVE sums -> per-block means by differencing
                snap["pump_phi_sum_E"] = self.pump_phi_sum_E.copy()
                snap["pump_spike_count_E"] = self.pump_spike_count_E.copy()
                snap["pump_phi_count"] = int(self.pump_phi_count)
        self.snapshots[label] = snap

    @property
    def n_steps_run(self):
        """Number of step() calls executed (== simulate_kick iterations run, honoring early-stop)."""
        return self._step_i

    def _validate_config(self):
        c = self.cfg
        if c.membrane_mode not in ("current", "conductance", "full_conductance"):
            raise ValueError("membrane_mode must be 'current', 'conductance' or 'full_conductance'")
        if not 0.0 <= c.global_gaba_fraction <= 1.0:
            raise ValueError("global_gaba_fraction must be in [0,1]")
        if c.global_gaba_mode not in ("replace", "additive"):
            raise ValueError("global_gaba_mode must be 'replace' or 'additive'")
        if c.z_scope not in ("total", "local_only"):
            raise ValueError("z_scope must be 'total' or 'local_only'")
        finite = (c.tau_z, c.I_th_EI, c.tau_adp, c.eta_m, c.tau_phi, c.delta_phi,
                  c.gaba_gain, c.m_conductance_gain, c.global_gaba_fraction,
                  c.v_match, c.e_gaba, c.e_k, c.max_total_conductance)
        if not all(np.isfinite(x) for x in finite[:-1]) or np.isnan(c.max_total_conductance):
            raise ValueError("MZ numeric configuration must be finite (max_total_conductance may be +inf)")
        if c.gaba_gain < 0.0 or c.m_conductance_gain < 0.0 or c.eta_m < 0.0:
            raise ValueError("conductance gains must be non-negative")
        if c.v_match <= c.e_gaba or c.v_match <= c.e_k:
            raise ValueError("v_match must exceed e_gaba and e_k for positive force matching")
        if c.max_total_conductance <= 0.0:
            raise ValueError("max_total_conductance must be positive")
        if c.use_z and c.tau_z <= 0.0:
            raise ValueError("use_z requires tau_z>0")
        if c.z_frozen_E is not None:
            zf = np.asarray(c.z_frozen_E, float)
            if zf.ndim != 1 or not np.all(np.isfinite(zf)) or zf.min() < 0.0 or zf.max() > 1.0:
                raise ValueError("z_frozen_E must be a finite 1-D field with values in [0,1]")
            if c.use_z:
                raise ValueError("z_frozen_E (frozen field) requires use_z=False; a frozen field must not evolve")
        if c.use_m and c.tau_adp <= 0.0:
            raise ValueError("use_m requires tau_adp>0")
        if c.m_enable_ms is not None and not c.use_m:
            raise ValueError("m_enable_ms (delayed adaptation onset) requires use_m=True")
        if c.m_frozen_E is not None:
            if c.membrane_mode != "full_conductance":
                raise ValueError("m_frozen_E (static-K control) requires membrane_mode='full_conductance'")
            if c.use_m or c.m_enable_ms is not None:
                raise ValueError("m_frozen_E requires use_m=False and m_enable_ms=None (a frozen field must not evolve)")
            mf = np.asarray(c.m_frozen_E, float)
            if mf.ndim != 1 or not np.all(np.isfinite(mf)) or mf.min() < 0.0:
                raise ValueError("m_frozen_E must be a finite 1-D field with values >= 0")
        if c.m_frozen_enable_ms is not None and c.m_frozen_E is None:
            raise ValueError("m_frozen_enable_ms (delayed static-K) requires m_frozen_E")
        for nm, fld, strict_pos in (("tau_adp_E", c.tau_adp_E, True), ("eta_m_E", c.eta_m_E, False)):
            if fld is None:
                continue
            v = np.asarray(fld, float)
            bad_sign = bool((v <= 0).any()) if strict_pos else bool((v < 0).any())
            if v.ndim != 1 or not bool(np.all(np.isfinite(v))) or bad_sign:
                raise ValueError(f"{nm} must be a finite 1-D field ({'>0' if strict_pos else '>=0'})")
        if (c.tau_adp_E is not None or c.eta_m_E is not None or c.m_mean_field) and not c.use_m:
            raise ValueError("tau_adp_E / eta_m_E / m_mean_field require use_m=True")
        if c.use_phi and (c.tau_phi <= 0.0 or c.delta_phi < 0.0):
            raise ValueError("use_phi requires tau_phi>0 and delta_phi>=0")
        if c.membrane_mode == "full_conductance":
            if not (np.isfinite(c.E_E) and np.isfinite(c.c_E)):
                raise ValueError("full_conductance requires finite E_E and c_E")
            if c.E_E <= c.v_match:
                raise ValueError("full_conductance requires E_E > v_match for positive AMPA force matching")
            if c.c_E < 0.0:
                raise ValueError("c_E must be non-negative")
        if c.rec_sat_g < 0.0:
            raise ValueError("rec_sat_g must be non-negative (0 = off)")
        if c.rec_sat_g > 0.0 and not (c.membrane_mode == "full_conductance" and c.rec_conductance):
            raise ValueError("rec_sat_g>0 (recurrent smooth saturation) requires full_conductance + rec_conductance")
        if c.coop_A < 0.0:
            raise ValueError("coop_A must be non-negative (0 = cooperative gate off)")
        if c.coop_A > 0.0:
            # cooperative gain with no saturation would be unbounded -> require the tanh anchor + recurrent path
            if not (c.membrane_mode == "full_conductance" and c.rec_conductance and c.rec_sat_g > 0.0):
                raise ValueError("coop_A>0 (cooperative recurrent gate) requires full_conductance + "
                                 "rec_conductance + rec_sat_g>0 (bounded by saturation)")
            if c.coop_uc <= 0.0 or c.coop_Kc <= 0.0:
                raise ValueError("coop_A>0 requires coop_uc>0 and coop_Kc>0")
            if int(c.coop_n) < 1:
                raise ValueError("coop_A>0 requires coop_n>=1")
        if c.use_x:
            if c.membrane_mode != "full_conductance":
                raise ValueError("use_x (E->E relay) requires membrane_mode='full_conductance'")
            if c.tau_y <= 0.0 or c.tau_x <= 0.0:
                raise ValueError("use_x requires tau_y>0 and tau_x>0")
            if not 0.0 <= c.x_min <= 1.0:
                raise ValueError("x_min must be in [0,1]")
            if c.K_y <= 0.0:
                raise ValueError("use_x requires K_y>0")
            if int(c.hill_n) < 1:
                raise ValueError("use_x requires hill_n>=1")
            if not np.isfinite(c.y_gate):
                raise ValueError("y_gate must be finite")
            # FCXR-LC1 asymmetric relay kinetics: both-or-neither + positive (design invariant
            # tau_x_down < tau_z <= tau_x_up is a RUN-parameter check enforced by the runner, not here).
            if (c.tau_x_down is not None) or (c.tau_x_up is not None):
                if c.tau_x_down is None or c.tau_x_up is None:
                    raise ValueError("asymmetric relay kinetics require BOTH tau_x_down and tau_x_up (or neither)")
                if c.tau_x_down <= 0.0 or c.tau_x_up <= 0.0:
                    raise ValueError("tau_x_down and tau_x_up must be > 0 (asymmetric relay kinetics)")
        # ---- FCXR pump lifecycle (per-E activity-dependent load u_i + electrogenic pump) ----
        if not c.use_pump:
            if (c.pump_sensor_only or c.pump_Imax > 0.0 or c.pump_record_calibration
                    or c.pump_interventions or c.pump_u_init_E is not None):
                raise ValueError("pump_* options require use_pump=True")
        else:
            if not (c.pump_tau_ms > 0.0 and np.isfinite(c.pump_tau_ms)):
                raise ValueError("use_pump requires a finite pump_tau_ms>0 (load release time)")
            if not (c.pump_a_load >= 0.0 and np.isfinite(c.pump_a_load)):
                raise ValueError("pump_a_load must be finite and >= 0")
            if not (c.pump_Imax >= 0.0 and np.isfinite(c.pump_Imax)):
                raise ValueError("pump_Imax must be finite and >= 0")
            if int(c.pump_h) < 1:
                raise ValueError("pump_h must be >= 1 (primary tier fixes 3)")
            if c.pump_sensor_only and c.pump_Imax > 0.0:
                raise ValueError("pump_sensor_only requires pump_Imax=0 (the membrane must stay untouched)")
