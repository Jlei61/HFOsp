"""M3A-v2 spatial slow-variable field (off-by-default; plugs into simulate_kick).

Spatial generalization of `slow_vars.RegionalResource` (§B2): the two SCALAR tanks
q_core/q_global become two spatial FIELDS on an n_grid x n_grid lattice over the
L x L sheet --

    q_I(x,t)  inhibitory-resource field   (depletes with local activity, refills to 1)
    g_K(x,t)  fatigue / sAHP field         (builds with local E activity, decays to 0)

so "axis fatigues while off-axis permissiveness rises" becomes representable, which a
two-scalar model cannot carry (§B5.0). Each neuron is coupled at its own position:

    I_net,i = I_E,i - q_I(x_i,t) * I_I,i - eta_K * g_K(x_i,t)        (i in E)
    I_net,i = I_E,i - I_I,i                                          (i in I)

OFF-BY-DEFAULT: k_q=0, k_K=0, q_init=1  =>  q_I==1, g_K==0  =>  apply_currents returns
I_E - I_I  =>  BYTE-PARITY with slow=None (the BASELINE_SHA regression).

Canonical math: docs/snn_core_model_equations.md §B5.
Plan / TDD:     docs/superpowers/plans/2026-06-28-sef-hfo-m3a-v2-spatial-slowvar-field-plan.md

STATUS: IMPLEMENTED (2026-06-28; Tasks 1-5 green). The default values in
SpatialSlowFieldConfig ARE the locked spec defaults (§B5.2-B5.3). Off-by-default
(k_q=0, k_K=0, q_init=1) leaves q_I==1, g_K==0 -> byte-identical to slow=None.
Mechanism SCREEN only: the four-state label is a detector, not a seizure claim;
calibration / ablation deferred (plan "Deferred").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.sef_hfo_field import isotropic_gaussian, convolve_periodic
from src.topic4_m3a_v2_2_sensors import global_M, global_B, global_participation, chi_G  # §B6 sensors
from src.sef_hfo_m4_load_shunt import LoadShuntParams, load_shunt_step  # M4-3A n->a load/shunt ODE
from src.snn_engine.zm_conductance import (
    ZMConductanceConfig,
    conductance_membrane_step,
    state_dependent_homotopy_step,
)


# ---------------------------------------------------------------------------
# Config. Data only -- defaults are the §B5.2/§B5.3 locked spec values. The
# structural invariant sigma_q > sigma_K (wide disinhibition footprint, narrow
# fatigue footprint, §B5.3) and eta_I >= eta_E are enforced by validate().
# ---------------------------------------------------------------------------
@dataclass
class SpatialSlowFieldConfig:
    # ---- field grid ----
    n_grid: int = 32          # field lattice resolution per axis (n_grid x n_grid)
    # ---- firing-rate field K_r (spikes -> r_E, r_I), §B5.1 ----
    sigma_r: float = 0.5      # mm, spatial smoothing of spikes into the rate field
    tau_a: float = 100.0      # ms, temporal EMA of the rate field
    # ---- q_I(x,t) inhibitory-resource field, §B5.2 ----
    use_qI: bool = True
    tau_q: float = 5000.0     # ms, recovery toward q_I=1
    k_q: float = 0.0          # depletion rate; 0 -> OFF -> byte-parity
    q_min: float = 0.25       # floor
    q_init: float = 1.0       # initial / full-tank value
    sigma_q: float = 1.5      # mm, K_q width (WIDE; must be > sigma_K)
    eta_E: float = 0.3        # r_E weight in a_q
    eta_I: float = 1.0        # r_I weight in a_q (>= eta_E: resource tracks inhib usage)
    a0_q: float = 0.0         # saturation onset
    a50_q: float = 1.0        # saturation half-saturation
    # ---- g_K(x,t) fatigue / sAHP field, §B5.3 ----
    use_gK: bool = True
    tau_K: float = 5000.0     # ms, g_K decay
    k_K: float = 0.0          # build-rate STRENGTH knob; 0 -> OFF -> byte-parity
    gK_max: float = 1.0       # ceiling
    sigma_K: float = 0.5      # mm, K_K width (NARROW; must be < sigma_q)
    eta_K: float = 1.0        # coupling strength of g_K into the membrane
    a0_K: float = 0.0
    a50_K: float = 1.0
    # ---- h_G(t) global inhibitory recovery scalar (M3A-v2.2, §B6) ----
    use_hG: bool = False       # OFF by default -> h_G stays hG_init -> byte-parity
    eta_G: float = 0.0         # coupling of h_G into E membrane
    tau_G: float = 600.0       # ms, h_G decay
    k_G: float = 0.0           # build-rate STRENGTH knob; 0 -> no build (still decays)
    hG_max: float = 1.0        # ceiling
    hG_init: float = 0.0       # initial h_G (parity requires 0)
    # ---- h_G sensor (fast EMA + soft-area + Hill thresholds), §B6 ----
    tau_s: float = 15.0        # ms, FAST EMA for the recovery sensor (separate from tau_a)
    r_A: float = 0.0           # soft recruited-area reference level (B sensor)
    Delta_A: float = 1.0       # soft recruited-area slope (B sensor)
    M50: float = 1.0           # Hill half-trigger for M
    B50: float = 0.5           # Hill half-trigger for B
    Pi50: float = 0.45         # Hill half-trigger for Pi
    n_M: float = 4.0           # Hill exponent M
    n_B: float = 4.0           # Hill exponent B
    n_Pi: float = 4.0          # Hill exponent Pi
    # ---- optional q_I replenish driven by h_G (arm F only; ablated separately) ----
    lambda_G: float = 0.0      # 0 -> primary arm E; >0 -> arm F
    # ---- proxy phase-plane h_G term ----
    beta_G: float = 1.0        # weight in Y_new = P_global - beta_G*h_G
    # ---- M4 divisive shared inhibitory pool S_G (rev4 spec §3-§5; OFF by default -> byte-parity) ----
    use_SG: bool = False       # master gate; False -> no pool alloc/evolution, apply_currents unchanged
    alpha_G: float = 0.0       # divisive strength: I_rec_E -> I_rec_E/(1+alpha_G*S_G)
    beta_SG: float = 0.0       # OPTIONAL small subtractive pool current (arm 1/3); NOT beta_G (h_G proxy)
    r0_psi: float = 0.0        # Psi_G recruitment onset
    r50_psi: float = 1.0       # Psi_G half-recruitment
    n_psi: float = 2.0         # Psi_G steepness
    p_pool: float = 3.0        # A_G p-norm exponent (2-4 focal; 1 = area/mean); swept diagnostic
    tau_mu: float = 40.0       # ms, pool activation low-pass (fast)
    tau_S: float = 120.0       # ms, pool output low-pass
    S_max: float = 1.0         # pool output ceiling
    clamp_SG: float = None     # mechanism control: if set, S_G is FROZEN at this value (static-pool arm); None -> normal dynamics
    # ---- n(x,t) load -> a(x,t) shunt field, M4-3A ----
    use_A: bool = False        # master gate; False -> byte-parity
    k_n: float = 0.0           # load build rate; 0 -> OFF -> byte-parity
    tau_n: float = 20000.0     # ms, load recovery (SLOW; keep > tau_q)
    rho_n: float = 0.0         # load consumption via Pi(n)
    n_base: float = 0.0        # Hill offset / baseline load
    n50: float = 0.5           # Hill half-point
    hill_h: float = 2.0        # Hill exponent
    a_max: float = 1.0         # shunt ceiling
    alpha_A: float = 0.0       # divisive conductance gain: g_A = alpha_A * a
    eta_A: float = 0.0         # subtractive bias gain: -eta_A * a (E cells)
    sigma_n: float = 1.5       # mm, K_n width (default = sigma_q, WIDE)
    u_n0: float = 0.0          # baseline drive set-point (constant, from Arm0)
    n_min: float = 0.0
    n_max: float = 10.0
    g_A_max: float = 20.0      # conductance cap
    # ---- persistence-gated local recovery field p(x,t) (SNN-native M4 exit, spec 2026-07-21 §5) ----
    # p is a SLOW leaky integral of supra-theta_p local activity: short IEDs (<< tau_p) barely charge it,
    # a sustained bounded state saturates it. It gates a LOCAL OUTWARD recovery current on E cells only
    # (g_K-type actuator, right direction per M3A Step-3 lineage; but persistence-gated, never before tested).
    use_persist: bool = False  # master gate; False -> no p coupling / no advance -> byte-parity
    tau_p: float = 5000.0      # ms, persistence leak (>> IED duration -> duration selectivity; ~ q_I refill)
    theta_p: float = 0.0       # activity threshold: p charges only where K_p*r_E > theta_p
    a50_p: float = 1.0         # Psi half-saturation: Psi=[a-theta]_+/(a50+[a-theta]_+)
    sigma_p: float = 1.5       # mm, K_p persistence-sensor footprint
    eta_r: float = 0.0         # recovery-current strength (mV); 0 -> OFF -> byte-parity
    p50_r: float = 0.0         # Phi(p) Hill half-point; <=0 -> linear Phi(p)=p
    n_r: float = 2.0           # Phi(p) Hill exponent (used only if p50_r>0)
    p_init: float = 0.0        # initial p (parity requires 0 when off)
    clamp_persist: float = None  # open-loop probe / E4 ablation: freeze p at this value (None -> normal dynamics)
    tau_p_down: float = None   # asymmetric decay time (ms); None -> symmetric (== tau_p). When set, p CHARGES with
                               # tau_p (fast, short ictal) but DECAYS with tau_p_down (slow) once activity drops --
                               # a long hold that lets q_I refill (continuous analog of R4's active-low M freeze).
    persist_onset_ms: float = 0.0  # p stays 0 (no accumulation, no current) until t >= this. 0 -> active from t=0.
                               # >0 -> established-state fork: let the M4 state FORM first, THEN engage the recovery
                               # current -> distinguishes termination-of-formed-state from prevention-of-formation.
    # ---- containment memory H (Phase-3 vNext, spec §6 review 2026-07-22; scalar; OFF by default -> byte-parity).
    # H builds with the mean recovery-current gate <Phi(p)> and decays with tau_H (tau_H dH/dt = <Phi(p)> - H),
    # so it HOLDS the divisive containment through the q_I-refill window AFTER activity (and thus the activity-
    # driven S_G) drops: I_EE_eff = I_EE / (1 + alpha_G*S_G + alpha_H*H). Requires use_SG (kick_probe tracks the
    # recurrent-only current I_E_rec only when use_SG; kick_probe is guarded, so H piggybacks on that path). ----
    use_H: bool = False        # master gate; False -> H not in the denominator, no advance -> byte-parity
    alpha_H: float = 0.0       # divisive coupling of H (adds to the 1+alpha_G*S_G denominator)
    tau_H: float = 5000.0      # ms, H build/decay time (SLOW; must outlast the q_I-refill so containment holds)
    H_init: float = 0.0        # initial H (parity requires 0)
    H_max: float = 1.0         # ceiling (<Phi(p)> in [0,1] -> H in [0,1])
    H_sensor: str = "global"   # H build drive: 'global' = spatial-mean Phi(p) (broad q_I state; DEFAULT -> the
                               # existing byte-parity path); 'active' = 90th-pct of Phi(p) = active-focus intensity,
                               # NOT diluted by inactive cortex (needed for a spatially-localized Z/M bursting focus,
                               # where the global mean starves H -- Z/M migration 2026-07-22)
    # ---- Z/M per-neuron slow variables (ported byte-identical from mz_slow_vars.py; Z/M migration 2026-07-22).
    # Per-E-cell z_i (inhibitory efficacy: tau_z dz/dt = H(I_th_EI - I_I^E) - z; effective E inhibition = z_i*I_I)
    # and m_i (spike-frequency adaptation: dm/dt = -m/tau_adp + spikes; current eta_m*m_i subtracted). Both act on
    # E CELLS ONLY (z=1, m=0 on I cells). Both OFF -> byte-parity with slow=None. For the pure Z/M substrate set
    # use_qI=False (q_I frozen at 1) so z_i*q_I*I_I == z_i*I_I, matching mz's z*I_I exactly. Defaults == mz. ----
    use_z: bool = False        # master gate; False -> z stays 1 -> inhibition unscaled -> byte-parity
    use_m: bool = False        # master gate; False -> no adaptation current -> byte-parity
    tau_z: float = 5000.0      # ms, z (inhibitory-efficacy) time constant
    I_th_EI: float = 0.0       # z_inf = H(I_th_EI - I_I): z depletes where E-cell inhibitory current I_I >= I_th_EI
    tau_adp: float = 2000.0    # ms, m (adaptation) decay time
    eta_m: float = 0.0         # adaptation-current strength (mV per unit m)
    # ---- state-selective local recurrent feedback for the next lifecycle prototype.
    # A local activity memory h_mode(x) is always causal and detector-free.  Its
    # recurrent-E gain is exactly zero at healthy zeta=0, opens as z depletes,
    # and closes as the native per-E M state accumulates:
    # I_EE_eff = I_EE * (1 + rho*H*S_zeta*S_M) / (1 + alpha_G*S_G).
    use_mode_H: bool = False
    tau_mode_H: float = 250.0       # ms; bridges sub-second burst gaps
    tau_mode_H_down: float = 250.0  # ms; set slower than tau_mode_H to retain gain across gaps
    theta_mode_H_hz: float = 40.0   # local rE drive onset
    half_mode_H_hz: float = 40.0    # drive half-saturation above onset
    rho_mode_H: float = 0.0         # maximum multiplicative recurrent-E increment
    mode_H_common_subtraction: float = 0.0  # 0=native local H; 1=remove its E-population common mode
    z_mode_base: float = 1.0
    z_mode_susceptible: float = 0.50
    zeta_mode_center: float = 0.50
    zeta_mode_slope: float = 0.10
    m_mode_half: float = 45.0
    m_mode_power: float = 4.0
    # Optional collective M brake on the *same recurrent-E denominator*.
    # The p-norm is detector-free and has no anatomical mask: recruited cells
    # with high native m can stabilise the whole pathological mode instead of
    # merely closing H locally and letting amplification escape to surround.
    use_mode_M_divisive: bool = False
    kappa_mode_M: float = 0.0
    m_mode_div_ref: float = 30.0
    m_mode_div_power: float = 4.0
    m_mode_div_hill_power: float = 4.0
    use_mode_M_memory: bool = False
    tau_mode_M_memory_up: float = 3000.0
    tau_mode_M_memory_down: float = 8000.0
    # ---- Phase-D fast carrier: E-only dynamic-threshold INCREMENT.
    # This is not the legacy absolute-threshold SlowVars.phi. The heterogeneous
    # V_th substrate remains the base and phi_increment is added on top. ----
    use_phi: bool = False       # OFF -> threshold() is a literal passthrough
    tau_phi: float = 100.0      # ms, exact exponential recovery
    delta_phi: float = 0.0      # mV added after each E spike
    # ---- lifecycle prototype: local inhibitory-state mechanisms (OFF by default).
    # I->E depression is presynaptic and therefore coupled at the inhibitory
    # delay-ring scatter, not as a postsynaptic scalar in apply_currents. ----
    use_i2e_depression: bool = False
    tau_i2e_depression: float = 300.0  # ms, recovery of per-I-neuron resource toward one
    U_i2e_depression: float = 0.0      # fractional resource use per I spike
    d_i2e_min: float = 0.20            # safety floor; I->I edges are never scaled
    i2e_tau_cv: float = 0.0            # quenched per-I recovery heterogeneity
    i2e_tau_seed: int = 0
    use_i_adaptation: bool = False
    tau_i_adaptation: float = 300.0    # ms, exact recovery of I threshold increment
    delta_i_adaptation: float = 0.0    # mV added after each I spike
    # ---- Phase-D unit-safe conductance membrane (OFF by default).
    # The existing mV-equivalent drives are converted by kappa before entering
    # a conductance sum; the old S_G recurrent-E divisor is forbidden here. ----
    use_zm_conductance: bool = False
    # Phase-E baseline-preserving variant.  High-z cells retain the native
    # current membrane exactly; only depleted cells cross to conductance form.
    use_zm_conductance_homotopy: bool = False
    cond_homotopy_z_native: float = 0.60
    cond_homotopy_z_conductance: float = 0.40
    cond_kappa_E: float = 0.1
    cond_kappa_I: float = 0.25
    cond_g_M: float = 0.001 / 15.0
    cond_gamma: float = 0.0
    cond_z_spares_global: bool = False
    cond_g_L: float = 1.0
    cond_E_L: float = 0.0
    cond_E_E: float = 25.0
    cond_E_I: float = 11.0
    cond_E_K: float = 0.0
    cond_tau_m_E: float = 20.0

    def validate(self) -> None:
        """Raise ValueError on any breached structural invariant (§B5.2-B5.3):
        sigma_q > sigma_K, eta_I >= eta_E, q_min in [0,1], gK_max >= 0, n_grid >= 2.
        (q_min=0 is valid: full depletion; §B5.2 only requires q_min <= q_I <= 1.)"""
        if not (self.sigma_q > self.sigma_K):
            raise ValueError(f"sigma_q ({self.sigma_q}) must be > sigma_K ({self.sigma_K}) "
                             "(wide disinhibition footprint, narrow fatigue footprint; §B5.3)")
        if self.eta_I < self.eta_E:
            raise ValueError(f"eta_I ({self.eta_I}) must be >= eta_E ({self.eta_E}) (§B5.2)")
        if not (0.0 <= self.q_min <= 1.0):
            raise ValueError(f"q_min must be in [0, 1], got {self.q_min}")
        if self.gK_max < 0.0:
            raise ValueError(f"gK_max must be >= 0, got {self.gK_max}")
        if self.n_grid < 2:
            raise ValueError(f"n_grid must be >= 2, got {self.n_grid}")
        # ---- h_G (M3A-v2.2, §B6) ----
        if self.tau_s <= 0.0:
            raise ValueError(f"tau_s must be > 0, got {self.tau_s}")
        if self.tau_G <= 0.0:
            raise ValueError(f"tau_G must be > 0, got {self.tau_G}")
        if self.k_G < 0.0:
            raise ValueError(f"k_G must be >= 0, got {self.k_G}")
        if self.hG_max < 0.0:
            raise ValueError(f"hG_max must be >= 0, got {self.hG_max}")
        if self.eta_G < 0.0:
            raise ValueError(f"eta_G must be >= 0, got {self.eta_G}")
        if self.lambda_G < 0.0:
            raise ValueError(f"lambda_G must be >= 0, got {self.lambda_G}")
        if self.Delta_A <= 0.0:
            raise ValueError(f"Delta_A must be > 0, got {self.Delta_A}")
        for nm, v in (("M50", self.M50), ("B50", self.B50), ("Pi50", self.Pi50),
                      ("n_M", self.n_M), ("n_B", self.n_B), ("n_Pi", self.n_Pi)):
            if v <= 0.0:
                raise ValueError(f"{nm} must be > 0, got {v}")
        # ---- M4 pool (rev4 §3-§5) ----
        if self.alpha_G < 0.0:
            raise ValueError(f"alpha_G must be >= 0, got {self.alpha_G}")
        if self.beta_SG < 0.0:
            raise ValueError(f"beta_SG must be >= 0, got {self.beta_SG}")
        for nm, v in (("p_pool", self.p_pool), ("tau_mu", self.tau_mu), ("tau_S", self.tau_S),
                      ("S_max", self.S_max), ("r50_psi", self.r50_psi), ("n_psi", self.n_psi)):
            if v <= 0.0:
                raise ValueError(f"{nm} must be > 0, got {v}")
        # ---- M4-3A n->a load/shunt field ----
        if self.use_A and self.sigma_n <= 0.0:
            raise ValueError("sigma_n must be > 0 when use_A")
        # ---- persistence-gated recovery field (spec 2026-07-21 §5); checked only when the field is on ----
        if self.use_persist:
            if self.tau_p <= 0.0:
                raise ValueError(f"tau_p must be > 0, got {self.tau_p}")
            if self.a50_p <= 0.0:
                raise ValueError(f"a50_p must be > 0, got {self.a50_p}")
            if self.sigma_p <= 0.0:
                raise ValueError(f"sigma_p must be > 0, got {self.sigma_p}")
            if self.eta_r < 0.0:
                raise ValueError(f"eta_r must be >= 0, got {self.eta_r}")
            if self.n_r <= 0.0:
                raise ValueError(f"n_r must be > 0, got {self.n_r}")
            if not (0.0 <= self.p_init <= 1.0):
                raise ValueError(f"p_init must be in [0, 1], got {self.p_init}")
            if self.clamp_persist is not None and not (0.0 <= self.clamp_persist <= 1.0):
                raise ValueError(f"clamp_persist must be in [0, 1], got {self.clamp_persist}")
            if self.tau_p_down is not None and self.tau_p_down <= 0.0:
                raise ValueError(f"tau_p_down must be > 0 when set, got {self.tau_p_down}")
            if self.persist_onset_ms < 0.0:
                raise ValueError(f"persist_onset_ms must be >= 0, got {self.persist_onset_ms}")
        # ---- containment memory H (Phase-3 vNext) ----
        if self.use_H:
            if not self.use_SG:
                raise ValueError("use_H requires use_SG (H rides the recurrent-divisive term; kick_probe "
                                 "tracks I_E_rec only when use_SG)")
            if self.tau_H <= 0.0:
                raise ValueError(f"tau_H must be > 0, got {self.tau_H}")
            if self.alpha_H < 0.0:
                raise ValueError(f"alpha_H must be >= 0, got {self.alpha_H}")
            if self.H_sensor not in ("global", "active"):
                raise ValueError(f"H_sensor must be 'global' or 'active', got {self.H_sensor!r}")
        # ---- Z/M per-neuron slow variables (ported from mz_slow_vars.py) ----
        if self.use_z and self.tau_z <= 0.0:
            raise ValueError(f"tau_z must be > 0 when use_z, got {self.tau_z}")
        if self.use_m:
            if self.tau_adp <= 0.0:
                raise ValueError(f"tau_adp must be > 0 when use_m, got {self.tau_adp}")
            if self.eta_m < 0.0:
                raise ValueError(f"eta_m must be >= 0 when use_m, got {self.eta_m}")
        if self.use_mode_H:
            if not self.use_SG:
                raise ValueError("use_mode_H requires use_SG so recurrent-only E current is available")
            if not self.use_z or not self.use_m:
                raise ValueError("use_mode_H requires the native Z and M coordinates")
            positive = {
                "tau_mode_H": self.tau_mode_H,
                "tau_mode_H_down": self.tau_mode_H_down,
                "half_mode_H_hz": self.half_mode_H_hz,
                "zeta_mode_slope": self.zeta_mode_slope,
                "m_mode_half": self.m_mode_half,
                "m_mode_power": self.m_mode_power,
            }
            for name, value in positive.items():
                if not np.isfinite(value) or value <= 0.0:
                    raise ValueError(f"{name} must be finite and >0")
            if not np.isfinite(self.theta_mode_H_hz) or self.theta_mode_H_hz < 0.0:
                raise ValueError("theta_mode_H_hz must be finite and >=0")
            if not np.isfinite(self.rho_mode_H) or self.rho_mode_H < 0.0:
                raise ValueError("rho_mode_H must be finite and >=0")
            if (
                not np.isfinite(self.mode_H_common_subtraction)
                or not 0.0 <= self.mode_H_common_subtraction <= 1.0
            ):
                raise ValueError("mode_H_common_subtraction must lie in [0,1]")
            if not (np.isfinite(self.z_mode_base) and np.isfinite(self.z_mode_susceptible)
                    and self.z_mode_base > self.z_mode_susceptible):
                raise ValueError("z_mode_base must exceed z_mode_susceptible")
            if not np.isfinite(self.zeta_mode_center) or not 0.0 < self.zeta_mode_center < 1.0:
                raise ValueError("zeta_mode_center must lie in (0,1)")
        if self.use_mode_M_divisive:
            if not self.use_SG or not self.use_m:
                raise ValueError("use_mode_M_divisive requires recurrent-only current and native M")
            for name, value in {
                "m_mode_div_ref": self.m_mode_div_ref,
                "m_mode_div_power": self.m_mode_div_power,
                "m_mode_div_hill_power": self.m_mode_div_hill_power,
            }.items():
                if not np.isfinite(value) or value <= 0.0:
                    raise ValueError(f"{name} must be finite and >0")
            if not np.isfinite(self.kappa_mode_M) or self.kappa_mode_M < 0.0:
                raise ValueError("kappa_mode_M must be finite and >=0")
            if self.use_mode_M_memory:
                for name, value in {
                    "tau_mode_M_memory_up": self.tau_mode_M_memory_up,
                    "tau_mode_M_memory_down": self.tau_mode_M_memory_down,
                }.items():
                    if not np.isfinite(value) or value <= 0.0:
                        raise ValueError(f"{name} must be finite and >0")
        if self.use_phi:
            if self.tau_phi <= 0.0:
                raise ValueError(
                    f"tau_phi must be > 0 when use_phi, got {self.tau_phi}"
                )
            if self.delta_phi < 0.0:
                raise ValueError(
                    f"delta_phi must be >= 0 when use_phi, got {self.delta_phi}"
                )
        if self.use_i2e_depression:
            if self.tau_i2e_depression <= 0.0:
                raise ValueError("tau_i2e_depression must be > 0")
            if not (0.0 <= self.U_i2e_depression < 1.0):
                raise ValueError("U_i2e_depression must lie in [0,1)")
            if not (0.0 < self.d_i2e_min <= 1.0):
                raise ValueError("d_i2e_min must lie in (0,1]")
            if not np.isfinite(self.i2e_tau_cv) or self.i2e_tau_cv < 0.0:
                raise ValueError("i2e_tau_cv must be finite and >=0")
        if self.use_i_adaptation:
            if self.tau_i_adaptation <= 0.0:
                raise ValueError("tau_i_adaptation must be > 0")
            if self.delta_i_adaptation < 0.0:
                raise ValueError("delta_i_adaptation must be >= 0")
        if self.use_zm_conductance:
            if not self.use_z or not self.use_m:
                raise ValueError(
                    "use_zm_conductance requires use_z=True and use_m=True"
                )
            forbidden = {
                "use_qI": self.use_qI,
                "use_gK": self.use_gK,
                "use_hG": self.use_hG,
                "use_SG": self.use_SG,
                "use_A": self.use_A,
                "use_persist": self.use_persist,
                "use_H": self.use_H,
            }
            active = [name for name, value in forbidden.items() if value]
            if active:
                raise ValueError(
                    "use_zm_conductance requires the clean Z/M substrate; "
                    f"disable {active}"
                )
            self.zm_conductance_config().validate()
        if self.use_zm_conductance_homotopy:
            if self.use_zm_conductance:
                raise ValueError(
                    "full conductance and conductance homotopy are mutually exclusive"
                )
            if not self.use_z or not self.use_m:
                raise ValueError(
                    "use_zm_conductance_homotopy requires use_z=True and use_m=True"
                )
            forbidden = {
                "use_qI": self.use_qI,
                "use_gK": self.use_gK,
                "use_hG": self.use_hG,
                "use_A": self.use_A,
                "use_persist": self.use_persist,
                "use_H": self.use_H,
                "use_mode_M_divisive": self.use_mode_M_divisive,
            }
            active = [name for name, value in forbidden.items() if value]
            if active:
                raise ValueError(
                    "conductance homotopy requires the clean Z/M fast path; "
                    f"disable {active}"
                )
            if not (
                0.0 <= self.cond_homotopy_z_conductance
                < self.cond_homotopy_z_native <= 1.0
            ):
                raise ValueError(
                    "homotopy thresholds must satisfy 0 <= z_conductance < "
                    "z_native <= 1"
                )
            self.zm_conductance_config().validate()

    def zm_conductance_config(self) -> ZMConductanceConfig:
        """Materialise the pure, immutable conductance configuration."""
        return ZMConductanceConfig(
            kappa_E=self.cond_kappa_E,
            kappa_I=self.cond_kappa_I,
            g_M=self.cond_g_M,
            gamma=self.cond_gamma,
            z_spares_global=self.cond_z_spares_global,
            g_L=self.cond_g_L,
            E_L=self.cond_E_L,
            E_E=self.cond_E_E,
            E_I=self.cond_E_I,
            E_K=self.cond_E_K,
            tau_m_E=self.cond_tau_m_E,
        )


# ---------------------------------------------------------------------------
# Stateless helpers (unit-testable in isolation), §B5.1-B5.2.
# ---------------------------------------------------------------------------
def saturation(a, a0, a50):
    """f(a) = [a-a0]_+ / (a50 + [a-a0]_+).  Hill-like; f(a0)=0, f(a0+a50)=0.5,
    f -> 1 as a -> inf. Elementwise on arrays. Implemented per the plan (Task 2)."""
    x = np.maximum(np.asarray(a, dtype=float) - a0, 0.0)   # [a - a0]_+
    return x / (a50 + x)


def zero_baseline_sigmoid(x, center, slope):
    """Smooth susceptibility gate with the exact scientific baseline S(0)=0."""
    x = np.asarray(x, dtype=float)
    raw0 = 1.0 / (1.0 + np.exp(float(center) / float(slope)))
    arg = np.clip((x - float(center)) / float(slope), -60.0, 60.0)
    raw = 1.0 / (1.0 + np.exp(-arg))
    out = (raw - raw0) / (1.0 - raw0)
    out = np.where(x <= 0.0, 0.0, out)
    return np.clip(out, 0.0, 1.0)


def local_rate_field_hz(rate_field, *, dt_ms, n_population, n_grid):
    """Convert the legacy per-step grid count field to approximate local Hz.

    ``firing_rate_field`` deliberately returns spatially smoothed spike counts
    per integration step because the older q/S_G mechanisms were calibrated in
    that native scale.  A threshold named in Hz must therefore divide by both
    the step duration and the expected population per grid cell.  Positions are
    near-uniform on this sheet; the conversion leaves all legacy fields intact.
    """
    if not np.isfinite(dt_ms) or float(dt_ms) <= 0.0:
        raise ValueError("dt_ms must be finite and positive")
    if int(n_population) <= 0 or int(n_grid) <= 0:
        raise ValueError("n_population and n_grid must be positive")
    neurons_per_cell = float(n_population) / float(int(n_grid) ** 2)
    return np.asarray(rate_field, dtype=float) * (1000.0 / float(dt_ms)) / neurons_per_cell


def aq_drive(rE, rI, eta_E, eta_I):
    """Pre-convolution q_I depletion drive: eta_E*r_E + eta_I*r_I (§B5.2). The inhibitory
    resource depletes mainly with inhibitory USE, so eta_I >= eta_E (config.validate enforces
    it). Factored out + unit-pinned so `step` cannot silently drop the r_I term and degrade
    into pure-E depletion. Implemented per the plan (Task 5)."""
    return eta_E * np.asarray(rE, float) + eta_I * np.asarray(rI, float)


def recover_i2e_resource(resource, dt_ms, tau_ms):
    """Exact between-spike recovery for tau*d' = 1-d."""
    d = np.asarray(resource, dtype=float)
    tau = np.asarray(tau_ms, dtype=float)
    if np.any(tau <= 0.0) or not np.all(np.isfinite(tau)):
        raise ValueError("tau_ms must be finite and >0")
    return 1.0 - (1.0 - d) * np.exp(-float(dt_ms) / tau)


def deplete_i2e_resource(resource, use_fraction, d_min):
    """Multiplicative presynaptic use with a strictly positive safety floor."""
    return np.maximum(
        np.asarray(resource, dtype=float) * (1.0 - float(use_fraction)),
        float(d_min),
    )


def psi_recruit(r, r0, r50, n):
    """M4 per-location recruitment nonlinearity Psi_G(r)=[r-r0]_+^n/(r50^n+[r-r0]_+^n) (rev4 spec §3).
    Elementwise; range [0,1). Sub-threshold background (r<=r0) -> 0. This is the natural single readout
    that replaces the old M/B/P95 sensors: <Psi_G(r_E)>_x is a soft recruited-area measure."""
    x = np.maximum(np.asarray(r, dtype=float) - r0, 0.0)
    xn = x ** n
    return xn / (r50 ** n + xn)


def pnorm_pool(z, p):
    """M4 pooled drive A_G=[<z^p>_x]^(1/p) over ALL elements (rev4 spec §3). z in [0,1], p>=1.
    p=1 -> soft recruited-area mean; larger p (2-4) -> focal-sensitive (mean<->max knob), so a strong
    focal/core recruits the pool before participation is global."""
    z = np.asarray(z, dtype=float)
    return float(np.mean(z ** p)) ** (1.0 / p)


def _grid_index(pos, L, n_grid):
    ix = np.clip((np.asarray(pos)[:, 0] / L * n_grid).astype(int), 0, n_grid - 1)
    iy = np.clip((np.asarray(pos)[:, 1] / L * n_grid).astype(int), 0, n_grid - 1)
    return ix, iy


def firing_rate_field(spk_bool, pos, L, n_grid, sigma):
    """One timestep's INSTANTANEOUS rate field (pre-EMA): bin the spikes `spk_bool`
    (1-D bool over a subpopulation) at continuous positions `pos` (n,2 mm) onto an
    n_grid x n_grid lattice over [0,L]^2, then convolve with an isotropic Gaussian of
    width `sigma` mm (periodic, via src.sef_hfo_field). Returns (n_grid, n_grid).
    Implemented per the plan (Task 3)."""
    counts = np.zeros((n_grid, n_grid))
    spk_bool = np.asarray(spk_bool, bool)
    if spk_bool.any():
        ix, iy = _grid_index(pos[spk_bool], L, n_grid)
        np.add.at(counts, (iy, ix), 1.0)                       # field[iy, ix]
    return convolve_periodic(counts, isotropic_gaussian(n_grid, L, sigma))


def sample_field_at(field, pos, L, n_grid):
    """Sample a (n_grid, n_grid) field at continuous positions `pos` (n,2 mm) on the
    L x L sheet (nearest-lattice). Returns (n,). Inverse of firing_rate_field's
    binning. Implemented per the plan (Task 3)."""
    ix, iy = _grid_index(pos, L, n_grid)
    return np.asarray(field)[iy, ix]


# ---------------------------------------------------------------------------
# The field object. Implements the simulate_kick `slow` protocol:
#   apply_currents(I_E, I_I, labels) -> I_net   (called each step, pre-membrane)
#   threshold(V_th_base)            -> V_th_eff (keeps the heterogeneous core)
#   step(spk, labels, dt)           -> None     (advances the fields, called post-spike)
# ---------------------------------------------------------------------------
class SpatialSlowField:
    """Spatial slow-variable field; off-by-default == slow=None (byte-parity)."""

    def __init__(self, N, V_th0, posE, posI, L, core_mask_E=None,
                 cfg: SpatialSlowFieldConfig | None = None):
        """Allocate q_I (== q_init) and g_K (== 0) on the n_grid lattice, store the E/I
        positions for binning + sampling, validate cfg. Implemented per the plan (Task 4)."""
        self.cfg = cfg or SpatialSlowFieldConfig()
        self.cfg.validate()
        if self.cfg.use_A:
            self._load_shunt_params().validate()   # Task 1 LoadShuntParams.validate(), never auto-invoked otherwise
        self.N = int(N); self.nE = int(np.asarray(posE).shape[0])
        self.nI = int(np.asarray(posI).shape[0]); self.L = float(L)
        self.posE = np.asarray(posE, float); self.posI = np.asarray(posI, float)
        n = self.cfg.n_grid
        self.q_I = np.full((n, n), self.cfg.q_init, dtype=float)
        self.g_K = np.zeros((n, n), dtype=float)
        self.rE = np.zeros((n, n)); self.rI = np.zeros((n, n))         # EMA rate fields
        self._Kq = isotropic_gaussian(n, L, self.cfg.sigma_q)
        self._Kk = isotropic_gaussian(n, L, self.cfg.sigma_K)
        self._ixE, self._iyE = _grid_index(self.posE, L, n)            # fixed E->grid map
        self._alpha_a = None
        self.trace_qI_mean = []; self.trace_gK_mean = []
        # ---- h_G(t) global recovery (M3A-v2.2, §B6) ----
        self.h_G = float(self.cfg.hG_init)
        self.rE_fast = np.zeros((n, n))                               # FAST (tau_s) EMA for sensors
        self._t = 0.0                                                 # absolute time (for hG_script)
        self._alpha_s = None
        self.hG_script = None                                        # clamp/surrogate override (Deferred)
        self.trace_hG = []; self.trace_M = []; self.trace_B = []; self.trace_Pi = []
        # ---- M4 shared inhibitory pool (rev4 spec §4); mu_G=pool activation, S_G=pool output ----
        self.mu_G = 0.0
        self.S_G = 0.0
        self.trace_muG = []; self.trace_SG = []; self.trace_AG = []
        self.trace_Irec_mean = []                                     # mean recurrent-E current (matched-subtractive calib)
        if cfg.clamp_SG is not None:
            self.S_G = float(cfg.clamp_SG)                            # static-pool arm: S_G frozen from t=0
        self.trace_rEfast_max = []      # per-step spatial-max of rE_fast (for r50 sensor-scale calibration)
        # ---- n(x,t) load -> a(x,t) shunt field (M4-3A) ----
        self.n_load = np.full((n, n), self.cfg.n_base, dtype=float)
        self.a_shunt = np.zeros((n, n), dtype=float)
        self._Kn = isotropic_gaussian(n, L, self.cfg.sigma_n)
        self.trace_n_mean = []
        self.trace_a_mean = []
        self.trace_un_mean = []                     # P1-4: field-derived drive u_n (for P0b param lock)
        # ---- persistence-gated recovery field p(x,t) (spec 2026-07-21 §5) ----
        self.p = np.full((n, n), self.cfg.p_init, dtype=float)
        if self.cfg.clamp_persist is not None:
            self.p[:] = float(self.cfg.clamp_persist)     # open-loop / ablation: frozen p (step skips its ODE)
        self._Kp = isotropic_gaussian(n, L, self.cfg.sigma_p)
        self.trace_p_mean = []
        self.trace_p_max = []
        # ---- core/surround field split (SNN-native exit Phase 1 readout; core_mask_E=None -> no split
        # recorded -> zero overhead. These traces only READ q_I/p so E_spk_bool is unchanged -> BASELINE_SHA
        # preserved). q_core/q_surround = mean q_I sampled at core / non-core E-neuron cells. ----
        self._core_mask_E = None if core_mask_E is None else np.asarray(core_mask_E, bool)
        if self._core_mask_E is not None:
            self._iyE_core, self._ixE_core = self._iyE[self._core_mask_E], self._ixE[self._core_mask_E]
            _surr = ~self._core_mask_E
            self._iyE_surr, self._ixE_surr = self._iyE[_surr], self._ixE[_surr]
        self.trace_q_core = []
        self.trace_q_surround = []
        self.trace_p_core = []
        self.trace_p_surround = []
        # ---- containment memory H (Phase-3 vNext) ----
        self.H = float(self.cfg.H_init)
        self.trace_H = []
        # H-drive diagnostics (Phase-0.2 carrier hardening 2026-07-24): the ACTUAL sensor input phi_drive
        # (NOT p_max), and the active-focus mask fraction. Observation-only -> spike output unchanged.
        self.trace_phi_drive = []; self.trace_active_frac = []
        # ---- Z/M per-neuron slow variables (ported from mz_slow_vars.py; Z/M migration 2026-07-22) ----
        self.is_E = np.zeros(self.N, dtype=bool); self.is_E[:self.nE] = True   # E cells are [:nE]
        self.z = np.ones(self.N, dtype=float)     # inhibitory efficacy in [0,1]; 1 on I cells (never updated)
        self.m = np.zeros(self.N, dtype=float)    # adaptation; 0 on I cells (never updated)
        self._I_I_last = None                     # E-cell I_I from the last apply_currents (z_inf Heaviside input)
        self.trace_z_mean = []; self.trace_z_min = []
        self.trace_z_core_mean = []; self.trace_z_surround_mean = []
        self.trace_m_mean = []; self.trace_m_max = []
        self.trace_m_core_mean = []; self.trace_m_surround_mean = []
        # ---- local state-selective recurrent memory (allocated only on-path) ----
        self.mode_H = np.zeros((n, n), dtype=float) if self.cfg.use_mode_H else None
        self.trace_mode_H_mean = []; self.trace_mode_H_max = []
        self.trace_mode_H_rate_max_hz = []
        self.trace_mode_H_drive_mean = []; self.trace_mode_H_drive_max = []
        self.trace_mode_H_gain_mean = []; self.trace_mode_H_gain_max = []
        self.trace_mode_H_gain_core_mean = []
        self.trace_mode_M_raw_pool = []; self.trace_mode_M_pool = []; self.trace_mode_M_divisor = []
        self.mode_M_memory = 0.0
        self.trace_mode_M_memory = []
        # ---- Phase-D dynamic threshold increment (E-only, fast) ----
        self.phi_increment = np.zeros(self.N, dtype=float)
        self.trace_phi_mean = []; self.trace_phi_max = []
        self.trace_phi_core_mean = []; self.trace_phi_surround_mean = []
        # ---- lifecycle prototype inhibitory state (all off-path values are neutral) ----
        self.i2e_resource = np.ones(self.nI, dtype=float)
        if self.cfg.i2e_tau_cv > 0.0:
            rng_tau = np.random.default_rng(int(self.cfg.i2e_tau_seed))
            sigma_tau = np.sqrt(np.log1p(self.cfg.i2e_tau_cv ** 2))
            factors_tau = np.exp(
                sigma_tau * rng_tau.standard_normal(self.nI) - 0.5 * sigma_tau ** 2
            )
            factors_tau /= factors_tau.mean()
            self.i2e_tau_recovery = self.cfg.tau_i2e_depression * factors_tau
        else:
            self.i2e_tau_recovery = None
        self.i_adaptation_increment = np.zeros(self.N, dtype=float)
        self.trace_i2e_resource_mean = []; self.trace_i2e_resource_min = []
        self.trace_i_adaptation_mean = []; self.trace_i_adaptation_max = []
        self.trace_cond_vinf_mean = []; self.trace_cond_tau_eff_mean = []
        self.trace_cond_gE_mean = []; self.trace_cond_gI_local_mean = []
        self.trace_cond_gI_global = []; self.trace_cond_gI_eff_mean = []
        self.trace_cond_gMm_mean = []
        self.trace_cond_Iexc_mean = []; self.trace_cond_Iinh_mean = []
        self.trace_cond_Isahp_mean = []
        self.trace_cond_lambda_mean = []; self.trace_cond_lambda_max = []
        self.trace_cond_lambda_core_mean = []

    def uses_zm_conductance(self) -> bool:
        """True only for the explicit Phase-D conductance arm."""
        return bool(self.cfg.use_zm_conductance)

    def uses_zm_conductance_homotopy(self) -> bool:
        return bool(self.cfg.use_zm_conductance_homotopy)

    def zm_conductance_config(self) -> ZMConductanceConfig:
        return self.cfg.zm_conductance_config()

    def zm_conductance_step(self, V, I_E, I_I, decay_V):
        """Delegate one membrane step to the tested pure conductance module.

        Raw pre-z GABA is stashed for the original z sensor before any
        conductance scaling.  Neither ``apply_currents`` nor the old S_G
        recurrent-E divisor is involved.
        """
        if not self.uses_zm_conductance():
            raise RuntimeError("Z/M conductance membrane is not enabled")
        self._I_I_last = np.asarray(I_I, dtype=float)[:self.nE].copy()
        out = conductance_membrane_step(
            V,
            I_E,
            I_I,
            self.z,
            self.m,
            decay_V,
            self.is_E,
            self.zm_conductance_config(),
        )
        e = self.is_E
        self.trace_cond_vinf_mean.append(float(out["V_inf"][e].mean()))
        self.trace_cond_tau_eff_mean.append(
            float(out["tau_eff_ms"][e].mean())
        )
        self.trace_cond_gE_mean.append(float(out["g_E"][e].mean()))
        self.trace_cond_gI_local_mean.append(
            float(out["g_I_local"][e].mean())
        )
        self.trace_cond_gI_global.append(float(out["g_I_global"]))
        self.trace_cond_gI_eff_mean.append(
            float(out["g_I_eff"][e].mean())
        )
        self.trace_cond_gMm_mean.append(float(out["g_Mm"][e].mean()))
        self.trace_cond_Iexc_mean.append(float(out["I_exc"][e].mean()))
        self.trace_cond_Iinh_mean.append(float(out["I_inh"][e].mean()))
        self.trace_cond_Isahp_mean.append(float(out["I_sahp"][e].mean()))
        return out

    def zm_conductance_homotopy_step(self, V, I_E, I_I, I_native, decay_V):
        """Apply the Phase-E Z-gated vector-field homotopy and record it."""
        if not self.uses_zm_conductance_homotopy():
            raise RuntimeError("Z/M conductance homotopy is not enabled")
        out = state_dependent_homotopy_step(
            V,
            I_E,
            I_I,
            I_native,
            self.z,
            self.m,
            decay_V,
            self.is_E,
            self.zm_conductance_config(),
            z_native=self.cfg.cond_homotopy_z_native,
            z_conductance=self.cfg.cond_homotopy_z_conductance,
        )
        e = self.is_E
        lam_e = out["lambda"][e]
        self.trace_cond_lambda_mean.append(float(lam_e.mean()))
        self.trace_cond_lambda_max.append(float(lam_e.max()))
        if self._core_mask_E is not None:
            self.trace_cond_lambda_core_mean.append(
                float(lam_e[self._core_mask_E].mean())
            )
        self.trace_cond_vinf_mean.append(float(out["V_inf"][e].mean()))
        self.trace_cond_tau_eff_mean.append(float(out["tau_eff_ms"][e].mean()))
        self.trace_cond_gE_mean.append(float(out["g_E"][e].mean()))
        self.trace_cond_gI_local_mean.append(float(out["g_I_local"][e].mean()))
        self.trace_cond_gI_global.append(float(out["g_I_global"]))
        self.trace_cond_gI_eff_mean.append(float(out["g_I_eff"][e].mean()))
        self.trace_cond_gMm_mean.append(float(out["g_Mm"][e].mean()))
        self.trace_cond_Iexc_mean.append(float(out["I_exc"][e].mean()))
        self.trace_cond_Iinh_mean.append(float(out["I_inh"][e].mean()))
        self.trace_cond_Isahp_mean.append(float(out["I_sahp"][e].mean()))
        return out

    def apply_currents(self, I_E, I_I, labels=None, I_E_rec=None):
        """I_net = I_E - q_I(x_i,t)*I_I - eta_K*g_K(x_i,t) - eta_G*h_G for E cells; I_E - I_I for I cells.
        q_I==1, g_K==0, h_G off -> returns I_E - I_I exactly (parity).
        M4 (use_SG AND I_E_rec given): additionally subtract the removed recurrent current on E cells
        dI_rec = I_E_rec[:nE]*(alpha_G*S_G/(1+alpha_G*S_G)) + beta_SG*S_G  (divide recurrent E only,
        rev4 §5). alpha_G*S_G==0 AND beta_SG==0 -> dI_rec exactly 0 -> byte-parity."""
        qI_E = self.q_I[self._iyE, self._ixE]                          # (nE,)
        gK_E = self.g_K[self._iyE, self._ixE]
        out = np.asarray(I_E, float) - np.asarray(I_I, float)          # I cells: I_E - I_I
        nE = self.nE
        if self.cfg.use_z:                                             # Z/M: stash E-cell I_I for the z_inf Heaviside (step)
            self._I_I_last = np.asarray(I_I, float)[:nE]
        hG_eff = self.h_G if self.cfg.use_hG else 0.0                  # HARD gate: use_hG=False -> no h_G
        inh_E = qI_E * I_I[:nE]                                        # q_I-model inhibition (q_I==1 for pure Z/M)
        if self.cfg.use_z:                                            # Z/M: z_i scales E inhibition (z*q_I*I_I == z*I_I)
            inh_E = self.z[:nE] * inh_E
        out[:nE] = (I_E[:nE] - inh_E
                    - self.cfg.eta_K * gK_E
                    - self.cfg.eta_G * hG_eff)                         # global recovery scalar (E only)
        if self.cfg.use_m:                                            # Z/M: adaptation current eta_m*m subtracted (E-only)
            out[:nE] -= self.cfg.eta_m * self.m[:nE]
        if self.cfg.use_A and self.cfg.eta_A != 0.0:               # M4-3A subtractive bias (E only)
            aE = self.a_shunt[self._iyE, self._ixE]
            out[:nE] -= self.cfg.eta_A * aE
        if self.cfg.use_persist and self.cfg.eta_r != 0.0:         # §5 persistence-gated recovery current (E only)
            pE = self.p[self._iyE, self._ixE]                      # local: p sampled at each E neuron's cell
            phi = (pE if self.cfg.p50_r <= 0.0                     # linear Phi, or Hill if p50_r>0
                   else pE ** self.cfg.n_r / (self.cfg.p50_r ** self.cfg.n_r + pE ** self.cfg.n_r))
            out[:nE] -= self.cfg.eta_r * phi                       # outward (g_K-type) displacement, E only
        if self.cfg.use_SG and I_E_rec is None and (self.cfg.alpha_G > 0.0 or self.cfg.beta_SG > 0.0):
            raise RuntimeError(                                        # §6: loud failure beats silent contamination
                "use_SG with alpha_G>0 or beta_SG>0 requires I_E_rec (recurrent-only current), got None. "
                "The pool would build S_G with NO membrane effect (silent false negative). Drive M4 through "
                "simulate_kick (which tracks I_E_rec); do not use a caller that omits it.")
        if self.cfg.use_SG and I_E_rec is not None:                   # §5 divisive recurrent-gain (E only)
            aS = self.cfg.alpha_G * self.S_G
            if self.cfg.use_H:                                        # + containment memory H (Phase-3 vNext)
                aH = self.cfg.alpha_H * self.H
            else:
                aH = 0.0
            if self.cfg.use_mode_M_divisive:
                raw_pool_M = self.mode_M_raw_pool()
                drive_M = self.mode_M_pool()
                pool_M = self.mode_M_memory if self.cfg.use_mode_M_memory else drive_M
                aM = self.cfg.kappa_mode_M * pool_M
                self.trace_mode_M_raw_pool.append(float(raw_pool_M))
                self.trace_mode_M_pool.append(float(drive_M))
                self.trace_mode_M_memory.append(float(self.mode_M_memory))
                self.trace_mode_M_divisor.append(float(1.0 + aM))
            else:
                aM = 0.0
            denom = 1.0 + aS + aH + aM
            frac = (aS + aH + aM) / denom
            out[:nE] -= np.asarray(I_E_rec, float)[:nE] * frac + self.cfg.beta_SG * self.S_G
            if self.cfg.use_mode_H:
                if self.cfg.rho_mode_H > 0.0:
                    gain = self.mode_H_gain_at_E()
                    out[:nE] += np.asarray(I_E_rec, float)[:nE] * gain / denom
                else:  # sensor-only path: no new membrane arithmetic
                    gain = np.zeros(nE, dtype=float)
                self.trace_mode_H_gain_mean.append(float(np.mean(gain)))
                self.trace_mode_H_gain_max.append(float(np.max(gain)))
                if self._core_mask_E is not None:
                    self.trace_mode_H_gain_core_mean.append(
                        float(np.mean(gain[self._core_mask_E]))
                    )
            self.trace_Irec_mean.append(float(np.asarray(I_E_rec, float)[:nE].mean()))  # for matched-subtractive calib
        return out

    def mode_H_gain_at_E(self) -> np.ndarray:
        """Current local recurrent-E gain; pure read of H, Z and M state."""
        if not self.cfg.use_mode_H or self.cfg.rho_mode_H <= 0.0:
            return np.zeros(self.nE, dtype=float)
        zeta = np.clip(
            (self.cfg.z_mode_base - self.z[:self.nE])
            / (self.cfg.z_mode_base - self.cfg.z_mode_susceptible),
            0.0, 1.0,
        )
        gate_z = zero_baseline_sigmoid(
            zeta, self.cfg.zeta_mode_center, self.cfg.zeta_mode_slope
        )
        mE = self.m[:self.nE]
        gate_m = 1.0 / (1.0 + (mE / self.cfg.m_mode_half) ** self.cfg.m_mode_power)
        hE = self.mode_H[self._iyE, self._ixE]
        if self.cfg.mode_H_common_subtraction > 0.0:
            hE = np.maximum(
                hE - self.cfg.mode_H_common_subtraction * float(np.mean(hE)),
                0.0,
            )
        return self.cfg.rho_mode_H * hE * gate_z * gate_m

    def mode_M_raw_pool(self) -> float:
        """Mask-free p-norm of native E-cell M, normalised to a reference."""
        if not self.cfg.use_mode_M_divisive or self.cfg.kappa_mode_M <= 0.0:
            return 0.0
        scaled = np.maximum(self.m[:self.nE], 0.0) / self.cfg.m_mode_div_ref
        return float(np.mean(scaled ** self.cfg.m_mode_div_power)) ** (
            1.0 / self.cfg.m_mode_div_power
        )

    def mode_M_pool(self) -> float:
        """Low-baseline Hill activation of the collective native-M pool."""
        raw = self.mode_M_raw_pool()
        power = self.cfg.m_mode_div_hill_power
        raised = raw ** power
        return raised / (1.0 + raised)

    def _load_shunt_params(self):
        c = self.cfg
        return LoadShuntParams(tau_n=c.tau_n, k_n=c.k_n, rho_n=c.rho_n, n_base=c.n_base,
                               n50=c.n50, hill_h=c.hill_h, a_max=c.a_max, u_n0=c.u_n0,
                               n_min=c.n_min, n_max=c.n_max)

    def uses_shunt(self) -> bool:
        """True iff the conductance shunt actually couples (P1-1). Needs use_A AND k_n!=0
        (else a==0 forever -> must take the literal parity path in kick_probe) AND alpha_A!=0.
        The parity gate keys on uses_shunt()==False; do NOT drop the k_n!=0 term."""
        return bool(self.cfg.use_A and self.cfg.k_n != 0.0 and self.cfg.alpha_A != 0.0)

    def shunt_g_at_E(self) -> np.ndarray:
        """Per-E-neuron conductance g_A = alpha_A * a, clipped to [0, g_A_max].
        Returns zeros(nE) when shunt off -> kick_probe takes the literal parity branch."""
        if not self.uses_shunt():
            return np.zeros(self.nE, dtype=float)
        aE = self.a_shunt[self._iyE, self._ixE]
        return np.clip(self.cfg.alpha_A * aE, 0.0, self.cfg.g_A_max)

    def threshold(self, V_th_base):
        """Return the heterogeneous substrate plus enabled E/I increments."""
        if not self.cfg.use_phi and not self.cfg.use_i_adaptation:
            return V_th_base
        return (
            np.asarray(V_th_base, dtype=float)
            + self.phi_increment
            + self.i_adaptation_increment
        )

    def uses_i2e_depression(self) -> bool:
        return bool(self.cfg.use_i2e_depression)

    def i2e_resource_at_sources(self, source_indices) -> np.ndarray:
        """Presynaptic availability immediately before the current I-spike use."""
        if not self.uses_i2e_depression():
            return np.ones(len(source_indices), dtype=float)
        return self.i2e_resource[np.asarray(source_indices, dtype=int)]

    def consume_i2e_sources(self, source_indices) -> None:
        """Apply one multiplicative resource use per firing I neuron."""
        if not self.uses_i2e_depression():
            return
        idx = np.asarray(source_indices, dtype=int)
        self.i2e_resource[idx] = deplete_i2e_resource(
            self.i2e_resource[idx],
            self.cfg.U_i2e_depression,
            self.cfg.d_i2e_min,
        )

    def step(self, spk, labels, dt):
        """Advance the fields one dt: (1) EMA-update r_E,r_I from this step's spikes,
        (2) form a_q = K_q*aq_drive(r_E,r_I,eta_E,eta_I), a_K = K_K*r_E, (3) integrate the
        q_I ODE (depletion ~ k_q*f*q_I) and the BOUNDED-build g_K ODE
        (build ~ k_K*f*(gK_max-g_K)) on the lattice (bounds [q_min,1] and [0,gK_max]).
        q_I/g_K are read directly in apply_currents (no per-neuron cache). Task 5."""
        cfg = self.cfg
        spk = np.asarray(spk, bool)
        if cfg.use_i2e_depression:
            self.i2e_resource[:] = recover_i2e_resource(
                self.i2e_resource,
                dt,
                cfg.tau_i2e_depression
                if self.i2e_tau_recovery is None
                else self.i2e_tau_recovery,
            )
            self.trace_i2e_resource_mean.append(float(self.i2e_resource.mean()))
            self.trace_i2e_resource_min.append(float(self.i2e_resource.min()))
        if cfg.use_phi:
            # Exact decay between spikes, then the post-spike jump. I-cell
            # entries stay exactly zero and never enter the E threshold.
            phiE = self.phi_increment[:self.nE]
            phiE *= np.exp(-dt / cfg.tau_phi)
            phiE[spk[:self.nE]] += cfg.delta_phi
            self.phi_increment[self.nE:] = 0.0
            self.trace_phi_mean.append(float(phiE.mean()))
            self.trace_phi_max.append(float(phiE.max()))
            if self._core_mask_E is not None:
                self.trace_phi_core_mean.append(
                    float(phiE[self._core_mask_E].mean())
                )
                self.trace_phi_surround_mean.append(
                    float(phiE[~self._core_mask_E].mean())
                )
        if cfg.use_i_adaptation:
            aI = self.i_adaptation_increment[self.nE:]
            aI *= np.exp(-dt / cfg.tau_i_adaptation)
            aI[spk[self.nE:]] += cfg.delta_i_adaptation
            self.i_adaptation_increment[:self.nE] = 0.0
            self.trace_i_adaptation_mean.append(float(aI.mean()))
            self.trace_i_adaptation_max.append(float(aI.max()))
        if cfg.use_z:                                            # Z/M z_i update (ported byte-identical from mz_slow_vars.py)
            z_inf_E = (self._I_I_last < cfg.I_th_EI).astype(float)   # z_inf = H(I_th_EI - I_I): 1 iff I_I < I_th_EI (strict)
            zE = self.z[self.is_E]
            zE = zE + (dt / cfg.tau_z) * (z_inf_E - zE)
            self.z[self.is_E] = np.clip(zE, 0.0, 1.0)           # z in [0,1]
        if cfg.use_m:                                            # Z/M m_i update (ported byte-identical from mz_slow_vars.py)
            mE = self.m[self.is_E]
            mE = mE - (mE / cfg.tau_adp) * dt                    # decay
            self.m[self.is_E] = np.maximum(mE, 0.0)             # m >= 0
            self.m[spk & self.is_E] += 1.0                      # E spike -> +1 ; I spikes ignored (E-only)
        if cfg.use_z or cfg.use_m:                               # Z/M traces (skipped when both off -> parity)
            zE_t = self.z[:self.nE]; mE_t = self.m[:self.nE]
            self.trace_z_mean.append(float(zE_t.mean())); self.trace_z_min.append(float(zE_t.min()))
            self.trace_m_mean.append(float(mE_t.mean())); self.trace_m_max.append(float(mE_t.max()))
            if self._core_mask_E is not None:
                self.trace_z_core_mean.append(float(zE_t[self._core_mask_E].mean()))
                self.trace_z_surround_mean.append(float(zE_t[~self._core_mask_E].mean()))
                self.trace_m_core_mean.append(float(mE_t[self._core_mask_E].mean()))
                self.trace_m_surround_mean.append(float(mE_t[~self._core_mask_E].mean()))
        rE_inst = firing_rate_field(spk[:self.nE], self.posE, self.L, cfg.n_grid, cfg.sigma_r)
        rI_inst = firing_rate_field(spk[self.nE:], self.posI, self.L, cfg.n_grid, cfg.sigma_r)
        if self._alpha_a is None:
            self._alpha_a = 1.0 - np.exp(-dt / cfg.tau_a)
        a = self._alpha_a
        self.rE += a * (rE_inst - self.rE)                            # EMA (§B5.1)
        self.rI += a * (rI_inst - self.rI)
        if cfg.use_mode_H:
            rE_mode_hz = local_rate_field_hz(
                self.rE, dt_ms=dt, n_population=self.nE, n_grid=cfg.n_grid
            )
            drive_H = saturation(
                rE_mode_hz, cfg.theta_mode_H_hz, cfg.half_mode_H_hz
            )
            tau_H_mode = np.where(
                drive_H > self.mode_H, cfg.tau_mode_H, cfg.tau_mode_H_down
            )
            alpha_H_mode = 1.0 - np.exp(-dt / tau_H_mode)
            self.mode_H += alpha_H_mode * (drive_H - self.mode_H)
            np.clip(self.mode_H, 0.0, 1.0, out=self.mode_H)
            self.trace_mode_H_mean.append(float(self.mode_H.mean()))
            self.trace_mode_H_max.append(float(self.mode_H.max()))
            self.trace_mode_H_rate_max_hz.append(float(rE_mode_hz.max()))
            self.trace_mode_H_drive_mean.append(float(drive_H.mean()))
            self.trace_mode_H_drive_max.append(float(drive_H.max()))
        if cfg.use_mode_M_divisive and cfg.use_mode_M_memory:
            drive_M = self.mode_M_pool()
            tau_M_memory = (
                cfg.tau_mode_M_memory_up
                if drive_M > self.mode_M_memory
                else cfg.tau_mode_M_memory_down
            )
            alpha_M_memory = 1.0 - np.exp(-dt / tau_M_memory)
            self.mode_M_memory += alpha_M_memory * (drive_M - self.mode_M_memory)
            self.mode_M_memory = float(np.clip(self.mode_M_memory, 0.0, 1.0))
        if cfg.use_qI and cfg.k_q != 0.0:                            # §B5.2 (depletion ~ k_q*f*q_I)
            a_q = convolve_periodic(aq_drive(self.rE, self.rI, cfg.eta_E, cfg.eta_I), self._Kq)
            fq = saturation(a_q, cfg.a0_q, cfg.a50_q)
            self.q_I += dt * ((1.0 - self.q_I) / cfg.tau_q - cfg.k_q * fq * self.q_I)
            np.clip(self.q_I, cfg.q_min, 1.0, out=self.q_I)
        if cfg.use_gK and cfg.k_K != 0.0:                            # §B5.3 BOUNDED build (k_K is the knob)
            a_K = convolve_periodic(self.rE, self._Kk)
            fk = saturation(a_K, cfg.a0_K, cfg.a50_K)
            self.g_K += dt * (-self.g_K / cfg.tau_K + cfg.k_K * fk * (cfg.gK_max - self.g_K))
            np.clip(self.g_K, 0.0, cfg.gK_max, out=self.g_K)
        if cfg.use_persist:                                         # §5 persistence sensor: slow leaky integral
            if cfg.clamp_persist is None and self._t >= cfg.persist_onset_ms:  # of supra-theta_p activity (frozen when
                                                                    # clamped; inactive before persist_onset_ms so the
                                                                    # M4 state forms first = established-state fork)
                a_p = convolve_periodic(self.rE, self._Kp)         # smoothed EMA rate (sustained activity)
                drive = saturation(a_p, cfg.theta_p, cfg.a50_p)    # Psi(K_p*r_E - theta_p) in [0,1)
                if cfg.tau_p_down is None:
                    self.p += dt * (drive - self.p) / cfg.tau_p    # tau_p dp/dt = Psi - p (symmetric)
                else:                                              # asymmetric: charge tau_p (fast), decay tau_p_down (slow)
                    tau_eff = np.where(drive >= self.p, cfg.tau_p, cfg.tau_p_down)
                    self.p += dt * (drive - self.p) / tau_eff      # long hold once activity drops -> q_I refills
                np.clip(self.p, 0.0, 1.0, out=self.p)
            self.trace_p_mean.append(float(self.p.mean()))
            self.trace_p_max.append(float(self.p.max()))
        if cfg.use_H:                                              # §Phase-3: tau_H dH/dt = <Phi(p)> - H
            phi = (self.p if cfg.p50_r <= 0.0                      # linear Phi, or Hill (same Phi as the actuator)
                   else self.p ** cfg.n_r / (cfg.p50_r ** cfg.n_r + self.p ** cfg.n_r))
            if cfg.H_sensor == "active":                          # active-focus mean: mean Phi(p) over cells within
                pmax = float(phi.max())                           # 20% of the peak -> tracks a localized focus's
                if pmax > 0.0:                                    # intensity, NOT grid-diluted
                    mask = phi > 0.2 * pmax                       # (phi[mask]==phi[phi>0.2*pmax] -> byte-parity phi_drive)
                    phi_drive = float(phi[mask].mean())
                    active_frac = float(mask.mean())
                else:
                    phi_drive = 0.0; active_frac = 0.0
            else:                                                # (reduces to the mean for a broad/uniform state)
                phi_drive = float(phi.mean())                    # global spatial mean (byte-parity path)
                active_frac = 1.0
            self.H += dt * (phi_drive - self.H) / cfg.tau_H
            self.H = float(np.clip(self.H, 0.0, cfg.H_max))
            self.trace_H.append(self.H)
            self.trace_phi_drive.append(phi_drive)                # the REAL H input (was untraced; NOT p_max)
            self.trace_active_frac.append(active_frac)            # active-focus size = frac of grid cells sensed
        if cfg.use_A:                                               # M4-3A load -> shunt
            u_n = convolve_periodic(self.rE, self._Kn)              # field-derived drive K_n * rE (EMA rE)
            self.trace_un_mean.append(float(u_n.mean()))            # P1-4: dump real u_n even when k_n=0 (P0b lock)
            if cfg.k_n != 0.0:                                      # evolve load ONLY when active -> parity when k_n=0
                self.n_load, self.a_shunt = load_shunt_step(self.n_load, u_n, dt, self._load_shunt_params())
        self.trace_n_mean.append(float(self.n_load.mean()))
        self.trace_a_mean.append(float(self.a_shunt.mean()))
        self.trace_qI_mean.append(float(self.q_I.mean()))
        self.trace_gK_mean.append(float(self.g_K.mean()))
        if cfg.use_hG or cfg.use_SG:                              # FAST (tau_s) EMA needed by h_G and/or S_G
            if self._alpha_s is None:
                self._alpha_s = 1.0 - np.exp(-dt / cfg.tau_s)
            self.rE_fast += self._alpha_s * (rE_inst - self.rE_fast)   # FAST sensor EMA
        if cfg.use_hG:                                            # §B6 global recovery
            M = global_M(self.rE_fast)                            # sensors ALWAYS computed (trace sync)
            B = global_B(self.rE_fast, cfg.r_A, cfg.Delta_A)
            Pi = global_participation(self.rE_fast)
            chi = chi_G(M, B, Pi, cfg.M50, cfg.B50, cfg.Pi50, cfg.n_M, cfg.n_B, cfg.n_Pi)
            if self.hG_script is not None:                       # clamp/surrogate path (skip ODE)
                self.h_G = float(np.clip(self.hG_script(self._t), 0.0, cfg.hG_max))
            else:                                                # ALWAYS integrate: k_G=0 -> decay ONLY
                dh = -self.h_G / cfg.tau_G + cfg.k_G * chi * (cfg.hG_max - self.h_G)
                self.h_G = float(np.clip(self.h_G + dt * dh, 0.0, cfg.hG_max))
            if cfg.lambda_G != 0.0 and cfg.use_qI:               # arm F: h_G accelerates q_I refill
                self.q_I += dt * (cfg.lambda_G * self.h_G * (1.0 - self.q_I))
                np.clip(self.q_I, cfg.q_min, 1.0, out=self.q_I)
            self.trace_M.append(M); self.trace_B.append(B); self.trace_Pi.append(Pi)
            self.trace_hG.append(self.h_G)                       # traces ALWAYS synced under use_hG
        if cfg.use_SG:                                            # §4 M4 shared inhibitory pool advance
            z_G = psi_recruit(self.rE_fast, cfg.r0_psi, cfg.r50_psi, cfg.n_psi)
            A_G = pnorm_pool(z_G, cfg.p_pool)                     # single natural readout (no M/B/P95)
            self.mu_G += dt * (-self.mu_G + A_G) / cfg.tau_mu     # forward Euler (h_G-style integration)
            self.mu_G = float(np.clip(self.mu_G, 0.0, 1.0))
            self.S_G += dt * (-self.S_G + cfg.S_max * self.mu_G) / cfg.tau_S
            self.S_G = float(np.clip(self.S_G, 0.0, cfg.S_max))
            if cfg.clamp_SG is not None:
                self.S_G = float(cfg.clamp_SG)                        # static-pool arm: freeze S_G (mu_G still advances internally)
            self.trace_AG.append(A_G); self.trace_muG.append(self.mu_G); self.trace_SG.append(self.S_G)
            self.trace_rEfast_max.append(float(self.rE_fast.max()))   # time trace of the sensor-field peak
        if self._core_mask_E is not None:                            # Phase 1 readout: core vs surround field split
            self.trace_q_core.append(float(self.q_I[self._iyE_core, self._ixE_core].mean()))
            self.trace_q_surround.append(float(self.q_I[self._iyE_surr, self._ixE_surr].mean()))
            if cfg.use_persist:
                self.trace_p_core.append(float(self.p[self._iyE_core, self._ixE_core].mean()))
                self.trace_p_surround.append(float(self.p[self._iyE_surr, self._ixE_surr].mean()))
        self._t += dt
