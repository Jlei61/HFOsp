"""Topic 4 MZ early-onset dynamics — projected phase diagram, nonlinear ignition, causal bridge.

Design contract (BINDING):
  docs/superpowers/specs/2026-07-19-topic4-mz-onset-dynamics-phase-portrait-design.md

Tier = model-side mechanism analysis. NOT seizure validation; every phenotype is a detection label.

This module is IMPORT-SAFE and SIDE-EFFECT-FREE: no simulations run on import, no file writes (those
live in scripts/run_topic4_mz_onset_dynamics.py). It provides:

  1. MZOnsetProbe(MZSlowVars) — an off-by-default scheduled-intervention subclass of the accepted MZ slow
     object (spec §7.1/§13). With NO schedule it is byte-identical to MZSlowVars (parity gate). Rides the
     engine `slow=` protocol (apply_currents/threshold/step) polymorphically -> ZERO edits to the 6 guarded
     engine files, no re-bless. Adds: state freeze, z counterfactual transform, deterministic threshold
     probe (lowering, ignition §7) and pulse (raising, event suppression §9), and current-aware q_eff
     window accumulators (§5.2, per-neuron sums only -- never N x T).

  2. A faithful RESUMABLE copy of the simulate_kick default+slow integration loop (`run_loop`) that reuses
     kick_probe's exact numerics (_flatten_by_source, membrane_step) and can checkpoint full engine state
     mid-loop so amplitude/location forks resume cheaply (spec §7.1 "checkpoint/resume is an optional
     performance optimization only after the replay implementation is scientifically complete"). Bit-identity
     vs the guarded simulate_kick is a TEST gate (tests/test_topic4_mz_onset_dynamics.py).

  3. Slow-state coordinates (D_z, A, q_eff, p_deplete) per region, z_bar-vs-q_eff mapping audit (§5),
     realized (D,A) grid field construction (§6.1), result-neutral ignition classification (§7.2),
     and projected-flow (drift/nullcline) eligibility (§5.3).

Reuse (do not reinvent): src/snn_engine/mz_slow_vars.MZSlowVars, src/snn_engine/kick_probe.{_flatten_by_source,
membrane_step, ee_std_recover_factor}, params.compute_nu_theta, src/topic4_state_conditioned_susceptibility
(coarse binning + operator), src/topic4_m3b_spectral_phase (operator).
"""
from __future__ import annotations

import copy
import os
import sys
from dataclasses import dataclass, field

import numpy as np

# --- snn_engine on path (mirrors scripts/run_topic4_mz_slowvars.py import block) ---
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENG = os.path.join(_ROOT, "src", "snn_engine")
if _ENG not in sys.path:
    sys.path.insert(0, _ENG)

from mz_slow_vars import MZSlowVars, MZSlowVarsConfig  # noqa: E402
from kick_probe import _flatten_by_source, membrane_step  # noqa: E402  (reuse exact engine numerics)
from params import compute_nu_theta  # noqa: E402

SCHEMA_VERSION = "mz-onset-dynamics-1.0"
DUR_KICK = 18.0  # engine kick duration (unused here: no kick; kept for stream fidelity documentation)


# ======================================================================== scheduled-intervention subclass
class MZOnsetProbe(MZSlowVars):
    """Off-by-default scheduled interventions on the accepted MZ slow object (spec §7.1/§8/§9).

    Every intervention is OFF unless explicitly configured. With no schedule, all three hooks
    (apply_currents/threshold/step) reduce to MZSlowVars EXACTLY -> byte parity with the accepted MZ run.

    Interventions (all keyed on self._step_i, which equals the engine loop index t because the engine
    calls apply_currents()/threshold() BEFORE step() and step() increments _step_i at its end):
      - freeze:        from branch_step onward, skip the z/m Euler update (hold z,m at branch values).
      - z_transform:   applied ONCE at branch_step to z_E (counterfactual: uniform / shuffle / reset / rotate).
      - probe:         threshold LOWERING on registered E target over [lo,hi) (nonlinear ignition, §7).
      - supp_pulse:    threshold RAISING on registered E target over [lo,hi) (event suppression, §9).
      - q_eff_windows: accumulate per-E-neuron sum(z*I_I), sum(I_I), count(I_I>=I_th) over 20-ms windows (§5.2).
    """

    def __init__(self, N, V_th0, cfg=None, *, NE, core_mask_E=None, snapshot_steps=None):
        super().__init__(N, V_th0, cfg, NE=NE, core_mask_E=core_mask_E, snapshot_steps=snapshot_steps)
        # --- schedule (all OFF by default -> parity) ---
        self._branch_step = None
        self._freeze = False
        self._z_transform = None            # callable(z_E) -> z_E, applied once at branch_step
        self._z_transform_done = False
        self._probe = None                  # dict(lo, hi, target_E, delta) threshold LOWERING
        self._supp = None                   # dict(lo, hi, target_E, delta) threshold RAISING
        # --- current-aware q_eff observer (§5.2): per-E-neuron accumulators over registered windows ---
        self._qeff_windows = []             # list of (lo, hi, label)
        self.qeff_accum = {}                # label -> dict(S_zI, S_I, cnt_deplete, n_steps)

    # ---------------------------------------------------------------- schedule setters
    def set_branch(self, branch_step, *, freeze=False, z_transform=None):
        self._branch_step = None if branch_step is None else int(branch_step)
        self._freeze = bool(freeze)
        self._z_transform = z_transform
        self._z_transform_done = False
        return self

    def set_probe(self, *, lo, hi, target_E, delta):
        """Threshold-LOWERING probe: subtract `delta` (mV) from V_th on E cells in `target_E` over [lo,hi)."""
        self._probe = dict(lo=int(lo), hi=int(hi), target_E=np.asarray(target_E, bool), delta=float(delta))
        return self

    def set_suppression(self, *, lo, hi, target_E, delta):
        """Threshold-RAISING pulse (inhibitory): add `delta` (mV) to V_th on E cells in `target_E` over [lo,hi)."""
        self._supp = dict(lo=int(lo), hi=int(hi), target_E=np.asarray(target_E, bool), delta=float(delta))
        return self

    def set_qeff_windows(self, windows, I_th):
        """windows: list of (lo, hi, label). I_th for p_deplete = P(I_I >= I_th). Accumulate per-E-neuron only."""
        self._qeff_windows = [(int(a), int(b), str(c)) for (a, b, c) in windows]
        self._qeff_I_th = float(I_th)
        for _, _, lab in self._qeff_windows:
            self.qeff_accum[lab] = dict(S_zI=np.zeros(self.NE), S_I=np.zeros(self.NE),
                                        cnt_deplete=np.zeros(self.NE), n_steps=0)
        return self

    # ---------------------------------------------------------------- hooks
    def apply_currents(self, I_E, I_I, labels=None, I_E_rec=None):
        # q_eff accumulation uses the SAME z that the membrane sees at this step (pre-update z, current I_I).
        if self._qeff_windows:
            t = self._step_i
            for lo, hi, lab in self._qeff_windows:
                if lo <= t < hi:
                    zE = self.z[self.is_E]
                    iI = I_I[self.is_E]
                    acc = self.qeff_accum[lab]
                    acc["S_zI"] += zE * iI
                    acc["S_I"] += iI
                    acc["cnt_deplete"] += (iI >= self._qeff_I_th).astype(float)
                    acc["n_steps"] += 1
                    break
        return super().apply_currents(I_E, I_I, labels, I_E_rec)

    def threshold(self, V_th_base):
        t = self._step_i
        vth = None
        if self._probe is not None and self._probe["lo"] <= t < self._probe["hi"]:
            vth = np.array(V_th_base, float, copy=True)
            vth[:self.NE][self._probe["target_E"]] -= self._probe["delta"]   # LOWER threshold on E target
        if self._supp is not None and self._supp["lo"] <= t < self._supp["hi"]:
            if vth is None:
                vth = np.array(V_th_base, float, copy=True)
            vth[:self.NE][self._supp["target_E"]] += self._supp["delta"]      # RAISE threshold on E target
        return V_th_base if vth is None else vth

    def step(self, spk, labels, dt):
        # z counterfactual transform: applied ONCE at branch_step, before the freeze record (§8).
        if (self._branch_step is not None and self._step_i == self._branch_step
                and self._z_transform is not None and not self._z_transform_done):
            self.z[self.is_E] = np.clip(self._z_transform(self.z[self.is_E].copy()), 0.0, 1.0)
            self._z_transform_done = True
        # freeze: from branch_step onward, hold z/m constant (record traces + snapshot, skip ODE) (§7.2).
        if self._freeze and self._branch_step is not None and self._step_i >= self._branch_step:
            self._record_traces(np.asarray(spk, bool))
            if self._snap_steps is not None and self._step_i in self._snap_steps:
                self._capture(self._snap_steps[self._step_i])
            self._step_i += 1
            return
        super().step(spk, labels, dt)

    # ---------------------------------------------------------------- q_eff readout (§5.2)
    def qeff_fields(self):
        """Per-E-neuron q_eff = sum(z*I_I)/(sum(I_I)+eps) and p_deplete = mean(I_I>=I_th), per registered
        window label. Returns {label: dict(q_eff[NE], p_deplete[NE], n_steps)}. Empty windows -> NaN."""
        out = {}
        for lab, acc in self.qeff_accum.items():
            n = acc["n_steps"]
            with np.errstate(invalid="ignore", divide="ignore"):
                q = acc["S_zI"] / (acc["S_I"] + 1e-9)
            q = np.where(acc["S_I"] > 0, q, np.nan)
            pdep = acc["cnt_deplete"] / n if n > 0 else np.full(self.NE, np.nan)
            out[lab] = dict(q_eff=q, p_deplete=pdep, n_steps=int(n))
        return out


# ======================================================================== resumable engine loop (spec §7.1)
@dataclass
class LoopState:
    """Full mutable engine state captured at a branch step so amplitude/location forks resume identically.
    Deep-copied on capture and on each fork (forks mutate V/currents/rings)."""
    t: int
    V: np.ndarray
    ref: np.ndarray
    s_E: np.ndarray
    I_E: np.ndarray
    s_I: np.ndarray
    I_I: np.ndarray
    ring_sE: np.ndarray
    ring_sI: np.ndarray
    xi: float
    rng_state: dict
    slow: object


def _loop_consts(p, net):
    """Precompute the decays / delays / external scale EXACTLY as simulate_kick (bit-identical numerics)."""
    NE, NI = net["NE"], net["NI"]
    N = NE + NI
    labels = net["labels"]
    ampa = net["ampa_by_delay"]; gaba = net["gaba_by_delay"]
    M = net["max_delay_steps"] + 1
    dt = p.dt
    decay_sE = np.exp(-dt / p.tau_r_AMPA); decay_IE = np.exp(-dt / p.tau_d_AMPA)
    decay_sI = np.exp(-dt / p.tau_r_GABA); decay_II = np.exp(-dt / p.tau_d_GABA)
    tau_m = np.where(labels == 0, p.tau_m_E, p.tau_m_I).astype(np.float64)
    decay_V = np.exp(-dt / tau_m)
    ref_steps = np.where(labels == 0, int(round(p.tau_ref_E / dt)), int(round(p.tau_ref_I / dt))).astype(np.int32)
    ext_incr = (tau_m / p.tau_r_AMPA) * np.where(labels == 0, p.J_ext_E, p.J_ext_I)
    ampa_bins = [d for d in range(M) if ampa[d].nnz > 0]
    gaba_bins = [d for d in range(M) if gaba[d].nnz > 0]
    if "ampa_flat" not in net:
        net["ampa_flat"] = _flatten_by_source(ampa, ampa_bins, NE)
        net["gaba_flat"] = _flatten_by_source(gaba, gaba_bins, NI)
    nu_theta, _, _ = compute_nu_theta(p)
    sigma_n_inv_ms = p.sigma_n * 1e-3
    sigma_xi = sigma_n_inv_ms * np.sqrt(p.tau_n / 2.0)
    ou_a = np.exp(-dt / p.tau_n)
    ou_b = sigma_xi * np.sqrt(1.0 - ou_a * ou_a)
    return dict(NE=NE, NI=NI, N=N, labels=labels, M=M, dt=dt, decay_sE=decay_sE, decay_IE=decay_IE,
                decay_sI=decay_sI, decay_II=decay_II, decay_V=decay_V, ref_steps=ref_steps,
                ext_incr=ext_incr, nu_theta=nu_theta, ou_a=ou_a, ou_b=ou_b,
                nu_sig_const=p.nu_ext_ratio * nu_theta)


def run_loop(p, net, slow, V_th_per_neuron, *, n_steps, start=None, capture_final=False,
             store_spikes=True, early_stop_runaway=False, es_thresh_hz=120.0, es_dur_ms=100.0):
    """Faithful resumable copy of simulate_kick's default+slow integration loop (no kick, no LFP, no
    STD/SG/shunt/fb/perturb). Reuses kick_probe numerics so a fresh run is bit-identical to simulate_kick
    with the same seed/substrate/slow (parity test). Reproduces the exact RNG draw order:
      pre-loop: ras_keepE = rng.choice(NE, min(80,NE)); ras_keepI = NE + rng.choice(NI, min(20,NI))
      per step: xi update (1 standard_normal); ext = rng.poisson(nu_vec*dt, size=N).

    start=None -> fresh run from t=0 (draws ras_keep). start=LoopState -> resume from the checkpoint
    (NO ras_keep redraw; rng continues from the saved state). capture_final=True -> after executing all
    n_steps (steps t0..t0+n_steps-1), deep-copy the full engine state at t=t0+n_steps (the BRANCH state,
    ready for a frozen/probed continuation) into res['checkpoint']. Not captured if an early-stop truncated.
    """
    c = _loop_consts(p, net)
    NE, NI, N, M, dt = c["NE"], c["NI"], c["N"], c["M"], c["dt"]
    labels = c["labels"]
    a_indptr, a_dst, a_dly, a_w = net["ampa_flat"]
    g_indptr, g_dst, g_dly, g_w = net["gaba_flat"]
    rng = net["rng"]
    base_vth = p.V_th if V_th_per_neuron is None else np.asarray(V_th_per_neuron, float)

    if start is None:
        t0 = 0
        V = np.full(N, p.V_reset, dtype=np.float64)
        ref = np.zeros(N, dtype=np.int32)
        s_E = np.zeros(N); I_E = np.zeros(N); s_I = np.zeros(N); I_I = np.zeros(N)
        ring_sE = np.zeros((M, N)); ring_sI = np.zeros((M, N))
        xi = 0.0
        _ = rng.choice(NE, size=min(80, NE), replace=False)          # stream-fidelity: match simulate_kick
        _ = NE + rng.choice(NI, size=min(20, NI), replace=False)
    else:
        t0 = int(start.t)
        V = start.V.copy(); ref = start.ref.copy()
        s_E = start.s_E.copy(); I_E = start.I_E.copy(); s_I = start.s_I.copy(); I_I = start.I_I.copy()
        ring_sE = start.ring_sE.copy(); ring_sI = start.ring_sI.copy()
        xi = float(start.xi)
        rng.bit_generator.state = copy.deepcopy(start.rng_state)

    rate_E = np.zeros(n_steps); rate_I = np.zeros(n_steps)
    E_spk_bool = np.zeros((n_steps, NE), dtype=bool) if store_spikes else None
    _es_alpha = 1.0 - np.exp(-dt / 20.0); _es_ema = 0.0
    _es_dur = int(round(es_dur_ms / dt)); _es_run = 0; _stop_k = n_steps
    checkpoint = None

    for k in range(n_steps):
        t = t0 + k
        tm = t * dt
        # ----- external OU + Poisson (exact draw order) -----
        xi = c["ou_a"] * xi + c["ou_b"] * rng.standard_normal()
        nu_now = c["nu_sig_const"] + xi
        if nu_now < 0.0:
            nu_now = 0.0
        s_E *= c["decay_sE"]; s_I *= c["decay_sI"]
        slot = t % M
        s_E += ring_sE[slot]; ring_sE[slot] = 0.0
        s_I += ring_sI[slot]; ring_sI[slot] = 0.0
        nu_vec = np.full(N, max(nu_now, 0.0))
        ext = rng.poisson(nu_vec * dt, size=N).astype(np.float64)
        s_E += ext * c["ext_incr"]
        # ----- synaptic currents -----
        I_E = s_E + (I_E - s_E) * c["decay_IE"]
        I_I = s_I + (I_I - s_I) * c["decay_II"]
        # ----- slow layer -----
        if slow is not None:
            I_net = slow.apply_currents(I_E, I_I, labels)
            V_th_eff = slow.threshold(base_vth)
        else:
            I_net = I_E - I_I
            V_th_eff = base_vth
        # ----- membrane + spike + refractory -----
        ref -= 1
        np.maximum(ref, 0, out=ref)
        free = ref == 0
        if slow is not None:
            Vtmp = I_net + (V - I_net) * c["decay_V"]
        else:
            Vtmp = membrane_step(V, I_E, I_I, c["decay_V"])
        V = np.where(free, Vtmp, p.V_reset)
        spk = free & (V >= (V_th_eff if np.isscalar(V_th_eff) else V_th_eff))
        V[spk] = p.V_reset
        ref[spk] = c["ref_steps"][spk]
        if slow is not None:
            slow.step(spk, labels, dt)
        rate_E[k] = spk[:NE].sum(); rate_I[k] = spk[NE:].sum()
        if store_spikes:
            E_spk_bool[k] = spk[:NE]
        # ----- scatter (sparse, exact) -----
        spE = np.where(spk[:NE])[0]; spI = np.where(spk[NE:])[0]
        if spE.size:
            st = a_indptr[spE]; cnt = a_indptr[spE + 1] - st; tot = int(cnt.sum())
            if tot:
                idx = (np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt) + np.repeat(st, cnt))
                np.add.at(ring_sE, ((t + a_dly[idx]) % M, a_dst[idx]), a_w[idx])
        if spI.size:
            st = g_indptr[spI]; cnt = g_indptr[spI + 1] - st; tot = int(cnt.sum())
            if tot:
                idx = (np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt) + np.repeat(st, cnt))
                np.add.at(ring_sI, ((t + g_dly[idx]) % M, g_dst[idx]), g_w[idx])
        if early_stop_runaway:
            _es_ema += _es_alpha * (rate_E[k] / NE / dt * 1e3 - _es_ema)
            _es_run = _es_run + 1 if _es_ema >= es_thresh_hz else 0
            if _es_run >= _es_dur:
                _stop_k = k + 1
                break

    if capture_final and _stop_k >= n_steps:
        checkpoint = LoopState(
            t=t0 + n_steps, V=V.copy(), ref=ref.copy(), s_E=s_E.copy(), I_E=I_E.copy(), s_I=s_I.copy(),
            I_I=I_I.copy(), ring_sE=ring_sE.copy(), ring_sI=ring_sI.copy(), xi=float(xi),
            rng_state=copy.deepcopy(rng.bit_generator.state), slow=copy.deepcopy(slow))
    if _stop_k < n_steps:
        rate_E = rate_E[:_stop_k]; rate_I = rate_I[:_stop_k]
        if store_spikes:
            E_spk_bool = E_spk_bool[:_stop_k]
    return dict(rate_E=rate_E / NE / dt * 1e3, rate_I=rate_I / NI / dt * 1e3,
                E_spk_bool=E_spk_bool, n_steps=len(rate_E), t0=t0,
                runaway_early_stop_step=(None if _stop_k >= n_steps else t0 + _stop_k),
                checkpoint=checkpoint)


# ======================================================================== operational-runaway scorer (§7.2)
def score_runaway(rate_E_hz, dt, *, thresh_hz=120.0, dur_ms=100.0):
    """First time (ms, relative to trace start) the 20-ms-EMA per-neuron E-rate is sustained >= thresh_hz
    for >= dur_ms. Returns None if never. Mirrors run_m4_dynamic_qi._smooth + _first_sustained (the locked
    120 Hz / 100 ms criterion) with an in-module EMA so scoring works on a resumed continuation trace."""
    r = np.asarray(rate_E_hz, float)
    if r.size == 0:
        return None
    alpha = 1.0 - np.exp(-dt / 20.0)
    ema = np.zeros_like(r)
    acc = 0.0
    for i in range(r.size):
        acc += alpha * (r[i] - acc)
        ema[i] = acc
    need = int(round(dur_ms / dt))
    run = 0
    for i in range(r.size):
        run = run + 1 if ema[i] >= thresh_hz else 0
        if run >= need:
            return float((i - need + 1) * dt)
    return None


# ======================================================================== region masks (spec §5.1)
def build_region_masks(pos_E, src_xy, snk_xy, axis_unit, core_r, *, corridor_halfwidth):
    """E-indexed boolean masks for the six required regions (spec §5.1). corridor = band of `corridor_halfwidth`
    (same physical units as pos) perpendicular distance to the source->sink axis line, between the two cores."""
    pos = np.asarray(pos_E, float)
    src = np.asarray(src_xy, float); snk = np.asarray(snk_xy, float)
    u = np.asarray(axis_unit, float)
    source_core = np.linalg.norm(pos - src, axis=1) <= core_r
    sink_core = np.linalg.norm(pos - snk, axis=1) <= core_r
    core = source_core | sink_core
    rel = pos - src
    proj = rel @ u                                  # signed distance along axis from source
    axis_len = float(np.linalg.norm(snk - src))
    perp = np.linalg.norm(rel - np.outer(proj, u), axis=1)
    corridor = (perp <= corridor_halfwidth) & (proj >= -core_r) & (proj <= axis_len + core_r)
    return dict(all_E=np.ones(len(pos), bool), source_core=source_core, sink_core=sink_core,
                axis_corridor=corridor, off_axis=~corridor, core_excluded=~core)


# ======================================================================== slow-state coordinates (§5.1)
def natural_zm_trajectory(z_mean, adap_current, rate_hz, dt, *, I_EE_scale, downsample_ms):
    """Continuous MZ D–a state trajectory from the engine's streaming all-E traces (temporal
    phase-diagram §5.3). ``z_mean`` = MZSlowVars.trace_z_mean (mean z over E cells); ``adap_current``
    = trace_adap_current (== eta_m·mean(m over E) == A_abs); ``rate_hz`` = E population rate (Hz).

    Returns downsampled arrays: ``t_ms``, ``D_allE`` = 1 - z̄, ``a_allE`` = A_abs/I_EE_scale,
    ``rate_E_hz``. Downsampling averages within each ``downsample_ms`` window; a trailing partial
    (< one bin) remainder is dropped."""
    z_mean = np.asarray(z_mean, float)
    adap_current = np.asarray(adap_current, float)
    rate_hz = np.asarray(rate_hz, float)
    step = max(1, int(round(float(downsample_ms) / float(dt))))
    n_bins = z_mean.shape[0] // step

    def _avg(x):
        return x[:n_bins * step].reshape(n_bins, step).mean(axis=1)

    return dict(
        t_ms=np.arange(n_bins) * (step * float(dt)),
        D_allE=1.0 - _avg(z_mean),
        a_allE=_avg(adap_current) / float(I_EE_scale),
        rate_E_hz=_avg(rate_hz),
    )


def slow_state_coordinates(z_E, m_E, eta_m, region_masks):
    """D_z = 1 - mean(z), A = eta_m * mean(m) per region (spec §5.1). Returns {region: dict(D_z, A, n)}."""
    z = np.asarray(z_E, float); m = np.asarray(m_E, float)
    out = {}
    for name, mask in region_masks.items():
        mm = np.asarray(mask, bool)
        n = int(mm.sum())
        if n == 0:
            out[name] = dict(D_z=float("nan"), A=float("nan"), n=0)
            continue
        out[name] = dict(D_z=float(1.0 - z[mm].mean()), A=float(eta_m * m[mm].mean()), n=n)
    return out


def qeff_region_summary(q_eff_E, p_deplete_E, region_masks):
    """Region means of the current-aware q_eff and p_deplete fields (per-neuron -> region scalar)."""
    q = np.asarray(q_eff_E, float); pd = np.asarray(p_deplete_E, float)
    out = {}
    for name, mask in region_masks.items():
        mm = np.asarray(mask, bool)
        qv = q[mm]; pv = pd[mm]
        qv = qv[np.isfinite(qv)]; pv = pv[np.isfinite(pv)]
        out[name] = dict(q_eff=float(qv.mean()) if qv.size else float("nan"),
                         p_deplete=float(pv.mean()) if pv.size else float("nan"), n=int(mm.sum()))
    return out


# ======================================================================== z_bar vs q_eff mapping audit (§5.2)
def zbar_qeff_field_audit(zbar_field, qeff_field, region_field_masks=None):
    """Spatial agreement between the z_bar-derived and current-aware q_eff coarse fields (spec §5.2):
    Spearman rho, cosine similarity, mean/max abs difference (global + optional region masks), and whether
    the inferred preferred orientation flips. `*_field` are 2-D (n,n) grids on the SAME coarse grid."""
    from scipy.stats import spearmanr
    a = np.asarray(zbar_field, float).ravel()
    b = np.asarray(qeff_field, float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    a2, b2 = a[ok], b[ok]

    def _cos(x, y):
        nx, ny = np.linalg.norm(x), np.linalg.norm(y)
        return float(x @ y / (nx * ny)) if nx > 0 and ny > 0 else float("nan")

    rho = float(spearmanr(a2, b2).statistic) if a2.size > 2 else float("nan")
    out = dict(spearman=rho, cosine=_cos(a2, b2),
               mean_abs_diff=float(np.mean(np.abs(a2 - b2))) if a2.size else float("nan"),
               max_abs_diff=float(np.max(np.abs(a2 - b2))) if a2.size else float("nan"),
               n=int(a2.size))
    if region_field_masks:
        out["regions"] = {}
        for name, m in region_field_masks.items():
            mm = np.asarray(m, bool).ravel() & ok
            if mm.sum():
                out["regions"][name] = dict(mean_abs_diff=float(np.mean(np.abs(a[mm] - b[mm]))),
                                            max_abs_diff=float(np.max(np.abs(a[mm] - b[mm]))), n=int(mm.sum()))
    return out


# ======================================================================== realized (D,A) grid fields (§6.1)
def realized_D_grid(baseline_D, max_onset_D, *, n_D, clip, overshoot):
    """Nine equally spaced realized-D values from pooled baseline D to overshoot*max onset D, clipped (§6.1)."""
    hi = min(clip[1], overshoot * max_onset_D)
    lo = max(clip[0], baseline_D)
    return np.linspace(lo, hi, int(n_D))


def build_DA_q_field(onset_depletion_field, target_D):
    """Primary z-pattern (§6.1): pooled primary-onset DEPLETION field (1-z) normalized to mean depletion 1,
    then scaled to the requested mean D, converted to q = 1 - D_field, clipped [0,1]. Returns (q_field, D_field)."""
    dep = np.asarray(onset_depletion_field, float)
    m = np.nanmean(dep)
    shape = dep / m if m > 0 else np.ones_like(dep)          # mean-1 normalized depletion pattern
    D_field = np.clip(shape * float(target_D), 0.0, 1.0)     # scale to requested mean D
    q_field = np.clip(1.0 - D_field, 0.0, 1.0)
    return q_field, D_field


def DA_controls(depletion_field, *, shuffle_seed):
    """Control z-patterns for the (D,A) grid (§6.1): uniform, rotated_90, spatial_shuffle, z_blocked.
    Operate on the mean-1 normalized depletion PATTERN so `build_DA_q_field(control, D)` scales consistently."""
    dep = np.asarray(depletion_field, float)
    m = np.nanmean(dep)
    shape = dep / m if m > 0 else np.ones_like(dep)
    rng = np.random.default_rng(int(shuffle_seed))
    flat = shape.ravel().copy(); rng.shuffle(flat)
    return dict(primary=shape.copy(), uniform=np.ones_like(shape),
                rotated_90=np.rot90(shape).copy(), spatial_shuffle=flat.reshape(shape.shape),
                z_blocked=np.zeros_like(shape))   # z_blocked = no depletion pattern (D applied uniformly=0 -> q=1)


# ======================================================================== ignition classification (§7.2)
def epsilon_c_from_ladder(ladder, ran_away):
    """epsilon_c = smallest amplitude in `ladder` that produced runaway (spec §7.2). ran_away is a bool
    list aligned with ladder (sorted ascending). Returns dict(epsilon_c, censored, zero_runaway, bracket).
    zero-probe runaway -> epsilon_c=0; no amplitude ignites -> right-censored (epsilon_c=None)."""
    lad = list(ladder)
    ra = [bool(x) for x in ran_away]
    if ra and ra[0] and lad[0] == 0.0:
        return dict(epsilon_c=0.0, censored=False, zero_runaway=True, bracket=None)
    first = next((i for i, x in enumerate(ra) if x), None)
    if first is None:
        return dict(epsilon_c=None, censored=True, zero_runaway=False, bracket=None)
    bracket = (lad[first - 1], lad[first]) if first > 0 else (0.0, lad[first])
    return dict(epsilon_c=float(lad[first]), censored=False, zero_runaway=False, bracket=bracket)


def classify_ignition(per_state):
    """Result-NEUTRAL ignition trajectory label across registered states (spec §7.2). No category is a gate.

    per_state: list of dicts ordered baseline->onset, each with keys:
      alpha1 (float or None), epsilon_c (float or None -> censored), axial_gain, perp_gain, global_gain,
      seed_consistent (bool). Returns one of the spec labels."""
    resolved = [s for s in per_state if s.get("epsilon_c") is not None]
    if not resolved:
        return "unresolved"
    if any(not s.get("seed_consistent", True) for s in per_state):
        return "seed-inconsistent"
    eps = [s["epsilon_c"] for s in resolved]
    decreasing = len(eps) >= 2 and eps[-1] < eps[0] - 1e-9
    near_zero = min(eps) <= 1e-9
    a1 = [s.get("alpha1") for s in resolved if s.get("alpha1") is not None]
    a1_crosses = bool(a1) and max(a1) >= 0.0
    axial_pref = np.median([s.get("axial_gain", 0.0) - s.get("perp_gain", 0.0) for s in resolved]) > 1e-3
    global_dom = np.median([s.get("global_gain", 0.0) - max(s.get("axial_gain", 0.0), s.get("perp_gain", 0.0))
                            for s in resolved]) > 1e-3
    if near_zero and a1_crosses:
        return "linear_crossing"
    if decreasing and not a1_crosses:
        return "finite_amplitude_escape"
    if not decreasing and not near_zero:
        return "state-invariant threshold"
    if global_dom:
        return "uniform_global_amplification"
    if axial_pref:
        return "axis-selective susceptibility"
    return "unresolved"


# ======================================================================== projected-flow eligibility (§5.3)
def projected_flow_eligibility(visits, *, min_visits=3, min_seeds=2, min_sign_agree=2, n_seeds_total=3):
    """Decide whether a (D,A) bin can carry a drift arrow / nullcline (spec §5.3 rules 1-5).

    visits: list of dicts per visit to a bin, each dict(seed, dD, dA) (finite-difference drift components).
    Eligible for an ARROW iff: >= min_visits independent visits from >= min_seeds seeds AND, for EACH
    displayed component (dD, dA), the drift sign agrees in >= min_sign_agree of the seeds. Returns
    dict(eligible, n_visits, n_seeds, dD_mean, dA_mean, sign_ok_dD, sign_ok_dA)."""
    vs = list(visits)
    seeds = sorted({v["seed"] for v in vs})

    def _sign_agree(key):
        per_seed_sign = {}
        for v in vs:
            s = np.sign(v[key])
            per_seed_sign.setdefault(v["seed"], []).append(s)
        seed_signs = [np.sign(np.sum(sl)) for sl in per_seed_sign.values() if np.sum(np.abs(sl)) > 0]
        if not seed_signs:
            return False
        pos = sum(1 for s in seed_signs if s > 0)
        neg = sum(1 for s in seed_signs if s < 0)
        return max(pos, neg) >= min_sign_agree

    sign_ok_dD = _sign_agree("dD")
    sign_ok_dA = _sign_agree("dA")
    eligible = (len(vs) >= min_visits and len(seeds) >= min_seeds and sign_ok_dD and sign_ok_dA)
    return dict(eligible=bool(eligible), n_visits=len(vs), n_seeds=len(seeds),
                dD_mean=float(np.mean([v["dD"] for v in vs])) if vs else float("nan"),
                dA_mean=float(np.mean([v["dA"] for v in vs])) if vs else float("nan"),
                sign_ok_dD=bool(sign_ok_dD), sign_ok_dA=bool(sign_ok_dA))


# ======================================================================== focused-m aggregation (task §4)
FOCUSED_M_REQUIRED_FIELDS = ("seed", "z_regime", "A_frac", "tau_adp_ms", "eta_m", "realized_a_max",
                             "D_max", "phenotype", "runaway_ms", "n_events", "event_bar")
FOCUSED_M_MAIN_SEEDS = (1, 3, 4)
FOCUSED_M_MAIN_A_FRACS = (0.0, 0.001, 0.0025, 0.005, 0.0075, 0.01)
FOCUSED_M_MAIN_TAU_ADP_MS = 2000.0


def validate_focused_m_grid(rows, *, seeds=FOCUSED_M_MAIN_SEEDS, a_fracs=FOCUSED_M_MAIN_A_FRACS,
                            tau_adp_ms=FOCUSED_M_MAIN_TAU_ADP_MS):
    """Fail-loud validation of the aggregated focused-m MAIN gap grid (task §4.1).

    RAISES on: a row missing a required field (schema misalignment / stale old-format file), a tau_adp that
    is not the grid tau (tau contamination), a duplicate (seed, A_frac), or any missing/extra (seed, A_frac)
    cell versus the expected len(seeds) x len(a_fracs) grid. Returns the rows sorted by (seed, A_frac); does
    not mutate the inputs.
    """
    seen = {}
    for r in rows:
        for f in FOCUSED_M_REQUIRED_FIELDS:
            if f not in r:
                raise ValueError(f"focused_m row missing required field {f!r} (schema misalignment / stale file): {r}")
        if float(r["tau_adp_ms"]) != float(tau_adp_ms):
            raise ValueError(f"focused_m tau contamination: expected tau_adp_ms={tau_adp_ms}, got "
                             f"{r['tau_adp_ms']} for seed={r.get('seed')} A_frac={r.get('A_frac')}")
        key = (int(r["seed"]), round(float(r["A_frac"]), 6))
        if key in seen:
            raise ValueError(f"focused_m duplicate (seed, A_frac)={key}")
        seen[key] = r
    expected = {(int(s), round(float(a), 6)) for s in seeds for a in a_fracs}
    got = set(seen)
    if got != expected:
        raise ValueError(f"focused_m grid mismatch: missing={sorted(expected - got)} extra={sorted(got - expected)}")
    return sorted(seen.values(), key=lambda r: (int(r["seed"]), float(r["A_frac"])))


def tau_phenotype_denominators(rows):
    """Per-tau phenotype counts + runaway denominator for the tau-sensitivity summary (task §4.2)."""
    from collections import Counter
    out = {}
    for tau in sorted({float(r["tau_adp_ms"]) for r in rows}, reverse=True):
        sub = [r for r in rows if float(r["tau_adp_ms"]) == tau]
        out[f"tau{int(tau)}"] = {"n": len(sub),
                                 "n_runaway": int(sum(1 for r in sub if r["phenotype"] == "runaway")),
                                 "phenotypes": dict(Counter(r["phenotype"] for r in sub))}
    return out


def build_tau_sensitivity(row_by_seed_tau, *, seeds=FOCUSED_M_MAIN_SEEDS, taus=(2000.0, 1000.0, 500.0),
                          a_frac=0.001):
    """Assemble + validate the tau-sensitivity summary (task §4.2): exactly one (seed, tau) row at a fixed
    A_frac. RAISES on a missing/duplicate/misplaced cell or a row whose A_frac/tau_adp_ms disagree with its
    slot. Returns (rows_sorted_by_seed_then_descending_tau, phenotype_denominators)."""
    rows = []
    for s in seeds:
        for tau in taus:
            key = (int(s), float(tau))
            if key not in row_by_seed_tau:
                raise ValueError(f"tau-sensitivity missing cell (seed, tau_adp)={key}")
            r = row_by_seed_tau[key]
            if abs(float(r["A_frac"]) - float(a_frac)) > 1e-12:
                raise ValueError(f"tau-sensitivity cell {key}: A_frac {r['A_frac']} != {a_frac}")
            if float(r["tau_adp_ms"]) != float(tau):
                raise ValueError(f"tau-sensitivity cell {key}: tau_adp_ms {r['tau_adp_ms']} != {tau}")
            rows.append(r)
    rows = sorted(rows, key=lambda r: (int(r["seed"]), -float(r["tau_adp_ms"])))
    return rows, tau_phenotype_denominators(rows)
