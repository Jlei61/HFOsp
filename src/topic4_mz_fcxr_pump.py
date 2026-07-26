"""FCXR pump lifecycle — dimensionless activity-dependent load u_i and its electrogenic pump current.

NAMING CONTRACT (spec §2.1): ``u_i`` is an **activity-dependent intracellular load
(Na/pump-inspired)**. It is NOT an intracellular sodium concentration, an ATP model, or a complete
ionic-homeostasis model, and must never be reported as one.

    phi(u) = u^h / (1 + u^h)                      h = 3 fixed for the primary tier
    du/dt  = a_load * S_i(t) - phi(u_i)/tau_N     S_i = spike train
    I_pump_excess = Imax * [phi(u_i) - p0_i]      distributionally baseline-centered, NO positive part

Three clauses that are science contract, not implementation detail (spec §2.2/§2.3):

  1. the spike jump is PER SPIKE (never scaled by dt); the clearance IS scaled by dt/tau_N;
  2. the SAME phi drives the clearance and the membrane current;
  3. the membrane effect is Imax*(phi-p0) with NO positive part -- ``+Imax*p0`` compensates the
     steady state already implicit in the FCXR baseline, so a negative excess only means "pump
     activation below the baseline reference", never "pump running backwards".

Contract enumerated 1:1 in tests/test_topic4_mz_fcxr_pump.py.
Design: docs/superpowers/specs/2026-07-26-topic4-mz-fcxr-pump-lifecycle-design.md
"""
from __future__ import annotations

import numpy as np

PRIMARY_H = 3


def require_primary_h(h):
    """Primary tier fixes h=3 (spec §2.1); h in {2,4} is a DEFERRED sensitivity, not a sweep axis."""
    if int(h) != PRIMARY_H:
        raise ValueError(f"primary tier requires h={PRIMARY_H} (got {h}); h in {{2,4}} is a "
                         "deferred Tier-A sensitivity, not a primary sweep axis")


def pump_activation(u, h=PRIMARY_H):
    """phi(u) = u^h/(1+u^h): monotone, smooth, in [0,1), phi(0)=0. Same phi for clearance + membrane."""
    uh = np.asarray(u, float) ** h
    return uh / (1.0 + uh)


def step_spike_load(u, spikes, *, a_load, tau_N, dt, h=PRIMARY_H):
    """One discrete load step (spec §2.2), evaluated at the PRE-step load u(t^-):

        u(t+dt) = max[0, u(t) + a_load*N_spike - (dt/tau_N)*phi(u(t))]

    ``spikes`` is a per-cell spike COUNT for this step (a bool mask is the count 0/1 case). The jump
    carries no dt; the clearance carries dt/tau_N. Non-finite load fails fast -- a candidate that
    blows up is a failed candidate, not something to clamp (spec §2.2 "safety cap = fail-fast").
    """
    u = np.asarray(u, float)
    if not np.all(np.isfinite(u)):
        raise FloatingPointError("non-finite activity-dependent load u")
    jump = a_load * np.asarray(spikes, float)                 # per-spike, NOT scaled by dt
    clearance = (dt / tau_N) * pump_activation(u, h)          # scaled by dt/tau_N, at u(t^-)
    return np.maximum(u + jump - clearance, 0.0)


def excess_pump_current(u, p0, *, Imax, h=PRIMARY_H):
    """Baseline-centered electrogenic pump current Imax*[phi(u)-p0] subtracted from the E drive.

    NO positive part: phi<p0 gives a negative excess (pump activation below the baseline reference),
    which cancels the mean bias that a rectifier would inject from baseline fluctuations alone.
    """
    return Imax * (pump_activation(u, h) - np.asarray(p0, float))


# =====================================================================================
# Offline load integration — with Imax=0 (sensor-only) the load is a PURE FUNCTION of the
# recorded spike raster, so one simulation calibrates a whole (a_load, tau_N) candidate set.
# =====================================================================================
def integrate_load_from_raster(E_spk_bool, *, a_load, tau_N, dt, h=PRIMARY_H, u0=None,
                               snapshot_steps=(), block_edges=()):
    """Replay the sensor-only load mass balance over a recorded E raster (n_steps x NE).

    Returns ``(u_final, snapshots, block_phi_mean, block_spike_count, u_mean_trace)`` where
    ``block_phi_mean[b]`` is the per-cell mean of phi(u(t^-)) over block b (the activation the
    membrane WOULD have seen), and ``snapshots[k]`` is the load field after step ``snapshot_steps[k]``.
    Only landmark fields and per-block means are kept -- never an n_steps x NE load matrix.
    """
    E_spk_bool = np.asarray(E_spk_bool)
    n_steps, NE = E_spk_bool.shape
    u = np.zeros(NE) if u0 is None else np.asarray(u0, float).copy()
    edges = [(int(a), int(b)) for a, b in block_edges]
    phi_sum = np.zeros((len(edges), NE)); spk_sum = np.zeros((len(edges), NE), dtype=np.int64)
    n_in_block = np.zeros(len(edges), dtype=np.int64)
    snaps, want = {}, {int(s): k for k, s in enumerate(snapshot_steps)}
    u_mean = np.empty(n_steps)
    for t in range(n_steps):
        uh = u ** h
        phi = uh / (1.0 + uh)                                  # phi(u(t^-))
        spk = E_spk_bool[t]
        for b, (lo, hi) in enumerate(edges):
            if lo <= t < hi:
                phi_sum[b] += phi; spk_sum[b] += spk; n_in_block[b] += 1
        u = np.maximum(u + a_load * spk - (dt / tau_N) * phi, 0.0)
        u_mean[t] = u.mean()
        if t in want:
            snaps[want[t]] = u.copy()
    block_phi_mean = phi_sum / np.maximum(n_in_block, 1)[:, None]
    return u, snaps, block_phi_mean, spk_sum, u_mean


def analytic_steady_load(rate_E_hz, *, a_load, tau_N, h=PRIMARY_H):
    """Per-cell equilibrium load from the TIME-AVERAGED mass balance a_load*r_i = phi(u_i*)/tau_N.

    Exact in the mean, so it removes the multi-second startup transient without simulating it: a
    tau_N=2000 ms candidate would otherwise still be climbing at the end of an 8 s baseline and
    every p0_i would be a transient, not a baseline expectation.

    Returns ``(u_star, frac_divergent)``. ``y_i = a_load*r_i*tau_N >= 1`` means the cell's clearance
    saturates (phi pinned at 1) and its load has NO steady state -- a candidate with any such cell at
    baseline is inadmissible, not something to clamp.
    """
    r = np.asarray(rate_E_hz, float) * 1e-3                    # Hz -> spikes/ms
    y = a_load * r * tau_N
    div = y >= 1.0
    y_safe = np.clip(y, 0.0, 1.0 - 1e-12)
    return (y_safe / (1.0 - y_safe)) ** (1.0 / h), float(div.mean())


def event_locked_load_visibility(phi_on, phi_off, participating, phi_quiet_a, phi_quiet_b,
                                 k_visible=3.0):
    """Spec §I1 clause A1: does an ordinary interictal event visibly move the load OF THE CELLS THAT
    PARTICIPATE IN IT?

    PRE-REGISTRATION CORRECTION (2026-07-26, locked before the re-run): the first formulation
    compared the rise of the POPULATION-MEAN activation against its own fluctuation. That is not a
    faithful reading of "an isolated IED must produce a measurable load excursion": an event recruits
    only a few percent of E cells, so the population mean dilutes a participating cell's excursion by
    ~25x, and the population-mean fluctuation is dominated by residual equilibration drift rather
    than by events. The corrected clause is per cell and event-locked, with a MATCHED control: the
    same participating cells over an equally long interval in which no event was detected.

        rise_k  = median over cells participating in event k of  phi_i(t_off) - phi_i(t_on)
        quiet_k = median over the SAME cells of |phi_i(b) - phi_i(a)| on a matched-length quiet gap
        visible <=> median_k(rise_k) >= k_visible * median_k(quiet_k)

    All arrays are (n_events, NE); `participating` is the boolean per-event participation mask.
    """
    phi_on, phi_off = np.asarray(phi_on, float), np.asarray(phi_off, float)
    phi_qa, phi_qb = np.asarray(phi_quiet_a, float), np.asarray(phi_quiet_b, float)
    part = np.asarray(participating, bool)
    rises, quiets = [], []
    for k in range(part.shape[0]):
        m = part[k]
        if not m.any():
            continue
        rises.append(float(np.median(phi_off[k][m] - phi_on[k][m])))
        quiets.append(float(np.median(np.abs(phi_qb[k][m] - phi_qa[k][m]))))
    if not rises:
        return dict(n_events_scored=0, rise_median=float("nan"), quiet_median=float("nan"),
                    ratio=float("nan"), visible=False)
    rm, qm = float(np.median(rises)), float(np.median(quiets))
    return dict(n_events_scored=len(rises), rise_median=rm, quiet_median=qm,
                ratio=float(rm / qm) if qm > 0 else float("inf"),
                visible=bool(rm >= k_visible * qm))


def matched_quiet_intervals(event_steps, quiet_segments):
    """Pair each event with an equally long interval drawn from the no-event segments (deterministic
    round-robin over the available quiet segments). Events with no quiet segment long enough are
    dropped -- never silently paired with a shorter, easier control."""
    segs = [(int(a), int(b)) for a, b in quiet_segments]
    out, j = [], 0
    for a, b in event_steps:
        L = int(b) - int(a)
        if L <= 0 or not segs:
            out.append(None)
            continue
        picked = None
        for _ in range(len(segs)):
            s0, s1 = segs[j % len(segs)]
            j += 1
            if s1 - s0 >= L:
                picked = (s0, s0 + L)
                break
        out.append(picked)
    return out


# =====================================================================================
# p0 calibration: rate-decile grouping + empirical shrinkage (spec §2.4)
# =====================================================================================
def rate_decile_groups(baseline_rate_E, n_groups=10):
    """Group E cells by pump-OFF baseline firing-rate decile. Grouping never uses source/sink/axis
    labels -- p0 must not be allowed to encode a permanent source->sink gradient (spec §2.4/§7)."""
    r = np.asarray(baseline_rate_E, float)
    edges = np.quantile(r, np.linspace(0.0, 1.0, n_groups + 1)[1:-1])
    return np.searchsorted(edges, r, side="right").astype(int)


def _group_means(values, groups, n_groups):
    out = np.zeros(n_groups)
    for g in range(n_groups):
        m = groups == g
        out[g] = values[m].mean() if m.any() else float(values.mean())
    return out


def apply_p0_shrinkage(raw_p0, group_p0, groups, weight):
    """p0_i = (1-w)*raw_i + w*groupmean(i). w=0 keeps the noisy per-cell mean, w=1 the decile mean."""
    return (1.0 - weight) * np.asarray(raw_p0, float) + weight * np.asarray(group_p0, float)[np.asarray(groups)]


def fit_p0_shrinkage(phi_block_means, groups, *, weights=None, n_groups=10):
    """Choose the shrinkage weight by PRE-REGISTERED inner leave-one-block-out CV inside the
    CALIBRATION blocks only (spec §2.4 clause 4). For each held-out block b, p0 is built from the
    remaining blocks and scored by squared prediction error against block b's per-cell mean phi.
    The final held-out trajectory never enters this fit.
    """
    P = np.asarray(phi_block_means, float)
    nb = P.shape[0]
    if nb < 3:
        raise ValueError(f"shrinkage CV needs >=3 calibration blocks, got {nb}")
    weights = np.linspace(0.0, 1.0, 11) if weights is None else np.asarray(weights, float)
    err = np.zeros(len(weights))
    for b in range(nb):
        tr = np.delete(P, b, axis=0).mean(axis=0)
        gm = _group_means(tr, groups, n_groups)
        for k, w in enumerate(weights):
            err[k] += float(np.mean((apply_p0_shrinkage(tr, gm, groups, w) - P[b]) ** 2))
    k_best = int(np.argmin(err))
    raw = P.mean(axis=0)
    gm = _group_means(raw, groups, n_groups)
    return dict(weight=float(weights[k_best]), cv_weights=weights.tolist(),
                cv_error=(err / nb).tolist(), raw_p0=raw, group_p0=gm,
                p0=apply_p0_shrinkage(raw, gm, groups, weights[k_best]))


# =====================================================================================
# Baseline equivalence: margins from pump-OFF block-to-block variability (spec §I2)
# =====================================================================================
def block_equivalence_margins(baseline_blocks, *, k=2.0):
    """Per-metric equivalence margin = k * (pump-off block-to-block SD) of that metric. Locked and
    written to disk BEFORE the held-out pump-ON result is looked at; "not significant" is never
    accepted as equivalence.  ``baseline_blocks``: list of {metric: value} dicts (calibration only).
    """
    keys = sorted({k2 for b in baseline_blocks for k2 in b})
    out = {}
    for key in keys:
        vals = np.array([b[key] for b in baseline_blocks if key in b and np.isfinite(b[key])], float)
        if vals.size < 3:
            raise ValueError(f"metric {key!r} needs >=3 finite calibration blocks, got {vals.size}")
        mu, sd = float(vals.mean()), float(vals.std(ddof=1))
        rel = abs(k * sd / mu) if mu != 0 else float("inf")
        # A margin wider than half the metric's own mean cannot distinguish "equivalent" from
        # "we could not tell": the metric is reported as UNDERPOWERED so a within-margin result on
        # it is never read as tight equivalence (spec §I2 "not significant" is not equivalence).
        out[key] = dict(mean=mu, sd=sd, margin=float(k * sd), n_blocks=int(vals.size), k=float(k),
                        margin_over_mean=rel, underpowered=bool(rel > 0.5))
    return out


def evaluate_baseline_equivalence(off_metrics, on_metrics, margins):
    """One-shot judgement on the FINAL held-out trajectory: |on-off| <= margin for every primary
    metric. Any metric outside its pre-locked margin fails Gate I-a (spec §I2)."""
    rows = {}
    for key, m in margins.items():
        if key not in off_metrics or key not in on_metrics:
            rows[key] = dict(status="MISSING", within=False)
            continue
        d = float(on_metrics[key]) - float(off_metrics[key])
        rows[key] = dict(off=float(off_metrics[key]), on=float(on_metrics[key]), delta=d,
                         margin=float(m["margin"]), within=bool(abs(d) <= m["margin"]),
                         underpowered=bool(m.get("underpowered", False)),
                         margin_over_mean=m.get("margin_over_mean"),
                         status="WITHIN" if abs(d) <= m["margin"] else "OUTSIDE")
    return dict(per_metric=rows, all_within=bool(all(r["within"] for r in rows.values())),
                n_outside=int(sum(not r["within"] for r in rows.values())),
                n_underpowered=int(sum(bool(r.get("underpowered")) for r in rows.values())),
                underpowered_metrics=sorted(k for k, r in rows.items() if r.get("underpowered")))


def required_ied_count(block_event_counts, minimum=20):
    """Event budget for the statistical-return window (spec §C6): at least `minimum` events, and at
    least the count a baseline block typically delivers. A no-event tail can never count as return."""
    counts = np.asarray(block_event_counts, float)
    return dict(n_ied_required=int(max(minimum, np.ceil(np.median(counts)))),
                minimum=int(minimum), block_median=float(np.median(counts)) if counts.size else 0.0)


# =====================================================================================
# Virtual-SEEG component observer (Gate I-a readout audit, spec §I3)
# =====================================================================================
class VirtualSeegComponentObserver:
    """Online per-contact aggregation of virtual-SEEG PROXY components. NOT a physical forward
    voltage solution.

    The blessed ``LFPRecorder`` proxy is ``|I_E|+|I_I|`` -- unsigned, so it cannot separate
    excitation from inhibition and cannot expose a pump term. This observer reuses that recorder's
    electrode weights but aggregates SIGNED components taken from the model's own conductances at
    the force-match anchor ``v_match`` (where the FCXR membrane defines its current<->conductance
    correspondence):

        excitatory  = sum_j w_j * gE_j * (E_E - v_match)          > 0
        inhibitory  = sum_j w_j * gI_j * (e_gaba - v_match)       < 0
        adaptation  = sum_j w_j * gM_j * (e_k - v_match)          == 0 this sprint (M off)
        pump        = sum_j w_j * (-I_pump_excess_j)
        no_direct_pump = excitatory + inhibitory + adaptation
        all_components = no_direct_pump + pump

    IDENTIFIABILITY CAVEAT (recorded in the audit artifact): the slow protocol never sees V, so the
    driving force is evaluated at v_match rather than at the instantaneous membrane potential. The
    components are therefore state-independent off v_match -- a synaptic-current proxy, exactly like
    the legacy |I_E|+|I_I| readout, but signed and pump-separable. No sign is inferred from a
    magnitude: every sign comes from the model's own reversal potentials.
    """

    COMPONENTS = ("legacy_abs", "excitatory", "inhibitory", "adaptation", "pump",
                  "no_direct_pump", "all_components")

    def __init__(self, lfp_recorder, cfg, z_threshold=None):
        # Reuse the blessed recorder's per-site neuron indices and normalized shape weights, and its
        # exact per-site np.dot reduction, so `legacy_abs` is BITWISE identical to LFPRecorder.sample.
        self._idx = list(lfp_recorder._idx)
        self._w = list(lfp_recorder._w)
        self.n_sites = len(lfp_recorder.sites)
        # Gate T slow flow: with global_gaba_fraction=0 and z_scope='local_only' the z depletion
        # sensor IS max(I_I[:NE],0), so the fraction of E cells BELOW I_th_EI (z_inf=1, recovering)
        # can be accumulated here without touching the engine. None -> not accumulated.
        self.z_threshold = z_threshold
        self.z_inf_high_sum = 0.0
        self.z_inf_n = 0
        self.NE = int(lfp_recorder.NE)
        self.f_E = float(cfg.E_E - cfg.v_match)
        self.f_I = float(cfg.e_gaba - cfg.v_match)
        self.f_M = float(cfg.e_k - cfg.v_match)
        self.traces = {c: [] for c in self.COMPONENTS}

    def _agg(self, per_cell):
        out = np.empty(self.n_sites)
        for k in range(self.n_sites):
            out[k] = np.dot(self._w[k], per_cell[self._idx[k]])
        return out

    def sample(self, I_E, I_I, gE, gI, gM, ex_pump):
        NE = self.NE
        if self.z_threshold is not None:
            self.z_inf_high_sum += float(np.mean(np.maximum(I_I[:NE], 0.0) < self.z_threshold))
            self.z_inf_n += 1
        legacy = self._agg(np.abs(I_E[:NE]) + np.abs(I_I[:NE]))
        exc = self._agg(gE) * self.f_E
        inh = self._agg(gI) * self.f_I
        adp = self._agg(gM) * self.f_M
        pmp = np.zeros(self.n_sites) if ex_pump is None else -self._agg(ex_pump)
        nodp = exc + inh + adp
        for name, val in (("legacy_abs", legacy), ("excitatory", exc), ("inhibitory", inh),
                          ("adaptation", adp), ("pump", pmp), ("no_direct_pump", nodp),
                          ("all_components", nodp + pmp)):
            self.traces[name].append(val)

    def stack(self):
        return {c: np.asarray(v, float) for c, v in self.traces.items()}

    def frac_z_inf_high(self):
        """Time-averaged fraction of E cells whose received inhibition sits below the depletion
        threshold (z_inf=1). Feeds branch_slow_flow's dZ/dt on the branch actually observed."""
        return self.z_inf_high_sum / self.z_inf_n if self.z_inf_n else float("nan")


def component_audit(traces, dt, *, band=(1.0, 80.0)):
    """Gate I-a readout verdict inputs: the component identity plus where any broadband change lives.

    ``READOUT_CONTAMINATION`` is the failure in which the 1-80 Hz elevation exists in
    ``all_components`` but NOT in ``no_direct_pump`` -- i.e. the slow pump current painted the
    spectrum directly instead of the network activity producing it.
    """
    tr = {k: np.asarray(v, float) for k, v in traces.items()}
    ident_pump = float(np.max(np.abs((tr["all_components"] - tr["no_direct_pump"]) - tr["pump"])))
    ident_sum = float(np.max(np.abs(tr["no_direct_pump"]
                                    - (tr["excitatory"] + tr["inhibitory"] + tr["adaptation"]))))
    out = dict(identity_all_minus_nodp_equals_pump_max_abs_err=ident_pump,
               identity_component_sum_max_abs_err=ident_sum,
               n_steps=int(tr["legacy_abs"].shape[0]), n_sites=int(tr["legacy_abs"].shape[1]))
    for name in ("legacy_abs", "no_direct_pump", "all_components", "pump"):
        out[f"band_power_{name}"] = float(band_power(tr[name], dt, band))
        out[f"sd_{name}"] = float(np.mean(np.std(tr[name], axis=0)))
    return out


def band_power(x, dt, band):
    """Mean across contacts of the in-band power of a per-contact trace sampled every dt (ms)."""
    x = np.asarray(x, float)
    x = x - x.mean(axis=0, keepdims=True)
    fs = 1000.0 / dt
    freqs = np.fft.rfftfreq(x.shape[0], d=1.0 / fs)
    P = np.abs(np.fft.rfft(x, axis=0)) ** 2
    m = (freqs >= band[0]) & (freqs <= band[1])
    return float(P[m].sum() / x.shape[0] ** 2 / max(1, x.shape[1]))


def frozen_load_field(u0, u_high, rho_u):
    """Activity-shaped frozen load field u(rho_u) = u0 + rho_u*(u_high - u0) (spec §T1).

    ``rho_u`` is ONLY a field-construction parameter. The formal phase-diagram coordinate is the
    mean EXCESS PUMP ACTIVATION mean[phi(u_i)-p0_i], never raw u, raw rho_u or a cell count.
    """
    u0, u_high = np.asarray(u0, float), np.asarray(u_high, float)
    return np.maximum(u0 + float(rho_u) * (u_high - u0), 0.0)


def mean_excess_pump_activation(u, p0, h=PRIMARY_H):
    """The formal Gate T abscissa: P = mean_i[phi(u_i) - p0_i]."""
    return float(np.mean(pump_activation(u, h) - np.asarray(p0, float)))


def matched_uniform_field(u_shaped, p0, h=PRIMARY_H):
    """Control 1: a SPATIALLY UNIFORM load whose mean excess pump activation matches the shaped
    field's (spec §T2: uniform/shuffle must match mean[phi(u)-p0], NOT raw u)."""
    target = mean_excess_pump_activation(u_shaped, p0, h) + float(np.mean(p0))
    target = min(max(target, 0.0), 1.0 - 1e-12)
    return np.full(np.shape(u_shaped), (target / (1.0 - target)) ** (1.0 / h))


def value_matched_shuffle_field(u_shaped, rng):
    """Control 2: the same multiset of per-cell loads, randomly re-assigned in space. Matches every
    moment of the load distribution (hence the mean excess activation too) and destroys only the
    spatial arrangement."""
    u = np.asarray(u_shaped, float).copy()
    rng.shuffle(u)
    return u


def branch_slow_flow(rate_E_hz, u_frozen, p0, z_frozen, frac_z_inf_high, *, a_load, tau_N, tau_z,
                     h=PRIMARY_H):
    """Branch-conditioned MEAN-FIELD slow flow G_branch(Z,P) = <dZbar/dt, dPbar/dt> (spec §T3).

    Evaluated ON the fast branch the frozen cell actually settled into, from that branch's own
    per-cell firing rates and received inhibition:

        dP/dt   = mean_i phi'(u_i) * [a_load*r_i - phi(u_i)/tau_N]
        dZ/dt   = (mean_i z_inf,i - mean_i z_i) / tau_z

    ``frac_z_inf_high`` is the branch's time-averaged fraction of E cells whose received inhibition
    sits BELOW the depletion threshold (z_inf=1, i.e. recovering). This is a mean-field flow, not a
    full slow vector field, and is reported as such.
    """
    r = np.asarray(rate_E_hz, float) * 1e-3                    # Hz -> spikes/ms
    u = np.asarray(u_frozen, float)
    phi = pump_activation(u, h)
    dphi = h * u ** (h - 1) / (1.0 + u ** h) ** 2
    dP = float(np.mean(dphi * (a_load * r - phi / tau_N)))
    dZ = float((float(frac_z_inf_high) - float(np.mean(z_frozen))) / tau_z)
    return dict(dP_dt=dP, dZ_dt=dZ,
                P=mean_excess_pump_activation(u, p0, h), Z=float(np.mean(z_frozen)))


def readout_identifiability_note():
    """Capability audit (Gate I-a, spec §I3): what the NON-BLESSED observer can actually obtain from
    the slow protocol without touching the blessed engine, and what it cannot.

    AVAILABLE per E cell, per step, inside MZSlowVars.membrane_terms:
        I_E, I_I (synaptic input currents), I_E_rec (recurrent AMPA component),
        gE / gI / gM (the post-clip leak-relative conductances the membrane actually used),
        the reversal potentials E_E / e_gaba / e_k, the force-match anchor v_match,
        the pump excess Imax*(phi(u)-p0), and the spike mask (in step()).

    NOT AVAILABLE: the membrane potential V. The engine computes V AFTER the slow hook returns and
    never passes it back, so an instantaneous driving force (E_k - V_i) cannot be formed without
    editing the blessed engine.

    CONSEQUENCE: `no_direct_pump` IS constructible as a SIGNED, pump-separable synaptic-current
    proxy by evaluating each conductance against its own reversal at the model's force-match anchor
    v_match -- the same anchor the FCXR membrane uses to define its current<->conductance mapping.
    Every sign comes from the model's reversal potentials (E_E > v_match > e_gaba), never from
    guessing the sign of a magnitude. The cost is that the components are state-independent off
    v_match, exactly like the legacy |I_E|+|I_I| proxy. They are therefore reported as virtual-SEEG
    PROXY components, never as a physical forward-voltage solution, and Gate E remains a
    proxy-level, not a volt-level, statement.
    """
    return dict(
        status="IDENTIFIABLE_AS_PROXY",
        available=["I_E", "I_I", "I_E_rec", "gE", "gI", "gM", "E_E", "e_gaba", "e_k",
                   "v_match", "pump_excess", "spike_mask"],
        unavailable=["V (membrane potential, computed after the slow hook returns)"],
        driving_force_anchor="v_match (force-match reference), NOT the instantaneous V",
        signs_from="model reversal potentials (E_E>v_match>e_gaba); never inferred from magnitudes",
        blessed_engine_modified=False,
        proxy_only=True,
        gate_E_primary_component="no_direct_pump",
    )
