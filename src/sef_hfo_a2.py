"""M3A-A2 Abbott-LG helpers: builder, k_use derivation, mutual-exclusion guard, rho coordinate,
per-event R-class (canonical), bout detection. Isolated from the M3B-edited scripts: R-class reuses
event_props/classify_event from src.sef_hfo_mu_basin and faithfully reimplements the spatial-extent
helpers (run_m3_kick_calibration.py:241/375) rather than importing them.
See docs/archive/topic4/sef_hfo/m3a_a2_abbott_lg_dynamic_slowvars_spec_2026-06-25.md.
"""
from __future__ import annotations
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "snn_engine"))
from slow_vars import RegionalResource, RegionalResourceConfig  # noqa: E402
sys.path.insert(0, os.path.dirname(__file__))
from sef_hfo_mu_basin import classify_event, DEFAULT_CAPS, event_props  # noqa: E402

DEFAULT_FAR = 6.0   # far-radius (mm) for far_ea, matches A1 spatial-extent default


# --------------------------------------------------------------------------- builder + guards
def k_use_from_target(q_target, a_bar, tau_rec):
    """Invert the ODE fixed point q* = 1/(1 + k_use*a_bar*tau_rec) for k_use. a_bar = baseline core E
    firing fraction (Task-0). Sets the RESTING operating point; bout activity overshoots beyond q*."""
    if a_bar <= 0 or tau_rec <= 0:
        raise ValueError("a_bar and tau_rec must be > 0 to derive k_use")
    return (1.0 / float(q_target) - 1.0) / (float(a_bar) * float(tau_rec))


def assert_a2_exclusive(slow_var, shunt_gaba, feedback_gain):
    """A2 RegionalResource is mutually exclusive with the frozen slow-var, GABA-shunt, and A1c paths."""
    if str(slow_var) != "none":
        raise ValueError("A2 (a2-mode != off) requires --slow-var none")
    if shunt_gaba:
        raise ValueError("A2 is incompatible with --shunt-gaba")
    if float(feedback_gain) > 0.0:
        raise ValueError("A2 is incompatible with --feedback-gain > 0 (A1c)")


def build_regional_resource(N, V_th0, core_mask, NE, *, mode, q_target=None, k_use=None,
                            tau_rec=5000.0, tau_a=100.0, q_min=0.25, a_bar=None,
                            frozen=False, frozen_q_core=1.0, frozen_q_global=1.0, foci_masks=None,
                            gk_max=0.0, tau_k=5000.0):
    """core_mask: length-N bool (E core True). per_core: foci_masks = [left_mask, right_mask].
    gk_max>0 adds the optional sAHP recovery term (§8.4 next-ingredient probe)."""
    if not frozen and k_use is None:
        if q_target is None or a_bar is None:
            raise ValueError("dynamic build needs k_use OR (q_target AND a_bar)")
        k_use = k_use_from_target(q_target, a_bar, tau_rec)
    cfg = RegionalResourceConfig(mode=mode, k_use=float(k_use or 0.0), tau_rec=tau_rec, tau_a=tau_a,
                                 q_min=q_min, frozen=frozen,
                                 q_core_init=float(frozen_q_core), q_global_init=float(frozen_q_global),
                                 gk_max=float(gk_max), tau_k=float(tau_k))
    left = foci_masks[0] if (mode == "per_core" and foci_masks is not None) else None
    right = foci_masks[1] if (mode == "per_core" and foci_masks is not None) else None
    return RegionalResource(N, V_th0, core_mask, cfg, NE=NE, left_core_E=left, right_core_E=right)


# --------------------------------------------------------------------------- rho coordinate
def compute_rho(q_core, q_global, lgr_static):
    """Dynamic image of A1b's local_global_ratio along the inhibition axis (spec §2.5)."""
    return float(lgr_static) / (float(q_core) * float(q_global))


# --------------------------------------------------------------------------- canonical R-class
def _bin_spike_counts(spk, bin_of_cell, n_bins, lo_step, hi_step):
    """Total E spikes per bin in [lo_step, hi_step). EXACT mirror of
    run_m3_kick_calibration._bin_spike_counts_in_window:241 (NOT imported — M3B edits that script)."""
    per_cell = np.asarray(spk[lo_step:hi_step], bool).sum(axis=0).astype(float)
    return np.bincount(np.asarray(bin_of_cell, int), weights=per_cell, minlength=n_bins)


def _spatial_extent(net_bins, bin_centers, src_bin, far_radius):
    """(n_activated, r95_mm, far_field_frac) over NON-source bins. EXACT mirror of
    run_m3_kick_calibration._spatial_extent:375-432; source bin excluded from all three. bin_centers 2D."""
    bc = np.asarray(bin_centers, float)
    radii = np.linalg.norm(bc - bc[src_bin], axis=1)
    non_source = np.ones(len(net_bins), bool); non_source[src_bin] = False
    nonsrc = net_bins[non_source]
    floor = max(2.0, 0.05 * float(nonsrc.max() if nonsrc.size else 0.0))
    activated = non_source & (net_bins > floor)
    n_act = int(activated.sum())
    if n_act > 0:
        order = np.argsort(radii[activated]); r_s = radii[activated][order]; w_s = net_bins[activated][order]
        cw = np.cumsum(w_s)
        r95 = float(np.interp(0.95 * cw[-1], cw, r_s)) if cw[-1] > 0 else float(np.percentile(radii[activated], 95))
    else:
        r95 = 0.0
    total_ns = float(net_bins[non_source].sum())
    far = float(net_bins[non_source & (radii > far_radius)].sum()) / total_ns if total_ns > 0 else 0.0
    return n_act, r95, far


def event_rclass(af, spk, bin_of_cell, n_bins, bin_centers, bin_w, t_on, t_off, dt,
                 foci=None, src_window_ms=10.0, far_radius=DEFAULT_FAR):
    """Canonical per-event R-class. Source bin = the early-activity peak bin (first src_window_ms after
    t_on; tie -> nearest focus), NOT the geometric center. peak_active via event_props (FRACTION); front
    via a real 50ms window; spatial extent = exact A1 _spatial_extent.
    Returns (R_class, metrics_dict, n_activated, src_bin)."""
    s, e = int(t_on / bin_w), int(t_off / bin_w)
    ep = event_props(af, (s, e), bin_w, len(af))                      # peak_active = max FRACTION; returned
    lo, hi = int(round(t_on / dt)), int(round(t_off / dt))
    early_hi = max(int(round(min(t_off, t_on + src_window_ms) / dt)), lo + 1)
    early = _bin_spike_counts(spk, bin_of_cell, n_bins, lo, early_hi)
    cand = np.flatnonzero(early == early.max())
    if foci is not None and len(cand) > 1:
        bc = np.asarray(bin_centers, float); fc = np.asarray(foci, float)
        src_bin = int(cand[int(np.argmin([min(np.linalg.norm(bc[b] - f) for f in fc) for b in cand]))])
    else:
        src_bin = int(cand[0])
    bins = _bin_spike_counts(spk, bin_of_cell, n_bins, lo, hi)
    n_act, r95, far = _spatial_extent(bins, bin_centers, src_bin, far_radius)
    tail_lo = max(t_on, t_off - 50.0)
    tlo, thi = int(round(tail_lo / dt)), int(round(t_off / dt))
    tail_bins = _bin_spike_counts(spk, bin_of_cell, n_bins, tlo, thi)
    front_score = 1.0 - int(np.sum(tail_bins > 0)) / n_bins
    m = {"event_detected": True, "returned": bool(ep["returned"]), "runaway": bool(ep["sustained"]),
         "r95_ea": float(r95), "far_ea": float(far), "active_peak": float(ep["peak_active"]),
         "sustained_front_score": float(front_score)}
    return classify_event(m, DEFAULT_CAPS), m, n_act, src_bin


def detect_bouts(rho_bin, B):
    """Maximal contiguous index ranges where rho_bin >= B (the seizure-band entry)."""
    above = np.asarray(rho_bin, float) >= float(B)
    bouts = []; i = 0; n = len(above)
    while i < n:
        if above[i]:
            j = i
            while j + 1 < n and above[j + 1]:
                j += 1
            bouts.append((i, j)); i = j + 1
        else:
            i += 1
    return bouts


# --- M3A-A2 dynamic recorder: per-event landmark sampler --------------------
LANDMARK_PRE_MS = 200.0
LANDMARK_POST_OFFSETS_MS = (("post_50ms", 50.0), ("post_200ms", 200.0), ("post_1s", 1000.0))


def sample_event_landmarks(traces, dt_ms, events, *, pre_ms=LANDMARK_PRE_MS,
                           post_offsets_ms=LANDMARK_POST_OFFSETS_MS):
    """Per-event landmark samples (pre/onset/peak/end/post_*) from per-step traces.

    traces: {name: per-step sequence}, all the same length T (e.g. trace_core,
            trace_global, trace_gk from a RegionalResource run).
    dt_ms:  ms per simulation step.
    events: [{event_id, onset_ms, peak_ms, end_ms}, ...].

    Returns rows [{event_id, event_stage, time_ms, <name>: value, ...}], one per
    (event, stage). Each landmark time is clamped into [0, (T-1)*dt] and the value
    is read at that step, so samples align with simulation time; a post landmark
    past the trace end clamps to the last step. Feeds the M3B phase exporter.
    """
    if not traces:
        raise ValueError("traces must be a non-empty mapping name->sequence")
    lengths = {len(v) for v in traces.values()}
    if len(lengths) != 1:
        raise ValueError(f"all traces must share one length, got {sorted(lengths)}")
    T = lengths.pop()
    if T == 0 or dt_ms <= 0:
        raise ValueError("traces must be non-empty and dt_ms must be > 0")

    def _step(t_ms):
        return min(T - 1, max(0, int(round(t_ms / dt_ms))))

    rows = []
    for ev in events:
        eid = ev["event_id"]
        onset, peak, end = float(ev["onset_ms"]), float(ev["peak_ms"]), float(ev["end_ms"])
        stage_times = [("pre", onset - pre_ms), ("onset", onset),
                       ("peak", peak), ("end", end)]
        stage_times += [(name, end + off) for name, off in post_offsets_ms]
        for stage, t_ms in stage_times:
            step = _step(t_ms)
            row = {"event_id": eid, "event_stage": stage, "time_ms": step * dt_ms}
            for name, seq in traces.items():
                row[name] = seq[step]
            rows.append(row)
    return rows


# --- science-decided helpers (user 2026-06-27): A absolute tail, C real peak ----
def tail_to_baseline_absolute(rate, dt_ms, t_off_ms, baseline_window_ms=(5.0, 50.0),
                              tail_len_ms=200.0, return_threshold=1.5):
    """ABSOLUTE return-to-baseline (user decision A).

    ratio = mean(rate in the event tail) / mean(rate in the FIXED quiet baseline
    window). The denominator is the recording-baseline window (BASELINE_MS=(5,50) ms),
    NOT the event's own peak -- so a tall event cannot self-license "returned".
    returned := ratio <= return_threshold (the existing analyze_a2_pilot 1.5 gate).
    Returns (ratio, returned).
    """
    rate = np.asarray(rate, float)
    b0 = int(round(baseline_window_ms[0] / dt_ms))
    b1 = int(round(baseline_window_ms[1] / dt_ms))
    baseline = float(rate[b0:b1].mean()) if b1 > b0 else float(rate[0])
    t0 = int(round(t_off_ms / dt_ms))
    t1 = min(int(round((t_off_ms + tail_len_ms) / dt_ms)), len(rate))
    tail = float(rate[t0:t1].mean()) if t1 > t0 else float(rate[-1])
    ratio = tail / max(baseline, 1e-9)
    return ratio, bool(ratio <= return_threshold)


def event_peak_ms(af, bin_w, t_on_ms, t_off_ms):
    """Real activity-fraction peak time (ms) within [t_on, t_off] (user decision C).

    Returns the bin time of the maximum activity fraction inside the event window --
    the canonical 'peak' landmark, replacing the window-midpoint placeholder.
    """
    af = np.asarray(af, float)
    i0 = max(0, int(round(t_on_ms / bin_w)))
    i1 = min(len(af), max(i0 + 1, int(round(t_off_ms / bin_w))))
    rel = int(np.argmax(af[i0:i1]))
    return float((i0 + rel) * bin_w)
