"""Slow-state bins and natural fast-phase selection on a real Z/M+S_G trajectory
(spec rev3.1 §5.1-§5.3, plan Task 6).

Nothing here manufactures a state. Bins are chosen along the ARC LENGTH of the observed slow
trajectory (so an unevenly-paced trajectory is not sampled by wall-clock), and the fast phase inside
a bin is chosen from the local temporal derivative of the core rate -- trough / rising / peak are
naturally occurring microstates, never a reset of membrane, synaptic, refractory or delay state.

Anchor eligibility follows §5.1: returning events, escalation, and at least 4 s of the contained
state, with no runaway truncation. Fewer than three eligible seeds => `insufficient_bounded_anchors`,
never Branch F.
"""
from __future__ import annotations

import numpy as np

SELECTION_VERSION = "zm_anchor_states_v1_2026-07-26"

SLOW_KEYS = ("z_core", "z_surround", "dz_axis", "m_core", "m_surround", "dm_axis", "S_G")
BOUNDED_BINS = ("bounded_early", "bounded_mid", "bounded_late")
BOUNDED_QUANTILES = {"bounded_early": 1.0 / 6.0, "bounded_mid": 0.5, "bounded_late": 5.0 / 6.0}
FAST_PHASES = ("trough", "rising", "peak")

MIN_BOUNDED_MS = 4000.0        # §5.1 "at least 4 s of the contained burst-train state"
ESCALATION_FRAC = 0.5          # supra-threshold level defining entry into the contained regime
ESCALATION_SUSTAIN_MS = 100.0  # ...sustained this long (same 100 ms rule as the carrier onset gate)
PHASE_WINDOW_MS = 200.0        # search half-window for the natural fast phase inside a bin
REST_MARGIN_FRAC = 0.15        # pre-entry bin sits this far back into the quiet window


def _smooth(x, win_bins):
    w = max(1, int(round(win_bins)))
    if w == 1:
        return np.asarray(x, float)
    k = np.ones(w) / w
    return np.convolve(np.asarray(x, float), k, mode="same")


def first_sustained(above, need_bins):
    run = 0
    for i, b in enumerate(np.asarray(above, bool)):
        run = run + 1 if b else 0
        if run >= need_bins:
            return i - need_bins + 1
    return None


def escalation_bin(r_core, bin_ms, smooth_ms=100.0):
    """First bin from which the core rate stays above baseline + 0.5*(peak-baseline) for >=100 ms.

    This is the entry into the CONTAINED regime, not a claim of ictal onset; the spec's slow-state
    bins only need a reproducible, trajectory-derived split between the quiet and the active phase.
    """
    e = _smooth(r_core, smooth_ms / bin_ms)
    n = e.size
    if n < 8:
        return None
    base = float(np.median(e[:max(2, n // 20)]))
    peak = float(e.max())
    if peak - base <= 1e-12:
        return None
    thr = base + ESCALATION_FRAC * (peak - base)
    return first_sustained(e >= thr, max(1, int(round(ESCALATION_SUSTAIN_MS / bin_ms))))


def returning_event_stats(r_core, bin_ms, hi_bin):
    """Duration/peak of the returning interictal events BEFORE escalation (the IED reference the
    carrier lifetime has to beat, spec §6.3)."""
    e = _smooth(r_core[:hi_bin], 100.0 / bin_ms) if hi_bin > 4 else np.asarray(r_core[:hi_bin], float)
    if e.size < 4:
        return dict(n_events=0, median_duration_ms=float("nan"), median_peak_hz=float("nan"))
    # 20th percentile, NOT the median: on a trace where returning events occupy a large share of the
    # window the median sits ON the events and the amplitude collapses to zero.
    base = float(np.percentile(e, 20))
    amp = float(e.max()) - base
    if amp <= 1e-12:
        return dict(n_events=0, median_duration_ms=float("nan"), median_peak_hz=float("nan"))
    on = e >= base + 0.5 * amp
    durs, peaks, start = [], [], None
    for i, b in enumerate(on):
        if b and start is None:
            start = i
        elif not b and start is not None:
            durs.append((i - start) * bin_ms)
            peaks.append(float(e[start:i].max()))
            start = None
    if start is not None:
        durs.append((on.size - start) * bin_ms)
        peaks.append(float(e[start:].max()))
    if not durs:
        return dict(n_events=0, median_duration_ms=float("nan"), median_peak_hz=float("nan"))
    return dict(n_events=len(durs), median_duration_ms=float(np.median(durs)),
                median_peak_hz=float(np.median(peaks)))


def slow_feature_matrix(sc, axis_gradient=None):
    """[z_core, z_surround, dz_axis, m_core, m_surround, dm_axis, S_G] per bin (spec §7.1).

    dz_axis / dm_axis are the core-minus-surround contrasts, i.e. the axial gradient available from
    the recorded core/surround split; a full-field axis projection is added in Task 8 from the
    snapshot fields themselves.
    """
    z_c, z_s = np.asarray(sc["z_core"], float), np.asarray(sc["z_surround"], float)
    m_c, m_s = np.asarray(sc["m_core"], float), np.asarray(sc["m_surround"], float)
    dz = z_c - z_s if axis_gradient is None else np.asarray(axis_gradient[0], float)
    dm = m_c - m_s if axis_gradient is None else np.asarray(axis_gradient[1], float)
    return np.column_stack([z_c, z_s, dz, m_c, m_s, dm, np.asarray(sc["S_G"], float)])


def robust_standardize(Q, ref=None):
    """Median / IQR standardization fitted on the locked anchor trajectory only."""
    if ref is None:
        med = np.median(Q, axis=0)
        iqr = np.subtract(*np.percentile(Q, [75, 25], axis=0))
        scale = np.where(iqr > 1e-12, iqr, np.maximum(np.std(Q, axis=0), 1e-12))
        ref = dict(median=med.tolist(), scale=scale.tolist())
    return (Q - np.asarray(ref["median"])) / np.asarray(ref["scale"]), ref


def arclength_bins(Q_std, lo, hi):
    """Arc-length quantile positions inside [lo, hi) of the standardized slow trajectory."""
    seg = Q_std[lo:hi]
    if seg.shape[0] < 3:
        return {}
    step = np.linalg.norm(np.diff(seg, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(step)])
    total = float(s[-1])
    out = {}
    for name, q in BOUNDED_QUANTILES.items():
        target = q * total
        out[name] = int(lo + int(np.searchsorted(s, target)))
    return out, dict(total_arclength=total, mean_step=float(step.mean()) if step.size else 0.0)


def natural_fast_phase(r_core, bin_ms, center_bin, phase, window_ms=PHASE_WINDOW_MS):
    """Pick a naturally occurring microstate near `center_bin` from local temporal derivatives."""
    r = np.asarray(r_core, float)
    w = max(2, int(round(window_ms / bin_ms)))
    lo, hi = max(0, center_bin - w), min(r.size, center_bin + w + 1)
    seg = r[lo:hi]
    if seg.size < 3:
        return int(center_bin)
    if phase == "peak":
        return int(lo + int(np.argmax(seg)))
    if phase == "trough":
        return int(lo + int(np.argmin(seg)))
    d = np.gradient(seg)
    return int(lo + int(np.argmax(d)))       # steepest rise = the recruiting front


def anchor_eligibility(met, bin_ms, runaway_ms):
    """§5.1 gate. Returns (eligible, info)."""
    r_core = np.asarray(met["r_core"], float)
    n = r_core.size
    esc = escalation_bin(r_core, bin_ms)
    info = dict(n_bins=int(n), escalation_bin=(None if esc is None else int(esc)),
                escalation_ms=(None if esc is None else float(esc * bin_ms)),
                runaway_early_stop_ms=runaway_ms,
                bounded_ms=(0.0 if esc is None else float((n - esc) * bin_ms)))
    info["returning_events"] = returning_event_stats(r_core, bin_ms, esc if esc else max(2, n // 10))
    reasons = []
    if esc is None:
        reasons.append("no escalation into a contained regime")
    if runaway_ms is not None:
        reasons.append(f"runaway truncation at {runaway_ms} ms")
    if info["bounded_ms"] < MIN_BOUNDED_MS:
        reasons.append(f"contained segment {info['bounded_ms']:.0f} ms < {MIN_BOUNDED_MS:.0f} ms")
    if info["returning_events"]["n_events"] < 1:
        reasons.append("no returning events before escalation")
    info["reasons"] = reasons
    info["eligible"] = not reasons
    return info["eligible"], info


def select_states(met, sc, bin_ms, runaway_ms):
    """The full §5.2/§5.3 selection: eligibility, slow bins, natural fast phases, rest window."""
    ok, elig = anchor_eligibility(met, bin_ms, runaway_ms)
    out = dict(version=SELECTION_VERSION, eligibility=elig, states=[])
    if not ok:
        return out
    n = met["n_bins"]
    esc = elig["escalation_bin"]
    Q = slow_feature_matrix(sc)
    Q_std, std_ref = robust_standardize(Q)
    bins, arc = arclength_bins(Q_std, esc, n)
    out["standardization"] = std_ref
    out["arclength"] = arc
    out["rest_window"] = dict(lo_bin=0, hi_bin=int(esc))
    r_core = np.asarray(met["r_core"], float)
    pre = max(0, int(esc * (1.0 - REST_MARGIN_FRAC)) - 1)
    out["states"].append(dict(bin_name="pre_entry", fast_phase="natural", bin_index=int(pre)))
    out["states"].append(dict(bin_name="onset_adjacent", fast_phase="natural", bin_index=int(esc)))
    for name, b in bins.items():
        for ph in FAST_PHASES:
            out["states"].append(dict(bin_name=name, fast_phase=ph,
                                      bin_index=int(natural_fast_phase(r_core, bin_ms, b, ph)),
                                      bin_center=int(b)))
    for st in out["states"]:
        st["t_ms"] = float(st["bin_index"] * bin_ms)
        st["slow_coord"] = {k: float(v) for k, v in zip(SLOW_KEYS, Q[min(st["bin_index"], n - 1)])}
    return out
