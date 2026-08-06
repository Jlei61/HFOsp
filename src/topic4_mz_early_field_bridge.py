"""MZ early-field bridge — pure readout functions (no SNN dynamics, no I/O).

Design contract: docs/superpowers/specs/2026-07-19-topic4-mz-early-field-bridge-design.md.
Every function maps to a spec clause (referenced in comments). Engine calls, simulation
scheduling, provenance and file writes live in scripts/run_topic4_mz_early_field_bridge.py;
this module operates only on arrays so the 10 required contract tests run on synthetic fixtures.

Reuse (not reinvent, design §0/§5): src.early_recruitment_readout supplies
``early_energy_field`` (§8.2 fail-closed + §8.3 mean-sq excess), ``compare_arrival_to_energy``
(§9 signed Spearman/cosine/top-k with earliness = -rho), and the permutation primitives
(``_permutation_groups`` / ``_permutation_indices`` / ``_exact_permutation_indices`` /
``_spearman``). The maxAB null, the source-grid toroidal-shift null, the quartile contrast,
the 5*MAD readable rule and the t_recruit-contains-t120 component logic are NOT in that
library (they are bridge-specific) and are written here.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import factorial

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

from src.early_recruitment_readout import (
    _exact_permutation_indices,
    _permutation_groups,
    _permutation_indices,
    _spearman,
    EnergyField,
    compare_arrival_to_energy,
    early_energy_field,
)


# ======================================================================== §6 fixed-bar detector
@dataclass(frozen=True)
class EventBar:
    floor: float
    bar: float
    af_max: float
    baseline_ms: tuple[float, float]
    cal_frac: float


def compute_event_bar(af, bin_w, baseline_ms=(5.0, 50.0), cal_frac=0.5) -> EventBar:
    """Freeze the event threshold from ONE active-fraction series (design §6).

    floor = P95(af in the baseline interval); bar = floor + cal_frac*(max(af) - floor).
    The runner computes this ONCE on slow-off and passes ``bar`` to detect_events for BOTH
    slow-off and native. It must NEVER be recomputed from the native target's own af.max()
    (the exact bug in run_topic4_mz_slowvars._events_from_res). Test 1 guards this.
    """
    af = np.asarray(af, float)
    nb0, nb1 = int(baseline_ms[0] / bin_w), int(baseline_ms[1] / bin_w)
    floor = float(np.percentile(af[nb0:nb1], 95)) if nb1 > nb0 and af[nb0:nb1].size else float(af.min())
    af_max = float(af.max())
    bar = floor + float(cal_frac) * (af_max - floor)
    return EventBar(floor, bar, af_max, (float(baseline_ms[0]), float(baseline_ms[1])), float(cal_frac))


# ======================================================================== §7.2 burst envelope + quiet baseline
def burst_envelope(lfp_trace, times_ms, band=(30.0, 80.0), order=4) -> np.ndarray:
    """30-80 Hz zero-phase band-pass analytic envelope of the virtual LFP, shape (nT, nC).

    Mirrors the accepted M3 readout helper (scripts/run_topic4_m3_runaway_readout._burst_readout):
    butter(order, band, fs=1000/median(dt_ms)) -> sosfiltfilt -> |hilbert|. fs is derived from the
    time axis so it is correct at the LFPRecorder's 10 kHz (dt=0.1 ms) sampling.
    """
    lfp = np.asarray(lfp_trace, float)
    t = np.asarray(times_ms, float)
    if lfp.ndim != 2 or lfp.shape[0] != t.size:
        raise ValueError(f"lfp_trace must be (nT, nC) aligned to times_ms; got {lfp.shape} vs {t.shape}")
    dt_ms = float(np.median(np.diff(t)))
    fs_hz = 1000.0 / dt_ms
    sos = butter(order, band, btype="bandpass", fs=fs_hz, output="sos")
    burst = sosfiltfilt(sos, lfp, axis=0)
    return np.abs(hilbert(burst, axis=0))


def quiet_mask(times_ms, events, guard_ms=0.0) -> np.ndarray:
    """Boolean over LFP samples OUTSIDE every detected event (the slow-off quiet set, design §7.2/§8.3).

    ``events`` are detect_events dicts with t_on/t_off (ms). ``guard_ms`` pads each event so
    edge samples do not leak into the quiet baseline.
    """
    t = np.asarray(times_ms, float)
    m = np.ones(t.size, bool)
    for e in events:
        m &= ~((t >= e["t_on"] - guard_ms) & (t <= e["t_off"] + guard_ms))
    return m


def quiet_baseline(envelope, quiet_samples) -> tuple[np.ndarray, np.ndarray]:
    """Per-contact quiet median and MAD of the burst envelope over slow-off quiet samples.

    MAD is the median absolute deviation scaled to a std estimate (*1.4826), used by the
    5*MAD readable rule (design §7.2). Returns (median (nC,), mad (nC,))."""
    env = np.asarray(envelope, float)
    q = np.asarray(quiet_samples, bool)
    if q.sum() < 2:
        raise ValueError("fewer than two slow-off quiet samples for the envelope baseline")
    sub = env[q]
    med = np.median(sub, axis=0)
    mad = 1.4826 * np.median(np.abs(sub - med[None, :]), axis=0)
    return med, mad


# ======================================================================== §7.2 per-event contact timing field
@dataclass(frozen=True)
class ContactTiming:
    latency_ms: np.ndarray      # (nC,) peak latency rel. to event onset; NaN where not readable
    readable: np.ndarray        # (nC,) bool
    rank: np.ndarray            # (nC,) ordinal rank over readable contacts (1=earliest); NaN elsewhere
    axis_spearman: float        # Spearman(contact axis coord, latency rank) over readable
    direction: str              # A_to_B | B_to_A | unresolved
    n_readable: int
    eligible: bool              # n_readable >= min_readable_contacts


def _ordinal_rank(values) -> np.ndarray:
    """Deterministic 1..N ordinal rank over finite entries; NaN elsewhere (ties by first-seen)."""
    v = np.asarray(values, float)
    out = np.full(v.shape, np.nan, float)
    idx = np.flatnonzero(np.isfinite(v))
    order = idx[np.argsort(v[idx], kind="mergesort")]
    out[order] = np.arange(1, order.size + 1, dtype=float)
    return out


def event_contact_timing(envelope, times_ms, event, *, next_event_t_on, record_end_ms,
                         quiet_med, quiet_mad, contact_axis,
                         event_offset_ms=40.0, mad_k=5.0, rel_peak=0.10,
                         min_readable=6, direction_abs=0.30,
                         axis_src_to_snk_sign=+1) -> ContactTiming:
    """Per-contact 30-80 Hz burst-envelope PEAK latency for one fixed-bar returning event (design §7.2).

    Window = [t_on, min(t_off + event_offset_ms, next_event_t_on, record_end_ms)] (capped before the
    next event and the record end). Readable contact: (event peak envelope) exceeds its slow-off quiet
    median by mad_k*MAD AND its excess peak is >= rel_peak of the largest contact excess peak in this
    event. Latency = argmax(envelope) in the window minus t_on; missing contacts stay NaN (never imputed,
    test 3). Direction from Spearman(contact axis coord, latency rank).

    ``axis_src_to_snk_sign`` records that increasing contact_axis == source->sink (from src_xy/snk_xy/
    axis_unit); the sign->endpoint mapping goes to metadata and is NOT read off a plot (design §7.2).
    """
    env = np.asarray(envelope, float)
    t = np.asarray(times_ms, float)
    med = np.asarray(quiet_med, float)
    mad = np.asarray(quiet_mad, float)
    axis = np.asarray(contact_axis, float)
    nC = env.shape[1]
    w1 = min(float(event["t_off"]) + event_offset_ms,
             float(next_event_t_on) if next_event_t_on is not None else np.inf,
             float(record_end_ms))
    win = np.flatnonzero((t >= float(event["t_on"])) & (t <= w1))
    latency = np.full(nC, np.nan, float)
    readable = np.zeros(nC, bool)
    if win.size >= 2:
        seg = env[win]                                   # (nWin, nC)
        peak_idx = np.argmax(seg, axis=0)
        peak_env = seg[peak_idx, np.arange(nC)]
        excess = np.maximum(peak_env - med, 0.0)         # excess over slow-off quiet median
        max_excess = float(excess.max()) if excess.size else 0.0
        # clause: readable iff peak > quiet_median + mad_k*MAD AND excess >= rel_peak*max_excess_in_event
        thr = med + mad_k * np.where(mad > 0, mad, np.inf)   # MAD==0 -> unreadable (can't clear 5*MAD)
        readable = (peak_env > thr) & (excess >= rel_peak * max_excess) & (max_excess > 0.0)
        latency[readable] = t[win[peak_idx[readable]]] - float(event["t_on"])
    rank = _ordinal_rank(latency)
    n_readable = int(readable.sum())
    # axis-latency Spearman over readable contacts (rank vs axis coord)
    if n_readable >= 2 and np.ptp(axis[readable]) > 0:
        rho = float(_spearman(axis[readable], rank[readable]))
    else:
        rho = float("nan")
    rho_signed = rho * axis_src_to_snk_sign if np.isfinite(rho) else rho
    if np.isfinite(rho_signed) and rho_signed >= direction_abs:
        direction = "A_to_B"
    elif np.isfinite(rho_signed) and rho_signed <= -direction_abs:
        direction = "B_to_A"
    else:
        direction = "unresolved"
    return ContactTiming(latency, readable, rank, rho_signed, direction,
                         n_readable, n_readable >= int(min_readable))


# ======================================================================== §7.3 train / held-out templates
@dataclass(frozen=True)
class DirectionTemplate:
    direction: str
    full_template: np.ndarray       # (nLoc,) median ordinal rank over ALL events (used for §9 association)
    train_template: np.ndarray      # (nLoc,) median ordinal rank over training events (held-out anchor)
    heldout_scores: list            # per held-out event: Spearman(train_template, event rank) on shared support
    n_train: int
    n_heldout: int
    n_shared_contacts: int          # locations finite in the training template AND in >=1 held-out event
    template_variance_ok: bool
    eligible: bool


def build_template_from_ranks(ranks_chrono, *, min_train=3, min_heldout=2, min_shared=6) -> dict:
    """Chronological odd/even held-out template from per-location rank vectors (design §7.3).

    ``ranks_chrono`` is the chronological list of (nLoc,) ordinal-rank arrays (NaN where a location
    was not readable in that event). Training = even indices, held-out = odd indices (no leakage,
    test 4). full_template = per-location median over ALL events (the §9 association template);
    train_template = median over training events; held-out score = Spearman(train_template, event
    rank) on matched finite support. Generic over contacts and source-grid bins.
    """
    chrono = [np.asarray(r, float) for r in ranks_chrono]
    nLoc = chrono[0].size if chrono else 0
    if not chrono:
        return {"full_template": np.full(nLoc, np.nan), "train_template": np.full(nLoc, np.nan),
                "heldout_scores": [], "n_train": 0, "n_heldout": 0, "n_shared": 0,
                "variance_ok": False, "eligible": False}
    all_stack = np.vstack(chrono)
    with np.errstate(invalid="ignore"):
        full_template = np.nanmedian(all_stack, axis=0)
    full_template[np.all(~np.isfinite(all_stack), axis=0)] = np.nan
    train = chrono[0::2]
    held = chrono[1::2]
    train_stack = np.vstack(train)
    with np.errstate(invalid="ignore"):
        train_template = np.nanmedian(train_stack, axis=0)
    train_template[np.all(~np.isfinite(train_stack), axis=0)] = np.nan
    tmpl_support = np.isfinite(train_template)
    scores = []
    shared_any = np.zeros(nLoc, bool)
    for r in held:
        shared = tmpl_support & np.isfinite(r)
        shared_any |= shared
        if int(shared.sum()) >= 2 and np.ptp(train_template[shared]) > 0 and np.ptp(r[shared]) > 0:
            scores.append(float(_spearman(train_template[shared], r[shared])))
        else:
            scores.append(float("nan"))
    n_shared = int((tmpl_support & shared_any).sum())
    var_ok = bool(np.isfinite(train_template).sum() >= 2 and np.ptp(train_template[tmpl_support]) > 0)
    eligible = (len(train) >= int(min_train) and len(held) >= int(min_heldout)
                and n_shared >= int(min_shared) and var_ok)
    return {"full_template": full_template, "train_template": train_template,
            "heldout_scores": scores, "n_train": len(train), "n_heldout": len(held),
            "n_shared": n_shared, "variance_ok": var_ok, "eligible": eligible}


def build_direction_template(timings, direction, *, min_train=3, min_heldout=2,
                             min_shared=6) -> DirectionTemplate:
    """Contact-space wrapper over build_template_from_ranks for one direction (design §7.3)."""
    d = build_template_from_ranks([c.rank for c in timings], min_train=min_train,
                                  min_heldout=min_heldout, min_shared=min_shared)
    return DirectionTemplate(direction, d["full_template"], d["train_template"], d["heldout_scores"],
                             d["n_train"], d["n_heldout"], d["n_shared"], d["variance_ok"], d["eligible"])


# ======================================================================== §8.1 onset markers
def smooth_rate(rate, dt, win_ms=20.0) -> np.ndarray:
    """20 ms box-car of the E rate (mirrors run_m4_dynamic_qi._smooth exactly)."""
    n = max(1, int(round(win_ms / dt)))
    return np.convolve(np.asarray(rate, float), np.ones(n) / n, mode="same")


def compute_t_recruit(r20_native, r20_slowoff, dt, t120, *, theta_pct=99.9, gap_ms=5.0) -> dict:
    """Baseline-relative recruitment onset that must contain t120 (design §8.1, test 8).

    theta_recruit = P(theta_pct) of the 20 ms-smoothed slow-off E rate. t_recruit = start of the
    contiguous (<=gap_ms subthreshold gap) native supra-theta component that CONTAINS t120. If t120
    is None or no supra-theta component contains it -> onset_unresolved (no early-field claim).
    t120 stays the reproducibility anchor; t_recruit is the field-locking time.
    """
    rn = np.asarray(r20_native, float)
    theta = float(np.percentile(np.asarray(r20_slowoff, float), theta_pct))
    if t120 is None:
        return {"status": "onset_unresolved", "reason": "no_t120",
                "t120_ms": None, "theta_recruit": theta, "t_recruit_ms": None, "onset_diff_ms": None}
    above = rn >= theta
    gap = max(0, int(round(gap_ms / dt)))
    i120 = int(round(float(t120) / dt))
    if i120 >= rn.size or not above[i120]:
        # t120 sample itself not supra-theta: bridge one small gap around it before giving up
        lo = max(0, i120 - gap)
        hi = min(rn.size, i120 + gap + 1)
        if not above[lo:hi].any():
            return {"status": "onset_unresolved", "reason": "t120_not_in_supra_theta_component",
                    "t120_ms": float(t120), "theta_recruit": theta, "t_recruit_ms": None,
                    "onset_diff_ms": None}
        i120 = lo + int(np.flatnonzero(above[lo:hi])[0])
    # walk left from i120, allowing subthreshold runs up to `gap` samples
    start = i120
    run = 0
    i = i120 - 1
    while i >= 0:
        if above[i]:
            start = i
            run = 0
        else:
            run += 1
            if run > gap:
                break
        i -= 1
    t_recruit = round(float(start * dt), 1)
    return {"status": "eligible", "t120_ms": float(t120), "theta_recruit": theta,
            "t_recruit_ms": t_recruit, "onset_diff_ms": round(float(t120) - t_recruit, 1)}


# ======================================================================== §8.3 contact energy field
def contact_energy_field(envelope, times_ms, quiet_med, t_recruit_ms, window_ms, *, record_end_ms):
    """Mean squared positive 30-80 Hz envelope excess over slow-off quiet, in a window relative to
    t_recruit (design §8.3 + §8.2 fail-closed). Baseline = slow-off quiet median (never native, §8.3).

    Float-robust: the window bounds t_recruit+offset are snapped to integer step indices and then to the
    exact grid samples t[s]/t[e] before calling early_energy_field, so its inclusive coverage check
    (t[idx[-1]] < w1) never spuriously fails on the float representation of a value that IS on the grid
    (e.g. 9128.3 vs 91283*0.1 = 9128.30000000001). A window running off the recorded trace fails closed.
    """
    env = np.asarray(envelope, float)
    t = np.asarray(times_ms, float)
    med = np.asarray(quiet_med, float)
    excess = np.maximum(env - med[None, :], 0.0)
    dt = float(np.median(np.diff(t)))
    w0 = float(t_recruit_ms) + float(window_ms[0])
    w1 = float(t_recruit_ms) + float(window_ms[1])
    s = int(round(w0 / dt))
    e = int(round(w1 / dt))
    if s < 0 or e <= s or e >= t.size or t[e] > float(record_end_ms):     # off recorded trace -> fail closed
        return EnergyField(np.full(env.shape[1], np.nan), (w0, w1), "ineligible_incomplete_window", 0, False)
    return early_energy_field(excess, t, (float(t[s]), float(t[e])),
                              require_complete_presaturation_window=True)


# ======================================================================== §7.3/§8.4 source-grid fields
def source_bins(posE, L, n=24) -> tuple[np.ndarray, np.ndarray]:
    """Flat 24x24 bin index per E neuron + per-bin neuron count (mirrors run_m4_dynamic_qi._spatial_movie).

    ix = clip(floor(posE[:,0]/L*n), 0, n-1); iy likewise; cell = iy*n + ix. Returns (cell (NE,), counts (n*n,))."""
    pos = np.asarray(posE, float)
    ix = np.clip((pos[:, 0] / L * n).astype(int), 0, n - 1)
    iy = np.clip((pos[:, 1] / L * n).astype(int), 0, n - 1)
    cell = iy * n + ix
    counts = np.bincount(cell, minlength=n * n).astype(float)
    return cell, counts


def source_timing_field(E_spk_bool, cell, event, dt, n=24, *, min_active=5) -> np.ndarray:
    """Per-bin first-spike latency within a returning event, on the fixed 24x24 grid (design §7.3).

    For each E neuron active in [t_on, t_off], take its first-spike latency (ms rel. t_on); per bin
    aggregate by median over active neurons; require >=min_active active neurons per occupied bin else
    the bin is NaN. Returns (n*n,) latency field (source-space projection-control layer)."""
    spk = np.asarray(E_spk_bool)
    s, e = int(round(event["t_on"] / dt)), int(round(event["t_off"] / dt)) + 1
    seg = spk[s:min(e, spk.shape[0])]
    if seg.shape[0] == 0:
        return np.full(n * n, np.nan)
    active = seg.any(axis=0)
    first = np.full(spk.shape[1], -1, int)
    aidx = np.flatnonzero(active)
    first[aidx] = np.argmax(seg[:, aidx], axis=0)          # first True step within window
    lat = np.full(n * n, np.nan)
    for b in range(n * n):
        members = np.flatnonzero((cell == b) & active)
        if members.size >= int(min_active):
            lat[b] = float(np.median(first[members])) * dt
    return lat


def _binned_rate_ts(spk_slice, cell, counts, dt, time_bin_ms, n) -> np.ndarray:
    """Per-bin E firing-rate time series (Hz) over sub-time-bins of a spike-raster slice. (n_tbins, n*n)."""
    spk = np.asarray(spk_slice)
    bs = max(1, int(round(time_bin_ms / dt)))
    nT = spk.shape[0]
    dur_s = bs * dt / 1000.0
    safe_counts = np.where(counts > 0, counts, 1.0)
    out = []
    for b0 in range(0, nT - bs + 1, bs):
        active_counts = np.bincount(cell[spk[b0:b0 + bs].any(axis=0)], minlength=n * n).astype(float)
        out.append(active_counts / safe_counts / dur_s)
    return np.asarray(out, float) if out else np.zeros((0, n * n))


def source_energy_field(E_spk_bool, cell, counts, t_recruit_ms, window_ms, dt, n=24, *,
                        quiet_ref, time_bin_ms=10.0, record_end_ms) -> dict:
    """Per-bin early-window activation energy on the 24x24 grid (design §8.4).

    Per-bin E firing rate time series in the window minus the slow-off quiet per-bin mean rate,
    excess clamped at zero, squared and averaged over the window's sub-time-bins. Fails closed
    (status 'ineligible_incomplete_window') if the window runs off the recorded trace."""
    spk = np.asarray(E_spk_bool)
    w0 = float(t_recruit_ms) + float(window_ms[0])
    w1 = float(t_recruit_ms) + float(window_ms[1])
    if w1 > float(record_end_ms):
        return {"status": "ineligible_incomplete_window", "energy": np.full(n * n, np.nan),
                "window_ms": [w0, w1], "n_timebins": 0}
    s, e = int(round(w0 / dt)), int(round(w1 / dt))
    seg = spk[s:min(e, spk.shape[0])]
    rate_ts = _binned_rate_ts(seg, cell, counts, dt, time_bin_ms, n)      # (nTbins, n*n)
    if rate_ts.shape[0] < 1:
        return {"status": "ineligible_incomplete_window", "energy": np.full(n * n, np.nan),
                "window_ms": [w0, w1], "n_timebins": 0}
    excess = np.maximum(rate_ts - np.asarray(quiet_ref, float)[None, :], 0.0)
    energy = np.mean(excess ** 2, axis=0)
    energy[counts <= 0] = np.nan                                          # never-occupied bins carry no field
    return {"status": "eligible", "energy": energy, "window_ms": [w0, w1],
            "n_timebins": int(rate_ts.shape[0])}


def source_quiet_ref(E_spk_bool, cell, counts, quiet_samples, dt, n=24, time_bin_ms=10.0) -> np.ndarray:
    """Per-bin mean quiet (inter-event, slow-off) firing rate (Hz), the §8.4 subtraction reference."""
    spk = np.asarray(E_spk_bool)[np.asarray(quiet_samples, bool)]
    rate_ts = _binned_rate_ts(spk, cell, counts, dt, time_bin_ms, n)
    return rate_ts.mean(axis=0) if rate_ts.shape[0] else np.zeros(n * n)


# ======================================================================== §8.5 field diagnostics
def field_diagnostics(field, slowoff_p95=None, eps=1e-12) -> dict:
    """Support, dynamic range, recruited count, degeneracy (design §8.5)."""
    f = np.asarray(field, float)
    valid = np.isfinite(f)
    fv = f[valid]
    n = int(valid.sum())
    if n == 0:
        return {"support": 0, "std": None, "dynamic_range": None, "recruited": 0,
                "degenerate": True, "status": "empty_field"}
    p90, p10, med = np.percentile(fv, 90), np.percentile(fv, 10), np.median(fv)
    dyn = float((p90 - p10) / (abs(med) + eps))
    std = float(np.std(fv))
    recruited = int(np.sum(fv > slowoff_p95)) if slowoff_p95 is not None else None
    degenerate = bool(std <= eps or np.ptp(fv) <= eps)
    return {"support": n, "std": std, "dynamic_range": dyn, "recruited": recruited,
            "degenerate": degenerate, "status": "degenerate_field" if degenerate else "eligible"}


# ======================================================================== §9 association + maxAB
def quartile_contrast(template_rank, energy, support=None, eps=1e-12) -> float:
    """Earliest-quartile minus latest-quartile mean energy, normalized by the field IQR (design §9).

    Positive => the earliest-ranked quartile of contacts is hotter than the latest quartile."""
    r = np.asarray(template_rank, float)
    y = np.asarray(energy, float)
    sup = np.ones(r.shape, bool) if support is None else np.asarray(support, bool)
    v = sup & np.isfinite(r) & np.isfinite(y)
    if int(v.sum()) < 4:
        return float("nan")
    rr, yy = r[v], y[v]
    order = np.argsort(rr, kind="mergesort")
    k = max(1, int(round(0.25 * order.size)))
    early = yy[order[:k]].mean()
    late = yy[order[-k:]].mean()
    iqr = np.percentile(yy, 75) - np.percentile(yy, 25)
    return float((early - late) / (iqr + eps))


def associate(template_rank, energy, *, support=None, top_k=3, min_points=3) -> dict:
    """Signed Spearman / cosine / top-k (reuse compare_arrival_to_energy) + quartile contrast (design §9).

    template_rank is passed as 'arrival' so earliness_energy_spearman = corr(-template_rank, energy)."""
    out = compare_arrival_to_energy(template_rank, energy, support_mask=support,
                                    min_points=min_points, top_k=top_k)
    out = {k: v for k, v in out.items() if k != "valid_mask"}
    out["quartile_contrast"] = quartile_contrast(template_rank, energy, support)
    return out


def _earliness(rank, energy, support, min_points):
    v = np.asarray(support, bool) & np.isfinite(rank) & np.isfinite(energy)
    if int(v.sum()) < min_points:
        return float("nan")
    return -float(_spearman(rank[v], energy[v]))         # earliness = -corr(rank, energy) = corr(-rank, energy)


def maxab_observed(rank_a, rank_b, energy, *, support_a, support_b, min_points=3) -> dict:
    """rho_A, rho_B, rho_maxAB (design §9). rho_maxAB eligible only when BOTH direction templates are
    eligible; if only one survives, its signed association is secondary and maxAB is unresolved."""
    ra = np.asarray(rank_a, float); rb = np.asarray(rank_b, float); e = np.asarray(energy, float)
    rho_a = _earliness(ra, e, np.asarray(support_a, bool), min_points) if rank_a is not None else float("nan")
    rho_b = _earliness(rb, e, np.asarray(support_b, bool), min_points) if rank_b is not None else float("nan")
    a_ok, b_ok = np.isfinite(rho_a), np.isfinite(rho_b)
    if a_ok and b_ok:
        return {"rho_a": rho_a, "rho_b": rho_b, "rho_maxab": float(max(rho_a, rho_b)),
                "maxab_eligible": True, "single_direction": None}
    if a_ok or b_ok:
        return {"rho_a": rho_a if a_ok else None, "rho_b": rho_b if b_ok else None,
                "rho_maxab": None, "maxab_eligible": False,
                "single_direction": "A_to_B" if a_ok else "B_to_A"}
    return {"rho_a": None, "rho_b": None, "rho_maxab": None, "maxab_eligible": False,
            "single_direction": None}


def maxab_permutation_null(rank_a, rank_b, energy, *, support_a, support_b, groups=None,
                           n_permutations=10000, seed=0, min_points=3,
                           max_exact_permutations=50000) -> dict:
    """One-sided null for rho_maxAB by permuting the target energy labels and RECOMPUTING max(rho_A,rho_B)
    inside EACH permutation (design §9, test 5). ``groups`` (shaft labels over ALL contacts) gives the
    within-shaft null preserving shaft membership (test 6); groups=None gives the unrestricted shuffle.
    Enumerate exactly when the constrained space <= max_exact_permutations, else Monte Carlo.
    Never reports the better single direction's one-template p as the maxAB p."""
    ra = np.asarray(rank_a, float); rb = np.asarray(rank_b, float); e = np.asarray(energy, float)
    sa = np.asarray(support_a, bool); sb = np.asarray(support_b, bool)
    obs = maxab_observed(ra, rb, e, support_a=sa, support_b=sb, min_points=min_points)
    if not obs["maxab_eligible"]:
        return {"status": "maxab_ineligible", "observed": obs["rho_maxab"], "p_one_sided": None,
                "single_direction": obs["single_direction"], "n_permutations": 0}
    obs_stat = obs["rho_maxab"]
    n = e.size
    locs = _permutation_groups(n, None if groups is None else np.asarray(groups))
    n_possible = int(np.prod([factorial(len(loc)) for loc in locs], dtype=object))

    def stat(perm):
        ep = e[perm]
        rho_a = _earliness(ra, ep, sa, min_points)
        rho_b = _earliness(rb, ep, sb, min_points)
        both = [x for x in (rho_a, rho_b) if np.isfinite(x)]
        return max(both) if both else np.nan

    if n_possible <= int(max_exact_permutations):
        method = "exact"
        null = np.asarray([stat(p) for p in _exact_permutation_indices(
            n, None if groups is None else np.asarray(groups))], float)
    else:
        method = "monte_carlo"
        rng = np.random.default_rng(int(seed))
        null = np.asarray([stat(_permutation_indices(n, rng, None if groups is None else np.asarray(groups)))
                           for _ in range(int(n_permutations))], float)
    null = null[np.isfinite(null)]
    if null.size == 0:
        return {"status": "degenerate_null", "observed": float(obs_stat), "p_one_sided": None,
                "method": method, "n_permutations": 0}
    p = (float(np.sum(null >= obs_stat) / null.size) if method == "exact"
         else float((1 + np.sum(null >= obs_stat)) / (1 + null.size)))
    return {"status": "eligible", "observed": float(obs_stat), "p_one_sided": p,
            "null_median": float(np.median(null)), "null_p95": float(np.percentile(null, 95)),
            "method": method, "n_permutations": int(null.size), "n_unique_possible": n_possible}


def toroidal_maxab_null(rank_a_grid, rank_b_grid, energy_grid, *, support_a_grid, support_b_grid,
                        n=24, min_points=3) -> dict:
    """Source-grid null: non-zero toroidal translations of the energy field, recomputing rho_maxAB
    (design §9, test 7). Excludes the zero shift and preserves the energy multiset (np.roll). Reports
    the number of unique shifts."""
    ea = np.asarray(rank_a_grid, float); eb = np.asarray(rank_b_grid, float)
    E = np.asarray(energy_grid, float).reshape(n, n)
    sa = np.asarray(support_a_grid, bool); sb = np.asarray(support_b_grid, bool)

    def stat(flat_energy):
        rho_a = _earliness(ea, flat_energy, sa, min_points)
        rho_b = _earliness(eb, flat_energy, sb, min_points)
        both = [x for x in (rho_a, rho_b) if np.isfinite(x)]
        return max(both) if both else np.nan

    obs = stat(E.ravel())
    if not np.isfinite(obs):
        return {"status": "maxab_ineligible", "observed": None, "p_one_sided": None, "n_shifts": 0}
    null = []
    for dx in range(n):
        for dy in range(n):
            if dx == 0 and dy == 0:
                continue                                     # exclude the identity shift
            null.append(stat(np.roll(np.roll(E, dx, axis=0), dy, axis=1).ravel()))
    null = np.asarray(null, float)
    null = null[np.isfinite(null)]
    if null.size == 0:
        return {"status": "degenerate_null", "observed": float(obs), "p_one_sided": None, "n_shifts": 0}
    p = float((1 + np.sum(null >= obs)) / (1 + null.size))
    return {"status": "eligible", "observed": float(obs), "p_one_sided": p,
            "null_median": float(np.median(null)), "null_p95": float(np.percentile(null, 95)),
            "n_shifts": int(null.size)}


# ======================================================================== §9 core-loading + participation
def contact_core_loading(contacts, posE, core_mask_E, kernel_width_mm=1.5) -> np.ndarray:
    """Per-contact Gaussian core loading (design §9): Gaussian(distance)-weighted fraction of nearby
    E neurons that belong to either low-V_th core. Reimplements the early-readout adapter's
    ``_project(core.mask, ...)`` definition for the SNN frame. Contact is 'direct-core' when loading
    >= threshold (the runner keeps loading < threshold; threshold saved to metadata)."""
    C = np.asarray(contacts, float)
    P = np.asarray(posE, float)
    core = np.asarray(core_mask_E, float)
    sig = float(kernel_width_mm)
    out = np.empty(C.shape[0])
    for k in range(C.shape[0]):
        d2 = np.sum((P - C[k]) ** 2, axis=1)
        w = np.exp(-0.5 * d2 / (sig ** 2))
        s = w.sum()
        out[k] = float(np.dot(w, core) / s) if s > 0 else 0.0
    return out


def local_participation(contacts, posE, E_spk_bool, window_steps, radius_mm=1.5) -> np.ndarray:
    """Per-contact fraction of E neurons within radius_mm that fire in the window (design §9 audit).

    A readable contact signal is NOT direct local-tissue recruitment when this fraction is low.
    window_steps = (start_step, end_step) into E_spk_bool."""
    C = np.asarray(contacts, float)
    P = np.asarray(posE, float)
    s, e = int(window_steps[0]), int(window_steps[1])
    fired = np.asarray(E_spk_bool)[s:e].any(axis=0)
    out = np.full(C.shape[0], np.nan)
    for k in range(C.shape[0]):
        near = np.flatnonzero(np.sum((P - C[k]) ** 2, axis=1) <= radius_mm ** 2)
        if near.size:
            out[k] = float(fired[near].mean())
    return out


# ======================================================================== §10 (optional) global/local depletion
def global_local_fractions(z_E) -> dict:
    """Descriptive depletion-field decomposition d_i = 1 - z_i (design §10). global = uniform gain share,
    local = patterned share; they sum to one. NOT fed back into the dynamics."""
    z = np.asarray(z_E, float)
    d = 1.0 - z
    ss = float(np.sum(d ** 2))
    if ss <= 0:
        return {"global_fraction": float("nan"), "local_fraction": float("nan"), "sum_sq": ss}
    N = d.size
    g = float(N * (d.mean() ** 2) / ss)
    l = float(np.sum((d - d.mean()) ** 2) / ss)
    return {"global_fraction": g, "local_fraction": l, "sum_sq": ss}


# ======================================================================== §13 serialization + resume
def to_jsonable(obj):
    """Recursively convert numpy types to Python and non-finite floats to None (design §13, test 9)."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def resume_should_skip(existing_json_path, expected_fingerprint) -> bool:
    """--resume: skip a seed iff its bridge_metrics.json exists, is status 'complete', and its
    provenance fingerprint matches (design §13, test 10). Any mismatch/absence -> re-run."""
    import json
    import os
    if not os.path.exists(existing_json_path):
        return False
    try:
        with open(existing_json_path) as f:
            d = json.load(f)
    except Exception:
        return False
    return d.get("status") == "complete" and d.get("provenance_fingerprint") == expected_fingerprint
