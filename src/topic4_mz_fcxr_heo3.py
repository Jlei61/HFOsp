"""FCXR-HEO3 instruments — the joint sliding-window gate + source-space audit the HEO2.1 review requires.

Two gaps in HEO2.1 that this module closes:
 (c) whole-window summaries can't tell whether broadband / desync / energy hold AT THE SAME TIME, and a
     whole-window coherence drop can come from on-off switching rather than within-burst spatial phase
     spread -> `joint_target_windows` judges recruited ∧ broadband ∧ phase-dispersed ∧ high-energy per
     200-300 ms window, on ACTIVE samples only, and reports whether such windows persist consecutively.
 (b) sensor-space 15/15 may just be one core seen by 15 contacts -> `source_space_audit` reports
     per-region (core-source / core-sink / axis-corridor / off-axis) activity, participation ratio,
     activity centroid and its propagation order, from the E-cell spike field.

Band order matches HEO1/HEO2: [1-4, 4-8, 8-13, 13-30, 30-80, 80-150] Hz; indices 0..4 = 1-80 Hz.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt, spectrogram

BANDS = [(1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 80.0), (80.0, 150.0)]


# ------------------------------------------------------------------ per-window spectral field
def band_power_windows(lfp_dec, fs, win_ms=250.0, hop_ms=50.0):
    """(n_windows, n_contacts, n_bands) band power on a SHORT Hann window (HEO1's is fixed at 1 s).
    Density scaling x df -> band power is window-length independent in expectation, so these windows
    stay comparable to the 1 s baseline reference. Returns (bandpower, window_center_times_s)."""
    x = np.asarray(lfp_dec, float)
    if x.ndim == 1:
        x = x[:, None]
    nper = min(x.shape[0], max(16, int(round(win_ms * fs / 1000.0))))
    nover = max(0, min(nper - int(round(hop_ms * fs / 1000.0)), nper - 1))
    out, tcen = None, None
    for c in range(x.shape[1]):
        f, t, Sxx = spectrogram(x[:, c], fs=fs, window="hann", nperseg=nper, noverlap=nover,
                                scaling="density", mode="psd")
        if out is None:
            out = np.zeros((Sxx.shape[1], x.shape[1], len(BANDS)))
            tcen, df = t, f[1] - f[0]
        for b, (lo, hi) in enumerate(BANDS):
            m = (f >= lo) & (f < hi)
            out[:, c, b] = Sxx[m, :].sum(axis=0) * df
    return out, tcen


def phase_order_parameter(lfp_dec, fs, band=(4.0, 30.0)):
    """Cross-contact Kuramoto order parameter R(t) from Hilbert phase in `band`.
    R=1 -> every contact in phase (fully synchronous); R->0 -> phases dispersed across contacts.
    Per-sample (not a windowed coherence), so it can be restricted to ACTIVE samples — this is what
    keeps silent gaps from masquerading as spatial desynchronization (review P1-c)."""
    x = np.asarray(lfp_dec, float)
    if x.ndim == 1:
        x = x[:, None]
    lo, hi = band
    sos = butter(4, [lo / (fs / 2.0), min(hi / (fs / 2.0), 0.99)], btype="band", output="sos")
    xf = sosfiltfilt(sos, x - x.mean(axis=0), axis=0)
    phi = np.angle(hilbert(xf, axis=0))
    return np.abs(np.exp(1j * phi).mean(axis=1))


# ------------------------------------------------------------------ the joint gate
def joint_target_windows(bp, ref_med_power, rate_dec, order_R, fs_win, *, thr_db=3.0,
                         min_recruit=12, min_broadband=8, max_order=0.80, min_rate_hz=60.0,
                         k_bands=3, active_rate_hz=20.0):
    """Judge recruited ∧ broadband ∧ phase-dispersed ∧ high-energy PER WINDOW (review P1-c).

    bp: (nw, C, B) short-window band power; ref_med_power: (C, B) baseline median; rate_dec/order_R:
    per-sample population rate and order parameter (resampled onto the windows); fs_win: windows/s.
    Phase dispersion uses the ACTIVE-sample mean of R inside each window (silence excluded), so an
    on-off burst train cannot score as 'dispersed' just by having gaps.
    Returns per-window arrays + the fraction of target windows + the longest CONSECUTIVE run (ms)."""
    ddb = 10.0 * np.log10(np.maximum(bp, 1e-300) / np.maximum(ref_med_power[None], 1e-300))
    up = ddb[:, :, :5] >= thr_db                                  # (nw, C, 5) 1-80 Hz only
    recruit = up.any(axis=2).sum(axis=1)                          # contacts with ANY 1-80 band up
    broadband = (up.sum(axis=2) >= k_bands).sum(axis=1)           # contacts with >=k of 5 bands up
    crit = dict(recruited=recruit >= min_recruit, broadband=broadband >= min_broadband,
                dispersed=order_R <= max_order, high_energy=rate_dec >= min_rate_hz)
    target = crit["recruited"] & crit["broadband"] & crit["dispersed"] & crit["high_energy"]
    # longest consecutive run of target windows -> ms
    best = run = 0
    for t in target:
        run = run + 1 if t else 0
        best = max(best, run)
    return dict(recruit=recruit, broadband=broadband, order_R=order_R, rate=rate_dec,
                target=target, criteria={k: v for k, v in crit.items()},
                frac_target=float(target.mean()) if target.size else 0.0,
                longest_run_ms=float(best / fs_win * 1000.0),
                frac_by_criterion={k: float(v.mean()) if v.size else 0.0 for k, v in crit.items()})


def resample_to_windows(sig, tcen, dt_sig_ms, reduce="mean", gate=None):
    """Average a per-sample signal into the spectrogram windows (optionally only over `gate` samples,
    e.g. active samples). Returns one value per window (nan-safe: falls back to the plain mean)."""
    x = np.asarray(sig, float)
    idx = np.clip((np.asarray(tcen, float) * 1000.0 / dt_sig_ms).astype(int), 0, len(x) - 1)
    half = max(1, int(np.median(np.diff(idx)) / 2)) if len(idx) > 1 else 1
    out = np.empty(len(idx))
    for i, c in enumerate(idx):
        lo, hi = max(0, c - half), min(len(x), c + half + 1)
        seg = x[lo:hi]
        if gate is not None:
            g = np.asarray(gate, bool)[lo:hi]
            if g.any():
                seg = seg[g]
        out[i] = seg.max() if reduce == "max" else seg.mean()
    return out


# ------------------------------------------------------------------ source-space audit
def build_regions(posE, src_xy, snk_xy, core_r, corridor_w=None):
    """E-indexed boolean masks: core_source / core_sink / axis_corridor / off_axis (review P1-b).
    The corridor is the band of width `corridor_w` (default core_r) around the source->sink segment,
    excluding the two cores; off_axis is everything else."""
    p = np.asarray(posE, float)
    a, b = np.asarray(src_xy, float), np.asarray(snk_xy, float)
    w = float(core_r if corridor_w is None else corridor_w)
    core_s = np.linalg.norm(p - a, axis=1) <= core_r
    core_k = np.linalg.norm(p - b, axis=1) <= core_r
    ab = b - a
    L2 = float(ab @ ab) or 1.0
    s = np.clip(((p - a) @ ab) / L2, 0.0, 1.0)                     # projection along the axis in [0,1]
    perp = np.linalg.norm(p - (a[None] + s[:, None] * ab[None]), axis=1)
    corridor = (perp <= w) & ~core_s & ~core_k
    return dict(core_source=core_s, core_sink=core_k, axis_corridor=corridor,
                off_axis=~(core_s | core_k | corridor), axis_coord=s)


def participation_ratio(rates):
    """Effective FRACTION of E-cells carrying the activity: (Σr)²/(N·Σr²). 1 = perfectly uniform,
    ~n/N = only n cells active. Distinguishes 'the whole tissue is recruited' from 'one core is loud'."""
    r = np.asarray(rates, float)
    s2 = float((r ** 2).sum())
    return float((r.sum() ** 2) / (len(r) * s2)) if s2 > 0 else 0.0


def source_space_audit(E_spk_bool, posE, regions, dt, win_ms=250.0, hop_ms=50.0):
    """Per-window source-space readout from the E spike field (nsteps, NE): per-region mean rate (Hz)
    and active-cell fraction, whole-field participation ratio, rate-weighted centroid + its axis
    coordinate. Answers 'is the tissue recruited, or is one core just loud?' (review P1-b)."""
    spk = np.asarray(E_spk_bool)
    nsteps, NE = spk.shape
    w = max(1, int(round(win_ms / dt))); h = max(1, int(round(hop_ms / dt)))
    p = np.asarray(posE, float)
    rows = []
    for s0 in range(0, max(1, nsteps - w + 1), h):
        seg = spk[s0:s0 + w]
        cnt = seg.sum(axis=0).astype(float)
        rate = cnt / (w * dt / 1000.0)                              # per-neuron Hz
        tot = float(rate.sum())
        cen = (p * rate[:, None]).sum(axis=0) / tot if tot > 0 else np.full(2, np.nan)
        row = dict(t_ms=float((s0 + w / 2) * dt), participation_ratio=participation_ratio(rate),
                   mean_rate_hz=float(rate.mean()),
                   centroid_x=float(cen[0]), centroid_y=float(cen[1]))
        for name, m in regions.items():
            if name == "axis_coord":
                continue
            row[f"rate_{name}"] = float(rate[m].mean()) if m.any() else 0.0
            row[f"activefrac_{name}"] = float((cnt[m] > 0).mean()) if m.any() else 0.0
        ac = regions["axis_coord"]
        row["centroid_axis_coord"] = float((ac * rate).sum() / tot) if tot > 0 else float("nan")
        rows.append(row)
    return rows
