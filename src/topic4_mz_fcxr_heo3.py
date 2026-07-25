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
def band_power_windows(lfp_dec, fs, win_ms=1000.0, hop_ms=100.0):
    """(n_windows, n_contacts, n_bands) band power on a Hann window of `win_ms` (HEO1's is fixed at 1 s).

    ⚠️ WINDOW-LENGTH FLOOR: the joint gate must NOT use a 200-300 ms window. At 250 ms the spectrogram
    has no bin below 4 Hz (the 1-4 Hz band comes out empty -> -inf dB) and the Hann main lobe is ~16 Hz
    wide, so a strong 16 Hz peak leaks +20 dB into 8-13 Hz and fakes 'broadband'. The target phenotype
    is defined by ~3 Hz content, so 1 s (>=3 cycles at 3 Hz, ~1 Hz bins) is the shortest defensible
    window; the joint judgement is kept dense via the 100 ms hop. Density scaling x df keeps band power
    comparable to the 1 s baseline reference. Returns (bandpower, window_center_times_s)."""
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


def band_phase(lfp_dec, fs, band=(4.0, 30.0)):
    """Instantaneous Hilbert phase per contact, band-limited. (n_samples, C)."""
    x = np.asarray(lfp_dec, float)
    if x.ndim == 1:
        x = x[:, None]
    lo, hi = band
    sos = butter(4, [lo / (fs / 2.0), min(hi / (fs / 2.0), 0.99)], btype="band", output="sos")
    xf = sosfiltfilt(sos, x - x.mean(axis=0), axis=0)
    return np.angle(hilbert(xf, axis=0))


def phase_order_parameter(lfp_dec, fs, band=(4.0, 30.0)):
    """Cross-contact Kuramoto order parameter R(t) = instantaneous phase ALIGNMENT.

    ⚠️ NOT the desynchronization instrument. A travelling wave (fixed inter-contact lag, phase spread
    over ~180°) is perfectly ORGANIZED yet scores R~0.4 — the FCXR 16 Hz reference state does exactly
    this (coherence 0.98, phase span 186°, R 0.43). Use `pairwise_plv_windows` to judge whether the
    phase ORGANIZATION broke up; R is reported only as a descriptive alignment/travelling-wave readout."""
    return np.abs(np.exp(1j * band_phase(lfp_dec, fs, band)).mean(axis=1))


def pairwise_plv_windows(phi, tcen, dt_ms, win_ms, active=None):
    """Per-window mean pairwise phase-locking value = consistency of inter-contact phase RELATIONSHIPS.

    PLV_cd = |mean_t exp(i(φ_d - φ_c))| over the window's samples; returns the mean over contact pairs.
    Invariant to fixed lags, so a travelling wave scores ~1 (organized) while phases that drift relative
    to one another score low (genuinely desynchronized) — this is the criterion HEO3 needs. `active`
    restricts the average to supra-threshold samples so silent gaps cannot fake desynchronization."""
    p = np.asarray(phi, float)
    n, C = p.shape
    half = max(1, int(round(win_ms / dt_ms / 2)))
    out = np.full(len(tcen), np.nan)
    for i, tc in enumerate(np.asarray(tcen, float)):
        c0 = int(round(tc * 1000.0 / dt_ms))
        lo, hi = max(0, c0 - half), min(n, c0 + half + 1)
        seg = p[lo:hi]
        if active is not None:
            g = np.asarray(active, bool)[lo:hi]
            if g.sum() < 8:
                continue                                   # too little ACTIVE signal -> undefined (nan),
            seg = seg[g]                                   #   NOT "desynchronized": silence must not pass
        if seg.shape[0] < 8:
            continue
        z = np.exp(1j * seg)                                   # (m, C)
        M = np.abs(z.conj().T @ z) / seg.shape[0]              # (C, C) pairwise PLV
        iu = np.triu_indices(C, k=1)
        out[i] = float(M[iu].mean())
    return out


# ------------------------------------------------------------------ the joint gate
def joint_target_windows(bp, ref_med_power, rate_dec, plv, fs_win, *, thr_db=3.0,
                         min_recruit=12, min_broadband=8, max_plv=0.60, min_rate_hz=60.0,
                         k_bands=3):
    """Judge recruited ∧ broadband ∧ phase-DESYNCHRONIZED ∧ high-energy PER WINDOW (review P1-c).

    bp: (nw, C, B) band power (>=1 s window, see band_power_windows); ref_med_power: (C, B) baseline
    median; rate_dec: per-window population rate; `plv`: per-window mean pairwise PLV on ACTIVE samples
    (pairwise_plv_windows) — NOT the instantaneous order parameter, which a travelling wave fails for
    the wrong reason. Desynchronized := PLV <= max_plv (phase relationships stopped being consistent).
    Returns per-window arrays + fraction of target windows + longest CONSECUTIVE run (ms)."""
    ddb = 10.0 * np.log10(np.maximum(bp, 1e-300) / np.maximum(ref_med_power[None], 1e-300))
    up = ddb[:, :, :5] >= thr_db                                  # (nw, C, 5) 1-80 Hz only
    recruit = up.any(axis=2).sum(axis=1)                          # contacts with ANY 1-80 band up
    broadband = (up.sum(axis=2) >= k_bands).sum(axis=1)           # contacts with >=k of 5 bands up
    plv = np.asarray(plv, float)
    crit = dict(recruited=recruit >= min_recruit, broadband=broadband >= min_broadband,
                desynchronized=np.nan_to_num(plv, nan=1.0) <= max_plv,   # nan (silent) -> not desync
                high_energy=rate_dec >= min_rate_hz)
    target = crit["recruited"] & crit["broadband"] & crit["desynchronized"] & crit["high_energy"]
    best = run = 0
    for t in target:                                              # longest consecutive run
        run = run + 1 if t else 0
        best = max(best, run)
    return dict(recruit=recruit, broadband=broadband, plv=plv, rate=rate_dec,
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
