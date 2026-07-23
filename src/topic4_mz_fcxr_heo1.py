"""FCXR-HEO1 high-energy-oscillatory-branch spectral classifier (pure; no engine, no sim).

Consumes a virtual-SEEG LFP trace (nsteps, n_contacts) + population rate_E (nsteps,) recorded at the
engine dt, plus a slow-off baseline reference, and decides HEO_BRANCH via the pre-registered gates
A-E of the 2026-07-24 design lock (§3). All thresholds are locked module constants here; NOTHING is
tuned to a run's outcome. Gates:

  B broadband (per contact,window): >=5/6 bands pass AND (30-80) & (80-150) pass AND median 6-band
    dB gain >= 6 dB, where a band "passes" iff robust-z >= Z_GATE AND raw power >= baseline q99.
  C platform (per window): >=11/15 contacts broadband-high AND >=3/4 SCL contacts broadband-high.
  A plateau: longest platform-high run (100 ms hop) with occupancy >= 0.80, no >=2 consecutive low
    windows (return-to-baseline < ~100 ms), duration >= 1000 ms.
  D oscillation (on the plateau): population-rate PSD AND >=half passing contacts have a non-DC
    30-150 Hz peak with prominence >= 6 dB over local background; >=half share a center +-15 Hz.
  E numerical: finite, clip_frac==0, tau_eff_min>=2dt, not runaway, plateau mean rate < 400 Hz.

HEO_BRANCH = A AND B AND C AND D AND E.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import decimate, spectrogram, welch

# ---- locked spectral + gate constants (design lock §3) ----
FS_WORK = 1000.0
WIN_MS = 1000.0
HOP_MS = 100.0
BANDS = [(1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 80.0), (80.0, 150.0)]
BROADBAND_IDX = (4, 5)            # (30,80) and (80,150) must both pass
Z_GATE = 3.0
DB_GAIN_GATE = 6.0
N_BANDS_GATE = 5
N_CONTACTS_GATE = 11
N_SCL_GATE = 3
PLATEAU_MIN_MS = 1000.0
PLATEAU_OCC = 0.80
PLATEAU_MAX_GAP_MS = 100.0        # operationalized at hop resolution: no >=2 consecutive low windows
OSC_LO, OSC_HI = 30.0, 150.0
OSC_PROM_DB = 6.0
OSC_CENTER_TOL_HZ = 15.0
CEIL_HZ = 400.0                   # plateau mean rate >= this -> pinned near the 500 Hz refractory ceiling


# ----------------------------------------------------------------- decimation + spectrogram
def _factorize(n):
    """Factors <=10 whose product is n (staged FIR decimation)."""
    out = []
    for f in (5, 4, 3, 2):
        while n % f == 0 and n > 1:
            out.append(f); n //= f
    if n > 1:
        out.append(n)
    return out or [1]


def decimate_to_work(sig, dt):
    """Decimate a (nsteps,) or (nsteps,C) signal from fs=1000/dt down to ~FS_WORK (staged FIR)."""
    x = np.asarray(sig, float)
    fs_raw = 1000.0 / float(dt)
    factor = int(round(fs_raw / FS_WORK))
    if factor <= 1:
        return x, fs_raw
    for f in _factorize(factor):
        x = decimate(x, f, axis=0, ftype="fir")
    return x, fs_raw / factor


def band_power_spectrogram(sig, fs):
    """(n_windows, n_contacts, n_bands) band power via a 1 s Hann spectrogram at HOP_MS hop.

    Returns (bandpower, window_center_times_s). Falls back to a shorter window if the signal is
    shorter than WIN_MS (keeps unit tests + short runs analysable)."""
    x = np.asarray(sig, float)
    if x.ndim == 1:
        x = x[:, None]
    C = x.shape[1]
    nper = min(x.shape[0], int(round(WIN_MS * fs / 1000.0)))
    nover = nper - int(round(HOP_MS * fs / 1000.0))
    nover = max(0, min(nover, nper - 1))
    out = None
    tcen = None
    df = None
    for c in range(C):
        f, t, Sxx = spectrogram(x[:, c], fs=fs, window="hann", nperseg=nper, noverlap=nover,
                                scaling="density", mode="psd")
        if out is None:
            out = np.zeros((Sxx.shape[1], C, len(BANDS)))
            tcen = t
            df = f[1] - f[0]
        for b, (lo, hi) in enumerate(BANDS):
            m = (f >= lo) & (f < hi)
            out[:, c, b] = Sxx[m, :].sum(axis=0) * df
    return out, tcen


# ----------------------------------------------------------------- baseline reference
def build_baseline_reference(lfp, rate, dt):
    """Per-(contact,band) slow-off reference from the F0 windows: median/MAD of log10 power, q99, median
    power. rate is accepted for schema symmetry (not used to build the reference)."""
    ldec, fs = decimate_to_work(lfp, dt)
    bp, _ = band_power_spectrogram(ldec, fs)               # (nw, C, B)
    logbp = np.log10(np.maximum(bp, 1e-300))
    med_log = np.median(logbp, axis=0)
    return dict(
        med_log=med_log,
        mad_log=np.median(np.abs(logbp - med_log[None]), axis=0),
        q99_power=np.percentile(bp, 99.0, axis=0),
        med_power=np.median(bp, axis=0),
        fs=float(fs), n_windows=int(bp.shape[0]),
    )


# ----------------------------------------------------------------- gate helpers
def longest_plateau(platform):
    """Gate A. Longest [i,j] on the HOP_MS grid with occupancy>=PLATEAU_OCC, no >=2 consecutive low
    windows, endpoints high, duration>=PLATEAU_MIN_MS. Returns dict or None."""
    max_gap_win = max(1, int(PLATEAU_MAX_GAP_MS // HOP_MS))     # =1 -> single-window flicker tolerated
    n = len(platform)
    highs = np.flatnonzero(platform)
    best = None
    for a in highs:
        run_low = 0
        occ_hi = 0
        for j in range(a, n):
            if platform[j]:
                run_low = 0; occ_hi += 1
            else:
                run_low += 1
                if run_low > max_gap_win:
                    break
            length = j - a + 1
            occ = occ_hi / length
            if platform[j] and occ >= PLATEAU_OCC and length * HOP_MS >= PLATEAU_MIN_MS:
                if best is None or length > (best["j"] - best["i"] + 1):
                    best = dict(i=int(a), j=int(j), duration_ms=float(length * HOP_MS), occupancy=float(occ))
    return best


def _osc_peak(f, psd, lo=OSC_LO, hi=OSC_HI):
    """Strongest non-DC peak in [lo,hi]: (peak_hz, prominence_db over +-20Hz background excl +-5Hz)."""
    band = (f >= lo) & (f <= hi)
    if band.sum() < 3:
        return np.nan, -np.inf
    fb, pb = f[band], psd[band]
    k = int(np.argmax(pb))
    peak_hz, peak_p = float(fb[k]), float(pb[k])
    near = (f >= peak_hz - 20) & (f <= peak_hz + 20)
    excl = (f >= peak_hz - 5) & (f <= peak_hz + 5)
    bgmask = near & ~excl
    bg = float(np.median(psd[bgmask])) if np.any(bgmask) else float(np.median(pb))
    prom = 10.0 * np.log10(peak_p / max(bg, 1e-300))
    return peak_hz, float(prom)


# ----------------------------------------------------------------- classifier
def classify_heo(lfp, rate, dt, scl_mask, baseline_ref, safety=None):
    """Full HEO gate. safety = numerical row (numerical_unsafe, runaway_early_stop_ms). Returns a
    verdict dict with HEO_BRANCH + per-gate booleans + plateau/oscillation/coverage diagnostics."""
    scl_mask = np.asarray(scl_mask, bool)
    safety = safety or {}
    ldec, fs = decimate_to_work(lfp, dt)
    rdec, _ = decimate_to_work(rate, dt)
    bp, tcen = band_power_spectrogram(ldec, fs)            # (nw, C, B)
    logbp = np.log10(np.maximum(bp, 1e-300))
    med_log = baseline_ref["med_log"]; mad_log = baseline_ref["mad_log"]
    q99 = baseline_ref["q99_power"]; med_power = baseline_ref["med_power"]

    denom = 1.4826 * mad_log                               # (C,B)
    with np.errstate(divide="ignore", invalid="ignore"):
        zscore = (logbp - med_log[None]) / denom[None]
    zscore = np.where(np.isfinite(zscore), zscore, -np.inf)
    zscore[:, denom == 0] = -np.inf                        # MAD==0 -> fail closed
    band_pass = (zscore >= Z_GATE) & (bp >= q99[None])     # (nw,C,B)
    db = 10.0 * np.log10(np.maximum(bp, 1e-300) / np.maximum(med_power[None], 1e-300))
    med_db = np.median(db, axis=2)                         # (nw,C)
    n_bands = band_pass.sum(axis=2)                        # (nw,C)
    broad_ok = band_pass[:, :, BROADBAND_IDX[0]] & band_pass[:, :, BROADBAND_IDX[1]]
    contact_high = (n_bands >= N_BANDS_GATE) & broad_ok & (med_db >= DB_GAIN_GATE)   # (nw,C) -> Gate B
    n_high = contact_high.sum(axis=1)
    scl_high = contact_high[:, scl_mask].sum(axis=1)
    platform = (n_high >= N_CONTACTS_GATE) & (scl_high >= N_SCL_GATE)                # (nw,) -> Gate C

    plateau = longest_plateau(platform)                                             # Gate A

    # Gate D — oscillation on the plateau window (else on the whole run for reporting)
    osc = dict(rate_peak_hz=np.nan, rate_prom_db=float("-inf"), n_contacts_osc=0,
               frac_contacts_osc=0.0, common_center=False, center_hz=np.nan, plateau_mean_rate_hz=np.nan)
    if plateau is not None:
        i, j = plateau["i"], plateau["j"]
        t0 = max(0.0, tcen[i] - WIN_MS / 2000.0)
        t1 = tcen[j] + WIN_MS / 2000.0
        s0, s1 = int(t0 * fs), min(ldec.shape[0], int(t1 * fs))
        seg_lfp, seg_rate = ldec[s0:s1], rdec[s0:s1]
        nper = max(8, min(len(seg_rate), int(round(0.5 * fs))))
        fr, pr = welch(seg_rate, fs=fs, nperseg=nper)
        osc["rate_peak_hz"], osc["rate_prom_db"] = _osc_peak(fr, pr)
        osc["plateau_mean_rate_hz"] = float(seg_rate.mean())
        passing = np.flatnonzero(contact_high[i:j + 1].mean(axis=0) >= 0.5)
        peaks = []
        for c in passing:
            fc, pc = welch(seg_lfp[:, c], fs=fs, nperseg=nper)
            hz, prom = _osc_peak(fc, pc)
            if prom >= OSC_PROM_DB:
                peaks.append(hz)
        osc["n_contacts_osc"] = len(peaks)
        osc["frac_contacts_osc"] = len(peaks) / max(len(passing), 1)
        if peaks:
            center = float(np.median(peaks))
            osc["center_hz"] = center
            osc["common_center"] = bool(np.mean(np.abs(np.array(peaks) - center) <= OSC_CENTER_TOL_HZ) >= 0.5)
    osc_ok = bool(plateau is not None and osc["rate_prom_db"] >= OSC_PROM_DB
                  and osc["frac_contacts_osc"] >= 0.5 and osc["common_center"])

    gate_A = plateau is not None
    gate_C = bool(platform.any())
    gate_B = bool(contact_high[platform].any()) if gate_C else bool(contact_high.any())
    gate_D = osc_ok
    ceil_ok = plateau is None or osc["plateau_mean_rate_hz"] < CEIL_HZ
    gate_E = bool((not safety.get("numerical_unsafe", False))
                  and safety.get("runaway_early_stop_ms") is None and ceil_ok)
    heo = bool(gate_A and gate_B and gate_C and gate_D and gate_E)
    return dict(
        HEO_BRANCH=heo,
        gate_A_plateau=bool(gate_A), gate_B_broadband=bool(gate_B), gate_C_platform=bool(gate_C),
        gate_D_oscillation=bool(gate_D), gate_E_numerical=bool(gate_E),
        plateau=plateau, oscillation=osc,
        max_platform_contacts=int(n_high.max()) if n_high.size else 0,
        max_platform_scl=int(scl_high.max()) if scl_high.size else 0,
        platform_window_frac=float(platform.mean()) if platform.size else 0.0,
        n_windows=int(bp.shape[0]), fs_work=float(fs),
    )
