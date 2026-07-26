"""Empirical carrier / readout lock (spec rev3.1 §4.1-§4.3, plan Task 5).

Two separable jobs:

1. A metric battery that can tell a genuine broadband recruiting carrier apart from the two things
   that mimic it -- a SHARP HARMONIC PULSE TRAIN (whose comb fakes broadband power) and a
   STATIONARY GLOBAL OSCILLATOR (whose occupancy fakes persistence). These discriminators are
   validated on synthetic fixtures, which is where they can actually be falsified.

2. A resolver for the REAL E1146 reference classes -- early-ictal windows and returning interictal
   group events -- which supply the empirical intervals the model must fall inside. If those
   artifacts cannot be resolved with provenance, the caller writes `blocked_reference_artifacts` and
   the top-level verdict keeps `observation_layer_blocked`. Model-derived thresholds are NEVER
   substituted for the missing real distribution.

The band machinery is reused from src/topic4_zm_ictal_carrier (STFT band envelopes, dB, Nyquist
gate) and src/topic4_zm_carrier_gate_v2 (macroepisode) rather than reinvented: the question those
helpers answer -- "is there a sustained, energy-enhanced episode on this contact?" -- is the same
question here, only the input is real SEEG instead of the model's virtual SEEG.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import os
import zoneinfo

import numpy as np
import scipy.signal as ss

import src.topic4_zm_ictal_carrier as CG
import src.topic4_zm_carrier_gate_v2 as CGV2

LOCK_VERSION = "zm_empirical_carrier_v1_2026-07-26"

EPI_ROOT = "/mnt/epilepsia_data"
E1146_REC = os.path.join(EPI_ROOT, "inv", "pat_114602", "adm_1146102", "rec_114600102")
E1146_TZ = "Europe/Berlin"
SUBJECT = "1146"

#: metrics the observation gate is allowed to use; anything not listed is not silently invented
GATE_METRICS = ("duration_ms", "occupancy", "peak_lowgamma_db", "n_independent_contacts",
                "harmonic_comb", "spectral_entropy", "inst_freq_drift_hz", "burst_interval_cv",
                "phase_coherence")
#: spec §4.2 asks for these too; they are SOURCE-space constructs in this model and are reported
#: from the source metrics instead, so the observation lock records them as deliberately absent
GATE_METRICS_NOT_IN_OBSERVATION = ("wavefront_velocity_variability", "spatial_phase_entropy",
                                   "axial_first_passage")


# ================================================================ metric battery
def harmonic_comb_concentration(x, fs, fmin=1.0, fmax=150.0, n_harm=8, tol_hz=1.5):
    """Fraction of 1-150 Hz power sitting within +-tol of the fundamental and its harmonics.

    A sharp pulse train at f0 puts nearly all its power on the comb -> ~1. Broadband recruiting
    activity spreads power off the comb -> low. This is the statistic that stops harmonic
    pseudo-broadband from passing as ictal broadband (upstream acceptance doc §2.2).
    """
    f, P = ss.welch(np.asarray(x, float), fs=fs, nperseg=min(len(x), int(fs)))
    band = (f >= fmin) & (f <= fmax)
    f, P = f[band], P[band]
    if P.sum() <= 0 or f.size < 8:
        return float("nan")
    f0 = float(f[int(np.argmax(P))])
    if f0 <= 0:
        return float("nan")
    m = np.zeros_like(f, bool)
    for h in range(1, n_harm + 1):
        m |= np.abs(f - h * f0) <= tol_hz
    return float(P[m].sum() / P.sum())


def spectral_entropy(x, fs, fmin=1.0, fmax=150.0):
    """Normalized entropy of the 1-150 Hz PSD. 1 = flat/broadband, ~0 = one narrow line."""
    f, P = ss.welch(np.asarray(x, float), fs=fs, nperseg=min(len(x), int(fs)))
    band = (f >= fmin) & (f <= fmax)
    P = P[band]
    if P.sum() <= 0 or P.size < 4:
        return float("nan")
    p = P / P.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(P.size))


def inst_freq_drift(x, fs, fmin=1.0, fmax=150.0, win_s=0.25):
    """SD (Hz) of the spectral centroid over time -- a stationary oscillator does not drift."""
    nper = max(16, int(round(win_s * fs)))
    f, t, S = ss.spectrogram(np.asarray(x, float), fs=fs, nperseg=nper,
                             noverlap=nper // 2, detrend="linear")
    band = (f >= fmin) & (f <= fmax)
    f, S = f[band], S[band]
    tot = S.sum(axis=0)
    ok = tot > 0
    if ok.sum() < 3:
        return float("nan")
    cen = (f[:, None] * S).sum(axis=0)[ok] / tot[ok]
    return float(np.std(cen))


def burst_interval_cv(env, dt_ms):
    """CV of inter-peak intervals of an envelope. A metronomic pulse train -> ~0."""
    e = np.asarray(env, float)
    if e.size < 8:
        return float("nan")
    pk, _ = ss.find_peaks(e, height=float(np.median(e) + 0.5 * (e.max() - np.median(e))))
    if pk.size < 3:
        return float("nan")
    iv = np.diff(pk) * dt_ms
    return float(np.std(iv) / np.mean(iv)) if np.mean(iv) > 0 else float("nan")


def dominant_band(x, fs, fmin=3.0, fmax=150.0, rel_width=0.35):
    """Band around the dominant rhythmic component. Coherence has to be measured WHERE the rhythm
    is: a fixed 30-80 Hz window scores a 5 Hz whole-field oscillator as incoherent. The search
    starts at 3 Hz on a DETRENDED signal -- these readouts are rectified power proxies, whose shared
    near-DC component is coherent by construction and would otherwise capture every band choice.
    """
    x = np.asarray(x, float)
    m = x.mean(axis=1) if x.ndim > 1 else x
    f, P = ss.welch(ss.detrend(m), fs=fs, nperseg=min(len(m), int(fs)))
    band = (f >= fmin) & (f <= fmax)
    if not band.any():
        return (30.0, 80.0)
    f0 = float(f[band][int(np.argmax(P[band]))])
    f0 = max(f0, fmin + 1.0)
    return (max(fmin, f0 * (1 - rel_width)), min(fs / 2 * 0.95, f0 * (1 + rel_width) + 1.0))


def phase_coherence(lfp, fs, band=None):
    """Mean pairwise phase-locking across contacts in the dominant band. Whole-field rhythm -> ~1."""
    x = np.asarray(lfp, float)
    if x.ndim == 1 or x.shape[1] < 2:
        return float("nan")
    band = dominant_band(x, fs) if band is None else band
    b, a = ss.butter(4, [band[0] / (fs / 2), min(band[1] / (fs / 2), 0.99)], btype="band")
    ph = np.angle(ss.hilbert(ss.filtfilt(b, a, x, axis=0), axis=0))
    n = ph.shape[1]
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            vals.append(abs(np.mean(np.exp(1j * (ph[:, i] - ph[:, j])))))
    return float(np.mean(vals)) if vals else float("nan")


def independent_contacts(active_idx, coords, kernel_width):
    """Greedy count of active contacts separated by more than one readout-kernel width (§4.3).

    Two adjacent contacts fed by the same hotspot are ONE independent contact.
    """
    idx = list(active_idx)
    if not idx:
        return 0
    pts = np.asarray(coords, float)[idx]
    kept = [0]
    for i in range(1, len(idx)):
        if all(np.linalg.norm(pts[i] - pts[k]) > kernel_width for k in kept):
            kept.append(i)
    return len(kept)


def metric_battery(lfp, fs, coords=None, kernel_width=0.0, baseline_ms=CGV2.OBS_BASELINE_MS):
    """The locked observation battery, computed identically on real SEEG and on the model vSEEG."""
    x = np.asarray(lfp, float)
    if x.ndim == 1:
        x = x[:, None]
    CG.assert_nyquist(fs)
    obs = CGV2.compute_observed_gate_v2(x, fs, baseline_ms=baseline_ms)
    best = obs["best_macro"]
    bc = obs["best_contact_idx"]
    lg = obs["lg_db"]
    active = [c for c in range(x.shape[1]) if obs["contacts"][c]["sustained"]]
    dt_frame = obs["frame_dt_ms"]
    return dict(
        duration_ms=float(best["duration_ms"]), occupancy=float(best["occupancy"]),
        peak_lowgamma_db=float(obs["contacts"][bc]["peak_lowgamma_db"]),
        n_sustained_contacts=int(obs["n_sustained_contacts"]),
        n_independent_contacts=int(independent_contacts(active, coords, kernel_width)
                                  if coords is not None else len(active)),
        harmonic_comb=harmonic_comb_concentration(x[:, bc], fs),
        spectral_entropy=spectral_entropy(x[:, bc], fs),
        inst_freq_drift_hz=inst_freq_drift(x[:, bc], fs),
        burst_interval_cv=burst_interval_cv(lg[:, bc], dt_frame),
        phase_coherence=phase_coherence(x, fs),
        highfreq_enhanced=bool(obs["highfreq_enhanced"]),
        onset_ms=best["onset_ms"], best_contact_idx=int(bc))


# ================================================================ synthetic nulls
def synth_pulse_train(fs, dur_s, f0=10.0, n_contacts=6, width_ms=3.0, seed=0):
    """Sharp harmonic pulse train: high occupancy, high peak power, comb spectrum."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur_s * fs)) / fs
    p = np.zeros_like(t)
    w = max(1, int(round(width_ms * 1e-3 * fs)))
    for k in range(int(dur_s * f0)):
        i = int(k / f0 * fs)
        p[i:i + w] = 1.0
    x = np.outer(p, np.ones(n_contacts)) * 8.0 + 0.4 * rng.standard_normal((t.size, n_contacts))
    return np.abs(x)


def synth_global_oscillator(fs, dur_s, f0=5.0, n_contacts=6, seed=0):
    """Stationary whole-field rhythm: high occupancy, near-perfect phase coherence, no drift."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur_s * fs)) / fs
    base = 4.0 + 3.0 * np.sin(2 * np.pi * f0 * t)
    x = np.outer(base, np.ones(n_contacts)) + 0.3 * rng.standard_normal((t.size, n_contacts))
    return np.abs(x)


def synth_broadband_carrier(fs, dur_s, n_contacts=6, seed=0):
    """Continuous noisy broadband recruitment with a drifting centre frequency and desynchronised
    contacts -- the thing the gate is supposed to accept."""
    rng = np.random.default_rng(seed)
    n = int(dur_s * fs)
    out = np.zeros((n, n_contacts))
    for c in range(n_contacts):
        # INDEPENDENT frequency random walk per contact. A shared frequency ramp would give every
        # contact a constant phase OFFSET and hence PLV ~ 1 -- i.e. it would look like the very
        # whole-field oscillator this class is supposed to be distinguishable from.
        drift = 25.0 + 30.0 * np.cumsum(rng.standard_normal(n)) / np.sqrt(n)
        ph = np.cumsum(2 * np.pi * np.clip(drift, 5.0, 140.0) / fs) + rng.uniform(0, 2 * np.pi)
        out[:, c] = 5.0 + 3.0 * np.sin(ph) + 2.5 * rng.standard_normal(n)
    out[: int(0.3 * fs)] *= 0.15                       # a quiet pre-onset baseline
    return np.abs(out)


# ================================================================ real E1146 references
def _block_index():
    """(path_stem, start_epoch, n_samples, fs) for every E1146 raw block, from the .head files."""
    tz = zoneinfo.ZoneInfo(E1146_TZ)
    rows = []
    if not os.path.isdir(E1146_REC):
        return rows
    for name in sorted(os.listdir(E1146_REC)):
        if not name.endswith(".head"):
            continue
        info = {}
        with open(os.path.join(E1146_REC, name)) as f:
            for line in f:
                k, _, v = line.partition("=")
                info[k.strip()] = v.strip()
        try:
            dtm = _dt.datetime.strptime(info["start_ts"], "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=tz)
        except (KeyError, ValueError):
            continue
        rows.append(dict(stem=os.path.join(E1146_REC, name[:-5]),
                         start_epoch=dtm.timestamp(), n_samples=int(info["num_samples"]),
                         fs=float(info["sample_freq"]),
                         elec_names=info.get("elec_names", "").strip("[]").split(",")))
    return rows


def resolve_early_ictal_windows(seizure_rows, pre_s=5.0, post_s=25.0, max_n=6):
    """Map real seizure EEG onsets onto (block, crop_start_sec, duration) windows."""
    blocks = _block_index()
    out = []
    for r in seizure_rows:
        try:
            onset = float(r["eeg_onset_epoch"])
        except (KeyError, TypeError, ValueError):
            continue
        for b in blocks:
            end = b["start_epoch"] + b["n_samples"] / b["fs"]
            if b["start_epoch"] <= onset - pre_s and onset + post_s <= end:
                out.append(dict(kind="early_ictal", seizure_id=r.get("seizure_id"),
                                stem=b["stem"], crop_start_sec=onset - pre_s - b["start_epoch"],
                                duration_sec=pre_s + post_s, fs=b["fs"], onset_epoch=onset,
                                pre_s=pre_s))
                break
        if len(out) >= max_n:
            break
    return out


def resolve_interictal_windows(seizure_rows, dur_s=30.0, max_n=6, guard_s=3600.0):
    """Interictal windows at least `guard_s` from any seizure, spread across the recording."""
    blocks = _block_index()
    onsets = []
    for r in seizure_rows:
        try:
            onsets.append(float(r["eeg_onset_epoch"]))
        except (KeyError, TypeError, ValueError):
            continue
    out = []
    for b in blocks[:: max(1, len(blocks) // max(1, max_n))]:
        mid = b["start_epoch"] + b["n_samples"] / b["fs"] / 2.0
        if all(abs(mid - o) > guard_s for o in onsets):
            out.append(dict(kind="interictal", stem=b["stem"],
                            crop_start_sec=b["n_samples"] / b["fs"] / 2.0,
                            duration_sec=dur_s, fs=b["fs"]))
        if len(out) >= max_n:
            break
    return out


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
            if h.name and f.tell() > (1 << 28):     # 256 MB prefix is enough to pin identity
                break
    return h.hexdigest()


# ================================================================ the lock
def build_lock(ictal_metrics, interictal_metrics, pulse_metrics, osc_metrics, *,
               null_quantile=0.95, ictal_interval=(0.05, 0.95), min_n=3):
    """Turn the four reference classes into the data-locked observation gate.

    A model metric must (a) exceed the null quantile where larger is more ictal-like (or fall below
    it where smaller is), (b) lie inside the broad empirical early-ictal interval, and (c) sit
    outside the returning-interictal / pulse-train region. Quantiles, the multiplicity rule, the
    missing-data rule and the minimum sample count are fixed HERE, before any model fork is read.
    """
    def col(rows, k):
        v = np.array([r[k] for r in rows if r.get(k) is not None and np.isfinite(r.get(k, np.nan))],
                     float)
        return v

    #: direction[k] = +1 when larger is more ictal-like, -1 when smaller is
    direction = dict(duration_ms=+1, occupancy=+1, peak_lowgamma_db=+1, n_independent_contacts=+1,
                     harmonic_comb=-1, spectral_entropy=+1, inst_freq_drift_hz=+1,
                     burst_interval_cv=+1, phase_coherence=-1)
    enough = len(ictal_metrics) >= min_n and len(interictal_metrics) >= min_n
    lock = dict(version=LOCK_VERSION, null_quantile=null_quantile,
                ictal_interval=list(ictal_interval), min_n=min_n,
                multiplicity_rule="every listed metric must pass; no weighted score",
                missing_data_rule="a metric that is nan in the model is a FAIL, not a skip",
                n_ictal=len(ictal_metrics), n_interictal=len(interictal_metrics),
                n_pulse=len(pulse_metrics), n_oscillator=len(osc_metrics),
                sufficient_reference_sample=bool(enough),
                metrics_not_in_observation_layer=list(GATE_METRICS_NOT_IN_OBSERVATION),
                thresholds={})
    for k in GATE_METRICS:
        d = direction[k]
        ic, ii = col(ictal_metrics, k), col(interictal_metrics, k)
        nulls = np.concatenate([col(pulse_metrics, k), col(osc_metrics, k), ii]) \
            if (len(pulse_metrics) or len(osc_metrics) or ii.size) else np.zeros(0)
        entry = dict(direction=int(d))
        if nulls.size:
            entry["null_bound"] = float(np.quantile(nulls, null_quantile if d > 0
                                                    else 1 - null_quantile))
        if ic.size:
            entry["ictal_lo"] = float(np.quantile(ic, ictal_interval[0]))
            entry["ictal_hi"] = float(np.quantile(ic, ictal_interval[1]))
        if ii.size:
            entry["interictal_median"] = float(np.median(ii))
        lock["thresholds"][k] = entry
    return lock


def evaluate_against_lock(model_metrics, lock):
    """Apply the locked gate to one model readout. Fail-closed on missing/NaN metrics."""
    if not lock.get("sufficient_reference_sample"):
        return dict(verdict="observation_layer_blocked", per_metric={},
                    reason="reference sample below the locked minimum")
    per = {}
    for k, e in lock["thresholds"].items():
        v = model_metrics.get(k)
        if v is None or not np.isfinite(v):
            per[k] = dict(value=None, passed=False, why="missing/nan (fail-closed)")
            continue
        d = e["direction"]
        ok = True
        why = []
        if "null_bound" in e:
            beats = v > e["null_bound"] if d > 0 else v < e["null_bound"]
            ok &= beats
            if not beats:
                why.append(f"does not beat null bound {e['null_bound']:.4g}")
        if "ictal_lo" in e:
            inside = e["ictal_lo"] <= v <= e["ictal_hi"]
            ok &= inside
            if not inside:
                why.append(f"outside empirical early-ictal [{e['ictal_lo']:.4g},{e['ictal_hi']:.4g}]")
        per[k] = dict(value=float(v), passed=bool(ok), why="; ".join(why))
    passed = all(p["passed"] for p in per.values())
    return dict(verdict="observation_carrier" if passed else "fails_observation_gate",
                per_metric=per, n_failed=sum(not p["passed"] for p in per.values()))
