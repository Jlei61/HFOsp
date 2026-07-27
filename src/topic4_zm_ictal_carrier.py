"""Z/M ictal-carrier gate: virtual-SEEG high-frequency readout + source-space + observed metric
extraction (spec docs/superpowers/specs/2026-07-24-topic4-zm-ictal-carrier-gate-design.md §1-§4).

Pure functions (no simulation) so the sustained-carrier vs HFO-burst-train discrimination, the Nyquist
gate, and the provenance/resume contract are unit-testable. The runner (scripts/run_topic4_zm_ictal_
carrier.py) drives the SNN via the engine's built-in `lfp_recorder=` hook (observation-only, no engine
edit) and calls these to produce `ictal_carrier_verdict` / `lifecycle_verdict`.

Thresholds live in src/topic4_zm_carrier_verdict.py (the pre-registered authority); this module imports
them and reuses `analyze_macroepisode` -- it does NOT reinvent the macroepisode machinery (CLAUDE.md §6.1).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile

import numpy as np
import scipy.signal as ss

from src.topic4_zm_carrier_verdict import (
    analyze_macroepisode, is_sustained,
    BIN_FINE_MS, SMOOTH_MS, ON_FRAC, SEP_FACTOR, A7_DIMS_REQUIRED, ENH_DB, N_CONTACTS_MIN,
    FLASH_FRAC, FLASH_WINDOW_MS, MIN_MACRO_MS)

CLASSIFIER_VERSION = "zm_carrier_gate_v1_2026-07-24"
LFP_SAMPLE_HZ = 10000.0        # engine samples every dt=0.1 ms
DECIMATE_Q = 5                 # 10 kHz -> 2 kHz stored
STFT_WIN_MS = 250.0
STFT_HOP_MS = 25.0
BANDS = {"lowgamma": (30.0, 80.0), "highfreq": (80.0, 150.0), "broadband": (1.0, 150.0)}
FMAX_HZ = 150.0                # highest band edge we must resolve
RUNAWAY_ALLE_HZ = 150.0        # full-T (un-truncated) runaway: sustained all-E population rate this high
PLATEAU_AREA_FRAC = 0.50       # saturated whole-field plateau: >= this frac of ALL E cells active, sustained


# ================================================================ LFP / spectral (CM1-CM4)
def lfp_sample_hz(dt_ms):
    """Native per-step recorder sampling rate for an integration step in ms."""
    dt_ms = float(dt_ms)
    if not np.isfinite(dt_ms) or dt_ms <= 0:
        raise ValueError("dt_ms must be finite and positive")
    return 1000.0 / dt_ms


def decimate_lfp(lfp, fs_in=LFP_SAMPLE_HZ, q=DECIMATE_Q):
    """10 kHz -> fs_in/q with an anti-alias FIR (scipy.signal.decimate). Returns (lfp_ds, fs_out)."""
    lfp = np.asarray(lfp, float)
    if lfp.ndim == 1:
        lfp = lfp[:, None]
    ds = ss.decimate(lfp, q, ftype="fir", axis=0)
    return ds.astype(np.float32), fs_in / q


def assert_nyquist(fs, fmax=FMAX_HZ):
    """Hard Nyquist gate (spec §1.2 / stop #1): the stored LFP must resolve up to fmax Hz."""
    if fs / 2.0 <= fmax:
        raise ValueError(f"stored LFP fs={fs} Hz -> Nyquist {fs/2.0} Hz <= {fmax} Hz: cannot resolve the "
                         f"HFO band; refusing to analyse (spec §1.2, stop condition #1).")


def band_envelopes(lfp, fs, win_ms=STFT_WIN_MS, hop_ms=STFT_HOP_MS, bands=BANDS):
    """Per-contact band power over STFT frames. Window Hann `win_ms`, hop `hop_ms`, per-window LINEAR
    detrend (removes DC + slope of the rectified proxy -- spec §1.3). Returns (frame_times_ms, {band:
    (n_frames, n_contacts)})."""
    lfp = np.asarray(lfp, float)
    if lfp.ndim == 1:
        lfp = lfp[:, None]
    nper = max(16, int(round(win_ms / 1000.0 * fs)))
    nover = nper - max(1, int(round(hop_ms / 1000.0 * fs)))
    n_contacts = lfp.shape[1]
    env = {b: [] for b in bands}
    tvec = None
    for c in range(n_contacts):
        f, tt, Sxx = ss.spectrogram(lfp[:, c], fs=fs, window="hann", nperseg=nper, noverlap=nover,
                                    detrend="linear", scaling="density", mode="psd")
        tvec = tt
        for b, (lo, hi) in bands.items():
            m = (f >= lo) & (f < hi)
            env[b].append(np.trapz(Sxx[m, :], f[m], axis=0))
    for b in bands:
        env[b] = np.stack(env[b], axis=1)          # (n_frames, n_contacts)
    return tvec * 1000.0, env


def to_db(env, pre_frames):
    """dB of a band-power envelope relative to its PRE-ONSET median (per contact). Also returns robust-z
    (reported, NOT gated -- MAD can be ~0 in a quiet baseline and inflate z; spec §1.3)."""
    env = np.asarray(env, float)
    tiny = np.finfo(float).tiny
    pre = env[:max(1, pre_frames)]
    med = np.median(pre, axis=0)
    med_safe = np.where(med <= 0, tiny, med)
    db = 10.0 * np.log10(np.maximum(env, tiny) / med_safe)
    mad = np.median(np.abs(pre - med), axis=0)
    z = (env - med) / np.where(mad <= 0, tiny, 1.4826 * mad)
    return db, z


# ================================================================ source-space (CM6)
def moving_average(x, win_bins):
    x = np.asarray(x, float)
    w = max(1, int(round(win_bins)))
    if w <= 1 or x.size < w:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def fine_rates(E_spk_bool, core_mask, dt_ms, bin_ms=BIN_FINE_MS):
    """core / surround / all-E population rate (Hz) + active-fraction per `bin_ms` bin, from the raster."""
    E = np.asarray(E_spk_bool)
    n, NE = E.shape
    core_mask = np.asarray(core_mask, bool)
    bs = max(1, int(round(bin_ms / dt_ms)))

    def rate(mask):
        nsel = max(1, int(mask.sum()))
        out = []
        for b0 in range(0, n, bs):
            seg = E[b0:b0 + bs][:, mask]
            out.append(float(seg.sum()) / nsel / (seg.shape[0] * dt_ms) * 1e3)
        return np.asarray(out, float)

    af = []
    for b0 in range(0, n, bs):
        seg = E[b0:b0 + bs]
        af.append(float((seg.sum(axis=0) > 0).mean()))
    core = rate(core_mask)
    return dict(core=core, surround=rate(~core_mask), allE=rate(np.ones(NE, bool)),
                active_frac=np.asarray(af, float), t_ms=np.arange(len(core)) * bin_ms, bin_ms=float(bin_ms))


def _events(trace, dt_ms, baseline, amp, on_frac=ON_FRAC):
    """Contiguous supra-ON runs (returns list of (dur_ms, peak, energy))."""
    on = baseline + on_frac * amp
    above = np.asarray(trace) >= on
    idx = np.flatnonzero(above)
    if idx.size == 0:
        return []
    ev, start, prev = [], int(idx[0]), int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i - prev > 1:
            ev.append((start, prev + 1))
            start = i
        prev = i
    ev.append((start, prev + 1))
    out = []
    for i0, i1 in ev:
        seg = np.asarray(trace)[i0:i1]
        out.append(((i1 - i0) * dt_ms, float(seg.max()), float(np.clip(seg - baseline, 0, None).sum() * dt_ms)))
    return out


def _tail_escalates(all_rate):
    """Full-T (un-truncated) runaway backup: the last 5% of the all-E rate sits at a sustained very-high
    population rate (a genuine bounded carrier's all-E MEAN stays modest even with high-rate focal bursts)."""
    tail = all_rate[-max(1, len(all_rate) // 20):]
    return bool(tail.mean() >= RUNAWAY_ALLE_HZ)


def compute_source_metrics(E_spk_bool, core_mask, posE, src_xy, axis_unit, L, dt_ms,
                           runaway_early_stop_ms):
    """Gate-A metrics from the raster (spec §3). e_A = smoothed core rate; macroepisode via the shared
    machinery; recruitment / whole-field-flash / saturation from the active-area; A7 separation vs the
    pre-onset interictal events."""
    fr = fine_rates(E_spk_bool, core_mask, dt_ms)
    e_A = moving_average(fr["core"], SMOOTH_MS / fr["bin_ms"])
    macro = analyze_macroepisode(e_A, fr["bin_ms"])
    af_macro = analyze_macroepisode(fr["active_frac"], fr["bin_ms"])
    onset_ms = macro["onset_ms"]

    # ---- A6 recruitment / A8 whole-field flash from the active-area around onset ----
    flash = recruit = False
    onset_area = peak_area = 0.0
    if onset_ms is not None:
        oi = int(round(onset_ms / fr["bin_ms"]))
        fw = max(1, int(round(FLASH_WINDOW_MS / fr["bin_ms"])))
        onset_area = float(fr["active_frac"][oi:oi + fw].mean())
        peak_area = float(fr["active_frac"][oi:].max())
        flash = bool(peak_area > 0 and onset_area >= FLASH_FRAC * peak_area)     # ignites most of eventual area at once
        recruit = bool((not flash) and (peak_area >= 1.2 * max(onset_area, 1e-9) or is_sustained(af_macro)))

    # ---- saturated whole-field plateau: sustained AND active area ~ whole sheet ----
    sat_area = float(fr["active_frac"][int(round((onset_ms or 0) / fr["bin_ms"])):].mean()) if onset_ms else 0.0
    saturated = bool(is_sustained(macro) and sat_area >= PLATEAU_AREA_FRAC)

    # ---- A7: separation of the macroepisode from pre-onset interictal events (peak, duration, area) ----
    src_sep_count = A7_DIMS_REQUIRED    # default: if there are no pre-onset events, an ictal macro is trivially distinct
    if onset_ms is not None:
        oi = int(round(onset_ms / fr["bin_ms"]))
        pre = e_A[:oi]
        if pre.size:
            b = float(np.median(pre[:max(1, pre.size)]))
            ev = _events(pre, fr["bin_ms"], b, macro["peak"] - b)
            if ev:
                med_dur = float(np.median([e[0] for e in ev]))
                med_peak = float(np.median([e[1] for e in ev]))
                med_area = float(np.median([e[2] for e in ev]))
                macro_energy = float(np.clip(e_A[oi:oi + int(macro["duration_ms"] / fr["bin_ms"])] - macro["baseline"],
                                             0, None).sum() * fr["bin_ms"])
                dims = 0
                dims += macro["duration_ms"] >= SEP_FACTOR * max(med_dur, 1e-9)
                dims += macro["peak"] >= SEP_FACTOR * max(med_peak, 1e-9)
                dims += macro_energy >= SEP_FACTOR * max(med_area, 1e-9)
                src_sep_count = int(dims)

    return dict(macro=macro, af_macro=af_macro, onset_ms=onset_ms,
                whole_field_flash=flash, has_recruitment=recruit, saturated_plateau=saturated,
                tail_escalating=_tail_escalates(fr["allE"]), src_sep_count=src_sep_count,
                runaway_early_stop_ms=runaway_early_stop_ms, onset_area=onset_area, peak_area=peak_area,
                rates=fr, e_A=e_A)


# ================================================================ observed virtual-SEEG (Gate B)
def compute_observed_metrics(lfp, fs, onset_ms):
    """Gate-B metrics from the (decimated) virtual SEEG (spec §4). Per-contact low-gamma dB envelope ->
    macroepisode + peak dB; count sustained contacts (B1); high-freq co-enhancement (B2); best contact
    (B3-B5); B6 separation left to carrier_metrics_from."""
    assert_nyquist(fs)
    tms, env = band_envelopes(lfp, fs)
    dt_frame_ms = float(np.median(np.diff(tms))) if tms.size > 1 else STFT_HOP_MS
    pre_frames = max(1, int(np.sum(tms < onset_ms)))
    lg_db, _ = to_db(env["lowgamma"], pre_frames)
    hf_db, _ = to_db(env["highfreq"], pre_frames)
    bb_db, _ = to_db(env["broadband"], pre_frames)
    pre_onset_ms = pre_frames * dt_frame_ms      # aligns analyze_macroepisode's baseline window to the true pre-onset

    contacts = []
    for c in range(lg_db.shape[1]):
        macro = analyze_macroepisode(lg_db[:, c], dt_frame_ms, pre_onset_ms=pre_onset_ms)
        peak_lg = float(lg_db[:, c].max())
        contacts.append(dict(macro=macro, peak_lowgamma_db=peak_lg,
                             peak_highfreq_db=float(hf_db[:, c].max()),
                             peak_broadband_db=float(bb_db[:, c].max()),
                             sustained=bool(is_sustained(macro) and peak_lg >= ENH_DB)))
    sustained = [c for c in contacts if c["sustained"]]
    highfreq_enhanced = any(c["peak_highfreq_db"] >= ENH_DB or c["peak_broadband_db"] >= ENH_DB for c in contacts)
    pool = sustained if sustained else contacts
    best_idx = int(max(range(len(contacts)),
                       key=lambda i: (contacts[i]["sustained"], contacts[i]["macro"]["duration_ms"])))
    return dict(n_sustained_contacts=len(sustained), highfreq_enhanced=bool(highfreq_enhanced),
                best_macro=contacts[best_idx]["macro"], best_contact_idx=best_idx, contacts=contacts,
                times_ms=tms, lowgamma_db=lg_db, highfreq_db=hf_db, broadband_db=bb_db,
                frame_dt_ms=dt_frame_ms, pre_frames=pre_frames)


def _observed_sep_count(observed):
    """B6: macroepisode vs pre-onset returning events on the best contact, over 4 dims
    {duration, duty-cycle, energy, spatial-extent}. Returns count of separated dims (0..4)."""
    bc = observed["best_contact_idx"]
    lg = observed["lowgamma_db"][:, bc]
    dt = observed["frame_dt_ms"]
    pre = observed["pre_frames"]
    macro = observed["best_macro"]
    if macro["onset_ms"] is None:
        return 0
    b = float(np.median(lg[:pre])) if pre else 0.0
    amp = macro["peak"] - b
    ev = _events(lg[:pre], dt, b, amp)
    med_dur = float(np.median([e[0] for e in ev])) if ev else 0.0
    med_energy = float(np.median([e[2] for e in ev])) if ev else 0.0
    on = b + ON_FRAC * amp
    pre_duty = float((lg[:pre] >= on).mean()) if pre else 0.0
    oi = int(round(macro["onset_ms"] / dt))
    macro_energy = float(np.clip(lg[oi:oi + int(macro["duration_ms"] / dt)] - b, 0, None).sum() * dt)
    dims = 0
    dims += med_dur <= 0 or macro["duration_ms"] >= SEP_FACTOR * med_dur            # duration
    dims += pre_duty <= 0 or macro["occupancy"] >= SEP_FACTOR * pre_duty            # duty-cycle
    dims += med_energy <= 0 or macro_energy >= SEP_FACTOR * med_energy              # energy
    dims += observed["n_sustained_contacts"] >= N_CONTACTS_MIN                      # spatial extent
    return int(dims)


def carrier_metrics_from(source, observed):
    """Assemble the metric dict consumed by ictal_carrier_verdict (spec §5)."""
    return dict(
        runaway_early_stop_ms=source.get("runaway_early_stop_ms"),
        tail_escalating=source["tail_escalating"],
        whole_field_flash=source["whole_field_flash"],
        saturated_plateau=source["saturated_plateau"],
        has_recruitment=source["has_recruitment"],
        src_macro=source["macro"],
        src_sep_count=source["src_sep_count"],
        obs_n_sustained_contacts=observed["n_sustained_contacts"],
        obs_highfreq_enhanced=observed["highfreq_enhanced"],
        obs_best_macro=observed["best_macro"],
        obs_sep_count=_observed_sep_count(observed),
    )


# ================================================================ kymographs / snapshots (figures)
def axis_transverse_coords(posE, src_xy, axis_unit):
    """Per-E-neuron (axial, transverse) coordinate in mm relative to the source centroid along axis_unit."""
    d = np.asarray(posE, float) - np.asarray(src_xy, float)
    ax = d @ np.asarray(axis_unit, float)
    perp = np.array([-axis_unit[1], axis_unit[0]])
    tr = d @ perp
    return ax, tr


def kymograph(E_spk_bool, coord_mm, dt_ms, bin_ms=5.0, n_pos=40):
    """Spike-density kymograph: E spikes binned in (position-along-`coord_mm` × time). Returns
    (density (n_pos, n_time), pos_edges_mm, t_ms)."""
    E = np.asarray(E_spk_bool)
    n, NE = E.shape
    bs = max(1, int(round(bin_ms / dt_ms)))
    lo, hi = float(np.min(coord_mm)), float(np.max(coord_mm))
    edges = np.linspace(lo, hi, n_pos + 1)
    pos_bin = np.clip(np.digitize(coord_mm, edges) - 1, 0, n_pos - 1)
    nt = (n + bs - 1) // bs
    dens = np.zeros((n_pos, nt))
    for ti, b0 in enumerate(range(0, n, bs)):
        seg = E[b0:b0 + bs]
        counts = seg.sum(axis=0)                          # per-neuron spike count in the time bin
        np.add.at(dens[:, ti], pos_bin, counts)
    return dens, edges, np.arange(nt) * bin_ms


def field_snapshot(E_spk_bool, posE, L, t0_ms, t1_ms, dt_ms, n_grid=32):
    """E spike-density on the L×L sheet integrated over [t0_ms, t1_ms). Returns (n_grid, n_grid)."""
    E = np.asarray(E_spk_bool)
    i0, i1 = int(round(t0_ms / dt_ms)), int(round(t1_ms / dt_ms))
    i0, i1 = max(0, i0), min(E.shape[0], max(i0 + 1, i1))
    counts = E[i0:i1].sum(axis=0)
    ix = np.clip((posE[:, 0] / L * n_grid).astype(int), 0, n_grid - 1)
    iy = np.clip((posE[:, 1] / L * n_grid).astype(int), 0, n_grid - 1)
    field = np.zeros((n_grid, n_grid))
    np.add.at(field, (iy, ix), counts)
    return field


# ================================================================ provenance (CM11)
def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha(root):
    return subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()


def write_json_atomic(path, obj):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2, default=_json_default)
        os.replace(tmp, path)                             # atomic on POSIX
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def read_manifest(path):
    if os.path.exists(path):
        try:
            return json.load(open(path))
        except Exception:
            pass
    return dict(arms={})


def write_arm_to_manifest(path, arm_record):
    """Merge one arm's record into the manifest WITHOUT clobbering the other arms (crash-safe, atomic)."""
    man = read_manifest(path)
    man.setdefault("arms", {})[arm_record["arm"]] = arm_record
    write_json_atomic(path, man)


def arm_completed(manifest, arm):
    a = manifest.get("arms", {}).get(arm)
    return bool(a and a.get("status") == "complete")
