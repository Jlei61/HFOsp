#!/usr/bin/env python3
"""One-run spatial capture for the locked MZ-divisive v2 representative.

This is a readout-only producer.  It reuses the accepted E1146 substrate, the existing
``MZDivisivePoolSlowVars`` composite and ``kick_probe.simulate_kick`` without changing the
engine.  The full E-neuron spike raster exists only in RAM while the existing engine runs;
the written artifact contains compact derived fields and never serializes ``E_spk_bool``.

Locked cell (the finite-window v2 opening, not a lifecycle pass): seed=1, T=20 s,
alpha_G=2, alpha_TG=4, tau_TG=750 ms, M off, spontaneous/no kick.
"""
from __future__ import annotations

import os

# Force one BLAS thread before importing NumPy/SciPy.  This capture is deliberately single-process.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/hfosp_mpl_cache")

import argparse  # noqa: E402
import contextlib  # noqa: E402
import dataclasses  # noqa: E402
import datetime as dt_datetime  # noqa: E402
import fcntl  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import resource  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from scipy.signal import butter, hilbert, sosfiltfilt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

import run_m4_dynamic_qi as M4  # noqa: E402
import run_m4_phaseplane as PP  # noqa: E402
import run_sef_hfo_snn_cm_spontaneous_readout as CM  # noqa: E402
import run_topic4_mz_divisive_lifecycle as LIFE  # noqa: E402
from kick_probe import simulate_kick  # noqa: E402
from lfp import LFPRecorder  # noqa: E402
from src.topic4_mz_early_field_bridge import event_contact_timing  # noqa: E402


SEED = 1
T_MS = 20_000.0
DT_MS = float(PP.DT)
MIN_RESERVE_GIB = 96.0
DEFAULT_WORKER_GIB = 16.0
RESULT_DIR = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_divisive_lifecycle", "figure_capture")
NPZ_PATH = os.path.join(RESULT_DIR, "current_stage_capture.npz")
JSON_PATH = os.path.join(RESULT_DIR, "current_stage_capture.json")

# Exact current-stage state/event contract.
RECRUITED_HZ = 20.0
STATE_ENVELOPE_MS = 250.0
MIN_RECRUITED_MS = 1_000.0
EVENT_ENVELOPE_MS = 50.0
EVENT_MAX_MS = 200.0
EVENT_RETURN_MS = 250.0
EVENT_TO_ONSET_GUARD_MS = 500.0
EVENT_FILTER_GUARD_MS = 100.0
RECRUITED_WINDOW_MS = 1_000.0
MOVIE_GRID = 24
MOVIE_BIN_MS = 25.0
AXIAL_BINS = 48
AXIAL_TIME_BIN_MS = 10.0
LFP_BAND_HZ = (30.0, 80.0)
LFP_EXPORT_DT_MS = 1.0


def _moving_average(x, n):
    """Centered edge-padded mean, identical to the strict lifecycle audit."""
    x = np.asarray(x, float)
    n = max(1, int(n))
    if n == 1:
        return x.copy()
    left = n // 2
    right = n - 1 - left
    padded = np.pad(x, (left, right), mode="edge")
    csum = np.r_[0.0, np.cumsum(padded, dtype=float)]
    return (csum[n:] - csum[:-n]) / float(n)


def _episodes(mask):
    idx = np.flatnonzero(np.asarray(mask, bool))
    if idx.size == 0:
        return []
    out = []
    start = prev = int(idx[0])
    for raw in idx[1:]:
        i = int(raw)
        if i != prev + 1:
            out.append((start, prev + 1))
            start = i
        prev = i
    out.append((start, prev + 1))
    return out


def strict_recruited_episode(rate_hz, dt_ms=DT_MS):
    """First 250-ms-envelope >=20-Hz component lasting at least 1 s.

    Returns a JSON-ready record plus the state envelope.  No gaps are merged, matching the
    strict post-hoc lifecycle audit.  An episode ending at the last sample is explicitly marked
    ``reaches_record_end`` and must not be interpreted as observed termination.
    """
    rate = np.asarray(rate_hz, float)
    if rate.ndim != 1 or rate.size == 0 or not np.all(np.isfinite(rate)):
        raise ValueError("rate_hz must be a non-empty finite 1D array")
    if dt_ms <= 0:
        raise ValueError("dt_ms must be > 0")
    n_env = max(1, int(round(STATE_ENVELOPE_MS / dt_ms)))
    min_n = max(1, int(round(MIN_RECRUITED_MS / dt_ms)))
    env = _moving_average(rate, n_env)
    candidates = [(a, b) for a, b in _episodes(env >= RECRUITED_HZ) if b - a >= min_n]
    if not candidates:
        return {
            "status": "no_recruited_macrostate",
            "onset_ms": None,
            "episode_end_ms": None,
            "duration_ms": 0.0,
            "reaches_record_end": False,
        }, env
    a, b = candidates[0]
    reaches_end = b >= rate.size
    return {
        "status": "recruited_macrostate",
        "onset_ms": float(a * dt_ms),
        "episode_end_ms": float(b * dt_ms),
        "duration_ms": float((b - a) * dt_ms),
        "reaches_record_end": bool(reaches_end),
    }, env


def select_pre_onset_returning_event(rate_hz, dt_ms, macro_onset_ms, state_envelope=None):
    """Deterministically select the latest clearly returned pre-onset event.

    Candidate: a contiguous 50-ms rate envelope crossing 20 Hz for <=200 ms, ending at
    least 500 ms before recruited-state onset.  Return: after the centered-envelope half-width,
    the following 250-ms state-envelope segment has median <20 Hz and ends <20 Hz.  The latest
    eligible event is selected (ties: higher peak, then earlier onset).  There is no visual pick.
    """
    rate = np.asarray(rate_hz, float)
    if macro_onset_ms is None:
        raise ValueError("cannot select a pre-onset event without a recruited-state onset")
    state = (_moving_average(rate, max(1, int(round(STATE_ENVELOPE_MS / dt_ms))))
             if state_envelope is None else np.asarray(state_envelope, float))
    event_env = _moving_average(rate, max(1, int(round(EVENT_ENVELOPE_MS / dt_ms))))
    onset_i = int(round(float(macro_onset_ms) / dt_ms))
    half_state = 0.5 * STATE_ENVELOPE_MS
    candidates = []
    all_events = []
    for a, b in _episodes(event_env >= RECRUITED_HZ):
        t0, t1 = float(a * dt_ms), float(b * dt_ms)
        duration = t1 - t0
        rec = {
            "t_on": t0,
            "t_off": t1,
            "duration_ms": duration,
            "peak_50ms_rate_hz": float(np.max(event_env[a:b])),
        }
        # Keep only components that begin before strict recruited-state onset.  The long
        # recruited component may begin a few samples before that onset because the 50-ms and
        # 250-ms envelopes have different support; retaining it here is useful for the quiet-mask
        # audit, while components that begin after onset are not "pre-onset events".
        if t0 < float(macro_onset_ms):
            all_events.append(rec)
        if duration > EVENT_MAX_MS or t1 > float(macro_onset_ms) - EVENT_TO_ONSET_GUARD_MS:
            continue
        q0 = int(round((t1 + half_state) / dt_ms))
        q1 = int(round((t1 + half_state + EVENT_RETURN_MS) / dt_ms))
        q1 = min(q1, onset_i, state.size)
        if q1 <= q0:
            continue
        returned = bool(float(np.median(state[q0:q1])) < RECRUITED_HZ
                        and float(state[q1 - 1]) < RECRUITED_HZ)
        if returned:
            rec = dict(rec, return_check_start_ms=float(q0 * dt_ms),
                       return_check_end_ms=float(q1 * dt_ms),
                       return_state_median_hz=float(np.median(state[q0:q1])), returned=True)
            candidates.append(rec)
    if not candidates:
        raise RuntimeError("no machine-eligible returning event before recruited-state onset")
    candidates.sort(key=lambda e: (e["t_off"], e["peak_50ms_rate_hz"], -e["t_on"]))
    return candidates[-1], candidates, all_events, event_env


def pre_onset_quiet_mask(times_ms, events, macro_onset_ms):
    """Quiet samples used only to normalize the 30-80-Hz contact readout."""
    t = np.asarray(times_ms, float)
    quiet = t < float(macro_onset_ms) - EVENT_TO_ONSET_GUARD_MS
    for event in events:
        quiet &= ~((t >= event["t_on"] - EVENT_FILTER_GUARD_MS)
                   & (t <= event["t_off"] + EVENT_FILTER_GUARD_MS))
    return quiet


def bandpass_lfp(lfp_trace, times_ms, band=LFP_BAND_HZ, order=4):
    """Signed zero-phase 30-80-Hz virtual LFP and its analytic envelope."""
    lfp = np.asarray(lfp_trace, float)
    times = np.asarray(times_ms, float)
    if lfp.ndim != 2 or lfp.shape[0] != times.size or times.size < 2:
        raise ValueError("lfp_trace must be (n_time,n_contact) and align to times_ms")
    dt_ms = float(np.median(np.diff(times)))
    fs_hz = 1000.0 / dt_ms
    if not (0.0 < band[0] < band[1] < fs_hz / 2.0):
        raise ValueError(f"invalid LFP band {band} for fs={fs_hz}")
    sos = butter(int(order), tuple(float(v) for v in band), btype="bandpass", fs=fs_hz, output="sos")
    signed = sosfiltfilt(sos, lfp, axis=0)
    envelope = np.abs(hilbert(signed, axis=0))
    return signed, envelope


def _quiet_baseline(envelope, quiet):
    env = np.asarray(envelope, float)
    q = np.asarray(quiet, bool)
    if q.shape != (env.shape[0],) or int(q.sum()) < 2:
        raise ValueError("quiet mask must provide at least two aligned samples")
    sub = env[q]
    med = np.median(sub, axis=0)
    mad = 1.4826 * np.median(np.abs(sub - med[None, :]), axis=0)
    return med, mad


def per_neuron_first_spike_latency(E_spk_bool, event, dt_ms):
    spk = np.asarray(E_spk_bool, bool)
    s = max(0, int(round(float(event["t_on"]) / dt_ms)))
    e = min(spk.shape[0], int(round(float(event["t_off"]) / dt_ms)))
    out = np.full(spk.shape[1], np.nan, np.float32)
    if e <= s:
        return out
    seg = spk[s:e]
    active = seg.any(axis=0)
    idx = np.flatnonzero(active)
    out[idx] = (np.argmax(seg[:, idx], axis=0) * dt_ms).astype(np.float32)
    return out


def per_neuron_window_rate(E_spk_bool, t0_ms, t1_ms, dt_ms):
    spk = np.asarray(E_spk_bool, bool)
    s = max(0, int(round(float(t0_ms) / dt_ms)))
    e = min(spk.shape[0], int(round(float(t1_ms) / dt_ms)))
    if e <= s:
        raise ValueError("empty per-neuron rate window")
    duration_s = (e - s) * dt_ms * 1e-3
    return (spk[s:e].sum(axis=0) / duration_s).astype(np.float32)


def contact_window_energy(envelope, times_ms, quiet_med, t0_ms, t1_ms, valid_mask=None):
    """Mean squared positive envelope excess over the pre-onset quiet median."""
    env = np.asarray(envelope, float)
    t = np.asarray(times_ms, float)
    med = np.asarray(quiet_med, float)
    win = (t >= float(t0_ms)) & (t < float(t1_ms))
    if not np.any(win):
        raise ValueError("empty contact energy window")
    energy = np.mean(np.maximum(env[win] - med[None, :], 0.0) ** 2, axis=0)
    if valid_mask is not None:
        energy = np.where(np.asarray(valid_mask, bool), energy, np.nan)
    return energy.astype(np.float32)


def axial_space_time(E_spk_bool, posE, center, axis_unit, dt_ms, *, n_space=AXIAL_BINS,
                     time_bin_ms=AXIAL_TIME_BIN_MS):
    """Distinct-neuron active fraction in source->sink axial bins over time."""
    spk = np.asarray(E_spk_bool, bool)
    pos = np.asarray(posE, float)
    axis = np.asarray(axis_unit, float)
    axis = axis / np.linalg.norm(axis)
    coord = (pos - np.asarray(center, float)) @ axis
    lo, hi = float(coord.min()), float(coord.max())
    edges = np.linspace(lo, hi + np.finfo(float).eps, int(n_space) + 1)
    cell = np.clip(np.digitize(coord, edges[1:-1], right=False), 0, int(n_space) - 1)
    occupancy = np.bincount(cell, minlength=int(n_space)).astype(np.int32)
    safe = np.maximum(occupancy, 1)
    bs = max(1, int(round(float(time_bin_ms) / dt_ms)))
    n_frames = int(np.ceil(spk.shape[0] / bs))
    out = np.zeros((n_frames, int(n_space)), np.float32)
    for frame, b0 in enumerate(range(0, spk.shape[0], bs)):
        active = spk[b0:min(b0 + bs, spk.shape[0])].any(axis=0)
        counts = np.bincount(cell[active], minlength=int(n_space))
        out[frame] = counts / safe
    out[:, occupancy == 0] = np.nan
    frame_times = np.arange(n_frames, dtype=np.float32) * np.float32(bs * dt_ms)
    centers = (0.5 * (edges[:-1] + edges[1:])).astype(np.float32)
    return out, frame_times, centers, edges.astype(np.float32), occupancy


def movie_occupancy(posE, L, n=MOVIE_GRID):
    pos = np.asarray(posE, float)
    ix = np.clip((pos[:, 0] / float(L) * int(n)).astype(int), 0, int(n) - 1)
    iy = np.clip((pos[:, 1] / float(L) * int(n)).astype(int), 0, int(n) - 1)
    return np.bincount(iy * int(n) + ix, minlength=int(n) ** 2).reshape(int(n), int(n)).astype(np.int32)


def _meminfo_gib():
    values = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, value = line.split(":", 1)
            values[key] = float(value.strip().split()[0]) / 1024.0 / 1024.0
    return {
        "mem_total_gib": values["MemTotal"],
        "mem_available_gib": values["MemAvailable"],
        "swap_total_gib": values.get("SwapTotal", 0.0),
        "swap_free_gib": values.get("SwapFree", 0.0),
    }


def memory_gate(reserve_gib=MIN_RESERVE_GIB, worker_gib=DEFAULT_WORKER_GIB):
    reserve = float(reserve_gib)
    worker = float(worker_gib)
    if reserve < MIN_RESERVE_GIB:
        raise ValueError(f"reserve_gib must be >= {MIN_RESERVE_GIB:g}")
    if worker <= 0:
        raise ValueError("worker_gib must be > 0")
    mem = _meminfo_gib()
    predicted_after_launch = mem["mem_available_gib"] - worker
    audit = dict(**mem, reserve_gib=reserve, assumed_capture_peak_gib=worker,
                 predicted_available_after_launch_gib=predicted_after_launch)
    if predicted_after_launch < reserve:
        raise RuntimeError(f"memory gate refused capture: {audit}")
    return audit


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_info():
    sha = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True,
                         text=True, check=False).stdout.strip() or None
    status = subprocess.run(["git", "-C", ROOT, "status", "--short"], capture_output=True,
                            text=True, check=False).stdout.splitlines()
    return sha, status


def _atomic_json(payload, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_capture_", suffix=".json", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, allow_nan=False)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _atomic_npz(payload, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_capture_", suffix=".npz", dir=os.path.dirname(path))
    os.close(fd)
    try:
        np.savez_compressed(tmp, **payload)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


@contextlib.contextmanager
def _exclusive_output_lock():
    os.makedirs(RESULT_DIR, exist_ok=True)
    path = os.path.join(RESULT_DIR, ".current_stage_capture.lock")
    lock = open(path, "a+")
    try:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another figure capture holds {path}") from exc
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def _locked_cell_config(cfg):
    cell = LIFE._base_cfg(
        cfg,
        use_z=True,
        use_m=False,
        eta_m=0.0,
        use_SG=True,
        alpha_G=2.0,
        p_pool=1.0,
        use_TG=True,
        alpha_TG=4.0,
        tau_TG=750.0,
        tau_S=80.0,
    )
    expected = {
        "use_z": True, "use_m": False, "alpha_G": 2.0, "p_pool": 1.0,
        "use_TG": True, "alpha_TG": 4.0, "tau_TG": 750.0, "eta_m": 0.0,
    }
    for key, value in expected.items():
        if cell[key] != value:
            raise RuntimeError(f"locked cell drift: {key}={cell[key]!r}, expected {value!r}")
    return cell


def _expand_contact_timing(timing, valid, n_contacts):
    latency = np.full(n_contacts, np.nan, np.float32)
    rank = np.full(n_contacts, np.nan, np.float32)
    readable = np.zeros(n_contacts, bool)
    latency[valid] = np.asarray(timing.latency_ms, np.float32)
    rank[valid] = np.asarray(timing.rank, np.float32)
    readable[valid] = np.asarray(timing.readable, bool)
    return latency, rank, readable


def run_capture(*, reserve_gib=MIN_RESERVE_GIB, worker_gib=DEFAULT_WORKER_GIB, overwrite=False):
    if not overwrite and (os.path.exists(NPZ_PATH) or os.path.exists(JSON_PATH)):
        raise FileExistsError(f"capture artifact already exists; pass --overwrite to replace: {RESULT_DIR}")
    initial_memory = memory_gate(reserve_gib, worker_gib)
    cfg = LIFE._load_config(LIFE.DEFAULT_CONFIG)
    cfg["_config_path"] = LIFE.DEFAULT_CONFIG
    cell_cfg = _locked_cell_config(cfg)

    print(f"[capture] build E1146 substrate seed={SEED}", flush=True)
    S = PP.build_substrate(SEED)
    LIFE._prepare_flat_cache(S)
    pre_sim_memory = memory_gate(reserve_gib, worker_gib)
    montage = S["reg"]["montage_sheet"]
    contacts = np.asarray(montage.contacts, float)
    names = np.asarray([str(x) for x in montage.names], dtype="U")
    valid_contacts = CM.valid_mask(montage, S["posE"], S["L"], S["p"].Rr)
    recorder = LFPRecorder(S["p"], S["net"]["pos"], S["net"]["labels"], sites=contacts)
    slow = LIFE._slow_from_spec(S, {"cfg": cell_cfg})
    p = dataclasses.replace(S["p"], T=T_MS)
    S["net"]["rng"] = np.random.default_rng(SEED)

    print("[capture] start locked 20-s single-process simulation", flush=True)
    started = time.time()
    res = simulate_kick(
        p,
        S["net"],
        0.0,
        slow=slow,
        kick_center=list(S["src_xy"]),
        r_kick=PP.R_KICK,
        t_kick=1e9,
        V_th_per_neuron=S["vth"],
        early_stop_runaway=False,
        lfp_recorder=recorder,
    )
    sim_wall_s = time.time() - started
    rate = np.asarray(res["rate_E"], float)
    times = np.asarray(res["times"], float)
    spk = np.asarray(res["E_spk_bool"], bool)
    if rate.size != int(round(T_MS / DT_MS)) or spk.shape != (rate.size, S["NE"]):
        raise RuntimeError(f"incomplete capture: rate={rate.shape}, spikes={spk.shape}")

    macro, state_env = strict_recruited_episode(rate, DT_MS)
    if macro["status"] != "recruited_macrostate":
        raise RuntimeError(f"locked v2 phenotype drifted: {macro}")
    selected, returning, all_pre_events, event_env = select_pre_onset_returning_event(
        rate, DT_MS, macro["onset_ms"], state_env
    )

    signed_lfp, lfp_env = bandpass_lfp(res["lfp_trace"], times)
    quiet = pre_onset_quiet_mask(times, all_pre_events, macro["onset_ms"])
    quiet_med, quiet_mad = _quiet_baseline(lfp_env, quiet)
    contact_axis = (contacts - np.asarray(S["center"], float)) @ np.asarray(S["axis_unit"], float)
    valid_idx = np.flatnonzero(valid_contacts)
    timing = event_contact_timing(
        lfp_env[:, valid_contacts],
        times,
        selected,
        next_event_t_on=macro["onset_ms"],
        record_end_ms=float(times[-1]),
        quiet_med=quiet_med[valid_contacts],
        quiet_mad=quiet_mad[valid_contacts],
        contact_axis=contact_axis[valid_contacts],
        event_offset_ms=40.0,
        mad_k=5.0,
        rel_peak=0.10,
        min_readable=6,
        direction_abs=0.30,
        axis_src_to_snk_sign=+1,
    )
    contact_latency, contact_rank, contact_readable = _expand_contact_timing(
        timing, valid_idx, len(contacts)
    )
    event_neuron_latency = per_neuron_first_spike_latency(spk, selected, DT_MS)

    recruit_t0 = float(macro["onset_ms"])
    recruit_t1 = recruit_t0 + RECRUITED_WINDOW_MS
    if recruit_t1 > T_MS:
        raise RuntimeError("locked recruited window runs beyond record end")
    recruited_neuron_rate = per_neuron_window_rate(spk, recruit_t0, recruit_t1, DT_MS)
    recruited_contact_energy = contact_window_energy(
        lfp_env, times, quiet_med, recruit_t0, recruit_t1, valid_contacts
    )

    movie = M4._spatial_movie(spk, S["posE"], S["L"], DT_MS)
    expected_frames = int(np.ceil(rate.size / int(round(MOVIE_BIN_MS / DT_MS))))
    if movie.shape != (expected_frames, MOVIE_GRID, MOVIE_GRID):
        raise RuntimeError(f"unexpected 24x24 movie shape: {movie.shape}")
    movie_times = np.arange(movie.shape[0], dtype=np.float32) * np.float32(MOVIE_BIN_MS)
    axial, axial_times, axial_centers, axial_edges, axial_occupancy = axial_space_time(
        spk, S["posE"], S["center"], S["axis_unit"], DT_MS
    )

    # The LFP is filtered at the native 10-kHz sampling rate, then stride-decimated after filtering.
    lfp_stride = max(1, int(round(LFP_EXPORT_DT_MS / DT_MS)))
    slow_payload = {
        "slow_z_mean": np.asarray(slow.trace_z_mean, np.float32),
        "slow_z_min": np.asarray(slow.trace_z_min, np.float32),
        "slow_m_mean": np.asarray(slow.trace_m_mean, np.float32),
        "slow_adaptation_current": np.asarray(slow.trace_adap_current, np.float32),
        "slow_SG": np.asarray(slow.trace_SG, np.float32),
        "slow_AG": np.asarray(slow.trace_AG, np.float32),
        "slow_muG": np.asarray(slow.trace_muG, np.float32),
        "slow_TG": np.asarray(slow.trace_TG, np.float32),
        "slow_UTG": np.asarray(slow.trace_UTG, np.float32),
        "slow_rEfast_max": np.asarray(slow.trace_rEfast_max, np.float32),
    }
    for key, values in slow_payload.items():
        if values.shape != rate.shape:
            raise RuntimeError(f"unaligned slow trace {key}: {values.shape} vs {rate.shape}")

    payload = {
        "times_ms": times.astype(np.float32),
        "rate_E_hz": rate.astype(np.float32),
        "rate_I_hz": np.asarray(res["rate_I"], np.float32),
        "rate_state_envelope_250ms_hz": state_env.astype(np.float32),
        "rate_event_envelope_50ms_hz": event_env.astype(np.float32),
        **slow_payload,
        "slow_z_final_E": np.asarray(slow.z[: S["NE"]], np.float32),
        "slow_m_final_E": np.asarray(slow.m[: S["NE"]], np.float32),
        "lfp_times_ms": times[::lfp_stride].astype(np.float32),
        "lfp_gamma_30_80": signed_lfp[::lfp_stride].astype(np.float32),
        "lfp_gamma_envelope": lfp_env[::lfp_stride].astype(np.float32),
        "lfp_quiet_median": quiet_med.astype(np.float32),
        "lfp_quiet_mad": quiet_mad.astype(np.float32),
        "posE": np.asarray(S["posE"], np.float32),
        "contacts": contacts.astype(np.float32),
        "contact_names": names,
        "valid_contacts": np.asarray(valid_contacts, bool),
        "contact_axis_mm": contact_axis.astype(np.float32),
        "src_xy": np.asarray(S["src_xy"], np.float32),
        "snk_xy": np.asarray(S["snk_xy"], np.float32),
        "center_xy": np.asarray(S["center"], np.float32),
        "axis_unit": np.asarray(S["axis_unit"], np.float32),
        "pre_event_first_spike_latency_ms": event_neuron_latency,
        "pre_event_contact_latency_ms": contact_latency,
        "pre_event_contact_rank": contact_rank,
        "pre_event_contact_readable": contact_readable,
        "recruited_neuron_rate_hz": recruited_neuron_rate,
        "recruited_contact_energy": recruited_contact_energy,
        "movie_active_fraction": np.asarray(movie, np.float32),
        "movie_times_ms": movie_times,
        "movie_occupancy": movie_occupancy(S["posE"], S["L"]),
        "axial_active_fraction": axial,
        "axial_times_ms": axial_times,
        "axial_centers_mm": axial_centers,
        "axial_edges_mm": axial_edges,
        "axial_occupancy": axial_occupancy,
    }
    forbidden = {key for key in payload if "spk_bool" in key.lower() or "spike_raster" in key.lower()}
    if forbidden:
        raise AssertionError(f"full spike raster must not be serialized: {sorted(forbidden)}")

    git_sha, git_status = _git_info()
    metadata = {
        "schema_version": "topic4_mz_divisive_current_stage_capture_v1",
        "scientific_status": "finite_window_bounded_bursting_opening_not_recovered_lifecycle",
        "simulation": {
            "subject": PP.SUBJECT,
            "montage": PP.MONTAGE,
            "seed": SEED,
            "T_ms": T_MS,
            "dt_ms": DT_MS,
            "spontaneous_no_kick": True,
            "early_stop_runaway": False,
            "cell_config": cell_cfg,
            "wall_s": round(sim_wall_s, 3),
            "peak_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0, 3),
        },
        "recruited_state": macro,
        "representative_pre_onset_event": selected,
        "representative_event_selection": {
            "rule": "latest machine-eligible returned event; no visual selection",
            "candidate_event_envelope_ms": EVENT_ENVELOPE_MS,
            "candidate_threshold_hz": RECRUITED_HZ,
            "candidate_max_duration_ms": EVENT_MAX_MS,
            "minimum_gap_to_recruited_onset_ms": EVENT_TO_ONSET_GUARD_MS,
            "return_rule": ("after the 125-ms centered-envelope half-width, the next 250-ms "
                            "state-envelope median and final sample are both below 20 Hz"),
            "n_returning_candidates": len(returning),
            "n_all_pre_onset_threshold_components": len(all_pre_events),
        },
        "strict_onset_contract": {
            "state_envelope_ms": STATE_ENVELOPE_MS,
            "threshold_hz": RECRUITED_HZ,
            "minimum_contiguous_duration_ms": MIN_RECRUITED_MS,
            "gap_merge_ms": 0.0,
        },
        "readout_contract": {
            "lfp_definition": "existing LFPRecorder weighted |I_E|+|I_I| on E neurons",
            "lfp_band_hz": list(LFP_BAND_HZ),
            "lfp_filter": "4th-order Butterworth SOS, zero-phase sosfiltfilt at native 10 kHz",
            "lfp_export_dt_ms": LFP_EXPORT_DT_MS,
            "contact_timing": ("30-80-Hz envelope peak latency; readable if > quiet median + 5*MAD "
                               "and >=10% of event max excess; ranks restricted to valid contacts"),
            "n_valid_contacts": int(np.sum(valid_contacts)),
            "n_contacts": int(len(valid_contacts)),
            "n_readable_pre_event_contacts": int(timing.n_readable),
            "pre_event_direction": timing.direction,
            "pre_event_axis_spearman": None if not np.isfinite(timing.axis_spearman) else float(timing.axis_spearman),
            "pre_event_contact_timing_eligible": bool(timing.eligible),
            "recruited_window_ms": [recruit_t0, recruit_t1],
            "recruited_neuron_value": "per-neuron spike count / exact 1-s window (Hz)",
            "recruited_contact_value": "mean squared positive 30-80-Hz envelope excess over pre-onset quiet median",
            "movie": {"grid": [MOVIE_GRID, MOVIE_GRID], "frame_ms": MOVIE_BIN_MS,
                      "value": "fraction of neurons with >=1 spike per occupied cell"},
            "axial_space_time": {"n_bins": AXIAL_BINS, "frame_ms": AXIAL_TIME_BIN_MS,
                                  "value": "fraction of neurons with >=1 spike per source-to-sink axial bin"},
        },
        "resource_gate": {"initial": initial_memory, "pre_simulation": pre_sim_memory,
                          "blas_threads": {name: os.environ.get(name) for name in
                                           ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
                          "processes": 1},
        "artifact_contract": {
            "npz_path": os.path.relpath(NPZ_PATH, ROOT),
            "full_E_spk_bool_saved": False,
            "arrays": {key: {"shape": list(np.asarray(value).shape), "dtype": str(np.asarray(value).dtype)}
                       for key, value in payload.items()},
        },
        "provenance": {
            "git_sha": git_sha,
            "git_status_short": git_status,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "utc": dt_datetime.datetime.now(dt_datetime.timezone.utc).isoformat(),
            "argv": sys.argv,
            "config_path": os.path.relpath(LIFE.DEFAULT_CONFIG, ROOT),
            "source_files": {
                "capture": os.path.relpath(__file__, ROOT),
                "runner": os.path.relpath(LIFE.__file__, ROOT),
                "substrate": os.path.relpath(PP.__file__, ROOT),
            },
        },
    }

    _atomic_npz(payload, NPZ_PATH)
    metadata["artifact_contract"]["npz_sha256"] = _sha256(NPZ_PATH)
    metadata["artifact_contract"]["npz_bytes"] = os.path.getsize(NPZ_PATH)
    _atomic_json(metadata, JSON_PATH)
    # Release the only full-raster reference before reporting success.
    del spk, res
    print(f"[capture] wrote {NPZ_PATH}", flush=True)
    print(f"[capture] wrote {JSON_PATH}", flush=True)
    return metadata


def build_parser():
    parser = argparse.ArgumentParser(description="Locked v2 MZ-divisive spatial figure capture")
    parser.add_argument("--confirm-run", action="store_true",
                        help="required: launches one full E1146 20-s simulation")
    parser.add_argument("--reserve-gib", type=float, default=MIN_RESERVE_GIB,
                        help="minimum predicted available RAM after launch; cannot be below 96 GiB")
    parser.add_argument("--worker-gib", type=float, default=DEFAULT_WORKER_GIB,
                        help="conservative assumed peak RSS for the single capture process")
    parser.add_argument("--overwrite", action="store_true",
                        help="explicitly replace an existing current_stage_capture artifact")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not args.confirm_run:
        print("REFUSING: this command launches a 20-s E1146 simulation; pass --confirm-run.",
              file=sys.stderr)
        return 2
    with _exclusive_output_lock():
        run_capture(reserve_gib=args.reserve_gib, worker_gib=args.worker_gib,
                    overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
