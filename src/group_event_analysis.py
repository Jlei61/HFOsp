"""
Module 3: Group Event Analysis (Yuquan 24h SEEG)

Goal
----
Assemble per-channel HFO detections into aligned group-event windows (typically 500ms),
then compute per-channel timing within each window:
- absolute timing: centroid time relative to window start (lagPatRaw-like)
- relative ordering: rank within the window (lagPatRank-like)

Design principles
-----------------
- No guessing: caller provides detections and/or packedTimes windows explicitly.
- Streaming-friendly: on-demand waveform loading via callback (no full-record preload required).
- Backwards compatibility: this module is additive and does not change existing APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .preprocessing import ElectrodeParser
from .utils import bqk_utils
from .utils.logging_utils import get_run_logger, log_section


Seconds = float


@dataclass(frozen=True)
class SingleEvent:
    """Single-channel detection in seconds (relative to record start)."""

    start: Seconds
    end: Seconds
    ch_name: str


@dataclass(frozen=True)
class EventWindow:
    """
    An aligned group-event window [start, end] in seconds (relative to record start).

    Note: Yuquan packedTimes windows are fixed-length (typically 0.5s).
    """

    start: Seconds
    end: Seconds
    event_id: int

    @property
    def duration(self) -> float:
        return float(self.end - self.start)


@dataclass(frozen=True)
class LagMatrices:
    """
    Output compatible with Yuquan lagPat-like representation.

    lag_raw:
        (n_channels, n_events) float seconds, relative to window start.
        Non-participating channels are NaN.
    events_bool:
        (n_channels, n_events) bool, True if channel has a detection overlapping the window.
    lag_rank:
        (n_channels, n_events) int, dense rank within each event among participating channels.
        Convention: 0 = earliest (source-like), -1 = non-participating.
    """

    ch_names: List[str]
    windows: List[EventWindow]
    lag_raw: np.ndarray
    events_bool: np.ndarray
    lag_rank: np.ndarray


def _interval_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    """Return overlap length between [a0,a1] and [b0,b1] (seconds)."""
    left = max(a0, b0)
    right = min(a1, b1)
    return max(0.0, right - left)


def _ensure_2col_times(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(f"Expected shape (n, 2) times array, got {x.shape}")
    return x.astype(np.float64, copy=False)


def _moving_mean_1d(x: np.ndarray, win: int) -> np.ndarray:
    """Fast moving mean for 1D arrays."""
    if win <= 1:
        return x.astype(np.float64, copy=False)
    x = x.astype(np.float64, copy=False)
    c = np.cumsum(x)
    out = c[win - 1 :] - np.concatenate(([0.0], c[:-win]))
    return out / float(win)


def _moving_sum_1d(x: np.ndarray, win: int) -> np.ndarray:
    """Fast moving sum for 1D arrays."""
    if win <= 1:
        return x.astype(np.float64, copy=False)
    x = x.astype(np.float64, copy=False)
    c = np.cumsum(x)
    out = c[win - 1 :] - np.concatenate(([0.0], c[:-win]))
    return out


def _select_nearest_pool_indices(
    pool_starts: np.ndarray,
    event_time: float,
    *,
    n_select: int = 10,
    min_distance_sec: float = 2.0,
) -> np.ndarray:
    pool_starts = np.asarray(pool_starts, dtype=np.float64)
    if pool_starts.size == 0:
        return np.zeros((0,), dtype=np.int64)
    dist = np.abs(pool_starts - float(event_time))
    mask = dist >= float(min_distance_sec)
    if not np.any(mask):
        return np.zeros((0,), dtype=np.int64)
    idx = np.where(mask)[0]
    order = np.argsort(dist[idx])
    keep = idx[order[: int(n_select)]]
    return keep.astype(np.int64, copy=False)


def compute_baseline_pool_indices(
    *,
    data: np.ndarray,
    env: np.ndarray,
    sfreq: float,
    detections: Mapping[str, np.ndarray],
    ictal_mask: Optional[np.ndarray],
    window_sec: float = 2.0,
    step_sec: float = 1.0,
    line_length_q: float = 0.90,
    ripple_env_q: float = 0.80,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Build baseline pool using line length + ripple envelope, excluding ictal/HFO windows.
    Returns (pool_starts_sec, pool_start_indices, params).
    """
    data = np.asarray(data, dtype=np.float64)
    env = np.asarray(env, dtype=np.float64)
    sfreq = float(sfreq)
    if data.ndim != 2 or env.ndim != 2:
        raise ValueError("data/env must be 2D (n_channels, n_samples)")
    n_samples = data.shape[1]
    win = max(1, int(round(float(window_sec) * sfreq)))
    step = max(1, int(round(float(step_sec) * sfreq)))
    if n_samples < win:
        return (
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.int64),
            {
                "window_sec": float(window_sec),
                "step_sec": float(step_sec),
                "line_length_q": float(line_length_q),
                "ripple_env_q": float(ripple_env_q),
                "exclude_ictal": bool(ictal_mask is not None),
                "exclude_hfo": True,
            },
        )

    starts = np.arange(0, n_samples - win + 1, step, dtype=np.int64)

    # Line length on channel-averaged signal
    x_mean = np.mean(data, axis=0)
    d = np.abs(np.diff(x_mean))
    ll = _moving_sum_1d(d, max(1, win - 1))
    ll_vals = ll[starts]

    # Ripple envelope metric: median across channels, then windowed mean
    env_med = np.median(env, axis=0)
    env_mean = _moving_mean_1d(env_med, win)
    env_vals = env_mean[starts]

    valid = np.ones((starts.shape[0],), dtype=bool)

    # Exclude ictal windows
    if ictal_mask is not None and ictal_mask.size == n_samples:
        ictal = ictal_mask.astype(np.int64)
        prefix = np.concatenate(([0], np.cumsum(ictal)))
        win_hits = prefix[starts + win] - prefix[starts]
        valid &= (win_hits == 0)

    # Exclude windows overlapping any HFO detection (across all channels)
    if detections:
        hfo_bad = np.zeros_like(valid)
        for times in detections.values():
            if times is None:
                continue
            arr = np.asarray(times)
            if arr.size == 0:
                continue
            arr = _ensure_2col_times(arr)
            for s, e in arr:
                w0 = int(np.ceil((float(s) - float(window_sec)) / float(step_sec)))
                w1 = int(np.floor(float(e) / float(step_sec)))
                w0 = max(0, w0)
                w1 = min(len(starts) - 1, w1)
                if w1 >= w0:
                    hfo_bad[w0 : w1 + 1] = True
        valid &= ~hfo_bad

    # Threshold by quantiles on remaining pool
    pre_thresh_valid = valid.copy()
    if np.any(valid):
        ll_thr = float(np.quantile(ll_vals[valid], float(line_length_q)))
        env_thr = float(np.quantile(env_vals[valid], float(ripple_env_q)))
        valid &= (ll_vals <= ll_thr) & (env_vals <= env_thr)
        if not np.any(valid):
            # Fallback: keep pre-threshold pool if thresholds wipe out all windows
            valid = pre_thresh_valid

    pool_starts = starts[valid].astype(np.float64) / sfreq
    pool_indices = starts[valid].astype(np.int64)
    params = {
        "window_sec": float(window_sec),
        "step_sec": float(step_sec),
        "line_length_q": float(line_length_q),
        "ripple_env_q": float(ripple_env_q),
        "exclude_ictal": bool(ictal_mask is not None),
        "exclude_hfo": True,
    }
    return pool_starts, pool_indices, params

def build_windows_from_packed_times(packed_times: np.ndarray) -> List[EventWindow]:
    """
    Convert Yuquan packedTimes.npy (n_events, 2) -> List[EventWindow].
    """
    t = _ensure_2col_times(packed_times)
    return [EventWindow(float(s), float(e), int(i)) for i, (s, e) in enumerate(t)]


def build_windows_from_detections(
    detections: Mapping[str, np.ndarray],
    window_sec: float = 0.5,
    *,
    strategy: str = "fixed_from_earliest_start",
) -> List[EventWindow]:
    """
    Build fixed-length windows from per-channel detections.

    This is a pragmatic approximation when packedTimes is unavailable.
    For validation against Yuquan lagPat, prefer using packedTimes directly.

    Strategy
    --------
    fixed_from_earliest_start:
        Take earliest unassigned event start as window start; window is [start, start+window_sec).
        Assign all events whose start lies in the window, mark them assigned, repeat.
    """
    if window_sec <= 0:
        raise ValueError("window_sec must be > 0")
    if strategy != "fixed_from_earliest_start":
        raise ValueError(f"Unknown strategy='{strategy}'")

    flat: List[SingleEvent] = []
    for ch, times in detections.items():
        if times is None:
            continue
        arr = np.asarray(times)
        if arr.size == 0:
            continue
        arr = _ensure_2col_times(arr)
        for s, e in arr:
            flat.append(SingleEvent(float(s), float(e), str(ch)))

    flat.sort(key=lambda ev: ev.start)
    if not flat:
        return []

    assigned = np.zeros(len(flat), dtype=bool)
    windows: List[EventWindow] = []
    i = 0
    while True:
        while i < len(flat) and assigned[i]:
            i += 1
        if i >= len(flat):
            break

        w_start = flat[i].start
        w_end = w_start + float(window_sec)
        # Mark all events starting within window as assigned.
        for j in range(i, len(flat)):
            if assigned[j]:
                continue
            if flat[j].start < w_end:
                assigned[j] = True
            else:
                break

        windows.append(EventWindow(float(w_start), float(w_end), int(len(windows))))

    return windows


def compare_window_sets(
    a: Sequence[EventWindow],
    b: Sequence[EventWindow],
    *,
    min_overlap_sec: float = 0.25,
) -> Dict[str, float]:
    """
    Compare two sorted window lists by interval overlap.

    Matching rule:
      a window matches b window if overlap(a,b) >= min_overlap_sec.
    We do greedy two-pointer matching (sufficient because windows are fixed-length and sorted).

    Returns metrics:
      - n_a, n_b
      - n_match
      - precision (= match / n_a)
      - recall (= match / n_b)
      - mean_abs_start_diff_s for matched pairs
    """
    a = list(a)
    b = list(b)
    i = 0
    j = 0
    matches: List[Tuple[int, int]] = []

    while i < len(a) and j < len(b):
        wa = a[i]
        wb = b[j]
        ov = _interval_overlap(wa.start, wa.end, wb.start, wb.end)
        if ov >= min_overlap_sec:
            matches.append((i, j))
            i += 1
            j += 1
            continue
        # advance pointer of the earlier-ending window
        if wa.end <= wb.end:
            i += 1
        else:
            j += 1

    n_match = len(matches)
    start_diffs = [abs(a[i].start - b[j].start) for i, j in matches]
    mean_abs_start_diff = float(np.mean(start_diffs)) if start_diffs else float("nan")

    n_a = float(len(a))
    n_b = float(len(b))
    precision = float(n_match) / n_a if n_a > 0 else 0.0
    recall = float(n_match) / n_b if n_b > 0 else 0.0

    return {
        "n_a": n_a,
        "n_b": n_b,
        "n_match": float(n_match),
        "precision": precision,
        "recall": recall,
        "mean_abs_start_diff_s": mean_abs_start_diff,
    }


def filter_windows_by_min_channels(
    windows: Sequence[EventWindow],
    detections: Mapping[str, np.ndarray],
    *,
    min_channels: int,
) -> List[EventWindow]:
    """
    Keep only windows where at least `min_channels` distinct channels have an event overlapping it.
    """
    if min_channels <= 1:
        return list(windows)
    out: List[EventWindow] = []
    ch_list = list(detections.keys())
    for w in windows:
        cnt = 0
        for ch in ch_list:
            if _overlaps_any_event(detections.get(ch, None), w):
                cnt += 1
                if cnt >= min_channels:
                    out.append(w)
                    break
    return out


def select_core_channels_by_event_count(
    *,
    events_count: np.ndarray,
    ch_names: Sequence[str],
    method: str = "log_mean_std",
    k: float = 2.0,
    top_n: Optional[int] = None,
    min_count: int = 1,
) -> List[str]:
    """
    Heuristic "core channel" selection based on per-channel event counts.

    Common patterns seen in Yuquan:
      - use log scale because counts are heavy-tailed
      - threshold by mean + k*std (or median + k*MAD)
      - then optionally cap to top_n

    method:
      - 'mean_std'           : threshold on raw counts
      - 'log_mean_std'       : threshold on log1p(count)
      - 'log_median_mad'     : threshold on log1p(count) using MAD
      - 'top_n'              : take top_n by raw count
      - 'log_top_n'          : take top_n by log1p(count)
    """
    counts = np.asarray(events_count, dtype=np.float64)
    names = [str(x) for x in ch_names]
    if counts.shape[0] != len(names):
        raise ValueError("events_count length must match ch_names length")

    # mask out dead channels
    alive = counts >= float(min_count)
    if not np.any(alive):
        return []

    if method in ("top_n", "log_top_n"):
        if top_n is None:
            raise ValueError("top_n is required for method='top_n'/'log_top_n'")
        score = np.log1p(counts) if method == "log_top_n" else counts
        idx = np.argsort(score)[::-1]
        idx = [i for i in idx if alive[i]][: int(top_n)]
        return [names[i] for i in idx]

    if method == "mean_std":
        score = counts
        mu = float(np.mean(score[alive]))
        sd = float(np.std(score[alive], ddof=0))
        thr = mu + float(k) * sd
        sel_idx = [i for i in range(len(names)) if alive[i] and score[i] > thr]
    elif method == "log_mean_std":
        score = np.log1p(counts)
        mu = float(np.mean(score[alive]))
        sd = float(np.std(score[alive], ddof=0))
        thr = mu + float(k) * sd
        sel_idx = [i for i in range(len(names)) if alive[i] and score[i] > thr]
    elif method == "log_median_mad":
        score = np.log1p(counts)
        med = float(np.median(score[alive]))
        mad = float(np.median(np.abs(score[alive] - med)))
        thr = med + float(k) * mad
        sel_idx = [i for i in range(len(names)) if alive[i] and score[i] > thr]
    else:
        raise ValueError(f"Unknown method='{method}'")

    if top_n is not None and len(sel_idx) > int(top_n):
        # cap by raw count (stable and interpretable)
        sel_idx = sorted(sel_idx, key=lambda i: counts[i], reverse=True)[: int(top_n)]

    return [names[i] for i in sel_idx]


def compare_channel_sets(a: Sequence[str], b: Sequence[str]) -> Dict[str, float]:
    """
    Compare two channel sets and return simple overlap metrics.
    """
    sa = set(str(x) for x in a)
    sb = set(str(x) for x in b)
    inter = sa & sb
    union = sa | sb
    return {
        "n_a": float(len(sa)),
        "n_b": float(len(sb)),
        "n_intersection": float(len(inter)),
        "jaccard": float(len(inter) / len(union)) if union else 1.0,
    }


def _overlaps_any_event(times: np.ndarray, win: EventWindow) -> bool:
    """
    Return True if any detection [s,e] overlaps [win.start, win.end).
    """
    if times is None:
        return False
    arr = np.asarray(times)
    if arr.size == 0:
        return False
    arr = _ensure_2col_times(arr)
    s = arr[:, 0]
    e = arr[:, 1]
    return bool(np.any((e > win.start) & (s < win.end)))


def _compute_energy_centroid(
    env: np.ndarray,
    t: np.ndarray,
    *,
    power: float = 2.0,
    eps: float = 1e-12,
) -> float:
    """
    Centroid = sum(t * env^power) / sum(env^power).
    """
    env = np.asarray(env, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    w = np.power(np.maximum(env, 0.0), power)
    denom = float(np.sum(w))
    if denom <= eps:
        return float(np.mean(t))
    return float(np.sum(t * w) / denom)


def _gpu_hilbert_envelope_filterbank(
    x: np.ndarray,
    sfreq: float,
    freqband: Tuple[float, float],
    *,
    subband_width_hz: float = 20.0,
) -> np.ndarray:
    """
    GPU version of bqk_utils.return_hil_enve_norm:
      - split freqband into <=20Hz sub-bands
      - bandpass + Hilbert envelope per sub-band
      - sum envelopes across sub-bands

    Returns numpy array on CPU (to keep downstream code simple).
    """
    import cupy as cp
    import cupyx.scipy.signal as cs

    x_gpu = cp.asarray(x, dtype=cp.float32)
    fs = float(sfreq)
    nyq = fs / 2.0
    lo, hi = float(freqband[0]), float(freqband[1])
    if hi >= nyq:
        hi = max(nyq - 1.0, lo + 1.0)

    if (hi - lo) <= float(subband_width_hz):
        bands = [(lo, hi)]
    else:
        edges = np.arange(lo, hi, float(subband_width_hz), dtype=np.float64)
        edges = np.append(edges, hi)
        bands = list(zip(edges[:-1].tolist(), edges[1:].tolist()))

    out = cp.zeros_like(x_gpu, dtype=cp.float32)
    for b0, b1 in bands:
        sos = cs.butter(3, [b0 / nyq, b1 / nyq], btype="bandpass", output="sos")
        xf = cs.sosfiltfilt(sos, x_gpu)
        env = cp.abs(cs.hilbert(xf))
        out += env.astype(cp.float32, copy=False)

    return cp.asnumpy(out)


def _maybe_gpu_envelope(
    x: np.ndarray,
    sfreq: float,
    freqband: Tuple[float, float],
    *,
    use_gpu: bool,
) -> np.ndarray:
    if not use_gpu:
        return bqk_utils.return_hil_enve_norm(x, sfreq, freqband)
    try:
        return _gpu_hilbert_envelope_filterbank(x, sfreq, freqband)
    except Exception:
        # Fail closed: keep correctness; callers can inspect timing instead.
        return bqk_utils.return_hil_enve_norm(x, sfreq, freqband)


def precompute_envelope_cache(
    *,
    edf_path: str,
    out_npz_path: str,
    band: str = "ripple",
    crop_seconds: float = 120.0,
    reference: str = "bipolar",
    include_channels: Optional[List[str]] = None,
    alias_bipolar_to_left: bool = True,
    alias_filter_using_gpu_npz: Optional[str] = None,
    use_gpu: bool = True,
    dtype: str = "float32",
    save_bandpass: bool = False,
) -> str:
    """
    Precompute Hilbert envelope (bqk-style 20Hz filterbank sum) for an entire crop and save to disk.

    This is the recommended acceleration path: compute envelope once per channel for the crop,
    then centroid/lag/rank only need fast slicing per window.
    """
    from .preprocessing import SEEGPreprocessor

    pre = SEEGPreprocessor(
        target_band="fast_ripple" if band in ("fast_ripple", "fast-ripple", "fr", "fast") else "ripple",
        reference=reference,
        include_channels=include_channels,
        crop_seconds=float(crop_seconds),
        check_quality=False,
        use_gpu=False,
    )
    pre_out = pre.run(edf_path)

    data = np.asarray(pre_out.data)
    names = [str(x) for x in pre_out.ch_names]
    sfreq = float(pre_out.sfreq)

    # Optionally alias bipolar pairs to left contact name ('A1-A2' -> 'A1')
    if alias_bipolar_to_left:
        allowed = None
        if alias_filter_using_gpu_npz is not None:
            g = np.load(alias_filter_using_gpu_npz, allow_pickle=True)
            allowed = set(str(x).upper() for x in g["chns_names"].tolist())

        keep_idx: List[int] = []
        alias_names: List[str] = []
        for i, nm in enumerate(names):
            if "-" not in nm:
                continue
            left, right = nm.split("-", 1)
            left = left.strip().upper()
            right = right.strip().upper()
            if allowed is not None and (left not in allowed or right not in allowed):
                continue
            keep_idx.append(i)
            alias_names.append(left)

        data = data[keep_idx] if keep_idx else np.zeros((0, data.shape[1]), dtype=np.float64)
        names = alias_names

    b = band.lower().strip()
    if b in ("ripple", "ripples"):
        fb = (80.0, 250.0)
    elif b in ("fast_ripple", "fast-ripple", "fr", "fast"):
        fb = (250.0, 500.0)
    else:
        raise ValueError(f"Unknown band='{band}'. Use 'ripple' or 'fast_ripple'.")

    # Optional: also compute broad bandpass (for visualization figure 1)
    x_band = None
    if bool(save_bandpass):
        try:
            import cupy as cp
            import cupyx.scipy.signal as cs

            nyq = sfreq / 2.0
            lo, hi = fb
            hi2 = hi if hi < nyq else max(nyq - 1.0, lo + 1.0)
            sos = cs.butter(4, [float(lo) / nyq, float(hi2) / nyq], btype="bandpass", output="sos")

            x_band = np.zeros_like(data, dtype=np.float32)
            for ci in range(data.shape[0]):
                xg = cp.asarray(data[ci], dtype=cp.float32)
                xf = cs.sosfiltfilt(sos, xg)
                x_band[ci] = cp.asnumpy(xf)
        except Exception:
            # If GPU bandpass fails, keep going (envelope cache still useful)
            x_band = None

    env = np.zeros_like(data, dtype=np.float32)
    for ci in range(data.shape[0]):
        env[ci] = _maybe_gpu_envelope(data[ci], sfreq, fb, use_gpu=bool(use_gpu)).astype(np.float32, copy=False)

    if dtype == "float16":
        env = env.astype(np.float16)
    elif dtype != "float32":
        raise ValueError("dtype must be 'float32' or 'float16'")

    np.savez_compressed(
        out_npz_path,
        env=env,
        x_band=x_band if x_band is not None else np.zeros((0, 0), dtype=np.float32),
        sfreq=np.array([sfreq], dtype=np.float64),
        ch_names=np.array(names, dtype=object),
        band=np.array([band], dtype=object),
        reference=np.array([reference], dtype=object),
        crop_seconds=np.array([float(crop_seconds)], dtype=np.float64),
        start_sec=np.array([0.0], dtype=np.float64),
        alias_bipolar_to_left=np.array([bool(alias_bipolar_to_left)], dtype=bool),
        has_x_band=np.array([x_band is not None], dtype=bool),
    )
    return out_npz_path


def load_envelope_cache(npz_path: str) -> Dict:
    """Load envelope cache saved by precompute_envelope_cache()."""
    d = np.load(npz_path, allow_pickle=True)
    xb = np.asarray(d["x_band"])
    has_xb = bool(np.asarray(d.get("has_x_band", np.array([False]))).ravel()[0])
    return {
        "env": np.asarray(d["env"]),
        "x_band": xb if (has_xb and xb.size > 0) else None,
        "sfreq": float(np.asarray(d["sfreq"]).ravel()[0]),
        "ch_names": [str(x) for x in np.asarray(d["ch_names"]).tolist()],
        "band": str(np.asarray(d["band"]).ravel()[0]),
        "reference": str(np.asarray(d["reference"]).ravel()[0]),
        "crop_seconds": float(np.asarray(d["crop_seconds"]).ravel()[0]),
        "start_sec": float(np.asarray(d["start_sec"]).ravel()[0]),
        "alias_bipolar_to_left": bool(np.asarray(d["alias_bipolar_to_left"]).ravel()[0]),
    }


def _safe_filename_component(s: str) -> str:
    """Make a string safe for filenames (ASCII-ish, no spaces/slashes)."""
    s = str(s)
    out = []
    for c in s:
        if c.isalnum() or c in ("-", "_", "."):
            out.append(c)
        else:
            out.append("_")
    # Collapse long runs of underscores
    ss = "".join(out)
    while "__" in ss:
        ss = ss.replace("__", "_")
    return ss.strip("_.") or "x"


def compute_and_save_hfo_event_wavelet_tfr(
    *,
    raw_cache_npz_path: str,
    target_channel: str,
    event_start: float,
    event_end: float,
    margin_sec: float = 0.2,
    band: Optional[str] = None,
    fmin: float = 10.0,
    fmax: float = 400.0,
    n_freqs: int = 180,
    n_cycles: float = 4.0,
    n_cycles_mode: str = "linear",  # 'fixed'|'linear'|'f_over_2'
    n_cycles_min: float = 3.0,
    n_cycles_max: float = 10.0,
    decim: int = 1,
    baseline_mode: str = "robust_z",  # 'none'|'robust_z'|'event_pctl'
    baseline_pctl_low: float = 5.0,
    baseline_pctl_high: float = 95.0,
    smooth_time_ms: float = 8.0,
    smooth_freq_bins: float = 1.5,
    output_npz_path: Optional[str] = None,
    overwrite: bool = False,
    baseline_pool_starts: Optional[np.ndarray] = None,
    baseline_pool_indices: Optional[np.ndarray] = None,
    baseline_window_sec: float = 2.0,
    baseline_n_select: int = 10,
    baseline_min_distance_sec: float = 2.0,
) -> str:
    """
    Precompute and save a single-event Morlet-wavelet TFR for later visualization.

    Design principle
    ----------------
    - This function does the computation and writes an on-disk artifact (npz).
    - `src/visualization.py` should only read the saved results and plot.

    Parameters
    ----------
    raw_cache_npz_path : str
        Path to *_rawCache_*.npz created by preprocessing.save_raw_cache().
    target_channel : str
        Channel name in raw cache.
    event_start, event_end : float
        Event window in seconds (absolute time w.r.t. the cached crop start).
    margin_sec : float
        Context window before/after event for plotting and TF computation.
    band : str, optional
        Convenience band selector: 'ripple' (80-250Hz) or 'fast_ripple' (250-500Hz).
        If provided, overrides fmin/fmax and is stored in metadata.
    fmin, fmax : float
        Frequency range (Hz) for wavelet transform.
    n_freqs : int
        Number of frequencies between [fmin, fmax] (log-spaced).
    n_cycles : float
        Morlet cycles (tradeoff between time/frequency resolution).
        For HFOs, smaller values (3-6) usually highlight short bursts better.
    n_cycles_mode : str
        How to set cycles across frequencies:
        - 'fixed'    : use n_cycles for all freqs
        - 'linear'   : linearly spaced from n_cycles_min to n_cycles_max
        - 'f_over_2' : n_cycles = f / 2 (f in Hz)
    n_cycles_min, n_cycles_max : float
        Range for 'linear' mode.
    decim : int
        Decimation factor on time axis for TF output (>=1).
    baseline_mode : str
        Baseline normalization stored in cache for better contrast.
        - 'none'        : only save power_db
        - 'robust_z'    : per-frequency robust z-score using baseline (outside [event_start,event_end])
        - 'event_pctl'  : per-frequency percentile scaling within the event window (paper-like contrast)
        - 'dynamic_pool': median baseline from nearest clean pool windows
    baseline_pctl_low, baseline_pctl_high : float
        Percentile range used by 'event_pctl' scaling.
    smooth_time_ms : float
        Gaussian smoothing (ms) applied to power_db for visualization ("blob" look).
    smooth_freq_bins : float
        Gaussian smoothing (bins) along frequency axis.
    output_npz_path : str, optional
        Output path. If None, saves next to raw cache with a deterministic name.
    overwrite : bool
        If False and output exists, returns the existing path without recomputing.

    Returns
    -------
    out_npz_path : str
        Saved artifact path.
    """
    from pathlib import Path
    from .preprocessing import load_raw_cache

    try:
        from scipy.signal import fftconvolve
    except Exception as exc:
        raise ImportError("SciPy is required for wavelet TFR (scipy.signal.fftconvolve).") from exc

    raw_cache = load_raw_cache(str(raw_cache_npz_path))
    data = np.asarray(raw_cache["data"])
    sfreq = float(np.asarray(raw_cache["sfreq"]).ravel()[0])
    ch_names = [str(x) for x in raw_cache["ch_names"]]
    start_sec = float(np.asarray(raw_cache.get("start_sec", np.array([0.0]))).ravel()[0])

    if target_channel not in ch_names:
        raise ValueError(f"Channel '{target_channel}' not found in raw cache.")
    if data.ndim != 2:
        raise ValueError("raw cache 'data' must be 2D (n_channels, n_samples).")
    if sfreq <= 0:
        raise ValueError("Invalid sfreq in raw cache.")

    event_start = float(event_start)
    event_end = float(event_end)
    if event_end <= event_start:
        raise ValueError("event_end must be > event_start.")
    margin_sec = float(margin_sec)
    if margin_sec < 0:
        raise ValueError("margin_sec must be >= 0.")

    plot_start = max(float(start_sec), event_start - margin_sec)
    plot_end = min(float(start_sec) + (data.shape[1] / sfreq), event_end + margin_sec)
    if plot_end <= plot_start:
        raise ValueError("Invalid plot window after clipping to raw cache bounds.")

    i0 = int(round((plot_start - start_sec) * sfreq))
    i1 = int(round((plot_end - start_sec) * sfreq))
    i0 = max(0, i0)
    i1 = min(int(data.shape[1]), i1)
    if i1 <= i0:
        raise ValueError("Empty signal segment after index conversion.")

    ch_idx = ch_names.index(target_channel)
    x = np.asarray(data[ch_idx, i0:i1], dtype=np.float64)
    # Keep a stable time base in seconds (absolute time w.r.t. crop)
    t = (np.arange(x.shape[0], dtype=np.float64) / sfreq) + plot_start

    decim = int(decim)
    if decim < 1:
        raise ValueError("decim must be >= 1.")

    # Log-spaced frequencies are usually more readable for HFO blobs.
    band_name = "custom"
    if band is not None:
        b = str(band).lower().strip()
        if b in ("ripple", "rp", "ripples"):
            fmin, fmax = 80.0, 250.0
            band_name = "ripple"
        elif b in ("fast_ripple", "fast-ripple", "fr", "fast"):
            fmin, fmax = 250.0, 500.0
            band_name = "fast_ripple"
        else:
            raise ValueError("band must be 'ripple' or 'fast_ripple' (or None).")

    fmin = float(fmin)
    fmax = float(fmax)
    if fmin <= 0 or fmax <= 0 or fmax <= fmin:
        raise ValueError("Require 0 < fmin < fmax.")
    n_freqs = int(n_freqs)
    if n_freqs < 4:
        raise ValueError("n_freqs must be >= 4.")
    freqs = bqk_utils.get_wavelet_freqs(fmin, fmax, n_freqs, scale="log")

    # SciPy-only Morlet wavelet power using fftconvolve (stable, no MNE dependency).
    # Wavelet definition:
    #   psi(t) = exp(2*pi*i*f*t) * exp(-t^2 / (2*sigma_t^2)), sigma_t = n_cycles/(2*pi*f)
    method = "scipy_fftconvolve_morlet"
    n_sig = int(x.shape[0])
    x_full = np.asarray(data[ch_idx], dtype=np.float64)
    n_sig_full = int(x_full.shape[0])
    power = np.zeros((freqs.shape[0], n_sig), dtype=np.float64)
    n_cycles = float(n_cycles)
    if n_cycles <= 0:
        raise ValueError("n_cycles must be > 0.")
    n_cycles_mode = str(n_cycles_mode).lower().strip()
    if n_cycles_mode not in ("fixed", "linear", "f_over_2"):
        raise ValueError("n_cycles_mode must be 'fixed', 'linear', or 'f_over_2'.")
    if n_cycles_mode == "fixed":
        cycles = np.full(freqs.shape[0], float(n_cycles), dtype=np.float64)
    elif n_cycles_mode == "linear":
        cmin = float(n_cycles_min)
        cmax = float(n_cycles_max)
        if cmin <= 0 or cmax <= 0 or cmax < cmin:
            raise ValueError("n_cycles_min/max must be >0 and max>=min.")
        cycles = np.linspace(cmin, cmax, freqs.shape[0]).astype(np.float64)
    else:
        cycles = (freqs / 2.0).astype(np.float64)

    for fi, f0 in enumerate(freqs):
        c = float(cycles[fi])
        sigma_t = c / (2.0 * np.pi * float(f0))
        # +/- 3 sigma covers 99.7% of Gaussian energy
        half_support = int(np.ceil(3.0 * sigma_t * sfreq))
        tt = (np.arange(-half_support, half_support + 1, dtype=np.float64) / sfreq)
        w = np.exp(2j * np.pi * float(f0) * tt) * np.exp(-(tt**2) / (2.0 * sigma_t**2))
        # L2 normalize (keeps scale comparable across freqs)
        wn = np.sqrt(np.sum(np.abs(w) ** 2)) + 1e-30
        w = w / wn
        y = fftconvolve(x, w, mode="same")
        power[fi, :] = (np.abs(y) ** 2).astype(np.float64, copy=False)

    if decim > 1:
        power = power[:, ::decim]

    # Align time vector with decimated output
    t_out = t[::decim] if decim > 1 else t
    if power.shape[1] != t_out.shape[0]:
        # guard for off-by-one from backend
        n = min(power.shape[1], t_out.shape[0])
        power = power[:, :n]
        t_out = t_out[:n]

    # Convert to dB for visualization (avoid -inf).
    power_db = 10.0 * np.log10(power + 1e-30)

    # Optional smoothing to reduce striping and highlight blobs.
    power_db_smooth = None
    try:
        from scipy.ndimage import gaussian_filter

        sigma_t = max(0.0, float(smooth_time_ms)) * 1e-3 * (sfreq / float(decim))
        sigma_f = max(0.0, float(smooth_freq_bins))
        if sigma_t > 0.0 or sigma_f > 0.0:
            power_db_smooth = gaussian_filter(power_db, sigma=(sigma_f, sigma_t), mode="nearest")
    except Exception:
        power_db_smooth = None

    # Optional baseline normalization for better HFO contrast.
    baseline_mode = str(baseline_mode).lower().strip()
    if baseline_mode not in ("none", "robust_z", "event_pctl", "dynamic_pool"):
        raise ValueError("baseline_mode must be 'none', 'robust_z', 'event_pctl', or 'dynamic_pool'.")
    power_z = None
    power_pctl = None
    power_db_baseline = None
    if baseline_mode == "robust_z":
        base_mask = (t_out < float(event_start)) | (t_out > float(event_end))
        # If margin is too small, base_mask could be empty; fall back to using full window.
        if not np.any(base_mask):
            base_mask = np.ones_like(t_out, dtype=bool)
        # robust z-score per frequency: (x - median) / MAD
        med = np.median(power_db[:, base_mask], axis=1, keepdims=True)
        mad = np.median(np.abs(power_db[:, base_mask] - med), axis=1, keepdims=True) + 1e-9
        power_z = (power_db - med) / mad
    elif baseline_mode == "event_pctl":
        low = float(baseline_pctl_low)
        high = float(baseline_pctl_high)
        if not (0.0 <= low < high <= 100.0):
            raise ValueError("baseline_pctl_low/high must satisfy 0 <= low < high <= 100.")
        ev_mask = (t_out >= float(event_start)) & (t_out <= float(event_end))
        if not np.any(ev_mask):
            ev_mask = np.ones_like(t_out, dtype=bool)
        p_lo = np.percentile(power_db[:, ev_mask], low, axis=1, keepdims=True)
        p_hi = np.percentile(power_db[:, ev_mask], high, axis=1, keepdims=True)
        denom = (p_hi - p_lo) + 1e-9
        power_pctl = (power_db - p_lo) / denom
        power_pctl = np.clip(power_pctl, 0.0, 1.0)
    elif baseline_mode == "dynamic_pool":
        if baseline_pool_starts is None or baseline_pool_indices is None:
            raise ValueError("dynamic_pool requires baseline_pool_starts and baseline_pool_indices.")
        pool_starts = np.asarray(baseline_pool_starts, dtype=np.float64)
        pool_indices = np.asarray(baseline_pool_indices, dtype=np.int64)
        if pool_starts.size == 0 or pool_indices.size == 0:
            power_db_baseline = power_db.copy()
        else:
            event_time = 0.5 * (float(event_start) + float(event_end))
            sel = _select_nearest_pool_indices(
                pool_starts,
                event_time,
                n_select=int(baseline_n_select),
                min_distance_sec=float(baseline_min_distance_sec),
            )
            if sel.size == 0:
                power_db_baseline = power_db.copy()
            else:
                win = int(round(float(baseline_window_sec) * sfreq))
                spectra = []
                for pi in pool_indices[sel]:
                    s0 = int(pi)
                    s1 = min(n_sig_full, s0 + win)
                    if s1 <= s0:
                        continue
                    xb = np.asarray(x_full[s0:s1], dtype=np.float64)
                    power_b = np.zeros((freqs.shape[0], xb.shape[0]), dtype=np.float64)
                    for fi, f0 in enumerate(freqs):
                        c = float(cycles[fi])
                        sigma_t = c / (2.0 * np.pi * float(f0))
                        half_support = int(np.ceil(3.0 * sigma_t * sfreq))
                        tt = (np.arange(-half_support, half_support + 1, dtype=np.float64) / sfreq)
                        w = np.exp(2j * np.pi * float(f0) * tt) * np.exp(-(tt**2) / (2.0 * sigma_t**2))
                        wn = np.sqrt(np.sum(np.abs(w) ** 2)) + 1e-30
                        w = w / wn
                        yb = fftconvolve(xb, w, mode="same")
                        power_b[fi, :] = (np.abs(yb) ** 2).astype(np.float64, copy=False)
                    spectra.append(np.mean(power_b, axis=1))
                if spectra:
                    base = np.median(np.stack(spectra, axis=1), axis=1, keepdims=True) + 1e-30
                    power_db_baseline = 10.0 * np.log10(power / base)
                else:
                    power_db_baseline = power_db.copy()

    if output_npz_path is None:
        raw_path = Path(str(raw_cache_npz_path))
        base = raw_path.name
        # "FC10477Q_rawCache_none.npz" -> "FC10477Q"
        prefix = base.split("_rawCache_", 1)[0]
        ch_tag = _safe_filename_component(target_channel)
        ev_tag = f"{event_start:.3f}_{event_end:.3f}".replace(".", "p")
        band_tag = _safe_filename_component(band_name)
        out_name = f"{prefix}_hfoVerify_wavelet_{band_tag}_{ch_tag}_{ev_tag}.npz"
        output_npz_path = str(raw_path.parent / out_name)

    out_p = Path(str(output_npz_path))
    if out_p.exists() and not bool(overwrite):
        return str(out_p)

    np.savez_compressed(
        str(out_p),
        method=np.array([method], dtype=object),
        band=np.array([band_name], dtype=object),
        freq_band_hz=np.array([float(fmin), float(fmax)], dtype=np.float64),
        sfreq=np.array([sfreq], dtype=np.float64),
        start_sec=np.array([start_sec], dtype=np.float64),
        target_channel=np.array([str(target_channel)], dtype=object),
        plot_start=np.array([float(plot_start)], dtype=np.float64),
        plot_end=np.array([float(plot_end)], dtype=np.float64),
        event_start=np.array([float(event_start)], dtype=np.float64),
        event_end=np.array([float(event_end)], dtype=np.float64),
        margin_sec=np.array([float(margin_sec)], dtype=np.float64),
        freqs_hz=freqs.astype(np.float64),
        t_sec=t_out.astype(np.float64),
        power_db=power_db.astype(np.float32),
        power_db_smooth=(power_db_smooth.astype(np.float32) if power_db_smooth is not None else np.zeros((0, 0), dtype=np.float32)),
        power_z=(power_z.astype(np.float32) if power_z is not None else np.zeros((0, 0), dtype=np.float32)),
        power_pctl=(power_pctl.astype(np.float32) if power_pctl is not None else np.zeros((0, 0), dtype=np.float32)),
        power_db_baseline=(power_db_baseline.astype(np.float32) if power_db_baseline is not None else np.zeros((0, 0), dtype=np.float32)),
        baseline_mode=np.array([baseline_mode], dtype=object),
        baseline_pctl_low=np.array([float(baseline_pctl_low)], dtype=np.float64),
        baseline_pctl_high=np.array([float(baseline_pctl_high)], dtype=np.float64),
        smooth_time_ms=np.array([float(smooth_time_ms)], dtype=np.float64),
        smooth_freq_bins=np.array([float(smooth_freq_bins)], dtype=np.float64),
        n_cycles=np.array([float(n_cycles)], dtype=np.float64),
        n_cycles_mode=np.array([str(n_cycles_mode)], dtype=object),
        n_cycles_min=np.array([float(n_cycles_min)], dtype=np.float64),
        n_cycles_max=np.array([float(n_cycles_max)], dtype=np.float64),
        n_cycles_vec=cycles.astype(np.float64),
        decim=np.array([int(decim)], dtype=np.int64),
        fmin=np.array([float(fmin)], dtype=np.float64),
        fmax=np.array([float(fmax)], dtype=np.float64),
        source_raw_cache=np.array([str(raw_cache_npz_path)], dtype=object),
    )
    return str(out_p)


def compute_and_save_hfo_event_wavelet_tfr_two_bands(
    *,
    raw_cache_npz_path: str,
    target_channel: str,
    event_start: float,
    event_end: float,
    margin_sec: float = 0.2,
    n_freqs: int = 180,
    n_cycles_ripple: float = 4.0,
    n_cycles_fast_ripple: float = 4.0,
    n_cycles_mode: str = "linear",
    n_cycles_min: float = 3.0,
    n_cycles_max: float = 10.0,
    decim: int = 1,
    baseline_mode: str = "robust_z",
    baseline_pctl_low: float = 5.0,
    baseline_pctl_high: float = 95.0,
    smooth_time_ms: float = 8.0,
    smooth_freq_bins: float = 1.5,
    overwrite: bool = False,
    baseline_pool_starts: Optional[np.ndarray] = None,
    baseline_pool_indices: Optional[np.ndarray] = None,
    baseline_window_sec: float = 2.0,
    baseline_n_select: int = 10,
    baseline_min_distance_sec: float = 2.0,
) -> Dict[str, str]:
    """
    Convenience wrapper: precompute and save BOTH ripple (80-250) and fast-ripple (250-500) TFR caches.
    Returns dict with keys: 'ripple', 'fast_ripple'.
    """
    out: Dict[str, str] = {}
    out["ripple"] = compute_and_save_hfo_event_wavelet_tfr(
        raw_cache_npz_path=str(raw_cache_npz_path),
        target_channel=str(target_channel),
        event_start=float(event_start),
        event_end=float(event_end),
        margin_sec=float(margin_sec),
        band="ripple",
        n_freqs=int(n_freqs),
        n_cycles=float(n_cycles_ripple),
        n_cycles_mode=str(n_cycles_mode),
        n_cycles_min=float(n_cycles_min),
        n_cycles_max=float(n_cycles_max),
        decim=int(decim),
        baseline_mode=str(baseline_mode),
        baseline_pctl_low=float(baseline_pctl_low),
        baseline_pctl_high=float(baseline_pctl_high),
        smooth_time_ms=float(smooth_time_ms),
        smooth_freq_bins=float(smooth_freq_bins),
        overwrite=bool(overwrite),
        baseline_pool_starts=baseline_pool_starts,
        baseline_pool_indices=baseline_pool_indices,
        baseline_window_sec=float(baseline_window_sec),
        baseline_n_select=int(baseline_n_select),
        baseline_min_distance_sec=float(baseline_min_distance_sec),
    )
    out["fast_ripple"] = compute_and_save_hfo_event_wavelet_tfr(
        raw_cache_npz_path=str(raw_cache_npz_path),
        target_channel=str(target_channel),
        event_start=float(event_start),
        event_end=float(event_end),
        margin_sec=float(margin_sec),
        band="fast_ripple",
        n_freqs=int(n_freqs),
        n_cycles=float(n_cycles_fast_ripple),
        n_cycles_mode=str(n_cycles_mode),
        n_cycles_min=float(n_cycles_min),
        n_cycles_max=float(n_cycles_max),
        decim=int(decim),
        baseline_mode=str(baseline_mode),
        baseline_pctl_low=float(baseline_pctl_low),
        baseline_pctl_high=float(baseline_pctl_high),
        smooth_time_ms=float(smooth_time_ms),
        smooth_freq_bins=float(smooth_freq_bins),
        overwrite=bool(overwrite),
        baseline_pool_starts=baseline_pool_starts,
        baseline_pool_indices=baseline_pool_indices,
        baseline_window_sec=float(baseline_window_sec),
        baseline_n_select=int(baseline_n_select),
        baseline_min_distance_sec=float(baseline_min_distance_sec),
    )
    return out


def compute_centroid_matrix_from_envelope_cache(
    *,
    windows: Sequence[EventWindow],
    detections: Mapping[str, np.ndarray],
    ch_names: Sequence[str],
    env: np.ndarray,
    env_ch_names: Optional[Sequence[str]] = None,
    sfreq: float,
    start_sec: float = 0.0,
    centroid_power: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute centroids using a precomputed envelope cache (env per channel for the whole crop).
    """
    ch_names = [str(c) for c in ch_names]
    windows = list(windows)
    env = np.asarray(env)
    sf = float(sfreq)
    start_sec = float(start_sec)

    if env.ndim != 2:
        raise ValueError("env must be 2D (n_channels, n_samples)")
    # If cache contains more channels than requested, select by name.
    if env_ch_names is not None:
        env_names = [str(x) for x in env_ch_names]
        name_to_row = {n: i for i, n in enumerate(env_names)}
        rows = []
        for nm in ch_names:
            if nm not in name_to_row:
                raise KeyError(f"Channel '{nm}' not found in envelope cache.")
            rows.append(name_to_row[nm])
        env = env[np.array(rows, dtype=int), :]
    else:
        if env.shape[0] != len(ch_names):
            raise ValueError("env.shape[0] must match len(ch_names) (or pass env_ch_names).")

    n_ch = len(ch_names)
    n_ev = len(windows)
    events_bool = np.zeros((n_ch, n_ev), dtype=bool)
    centroids = np.full((n_ch, n_ev), np.nan, dtype=np.float64)

    for ei, win in enumerate(windows):
        i0 = int(round((win.start - start_sec) * sf))
        i1 = int(round((win.end - start_sec) * sf))
        if i1 <= 0 or i0 >= env.shape[1]:
            continue
        i0c = max(0, i0)
        i1c = min(env.shape[1], i1)
        if i1c <= i0c:
            continue

        for ci, ch in enumerate(ch_names):
            times = detections.get(ch, None)
            if not _overlaps_any_event(times, win):
                continue
            events_bool[ci, ei] = True
            env_win = np.asarray(env[ci, i0c:i1c], dtype=np.float64)
            t_vec = (np.arange(env_win.shape[-1], dtype=np.float64) / sf) + float(win.start)
            c_time = _compute_energy_centroid(env_win, t_vec, power=float(centroid_power))
            centroids[ci, ei] = float(c_time - win.start)

    return centroids, events_bool

def _xcorr_lag(env1: np.ndarray, env2: np.ndarray, sfreq: float) -> float:
    """
    Compute lag of env2 relative to env1 using cross-correlation on Hilbert envelopes.
    
    Returns
    -------
    lag_sec : float
        Positive if env2 lags env1 (env1 leads), negative otherwise.
    """
    from scipy.signal import correlate
    
    env1 = np.asarray(env1, dtype=np.float64)
    env2 = np.asarray(env2, dtype=np.float64)
    
    # Normalize to zero mean
    env1 = env1 - np.mean(env1)
    env2 = env2 - np.mean(env2)
    
    # Cross-correlation (full mode)
    corr = correlate(env2, env1, mode='full')
    
    # The peak index in 'full' mode: index = len(env1) - 1 + lag
    # So lag = peak_index - (len(env1) - 1)
    n = len(env1)
    peak_idx = int(np.argmax(corr))
    lag_samples = peak_idx - (n - 1)
    
    return float(lag_samples) / sfreq


def _match_raw_channel_name(raw_ch_names: Sequence[str], target_clean: str) -> Optional[str]:
    """
    Map a clean channel name like 'K10' to one of raw channel names like:
    'POL K10', 'EEG K10-Ref', etc.
    """
    target_clean = target_clean.strip().upper()
    # Exact clean match first
    for rn in raw_ch_names:
        if ElectrodeParser.clean_name(rn).upper() == target_clean:
            return rn
    # Fallback: substring match (last resort)
    for rn in raw_ch_names:
        if target_clean in rn.upper():
            return rn
    return None


class MNERawOnDemandLoader:
    """
    On-demand loader for one-channel segments from an mne.io.Raw instance.

    Notes
    -----
    - raw should be opened with preload=False to avoid memory blowup.
    - Channel name mapping uses ElectrodeParser.clean_name to match GPU names.
    """

    def __init__(self, raw, *, channel_map: Optional[Mapping[str, str]] = None):
        self.raw = raw
        self.sfreq = float(raw.info["sfreq"])
        self._raw_names = list(raw.ch_names)
        self._channel_map = dict(channel_map) if channel_map is not None else {}
        # Pre-build name-to-index map for faster lookup
        self._name_to_idx = {n: i for i, n in enumerate(self._raw_names)}

    def resolve_channel(self, ch_name: str) -> Tuple[str, int]:
        """Return (raw_channel_name, raw_channel_index)."""
        clean = ElectrodeParser.clean_name(ch_name).upper()
        if clean in self._channel_map:
            rn = self._channel_map[clean]
            return rn, self._name_to_idx[rn]
        matched = _match_raw_channel_name(self._raw_names, clean)
        if matched is None:
            raise KeyError(f"Cannot match channel '{ch_name}' to EDF raw channel list.")
        self._channel_map[clean] = matched
        return matched, self._name_to_idx[matched]

    def load(self, ch_name: str, t_start: float, t_end: float) -> np.ndarray:
        if t_end <= t_start:
            raise ValueError("t_end must be > t_start")
        sf = self.sfreq
        start_samp = int(np.floor(t_start * sf))
        end_samp = int(np.ceil(t_end * sf))
        start_samp = max(start_samp, 0)
        end_samp = max(end_samp, start_samp + 1)

        _, ch_idx = self.resolve_channel(ch_name)
        # Use get_data for efficiency (no copy of entire raw object)
        data = self.raw.get_data(picks=[ch_idx], start=start_samp, stop=end_samp)
        return np.asarray(data[0], dtype=np.float64, order="C")


class BipolarAliasOnDemandLoader:
    """
    On-demand loader that exposes "alias" bipolar channels:
      alias 'A1' means 'A1 - A2' (or nearest next contact within allow_gap).

    This matches the common legacy convention in some pipelines where bipolar pairs
    are named by the first contact only.
    """

    def __init__(self, raw, *, allow_gap: int = 2):
        self.raw = raw
        self.sfreq = float(raw.info["sfreq"])
        self._raw_names = list(raw.ch_names)
        self._name_to_idx = {n: i for i, n in enumerate(self._raw_names)}

        # Map clean contact name -> raw index
        clean_to_raw: Dict[str, int] = {}
        for rn in self._raw_names:
            cn = ElectrodeParser.clean_name(rn).upper()
            if cn not in clean_to_raw:
                clean_to_raw[cn] = self._name_to_idx[rn]
        self._clean_to_raw = clean_to_raw

        # Build alias -> (idx_a, idx_b) mapping from available contacts
        by_prefix: Dict[str, List[Tuple[int, str]]] = {}
        for cn, idx in clean_to_raw.items():
            prefix, num = ElectrodeParser.parse(cn)
            if prefix is None or num is None:
                continue
            by_prefix.setdefault(prefix, []).append((num, cn))

        alias_to_pair: Dict[str, Tuple[int, int]] = {}
        for prefix, items in by_prefix.items():
            items.sort(key=lambda x: x[0])
            for (n1, c1), (n2, c2) in zip(items[:-1], items[1:]):
                if (n2 - n1) <= int(allow_gap):
                    a = clean_to_raw[c1]
                    b = clean_to_raw[c2]
                    alias_to_pair[c1] = (a, b)  # alias is first contact name
        self._alias_to_pair = alias_to_pair

    def load(self, alias_name: str, t_start: float, t_end: float) -> np.ndarray:
        if t_end <= t_start:
            raise ValueError("t_end must be > t_start")
        sf = self.sfreq
        start_samp = int(np.floor(t_start * sf))
        end_samp = int(np.ceil(t_end * sf))
        start_samp = max(start_samp, 0)
        end_samp = max(end_samp, start_samp + 1)

        key = ElectrodeParser.clean_name(alias_name).upper()
        if key not in self._alias_to_pair:
            raise KeyError(f"No bipolar pair available for alias '{alias_name}'")
        ia, ib = self._alias_to_pair[key]
        a = self.raw.get_data(picks=[ia], start=start_samp, stop=end_samp)[0]
        b = self.raw.get_data(picks=[ib], start=start_samp, stop=end_samp)[0]
        return np.asarray(a - b, dtype=np.float64, order="C")


class GroupEventAnalyzer:
    """
    Compute lag matrices (lagPatRaw-like) and ranks from aligned event windows.
    
    Supports two timing methods:
    - 'centroid': Energy-weighted centroid time within window (default)
    - 'xcorr': Cross-correlation of Hilbert envelopes between channels
    """

    def __init__(
        self,
        *,
        sfreq: float,
        band: str = "ripple",
        window_sec: float = 0.5,
        pad_sec: float = 0.1,
        centroid_power: float = 2.0,
        method: str = "centroid",  # 'centroid' or 'xcorr'
    ):
        self.sfreq = float(sfreq)
        self.band = band
        self.window_sec = float(window_sec)
        self.pad_sec = float(pad_sec)
        self.centroid_power = float(centroid_power)
        if method not in ("centroid", "xcorr"):
            raise ValueError(f"method must be 'centroid' or 'xcorr', got '{method}'")
        self.method = method

    def _freqband(self) -> Tuple[float, float]:
        b = self.band.lower().strip()
        if b in ("ripple", "ripples"):
            return (80.0, 250.0)
        if b in ("fast_ripple", "fast-ripple", "fr", "fast"):
            return (250.0, 500.0)
        raise ValueError(f"Unknown band='{self.band}'. Use 'ripple' or 'fast_ripple'.")

    def compute_lag_matrices(
        self,
        *,
        windows: Sequence[EventWindow],
        detections: Mapping[str, np.ndarray],
        ch_names: Sequence[str],
        data_loader: Callable[[str, float, float], np.ndarray],
    ) -> LagMatrices:
        """
        Core routine:
        - events_bool: overlap(detections[ch], window)
        - For 'centroid': lag_raw[ch, ev] = centroid_time - window.start
        - For 'xcorr': lag_raw[ch, ev] = xcorr lag relative to earliest-peak channel

        data_loader signature:
            (ch_name, t_start, t_end) -> 1D np.ndarray voltage
        """
        ch_names = [str(c) for c in ch_names]
        windows = list(windows)
        n_ch = len(ch_names)
        n_ev = len(windows)

        events_bool = np.zeros((n_ch, n_ev), dtype=bool)
        lag_raw = np.full((n_ch, n_ev), np.nan, dtype=np.float64)

        fb = self._freqband()
        sf = self.sfreq
        pad = self.pad_sec

        for ei, win in enumerate(windows):
            # First pass: determine which channels participate and load their envelopes
            participating_idx = []
            envelopes = {}
            peak_times = {}
            
            for ci, ch in enumerate(ch_names):
                times = detections.get(ch, None)
                if not _overlaps_any_event(times, win):
                    continue

                events_bool[ci, ei] = True
                participating_idx.append(ci)

                t0 = float(win.start - pad)
                t1 = float(win.end + pad)
                x = data_loader(ch, t0, t1)

                # Envelope: match legacy bqk (20Hz sub-bands sum) for robustness.
                env = bqk_utils.return_hil_enve_norm(x, sf, fb)

                # Restrict to the true window (exclude padding).
                pad_samp = int(round(pad * sf))
                win_samp = int(round((win.end - win.start) * sf))
                i0 = max(pad_samp, 0)
                i1 = min(i0 + win_samp, env.shape[-1])
                env_win = env[i0:i1]
                envelopes[ci] = env_win
                
                # Track peak time for reference selection
                peak_idx = int(np.argmax(env_win))
                peak_times[ci] = float(peak_idx) / sf

            if not participating_idx:
                continue

            if self.method == "centroid":
                # Centroid method: absolute timing relative to window start
                for ci in participating_idx:
                    env_win = envelopes[ci]
                    t_vec = (np.arange(env_win.shape[-1], dtype=np.float64) / sf) + float(win.start)
                    c_time = _compute_energy_centroid(env_win, t_vec, power=self.centroid_power)
                    lag_raw[ci, ei] = float(c_time - win.start)

            elif self.method == "xcorr":
                # XCorr method: relative timing via cross-correlation
                # Reference channel = earliest peak (like original algorithm)
                ref_ci = min(participating_idx, key=lambda c: peak_times[c])
                ref_env = envelopes[ref_ci]
                
                for ci in participating_idx:
                    if ci == ref_ci:
                        lag_raw[ci, ei] = 0.0
                    else:
                        # Lag = how much ci lags behind ref_ci
                        lag_raw[ci, ei] = _xcorr_lag(ref_env, envelopes[ci], sf)

        lag_rank = compute_dense_rank(lag_raw, events_bool)
        return LagMatrices(
            ch_names=list(ch_names),
            windows=windows,
            lag_raw=lag_raw,
            events_bool=events_bool,
            lag_rank=lag_rank,
        )


def compute_centroid_matrix(
    *,
    windows: Sequence[EventWindow],
    detections: Mapping[str, np.ndarray],
    ch_names: Sequence[str],
    data_loader: Callable[[str, float, float], np.ndarray],
    sfreq: float,
    band: str,
    pad_sec: float = 0.1,
    centroid_power: float = 2.0,
    use_gpu: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Step-2: compute per-window, per-channel centroid times.

    Returns
    -------
    centroids_sec : (n_channels, n_events) float
        Centroid time relative to window start (seconds). NaN if no event.
    events_bool : (n_channels, n_events) bool
        Channel participation based on overlap with the window.
    """
    ch_names = [str(c) for c in ch_names]
    windows = list(windows)
    n_ch = len(ch_names)
    n_ev = len(windows)

    events_bool = np.zeros((n_ch, n_ev), dtype=bool)
    centroids = np.full((n_ch, n_ev), np.nan, dtype=np.float64)

    b = band.lower().strip()
    if b in ("ripple", "ripples"):
        fb = (80.0, 250.0)
    elif b in ("fast_ripple", "fast-ripple", "fr", "fast"):
        fb = (250.0, 500.0)
    else:
        raise ValueError(f"Unknown band='{band}'. Use 'ripple' or 'fast_ripple'.")

    sf = float(sfreq)
    pad = float(pad_sec)

    for ei, win in enumerate(windows):
        for ci, ch in enumerate(ch_names):
            times = detections.get(ch, None)
            if not _overlaps_any_event(times, win):
                continue

            events_bool[ci, ei] = True

            t0 = float(win.start - pad)
            t1 = float(win.end + pad)
            x = data_loader(ch, t0, t1)

            env = _maybe_gpu_envelope(x, sf, fb, use_gpu=bool(use_gpu))

            pad_samp = int(round(pad * sf))
            win_samp = int(round((win.end - win.start) * sf))
            i0 = max(pad_samp, 0)
            i1 = min(i0 + win_samp, env.shape[-1])
            env_win = env[i0:i1]

            t_vec = (np.arange(env_win.shape[-1], dtype=np.float64) / sf) + float(win.start)
            c_time = _compute_energy_centroid(env_win, t_vec, power=float(centroid_power))
            centroids[ci, ei] = float(c_time - win.start)

    return centroids, events_bool


def lag_rank_from_centroids(
    centroids_sec: np.ndarray,
    events_bool: np.ndarray,
    *,
    align: str = "first_centroid",
    tie_tol_ms: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Step-3: convert centroid matrix into lag and rank.

    align:
      - 'window_start': lag == centroid_sec (relative to window start)
      - 'first_centroid': lag == centroid_sec - min_centroid_in_event
    """
    centroids_sec = np.asarray(centroids_sec, dtype=np.float64)
    events_bool = np.asarray(events_bool, dtype=bool)
    if centroids_sec.shape != events_bool.shape:
        raise ValueError("centroids_sec and events_bool must have same shape")

    lag = centroids_sec.copy()
    if align not in ("window_start", "first_centroid"):
        raise ValueError("align must be 'window_start' or 'first_centroid'")

    if align == "first_centroid":
        for ei in range(lag.shape[1]):
            mask = events_bool[:, ei] & np.isfinite(lag[:, ei])
            if not np.any(mask):
                continue
            lag[:, ei] = lag[:, ei] - np.min(lag[mask, ei])

    if float(tie_tol_ms) > 0.0:
        rank = compute_dense_rank_with_ties(lag, events_bool, tie_tol_ms=float(tie_tol_ms))
    else:
        rank = compute_dense_rank(lag, events_bool)
    return lag, rank


def compute_dense_rank(lag_raw: np.ndarray, events_bool: np.ndarray) -> np.ndarray:
    """
    Compute dense ranks per event (column) for participating channels only.

    Returns rank matrix with:
    - 0 = earliest lag
    - -1 = non-participant or NaN
    """
    lag_raw = np.asarray(lag_raw, dtype=np.float64)
    events_bool = np.asarray(events_bool, dtype=bool)
    if lag_raw.shape != events_bool.shape:
        raise ValueError("lag_raw and events_bool must have same shape")

    n_ch, n_ev = lag_raw.shape
    out = np.full((n_ch, n_ev), -1, dtype=np.int64)

    for ei in range(n_ev):
        mask = events_bool[:, ei] & np.isfinite(lag_raw[:, ei])
        if not np.any(mask):
            continue
        idx = np.where(mask)[0]
        vals = lag_raw[idx, ei]
        order = np.argsort(vals, kind="stable")
        out[idx[order], ei] = np.arange(order.size, dtype=np.int64)

    return out


def compute_dense_rank_with_ties(
    lag_raw: np.ndarray,
    events_bool: np.ndarray,
    *,
    tie_tol_ms: float = 0.0,
) -> np.ndarray:
    """
    Dense rank per event with optional tie tolerance.

    If tie_tol_ms > 0, lags within tie_tol_ms are treated as the same rank (in sorted order).
    This is reasonable when the timing estimator's precision is limited by sampling interval,
    envelope smoothness, SNR, and residual preprocessing differences.
    """
    lag_raw = np.asarray(lag_raw, dtype=np.float64)
    events_bool = np.asarray(events_bool, dtype=bool)
    if lag_raw.shape != events_bool.shape:
        raise ValueError("lag_raw and events_bool must have same shape")

    tol = float(tie_tol_ms) * 1e-3
    n_ch, n_ev = lag_raw.shape
    out = np.full((n_ch, n_ev), -1, dtype=np.int64)

    for ei in range(n_ev):
        mask = events_bool[:, ei] & np.isfinite(lag_raw[:, ei])
        if not np.any(mask):
            continue
        idx = np.where(mask)[0]
        vals = lag_raw[idx, ei]
        order = np.argsort(vals, kind="stable")
        idx_s = idx[order]
        vals_s = vals[order]

        rank = 0
        out[idx_s[0], ei] = 0
        last_val = float(vals_s[0])
        for k in range(1, len(idx_s)):
            v = float(vals_s[k])
            if tol > 0.0 and (v - last_val) <= tol:
                out[idx_s[k], ei] = rank
            else:
                rank += 1
                out[idx_s[k], ei] = rank
                last_val = v

    return out


def compare_to_reference_lagpat(
    *,
    computed: LagMatrices,
    ref_lag_raw: np.ndarray,
    ref_lag_rank: Optional[np.ndarray] = None,
    ref_events_bool: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compare computed matrices to reference (Yuquan *_lagPat.npz).

    Returns a dict of simple, robust metrics.
    """
    ref_lag_raw = np.asarray(ref_lag_raw, dtype=np.float64)
    if ref_lag_raw.shape != computed.lag_raw.shape:
        raise ValueError(f"Shape mismatch: ref {ref_lag_raw.shape} vs computed {computed.lag_raw.shape}")

    mask = np.isfinite(ref_lag_raw) & np.isfinite(computed.lag_raw)
    if ref_events_bool is not None:
        ref_events_bool = np.asarray(ref_events_bool).astype(bool)
        mask &= ref_events_bool
    if not np.any(mask):
        return {"n_compare": 0.0}

    diff = computed.lag_raw[mask] - ref_lag_raw[mask]
    metrics: Dict[str, float] = {
        "n_compare": float(mask.sum()),
        "lag_mae_s": float(np.mean(np.abs(diff))),
        "lag_rmse_s": float(np.sqrt(np.mean(diff**2))),
    }

    if ref_lag_rank is not None:
        ref_lag_rank = np.asarray(ref_lag_rank)
        if ref_lag_rank.shape == computed.lag_rank.shape:
            # Handle common convention differences: 0-based vs 1-based ranks.
            match0 = (computed.lag_rank == ref_lag_rank) & (computed.lag_rank >= 0)
            match1 = (computed.lag_rank + 1 == ref_lag_rank) & (computed.lag_rank >= 0)
            denom = float(np.sum(computed.lag_rank >= 0))
            if denom > 0:
                metrics["rank_match_rate_0based"] = float(np.sum(match0) / denom)
                metrics["rank_match_rate_1based"] = float(np.sum(match1) / denom)

    return metrics


def _detections_dict_from_hfo_result(hfo_result) -> Dict[str, np.ndarray]:
    """
    Convert HFODetectionResult -> Dict[ch_name] = (n_events,2) seconds array.
    """
    dets: Dict[str, np.ndarray] = {}
    ch_names = list(getattr(hfo_result, "ch_names"))
    events_by_channel = list(getattr(hfo_result, "events_by_channel"))
    for ch, ev in zip(ch_names, events_by_channel):
        dets[str(ch)] = np.asarray(ev, dtype=np.float64)
    return dets


def bqk_detect_and_compare_windows_to_packed(
    *,
    edf_path: str,
    packed_times_path: str,
    band: str,
    crop_seconds: float,
    reference: str = "none",
    include_channels: Optional[List[str]] = None,
    alias_bipolar_to_left: bool = False,
    alias_filter_using_gpu_npz: Optional[str] = None,
    rel_thresh: float = 3.0,
    abs_thresh: float = 3.0,
    min_gap_ms: float = 20.0,
    min_last_ms: float = 50.0,
    window_sec: Optional[float] = None,
    min_overlap_sec: Optional[float] = None,
    min_channels: int = 1,
    restrict_channels: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """
    Step-1 validation:
      Our bqk detector -> build_windows_from_detections
      vs
      Dataset packedTimes (within the same crop time).

    This validates *our detection + window assembly*, not other people's lag code.
    """
    from .preprocessing import SEEGPreprocessor
    from .hfo_detector import HFODetector, HFODetectionConfig

    packed = np.load(packed_times_path, allow_pickle=True)
    packed_windows_all = build_windows_from_packed_times(packed)
    packed_windows = [w for w in packed_windows_all if w.start < float(crop_seconds)]
    if window_sec is None:
        # Infer from packedTimes (robust to different datasets)
        durs = packed[:, 1] - packed[:, 0]
        window_sec = float(np.median(durs))
    if min_overlap_sec is None:
        # Require at least half-window overlap by default
        min_overlap_sec = float(window_sec) * 0.5

    pre = SEEGPreprocessor(
        target_band="fast_ripple" if band in ("fast_ripple", "fast-ripple", "fr", "fast") else "ripple",
        reference=reference,
        include_channels=include_channels,
        crop_seconds=float(crop_seconds),
        check_quality=False,
        use_gpu=False,
    )
    pre_out = pre.run(edf_path)

    cfg = HFODetectionConfig(
        algorithm="bqk",
        band=band,
        rel_thresh=float(rel_thresh),
        abs_thresh=float(abs_thresh),
        min_gap_ms=float(min_gap_ms),
        min_last_ms=float(min_last_ms),
        chunk_sec=30.0,
        chunk_overlap_sec=1.0,
        use_gpu=False,
    )
    det = HFODetector(cfg)
    if alias_bipolar_to_left:
        # Expect bipolar names like 'A1-A2' from preprocessing; alias to left contact 'A1'.
        data = np.asarray(pre_out.data)
        names = [str(x) for x in pre_out.ch_names]

        allowed = None
        if alias_filter_using_gpu_npz is not None:
            g = np.load(alias_filter_using_gpu_npz, allow_pickle=True)
            allowed = set(str(x).upper() for x in g["chns_names"].tolist())

        keep_idx: List[int] = []
        alias_names: List[str] = []
        for i, nm in enumerate(names):
            if "-" not in nm:
                continue
            left, right = nm.split("-", 1)
            left = left.strip().upper()
            right = right.strip().upper()
            if allowed is not None:
                # Keep only pairs whose both contacts exist in allowed set (drops distal contacts).
                if left not in allowed or right not in allowed:
                    continue
            keep_idx.append(i)
            alias_names.append(left)

        data2 = data[keep_idx] if keep_idx else np.zeros((0, data.shape[1]), dtype=np.float64)
        res = det.detect(data2, sfreq=float(pre_out.sfreq), ch_names=alias_names)
    else:
        res = det.detect(pre_out)
    dets = _detections_dict_from_hfo_result(res)
    if restrict_channels is not None:
        r = set(str(x) for x in restrict_channels)
        dets = {k: v for k, v in dets.items() if k in r}

    our_windows = build_windows_from_detections(dets, window_sec=float(window_sec))
    # crop our windows too, in case detection creates a late window start close to crop boundary
    our_windows = [w for w in our_windows if w.start < float(crop_seconds)]
    our_windows = filter_windows_by_min_channels(our_windows, dets, min_channels=int(min_channels))

    metrics = compare_window_sets(our_windows, packed_windows, min_overlap_sec=float(min_overlap_sec))
    metrics.update(
        {
            "crop_seconds": float(crop_seconds),
            "band_str": band,
            "min_channels": float(min_channels),
            "alias_bipolar_to_left": 1.0 if alias_bipolar_to_left else 0.0,
            "window_sec": float(window_sec),
            "min_overlap_sec": float(min_overlap_sec),
        }
    )
    return metrics


def gpu_detections_and_compare_windows_to_packed(
    *,
    gpu_npz_path: str,
    packed_times_path: str,
    crop_seconds: float,
    window_sec: Optional[float] = None,
    min_overlap_sec: Optional[float] = None,
    restrict_channels: Optional[Sequence[str]] = None,
    min_channels: int = 1,
) -> Dict[str, float]:
    """
    Step-1 control experiment:
      Use dataset-provided GPU detections -> build_windows_from_detections
      and compare to packedTimes.

    This isolates whether our window-building logic matches the dataset's packedTimes logic.
    """
    gpu = np.load(gpu_npz_path, allow_pickle=True)
    packed = np.load(packed_times_path, allow_pickle=True)

    packed_windows_all = build_windows_from_packed_times(packed)
    packed_windows = [w for w in packed_windows_all if w.start < float(crop_seconds)]
    if window_sec is None:
        durs = packed[:, 1] - packed[:, 0]
        window_sec = float(np.median(durs))
    if min_overlap_sec is None:
        min_overlap_sec = float(window_sec) * 0.5

    gpu_names = [str(x) for x in gpu["chns_names"].tolist()]
    dets_obj = gpu["whole_dets"]

    dets: Dict[str, np.ndarray] = {}
    if restrict_channels is None:
        use_names = gpu_names
    else:
        s = set(str(x) for x in restrict_channels)
        use_names = [n for n in gpu_names if n in s]

    name_to_idx = {n: i for i, n in enumerate(gpu_names)}
    for ch in use_names:
        dets[ch] = np.asarray(dets_obj[name_to_idx[ch]], dtype=np.float64)

    our_windows = build_windows_from_detections(dets, window_sec=float(window_sec))
    our_windows = [w for w in our_windows if w.start < float(crop_seconds)]
    our_windows = filter_windows_by_min_channels(our_windows, dets, min_channels=int(min_channels))

    metrics = compare_window_sets(our_windows, packed_windows, min_overlap_sec=float(min_overlap_sec))
    metrics.update(
        {
            "crop_seconds": float(crop_seconds),
            "min_channels": float(min_channels),
            "window_sec": float(window_sec),
            "min_overlap_sec": float(min_overlap_sec),
        }
    )
    return metrics


def validate_yuquan_record(
    *,
    edf_path: str,
    gpu_npz_path: str,
    packed_times_path: str,
    lagpat_npz_path: str,
    band: str = "ripple",
    pad_sec: float = 0.1,
    centroid_power: float = 2.0,
) -> Dict[str, float]:
    """
    End-to-end validation against Yuquan reference files for a single record.

    Uses:
    - packedTimes as authoritative aligned event windows
    - lagPatRaw/Rank/Bool as reference
    - GPU detections only to decide eventsBool (whether channel participates in a window)

    Returns a dict of metrics (MAE/RMSE and rank match rates).
    """
    import mne  # local import to keep module import light

    gpu = np.load(gpu_npz_path, allow_pickle=True)
    lag = np.load(lagpat_npz_path, allow_pickle=True)
    packed = np.load(packed_times_path, allow_pickle=True)

    core_ch = [str(x) for x in lag["chnNames"].tolist()]

    # Build detections dict for core channels only (empty if missing)
    gpu_names = [str(x) for x in gpu["chns_names"].tolist()]
    name_to_idx = {n: i for i, n in enumerate(gpu_names)}
    dets: Dict[str, np.ndarray] = {}
    for ch in core_ch:
        if ch in name_to_idx:
            dets[ch] = np.asarray(gpu["whole_dets"][name_to_idx[ch]])
        else:
            dets[ch] = np.zeros((0, 2), dtype=np.float64)

    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False, encoding="latin1")
    loader = MNERawOnDemandLoader(raw)

    windows = build_windows_from_packed_times(packed)
    analyzer = GroupEventAnalyzer(
        sfreq=float(raw.info["sfreq"]),
        band=band,
        window_sec=float(windows[0].duration) if windows else 0.5,
        pad_sec=pad_sec,
        centroid_power=centroid_power,
    )
    computed = analyzer.compute_lag_matrices(
        windows=windows, detections=dets, ch_names=core_ch, data_loader=loader.load
    )

    metrics = compare_to_reference_lagpat(
        computed=computed,
        ref_lag_raw=np.asarray(lag["lagPatRaw"]),
        ref_lag_rank=np.asarray(lag["lagPatRank"]) if "lagPatRank" in lag else None,
        ref_events_bool=np.asarray(lag["eventsBool"]) if "eventsBool" in lag else None,
    )
    return metrics


def validate_packedtimes_centroid_lagrank_against_lagpat(
    *,
    edf_path: str,
    gpu_npz_path: str,
    packed_times_path: str,
    lagpat_npz_path: str,
    band: str = "ripple",
    crop_seconds: Optional[float] = None,
    align: str = "first_centroid",
    use_bipolar_alias_loader: bool = True,
    allow_gap: int = 2,
    pad_sec: float = 0.1,
    centroid_power: float = 2.0,
    tie_tol_ms: float = 0.0,
    use_gpu: bool = False,
) -> Dict[str, float]:
    """
    Step-2/3 validation (core 8 channels):
      packedTimes windows -> compute centroid per channel -> align -> lag/rank
      compare against lagPatRank and relative lag derived from lagPatRaw.

    Notes
    -----
    - lagPatRaw absolute values appear to be on a stitched timeline; we compare *relative* lag
      within each event: lagPatRaw - min(lagPatRaw among participating channels).
    - eventsBool from lagPat is used as the participation mask for comparison.
    - GPU detections are used only to compute our events_bool (overlap), which should match.
    """
    import mne  # local import

    gpu = np.load(gpu_npz_path, allow_pickle=True)
    lag = np.load(lagpat_npz_path, allow_pickle=True)
    packed = np.load(packed_times_path, allow_pickle=True)

    core_ch = [str(x) for x in lag["chnNames"].tolist()]
    windows = build_windows_from_packed_times(packed)
    if crop_seconds is not None:
        windows = [w for w in windows if w.start < float(crop_seconds)]

    # detections from GPU for core channels
    gpu_names = [str(x) for x in gpu["chns_names"].tolist()]
    name_to_idx = {n: i for i, n in enumerate(gpu_names)}
    dets: Dict[str, np.ndarray] = {}
    for ch in core_ch:
        if ch in name_to_idx:
            dets[ch] = np.asarray(gpu["whole_dets"][name_to_idx[ch]], dtype=np.float64)
        else:
            dets[ch] = np.zeros((0, 2), dtype=np.float64)

    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False, encoding="latin1")
    if use_bipolar_alias_loader:
        loader = BipolarAliasOnDemandLoader(raw, allow_gap=int(allow_gap))
        load = loader.load
        sfreq = loader.sfreq
    else:
        loader2 = MNERawOnDemandLoader(raw)
        load = loader2.load
        sfreq = loader2.sfreq

    cent, events_bool = compute_centroid_matrix(
        windows=windows,
        detections=dets,
        ch_names=core_ch,
        data_loader=load,
        sfreq=float(sfreq),
        band=band,
        pad_sec=float(pad_sec),
        centroid_power=float(centroid_power),
        use_gpu=bool(use_gpu),
    )
    lag_ours, rank_ours = lag_rank_from_centroids(cent, events_bool, align=align, tie_tol_ms=float(tie_tol_ms))

    # Reference relative lag and rank
    ref_raw = np.asarray(lag["lagPatRaw"], dtype=np.float64)
    ref_rank = np.asarray(lag["lagPatRank"], dtype=np.int64)
    ref_bool = np.asarray(lag["eventsBool"]) > 0
    if crop_seconds is not None:
        # packedTimes index aligns with lagPat columns; so crop by number of windows kept
        n_keep = len(windows)
        ref_raw = ref_raw[:, :n_keep]
        ref_rank = ref_rank[:, :n_keep]
        ref_bool = ref_bool[:, :n_keep]

    # relative within-event reference
    ref_rel = np.full_like(ref_raw, np.nan, dtype=np.float64)
    for ei in range(ref_raw.shape[1]):
        mask = ref_bool[:, ei] & np.isfinite(ref_raw[:, ei])
        if not np.any(mask):
            continue
        ref_rel[:, ei] = ref_raw[:, ei] - np.min(ref_raw[mask, ei])

    # Compare only where both sides participate
    both = ref_bool & events_bool & np.isfinite(ref_rel) & np.isfinite(lag_ours)
    n_cmp = int(np.sum(both))
    out: Dict[str, float] = {"n_compare": float(n_cmp)}
    if n_cmp == 0:
        return out

    diff = lag_ours[both] - ref_rel[both]
    out["lag_rel_mae_ms"] = float(np.mean(np.abs(diff)) * 1000.0)
    out["lag_rel_rmse_ms"] = float(np.sqrt(np.mean(diff**2)) * 1000.0)

    # Rank match (0-based)
    denom = int(np.sum(ref_bool & events_bool))
    if denom > 0:
        out["rank_match_rate"] = float(np.mean((rank_ours == ref_rank)[ref_bool & events_bool]))
    else:
        out["rank_match_rate"] = 0.0

    # Participation agreement (sanity)
    out["eventsBool_match_rate"] = float(np.mean(ref_bool == events_bool))
    out["tie_tol_ms"] = float(tie_tol_ms)

    # Tie-aware rank agreement (fair: apply same tolerance to reference)
    if float(tie_tol_ms) > 0.0 and denom > 0:
        rank_ref_tie = compute_dense_rank_with_ties(ref_rel, ref_bool, tie_tol_ms=float(tie_tol_ms))
        rank_ours_tie = compute_dense_rank_with_ties(lag_ours, events_bool, tie_tol_ms=float(tie_tol_ms))
        out["rank_match_rate_tie"] = float(
            np.mean((rank_ours_tie == rank_ref_tie)[ref_bool & events_bool])
        )

        # Pairwise concordance with tolerance (order agreement)
        tol = float(tie_tol_ms) * 1e-3
        total_pairs = 0
        agree = 0
        for ei in range(ref_rel.shape[1]):
            mask = ref_bool[:, ei] & events_bool[:, ei] & np.isfinite(ref_rel[:, ei]) & np.isfinite(lag_ours[:, ei])
            idx = np.where(mask)[0]
            if idx.size < 2:
                continue
            rv = ref_rel[idx, ei]
            ov = lag_ours[idx, ei]
            # Compare all pairs (small: core=8; ok)
            for a in range(idx.size):
                for b in range(a + 1, idx.size):
                    dr = rv[a] - rv[b]
                    do = ov[a] - ov[b]
                    # treat near-zero as tie
                    if abs(dr) <= tol or abs(do) <= tol:
                        continue
                    total_pairs += 1
                    if (dr > 0 and do > 0) or (dr < 0 and do < 0):
                        agree += 1
        out["pairwise_concordance"] = float(agree / total_pairs) if total_pairs > 0 else float("nan")
        out["pairwise_n"] = float(total_pairs)
    return out


# =============================================================================
# NEW: TF Centroid Computation (wavelet + dynamic baseline)
# =============================================================================


def _compute_wavelet_power(
    x: np.ndarray,
    sfreq: float,
    freqs: np.ndarray,
    cycles: np.ndarray,
) -> np.ndarray:
    """Compute Morlet wavelet power (freqs x time)."""
    from scipy.signal import fftconvolve

    x = np.asarray(x, dtype=np.float64)
    sfreq = float(sfreq)
    freqs = np.asarray(freqs, dtype=np.float64)
    cycles = np.asarray(cycles, dtype=np.float64)
    power = np.zeros((freqs.shape[0], x.shape[0]), dtype=np.float64)

    for fi, f0 in enumerate(freqs):
        c = float(cycles[fi])
        sigma_t = c / (2.0 * np.pi * float(f0))
        half_support = int(np.ceil(3.0 * sigma_t * sfreq))
        tt = (np.arange(-half_support, half_support + 1, dtype=np.float64) / sfreq)
        w = np.exp(2j * np.pi * float(f0) * tt) * np.exp(-(tt**2) / (2.0 * sigma_t**2))
        wn = np.sqrt(np.sum(np.abs(w) ** 2)) + 1e-30
        w = w / wn
        y = fftconvolve(x, w, mode="same")
        power[fi, :] = (np.abs(y) ** 2).astype(np.float64, copy=False)

    return power


def compute_baseline_pool_spectra(
    *,
    data: np.ndarray,
    sfreq: float,
    pool_indices: np.ndarray,
    window_sec: float,
    freqs_hz: np.ndarray,
    n_cycles_vec: np.ndarray,
) -> np.ndarray:
    """
    Precompute baseline spectra for each channel and baseline window.
    Returns shape (n_channels, n_freqs, n_pool).
    """
    data = np.asarray(data, dtype=np.float64)
    pool_indices = np.asarray(pool_indices, dtype=np.int64)
    sfreq = float(sfreq)
    win = max(1, int(round(float(window_sec) * sfreq)))
    n_ch = data.shape[0]
    n_pool = pool_indices.shape[0]
    n_freqs = int(np.asarray(freqs_hz).shape[0])
    out = np.zeros((n_ch, n_freqs, n_pool), dtype=np.float32)
    if n_pool == 0:
        return out

    for ci in range(n_ch):
        for pj, s0 in enumerate(pool_indices):
            s1 = min(data.shape[1], int(s0) + win)
            if s1 <= s0:
                continue
            seg = data[ci, int(s0) : s1]
            power = _compute_wavelet_power(seg, sfreq, freqs_hz, n_cycles_vec)
            out[ci, :, pj] = np.mean(power, axis=1).astype(np.float32, copy=False)
    return out


def compute_tf_centroids(
    *,
    data: np.ndarray,
    sfreq: float,
    event_windows: np.ndarray,
    events_bool: np.ndarray,
    freqs_hz: np.ndarray,
    n_cycles_vec: np.ndarray,
    baseline_pool_starts: np.ndarray,
    baseline_pool_indices: np.ndarray,
    baseline_pool_spectra: Optional[np.ndarray] = None,
    baseline_n_select: int = 10,
    baseline_min_distance_sec: float = 2.0,
    baseline_window_sec: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute TF 2D centroids (time, freq) using wavelet + dynamic baseline.
    """
    data = np.asarray(data, dtype=np.float64)
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
    n_cycles_vec = np.asarray(n_cycles_vec, dtype=np.float64)
    n_ch, n_samples = data.shape
    n_events = event_windows.shape[0]

    tf_centroid_time = np.full((n_ch, n_events), np.nan, dtype=np.float64)
    tf_centroid_freq = np.full((n_ch, n_events), np.nan, dtype=np.float64)

    for ei in range(n_events):
        win_start, win_end = float(event_windows[ei, 0]), float(event_windows[ei, 1])
        i0 = int(round(win_start * sfreq))
        i1 = int(round(win_end * sfreq))
        i0 = max(0, i0)
        i1 = min(n_samples, i1)
        if i1 <= i0:
            continue

        event_time = 0.5 * (win_start + win_end)
        sel = _select_nearest_pool_indices(
            np.asarray(baseline_pool_starts, dtype=np.float64),
            event_time,
            n_select=int(baseline_n_select),
            min_distance_sec=float(baseline_min_distance_sec),
        )
        if sel.size == 0:
            continue

        for ci in range(n_ch):
            if not events_bool[ci, ei]:
                continue

            seg = data[ci, i0:i1]
            power = _compute_wavelet_power(seg, sfreq, freqs_hz, n_cycles_vec)

            if baseline_pool_spectra is not None and baseline_pool_spectra.size > 0:
                base_pool = np.asarray(baseline_pool_spectra[ci, :, sel], dtype=np.float64)
            else:
                base_pool = []
                win_b = max(1, int(round(float(baseline_window_sec) * sfreq)))
                for pj in np.asarray(baseline_pool_indices, dtype=np.int64)[sel]:
                    s0 = int(pj)
                    s1 = min(n_samples, s0 + win_b)
                    if s1 <= s0:
                        continue
                    base_seg = data[ci, s0:s1]
                    base_power = _compute_wavelet_power(base_seg, sfreq, freqs_hz, n_cycles_vec)
                    base_pool.append(np.mean(base_power, axis=1))
                if not base_pool:
                    continue
                base_pool = np.stack(base_pool, axis=1)

            base = np.median(base_pool, axis=1, keepdims=True) + 1e-30
            power_db = 10.0 * np.log10(power / base)
            W = np.maximum(power_db, 0.0)
            denom = float(np.sum(W))
            if denom <= 1e-30:
                continue

            t_rel = np.arange(seg.shape[0], dtype=np.float64) / float(sfreq)
            t_c = float(np.sum(W * t_rel[None, :]) / denom)
            f_c = float(np.sum(W * freqs_hz[:, None]) / denom)
            tf_centroid_time[ci, ei] = t_c
            tf_centroid_freq[ci, ei] = f_c

    return tf_centroid_time, tf_centroid_freq


# =============================================================================
# NEW: Group Analysis Results I/O
# =============================================================================


def save_group_analysis_results(
    npz_path: str,
    *,
    sfreq: float,
    band: str,
    ch_names: Sequence[str],
    event_windows: np.ndarray,
    centroid_time: np.ndarray,
    events_bool: np.ndarray,
    lag_raw: np.ndarray,
    lag_rank: np.ndarray,
    tf_centroid_time: Optional[np.ndarray] = None,
    tf_centroid_freq: Optional[np.ndarray] = None,
    baseline_pool_starts: Optional[np.ndarray] = None,
    baseline_pool_indices: Optional[np.ndarray] = None,
    baseline_params: Optional[Dict] = None,
    tf_freqs_hz: Optional[np.ndarray] = None,
    tf_n_cycles_vec: Optional[np.ndarray] = None,
    tf_n_cycles_mode: Optional[np.ndarray] = None,
    tf_freq_scale: Optional[np.ndarray] = None,
    tf_baseline_n_select: Optional[np.ndarray] = None,
    tf_baseline_min_distance_sec: Optional[np.ndarray] = None,
) -> str:
    """
    Save group analysis results to a standardized npz file.

    This is the canonical output of group_event_analysis.
    Visualization and network_analysis should read from this file.

    Parameters
    ----------
    npz_path : str
        Output file path.
    sfreq : float
        Sampling frequency.
    band : str
        'ripple' or 'fast_ripple'.
    ch_names : sequence of str
        Channel names (n_ch,).
    event_windows : np.ndarray
        (n_events, 2) [start, end] in seconds.
    centroid_time : np.ndarray
        (n_ch, n_events) Hilbert envelope centroid (seconds, relative to window start).
    events_bool : np.ndarray
        (n_ch, n_events) channel participation.
    lag_raw : np.ndarray
        (n_ch, n_events) relative lag (aligned to earliest channel).
    lag_rank : np.ndarray
        (n_ch, n_events) dense rank (0=earliest, -1=non-participant).
    tf_centroid_time : np.ndarray, optional
        (n_ch, n_events) TF time centroid.
    tf_centroid_freq : np.ndarray, optional
        (n_ch, n_events) TF frequency centroid.

    Returns
    -------
    npz_path : str
        The output path (for chaining).
    """
    n_ch = len(ch_names)
    n_events = event_windows.shape[0]
    window_sec = float(np.median(event_windows[:, 1] - event_windows[:, 0]))

    data = {
        "sfreq": np.array([float(sfreq)], dtype=np.float64),
        "band": np.array([str(band)], dtype=object),
        "ch_names": np.array([str(x) for x in ch_names], dtype=object),
        "window_sec": np.array([window_sec], dtype=np.float64),
        "n_events": np.array([n_events], dtype=np.int64),
        "n_channels": np.array([n_ch], dtype=np.int64),
        "event_windows": np.asarray(event_windows, dtype=np.float64),
        "centroid_time": np.asarray(centroid_time, dtype=np.float64),
        "events_bool": np.asarray(events_bool, dtype=bool),
        "lag_raw": np.asarray(lag_raw, dtype=np.float64),
        "lag_rank": np.asarray(lag_rank, dtype=np.int64),
    }

    if tf_centroid_time is not None:
        data["tf_centroid_time"] = np.asarray(tf_centroid_time, dtype=np.float64)
    if tf_centroid_freq is not None:
        data["tf_centroid_freq"] = np.asarray(tf_centroid_freq, dtype=np.float64)
    if baseline_pool_starts is not None:
        data["baseline_pool_starts"] = np.asarray(baseline_pool_starts, dtype=np.float32)
    if baseline_pool_indices is not None:
        data["baseline_pool_indices"] = np.asarray(baseline_pool_indices, dtype=np.int64)
    if baseline_params is not None:
        data["baseline_params"] = np.array([baseline_params], dtype=object)
    if tf_freqs_hz is not None:
        data["tf_freqs_hz"] = np.asarray(tf_freqs_hz, dtype=np.float64)
    if tf_n_cycles_vec is not None:
        data["tf_n_cycles_vec"] = np.asarray(tf_n_cycles_vec, dtype=np.float64)
    if tf_n_cycles_mode is not None:
        data["tf_n_cycles_mode"] = tf_n_cycles_mode
    if tf_freq_scale is not None:
        data["tf_freq_scale"] = tf_freq_scale
    if tf_baseline_n_select is not None:
        data["tf_baseline_n_select"] = tf_baseline_n_select
    if tf_baseline_min_distance_sec is not None:
        data["tf_baseline_min_distance_sec"] = tf_baseline_min_distance_sec

    np.savez_compressed(npz_path, **data)
    return npz_path


def load_group_analysis_results(npz_path: str) -> Dict:
    """
    Load group analysis results from npz file.

    Returns a dict with standardized keys.
    """
    d = np.load(npz_path, allow_pickle=True)

    out = {
        "sfreq": float(np.asarray(d["sfreq"]).ravel()[0]),
        "band": str(np.asarray(d["band"]).ravel()[0]),
        "ch_names": [str(x) for x in np.asarray(d["ch_names"]).tolist()],
        "window_sec": float(np.asarray(d["window_sec"]).ravel()[0]),
        "n_events": int(np.asarray(d["n_events"]).ravel()[0]),
        "n_channels": int(np.asarray(d["n_channels"]).ravel()[0]),
        "event_windows": np.asarray(d["event_windows"]),
        "centroid_time": np.asarray(d["centroid_time"]),
        "events_bool": np.asarray(d["events_bool"]).astype(bool),
        "lag_raw": np.asarray(d["lag_raw"]),
        "lag_rank": np.asarray(d["lag_rank"]),
    }

    if "tf_centroid_time" in d:
        out["tf_centroid_time"] = np.asarray(d["tf_centroid_time"])
    if "tf_centroid_freq" in d:
        out["tf_centroid_freq"] = np.asarray(d["tf_centroid_freq"])

    return out


def compute_and_save_group_analysis(
    *,
    edf_path: str,
    output_dir: Optional[str] = None,
    output_prefix: Optional[str] = None,
    packed_times_path: Optional[str] = None,
    gpu_npz_path: Optional[str] = None,
    core_channels: Optional[List[str]] = None,
    band: str = "ripple",
    reference: str = "bipolar",
    alias_bipolar_to_left: bool = True,
    crop_seconds: Optional[float] = None,
    use_gpu: bool = True,
    save_env_cache: bool = True,
    centroid_power: float = 2.0,
    tf_n_freqs: int = 180,
    tf_n_cycles: float = 4.0,
    tf_n_cycles_mode: str = "linear",
    tf_n_cycles_min: float = 3.0,
    tf_n_cycles_max: float = 10.0,
    tf_freq_scale: str = "log",
    baseline_window_sec: float = 2.0,
    baseline_step_sec: float = 1.0,
    baseline_n_select: int = 10,
    baseline_min_distance_sec: float = 2.0,
    baseline_line_length_q: float = 0.90,
    baseline_ripple_env_q: float = 0.80,
    seizure_ll_k: float = 6.0,
    seizure_rms_k: float = 6.0,
    seizure_min_duration_sec: float = 5.0,
) -> Dict[str, str]:
    """
    One-stop function to compute and save all group analysis results.

    This is the recommended entry point for Module 3.

    Workflow:
    1. Load EDF and preprocess (bipolar/filter)
    2. Get detections (from gpu_npz or run HFO detector)
    3. Get event windows (from packed_times or build from detections)
    4. Precompute envelope cache (optional save)
    5. Compute centroid, lag, rank
    6. Compute TF centroids
    7. Save groupAnalysis.npz

    Parameters
    ----------
    edf_path : str
        Path to EDF file.
    output_dir : str, optional
        Output directory. Default: same as EDF.
    output_prefix : str, optional
        File name prefix. Default: EDF basename.
    packed_times_path : str, optional
        Path to packedTimes.npy. If None, build windows from detections.
    gpu_npz_path : str, optional
        Path to *_gpu.npz. If None, run HFO detector.
    core_channels : list of str, optional
        Core channels to analyze. If None, use all channels.
    band : str
        'ripple' or 'fast_ripple'.
    reference : str
        'bipolar', 'car', or 'none'.
    alias_bipolar_to_left : bool
        If True, alias 'A1-A2' to 'A1'.
    crop_seconds : float, optional
        Limit analysis to first N seconds.
    use_gpu : bool
        Use GPU for envelope computation.
    save_env_cache : bool
        If True, save envelope cache.
    centroid_power : float
        Power for envelope centroid (default 2.0).
    tf_nperseg : int
        STFT window size for TF centroids.
    tf_noverlap : int
        STFT overlap for TF centroids.

    Returns
    -------
    dict with keys:
        'group_analysis_path': str
        'env_cache_path': str (if saved)
    """
    from pathlib import Path
    from .preprocessing import (
        SEEGPreprocessor,
        detect_seizure_onsets_from_data,
        save_seizure_onsets_json,
    )

    edf_path = str(edf_path)
    edf_stem = Path(edf_path).stem

    if output_dir is None:
        output_dir = str(Path(edf_path).parent)
    if output_prefix is None:
        output_prefix = edf_stem

    t_start = time.time()
    logger = get_run_logger(f"group_analysis_{output_prefix}")
    log_section(logger, "GROUP ANALYSIS START")
    logger.info("edf_path=%s", str(edf_path))
    logger.info("output_dir=%s", str(output_dir))
    logger.info("output_prefix=%s", str(output_prefix))
    logger.info("band=%s", str(band))
    logger.info("reference=%s", str(reference))
    logger.info("crop_seconds=%s", str(crop_seconds))
    logger.info("baseline_window_sec=%.3f", float(baseline_window_sec))
    logger.info("baseline_step_sec=%.3f", float(baseline_step_sec))
    logger.info("baseline_n_select=%d", int(baseline_n_select))

    # Determine freq band
    b = band.lower().strip()
    if b in ("ripple", "ripples"):
        freq_band = (80.0, 250.0)
    elif b in ("fast_ripple", "fast-ripple", "fr", "fast"):
        freq_band = (250.0, 500.0)
    else:
        raise ValueError(f"Unknown band='{band}'")

    # Step 1: Preprocess
    t_step = time.time()
    pre = SEEGPreprocessor(
        target_band="fast_ripple" if "fast" in b else "ripple",
        reference=reference,
        crop_seconds=crop_seconds,
        check_quality=False,
        use_gpu=False,
    )
    pre_out = pre.run(edf_path)
    data = np.asarray(pre_out.data)
    names = [str(x) for x in pre_out.ch_names]
    sfreq = float(pre_out.sfreq)
    logger.info("step=preprocess elapsed_sec=%.3f", float(time.time() - t_step))

    # Step 2: Alias bipolar names if needed
    if alias_bipolar_to_left:
        allowed = None
        if gpu_npz_path is not None:
            g = np.load(gpu_npz_path, allow_pickle=True)
            allowed = set(str(x).upper() for x in g["chns_names"].tolist())

        keep_idx: List[int] = []
        alias_names: List[str] = []
        for i, nm in enumerate(names):
            if "-" not in nm:
                continue
            left, right = nm.split("-", 1)
            left = left.strip().upper()
            right = right.strip().upper()
            if allowed is not None and (left not in allowed or right not in allowed):
                continue
            keep_idx.append(i)
            alias_names.append(left)

        data = data[keep_idx] if keep_idx else np.zeros((0, data.shape[1]), dtype=np.float64)
        names = alias_names

    # Step 3: Get detections
    t_step = time.time()
    if gpu_npz_path is not None:
        gpu = np.load(gpu_npz_path, allow_pickle=True)
        gpu_names = [str(x) for x in gpu["chns_names"].tolist()]
        name_to_idx = {n: i for i, n in enumerate(gpu_names)}
        dets: Dict[str, np.ndarray] = {}
        for ch in names:
            ch_up = ch.upper()
            if ch_up in name_to_idx:
                dets[ch] = np.asarray(gpu["whole_dets"][name_to_idx[ch_up]], dtype=np.float64)
            else:
                dets[ch] = np.zeros((0, 2), dtype=np.float64)
    else:
        from .hfo_detector import HFODetector, HFODetectionConfig
        cfg = HFODetectionConfig(algorithm="bqk", band=band, use_gpu=False)
        det = HFODetector(cfg)
        res = det.detect(data, sfreq=sfreq, ch_names=names)
        dets = _detections_dict_from_hfo_result(res)
    logger.info("step=detections elapsed_sec=%.3f", float(time.time() - t_step))

    # Step 4: Get event windows
    t_step = time.time()
    if packed_times_path is not None:
        packed = np.load(packed_times_path, allow_pickle=True)
        windows = build_windows_from_packed_times(packed)
        if crop_seconds is not None:
            windows = [w for w in windows if w.start < float(crop_seconds)]
        event_windows = np.array([[w.start, w.end] for w in windows], dtype=np.float64)
    else:
        window_sec = 0.5
        windows = build_windows_from_detections(dets, window_sec=window_sec)
        if crop_seconds is not None:
            windows = [w for w in windows if w.start < float(crop_seconds)]
        event_windows = np.array([[w.start, w.end] for w in windows], dtype=np.float64)
    logger.info("step=event_windows elapsed_sec=%.3f", float(time.time() - t_step))

    # Step 5: Filter to core channels if specified
    t_step = time.time()
    if core_channels is not None:
        core_set = set(str(x).upper() for x in core_channels)
        keep_idx = [i for i, n in enumerate(names) if n.upper() in core_set]
        data = data[keep_idx]
        names = [names[i] for i in keep_idx]
        dets = {n: dets.get(n, np.zeros((0, 2))) for n in names}
    logger.info("step=core_channels elapsed_sec=%.3f", float(time.time() - t_step))

    n_ch = len(names)
    n_events = len(windows)

    if n_ch == 0 or n_events == 0:
        raise ValueError(f"No channels ({n_ch}) or no events ({n_events}) to analyze.")

    # Step 6: Detect seizure onsets (subject-level) + ictal mask
    t_step = time.time()
    seizure = detect_seizure_onsets_from_data(
        data,
        sfreq,
        ll_k=float(seizure_ll_k),
        rms_k=float(seizure_rms_k),
        min_duration_sec=float(seizure_min_duration_sec),
    )
    ictal_mask = seizure.get("ictal_mask", None)
    save_seizure_onsets_json(
        subject_id=str(output_prefix),
        onsets_sec=seizure.get("onsets_sec", np.zeros((0,), dtype=np.float64)),
        offsets_sec=seizure.get("offsets_sec", np.zeros((0,), dtype=np.float64)),
        sfreq=sfreq,
        output_dir="results/seizure_onset",
        params={
            "ll_k": float(seizure_ll_k),
            "rms_k": float(seizure_rms_k),
            "min_duration_sec": float(seizure_min_duration_sec),
            "ll_win_sec": 1.0,
            "ll_step_sec": 0.2,
            "rms_win_sec": 1.0,
            "rms_step_sec": 0.2,
        },
    )
    logger.info(
        "step=seizure_onsets elapsed_sec=%.3f onsets=%d",
        float(time.time() - t_step),
        int(len(seizure.get("onsets_sec", []))),
    )

    # Step 7: Compute envelope (bandpass + Hilbert)
    t_step = time.time()
    env = np.zeros((n_ch, data.shape[1]), dtype=np.float32)
    x_band = np.zeros_like(data, dtype=np.float32)

    for ci in range(n_ch):
        env[ci] = _maybe_gpu_envelope(data[ci], sfreq, freq_band, use_gpu=use_gpu).astype(np.float32)

    # Bandpass for TF
    try:
        from scipy.signal import butter, sosfiltfilt
        nyq = sfreq / 2.0
        lo, hi = freq_band
        hi2 = hi if hi < nyq else max(nyq - 1.0, lo + 1.0)
        sos = butter(4, [lo / nyq, hi2 / nyq], btype="bandpass", output="sos")
        for ci in range(n_ch):
            x_band[ci] = sosfiltfilt(sos, data[ci]).astype(np.float32)
    except Exception:
        x_band = data.astype(np.float32)
    logger.info("step=envelope elapsed_sec=%.3f", float(time.time() - t_step))

    # Step 8: Compute centroids
    t_step = time.time()
    centroids, events_bool = compute_centroid_matrix_from_envelope_cache(
        windows=windows,
        detections=dets,
        ch_names=names,
        env=env,
        env_ch_names=names,
        sfreq=sfreq,
        start_sec=0.0,
        centroid_power=centroid_power,
    )
    logger.info("step=centroid elapsed_sec=%.3f", float(time.time() - t_step))

    # Step 9: Compute lag/rank
    t_step = time.time()
    lag_raw, lag_rank = lag_rank_from_centroids(centroids, events_bool, align="first_centroid")
    logger.info("step=lag_rank elapsed_sec=%.3f", float(time.time() - t_step))

    # Step 10: Build baseline pool
    t_step = time.time()
    pool_starts, pool_indices, pool_params = compute_baseline_pool_indices(
        data=data,
        env=env,
        sfreq=sfreq,
        detections=dets,
        ictal_mask=ictal_mask,
        window_sec=float(baseline_window_sec),
        step_sec=float(baseline_step_sec),
        line_length_q=float(baseline_line_length_q),
        ripple_env_q=float(baseline_ripple_env_q),
    )
    logger.info(
        "step=baseline_pool elapsed_sec=%.3f pool_windows=%d",
        float(time.time() - t_step),
        int(pool_starts.shape[0]),
    )

    # Step 11: Wavelet params (shared across analysis + visualization)
    freqs_hz = bqk_utils.get_wavelet_freqs(freq_band[0], freq_band[1], int(tf_n_freqs), scale=tf_freq_scale)
    n_cycles_mode = str(tf_n_cycles_mode).lower().strip()
    if n_cycles_mode not in ("fixed", "linear", "f_over_2"):
        raise ValueError("tf_n_cycles_mode must be 'fixed', 'linear', or 'f_over_2'.")
    if n_cycles_mode == "fixed":
        n_cycles_vec = np.full(freqs_hz.shape[0], float(tf_n_cycles), dtype=np.float64)
    elif n_cycles_mode == "linear":
        cmin = float(tf_n_cycles_min)
        cmax = float(tf_n_cycles_max)
        if cmin <= 0 or cmax <= 0 or cmax < cmin:
            raise ValueError("tf_n_cycles_min/max must be >0 and max>=min.")
        n_cycles_vec = np.linspace(cmin, cmax, freqs_hz.shape[0]).astype(np.float64)
    else:
        n_cycles_vec = (freqs_hz / 2.0).astype(np.float64)

    # Step 12: Precompute baseline spectra (in-memory)
    t_step = time.time()
    baseline_pool_spectra = compute_baseline_pool_spectra(
        data=data,
        sfreq=sfreq,
        pool_indices=pool_indices,
        window_sec=float(baseline_window_sec),
        freqs_hz=freqs_hz,
        n_cycles_vec=n_cycles_vec,
    )
    logger.info("step=baseline_spectra elapsed_sec=%.3f", float(time.time() - t_step))

    # Step 13: Compute TF centroids (wavelet + dynamic baseline)
    t_step = time.time()
    tf_time, tf_freq = compute_tf_centroids(
        data=data,
        sfreq=sfreq,
        event_windows=event_windows,
        events_bool=events_bool,
        freqs_hz=freqs_hz,
        n_cycles_vec=n_cycles_vec,
        baseline_pool_starts=pool_starts,
        baseline_pool_indices=pool_indices,
        baseline_pool_spectra=baseline_pool_spectra,
        baseline_n_select=int(baseline_n_select),
        baseline_min_distance_sec=float(baseline_min_distance_sec),
        baseline_window_sec=float(baseline_window_sec),
    )
    logger.info("step=tf_centroid elapsed_sec=%.3f", float(time.time() - t_step))

    # Step 14: Save results
    t_step = time.time()
    out_paths: Dict[str, str] = {}

    group_path = f"{output_dir}/{output_prefix}_groupAnalysis.npz"
    save_group_analysis_results(
        group_path,
        sfreq=sfreq,
        band=band,
        ch_names=names,
        event_windows=event_windows,
        centroid_time=centroids,
        events_bool=events_bool,
        lag_raw=lag_raw,
        lag_rank=lag_rank,
        tf_centroid_time=tf_time,
        tf_centroid_freq=tf_freq,
        baseline_pool_starts=pool_starts,
        baseline_pool_indices=pool_indices,
        baseline_params=pool_params,
        tf_freqs_hz=freqs_hz,
        tf_n_cycles_vec=n_cycles_vec,
        tf_n_cycles_mode=np.array([str(tf_n_cycles_mode)], dtype=object),
        tf_freq_scale=np.array([str(tf_freq_scale)], dtype=object),
        tf_baseline_n_select=np.array([int(baseline_n_select)], dtype=np.int64),
        tf_baseline_min_distance_sec=np.array([float(baseline_min_distance_sec)], dtype=np.float64),
    )
    out_paths["group_analysis_path"] = group_path

    if save_env_cache:
        env_path = f"{output_dir}/{output_prefix}_envCache_{band}_{reference}.npz"
        np.savez_compressed(
            env_path,
            env=env,
            x_band=x_band,
            sfreq=np.array([sfreq], dtype=np.float64),
            ch_names=np.array(names, dtype=object),
            band=np.array([band], dtype=object),
            reference=np.array([reference], dtype=object),
            crop_seconds=np.array([crop_seconds if crop_seconds else data.shape[1] / sfreq], dtype=np.float64),
            start_sec=np.array([0.0], dtype=np.float64),
            alias_bipolar_to_left=np.array([alias_bipolar_to_left], dtype=bool),
            has_x_band=np.array([True], dtype=bool),
        )
        out_paths["env_cache_path"] = env_path

    logger.info("step=save elapsed_sec=%.3f", float(time.time() - t_step))
    log_section(logger, "GROUP ANALYSIS SUMMARY")
    logger.info("channels=%d", int(n_ch))
    logger.info("events=%d", int(n_events))
    logger.info("group_analysis_path=%s", str(group_path))
    if save_env_cache:
        logger.info("env_cache_path=%s", str(env_path))
    logger.info("elapsed_sec=%.3f", float(time.time() - t_start))
    log_section(logger, "GROUP ANALYSIS END")

    return out_paths

