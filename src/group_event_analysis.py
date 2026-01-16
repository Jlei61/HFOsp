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
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .preprocessing import ElectrodeParser
from .utils import bqk_utils


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
# NEW: TF Centroid Computation (moved from visualization.py)
# =============================================================================


def compute_tf_centroids(
    *,
    x_band: np.ndarray,
    sfreq: float,
    event_windows: np.ndarray,
    events_bool: np.ndarray,
    freq_band: Tuple[float, float] = (80.0, 250.0),
    nperseg: int = 128,
    noverlap: int = 96,
    centroid_power: str = "power2",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute TF 2D centroids (time, freq) for each (channel, event).

    This function performs STFT on bandpassed signal and computes energy-weighted
    centroid in time-frequency space.

    Parameters
    ----------
    x_band : np.ndarray
        Bandpassed signal (n_channels, n_samples).
    sfreq : float
        Sampling frequency.
    event_windows : np.ndarray
        (n_events, 2) [start, end] in seconds.
    events_bool : np.ndarray
        (n_channels, n_events) bool, True if channel participates.
    freq_band : tuple
        (f_low, f_high) for TF analysis.
    nperseg : int
        STFT window size.
    noverlap : int
        STFT overlap.
    centroid_power : str
        'power' or 'power2' for weighting.

    Returns
    -------
    tf_centroid_time : np.ndarray
        (n_channels, n_events) time centroid relative to window start (seconds).
    tf_centroid_freq : np.ndarray
        (n_channels, n_events) frequency centroid (Hz).
    """
    from scipy.signal import stft

    n_ch, n_samples = x_band.shape
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

        for ci in range(n_ch):
            if not events_bool[ci, ei]:
                continue

            seg = x_band[ci, i0:i1]
            nwin = seg.shape[0]
            nps = min(int(nperseg), max(8, nwin))
            nov = min(int(noverlap), max(0, nps - 1))

            f, t, Z = stft(seg, fs=sfreq, nperseg=nps, noverlap=nov, boundary=None)
            P = (np.abs(Z) ** 2).astype(np.float64)

            # Mask to freq_band
            band_mask = (f >= freq_band[0]) & (f <= freq_band[1])
            f_band = f[band_mask]
            P_band = P[band_mask, :]

            if P_band.size == 0:
                continue

            # Apply power weighting
            if centroid_power == "power2":
                W = P_band ** 2
            elif centroid_power == "power":
                W = P_band
            else:
                raise ValueError(f"centroid_power must be 'power' or 'power2', got '{centroid_power}'")

            denom = float(np.sum(W))
            if denom <= 1e-30:
                continue

            # 2D centroid
            t_c = float(np.sum(W * t[None, :]) / denom)
            f_c = float(np.sum(W * f_band[:, None]) / denom)

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
    tf_nperseg: int = 128,
    tf_noverlap: int = 96,
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
    from .preprocessing import SEEGPreprocessor

    edf_path = str(edf_path)
    edf_stem = Path(edf_path).stem

    if output_dir is None:
        output_dir = str(Path(edf_path).parent)
    if output_prefix is None:
        output_prefix = edf_stem

    # Determine freq band
    b = band.lower().strip()
    if b in ("ripple", "ripples"):
        freq_band = (80.0, 250.0)
    elif b in ("fast_ripple", "fast-ripple", "fr", "fast"):
        freq_band = (250.0, 500.0)
    else:
        raise ValueError(f"Unknown band='{band}'")

    # Step 1: Preprocess
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

    # Step 4: Get event windows
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

    # Step 5: Filter to core channels if specified
    if core_channels is not None:
        core_set = set(str(x).upper() for x in core_channels)
        keep_idx = [i for i, n in enumerate(names) if n.upper() in core_set]
        data = data[keep_idx]
        names = [names[i] for i in keep_idx]
        dets = {n: dets.get(n, np.zeros((0, 2))) for n in names}

    n_ch = len(names)
    n_events = len(windows)

    if n_ch == 0 or n_events == 0:
        raise ValueError(f"No channels ({n_ch}) or no events ({n_events}) to analyze.")

    # Step 6: Compute envelope (bandpass + Hilbert)
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

    # Step 7: Compute centroids
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

    # Step 8: Compute lag/rank
    lag_raw, lag_rank = lag_rank_from_centroids(centroids, events_bool, align="first_centroid")

    # Step 9: Compute TF centroids
    tf_time, tf_freq = compute_tf_centroids(
        x_band=x_band,
        sfreq=sfreq,
        event_windows=event_windows,
        events_bool=events_bool,
        freq_band=freq_band,
        nperseg=tf_nperseg,
        noverlap=tf_noverlap,
        centroid_power="power2",
    )

    # Step 10: Save results
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

    return out_paths

