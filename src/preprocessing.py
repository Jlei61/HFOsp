"""
SEEG Data Preprocessing Pipeline for HFO Analysis

This module handles:
- EDF file loading with electrode name parsing
- Re-referencing strategies:
  - Bipolar (adjacent contacts on same shaft)
  - Common Average Reference (CAR) per shaft
  - Detection of pre-bipolar data (A1-A2 labeled as A1)
- Resampling with Nyquist-aware warnings
- Notch and bandpass filtering
- Channel quality assessment
- GPU acceleration support (optional, via CuPy)

Author: HFOsp Team
Date: 2026-01-14
"""

import json
import time
import numpy as np
import mne
import re
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt
from scipy import signal
from .utils.logging_utils import get_run_logger, log_section

# GPU support (optional)
try:
    import cupy as cp
    from cupyx.scipy.signal import sosfilt as cupy_sosfilt
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    cp = None


@dataclass
class PreprocessingResult:
    """Container for preprocessing output"""
    data: np.ndarray              # (n_channels, n_samples)
    sfreq: float                  # Sampling frequency
    ch_names: List[str]           # Channel names (monopolar or bipolar)
    original_ch_names: List[str]  # Original EDF channel names
    bipolar_pairs: Optional[List[Tuple[str, str]]] = None  # (ch1, ch2) pairs if bipolar
    shaft_mapping: Dict[str, List[str]] = field(default_factory=dict)
    bad_channels: List[str] = field(default_factory=list)
    quality_scores: Dict[str, float] = field(default_factory=dict)
    reference_type: str = 'unknown'  # 'monopolar', 'bipolar', 'car'
    excluded_channels: List[str] = field(default_factory=list)
    used_gpu: bool = False        # Whether GPU was used for processing
    outer_contact_mode: str = "none"
    dropped_outer_contacts: List[str] = field(default_factory=list)


def _moving_sum_1d(x: np.ndarray, win: int) -> np.ndarray:
    """Fast moving sum for 1D arrays."""
    if win <= 1:
        return x.astype(np.float64, copy=False)
    x = x.astype(np.float64, copy=False)
    c = np.cumsum(x)
    out = c[win - 1 :] - np.concatenate(([0.0], c[:-win]))
    return out


def _robust_z(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Robust z-score using median and MAD."""
    x = np.asarray(x, dtype=np.float64)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) + eps
    return (x - med) / mad


def detect_seizure_onsets_from_data(
    data: np.ndarray,
    sfreq: float,
    *,
    ll_win_sec: float = 1.0,
    ll_step_sec: float = 0.2,
    ll_k: float = 6.0,
    rms_win_sec: float = 1.0,
    rms_step_sec: float = 0.2,
    rms_k: float = 6.0,
    min_duration_sec: float = 5.0,
) -> Dict[str, np.ndarray]:
    """
    Detect ictal segments using channel-averaged line length + RMS.
    Returns onset/offset times and sample-level ictal mask.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("data must be 2D (n_channels, n_samples)")
    sfreq = float(sfreq)
    if sfreq <= 0:
        raise ValueError("sfreq must be > 0")

    n_samples = data.shape[1]
    x_mean = np.mean(data, axis=0)

    def _windowed_values(x: np.ndarray, win_sec: float, step_sec: float, func) -> Tuple[np.ndarray, np.ndarray]:
        win = max(1, int(round(win_sec * sfreq)))
        step = max(1, int(round(step_sec * sfreq)))
        if n_samples < win:
            return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)
        starts = np.arange(0, n_samples - win + 1, step, dtype=np.int64)
        vals = np.zeros((starts.shape[0],), dtype=np.float64)
        for i, s in enumerate(starts):
            seg = x[s : s + win]
            vals[i] = func(seg)
        times = starts.astype(np.float64) / sfreq
        return times, vals

    ll_t, ll_vals = _windowed_values(
        x_mean, ll_win_sec, ll_step_sec, lambda seg: float(np.sum(np.abs(np.diff(seg))))
    )
    rms_t, rms_vals = _windowed_values(
        x_mean, rms_win_sec, rms_step_sec, lambda seg: float(np.sqrt(np.mean(seg**2))))

    if ll_vals.size == 0 or rms_vals.size == 0:
        return {
            "onsets_sec": np.zeros((0,), dtype=np.float64),
            "offsets_sec": np.zeros((0,), dtype=np.float64),
            "ictal_mask": np.zeros((n_samples,), dtype=bool),
        }

    ll_z = _robust_z(ll_vals)
    rms_z = _robust_z(rms_vals)

    ll_flag = ll_z >= float(ll_k)
    rms_flag = rms_z >= float(rms_k)

    # Align to a common timeline by mapping onto the denser grid (use ll_t).
    rms_interp = np.interp(ll_t, rms_t, rms_flag.astype(np.float64), left=0.0, right=0.0) >= 0.5
    ictal_flag = ll_flag | rms_interp

    # Enforce min duration on ll timeline
    runs = []
    cur = None
    for i, val in enumerate(ictal_flag):
        if val and cur is None:
            cur = i
        elif not val and cur is not None:
            runs.append((cur, i))
            cur = None
    if cur is not None:
        runs.append((cur, len(ictal_flag)))

    onsets = []
    offsets = []
    ictal_mask = np.zeros((n_samples,), dtype=bool)
    for s_idx, e_idx in runs:
        t0 = ll_t[s_idx]
        t1 = ll_t[e_idx - 1] + float(ll_step_sec) + float(ll_win_sec)
        if (t1 - t0) < float(min_duration_sec):
            continue
        onsets.append(float(t0))
        offsets.append(float(t1))
        i0 = max(0, int(round(t0 * sfreq)))
        i1 = min(n_samples, int(round(t1 * sfreq)))
        if i1 > i0:
            ictal_mask[i0:i1] = True

    return {
        "onsets_sec": np.asarray(onsets, dtype=np.float64),
        "offsets_sec": np.asarray(offsets, dtype=np.float64),
        "ictal_mask": ictal_mask,
    }


def save_seizure_onsets_json(
    *,
    subject_id: str,
    onsets_sec: np.ndarray,
    offsets_sec: np.ndarray,
    sfreq: float,
    output_dir: Union[str, Path] = "results/seizure_onset",
    params: Optional[Dict] = None,
    method: str = "line_length_rms_channel_avg",
) -> str:
    """
    Save subject-level seizure onset JSON.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{subject_id}.json"
    payload = {
        "subject_id": str(subject_id),
        "sfreq": float(sfreq),
        "method": str(method),
        "params": params or {},
        "onsets_sec": [float(x) for x in np.asarray(onsets_sec).ravel().tolist()],
        "offsets_sec": [float(x) for x in np.asarray(offsets_sec).ravel().tolist()],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
    return str(out_path)


# =====================================================================
# PR2 — Streaming seizure detection (direct binary EDF, NFS-friendly)
# =====================================================================

def read_edf_start_time(edf_path: Union[str, Path]) -> float:
    """Read recording start epoch from EDF header bytes 168–184 (UTC assumed)."""
    edf_path = Path(edf_path)
    with open(edf_path, "rb") as f:
        hdr = f.read(256)
    if len(hdr) < 184:
        raise ValueError(f"EDF header too short: {edf_path}")
    date_str = hdr[168:176].decode("ascii", errors="ignore").strip()
    time_str = hdr[176:184].decode("ascii", errors="ignore").strip()
    dd, mm, yy = date_str.split(".")
    hh, mi, ss = time_str.split(".")
    yy_int = int(yy)
    year = 2000 + yy_int if yy_int < 85 else 1900 + yy_int
    dt = datetime(year, int(mm), int(dd), int(hh), int(mi), int(ss),
                  tzinfo=timezone.utc)
    return dt.timestamp()


def _parse_edf_header_for_streaming(edf_path: Path) -> Dict:
    """Parse EDF header and return everything needed for binary streaming."""
    with open(edf_path, "rb") as f:
        fixed = f.read(256)
        header_n_bytes = int(float(fixed[184:192].decode("ascii", errors="ignore").strip()))
        n_records_raw = int(float(fixed[236:244].decode("ascii", errors="ignore").strip()))
        record_dur_str = fixed[244:252].decode("ascii", errors="ignore").strip()
        record_duration = float(record_dur_str) if record_dur_str else 1.0
        n_signals = int(float(fixed[252:256].decode("ascii", errors="ignore").strip()))
        var = f.read(header_n_bytes - 256)

    c = 0
    labels = [var[c + i * 16: c + (i + 1) * 16].decode("ascii", errors="ignore").strip()
              for i in range(n_signals)]
    c += 16 * n_signals
    c += 80 * n_signals   # transducer
    c += 8 * n_signals    # phys dim
    phys_min = [float(var[c + i * 8: c + (i + 1) * 8].decode("ascii", errors="ignore").strip() or "0")
                for i in range(n_signals)]
    c += 8 * n_signals
    phys_max = [float(var[c + i * 8: c + (i + 1) * 8].decode("ascii", errors="ignore").strip() or "0")
                for i in range(n_signals)]
    c += 8 * n_signals
    dig_min = [float(var[c + i * 8: c + (i + 1) * 8].decode("ascii", errors="ignore").strip() or "-32768")
               for i in range(n_signals)]
    c += 8 * n_signals
    dig_max = [float(var[c + i * 8: c + (i + 1) * 8].decode("ascii", errors="ignore").strip() or "32767")
               for i in range(n_signals)]
    c += 8 * n_signals
    c += 80 * n_signals   # prefilter
    nsamp = [int(float(var[c + i * 8: c + (i + 1) * 8].decode("ascii", errors="ignore").strip() or "0"))
             for i in range(n_signals)]

    gains = np.array([(phys_max[i] - phys_min[i]) / max(1e-12, dig_max[i] - dig_min[i])
                       for i in range(n_signals)], dtype=np.float64)
    offsets = np.array([phys_min[i] - dig_min[i] * gains[i]
                        for i in range(n_signals)], dtype=np.float64)

    seeg_idx = [i for i, lab in enumerate(labels) if ElectrodeParser.is_valid_seeg(lab)]
    if not seeg_idx:
        raise ValueError(f"No SEEG channels in {edf_path.name}")

    sprs = [nsamp[i] for i in seeg_idx]
    if len(set(sprs)) != 1:
        raise ValueError("SEEG channels have differing sample rates — cannot stream")
    spr = sprs[0]

    sample_offsets = np.array([sum(nsamp[:i]) for i in seeg_idx], dtype=np.int64)
    record_total_samples = sum(nsamp)
    record_total_bytes = record_total_samples * 2

    n_records = n_records_raw
    if n_records <= 0:
        file_bytes = edf_path.stat().st_size
        n_records = max(0, (file_bytes - header_n_bytes) // record_total_bytes)

    return {
        "header_n_bytes": header_n_bytes,
        "n_records": n_records,
        "record_duration": record_duration,
        "record_total_bytes": record_total_bytes,
        "seeg_idx": seeg_idx,
        "n_seeg": len(seeg_idx),
        "spr": spr,
        "sample_offsets": sample_offsets,
        "gains": gains[seeg_idx],
        "offsets": offsets[seeg_idx],
        "sfreq": float(spr / record_duration),
    }


def _stream_edf_channel_mean(edf_path: Path) -> Tuple[np.ndarray, float, int]:
    """
    Read EDF binary sequentially, return SEEG channel mean.

    Single pass, no seeks, ~2.5 MB peak per record — optimal for NFS.
    """
    h = _parse_edf_header_for_streaming(edf_path)
    n_rec = h["n_records"]
    spr = h["spr"]
    total = n_rec * spr
    x_mean = np.empty(total, dtype=np.float64)
    gains = h["gains"][:, None]       # (n_ch, 1)
    offs = h["offsets"][:, None]      # (n_ch, 1)
    so = h["sample_offsets"]
    rtb = h["record_total_bytes"]

    with open(edf_path, "rb") as f:
        f.seek(h["header_n_bytes"])
        for rec in range(n_rec):
            raw = f.read(rtb)
            if len(raw) < rtb:
                x_mean = x_mean[: rec * spr]
                break
            all_i16 = np.frombuffer(raw, dtype="<i2")
            ch_data = np.stack([all_i16[int(o): int(o) + spr] for o in so])
            physical = ch_data.astype(np.float64) * gains + offs
            start = rec * spr
            x_mean[start: start + spr] = physical.mean(axis=0)

    return x_mean, h["sfreq"], h["n_seeg"]


def detect_seizure_streaming(
    edf_path: Union[str, Path],
    *,
    ll_win_sec: float = 1.0,
    ll_step_sec: float = 0.2,
    ll_k: float = 6.0,
    rms_win_sec: float = 1.0,
    rms_step_sec: float = 0.2,
    rms_k: float = 6.0,
    min_duration_sec: float = 5.0,
    merge_gap_sec: float = 10.0,
    combine_mode: str = "and",
) -> Dict:
    """
    Streaming seizure detector — reads EDF binary directly (no mne).

    Single sequential pass → NFS-friendly, peak memory ≈ one data-record.
    Returns onset/offset times plus LL / RMS traces for plotting.
    """
    edf_path = Path(edf_path)
    if not edf_path.exists():
        raise FileNotFoundError(edf_path)

    x_mean, sfreq, n_channels = _stream_edf_channel_mean(edf_path)
    n_samples = len(x_mean)

    ll_win = max(1, int(round(ll_win_sec * sfreq)))
    ll_step = max(1, int(round(ll_step_sec * sfreq)))
    rms_win = max(1, int(round(rms_win_sec * sfreq)))
    rms_step = max(1, int(round(rms_step_sec * sfreq)))

    diff_abs = np.abs(np.diff(x_mean))
    ll_full = _moving_sum_1d(diff_abs, ll_win)
    ll_idx = np.arange(0, len(ll_full), ll_step, dtype=np.int64)
    ll_vals = ll_full[ll_idx]
    ll_t = ll_idx.astype(np.float64) / sfreq

    x2 = x_mean ** 2
    rms_sum = _moving_sum_1d(x2, rms_win)
    rms_full = np.sqrt(rms_sum / rms_win)
    rms_idx = np.arange(0, len(rms_full), rms_step, dtype=np.int64)
    rms_vals = rms_full[rms_idx]
    rms_t = rms_idx.astype(np.float64) / sfreq

    ll_z = _robust_z(ll_vals)
    rms_z = _robust_z(rms_vals)

    ll_flag = ll_z >= float(ll_k)
    rms_interp_z = np.interp(ll_t, rms_t, rms_z)
    rms_flag = rms_interp_z >= float(rms_k)

    combine_mode = str(combine_mode).strip().lower()
    if combine_mode == "and":
        ictal_flag = ll_flag & rms_flag
    elif combine_mode == "or":
        ictal_flag = ll_flag | rms_flag
    elif combine_mode == "sum":
        ictal_flag = (ll_z + rms_interp_z) >= float(ll_k + rms_k)
    else:
        raise ValueError("combine_mode must be 'and', 'or', or 'sum'")

    runs = _flag_to_runs(ictal_flag)
    merged = _merge_close_runs(runs, ll_t, merge_gap_sec)

    onsets: List[float] = []
    offsets_list: List[float] = []
    ictal_mask = np.zeros(n_samples, dtype=bool)
    for s_idx, e_idx in merged:
        t0 = float(ll_t[s_idx])
        t1 = float(ll_t[min(e_idx - 1, len(ll_t) - 1)] + ll_step_sec + ll_win_sec)
        if (t1 - t0) < float(min_duration_sec):
            continue
        onsets.append(t0)
        offsets_list.append(t1)
        i0 = max(0, int(round(t0 * sfreq)))
        i1 = min(n_samples, int(round(t1 * sfreq)))
        if i1 > i0:
            ictal_mask[i0:i1] = True

    peak_mb = round(n_samples * 8 / (1024 ** 2), 1)
    return {
        "onsets_sec": np.asarray(onsets, dtype=np.float64),
        "offsets_sec": np.asarray(offsets_list, dtype=np.float64),
        "ictal_mask": ictal_mask,
        "ll_t": ll_t, "ll_z": ll_z,
        "rms_t": rms_t, "rms_z": rms_z,
        "ll_k": float(ll_k), "rms_k": float(rms_k),
        "combine_mode": combine_mode,
        "sfreq": sfreq, "n_channels": n_channels,
        "duration_sec": float(n_samples / sfreq),
        "peak_mem_est_mb": peak_mb,
    }


def _flag_to_runs(flag: np.ndarray) -> List[Tuple[int, int]]:
    """Boolean flag → list of (start_idx, end_idx) contiguous True runs."""
    runs: List[Tuple[int, int]] = []
    cur = None
    for i, v in enumerate(flag):
        if v and cur is None:
            cur = i
        elif not v and cur is not None:
            runs.append((cur, i))
            cur = None
    if cur is not None:
        runs.append((cur, len(flag)))
    return runs


def _merge_close_runs(
    runs: List[Tuple[int, int]],
    time_axis: np.ndarray,
    gap_sec: float,
) -> List[Tuple[int, int]]:
    """Merge consecutive runs whose gap ≤ *gap_sec*."""
    if not runs:
        return []
    merged = [runs[0]]
    for s, e in runs[1:]:
        prev_s, prev_e = merged[-1]
        if float(time_axis[s] - time_axis[min(prev_e - 1, len(time_axis) - 1)]) <= gap_sec:
            merged[-1] = (prev_s, e)
        else:
            merged.append((s, e))
    return merged


def match_seizure_intervals(
    manual: List[Tuple[float, float]],
    detected: List[Tuple[float, float]],
    *,
    onset_tolerance_sec: float = 30.0,
) -> Dict:
    """
    Match detected vs manual seizure intervals via overlap.

    Returns dict: tp (list of match dicts), fp, fn, recall, precision, f1.
    """
    n_man = len(manual)
    n_det = len(detected)
    if n_man == 0 and n_det == 0:
        return {"tp": [], "fp": [], "fn": [], "recall": 1.0, "precision": 1.0, "f1": 1.0}
    if n_man == 0:
        return {"tp": [], "fp": list(range(n_det)), "fn": [],
                "recall": 1.0, "precision": 0.0, "f1": 0.0}
    if n_det == 0:
        return {"tp": [], "fp": [], "fn": list(range(n_man)),
                "recall": 0.0, "precision": 0.0, "f1": 0.0}

    matched_det: set = set()
    matched_man: set = set()
    tp: List[Dict] = []

    for mi, (m_on, m_off) in enumerate(manual):
        best_di = -1
        best_dist = float("inf")
        for di, (d_on, d_off) in enumerate(detected):
            if di in matched_det:
                continue
            if min(m_off, d_off) - max(m_on, d_on) > 0:
                dist = abs(d_on - m_on)
                if dist < best_dist:
                    best_dist = dist
                    best_di = di
        if best_di >= 0:
            d_on, d_off = detected[best_di]
            tp.append({"manual_idx": mi, "detected_idx": best_di,
                        "onset_err": float(d_on - m_on),
                        "offset_err": float(d_off - m_off)})
            matched_det.add(best_di)
            matched_man.add(mi)

    fp = [di for di in range(n_det) if di not in matched_det]
    fn = [mi for mi in range(n_man) if mi not in matched_man]
    n_tp = len(tp)
    recall = n_tp / n_man if n_man else 0.0
    precision = n_tp / n_det if n_det else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "recall": recall, "precision": precision, "f1": f1}


def _parse_fixed_width_ascii_int(raw: bytes, default: int = 0) -> int:
    txt = raw.decode("ascii", errors="ignore").strip()
    if not txt:
        return int(default)
    return int(float(txt))


def _split_fixed_fields(blob: bytes, width: int, count: int) -> List[bytes]:
    fields: List[bytes] = []
    for i in range(int(count)):
        start = i * int(width)
        end = start + int(width)
        fields.append(blob[start:end])
    return fields


def _parse_tal_annotations(raw_text: str) -> List[Tuple[float, float, str]]:
    out: List[Tuple[float, float, str]] = []
    for tal in raw_text.split("\x00"):
        if not tal or "\x14" not in tal:
            continue
        parts = tal.split("\x14")
        if not parts:
            continue
        onset_dur = parts[0]
        if not onset_dur:
            continue
        if "\x15" in onset_dur:
            onset_str, duration_str = onset_dur.split("\x15", 1)
        else:
            onset_str, duration_str = onset_dur, ""
        onset_str = onset_str.strip()
        duration_str = duration_str.strip()
        if not onset_str:
            continue
        try:
            onset = float(onset_str)
        except ValueError:
            continue
        try:
            duration = float(duration_str) if duration_str else 0.0
        except ValueError:
            duration = 0.0
        for desc in parts[1:]:
            desc_clean = desc.strip()
            if not desc_clean:
                continue
            out.append((float(onset), float(duration), desc_clean))
    return out


def _normalize_annotation_label(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def _merge_time_intervals(
    intervals: List[Tuple[float, float]],
    *,
    gap_sec: float = 0.0,
) -> List[Tuple[float, float]]:
    """Sort and merge overlapping / near-adjacent intervals."""
    if not intervals:
        return []
    gap_sec = max(0.0, float(gap_sec))
    cleaned = sorted(
        (float(start), float(end))
        for start, end in intervals
        if float(end) > float(start)
    )
    if not cleaned:
        return []
    merged: List[List[float]] = [[cleaned[0][0], cleaned[0][1]]]
    for start, end in cleaned[1:]:
        prev = merged[-1]
        if start <= (prev[1] + gap_sec):
            prev[1] = max(prev[1], end)
        else:
            merged.append([start, end])
    return [(float(start), float(end)) for start, end in merged]


def _maybe_collapse_utf16_annotation_bytes(payload_bytes: bytes) -> bytes:
    """
    Some EDF writers store TAL text as 16-bit little-endian characters.
    Detect that layout using only non-padding byte pairs.
    """
    if len(payload_bytes) < 2:
        return payload_bytes
    even_bytes = payload_bytes[0::2]
    odd_bytes = payload_bytes[1::2]
    active_pairs = [
        (int(lo), int(hi))
        for lo, hi in zip(even_bytes, odd_bytes)
        if lo != 0 or hi != 0
    ]
    if not active_pairs:
        return payload_bytes
    odd_zero_ratio = sum(1 for _, hi in active_pairs if hi == 0) / len(active_pairs)
    even_printable_ratio = sum(
        1
        for lo, _ in active_pairs
        if lo in (9, 10, 13, 20, 21) or 32 <= lo <= 126
    ) / len(active_pairs)
    if odd_zero_ratio >= 0.8 and even_printable_ratio >= 0.6:
        return bytes(lo for lo, _ in active_pairs)
    return payload_bytes


def fast_read_edf_annotations(edf_path: Union[str, Path]) -> List[Tuple[float, float, str]]:
    """
    Fast binary EDF+ TAL parser.

    Returns:
        List of (onset_sec, duration_sec, description), sorted by onset.
    """
    edf_path = Path(edf_path)
    if not edf_path.exists():
        raise FileNotFoundError(f"EDF file not found: {edf_path}")
    with open(edf_path, "rb") as f:
        header_fixed = f.read(256)
        if len(header_fixed) < 256:
            raise ValueError(f"Invalid EDF header (too short): {edf_path}")

        header_n_bytes = _parse_fixed_width_ascii_int(header_fixed[184:192], default=256)
        n_records = _parse_fixed_width_ascii_int(header_fixed[236:244], default=0)
        n_signals = _parse_fixed_width_ascii_int(header_fixed[252:256], default=0)
        if header_n_bytes < 256 or n_signals <= 0:
            return []

        signal_header_blob = f.read(header_n_bytes - 256)
        if len(signal_header_blob) < (header_n_bytes - 256):
            return []

        cursor = 0
        labels_blob = signal_header_blob[cursor : cursor + (16 * n_signals)]
        cursor += 16 * n_signals
        cursor += 80 * n_signals  # transducer
        cursor += 8 * n_signals   # phys dim
        cursor += 8 * n_signals   # phys min
        cursor += 8 * n_signals   # phys max
        cursor += 8 * n_signals   # dig min
        cursor += 8 * n_signals   # dig max
        cursor += 80 * n_signals  # prefilter
        nsamples_blob = signal_header_blob[cursor : cursor + (8 * n_signals)]

        labels = [b.decode("ascii", errors="ignore").strip() for b in _split_fixed_fields(labels_blob, 16, n_signals)]
        nsamples_per_record = [_parse_fixed_width_ascii_int(b, default=0) for b in _split_fixed_fields(nsamples_blob, 8, n_signals)]
        if not nsamples_per_record or any(v < 0 for v in nsamples_per_record):
            return []

        ann_indices = [i for i, lab in enumerate(labels) if lab.lower().startswith("edf annotations")]
        if not ann_indices:
            return []

        bytes_per_signal = [2 * int(v) for v in nsamples_per_record]
        record_n_bytes = int(sum(bytes_per_signal))
        if record_n_bytes <= 0:
            return []

        if n_records < 0:
            file_size = edf_path.stat().st_size
            payload_n_bytes = max(0, file_size - header_n_bytes)
            n_records = int(payload_n_bytes // record_n_bytes)
        if n_records <= 0:
            return []

        byte_offsets = np.cumsum([0, *bytes_per_signal[:-1]], dtype=np.int64)
        payload_start = int(header_n_bytes)
        ann_spans = [
            (int(byte_offsets[ann_idx]), int(bytes_per_signal[ann_idx]))
            for ann_idx in ann_indices
        ]
        ann_bytes_per_record = int(sum(n for _, n in ann_spans))
        if ann_bytes_per_record <= 0:
            return []

    file_size = edf_path.stat().st_size
    available = max(0, int(file_size) - payload_start)
    actual_records = min(int(n_records), available // record_n_bytes)
    if actual_records <= 0:
        return []

    first_ann_off = min(off for off, _ in ann_spans)
    last_ann_end = max(off + sz for off, sz in ann_spans)
    tail_offset = int(first_ann_off)
    tail_size = int(last_ann_end - first_ann_off)
    local_spans = [(int(off - first_ann_off), int(sz)) for off, sz in ann_spans]

    import os as _os
    from concurrent.futures import ThreadPoolExecutor

    fd = _os.open(str(edf_path), _os.O_RDONLY)
    try:
        positions = [
            payload_start + i * record_n_bytes + tail_offset
            for i in range(actual_records)
        ]
        _N_WORKERS = min(128, actual_records)

        def _pread_tail(pos: int) -> bytes:
            return _os.pread(fd, tail_size, pos)

        with ThreadPoolExecutor(max_workers=_N_WORKERS) as pool:
            chunks = list(pool.map(_pread_tail, positions))
    finally:
        _os.close(fd)

    buf = b"".join(chunks)
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(actual_records, tail_size)
    parts = [arr[:, lo : lo + sz] for lo, sz in local_spans]
    ann_all = np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
    payload_bytes = _maybe_collapse_utf16_annotation_bytes(
        np.ascontiguousarray(ann_all).tobytes()
    )
    tal_text = payload_bytes.decode("latin1", errors="ignore")
    annotations = _parse_tal_annotations(tal_text)
    return sorted(annotations, key=lambda x: (float(x[0]), float(x[1]), x[2]))


def parse_seizure_annotation_events(
    edf_path: Union[str, Path],
    target_labels: List[str],
    start_t_epoch: float,
    *,
    merge_gap_sec: float = 0.0,
) -> Dict[str, object]:
    """
    Parse seizure annotations into normalized intervals plus orphan onset markers.

    Returns a dict with:
    - intervals: merged List[(onset_epoch, offset_epoch)]
    - orphan_onsets: List[onset_epoch]
    - raw_interval_details: one entry per raw onset that produced an interval
    - merged_interval_details: merged interval summaries with provenance
    """
    labels_norm = {
        _normalize_annotation_label(x)
        for x in target_labels
        if str(x).strip()
    }
    if not labels_norm:
        return {
            "intervals": [],
            "orphan_onsets": [],
            "raw_interval_details": [],
            "merged_interval_details": [],
        }

    events = fast_read_edf_annotations(edf_path)
    base_epoch = float(start_t_epoch)
    end_labels = {"end", "eeg end", "seizure end", "sz end", "ictal end", "offset"}
    raw_interval_details: List[Dict[str, object]] = []
    orphan_onsets: List[float] = []

    for idx, (onset_sec, duration_sec, desc) in enumerate(events):
        desc_norm = _normalize_annotation_label(desc)
        if not desc_norm or desc_norm not in labels_norm:
            continue

        onset_epoch = base_epoch + float(onset_sec)
        dur = max(0.0, float(duration_sec))
        detail: Dict[str, object] = {
            "label": str(desc),
            "label_norm": desc_norm,
            "onset_epoch": float(onset_epoch),
            "onset_rel_sec": float(onset_sec),
            "offset_epoch": None,
            "offset_rel_sec": None,
            "duration_sec": None,
            "offset_source": None,
            "paired_end_label": None,
            "paired_end_rel_sec": None,
        }

        if dur > 0.0:
            offset_epoch = onset_epoch + dur
            detail["offset_epoch"] = float(offset_epoch)
            detail["offset_rel_sec"] = float(onset_sec + dur)
            detail["duration_sec"] = float(dur)
            detail["offset_source"] = "duration"
            raw_interval_details.append(detail)
            continue

        offset_epoch = None
        for next_onset_sec, _, next_desc in events[idx + 1 :]:
            next_desc_norm = _normalize_annotation_label(next_desc)
            if next_desc_norm in end_labels and float(next_onset_sec) >= float(onset_sec):
                offset_epoch = base_epoch + float(next_onset_sec)
                detail["offset_epoch"] = float(offset_epoch)
                detail["offset_rel_sec"] = float(next_onset_sec)
                detail["duration_sec"] = float(next_onset_sec - float(onset_sec))
                detail["offset_source"] = "end_label"
                detail["paired_end_label"] = str(next_desc)
                detail["paired_end_rel_sec"] = float(next_onset_sec)
                break

        if offset_epoch is None or float(offset_epoch) <= float(onset_epoch):
            orphan_onsets.append(float(onset_epoch))
            continue
        raw_interval_details.append(detail)

    raw_interval_details.sort(key=lambda x: (float(x["onset_epoch"]), float(x["offset_epoch"])))

    merged_interval_details: List[Dict[str, object]] = []
    gap_sec = max(0.0, float(merge_gap_sec))
    for detail in raw_interval_details:
        onset_epoch = float(detail["onset_epoch"])
        offset_epoch = float(detail["offset_epoch"])
        if (
            merged_interval_details
            and onset_epoch <= float(merged_interval_details[-1]["offset_epoch"]) + gap_sec
        ):
            prev = merged_interval_details[-1]
            prev["offset_epoch"] = max(float(prev["offset_epoch"]), offset_epoch)
            prev["duration_sec"] = float(prev["offset_epoch"]) - float(prev["onset_epoch"])
            prev["raw_onset_count"] = int(prev["raw_onset_count"]) + 1
            prev["labels"].append(str(detail["label"]))
            prev["offset_sources"].append(str(detail["offset_source"]))
            if detail["paired_end_label"] is not None:
                prev["paired_end_labels"].append(str(detail["paired_end_label"]))
        else:
            merged_interval_details.append(
                {
                    "onset_epoch": onset_epoch,
                    "offset_epoch": offset_epoch,
                    "duration_sec": offset_epoch - onset_epoch,
                    "raw_onset_count": 1,
                    "labels": [str(detail["label"])],
                    "offset_sources": [str(detail["offset_source"])],
                    "paired_end_labels": (
                        [str(detail["paired_end_label"])]
                        if detail["paired_end_label"] is not None
                        else []
                    ),
                }
            )

    intervals = [
        (float(x["onset_epoch"]), float(x["offset_epoch"]))
        for x in merged_interval_details
    ]
    orphan_onsets = sorted(float(x) for x in orphan_onsets)
    return {
        "intervals": intervals,
        "orphan_onsets": orphan_onsets,
        "raw_interval_details": raw_interval_details,
        "merged_interval_details": merged_interval_details,
    }


def parse_seizure_onsets_from_annotations(
    edf_path: Union[str, Path],
    target_labels: List[str],
    start_t_epoch: float,
    *,
    merge_gap_sec: float = 0.0,
) -> List[Tuple[float, float]]:
    """
    Parse seizure intervals from EDF+ annotations and convert to absolute epoch.

    Returns:
        List[(onset_epoch, offset_epoch)].

    Notes:
        - Exact label match after normalization.
        - Zero-duration onset markers are paired to the next explicit END label.
        - Orphan onset markers are preserved by `parse_seizure_annotation_events()`
          but omitted here because this legacy API returns intervals only.
        - Overlapping duplicate intervals are merged.
    """
    parsed = parse_seizure_annotation_events(
        edf_path=edf_path,
        target_labels=target_labels,
        start_t_epoch=start_t_epoch,
        merge_gap_sec=merge_gap_sec,
    )
    return list(parsed["intervals"])


def parse_seizure_annotations(
    edf_path: Union[str, Path],
    target_labels: List[str],
    start_t_epoch: float,
    *,
    merge_gap_sec: float = 0.0,
) -> List[Tuple[float, float]]:
    """
    Backward-compatible alias for parse_seizure_onsets_from_annotations().
    """
    return parse_seizure_onsets_from_annotations(
        edf_path=edf_path,
        target_labels=target_labels,
        start_t_epoch=start_t_epoch,
        merge_gap_sec=merge_gap_sec,
    )


def epoch_to_local_hour(epoch_ts: float, timezone_str: str) -> int:
    """
    Convert epoch timestamp to local hour in explicit timezone.
    """
    tz_name = str(timezone_str).strip()
    if not tz_name:
        raise ValueError("timezone_str must be non-empty.")
    dt_utc = datetime.fromtimestamp(float(epoch_ts), tz=timezone.utc)
    return int(dt_utc.astimezone(ZoneInfo(tz_name)).hour)


def read_edf_record_info(edf_path: Union[str, Path]) -> Dict[str, Union[str, int, float]]:
    """
    Read start time and duration from EDF header.

    Returns a small header-driven summary suitable for subject-level timelines.
    """
    edf_path = Path(edf_path)
    if not edf_path.exists():
        raise FileNotFoundError(f"EDF file not found: {edf_path}")
    with open(edf_path, "rb") as f:
        hdr = f.read(256)
    if len(hdr) < 256:
        raise ValueError(f"Invalid EDF header (too short): {edf_path}")
    date_str = hdr[168:176].decode("ascii", errors="ignore").strip()
    time_str = hdr[176:184].decode("ascii", errors="ignore").strip()
    dd, mm, yy = date_str.split(".")
    hh, mi, ss = time_str.split(".")
    yy_int = int(yy)
    year = 2000 + yy_int if yy_int < 85 else 1900 + yy_int
    start_dt = datetime(
        year, int(mm), int(dd), int(hh), int(mi), int(ss), tzinfo=timezone.utc
    )
    n_records = _parse_fixed_width_ascii_int(hdr[236:244], default=0)
    record_duration_sec = float(
        hdr[244:252].decode("ascii", errors="ignore").strip() or "0"
    )
    header_n_bytes = _parse_fixed_width_ascii_int(hdr[184:192], default=256)
    duration_sec = max(0.0, float(n_records) * float(record_duration_sec))
    return {
        "record": edf_path.stem,
        "path": str(edf_path),
        "start_epoch": float(start_dt.timestamp()),
        "end_epoch": float(start_dt.timestamp() + duration_sec),
        "start_iso_utc": start_dt.isoformat(),
        "duration_sec": float(duration_sec),
        "duration_hours": float(duration_sec / 3600.0),
        "n_records": int(n_records),
        "record_duration_sec": float(record_duration_sec),
        "header_n_bytes": int(header_n_bytes),
        "file_size_bytes": int(edf_path.stat().st_size),
    }


def build_recording_timeline(
    edf_paths: List[Union[str, Path]],
    *,
    continuity_gap_tolerance_sec: float = 120.0,
) -> Dict[str, Union[float, int, bool, List[Dict[str, Union[str, int, float, bool]]]]]:
    """
    Build a header-driven subject timeline from EDF files.

    This is the canonical way to reason about subject duration / continuity.
    Do not assume "12 files == 24h".
    """
    entries = [read_edf_record_info(p) for p in edf_paths]
    if not entries:
        return {
            "records": [],
            "n_records": 0,
            "sum_duration_sec": 0.0,
            "span_sec": 0.0,
            "max_abs_gap_sec": 0.0,
            "n_gap_violations": 0,
            "is_continuous": True,
        }
    entries.sort(key=lambda x: float(x["start_epoch"]))
    tol = max(0.0, float(continuity_gap_tolerance_sec))
    max_abs_gap_sec = 0.0
    n_gap_violations = 0
    for idx, entry in enumerate(entries):
        prev = entries[idx - 1] if idx > 0 else None
        if prev is None:
            gap_prev_sec = 0.0
            continuous_prev = True
        else:
            gap_prev_sec = float(entry["start_epoch"]) - float(prev["end_epoch"])
            continuous_prev = abs(gap_prev_sec) <= tol
            max_abs_gap_sec = max(max_abs_gap_sec, abs(gap_prev_sec))
            if not continuous_prev:
                n_gap_violations += 1
        entry["gap_prev_sec"] = float(gap_prev_sec)
        entry["continuous_prev"] = bool(continuous_prev)
    sum_duration_sec = sum(float(x["duration_sec"]) for x in entries)
    span_sec = float(entries[-1]["end_epoch"]) - float(entries[0]["start_epoch"])
    return {
        "records": entries,
        "n_records": len(entries),
        "sum_duration_sec": float(sum_duration_sec),
        "sum_duration_hours": float(sum_duration_sec / 3600.0),
        "span_sec": float(span_sec),
        "span_hours": float(span_sec / 3600.0),
        "max_abs_gap_sec": float(max_abs_gap_sec),
        "n_gap_violations": int(n_gap_violations),
        "is_continuous": bool(n_gap_violations == 0),
    }

def get_array_module(use_gpu: bool = False):
    """Get numpy or cupy based on availability and preference."""
    if use_gpu and HAS_GPU:
        return cp
    return np


def to_numpy(arr):
    """Convert cupy array to numpy if needed."""
    if HAS_GPU and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return arr


def to_gpu(arr):
    """Convert numpy array to cupy if GPU available."""
    if HAS_GPU:
        return cp.asarray(arr)
    return arr
    
    
class ElectrodeParser:
    """
    Parses SEEG electrode names into (prefix, number) tuples.
    
    Handles formats:
    - Standard: A1, B10, K3
    - With prime: A'1, B'10
    - With prefixes: POL A1, EEG A1-Ref
    """
    
    # Common prefixes to strip
    STRIP_PREFIXES = ['POL ', 'EEG ', 'SEEG ']
    STRIP_SUFFIXES = ['-Ref', '-REF', '-ref']
    
    # Channels to exclude (case-insensitive patterns)
    EXCLUDE_PATTERNS = ['ECG', 'EKG', 'EMG', 'EOG', 'MK', 'DC', 'Annotations', 
                        'Status', 'Trigger', 'Event', 'STI', 'RESP', 'SpO2']
    
    # Valid SEEG pattern: letters + optional prime + numbers
    SEEG_PATTERN = re.compile(r"^([a-zA-Z]+[']?)([0-9]+)$")
    
    @classmethod
    def clean_name(cls, raw_name: str) -> str:
        """Remove common prefixes and suffixes from channel name."""
        name = raw_name
        for prefix in cls.STRIP_PREFIXES:
            if name.startswith(prefix):
                name = name[len(prefix):]
        for suffix in cls.STRIP_SUFFIXES:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
        return name.strip()
    
    @classmethod
    def is_excluded(cls, name: str) -> bool:
        """Check if channel should be excluded."""
        name_lower = name.lower()
        for pattern in cls.EXCLUDE_PATTERNS:
            if pattern.lower() in name_lower:
                return True
        return False
    
    @classmethod
    def parse(cls, ch_name: str) -> Tuple[Optional[str], Optional[int]]:
        """
        Parse electrode name into (prefix, number).
        
        Examples:
            'A1' -> ('A', 1)
            'K10' -> ('K', 10)
            "A'5" -> ("A'", 5)
            
        Returns:
            (prefix, number) or (None, None) if invalid
        """
        clean = cls.clean_name(ch_name)
        match = cls.SEEG_PATTERN.match(clean)
        if match:
            return match.group(1).upper(), int(match.group(2))
        return None, None
    
    @classmethod
    def is_valid_seeg(cls, ch_name: str) -> bool:
        """Check if channel name is valid SEEG format."""
        if cls.is_excluded(ch_name):
            return False
        prefix, num = cls.parse(ch_name)
        return prefix is not None


class CommonAverageReferencer:
    """
    Applies Common Average Reference (CAR) per electrode shaft.
    
    For each shaft (e.g., all A contacts), subtracts the mean of all
    contacts on that shaft from each contact.
    """
    
    def compute_car(self, data: np.ndarray, ch_names: List[str]
                    ) -> Tuple[np.ndarray, List[str]]:
        """
        Apply CAR per shaft.
        
        Args:
            data: (n_channels, n_samples)
            ch_names: Channel names
            
        Returns:
            car_data: (n_channels, n_samples) CAR referenced data
            ch_names: Same channel names (unchanged)
        """
        car_data = data.copy()
        ch_indices = {name: i for i, name in enumerate(ch_names)}
        
        # Group by shaft
        shaft_groups = {}
        for name in ch_names:
            prefix, num = ElectrodeParser.parse(name)
            if prefix:
                if prefix not in shaft_groups:
                    shaft_groups[prefix] = []
                shaft_groups[prefix].append(name)
        
        # Apply CAR per shaft
        for prefix, channels in shaft_groups.items():
            indices = [ch_indices[ch] for ch in channels]
            shaft_mean = np.mean(data[indices], axis=0, keepdims=True)
            car_data[indices] -= shaft_mean
        
        return car_data, ch_names


class FilterBackend:
    """
    Abstract interface for filtering operations.
    
    Eliminates runtime if/else branching between CPU and GPU implementations.
    """
    
    def apply_notch(self, data: np.ndarray, sfreq: float, freqs: List[float]) -> np.ndarray:
        """Apply notch filter at specified frequencies."""
        raise NotImplementedError
    
    def apply_bandpass(self, data: np.ndarray, sfreq: float, low: float, high: float) -> np.ndarray:
        """Apply bandpass filter."""
        raise NotImplementedError


class CpuFilterBackend(FilterBackend):
    """CPU-based filtering using scipy."""
    
    def apply_notch(self, data: np.ndarray, sfreq: float, freqs: List[float]) -> np.ndarray:
        """Apply notch filter for power line harmonics."""
        for freq in freqs:
            if freq < sfreq / 2:  # Must be below Nyquist
                Q = 30  # Quality factor
                b, a = iirnotch(freq, Q, sfreq)
                data = filtfilt(b, a, data, axis=-1)
        return data
    
    def apply_bandpass(self, data: np.ndarray, sfreq: float, low: float, high: float) -> np.ndarray:
        """Apply bandpass filter."""
        nyq = sfreq / 2
        
        if high >= nyq:
            warnings.warn(f"Bandpass high ({high}Hz) >= Nyquist ({nyq}Hz), clipping to {nyq-1}Hz")
            high = nyq - 1
            
        sos = butter(4, [low / nyq, high / nyq], btype='band', output='sos')
        data = sosfiltfilt(sos, data, axis=-1)
        return data


class GpuFilterBackend(FilterBackend):
    """GPU-based filtering using CuPy."""
    
    def __init__(self, chunk_sec: float = 20.0):
        """
        Args:
            chunk_sec: Chunk size in seconds for processing (avoids OOM on long recordings)
        """
        if not HAS_GPU:
            raise RuntimeError("GPU requested but CuPy not available")
        self.chunk_sec = float(chunk_sec)
    
    def _sosfiltfilt_reflect_gpu(self, x_gpu, sos_gpu, pad_len: int):
        """
        Approximate filtfilt on GPU via reflect padding + forward/backward sosfilt.
        Avoids worst edge transients from naive forward->flip->forward.
        """
        if pad_len <= 0:
            y = cupy_sosfilt(sos_gpu, x_gpu, axis=-1)
            y = cp.flip(y, axis=-1)
            y = cupy_sosfilt(sos_gpu, y, axis=-1)
            return cp.flip(y, axis=-1)

        x_pad = cp.pad(x_gpu, ((0, 0), (pad_len, pad_len)), mode="reflect")
        y = cupy_sosfilt(sos_gpu, x_pad, axis=-1)
        y = cp.flip(y, axis=-1)
        y = cupy_sosfilt(sos_gpu, y, axis=-1)
        y = cp.flip(y, axis=-1)
        return y[:, pad_len:-pad_len]
    
    def _filter_in_chunks(self, x_gpu, sos_gpu, pad_len: int, sfreq: float):
        """Filter data in chunks to avoid GPU OOM."""
        n_samples = int(x_gpu.shape[1])
        chunk_len = int(max(1, round(self.chunk_sec * sfreq)))
        
        if n_samples <= chunk_len:
            return self._sosfiltfilt_reflect_gpu(x_gpu, sos_gpu, pad_len)
        
        out = cp.zeros_like(x_gpu, dtype=x_gpu.dtype)
        step = max(1, chunk_len)
        
        for start in range(0, n_samples, step):
            end = min(n_samples, start + step)
            ext_start = max(0, start - pad_len)
            ext_end = min(n_samples, end + pad_len)
            chunk = x_gpu[:, ext_start:ext_end]
            filt = self._sosfiltfilt_reflect_gpu(chunk, sos_gpu, pad_len)
            s0 = start - ext_start
            s1 = s0 + (end - start)
            out[:, start:end] = filt[:, s0:s1]
        
        return out
    
    def apply_notch(self, data: np.ndarray, sfreq: float, freqs: List[float]) -> np.ndarray:
        """Apply notch filter on GPU."""
        # Transfer to GPU (float32 to reduce memory)
        data_gpu = cp.asarray(data, dtype=cp.float32)
        
        for freq in freqs:
            if freq < sfreq / 2:
                Q = 30
                b, a = iirnotch(freq, Q, sfreq)
                from scipy.signal import tf2sos
                sos = tf2sos(b, a)
                sos_gpu = cp.asarray(sos)
                pad_len = int(min(data_gpu.shape[1] // 4, max(1, round(0.5 * sfreq))))
                data_gpu = self._filter_in_chunks(data_gpu, sos_gpu, pad_len, sfreq)
        
        return cp.asnumpy(data_gpu)
    
    def apply_bandpass(self, data: np.ndarray, sfreq: float, low: float, high: float) -> np.ndarray:
        """Apply bandpass filter on GPU."""
        nyq = sfreq / 2
        
        if high >= nyq:
            high = nyq - 1
            
        sos = butter(4, [low / nyq, high / nyq], btype='band', output='sos')
        
        # Transfer to GPU
        data_gpu = cp.asarray(data, dtype=cp.float32)
        sos_gpu = cp.asarray(sos)
        pad_len = int(min(data_gpu.shape[1] // 4, max(1, round(0.5 * sfreq))))
        data_gpu = self._filter_in_chunks(data_gpu, sos_gpu, pad_len, sfreq)
        
        return cp.asnumpy(data_gpu)




class BipolarReferencer:
    """
    Applies bipolar re-referencing to SEEG data.
    
    Critical rules:
    1. Only subtract adjacent contacts on the SAME shaft (same prefix)
    2. Handle non-consecutive numbering (e.g., K3, K5, K6 -> K3-K5, K5-K6)
    3. Naming convention: explicit pair name (A1-A2, A2-A3, ...)
    """
    
    def __init__(self, allow_gap: int = 1):
        """
        Args:
            allow_gap: Maximum allowed gap in contact numbers.
                       1 = only consecutive (A1-A2), matching legacy yuquan pipeline
                       2 = allow one missing (A1-A3 if A2 doesn't exist)
        """
        self.allow_gap = allow_gap
        
    def group_by_shaft(self, ch_names: List[str]) -> Dict[str, List[Tuple[int, str]]]:
        """
        Group channels by electrode shaft prefix.
        
        Args:
            ch_names: List of channel names
        
        Returns:
            Dict mapping prefix -> [(number, name), ...] sorted by number
        """
        shaft_groups = {}
        
        for name in ch_names:
            prefix, num = ElectrodeParser.parse(name)
            if prefix is not None:
                if prefix not in shaft_groups:
                    shaft_groups[prefix] = []
                shaft_groups[prefix].append((num, name))
        
        # Sort each group by contact number
        for prefix in shaft_groups:
            shaft_groups[prefix].sort(key=lambda x: x[0])
            
        return shaft_groups
    
    def compute_bipolar(self, data: np.ndarray, ch_names: List[str]
                        ) -> Tuple[np.ndarray, List[str], List[Tuple[str, str]]]:
        """
        Compute bipolar referenced data.
        
        Args:
            data: (n_channels, n_samples) array
            ch_names: List of channel names
            
        Returns:
            bipolar_data: (n_bipolar, n_samples) array
            bipolar_names: List of bipolar channel names
            bipolar_pairs: List of (ch1, ch2) pairs that were subtracted
        """
        ch_indices = {name: i for i, name in enumerate(ch_names)}
        shaft_groups = self.group_by_shaft(ch_names)
        
        bipolar_data = []
        bipolar_names = []
        bipolar_pairs = []
        
        for prefix, contacts in shaft_groups.items():
            # contacts is [(num, name), ...] sorted by number
            for i in range(len(contacts) - 1):
                curr_num, curr_name = contacts[i]
                next_num, next_name = contacts[i + 1]
                
                # Check gap between contact numbers
                gap = next_num - curr_num
                
                if gap <= self.allow_gap:
                    idx_curr = ch_indices[curr_name]
                    idx_next = ch_indices[next_name]
                    
                    # Bipolar: proximal - distal (lower - higher number)
                    diff_signal = data[idx_curr] - data[idx_next]
                    
                    bipolar_data.append(diff_signal)
                    curr_clean = ElectrodeParser.clean_name(curr_name)
                    next_clean = ElectrodeParser.clean_name(next_name)
                    bipolar_names.append(f"{curr_clean}-{next_clean}")
                    bipolar_pairs.append((curr_name, next_name))
                else:
                    warnings.warn(
                        f"Skipping pair {curr_name}-{next_name}: gap={gap} > allow_gap={self.allow_gap}"
                    )
        
        if not bipolar_data:
            raise RuntimeError("Bipolar referencing resulted in 0 channels. Check channel naming.")
            
        return np.array(bipolar_data), bipolar_names, bipolar_pairs


class ChannelQualityChecker:
    """
    Assesses channel quality for SEEG data.
    
    Detects:
    - High amplitude outliers (artifacts)
    - Low variance channels (disconnected/dead)
    - High-frequency noise
    """
    
    def __init__(self, 
                 zscore_threshold: float = 5.0,
                 variance_percentile_low: float = 1.0,
                 variance_percentile_high: float = 99.0):
        """
        Args:
            zscore_threshold: Channels with |z-score| > threshold are flagged
            variance_percentile_low: Below this percentile = low variance (dead)
            variance_percentile_high: Above this percentile = high variance (noisy)
        """
        self.zscore_threshold = zscore_threshold
        self.var_pctl_low = variance_percentile_low
        self.var_pctl_high = variance_percentile_high
        
    def check_quality(self, data: np.ndarray, ch_names: List[str]
                      ) -> Tuple[List[str], Dict[str, float]]:
        """
        Check channel quality.
        
        Args:
            data: (n_channels, n_samples)
            ch_names: Channel names
            
        Returns:
            bad_channels: List of channels flagged as bad
            quality_scores: Dict of channel -> score (higher = better, 0-1)
        """
        n_channels = data.shape[0]
        bad_channels = []
        quality_scores = {}
        
        # Compute per-channel statistics
        channel_vars = np.var(data, axis=1)
        channel_means = np.mean(np.abs(data), axis=1)
        
        # Global statistics for z-scoring
        global_var = np.median(channel_vars)
        global_std = np.std(channel_vars)
        
        # Percentile thresholds
        var_low = np.percentile(channel_vars, self.var_pctl_low)
        var_high = np.percentile(channel_vars, self.var_pctl_high)
        
        for i, name in enumerate(ch_names):
            score = 1.0  # Start with perfect score
            reasons = []
            
            # Check 1: Variance z-score
            if global_std > 0:
                z = (channel_vars[i] - global_var) / global_std
                if np.abs(z) > self.zscore_threshold:
                    score *= 0.5
                    reasons.append(f"z-score={z:.1f}")
            
            # Check 2: Low variance (dead channel)
            if channel_vars[i] < var_low:
                score *= 0.3
                reasons.append("low_variance")
            
            # Check 3: High variance (noisy/artifact)
            if channel_vars[i] > var_high:
                score *= 0.5
                reasons.append("high_variance")
            
            # Check 4: Flat line detection (very low variance)
            if channel_vars[i] < 1e-12:
                score = 0.0
                reasons.append("flat_line")
            
            quality_scores[name] = score
            
            if score < 0.5:
                bad_channels.append(name)
                
        return bad_channels, quality_scores


class SEEGPreprocessor:
    """
    Complete preprocessing pipeline for SEEG HFO analysis.
    
    Pipeline:
    1. Load EDF and filter to valid SEEG channels
    2. Apply re-referencing strategy (bipolar, CAR, or none)
    3. Optionally apply explicit channel include/exclude lists
    5. Resample (with Nyquist warnings)
    6. Apply notch and bandpass filters (GPU accelerated if available)
    7. Check channel quality
    
    Re-referencing strategies:
    - 'none': Keep as-is (for pre-bipolar data or monopolar analysis)
    - 'bipolar': Apply bipolar referencing (A1-A2, A2-A3, ...)
    - 'car': Common Average Reference per shaft
    - 'auto': Alias for 'bipolar' (kept for backwards compatibility; no guessing)
    
    Example:
        >>> # Explicitly bipolar-reference (recommended for SEEG analysis)
        >>> preprocessor = SEEGPreprocessor(reference='bipolar')
        >>> result = preprocessor.run('/path/to/file.edf')
        
        >>> # With GPU acceleration
        >>> preprocessor = SEEGPreprocessor(use_gpu=True)
        >>> result = preprocessor.run('/path/to/file.edf')
    """
    
    def __init__(self,
                 target_sfreq: Optional[Union[float, str]] = None,
                 target_band: str = 'ripple',
                 reference: str = 'auto',
                 bipolar_gap: int = 1,
                 outer_contact_mode: str = "none",
                 include_channels: Optional[List[str]] = None,
                 exclude_channels: Optional[List[str]] = None,
                 notch_freqs: Optional[List[float]] = None,
                 bandpass: Optional[Tuple[float, float]] = None,
                 check_quality: bool = True,
                 crop_seconds: Optional[float] = None,
                 use_gpu: bool = False,
                 gpu_chunk_sec: float = 20.0,
                 use_fif_cache: bool = True,
                 fif_cache_dir: Optional[Union[str, Path]] = None):
        """
        Args:
            target_sfreq: Target sampling rate (Hz), or 'auto'/'none'.
                          If None or 'auto', auto-select based on target_band:
                          - 'ripple': 1000 Hz
                          - 'fast_ripple': 2000 Hz (or keep original if >= 2000)
            target_band: 'ripple' (80-250Hz) or 'fast_ripple' (250-500Hz)
            reference: Re-referencing strategy:
                       - 'none': Keep original (EDF raw; do not guess)
                       - 'bipolar': Apply bipolar referencing
                       - 'car': Common Average Reference per shaft
                       - 'auto': Alias for 'bipolar' (no guessing)
            bipolar_gap: Max gap in contact numbers for bipolar (default=1)
            outer_contact_mode: Optional legacy contact-edge handling.
                               - 'none': keep all contacts
                               - 'drop_shaft_edges': drop first/last contact per shaft
            include_channels: Explicit channel whitelist (after cleaning names). If set,
                              keeps only these channels (intersection, order preserved).
            exclude_channels: Explicit channel blacklist (after cleaning names).
            notch_freqs: Notch filter frequencies (default: 50Hz harmonics)
            bandpass: (low, high) bandpass filter (default: None, applied in detector)
            check_quality: Whether to run quality checks
            crop_seconds: If set, only load first N seconds (for testing/memory)
            use_gpu: Whether to use GPU acceleration (requires CuPy)
        """
        self.target_band = target_band
        self.reference = reference
        self.bipolar_gap = bipolar_gap
        self.outer_contact_mode = str(outer_contact_mode).strip().lower()
        self.include_channels = include_channels
        self.exclude_channels = exclude_channels
        self.check_quality = check_quality
        self.crop_seconds = crop_seconds
        self.use_gpu = use_gpu and HAS_GPU
        self.gpu_chunk_sec = float(gpu_chunk_sec)
        self.use_fif_cache = bool(use_fif_cache)
        self.fif_cache_dir = Path(fif_cache_dir) if fif_cache_dir is not None else None
        
        if use_gpu and not HAS_GPU:
            warnings.warn("GPU requested but CuPy not available. Using CPU.")
        
        # Initialize filter backend (决定 CPU/GPU 一次,消除运行时判断)
        if self.use_gpu:
            self.filter_backend: FilterBackend = GpuFilterBackend(chunk_sec=self.gpu_chunk_sec)
        else:
            self.filter_backend: FilterBackend = CpuFilterBackend()
        
        self._skip_resample = False
        if isinstance(target_sfreq, str):
            ts = target_sfreq.strip().lower()
            if ts == "none":
                self._skip_resample = True
                self.target_sfreq = None
            elif ts == "auto":
                target_sfreq = None
            else:
                target_sfreq = float(target_sfreq)

        # Auto-select sampling rate based on target band
        if target_sfreq is None and not self._skip_resample:
            if target_band == 'fast_ripple':
                self.target_sfreq = 2000.0  # Keep high for FR
            else:
                self.target_sfreq = 1000.0  # Safe for Ripple
        elif not self._skip_resample:
            self.target_sfreq = float(target_sfreq)

        # Validate Nyquist for target band
        if not self._skip_resample:
            self._validate_nyquist()
        
        # Default notch frequencies (50Hz power line and harmonics)
        if notch_freqs is None:
            self.notch_freqs = [50.0, 100.0, 150.0, 200.0]
        else:
            self.notch_freqs = notch_freqs
            
        self.bandpass = bandpass
        
        # Components
        self.bipolar_ref = BipolarReferencer(allow_gap=bipolar_gap)
        self.car_ref = CommonAverageReferencer()
        self.quality_checker = ChannelQualityChecker()
        
        # State
        self._raw = None
        self._original_sfreq = None
        self._original_ch_names = []
        self._excluded_channels = []
        self._used_gpu = False

        if self.outer_contact_mode not in {"none", "drop_shaft_edges"}:
            raise ValueError(
                "outer_contact_mode must be 'none' or 'drop_shaft_edges'. "
                f"Got: {outer_contact_mode!r}"
            )
        
    def _validate_nyquist(self):
        """Warn if sampling rate is too low for target band."""
        if self.target_band == 'fast_ripple':
            # FR needs up to 500Hz, Nyquist = 1000Hz minimum
            # But for clean signal, we want 2x margin
            if self.target_sfreq < 1200:
                warnings.warn(
                    f"Target sfreq {self.target_sfreq}Hz is marginal for Fast Ripple (250-500Hz). "
                    f"Recommend >= 1500Hz, ideally 2000Hz."
                )
        elif self.target_band == 'ripple':
            # Ripple needs up to 250Hz, Nyquist = 500Hz minimum
            if self.target_sfreq < 600:
                warnings.warn(
                    f"Target sfreq {self.target_sfreq}Hz is too low for Ripple (80-250Hz). "
                    f"Nyquist limit violated!"
                )

    def _edge_drop_key(self, name: str) -> Tuple[Optional[str], Optional[int]]:
        """Map a channel name to (shaft, order) for edge-contact dropping."""
        nm = str(name).strip().upper()
        if "-" in nm:
            left, right = nm.split("-", 1)
            p1, n1 = ElectrodeParser.parse(left.strip())
            p2, n2 = ElectrodeParser.parse(right.strip())
            if p1 is not None and p2 is not None and p1 == p2:
                return p1, min(int(n1), int(n2))
        p, n = ElectrodeParser.parse(nm)
        if p is None:
            return None, None
        return p, int(n)

    def _drop_outer_contacts_per_shaft(
        self,
        data: np.ndarray,
        ch_names: List[str],
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """Drop first/last ordered contact in each shaft group when possible."""
        groups: Dict[str, List[Tuple[int, int, str]]] = {}
        for idx, nm in enumerate(ch_names):
            shaft, order = self._edge_drop_key(nm)
            if shaft is None or order is None:
                continue
            groups.setdefault(str(shaft), []).append((int(order), int(idx), str(nm)))

        drop_idx: set = set()
        dropped: List[str] = []
        for _, items in groups.items():
            if len(items) <= 2:
                continue
            items_sorted = sorted(items, key=lambda x: (x[0], x[1]))
            for _, idx, nm in (items_sorted[0], items_sorted[-1]):
                if idx not in drop_idx:
                    drop_idx.add(idx)
                    dropped.append(nm)

        if not drop_idx:
            return data, ch_names, []

        keep_indices = [i for i in range(len(ch_names)) if i not in drop_idx]
        return data[keep_indices], [ch_names[i] for i in keep_indices], dropped
    
    def _get_fif_cache_path(self, edf_path: Path) -> Path:
        base = edf_path.stem
        if self.crop_seconds is not None:
            base = f"{base}_crop{int(self.crop_seconds)}s"
        cache_dir = self.fif_cache_dir if self.fif_cache_dir is not None else edf_path.parent
        return Path(cache_dir) / f"{base}.fif"

    def load_edf(self, file_path: Union[str, Path]) -> 'SEEGPreprocessor':
        """
        Load EDF file and filter to valid SEEG channels.
        
        Args:
            file_path: Path to EDF file
            
        Returns:
            self (for chaining)
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"EDF file not found: {file_path}")
            
        print(f"Loading: {file_path.name}")

        # Optional: use cached FIF to speed repeated reads
        if self.use_fif_cache:
            fif_path = self._get_fif_cache_path(file_path)
            if fif_path.exists():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self._raw = mne.io.read_raw_fif(
                        str(fif_path),
                        preload=True,
                        verbose=False,
                    )
                self._original_sfreq = self._raw.info['sfreq']
                self._original_ch_names = list(self._raw.ch_names)
                print(f"  Loaded cached FIF: {fif_path.name}")
                return self
        
        # First pass: read header to get channel list
        # Use latin1 encoding for Chinese hospital EDF files
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            temp_raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False, encoding='latin1')
        
        all_chs = temp_raw.ch_names
        self._original_ch_names = all_chs
        
        # Filter to valid SEEG channels
        keep_chs = [ch for ch in all_chs if ElectrodeParser.is_valid_seeg(ch)]
        
        if not keep_chs:
            raise ValueError(
                f"No valid SEEG channels found. "
                f"Sample channels: {all_chs[:5]}"
            )
        
        # Load only valid channels (initially without preload for cropping)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._raw = mne.io.read_raw_edf(
                file_path, 
                include=keep_chs, 
                preload=False, 
                verbose=False,
                encoding='latin1'
            )
        
        self._original_sfreq = self._raw.info['sfreq']
        total_duration = self._raw.times[-1]
        
        # Crop if requested (for memory management)
        if self.crop_seconds is not None and self.crop_seconds < total_duration:
            self._raw.crop(tmax=self.crop_seconds)
            print(f"  Cropped to first {self.crop_seconds} seconds")
        
        # Now load data into memory
        self._raw.load_data()
        
        # Clean channel names
        rename_map = {ch: ElectrodeParser.clean_name(ch) for ch in self._raw.ch_names}
        self._raw.rename_channels(rename_map)
        
        print(f"  Loaded {len(self._raw.ch_names)} SEEG channels")
        print(f"  Original sfreq: {self._original_sfreq} Hz")
        print(f"  Duration: {self._raw.times[-1]:.1f} s ({self._raw.times[-1]/3600:.2f} h)")

        if self.use_fif_cache:
            try:
                fif_path = self._get_fif_cache_path(file_path)
                fif_path.parent.mkdir(parents=True, exist_ok=True)
                self._raw.save(str(fif_path), overwrite=True, verbose=False)
                print(f"  Saved FIF cache: {fif_path.name}")
            except Exception:
                pass
        
        return self
    
    def run(self, file_path: Union[str, Path]) -> PreprocessingResult:
        """
        Run complete preprocessing pipeline.
        
        Args:
            file_path: Path to EDF file
            
        Returns:
            PreprocessingResult containing processed data and metadata
        """
        t_start = time.time()
        logger = get_run_logger(f"preprocess_{Path(file_path).stem}")
        log_section(logger, "PREPROCESS START")
        logger.info("file_path=%s", str(file_path))
        logger.info("target_band=%s", str(self.target_band))
        logger.info("reference=%s", str(self.reference))
        logger.info("target_sfreq=%s", str(self.target_sfreq))
        logger.info("skip_resample=%s", str(self._skip_resample))
        logger.info("crop_seconds=%s", str(self.crop_seconds))
        logger.info("use_gpu=%s", str(self.use_gpu))
        logger.info("outer_contact_mode=%s", str(self.outer_contact_mode))

        # Step 1: Load EDF
        self.load_edf(file_path)
        
        # Get raw data
        data = self._raw.get_data()  # (n_channels, n_samples)
        ch_names = list(self._raw.ch_names)
        sfreq = self._original_sfreq
        original_ch_count = len(ch_names)
        
        # Step 2: Apply re-referencing (explicit; no guessing)
        reference_type = 'monopolar'
        effective_reference = self.reference
        if effective_reference == 'auto':
            effective_reference = 'bipolar'

        # Step 3: Apply re-referencing
        bipolar_pairs = None
        
        if effective_reference == 'bipolar':
            print("Applying bipolar re-referencing...")
            pre_count = len(ch_names)
            data, ch_names, bipolar_pairs = self.bipolar_ref.compute_bipolar(data, ch_names)
            reference_type = 'bipolar'
            print(f"  {pre_count} -> {len(ch_names)} bipolar channels")
            
        elif effective_reference == 'car':
            print("Applying Common Average Reference per shaft...")
            data, ch_names = self.car_ref.compute_car(data, ch_names)
            reference_type = 'car'
            print(f"  Applied CAR to {len(ch_names)} channels")
            
        elif effective_reference == 'none':
            reference_type = 'monopolar'
            print("Keeping original reference (monopolar)")
        else:
            raise ValueError(f"Unknown reference='{self.reference}'. Use 'none'/'bipolar'/'car'.")

        dropped_outer_contacts: List[str] = []
        if self.outer_contact_mode == "drop_shaft_edges":
            pre_drop = len(ch_names)
            data, ch_names, dropped_outer_contacts = self._drop_outer_contacts_per_shaft(data, ch_names)
            if dropped_outer_contacts:
                print(
                    "Dropping shaft-edge contacts "
                    f"({len(dropped_outer_contacts)}): {dropped_outer_contacts[:5]}..."
                )
            print(f"  Edge-drop mode: {pre_drop} -> {len(ch_names)} channels")

        # Step 4: Apply explicit channel include/exclude lists (post-reference)
        excluded_channels: List[str] = []
        excluded_channels.extend(dropped_outer_contacts)
        if self.include_channels is not None:
            include_set = set(self.include_channels)
            keep_indices = [i for i, nm in enumerate(ch_names) if nm in include_set]
            excluded_channels.extend([nm for nm in ch_names if nm not in include_set])
            data = data[keep_indices]
            ch_names = [ch_names[i] for i in keep_indices]

        if self.exclude_channels is not None:
            exclude_set = set(self.exclude_channels)
            keep_indices = [i for i, nm in enumerate(ch_names) if nm not in exclude_set]
            excluded_channels.extend([nm for nm in ch_names if nm in exclude_set])
            data = data[keep_indices]
            ch_names = [ch_names[i] for i in keep_indices]

        self._excluded_channels = sorted(set(excluded_channels))
        
        # Step 5: Resample if needed
        if self._skip_resample:
            print(f"Skipping resample (target_sfreq=none); keeping {sfreq} Hz")
        elif sfreq != self.target_sfreq:
            print(f"Resampling: {sfreq} -> {self.target_sfreq} Hz...")
            data = self._resample(data, sfreq, self.target_sfreq)
            sfreq = self.target_sfreq
        else:
            print(f"Sampling rate matches target ({sfreq} Hz), skipping resample")
        
        # Step 6: Filtering (with optional GPU acceleration)
        gpu_status = "(GPU)" if self.use_gpu else "(CPU)"
        print(f"Filtering {gpu_status}: notch={self.notch_freqs}, bandpass={self.bandpass}")
        data = self._apply_filters(data, sfreq)
        self._used_gpu = self.use_gpu
        
        # Step 7: Quality check
        bad_channels = []
        quality_scores = {}
        if self.check_quality:
            print("Checking channel quality...")
            bad_channels, quality_scores = self.quality_checker.check_quality(data, ch_names)
            if bad_channels:
                print(f"  Flagged {len(bad_channels)} potentially bad channels: {bad_channels[:5]}...")
        
        # Build shaft mapping
        shaft_mapping = {}
        for name in ch_names:
            prefix, _ = ElectrodeParser.parse(name)
            if prefix:
                if prefix not in shaft_mapping:
                    shaft_mapping[prefix] = []
                shaft_mapping[prefix].append(name)
        
        print(f"Preprocessing complete: {data.shape[0]} channels, {data.shape[1]/sfreq:.1f}s at {sfreq}Hz")
        log_section(logger, "PREPROCESS SUMMARY")
        logger.info("input_channels=%d", int(original_ch_count))
        logger.info("output_channels=%d", int(data.shape[0]))
        logger.info("reference_type=%s", str(reference_type))
        logger.info("sfreq=%.3f", float(sfreq))
        logger.info("duration_sec=%.3f", float(data.shape[1] / sfreq))
        logger.info("excluded_channels=%d", int(len(set(excluded_channels))))
        logger.info("dropped_outer_contacts=%d", int(len(dropped_outer_contacts)))
        logger.info("bad_channels=%d", int(len(bad_channels)))
        logger.info("elapsed_sec=%.3f", float(time.time() - t_start))
        log_section(logger, "PREPROCESS END")
        
        return PreprocessingResult(
            data=data,
            sfreq=sfreq,
            ch_names=ch_names,
            original_ch_names=list(self._raw.ch_names),
            bipolar_pairs=bipolar_pairs,
            shaft_mapping=shaft_mapping,
            bad_channels=bad_channels,
            quality_scores=quality_scores,
            reference_type=reference_type,
            excluded_channels=excluded_channels,
            used_gpu=self._used_gpu,
            outer_contact_mode=str(self.outer_contact_mode),
            dropped_outer_contacts=dropped_outer_contacts,
        )
    
    def _resample(self, data: np.ndarray, orig_sfreq: float, target_sfreq: float) -> np.ndarray:
        """Resample data using polyphase filtering."""
        from fractions import Fraction
        
        # Find up/down factors
        frac = Fraction(int(target_sfreq), int(orig_sfreq)).limit_denominator(1000)
        up, down = frac.numerator, frac.denominator
        
        # Use scipy's resample_poly (handles anti-aliasing)
        from scipy.signal import resample_poly
        resampled = resample_poly(data, up, down, axis=-1)
        
        return resampled
    
    def _apply_filters(self, data: np.ndarray, sfreq: float) -> np.ndarray:
        """Apply notch and bandpass filters using configured backend."""
        # Notch filter for power line harmonics
        if self.notch_freqs:
            data = self.filter_backend.apply_notch(data, sfreq, self.notch_freqs)
        
        # Bandpass filter (optional - usually applied in detector)
        if self.bandpass is not None:
            low, high = self.bandpass
            data = self.filter_backend.apply_bandpass(data, sfreq, low, high)
        
        return data
    
    def get_shaft_info(self) -> Dict[str, int]:
        """Get number of contacts per electrode shaft."""
        if self._raw is None:
            raise RuntimeError("No data loaded")
        shaft_groups = self.bipolar_ref.group_by_shaft(list(self._raw.ch_names))
        return {k: len(v) for k, v in shaft_groups.items()}


# =============================================================================
# Convenience Functions
# =============================================================================

def save_raw_cache(
    edf_path: Union[str, Path],
    output_dir: Union[str, Path],
    output_prefix: str,
    *,
    reference: str = "none",
    include_channels: Optional[List[str]] = None,
    exclude_channels: Optional[List[str]] = None,
    crop_start_sec: float = 0.0,
    crop_seconds: Optional[float] = None,
) -> str:
    """
    Save raw waveform cache (no resample, no filtering) for visualization.

    The cache is generated using preprocessing utilities (clean names + reference),
    but skips resampling and filtering to preserve raw waveform.
    """
    edf_path = Path(edf_path)
    if not edf_path.exists():
        raise FileNotFoundError(f"EDF file not found: {edf_path}")

    output_dir = Path(output_dir)
    temp_dir = output_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    out_path = temp_dir / f"{output_prefix}_rawCache_{reference}.npz"

    # Read EDF header for channel list
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        temp_raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False, encoding="latin1")
    all_chs = temp_raw.ch_names

    # Keep valid SEEG channels
    keep_chs = [ch for ch in all_chs if ElectrodeParser.is_valid_seeg(ch)]
    if not keep_chs:
        raise ValueError("No valid SEEG channels found in EDF.")

    # Load data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = mne.io.read_raw_edf(
            edf_path,
            include=keep_chs,
            preload=False,
            verbose=False,
            encoding="latin1",
        )

    crop_start_sec = max(0.0, float(crop_start_sec))
    if crop_seconds is not None:
        crop_seconds = float(crop_seconds)
        if crop_seconds <= 0:
            raise ValueError("crop_seconds must be > 0 when provided.")
    if crop_start_sec > 0.0 or crop_seconds is not None:
        total_duration = float(raw.times[-1])
        tmin = min(crop_start_sec, total_duration)
        if crop_seconds is None:
            tmax = total_duration
        else:
            tmax = min(total_duration, crop_start_sec + crop_seconds)
        if tmax > tmin:
            raw.crop(tmin=tmin, tmax=tmax)
    raw.load_data()

    # Clean channel names
    rename_map = {ch: ElectrodeParser.clean_name(ch) for ch in raw.ch_names}
    raw.rename_channels(rename_map)

    data = raw.get_data()
    ch_names = list(raw.ch_names)
    sfreq = float(raw.info["sfreq"])

    # Apply explicit channel include/exclude after cleaning
    excluded_channels: List[str] = []
    if include_channels is not None:
        include_set = set(include_channels)
        keep_indices = [i for i, nm in enumerate(ch_names) if nm in include_set]
        excluded_channels.extend([nm for nm in ch_names if nm not in include_set])
        data = data[keep_indices]
        ch_names = [ch_names[i] for i in keep_indices]
    if exclude_channels is not None:
        exclude_set = set(exclude_channels)
        keep_indices = [i for i, nm in enumerate(ch_names) if nm not in exclude_set]
        excluded_channels.extend([nm for nm in ch_names if nm in exclude_set])
        data = data[keep_indices]
        ch_names = [ch_names[i] for i in keep_indices]

    # Apply reference (no filtering/resampling)
    reference_type = "monopolar"
    if reference == "bipolar":
        ref = BipolarReferencer(allow_gap=1)
        data, ch_names, _ = ref.compute_bipolar(data, ch_names)
        reference_type = "bipolar"
    elif reference == "car":
        ref = CommonAverageReferencer()
        data, ch_names = ref.compute_car(data, ch_names)
        reference_type = "car"
    elif reference == "none":
        reference_type = "monopolar"
    else:
        raise ValueError("reference must be 'none'|'bipolar'|'car'")

    np.savez_compressed(
        out_path,
        data=data.astype(np.float32, copy=False),
        sfreq=np.array([sfreq], dtype=np.float64),
        ch_names=np.array(ch_names),
        start_sec=np.array([crop_start_sec], dtype=np.float64),
        reference_type=np.array([reference_type]),
        original_ch_names=np.array(all_chs),
        excluded_channels=np.array(sorted(set(excluded_channels))),
    )
    return str(out_path)


def load_raw_cache(cache_npz_path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Load raw cache saved by save_raw_cache()."""
    meta = np.load(str(cache_npz_path), allow_pickle=True)
    return {k: meta[k] for k in meta.files}


def preprocess_edf(file_path: Union[str, Path],
                   target_band: str = 'ripple',
                   notch_freqs: Optional[List[float]] = None
                   ) -> Tuple[np.ndarray, float, List[str]]:
    """
    Convenience function for quick preprocessing.
    
    Args:
        file_path: Path to EDF file
        target_band: 'ripple' or 'fast_ripple'
        notch_freqs: Notch filter frequencies
        
    Returns:
        data: (n_channels, n_samples) preprocessed data
        sfreq: Sampling frequency
        ch_names: Channel names
    """
    preprocessor = SEEGPreprocessor(
        target_band=target_band,
        notch_freqs=notch_freqs
    )
    result = preprocessor.run(file_path)
    return result.data, result.sfreq, result.ch_names




# =============================================================================
# Main / Test
# =============================================================================

if __name__ == '__main__':
    # Minimal smoke test (local)
    edf_path = '/mnt/yuquan_data/yuquan_24h_edf/chengshuai/FC10477Q.edf'
    print("=" * 70)
    print("Smoke test: SEEGPreprocessor (bipolar)")
    print("=" * 70)
    pre = SEEGPreprocessor(reference='bipolar', crop_seconds=10)
    out = pre.run(edf_path)
    print(f"OK: {out.data.shape} @ {out.sfreq}Hz, ref={out.reference_type}")
