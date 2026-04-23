"""PR-6-A ictal onset rank extraction primitives (Step 2 contract).

This module owns the lowest layer of the PR-6-A pipeline:
energy-ratio (ER) feature extraction with per-channel sliding windows, a
per-channel baseline z-score normalizer, and a thin loader that crops a
seizure-centered intracranial signal block.

Design contract (locked in
``docs/archive/topic1/pr6a_template_ictal_alignment_plan_2026-04-21.md``):

* Two ER configurations run *in parallel* from Step 2 onward:
    - ``gamma_ER`` (fast=60-100 Hz, slow=4-20 Hz) — primary, HFO-centered.
    - ``broad_ER`` (fast=12-127 Hz, slow=4-20 Hz) — sensitivity, Bartolomei
      2008 baseline, covers burst-suppression / delta-brush onset patterns.
  No third runtime-configurable band is accepted; downstream callers must
  reuse these dicts.
* Baseline window: ``[-300 s, baseline_end_sec]`` relative to clinical onset,
  where ``baseline_end_sec = min(0, eeg_onset_rel_sec) - 60 s``. In words:
  back off 60 s from whichever onset annotation comes first (clinical or
  electrographic). The 60 s buffer is hard-coded; per-subject tuning is
  forbidden (Step 3 will add per-subject CUSUM lambda calibration, *not*
  baseline relocation). When ``eeg_onset_rel_sec`` is missing, this reduces
  to the legacy ``[-300, -60]`` window.
* If the EEG-aware clip leaves less than 60 s of valid baseline, the
  seizure is marked baseline-invalid and downstream code MUST drop it
  rather than silently fall back to the clinical-only window — falling
  back would re-introduce the pre-ictal contamination this contract
  exists to remove.
* Channels with insufficient baseline (< 60 s of valid samples within the
  resolved baseline window) get all-NaN z-ER and must be dropped by
  downstream ranking, never imputed.

Page-Hinkley CUSUM, tie / unreached flagging, ``r_sz`` construction and
the dual-cohort gate are implemented in subsequent steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import spectrogram


GAMMA_ER_BANDS: dict = {
    "key": "gamma_ER",
    "fast": (60.0, 100.0),
    "slow": (4.0, 20.0),
    "role": "primary",
}

BROAD_ER_BANDS: dict = {
    "key": "broad_ER",
    "fast": (12.0, 127.0),
    "slow": (4.0, 20.0),
    "role": "sensitivity",
}

ER_CONFIGS: Tuple[dict, dict] = (GAMMA_ER_BANDS, BROAD_ER_BANDS)

BASELINE_PRE_SEC: float = 300.0
BASELINE_BUFFER_SEC: float = 60.0
MIN_BASELINE_VALID_SEC: float = 60.0


def compute_er(
    signal: np.ndarray,
    fs: float,
    fast_band: Tuple[float, float],
    slow_band: Tuple[float, float],
    win_sec: float = 1.0,
    hop_sec: float = 0.1,
) -> np.ndarray:
    """Compute per-channel log energy ratio ``log(E_fast / E_slow)``.

    Bartolomei 2008-style energy ratio with sliding windows.
    Returns an ``(n_channels, n_time_frames)`` ER matrix.
    """

    sig = np.asarray(signal, dtype=np.float64)
    if sig.ndim != 2:
        raise ValueError("signal must be 2D (n_channels, n_samples)")
    fs = float(fs)
    if fs <= 0:
        raise ValueError("fs must be > 0")
    nyq = fs / 2.0
    for label, band in (("fast_band", fast_band), ("slow_band", slow_band)):
        lo, hi = float(band[0]), float(band[1])
        if not (0.0 <= lo < hi):
            raise ValueError(f"{label} must satisfy 0 <= lo < hi, got {band}")
        if hi >= nyq:
            raise ValueError(
                f"{label} upper bound {hi} Hz >= Nyquist {nyq} Hz for fs={fs}"
            )

    nperseg = max(1, int(round(win_sec * fs)))
    if nperseg > sig.shape[1]:
        raise ValueError(
            f"win_sec={win_sec}s requires {nperseg} samples but signal has only "
            f"{sig.shape[1]} samples"
        )
    noverlap = max(0, nperseg - max(1, int(round(hop_sec * fs))))

    f, _t, Sxx = spectrogram(
        sig,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="density",
        mode="psd",
        axis=-1,
    )

    fast_mask = (f >= float(fast_band[0])) & (f <= float(fast_band[1]))
    slow_mask = (f >= float(slow_band[0])) & (f <= float(slow_band[1]))
    if not fast_mask.any():
        raise ValueError(f"No FFT bins fall inside fast_band={fast_band}")
    if not slow_mask.any():
        raise ValueError(f"No FFT bins fall inside slow_band={slow_band}")

    e_fast = Sxx[..., fast_mask, :].sum(axis=-2)
    e_slow = Sxx[..., slow_mask, :].sum(axis=-2)
    er = np.log(np.maximum(e_fast, 1e-30) / np.maximum(e_slow, 1e-30))
    if er.ndim == 1:
        er = er[np.newaxis, :]
    return er.astype(np.float64, copy=False)


def baseline_zscore_er(
    er: np.ndarray,
    baseline_idx_window: Tuple[int, int],
    exclude_peaks: Optional[Iterable[Tuple[int, int]]] = None,
    *,
    hop_sec: float = 0.1,
    min_baseline_valid_sec: float = MIN_BASELINE_VALID_SEC,
) -> np.ndarray:
    """Per-channel z-score against a baseline window of ER frames.

    ``baseline_idx_window`` is a half-open ``(i_start, i_end)`` slice into
    the time axis of ``er``. Frames inside any range from
    ``exclude_peaks`` (sample index pairs in the same time axis) are
    masked out before computing mean/std. A channel with fewer than
    ``min_baseline_valid_sec`` worth of valid baseline frames returns
    all-NaN; downstream code must drop NaN channels rather than fall back
    to global statistics.
    """

    er = np.asarray(er, dtype=np.float64)
    if er.ndim != 2:
        raise ValueError("er must be 2D (n_channels, n_time_frames)")
    n_ch, n_t = er.shape

    i0, i1 = int(baseline_idx_window[0]), int(baseline_idx_window[1])
    if i0 < 0 or i1 > n_t or i0 >= i1:
        raise ValueError(
            f"baseline_idx_window={baseline_idx_window} out of range for er with n_t={n_t}"
        )

    mask = np.ones(n_t, dtype=bool)
    mask[:i0] = False
    mask[i1:] = False
    if exclude_peaks is not None:
        for p0, p1 in exclude_peaks:
            p0 = max(0, int(p0))
            p1 = min(n_t, int(p1))
            if p1 > p0:
                mask[p0:p1] = False

    n_valid = int(mask.sum())
    min_required_frames = int(np.ceil(float(min_baseline_valid_sec) / float(hop_sec)))
    if n_valid < min_required_frames:
        return np.full_like(er, np.nan)

    baseline_frames = er[:, mask]
    mu = baseline_frames.mean(axis=1, keepdims=True)
    sigma = baseline_frames.std(axis=1, ddof=1, keepdims=True)
    sigma = np.where(sigma > 0.0, sigma, np.nan)
    z = (er - mu) / sigma
    z[np.isnan(sigma).squeeze(axis=1)] = np.nan
    return z


# ---------------------------------------------------------------------------
# Real-data loader: extract a seizure-centered window from Epilepsiae.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeizureWindow:
    """Container for a seizure-centered intracranial recording slice."""

    signal: np.ndarray  # (n_channels, n_samples)
    fs: float
    t_axis: np.ndarray  # seconds, relative to ``clin_onset_epoch``
    ch_names: Sequence[str]
    subject: str
    seizure_id: str
    block_stem: str
    clin_onset_epoch: float
    eeg_onset_epoch: Optional[float]
    pre_sec: float
    post_sec: float


def _read_csv_rows(csv_path: Path) -> list[dict]:
    import csv

    with open(csv_path, "r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _resolve_inventory_paths(
    results_root: Path,
) -> Tuple[Path, Path]:
    """Return (seizure_inventory_csv, block_inventory_csv) under ``results_root``.

    Looks first in ``results/dataset_inventory/`` then falls back to
    ``results/`` (legacy layout that ``src.epilepsiae_dataset`` writes by
    default).
    """

    candidates = (
        results_root / "dataset_inventory" / "epilepsiae_seizure_inventory.csv",
        results_root / "epilepsiae_seizure_inventory.csv",
    )
    for sz in candidates:
        if sz.exists():
            blk = sz.with_name("epilepsiae_block_inventory.csv")
            if blk.exists():
                return sz, blk
    raise FileNotFoundError(
        f"epilepsiae_seizure_inventory.csv not found under {results_root}"
    )


def extract_seizure_window(
    subject: str,
    seizure_idx: int,
    *,
    pre_sec: float = BASELINE_PRE_SEC,
    post_sec: float = 30.0,
    results_root: Path | str = Path("results"),
    reference: str = "car",
) -> SeizureWindow:
    """Load an intracranial signal slice around the ``seizure_idx``-th seizure.

    Parameters
    ----------
    subject:
        ``"<dataset>/<id>"``; only the Epilepsiae dataset is supported in
        Step 2 (Yuquan ictal data lacks per-block onset annotation suitable
        for this pipeline).
    seizure_idx:
        Zero-based index into the chronologically sorted list of seizures
        with a complete clinical onset epoch for the subject.
    pre_sec, post_sec:
        Seconds before / after ``clin_onset_epoch`` to load.

    Raises
    ------
    ValueError if the requested window crosses block boundaries; the
    caller must filter such seizures upstream rather than have this
    function silently shrink the window.
    """

    if "/" not in subject:
        raise ValueError(f"subject must be '<dataset>/<id>', got {subject!r}")
    dataset, sid = subject.split("/", 1)
    if dataset != "epilepsiae":
        raise NotImplementedError(
            f"extract_seizure_window currently only supports epilepsiae; got {dataset}"
        )

    results_root = Path(results_root)
    sz_csv, blk_csv = _resolve_inventory_paths(results_root)
    sz_rows = [r for r in _read_csv_rows(sz_csv) if r["subject"] == sid]
    sz_rows = [r for r in sz_rows if r.get("clin_onset_epoch")]
    sz_rows.sort(key=lambda r: float(r["clin_onset_epoch"]))
    if not sz_rows:
        raise ValueError(f"No seizures with clin_onset_epoch found for {subject}")
    if not (0 <= seizure_idx < len(sz_rows)):
        raise IndexError(
            f"seizure_idx={seizure_idx} out of range for {subject} (n={len(sz_rows)})"
        )
    sz = sz_rows[seizure_idx]
    block_id = sz["block_id"]
    clin_onset_epoch = float(sz["clin_onset_epoch"])
    eeg_onset_epoch = float(sz["eeg_onset_epoch"]) if sz.get("eeg_onset_epoch") else None

    blk_rows = [r for r in _read_csv_rows(blk_csv) if r["subject"] == sid and r["block_id"] == block_id]
    if not blk_rows:
        raise ValueError(f"block_id={block_id} not found in block inventory for {subject}")
    blk = blk_rows[0]
    block_start_epoch = float(blk["block_start_epoch"])
    block_end_epoch = float(blk["block_end_epoch"])
    head_path = blk["head_path"]
    data_path = blk["data_path"]
    if not head_path or not data_path:
        raise ValueError(f"block {block_id} missing head/data path in inventory")

    win_start_epoch = clin_onset_epoch - float(pre_sec)
    win_end_epoch = clin_onset_epoch + float(post_sec)
    if win_start_epoch < block_start_epoch:
        raise ValueError(
            f"{subject} seizure {seizure_idx}: requested window starts "
            f"{block_start_epoch - win_start_epoch:.2f}s before block_start; "
            f"upstream caller must drop this seizure"
        )
    if win_end_epoch > block_end_epoch:
        raise ValueError(
            f"{subject} seizure {seizure_idx}: requested window ends "
            f"{win_end_epoch - block_end_epoch:.2f}s after block_end; "
            f"upstream caller must drop this seizure"
        )

    from src.preprocessing import load_epilepsiae_block

    pre = load_epilepsiae_block(
        data_path,
        head_path,
        reference=reference,
        segment_sec=200.0,
    )
    fs = float(pre.sfreq)
    rel_start_sec = win_start_epoch - block_start_epoch
    i0 = int(round(rel_start_sec * fs))
    i1 = i0 + int(round((float(pre_sec) + float(post_sec)) * fs))
    sliced = pre.data[:, i0:i1].copy()
    n_samples_actual = sliced.shape[1]
    t_axis = (np.arange(n_samples_actual) / fs) - float(pre_sec)

    return SeizureWindow(
        signal=sliced,
        fs=fs,
        t_axis=t_axis,
        ch_names=list(pre.ch_names),
        subject=subject,
        seizure_id=sz["seizure_id"],
        block_stem=blk["block_stem"],
        clin_onset_epoch=clin_onset_epoch,
        eeg_onset_epoch=eeg_onset_epoch,
        pre_sec=float(pre_sec),
        post_sec=float(post_sec),
    )


@dataclass(frozen=True)
class BaselineWindow:
    """Resolved baseline window for a single seizure / channel set.

    Attributes
    ----------
    start_idx, end_idx:
        ER-frame slice ``[start_idx:end_idx]`` (half-open) into the time
        axis returned by :func:`compute_er`.
    start_sec, end_sec:
        Same window expressed in seconds *relative to clinical onset*;
        useful for plotting and JSON diagnostics.
    valid_sec:
        ``end_sec - start_sec``. Negative when the EEG-aware clip would
        push the end before the start.
    clipped_by_eeg_onset:
        True iff ``eeg_onset_rel_sec`` was supplied, was negative
        (electrographic onset earlier than clinical), and actually moved
        ``end_sec`` earlier than ``-buffer_sec``.
    valid:
        ``valid_sec >= min_baseline_valid_sec``. Downstream code must
        treat the seizure as baseline-invalid and drop it when this is
        False; silently falling back to the clinical-only window would
        re-introduce the pre-ictal contamination this contract exists to
        remove.
    """

    start_idx: int
    end_idx: int
    start_sec: float
    end_sec: float
    valid_sec: float
    clipped_by_eeg_onset: bool
    valid: bool


def resolve_baseline_window(
    n_time_frames: int,
    *,
    hop_sec: float = 0.1,
    pre_sec: float = BASELINE_PRE_SEC,
    buffer_sec: float = BASELINE_BUFFER_SEC,
    eeg_onset_rel_sec: Optional[float] = None,
    min_baseline_valid_sec: float = MIN_BASELINE_VALID_SEC,
    onset_t_sec: float = 0.0,
) -> BaselineWindow:
    """Compute the EEG-onset-aware baseline window for a single seizure.

    The signal time axis is assumed to be ``t_axis = (i / fs) - pre_sec``
    (see :func:`extract_seizure_window`), so signal time 0 = clinical
    onset and ER frame 0 corresponds to ``-pre_sec`` seconds.

    Window definition (relative to clinical onset):
    ``baseline_end_sec = min(0, eeg_onset_rel_sec) - buffer_sec``;
    ``baseline_start_sec = -pre_sec``.

    A missing ``eeg_onset_rel_sec`` (None or NaN) reduces to the legacy
    ``[-pre_sec, -buffer_sec]`` window.
    """

    if pre_sec <= buffer_sec:
        raise ValueError(
            f"pre_sec ({pre_sec}) must exceed buffer_sec ({buffer_sec})"
        )

    if eeg_onset_rel_sec is None or (
        isinstance(eeg_onset_rel_sec, float) and np.isnan(eeg_onset_rel_sec)
    ):
        earliest_onset_rel_sec = 0.0
        clipped = False
    else:
        earliest_onset_rel_sec = min(0.0, float(eeg_onset_rel_sec))
        clipped = earliest_onset_rel_sec < 0.0

    start_sec = -float(pre_sec) + float(onset_t_sec)
    end_sec = earliest_onset_rel_sec - float(buffer_sec) + float(onset_t_sec)
    valid_sec = end_sec - start_sec

    onset_frame_offset = int(round(float(onset_t_sec) / float(hop_sec)))
    bl_start_frame = max(0, onset_frame_offset)
    bl_end_frame_unclamped = int(
        round((float(pre_sec) + earliest_onset_rel_sec - float(buffer_sec)) / float(hop_sec))
    ) + onset_frame_offset
    bl_end_frame = min(n_time_frames, max(bl_start_frame, bl_end_frame_unclamped))

    valid = valid_sec >= float(min_baseline_valid_sec) and bl_end_frame > bl_start_frame

    return BaselineWindow(
        start_idx=int(bl_start_frame),
        end_idx=int(bl_end_frame),
        start_sec=float(start_sec),
        end_sec=float(end_sec),
        valid_sec=float(valid_sec),
        clipped_by_eeg_onset=bool(clipped),
        valid=bool(valid),
    )


@dataclass(frozen=True)
class DetectionWindow:
    """Preview detection window for per-channel ER onset timing.

    This is intentionally a preview-only helper for sentinel inspection
    before formal Step 3 lambda calibration exists. The start of the
    window is clamped to the later of:

    1. the resolved EEG-aware ``baseline_end_sec``; and
    2. a user-chosen search floor (default ``-120s`` relative to
       clinical onset).
    """

    start_idx: int
    end_idx: int
    start_sec: float
    end_sec: float
    valid_sec: float
    valid: bool


@dataclass(frozen=True)
class EROnsetPreview:
    """Preview-only per-channel ER onset estimate.

    ``detected=False`` means the search window was valid but the
    cumulative statistic never crossed ``threshold``. That is different
    from an invalid window and should be treated as "unreached", not as a
    fabricated onset time.
    """

    detected: bool
    onset_idx: Optional[int]
    onset_sec: Optional[float]
    peak_stat: float
    bias: float
    threshold: float


def _peak_positive_cusum(x_1d: np.ndarray, *, bias: float) -> float:
    """Return the peak one-sided cumulative statistic over ``x_1d``."""

    x_1d = np.asarray(x_1d, dtype=np.float64)
    if x_1d.ndim != 1:
        raise ValueError("x_1d must be 1D")

    stat = 0.0
    peak_stat = 0.0
    for val in x_1d:
        if not np.isfinite(val):
            stat = 0.0
            continue
        stat = max(0.0, stat + float(val) - float(bias))
        peak_stat = max(peak_stat, stat)
    return float(peak_stat)


def preview_threshold_from_baseline(
    z_1d: np.ndarray,
    baseline_idx_window: Tuple[int, int],
    *,
    bias: float = 0.5,
    threshold_margin: float = 1.0,
) -> float:
    """Channel-specific preview threshold from the baseline peak statistic.

    Preview mode should not use a single global magic threshold across all
    channels and seizures. The whole point of the baseline z-score is that
    channels differ. We therefore take the largest one-sided cumulative
    deviation observed inside the resolved baseline window and require the
    detection window to exceed that by ``threshold_margin``.
    """

    z_1d = np.asarray(z_1d, dtype=np.float64)
    if z_1d.ndim != 1:
        raise ValueError("z_1d must be 1D")
    i0, i1 = int(baseline_idx_window[0]), int(baseline_idx_window[1])
    if i0 < 0 or i1 > z_1d.shape[0] or i0 >= i1:
        raise ValueError(
            f"baseline_idx_window={baseline_idx_window} out of range for n_t={z_1d.shape[0]}"
        )

    baseline_peak = _peak_positive_cusum(z_1d[i0:i1], bias=float(bias))
    return float(baseline_peak + float(threshold_margin))


def resolve_detection_window(
    n_time_frames: int,
    *,
    hop_sec: float = 0.1,
    pre_sec: float = BASELINE_PRE_SEC,
    baseline_end_sec: float = -60.0,
    start_floor_sec: float = -120.0,
    end_sec: float = 30.0,
    onset_t_sec: float = 0.0,
) -> DetectionWindow:
    """Resolve a preview detection window relative to clinical onset.

    The preview use-case is: inspect whether ER can recover a recruitment
    order *before* clinical onset. We therefore search after the resolved
    baseline ends, but allow the caller to cap how far back the search can
    go using ``start_floor_sec``.
    """

    start_sec = max(float(baseline_end_sec), float(start_floor_sec)) + float(onset_t_sec)
    end_sec_abs = float(end_sec) + float(onset_t_sec)
    valid_sec = end_sec_abs - start_sec

    onset_frame_offset = int(round(float(onset_t_sec) / float(hop_sec)))
    start_idx = int(round((float(pre_sec) + start_sec - float(onset_t_sec)) / float(hop_sec)))
    end_idx = int(round((float(pre_sec) + end_sec_abs - float(onset_t_sec)) / float(hop_sec)))
    start_idx = max(0, min(n_time_frames, start_idx + onset_frame_offset))
    end_idx = max(start_idx, min(n_time_frames, end_idx + onset_frame_offset))
    valid = valid_sec > 0.0 and end_idx > start_idx

    return DetectionWindow(
        start_idx=int(start_idx),
        end_idx=int(end_idx),
        start_sec=float(start_sec),
        end_sec=float(end_sec_abs),
        valid_sec=float(valid_sec),
        valid=bool(valid),
    )


def detect_er_onset_preview(
    z_1d: np.ndarray,
    t_axis_sec: np.ndarray,
    detection_idx_window: Tuple[int, int],
    *,
    bias: float = 0.5,
    threshold: float = 5.0,
) -> EROnsetPreview:
    """Preview-only one-sided ER onset detector.

    This is *not* the final Step 3 implementation. It is a minimal,
    deterministic preview that answers one question for sentinel seizures:
    does a channel show a sustained positive ER deviation inside the
    pre-clinical search window, and if so, when is the first threshold
    crossing?
    """

    z_1d = np.asarray(z_1d, dtype=np.float64)
    t_axis_sec = np.asarray(t_axis_sec, dtype=np.float64)
    if z_1d.ndim != 1:
        raise ValueError("z_1d must be 1D")
    if t_axis_sec.ndim != 1 or t_axis_sec.shape[0] != z_1d.shape[0]:
        raise ValueError("t_axis_sec must be 1D and aligned with z_1d")

    i0, i1 = int(detection_idx_window[0]), int(detection_idx_window[1])
    if i0 < 0 or i1 > z_1d.shape[0] or i0 >= i1:
        raise ValueError(
            f"detection_idx_window={detection_idx_window} out of range for n_t={z_1d.shape[0]}"
        )

    stat = 0.0
    peak_stat = 0.0
    for idx in range(i0, i1):
        val = z_1d[idx]
        if not np.isfinite(val):
            stat = 0.0
            continue
        stat = max(0.0, stat + float(val) - float(bias))
        peak_stat = max(peak_stat, stat)
        if stat >= float(threshold):
            return EROnsetPreview(
                detected=True,
                onset_idx=int(idx),
                onset_sec=float(t_axis_sec[idx]),
                peak_stat=float(peak_stat),
                bias=float(bias),
                threshold=float(threshold),
            )

    return EROnsetPreview(
        detected=False,
        onset_idx=None,
        onset_sec=None,
        peak_stat=float(peak_stat),
        bias=float(bias),
        threshold=float(threshold),
    )
