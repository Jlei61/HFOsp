"""Topic 5 V3a mode-transition helpers.

This module intentionally stays on pure configuration for Task 0. Later
tasks add event-window extraction, geometry, dynamics, and avalanche-flux
estimators on top of this config contract. See
docs/superpowers/plans/2026-07-02-topic5-v3a-mode-transition.md for the
full task list; treat this line as exploratory pending the pilot-lock gate.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CFG = _ROOT / "config" / "topic5_v3.yaml"


def load_v3_config(path: str | Path | None = None) -> dict:
    """Load the V3a mode-transition YAML config as a plain dict."""
    cfg_path = Path(path) if path is not None else _DEFAULT_CFG
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, Mapping):
        raise ValueError(f"V3a config must be a mapping: {cfg_path}")
    return dict(cfg)


def _window_index_range(relt: np.ndarray, lo: float, hi: float) -> tuple[int, int] | None:
    """Half-open ``(start, stop)`` sample indices where ``lo <= relt <= hi``.

    ``relt`` is monotone increasing, so the mask is contiguous. Returns
    ``None`` if the window is empty. Local copy of the pattern in
    ``scripts/_topic5_v2_crit_io.py::window_index_range``.
    """
    relt = np.asarray(relt, dtype=float)
    mask = (relt >= float(lo)) & (relt <= float(hi))
    if not mask.any():
        return None
    idx = np.flatnonzero(mask)
    return int(idx[0]), int(idx[-1] + 1)


def i1_range(
    eeg_onset_rel: float, eeg_offset_rel: float, duration: float, cfg: dict
) -> tuple[float, float, bool]:
    """Early-ictal I1 window relative to eeg onset.

    Primary (``duration >= I1_min_duration_sec``): ``[onset+I1_rel[0],
    onset+I1_rel[1]]``. Short-seizure fallback: ``[onset+I1_rel[0],
    offset - I1_post_guard_sec]`` — offset-based, never ``0.25*duration``
    (plan rev2). ``i1_eligible`` requires at least one full ``window_sec``.
    """
    ph = cfg["phases"]
    onset = float(eeg_onset_rel)
    offset = float(eeg_offset_rel)
    dur = float(duration)
    lo = onset + ph["I1_rel"][0]
    if dur >= ph["I1_min_duration_sec"]:
        hi = onset + ph["I1_rel"][1]
    else:
        hi = offset - ph["I1_post_guard_sec"]
    i1_eligible = bool((hi - lo) >= ph["window_sec"])
    return lo, hi, i1_eligible


def phase_bin_range(
    relt: np.ndarray,
    eeg_onset_rel: float,
    eeg_offset_rel: float,
    duration: float,
    phase: str,
    cfg: dict,
    onset_shift: float = 0.0,
) -> tuple[int, int] | None:
    """Half-open sample-index range for one named phase bin.

    Anchored on ``eeg_onset_rel + onset_shift`` (onset jitter perturbs the
    anchor) for P0..O and I1; I2/I3 are ictal-fraction of ``[anchor,
    offset]`` (offset itself does not shift); Post is relative to
    ``eeg_offset_rel`` only and is never shifted by onset jitter.
    """
    ph = cfg["phases"]
    anchor = float(eeg_onset_rel) + float(onset_shift)
    offset = float(eeg_offset_rel)

    if phase == "P0":
        lo, hi = anchor - 120.0, anchor - 90.0
    elif phase == "P1":
        lo, hi = anchor - 90.0, anchor - 60.0
    elif phase == "P2":
        lo, hi = anchor - 60.0, anchor - 30.0
    elif phase == "P3":
        lo, hi = anchor + ph["P3_rel"][0], anchor + ph["P3_rel"][1]
    elif phase == "O":
        lo, hi = anchor + ph["O_rel"][0], anchor + ph["O_rel"][1]
    elif phase == "I1":
        lo, hi, _ = i1_range(anchor, offset, duration, cfg)
    elif phase == "I2":
        lo, hi = anchor + 0.25 * (offset - anchor), anchor + 0.75 * (offset - anchor)
    elif phase == "I3":
        lo, hi = anchor + 0.75 * (offset - anchor), offset
    elif phase == "Post":
        lo, hi = offset, offset + ph["span_post_sec"]
    else:
        raise ValueError(f"unknown phase: {phase!r}")

    return _window_index_range(relt, lo, hi)


def sliding_windows(
    relt: np.ndarray, start: int, stop: int, window_sec: float, step_sec: float
) -> list[tuple[int, int]]:
    """Sliding ``(window_start_idx, window_end_idx)`` half-open pairs over ``[start, stop)``.

    Samples-per-second is derived from the median spacing of ``relt``. Only
    full-length windows are emitted: a window is kept only if it spans the
    complete ``window_sec`` within ``[start, stop)``; the partial trailing
    tail is dropped rather than clipped to ``stop``. The ``>= 3``-sample
    guard is kept as a defensive floor (subsumed for realistic configs, but
    still checked).
    """
    relt = np.asarray(relt, dtype=float)
    dt = float(np.median(np.diff(relt)))
    window_n = int(round(window_sec / dt))
    step_n = int(round(step_sec / dt))
    windows: list[tuple[int, int]] = []
    ws = start
    while ws + window_n <= stop:
        we = ws + window_n
        if we - ws >= 3:
            windows.append((ws, we))
        ws += step_n
    return windows
