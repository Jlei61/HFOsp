"""PR-T3-1 v2.1 — Layer A: Ictal ER-rank producer (scope-restricted).

This module implements Layer A of the v2.1 pivot
(`docs/archive/topic3/pr_t3_1_pivot_to_pr6a_er_ranking_2026-05-03.md`):

- Per-channel Page-Hinkley CUSUM alarm time ``n_d`` against PR-6A's
  baseline-z-scored ER (gamma_ER + broad_ER).
- Per-subject λ calibration on the baseline window targeting a given
  false-positive rate.
- Per-seizure rank vector (fractional rank for ties), per-seizure status
  (``ok`` / ``onset_tied`` / ``onset_unreached`` / ``baseline_invalid``).
- Cross-seizure ``r_sz`` (median rank per channel) and stability
  ``s_sz`` (median pairwise Spearman of rank vectors).
- Time-shifted ``r_sz`` null surrogate (shifted onsets inside baseline
  window, avoiding real seizures by ± exclusion radius).

**SCOPE-RESTRICTED**: Layer A is SOZ-purpose only. It does NOT
implement template-ictal alignment / Smith 2022 H1 / H1' / r_template
correlation. PR-6A's H1/H1' main line is superseded; this module
reuses PR-6A Step 0-2 ER + baseline z-score helpers via
``src.ictal_onset_extraction`` but does NOT extend the H1/H1' machinery.

**CUSUM formulation — deviation from plan §3.3**:

The pivot plan §3.3 lists the alarm criterion as ``M[n] - U[n] >= λ``
where ``U[n] = max(0, U[n-1] + z[n] - bias)`` and ``M[n] = max_{k<=n}
U[k]``. That formulation does NOT fire for a sustained upward step
(during a step, U is monotonically non-decreasing, so ``M - U = 0`` and
no alarm ever triggers). The correct upward-change formulation for the
clamped CUSUM ``U`` is simply ``U[n] >= λ`` (which is what PR-6A
Step 2's ``detect_er_onset_preview`` already uses); for the unclamped
``S = sum(z - bias)`` form it would be ``S[n] - min(S[k<=n]) >= λ``.
Both reduce to the same alarm time on a clean step. We adopt the
clamped ``U >= λ`` form here and note the plan-text deviation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.ictal_onset_extraction import (
    BROAD_ER_BANDS,
    GAMMA_ER_BANDS,
    baseline_zscore_er,
    compute_er,
    resolve_baseline_window,
    resolve_detection_window,
)


__all__ = [
    "compute_cusum_n_d",
    "calibrate_lambda_per_subject",
    "rank_channels_by_n_d",
    "compute_seizure_status",
    "SeizureStatus",
]


SeizureStatus = str  # one of: "ok" / "onset_tied" / "onset_unreached" / "baseline_invalid"


def compute_cusum_n_d(
    z_er_1d: np.ndarray,
    lambda_thresh: float,
    *,
    bias: float = 0.5,
    detection_idx_window: Optional[Tuple[int, int]] = None,
) -> Optional[int]:
    """Page-Hinkley alarm frame index for upward change in z-ER.

    Clamped CUSUM: ``U[n] = max(0, U[n-1] + z[n] - bias)``. Alarm fires
    at the first frame inside ``detection_idx_window`` where
    ``U[n] >= lambda_thresh``.

    Parameters
    ----------
    z_er_1d
        Per-channel baseline-z-scored ER trace (1D, length n_frames).
    lambda_thresh
        Detection threshold on ``U``. Calibrated per subject from
        baseline ``U`` percentiles via ``calibrate_lambda_per_subject``.
    bias
        Drift correction (Bartolomei 2008 default 0.5). A pure-noise z
        with zero mean shrinks ``U`` toward zero with this bias; raising
        bias makes the detector less sensitive.
    detection_idx_window
        ``(i_start, i_end)`` half-open frame slice where alarms count.
        ``None`` means "anywhere in z_er_1d". CUSUM accumulates from
        frame 0 (so excursions starting before the window can still
        trigger inside the window).

    Returns
    -------
    int or None
        Frame index of first alarm, or ``None`` if no alarm fires
        inside the detection window.

    Notes
    -----
    NaN / non-finite values reset ``U`` to 0 (matching PR-6A Step 2's
    ``detect_er_onset_preview`` behavior). This is conservative — a
    drop-out in the input does not generate a spurious alarm but also
    does not propagate the prior accumulation across the gap.
    """
    z = np.asarray(z_er_1d, dtype=np.float64)
    if z.ndim != 1:
        raise ValueError("z_er_1d must be 1D")
    if not np.isfinite(lambda_thresh) or lambda_thresh <= 0:
        raise ValueError(
            f"lambda_thresh must be positive finite, got {lambda_thresh}"
        )
    n = z.shape[0]
    if detection_idx_window is None:
        i0, i1 = 0, n
    else:
        i0, i1 = int(detection_idx_window[0]), int(detection_idx_window[1])
        if i0 < 0 or i1 > n or i0 >= i1:
            raise ValueError(
                f"detection_idx_window={detection_idx_window} out of range for n={n}"
            )

    bias_f = float(bias)
    lam = float(lambda_thresh)
    stat = 0.0
    for idx in range(n):
        val = z[idx]
        if not np.isfinite(val):
            stat = 0.0
            continue
        stat = max(0.0, stat + float(val) - bias_f)
        if i0 <= idx < i1 and stat >= lam:
            return int(idx)
    return None


def calibrate_lambda_per_subject(
    z_er_baseline: np.ndarray,
    *,
    fpr_target_per_hour: float = 1.0,
    bias: float = 0.5,
    hop_sec: float = 0.1,
    lambda_min: float = 1.0,
    lambda_max: float = 100.0,
    lambda_n_grid: int = 199,
) -> float:
    """Per-subject λ calibration on baseline z-ER frames.

    Finds the smallest ``λ`` such that running ``compute_cusum_n_d`` on
    every channel of the baseline window emits **at most** ``fpr_target
    * total_baseline_hours`` alarms in expectation. Multi-channel
    baseline alarms are pooled (we want subject-level FPR, not
    per-channel).

    Parameters
    ----------
    z_er_baseline
        ``(n_channels, n_baseline_frames)`` baseline-z-scored ER.
    fpr_target_per_hour
        Subject-level false-positive rate target. Plan §3.3 default = 1.
    bias
        CUSUM drift correction (must match the value used in
        ``compute_cusum_n_d`` at detection time).
    hop_sec
        Frame hop in seconds (PR-6A default 0.1).
    lambda_min, lambda_max, lambda_n_grid
        Search grid for ``λ``. The output is the smallest grid value
        whose pooled FP count is ≤ the target.

    Returns
    -------
    float
        Calibrated ``λ``. Falls back to ``lambda_max`` if every grid
        value exceeds the target (e.g. very noisy baseline).
    """
    z = np.asarray(z_er_baseline, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError(
            f"z_er_baseline must be 2D (n_channels, n_frames), got shape {z.shape}"
        )
    if z.shape[1] < 2:
        raise ValueError(
            f"z_er_baseline must have at least 2 frames, got {z.shape[1]}"
        )
    if fpr_target_per_hour <= 0:
        raise ValueError("fpr_target_per_hour must be > 0")
    if hop_sec <= 0:
        raise ValueError("hop_sec must be > 0")
    if lambda_min <= 0 or lambda_max <= lambda_min:
        raise ValueError("require 0 < lambda_min < lambda_max")
    if lambda_n_grid < 2:
        raise ValueError("lambda_n_grid must be >= 2")

    n_channels, n_frames = z.shape
    baseline_hours = (n_frames * hop_sec) / 3600.0
    fp_budget = float(fpr_target_per_hour) * baseline_hours

    # Per-channel running CUSUM peak — counting an alarm any frame the
    # statistic crosses λ. We re-arm the test by resetting U to 0 after
    # each alarm so we get an honest "alarms per hour" estimate rather
    # than a single first-crossing count.
    bias_f = float(bias)

    def count_alarms_at_lambda(lam: float) -> int:
        total = 0
        for ch in range(n_channels):
            stat = 0.0
            for idx in range(n_frames):
                val = z[ch, idx]
                if not np.isfinite(val):
                    stat = 0.0
                    continue
                stat = max(0.0, stat + float(val) - bias_f)
                if stat >= lam:
                    total += 1
                    stat = 0.0
        return total

    grid = np.linspace(float(lambda_min), float(lambda_max), int(lambda_n_grid))
    chosen = float(lambda_max)
    for lam in grid:
        if count_alarms_at_lambda(float(lam)) <= fp_budget:
            chosen = float(lam)
            break
    return chosen


def rank_channels_by_n_d(
    n_d_per_channel: Mapping[str, Optional[float]],
    *,
    tie_eps_frames: float = 0.5,
) -> Dict[str, Optional[float]]:
    """Convert per-channel alarm frame indices to fractional ranks.

    Plan §3.4 contract:

    - Channels with ``None`` n_d (no alarm) get ``None`` rank.
    - Among channels with finite ``n_d``, smaller ``n_d`` = earlier =
      lower rank (rank 0 is earliest).
    - Ties: channels whose ``n_d`` values differ by at most
      ``tie_eps_frames`` (default 0.5 frame ≈ 50 ms at hop=100 ms) get
      the average rank.

    Parameters
    ----------
    n_d_per_channel
        ``{channel_name: n_d_frame_index_or_None}``.
    tie_eps_frames
        Window (in frames) within which two ``n_d`` values are
        considered tied. Default 0.5 corresponds to 50 ms at PR-6A's
        100 ms hop.

    Returns
    -------
    dict
        ``{channel_name: rank_or_None}`` preserving input keys.
    """
    finite_items: List[Tuple[str, float]] = []
    none_channels: List[str] = []
    for ch, val in n_d_per_channel.items():
        if val is None:
            none_channels.append(ch)
            continue
        v = float(val)
        if not np.isfinite(v):
            none_channels.append(ch)
            continue
        finite_items.append((ch, v))

    if not finite_items:
        return {ch: None for ch in n_d_per_channel}

    finite_items.sort(key=lambda x: (x[1], x[0]))
    ranks: Dict[str, float] = {}
    n = len(finite_items)
    i = 0
    while i < n:
        j = i + 1
        while j < n and finite_items[j][1] - finite_items[i][1] <= float(tie_eps_frames):
            j += 1
        avg_rank = (i + j - 1) / 2.0
        for k in range(i, j):
            ranks[finite_items[k][0]] = float(avg_rank)
        i = j

    out: Dict[str, Optional[float]] = {}
    for ch in n_d_per_channel:
        if ch in ranks:
            out[ch] = ranks[ch]
        else:
            out[ch] = None
    return out


@dataclass(frozen=True)
class SeizureStatusResult:
    """Output of ``compute_seizure_status``."""

    status: SeizureStatus
    n_active: int
    n_total: int
    fast_recruit_fraction: float


def compute_seizure_status(
    n_d_per_channel: Mapping[str, Optional[float]],
    *,
    n_total: int,
    onset_idx: int,
    fast_recruit_window_frames: int = 10,
    tied_fraction_threshold: float = 0.6,
    unreached_active_fraction_threshold: float = 0.3,
) -> SeizureStatusResult:
    """Three-state seizure status classification (plan §3.3).

    Parameters
    ----------
    n_d_per_channel
        ``{channel_name: n_d_frame_index_or_None}``.
    n_total
        Total number of analysis channels (denominator for active /
        recruited fraction). Pass the cohort channel count, NOT
        ``len(n_d_per_channel)`` — this matters when channels were
        excluded earlier (e.g. baseline-invalid).
    onset_idx
        Frame index of clinical onset.
    fast_recruit_window_frames
        Frames after onset within which a channel counts as
        "fast-recruited" (default 10 frames = 1 s at PR-6A hop).
    tied_fraction_threshold
        If the fast-recruited fraction exceeds this, status =
        ``onset_tied``.
    unreached_active_fraction_threshold
        If the active (n_d not None) fraction is below this, status =
        ``onset_unreached``.

    Returns
    -------
    SeizureStatusResult
        Status + diagnostic counts.
    """
    if n_total <= 0:
        raise ValueError(f"n_total must be > 0, got {n_total}")

    active_n_ds: List[float] = []
    for v in n_d_per_channel.values():
        if v is None:
            continue
        fv = float(v)
        if not np.isfinite(fv):
            continue
        active_n_ds.append(fv)
    n_active = len(active_n_ds)
    active_fraction = n_active / float(n_total)

    if active_fraction < float(unreached_active_fraction_threshold):
        return SeizureStatusResult(
            status="onset_unreached",
            n_active=n_active,
            n_total=int(n_total),
            fast_recruit_fraction=0.0,
        )

    fast_recruited = sum(
        1 for v in active_n_ds
        if int(onset_idx) <= v <= int(onset_idx) + int(fast_recruit_window_frames)
    )
    fast_recruit_fraction = fast_recruited / float(n_total)

    if fast_recruit_fraction > float(tied_fraction_threshold):
        return SeizureStatusResult(
            status="onset_tied",
            n_active=n_active,
            n_total=int(n_total),
            fast_recruit_fraction=fast_recruit_fraction,
        )

    return SeizureStatusResult(
        status="ok",
        n_active=n_active,
        n_total=int(n_total),
        fast_recruit_fraction=fast_recruit_fraction,
    )
