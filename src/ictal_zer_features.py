"""z-ER binned tensor feature extraction for topic5 PR-1 Step 2.

Upgrades clustering feature from per-channel ``t_ER_onset`` (1 scalar
per channel) to per-channel × time-bin ``z-ER`` mean (B scalars per
channel). The new feature captures both:

- pre-onset rank   (in the (-50, 0)s bin)
- post-onset spread + intensity (in the (0, 50), (50, 150), (150, 200)s bins)

Distance / linkage / k-selection / outlier-split unchanged — only the
feature vector grows from N → N×B dim. Spearman + UPGMA + silhouette
remain rank-robust at this dimensionality.

The z-ER tensor is NOT persisted in v2.3 Layer A JSON (only ``t_onset_sec``
is). Must re-extract from raw via ``extract_seizure_window`` →
``compute_er`` → ``baseline_zscore_er``. Adaptive ``post_sec`` retry
matches ``scripts/plot_ictal_er_atlas.py`` behavior.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


# Default bin edges (seconds, relative to clinical onset).
# Schroeder/Panagiotopoulou-inspired: pre-onset baseline / pre-onset
# imminent / post-onset early / post-onset spread / post-onset late.
DEFAULT_BINS_SEC: Tuple[Tuple[float, float], ...] = (
    (-200.0, -50.0),
    (-50.0, 0.0),
    (0.0, 50.0),
    (50.0, 150.0),
    (150.0, 200.0),
)

DEFAULT_BAND_KEYS = ("gamma_ER", "broad_ER")


def bin_zer_to_features(
    z_er: np.ndarray,
    t_axis: np.ndarray,
    bins: Sequence[Tuple[float, float]],
) -> Tuple[np.ndarray, List[Optional[Tuple[int, int]]]]:
    """Bin a z-ER tensor (n_channels, n_frames) into mean-per-bin features.

    Parameters
    ----------
    z_er : (n_channels, n_frames)
    t_axis : (n_frames,) seconds relative to clinical onset
    bins : sequence of (t_lo, t_hi) tuples in seconds

    Returns
    -------
    feature : (n_channels * n_bins,) flat vector, channel-major:
        feature[ch * n_bins + b] = mean(z_er[ch, frames in bin b])
    bin_idx_used : list of (i_start, i_end) per bin (None if bin uncovered)

    NaN handling: input NaN propagates (np.nanmean), so a channel with all-
    NaN row produces all-NaN bins. Bins with zero overlapping frames also
    return NaN.
    """
    z = np.asarray(z_er, dtype=np.float64)
    t = np.asarray(t_axis, dtype=np.float64)
    if z.ndim != 2 or t.ndim != 1 or z.shape[1] != t.shape[0]:
        raise ValueError(
            f"shape mismatch: z {z.shape}, t {t.shape}"
        )
    n_ch = z.shape[0]
    n_bins = len(bins)
    out = np.full(n_ch * n_bins, np.nan, dtype=np.float64)
    bin_idx_used: List[Optional[Tuple[int, int]]] = []
    for b, (t_lo, t_hi) in enumerate(bins):
        mask = (t >= float(t_lo)) & (t < float(t_hi))
        if not mask.any():
            bin_idx_used.append(None)
            continue
        idx = np.where(mask)[0]
        bin_idx_used.append((int(idx[0]), int(idx[-1] + 1)))
        sub = z[:, idx]
        # nanmean: empty all-NaN row → NaN with RuntimeWarning; suppress.
        with np.errstate(invalid="ignore"):
            means = np.nanmean(sub, axis=1)
        # All-NaN slice (no finite frame) → leave NaN; np.nanmean already
        # returns NaN in that case (older numpy may issue warning).
        for ch in range(n_ch):
            out[ch * n_bins + b] = float(means[ch])
    return out, bin_idx_used


def stack_features_to_matrix(
    features: Sequence[Optional[np.ndarray]],
    seizure_ids: Sequence[str],
) -> Tuple[np.ndarray, List[str]]:
    """Stack per-seizure feature vectors into a (n_features, n_kept_seizures)
    matrix, dropping seizures with None feature (extraction failed).
    """
    if len(features) != len(seizure_ids):
        raise ValueError("features and seizure_ids length mismatch")
    kept: List[Tuple[str, np.ndarray]] = []
    for sid, f in zip(seizure_ids, features):
        if f is None:
            continue
        kept.append((sid, np.asarray(f, dtype=np.float64)))
    if not kept:
        return np.zeros((0, 0), dtype=np.float64), []
    n_feat = kept[0][1].shape[0]
    for _, f in kept:
        if f.shape[0] != n_feat:
            raise ValueError("inconsistent feature length across seizures")
    matrix = np.column_stack([f for _, f in kept])
    kept_ids = [sid for sid, _ in kept]
    return matrix, kept_ids


def extract_zer_binned_for_subject(
    subject: str,
    band_key: str,
    seizure_indices: Iterable[int],
    *,
    bins: Sequence[Tuple[float, float]] = DEFAULT_BINS_SEC,
    pre_sec: float = 300.0,
    post_sec_attempts: Sequence[float] = (300.0, 200.0, 100.0, 60.0, 30.0),
    win_sec: float = 1.0,
    hop_sec: float = 0.1,
    results_root=None,
    reference: str = "car",
) -> Tuple[List[Optional[np.ndarray]], List[str], List[str], List[str]]:
    """For each seizure index, extract z-ER, bin it, return feature vector.

    Returns
    -------
    features : list of (n_ch * n_bins,) np.ndarray or None (if extraction
        failed for any reason — boundary, baseline_invalid, etc.)
    seizure_ids : list of seizure_id strings (same length as features)
    channel_names : list of channel names (from the FIRST successful
        extraction; assumed consistent across seizures of the same subject)
    drop_reasons : list of failure reason strings (empty for successful)

    Each feature vector is computed on the band-specific z-ER (gamma or broad).
    """
    from pathlib import Path
    from src.ictal_onset_extraction import (
        BROAD_ER_BANDS,
        GAMMA_ER_BANDS,
        baseline_zscore_er,
        compute_er,
        extract_seizure_window,
        resolve_baseline_window,
    )

    band_cfg = {"gamma_ER": GAMMA_ER_BANDS, "broad_ER": BROAD_ER_BANDS}[band_key]
    if results_root is None:
        results_root = Path(__file__).resolve().parents[1] / "results"

    features: List[Optional[np.ndarray]] = []
    seizure_ids: List[str] = []
    channel_names: List[str] = []
    drop_reasons: List[str] = []

    for sz_idx in seizure_indices:
        seizure_ids.append(str(sz_idx))
        sw = None
        last_exc = None
        for post_sec in post_sec_attempts:
            try:
                sw = extract_seizure_window(
                    subject, int(sz_idx),
                    pre_sec=pre_sec, post_sec=float(post_sec),
                    results_root=results_root, reference=reference,
                )
                break
            except (ValueError, IndexError) as exc:
                last_exc = exc
                continue
        if sw is None:
            features.append(None)
            drop_reasons.append(f"window_extraction_failed: {last_exc}")
            continue
        # update seizure_ids[-1] to actual id
        seizure_ids[-1] = sw.seizure_id

        eeg_rel = (
            sw.eeg_onset_epoch - sw.clin_onset_epoch
            if sw.eeg_onset_epoch is not None else None
        )
        try:
            er = compute_er(
                sw.signal, fs=sw.fs,
                fast_band=band_cfg["fast"], slow_band=band_cfg["slow"],
                win_sec=win_sec, hop_sec=hop_sec,
            )
        except Exception as exc:
            features.append(None)
            drop_reasons.append(f"compute_er_failed: {exc}")
            continue
        n_t = er.shape[1]
        bw = resolve_baseline_window(
            n_t, hop_sec=hop_sec, pre_sec=sw.pre_sec,
            eeg_onset_rel_sec=eeg_rel,
        )
        if not bw.valid:
            features.append(None)
            drop_reasons.append("baseline_invalid")
            continue
        try:
            z = baseline_zscore_er(
                er, (bw.start_idx, bw.end_idx), hop_sec=hop_sec,
            )
        except Exception as exc:
            features.append(None)
            drop_reasons.append(f"baseline_zscore_failed: {exc}")
            continue
        # STRICT channel-order check (audit 2026-05-10): downstream
        # stack_features_to_matrix only verifies feature length, not
        # channel name/order. If a later seizure's ch_names disagrees with
        # the first successful one, the bin-major flat vectors point to
        # different channels at the same row index — silent contamination.
        cur_ch = list(sw.ch_names)
        if not channel_names:
            channel_names = cur_ch
        elif cur_ch != channel_names:
            features.append(None)
            drop_reasons.append(
                f"channel_order_mismatch (n_first={len(channel_names)}, "
                f"n_this={len(cur_ch)})"
            )
            continue
        t_axis = (np.arange(n_t) * hop_sec + win_sec / 2.0) - sw.pre_sec
        feat, _ = bin_zer_to_features(z, t_axis, bins)
        features.append(feat)
        drop_reasons.append("")

    return features, seizure_ids, channel_names, drop_reasons
