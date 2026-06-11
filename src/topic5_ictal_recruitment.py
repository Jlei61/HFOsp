"""Topic 5 Stage-2 early-ictal recruitment-time instrument (PURE math, no I/O).

Spec: docs/superpowers/specs/2026-06-10-topic5-ictal-recruitment-stage2-design.md
Plan: docs/superpowers/plans/2026-06-10-topic5-ictal-recruitment-stage2.md

All feature traces land on a COMMON hop grid (HOP=0.1s) so the 4 fused detectors
(line-length / broadband / HFA / spectral-edge) and the held-out ER reference are
directly comparable. Frame j covers [j*hop, j*hop + win]; the authoritative cross-feature
onset TIME is the CUSUM wrapper's t_onset_sec = frame*hop + win/2 - pre (center, rel
clinical onset), so onsets from features with different `win` are still comparable.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from scipy.signal import spectrogram
from scipy.stats import rankdata, spearmanr

from src.ictal_er_rank import compute_cusum_n_d_with_time, calibrate_lambda_per_subject

Z_SUSTAIN = 2.0   # robust-z floor the feature must HOLD post-onset (a feature-z, NOT lambda)


# ---------------------------------------------------------------------------
# Task 1 — line-length trace
# ---------------------------------------------------------------------------
def _frame_starts(n_samples: int, fs: float, win_sec: float, hop_sec: float) -> np.ndarray:
    """Sample indices where each analysis frame starts; last frame must fit win."""
    win = int(round(win_sec * fs))
    hop = int(round(hop_sec * fs))
    if win > n_samples:
        raise ValueError(f"win {win} samples > signal {n_samples}")
    last = n_samples - win
    return np.arange(0, last + 1, hop)


def line_length_trace(signal, fs, *, win_sec: float = 1.0, hop_sec: float = 0.1):
    """F1 line-length: per-channel sliding-window sum of |diff(signal)| (time domain).

    Returns (trace[n_ch, n_frames], t_frames[n_frames]) where t = frame_start/fs.
    """
    sig = np.asarray(signal, dtype=np.float64)
    if sig.ndim != 2:
        raise ValueError("signal must be 2D (n_channels, n_samples)")
    fs = float(fs)
    win = int(round(win_sec * fs))
    starts = _frame_starts(sig.shape[1], fs, win_sec, hop_sec)
    absdiff = np.abs(np.diff(sig, axis=1))                    # (n_ch, n_samp-1)
    csum = np.concatenate([np.zeros((sig.shape[0], 1)), np.cumsum(absdiff, axis=1)], axis=1)
    out = np.empty((sig.shape[0], starts.size), dtype=np.float64)
    for j, s in enumerate(starts):
        e = min(s + win - 1, absdiff.shape[1])               # diff has n_samp-1 cols
        out[:, j] = csum[:, e] - csum[:, s]
    t = starts / fs
    return out, t


# ---------------------------------------------------------------------------
# Task 2 — band-power + spectral-edge traces (shared spectrogram)
# ---------------------------------------------------------------------------
def _spectrogram_on_hop(sig, fs, win_sec, hop_sec):
    """Shared spectrogram on the common hop grid. Returns f, t, Sxx (n_ch,n_f,n_t)."""
    sig = np.asarray(sig, dtype=np.float64)
    if sig.ndim != 2:
        raise ValueError("signal must be 2D (n_channels, n_samples)")
    fs = float(fs)
    nperseg = max(1, int(round(win_sec * fs)))
    if nperseg > sig.shape[1]:
        raise ValueError(f"win_sec={win_sec}s needs {nperseg} samples; have {sig.shape[1]}")
    noverlap = max(0, nperseg - max(1, int(round(hop_sec * fs))))
    f, t, Sxx = spectrogram(sig, fs=fs, nperseg=nperseg, noverlap=noverlap,
                            scaling="density", mode="psd", axis=-1)
    if Sxx.ndim == 2:        # single channel -> add ch axis
        Sxx = Sxx[np.newaxis, ...]
    return f, t, Sxx


def band_power_trace(signal, fs, *, band, win_sec=1.0, hop_sec=0.1):
    """F2/F3: per-channel log power summed over `band`. Nyquist-gated."""
    nyq = float(fs) / 2.0
    lo, hi = float(band[0]), float(band[1])
    if not (0.0 <= lo < hi):
        raise ValueError(f"band must be 0<=lo<hi, got {band}")
    if hi >= nyq:
        raise ValueError(f"band hi {hi} Hz >= Nyquist {nyq} Hz for fs={fs} (feature unavailable)")
    f, t, Sxx = _spectrogram_on_hop(signal, fs, win_sec, hop_sec)
    mask = (f >= lo) & (f <= hi)
    if not mask.any():
        raise ValueError(f"no FFT bins inside band={band}")
    power = Sxx[:, mask, :].sum(axis=1)                      # (n_ch, n_t)
    return np.log(np.maximum(power, 1e-30)), t


def spectral_edge_trace(signal, fs, *, edge=0.9, win_sec=1.0, hop_sec=0.1):
    """F5: per-channel spectral-edge frequency (cumulative PSD reaches `edge`)."""
    nyq = float(fs) / 2.0
    hi = min(127.0, nyq - 1.0)
    f, t, Sxx = _spectrogram_on_hop(signal, fs, win_sec, hop_sec)
    band = (f >= 1.0) & (f <= hi)
    fb = f[band]                                             # (n_f',)
    P = Sxx[:, band, :]                                      # (n_ch, n_f', n_t)
    csum = np.cumsum(P, axis=1)
    total = csum[:, -1:, :]
    frac = np.divide(csum, np.maximum(total, 1e-30))        # (n_ch, n_f', n_t)
    reached = frac >= float(edge)
    idx = np.argmax(reached, axis=1)                        # (n_ch, n_t)
    sef = fb[idx]
    return sef.astype(np.float64), t


# ---------------------------------------------------------------------------
# Task 3 — robust-z normalization
# ---------------------------------------------------------------------------
def baseline_robust_z(trace, baseline_idx_window, *, hop_sec=0.1, min_baseline_valid_sec=60.0):
    """Per-channel robust z against a baseline frame window. MAD=0 or too-short -> NaN row."""
    tr = np.asarray(trace, dtype=np.float64)
    if tr.ndim != 2:
        raise ValueError("trace must be 2D (n_ch, n_frames)")
    i0, i1 = int(baseline_idx_window[0]), int(baseline_idx_window[1])
    n_t = tr.shape[1]
    if i0 < 0 or i1 > n_t or i0 >= i1:
        raise ValueError(f"baseline_idx_window={baseline_idx_window} bad for n_t={n_t}")
    n_valid = i1 - i0
    min_frames = int(np.ceil(float(min_baseline_valid_sec) / float(hop_sec)))
    if n_valid < min_frames:
        return np.full_like(tr, np.nan)
    base = tr[:, i0:i1]
    med = np.median(base, axis=1, keepdims=True)
    mad = np.median(np.abs(base - med), axis=1, keepdims=True) * 1.4826
    mad = np.where(mad > 0.0, mad, np.nan)
    z = (tr - med) / mad
    z[np.isnan(mad).squeeze(axis=1)] = np.nan
    return z


# ---------------------------------------------------------------------------
# Task 4 — per-contact onset (CUSUM wrapper + no-onset / ambiguous)
# ---------------------------------------------------------------------------
def detect_contact_onset(z_trace_1d, *, lam, detection_idx_window, hop_sec=0.1,
                         win_sec=1.0, pre_sec=0.0, bias=0.5, sustain_sec=1.0,
                         z_sustain=Z_SUSTAIN):
    """Per-contact onset on a 1-D robust-z trace via clamped Page-Hinkley CUSUM.

    Wraps compute_cusum_n_d_with_time (src/ictal_er_rank.py): returns
    CusumOnsetResult(frame_idx, t_onset_sec) with
    t_onset_sec = frame_idx*hop + win/2 - pre (center convention, rel clinical onset).
    Ambiguous (§5.4) is judged on the FEATURE z, NOT lambda: a transient false alarm =
    the post-onset feature z does not HOLD >= z_sustain over sustain_sec -> void.
    Returns {detected, onset_frame, onset_sec, reason in {ok, unreached, ambiguous}}.
    """
    z = np.asarray(z_trace_1d, dtype=np.float64)
    res = compute_cusum_n_d_with_time(
        z, float(lam), bias=float(bias), detection_idx_window=detection_idx_window,
        hop_sec=float(hop_sec), win_sec=float(win_sec), pre_sec=float(pre_sec),
    )
    if res.frame_idx is None:
        return {"detected": False, "onset_frame": float("nan"),
                "onset_sec": float("nan"), "reason": "unreached"}
    idx = int(res.frame_idx)
    n_sus = max(1, int(round(float(sustain_sec) / float(hop_sec))))
    post = z[idx: idx + n_sus]
    post_med = np.nanmedian(post) if post.size else np.nan
    if not np.isfinite(post_med) or post_med < float(z_sustain):
        return {"detected": False, "onset_frame": float("nan"),
                "onset_sec": float("nan"), "reason": "ambiguous"}
    return {"detected": True, "onset_frame": idx,
            "onset_sec": float(res.t_onset_sec), "reason": "ok"}


# ---------------------------------------------------------------------------
# Task 5 — pooled per-(subject,feature) lambda calibration
# ---------------------------------------------------------------------------
def calibrate_feature_lambda(pooled_baseline_z, *, fpr_target_per_hour=1.0, hop_sec=0.1,
                             min_pooled_baseline_sec=600.0, bias=0.5):
    """Per-(subject,feature) lambda on POOLED baseline z-frames.

    Duration is TIME-based: pooled_sec = n_time_frames * hop. Below
    min_pooled_baseline_sec -> calibration_unstable (lambda NaN). Dead (all-NaN)
    channels are dropped as ROWS (frames kept); the underlying calibrator skips
    non-finite values internally.
    """
    z = np.asarray(pooled_baseline_z, dtype=np.float64)
    pooled_sec = z.shape[1] * float(hop_sec)
    if pooled_sec < float(min_pooled_baseline_sec):
        return {"lambda": float("nan"), "calibration_unstable": True,
                "pooled_baseline_sec": float(pooled_sec)}
    keep = ~np.all(np.isnan(z), axis=1)            # drop dead channels, KEEP all frames
    z = z[keep]
    n_ch = int(z.shape[0])
    if n_ch < 1 or z.shape[1] < 2:
        return {"lambda": float("nan"), "calibration_unstable": True,
                "pooled_baseline_sec": float(pooled_sec)}
    # calibrate_lambda_per_subject pools alarms across channels but normalizes the FP
    # budget by SINGLE-channel baseline-hours, i.e. its target is a SUBJECT-pooled FPR.
    # For a multi-channel recruitment instrument that allows ~0 alarms across the whole
    # array and saturates lambda_max. Scale the target by n_ch so the effective criterion
    # is a PER-CHANNEL FPR of fpr_target_per_hour (the spec's intended per-hour rate).
    lam = calibrate_lambda_per_subject(
        z, fpr_target_per_hour=float(fpr_target_per_hour) * n_ch, bias=float(bias),
        hop_sec=float(hop_sec),
    )
    return {"lambda": float(lam), "calibration_unstable": False,
            "pooled_baseline_sec": float(pooled_sec), "n_channels": n_ch}


# ---------------------------------------------------------------------------
# Task 6 — two-pass global onset (non-vacuous)
# ---------------------------------------------------------------------------
def resolve_global_onset(provisional_onsets, n_valid, *, frac=0.15):
    """t_global = earliest value where >= frac of n_valid contacts have a
    (non-ambiguous, non-NaN) provisional onset <= that value. UNIT-AGNOSTIC: the runner
    passes onset SECONDS, so the returned `t_global` is in seconds. No persist clause —
    transient false alarms are already voided by the ambiguous rule (detect_contact_onset)."""
    onsets = np.asarray(provisional_onsets, dtype=np.float64)
    finite = onsets[np.isfinite(onsets)]
    need = int(np.ceil(float(frac) * int(n_valid)))
    if finite.size < need or need < 1:
        return {"global_onset_resolved": False, "t_global": float("nan"),
                "n_recruited": int(finite.size), "n_needed": int(need)}
    t_global = float(np.sort(finite)[need - 1])      # the need-th earliest onset
    return {"global_onset_resolved": True, "t_global": t_global,
            "n_recruited": int(finite.size), "n_needed": int(need)}


# ---------------------------------------------------------------------------
# Task 7 — fusion + family-structured agreement
# ---------------------------------------------------------------------------
def fuse_recruitment_rank(per_feature_onset_frames):
    """Fused recruitment rank from the fused features (ER excluded by caller).

    per_feature_onset_frames: dict feature_key -> onset array (NaN = no onset; the runner
    passes onset SECONDS). fused_onset(c) = nanmedian over features; rank ascending
    (ties='average'). Contacts with <2 available features -> NaN (out of echo)."""
    keys = list(per_feature_onset_frames)
    M = np.vstack([np.asarray(per_feature_onset_frames[k], float) for k in keys])  # (n_feat,n_ch)
    avail = np.sum(np.isfinite(M), axis=0)
    with np.errstate(invalid="ignore"):
        fused_onset = np.where(avail >= 2, np.nanmedian(M, axis=0), np.nan)
    ranks = np.full(fused_onset.shape, np.nan)
    fin = np.isfinite(fused_onset)
    if fin.any():
        ranks[fin] = rankdata(fused_onset[fin], method="average")
    return ranks, fused_onset


def _rank_of(onset):
    o = np.asarray(onset, float)
    r = np.full(o.shape, np.nan)
    fin = np.isfinite(o)
    if fin.any():
        r[fin] = rankdata(o[fin], method="average")
    return r


def _pair_rho(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    common = np.isfinite(a) & np.isfinite(b)
    if common.sum() < 3:
        return np.nan
    return float(spearmanr(a[common], b[common]).statistic)


def _early_overlap(rank_dict, keys, early_k):
    sets = []
    for k in keys:
        r = rank_dict[k]
        fin = np.where(np.isfinite(r))[0]
        if fin.size < early_k:
            return np.nan
        order = fin[np.argsort(r[fin])][:early_k]
        sets.append(set(order.tolist()))
    jac = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            u = sets[i] | sets[j]
            jac.append(len(sets[i] & sets[j]) / len(u) if u else np.nan)
    return float(np.nanmedian(jac)) if jac else np.nan


def feature_agreement(per_feature_ranks, *, amplitude, spectral, early_k=3,
                      amp_agree_min=0.5, early_overlap_min=0.3):
    """Family-structured agreement. HARD gate uses ONLY the amplitude family (§6.2)."""
    rk = {k: np.asarray(v, float) for k, v in per_feature_ranks.items()}
    amp_pairs = [_pair_rho(rk[a], rk[b]) for i, a in enumerate(amplitude)
                 for b in amplitude[i + 1:]]
    amp_agree = float(np.nanmedian(amp_pairs)) if amp_pairs else np.nan
    amp_med_rank = _rank_of(np.nanmedian(np.vstack([rk[a] for a in amplitude]), axis=0))
    spectral_support = _pair_rho(rk[spectral], amp_med_rank) if spectral in rk else np.nan
    early_amp = _early_overlap(rk, list(amplitude), early_k)
    early_with_spec = (_early_overlap(rk, list(amplitude) + [spectral], early_k)
                       if spectral in rk else np.nan)
    flag = bool(np.isfinite(amp_agree) and amp_agree >= amp_agree_min
                and np.isfinite(early_amp) and early_amp >= early_overlap_min)
    return {"amplitude_family_agreement": amp_agree,
            "spectral_support": spectral_support,
            "spectral_conflict_flag": bool(np.isfinite(spectral_support)
                                           and spectral_support < 0),
            "early_K_overlap": early_amp,
            "early_K_overlap_with_spectral": early_with_spec,
            "feature_agreement_flag": flag}


# ---------------------------------------------------------------------------
# Task 8 — montage helpers (channel-identity hard contract)
# ---------------------------------------------------------------------------
def bipolar_alias_label(pair_label: str) -> str:
    """'HRA1-HRA2' -> 'HRA1' (alias-left), matching the legacy template convention."""
    return str(pair_label).split("-", 1)[0]


def assert_channel_identity(*, template_montage: str, ictal_montage: str) -> None:
    """§3.4: name equality is NOT identity. The ictal montage semantics must match the
    template's, or the same name refers to a different signal object."""
    if template_montage != ictal_montage:
        raise ValueError(
            f"montage/channel-identity mismatch: template={template_montage!r} "
            f"!= ictal={ictal_montage!r}. Same channel name != same signal object; "
            f"Main-A requires montage-semantics match (spec §3.4). Refusing to align."
        )
