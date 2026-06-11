# Topic 5 Stage 2 — Early-Ictal Recruitment-Time Instrument Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. When implementing any function whose spec section states multi-clause invariants (montage contract, pooled-λ, two-pass onset, fusion families), invoke `hfosp-deep-contract-verify` first.

**Goal:** Build a real early-ictal recruitment-time instrument — per-contact onset from 4 independent feature detectors on the raw seizure EEG → fused recruitment rank — and re-run Stage 1's echo statistic against the interictal template, replacing the Stage-1 ER proxy.

**Architecture:** One pure-math module (`src/topic5_ictal_recruitment.py`, no I/O) holding feature traces, robust-z, per-contact CUSUM onset, pooled-λ calibration, two-pass global onset, family-structured fusion, and montage-alias helpers; one runner (`scripts/run_topic5_ictal_recruitment.py`) that loads bipolar raw EEG via `extract_seizure_window`, masked templates via `results/interictal_propagation_masked/`, computes the instrument per seizure, and reuses `src/topic5_echo_gate.py` for echo + nulls; one plotter. **Staged gate:** Phase 1 (pure math + synthetic tests) → Phase 2 (montage trace + B0 audit + sentinel, MANUAL GATE) → Phase 3 (per-subject + cohort) → Phase 4 (figures + archive). **Cohort/per-seizure inference does NOT run until the sentinel gate passes.**

**Tech Stack:** Python, numpy, scipy.signal (spectrogram), scipy.stats (rankdata, spearmanr). Reuses `src.ictal_onset_extraction` (raw EEG window, compute_er, baseline windows, CUSUM), `src.ictal_er_rank.calibrate_lambda_per_subject`, `src.topic5_echo_gate` (echo + nulls + pooling), `src.propagation_skeleton_geometry.parse_shaft`, `src.seeg_coord_loader`. pytest for TDD. Matplotlib for figures.

**Spec:** `docs/superpowers/specs/2026-06-10-topic5-ictal-recruitment-stage2-design.md` (v2, committed). Re-read each referenced spec section at the matching task boundary (CLAUDE.md §5).

> **Note on test counts:** `Expected: PASS (N tests)` numbers are cumulative and indicative; the real gate is `pytest tests/test_topic5_ictal_recruitment.py -v` fully green, not the exact integer.

---

## File Structure

- **Create `src/topic5_ictal_recruitment.py`** — pure functions, no file I/O. Feature traces (`line_length_trace`, `band_power_trace`, `spectral_edge_trace`), `baseline_robust_z`, `detect_contact_onset`, `calibrate_feature_lambda`, `resolve_global_onset`, `fuse_recruitment_rank`, `feature_agreement`, montage helpers (`bipolar_alias_label`, `assert_channel_identity`).
- **Create `scripts/run_topic5_ictal_recruitment.py`** — I/O + orchestration. `trace-montage` / `audit` / `sentinel` / `per-subject` / `cohort` subcommands. Bipolar raw EEG via `extract_seizure_window(reference="bipolar")`; masked templates from `results/interictal_propagation_masked/`; echo via `src.topic5_echo_gate`.
- **Create `scripts/plot_topic5_ictal_recruitment.py`** — sentinel overlays + 4 cohort figures + `figures/README.md`.
- **Create `tests/test_topic5_ictal_recruitment.py`** — unit tests on synthetic signals with known answers.

**Locked constants (spec):** `HOP=0.1s`, `MIN_CH=8`, `RECRUIT_POST_SEC=15`, `GLOBAL_ONSET_FRAC=0.15`, `FPR_TARGET_PER_HOUR=1.0`, `MIN_POOLED_BASELINE_SEC=600`, `MIN_BASELINE_SEC=60`, `BASELINE_PRE_SEC=300`, `PRE_ONSET_CHANGE_SEC=-10`, `EARLY_K=3`, `ties='average'`, `B=2000`, `RNG_SEED=20260610`. Feature WINs: LL=1.0, broadband=1.0, HFA=0.5, spectral_edge=1.0, ER=1.0. Bands: broadband=(1,45), HFA=(80,150), ER fast=(60,100)/slow=(4,20), SEF=(1,min(127,nyq)) edge=0.9.

---

# PHASE 0 — Pre-flight (before any code, reviewer §5/§7)

- [ ] **Clean the worktree.** This work is on branch `topic5-ictal-recruitment-stage2`, but the working tree still carries uncommitted Topic 4 changes + untracked Topic 4 scripts. Before Phase 1, isolate Topic 5: either (a) commit/stash the Topic 4 changes, or (b) start a clean worktree from a Topic-4-clean commit. Do NOT let Topic 4 edits land in Topic 5 commits. Verify with `git status` showing only Topic 5 files staged per task.
- [ ] **Skill availability.** The header names Claude Code skills (`superpowers:subagent-driven-development`, `hfosp-deep-contract-verify`). In Claude Code they are available. In environments without them (e.g. Codex), treat each as a **manual pre-implementation contract check**: before writing a function with multi-clause invariants (montage §3.4, pooled-λ §5.3, two-pass onset §5.2, fusion families §6), write the invariants out and verify the implementation against each — same discipline, no skill required.

---

# PHASE 1 — Pure-math instrument + synthetic TDD

> Phase 1 has NO file I/O and NO real data. Every test runs on synthetic signals with a known injected recruitment order. This is the only phase that can be fully verified before touching the cohort.

## Task 1: Module scaffold + `line_length_trace`

Re-read spec §5.0 + §5.1 (F1 line-length, time-domain, common hop grid) before this task.

**Files:**
- Create: `src/topic5_ictal_recruitment.py`
- Test: `tests/test_topic5_ictal_recruitment.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pytest
from src.topic5_ictal_recruitment import line_length_trace


def test_line_length_trace_shape_and_hop():
    fs = 200.0
    sig = np.zeros((3, int(round(10 * fs))))  # 3 ch, 10 s
    tr, t = line_length_trace(sig, fs, win_sec=1.0, hop_sec=0.1)
    assert tr.shape[0] == 3
    # frames span [0, 9s] at 0.1 hop -> ~91 frames (last full 1s window starts at 9.0s)
    assert tr.shape[1] == t.shape[0]
    assert tr.shape[1] == pytest.approx(91, abs=1)
    assert np.allclose(t[:3], [0.0, 0.1, 0.2])


def test_line_length_trace_rises_with_fast_activity():
    fs = 200.0
    n = int(round(10 * fs))
    rng = np.random.default_rng(0)
    sig = 0.01 * rng.standard_normal((1, n))           # quiet baseline
    # inject fast oscillation in the second half -> line length jumps
    tt = np.arange(n) / fs
    sig[0, n // 2:] += np.sin(2 * np.pi * 40 * tt[n // 2:])
    tr, t = line_length_trace(sig, fs, win_sec=1.0, hop_sec=0.1)
    early = tr[0, t < 4.0].mean()
    late = tr[0, t > 6.0].mean()
    assert late > 5 * early
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic5_ictal_recruitment.py -v`
Expected: FAIL with `ImportError: cannot import name 'line_length_trace'`.

- [ ] **Step 3: Write minimal implementation**

```python
"""Topic 5 Stage-2 early-ictal recruitment-time instrument (PURE math, no I/O).

Spec: docs/superpowers/specs/2026-06-10-topic5-ictal-recruitment-stage2-design.md

All feature traces land on a COMMON hop grid (HOP=0.1s) so the 4 fused
detectors (line-length / broadband / HFA / spectral-edge) and the held-out ER
reference are directly comparable. Frame j covers [j*hop, j*hop + win].
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import spectrogram
from scipy.stats import rankdata


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
    # cumulative sum trick for fast sliding-window sums
    csum = np.concatenate([np.zeros((sig.shape[0], 1)), np.cumsum(absdiff, axis=1)], axis=1)
    out = np.empty((sig.shape[0], starts.size), dtype=np.float64)
    for j, s in enumerate(starts):
        e = min(s + win - 1, absdiff.shape[1])               # diff has n_samp-1 cols
        out[:, j] = csum[:, e] - csum[:, s]
    t = starts / fs
    return out, t
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_topic5_ictal_recruitment.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/topic5_ictal_recruitment.py tests/test_topic5_ictal_recruitment.py
git commit -m "feat(topic5-recruit): module scaffold + line_length_trace"
```

---

## Task 2: `band_power_trace` (broadband + HFA) and `spectral_edge_trace`

Re-read spec §5.1 (F2/F3 single-band log power; F5 90% spectral-edge frequency; all on the same hop grid via spectrogram). Note the Nyquist gate: HFA hi=150 needs fs>300.

**Files:**
- Modify: `src/topic5_ictal_recruitment.py`
- Test: `tests/test_topic5_ictal_recruitment.py`

- [ ] **Step 1: Write the failing test**

```python
from src.topic5_ictal_recruitment import band_power_trace, spectral_edge_trace


def test_band_power_trace_hfa_requires_nyquist():
    fs = 200.0  # Nyquist 100 < 150 -> HFA band invalid
    sig = np.random.default_rng(0).standard_normal((2, int(round(5 * fs))))
    with pytest.raises(ValueError):
        band_power_trace(sig, fs, band=(80.0, 150.0), win_sec=0.5, hop_sec=0.1)


def test_band_power_trace_tracks_injected_band():
    fs = 500.0
    n = int(round(8 * fs))
    tt = np.arange(n) / fs
    sig = np.zeros((1, n))
    sig[0, n // 2:] += np.sin(2 * np.pi * 100 * tt[n // 2:])  # 100 Hz burst late
    tr, t = band_power_trace(sig, fs, band=(80.0, 150.0), win_sec=0.5, hop_sec=0.1)
    assert tr[0, t > 5].mean() > tr[0, t < 3].mean() + 2.0   # log power up


def test_spectral_edge_trace_rises_to_fast():
    fs = 500.0
    n = int(round(8 * fs))
    tt = np.arange(n) / fs
    rng = np.random.default_rng(1)
    sig = np.zeros((1, n))
    sig[0, :n // 2] = np.sin(2 * np.pi * 5 * tt[:n // 2])     # slow first half
    sig[0, n // 2:] = np.sin(2 * np.pi * 90 * tt[n // 2:])    # fast second half
    sig += 0.01 * rng.standard_normal((1, n))
    sef, t = spectral_edge_trace(sig, fs, edge=0.9, win_sec=1.0, hop_sec=0.1)
    assert sef[0, t > 6].mean() > sef[0, t < 2].mean() + 30.0  # SEF jumps up (Hz)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic5_ictal_recruitment.py -k "band_power or spectral_edge" -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation**

```python
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
    # first freq index whose cumulative fraction >= edge
    reached = frac >= float(edge)
    idx = np.argmax(reached, axis=1)                         # (n_ch, n_t)
    sef = fb[idx]
    return sef.astype(np.float64), t
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_topic5_ictal_recruitment.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/topic5_ictal_recruitment.py tests/test_topic5_ictal_recruitment.py
git commit -m "feat(topic5-recruit): band_power + spectral_edge traces (Nyquist-gated)"
```

---

## Task 3: `baseline_robust_z` (MAD normalization)

Re-read spec §5.3 (robust z = (x − median) / (1.4826·MAD) over a baseline frame window; MAD=0 → NaN; reuse the baseline-frame mask + min-valid logic of `baseline_zscore_er`).

**Files:**
- Modify: `src/topic5_ictal_recruitment.py`
- Test: `tests/test_topic5_ictal_recruitment.py`

- [ ] **Step 1: Write the failing test**

```python
from src.topic5_ictal_recruitment import baseline_robust_z


def test_baseline_robust_z_centers_and_scales():
    # baseline frames 0..49 ~ N(0,1); a spike later should be many robust-z up
    rng = np.random.default_rng(2)
    tr = rng.standard_normal((1, 200))
    tr[0, 150] += 50.0
    z = baseline_robust_z(tr, (0, 50), hop_sec=0.1, min_baseline_valid_sec=2.0)
    assert abs(np.median(z[0, :50])) < 0.5
    assert z[0, 150] > 10.0


def test_baseline_robust_z_zero_mad_returns_nan():
    tr = np.ones((1, 100))                       # constant baseline -> MAD 0
    z = baseline_robust_z(tr, (0, 50), hop_sec=0.1, min_baseline_valid_sec=2.0)
    assert np.all(np.isnan(z[0]))


def test_baseline_robust_z_insufficient_baseline_returns_nan():
    tr = np.random.default_rng(3).standard_normal((1, 100))
    # only 5 frames * 0.1s = 0.5s < 2s required
    z = baseline_robust_z(tr, (0, 5), hop_sec=0.1, min_baseline_valid_sec=2.0)
    assert np.all(np.isnan(z[0]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic5_ictal_recruitment.py -k baseline_robust_z -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_topic5_ictal_recruitment.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add src/topic5_ictal_recruitment.py tests/test_topic5_ictal_recruitment.py
git commit -m "feat(topic5-recruit): baseline_robust_z (MAD)"
```

---

## Task 4: `detect_contact_onset` (CUSUM wrapper + no-onset / ambiguous rules)

Re-read spec §5.0 + §5.4. The change-point inner core is `src.ictal_er_rank.compute_cusum_n_d_with_time` — a GENERIC wrapper, not ER-specific. Ambiguous = CUSUM fires but the **feature z** does not HOLD ≥ `z_sustain` (a robust-z level) over `sustain_sec` → transient false alarm → onset void (NaN). No-onset = CUSUM never crosses λ → NaN. (The ambiguity test is on the feature z, never on λ — λ is a CUSUM accumulation threshold, a different quantity.)

**Files:**
- Modify: `src/topic5_ictal_recruitment.py`
- Test: `tests/test_topic5_ictal_recruitment.py`

- [ ] **Step 1: Write the failing test**

```python
from src.topic5_ictal_recruitment import detect_contact_onset


def test_detect_contact_onset_sustained_step_fires():
    z = np.zeros(200)
    z[80:] = 6.0                                  # sustained step at frame 80
    res = detect_contact_onset(z, lam=5.0, detection_idx_window=(0, 200),
                               hop_sec=0.1, win_sec=1.0, pre_sec=0.0)
    assert res["detected"] is True
    assert 78 <= res["onset_frame"] <= 90
    # onset_sec = frame*hop + win/2 - pre (center convention, rel clinical onset)
    assert res["onset_sec"] == pytest.approx(res["onset_frame"] * 0.1 + 0.5, abs=0.2)


def test_detect_contact_onset_transient_is_ambiguous():
    z = np.zeros(200)
    z[80:83] = 6.0                                # 0.3s spike then back to baseline
    res = detect_contact_onset(z, lam=5.0, detection_idx_window=(0, 200),
                               hop_sec=0.1, win_sec=1.0, pre_sec=0.0)
    assert res["detected"] is False
    assert res["reason"] == "ambiguous"            # feature-z not sustained post-onset


def test_detect_contact_onset_never_crosses():
    z = 0.5 * np.ones(200)
    res = detect_contact_onset(z, lam=5.0, detection_idx_window=(0, 200),
                               hop_sec=0.1, win_sec=1.0, pre_sec=0.0)
    assert res["detected"] is False
    assert res["reason"] == "unreached"
    assert np.isnan(res["onset_frame"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic5_ictal_recruitment.py -k detect_contact_onset -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation**

```python
from src.ictal_er_rank import compute_cusum_n_d_with_time

Z_SUSTAIN = 2.0   # robust-z floor the feature must HOLD post-onset (a feature-z, NOT lambda)


def detect_contact_onset(z_trace_1d, *, lam, detection_idx_window, hop_sec=0.1,
                         win_sec=1.0, pre_sec=0.0, bias=0.5, sustain_sec=1.0,
                         z_sustain=Z_SUSTAIN):
    """Per-contact onset on a 1-D robust-z trace via clamped Page-Hinkley CUSUM.

    REAL wrapper signature (src/ictal_er_rank.py:179):
      compute_cusum_n_d_with_time(z, lambda_thresh, *, bias, detection_idx_window,
                                  hop_sec, win_sec, pre_sec) -> CusumOnsetResult(frame_idx, t_onset_sec)
    where t_onset_sec = frame_idx*hop + win/2 - pre (CENTER convention, rel clinical
    onset). Because every feature uses this same formula, onset_sec is directly
    comparable ACROSS features despite different win — this is what makes the fused
    median well-defined (Task 11 fuses on onset_sec, NOT raw frame index).

    Ambiguous (§5.4) is judged on the FEATURE z (dimensionally correct), NOT on lambda:
    lambda is a CUSUM ACCUMULATION threshold, not a feature-z level. A transient false
    alarm = the post-onset feature z does not HOLD >= z_sustain over sustain_sec -> void.
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
```

> **Signature aligned (reviewer P0-2):** this matches the real `compute_cusum_n_d_with_time` (positional `lambda_thresh`; kwargs `detection_idx_window`, `hop_sec`, `win_sec`, `pre_sec`; returns `CusumOnsetResult(frame_idx, t_onset_sec)`). No "adapt at execution" hand-wave — the core detector signature is locked here. The ambiguous rule compares the **feature z** to `z_sustain` (a robust-z level), never to `lambda` (a CUSUM accumulation threshold) — the v1 `z < λ/2` was a dimensional error (reviewer P1).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_topic5_ictal_recruitment.py -v`
Expected: PASS (11 tests).

- [ ] **Step 5: Commit**

```bash
git add src/topic5_ictal_recruitment.py tests/test_topic5_ictal_recruitment.py
git commit -m "feat(topic5-recruit): detect_contact_onset (CUSUM + no-onset/ambiguous)"
```

---

## Task 5: `calibrate_feature_lambda` (pooled baseline, per-hour FPR, calibration_unstable)

Re-read spec §5.3 (λ unit = `fpr_target_per_hour`; pool baseline z-frames across a subject's seizures; pooled < `MIN_POOLED_BASELINE_SEC` → `calibration_unstable=True`).

**Files:**
- Modify: `src/topic5_ictal_recruitment.py`
- Test: `tests/test_topic5_ictal_recruitment.py`

- [ ] **Step 1: Write the failing test**

```python
from src.topic5_ictal_recruitment import calibrate_feature_lambda


def test_calibrate_feature_lambda_pooled_ok():
    # 7000 baseline frames * 0.1s = 700s pooled > 600 -> stable (duration = n_FRAMES*hop)
    rng = np.random.default_rng(4)
    pooled = rng.standard_normal((10, 7000))     # (n_ch, n_frames) baseline z
    out = calibrate_feature_lambda(pooled, fpr_target_per_hour=1.0, hop_sec=0.1,
                                   min_pooled_baseline_sec=600.0)
    assert out["calibration_unstable"] is False
    assert out["pooled_baseline_sec"] == pytest.approx(700.0)
    assert 1.0 <= out["lambda"] <= 100.0


def test_calibrate_feature_lambda_nan_frames_do_not_shrink_duration():
    # a few NaN channels must NOT delete whole time frames (duration is time-based)
    rng = np.random.default_rng(9)
    pooled = rng.standard_normal((10, 7000))
    pooled[3, :] = np.nan                          # one dead channel
    out = calibrate_feature_lambda(pooled, fpr_target_per_hour=1.0, hop_sec=0.1,
                                   min_pooled_baseline_sec=600.0)
    assert out["pooled_baseline_sec"] == pytest.approx(700.0)   # NOT shrunk to ~0
    assert out["calibration_unstable"] is False


def test_calibrate_feature_lambda_too_short_is_unstable():
    rng = np.random.default_rng(5)
    pooled = rng.standard_normal((10, 100))      # 10s pooled << 600
    out = calibrate_feature_lambda(pooled, fpr_target_per_hour=1.0, hop_sec=0.1,
                                   min_pooled_baseline_sec=600.0)
    assert out["calibration_unstable"] is True
    assert np.isnan(out["lambda"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic5_ictal_recruitment.py -k calibrate_feature_lambda -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation**

```python
from src.ictal_er_rank import calibrate_lambda_per_subject


def calibrate_feature_lambda(pooled_baseline_z, *, fpr_target_per_hour=1.0, hop_sec=0.1,
                             min_pooled_baseline_sec=600.0, bias=0.5):
    """Per-(subject,feature) lambda on POOLED baseline z-frames.

    pooled_baseline_z: (n_ch, n_frames_pooled) robust-z baseline frames concatenated
    across the subject's eligible seizures. Total duration = n_frames * hop. Below
    min_pooled_baseline_sec -> calibration_unstable (lambda NaN), subject->sensitivity.
    """
    z = np.asarray(pooled_baseline_z, dtype=np.float64)
    # Duration is TIME-based: pooled_sec = n_time_frames * hop. Do NOT drop frames just
    # because some channel is NaN (reviewer P1-1) — that would collapse the duration.
    # Drop only fully-NaN CHANNELS (rows); calibrate_lambda_per_subject already skips
    # non-finite values internally (src/ictal_er_rank.py count_alarms `if not np.isfinite`).
    pooled_sec = z.shape[1] * float(hop_sec)
    if pooled_sec < float(min_pooled_baseline_sec):
        return {"lambda": float("nan"), "calibration_unstable": True,
                "pooled_baseline_sec": float(pooled_sec)}
    keep = ~np.all(np.isnan(z), axis=1)            # drop dead channels, KEEP all frames
    z = z[keep]
    if z.shape[0] < 1 or z.shape[1] < 2:
        return {"lambda": float("nan"), "calibration_unstable": True,
                "pooled_baseline_sec": float(pooled_sec)}
    lam = calibrate_lambda_per_subject(
        z, fpr_target_per_hour=float(fpr_target_per_hour), bias=float(bias),
        hop_sec=float(hop_sec),
    )
    return {"lambda": float(lam), "calibration_unstable": False,
            "pooled_baseline_sec": float(pooled_sec)}
```

> **Signature confirmed:** `calibrate_lambda_per_subject(z_er_baseline (n_ch,n_frames), *, fpr_target_per_hour, bias, hop_sec, ...)` takes a 2-D array and pools multi-channel baseline alarms internally, skipping non-finite values (src/ictal_er_rank.py:208). No flattening needed.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_topic5_ictal_recruitment.py -v`
Expected: PASS (13 tests).

- [ ] **Step 5: Commit**

```bash
git add src/topic5_ictal_recruitment.py tests/test_topic5_ictal_recruitment.py
git commit -m "feat(topic5-recruit): calibrate_feature_lambda (pooled, per-hour, unstable flag)"
```

---

## Task 6: `resolve_global_onset` (two-pass — non-vacuous fraction threshold)

Re-read spec §4.2 + §5.2. `t_global` = first frame where the fraction of contacts with a *non-ambiguous* provisional onset ≤ t reaches `GLOBAL_ONSET_FRAC`. The removed v1 persist clause is NOT reintroduced — sustained-ness is already enforced by the ambiguous rule (Task 4).

**Files:**
- Modify: `src/topic5_ictal_recruitment.py`
- Test: `tests/test_topic5_ictal_recruitment.py`

- [ ] **Step 1: Write the failing test**

```python
from src.topic5_ictal_recruitment import resolve_global_onset


def test_resolve_global_onset_reaches_fraction():
    # 20 contacts; onsets clustered at value 100; 1 early straggler at 10.
    # NOTE: resolve_global_onset is UNIT-AGNOSTIC — Task 11 feeds it onset SECONDS,
    # so the returned `t_global` carries whatever unit the input had (here arbitrary).
    onsets = np.full(20, 100.0)
    onsets[0] = 10.0
    res = resolve_global_onset(onsets, n_valid=20, frac=0.15)
    # need >=3 of 20 (15%); only 1 by 10 -> not global; 3rd-earliest is ~100
    assert res["global_onset_resolved"] is True
    assert res["t_global"] == pytest.approx(100, abs=1)


def test_resolve_global_onset_single_transient_not_global():
    onsets = np.full(20, np.nan)                  # nobody recruits except one
    onsets[0] = 10.0
    res = resolve_global_onset(onsets, n_valid=20, frac=0.15)
    assert res["global_onset_resolved"] is False
    assert np.isnan(res["t_global"])


def test_resolve_global_onset_unresolved_when_too_few():
    onsets = np.full(20, np.nan)
    onsets[:2] = [10.0, 11.0]                      # 2/20 = 10% < 15%
    res = resolve_global_onset(onsets, n_valid=20, frac=0.15)
    assert res["global_onset_resolved"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic5_ictal_recruitment.py -k resolve_global_onset -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation**

```python
def resolve_global_onset(provisional_onsets, n_valid, *, frac=0.15):
    """t_global = earliest value where >= frac of n_valid contacts have a
    (non-ambiguous, non-NaN) provisional onset <= that value. UNIT-AGNOSTIC: Task 11
    passes onset SECONDS, so the returned `t_global` is in seconds. No persist clause —
    transient false alarms are already voided by the ambiguous rule (Task 4)."""
    onsets = np.asarray(provisional_onsets, dtype=np.float64)
    finite = onsets[np.isfinite(onsets)]
    need = int(np.ceil(float(frac) * int(n_valid)))
    if finite.size < need or need < 1:
        return {"global_onset_resolved": False, "t_global": float("nan"),
                "n_recruited": int(finite.size), "n_needed": int(need)}
    t_global = float(np.sort(finite)[need - 1])      # the need-th earliest onset
    return {"global_onset_resolved": True, "t_global": t_global,
            "n_recruited": int(finite.size), "n_needed": int(need)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_topic5_ictal_recruitment.py -v`
Expected: PASS (16 tests).

- [ ] **Step 5: Commit**

```bash
git add src/topic5_ictal_recruitment.py tests/test_topic5_ictal_recruitment.py
git commit -m "feat(topic5-recruit): resolve_global_onset (two-pass, non-vacuous)"
```

---

## Task 7: `fuse_recruitment_rank` + `feature_agreement` (family-structured)

Re-read spec §6.1–6.3. Fused = median over the 4 fused features' pass-2 onsets (ER NOT in fuse). The hard `feature_agreement_flag` uses ONLY the amplitude family F1/F2/F3 (`amplitude_family_agreement` + `early_K_overlap`); spectral-edge F5 contributes `spectral_support` and a diagnostic `early_K_overlap_with_spectral` but never enters the gate.

**Files:**
- Modify: `src/topic5_ictal_recruitment.py`
- Test: `tests/test_topic5_ictal_recruitment.py`

- [ ] **Step 1: Write the failing test**

```python
from src.topic5_ictal_recruitment import fuse_recruitment_rank, feature_agreement

# feature key order convention used throughout: F1 line_length, F2 broadband,
# F3 hfa, F5 spectral_edge. AMP = ("line_length","broadband","hfa"); SPECTRAL="spectral_edge".

def test_fuse_recruitment_rank_median_ignores_nan_and_excludes_er():
    # 4 contacts; per-feature onset frames; one feature NaN for contact 2
    per_feat = {
        "line_length": np.array([10.0, 20.0, 30.0, 40.0]),
        "broadband":   np.array([11.0, 19.0, np.nan, 41.0]),
        "hfa":         np.array([9.0, 21.0, 31.0, 39.0]),
        "spectral_edge": np.array([10.0, 20.0, 30.0, 40.0]),
    }
    fused_rank, fused_onset = fuse_recruitment_rank(per_feat)
    assert np.argsort(fused_rank).tolist() == [0, 1, 2, 3]   # ascending recruitment
    assert np.isfinite(fused_onset[2])                       # median over available


def test_feature_agreement_flag_amplitude_only():
    # amplitude family agree strongly; spectral-edge CONFLICTS (reversed) ->
    # flag must still be True (spectral out of the gate), spectral_support < 0.
    n = 10
    base = np.arange(n, dtype=float)
    per_rank = {
        "line_length": base.copy(),
        "broadband": base.copy(),
        "hfa": base.copy(),
        "spectral_edge": base[::-1].copy(),
    }
    ag = feature_agreement(per_rank, amplitude=("line_length", "broadband", "hfa"),
                           spectral="spectral_edge", early_k=3)
    assert ag["feature_agreement_flag"] is True
    assert ag["spectral_support"] < 0
    assert ag["spectral_conflict_flag"] is True


def test_feature_agreement_flag_false_when_amplitude_disagrees():
    n = 10
    rng = np.random.default_rng(6)
    per_rank = {
        "line_length": np.arange(n, dtype=float),
        "broadband": rng.permutation(n).astype(float),
        "hfa": rng.permutation(n).astype(float),
        "spectral_edge": np.arange(n, dtype=float),
    }
    ag = feature_agreement(per_rank, amplitude=("line_length", "broadband", "hfa"),
                           spectral="spectral_edge", early_k=3)
    assert ag["feature_agreement_flag"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic5_ictal_recruitment.py -k "fuse_recruitment or feature_agreement" -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation**

```python
from scipy.stats import spearmanr


def fuse_recruitment_rank(per_feature_onset_frames):
    """Fused recruitment rank from the 4 fused features (ER excluded by caller).

    per_feature_onset_frames: dict feature_key -> onset_frame array (NaN = no onset).
    fused_onset(c) = nanmedian over features; rank by ascending onset (ties='average').
    Contacts with <2 available features -> NaN (out of echo)."""
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
    """Family-structured agreement. HARD gate uses ONLY amplitude family (§6.2)."""
    rk = {k: np.asarray(v, float) for k, v in per_feature_ranks.items()}
    amp_pairs = [_pair_rho(rk[a], rk[b]) for i, a in enumerate(amplitude)
                 for b in amplitude[i + 1:]]
    amp_agree = float(np.nanmedian(amp_pairs)) if amp_pairs else np.nan
    amp_med_rank = _rank_of(np.nanmedian(
        np.vstack([rk[a] for a in amplitude]), axis=0))
    spectral_support = _pair_rho(rk[spectral], amp_med_rank)
    early_amp = _early_overlap(rk, list(amplitude), early_k)
    early_with_spec = _early_overlap(rk, list(amplitude) + [spectral], early_k)
    flag = bool(np.isfinite(amp_agree) and amp_agree >= amp_agree_min
                and np.isfinite(early_amp) and early_amp >= early_overlap_min)
    return {"amplitude_family_agreement": amp_agree,
            "spectral_support": spectral_support,
            "spectral_conflict_flag": bool(np.isfinite(spectral_support)
                                           and spectral_support < 0),
            "early_K_overlap": early_amp,
            "early_K_overlap_with_spectral": early_with_spec,
            "feature_agreement_flag": flag}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_topic5_ictal_recruitment.py -v`
Expected: PASS (19 tests).

- [ ] **Step 5: Commit**

```bash
git add src/topic5_ictal_recruitment.py tests/test_topic5_ictal_recruitment.py
git commit -m "feat(topic5-recruit): fuse_recruitment_rank + family-structured feature_agreement"
```

---

## Task 8: montage helpers — `bipolar_alias_label` + `assert_channel_identity`

Re-read spec §3.4 (P0). Template channel names are single-contact labels but detection is bipolar-aliased-left (`A1-A2 → A1`). Ictal bipolar pair labels from `_build_bipolar_pairs` look like `"HRA1-HRA2"`; alias them to the left contact `"HRA1"` to match the template. Name equality alone must NOT pass identity — montage semantics must match.

**Files:**
- Modify: `src/topic5_ictal_recruitment.py`
- Test: `tests/test_topic5_ictal_recruitment.py`

- [ ] **Step 1: Write the failing test**

```python
from src.topic5_ictal_recruitment import bipolar_alias_label, assert_channel_identity


def test_bipolar_alias_label_left_contact():
    assert bipolar_alias_label("HRA1-HRA2") == "HRA1"
    assert bipolar_alias_label("BFRA5-BFRA6") == "BFRA5"


def test_assert_channel_identity_ok_when_montage_matches():
    # both bipolar-aliased -> contract holds
    assert_channel_identity(template_montage="bipolar_aliased_left",
                            ictal_montage="bipolar_aliased_left")  # no raise


def test_assert_channel_identity_hard_fail_on_montage_mismatch():
    # same names but one CAR-monopolar, one bipolar-aliased -> different signal object
    with pytest.raises(ValueError, match="montage"):
        assert_channel_identity(template_montage="bipolar_aliased_left",
                                ictal_montage="car_monopolar")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic5_ictal_recruitment.py -k "alias_label or channel_identity" -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation**

```python
def bipolar_alias_label(pair_label: str) -> str:
    """'HRA1-HRA2' -> 'HRA1' (alias-left), matching the legacy template convention."""
    return str(pair_label).split("-", 1)[0]


def assert_channel_identity(*, template_montage: str, ictal_montage: str) -> None:
    """P0 §3.4: name equality is NOT identity. The ictal montage semantics must
    match the template's, or the same name refers to a different signal object."""
    if template_montage != ictal_montage:
        raise ValueError(
            f"montage/channel-identity mismatch: template={template_montage!r} "
            f"!= ictal={ictal_montage!r}. Same channel name != same signal object; "
            f"Main-A requires montage-semantics match (spec §3.4). Refusing to align."
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_topic5_ictal_recruitment.py -v`
Expected: PASS (22 tests).

- [ ] **Step 5: Commit**

```bash
git add src/topic5_ictal_recruitment.py tests/test_topic5_ictal_recruitment.py
git commit -m "feat(topic5-recruit): montage alias + channel-identity hard contract"
```

---

## Task 9: echo integration smoke (reuse `src.topic5_echo_gate` on a recruitment rank)

Re-read spec §7.1 (reuse Stage 1's `compute_echo_strength` / `pool_echo_subject_level` — do NOT reinvent the statistic). This task is a smoke test proving the recruitment rank feeds the Stage-1 echo core unchanged.

**Files:**
- Test: `tests/test_topic5_ictal_recruitment.py`

- [ ] **Step 1: Write the failing test**

```python
def test_recruitment_rank_feeds_stage1_echo_core():
    # a recruitment rank that matches a template -> high e_k via the SAME Stage-1 core
    from src.topic5_echo_gate import compute_echo_strength
    template = np.arange(12, dtype=float)
    recruitment_rank = np.arange(12, dtype=float)        # perfect echo
    res = compute_echo_strength(recruitment_rank, [template], B=1000,
                                rng=np.random.default_rng(7), min_ch=8)
    assert res["r_obs"] == pytest.approx(1.0)
    assert res["e_k"] > 3.0
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `pytest tests/test_topic5_ictal_recruitment.py -k feeds_stage1 -v`
Expected: PASS immediately (no new code — this asserts the reuse contract). If `compute_echo_strength` import fails, STOP: Stage 1's `src/topic5_echo_gate.py` must be present (it is, 36 tests).

- [ ] **Step 3: Commit**

```bash
git add tests/test_topic5_ictal_recruitment.py
git commit -m "test(topic5-recruit): echo-core reuse smoke (recruitment rank -> Stage-1 e_k)"
```

---

### PHASE 1 GATE

Run the whole pure-math suite: `pytest tests/test_topic5_ictal_recruitment.py -v`. **All green required.** Phase 1 is complete and self-contained (no real data touched). Do NOT proceed to Phase 2 until green.

---

# PHASE 2 — Runner: montage trace + B0 audit + sentinel (MANUAL GATE)

> Phase 2 touches real data but produces NO cohort inference. It ends at a manual sentinel gate (spec §13 step 4–5). Cohort/per-seizure stats (Phase 3) MUST NOT run until the user signs off on the sentinel.

## Task 10: Runner scaffold + template montage trace (`trace-montage`)

Re-read spec §3.4 (montage trace is the sentinel pre-step). Determine whether the masked interictal template channels are monopolar or bipolar-aliased-left by tracing the producer / provenance, and record `template_montage`.

**Files:**
- Create: `scripts/run_topic5_ictal_recruitment.py`

- [ ] **Step 1: Inspect the masked template + its producer provenance live**

Run:
```bash
python -c "import json,glob; f=sorted(glob.glob('results/interictal_propagation_masked/**/*.json',recursive=True))[0]; d=json.load(open(f)); print(f); print('keys', list(d.keys())); print('channel_names', d.get('channel_names')[:8]); print('src_diag', d.get('source_diagnostic'))"
grep -rnE "reference|bipolar|car|alias|monopolar" src/interictal_propagation.py | head
```
Expected: confirm whether `channel_names` are single-contact (bipolar-aliased-left, working hypothesis) or true monopolar. **Record the verdict** — it sets `TEMPLATE_MONTAGE` and the ictal `reference`. If genuinely ambiguous, STOP and ask the user before proceeding (do not guess).

- [ ] **Step 2: Write the runner scaffold + `cmd_trace_montage`**

```python
"""Topic 5 Stage-2 recruitment-time instrument runner (I/O + orchestration).

Spec: docs/superpowers/specs/2026-06-10-topic5-ictal-recruitment-stage2-design.md
Staged gate: trace-montage -> audit -> sentinel (MANUAL) -> per-subject -> cohort.

P0 invariants:
- §3.4 montage: ictal features computed on bipolar (reference='bipolar') aliased
  to the template's single-contact convention; assert_channel_identity hard-fails
  on a montage mismatch; CAR is sensitivity only.
- §7.1 echo reuses src.topic5_echo_gate (no reinvented statistic).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src import topic5_ictal_recruitment as recruit
from src import topic5_echo_gate as echo
from src.ictal_onset_extraction import (extract_seizure_window, compute_er,
                                        resolve_baseline_window, GAMMA_ER_BANDS)
from src.propagation_skeleton_geometry import parse_shaft

MASKED_ROOT = Path("results/interictal_propagation_masked")
OUT_ROOT = Path("results/topic5_ictal_recruitment")
HOP = 0.1
MIN_CH = 8
RECRUIT_POST_SEC = 15.0
GLOBAL_ONSET_FRAC = 0.15
FPR_TARGET_PER_HOUR = 1.0
MIN_POOLED_BASELINE_SEC = 600.0
MIN_BASELINE_SEC = 60.0
BASELINE_PRE_SEC = 300.0
PRE_ONSET_CHANGE_SEC = -10.0
EARLY_K = 3
B = 2000
RNG_SEED = 20260610
# set from Task 10 Step 1 trace; default = working hypothesis
TEMPLATE_MONTAGE = "bipolar_aliased_left"
ICTAL_REFERENCE = "bipolar"

FUSED_FEATURES = ("line_length", "broadband", "hfa", "spectral_edge")
AMP_FEATURES = ("line_length", "broadband", "hfa")
SPECTRAL_FEATURE = "spectral_edge"


def cmd_trace_montage(args):
    """Record the traced template montage for the cohort (human-confirmed in Step 1)."""
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rec = {"template_montage": TEMPLATE_MONTAGE, "ictal_reference": ICTAL_REFERENCE,
           "note": "set after tracing the masked-template producer (Task 10 Step 1)"}
    json.dump(rec, open(OUT_ROOT / "montage_trace.json", "w"), indent=2)
    print("montage trace:", rec)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("trace-montage").set_defaults(func=cmd_trace_montage)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run it**

Run: `python scripts/run_topic5_ictal_recruitment.py trace-montage`
Expected: writes `results/topic5_ictal_recruitment/montage_trace.json` with the traced montage.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_topic5_ictal_recruitment.py
git commit -m "feat(topic5-recruit): runner scaffold + template montage trace"
```

---

## Task 11: Per-seizure pipeline (`compute_seizure_recruitment`) — one seizure end-to-end

Re-read spec §4 (three-layer window, two-pass), §5 (5 detectors incl. ER held-out), §6 (fuse + agreement). This is the heart of the runner: one seizure → fused recruitment rank + agreement + ER reference.

**Files:**
- Modify: `scripts/run_topic5_ictal_recruitment.py`

- [ ] **Step 1: Add the per-seizure function**

```python
def _feature_traces(signal, fs):
    """Build the 4 fused-feature traces + the held-out ER trace on the common hop.
    HFA may be unavailable (low fs) -> key maps to (None, None); caller drops it."""
    traces = {}
    traces["line_length"] = recruit.line_length_trace(signal, fs, win_sec=1.0, hop_sec=HOP)
    traces["broadband"] = recruit.band_power_trace(signal, fs, band=(1.0, 45.0),
                                                    win_sec=1.0, hop_sec=HOP)
    try:
        traces["hfa"] = recruit.band_power_trace(signal, fs, band=(80.0, 150.0),
                                                 win_sec=0.5, hop_sec=HOP)
    except ValueError:
        traces["hfa"] = (None, None)                  # Nyquist: feature unavailable
    traces["spectral_edge"] = recruit.spectral_edge_trace(signal, fs, edge=0.9,
                                                           win_sec=1.0, hop_sec=HOP)
    er = compute_er(signal, fs, GAMMA_ER_BANDS["fast"], GAMMA_ER_BANDS["slow"],
                    win_sec=1.0, hop_sec=HOP)
    traces["er"] = (er, np.arange(er.shape[1]) * HOP)
    return traces


# per-feature analysis window (sec) — needed to convert between onset SECONDS and that
# feature's own frame grid, since features have different win (HFA=0.5, others=1.0).
FEATURE_WIN = {"line_length": 1.0, "broadband": 1.0, "hfa": 0.5,
               "spectral_edge": 1.0, "er": 1.0}


def _sec_to_frame(sec, *, pre_sec, win_sec, hop_sec, n_frames):
    """Inverse of t_onset_sec = frame*hop + win/2 - pre. Clamped to [0, n_frames]."""
    frame = int(round((float(sec) + float(pre_sec) - float(win_sec) / 2.0) / float(hop_sec)))
    return int(np.clip(frame, 0, n_frames))


def compute_seizure_recruitment(signal, fs, pre_sec, channels, lambdas, *,
                                eeg_onset_rel_sec=None):
    """One seizure -> fused recruitment rank (pass 2), agreement, ER reference.

    REVIEWER P0-1: operate entirely in onset SECONDS (rel clinical onset, center
    convention = CUSUM wrapper's t_onset_sec), NOT raw frame indices — the raw sample
    t_axis must NEVER index feature frames. pre_sec = extraction pre (signal frame 0 is
    t = -pre_sec). lambdas: {feature_key: lambda or NaN}. Returns dict or None.
    Window contract §4: baseline from the long pre-onset segment; pass-1 onset over the
    [-30,+30]s band -> t_global (sec); pass-2 onset over [t_global-2, t_global+RECRUIT_POST].
    """
    traces = _feature_traces(signal, fs)
    avail = [k for k in FUSED_FEATURES if traces[k][0] is not None
             and np.isfinite(lambdas.get(k, np.nan))]
    if len(avail) < 2:
        return None
    # robust-z each available feature + ER on its OWN frame grid + its OWN baseline window
    z, nfr = {}, {}
    for k in avail + ["er"]:
        tr = traces[k][0]
        nfr[k] = tr.shape[1]
        bl = resolve_baseline_window(nfr[k], hop_sec=HOP, pre_sec=pre_sec,
                                     eeg_onset_rel_sec=eeg_onset_rel_sec)
        if not bl.valid:
            return None
        z[k] = recruit.baseline_robust_z(tr, (bl.start_idx, bl.end_idx),
                                         hop_sec=HOP, min_baseline_valid_sec=MIN_BASELINE_SEC)
    n_valid = z[avail[0]].shape[0]

    def onsets_in(lo_sec, hi_sec, keys):
        """Per-feature onset SECONDS (rel clinical onset) within [lo_sec, hi_sec]."""
        out = {}
        for k in keys:
            win = FEATURE_WIN[k]
            lo_f = _sec_to_frame(lo_sec, pre_sec=pre_sec, win_sec=win, hop_sec=HOP, n_frames=nfr[k])
            hi_f = _sec_to_frame(hi_sec, pre_sec=pre_sec, win_sec=win, hop_sec=HOP, n_frames=nfr[k])
            arr = np.full(n_valid, np.nan)
            for c in range(z[k].shape[0]):
                if not np.isfinite(z[k][c]).any():
                    continue                       # baseline-invalid channel (NaN row)
                r = recruit.detect_contact_onset(
                    z[k][c], lam=lambdas[k], detection_idx_window=(lo_f, hi_f),
                    hop_sec=HOP, win_sec=win, pre_sec=pre_sec)
                if r["detected"]:
                    arr[c] = r["onset_sec"]        # SECONDS, comparable across features
            out[k] = arr
        return out

    # PASS 1: provisional onset SECONDS over [-30,+30]s -> fused provisional -> t_global (sec)
    p1 = onsets_in(-30.0, 30.0, avail)
    prov_rank, prov_onset = recruit.fuse_recruitment_rank(p1)
    g = recruit.resolve_global_onset(prov_onset, n_valid=n_valid, frac=GLOBAL_ONSET_FRAC)
    if not g["global_onset_resolved"]:
        return None
    t_global = g["t_global"]                       # SECONDS (resolve_global_onset is unit-agnostic)

    # PASS 2: per-contact onset SECONDS in [t_global-2, t_global+RECRUIT_POST]
    p2 = onsets_in(t_global - 2.0, t_global + RECRUIT_POST_SEC, avail + ["er"])
    # pre-onset-change: pass-1 onset before annotation-10s -> exclude from recruitment rank
    for k in avail:
        mask_pre = np.isfinite(p1[k]) & (p1[k] < PRE_ONSET_CHANGE_SEC)
        p2[k][mask_pre] = np.nan
    fused_rank, fused_onset = recruit.fuse_recruitment_rank({k: p2[k] for k in avail})
    per_rank = {k: recruit._rank_of(p2[k]) for k in avail}
    have_amp = [k for k in AMP_FEATURES if k in per_rank]
    ag = (recruit.feature_agreement(
            {**per_rank, SPECTRAL_FEATURE: per_rank.get(SPECTRAL_FEATURE, np.full(n_valid, np.nan))},
            amplitude=tuple(have_amp), spectral=SPECTRAL_FEATURE, early_k=EARLY_K)
          if len(have_amp) >= 2 else {"feature_agreement_flag": False})
    return {"recruitment_rank": fused_rank, "er_rank": recruit._rank_of(p2["er"]),
            "t_global_sec": float(t_global), "n_recruited": int(np.isfinite(fused_rank).sum()),
            "n_preonset_change": int(sum(int((np.isfinite(p1[k]) & (p1[k] < PRE_ONSET_CHANGE_SEC)).sum())
                                         for k in avail)),
            "agreement": ag, "channels": list(channels), "available_features": avail}
```

> **Signature confirmed:** `resolve_baseline_window(n_time_frames, *, hop_sec, pre_sec, buffer_sec=60, eeg_onset_rel_sec, min_baseline_valid_sec=60, onset_t_sec=0)` — `n_time_frames` is the first positional; **pre_sec must exceed buffer_sec (60s)**, so the extraction `pre_sec` (default 300, §4.1) is well above the 60s buffer. Each feature gets its own baseline window from its own frame count. `t_global` is in SECONDS (the value `resolve_global_onset` returns just follows the unit of its input array — here seconds, not frames). Cross-feature comparability holds because every `onset_sec` comes from the same `frame*hop + win/2 - pre` formula.

- [ ] **Step 2: Add a tiny smoke (synthetic seizure) to the test file**

```python
def test_compute_seizure_recruitment_smoke():
    # synthetic seizure with a KNOWN recruitment order; run the per-seizure function
    # directly (no disk). Guards the two-pass wiring + the SECONDS time axis (P0-1).
    # pre_sec=120 > buffer(60) so resolve_baseline_window yields a valid 60s baseline.
    import scripts.run_topic5_ictal_recruitment as R
    fs = 500.0
    pre_sec = 120.0
    n = int(round((pre_sec + 30.0) * fs))      # [-120, +30]s
    rng = np.random.default_rng(8)
    sig = 0.01 * rng.standard_normal((10, n))
    onset0 = int(round(pre_sec * fs))          # clinical onset (t=0) sample
    # recruit channels 0..9 in order, 0.2s apart, with sustained fast activity
    for c in range(10):
        s = onset0 + int(round(c * 0.2 * fs))
        tt = np.arange(n - s) / fs
        sig[c, s:] += np.sin(2 * np.pi * 90 * tt)
    lambdas = {k: 5.0 for k in R.FUSED_FEATURES}
    lambdas["er"] = 5.0
    out = R.compute_seizure_recruitment(sig, fs, pre_sec, [f"A{i}" for i in range(10)], lambdas)
    assert out is not None
    rr = out["recruitment_rank"]
    assert np.nanargmin(rr) in (0, 1)          # earliest-recruited channels rank lowest
    assert out["t_global_sec"] == pytest.approx(0.0, abs=2.0)   # global onset near t=0
```

Run: `pytest tests/test_topic5_ictal_recruitment.py -k compute_seizure_recruitment -v`
Expected: PASS (may need to relax the band/fs in the synthetic if onset detection is too strict — keep the order assertion).

- [ ] **Step 3: Commit**

```bash
git add scripts/run_topic5_ictal_recruitment.py tests/test_topic5_ictal_recruitment.py
git commit -m "feat(topic5-recruit): per-seizure two-pass recruitment pipeline + smoke"
```

---

## Task 12: B0 eligibility audit (`audit`)

Re-read spec §3.1 (audit columns; locked thresholds; report drops only) + §3.4 (montage columns). The audit enumerates subjects present in BOTH the masked-template tree AND a loadable ictal cohort, and computes per-subject pooled-λ (so `calibration_unstable` is known before sentinel).

**Files:**
- Modify: `scripts/run_topic5_ictal_recruitment.py`

- [ ] **Step 1: Add `cmd_audit`** (enumerate, load masked template per §3.4, load each seizure window bipolar, pool baseline z per feature, calibrate λ, fill columns)

```python
def _load_masked_template(ds_sid):
    """Return (channel_names, [template_rank vectors per cluster], template_k, swap_class,
    template_montage). Phantom-safe masked source (Stage-1 contract)."""
    # locate the masked per-subject JSON (rank_displacement or propagation tree)
    import glob
    cand = glob.glob(str(MASKED_ROOT / "**" / f"{ds_sid}.json"), recursive=True)
    if not cand:
        return None
    d = json.load(open(sorted(cand)[0]))
    ch = d["channel_names"]
    # cluster template ranks: reuse the same masked field Stage-1 used (centered_rank /
    # adaptive_cluster); confirm key live at execution and apply per-cluster valid_mask.
    templates = _extract_masked_cluster_templates(d)        # list of 1-D rank vectors (NaN=masked)
    return {"channels": ch, "templates": templates,
            "template_k": len(templates), "swap_class": d.get("swap_class", "na"),
            "template_montage": TEMPLATE_MONTAGE}


def cmd_audit(args):
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds_sid in _iter_cohort_subjects():                  # subjects in masked ∩ atlas/seizure inv
        info = _audit_one_subject(ds_sid)                   # returns the §3.1 column dict or None
        if info is not None:
            rows.append(info)
    import csv
    cols = ["subject_id", "dataset", "fs", "n_seizures_total", "n_seizures_loadable",
            "n_seizures_eligible", "n_channels_template_narrow", "n_channels_template_broad",
            "n_channels_recruited_min", "n_channels_recruited_median", "n_channels_recruited_max",
            "per_feature_available", "global_onset_resolved_fraction",
            "feature_agreement_flag_fraction", "template_k_narrow", "template_k_broad",
            "swap_class", "template_montage", "ictal_montage", "channel_identity_contract",
            "n_channels_montage_matched", "calibration_unstable_per_feature",
            "pooled_baseline_sec", "no_onset_rate_per_feature", "null_d_mode",
            "coord_available", "channel_name_normalization_status", "alignment_guard_pass",
            "n_preonset_change_contacts", "MIN_CH_pass"]
    with open(OUT_ROOT / "b0_recruitment_audit.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote b0_recruitment_audit.csv ({len(rows)} subjects)")
    drop = sum(1 for r in rows if r.get("MIN_CH_pass") is False)
    print(f"[MIN_CH={MIN_CH} locked] subjects failing MIN_CH: {drop} (reported, not tuned)")
```

(Implement `_iter_cohort_subjects`, `_extract_masked_cluster_templates`, `_audit_one_subject`, and the `ictal_montage` / `channel_identity_contract` fill at execution. `_audit_one_subject` must: load masked template; `assert_channel_identity(template_montage=TEMPLATE_MONTAGE, ictal_montage=ICTAL_REFERENCE-derived)`; for each loadable seizure pool the per-feature baseline z-frames; `calibrate_feature_lambda` per feature → record `calibration_unstable_per_feature` + `pooled_baseline_sec`; set `null_d_mode` = `mni_nn` for epilepsiae / `region_matched` for yuquan. **Locked thresholds, drops reported only — do NOT tune from the audit.**)

- [ ] **Step 2: Run the audit and eyeball**

Run: `python scripts/run_topic5_ictal_recruitment.py audit`
Expected: writes `b0_recruitment_audit.csv`; prints subject count + MIN_CH drop count. **Eyeball:** confirm `template_montage`/`ictal_montage` columns agree (no `MISMATCH`), `calibration_unstable_per_feature` is not pervasive, `null_d_mode` is `mni_nn` for epi / `region_matched` for yuquan.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_topic5_ictal_recruitment.py
git commit -m "feat(topic5-recruit): B0 eligibility audit (montage + pooled-lambda columns)"
```

---

## Task 13: Sentinel overlay (`sentinel`) — **MANUAL GATE**

Re-read spec §9 + §13 step 5. Pick 3–5 sentinel seizures (epi + yuquan, ≥1 low-voltage-fast candidate). For each, render raw trace + 5 detector traces + per-detector onsets + fused onset, and print the per-feature λ / pooled-baseline / no-onset-rate table. **This is a human visual gate — Phase 3 does not start until the user signs off.**

**Files:**
- Modify: `scripts/run_topic5_ictal_recruitment.py`
- Create: `scripts/plot_topic5_ictal_recruitment.py` (sentinel overlay only for now)

- [ ] **Step 1: Add `cmd_sentinel`** (loads N chosen seizures, runs `compute_seizure_recruitment`, dumps a per-seizure JSON + calls the overlay plotter; prints the λ/no-onset table)

```python
def cmd_sentinel(args):
    sentinel_ids = args.seizures        # e.g. ["epilepsiae/1146:0", "yuquan/litengsheng:1", ...]
    (OUT_ROOT / "sentinel").mkdir(parents=True, exist_ok=True)
    import importlib
    plotter = importlib.import_module("scripts.plot_topic5_ictal_recruitment")
    for sid_spec in sentinel_ids:
        subj, sz = sid_spec.rsplit(":", 1)
        sw = extract_seizure_window(subj, int(sz), pre_sec=BASELINE_PRE_SEC, post_sec=30.0,
                                    reference=ICTAL_REFERENCE)
        lambdas = _subject_lambdas(subj)            # from audit / recompute pooled
        # Task 11 signature: (signal, fs, pre_sec, channels, lambdas, *, eeg_onset_rel_sec).
        # NEVER pass sw.t_axis (raw sample axis) here — that was the P0-1 bug.
        eeg_rel = ((sw.eeg_onset_epoch - sw.clin_onset_epoch)
                   if sw.eeg_onset_epoch is not None else None)
        out = compute_seizure_recruitment(sw.signal, sw.fs, sw.pre_sec, sw.ch_names, lambdas,
                                          eeg_onset_rel_sec=eeg_rel)
        json.dump(_jsonable(out), open(OUT_ROOT / "sentinel" / f"{subj.replace('/','_')}_{sz}.json", "w"),
                  indent=2)
        plotter.plot_sentinel_overlay(sw, out, lambdas,
                                      OUT_ROOT / "sentinel" / f"{subj.replace('/','_')}_{sz}.png")
        print(f"[sentinel {sid_spec}] lambdas={lambdas} no_onset={_no_onset_rates(out)}")
    print("SENTINEL DONE — human visual gate. Do NOT run per-subject/cohort until sign-off.")
```

- [ ] **Step 2: Add the overlay plotter** in `scripts/plot_topic5_ictal_recruitment.py`

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_sentinel_overlay(sw, out, lambdas, path):
    """Raw trace + 5 detector traces with per-contact onsets + fused onset marker,
    for the earliest-recruited contacts. Paper-grade, self-contained (no codenames)."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    # ... raw EEG of the earliest K recruited contacts; feature traces; onset ticks ...
    # (concrete drawing wired at execution against the sentinel JSON shape)
    fig.suptitle(f"{sw.subject} seizure {sw.seizure_id}: early-ictal recruitment")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
```

- [ ] **Step 3: Run the sentinel and EYEBALL**

Run: `python scripts/run_topic5_ictal_recruitment.py sentinel --seizures epilepsiae/1146:0 yuquan/litengsheng:0 epilepsiae/1077:0`
Expected: 3–5 PNGs + JSONs under `results/topic5_ictal_recruitment/sentinel/`. **Human visual check (spec §13):** do the 5 detectors agree on the earliest contacts? Is `t_global` sensible? Does the pre-onset-change flag fire correctly? Is no-onset-rate acceptable under the chosen λ? Are bipolar-aliased channel names aligned to the template?

- [ ] **Step 4: Commit**

```bash
git add scripts/run_topic5_ictal_recruitment.py scripts/plot_topic5_ictal_recruitment.py
git commit -m "feat(topic5-recruit): sentinel overlay + lambda/no-onset table (manual gate)"
```

### PHASE 2 GATE (MANUAL)

**STOP. Present the sentinel overlays + λ/no-onset table to the user.** Per spec §13, Phase 3 (per-subject + cohort) does NOT run until the user confirms the instrument is sound. If the sentinel fails (detectors disagree, λ pushes no-onset too high, montage misaligns), iterate on feature/window/λ params and re-run the sentinel — do NOT run the cohort.

---

# PHASE 3 — Per-subject + cohort (only after sentinel sign-off)

## Task 14: `per-subject` (all null modes + de-anchor + ER held-out)

Re-read spec §7.1–7.2 (echo + nulls reuse Stage 1), §6.4 (ER held-out consistency), §4.3 (dual anchor). For each eligible subject: per-seizure recruitment rank → `compute_echo_strength` across null modes (channel / within_shaft / anchor_matched), LOO de-anchor (n_seizures≥4), `er_vs_fused_consistency`, both anchors (epi) / data-driven only (yuquan).

**Files:**
- Modify: `scripts/run_topic5_ictal_recruitment.py`

- [ ] **Step 1: Add `cmd_per_subject`** (mirrors Stage 1's `cmd_per_subject` shape, but the rank is the recruitment rank from Task 11, and templates come from `_load_masked_template`; null modes via `echo.compute_echo_strength(null_mode=...)`; shafts via `parse_shaft`; anchor_bins by distance-to-SOZ or mean ictal earliness per spec §4.6 — confirm which with the user before wiring, default mean-ictal-earliness)

```python
def cmd_per_subject(args):
    rng = np.random.default_rng(RNG_SEED)
    (OUT_ROOT / "per_subject").mkdir(parents=True, exist_ok=True)
    for ds_sid in _iter_cohort_subjects():
        sub = _assemble_subject(ds_sid)        # seizures' recruitment ranks + er_rank + templates + shafts + anchor_bins
        if sub is None or sub["n_eligible_seizures"] < 1:
            continue
        templates = sub["templates"]           # masked narrow (Main-A)
        shafts = sub["shafts"]; anchor_bins = sub["anchor_bins"]
        per_seizure = []
        rr_matrix = sub["recruitment_ranks"]   # (n_seiz, n_ch)
        cap = echo.shaft_block_capacity(shafts)
        for k, rr in enumerate(rr_matrix):
            rec = {"seizure_idx": k, "feature_agreement_flag": sub["agree_flags"][k],
                   "spectral_conflict_flag": sub["spectral_conflict"][k]}
            modes = [("channel", None), ("within_shaft", shafts),
                     ("anchor_matched", anchor_bins)]
            for mode, blocks in modes:
                rec[mode] = echo.compute_echo_strength(rr, templates, B=B, rng=rng,
                                                       min_ch=MIN_CH, null_mode=mode, blocks=blocks)
            rec["er_vs_fused_consistency"] = float(recruit._pair_rho(sub["er_ranks"][k], rr))
            per_seizure.append(rec)
        deanchor = (echo.compute_deanchor_echo(rr_matrix, templates, B=B, rng=rng, min_ch=MIN_CH)
                    if rr_matrix.shape[0] >= 4 else None)
        out = {"subject": ds_sid, "dataset": sub["dataset"], "swap_class": sub["swap_class"],
               "template_k": sub["template_k"], "anchor_reliability": echo.anchor_reliability(rr_matrix),
               "null_d_mode": sub["null_d_mode"], "per_seizure": per_seizure, "deanchor": deanchor,
               "recruitment_ranks": [list(r) for r in rr_matrix],
               "templates": [list(t) for t in templates], "channels": list(sub["channels"])}
        json.dump(_jsonable(out), open(OUT_ROOT / "per_subject" / f"{ds_sid}.json", "w"), indent=2)
    print("per-subject done")
```

- [ ] **Step 2: Run**

Run: `python scripts/run_topic5_ictal_recruitment.py per-subject`
Expected: per-subject JSONs under `results/topic5_ictal_recruitment/per_subject/`. Eyeball one: `feature_agreement_flag` present per seizure, `er_vs_fused_consistency` finite, null modes populated.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_topic5_ictal_recruitment.py
git commit -m "feat(topic5-recruit): per-subject echo across null modes + ER held-out consistency"
```

---

## Task 15: Null D coordinate/region control (`_pool_null_d`) — built BEFORE cohort

Re-read spec §7.3. Epilepsiae: MNI nearest-neighbor remap of other subjects' templates. Yuquan: region/shaft-matched remap. Forbid cross-subject NN on yuquan subject-native coords. **This task precedes the cohort task (reviewer P0-3): `cmd_cohort` calls `_pool_null_d`, so it must exist first.**

**Files:**
- Modify: `scripts/run_topic5_ictal_recruitment.py`
- Test: `tests/test_topic5_ictal_recruitment.py`

- [ ] **Step 1: Write the failing test** (the runner Null D dispatcher must route by dataset and never call coord-NN for yuquan)

```python
def test_null_d_routes_by_dataset(monkeypatch):
    import scripts.run_topic5_ictal_recruitment as R
    calls = {"mni_nn": 0, "region": 0}
    monkeypatch.setattr(R, "_null_d_mni_nn", lambda *a, **k: calls.__setitem__("mni_nn", calls["mni_nn"] + 1) or {"e_k": 0.0})
    monkeypatch.setattr(R, "_null_d_region_matched", lambda *a, **k: calls.__setitem__("region", calls["region"] + 1) or {"e_k": 0.0})
    subs = [{"subject": "epilepsiae_1", "dataset": "epilepsiae", "null_d_mode": "mni_nn",
             "recruitment_ranks": [[0.0] * 8], "templates": [[0.0] * 8], "channels": [f"C{i}" for i in range(8)]},
            {"subject": "yuquan_a", "dataset": "yuquan", "null_d_mode": "region_matched",
             "recruitment_ranks": [[0.0] * 8], "templates": [[0.0] * 8], "channels": [f"A{i}" for i in range(8)]}]
    R._pool_null_d(subs)
    assert calls["mni_nn"] >= 1 and calls["region"] >= 1   # both routed; yuquan never via mni_nn
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic5_ictal_recruitment.py -k null_d_routes -v`
Expected: FAIL with `AttributeError` (functions not defined).

- [ ] **Step 3: Implement `_pool_null_d`, `_null_d_mni_nn`, `_null_d_region_matched`** (mni_nn uses `seeg_coord_loader.load_subject_coords(dataset='epilepsiae')` → mni152 coords → nearest-neighbor remap of other subjects' templates; region_matched uses yuquan region/shaft labels). The dispatcher routes each subject by `null_d_mode` and MUST raise / skip rather than call `_null_d_mni_nn` for a yuquan subject (subject-native coords are not cross-subject comparable, spec §7.3). **`_pool_null_d` returns a `by_mode` dict so the verdict can check each coordinate space separately (spec §7.3, reviewer):**

```python
def _pool_null_d(subs):
    """Returns {"by_mode": {"mni_nn": {neutral, wilcoxon_p_onesided, n_subjects},
    "region_matched": {...}}, "neutral": <both modes neutral>}. A mode with no eligible
    subjects is OMITTED (the verdict treats a missing mode as neutral — it cannot fail).
    `inapplicable` (no coords/region) subjects are counted but NOT silently passed."""
    by_mode = {}
    for mode, fn in (("mni_nn", _null_d_mni_nn), ("region_matched", _null_d_region_matched)):
        recs = []
        for s in subs:
            if s["null_d_mode"] != mode:
                continue
            for seiz in s["recruitment_ranks"]:
                recs.append({"subject": s["subject"], "e_k": fn(s, seiz, subs).get("e_k")})
        if recs:
            pooled = echo.pool_echo_subject_level(recs)
            pooled["neutral"] = not (pooled.get("wilcoxon_p_onesided", 1) < 0.05)
            by_mode[mode] = pooled
    overall = all(v.get("neutral", True) for v in by_mode.values()) if by_mode else False
    inapplicable = [s["subject"] for s in subs if s.get("null_d_mode") == "inapplicable"]
    return {"by_mode": by_mode, "neutral": overall, "inapplicable_subjects": inapplicable}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_topic5_ictal_recruitment.py -k null_d_routes -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_topic5_ictal_recruitment.py tests/test_topic5_ictal_recruitment.py
git commit -m "feat(topic5-recruit): Null D split by coord space (epi MNI-NN / yuquan region)"
```

---

## Task 16: `cohort` — Main-A verdict + Main-B broad extension + bad-data

Re-read spec §7.4 (verdict), §8 (Main-A primary / Main-B broad with unsealed caveat), §7.3 (Null D by dataset, implemented in Task 15), §4.4 bad-data, §4.1b LOO de-anchor. Primary subset = seizures with `feature_agreement_flag=True`; spectral-conflict seizures stay in but flagged.

**Files:**
- Modify: `scripts/run_topic5_ictal_recruitment.py`

- [ ] **Step 1: Add `cmd_cohort`** (subject-level pooling via `echo.pool_echo_subject_level`; primary = channel null on agreement-flagged seizures; `_pool_null_d` from Task 15; bad-data via `e_k_baddata`; `_pool_deanchor` for the LOO de-anchor cohort pool; Main-B = re-run on broad templates with `broad_unsealed` epi → sensitivity)

```python
def cmd_cohort(args):
    subs = [json.load(open(p)) for p in sorted((OUT_ROOT / "per_subject").glob("*.json"))]

    def pool(mode, *, agreement_only=True, subset=None):
        recs = []
        for s in subs:
            if subset and s["dataset"] != subset:
                continue
            for ps in s["per_seizure"]:
                if agreement_only and not ps.get("feature_agreement_flag"):
                    continue
                m = ps.get(mode)
                if m is None:
                    continue
                recs.append({"subject": s["subject"], "e_k": m.get("e_k")})
        return echo.pool_echo_subject_level(recs)

    bd = [{"subject": s["subject"], "e_k_baddata": ps["channel"].get("e_k_baddata"),
           "e_k": ps["channel"].get("e_k")}
          for s in subs for ps in s["per_seizure"]
          if ps.get("feature_agreement_flag") and ps.get("channel")]

    summary = {
        "scope": "main_a_narrow_recruitment_echo",
        "primary_channel": pool("channel"),
        "primary_within_shaft": pool("within_shaft"),
        "primary_anchor_matched": pool("anchor_matched"),
        "epi_only": pool("channel", subset="epilepsiae"),
        "yuquan_only": pool("channel", subset="yuquan"),
        "bad_data_regression": echo.bad_data_regression(bd),
        "null_d": _pool_null_d(subs),                  # by_mode (§7.3, Task 15): mni_nn / region_matched
        "deanchor_all": _pool_deanchor(subs),          # §4.1b LOO de-anchor cohort pool (P1-2)
        "cluster_robust_sensitivity": _cluster_robust_sensitivity(subs),  # §7.1 per-seizure leg
        "er_vs_fused_consistency_dist": _consistency_dist(subs),
        "construct_validity": _construct_validity_dist(subs),
        "main_b_broad_extension": _main_b_broad(subs),  # broad templates; epi unsealed->sensitivity
    }
    summary["verdict"] = _assign_verdict(summary, subs)   # §7.4
    json.dump(_jsonable(summary), open(OUT_ROOT / "cohort_recruitment_summary.json", "w"), indent=2)
    print("verdict:", summary["verdict"]["label"])
```

`_pool_deanchor` + `_assign_verdict` (§7.4) — encode the conclusion into thresholds (mirror Stage 1's guard that `站住·*` requires sign/bootstrap sensitivities present + bad-data clean + Null D neutral). **含具体通路 additionally requires the LOO de-anchor leg (reviewer P1-2):**

```python
def _pool_deanchor(subs):
    """Cohort pool of the LOO de-anchor echo (§4.1b). Each subject's `deanchor` is a
    list of per-seizure echo records (compute_deanchor_echo output) or None."""
    recs = []
    for s in subs:
        for r in (s.get("deanchor") or []):
            recs.append({"subject": s["subject"], "e_k": r.get("e_k")})
    return echo.pool_echo_subject_level(recs)


def _cluster_robust_sensitivity(subs):
    """Spec §7.1 per-seizure cluster-robust leg: e_k ~ 1 with cluster-robust SE by
    subject. direction_ok = intercept > 0 AND one-sided p < 0.05 (same direction as the
    subject-level primary). Uses statsmodels OLS cov_type='cluster'."""
    import statsmodels.api as sm
    rows = [(s["subject"], ps["channel"]["e_k"]) for s in subs for ps in s["per_seizure"]
            if ps.get("feature_agreement_flag") and ps.get("channel")
            and np.isfinite(ps["channel"].get("e_k", np.nan))]
    subjects = [r[0] for r in rows]
    if len(rows) < 3 or len(set(subjects)) < 2:        # cov_type='cluster' needs >=2 groups
        return {"slope": float("nan"), "p_onesided": float("nan"), "direction_ok": False,
                "n_clusters": len(set(subjects))}
    y = np.array([r[1] for r in rows], dtype=float)
    _, groups = np.unique(subjects, return_inverse=True)   # stable int group codes
    res = sm.OLS(y, np.ones((y.size, 1))).fit(cov_type="cluster", cov_kwds={"groups": groups})
    coef = float(res.params[0]); p2 = float(res.pvalues[0])
    p1 = p2 / 2.0 if coef > 0 else 1.0 - p2 / 2.0       # one-sided (>0)
    return {"slope": coef, "p_onesided": p1, "direction_ok": bool(coef > 0 and p1 < 0.05)}


def _assign_verdict(summary, subs):
    p = summary["primary_channel"]
    n_primary = int(p.get("n_subjects", 0))            # subjects in the agreement-flagged channel pool
    has_sens = (np.isfinite(p.get("sign_p_onesided", np.nan)) and
                np.isfinite(p.get("boot_ci95", [np.nan])[0]))
    bad_clean = not (summary["bad_data_regression"].get("wilcoxon_p_onesided", 1) < 0.05)
    nd = summary["null_d"].get("by_mode", {})          # §7.3 per-mode, NOT one pooled neutral
    nd_clean = (nd.get("mni_nn", {}).get("neutral", True)        # missing mode = no subjects -> can't fail
                and nd.get("region_matched", {}).get("neutral", True))
    cluster_ok = bool(summary["cluster_robust_sensitivity"].get("direction_ok", False))  # §7.1 leg
    if n_primary < 6:
        return {"label": "没看清", "why": "primary agreement subjects < 6"}
    sig = (p.get("wilcoxon_p_onesided", 1) < 0.05 and p["median_E_s"] > 0
           and has_sens and cluster_ok and bad_clean and nd_clean)
    if not sig:
        return {"label": "真仪器阴性/没看清",
                "why": "primary not significant OR a leg missing (sign/bootstrap/cluster-robust/bad-data/Null-D-by-mode)"}
    if n_primary < 10:                                  # spec §7.4: 站住 needs >=10 subjects
        return {"label": "临界",
                "why": "all legs pass but 6<=primary subjects<10; spec requires >=10 to 站住 — Stage 2 continues"}
    # standing (>=10 subjects + all sensitivities/controls). 含具体通路 additionally needs
    # (within-shaft OR anchor-matched) AND LOO de-anchor same-direction significant (§7.4).
    a = summary["primary_within_shaft"]; c = summary["primary_anchor_matched"]
    da = summary["deanchor_all"]
    ac_sig = (a.get("wilcoxon_p_onesided", 1) < 0.05 or c.get("wilcoxon_p_onesided", 1) < 0.05)
    deanchor_sig = (da.get("wilcoxon_p_onesided", 1) < 0.05 and da.get("median_E_s", 0) > 0)
    specific = ac_sig and deanchor_sig
    return {"label": "站住·含具体通路" if specific else "站住·稳定锚为主",
            "why": "echo holds (>=10 subjects, all controls); " +
                   ("A/C survive AND LOO de-anchor significant" if specific
                    else "A/C or LOO de-anchor flatten")}
```

- [ ] **Step 2: Run**

Run: `python scripts/run_topic5_ictal_recruitment.py cohort`
Expected: writes `cohort_recruitment_summary.json`; prints a §7.4 verdict label.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_topic5_ictal_recruitment.py
git commit -m "feat(topic5-recruit): cohort Main-A verdict + Main-B broad + LOO de-anchor + Null D"
```

---

# PHASE 4 — Figures + archive

## Task 17: Cohort figures + `figures/README.md`

Re-read spec §11.3 (4 figures) + AGENTS.md Results Directory Standards (Chinese README, one paragraph per figure ending `**关注点**：`, written AFTER figures render).

**Files:**
- Modify: `scripts/plot_topic5_ictal_recruitment.py`
- Create: `results/topic5_ictal_recruitment/figures/README.md`

- [ ] **Step 1: Add the 4 cohort plotters** (`recruitment_echo_forest`, `construct_validity`, `null_mode_panel`, `narrow_vs_broad`) — each paper-grade self-contained (no internal codenames on axes; see `MEMORY.md feedback_figure_self_contained_paper_grade`).

- [ ] **Step 2: Generate and eyeball**

Run: `python scripts/plot_topic5_ictal_recruitment.py`
Expected: 4 PNGs under `results/topic5_ictal_recruitment/figures/`. Eyeball each before writing the README.

- [ ] **Step 3: Write `figures/README.md`** (Chinese, `### filename` + 2–4 sentences + `**关注点**：`), AFTER the figures render.

- [ ] **Step 4: Commit**

```bash
git add scripts/plot_topic5_ictal_recruitment.py results/topic5_ictal_recruitment/figures/README.md
git commit -m "feat(topic5-recruit): cohort figures + README"
```

---

## Task 18: Archive doc + topic5 doc backlink

Re-read AGENTS.md 文档与输出形式 (archive to `docs/archive/topic5/ictal_recruitment/`; main doc keeps only a backlink until all sensitivities pass) + CLAUDE.md §8 (plain-language recap) + `hfosp-plain-language-recap` skill.

**Files:**
- Create: `docs/archive/topic5/ictal_recruitment/stage2_results_2026-06-10.md`
- Modify: `docs/topic5_seizure_subtyping.md` (§5 history index backlink only)

- [ ] **Step 1: Write the archive doc** — plain-language 三段式 abstract (测了什么 / 怎么测的 / 揭示了什么), then the cohort numbers, Main-A verdict, Main-B caveats, construct-validity (`er_vs_fused_consistency`) distribution, null-mode table, sentinel record, sensitivity battery, all caveats. Lead with the plain-language recap; codenames only in parentheses.

- [ ] **Step 2: Add a backlink line to `docs/topic5_seizure_subtyping.md` §5** — point to the Stage-2 spec/plan/archive + results path; state it is the real recruitment instrument (Stage 2), exploratory, conclusions pending full sensitivity. No paper-level claim in the main doc yet.

- [ ] **Step 3: Commit**

```bash
git add docs/archive/topic5/ictal_recruitment/stage2_results_2026-06-10.md docs/topic5_seizure_subtyping.md
git commit -m "docs(topic5 stage2): recruitment instrument results archive + main-doc backlink"
```

---

## Self-Review (run after writing all tasks)

**Spec coverage (v2):** §3.4 montage hard contract (Tasks 8, 10, 12) ✓; §4.1 three-layer window + §4.2 non-vacuous global onset (Tasks 6, 11) ✓; §4.3 dual anchor (Task 11/14) ✓; §5.1 5 detectors incl. ER held-out (Tasks 1, 2, 11) ✓; §5.3 robust-z + pooled per-hour λ + calibration_unstable (Tasks 3, 5, 12) ✓; §5.4 no-onset/ambiguous (Task 4) ✓; §5.2 two-pass (Tasks 6, 11) ✓; §6 family-structured fusion + amplitude-only gate + ER held-out consistency (Tasks 7, 11, 14) ✓; §7.1–7.2 echo + nulls reuse Stage 1 (Tasks 9, 14) ✓; §7.3 Null D by coord space, checked per-mode in verdict (Task 15 built before cohort — reviewer P0-3 + by_mode) ✓; §7.4 verdict full gate — ≥10 primary subjects + sign/bootstrap + per-seizure cluster-robust + bad-data + Null-D-by-mode + (within-shaft/anchor) + LOO de-anchor (Task 16 — reviewer P1-2/subject-floor/cluster-robust) ✓; §8 Main-A/Main-B (Task 16) ✓; §9 cross-feature agreement construct validity (Tasks 7, 14, 17) ✓; §10 sensitivity battery (Task 16 epi/yuquan + sentinel) ✓; §13 staged gate (Phase gates) ✓; figures (Task 17) + archive (Task 18) ✓.

**Open implementation risks to verify at execution (flagged checks, not placeholders):**
- Masked-template cluster-rank field + per-cluster `valid_mask` key layout — Task 10/12 inspects live (`_extract_masked_cluster_templates`); reuse Stage 1's masked loader logic.
- `resolve_baseline_window` / `compute_cusum_n_d_with_time` / `calibrate_lambda_per_subject` exact signatures — Tasks 4/5/11 confirm live and adapt the call while preserving the contract.
- `anchor_matched` bins: distance-to-SOZ vs mean ictal earliness — **confirm with user before Task 14** (default mean ictal earliness); if no SOZ for a subject, that null is skipped for them (recorded, not dropped).
- Sentinel seizure IDs (Task 13) — pick from the audit; include ≥1 low-voltage-fast candidate. The Phase 2 manual gate is load-bearing; do not skip.
- Template montage (Task 10 Step 1) — if the trace is genuinely ambiguous (monopolar vs bipolar-aliased), STOP and ask the user; do not guess.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-10-topic5-ictal-recruitment-stage2.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration. Phase boundaries (esp. the Phase 2 sentinel manual gate) are natural review checkpoints.
2. **Inline Execution** — execute tasks in this session with checkpoints at each Phase gate.

Which approach?
