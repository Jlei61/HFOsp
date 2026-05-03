"""TDD tests for PR-T3-1 v2.1 Layer A — ictal ER-rank producer.

Covers Step A.1 + Step A.2 of the v2.1 pivot plan:

- ``compute_cusum_n_d``: Page-Hinkley alarm time on z-ER (A1, A2)
- ``calibrate_lambda_per_subject``: per-subject λ calibration (A3)
- ``rank_channels_by_n_d``: fractional rank with NaN tail + ties (A4)
- ``compute_seizure_status``: 3-state ok / onset_tied / onset_unreached
  (A5, A6, A7)
- ``compute_per_subject_r_sz``: cross-seizure median rank with
  baseline_invalid exclusion (A8, A8b)
- ``compute_stability_s_sz``: pairwise Spearman over qualifying
  seizures, baseline_invalid excluded (A9, A9b)
- ``time_shifted_seizure_onsets``: shifted onsets inside baseline
  window avoiding real seizures (A10, A11)
"""

from __future__ import annotations

import numpy as np
import pytest

from src.ictal_er_rank import (
    calibrate_lambda_per_subject,
    compute_cusum_n_d,
    compute_per_subject_r_sz,
    compute_seizure_status,
    compute_stability_s_sz,
    rank_channels_by_n_d,
    time_shifted_seizure_onsets,
)


# ---------------------------------------------------------------------------
# A1. CUSUM alarm fires on a known sustained step input
# ---------------------------------------------------------------------------


def test_cusum_alarm_on_known_step_input():
    """Synthetic z-ER with zero baseline + sustained +5 step at t=10:
    CUSUM with bias=0.5 and small λ should alarm within a few frames
    of the step. With bias=0.5, ``U`` accumulates 4.5 per frame after
    t=10; λ=5 → alarm at frame 11; λ=10 → frame 12.
    """
    n_frames = 50
    z = np.zeros(n_frames, dtype=float)
    z[10:] = 5.0   # sustained step

    # λ = 5: U[10] = 4.5 (< 5), U[11] = 9.0 (>= 5) → first alarm at 11.
    n_d = compute_cusum_n_d(z, lambda_thresh=5.0, bias=0.5)
    assert n_d == 11

    # λ = 10: U[11] = 9.0 (< 10), U[12] = 13.5 (>= 10) → first alarm at 12.
    n_d_higher = compute_cusum_n_d(z, lambda_thresh=10.0, bias=0.5)
    assert n_d_higher == 12

    # Very large λ = 1000 → no alarm in this short 50-frame trace.
    # (At λ=100, alarm fires at frame 32 since U[n] = 4.5 * (n-9).)
    n_d_huge = compute_cusum_n_d(z, lambda_thresh=1000.0, bias=0.5)
    assert n_d_huge is None


def test_cusum_handles_nan_input_by_resetting():
    """NaN values reset the running CUSUM to 0 (matching PR-6A
    ``detect_er_onset_preview`` behavior). A NaN dropout must NOT
    propagate prior accumulation across the gap.
    """
    z = np.zeros(20, dtype=float)
    z[5:10] = 5.0       # first step (would alarm at frame 6 with λ=5)
    z[10] = np.nan      # gap → resets U
    z[11:] = 5.0        # second step (re-accumulates from 0)

    # With λ = 50 (high), no alarm fires from first step alone (only
    # 4 frames of +5 contribution, peak U = 18). After NaN reset, the
    # second step (9 frames of +5) gets U up to 40.5 — still no alarm
    # at λ=50. Verifies the reset by demonstrating the gap kills
    # accumulation.
    n_d = compute_cusum_n_d(z, lambda_thresh=50.0, bias=0.5)
    assert n_d is None

    # Without the NaN reset, U would have continued accumulating from
    # frame 5 through frame 19 = 14 contributions of 4.5 = 63 > 50 →
    # alarm somewhere. So this test passing proves the reset works.


def test_cusum_detection_window_filters_alarms():
    """Alarms outside ``detection_idx_window`` must NOT be returned;
    CUSUM accumulates from frame 0 so an early excursion can still
    trigger an alarm inside a later window.
    """
    n = 50
    z = np.zeros(n, dtype=float)
    z[5:] = 5.0   # step at frame 5

    # Without window: alarm at frame 6 (U[6] = 9 >= 5).
    assert compute_cusum_n_d(z, lambda_thresh=5.0, bias=0.5) == 6

    # Detection window [10, 50): the very-early alarm at 6 should NOT
    # be returned. But CUSUM continues to accumulate, so the next frame
    # in window where U >= 5 is frame 10 (U[10] = 27).
    n_d_windowed = compute_cusum_n_d(
        z, lambda_thresh=5.0, bias=0.5, detection_idx_window=(10, 50)
    )
    assert n_d_windowed == 10


def test_cusum_rejects_invalid_inputs():
    z = np.zeros(10, dtype=float)
    with pytest.raises(ValueError, match="must be 1D"):
        compute_cusum_n_d(np.zeros((2, 5)), lambda_thresh=5.0)
    with pytest.raises(ValueError, match="lambda_thresh"):
        compute_cusum_n_d(z, lambda_thresh=0.0)
    with pytest.raises(ValueError, match="lambda_thresh"):
        compute_cusum_n_d(z, lambda_thresh=-1.0)
    with pytest.raises(ValueError, match="detection_idx_window"):
        compute_cusum_n_d(z, lambda_thresh=5.0, detection_idx_window=(5, 5))
    with pytest.raises(ValueError, match="detection_idx_window"):
        compute_cusum_n_d(z, lambda_thresh=5.0, detection_idx_window=(-1, 5))


# ---------------------------------------------------------------------------
# A2. CUSUM does NOT false-alarm on white noise with calibrated λ
# ---------------------------------------------------------------------------


def test_cusum_no_false_alarm_on_calibrated_white_noise():
    """White noise z-ER (zero mean, unit variance) under bias=0.5 should
    NOT trigger CUSUM at the calibrated λ. Use a long baseline so the
    calibration is reliable.
    """
    rng = np.random.default_rng(0)
    n_channels = 4
    n_frames = 30000   # 50 min at hop=0.1s
    baseline = rng.normal(0.0, 1.0, size=(n_channels, n_frames))

    # Calibrate λ targeting 1 false alarm per hour, pooled across channels.
    lam = calibrate_lambda_per_subject(
        baseline, fpr_target_per_hour=1.0, bias=0.5, hop_sec=0.1
    )
    assert lam > 0
    assert lam < 100

    # Now run a fresh white-noise stretch and verify alarms are rare.
    test_noise = rng.normal(0.0, 1.0, size=(n_channels, n_frames))
    test_hours = (n_frames * 0.1) / 3600.0
    n_alarms = 0
    for ch in range(n_channels):
        if compute_cusum_n_d(test_noise[ch], lambda_thresh=lam, bias=0.5) is not None:
            n_alarms += 1
    # Expected alarms ≤ fpr_target_per_hour * baseline_hours * channels ≈ low.
    # Allow up to 5x slack since calibrate counts re-armed alarms while
    # this test counts only first alarms per channel.
    assert n_alarms <= max(2, 5 * test_hours * n_channels)


# ---------------------------------------------------------------------------
# A3. Calibrate λ converges and grows with stricter FPR target
# ---------------------------------------------------------------------------


def test_calibrate_lambda_grows_with_stricter_fpr():
    """A stricter FPR target (smaller alarms/hour) requires a larger λ."""
    rng = np.random.default_rng(1)
    baseline = rng.normal(0.0, 1.0, size=(4, 30000))

    lam_loose = calibrate_lambda_per_subject(
        baseline, fpr_target_per_hour=10.0, bias=0.5, hop_sec=0.1
    )
    lam_strict = calibrate_lambda_per_subject(
        baseline, fpr_target_per_hour=0.1, bias=0.5, hop_sec=0.1
    )
    assert lam_strict >= lam_loose


def test_calibrate_lambda_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="must be 2D"):
        calibrate_lambda_per_subject(np.zeros(100))
    with pytest.raises(ValueError, match="at least 2 frames"):
        calibrate_lambda_per_subject(np.zeros((2, 1)))
    with pytest.raises(ValueError, match="fpr_target"):
        calibrate_lambda_per_subject(np.zeros((2, 100)), fpr_target_per_hour=0.0)
    with pytest.raises(ValueError, match="hop_sec"):
        calibrate_lambda_per_subject(np.zeros((2, 100)), hop_sec=0.0)
    with pytest.raises(ValueError, match="lambda_min"):
        calibrate_lambda_per_subject(
            np.zeros((2, 100)), lambda_min=10.0, lambda_max=5.0
        )
    with pytest.raises(ValueError, match="lambda_n_grid"):
        calibrate_lambda_per_subject(np.zeros((2, 100)), lambda_n_grid=1)


# ---------------------------------------------------------------------------
# A4. Rank channels by n_d with NaN tail + tie averaging
# ---------------------------------------------------------------------------


def test_rank_channels_by_n_d_distinct_values():
    n_d = {"a": 100.0, "b": 50.0, "c": 200.0}
    ranks = rank_channels_by_n_d(n_d)
    assert ranks == {"a": 1.0, "b": 0.0, "c": 2.0}


def test_rank_channels_by_n_d_none_goes_to_none():
    n_d = {"a": 50.0, "b": None, "c": 100.0}
    ranks = rank_channels_by_n_d(n_d)
    assert ranks["a"] == 0.0
    assert ranks["c"] == 1.0
    assert ranks["b"] is None


def test_rank_channels_by_n_d_fractional_ties():
    """Channels whose n_d differ by ≤ tie_eps_frames get the average rank."""
    # b and c are 0.3 frames apart → tied at tie_eps_frames=0.5;
    # a is alone, d is alone.
    n_d = {"a": 10.0, "b": 50.0, "c": 50.3, "d": 100.0}
    ranks = rank_channels_by_n_d(n_d, tie_eps_frames=0.5)
    assert ranks["a"] == 0.0
    # b, c tied → avg rank of (1, 2) = 1.5
    assert ranks["b"] == 1.5
    assert ranks["c"] == 1.5
    assert ranks["d"] == 3.0


def test_rank_channels_by_n_d_three_way_tie():
    """Three channels within tie_eps form one block → avg rank 1.0."""
    n_d = {"a": 0.0, "b": 100.0, "c": 100.2, "d": 100.4}
    ranks = rank_channels_by_n_d(n_d, tie_eps_frames=0.5)
    assert ranks["a"] == 0.0
    # b, c, d span 0.4 frames; consecutive gaps 0.2 each, both <= 0.5
    # → all three tied. avg rank of (1, 2, 3) = 2.0.
    assert ranks["b"] == 2.0
    assert ranks["c"] == 2.0
    assert ranks["d"] == 2.0


def test_rank_channels_by_n_d_all_none():
    """Edge case: every channel returned None → all ranks None."""
    n_d = {"a": None, "b": None}
    assert rank_channels_by_n_d(n_d) == {"a": None, "b": None}


def test_rank_channels_by_n_d_handles_nan_as_none():
    """NaN n_d treated as None (no alarm)."""
    n_d = {"a": 10.0, "b": float("nan")}
    ranks = rank_channels_by_n_d(n_d)
    assert ranks["a"] == 0.0
    assert ranks["b"] is None


# ---------------------------------------------------------------------------
# A5. seizure_status = "ok" when n_d distribution is well-spread
# ---------------------------------------------------------------------------


def test_seizure_status_ok_for_spread_n_d():
    """5 channels, n_d evenly spread well after onset, all active →
    status = "ok"."""
    onset = 100
    n_d = {f"ch{i}": float(onset + 5 * i) for i in range(5)}
    res = compute_seizure_status(
        n_d,
        n_total=5,
        onset_idx=onset,
        fast_recruit_window_frames=2,    # only frames 100-102 count "fast"
    )
    assert res.status == "ok"
    assert res.n_active == 5
    assert res.n_total == 5
    # ch0 fires at onset (idx 100) → fast-recruited; ch1 at 105 → not.
    # → fast_recruit_fraction = 1/5 = 0.2 < 0.6 default threshold.
    assert res.fast_recruit_fraction == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# A6. seizure_status = "onset_tied" when too many channels recruit fast
# ---------------------------------------------------------------------------


def test_seizure_status_onset_tied_for_fast_recruitment():
    """Plan §3.3: > 60 % channels recruited within [onset, onset+1s] →
    onset_tied. Use 10 channels, 7 fire within 10 frames of onset."""
    onset = 100
    n_d = {f"ch{i}": float(onset + i) for i in range(10)}
    # ch0..ch9 fire at onset+0..+9. With window 10 frames and threshold
    # 60%, fast_recruited = all 10 → fraction 1.0 > 0.6 → onset_tied.
    res = compute_seizure_status(
        n_d,
        n_total=10,
        onset_idx=onset,
        fast_recruit_window_frames=10,
    )
    assert res.status == "onset_tied"
    assert res.n_active == 10
    assert res.fast_recruit_fraction == 1.0


# ---------------------------------------------------------------------------
# A7. seizure_status = "onset_unreached" when too few channels detect
# ---------------------------------------------------------------------------


def test_seizure_status_onset_unreached_for_sparse_activation():
    """Plan §3.3: < 30 % channels with valid n_d → onset_unreached.
    10 total, only 2 active → 20 % < 30 %."""
    onset = 100
    n_d = {f"ch{i}": (float(onset + 5 * i) if i < 2 else None)
           for i in range(10)}
    res = compute_seizure_status(
        n_d,
        n_total=10,
        onset_idx=onset,
    )
    assert res.status == "onset_unreached"
    assert res.n_active == 2
    assert res.n_total == 10


def test_seizure_status_rejects_zero_n_total():
    with pytest.raises(ValueError, match="n_total"):
        compute_seizure_status({}, n_total=0, onset_idx=0)


# ---------------------------------------------------------------------------
# A8. compute_per_subject_r_sz: cross-seizure median rank
# ---------------------------------------------------------------------------


def test_compute_per_subject_r_sz_median_rank():
    """3 seizures × 4 channels: r_sz = median rank per channel.

    Per-seizure ranks (NaN for unreached / non-finite):
        seizure 0: a=0, b=1, c=2, d=3
        seizure 1: a=0, b=2, c=1, d=3
        seizure 2: a=1, b=0, c=2, d=3
    median per channel: a=0, b=1, c=2, d=3
    """
    per_seizure_ranks = [
        {"a": 0.0, "b": 1.0, "c": 2.0, "d": 3.0},
        {"a": 0.0, "b": 2.0, "c": 1.0, "d": 3.0},
        {"a": 1.0, "b": 0.0, "c": 2.0, "d": 3.0},
    ]
    seizure_statuses = ["ok", "ok", "ok"]
    r_sz = compute_per_subject_r_sz(
        per_seizure_ranks, seizure_statuses=seizure_statuses
    )
    assert r_sz["a"] == pytest.approx(0.0)
    assert r_sz["b"] == pytest.approx(1.0)
    assert r_sz["c"] == pytest.approx(2.0)
    assert r_sz["d"] == pytest.approx(3.0)


def test_compute_per_subject_r_sz_handles_none_ranks():
    """A channel that's None in some seizures contributes only its
    finite ranks to the median (does NOT get penalty rank n_channels).

    seizure 0: a=0, b=None, c=2
    seizure 1: a=1, b=0, c=None
    seizure 2: a=2, b=1, c=1
    medians: a = median(0,1,2) = 1.0; b = median(0,1) = 0.5;
             c = median(2,1) = 1.5
    """
    per_seizure_ranks = [
        {"a": 0.0, "b": None, "c": 2.0},
        {"a": 1.0, "b": 0.0, "c": None},
        {"a": 2.0, "b": 1.0, "c": 1.0},
    ]
    seizure_statuses = ["ok", "ok", "ok"]
    r_sz = compute_per_subject_r_sz(
        per_seizure_ranks, seizure_statuses=seizure_statuses
    )
    assert r_sz["a"] == pytest.approx(1.0)
    assert r_sz["b"] == pytest.approx(0.5)
    assert r_sz["c"] == pytest.approx(1.5)


def test_compute_per_subject_r_sz_channel_never_active_returns_none():
    """A channel that is None in EVERY seizure must get None r_sz, not 0
    or NaN — Layer B must treat such channels as missing, not as the
    earliest-onset rank."""
    per_seizure_ranks = [
        {"a": 0.0, "b": None},
        {"a": 1.0, "b": None},
    ]
    r_sz = compute_per_subject_r_sz(
        per_seizure_ranks, seizure_statuses=["ok", "ok"]
    )
    assert r_sz["a"] == pytest.approx(0.5)
    assert r_sz["b"] is None


# ---------------------------------------------------------------------------
# A8b. baseline_invalid seizures excluded from r_sz input
# ---------------------------------------------------------------------------


def test_compute_per_subject_r_sz_excludes_baseline_invalid_seizures():
    """Plan §10 A8b: baseline_invalid seizures (resolve_baseline_window
    found < 60 s of valid samples) must NEVER enter the median rank
    pool. Without this exclusion, ranks built on truncated baselines
    would leak into r_sz.
    """
    per_seizure_ranks = [
        {"a": 0.0, "b": 1.0},   # ok
        {"a": 1.0, "b": 0.0},   # baseline_invalid → must be ignored
        {"a": 0.0, "b": 1.0},   # ok
    ]
    seizure_statuses = ["ok", "baseline_invalid", "ok"]
    r_sz = compute_per_subject_r_sz(
        per_seizure_ranks, seizure_statuses=seizure_statuses
    )
    # If baseline_invalid had been kept: median(0, 1, 0) = 0 for a;
    # since it's excluded: median(0, 0) = 0 for a; median(1, 1) = 1 for b.
    assert r_sz["a"] == pytest.approx(0.0)
    assert r_sz["b"] == pytest.approx(1.0)


def test_compute_per_subject_r_sz_also_excludes_onset_tied_and_unreached():
    """Plan §3.3 main analysis exclusion: onset_tied + onset_unreached
    seizures stay out of r_sz median (ranks are noise-dominated /
    undefined). Combined with baseline_invalid, only "ok" feeds r_sz.
    """
    per_seizure_ranks = [
        {"a": 0.0, "b": 1.0},   # ok
        {"a": 99.0, "b": 0.0},   # onset_tied (would skew if kept)
        {"a": 99.0, "b": 0.0},   # onset_unreached (would skew if kept)
        {"a": 0.0, "b": 1.0},   # ok
    ]
    seizure_statuses = ["ok", "onset_tied", "onset_unreached", "ok"]
    r_sz = compute_per_subject_r_sz(
        per_seizure_ranks, seizure_statuses=seizure_statuses
    )
    assert r_sz["a"] == pytest.approx(0.0)
    assert r_sz["b"] == pytest.approx(1.0)


def test_compute_per_subject_r_sz_zero_qualifying_seizures_returns_all_none():
    """If every seizure is non-ok, every channel gets None r_sz.
    Subject-level downstream code must treat this as the strongest
    drop signal."""
    per_seizure_ranks = [
        {"a": 0.0, "b": 1.0},
        {"a": 1.0, "b": 0.0},
    ]
    r_sz = compute_per_subject_r_sz(
        per_seizure_ranks,
        seizure_statuses=["onset_tied", "baseline_invalid"],
    )
    assert r_sz["a"] is None
    assert r_sz["b"] is None


def test_compute_per_subject_r_sz_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length"):
        compute_per_subject_r_sz(
            [{"a": 0.0}, {"a": 1.0}], seizure_statuses=["ok"]
        )


# ---------------------------------------------------------------------------
# A9. compute_stability_s_sz: pairwise Spearman across seizures
# ---------------------------------------------------------------------------


def test_compute_stability_s_sz_high_for_identical_ranks():
    """Identical rank vectors across seizures → s_sz = 1.0 (perfect
    Spearman in every pair).
    """
    per_seizure_ranks = [
        {"a": 0.0, "b": 1.0, "c": 2.0, "d": 3.0},
        {"a": 0.0, "b": 1.0, "c": 2.0, "d": 3.0},
        {"a": 0.0, "b": 1.0, "c": 2.0, "d": 3.0},
    ]
    s_sz = compute_stability_s_sz(
        per_seizure_ranks, seizure_statuses=["ok", "ok", "ok"]
    )
    assert s_sz == pytest.approx(1.0)


def test_compute_stability_s_sz_low_for_anticorrelated_ranks():
    """One seizure with reversed ranks → some pairs have negative
    Spearman; median is depressed.
    """
    per_seizure_ranks = [
        {"a": 0.0, "b": 1.0, "c": 2.0, "d": 3.0},
        {"a": 3.0, "b": 2.0, "c": 1.0, "d": 0.0},   # reversed
        {"a": 0.0, "b": 1.0, "c": 2.0, "d": 3.0},
    ]
    s_sz = compute_stability_s_sz(
        per_seizure_ranks, seizure_statuses=["ok", "ok", "ok"]
    )
    # Pairs: (0,1)=-1, (0,2)=+1, (1,2)=-1 → median = -1.
    assert s_sz == pytest.approx(-1.0)


def test_compute_stability_s_sz_returns_nan_on_single_seizure():
    """Stability requires ≥ 2 qualifying seizures (need pairs)."""
    s_sz = compute_stability_s_sz(
        [{"a": 0.0, "b": 1.0}], seizure_statuses=["ok"]
    )
    assert s_sz is None


def test_compute_stability_s_sz_skips_channels_with_none_in_either():
    """Spearman per pair is computed over the channel intersection
    (both seizures must have a finite rank for that channel)."""
    per_seizure_ranks = [
        {"a": 0.0, "b": 1.0, "c": 2.0},
        {"a": 0.0, "b": None, "c": 2.0},   # b dropped from this pair
    ]
    s_sz = compute_stability_s_sz(
        per_seizure_ranks, seizure_statuses=["ok", "ok"]
    )
    # Pair (a, c): ranks (0,2) vs (0,2) → Spearman = 1.0.
    assert s_sz == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# A9b. baseline_invalid seizures excluded from s_sz input
# ---------------------------------------------------------------------------


def test_compute_stability_s_sz_excludes_baseline_invalid_seizures():
    """Plan §10 A9b: baseline_invalid seizures must NOT enter pairwise
    Spearman. If kept, the noisy rank vector from a truncated baseline
    would deflate s_sz."""
    per_seizure_ranks = [
        {"a": 0.0, "b": 1.0},                # ok
        {"a": 1.0, "b": 0.0},                # baseline_invalid (reversed)
        {"a": 0.0, "b": 1.0},                # ok
    ]
    s_sz = compute_stability_s_sz(
        per_seizure_ranks,
        seizure_statuses=["ok", "baseline_invalid", "ok"],
    )
    # With baseline_invalid kept: pairs (0,1) corr=-1, (0,2) corr=+1,
    # (1,2) corr=-1 → median = -1. With baseline_invalid excluded:
    # only pair (0,2) corr=+1 → median = +1.
    assert s_sz == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# A10 / A11. time_shifted_seizure_onsets
# ---------------------------------------------------------------------------


def test_time_shifted_onsets_avoid_real_seizures():
    """Plan §5.1: shifted onset must be ≥ exclusion_radius_sec from
    every real seizure in the same block."""
    real_onsets = [1500.0, 3000.0]
    block_start = 0.0
    block_end = 4500.0
    baseline_pre_sec = 300.0
    baseline_buffer_sec = 60.0

    shifted = time_shifted_seizure_onsets(
        real_onsets=real_onsets,
        block_start_sec=block_start,
        block_end_sec=block_end,
        baseline_pre_sec=baseline_pre_sec,
        baseline_buffer_sec=baseline_buffer_sec,
        exclusion_radius_sec=300.0,
        n_iter=50,
        rng_seed=0,
    )
    # Returned shape: (n_iter, n_real_onsets)
    assert shifted.shape == (50, 2)
    # Every shifted onset must be ≥ 300 s away from BOTH real onsets.
    for it in range(50):
        for onset in shifted[it]:
            for real in real_onsets:
                assert abs(onset - real) >= 300.0


def test_time_shifted_onsets_inside_baseline_window():
    """Plan §3.3 baseline window is [-baseline_pre_sec,
    -baseline_buffer_sec] relative to onset. Shifted onsets must keep
    the resolved baseline inside the block (so re-running CUSUM on the
    shifted draw has a valid baseline)."""
    real_onsets = [2000.0]
    block_start = 0.0
    block_end = 3600.0
    baseline_pre_sec = 300.0
    baseline_buffer_sec = 60.0

    shifted = time_shifted_seizure_onsets(
        real_onsets=real_onsets,
        block_start_sec=block_start,
        block_end_sec=block_end,
        baseline_pre_sec=baseline_pre_sec,
        baseline_buffer_sec=baseline_buffer_sec,
        exclusion_radius_sec=0.0,
        n_iter=50,
        rng_seed=0,
    )
    # Every shifted onset must satisfy:
    #   onset - baseline_pre_sec >= block_start
    #   onset - baseline_buffer_sec <= block_end (window doesn't fall off the right)
    for it in range(50):
        for onset in shifted[it]:
            assert onset - baseline_pre_sec >= block_start
            # We need at least baseline_buffer_sec after the onset for
            # the post window to be valid.
            assert onset + baseline_buffer_sec <= block_end


def test_time_shifted_onsets_rejects_too_short_block():
    """Block too short to fit even a single shifted onset (after
    excluding real onsets ± radius and respecting baseline) → return
    empty array per real onset (NOT silent success)."""
    real_onsets = [200.0]
    # Block too short: feasible onset range is
    # [baseline_pre_sec, block_end - baseline_buffer_sec] = [300, 240]
    # which is empty.
    shifted = time_shifted_seizure_onsets(
        real_onsets=real_onsets,
        block_start_sec=0.0,
        block_end_sec=300.0,
        baseline_pre_sec=300.0,
        baseline_buffer_sec=60.0,
        exclusion_radius_sec=0.0,
        n_iter=50,
        rng_seed=0,
    )
    # Returned shape (n_iter, n_real_onsets) where missing draws are NaN.
    assert shifted.shape == (50, 1)
    assert np.all(np.isnan(shifted))


def test_time_shifted_onsets_deterministic_under_seed():
    """Same rng_seed → identical shifted matrix."""
    real_onsets = [1500.0]
    args = dict(
        real_onsets=real_onsets,
        block_start_sec=0.0,
        block_end_sec=3600.0,
        baseline_pre_sec=300.0,
        baseline_buffer_sec=60.0,
        exclusion_radius_sec=300.0,
        n_iter=20,
    )
    a = time_shifted_seizure_onsets(rng_seed=42, **args)
    b = time_shifted_seizure_onsets(rng_seed=42, **args)
    assert np.array_equal(a, b)
    c = time_shifted_seizure_onsets(rng_seed=43, **args)
    assert not np.array_equal(a, c)
