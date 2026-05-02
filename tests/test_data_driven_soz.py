"""TDD tests for PR-T3-1 ``src/data_driven_soz.py``.

- Step 1 covers M1 (HFO-onset rate) three variants + ranking +
  aggregation: T1–T10 (plus follow-up T6b/T6c for the ``rank_last``
  contract + T10b for missing-seizure rank).
- Step 2 covers M2 (ER log-ratio) + Nyquist / filter padding guards
  + per-channel eps: T11–T18.

See ``docs/archive/topic3/pr_t3_1_data_driven_soz_audit_plan_2026-04-30.md``
§10 for the full TDD list.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.data_driven_soz import (
    FilterPaddingError,
    NyquistGuardError,
    PerChannelEps,
    _bandpass_power,
    aggregate_consensus,
    aggregate_median_rank,
    annotate_clinical_soz,
    check_channel_schema_consistency,
    compute_er_logratio,
    compute_hfo_onset_metrics,
    estimate_per_channel_eps,
    matched_clinical_contacts,
    rank_top_k_per_seizure,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _events_with_n(n_pre: int, n_post: int, t_s: float, w_pre: float, w_post: float):
    """Return absolute timestamps placing ``n_pre`` events strictly inside
    ``[t_s - w_pre, t_s)`` and ``n_post`` events strictly inside
    ``(t_s, t_s + w_post]``.
    """
    pre = np.linspace(t_s - w_pre + 0.1, t_s - 0.1, num=n_pre) if n_pre > 0 else np.array([])
    post = np.linspace(t_s + 0.1, t_s + w_post - 0.1, num=n_post) if n_post > 0 else np.array([])
    return np.concatenate([pre, post])


# ---------------------------------------------------------------------------
# T1 — M1_raw simple arithmetic
# ---------------------------------------------------------------------------


def test_t1_m1_raw_arithmetic():
    t_s = 1000.0
    w_pre, w_post = 30.0, 10.0
    # post = 5 events / 10s, pre = 1 event / 30s
    events = {"chA": _events_with_n(1, 5, t_s, w_pre, w_post)}
    out = compute_hfo_onset_metrics(events, t_s, w_pre=w_pre, w_post=w_post)
    rec = out["chA"]
    # rate_post = 0.5 ; rate_pre = 1/30 ≈ 0.0333
    np.testing.assert_allclose(rec["M1_raw"], 0.5 - 1.0 / 30.0, rtol=1e-9)
    np.testing.assert_allclose(rec["rate_post"], 0.5, rtol=1e-9)
    np.testing.assert_allclose(rec["rate_pre"], 1.0 / 30.0, rtol=1e-9)


# ---------------------------------------------------------------------------
# T2 — M1_log formula with W_post / W_pre correction
# ---------------------------------------------------------------------------


def test_t2_m1_log_formula():
    t_s = 1000.0
    w_pre, w_post = 30.0, 10.0
    # post = 5, pre = 1 → log(6) - log(2) - log(10/30) = 2*log(3) ≈ 2.1972
    events = {"chA": _events_with_n(1, 5, t_s, w_pre, w_post)}
    out = compute_hfo_onset_metrics(events, t_s, w_pre=w_pre, w_post=w_post)
    expected = math.log(6.0) - math.log(2.0) - math.log(10.0 / 30.0)
    np.testing.assert_allclose(out["chA"]["M1_log"], expected, rtol=1e-9)
    np.testing.assert_allclose(out["chA"]["M1_log"], 2.0 * math.log(3.0), rtol=1e-9)


# ---------------------------------------------------------------------------
# T3 — M1_pois Poisson z arithmetic
# ---------------------------------------------------------------------------


def test_t3_m1_pois_arithmetic():
    t_s = 1000.0
    w_pre, w_post = 30.0, 10.0
    # post=5, pre=1, μ_pre = (1/30)*10 = 1/3
    # M1_pois = (5 - 1/3) / sqrt(1/3 + 1) ≈ 4.041
    events = {"chA": _events_with_n(1, 5, t_s, w_pre, w_post)}
    out = compute_hfo_onset_metrics(events, t_s, w_pre=w_pre, w_post=w_post)
    mu_pre = (1.0 / 30.0) * 10.0
    expected = (5 - mu_pre) / math.sqrt(mu_pre + 1.0)
    np.testing.assert_allclose(out["chA"]["M1_pois"], expected, rtol=1e-9)
    np.testing.assert_allclose(out["chA"]["M1_pois"], 4.041, atol=1e-2)


# ---------------------------------------------------------------------------
# T4 — channel with zero events → all three variants 0
# ---------------------------------------------------------------------------


def test_t4_no_events_all_zero():
    t_s = 1000.0
    events = {"silent": np.array([])}
    out = compute_hfo_onset_metrics(events, t_s, w_pre=30.0, w_post=10.0)
    rec = out["silent"]
    np.testing.assert_allclose(rec["M1_raw"], 0.0, atol=1e-12)
    # M1_log: log(0+1) - log(0+1) - log(10/30) = -log(1/3) = log(3)
    # The plan T4 states "全无 events → 三 variant 全 0".
    # That requires M1_log = 0 when n_pre = n_post = 0. The cleanest way is
    # to subtract the rate-correction term only when there are events on the
    # channel; or to set all three to 0 explicitly when both windows are
    # empty. The implementation chooses the explicit short-circuit per the
    # plan §3.3.
    np.testing.assert_allclose(rec["M1_log"], 0.0, atol=1e-12)
    # M1_pois: (0 - 0) / sqrt(0 + 1) = 0
    np.testing.assert_allclose(rec["M1_pois"], 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# T5 — rank_top_k 5 channels → top 3 deterministic
# ---------------------------------------------------------------------------


def test_t5_rank_top_k_basic():
    scores = {"a": 1.0, "b": 5.0, "c": 3.0, "d": 4.0, "e": 2.0}
    top3 = rank_top_k_per_seizure(scores, k=3)
    assert top3 == ["b", "d", "c"]


# ---------------------------------------------------------------------------
# T6 — NaN channels routed to bottom
# ---------------------------------------------------------------------------


def test_t6_rank_top_k_nan_bottom():
    scores = {"a": 1.0, "b": float("nan"), "c": 3.0, "d": 4.0, "e": float("nan")}
    top3 = rank_top_k_per_seizure(scores, k=3)
    # Non-NaN sorted: d, c, a; NaN channels b/e never enter top3.
    assert top3 == ["d", "c", "a"]


def test_t6b_rank_top_k_nan_tail_when_k_exceeds_finite_count():
    """Plan §3.5 ``rank_last`` contract: NaN scores rank LAST, not dropped.

    If ``k`` exceeds the number of finite-score channels, the NaN
    channels must fill the tail in alphabetical order — otherwise the
    size-matched primary k = ``|clinical_matched|`` (plan §3.6) would
    silently truncate when many channels have zero baseline rate.
    """
    scores = {"a": 1.0, "b": float("nan"), "c": 3.0, "d": 4.0, "e": float("nan")}
    top5 = rank_top_k_per_seizure(scores, k=5)
    # Finite descending: d (4), c (3), a (1); NaN tail alphabetical: b, e
    assert top5 == ["d", "c", "a", "b", "e"]


def test_t6c_rank_top_k_only_nan_returns_alphabetical_tail():
    scores = {"zeta": float("nan"), "alpha": float("nan"), "delta": float("nan")}
    top2 = rank_top_k_per_seizure(scores, k=2)
    assert top2 == ["alpha", "delta"]


# ---------------------------------------------------------------------------
# T7 — tie-break deterministic by ascending channel name
# ---------------------------------------------------------------------------


def test_t7_rank_top_k_tie_breaks_by_name():
    # All scores tied → top 3 must be alphabetically smallest 3
    scores = {"zeta": 1.0, "alpha": 1.0, "gamma": 1.0, "beta": 1.0, "delta": 1.0}
    top3 = rank_top_k_per_seizure(scores, k=3)
    assert top3 == ["alpha", "beta", "delta"]


# ---------------------------------------------------------------------------
# T8 — aggregate_consensus 50% threshold positive case
# ---------------------------------------------------------------------------


def test_t8_aggregate_consensus_positive():
    # 4 seizures, channel A appears in 3 (75% ≥ 50%) → IN
    per_seizure_topk = [
        ["A", "B", "C"],
        ["A", "B", "D"],
        ["A", "E", "F"],
        ["B", "G", "H"],
    ]
    consensus = aggregate_consensus(per_seizure_topk, min_seizure_fraction=0.5)
    assert "A" in consensus
    assert "B" in consensus  # B in 3/4 too


# ---------------------------------------------------------------------------
# T9 — aggregate_consensus 50% threshold negative case
# ---------------------------------------------------------------------------


def test_t9_aggregate_consensus_negative():
    # 4 seizures, channel A in 1 (25% < 50%) → OUT
    per_seizure_topk = [
        ["A", "B", "C"],
        ["X", "Y", "Z"],
        ["P", "Q", "R"],
        ["M", "N", "O"],
    ]
    consensus = aggregate_consensus(per_seizure_topk, min_seizure_fraction=0.5)
    assert "A" not in consensus


# ---------------------------------------------------------------------------
# T10 — aggregate_median_rank with median rank + missing → bottom rank
# ---------------------------------------------------------------------------


def test_t10_aggregate_median_rank():
    # 4 seizures × 5 channels. A has rank=2 in all 4 → median 2.
    # B has rank 1, 1, 5, 5 → median 3. C consistently rank 3 → median 3.
    # D and E rotate at the bottom.
    per_seizure_ranks = [
        {"A": 2, "B": 1, "C": 3, "D": 4, "E": 5},
        {"A": 2, "B": 1, "C": 3, "D": 4, "E": 5},
        {"A": 2, "B": 5, "C": 3, "D": 1, "E": 4},
        {"A": 2, "B": 5, "C": 3, "D": 1, "E": 4},
    ]
    top3 = aggregate_median_rank(per_seizure_ranks, k=3)
    # Medians: A=2, B=3, C=3, D=2.5, E=4.5 → smallest 3: {A=2, D=2.5, then B=3 or C=3}
    # Tie between B & C resolved by alphabetical order → B
    assert "A" in top3
    assert "D" in top3
    assert "B" in top3
    assert top3 == {"A", "B", "D"}


# ---------------------------------------------------------------------------
# Audit helpers (Step 0 hardening)
# ---------------------------------------------------------------------------


def test_matched_clinical_contacts_bipolar_partial_overlap():
    """Plan §3.2: matched contacts are clinical SOZ entries that touch
    at least one analysis channel. Contacts with no matching pair are
    reported via the unmatched complement.
    """
    analysis = ["A1-A2", "A2-A3", "B1-B2"]
    clinical = ["A1", "A3", "C1"]  # C1 not present anywhere
    matched = matched_clinical_contacts(analysis, clinical)
    assert matched == {"A1", "A3"}


def test_matched_clinical_contacts_normalizes_eeg_prefix():
    """``_normalize_channel_name`` strips ``EEG `` / ``EEG_`` and
    upper-cases. Reusing it for the unmatched stat keeps Step 0 audit
    aligned with the canonical matcher (plan §3.2 hard requirement).
    """
    analysis = ["EEG A1-EEG A2", "eeg_a3-eeg_a4"]
    clinical = ["A1", "a4"]
    matched = matched_clinical_contacts(analysis, clinical)
    assert matched == {"A1", "A4"}


def test_matched_clinical_contacts_car_channel():
    """CAR / monopolar analysis channels: single contact per channel."""
    analysis = ["GA1", "GA2", "GA3"]
    clinical = ["GA2"]
    matched = matched_clinical_contacts(analysis, clinical)
    assert matched == {"GA2"}


def test_annotate_clinical_soz_unknown_for_malformed_bipolar():
    """Plan §3.2: ``X-`` (empty side) → unknown."""
    analysis = ["A1-A2", "B1-", "-C2", "A1"]
    clinical = ["A1"]
    labels = annotate_clinical_soz(analysis, clinical)
    assert labels["A1-A2"] == "soz"
    assert labels["B1-"] == "unknown"
    assert labels["-C2"] == "unknown"
    assert labels["A1"] == "soz"


def test_check_channel_schema_consistency_consistent():
    blocks = [["A1", "A2", "A3"], ["A1", "A2", "A3"]]
    res = check_channel_schema_consistency(blocks)
    assert res["all_consistent"] is True
    assert res["mismatched_block_indices"] == []
    assert res["n_channels_min"] == 3
    assert res["n_channels_max"] == 3


def test_check_channel_schema_consistency_order_mismatch():
    """Channel ordering mismatch is a real problem because every other
    artifact in the pipeline indexes by position. Must flag, not silent."""
    blocks = [["A1", "A2", "A3"], ["A2", "A1", "A3"]]
    res = check_channel_schema_consistency(blocks)
    assert res["all_consistent"] is False
    assert res["mismatched_block_indices"] == [1]


def test_check_channel_schema_consistency_partial_blocks():
    blocks = [["A1", "A2", "A3"], ["A1", "A2"]]
    res = check_channel_schema_consistency(blocks)
    assert res["all_consistent"] is False
    assert res["n_channels_min"] == 2
    assert res["n_channels_max"] == 3
    assert res["mismatched_block_indices"] == [1]


def test_check_channel_schema_consistency_empty():
    res = check_channel_schema_consistency([])
    assert res["all_consistent"] is True
    assert res["mismatched_block_indices"] == []
    assert res["n_channels_min"] == 0
    assert res["n_channels_max"] == 0


def test_t10b_aggregate_median_rank_missing_seizure_goes_to_bottom():
    # 4 seizures, channel set varies. n_channels = 4.
    # X is ranked 1 in only 1 seizure, missing in others → median > 1.
    per_seizure_ranks = [
        {"X": 1, "Y": 2, "Z": 3, "W": 4},
        {"Y": 1, "Z": 2, "W": 3},  # X missing → counted as rank=4
        {"Y": 1, "Z": 2, "W": 3},  # X missing
        {"Y": 1, "Z": 2, "W": 3},  # X missing
    ]
    top3 = aggregate_median_rank(per_seizure_ranks, k=3)
    # Y median 1.5, Z median 2, W median 3, X median ≈ 4 → top3 = Y/Z/W
    assert top3 == {"Y", "Z", "W"}


# ===========================================================================
# Step 2 — M2 (ER log-ratio + Nyquist + per-channel eps)
# ===========================================================================


W_PRE = 30.0
W_POST = 10.0
EDGE_BUFFER = 2.0


def _make_const_loader(signal_array: np.ndarray, sfreq: float):
    """Return a ``signal_loader(t_start, t_end, channels)`` that always
    returns the same ``(signal_array, sfreq)`` tuple regardless of the
    requested time range. Useful for tests that pre-compute a windowed
    signal aligned to ``t_start = seizure_onset - W_PRE - EDGE_BUFFER``.
    """

    def loader(t_start, t_end, channels):  # noqa: ARG001
        return signal_array, sfreq

    return loader


def _build_pre_post_signal(
    sfreq: float,
    n_channels: int,
    *,
    rng_seed: int,
    pre_factory=None,
    burst_window=None,
    burst_factory=None,
    post_factory=None,
) -> np.ndarray:
    """Synthesize a (T, n_channels) signal that spans
    ``[seizure_onset - W_PRE - EDGE_BUFFER, seizure_onset + W_POST + EDGE_BUFFER]``
    so the pre window is samples [0 : W_PRE * sfreq], the post window is
    samples [(W_PRE + 2*EDGE_BUFFER) * sfreq : ...], and the edge buffer
    (the W_PRE..W_PRE+2*EDGE_BUFFER samples) sits in the middle.

    Each *factory* receives ``(rng, n_samples_in_window)`` and returns a
    ``(n_samples, n_channels)`` block to overwrite that window. Anything
    not overwritten stays at the rng-generated baseline (white noise).
    """
    rng = np.random.default_rng(rng_seed)
    n_total = int(round((W_PRE + 2 * EDGE_BUFFER + W_POST) * sfreq))
    sig = rng.standard_normal((n_total, n_channels))

    pre_end = int(round(W_PRE * sfreq))
    edge_end = int(round((W_PRE + 2 * EDGE_BUFFER) * sfreq))

    if pre_factory is not None:
        sig[:pre_end, :] = pre_factory(rng, pre_end)
    if post_factory is not None:
        sig[edge_end:, :] = post_factory(rng, n_total - edge_end)
    if burst_window is not None and burst_factory is not None:
        b_start_t, b_end_t = burst_window
        # Convert from "seconds since t_start" to sample indices.
        b_start = int(round(b_start_t * sfreq))
        b_end = int(round(b_end_t * sfreq))
        sig[b_start:b_end, :] = burst_factory(rng, b_end - b_start)

    return sig


# ---------------------------------------------------------------------------
# T11 — _bandpass_power: in-band sine has high power, out-of-band ~0
# ---------------------------------------------------------------------------


def test_t11_bandpass_power_in_band_vs_out_of_band():
    """Plan §3.4: filter is Butter order 4 + filtfilt zero-phase. An
    in-band 150 Hz sine is preserved (mean power ~ 0.5 for unit
    amplitude); an out-of-band 5 Hz sine is suppressed (mean power
    < 0.05).
    """
    sfreq = 1000.0
    n = int(5 * sfreq)
    t = np.arange(n) / sfreq
    in_band = np.sin(2 * np.pi * 150 * t)
    out_band = np.sin(2 * np.pi * 5 * t)
    signal = np.column_stack([in_band, out_band])

    inst_power = _bandpass_power(signal, sfreq, (80.0, 250.0))

    p_in = float(np.mean(inst_power[:, 0]))
    p_out = float(np.mean(inst_power[:, 1]))
    assert p_in > 0.4, f"in-band power={p_in:.3f}, expected > 0.4"
    assert p_out < 0.05, f"out-of-band power={p_out:.3f}, expected < 0.05"


# ---------------------------------------------------------------------------
# T12 — compute_er_logratio: post burst → logratio > log(10) ≈ 2.3
# ---------------------------------------------------------------------------


def test_t12_compute_er_logratio_post_burst_drives_high_logratio():
    sfreq = 1000.0
    seizure_onset = 100.0
    eps = 1e-12

    def post_burst(rng, n_samples):
        t = np.arange(n_samples) / sfreq
        burst = 5.0 * np.sin(2 * np.pi * 150 * t)
        return burst[:, None]

    signal = _build_pre_post_signal(
        sfreq=sfreq,
        n_channels=1,
        rng_seed=0,
        post_factory=post_burst,
    )
    out = compute_er_logratio(
        signal_loader=_make_const_loader(signal, sfreq),
        channels=["ch0"],
        seizure_onset=seizure_onset,
        eps_per_channel={"ch0": eps},
    )
    assert out["ch0"] > math.log(10), (
        f"expected logratio > {math.log(10):.3f}, got {out['ch0']:.3f}"
    )


# ---------------------------------------------------------------------------
# T13 — compute_er_logratio: same-distribution noise → logratio ≈ 0
# ---------------------------------------------------------------------------


def test_t13_compute_er_logratio_same_noise_returns_near_zero():
    sfreq = 1000.0
    seizure_onset = 100.0
    eps = 1e-9

    signal = _build_pre_post_signal(
        sfreq=sfreq,
        n_channels=1,
        rng_seed=42,
    )
    out = compute_er_logratio(
        signal_loader=_make_const_loader(signal, sfreq),
        channels=["ch0"],
        seizure_onset=seizure_onset,
        eps_per_channel={"ch0": eps},
    )
    assert abs(out["ch0"]) < 0.5, (
        f"expected |logratio| < 0.5 for same-noise pre/post, got {out['ch0']:.3f}"
    )


# ---------------------------------------------------------------------------
# T14 — edge buffer honored: burst inside ±edge_buffer doesn't enter windows
# ---------------------------------------------------------------------------


def test_t14_compute_er_logratio_edge_buffer_excludes_burst():
    """Plan §3.4: edge_buffer keeps onset filter ringing out of pre/post
    windows. A 1-second burst centered on onset must NOT raise either
    power estimate above the surrounding noise floor.
    """
    sfreq = 1000.0
    seizure_onset = 100.0
    eps = 1e-9

    # Burst lives at [t_s - 0.5, t_s + 0.5] = [99.5, 100.5] sec, fully
    # inside the 4-sec edge buffer band. In "seconds since t_start"
    # (where t_start = 100 - 30 - 2 = 68 sec) this is [31.5, 32.5].
    def burst_factory(rng, n_samples):  # noqa: ARG001
        t = np.arange(n_samples) / sfreq
        return (50.0 * np.sin(2 * np.pi * 150 * t))[:, None]

    signal = _build_pre_post_signal(
        sfreq=sfreq,
        n_channels=1,
        rng_seed=7,
        burst_window=(31.5, 32.5),
        burst_factory=burst_factory,
    )
    out = compute_er_logratio(
        signal_loader=_make_const_loader(signal, sfreq),
        channels=["ch0"],
        seizure_onset=seizure_onset,
        eps_per_channel={"ch0": eps},
    )
    # Without edge buffer the burst would dominate the post window and
    # drive logratio above 5. With edge buffer honored the result must
    # stay near baseline (similar to the same-noise T13 case).
    assert abs(out["ch0"]) < 1.0, (
        f"edge buffer not honored: logratio={out['ch0']:.3f}"
    )


# ---------------------------------------------------------------------------
# T15 — power_pre[ch] = 0 → eps_ch fallback, finite result, no inf/NaN
# ---------------------------------------------------------------------------


def test_t15_compute_er_logratio_zero_pre_falls_back_to_eps():
    sfreq = 1000.0
    seizure_onset = 100.0
    eps = 1e-9

    def zero_pre(rng, n_samples):  # noqa: ARG001
        return np.zeros((n_samples, 1))

    def post_burst(rng, n_samples):  # noqa: ARG001
        t = np.arange(n_samples) / sfreq
        return (1.0 * np.sin(2 * np.pi * 150 * t))[:, None]

    signal = _build_pre_post_signal(
        sfreq=sfreq,
        n_channels=1,
        rng_seed=0,
        pre_factory=zero_pre,
        post_factory=post_burst,
    )
    out = compute_er_logratio(
        signal_loader=_make_const_loader(signal, sfreq),
        channels=["ch0"],
        seizure_onset=seizure_onset,
        eps_per_channel={"ch0": eps},
    )
    val = out["ch0"]
    assert math.isfinite(val), f"logratio not finite: {val!r}"
    # power_post is ~0.5 (unit sine), so logratio = log((0.5 + eps)/eps)
    # ≈ log(0.5 / 1e-9) ≈ log(5e8) ≈ 20.
    assert val > 5.0, f"expected large positive logratio, got {val:.3f}"


# ---------------------------------------------------------------------------
# T16 — Nyquist guard: sfreq=512 with band=(80,250) → NyquistGuardError
# ---------------------------------------------------------------------------


def test_t16_compute_er_logratio_nyquist_guard():
    """sfreq/2 = 256 < 250 * 1.05 = 262.5 → must raise."""
    sfreq = 512.0
    seizure_onset = 100.0

    def loader(t_start, t_end, channels):  # noqa: ARG001
        n = int(round((t_end - t_start) * sfreq))
        return np.zeros((n, len(channels))), sfreq

    with pytest.raises(NyquistGuardError):
        compute_er_logratio(
            signal_loader=loader,
            channels=["ch0"],
            seizure_onset=seizure_onset,
            eps_per_channel={"ch0": 1e-12},
        )


# ---------------------------------------------------------------------------
# T17 — filter padding: signal too short → FilterPaddingError
# ---------------------------------------------------------------------------


def test_t17_bandpass_power_filter_padding_too_short():
    sfreq = 1000.0
    band = (80.0, 250.0)
    # Required padlen = max(default=15, int(1.5 * 1000 / 80) = 18) = 18.
    # 10 samples is below that → must raise.
    short_signal = np.zeros((10, 1))
    with pytest.raises(FilterPaddingError):
        _bandpass_power(short_signal, sfreq, band)


def test_t17b_compute_er_logratio_filter_padding_propagates():
    """A pathological loader returning 5 samples must surface
    FilterPaddingError, not silently corrupt the per-seizure metric."""
    sfreq = 1000.0

    def short_loader(t_start, t_end, channels):  # noqa: ARG001
        return np.zeros((5, len(channels))), sfreq

    with pytest.raises(FilterPaddingError):
        compute_er_logratio(
            signal_loader=short_loader,
            channels=["ch0"],
            seizure_onset=100.0,
            eps_per_channel={"ch0": 1e-12},
        )


# ---------------------------------------------------------------------------
# T18 — estimate_per_channel_eps: 1st percentile per channel, floor
# ---------------------------------------------------------------------------


def test_t18_estimate_per_channel_eps_basic_percentile():
    """Plan §3.4 contract: 1st percentile per channel, floored at
    ``floor``. The result is a ``PerChannelEps`` NamedTuple so the
    Step 3 runner can also see (a) which channels were floor-active
    (raw < floor) and (b) which channels are strictly M2-ineligible
    (every pre-power value == 0)."""
    rng = np.random.default_rng(0)
    n_seizures = 50
    n_channels = 3
    pre_matrix = np.empty((n_seizures, n_channels))
    pre_matrix[:, 0] = rng.uniform(1e-6, 1e-3, n_seizures)
    pre_matrix[:, 1] = rng.uniform(1.0, 2.0, n_seizures)
    pre_matrix[:, 2] = rng.uniform(0.0, 1e-12, n_seizures)

    result = estimate_per_channel_eps(pre_matrix, floor=1e-18)

    assert isinstance(result, PerChannelEps)
    assert result.eps.shape == (3,)
    assert result.m2_ineligible.shape == (3,)
    assert result.raw_percentile.shape == (3,)
    np.testing.assert_allclose(result.eps[0], np.percentile(pre_matrix[:, 0], 1))
    np.testing.assert_allclose(result.eps[1], np.percentile(pre_matrix[:, 1], 1))
    np.testing.assert_allclose(result.eps[2], np.percentile(pre_matrix[:, 2], 1))
    # None of these channels are strictly all-zero → none are
    # M2-ineligible (the strict drop criterion). channel 2 is
    # floor-active (raw 1st percentile may be ~ 0 but not strictly
    # all-zero), but that's not the same as ineligible.
    assert not result.m2_ineligible.any()


def test_t18b_estimate_per_channel_eps_all_zero_channel_marked_ineligible():
    """Plan §3.4: a channel whose pre-power is zero across every
    seizure must be flagged ``m2_ineligible=True`` so the per-subject
    runner can drop it. Floor still kicks in to keep the eps array
    finite for any callers that index without filtering, but the test
    must verify the **drop signal**, not just the floored value, so
    Step 3 can route the channel into ``m2_ineligible_channels`` per
    plan §3.4 ("如果某通道全 cohort 都 0，drop 该通道").
    """
    pre_matrix = np.zeros((10, 3))
    pre_matrix[:, 1] = 1.0          # healthy
    pre_matrix[5, 2] = 1e-15        # tiny but non-zero on one seizure
    result = estimate_per_channel_eps(pre_matrix, floor=1e-18)

    # Channel 0 (all-zero across cohort) is the only ineligible one.
    np.testing.assert_array_equal(result.m2_ineligible, [True, False, False])
    # Floor still applied for safety (eps must remain positive).
    assert result.eps[0] == 1e-18
    # Raw percentile should be zero for ch0; healthy values for ch1.
    assert result.raw_percentile[0] == 0.0
    assert result.raw_percentile[1] == 1.0


def test_t18c_estimate_per_channel_eps_custom_floor():
    """Floor honored when raw percentile falls below it. Channel is
    floor-active but not strictly ineligible (one seizure had a
    non-zero pre-power)."""
    pre_matrix = np.full((5, 1), 1e-20)  # below default floor
    pre_matrix[2, 0] = 1e-19             # one non-zero, still below custom floor
    result = estimate_per_channel_eps(pre_matrix, floor=1e-15)
    assert result.eps[0] == 1e-15
    # Not strictly all-zero, so not M2-ineligible by the drop rule.
    assert not result.m2_ineligible[0]


def test_t18d_estimate_per_channel_eps_zero_seizures_raises():
    """Defensive: estimating eps from an empty seizure list is a
    usage error in the runner pipeline, not a silent no-op."""
    with pytest.raises(ValueError, match="0 rows"):
        estimate_per_channel_eps(np.empty((0, 5)))


# ---------------------------------------------------------------------------
# Helper-level Nyquist safety + missing-eps strictness (Step 2 hardening)
# ---------------------------------------------------------------------------


def test_t16b_bandpass_power_nyquist_safety_margin():
    """Plan §3.4 5% safety margin must hold at the helper level too —
    Step 3 will pre-compute power_pre_matrix by calling _bandpass_power
    directly, and a soft-Nyquist subject (e.g. sfreq=510 with
    band=(80,250): nyq=255 > band_hi=250 but < band_hi*1.05=262.5)
    must NOT silently slip past the helper.
    """
    signal = np.zeros((50000, 1))
    with pytest.raises(NyquistGuardError):
        _bandpass_power(signal, sfreq=510.0, band=(80.0, 250.0))


def test_compute_er_logratio_missing_eps_raises_value_error():
    """compute_er_logratio must require an eps entry for every queried
    channel. Silently falling back to ``1e-18`` would let a Step 3 bug
    (forgetting to estimate eps for a freshly-added channel) produce
    plausible-looking science output. Plan §3.4: eps is per-channel
    contract input, not optional.
    """
    sfreq = 1000.0
    signal = _build_pre_post_signal(sfreq=sfreq, n_channels=2, rng_seed=0)
    with pytest.raises(ValueError, match="eps_per_channel"):
        compute_er_logratio(
            signal_loader=_make_const_loader(signal, sfreq),
            channels=["ch0", "ch1"],
            seizure_onset=100.0,
            eps_per_channel={"ch0": 1e-12},  # ch1 missing
        )


# ---------------------------------------------------------------------------
# Step 2 acceptance v2 — matcher consistency, consensus output size, and
# m2_ineligible enforcement helper. These three were flagged after the
# initial Step 2 commit and must hold before Step 3 builds on them.
# ---------------------------------------------------------------------------


def test_annotate_clinical_soz_dual_prefix_bipolar_matches_audit():
    """``annotate_clinical_soz`` must agree with ``matched_clinical_contacts``
    on dual-prefix bipolar names like ``EEG A1-EEG A2``. Otherwise audit
    rows can report ``n_clinical_unmatched=0`` (matched_clinical_contacts
    found ``A2``) while annotate_clinical_soz silently labels the channel
    ``non_soz`` because ``match_bipolar_soz`` only strips the leading
    prefix from the whole string. The two paths must use the same
    per-endpoint normalization.
    """
    analysis = ["EEG A1-EEG A2", "EEG B1-EEG B2"]
    clinical = ["A2"]
    matched = matched_clinical_contacts(analysis, clinical)
    assert "A2" in matched
    labels = annotate_clinical_soz(analysis, clinical)
    assert labels["EEG A1-EEG A2"] == "soz"
    assert labels["EEG B1-EEG B2"] == "non_soz"


def test_annotate_clinical_soz_lowercase_dual_prefix():
    """Same matcher contract as above but the bipolar uses ``eeg_`` and
    lowercase per-endpoint — covers the second normalize path through
    ``_normalize_channel_name`` (uppercase + ``EEG_`` strip).
    """
    analysis = ["eeg_a1-eeg_a2"]
    clinical = ["A1"]
    labels = annotate_clinical_soz(analysis, clinical)
    assert labels["eeg_a1-eeg_a2"] == "soz"


def test_aggregate_consensus_output_can_be_smaller_than_topk_size():
    """Plan §3.5 enrichment uses ``len(B)``, not the per-seizure ``k``.
    Demonstrate the consensus rule can return fewer than ``k`` channels:
    only ``A`` clears the 50% threshold even though every per-seizure
    list has ``k=3`` members.
    """
    per_seizure = [["A", "B", "C"], ["A", "D", "E"], ["A", "F", "G"]]
    out = aggregate_consensus(per_seizure, min_seizure_fraction=0.5)
    assert out == {"A"}
    assert len(out) == 1  # NOT 3 (k)


def test_aggregate_consensus_output_can_be_larger_than_topk_size():
    """The other direction: per-seizure top-3 lists can yield a larger
    consensus set when many channels are stable. Step 3/4 enrichment
    must use ``len(B)`` (= 4 here), NOT some fixed ``k``.
    """
    # k=3 per seizure, but channels stable enough to clear 50%:
    #   A,B,C appear 3/3; D appears 2/3 (clears 0.5).
    per_seizure = [["A", "B", "C"], ["A", "B", "D"], ["A", "C", "D"]]
    out = aggregate_consensus(per_seizure, min_seizure_fraction=0.5)
    assert out == {"A", "B", "C", "D"}
    assert len(out) == 4  # NOT 3 (k)


def test_select_m2_eligible_channels_drops_ineligible():
    """Step 3 must filter channels by ``PerChannelEps.m2_ineligible``
    BEFORE calling ``compute_er_logratio``. Otherwise the 1e-18 floor
    on a strictly-zero-pre-power channel produces ``log(P_post / 1e-18)
    ≈ +40`` and dominates the M2 ranking. The helper formalises that
    contract so Step 3 cannot accidentally feed an ineligible channel
    into the log-ratio.
    """
    from src.data_driven_soz import select_m2_eligible_channels

    eps_result = PerChannelEps(
        eps=np.array([1e-18, 1e-12, 1e-18]),
        m2_ineligible=np.array([True, False, False]),
        raw_percentile=np.array([0.0, 1e-12, 0.0]),
    )
    channel_index = {"chA": 0, "chB": 1, "chC": 2}
    eligible, dropped = select_m2_eligible_channels(
        ["chA", "chB", "chC"], eps_result, channel_index
    )
    assert eligible == ["chB", "chC"]
    assert dropped == ["chA"]


def test_select_m2_eligible_channels_preserves_input_order():
    """Eligible channel order must match the input order — Step 3 indexes
    downstream M2 outputs by this list, so silent re-ordering would
    corrupt the per-channel score map.
    """
    from src.data_driven_soz import select_m2_eligible_channels

    eps_result = PerChannelEps(
        eps=np.array([1e-12, 1e-18, 1e-12, 1e-18]),
        m2_ineligible=np.array([False, True, False, True]),
        raw_percentile=np.array([1e-12, 0.0, 1e-12, 0.0]),
    )
    channel_index = {"d": 0, "c": 1, "b": 2, "a": 3}
    eligible, dropped = select_m2_eligible_channels(
        ["d", "c", "b", "a"], eps_result, channel_index
    )
    assert eligible == ["d", "b"]
    assert dropped == ["c", "a"]


def test_select_m2_eligible_channels_unknown_channel_raises():
    """A channel name missing from ``channel_index`` is a Step 3 caller
    bug — fail loudly rather than silently dropping the channel."""
    from src.data_driven_soz import select_m2_eligible_channels

    eps_result = PerChannelEps(
        eps=np.array([1e-12]),
        m2_ineligible=np.array([False]),
        raw_percentile=np.array([1e-12]),
    )
    with pytest.raises(KeyError, match="not_in_index"):
        select_m2_eligible_channels(
            ["not_in_index"], eps_result, {"present": 0}
        )


# ---------------------------------------------------------------------------
# Step 3 — per-subject runner contract tests (TDD before implementation).
#
# Acceptance gates the user enumerated:
#   (a) seizure window block-boundary prefilter via inventory timestamps,
#       NOT by catching FilterPaddingError inside compute_er_logratio
#   (b) per-subject runner invokes select_m2_eligible_channels and writes
#       the dropped set to the JSON's "m2_ineligible_channels" field
#   (c) overlap enrichment uses len(B) (consensus may yield != k)
#   (d) per-subject JSON schema contains "m2_ineligible_channels"
#   (e) --cohort-overlap still raises NotImplementedError
# ---------------------------------------------------------------------------


def test_prefilter_seizures_by_block_window_keeps_centered_drops_edges():
    """Plan §3.4 says the M2 window is [t_s − W_pre − edge, t_s + W_post + edge].
    A seizure within ``W_pre + edge`` of the block start, or within
    ``W_post + edge`` of the block end, has no signal to load. The
    correct gate is a forward inventory check, NOT catching
    ``FilterPaddingError`` inside compute_er_logratio (which would only
    trigger after a wasted partial signal load and would conflate
    boundary failures with sfreq problems).
    """
    from src.data_driven_soz import prefilter_seizures_by_block_window

    block_windows = {"BLK0": (1000.0, 4600.0)}  # 1 hour block
    # W_pre + edge = 32 s → onsets must be > 1032; W_post + edge = 12 s → < 4588
    seizure_onsets = [
        1010.0,  # too close to block start (1010 - 32 < 1000) → drop
        1100.0,  # safely inside → keep
        2500.0,  # safely inside → keep
        4595.0,  # too close to block end (4595 + 12 > 4600) → drop
    ]
    seizure_block_ids = ["BLK0", "BLK0", "BLK0", "BLK0"]
    kept, dropped, reasons = prefilter_seizures_by_block_window(
        seizure_onsets,
        seizure_block_ids,
        block_windows,
        w_pre=30.0,
        w_post=10.0,
        edge_buffer=2.0,
    )
    assert kept == [1, 2]
    assert dropped == [0, 3]
    assert reasons[0].startswith("boundary_pre")
    assert reasons[3].startswith("boundary_post")


def test_prefilter_seizures_by_block_window_missing_block_dropped():
    """A seizure whose ``block_id`` is not in ``block_windows`` (block
    inventory missing for that recording) is dropped with a distinct
    reason — silently keeping it would feed an unbounded window into
    the loader."""
    from src.data_driven_soz import prefilter_seizures_by_block_window

    block_windows = {"BLK0": (1000.0, 4600.0)}
    kept, dropped, reasons = prefilter_seizures_by_block_window(
        seizure_onsets=[1100.0, 9000.0],
        seizure_block_ids=["BLK0", "BLK_MISSING"],
        block_windows=block_windows,
        w_pre=30.0,
        w_post=10.0,
        edge_buffer=2.0,
    )
    assert kept == [0]
    assert dropped == [1]
    assert reasons[1] == "missing_block"


def test_random_expected_jaccard_closed_form():
    """Random expected Jaccard for two uniform random subsets of
    [n_total] of sizes ``a_size`` and ``b_size``: the helper must use
    the analytical formula, not the per-seizure ``k`` (Step 3/4
    enrichment depends on this).
    """
    from src.data_driven_soz import random_expected_jaccard

    # |A|*|B|/N expected intersection / (|A|+|B|-|A|*|B|/N) expected union
    # |A|=10, |B|=10, N=100 → expected intersection 1, union 19, jaccard ≈ 1/19
    j = random_expected_jaccard(10, 10, 100)
    assert abs(j - 1.0 / 19.0) < 1e-6
    # Edge: empty B → jaccard 0
    assert random_expected_jaccard(5, 0, 100) == 0.0
    # Edge: B == N → jaccard = a/N
    assert random_expected_jaccard(7, 100, 100) == pytest.approx(7.0 / 100.0)


def test_compute_overlap_enrichment_uses_len_B_not_k():
    """Plan §3.7 enrichment = observed_intersection / random_expected_intersection
    where random_expected_intersection = |A| * |B| / n_total. ``|B|`` is
    the actual basket size (consensus rule may yield != k). The helper
    must NOT take ``k`` as an argument and must not bake it in.
    """
    from src.data_driven_soz import compute_overlap

    A = {"x", "y", "z"}              # |A| = 3
    B = {"x", "y"}                   # |B| = 2 (consensus may shrink)
    n_total = 30
    res = compute_overlap(A, B, n_total)
    # observed_intersection = 2
    assert res["observed_intersection"] == 2
    # expected = |A| * |B| / n_total = 3 * 2 / 30 = 0.2
    assert res["random_expected_intersection"] == pytest.approx(0.2)
    # enrichment = observed / max(expected, 0.5) = 2 / 0.5 = 4.0
    # (plan §3.7 floor: max(expected_intersection, 0.5) prevents inf)
    assert res["enrichment"] == pytest.approx(2.0 / 0.5)
    # If we ran the same A against a consensus of size 5 instead, the
    # expected intersection would be 0.5 — different number — same
    # helper must give the right answer.
    res5 = compute_overlap(A, set("xyzab"), n_total)
    assert res5["random_expected_intersection"] == pytest.approx(0.5)


def test_compute_overlap_returns_jaccard_precision_recall_f1():
    """Per plan §3.7, the helper returns the full overlap quartet."""
    from src.data_driven_soz import compute_overlap

    A = {"a", "b", "c", "d"}    # |A| = 4
    B = {"c", "d", "e"}         # |B| = 3, A ∩ B = {c, d}
    res = compute_overlap(A, B, 20)
    assert res["jaccard"] == pytest.approx(2.0 / 5.0)
    assert res["precision"] == pytest.approx(2.0 / 3.0)
    assert res["recall"] == pytest.approx(2.0 / 4.0)
    expected_f1 = 2 * (2 / 3) * (2 / 4) / ((2 / 3) + (2 / 4))
    assert res["f1"] == pytest.approx(expected_f1)


def test_per_subject_runner_drops_m2_ineligible_channels_from_ranking():
    """The orchestrator MUST call select_m2_eligible_channels before
    compute_er_logratio. A channel whose pre-power is strictly zero
    across every seizure must NOT appear in any M2 ranking and MUST
    appear in the JSON's ``m2_ineligible_channels`` list. This is the
    tactical guard against the 1e-18 floor producing log(P_post/1e-18) ≈
    +40 noise-floor rankings.
    """
    from src.data_driven_soz import compute_per_subject_audit

    sfreq = 1000.0
    # Use 100 s blocks so seizures at t=50 (relative) sit safely inside
    # the [w_pre+edge, block_len - w_post - edge] = [32, 88] safe band.
    block_len_sec = 100.0
    n_samples = int(block_len_sec * sfreq)

    def make_signal(rng_seed: int) -> np.ndarray:
        rng_local = np.random.default_rng(rng_seed)
        sig = rng_local.normal(0, 1, (n_samples, 3))
        sig[:, 2] = 0.0
        return sig

    block_signals = {
        "BLK0": make_signal(1),
        "BLK1": make_signal(2),
    }
    block_windows = {
        "BLK0": (0.0, block_len_sec),
        "BLK1": (200.0, 200.0 + block_len_sec),
    }

    def signal_loader(t_start, t_end, channels):
        for blk, (b0, b1) in block_windows.items():
            if t_start >= b0 and t_end <= b1:
                s0 = int(round((t_start - b0) * sfreq))
                s1 = int(round((t_end - b0) * sfreq))
                ch_idx = {"chA": 0, "chB": 1, "chC": 2}
                cols = [ch_idx[ch] for ch in channels]
                return block_signals[blk][s0:s1, cols], sfreq
        raise ValueError(f"window {t_start}-{t_end} doesn't fit any block")

    hfo_events = {
        "chA": np.array([45.0, 50.5, 55.0, 245.0, 250.5, 255.0]),
        "chB": np.array([46.0, 51.0, 56.0, 246.0, 251.0, 256.0]),
        "chC": np.array([47.0, 52.0, 57.0, 247.0, 252.0, 257.0]),
    }
    seizure_onsets = [50.0, 250.0]
    seizure_block_ids = ["BLK0", "BLK1"]

    res = compute_per_subject_audit(
        dataset="testset",
        subject="synth",
        seizure_onsets=seizure_onsets,
        seizure_block_ids=seizure_block_ids,
        block_windows=block_windows,
        hfo_event_times_per_channel=hfo_events,
        signal_loader=signal_loader,
        sfreq=sfreq,
        clinical_soz=["chA"],
        analysis_channels=["chA", "chB", "chC"],
        m2_eligible=True,
        null_n_iter=0,   # skip null surrogate for speed in this unit test
    )
    assert "m2_ineligible_channels" in res
    assert res["m2_ineligible_channels"] == ["chC"]
    # chC must NOT appear in any M2 ranking output.
    m2_results = res["results"]["M2_logratio"]
    for agg, by_k in m2_results.items():
        for k_label, ranking in by_k.items():
            assert "chC" not in ranking, (
                f"M2 ranking {agg}/{k_label} contained ineligible channel: {ranking}"
            )


def test_per_subject_runner_json_schema_contains_required_fields():
    """Plan §9 Step 3.3 locks the per-subject JSON schema. Verify the
    orchestrator emits all the required top-level fields, including
    ``m2_ineligible_channels`` (Step 2 acceptance v2 addition)."""
    from src.data_driven_soz import compute_per_subject_audit

    sfreq = 1000.0
    block_len_sec = 100.0
    n_samples = int(block_len_sec * sfreq)
    rng = np.random.default_rng(0)
    block_windows = {
        "BLK0": (0.0, block_len_sec),
        "BLK1": (200.0, 200.0 + block_len_sec),
    }
    block_signals = {
        b: rng.normal(0, 1, (n_samples, 2)) for b in block_windows
    }

    def signal_loader(t_start, t_end, channels):
        for blk, (b0, b1) in block_windows.items():
            if t_start >= b0 and t_end <= b1:
                s0 = int(round((t_start - b0) * sfreq))
                s1 = int(round((t_end - b0) * sfreq))
                ch_idx = {"chA": 0, "chB": 1}
                cols = [ch_idx[ch] for ch in channels]
                return block_signals[blk][s0:s1, cols], sfreq
        raise ValueError(f"window {t_start}-{t_end} doesn't fit any block")

    hfo_events = {
        "chA": np.array([45.0, 50.5, 55.0, 245.0, 250.5, 255.0]),
        "chB": np.array([46.0, 51.0, 56.0, 246.0, 251.0, 256.0]),
    }
    res = compute_per_subject_audit(
        dataset="testset",
        subject="synth",
        seizure_onsets=[50.0, 250.0],
        seizure_block_ids=["BLK0", "BLK1"],
        block_windows=block_windows,
        hfo_event_times_per_channel=hfo_events,
        signal_loader=signal_loader,
        sfreq=sfreq,
        clinical_soz=["chA"],
        analysis_channels=["chA", "chB"],
        m2_eligible=True,
        null_n_iter=0,
    )
    required_top_level = {
        "dataset", "subject", "n_seizures_used", "n_channels_total", "sfreq",
        "m2_eligible", "channel_matching", "baseline_rate_per_channel",
        "k_primary_size_matched", "results", "overlap_with_clinical",
        "headline_primary", "per_seizure_consistency", "time_shifted_null",
        "m2_ineligible_channels", "n_seizures_dropped", "dropped_seizure_reasons",
    }
    missing = required_top_level - set(res.keys())
    assert not missing, f"per-subject JSON missing required keys: {missing}"
    # No verdict-style keys — plan §3.9.
    forbidden = {"verdict", "broadly_consistent", "partially_consistent",
                 "unreliable", "ground_truth", "true_soz"}
    leaks = forbidden & set(res.keys())
    assert not leaks, f"per-subject JSON leaked verdict-style keys: {leaks}"


def test_per_subject_runner_uses_inventory_prefilter_not_filter_padding_error():
    """If a seizure is too close to a block boundary, the orchestrator
    must drop it via ``prefilter_seizures_by_block_window`` BEFORE
    invoking ``compute_er_logratio``. This means: ``signal_loader`` is
    never called for the dropped seizure (so we cannot rely on
    ``FilterPaddingError`` to catch it).
    """
    from src.data_driven_soz import compute_per_subject_audit

    sfreq = 1000.0
    block_len_sec = 100.0
    n_samples = int(block_len_sec * sfreq)
    rng = np.random.default_rng(0)
    block_signals = {"BLK0": rng.normal(0, 1, (n_samples, 1))}
    block_windows = {"BLK0": (0.0, block_len_sec)}
    loader_calls = []

    def signal_loader(t_start, t_end, channels):
        loader_calls.append((t_start, t_end))
        s0 = int(round(t_start * sfreq))
        s1 = int(round(t_end * sfreq))
        return block_signals["BLK0"][s0:s1, [0]], sfreq

    hfo_events = {"chA": np.array([5.0, 50.0, 95.0])}

    # One seizure at t=50 fits the [32, 88] safe band; a second at t=5
    # does NOT (5 - 32 < 0 = boundary_pre).
    res = compute_per_subject_audit(
        dataset="testset",
        subject="synth",
        seizure_onsets=[50.0, 5.0],
        seizure_block_ids=["BLK0", "BLK0"],
        block_windows=block_windows,
        hfo_event_times_per_channel=hfo_events,
        signal_loader=signal_loader,
        sfreq=sfreq,
        clinical_soz=["chA"],
        analysis_channels=["chA"],
        m2_eligible=True,
        null_n_iter=0,
    )
    assert res["n_seizures_used"] == 1
    assert res["n_seizures_dropped"] == 1
    # Loader must have been called once for power_pre precompute and
    # once for the kept seizure's compute_er_logratio (= 2 calls). The
    # boundary-violating seizure must NOT trigger any loader call —
    # if the runner relied on FilterPaddingError, loader_calls would
    # contain a 3rd entry for the dropped seizure's t=5 window.
    assert len(loader_calls) == 2
    for t0, t1 in loader_calls:
        assert t0 < 50 < t1, f"loader window {t0}-{t1} should be centered on t=50"


def test_cohort_overlap_still_raises_not_implemented():
    """Step 3 only adds --per-subject. --cohort-overlap stays
    NotImplementedError until Step 4 lands."""
    import subprocess
    import sys as _sys

    proc = subprocess.run(
        [_sys.executable, "scripts/run_data_driven_soz.py", "--cohort-overlap"],
        capture_output=True,
        text=True,
        cwd=str(__import__("pathlib").Path(__file__).resolve().parents[1]),
    )
    assert proc.returncode != 0
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert "NotImplementedError" in combined or "not implemented" in combined.lower()


def test_epilepsiae_partial_window_loader_against_synthetic_data(tmp_path):
    """Write a synthetic .data + .head and verify the partial loader
    extracts the right sample count, applies CAR, and uses
    intracranial-only channels.
    """
    from scripts.run_data_driven_soz import _epilepsiae_partial_window_loader

    sfreq = 1000.0
    n_channels = 4
    duration_sec = 10.0
    n_samples = int(sfreq * duration_sec)
    # 2 intracranial (HLA1, HLA2), 2 scalp (EEG/EMG markers in the
    # EPILEPSIAE_SCALP_AUX_CHANNELS set).
    ch_names = ["HLA1", "HLA2", "FZ", "ECG"]
    # Synthetic int16 signal: known per-channel patterns.
    rng = np.random.default_rng(42)
    raw = rng.integers(-1000, 1000, size=(n_samples, n_channels), dtype=np.int16)

    head_path = tmp_path / "synth.head"
    data_path = tmp_path / "synth.data"
    head_path.write_text(
        "num_channels=4\n"
        f"sample_freq={sfreq}\n"
        f"duration_in_sec={duration_sec}\n"
        f"num_samples={n_samples}\n"
        "sample_bytes=2\n"
        "conversion_factor=1.0\n"
        f"elec_names=[{','.join(ch_names)}]\n"
    )
    data_path.write_bytes(raw.tobytes())

    # Read the middle 2 seconds.
    sig, sfreq_ret, ch_ret = _epilepsiae_partial_window_loader(
        str(data_path), str(head_path), rel_start_sec=4.0, rel_end_sec=6.0
    )
    assert sfreq_ret == sfreq
    assert ch_ret == ["HLA1", "HLA2"]      # scalp/aux dropped
    assert sig.shape == (2000, 2)          # 2 s × 1000 Hz, 2 intracranial ch
    # CAR property: row mean across kept channels is ≈ 0.
    assert np.allclose(sig.mean(axis=1), 0.0, atol=1e-6)
    # The conversion sign (-1.0 * factor) must be applied.
    expected_intra = raw[4000:6000, :2].astype(np.float32) * np.float32(-1.0)
    expected_car = expected_intra - expected_intra.mean(axis=1, keepdims=True)
    assert np.allclose(sig, expected_car, atol=1e-4)


def test_per_subject_runner_drops_low_sfreq_blocks_from_m2_only():
    """A subject whose blocks have mixed sample rates (e.g. epilepsiae 583
    has both 1024 Hz and 256 Hz blocks) is currently audit-eligible
    because the cohort sfreq_min restricts to retained blocks. But
    seizures landing inside the 256 Hz blocks fail the 525 Hz Nyquist
    guard inside compute_er_logratio. The orchestrator must drop those
    seizures **from M2 only** (M1 doesn't need signal) via a forward
    block-sfreq check, NOT by catching NyquistGuardError.
    """
    from src.data_driven_soz import compute_per_subject_audit

    sfreq = 1000.0
    block_len_sec = 100.0
    n_samples = int(block_len_sec * sfreq)
    rng = np.random.default_rng(0)

    block_windows = {
        "BLK_HI": (0.0, block_len_sec),
        "BLK_LO": (200.0, 200.0 + block_len_sec),
    }
    block_sfreqs = {"BLK_HI": 1000.0, "BLK_LO": 256.0}  # LO is M2-ineligible

    block_signals = {
        "BLK_HI": rng.normal(0, 1, (n_samples, 2)),
    }

    loader_calls = []

    def signal_loader(t_start, t_end, channels):
        loader_calls.append((t_start, t_end, tuple(channels)))
        for blk, (b0, b1) in block_windows.items():
            if t_start >= b0 and t_end <= b1:
                if blk != "BLK_HI":
                    raise AssertionError(
                        f"loader was called for low-sfreq block {blk!r} — "
                        f"the orchestrator should have dropped this seizure "
                        f"from the M2 path before any signal load"
                    )
                s0 = int(round((t_start - b0) * sfreq))
                s1 = int(round((t_end - b0) * sfreq))
                ch_idx = {"chA": 0, "chB": 1}
                cols = [ch_idx[ch] for ch in channels]
                return block_signals[blk][s0:s1, cols], sfreq
        raise ValueError(f"window {t_start}-{t_end} doesn't fit any block")

    hfo_events = {
        "chA": np.array([45.0, 50.5, 55.0, 245.0, 250.5, 255.0]),
        "chB": np.array([46.0, 51.0, 56.0, 246.0, 251.0, 256.0]),
    }
    res = compute_per_subject_audit(
        dataset="testset",
        subject="mixed_sfreq",
        seizure_onsets=[50.0, 250.0],
        seizure_block_ids=["BLK_HI", "BLK_LO"],
        block_windows=block_windows,
        block_sfreqs=block_sfreqs,
        hfo_event_times_per_channel=hfo_events,
        signal_loader=signal_loader,
        sfreq=1000.0,
        clinical_soz=["chA"],
        analysis_channels=["chA", "chB"],
        m2_eligible=True,
        null_n_iter=0,
    )
    # M1 must still cover BOTH seizures (no signal load needed).
    assert res["n_seizures_used"] == 2
    # The loader should never have been called for the low-sfreq block.
    for t_start, t_end, _ in loader_calls:
        assert t_start < block_windows["BLK_LO"][0] or t_end > block_windows["BLK_LO"][1]
    # The drop must be reported in the JSON for downstream cohort tracking.
    assert "n_seizures_m2_dropped_low_sfreq" in res
    assert res["n_seizures_m2_dropped_low_sfreq"] == 1


def test_per_subject_runner_records_preprocessing_metadata():
    """Step 4 sensitivity comparison must distinguish between the
    partial-loader / no-notch path and the legacy full-block + notch
    path. The orchestrator records this via a ``preprocessing`` dict
    with stable keys; downstream Step 4 code splits the cohort by
    ``preprocessing['signal_loader_path']`` rather than relying on
    directory names.
    """
    from src.data_driven_soz import compute_per_subject_audit

    sfreq = 1000.0
    n_samples = int(60 * sfreq)
    rng = np.random.default_rng(0)
    block_signals = {"BLK0": rng.normal(0, 1, (n_samples, 1))}
    block_windows = {"BLK0": (0.0, 60.0)}

    def signal_loader(t_start, t_end, channels):
        s0 = int(round(t_start * sfreq))
        s1 = int(round(t_end * sfreq))
        return block_signals["BLK0"][s0:s1, [0]], sfreq

    hfo_events = {"chA": np.array([])}
    res = compute_per_subject_audit(
        dataset="testset",
        subject="meta_test",
        seizure_onsets=[],
        seizure_block_ids=[],
        block_windows=block_windows,
        block_sfreqs={"BLK0": sfreq},
        hfo_event_times_per_channel=hfo_events,
        signal_loader=signal_loader,
        sfreq=sfreq,
        clinical_soz=["chA"],
        analysis_channels=["chA"],
        m2_eligible=True,
        null_n_iter=0,
        signal_loader_path="partial_no_notch",
    )
    assert "preprocessing" in res
    pp = res["preprocessing"]
    assert pp["signal_loader_path"] == "partial_no_notch"
    assert pp["band"] == [80.0, 250.0]
    assert pp["w_pre"] == 30.0
    assert pp["w_post"] == 10.0
    assert pp["edge_buffer"] == 2.0


def test_time_shifted_null_m2_skip_is_structured_flag():
    """The M2 surrogate is intentionally skipped per-subject (each
    shifted draw would re-bandpass the full block; deferred to cohort
    level). Step 4 must be able to gate M2 true-vs-shifted reporting
    deterministically — a structured boolean flag, not a free-text
    note. Replace the legacy ``time_shifted_null['note_M2']`` string
    with a typed ``m2_surrogate_skipped`` boolean and a stable
    ``skip_reason`` string.
    """
    from src.data_driven_soz import compute_per_subject_audit

    sfreq = 1000.0
    block_len_sec = 100.0
    n_samples = int(block_len_sec * sfreq)
    rng = np.random.default_rng(0)
    block_windows = {"BLK0": (0.0, block_len_sec)}
    block_signals = {"BLK0": rng.normal(0, 1, (n_samples, 2))}

    def signal_loader(t_start, t_end, channels):
        s0 = int(round(t_start * sfreq))
        s1 = int(round(t_end * sfreq))
        ch_idx = {"chA": 0, "chB": 1}
        cols = [ch_idx[ch] for ch in channels]
        return block_signals["BLK0"][s0:s1, cols], sfreq

    hfo_events = {
        "chA": np.array([45.0, 50.5, 55.0]),
        "chB": np.array([46.0, 51.0, 56.0]),
    }
    res = compute_per_subject_audit(
        dataset="testset",
        subject="m2_surrogate_test",
        seizure_onsets=[50.0],
        seizure_block_ids=["BLK0"],
        block_windows=block_windows,
        block_sfreqs={"BLK0": sfreq},
        hfo_event_times_per_channel=hfo_events,
        signal_loader=signal_loader,
        sfreq=sfreq,
        clinical_soz=["chA"],
        analysis_channels=["chA", "chB"],
        m2_eligible=True,
        null_n_iter=5,
    )
    null = res["time_shifted_null"]
    # Required structured fields:
    assert null["m2_surrogate_skipped"] is True
    assert isinstance(null.get("m2_skip_reason"), str)
    assert null["m2_skip_reason"]  # non-empty
    # M2 enrichment must be None (not NaN), so JSON serializes to null
    # and Step 4 can use ``is None`` checks.
    assert null["enrichment_true_over_shift_M2_logratio"] is None


def test_per_subject_runner_m2_ineligible_subject_emits_nan_not_zero():
    """When ``m2_eligible=False`` (e.g. epilepsiae 139 / 253 with 256 / 512
    Hz blocks), M2 was never computed. The previous behavior reported
    H_M2=0.0 and empty top-k lists, which Step 4 cohort code would
    average as a real zero into the M2 median — silently dragging the
    cohort headline down.

    The fix: emit ``None`` (JSON ``null``) for every M2-derived field
    when M2 was not run. Step 4 must skip these subjects from the M2
    distribution rather than treat them as observed zeros.
    """
    from src.data_driven_soz import compute_per_subject_audit

    sfreq_hi = 1000.0
    block_len_sec = 100.0
    n_samples = int(block_len_sec * sfreq_hi)
    rng = np.random.default_rng(0)

    block_windows = {"BLK0": (0.0, block_len_sec)}
    block_signals = {"BLK0": rng.normal(0, 1, (n_samples, 2))}

    def signal_loader(t_start, t_end, channels):  # never called
        raise AssertionError(
            "signal_loader must NOT be called when m2_eligible=False"
        )

    hfo_events = {
        "chA": np.array([45.0, 50.5, 55.0]),
        "chB": np.array([46.0, 51.0, 56.0]),
    }
    res = compute_per_subject_audit(
        dataset="testset",
        subject="m2_ineligible_subj",
        seizure_onsets=[50.0],
        seizure_block_ids=["BLK0"],
        block_windows=block_windows,
        block_sfreqs={"BLK0": 256.0},   # below 525 Hz
        hfo_event_times_per_channel=hfo_events,
        signal_loader=signal_loader,
        sfreq=256.0,
        clinical_soz=["chA"],
        analysis_channels=["chA", "chB"],
        m2_eligible=False,             # subject-level skip
        null_n_iter=0,
    )

    # H_M2 and H_concord must be None (JSON null), NOT 0.0.
    assert res["headline_primary"]["H_M2_logratio_medianrank_size_matched"] is None
    assert res["headline_primary"]["H_concord_M1_M2_size_matched"] is None
    # H_M1 still a real number (M1 ran).
    h_m1 = res["headline_primary"]["H_M1_pois_medianrank_size_matched"]
    assert h_m1 is not None and isinstance(h_m1, float)

    # M2 results dict must be None (or every k-list None), not empty.
    m2_results = res["results"]["M2_logratio"]
    for agg, by_k in m2_results.items():
        for k_label, ranking in by_k.items():
            assert ranking is None, (
                f"results.M2_logratio.{agg}.{k_label} must be None for "
                f"m2_eligible=False subject, got {ranking!r}"
            )

    # overlap_with_clinical M2 entries must be None.
    for key, ov in res["overlap_with_clinical"].items():
        if key.startswith("M2_logratio"):
            assert ov is None, (
                f"overlap_with_clinical.{key} must be None for "
                f"m2_eligible=False subject, got {ov!r}"
            )

    # Per-seizure consistency M2 must be None.
    assert res["per_seizure_consistency"]["M2_logratio_kPrimary"] is None


def test_epilepsiae_partial_window_loader_short_read_raises(tmp_path):
    """A request beyond the file's sample count must raise — never
    silently truncate (would corrupt M2 windows)."""
    from scripts.run_data_driven_soz import _epilepsiae_partial_window_loader

    sfreq = 1000.0
    n_channels = 2
    duration_sec = 1.0
    n_samples = int(sfreq * duration_sec)
    ch_names = ["HLA1", "HLA2"]
    raw = np.zeros((n_samples, n_channels), dtype=np.int16)

    head_path = tmp_path / "short.head"
    data_path = tmp_path / "short.data"
    head_path.write_text(
        "num_channels=2\n"
        f"sample_freq={sfreq}\n"
        f"duration_in_sec={duration_sec}\n"
        f"num_samples={n_samples}\n"
        "sample_bytes=2\n"
        "conversion_factor=1.0\n"
        f"elec_names=[{','.join(ch_names)}]\n"
    )
    data_path.write_bytes(raw.tobytes())
    with pytest.raises(ValueError, match="short read"):
        _epilepsiae_partial_window_loader(
            str(data_path), str(head_path), rel_start_sec=0.0, rel_end_sec=2.0
        )
