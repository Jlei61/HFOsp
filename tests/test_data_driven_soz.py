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
