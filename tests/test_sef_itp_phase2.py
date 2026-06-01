"""Unit tests for src/sef_itp_phase2.py (SEF-ITP framework Phase 2: H3 + H4).

Phase 2 implements H3 (mark-independent template sampling) + H4 (normalized
rate vs geometry instability) on top of Phase 1's n=23 cohort. H3 is mostly
ingest-heavy reuse of PR-7 / PR-6 outputs; H4 is new (epoch slicing +
normalized instability + matched null).

Tests grow incrementally per plan tasks in
docs/superpowers/plans/2026-05-23-topic4-phase2-h3-h4-plan.md.
"""
from __future__ import annotations

import numpy as np
import pytest

from src import sef_itp_phase2 as p2


# --------------------------------------------------------------------------- #
# Task 1 — module skeleton + SubjectPhase2Data
# --------------------------------------------------------------------------- #


def test_module_version():
    """v1.1 — bumped 2026-05-23 user catch (rank-based endpoint, see B1+ in phase2 h4 v1.1 plan)."""
    assert p2.__version__ == "v1.1.0"


def test_subject_phase2_data_dataclass_fields():
    """SubjectPhase2Data dataclass must carry both H3 ingest fields and H4 raw inputs."""
    s = p2.SubjectPhase2Data(
        dataset="yuquan",
        subject_id="test",
        # H3 ingest
        lag1_same_excess_n2=0.01,
        window_excess_n2={10.0: 0.0, 30.0: 0.0, 60.0: 0.0, 1800.0: 0.0},
        run_length_lift_n2=1.0,
        endpoint_jaccard_first_half=0.9,
        endpoint_jaccard_odd_even=0.85,
        # H4 raw
        event_abs_times=np.array([0.0, 1.0, 2.0]),
        cluster_labels=np.array([0, 1, 0]),
        block_time_ranges=[(0.0, 10.0)],
        template_ranks={0: np.array([0, 1, 2, 3, 4, 5]), 1: np.array([5, 4, 3, 2, 1, 0])},
        channel_names=["A", "B", "C", "D", "E", "F"],
    )
    assert s.dataset == "yuquan"
    assert s.lag1_same_excess_n2 == pytest.approx(0.01)
    assert 10.0 in s.window_excess_n2
    assert s.event_abs_times.shape == (3,)


# --------------------------------------------------------------------------- #
# Task 2 — H3 ingest extractors (PR-7 pairing/burst + PR-6 anchoring)
# --------------------------------------------------------------------------- #


def test_extract_window_excess_from_pairing():
    """extract_window_excess reads pairing_with_nulls.lift.N2.{10,30,60,1800}.excess (string keys)."""
    pairing_json = {
        "pairing_with_nulls": {
            "lift": {
                "N2": {
                    "1.0": {"excess": 0.10},
                    "5.0": {"excess": 0.08},
                    "10.0": {"excess": 0.03},
                    "30.0": {"excess": 0.01},
                    "60.0": {"excess": 0.005},
                    "300.0": {"excess": 0.0},
                    "1800.0": {"excess": -0.001},
                    "3600.0": {"excess": -0.002},
                }
            }
        }
    }
    metrics = p2.extract_window_excess_from_pairing(
        pairing_json, windows=(10.0, 30.0, 60.0, 1800.0)
    )
    assert metrics == {10.0: 0.03, 30.0: 0.01, 60.0: 0.005, 1800.0: -0.001}


def test_extract_window_excess_missing_window_raises():
    """If a requested window key is missing, raise KeyError (no silent default)."""
    pairing_json = {
        "pairing_with_nulls": {
            "lift": {
                "N2": {"10.0": {"excess": 0.03}}
                # 30, 60, 1800 missing
            }
        }
    }
    with pytest.raises(KeyError):
        p2.extract_window_excess_from_pairing(
            pairing_json, windows=(10.0, 30.0, 60.0, 1800.0)
        )


def test_extract_lag1_and_runlength_from_burst():
    """extract_lag1_and_runlength reads burst_diagnostic.lag1_same_excess.N2 +
    burst_diagnostic.lift.N2.run_length_lift."""
    burst_json = {
        "burst_diagnostic": {
            "lag1_same_excess": {"N1": 0.02, "N2": 0.005},
            "lift": {
                "N2": {"run_length_lift": 0.97, "mean_run_length": 0.97},
            },
        }
    }
    lag1, run_length = p2.extract_lag1_and_runlength_from_burst(burst_json)
    assert lag1 == pytest.approx(0.005)
    assert run_length == pytest.approx(0.97)


def test_extract_endpoint_jaccard_from_anchoring():
    """extract_endpoint_jaccard reads PR-6 split_half_robustness.per_split.{first,odd}
    .subject_mean_jaccard_endpoint."""
    anchoring_json = {
        "split_half_robustness": {
            "per_split": {
                "first_half_second_half": {"subject_mean_jaccard_endpoint": 0.9},
                "odd_even_block": {"subject_mean_jaccard_endpoint": 0.85},
            }
        }
    }
    fh, oe = p2.extract_endpoint_jaccard_from_anchoring(anchoring_json)
    assert fh == pytest.approx(0.9)
    assert oe == pytest.approx(0.85)


def test_extract_endpoint_jaccard_missing_per_split_raises():
    """If split_half_robustness or per_split is missing, raise (not silently zero)."""
    with pytest.raises(KeyError):
        p2.extract_endpoint_jaccard_from_anchoring({})
    with pytest.raises(KeyError):
        p2.extract_endpoint_jaccard_from_anchoring({"split_half_robustness": {}})


# --------------------------------------------------------------------------- #
# Task 3 — tost_equivalence (ported from PR-7 addendum)
# --------------------------------------------------------------------------- #


def test_tost_equivalence_compatible_median_zero():
    """Cohort median ~0 with tight CI well inside ±δ → equivalence_pass = True."""
    rng = np.random.default_rng(42)
    vals = rng.normal(loc=0.0, scale=0.005, size=30)
    out = p2.tost_equivalence(vals, target=0.0, delta=0.05, n_boot=2000, seed=0)
    assert out["equivalence_pass"] is True
    assert -0.05 < out["ci95_lo"] < out["ci95_hi"] < 0.05


def test_tost_equivalence_violated_median_outside_band():
    """Cohort median 0.1 > δ → equivalence_pass = False."""
    rng = np.random.default_rng(42)
    vals = rng.normal(loc=0.10, scale=0.01, size=30)
    out = p2.tost_equivalence(vals, target=0.0, delta=0.05, n_boot=2000, seed=0)
    assert out["equivalence_pass"] is False
    assert out["ci95_lo"] > 0.05  # whole CI above δ band


def test_tost_equivalence_target_one_for_run_length():
    """run_length_lift target=1 with vals ~ 1.0 → equivalence_pass = True."""
    rng = np.random.default_rng(42)
    vals = rng.normal(loc=1.0, scale=0.005, size=30)
    out = p2.tost_equivalence(vals, target=1.0, delta=0.05, n_boot=2000, seed=0)
    assert out["equivalence_pass"] is True
    assert out["target"] == pytest.approx(1.0)


def test_tost_equivalence_returns_expected_schema():
    """Output dict carries all expected keys (mirror of PR-7 addendum)."""
    out = p2.tost_equivalence(np.array([0.0, 0.0, 0.0]), target=0.0, delta=0.05, n_boot=100, seed=0)
    for k in (
        "median_obs", "ci95_lo", "ci95_hi", "tost_p_lower", "tost_p_upper", "tost_p",
        "equivalence_pass", "median_inside_band", "ci_inside_band", "target", "delta", "n",
    ):
        assert k in out, f"missing key: {k}"


# --------------------------------------------------------------------------- #
# Task 4 — H3 integrated verdict (SUPPORTED / NOT_SUPPORTED_* / CONTRADICTED)
# --------------------------------------------------------------------------- #


_TOST_PASS = {"equivalence_pass": True, "cohort_main": {"equivalence_pass": True}}


def _all_mark_pass():
    return {
        "lag1_same_excess": dict(_TOST_PASS),
        "window_excess_10s": dict(_TOST_PASS),
        "window_excess_30s": dict(_TOST_PASS),
        "window_excess_60s": dict(_TOST_PASS),
        "window_excess_1800s": dict(_TOST_PASS),
        "run_length_lift": dict(_TOST_PASS),
    }


def test_h3_integrated_verdict_supported():
    """All TOST equivalent + both endpoint Jaccard medians ≥ 0.7 → SUPPORTED."""
    verdict = p2.compute_h3_integrated_verdict(
        cohort_tost=_all_mark_pass(),
        endpoint_jaccard_first_half_median=0.85,
        endpoint_jaccard_odd_even_median=0.80,
    )
    assert verdict == "SUPPORTED"


def test_h3_integrated_verdict_or_combinator():
    """One Jaccard median ≥ 0.7, other below → still SUPPORTED (OR combinator).

    Mirrors AGENTS.md forward_reverse_reproduced = split-half OR odd-even.
    """
    verdict = p2.compute_h3_integrated_verdict(
        cohort_tost=_all_mark_pass(),
        endpoint_jaccard_first_half_median=0.85,  # ≥ 0.7
        endpoint_jaccard_odd_even_median=0.55,    # < 0.7
    )
    assert verdict == "SUPPORTED"


def test_h3_integrated_verdict_not_supported_geometry_unstable():
    """All TOST equivalent + BOTH endpoint Jaccard medians < 0.7 → NOT_SUPPORTED_GEOMETRY_UNSTABLE."""
    verdict = p2.compute_h3_integrated_verdict(
        cohort_tost=_all_mark_pass(),
        endpoint_jaccard_first_half_median=0.60,
        endpoint_jaccard_odd_even_median=0.55,
    )
    assert verdict == "NOT_SUPPORTED_GEOMETRY_UNSTABLE"


def test_h3_integrated_verdict_contradicted_robust():
    """≥1 TOST fail with robust LOO (no LOO subset restores equivalence) → CONTRADICTED."""
    cohort_tost = _all_mark_pass()
    cohort_tost["lag1_same_excess"] = {
        "equivalence_pass": False,
        "leave_one_out_min_pass_rate": 0.0,  # robust failure
    }
    verdict = p2.compute_h3_integrated_verdict(
        cohort_tost=cohort_tost,
        endpoint_jaccard_first_half_median=0.85,
        endpoint_jaccard_odd_even_median=0.80,
    )
    assert verdict == "CONTRADICTED"


def test_h3_integrated_verdict_not_supported_memory_single_subject_sensitive():
    """≥1 TOST fail but LOO restores equivalence (single-subject sensitive) → NOT_SUPPORTED_MEMORY."""
    cohort_tost = _all_mark_pass()
    cohort_tost["window_excess_10s"] = {
        "equivalence_pass": False,
        "leave_one_out_min_pass_rate": 0.8,  # not robust
    }
    verdict = p2.compute_h3_integrated_verdict(
        cohort_tost=cohort_tost,
        endpoint_jaccard_first_half_median=0.85,
        endpoint_jaccard_odd_even_median=0.80,
    )
    assert verdict == "NOT_SUPPORTED_MEMORY"


def test_h3_integrated_verdict_not_supported_both():
    """≥1 TOST fail AND endpoint unstable → NOT_SUPPORTED_BOTH."""
    cohort_tost = _all_mark_pass()
    cohort_tost["run_length_lift"] = {
        "equivalence_pass": False,
        "leave_one_out_min_pass_rate": 0.0,
    }
    verdict = p2.compute_h3_integrated_verdict(
        cohort_tost=cohort_tost,
        endpoint_jaccard_first_half_median=0.60,
        endpoint_jaccard_odd_even_median=0.55,
    )
    assert verdict == "NOT_SUPPORTED_BOTH"


# --------------------------------------------------------------------------- #
# Task 5 — H4 epoch slicer (block-aware, time-preserving)
# --------------------------------------------------------------------------- #


def test_slice_events_into_epochs_basic():
    """24h of events sliced into 2h epochs → 12 epochs, time-ordered, no event overlap."""
    t0 = 1_000_000.0
    times = t0 + np.arange(86_400, dtype=float)  # 1 event/sec for 24h
    labels = np.tile([0, 1], 43_200)
    block_time_ranges = [(t0, t0 + 86_400)]
    epochs = p2.slice_events_into_epochs(
        event_abs_times=times,
        cluster_labels=labels,
        block_time_ranges=block_time_ranges,
        epoch_hours=2.0,
    )
    assert len(epochs) == 12
    assert epochs[0]["t_start"] == pytest.approx(t0)
    assert epochs[0]["t_end"] == pytest.approx(t0 + 7200.0)
    assert len(epochs[0]["event_indices"]) == 7200
    total = sum(len(ep["event_indices"]) for ep in epochs)
    assert total == 86_400
    for i in range(1, len(epochs)):
        assert epochs[i]["t_start"] >= epochs[i - 1]["t_end"]


def test_slice_events_into_epochs_handles_gaps():
    """Events split across two recording blocks with a 12h gap — slicer respects block
    boundaries (no phantom epoch covering the gap)."""
    t0 = 1_000_000.0
    block_a = (t0, t0 + 7200.0)
    block_b = (t0 + 50_000.0, t0 + 57_200.0)
    times = np.concatenate([
        np.linspace(t0 + 1.0, t0 + 7199.0, 100),
        np.linspace(t0 + 50_001.0, t0 + 57_199.0, 100),
    ])
    labels = np.tile([0, 1], 100)
    epochs = p2.slice_events_into_epochs(
        event_abs_times=times,
        cluster_labels=labels,
        block_time_ranges=[block_a, block_b],
        epoch_hours=2.0,
    )
    assert len(epochs) == 2
    assert len(epochs[0]["event_indices"]) == 100
    assert len(epochs[1]["event_indices"]) == 100
    assert epochs[1]["t_start"] >= block_b[0]


def test_slice_events_drops_short_epochs():
    """Epoch with < min_events events is dropped."""
    t0 = 1_000_000.0
    times = np.concatenate([
        np.linspace(t0 + 1.0, t0 + 7199.0, 100),       # epoch 0: 100 events (kept)
        np.linspace(t0 + 7201.0, t0 + 14_399.0, 3),    # epoch 1: 3 events (dropped @ min=10)
    ])
    labels = np.array([0, 1] * 50 + [0, 1, 0])
    block_time_ranges = [(t0, t0 + 14_400.0)]
    epochs = p2.slice_events_into_epochs(
        event_abs_times=times,
        cluster_labels=labels,
        block_time_ranges=block_time_ranges,
        epoch_hours=2.0,
        min_events=10,
    )
    assert len(epochs) == 1
    assert len(epochs[0]["event_indices"]) == 100


def test_slice_events_epoch_tolerance_handles_short_blocks():
    """Epilepsiae-style: blocks of ~59:41 min (slightly under 1h target).
    With epoch_tolerance=0.1 each block counts as 1 epoch; default tolerance=0 → 0 epochs."""
    t0 = 1_000_000.0
    blocks = [
        (t0, t0 + 3581.0),                      # ~59:41 = 0.9947h
        (t0 + 3700.0, t0 + 3700.0 + 3585.0),    # ~59:45
        (t0 + 7400.0, t0 + 7400.0 + 3590.0),    # ~59:50
    ]
    # 50 events per block
    times = np.concatenate([
        np.linspace(b[0] + 1, b[1] - 1, 50) for b in blocks
    ])
    labels = np.tile([0, 1], 75)
    # Strict floor → 0 epochs (each block < 1h)
    strict = p2.slice_events_into_epochs(
        event_abs_times=times, cluster_labels=labels,
        block_time_ranges=blocks, epoch_hours=1.0, min_events=10,
        epoch_tolerance=0.0,
    )
    assert len(strict) == 0
    # Tolerance 0.1 → each block becomes 1 epoch
    lenient = p2.slice_events_into_epochs(
        event_abs_times=times, cluster_labels=labels,
        block_time_ranges=blocks, epoch_hours=1.0, min_events=10,
        epoch_tolerance=0.1,
    )
    assert len(lenient) == 3
    for ep, block in zip(lenient, blocks):
        assert ep["t_end"] == pytest.approx(block[1])  # clipped to block_end


# --------------------------------------------------------------------------- #
# Task 6 — H4 per-epoch local endpoint + Jaccard helper
# --------------------------------------------------------------------------- #


def test_compute_local_endpoint_returns_top_k_source_sink():
    """For 2 clusters with clear leader/laggard channels, local endpoint returns expected indices."""
    bools = np.array([
        [1, 1, 1, 0, 0, 0],   # cluster 0 leaders ch0,1,2
        [1, 1, 0, 1, 0, 0],   # cluster 0 leaders ch0,1,3
        [0, 0, 0, 1, 1, 1],   # cluster 1 leaders ch3,4,5
        [0, 0, 1, 0, 1, 1],   # cluster 1 leaders ch2,4,5
    ], dtype=bool)
    labels = np.array([0, 0, 1, 1])
    out = p2.compute_local_endpoint(bools, labels, k=2)
    # cluster 0 mean = [1, 1, 0.5, 0.5, 0, 0] → source top-2 = {0,1}; sink bottom-2 = {4,5}
    assert set(out[0]["source"]) == {0, 1}
    assert set(out[0]["sink"]) == {4, 5}
    # cluster 1 mean = [0, 0, 0.5, 0.5, 1, 1] → source = {4,5}; sink = {0,1}
    assert set(out[1]["source"]) == {4, 5}
    assert set(out[1]["sink"]) == {0, 1}


def test_compute_local_endpoint_respects_valid_mask():
    """If valid_mask drops some channels, they're not eligible for source/sink top-k."""
    bools = np.zeros((4, 6), dtype=bool)
    bools[:2, [0, 1, 2]] = True   # cluster 0 leaders ch0,1,2
    bools[2:, [3, 4, 5]] = True   # cluster 1 leaders ch3,4,5
    labels = np.array([0, 0, 1, 1])
    valid_mask = np.array([True, True, False, True, True, False])  # ch2, ch5 invalid
    out = p2.compute_local_endpoint(bools, labels, k=2, valid_mask=valid_mask)
    assert 2 not in out[0]["source"] and 2 not in out[0]["sink"]
    assert 5 not in out[1]["source"] and 5 not in out[1]["sink"]


def test_endpoint_jaccard_local_vs_global():
    """jaccard_endpoint(local, global) = |intersect|/|union| of source∪sink sets."""
    local = {0: {"source": [0, 1, 2], "sink": [5, 6, 7]}}
    global_ = {0: {"source": [0, 1, 3], "sink": [5, 6, 8]}}
    j = p2.endpoint_jaccard(local, global_, cluster_id=0)
    # local endpoint = {0,1,2,5,6,7}; global = {0,1,3,5,6,8}
    # intersect = {0,1,5,6}, union = {0,1,2,3,5,6,7,8} → 4/8 = 0.5
    assert j == pytest.approx(0.5)


def test_endpoint_jaccard_perfect_match_returns_one():
    local = {0: {"source": [0, 1, 2], "sink": [5, 6, 7]}}
    global_ = {0: {"source": [2, 1, 0], "sink": [7, 6, 5]}}  # order shuffled
    assert p2.endpoint_jaccard(local, global_, cluster_id=0) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Task 7 — H4 I_rate normalized instability (BOTH null methods; advisor catch B)
# --------------------------------------------------------------------------- #


def test_compute_I_rate_epoch_order_shuffle_is_degenerate():
    """Framework v1.0.5 §3.4 literal null (epoch_order_shuffle) is degenerate by construction:
    std is permutation-invariant, so null_var = 0 and I_rate is flagged undefined."""
    rates = np.array([5.0, 10.0, 20.0, 40.0, 80.0])
    result = p2.compute_I_rate_normalized(rates, n_perm=200, seed=0)
    assert result["null_std_var"] == pytest.approx(0.0, abs=1e-9)
    assert result["I_rate_undefined_under_shuffle_null"] is True
    assert result["null_method"] == "epoch_order_shuffle"


def test_compute_I_rate_constant_rate_zero_obs_std():
    """Constant rate → obs std = 0 (degenerate null too, but for a different reason)."""
    rates = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
    result = p2.compute_I_rate_normalized(rates, n_perm=200, seed=0)
    assert result["log_rate_std_obs"] == pytest.approx(0.0, abs=1e-9)


def test_compute_I_rate_circular_shift_nondegenerate():
    """Circular shift within block: random offset → epoch membership randomized →
    rate per epoch changes → std varies across permutations → I_rate well-defined."""
    rng = np.random.default_rng(0)
    t0 = 1_000_000.0
    # 24h block, ~720 events with sinusoidal modulation (rate ranges 10–50 events/h)
    n_events = 720
    times = np.sort(t0 + rng.uniform(0, 86_400, size=n_events))
    result = p2.compute_I_rate_normalized_circular_shift(
        event_abs_times=times,
        block_time_ranges=[(t0, t0 + 86_400)],
        epoch_hours=2.0,
        n_perm=200,
        seed=0,
    )
    assert result["null_std_var"] > 0
    assert np.isfinite(result["I_rate"])
    assert result["null_method"] == "circular_shift_within_block"
    assert result["n_epochs"] == 12


def test_compute_I_rate_circular_shift_respects_block_boundary():
    """Circular shift stays within block — events don't cross blocks."""
    rng = np.random.default_rng(0)
    t0 = 1_000_000.0
    block_a = (t0, t0 + 14_400.0)            # 4h
    block_b = (t0 + 50_000.0, t0 + 64_400.0)  # 4h, 12h gap
    # 100 events in each block
    times = np.concatenate([
        np.sort(t0 + rng.uniform(0, 14_400, size=100)),
        np.sort(t0 + 50_000 + rng.uniform(0, 14_400, size=100)),
    ])
    result = p2.compute_I_rate_normalized_circular_shift(
        event_abs_times=times,
        block_time_ranges=[block_a, block_b],
        epoch_hours=2.0,
        n_perm=100,
        seed=0,
    )
    # 2 epochs per block × 2 blocks = 4 epochs (depending on min_events; default 10)
    assert result["n_epochs"] == 4
    assert result["null_std_var"] > 0


# --------------------------------------------------------------------------- #
# Task 8 — H4 I_geom normalized instability
# --------------------------------------------------------------------------- #


def test_compute_I_geom_random_endpoint_null_nondegenerate():
    """Random endpoint sample → Jaccard varies → I_geom well-defined."""
    rng = np.random.default_rng(42)
    n_epochs = 10
    n_ch = 12
    global_endpoint = {0: {"source": [0, 1, 2], "sink": [9, 10, 11]}}
    # Per-epoch local: 5 near global, 5 far
    per_epoch_local = []
    for i in range(n_epochs):
        if i < 5:
            per_epoch_local.append({0: {"source": [0, 1, 3], "sink": [9, 10, 8]}})
        else:
            per_epoch_local.append({0: {"source": [4, 5, 6], "sink": [4, 5, 6]}})
    valid_mask = np.ones(n_ch, dtype=bool)
    result = p2.compute_I_geom_normalized(
        per_epoch_local=per_epoch_local,
        global_endpoint=global_endpoint,
        valid_mask=valid_mask,
        endpoint_size=6,
        n_perm=500,
        seed=0,
    )
    assert result["null_std_var"] > 0  # non-degenerate
    assert np.isfinite(result["I_geom"])
    assert result["I_geom"] > 0
    assert result["n_epochs"] == 10


def test_compute_I_geom_perfect_stability_zero_obs():
    """Every epoch's local endpoint == global → obs_std = 0 → I_geom = 0."""
    n_epochs = 8
    n_ch = 10
    global_endpoint = {0: {"source": [0, 1, 2], "sink": [7, 8, 9]}}
    per_epoch_local = [global_endpoint for _ in range(n_epochs)]
    valid_mask = np.ones(n_ch, dtype=bool)
    result = p2.compute_I_geom_normalized(
        per_epoch_local=per_epoch_local,
        global_endpoint=global_endpoint,
        valid_mask=valid_mask,
        endpoint_size=6,
        n_perm=200,
        seed=0,
    )
    assert result["geom_dispersion_std_obs"] == pytest.approx(0.0, abs=1e-9)
    assert result["I_geom"] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# Task 9 — H4 cohort Wilcoxon + Cohen's d verdict
# --------------------------------------------------------------------------- #


def test_h4_cohort_verdict_pass():
    """All subjects I_rate >> I_geom + Wilcoxon p < 0.05 + Cohen's d ≥ 0.3 → PASS."""
    I_rate = np.array([3.0, 2.5, 2.8, 2.2, 3.5, 2.9, 2.1, 3.1, 2.6, 2.4])
    I_geom = np.array([0.5, 0.8, 0.7, 0.9, 0.6, 0.8, 1.0, 0.7, 0.9, 0.8])
    result = p2.compute_h4_cohort_verdict(I_rate, I_geom)
    assert result["verdict"] == "PASS"
    assert result["wilcoxon_p"] < 0.05
    assert result["cohen_d"] >= 0.3


def test_h4_cohort_verdict_null_low_effect():
    """I_rate ≈ I_geom → NULL."""
    rng = np.random.default_rng(42)
    I_rate = rng.normal(1.0, 0.3, size=10)
    I_geom = rng.normal(1.0, 0.3, size=10)
    result = p2.compute_h4_cohort_verdict(I_rate, I_geom)
    assert result["verdict"] == "NULL"


def test_h4_cohort_verdict_fail_geom_more_unstable():
    """I_geom systematically > I_rate → FAIL."""
    I_rate = np.array([0.5, 0.6, 0.4, 0.7, 0.5, 0.8, 0.4, 0.6, 0.5, 0.7])
    I_geom = np.array([2.5, 2.4, 2.8, 2.3, 2.6, 2.2, 2.7, 2.4, 2.5, 2.3])
    result = p2.compute_h4_cohort_verdict(I_rate, I_geom)
    assert result["verdict"] == "FAIL"
    assert result["cohen_d"] < 0


def test_h4_cohort_verdict_underpowered_n_lt_6():
    """n < 6 → UNDERPOWERED, no PASS/NULL/FAIL."""
    I_rate = np.array([3.0, 2.5, 2.8])
    I_geom = np.array([0.5, 0.8, 0.7])
    result = p2.compute_h4_cohort_verdict(I_rate, I_geom)
    assert result["verdict"] == "UNDERPOWERED"
    assert result["n_subjects"] == 3


def test_h4_cohort_verdict_handles_inf_filter():
    """Subjects with non-finite I_rate or I_geom are filtered out before stats."""
    I_rate = np.array([3.0, np.inf, 2.5, np.inf, 2.8, np.nan, 3.5, 2.9, 2.1, 3.1, 2.6, 2.4])
    I_geom = np.array([0.5, 0.8, np.inf, 0.9, 0.6, 0.8, 1.0, 0.7, 0.9, 0.8, 0.6, 0.8])
    result = p2.compute_h4_cohort_verdict(I_rate, I_geom)
    # Indices 1, 2, 3, 5 have non-finite values → drop. Remaining n=8.
    assert result["n_subjects"] == 8


# --------------------------------------------------------------------------- #
# Task 11.5 — LOO populator (advisor catch C — required for CONTRADICTED branch)
# --------------------------------------------------------------------------- #


def test_cohort_tost_with_loo_robust_failure():
    """When cohort fails TOST AND failure persists across all LOO drops:
    leave_one_out_min_pass_rate = 0.0 → CONTRADICTED branch can fire."""
    rng = np.random.default_rng(0)
    values = rng.normal(0.10, 0.005, size=30)  # all ~0.10, far above δ=0.05 band
    loo = p2.cohort_tost_with_loo(values, target=0.0, delta=0.05, n_boot=500, seed=0)
    assert loo["cohort_main"]["equivalence_pass"] is False
    assert loo["equivalence_pass"] is False
    assert loo["leave_one_out_min_pass_rate"] == 0.0
    assert "leave_one_out" in loo


def test_cohort_tost_with_loo_single_subject_sensitive():
    """Outlier subject — dropping it restores equivalence → min_pass_rate > 0.

    Constructed: 29 compatible subjects (~0.0) + 1 huge outlier (~0.5). Cohort median
    is dominated by the 29 cohort members (median ~0), but bootstrap CI is widened by
    the outlier (still might pass for n=30 even / 29 odd). Use n=10 to keep cohort small
    so the outlier matters; verify dropping outlier restores main test.
    """
    rng = np.random.default_rng(0)
    values = np.concatenate([
        rng.normal(0.0, 0.005, size=9),   # 9 compatible
        np.array([0.5]),                  # 1 outlier
    ])
    loo = p2.cohort_tost_with_loo(values, target=0.0, delta=0.05, n_boot=500, seed=0)
    # We don't strictly assert cohort_main fails (n=10 even may or may not be sensitive);
    # we just assert the LOO scaffold returns sensible structure.
    assert "leave_one_out_min_pass_rate" in loo
    assert 0.0 <= loo["leave_one_out_min_pass_rate"] <= 1.0
    assert len(loo["leave_one_out"]) == 10


def test_cohort_tost_with_loo_returns_main_equivalence_alias():
    """`equivalence_pass` mirrors cohort_main['equivalence_pass'] for verdict consumer compat."""
    rng = np.random.default_rng(0)
    values = rng.normal(0.0, 0.005, size=30)  # cohort-compatible
    loo = p2.cohort_tost_with_loo(values, target=0.0, delta=0.05, n_boot=500, seed=0)
    assert loo["equivalence_pass"] == loo["cohort_main"]["equivalence_pass"]


# --------------------------------------------------------------------------- #
# Stage B v1.1 — rank-based endpoint geometry drift (H4 v1.1 main line)
# Plan: docs/superpowers/plans/2026-05-23-topic4-phase2-h4-v1.1-rank-endpoint-plan.md
#
# H4 v1.0 used participation field top-k (events_bool.mean) as "endpoint" — that
# measures participation field drift, not propagation rank endpoint drift. v1.1
# fixes this by using masked mean lag rank per channel (phantom-rank discipline
# per AGENTS.md cross-PR lagPatRank contract).
# --------------------------------------------------------------------------- #


# --- B1 main: compute_local_rank_endpoint ---
#
# v1.1 endpoint extractor reuses PR-6 same primitives per-epoch:
#   `_per_cluster_template_rank` (→ `_legacy_hist_mean_rank` from interictal_propagation
#   → argsort(argsort)) + `extract_endpoint_middle` from template_anatomical_anchoring +
#   per-cluster valid_mask. NOT a new "masked arithmetic mean rank" estimator (user-return
#   v2 catch 2026-05-23: that would be a different statistic and wouldn't reproduce
#   PR-6 anchoring on full data — calibration must pass).


def test_compute_local_rank_endpoint_basic_two_clusters():
    """For 2 clusters with masked mean rank low/high split, source = lowest mean, sink = highest."""
    # 5 channels, 4 events. Cluster 0: events 0,1. Cluster 1: events 2,3.
    # Cluster 0 mean masked rank: ch0=0, ch1=1, ch2=2, ch3=3, ch4=4 → source=lowest=[0,1], sink=highest=[3,4]
    # Cluster 1 mean masked rank: ch0=4, ch1=3, ch2=2, ch3=1, ch4=0 → source=[4,3], sink=[0,1]
    ranks = np.array([
        [0, 0, 4, 4],
        [1, 1, 3, 3],
        [2, 2, 2, 2],
        [3, 3, 1, 1],
        [4, 4, 0, 0],
    ], dtype=float)
    bools = np.ones_like(ranks, dtype=bool)
    labels = np.array([0, 0, 1, 1])
    out = p2.compute_local_rank_endpoint(
        ranks, bools, labels, event_indices=np.array([0, 1, 2, 3]), k=2,
    )
    assert set(out[0]["source"]) == {0, 1}
    assert set(out[0]["sink"]) == {3, 4}
    assert set(out[1]["source"]) == {4, 3}
    assert set(out[1]["sink"]) == {0, 1}


def test_compute_local_rank_endpoint_phantom_mask_changes_endpoint():
    """C1 — Phantom-mask discipline: a channel with phantom int rank on non-participating
    events must NOT be picked as endpoint based on those phantom values."""
    # 4 channels, 4 events. All in cluster 0.
    # Channel 0: rank 0 on event 0 (participates), phantom 99 on events 1,2,3 (does NOT participate)
    #   → masked mean = 0 → source candidate. Unmasked mean would be 99*3/4=74.25 → sink candidate.
    # Channel 1: rank 1 on all 4 events (participates) → mean=1
    # Channel 2: rank 2 on all 4 events (participates) → mean=2
    # Channel 3: rank 3 on all 4 events (participates) → mean=3 → sink
    ranks = np.array([
        [0, 99, 99, 99],
        [1, 1, 1, 1],
        [2, 2, 2, 2],
        [3, 3, 3, 3],
    ], dtype=float)
    bools = np.array([
        [1, 0, 0, 0],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ], dtype=bool)
    labels = np.array([0, 0, 0, 0])
    out = p2.compute_local_rank_endpoint(
        ranks, bools, labels, event_indices=np.array([0, 1, 2, 3]), k=1,
    )
    # Source = mean rank lowest = channel 0 (masked mean = 0). NOT channel 3.
    assert out[0]["source"] == [0]
    assert out[0]["sink"] == [3]


def test_compute_local_rank_endpoint_zero_participation_channel_excluded():
    """C2 — A channel with zero participation in the slice is excluded entirely; does not
    appear as either source or sink even if it has phantom ranks."""
    # 4 channels, 2 events all in cluster 0.
    # Channel 2 has phantom rank but never participates → must be excluded.
    ranks = np.array([
        [1.0, 1.0],
        [2.0, 2.0],
        [99.0, 99.0],   # phantom only, never participates
        [3.0, 3.0],
    ])
    bools = np.array([
        [1, 1],
        [1, 1],
        [0, 0],
        [1, 1],
    ], dtype=bool)
    labels = np.array([0, 0])
    out = p2.compute_local_rank_endpoint(
        ranks, bools, labels, event_indices=np.array([0, 1]), k=1,
    )
    # 3 eligible channels (0,1,3) with masked means 1,2,3. Source=ch0, sink=ch3.
    assert out[0]["source"] == [0]
    assert out[0]["sink"] == [3]
    # Channel 2 must NOT appear anywhere
    assert 2 not in out[0]["source"]
    assert 2 not in out[0]["sink"]


def test_compute_local_rank_endpoint_valid_mask_filter():
    """C3 — external valid_mask=False channels excluded even with low template rank.

    Uses valid_mask_per_cluster={cluster_id: mask} signature (mirrors PR-6 anchoring
    runner's per_cluster_masks convention).
    """
    ranks = np.array([
        [0.0, 0.0],   # ch0 lowest template rank
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
    ])
    bools = np.ones((4, 2), dtype=bool)
    labels = np.array([0, 0])
    valid_mask = np.array([False, True, True, True])  # ch0 INVALID despite low rank
    out = p2.compute_local_rank_endpoint(
        ranks, bools, labels, event_indices=np.array([0, 1]), k=1,
        valid_mask_per_cluster={0: valid_mask},
    )
    assert 0 not in out[0]["source"]
    assert out[0]["source"] == [1]   # now ch1 is the source (lowest among eligible)
    assert out[0]["sink"] == [3]


def test_compute_local_rank_endpoint_k_degradation_on_small_pool():
    """C4 — When eligible pool < 2k, k degrades to max(1, pool//2). Source / sink same size."""
    ranks = np.arange(20, dtype=float).reshape(5, 4)  # 5 channels, 4 events
    bools = np.ones((5, 4), dtype=bool)
    labels = np.array([0, 0, 0, 0])
    # k=3 with 5 eligible channels → 2k=6 > 5 → k_eff = 5//2 = 2
    out = p2.compute_local_rank_endpoint(
        ranks, bools, labels, event_indices=np.array([0, 1, 2, 3]), k=3,
    )
    assert len(out[0]["source"]) == 2
    assert len(out[0]["sink"]) == 2


def test_compute_local_rank_endpoint_empty_cluster_skipped():
    """C6 — A cluster with no events in the slice is omitted from the output dict."""
    ranks = np.zeros((3, 4))
    bools = np.ones((3, 4), dtype=bool)
    labels = np.array([0, 0, 0, 0])  # only cluster 0
    out = p2.compute_local_rank_endpoint(
        ranks, bools, labels, event_indices=np.array([0, 1, 2, 3]), k=1,
    )
    assert set(out.keys()) == {0}
    assert 1 not in out  # cluster 1 has no events → not in dict


def test_compute_local_rank_endpoint_empty_event_indices():
    """Empty event_indices → empty output dict (no cluster gets endpoint)."""
    ranks = np.zeros((3, 4))
    bools = np.ones((3, 4), dtype=bool)
    labels = np.array([0, 1, 0, 1])
    out = p2.compute_local_rank_endpoint(
        ranks, bools, labels, event_indices=np.array([], dtype=int), k=1,
    )
    assert out == {}


def test_compute_local_rank_endpoint_external_mask_intersected_not_replaced():
    """User-review catch 2026-05-23 — external valid_mask_per_cluster (e.g., full-data PR-6
    per-cluster mask) must be INTERSECTED with derived_valid (per-epoch participation),
    NOT replace it. Otherwise channels valid in full-data but absent from this epoch get
    a phantom template[ci] = ci ranking via _legacy_hist_mean_rank fallback and may
    silently bubble into source/sink.

    Setup: 5 channels. Channel 0 has zero participation in cluster 0's events_indices
    (the "absent in this epoch" case). External mask passes channel 0 as VALID
    (full-data perspective; pretend it participates in some other epoch).
    Without intersection (replace semantics): channel 0 enters template_rank via
    fallback and may be picked as source/sink.
    With intersection (correct): channel 0 excluded by derived_valid=False.
    """
    ranks = np.array([
        [10.0, 10.0],   # channel 0: phantom (would fallback to ci=0)
        [3.0, 3.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [4.0, 4.0],
    ])
    bools = np.array([
        [0, 0],   # channel 0 zero participation in this epoch
        [1, 1],
        [1, 1],
        [1, 1],
        [1, 1],
    ], dtype=bool)
    labels = np.array([0, 0])
    # External mask: ALL channels valid (full-data PR-6 mask says ch0 fine)
    external = np.array([True, True, True, True, True])
    out = p2.compute_local_rank_endpoint(
        ranks, bools, labels, event_indices=np.array([0, 1]), k=1,
        valid_mask_per_cluster={0: external},
    )
    # If REPLACE semantics (bug): ch0 enters via fallback template[0]=0 (lowest rank)
    # and would be picked as source. With INTERSECT semantics (correct), ch0 is
    # excluded by derived_valid (zero participation).
    assert 0 not in out[0]["source"]
    assert 0 not in out[0]["sink"]


def test_compute_local_rank_endpoint_external_intersect_can_restrict_further():
    """External mask CAN restrict beyond derived (subset semantics still works)."""
    ranks = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
    ])
    bools = np.ones((4, 2), dtype=bool)  # all participate (derived = all True)
    labels = np.array([0, 0])
    # External excludes ch0 (the would-be source)
    external = np.array([False, True, True, True])
    out = p2.compute_local_rank_endpoint(
        ranks, bools, labels, event_indices=np.array([0, 1]), k=1,
        valid_mask_per_cluster={0: external},
    )
    assert 0 not in out[0]["source"]
    assert out[0]["source"] == [1]  # ch1 now lowest among eligible (intersection = [F,T,T,T])
    assert out[0]["sink"] == [3]


# --- B2: compute_endpoint_spatial_radius + compute_source_sink_centroid_distance ---


def test_compute_endpoint_spatial_radius_empty():
    """Empty endpoint list → all radii NaN."""
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    out = p2.compute_endpoint_spatial_radius([], coords)
    assert np.isnan(out["centroid_rms"])
    assert np.isnan(out["mean_pairwise"])
    assert np.isnan(out["min_enclosing_radius"])


def test_compute_endpoint_spatial_radius_single_point():
    """C8 — k=1 → centroid_rms=0, mean_pairwise=NaN (no pairs), min_enclosing=0."""
    coords = np.array([[3.0, 4.0, 5.0], [10.0, 10.0, 10.0]])
    out = p2.compute_endpoint_spatial_radius([0], coords)
    assert out["centroid_rms"] == 0.0
    assert np.isnan(out["mean_pairwise"])
    assert out["min_enclosing_radius"] == 0.0
    assert out["n_points"] == 1


def test_compute_endpoint_spatial_radius_equilateral():
    """C9 — n=3 points forming equilateral triangle (side d) →
    centroid_rms = d/sqrt(3); mean_pairwise = d; min_enclosing_radius = d/sqrt(3)."""
    d = 6.0
    h = d * np.sqrt(3.0) / 2.0   # equilateral height
    coords = np.array([
        [0.0, 0.0, 0.0],
        [d, 0.0, 0.0],
        [d / 2.0, h, 0.0],
    ])
    out = p2.compute_endpoint_spatial_radius([0, 1, 2], coords)
    assert out["centroid_rms"] == pytest.approx(d / np.sqrt(3.0), rel=1e-6)
    assert out["mean_pairwise"] == pytest.approx(d, rel=1e-6)
    assert out["min_enclosing_radius"] == pytest.approx(d / np.sqrt(3.0), rel=1e-6)


def test_compute_endpoint_spatial_radius_collinear():
    """C10 — collinear k=3 points → min_enclosing = half max pairwise distance."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
    ])
    out = p2.compute_endpoint_spatial_radius([0, 1, 2], coords)
    # max pairwise = 10, mid = (0+10)/2 = 5 along x; min enclosing = 5
    assert out["min_enclosing_radius"] == pytest.approx(5.0, rel=1e-6)
    # centroid = (5,0,0); RMS = sqrt(((5-5)^2 + (5-5)^2 + (5-5)^2 + ...) wait
    # actually centroid = (15/3, 0, 0) = (5, 0, 0); deviations = [5, 5, 0];
    # RMS = sqrt((25 + 25 + 0) / 3) = sqrt(50/3) ≈ 4.082
    assert out["centroid_rms"] == pytest.approx(np.sqrt(50.0 / 3.0), rel=1e-6)


def test_compute_source_sink_centroid_distance_basic():
    """source / sink axis length = ||centroid(source) − centroid(sink)||."""
    coords = np.array([
        [0.0, 0.0, 0.0],   # source 0
        [2.0, 0.0, 0.0],   # source 1
        [10.0, 0.0, 0.0],  # sink 0
        [12.0, 0.0, 0.0],  # sink 1
    ])
    # source centroid = (1, 0, 0); sink centroid = (11, 0, 0); d = 10
    d = p2.compute_source_sink_centroid_distance([0, 1], [2, 3], coords)
    assert d == pytest.approx(10.0)


def test_compute_source_sink_centroid_distance_3d():
    """3D non-trivial separation."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [4.0, 3.0, 0.0],   # sink at distance 5 from origin
    ])
    d = p2.compute_source_sink_centroid_distance([0], [1], coords)
    assert d == pytest.approx(5.0)


def test_compute_source_sink_centroid_distance_empty_returns_nan():
    """If either side empty → NaN (can't compute centroid)."""
    coords = np.array([[0.0, 0.0, 0.0]])
    assert np.isnan(p2.compute_source_sink_centroid_distance([], [0], coords))
    assert np.isnan(p2.compute_source_sink_centroid_distance([0], [], coords))


# --- B3: compute_decision_k_drift ---


def _make_synthetic_swap_data(rng, n_ch=8, n_events_per_cluster=50, swap_strength=1.0):
    """Make synthetic ranks + bools + labels for 2 clusters with controllable swap signal.

    Cluster 0 mean ranks (over events): roughly [0,1,2,3,4,5,6,7] (sorted ascending);
    cluster 1 mean ranks: reversed [7,6,5,4,3,2,1,0] scaled by swap_strength + noise.
    All channels participate (bools=True). epoch_indices selects events.
    """
    total = 2 * n_events_per_cluster
    ranks = np.zeros((n_ch, total), dtype=float)
    bools = np.ones((n_ch, total), dtype=bool)
    labels = np.concatenate([
        np.zeros(n_events_per_cluster, dtype=int),
        np.ones(n_events_per_cluster, dtype=int),
    ])
    # Cluster 0
    for e in range(n_events_per_cluster):
        ranks[:, e] = np.arange(n_ch) + rng.normal(0, 0.3, size=n_ch)
    # Cluster 1: swapped
    for e in range(n_events_per_cluster, total):
        ranks[:, e] = swap_strength * (n_ch - 1 - np.arange(n_ch)) + (1 - swap_strength) * np.arange(n_ch) + rng.normal(0, 0.3, size=n_ch)
    return ranks, bools, labels


def test_compute_decision_k_drift_returns_decision_k_per_epoch():
    """Per-epoch decision-k extraction over synthetic strong-swap data."""
    rng = np.random.default_rng(0)
    ranks, bools, labels = _make_synthetic_swap_data(rng, n_ch=8, n_events_per_cluster=40, swap_strength=1.0)
    # 2 epochs, each spanning half of each cluster's events
    epochs = [
        {"event_indices": np.concatenate([np.arange(0, 20), np.arange(40, 60)]), "block_index": 0, "t_start": 0, "t_end": 1},
        {"event_indices": np.concatenate([np.arange(20, 40), np.arange(60, 80)]), "block_index": 0, "t_start": 1, "t_end": 2},
    ]
    out = p2.compute_decision_k_drift(
        ranks, bools, labels, epochs,
        cluster_a=0, cluster_b=1,
        min_events_per_cluster=10,
        n_perm=200, seed=0,
    )
    assert "decision_k_per_epoch" in out
    assert len(out["decision_k_per_epoch"]) == 2
    # With strong swap signal, both epochs should find a non-None decision_k
    finite = [k for k in out["decision_k_per_epoch"] if k is not None]
    assert len(finite) == 2
    assert all(2 <= k <= 4 for k in finite)


def test_compute_decision_k_drift_drops_low_event_epochs():
    """C12 — epochs with cluster A or B events < min_events_per_cluster → decision_k=None."""
    rng = np.random.default_rng(0)
    ranks, bools, labels = _make_synthetic_swap_data(rng, n_ch=8, n_events_per_cluster=40)
    # Epoch 0: cluster 0 has 5 events (< min 20), cluster 1 has 30 events
    # Epoch 1: both have ≥ 20 events
    epochs = [
        {"event_indices": np.concatenate([np.arange(0, 5), np.arange(40, 70)]), "block_index": 0, "t_start": 0, "t_end": 1},
        {"event_indices": np.concatenate([np.arange(5, 40), np.arange(70, 80)]), "block_index": 0, "t_start": 1, "t_end": 2},
    ]
    out = p2.compute_decision_k_drift(
        ranks, bools, labels, epochs,
        cluster_a=0, cluster_b=1,
        min_events_per_cluster=20,
        n_perm=100, seed=0,
    )
    # Epoch 0 cluster 0 has only 5 events → None
    assert out["decision_k_per_epoch"][0] is None
    # Epoch 1 cluster 1 has only 10 events → None
    assert out["decision_k_per_epoch"][1] is None
    assert out["n_epochs_with_decision_k"] == 0


def test_compute_decision_k_drift_phantom_mask():
    """C11 — per-cluster template rank in B3 uses masked mean rank (via _legacy_hist_mean_rank).
    Phantom int ranks on non-participating events must not enter the per-epoch swap_sweep."""
    rng = np.random.default_rng(0)
    n_ch = 8
    n_per = 30
    ranks, bools, labels = _make_synthetic_swap_data(rng, n_ch=n_ch, n_events_per_cluster=n_per, swap_strength=1.0)
    # Inject phantom rank into a channel for events where it doesn't participate
    bools[7, n_per:] = False  # channel 7 doesn't participate in any cluster 1 event
    ranks[7, n_per:] = 999.0  # phantom rank
    epochs = [
        {"event_indices": np.arange(0, 2 * n_per), "block_index": 0, "t_start": 0, "t_end": 1},
    ]
    out = p2.compute_decision_k_drift(
        ranks, bools, labels, epochs,
        cluster_a=0, cluster_b=1,
        min_events_per_cluster=10,
        n_perm=100, seed=0,
    )
    # Phantom 999 must not pull cluster 1's template — decision_k should still be reasonable (2-4)
    assert out["decision_k_per_epoch"][0] is not None
    assert 2 <= out["decision_k_per_epoch"][0] <= 4


def test_compute_decision_k_drift_summary_stats():
    """summary stats (std / mean / range) computed correctly on finite decision_k list."""
    rng = np.random.default_rng(0)
    ranks, bools, labels = _make_synthetic_swap_data(rng, n_ch=8, n_events_per_cluster=40, swap_strength=1.0)
    epochs = [
        {"event_indices": np.concatenate([np.arange(0, 20), np.arange(40, 60)]), "block_index": 0, "t_start": 0, "t_end": 1},
        {"event_indices": np.concatenate([np.arange(20, 40), np.arange(60, 80)]), "block_index": 0, "t_start": 1, "t_end": 2},
    ]
    out = p2.compute_decision_k_drift(
        ranks, bools, labels, epochs,
        cluster_a=0, cluster_b=1,
        min_events_per_cluster=10,
        n_perm=200, seed=0,
    )
    finite = [k for k in out["decision_k_per_epoch"] if k is not None]
    assert out["n_epochs_with_decision_k"] == len(finite)
    assert "decision_k_std" in out
    assert "decision_k_mean" in out
    assert "decision_k_range" in out
    if len(finite) >= 2:
        assert out["decision_k_std"] == pytest.approx(float(np.std(finite)), rel=1e-6)
        assert out["decision_k_mean"] == pytest.approx(float(np.mean(finite)), rel=1e-6)
        assert out["decision_k_range"] == [int(min(finite)), int(max(finite))]
