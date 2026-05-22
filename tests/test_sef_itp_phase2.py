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
    assert p2.__version__ == "v1.0.0"


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
