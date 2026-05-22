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
