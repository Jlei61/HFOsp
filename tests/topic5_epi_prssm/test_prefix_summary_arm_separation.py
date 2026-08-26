"""Regression tests for the ambiguous-prefix summary.

Two defects were found in review (2026-08-19):
  1. the raw field is ``swapped_nll - correct_nll``, so positive already means the
     correct state helped; the aggregator negated it a second time;
  2. all thirteen arms were pooled into one median, and seven of them have a
     frozen state whose gain is exactly zero, so the median was zero by
     construction and erased the real effect.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/topic5_epi_prssm"))

import aggregate_event_distribution as agg  # noqa: E402


def _frame(rows):
    return pd.DataFrame(rows)


def test_frozen_arms_do_not_dilute_the_moving_arms():
    rows = []
    for subject in [f"s{i}" for i in range(10)]:
        rows.append({"arm": "node_film_g2", "subject": subject, "prefix_depth": 1,
                     "state_gain": 0.02})
        for frozen in ("node_film_frozen", "edge_gate_frozen", "no_state"):
            rows.append({"arm": frozen, "subject": subject, "prefix_depth": 1,
                         "state_gain": 0.0})
    out = agg._prefix_summary(_frame(rows))
    moving = out["by_arm"]["node_film_g2"][1]
    assert moving["median_delta"] == pytest.approx(0.02)
    assert moving["n_favourable"] == 10
    for frozen in ("node_film_frozen", "edge_gate_frozen", "no_state"):
        assert out["negative_control_arms"][frozen][1]["median_delta"] == pytest.approx(0.0)


def test_moving_and_frozen_arms_are_classified_correctly():
    assert agg._is_moving_state_arm("node_film_g2")
    assert agg._is_moving_state_arm("edge_gate_g0")
    assert not agg._is_moving_state_arm("node_film_frozen")
    assert not agg._is_moving_state_arm("no_state")


def test_sign_convention_positive_means_the_correct_state_helped():
    """The producer writes swapped - correct; a positive value must stay positive."""
    correct, swapped = 1.11571, 1.19677
    produced = swapped - correct
    rows = [{"arm": "node_film_g2", "subject": "s0", "prefix_depth": 1,
             "state_gain": produced}]
    out = agg._prefix_summary(_frame(rows))
    assert out["by_arm"]["node_film_g2"][1]["median_delta"] > 0
