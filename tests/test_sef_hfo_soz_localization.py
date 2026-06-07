"""TDD for SEF-HFO SOZ-localization geometry/rate scores + comparison-A metrics.

Contract: plan §3.1 (geom endpoint primary + source/sink sensitivities),
§3.2 (rate alignment in-universe, missing!=0), §4 (comparison-A AUC + topk).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.sef_hfo_soz_localization import (
    geom_scores,
    align_rate_and_soz,
    comparison_a_subject,
)


def test_geom_endpoint_main_and_sensitivities():
    # plan §9 Task 2 Step 1 reference test
    chans = ["a", "b", "c", "d"]
    pair = {"channel_names": chans, "joint_valid": [True] * 4,
            "rank_a_dense_full": [0, 1, 2, 3], "rank_b_dense_full": [3, 2, 1, 0]}
    s = geom_scores(chans, pair)
    assert s["endpoint"][0] == s["endpoint"][3] == 1.0       # both ends -> 1
    assert abs(s["endpoint"][1] - s["endpoint"][2]) < 1e-9   # middle symmetric
    assert s["source"][0] == 1.0 and s["sink"][0] == 0.0     # directional sensitivities
    assert s["source"][3] == 0.0 and s["sink"][3] == 1.0


def test_geom_scores_realign_pair_channel_order():
    # §6.1: pair.channel_names may differ in order from the queried `channels`.
    chans = ["d", "a"]  # query order differs from pair order
    pair = {"channel_names": ["a", "b", "c", "d"], "joint_valid": [True] * 4,
            "rank_a_dense_full": [0, 1, 2, 3], "rank_b_dense_full": [3, 2, 1, 0]}
    s = geom_scores(chans, pair)
    assert s["source"][0] == 0.0   # 'd' is the sink in template A (rank_a=3)
    assert s["source"][1] == 1.0   # 'a' is the source in template A (rank_a=0)


def test_geom_scores_missing_or_invalid_channel_is_nan():
    chans = ["a", "x", "c"]  # 'x' not in pair; 'c' present but joint_valid False
    pair = {"channel_names": ["a", "b", "c"], "joint_valid": [True, True, False],
            "rank_a_dense_full": [0, 1, 0], "rank_b_dense_full": [1, 0, 0]}
    s = geom_scores(chans, pair)
    assert math.isfinite(s["endpoint"][0])      # 'a' valid
    assert math.isnan(s["endpoint"][1])         # 'x' absent -> NaN (not 0)
    assert math.isnan(s["endpoint"][2])         # 'c' invalid -> NaN


def test_geom_scores_uses_n_valid_not_total_for_normalization():
    # 5 channels but only 3 valid -> dense ranks 0..2, n_valid=3, normalize by (n_valid-1)=2
    chans = ["p", "q", "r"]
    pair = {"channel_names": ["p", "q", "r", "s", "t"],
            "joint_valid": [True, True, True, False, False],
            "rank_a_dense_full": [0, 1, 2, 0, 0], "rank_b_dense_full": [2, 1, 0, 0, 0]}
    s = geom_scores(chans, pair)
    assert s["source"][1] == pytest.approx(1 - 1 / 2)   # middle valid -> 0.5
    assert s["endpoint"][0] == pytest.approx(1.0)        # rank_a=0,rank_b=2 -> both ends


# ---------- rate + SOZ alignment in universe (§3.2 / Task 3) ----------

def test_align_rate_and_soz_in_universe_no_zero_fill():
    universe = ["A1", "A2", "A3"]
    rate_in_universe = {"A1": 50.0, "A2": 40.0, "A3": 30.0}
    soz_core = ["A1", "A3", "Z9"]  # Z9 is SOZ but outside U -> must not appear, not 0-filled
    rate_vec, y = align_rate_and_soz(universe, rate_in_universe, soz_core)
    assert list(rate_vec) == [50.0, 40.0, 30.0]          # aligned to universe order
    assert list(y) == [True, False, True]                # A1,A3 in SOZ; A2 not
    assert int(y.sum()) == 2                              # == |SOZ_core ∩ U|, Z9 excluded
    assert len(rate_vec) == len(y) == len(universe)


def test_align_rate_and_soz_raises_if_universe_channel_missing_rate():
    # contract: every universe channel must have a (bridged) rate; missing -> raise, never 0-fill
    universe = ["A1", "A2"]
    rate_in_universe = {"A1": 50.0}  # A2 missing
    with pytest.raises((KeyError, ValueError)):
        align_rate_and_soz(universe, rate_in_universe, ["A1"])


# ---------- comparison-A per-subject metrics (§4 / Task 4) ----------

def test_comparison_a_perfect_separation_auc_one():
    # SOZ channels get the highest scores -> AUC 1.0
    y = np.array([True, True, False, False, False, False])
    scores = {"rate": np.array([9.0, 8.0, 1.0, 2.0, 3.0, 0.5]),
              "endpoint": np.array([1.0, 0.9, 0.1, 0.2, 0.0, 0.3])}
    out = comparison_a_subject(scores, y)
    assert out["insufficient"] is False
    assert out["auc"]["rate"] == pytest.approx(1.0)
    assert out["auc"]["endpoint"] == pytest.approx(1.0)
    assert out["n_soz_in_u"] == 2 and out["n_universe"] == 6


def test_comparison_a_reversed_score_auc_zero():
    y = np.array([True, True, False, False, False, False])
    scores = {"rate": np.array([0.1, 0.2, 9.0, 8.0, 7.0, 6.0])}  # SOZ lowest
    out = comparison_a_subject(scores, y)
    assert out["auc"]["rate"] == pytest.approx(0.0)


def test_comparison_a_insufficient_when_few_soz_in_universe():
    y = np.array([True, False, False, False, False])  # |SOZ∩U| = 1 < 2
    scores = {"rate": np.array([9.0, 1.0, 2.0, 3.0, 4.0])}
    out = comparison_a_subject(scores, y)
    assert out["insufficient"] is True
    assert "soz" in out["reason"].lower()
    assert out["auc"] == {} or all(v is None for v in out["auc"].values())


def test_comparison_a_insufficient_when_small_universe():
    y = np.array([True, True, False, False])  # |U| = 4 < 5
    scores = {"rate": np.array([9.0, 8.0, 1.0, 2.0])}
    out = comparison_a_subject(scores, y)
    assert out["insufficient"] is True
    assert "universe" in out["reason"].lower() or "|u|" in out["reason"].lower()


def test_comparison_a_topk_overlap_at_k_equals_n_soz():
    # k = |SOZ∩U| = 2; top-2 by rate should be the 2 SOZ channels -> overlap 1.0
    y = np.array([True, True, False, False, False, False])
    scores = {"rate": np.array([9.0, 8.0, 1.0, 2.0, 3.0, 0.5])}
    out = comparison_a_subject(scores, y)
    assert out["topk_overlap"]["rate"] == pytest.approx(1.0)


# ---------- cohort aggregation for comparison-A (Task 5) ----------

def _ps(subject, ds, cov, rate_auc, endp_auc, insufficient=False):
    return {"subject": subject, "dataset": ds, "soz_coverage": cov,
            "insufficient": insufficient,
            "auc": {} if insufficient else {"rate": rate_auc, "endpoint": endp_auc}}


def test_aggregate_comparison_a_paired_wilcoxon_and_counts():
    from src.sef_hfo_soz_localization import aggregate_comparison_a
    per_subject = [
        _ps("s1", "epilepsiae", 1.0, 0.60, 0.75),   # endpoint > rate
        _ps("s2", "epilepsiae", 1.0, 0.55, 0.70),   # endpoint > rate
        _ps("s3", "epilepsiae", 1.0, 0.65, 0.80),   # endpoint > rate
        _ps("s4", "yuquan", 0.5, 0.70, 0.72),        # endpoint slightly > rate
        _ps("s5", "yuquan", 0.4, 0.80, 0.60),        # endpoint < rate
        _ps("bad", "yuquan", 0.0, None, None, insufficient=True),  # excluded
    ]
    agg = aggregate_comparison_a(per_subject)
    assert agg["n_eligible"] == 5            # 'bad' excluded
    assert agg["n_geom_ge_rate"]["endpoint"] == 4   # s1..s4
    assert agg["median_delta_auc"]["endpoint"] > 0  # endpoint median above rate
    assert 0.0 <= agg["wilcoxon_p_endpoint_ge_rate"] <= 1.0
    # per-dataset coverage reported
    assert agg["median_soz_coverage"]["epilepsiae"] == pytest.approx(1.0)
    assert "yuquan" in agg["median_soz_coverage"]
