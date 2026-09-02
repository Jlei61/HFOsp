"""The two statistical traps the previous round fell into, pinned here.

Both are properties of the *estimator*, not of any particular data, so they are tested
on constructed inputs where the right answer is known by hand.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location(
    "aggregate_v0_4", ROOT / "scripts/aggregate_topic5_motif_autonomous_v0_4.py")
aggregate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aggregate)


# -- a bound sitting on zero is not an exclusion ----------------------------


def test_a_lower_bound_of_exactly_zero_is_reported_as_touching_zero():
    """The warm-start spike puts the bound exactly on zero.

    Testing ``low < 0 < high`` reads that as excluding zero, which turned a null into
    a finding in the previous round.
    """
    assert aggregate.interval_verdict(0.0, 0.014) == "touches_zero"
    assert aggregate.interval_verdict(-0.014, 0.0) == "touches_zero"


def test_a_strictly_positive_bound_still_counts_as_an_exclusion():
    assert aggregate.interval_verdict(1e-9, 0.014) == "excludes_zero_strictly"
    assert aggregate.interval_verdict(-0.02, -1e-9) == "excludes_zero_strictly"


def test_an_interval_around_zero_spans_it():
    assert aggregate.interval_verdict(-0.01, 0.02) == "spans_zero"


def test_the_spike_actually_produces_a_bound_on_zero():
    """Not a hypothetical: a cohort where most children reproduce their parent."""
    values = np.array([0.0] * 18 + [0.004, 0.011, 0.002, 0.008, 0.001])
    low, high = aggregate.bootstrap_median(values)
    assert low == 0.0
    assert aggregate.interval_verdict(low, high) == "touches_zero"


# -- ties are evidence of no gain, not missing data -------------------------


def test_ties_stay_in_the_denominator_of_the_sign_test():
    """Dropping them would turn 5 wins out of 23 into 5 out of 5.

    This case also pins the direction guard: with ties in the denominator, 5 positives
    out of 23 is significantly *few*, so the two-sided p is small while the child is the
    worse model.  A verdict read off the p alone would invert the finding.
    """
    import pandas as pd

    values = [0.0] * 18 + [0.004, 0.011, 0.002, 0.008, 0.001]
    table = pd.DataFrame({
        "patient": [f"p{i}" for i in range(len(values))],
        "comparison": ["M1_over_M0"] * len(values),
        "plain_language": [""] * len(values),
        "improvement_nats": values,
        "start_spread_child": [0.001] * len(values),
    })
    block = aggregate.summarise(table)["comparisons"][0]
    assert block["n_patients"] == 23
    assert (block["positive"], block["negative"], block["tied"]) == (5, 0, 18)
    # dropping the ties would leave 5 of 5 and read as perfect consistency
    assert block["sign_test_p"] < 0.05          # small, but pointing the other way
    # with 18 of 23 patients showing literally no change the median is zero, which is
    # the honest summary: the extra mechanism did nothing on most of the cohort
    assert block["median_improvement_nats"] == 0.0
    assert block["direction"] == "no_difference"
    assert block["supports_improvement"] is False   # the interval only touches zero


# -- Holm ------------------------------------------------------------------


def test_a_cohort_where_the_child_is_mostly_worse_is_never_called_support():
    """The failure mode the direction guard exists for."""
    import pandas as pd

    values = [-0.01] * 18 + [0.002] * 5
    table = pd.DataFrame({
        "patient": [f"p{i}" for i in range(len(values))],
        "comparison": ["M1_over_M0"] * len(values),
        "plain_language": [""] * len(values),
        "improvement_nats": values,
        "start_spread_child": [0.001] * len(values),
    })
    block = aggregate.summarise(table)["comparisons"][0]
    assert block["sign_test_p"] < 0.05
    assert block["direction"] == "child_worse"
    assert block["supports_improvement"] is False


def test_holm_is_monotone_and_never_exceeds_one():
    adjusted = aggregate.holm([0.01, 0.02, 0.6])
    assert adjusted == pytest.approx([0.03, 0.04, 0.6])
    assert aggregate.holm([0.4, 0.5, 0.9]) == pytest.approx([1.0, 1.0, 1.0])


def test_holm_keeps_the_ordering_of_the_raw_values():
    raw = [0.2, 0.001, 0.05]
    adjusted = aggregate.holm(raw)
    assert np.argsort(adjusted).tolist() == np.argsort(raw).tolist()


# -- the improvement points the right way ----------------------------------


def test_a_child_with_a_lower_loss_scores_as_an_improvement():
    """Negative log likelihood: lower is better, so parent minus child must be positive."""
    import pandas as pd

    table = pd.DataFrame({
        "patient": ["p0", "p0"],
        "arm": ["M0_ISOTROPIC_DIFFUSION", "M1_AXIAL_CORRIDOR"],
        "test_primary_nll": [2.10, 2.05],
        "start_spread": [0.0, 0.001],
    })
    step = aggregate.increments(table)
    assert float(step.improvement_nats.iloc[0]) == pytest.approx(0.05)
