"""Regression tests for the two review fixes: contrast labels and multiplicity.

Both defects were invisible to every existing gate. The label defect shipped
144 blocks saying ``n_physical_better`` under a contrast that never involved
the physical clock, and the package's own audit passed because it only checks
that keys parse and values are finite. The multiplicity defect shipped 288 raw
sign-test p-values with no adjusted companion, one of which a machine reader
then lifted into a headline. A schema check cannot catch either; only a test
that asserts the meaning can.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.topic5_continuous_marked_state.multiplicity import (  # noqa: E402
    annotate_family, benjamini_hochberg, holm)


def _load_clock_analysis():
    """Import the analysis script by path -- it is a script, not a package."""
    path = REPO / "scripts/topic5_continuous_marked_state/analyze_physical_vs_event_count_clock.py"
    spec = importlib.util.spec_from_file_location("_clock_analysis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- labels ----

def test_summary_keys_name_the_arms_actually_differenced():
    summary = _load_clock_analysis()._summary
    values = np.array([-1.0, -1.0, -1.0, 2.0])

    physical = summary(values, left_arm="physical_time", right_arm="event_count")
    assert physical["difference"] == "physical_time_minus_event_count"
    assert physical["n_physical_time_better"] == 3
    assert physical["n_event_count_better"] == 1

    # The same numbers under a different contrast must NOT reuse the old keys.
    count = summary(values, left_arm="event_count", right_arm="current_event")
    assert count["difference"] == "event_count_minus_current_event"
    assert count["n_event_count_better"] == 3
    assert count["n_current_event_better"] == 1
    assert "n_physical_time_better" not in count, (
        "the physical clock is not one of the arms in this contrast")
    assert "median_physical_minus_event_count" not in count, (
        "the hard-typed key from the shipped defect came back")


def test_summary_sign_convention_is_lower_nll_wins_for_the_left_arm():
    summary = _load_clock_analysis()._summary
    out = summary(np.array([-3.0, -2.0, -1.0]),
                  left_arm="physical_time", right_arm="event_count")
    assert out["lower_is_better_for"] == "physical_time"
    assert out["median_delta"] < 0 and out["n_physical_time_better"] == 3
    assert out["n_negative"] == 3 and out["n_positive"] == 0


# ---------------------------------------------------------- multiplicity ----

def test_holm_matches_statsmodels_on_the_real_shape_of_this_family():
    statsmodels = pytest.importorskip("statsmodels.stats.multitest")
    rng = np.random.default_rng(0)
    # 288 tests with heavy ties is the actual family; ties are where a
    # hand-rolled step-down most easily goes wrong.
    p = np.round(rng.beta(0.3, 3.0, size=288), 4)
    expected = statsmodels.multipletests(p, method="holm")[1]
    assert np.allclose(holm(list(p)), expected, atol=1e-12)


def test_benjamini_hochberg_matches_statsmodels_including_ties():
    statsmodels = pytest.importorskip("statsmodels.stats.multitest")
    rng = np.random.default_rng(1)
    p = np.round(rng.beta(0.3, 3.0, size=288), 4)
    expected = statsmodels.multipletests(p, method="fdr_bh")[1]
    assert np.allclose(benjamini_hochberg(list(p)), expected, atol=1e-12)


def test_corrections_are_monotone_and_never_smaller_than_raw():
    p = [0.0008, 0.0008, 0.01, 0.04, 0.2, 0.9]
    h, q = holm(p), benjamini_hochberg(p)
    assert all(a >= b - 1e-15 for a, b in zip(h, p)), "Holm went below raw"
    assert all(a >= b - 1e-15 for a, b in zip(q, p)), "BH went below raw"
    assert all(a >= b - 1e-15 for a, b in zip(h, q)), "Holm must be >= BH"
    order = np.argsort(p, kind="stable")
    assert np.all(np.diff(np.asarray(h)[order]) >= -1e-15)
    assert np.all(np.diff(np.asarray(q)[order]) >= -1e-15)


def test_the_headline_cell_costs_what_the_review_said_it_costs():
    """The p=8.2e-4 cell is one of a family of 288 and must not read as 8e-4."""
    rng = np.random.default_rng(7)
    p = list(np.round(rng.beta(0.3, 3.0, size=287), 4)) + [0.00082]
    assert holm(p)[-1] > 0.05, "Holm should not clear this cell on its own"
    assert benjamini_hochberg(p)[-1] < holm(p)[-1], "BH must be the looser lens"


def test_none_passes_through_and_does_not_enter_the_denominator():
    p = [0.01, None, 0.02]
    assert holm(p)[1] is None and benjamini_hochberg(p)[1] is None
    assert np.isclose(holm(p)[0], holm([0.01, 0.02])[0]), (
        "a skipped cell inflated the family size")


def test_annotate_family_writes_back_and_counts_what_it_wrote():
    summaries = [{"two_sided_exact_sign_p_unadjusted": v}
                 for v in (0.001, 0.02, 0.2, None)]
    index = annotate_family(
        ((("cell", str(i)), s) for i, s in enumerate(summaries)),
        family_name="unit_test")
    assert index["n_tests"] == 3 and index["n_raw_below_0p05"] == 2
    for s in summaries:
        assert s["multiplicity_family"] == "unit_test"
        assert "holm_adjusted_p" in s and "bh_adjusted_q" in s
    assert summaries[-1]["holm_adjusted_p"] is None
    assert summaries[0]["holm_adjusted_p"] >= summaries[0][
        "two_sided_exact_sign_p_unadjusted"]
