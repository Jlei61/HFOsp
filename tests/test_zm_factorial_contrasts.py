"""The 2x2 connectivity factorial must censor, not delete.

An arm that fails to transition within the cap carries the strongest possible
evidence about that arm. Dropping it -- which `paired_onset_difference` does --
compares the arms on the subset that happened to enter, i.e. exactly the subset
the arms are supposed to differ on.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic4_zm_statistics import (  # noqa: E402
    factorial_contrasts, paired_sign_flip_test)

CAP = 20000.0
SEEDS = list(range(1811, 1823))


def _arms(node, ee, etoi, joint):
    return {"Node": {s: node for s in SEEDS}, "Node+EE": {s: ee for s in SEEDS},
            "Node+EtoI": {s: etoi for s in SEEDS}, "Joint": {s: joint for s in SEEDS}}


def test_a_non_entering_run_is_censored_at_the_cap_not_dropped():
    arms = _arms(9000.0, 6000.0, 4300.0, 4000.0)
    arms["Node"][1811] = None                     # never transitioned
    report = factorial_contrasts(arms, cap_ms=CAP)
    assert report["per_arm"]["Node"]["n_censored"] == 1
    assert report["per_arm"]["Node"]["entered_fraction"] == pytest.approx(11 / 12)
    # the censored seed still contributes, at the cap
    assert report["contrasts"]["delta_EE"]["per_seed_ms"][1811] == pytest.approx(
        6000.0 - CAP)
    assert report["contrasts"]["delta_EE"]["sign_flip"]["n"] == 12


def test_an_onset_beyond_the_cap_is_clamped():
    arms = _arms(25000.0, 6000.0, 4300.0, 4000.0)
    report = factorial_contrasts(arms, cap_ms=CAP)
    assert report["per_arm"]["Node"]["restricted_mean_ms"] == pytest.approx(CAP)


def test_the_interaction_is_the_declared_combination():
    arms = _arms(9000.0, 6000.0, 4300.0, 4000.0)
    report = factorial_contrasts(arms, cap_ms=CAP)
    expected = 4000.0 - 6000.0 - 4300.0 + 9000.0
    assert report["contrasts"]["interaction"]["mean_ms"] == pytest.approx(expected)
    assert report["contrasts"]["delta_EE"]["mean_ms"] == pytest.approx(-3000.0)
    assert report["contrasts"]["delta_EtoI"]["mean_ms"] == pytest.approx(-4700.0)


def test_unpaired_arms_are_refused():
    arms = _arms(9000.0, 6000.0, 4300.0, 4000.0)
    del arms["Joint"][1822]
    with pytest.raises(ValueError, match="paired by network seed"):
        factorial_contrasts(arms, cap_ms=CAP)


def test_a_missing_arm_is_refused():
    arms = _arms(9000.0, 6000.0, 4300.0, 4000.0)
    del arms["Node+EtoI"]
    with pytest.raises(ValueError, match="missing"):
        factorial_contrasts(arms, cap_ms=CAP)


def test_twelve_seeds_give_an_exact_sign_flip_test():
    values = np.full(12, -3000.0)
    out = paired_sign_flip_test(values, draws=4096)
    assert out["exact"] is True
    assert out["n_permutations"] == 4096
    assert out["p_two_sided"] == pytest.approx(2.0 / 4096, rel=0.2)


def test_a_null_contrast_is_not_significant():
    rng = np.random.default_rng(0)
    values = rng.normal(0.0, 500.0, 12)
    out = paired_sign_flip_test(values, draws=4096)
    assert out["p_two_sided"] > 0.05


def test_sign_flip_p_can_never_be_zero():
    out = paired_sign_flip_test(np.full(12, 1e9), draws=4096)
    assert out["p_two_sided"] > 0.0


def test_large_designs_fall_back_to_sampling():
    out = paired_sign_flip_test(np.full(30, -1.0), draws=4096)
    assert out["exact"] is False
    assert out["n_permutations"] == 4096


def test_every_contrast_carries_the_meaning_of_its_sign():
    """The endpoint is a time. Without this, "delta_EE = -3000 ms" reads as
    "E->E weakened the effect" when it means the opposite."""
    report = factorial_contrasts(_arms(9000.0, 6000.0, 4300.0, 4000.0), cap_ms=CAP)
    for name, entry in report["contrasts"].items():
        assert "EARLIER" in entry["sign_meaning"] or "earlier" in entry["sign_meaning"]
        assert entry["n_seeds_earlier"] == (12 if entry["mean_ms"] < 0 else 0)


def test_a_super_additive_interaction_is_labelled_as_such():
    # Joint much earlier than the two separate effects predict
    report = factorial_contrasts(_arms(9000.0, 6000.0, 4300.0, 500.0), cap_ms=CAP)
    interaction = report["contrasts"]["interaction"]
    assert interaction["mean_ms"] < 0
    assert "super-additive" in interaction["sign_meaning"]
