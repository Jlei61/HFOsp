"""TDD tests for the pure statistical functions in
scripts/analyze_topic5_scaffold_ab_cohort.py (Topic5 V3d cohort analysis).

SYNTHETIC ONLY: every test hand-builds small arrays. None of these tests read or
write results/topic5_ictal_recruitment/scaffold_ab_switching/ -- a full-cohort
producer batch was actively writing that directory when this test was authored
(see the script's module docstring), so the loaders/CLI are deliberately not
exercised here.
"""
from __future__ import annotations

import numpy as np

from scripts.analyze_topic5_scaffold_ab_cohort import (
    bimodality_coefficient,
    frac_near_zero,
    typing_purity_test,
)


# ---------------------------------------------------------------------------
# frac_near_zero
# ---------------------------------------------------------------------------

def test_frac_near_zero_basic_1d():
    C_AB = np.array([0.1, -0.1, 0.5, -0.5, 0.15])
    present = np.array([True, True, True, True, False])
    # present&finite -> {0.1, -0.1, 0.5, -0.5}; |.|<0.2 -> {0.1, -0.1} -> 2/4
    assert abs(frac_near_zero(C_AB, present) - 0.5) < 1e-9


def test_frac_near_zero_excludes_nan_even_when_present_true():
    C_AB = np.array([0.1, -0.1, 0.5, -0.5, np.nan])
    present = np.array([True, True, True, True, True])
    # NaN dropped from the denominator too -> same 2/4 as the basic case above.
    assert abs(frac_near_zero(C_AB, present) - 0.5) < 1e-9


def test_frac_near_zero_works_on_2d_subject_arrays():
    # Real usage passes whole-subject npz arrays shaped [n_seizures, n_time].
    C_AB = np.array([[0.1, 0.5, np.nan], [-0.5, -0.1, 0.3]])
    present = np.array([[True, True, True], [True, True, False]])
    # present&finite: 0.1, 0.5, -0.5, -0.1 (4 windows); near-zero: 0.1, -0.1 -> 2/4
    assert abs(frac_near_zero(C_AB, present) - 0.5) < 1e-9


def test_frac_near_zero_nan_when_nothing_present():
    C_AB = np.array([0.1, 0.5])
    present = np.array([False, False])
    assert np.isnan(frac_near_zero(C_AB, present))


# ---------------------------------------------------------------------------
# typing_purity_test
# ---------------------------------------------------------------------------

def test_typing_purity_strongly_typed_seizures_are_significant():
    rng = np.random.default_rng(0)
    n_win = 20
    # 4 seizures, alternating clean A / clean B (small jitter so values aren't
    # literally identical, but every window stays on its seizure's labeled side).
    seizures = []
    for i in range(4):
        base = 0.8 if i % 2 == 0 else -0.8
        seizures.append(base + rng.normal(scale=0.02, size=n_win))
    out = typing_purity_test(seizures, n_perm=1000, seed=0)
    assert out["n_seizures_used"] == 4
    assert out["obs_mean_purity"] > 0.95
    assert out["null_p"] < 0.05


def test_typing_purity_unstructured_windows_not_significant():
    rng = np.random.default_rng(1)
    n_win = 20
    n_seiz = 5
    # Each window independently +-0.8 with p=0.5, irrespective of seizure -- this
    # is literally a draw from the null hypothesis itself, so the observed purity
    # should look like a typical null draw, not an outlier.
    seizures = [rng.choice([0.8, -0.8], size=n_win) for _ in range(n_seiz)]
    out = typing_purity_test(seizures, n_perm=1000, seed=1)
    assert out["n_seizures_used"] == n_seiz
    assert out["null_p"] > 0.2


def test_typing_purity_empty_seizure_excluded_from_n_used():
    seizures = [np.array([0.8, 0.8, 0.8]), np.array([]), np.array([-0.8, -0.8, -0.8])]
    out = typing_purity_test(seizures, n_perm=200, seed=2)
    assert out["n_seizures_used"] == 2   # the empty seizure contributes no evidence


def test_typing_purity_all_empty_returns_nan():
    out = typing_purity_test([np.array([]), np.array([])], n_perm=100, seed=0)
    assert out["n_seizures_used"] == 0
    assert np.isnan(out["obs_mean_purity"])
    assert np.isnan(out["null_p"])


# ---------------------------------------------------------------------------
# bimodality_coefficient
# ---------------------------------------------------------------------------

def test_bimodality_coefficient_bimodal_exceeds_threshold():
    net_sides = np.concatenate([np.full(15, 0.7), np.full(15, -0.7)])
    assert bimodality_coefficient(net_sides) > 0.555


def test_bimodality_coefficient_unimodal_is_lower_than_bimodal():
    rng = np.random.default_rng(3)
    net_sides_unimodal = rng.normal(loc=0.6, scale=0.05, size=40)
    net_sides_bimodal = np.concatenate([np.full(15, 0.7), np.full(15, -0.7)])
    bc_uni = bimodality_coefficient(net_sides_unimodal)
    bc_bi = bimodality_coefficient(net_sides_bimodal)
    assert bc_uni < bc_bi
    assert bc_uni < 0.555


def test_bimodality_coefficient_nan_below_min_n():
    assert np.isnan(bimodality_coefficient([0.1, 0.2, 0.3]))


def test_bimodality_coefficient_nan_when_degenerate():
    assert np.isnan(bimodality_coefficient(np.full(10, 0.5)))
