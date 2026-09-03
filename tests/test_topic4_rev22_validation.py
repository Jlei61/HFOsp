import numpy as np

from src.topic4_rev22_validation import (
    C2ST_NOT_ESTIMABLE, RECALL_LOW_YIELD, RECALL_OK,
    REFERENCE_SUPPORT_BUDGET_NOT_ESTIMABLE, calibrate_coverage_radius,
    classifier_two_sample_auc, fixed_budget_recall, freeze_reference_support_budget,
    paired_unit_bootstrap,
)


def test_reference_support_budget_uses_minimum_and_fails_closed():
    result = freeze_reference_support_budget([38, 43, 22, 40], expected_units=4, min_events=12)
    assert result["status"] == "OK" and result["n_cov"] == 22
    missing = freeze_reference_support_budget([38, 43, 22], expected_units=4, min_events=12)
    assert missing["status"] == REFERENCE_SUPPORT_BUDGET_NOT_ESTIMABLE
    low = freeze_reference_support_budget([38, 43, 11, 40], expected_units=4, min_events=12)
    assert low["status"] == REFERENCE_SUPPORT_BUDGET_NOT_ESTIMABLE and low["n_cov"] == 11


def test_paired_unit_bootstrap_preserves_pairing_and_missing_units_fail_closed():
    left = np.asarray([4.0, 5.0, 8.0, 9.0])
    right = np.asarray([3.0, 4.0, 7.0, 8.0])
    result = paired_unit_bootstrap(left, right, draws=512, seed=3)
    assert result["status"] == "OK" and result["delta"] == 1.0
    lower_better = paired_unit_bootstrap(right, left, draws=512, seed=3, higher_is_better=False)
    assert lower_better["delta"] == 1.0
    missing = paired_unit_bootstrap([1.0, np.nan], [0.0, 0.0], draws=32, seed=1)
    assert missing["status"] == "NOT_ESTIMABLE_MISSING_UNIT" and missing["delta"] is None


def test_radius_calibration_is_deterministic_and_positive():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(2000, 5))
    a = calibrate_coverage_radius(z, 30, draws=20, seed=3)
    b = calibrate_coverage_radius(z, 30, draws=20, seed=3)
    assert a == b and a["r_cov"] > 0 and a["radius_q05"] <= a["r_cov"] <= a["radius_q95"]


def test_recall_fixed_budget_and_low_yield_status():
    rng = np.random.default_rng(1)
    train = rng.normal(size=(2000, 4))
    r = calibrate_coverage_radius(train, 30, draws=20, seed=1)["r_cov"]
    queries = rng.normal(size=(500, 4))
    full = fixed_budget_recall(rng.normal(size=(120, 4)), queries, n_cov=30, r_cov=r, subsamples=20, seed=2)
    shifted = fixed_budget_recall(rng.normal(size=(120, 4)) + 4.0, queries, n_cov=30, r_cov=r, subsamples=20, seed=2)
    assert full["status"] == RECALL_OK and shifted["status"] == RECALL_OK
    assert full["recall"] > shifted["recall"]
    low = fixed_budget_recall(rng.normal(size=(10, 4)), queries, n_cov=30, r_cov=r)
    assert low["status"] == RECALL_LOW_YIELD and low["recall"] is None and low["recall_all_events"] is not None
    exact = fixed_budget_recall(rng.normal(size=(30, 4)), queries, n_cov=30, r_cov=r)
    assert exact["status"] == RECALL_OK and exact["subsamples"] == 1


def test_c2st_separates_shifted_model_and_is_chance_for_matched():
    rng = np.random.default_rng(4)
    patient = rng.normal(size=(600, 6))
    blocks = rng.integers(0, 12, 600)
    matched = rng.normal(size=(200, 6))
    seeds = np.repeat(np.arange(4), 50)
    same = classifier_two_sample_auc(patient, blocks, matched, seeds, seed=0, resamples=5, permutations=5)
    shifted = classifier_two_sample_auc(patient, blocks, matched + 2.0, seeds, seed=0, resamples=5, permutations=5)
    assert same["status"] == "OK" and shifted["status"] == "OK"
    assert shifted["auc"] > 0.9 > same["auc"] + 0.3
    assert abs(same["permutation_auc_median"] - 0.5) < 0.15
    single = classifier_two_sample_auc(patient, blocks, matched, np.zeros(200, int), seed=0)
    assert single["status"] == C2ST_NOT_ESTIMABLE
