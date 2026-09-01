import numpy as np

from scripts.compute_topic4_zm_bridge_selection_null import (
    _permutation, _permute_target)


def test_within_shaft_null_never_moves_contact_between_shafts():
    shafts = np.array(["SCL"] * 4 + ["ICL"] * 11)
    permutation = _permutation(np.random.default_rng(4), shafts, "within_shaft")
    assert np.all(permutation[:4] < 4)
    assert np.all(permutation[4:] >= 4)


def test_target_permutation_moves_medians_and_scales_together():
    endpoint = {
        "median": np.arange(4.0), "q025": np.arange(4.0) - 1,
        "q975": np.arange(4.0) + 1, "bootstrap_iqr": np.arange(4.0) + 2,
    }
    target = {"pre": endpoint, "early": endpoint, "increment": endpoint,
              "global_early_per_seizure": [1, 2, 3],
              "positive_fraction_per_seizure": [1, 1, 1],
              "contact_iqr_per_seizure": [1, 2, 3]}
    permutation = np.array([3, 2, 1, 0])
    got = _permute_target(target, permutation)
    np.testing.assert_array_equal(got["early"]["median"], [3, 2, 1, 0])
    np.testing.assert_array_equal(got["early"]["bootstrap_iqr"], [5, 4, 3, 2])
