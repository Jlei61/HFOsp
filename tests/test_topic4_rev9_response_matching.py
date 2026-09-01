import numpy as np

from src.topic4_rev9_response_matching import (positive_map_js_distance,
                                                pseudo_huber_squared,
                                                robust_scale,
                                                scalar_pair_loss)


def test_robust_scale_falls_back_for_discrete_values():
    assert np.isclose(robust_scale([0.0, 0.0, 1.0]), np.std([0.0, 0.0, 1.0]))


def test_positive_map_js_distance_contract():
    assert positive_map_js_distance([0.0, 0.0], [0.0, 0.0]) == 0.0
    assert positive_map_js_distance([1.0, 0.0], [0.0, 1.0]) == 1.0
    assert np.isclose(
        positive_map_js_distance([1.0, 2.0], [1.0, 2.0]), 0.0)


def test_scalar_pair_loss_ignores_missing_but_reports_feature_count():
    loss, count = scalar_pair_loss(
        [0.0, np.nan, 2.0], [1.0, 5.0, 2.0], [1.0, 1.0, 2.0])
    assert count == 2
    assert np.isclose(loss, np.mean(pseudo_huber_squared([1.0, 0.0])))
