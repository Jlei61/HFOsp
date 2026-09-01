import numpy as np

from src.topic4_core_field import build_vth
from src.topic4_core_field_rev9 import (assign_frozen_modes,
                                       component_responsibilities,
                                       fit_frozen_mode_classifier,
                                       node_reconstruction_error,
                                       reconstruct_frozen_node)
from src.topic4_core_field_stage3 import n_free


def _theta():
    theta = np.zeros(n_free(3))
    theta[0:2] = [2.0, 2.0]
    theta[5:7] = [5.0, 5.0]
    theta[10:12] = [8.0, 8.0]
    theta[[2, 3, 7, 8, 12, 13]] = np.log(1.0)
    return theta


def test_frozen_node_rebuild_keeps_d_independent_of_h():
    pos = np.array([[1.0, 1.0], [2.0, 2.0], [5.0, 5.0], [8.0, 8.0]])
    out = reconstruct_frozen_node(
        _theta(), pos, n_total=6, target_count=2.0,
        quantile_seed=17, core_mean=17.5, core_std=1.0,
        v_base=18.0, K=3, L=10.0)
    expected = build_vth(out["h"], out["d"], n_total=6, n_E=4, v_base=18.0)
    np.testing.assert_array_equal(out["vtheta"], expected)
    np.testing.assert_array_equal(out["delta_vtheta"], -out["h"] * out["d"])
    assert out["hashes"]["h_vector_sha256"] != out["hashes"]["d_vector_sha256"]


def test_node_reconstruction_error_reports_exact_and_perturbed_cases():
    values = np.array([18.0, 17.5, 18.2])
    exact = node_reconstruction_error(values, values.copy())
    assert exact["exact"] and exact["max_abs_error"] == 0.0
    changed = values.copy()
    changed[1] += 1e-6
    report = node_reconstruction_error(values, changed)
    assert not report["exact"]
    assert np.isclose(report["max_abs_error"], 1e-6)


def test_component_responsibilities_are_soft_and_normalized():
    theta = np.asarray([
        3.0, 3.0, np.log(1.0), np.log(1.0), 0.0,
        7.0, 3.0, np.log(1.0), np.log(1.0), 0.0,
        5.0, 7.0, np.log(1.0), np.log(1.0), 0.0,
        0.0, 0.0,
    ])
    points = np.asarray([[3.0, 3.0], [7.0, 3.0], [5.0, 5.0]])
    result = component_responsibilities(theta, points, K=3, L=10.0)
    assert result["responsibilities"].shape == (3, 3)
    assert np.allclose(result["responsibilities"].sum(axis=1), 1.0)
    assert result["assignments"].tolist()[:2] == [0, 1]
    assert np.all((result["maximum_responsibility"] > 0.0)
                  & (result["maximum_responsibility"] <= 1.0))


def _identity_reference(n_features=2):
    return dict(
        center=np.zeros(n_features),
        components=np.eye(n_features),
        score_center=np.zeros(n_features),
        score_scale=np.ones(n_features),
    )


def test_frozen_mode_classifier_reassigns_and_flags_ood():
    curves = np.asarray([
        [-1.1, 0.0], [-0.9, 0.1], [-1.0, -0.1],
        [1.1, 0.0], [0.9, -0.1], [1.0, 0.1],
    ])
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    reference = _identity_reference()
    classifier = fit_frozen_mode_classifier(
        curves, labels, reference, ood_quantile=0.9)
    assigned = assign_frozen_modes(
        np.asarray([[-1.0, 0.0], [1.0, 0.0], [4.0, 0.0]]),
        classifier, reference)
    assert assigned["labels"].tolist() == [0, 1, 1]
    assert assigned["ood"].tolist() == [False, False, True]
