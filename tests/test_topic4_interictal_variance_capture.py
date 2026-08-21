import numpy as np

from scripts.audit_topic4_interictal_variance_capture import (
    align_model_modes,
    contrast_metrics,
    event_features,
    shaft_balanced_weights,
    weighted_r2,
)


def test_event_features_keep_missingness_separate_from_rank():
    ranks = np.asarray([[0.0, np.nan, 1.0], [np.nan, 0.5, 1.0]])
    features = event_features(ranks)
    assert features.shape == (2, 6)
    assert np.array_equal(features[:, :3], np.isfinite(ranks))
    assert np.array_equal(features[:, 3:], np.nan_to_num(ranks, nan=0.0))


def test_shaft_balanced_weights_protect_small_shaft():
    names = np.asarray(["ICL1", "ICL2", "ICL3", "SCL1"])
    weights = shaft_balanced_weights(names)
    assert np.isclose(weights.sum(), 1.0)
    assert np.isclose(weights[:4][[0, 1, 2]].sum(), 0.25)
    assert np.isclose(weights[:4][[3]].sum(), 0.25)
    assert np.isclose(weights[4:][[0, 1, 2]].sum(), 0.25)
    assert np.isclose(weights[4:][[3]].sum(), 0.25)


def test_weighted_r2_is_one_for_exact_mode_prototypes():
    prototypes = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    labels = np.asarray([0, 1, 0, 1])
    events = prototypes[labels]
    metric = weighted_r2(
        events, labels, prototypes, np.asarray([0.5, 0.5]),
        np.asarray([0.5, 0.5]),
    )
    assert metric["r2"] == 1.0


def test_weighted_r2_can_be_negative():
    events = np.asarray([[0.0], [1.0]])
    labels = np.asarray([0, 1])
    metric = weighted_r2(
        events, labels, np.asarray([[2.0], [-1.0]]),
        np.asarray([0.5]), np.asarray([1.0]),
    )
    assert metric["r2"] < 0.0


def test_model_mode_alignment_uses_patient_training_semantics():
    raw = {
        1: np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        2: np.asarray([[0.9, 0.1], [0.1, 0.9]]),
    }
    patient = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    aligned, audit = align_model_modes(raw, patient, np.asarray([0.5, 0.5]))
    assert audit["raw_model_order_for_patient_TA_TB"] == [1, 0]
    assert np.allclose(aligned[1], patient)


def test_contrast_scale_is_fit_on_train_and_scored_on_heldout():
    model = np.asarray([[0.0, 0.5], [0.0, -0.5]])
    patient_train = np.asarray([[0.0, 1.0], [0.0, -1.0]])
    patient_heldout = np.asarray([[0.0, 1.5], [0.0, -1.5]])
    metric = contrast_metrics(
        model, patient_train, patient_heldout, np.asarray([0.5, 0.5]),
    )
    assert metric["train_fitted_nonnegative_scale"] == 2.0
    assert metric["heldout_scale_calibrated_contrast_r2"] < 1.0
    assert metric["heldout_weighted_cosine"] == 1.0
