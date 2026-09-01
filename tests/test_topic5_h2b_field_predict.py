"""B2 contract tests: predicting a held-out seizure's early field from a state.

With a median of ~5 TRAIN seizures per patient, a state -> field regression is
unfittable. The estimator instead re-weights the patient's own TRAIN fields by
how similar the frozen state at ``onset - lead`` is to the state before each
TRAIN seizure. Both arms then use *identical* TRAIN fields and differ only in
whether the state informs the weights, which is what makes the comparison
nested and interpretable.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.topic5_h2b_transfer.field_predict import (
    predict_field,
    state_similarity_weights,
)


# --- weights -------------------------------------------------------------------


def test_uniform_weights_reproduce_the_patient_average():
    fields = np.array([[1.0, 2.0], [3.0, 4.0]])
    pred = predict_field(fields, weights=None)
    assert np.allclose(pred, [2.0, 3.0])


def test_weights_sum_to_one():
    q = np.array([1.0, 0.0])
    ref = np.array([[1.0, 0.0], [0.0, 1.0]])
    w = state_similarity_weights(q, ref, temperature=1.0)
    assert np.isclose(w.sum(), 1.0)


def test_the_most_similar_train_state_gets_the_largest_weight():
    q = np.array([1.0, 0.0])
    ref = np.array([[0.9, 0.1], [-1.0, 0.0]])
    w = state_similarity_weights(q, ref, temperature=1.0)
    assert w[0] > w[1]


def test_a_very_low_temperature_approaches_nearest_neighbour():
    q = np.array([1.0, 0.0])
    ref = np.array([[0.9, 0.1], [-1.0, 0.0]])
    w = state_similarity_weights(q, ref, temperature=0.01)
    assert w[0] > 0.99


def test_a_very_high_temperature_approaches_the_uniform_baseline():
    q = np.array([1.0, 0.0])
    ref = np.array([[0.9, 0.1], [-1.0, 0.0]])
    w = state_similarity_weights(q, ref, temperature=1e6)
    assert np.allclose(w, [0.5, 0.5], atol=1e-3)


def test_weighting_is_scale_invariant_in_the_state():
    """Cosine similarity, so overall state magnitude must not drive the weights."""
    q = np.array([1.0, 0.0])
    ref = np.array([[0.9, 0.1], [-1.0, 0.0]])
    a = state_similarity_weights(q, ref, temperature=1.0)
    b = state_similarity_weights(10.0 * q, 10.0 * ref, temperature=1.0)
    assert np.allclose(a, b)


# --- prediction ------------------------------------------------------------------


def test_prediction_is_the_weighted_combination_of_train_fields_only():
    fields = np.array([[1.0, 0.0], [0.0, 1.0]])
    pred = predict_field(fields, weights=np.array([1.0, 0.0]))
    assert np.allclose(pred, [1.0, 0.0])


def test_prediction_refuses_a_weight_vector_of_the_wrong_length():
    with pytest.raises(ValueError, match="weights"):
        predict_field(np.zeros((2, 3)), weights=np.array([1.0, 0.0, 0.0]))


def test_contacts_missing_in_some_train_fields_do_not_poison_the_average():
    fields = np.array([[1.0, np.nan], [3.0, 4.0]])
    pred = predict_field(fields, weights=None)
    assert np.isclose(pred[0], 2.0)
    assert np.isclose(pred[1], 4.0)


def test_no_train_field_yields_an_all_nan_prediction_not_zeros():
    pred = predict_field(np.empty((0, 3)), weights=None)
    assert pred.shape == (3,)
    assert np.all(np.isnan(pred))
