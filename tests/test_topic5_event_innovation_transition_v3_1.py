import numpy as np

from src.topic5_event_innovation_transition_v3_1 import (
    SharedLinearFilter,
    filter_sequence,
    fit_event_transition_from_latent_trace,
    observable_transition_impulse,
    simulate_innovation_transition,
    transition_prediction_error,
)


def _shared_filter():
    return SharedLinearFilter(
        transition=np.array([[0.8, 0.1], [0.0, 0.7]]),
        observation=np.eye(2),
        filter_gain=np.eye(2) * 0.4,
    )


def test_observer_and_event_driven_rollouts_share_filter_parameters():
    shared = _shared_filter()
    observations = np.array([[1.0, 0.0], [0.5, 0.3], [0.2, -0.1]])
    observer = filter_sequence(observations, shared)
    driven = filter_sequence(
        observations,
        shared,
        event_transition=np.array([[0.2, 0.0], [0.0, -0.1]]),
    )
    np.testing.assert_allclose(observer.innovation[0], driven.innovation[0])
    assert not np.allclose(observer.prior[1:], driven.prior[1:])


def test_synthetic_event_transition_is_recovered_and_improves_prediction():
    shared = _shared_filter()
    truth = np.array([[0.5, 0.0], [0.1, -0.4]])
    data = simulate_innovation_transition(
        5000,
        shared,
        truth,
        observation_noise=0.8,
        transition_noise=0.02,
        seed=2,
    )
    fitted = fit_event_transition_from_latent_trace(
        data.posterior[:-1],
        data.innovation[:-1],
        data.prior[1:],
        shared,
        alpha=1e-6,
    )
    np.testing.assert_allclose(fitted, truth, atol=0.01)
    observer_error = transition_prediction_error(
        data.posterior[:-1], data.innovation[:-1], data.prior[1:], shared
    )
    driven_error = transition_prediction_error(
        data.posterior[:-1],
        data.innovation[:-1],
        data.prior[1:],
        shared,
        fitted,
    )
    assert driven_error < observer_error * 0.01


def test_autonomous_system_does_not_create_material_event_transition():
    shared = _shared_filter()
    zero = np.zeros((2, 2))
    data = simulate_innovation_transition(
        5000,
        shared,
        zero,
        observation_noise=0.8,
        transition_noise=0.02,
        seed=3,
    )
    fitted = fit_event_transition_from_latent_trace(
        data.posterior[:-1],
        data.innovation[:-1],
        data.prior[1:],
        shared,
        alpha=1.0,
    )
    assert np.linalg.norm(fitted) < 0.01


def test_transition_impulse_maps_to_contact_rank_coordinates():
    loading = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    event = np.array([[0.2, 0.0], [0.0, -0.1]])
    observable = observable_transition_impulse(loading, event)
    assert observable.shape == (3, 2)
