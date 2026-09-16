"""Matched linear-filter transition primitives for Topic 5 v3.1.

The observer-only and event-driven arms share ``A``, ``C`` and ``K_filter``.
The sole event-driven term is ``B @ innovation`` in the transition.  The
functions here are intentionally small and first support synthetic
identifiability; they do not yet authorize a human transition test.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge


def _matrix(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be one finite matrix")
    return array


@dataclass(frozen=True)
class SharedLinearFilter:
    transition: np.ndarray
    observation: np.ndarray
    filter_gain: np.ndarray

    def __post_init__(self):
        transition = _matrix(self.transition, "transition")
        observation = _matrix(self.observation, "observation")
        gain = _matrix(self.filter_gain, "filter_gain")
        state_dim = transition.shape[0]
        if transition.shape != (state_dim, state_dim):
            raise ValueError("transition must be square")
        if observation.shape[1] != state_dim:
            raise ValueError("observation/state dimension mismatch")
        if gain.shape != (state_dim, observation.shape[0]):
            raise ValueError("filter gain dimension mismatch")

    @property
    def state_dimension(self) -> int:
        return int(self.transition.shape[0])

    @property
    def observation_dimension(self) -> int:
        return int(self.observation.shape[0])


@dataclass(frozen=True)
class FilterTrace:
    prior: np.ndarray
    posterior: np.ndarray
    innovation: np.ndarray
    predicted_observation: np.ndarray


def filter_sequence(
    observations: np.ndarray,
    shared: SharedLinearFilter,
    *,
    initial_state: np.ndarray | None = None,
    event_transition: np.ndarray | None = None,
) -> FilterTrace:
    """Filter one continuity unit under a fixed observer/transition pair."""

    values = _matrix(observations, "observations")
    if values.shape[1] != shared.observation_dimension:
        raise ValueError("observation dimension mismatch")
    state_dim = shared.state_dimension
    event = (
        np.zeros((state_dim, shared.observation_dimension), dtype=float)
        if event_transition is None
        else _matrix(event_transition, "event_transition")
    )
    if event.shape != (state_dim, shared.observation_dimension):
        raise ValueError("event transition dimension mismatch")
    posterior_previous = (
        np.zeros(state_dim, dtype=float)
        if initial_state is None
        else np.asarray(initial_state, dtype=float)
    )
    if posterior_previous.shape != (state_dim,):
        raise ValueError("initial_state dimension mismatch")

    prior = np.empty((len(values), state_dim), dtype=float)
    posterior = np.empty_like(prior)
    innovation = np.empty((len(values), shared.observation_dimension), dtype=float)
    predicted = np.empty_like(innovation)
    previous_innovation = np.zeros(shared.observation_dimension, dtype=float)
    for index, observation in enumerate(values):
        prior[index] = (
            shared.transition @ posterior_previous
            + event @ previous_innovation
        )
        predicted[index] = shared.observation @ prior[index]
        innovation[index] = observation - predicted[index]
        posterior[index] = prior[index] + shared.filter_gain @ innovation[index]
        posterior_previous = posterior[index]
        previous_innovation = innovation[index]
    return FilterTrace(
        prior=prior,
        posterior=posterior,
        innovation=innovation,
        predicted_observation=predicted,
    )


@dataclass(frozen=True)
class SyntheticTransitionData:
    observations: np.ndarray
    prior: np.ndarray
    posterior: np.ndarray
    innovation: np.ndarray
    transition_residual: np.ndarray


def simulate_innovation_transition(
    n_events: int,
    shared: SharedLinearFilter,
    event_transition: np.ndarray,
    *,
    observation_noise: float = 0.5,
    transition_noise: float = 0.05,
    seed: int = 0,
) -> SyntheticTransitionData:
    """Generate a known innovation-driven latent process for calibration."""

    event = _matrix(event_transition, "event_transition")
    if event.shape != (
        shared.state_dimension,
        shared.observation_dimension,
    ):
        raise ValueError("event transition dimension mismatch")
    rng = np.random.default_rng(int(seed))
    n = int(n_events)
    if n < 2:
        raise ValueError("at least two events are required")
    state = np.zeros(shared.state_dimension, dtype=float)
    prior = np.empty((n, shared.state_dimension), dtype=float)
    posterior = np.empty_like(prior)
    innovation = np.empty((n, shared.observation_dimension), dtype=float)
    observations = np.empty_like(innovation)
    transition_residual = np.empty_like(prior)
    previous_innovation = np.zeros(shared.observation_dimension, dtype=float)
    for index in range(n):
        residual = event @ previous_innovation
        residual += rng.normal(
            scale=float(transition_noise), size=shared.state_dimension
        )
        prior[index] = shared.transition @ state + residual
        observations[index] = shared.observation @ prior[index]
        observations[index] += rng.normal(
            scale=float(observation_noise),
            size=shared.observation_dimension,
        )
        innovation[index] = observations[index] - shared.observation @ prior[index]
        posterior[index] = prior[index] + shared.filter_gain @ innovation[index]
        transition_residual[index] = residual
        state = posterior[index]
        previous_innovation = innovation[index]
    return SyntheticTransitionData(
        observations=observations,
        prior=prior,
        posterior=posterior,
        innovation=innovation,
        transition_residual=transition_residual,
    )


def fit_event_transition_from_latent_trace(
    posterior: np.ndarray,
    innovation: np.ndarray,
    next_prior: np.ndarray,
    shared: SharedLinearFilter,
    *,
    alpha: float = 1.0,
) -> np.ndarray:
    """Recover ``B`` from a synthetic latent trace with shared ``A/C/K`` fixed."""

    post = _matrix(posterior, "posterior")
    event = _matrix(innovation, "innovation")
    target = _matrix(next_prior, "next_prior")
    if not (len(post) == len(event) == len(target)):
        raise ValueError("transition rows must match")
    residual = target - post @ shared.transition.T
    ridge = Ridge(alpha=float(alpha), fit_intercept=False)
    ridge.fit(event, residual)
    return np.asarray(ridge.coef_, dtype=float)


def transition_prediction_error(
    posterior: np.ndarray,
    innovation: np.ndarray,
    next_prior: np.ndarray,
    shared: SharedLinearFilter,
    event_transition: np.ndarray | None = None,
) -> float:
    """Mean squared latent transition error for synthetic identification."""

    post = _matrix(posterior, "posterior")
    event = _matrix(innovation, "innovation")
    target = _matrix(next_prior, "next_prior")
    prediction = post @ shared.transition.T
    if event_transition is not None:
        matrix = _matrix(event_transition, "event_transition")
        prediction += event @ matrix.T
    return float(np.mean((target - prediction) ** 2))


def observable_transition_impulse(
    rank_loadings: np.ndarray,
    event_transition: np.ndarray,
) -> np.ndarray:
    """Map ``B`` to observable contact-rank coordinates."""

    loading = _matrix(rank_loadings, "rank_loadings")
    event = _matrix(event_transition, "event_transition")
    if loading.shape[1] != event.shape[0]:
        raise ValueError("rank/state dimension mismatch")
    return loading @ event


__all__ = [
    "FilterTrace",
    "SharedLinearFilter",
    "SyntheticTransitionData",
    "filter_sequence",
    "fit_event_transition_from_latent_trace",
    "observable_transition_impulse",
    "simulate_innovation_transition",
    "transition_prediction_error",
]
