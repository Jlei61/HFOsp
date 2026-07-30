import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.topic5_constructive_event_generator import (
    categorical_from_uniform,
    constant_stop_hazard,
    event_length_wasserstein,
    remove_revealed_source,
    shaft_preserving_permutation,
    source_conditioned_rollout,
    train_progress_hazard,
    train_static_log_scaffold,
)
from src.topic5_rank_distribution import LinearStateSequenceRNN


def _toy_model():
    torch.manual_seed(4)
    model = LinearStateSequenceRNN(
        3,
        hidden_size=5,
        contact_embedding_dim=4,
        contact_encoder_hidden=4,
        local_offset_dim=2,
    )
    return model.eval()


def test_train_only_scaffold_and_hazard_are_finite():
    groups = np.array(
        [[0, 1, -1], [1, 0, 2], [0, -1, 1], [1, 0, -1]], dtype=np.int16
    )
    counts = np.array([2, 3, 2, 2])
    train = np.array([0, 1, 2])
    scaffold = train_static_log_scaffold(groups, train)
    hazard = train_progress_hazard(counts, train, max_groups=3)
    assert scaffold.shape == (3,)
    assert np.all(np.isfinite(scaffold))
    assert hazard.shape == (4,)
    assert np.all((hazard >= 0) & (hazard <= 1))
    assert hazard[-1] == 1
    assert 0 < constant_stop_hazard(counts, train) <= 1


def test_inverse_cdf_uses_shared_uniforms_deterministically():
    probability = np.array([[0.2, 0.3, 0.5], [0.8, 0.1, 0.1]])
    uniform = np.array([0.49, 0.81])
    np.testing.assert_array_equal(
        categorical_from_uniform(probability, uniform),
        np.array([1, 1]),
    )


def test_shaft_permutation_never_crosses_shafts():
    names = ["A1", "A2", "A3", "B1", "B2"]
    permutation = shaft_preserving_permutation(names, seed=8)
    assert set(permutation[:3]) == {0, 1, 2}
    assert set(permutation[3:]) == {3, 4}


@pytest.mark.parametrize(
    "condition",
    [
        "full_constructive",
        "static_only",
        "static_shuffle",
        "history_h1",
        "history_h2",
        "constant_stop",
        "no_termination",
    ],
)
def test_source_conditioned_rollout_retains_source_and_never_repeats(condition):
    model = _toy_model()
    n_events, n_contacts = 7, 4
    source = np.zeros((n_events, n_contacts), dtype=bool)
    source[np.arange(n_events), np.arange(n_events) % n_contacts] = True
    uniforms = np.random.default_rng(2).random((n_events, n_contacts))
    kwargs = {}
    if condition == "static_shuffle":
        kwargs["static_permutation"] = np.array([1, 0, 3, 2])
    if condition == "constant_stop":
        kwargs["constant_hazard"] = 0.3
    rollout = source_conditioned_rollout(
        model,
        torch.randn(1, n_contacts, 3),
        torch.ones((1, n_contacts), dtype=torch.bool),
        torch.zeros(n_contacts, 2),
        source,
        uniforms,
        np.log(np.array([0.3, 0.2, 0.4, 0.1])),
        np.array([0.0, 0.2, 0.4, 0.7, 1.0]),
        condition=condition,
        **kwargs,
    )
    assert np.all(rollout.event_group_ids[source] == 0)
    assert np.all(
        rollout.event_participant_count
        == np.sum(rollout.event_group_ids >= 0, axis=1)
    )
    for row in rollout.event_group_ids:
        observed = row[row >= 0]
        assert observed.size == np.unique(np.flatnonzero(row >= 0)).size


def test_no_termination_recruits_every_contact():
    model = _toy_model()
    source = np.array([[True, False, False, False]] * 3)
    rollout = source_conditioned_rollout(
        model,
        torch.randn(1, 4, 3),
        torch.ones((1, 4), dtype=torch.bool),
        torch.zeros(4, 2),
        source,
        np.random.default_rng(5).random((3, 4)),
        np.zeros(4),
        np.array([0.0, 0.2, 0.4, 0.7, 1.0]),
        condition="no_termination",
    )
    np.testing.assert_array_equal(rollout.event_participant_count, 4)
    np.testing.assert_array_equal(rollout.event_group_count, 4)


def test_suffix_removal_reindexes_ranks():
    groups = np.array([[0, 1, 2, -1], [2, 0, 1, -1]])
    source = groups == 0
    suffix = remove_revealed_source(groups, source)
    np.testing.assert_array_equal(
        suffix,
        np.array([[-1, 0, 1, -1], [1, -1, 0, -1]], dtype=np.int16),
    )


def test_event_length_wasserstein_is_zero_for_identical_samples():
    assert event_length_wasserstein([2, 3, 4], [2, 3, 4]) == pytest.approx(0)
