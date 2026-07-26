from types import SimpleNamespace

import numpy as np
import torch

from scripts.train_topic5_persistent_path_rnn import (
    PathModePrior,
    _shuffle_mode_graphs,
    _shuffle_weights_within_mode,
    calibrate_heldout_offset_coverage,
    select_outer_records,
    train_shared_coverage,
)
from src.topic5_persistent_path_rnn import PersistentPathModeRNN


def test_loso_outer_records_are_unique_and_exclude_heldout():
    records = {
        f"subject_{index}": SimpleNamespace(subject=f"subject_{index}")
        for index in range(34)
    }
    outer = select_outer_records(records, "subject_7")
    assert len(outer) == 33
    assert len({record.subject for record in outer}) == 33
    assert "subject_7" not in {record.subject for record in outer}


def test_mode_shuffle_preserves_prior_weighted_mean_graph_exactly():
    rng = np.random.default_rng(11)
    graphs = rng.uniform(size=(4, 7, 7)).astype(np.float32)
    graphs[:, np.arange(7), np.arange(7)] = 0.0
    prior = np.array([0.45, 0.25, 0.20, 0.10], np.float32)
    shuffled = _shuffle_mode_graphs(graphs, prior, seed=19)
    np.testing.assert_allclose(
        np.einsum("m,mij->ij", prior, shuffled),
        np.einsum("m,mij->ij", prior, graphs),
        atol=2e-6,
    )
    assert not np.allclose(shuffled, graphs)


def test_weight_shuffle_preserves_each_mode_total_and_axis_support():
    axis = np.linspace(-1.0, 1.0, 6, dtype=np.float32)
    raw = np.zeros((3, 6, 6), np.float32)
    allowed = axis[:, None] > axis[None, :]
    rng = np.random.default_rng(5)
    raw[:, allowed] = rng.uniform(size=(3, int(allowed.sum())))
    shuffled = _shuffle_weights_within_mode(raw, axis, seed=23)
    np.testing.assert_allclose(
        shuffled.sum((1, 2)), raw.sum((1, 2)), atol=2e-6
    )
    assert np.all(shuffled[:, ~allowed] == 0)


def _tiny_record(subject: str, dataset: str):
    groups = np.asarray(
        [
            [0, 1, -1],
            [1, 0, -1],
            [0, -1, 1],
            [1, -1, 0],
        ],
        np.int16,
    )
    return SimpleNamespace(
        subject=subject,
        dataset=dataset,
        contact_names=np.asarray(["A1", "A2", "A3"]),
        contact_features=np.asarray(
            [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]], np.float32
        ),
        group_ids=groups,
        group_count=np.full(len(groups), 2, np.int16),
        train_indices=np.arange(len(groups)),
    )


def _tiny_prior(subject: str) -> PathModePrior:
    forward = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        np.float32,
    )
    graphs = np.stack([forward, forward.T])
    return PathModePrior(
        subject=subject,
        axis=np.asarray([-1.0, 0.0, 1.0], np.float32),
        component_graphs=graphs,
        component_prior=np.asarray([0.5, 0.5], np.float32),
        component_mode=np.asarray([0, 0], np.int16),
        component_direction=np.asarray([1, -1], np.int8),
        aggregate_forward=forward,
        aggregate_reverse=forward.T.copy(),
        left=np.asarray([True, False, False]),
        right=np.asarray([False, False, True]),
        source_sha256="test",
        control="intact",
        mode_count=1,
        use_recurrence=True,
    )


def _tiny_config():
    return {
        "model": {
            "local_offset_dim": 1,
            "stop_calibration_weight": 0.1,
            "endpoint_source_weight": 0.0,
        },
        "training": {
            "learning_rate": 1e-3,
            "local_learning_rate": 2e-3,
            "weight_decay": 1e-4,
            "gradient_clip": 1.0,
        },
    }


def test_exact_coverage_counts_every_event_once_per_cycle():
    records = [
        _tiny_record("subject_a", "epilepsiae"),
        _tiny_record("subject_b", "yuquan"),
    ]
    priors = {record.subject: _tiny_prior(record.subject) for record in records}
    model = PersistentPathModeRNN(2)
    _, rows, coverage = train_shared_coverage(
        model,
        records,
        priors,
        _tiny_config(),
        coverage_cycles=2,
        updates_per_patient=2,
        batch_size=2,
        device=torch.device("cpu"),
        seed=17,
    )
    assert len(rows) == 8
    for record in records:
        assert coverage[record.subject]["events_drawn"] == 8
        assert coverage[record.subject]["completed_cycles"] == 2
        assert coverage[record.subject]["fraction_of_first_cycle"] == 1.0


def test_exact_coverage_calibration_freezes_shared_parameters():
    record = _tiny_record("subject_a", "epilepsiae")
    prior = _tiny_prior(record.subject)
    model = PersistentPathModeRNN(2)
    before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    offset, rows, coverage = calibrate_heldout_offset_coverage(
        model,
        record,
        prior,
        _tiny_config(),
        coverage_cycles=3,
        updates_per_cycle=2,
        batch_size=2,
        device=torch.device("cpu"),
        seed=19,
    )
    assert offset.shape == (3, 1)
    assert len(rows) == 6
    assert coverage["events_drawn"] == 12
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])
