from pathlib import Path

import numpy as np
import torch
import yaml

from scripts.run_topic5_shared_scaffold_rnn_unit_v0_2 import (
    weighted_training_loss,
    within_event_rank_shuffle,
)
from src.topic5_shared_scaffold_rnn import SharedScaffoldPropagationRNN


ROOT = Path(__file__).resolve().parents[1]


def _model() -> SharedScaffoldPropagationRNN:
    graph = np.zeros((5, 5), dtype=np.float32)
    for index in range(4):
        graph[index, index + 1] = 1.0
        graph[index + 1, index] = 1.0
    return SharedScaffoldPropagationRNN(
        fixed_adjacency=graph,
        participation_bias=np.linspace(-0.2, 0.2, 5, dtype=np.float32),
        low_rank=2,
    )


def _groups() -> tuple[torch.Tensor, torch.Tensor]:
    groups = torch.tensor(
        [
            [0, 1, 2, -1, -1],
            [0, 0, 1, 2, -1],
            [0, -1, -1, -1, -1],
            [0, 1, 1, 2, 3],
        ],
        dtype=torch.long,
    )
    return groups, torch.tensor([3, 3, 1, 4], dtype=torch.long)


def test_weighted_loss_is_exact_frozen_three_component_objective():
    model = _model()
    groups, counts = _groups()
    weights = {"contact": 1.0, "cardinality": 0.25, "stop": 0.25}
    observed, pieces = weighted_training_loss(
        model,
        groups,
        counts,
        n_macro_events=len(groups),
        n_macro_transition_events=int(torch.sum(counts > 1)),
        weights=weights,
    )
    reference = model.batched_event_nll(groups, counts, reduction="event_first")
    expected = (
        reference["conditional_contacts"]
        + 0.25 * reference["cardinality"]
        + 0.25 * reference["stop"]
    )
    torch.testing.assert_close(observed, expected)
    assert pieces["joint"] == float(observed.detach())


def test_micro_batches_sum_to_same_macro_event_first_objective():
    model = _model()
    groups, counts = _groups()
    weights = {"contact": 1.0, "cardinality": 0.25, "stop": 0.25}
    complete, _ = weighted_training_loss(
        model,
        groups,
        counts,
        n_macro_events=len(groups),
        n_macro_transition_events=int(torch.sum(counts > 1)),
        weights=weights,
    )
    first, _ = weighted_training_loss(
        model,
        groups[:2],
        counts[:2],
        n_macro_events=len(groups),
        n_macro_transition_events=int(torch.sum(counts > 1)),
        weights=weights,
    )
    second, _ = weighted_training_loss(
        model,
        groups[2:],
        counts[2:],
        n_macro_events=len(groups),
        n_macro_transition_events=int(torch.sum(counts > 1)),
        weights=weights,
    )
    torch.testing.assert_close(first + second, complete)


def test_rank_shuffle_preserves_participation_and_each_rank_multiset():
    original = np.asarray(
        [[0, 0, 1, 2, -1], [0, 1, 2, 2, 3], [0, -1, -1, -1, -1]],
        dtype=np.int16,
    )
    first = within_event_rank_shuffle(original, seed=11)
    second = within_event_rank_shuffle(original, seed=11)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first >= 0, original >= 0)
    for before, after in zip(original, first):
        np.testing.assert_array_equal(
            np.sort(before[before >= 0]), np.sort(after[after >= 0])
        )


def test_config_freezes_optimizer_and_separates_dataset_from_worktree_output():
    config = yaml.safe_load(
        (ROOT / "config/topic5_shared_scaffold_propagation_rnn_v0_2.yaml").read_text()
    )
    assert config["dataset_artifact_root"] == "/home/honglab/leijiaxin/HFOsp"
    assert not Path(config["output_root"]).is_absolute()
    assert config["training"]["learning_rate"] == 3e-4
    assert config["training"]["weight_decay"] == 0.0
    assert config["training"]["gradient_clip"] == 1.0
    assert config["training"]["coverage_cycles"] == 7
    assert config["training"]["optimizer_updates_per_cycle"] == 32
    assert config["training"]["micro_batch_events"] == 256
    assert config["training"]["hazard_pseudocount"] == 0.5
    assert config["training"]["contact_weight"] == 1.0
    assert config["training"]["cardinality_weight"] == 0.25
    assert config["training"]["stop_weight"] == 0.25
