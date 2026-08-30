from __future__ import annotations

import numpy as np
import pytest

from src.topic4_node_final_science import (
    _topology_summary_from_features,
    final_zero_simulation_decision,
    score_workers_against_patient_endpoint,
    topology_label_permutation_test,
    weakest_mode_topology_quality,
)
from src.topic4_node_dualmode import (
    source_topology_features,
    topology_network_reproducibility,
)


def _two_mode_maps(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    maps = []
    labels = []
    for mode, center in ((0, (1, 1)), (1, (4, 4))):
        for _ in range(10):
            onset = np.full((6, 6), np.nan)
            onset[center] = rng.uniform(0.0, 1.0)
            onset[center[0], max(0, center[1] - 1)] = rng.uniform(1.0, 2.0)
            onset[min(5, center[0] + 1), center[1]] = rng.uniform(2.0, 3.0)
            maps.append(onset)
            labels.append(mode)
    order = rng.permutation(len(labels))
    return np.asarray(maps)[order], np.asarray(labels)[order]


def test_matched_topology_null_detects_reproducible_two_mode_sources():
    bundles = [_two_mode_maps(seed) for seed in (1, 2, 3)]
    result = topology_label_permutation_test(
        [row[0] for row in bundles], [row[1] for row in bundles],
        draws=256, seed=9,
    )
    assert result["observed_weakest_mode_quality"] > result["null_q95"]
    assert result["above_null_q95"] is True
    assert result["upper_tail_p"] < 0.05


def test_cached_topology_summary_matches_canonical_implementation():
    bundles = [_two_mode_maps(seed) for seed in (1, 2, 3)]
    maps = [row[0] for row in bundles]
    labels = [row[1] for row in bundles]
    observed = topology_network_reproducibility(maps, labels)
    cached = _topology_summary_from_features(
        [source_topology_features(value) for value in maps], labels,
    )
    for key in (
        "mean_within_network_split_half_cosine",
        "mean_across_network_template_cosine",
        "equal_network_between_mode_distance",
    ):
        assert np.isclose(cached[key], observed[key])
    for mode in (0, 1):
        assert np.isclose(
            cached["modes"][str(mode)]["pairwise_network_cosine_mean"],
            observed["modes"][str(mode)]["pairwise_network_cosine_mean"],
        )


def test_weakest_mode_quality_does_not_average_away_a_bad_mode():
    summary = {
        "modes": {
            "0": {"pairwise_network_cosine_mean": 0.9},
            "1": {"pairwise_network_cosine_mean": 0.1},
        },
        "equal_network_between_mode_distance": 0.8,
    }
    assert np.isclose(weakest_mode_topology_quality(summary), 0.08)


def test_patient_endpoint_scoring_uses_supplied_heldout_distribution(monkeypatch):
    observed = []

    def fake_objective(model_ranks, model_labels, patient_ranks, patient_labels,
                       contact_names, **kwargs):
        observed.append(np.asarray(patient_ranks).copy())
        offset = float(np.mean(patient_ranks))
        return {
            "weakest_mode_lse": offset,
            "modes": {
                "0": {"mean": offset + 1.0},
                "1": {"mean": offset + 2.0},
            },
        }

    monkeypatch.setattr(
        "src.topic4_node_final_science.dual_mode_objective", fake_objective,
    )
    heldout = np.full((4, 3), 7.0)
    result = score_workers_against_patient_endpoint(
        [{"seed": 1, "ranks": np.zeros((2, 3)), "labels": np.asarray([0, 1])}],
        patient_ranks=heldout,
        patient_labels=np.asarray([0, 0, 1, 1]),
        contact_names=np.asarray(["a", "b", "c"]),
        projections=np.eye(6), calibration={},
    )
    assert np.array_equal(observed[0], heldout)
    assert result["mean_weakest_mode_lse"] == 7.0
    assert result["mean_mode_0_loss"] == 8.0
    assert result["mean_mode_1_loss"] == 9.0


def test_topology_null_requires_both_modes_in_every_network():
    maps, labels = _two_mode_maps(1)
    with pytest.raises(ValueError, match="each network"):
        topology_label_permutation_test(
            [maps, maps], [np.zeros_like(labels), labels], draws=4,
        )


def test_final_decision_requires_both_modes_and_positive_heldout_r2():
    reference = {
        "model_prototype_r2_on_heldout": -0.2,
        "mean_weakest_mode_lse": 1.0,
        "mean_mode_0_loss": 0.8,
        "mean_mode_1_loss": 0.9,
    }
    candidate = {
        "model_prototype_r2_on_heldout": 0.1,
        "mean_weakest_mode_lse": 0.7,
        "mean_mode_0_loss": 0.6,
        "mean_mode_1_loss": 0.7,
    }
    topology = {
        "observed_weakest_mode_quality": 0.3,
        "null_q95": 0.2,
        "upper_tail_p": 0.01,
        "above_null_q95": True,
    }
    accepted = final_zero_simulation_decision(candidate, reference, topology)
    assert accepted["accepted_for_same_checkpoint_intervention"] is True
    assert accepted["node_freeze_permitted"] is False

    candidate["mean_mode_1_loss"] = 1.1
    rejected = final_zero_simulation_decision(candidate, reference, topology)
    assert rejected["accepted_for_same_checkpoint_intervention"] is False
    assert rejected["clauses"][
        "both_patient_mode_losses_improve_reference"
    ]["pass"] is False
