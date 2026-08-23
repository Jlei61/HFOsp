import numpy as np
import pytest

from scripts.aggregate_topic4_rev12_cascade_fit import (
    cascade_selection_objective,
    equal_network_natural_kmeans,
)


def test_cascade_selection_objective_rewards_patient_and_kmeans_improvement():
    baseline = cascade_selection_objective(
        patient_loss=1.0, kmeans_balanced_alignment=0.5,
        ood_fraction=0.2, compound_fraction=0.1,
    )
    patient_better = cascade_selection_objective(
        patient_loss=0.8, kmeans_balanced_alignment=0.5,
        ood_fraction=0.2, compound_fraction=0.1,
    )
    kmeans_better = cascade_selection_objective(
        patient_loss=1.0, kmeans_balanced_alignment=0.8,
        ood_fraction=0.2, compound_fraction=0.1,
    )
    assert patient_better["objective"] < baseline["objective"]
    assert kmeans_better["objective"] < baseline["objective"]


def test_cascade_selection_objective_keeps_compound_and_ood_continuous():
    clean = cascade_selection_objective(
        patient_loss=1.0, kmeans_balanced_alignment=0.7,
        ood_fraction=0.0, compound_fraction=0.0,
    )
    dirty = cascade_selection_objective(
        patient_loss=1.0, kmeans_balanced_alignment=0.7,
        ood_fraction=0.2, compound_fraction=0.2,
    )
    assert dirty["objective"] > clean["objective"]
    with pytest.raises(ValueError):
        cascade_selection_objective(
            patient_loss=1.0, kmeans_balanced_alignment=1.1,
            ood_fraction=0.0, compound_fraction=0.0,
        )


def test_kmeans_auxiliary_gives_networks_equal_event_count():
    first = {
        "ranks": np.asarray([
            [0, 1, 2, 3], [3, 2, 1, 0], [0, 1, 2, 3],
        ], float),
        "labels": np.asarray([0, 1, 0]),
    }
    second = {
        "ranks": np.asarray([
            [0, 1, 2, 3], [3, 2, 1, 0], [0, 1, 2, 3],
            [3, 2, 1, 0], [0, 1, 2, 3],
        ], float),
        "labels": np.asarray([0, 1, 0, 1, 0]),
    }
    result = equal_network_natural_kmeans([first, second], seed=7)
    assert result["n_per_network"] == 3
    assert result["n_events"] == 6
    assert result["network_weighting"] == "equal event count per network"
