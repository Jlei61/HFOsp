import numpy as np
import pytest

from scripts.aggregate_topic4_rev12_cascade_fit import (
    cascade_selection_objective,
    equal_network_natural_kmeans,
    k2_support_score,
    per_network_natural_kmeans,
    validate_contact_readout,
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


def test_k2_support_is_continuous_and_can_contribute_without_a_hard_gate():
    assert k2_support_score(1.0) > k2_support_score(0.0)
    assert k2_support_score(0.0) > k2_support_score(-1.0)
    unsupported = cascade_selection_objective(
        patient_loss=1.0, kmeans_balanced_alignment=0.7,
        ood_fraction=0.2, compound_fraction=0.2,
        k2_support=0.0, k2_support_weight=0.25,
    )
    supported = cascade_selection_objective(
        patient_loss=1.0, kmeans_balanced_alignment=0.7,
        ood_fraction=0.2, compound_fraction=0.2,
        k2_support=1.0, k2_support_weight=0.25,
    )
    assert supported["objective"] < unsupported["objective"]


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
    by_network = per_network_natural_kmeans([
        {**first, "seed": 1}, {**second, "seed": 2},
    ], seed=7)
    assert by_network["n_networks"] == 2
    assert 0.0 <= by_network["equal_network_mean_k2_support"] <= 1.0


def test_aggregate_fails_closed_on_contact_readout_drift():
    expected = {
        "source": "lineage_restricted_sheet_activity",
        "minimum_full_trace_pearson": 0.98,
    }
    validate_contact_readout({"contact_readout": {
        "source": "lineage_restricted_sheet_activity",
        "parity_status": "PASS",
        "full_trace_pearson_minimum": 0.981,
    }}, expected)
    with pytest.raises(RuntimeError, match="wrong contact readout"):
        validate_contact_readout({"contact_readout": {
            "source": "full_contact_envelope_within_lineage_window",
        }}, expected)
    with pytest.raises(RuntimeError, match="parity"):
        validate_contact_readout({"contact_readout": {
            "source": "lineage_restricted_sheet_activity",
            "parity_status": "PASS",
            "full_trace_pearson_minimum": 0.97,
        }}, expected)
