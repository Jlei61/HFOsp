import json

import numpy as np

from scripts.rescore_topic4_rev12_node_historical import (
    _deduplicate_workers_by_seed,
    _formal_clean_mask,
    _legacy_variance_parity,
    _old_to_patient_label_map,
    _reorder_patient_contract,
    _substrate_stratum,
    pareto_front,
)


def test_pareto_front_uses_all_three_historical_axes():
    rows = [
        {
            "library_candidate_id": "balanced",
            "mean_patient_objective": 0.2,
            "model_prototype_r2_on_heldout": 0.1,
            "same_network_both_fraction": 0.8,
        },
        {
            "library_candidate_id": "dominated",
            "mean_patient_objective": 0.3,
            "model_prototype_r2_on_heldout": 0.0,
            "same_network_both_fraction": 0.7,
        },
        {
            "library_candidate_id": "r2-specialist",
            "mean_patient_objective": 0.4,
            "model_prototype_r2_on_heldout": 0.3,
            "same_network_both_fraction": 0.6,
        },
    ]
    assert set(pareto_front(rows)) == {"balanced", "r2-specialist"}


def test_patient_contact_reorder_rebuilds_feature_blocks():
    patient = {
        "contact_names": np.asarray(["ICL1", "SCL1"]),
        "train_ranks": np.asarray([[0.0, 1.0], [1.0, 0.0]]),
        "heldout_ranks": np.asarray([[0.25, 0.75], [0.75, 0.25]]),
        "train_labels": np.asarray([0, 1]),
        "heldout_labels": np.asarray([0, 1]),
        "train_features": np.zeros((2, 4)),
        "heldout_features": np.zeros((2, 4)),
        "train_prototypes": np.zeros((2, 4)),
        "heldout_prototypes": np.zeros((2, 4)),
        "global_mean": np.zeros(4),
    }
    reordered = _reorder_patient_contract(patient, np.asarray(["SCL1", "ICL1"]))
    assert reordered["train_ranks"].tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert reordered["train_features"][0].tolist() == [1.0, 1.0, 1.0, 0.0]


def test_old_classifier_label_semantics_are_frozen_from_shared_patient_events():
    patient = {
        "train_event_indices": np.asarray([10, 11, 12, 13]),
        "train_labels": np.asarray([1, 1, 0, 0]),
    }
    classifier = {
        "old_event_indices": np.asarray([10, 11, 12, 13]),
        "old_labels": np.asarray([0, 0, 1, 1]),
    }
    mapping = _old_to_patient_label_map(patient, classifier)
    assert mapping["raw_to_patient"].tolist() == [1, 0]
    assert mapping["swapped_matches"] == 4


def test_formal_clean_is_joint_shaft_and_in_support():
    onsets = np.asarray([
        [1.0, np.nan, 2.0],
        [1.0, 2.0, np.nan],
        [np.nan, np.nan, 1.0],
        [1.0, np.nan, 2.0],
    ])
    groups = {"ICL": np.asarray([0, 1]), "SCL": np.asarray([2])}
    ood = np.asarray([False, False, False, True])
    assert _formal_clean_mask(onsets, ood, groups).tolist() == [True, False, False, False]


def test_formal_clean_parity_uses_accepted_rev11_anchor(tmp_path):
    reference = {
        "arms": {
            "node_baseline": {
                "components": {"all": {"model_r2_on_patient_heldout": -0.2}},
                "between_mode_contrast": {
                    "heldout_scale_calibrated_contrast_r2": 0.4,
                },
            },
        },
    }
    path = tmp_path / "variance.json"
    path.write_text(json.dumps(reference))
    rows = [{
        "library": "topic4_rev11_nlc_frozen_substrate_confirmation",
        "candidate_id": "node_baseline",
        "estimands": {"formal_clean": {
            "model_prototype_r2_on_heldout": -0.2,
            "contrast": {"heldout_train_scaled_r2": 0.4},
        }},
    }]
    result = _legacy_variance_parity(rows, path)
    assert result["status"] == "EXACT_WITHIN_1E_12"
    assert result["maximum_absolute_error"] == 0.0


def test_network_seed_is_counted_once_and_longest_trajectory_wins():
    short = {"seed": 4, "duration_ms": 8000.0, "npz_sha256": "a"}
    long = {"seed": 4, "duration_ms": 20000.0, "npz_sha256": "b"}
    other = {"seed": 5, "duration_ms": 16000.0, "npz_sha256": "c"}
    selected = _deduplicate_workers_by_seed([short, other, long])
    assert [(row["seed"], row["duration_ms"]) for row in selected] == [
        (4, 20000.0), (5, 16000.0),
    ]


def test_only_current_spatial_ou_contract_enters_field_ranking():
    assert _substrate_stratum({"fixed_spatial_ou": {
        "mode": "local", "sigma_rate_per_ms": 0.1,
    }}) == "current_spatial_ou_node_only"
    assert _substrate_stratum({}) == "legacy_without_current_spatial_ou"
