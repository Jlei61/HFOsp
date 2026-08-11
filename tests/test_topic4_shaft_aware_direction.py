import numpy as np

from src.topic4_shaft_aware import fit_patient_embedding
from src.topic4_shaft_aware_direction import (
    all_event_shaft_participation,
    assign_direction_modes,
    fit_direction_classifier,
)


def _synthetic_events(seed=4):
    rng = np.random.default_rng(seed)
    values, labels, blocks = [], [], []
    for block in range(12):
        for mode in (0, 1):
            for _ in range(8):
                row = np.full(6, np.nan)
                if mode == 0:
                    row[:3] = [0.0, 1.0, 2.0]
                else:
                    row[:3] = [2.0, 1.0, 0.0]
                if rng.random() < 0.8:
                    row[3:] = row[:3] + rng.normal(0.2, 0.05, 3)
                values.append(row)
                labels.append(mode)
                blocks.append(block)
    return np.asarray(values), np.asarray(labels), np.asarray(blocks)


def test_block_validated_direction_classifier_assigns_every_event():
    values, labels, blocks = _synthetic_events()
    groups = {"ICL": np.arange(3), "SCL": np.arange(3, 6)}
    from src.topic4_shaft_aware import build_event_features

    features = build_event_features(values, groups)["features"]
    embedding = fit_patient_embedding(
        features, variance_fraction=0.99, max_components=8,
        reference_n=100, n_directions=8, seed=10,
    )
    classifier = fit_direction_classifier(
        values, labels, blocks, groups=groups, embedding=embedding,
        n_splits=4,
    )
    assigned = assign_direction_modes(
        values, groups=groups, embedding=embedding, classifier=classifier,
    )

    assert classifier["heldout_balanced_accuracy"] > 0.95
    assert len(assigned["labels"]) == len(values)
    assert np.mean(assigned["labels"] == labels) > 0.95


def test_all_event_shaft_participation_does_not_drop_single_shaft_events():
    groups = {"ICL": np.arange(2), "SCL": np.arange(2, 4)}
    values = np.asarray([
        [0.0, 1.0, np.nan, np.nan],
        [0.0, 1.0, 2.0, 3.0],
        [np.nan, np.nan, 0.0, 1.0],
        [np.nan, np.nan, np.nan, np.nan],
    ])
    result = all_event_shaft_participation(values, groups)

    assert result["n_events"] == 4
    assert result["n_icl_only"] == 1
    assert result["n_joint"] == 1
    assert result["n_scl_only"] == 1
    assert result["n_unreadable"] == 1
    assert result["joint_fraction"] == 0.25


def test_factorized_objective_rewards_joint_events_without_hard_deletion():
    from scripts.aggregate_topic4_rev10_sa_spline_field_search import _objective

    config = {
        "search": {"objective": {
            "fixed_events_per_mode": 6,
            "minimum_target_joint_fraction": 0.95,
            "joint_floor_q05": 2.0 / 3.0,
            "joint_weight": 1.0,
            "direction_support_weight": 2.0,
            "ood_weight": 0.5,
        }}
    }
    supported = {"status": "OK", "weak_mode_score": 5.0}
    insufficient = {"status": "INSUFFICIENT_MODE_SUPPORT"}
    low_joint = _objective(
        supported, insufficient, np.array([6, 6]),
        {"joint_fraction": 0.0}, 0.0, config,
    )
    higher_joint = _objective(
        supported, insufficient, np.array([6, 6]),
        {"joint_fraction": 0.5}, 0.0, config,
    )

    assert higher_joint["selection_score"] < low_joint["selection_score"]
    assert higher_joint["direction_support_penalty"] == 0.0
