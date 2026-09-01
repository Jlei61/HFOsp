import numpy as np

from scripts.freeze_topic4_rev20_dc_selection import freeze_selection
from src.topic4_rev20_dual_core_endpoint import (
    complete_distribution_distance, reference_embedding,
    score_complete_distribution,
)
from src.topic4_shaft_aware import build_contact_contract, build_event_features


def _contract():
    names = [f"ICL{i}" for i in range(1, 12)] + [f"SCL{i}" for i in range(6, 10)]
    xy = np.column_stack([np.arange(15), np.zeros(15)])
    return build_contact_contract(names, xy, np.arange(15), {"kind": "test"})


def test_complete_distribution_has_no_label_or_ood_dependency():
    rng = np.random.default_rng(3)
    onsets = rng.normal(size=(12, 15))
    onsets[rng.random(onsets.shape) < 0.2] = np.nan
    groups = {
        "ICL": np.arange(11),
        "SCL": np.arange(11, 15),
    }
    features = build_event_features(onsets, groups)["features"]
    center = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale == 0] = 1.0
    components = np.eye(features.shape[1])[:5]
    z = ((features - center) / scale) @ components.T
    directions = rng.normal(size=(8, 5))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    embedding = {
        "center": center,
        "scale": scale,
        "components": components,
        "directions": directions,
        "reference_z": z,
    }
    first = complete_distribution_distance(
        onsets, groups=groups, embedding=embedding,
    )
    labels = rng.integers(0, 2, len(onsets))
    ood = rng.random(len(onsets)) > 0.5
    labels[:] = labels[::-1]
    ood[:] = ~ood
    second = complete_distribution_distance(
        onsets, groups=groups, embedding=embedding,
    )
    assert first == second == 0.0


def test_selection_score_needs_no_classifier_or_labels():
    rng = np.random.default_rng(7)
    onsets = rng.normal(size=(10, 15))
    groups = {"ICL": np.arange(11), "SCL": np.arange(11, 15)}
    features = build_event_features(onsets, groups)["features"]
    center = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale == 0] = 1.0
    components = np.eye(features.shape[1])[:6]
    z = ((features - center) / scale) @ components.T
    directions = rng.normal(size=(12, 6))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    training = {
        "feature_center": center,
        "feature_scale": scale,
        "pca_components": components,
        "sw_directions": directions,
        "global_reference_z": z,
    }
    result = score_complete_distribution(
        onsets, np.ones(len(onsets), bool), contract=_contract(),
        training_arrays=training,
    )
    assert result["complete_distribution_distance_training"] == 0.0
    assert result["selection_used_labels"] is False
    assert result["selection_used_ood"] is False
    assert result["selection_used_heldout"] is False


def test_reference_embedding_changes_only_cloud_reference():
    rng = np.random.default_rng(9)
    onsets = rng.normal(size=(14, 15))
    shifted = onsets + np.linspace(0.0, 2.0, 15)
    groups = {"ICL": np.arange(11), "SCL": np.arange(11, 15)}
    features = build_event_features(onsets, groups)["features"]
    center = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale == 0] = 1.0
    components = rng.normal(size=(7, features.shape[1]))
    components /= np.linalg.norm(components, axis=1, keepdims=True)
    directions = rng.normal(size=(10, 7))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    embedding = {
        "center": center, "scale": scale, "components": components,
        "directions": directions,
        "reference_z": ((features - center) / scale) @ components.T,
    }
    heldout = reference_embedding(shifted, groups=groups, embedding=embedding)
    assert np.array_equal(heldout["center"], embedding["center"])
    assert not np.array_equal(heldout["reference_z"], embedding["reference_z"])


def test_selection_freeze_rejects_open_validation_and_uses_distance_only():
    config = {
        "search": {"fit_network_seeds": [1, 2]},
        "claim_boundary": "test",
    }
    manifest = {"candidates": [
        {"candidate_id": "ref", "family": "reference", "level": "reference",
         "is_reference": True},
        {"candidate_id": "a", "family": "gain", "level": 0.5,
         "is_reference": False},
        {"candidate_id": "b", "family": "gain", "level": 1.5,
         "is_reference": False},
    ]}
    rows = []
    for candidate, values in {"ref": [1.0, 1.0], "a": [0.8, 0.9],
                              "b": [0.7, 1.2]}.items():
        for seed, value in zip((1, 2), values):
            rows.append({
                "candidate_id": candidate, "seed": seed,
                "runaway_early_stop_ms": None,
                "selection": {
                    "complete_distribution_distance_training": value,
                    "selection_used_labels": False,
                    "selection_used_ood": False,
                    "selection_used_heldout": False,
                },
            })
    aggregate = {
        "phase": "screen", "heldout_opened": False,
        "validation_endpoint_status": "SEALED_UNTIL_SELECTION_FREEZE",
        "per_network": rows,
    }
    selected = freeze_selection(config, aggregate, manifest)
    assert selected["candidate_ids"] == ["ref", "a"]
    aggregate["per_network"][0]["validation"] = {"ood_all_returned": 0.0}
    try:
        freeze_selection(config, aggregate, manifest)
    except RuntimeError as error:
        assert "forbidden validation" in str(error)
    else:
        raise AssertionError("validation leakage was not rejected")
