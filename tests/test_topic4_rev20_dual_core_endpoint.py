import numpy as np

from src.topic4_rev20_dual_core_endpoint import complete_distribution_distance
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
