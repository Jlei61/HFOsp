import copy

import numpy as np

from src.topic4_shaft_aware import (
    align_cluster_labels,
    build_contact_contract,
    build_event_features,
    consensus_kmeans,
    contract_groups,
    contract_pairs,
    describe_events,
    descriptor_distances,
    fit_patient_embedding,
    pair_state_distributions,
    sliced_event_cloud_distance,
)


def _contract():
    names = [f"ICL{i}" for i in range(1, 12)] + [f"SCL{i}" for i in range(6, 10)]
    xy = np.column_stack([np.arange(15.0), np.zeros(15)])
    axial = np.arange(15.0)
    return build_contact_contract(names, xy, axial, {"rx_mm": 0.13, "Rr_mm": 0.278})


def test_contract_freezes_unordered_pair_counts_and_hashes():
    contract = _contract()
    assert contract["shaft_counts"] == {"ICL": 11, "SCL": 4}
    assert contract["pair_counts"] == {"ICL-ICL": 55, "SCL-SCL": 6, "ICL-SCL": 44}
    assert len(contract["pairs"]) == 105
    assert all(len(value) == 64 for value in contract["hashes"].values())

    changed = copy.deepcopy(contract["readout_parameters"])
    changed["rx_mm"] = 0.14
    rebuilt = build_contact_contract(
        [row["contact_name"] for row in contract["contacts"]],
        np.asarray([row["sheet_xy_mm"] for row in contract["contacts"]]),
        [row["shared_axis_coordinate_mm"] for row in contract["contacts"]],
        changed,
    )
    assert (contract["hashes"]["contact_geometry_sha256"]
            == rebuilt["hashes"]["contact_geometry_sha256"])
    assert (contract["hashes"]["readout_parameters_sha256"]
            != rebuilt["hashes"]["readout_parameters_sha256"])


def test_features_keep_shaft_identity_when_axis_positions_collapse():
    contract = _contract()
    groups = contract_groups(contract)
    onset_icl = np.full((1, 15), np.nan)
    onset_scl = np.full((1, 15), np.nan)
    onset_icl[0, groups["ICL"][:2]] = [0.0, 1.0]
    onset_scl[0, groups["SCL"][:2]] = [0.0, 1.0]
    left = build_event_features(onset_icl, groups)["features"]
    right = build_event_features(onset_scl, groups)["features"]
    assert not np.array_equal(left, right)
    assert left[0, groups["ICL"]].sum() == 2
    assert right[0, groups["SCL"]].sum() == 2


def test_missing_shaft_is_a_precedence_state_not_a_deleted_pair():
    contract = _contract()
    groups = contract_groups(contract)
    pairs = contract_pairs(contract)
    onsets = np.full((3, 15), np.nan)
    onsets[:, groups["ICL"][:2]] = [[0, 1], [1, 0], [0, 1]]
    tables = pair_state_distributions(onsets, pairs)
    assert tables["ICL-SCL"].shape == (44, 4)
    assert np.allclose(tables["ICL-SCL"][:, 3], 1.0)
    assert np.allclose(tables["SCL-SCL"][:, 3], 1.0)


def test_cross_shaft_shift_does_not_change_within_shaft_precedence():
    contract = _contract()
    groups = contract_groups(contract)
    pairs = contract_pairs(contract)
    original = np.full((4, 15), np.nan)
    original[:, groups["ICL"][:2]] = [[0, 1], [0, 1], [1, 0], [1, 0]]
    original[:, groups["SCL"][:2]] = [[2, 3], [2, 3], [3, 2], [3, 2]]
    shifted = original.copy()
    shifted[:, groups["SCL"]] -= 4.0
    left = pair_state_distributions(original, pairs)
    right = pair_state_distributions(shifted, pairs)
    assert np.array_equal(left["ICL-ICL"], right["ICL-ICL"])
    assert np.array_equal(left["SCL-SCL"], right["SCL-SCL"])
    assert not np.array_equal(left["ICL-SCL"], right["ICL-SCL"])


def test_scl_censoring_is_visible_to_recruitment_and_precedence():
    contract = _contract()
    groups = contract_groups(contract)
    pairs = contract_pairs(contract)
    full = np.tile(np.arange(15.0), (8, 1))
    censored = full.copy()
    censored[:, groups["SCL"]] = np.nan
    target = describe_events(full, groups, pairs)
    candidate = describe_events(censored, groups, pairs)
    distances = descriptor_distances(candidate, target)
    assert distances["recruitment"]["SCL"] == 1.0
    assert distances["recruitment"]["ICL"] == 0.0
    assert distances["precedence"]["SCL-SCL"] > 0.0
    assert distances["precedence"]["ICL-SCL"] > 0.0


def test_patient_embedding_transforms_without_refitting_and_detects_censoring():
    contract = _contract()
    groups = contract_groups(contract)
    rng = np.random.default_rng(5)
    onsets = rng.normal(size=(80, 15))
    onsets[rng.random(onsets.shape) < 0.15] = np.nan
    features = build_event_features(onsets, groups)["features"]
    embedding = fit_patient_embedding(
        features, max_components=6, reference_n=50, n_directions=16, seed=7,
    )
    baseline = sliced_event_cloud_distance(features[:20], embedding)
    censored = onsets[:20].copy()
    censored[:, groups["SCL"]] = np.nan
    shifted = sliced_event_cloud_distance(
        build_event_features(censored, groups)["features"], embedding,
    )
    assert np.isfinite(baseline)
    assert shifted > baseline


def test_consensus_kmeans_and_label_alignment_are_label_invariant():
    rng = np.random.default_rng(8)
    z = np.r_[rng.normal(-2, 0.1, size=(20, 2)), rng.normal(2, 0.1, size=(20, 2))]
    reference = np.r_[np.zeros(20, dtype=int), np.ones(20, dtype=int)]
    result = consensus_kmeans(z, n_clusters=2, seeds=range(6), n_init=5)
    aligned = align_cluster_labels(result["labels"], reference)
    assert result["minimum_pairwise_ami"] == 1.0
    assert aligned["ami"] == 1.0
    assert aligned["accuracy"] == 1.0
