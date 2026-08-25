import json

import numpy as np

from src.topic4_node_dualmode import (
    calibrate_component_scales,
    contrast_r2,
    dual_mode_objective,
    event_source_onset_maps,
    event_features,
    fixed_projection_matrix,
    lse_max,
    matched_sample_dual_mode_objective,
    normalize_components_floor_ratio,
    normalize_event_ranks,
    shaft_balanced_feature_weights,
    sliced_wasserstein,
    soft_dual_mode_objective,
    soft_topology_network_reproducibility,
    topology_network_reproducibility,
    topology_reproducibility,
    weighted_sliced_wasserstein,
    weighted_r2,
)


NAMES = np.asarray(["ICL1", "ICL2", "ICL3", "SCL1", "SCL2"])


def _patient_modes(repeats=8):
    a = np.asarray([0, 1, 2, 3, 4], float)
    b = np.asarray([4, 3, 2, 1, 0], float)
    ranks = np.vstack([a for _ in range(repeats)] + [b for _ in range(repeats)])
    labels = np.asarray([0] * repeats + [1] * repeats)
    return ranks, labels


def test_feature_weights_balance_blocks_and_shafts():
    weights = shaft_balanced_feature_weights(NAMES)
    assert np.isclose(weights.sum(), 1.0)
    assert np.isclose(weights[:len(NAMES)].sum(), 0.5)
    assert np.isclose(weights[len(NAMES):].sum(), 0.5)
    assert np.isclose(weights[:3].sum(), 0.25)
    assert np.isclose(weights[3:5].sum(), 0.25)


def test_patient_modes_explain_more_than_global_mean():
    ranks, labels = _patient_modes()
    features = event_features(normalize_event_ranks(ranks))
    weights = shaft_balanced_feature_weights(NAMES)
    prototypes = np.asarray([features[labels == mode].mean(axis=0) for mode in (0, 1)])
    assert weighted_r2(features, labels, prototypes, features.mean(axis=0), weights) == 1.0


def test_contrast_scale_is_fit_on_training_target_only():
    patient, labels = _patient_modes()
    features = event_features(normalize_event_ranks(patient))
    prototypes = np.asarray([features[labels == mode].mean(axis=0) for mode in (0, 1)])
    weights = shaft_balanced_feature_weights(NAMES)
    result = contrast_r2(0.5 * prototypes, prototypes, prototypes, weights)
    assert np.isclose(result["train_fitted_nonnegative_scale"], 2.0)
    assert np.isclose(result["heldout_train_scaled_r2"], 1.0)


def test_worst_mode_cannot_be_hidden_by_dominant_mode():
    patient, labels = _patient_modes()
    projections = fixed_projection_matrix(2 * len(NAMES), n_directions=24, seed=4)
    good = dual_mode_objective(
        patient, labels, patient, labels, NAMES,
        missing_mode_penalty=1.0, projections=projections,
    )
    collapsed = dual_mode_objective(
        patient[labels == 0], np.zeros(np.sum(labels == 0), int),
        patient, labels, NAMES, missing_mode_penalty=1.0,
        projections=projections,
    )
    assert good["objective"] < collapsed["objective"]
    assert collapsed["modes"]["1"]["missing"]


def test_zero_event_network_is_penalized_not_dropped():
    patient, labels = _patient_modes()
    projections = fixed_projection_matrix(2 * len(NAMES), n_directions=12, seed=2)
    empty = dual_mode_objective(
        np.empty((0, len(NAMES))), np.empty(0, int), patient, labels, NAMES,
        missing_mode_penalty=1.0, projections=projections,
    )
    assert empty["modes"]["0"]["missing"]
    assert empty["modes"]["1"]["missing"]
    assert np.isclose(empty["occupancy_js"], np.log(2.0))


def test_contact_permutation_worsens_dual_mode_objective():
    patient, labels = _patient_modes()
    projections = fixed_projection_matrix(2 * len(NAMES), n_directions=32, seed=8)
    exact = dual_mode_objective(
        patient, labels, patient, labels, NAMES,
        missing_mode_penalty=1.0, projections=projections,
    )
    permutation = np.asarray([0, 3, 1, 4, 2])
    shuffled = dual_mode_objective(
        patient[:, permutation], labels, patient, labels, NAMES,
        missing_mode_penalty=1.0, projections=projections,
    )
    assert exact["objective"] < shuffled["objective"]


def test_recording_block_calibration_normalizes_component_scales():
    patient, labels = _patient_modes()
    patient[labels == 1, -1] = np.nan
    blocks = np.tile(np.repeat(np.arange(4), 2), 2)
    projections = fixed_projection_matrix(2 * len(NAMES), n_directions=16, seed=11)
    calibration = calibrate_component_scales(
        patient, labels, blocks, NAMES, projections,
        sample_size=2, draws=32, seed=12,
    )
    exact = dual_mode_objective(
        patient, labels, patient, labels, NAMES,
        missing_mode_penalty=1.0, projections=projections,
        calibration=calibration,
    )
    shuffled = dual_mode_objective(
        patient[:, [0, 3, 1, 4, 2]], labels, patient, labels, NAMES,
        missing_mode_penalty=1.0, projections=projections,
        calibration=calibration,
    )
    collapsed = dual_mode_objective(
        patient[labels == 0], np.zeros(np.sum(labels == 0), int),
        patient, labels, NAMES, missing_mode_penalty=1.0,
        projections=projections, calibration=calibration,
    )
    assert calibration["floor_resampling"].startswith("different recording blocks")
    assert exact["objective"] < shuffled["objective"]
    assert collapsed["objective"] > exact["objective"]
    assert "raw" in exact["modes"]["0"]
    json.dumps(exact)


def _unit_floor_calibration():
    return {
        "modes": {
            str(mode): {
                key: {"floor_q95": 1.0}
                for key in ("recruitment", "precedence", "profile", "cloud")
            }
            for mode in (0, 1)
        }
    }


def test_continuous_floor_ratio_does_not_clip_subfloor_differences():
    values = {
        "recruitment": 0.2, "precedence": 0.3,
        "profile": 0.4, "cloud": 0.5,
    }
    normalized = normalize_components_floor_ratio(
        values, _unit_floor_calibration(), 0,
    )
    assert normalized == values


def test_matched_sampling_prevents_extra_model_events_from_buying_a_lower_score():
    patient, labels = _patient_modes()
    blocks = np.tile(np.repeat(np.arange(4), 2), 2)
    projections = fixed_projection_matrix(2 * len(NAMES), n_directions=16, seed=33)
    kwargs = dict(
        patient_ranks=patient, patient_labels=labels, patient_blocks=blocks,
        contact_names=NAMES, projections=projections,
        calibration=_unit_floor_calibration(), sample_size=2, draws=32, seed=34,
    )
    once = matched_sample_dual_mode_objective(
        patient, labels, **kwargs,
    )
    repeated = matched_sample_dual_mode_objective(
        np.repeat(patient, 5, axis=0), np.repeat(labels, 5), **kwargs,
    )
    assert np.isclose(once["objective"], repeated["objective"])
    shuffled = matched_sample_dual_mode_objective(
        patient[:, [0, 3, 1, 4, 2]], labels, **kwargs,
    )
    assert once["objective"] < shuffled["objective"]


def test_duplicate_event_count_does_not_change_sliced_wasserstein():
    patient, _ = _patient_modes()
    features = event_features(normalize_event_ranks(patient))
    weights = shaft_balanced_feature_weights(NAMES)
    projections = fixed_projection_matrix(features.shape[1], n_directions=16, seed=3)
    once = sliced_wasserstein(
        features, features, weights=weights, projections=projections,
    )
    repeated = sliced_wasserstein(
        np.repeat(features, 3, axis=0), features,
        weights=weights, projections=projections,
    )
    assert once < 1e-12
    assert repeated < 1e-12


def test_weighted_sliced_wasserstein_is_invariant_to_exact_replication():
    patient, _ = _patient_modes()
    features = event_features(normalize_event_ranks(patient))
    feature_weights = shaft_balanced_feature_weights(NAMES)
    projections = fixed_projection_matrix(features.shape[1], n_directions=16, seed=5)
    once = weighted_sliced_wasserstein(
        features, features, x_event_weights=np.ones(len(features)),
        weights=feature_weights, projections=projections,
    )
    repeated = weighted_sliced_wasserstein(
        np.repeat(features, 4, axis=0), features,
        x_event_weights=np.ones(4 * len(features)),
        weights=feature_weights, projections=projections,
    )
    assert once < 1e-12
    assert repeated < 1e-12


def test_soft_objective_prefers_two_confident_patient_modes_to_ambiguous_cloud():
    patient, labels = _patient_modes(repeats=12)
    projections = fixed_projection_matrix(2 * len(NAMES), n_directions=24, seed=41)
    kwargs = dict(
        patient_ranks=patient, patient_labels=labels, contact_names=NAMES,
        projections=projections, calibration=_unit_floor_calibration(),
    )
    confident = soft_dual_mode_objective(
        patient, labels.astype(float), **kwargs,
    )
    ambiguous = soft_dual_mode_objective(
        patient, np.full(len(patient), 0.5), **kwargs,
    )
    one_mode = soft_dual_mode_objective(
        patient[labels == 0], np.zeros(np.sum(labels == 0)), **kwargs,
    )
    assert confident["objective"] < ambiguous["objective"]
    assert confident["objective"] < one_mode["objective"]
    assert ambiguous["ambiguity"] == 1.0
    assert one_mode["modes"]["1"]["missing"]


def test_soft_objective_changes_continuously_across_classifier_midpoint():
    patient, labels = _patient_modes(repeats=10)
    projections = fixed_projection_matrix(2 * len(NAMES), n_directions=16, seed=42)
    kwargs = dict(
        patient_ranks=patient, patient_labels=labels, contact_names=NAMES,
        projections=projections, calibration=_unit_floor_calibration(),
    )
    below = soft_dual_mode_objective(
        patient, np.where(labels == 1, 0.4999, 0.0001), **kwargs,
    )
    above = soft_dual_mode_objective(
        patient, np.where(labels == 1, 0.5001, 0.0001), **kwargs,
    )
    assert abs(below["objective"] - above["objective"]) < 1e-3


def _topology_maps(*, random=False, identical=False, seed=0):
    rng = np.random.default_rng(seed)
    maps, labels = [], []
    for mode in (0, 1):
        for _ in range(12):
            onset = np.full((8, 8), np.nan)
            if random:
                sites = rng.choice(64, size=2, replace=False)
            elif identical:
                sites = np.asarray([9, 46])
            else:
                sites = np.asarray([9, 46]) if mode == 0 else np.asarray([14, 41])
            for rank, site in enumerate(sites):
                y, x = divmod(int(site), 8)
                onset[y, x] = float(rank)
                onset[max(0, y - 1):min(8, y + 2), max(0, x - 1):min(8, x + 2)] = rank + 1.0
            maps.append(onset)
            labels.append(mode)
    return np.asarray(maps), np.asarray(labels)


def test_stable_two_hotspot_modes_are_reproducible_without_single_source_assumption():
    maps, labels = _topology_maps()
    result = topology_reproducibility(maps, labels)
    assert result["mean_within_mode_reliability"] > 0.99
    assert result["between_mode_distance"] > 0.2


def test_random_hotspots_fail_topology_reproducibility():
    stable_maps, labels = _topology_maps()
    random_maps, _ = _topology_maps(random=True, seed=7)
    stable = topology_reproducibility(stable_maps, labels)
    random = topology_reproducibility(random_maps, labels)
    assert stable["mean_within_mode_reliability"] > random["mean_within_mode_reliability"]


def test_identical_mode_topologies_fail_separation():
    maps, labels = _topology_maps(identical=True)
    result = topology_reproducibility(maps, labels)
    assert result["mean_within_mode_reliability"] > 0.99
    assert result["between_mode_distance"] < 1e-12


def test_network_topology_reliability_weights_networks_equally():
    maps, labels = _topology_maps()
    result = topology_network_reproducibility(
        [maps[:12], maps[12:], maps],
        [labels[:12], labels[12:], labels],
    )
    assert result["modes"]["0"]["n_networks"] == 2
    assert result["modes"]["1"]["n_networks"] == 2
    assert result["mean_across_network_template_cosine"] > 0.99
    assert result["equal_network_between_mode_distance"] > 0.2


def test_soft_topology_requires_probability_specific_templates():
    maps, labels = _topology_maps()
    coupled = soft_topology_network_reproducibility(
        [maps, maps], [labels.astype(float), labels.astype(float)],
    )
    ambiguous = soft_topology_network_reproducibility(
        [maps, maps], [np.full(len(maps), 0.5), np.full(len(maps), 0.5)],
    )
    assert coupled["mean_across_network_template_cosine"] > 0.99
    assert coupled["equal_network_between_mode_distance"] > 0.2
    assert ambiguous["equal_network_between_mode_distance"] < 1e-12


def test_lse_equal_inputs_has_no_log_two_offset():
    assert np.isclose(lse_max(np.asarray([2.0, 2.0]), tau=0.25), 2.0)


def test_full_spikes_reduce_to_evaluable_event_source_map():
    spikes = np.zeros((400, 4), bool)
    positions = np.asarray([[0.1, 0.1], [0.2, 0.2], [2.1, 2.1], [2.2, 2.2]])
    spikes[200:204, :2] = True
    spikes[210:214, 2:] = True
    result = event_source_onset_maps(
        spikes, positions, np.asarray([200.0, 250.0]),
        np.asarray([True, False]), dt_ms=1.0, sheet_mm=4.0,
    )
    onset = result["onset_maps_ms"][0]
    assert result["evaluable"].tolist() == [True, False]
    assert np.isfinite(onset[0, 0])
    assert np.isfinite(onset[2, 2])
    assert onset[0, 0] < onset[2, 2]
    assert np.isnan(result["onset_maps_ms"][1]).all()
    assert result["activity_counts"].dtype == np.uint16
    assert result["activity_counts"][0, :, 0, 0].max() == 2
    assert not np.any(result["activity_counts"][1])
