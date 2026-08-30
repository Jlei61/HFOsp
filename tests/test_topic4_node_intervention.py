import numpy as np
import pytest

from src.topic4_node_intervention import (
    crossed_hotspot_selectivity,
    early_support_probability,
    grid_covariates,
    intervention_footprint_covariates,
    network_balanced_early_support,
    representative_event_index,
    select_hotspot_triplet,
    select_representative_seed,
)


def test_early_support_probability_recovers_repeated_two_site_sources():
    maps = np.full((6, 8, 8), np.nan)
    maps[:, 2, 2] = 0.0
    maps[:, 6, 6] = 1.0
    maps[:, 3:5, 3:5] = 20.0
    probability = early_support_probability(maps, fraction=0.5)
    assert probability[2, 2] == 1.0
    assert probability[6, 6] == 1.0
    assert probability[0, 0] == 0.0


def test_early_support_is_equal_network_not_equal_event():
    many = np.full((10, 4, 4), np.nan)
    many[:, 0, 0] = 0.0
    one = np.full((1, 4, 4), np.nan)
    one[:, 3, 3] = 0.0
    probability = network_balanced_early_support(
        [many, one], [np.zeros(10, int), np.zeros(1, int)], mode=0,
    )
    assert probability[0, 0] == pytest.approx(0.5)
    assert probability[3, 3] == pytest.approx(0.5)


def test_representative_seed_uses_median_eligible_network():
    scores = [
        {"seed": 3, "objective": 0.9, "n_events": 50, "mode_counts": [20, 30]},
        {"seed": 1, "objective": 0.3, "n_events": 40, "mode_counts": [20, 20]},
        {"seed": 2, "objective": 0.5, "n_events": 45, "mode_counts": [20, 25]},
        {"seed": 4, "objective": 0.4, "n_events": 50, "mode_counts": [50, 0]},
    ]
    assert select_representative_seed(scores, {1: 10, 2: 10, 3: 10, 4: 10}) == 2


def test_hotspot_triplet_uses_spatially_separated_matched_control():
    probability = np.zeros((8, 8))
    probability[2, 2] = 1.0
    probability[6, 6] = 0.8
    h = np.linspace(0.0, 1.0, 64).reshape(8, 8)
    density = np.full((8, 8), 8.0)
    rate = np.full((8, 8), 2.0)
    h[0, 2] = h[2, 2]
    triplet = select_hotspot_triplet(
        probability,
        {"h_mean": h, "e_density": density, "baseline_rate_hz": rate},
        bin_mm=1.0,
        minimum_separation_mm=3.0,
    )
    assert (triplet["dominant"]["row"], triplet["dominant"]["column"]) == (2, 2)
    assert (triplet["secondary"]["row"], triplet["secondary"]["column"]) == (6, 6)
    control = triplet["matched_off_template"]
    assert control["early_probability"] == 0.0
    assert np.linalg.norm(
        np.asarray(control["xy_mm"]) - np.asarray(triplet["dominant"]["xy_mm"])
    ) >= 3.0


def test_hotspot_triplet_rejects_an_invented_zero_support_secondary():
    probability = np.zeros((8, 8))
    probability[2, 2] = 1.0
    covariates = {
        "h_mean": np.ones((8, 8)),
        "e_density": np.ones((8, 8)),
        "baseline_rate_hz": np.ones((8, 8)),
    }
    with pytest.raises(ValueError, match="supported spatially separated"):
        select_hotspot_triplet(probability, covariates, bin_mm=1.0)


def test_grid_covariates_preserve_density_h_and_rate_units():
    positions = np.asarray([[0.2, 0.2], [0.8, 0.7], [1.2, 1.4]])
    h = np.asarray([0.2, 0.6, 0.9])
    spikes = np.zeros((1000, 3), bool)
    spikes[[10, 20], 0] = True
    spikes[[30, 40], 1] = True
    spikes[[50], 2] = True
    result = grid_covariates(
        positions, h, spikes, dt_ms=1.0, sheet_mm=2.0, bin_mm=1.0,
    )
    assert result["e_density"][0, 0] == 2
    assert result["h_mean"][0, 0] == pytest.approx(0.4)
    assert result["baseline_rate_hz"][0, 0] == pytest.approx(2.0)
    assert result["baseline_rate_hz"][1, 1] == pytest.approx(1.0)


def test_intervention_covariates_match_the_actual_circular_target():
    positions = np.asarray([[0.2, 0.2], [0.8, 0.7], [1.2, 1.4]])
    h = np.asarray([0.2, 0.6, 0.9])
    spikes = np.zeros((1000, 3), bool)
    spikes[[10, 20], 0] = True
    spikes[[30, 40], 1] = True
    spikes[[50], 2] = True
    result = intervention_footprint_covariates(
        positions, h, spikes, dt_ms=1.0, sheet_mm=2.0, bin_mm=1.0,
        target_radius_mm=0.8,
    )
    assert result["e_density"][0, 0] == 2
    assert result["h_mean"][0, 0] == pytest.approx(0.4)
    assert result["baseline_rate_hz"][0, 0] == pytest.approx(2.0)
    assert result["covariate_footprint"] == "circular_intervention_target"


def test_mode_contrast_avoids_a_shared_absolute_hotspot():
    mode0 = np.zeros((8, 8))
    mode1 = np.zeros((8, 8))
    mode0[2, 2] = mode1[2, 2] = 1.0
    mode0[2, 6] = 0.8
    mode1[6, 2] = 0.9
    covariates = {
        "h_mean": np.ones((8, 8)),
        "e_density": np.full((8, 8), 8.0),
        "baseline_rate_hz": np.full((8, 8), 2.0),
    }
    target = select_hotspot_triplet(
        mode0, covariates, bin_mm=1.0, minimum_separation_mm=3.0,
        competing_probability=mode1, require_positive_contrast=True,
        maximum_standardized_l1=2.0,
        maximum_standardized_component=1.0,
    )
    assert (target["dominant"]["row"], target["dominant"]["column"]) == (2, 6)
    assert target["dominant"]["mode_probability_contrast"] == pytest.approx(0.8)
    assert target["match_quality"]["acceptable"] is True


def test_representative_event_is_joint_medoid():
    maps = np.full((3, 4, 4), np.nan)
    maps[0, 0, 0] = 0.0
    maps[1, 0, 0] = 0.0
    maps[1, 0, 1] = 1.0
    maps[2, 3, 3] = 0.0
    ranks = np.asarray([
        [0.0, 1.0, np.nan],
        [0.0, 1.0, 2.0],
        [2.0, 1.0, 0.0],
    ])
    assert representative_event_index(maps, ranks, np.zeros(3, int), 0) == 1


def test_empty_source_maps_are_not_silently_treated_as_controls():
    with pytest.raises(ValueError, match="no evaluable event"):
        early_support_probability(np.full((2, 4, 4), np.nan))


def _branch(event_occurred=True, latency=40.0):
    return {
        "event_occurred": event_occurred,
        "latency_from_checkpoint_ms": None if not event_occurred else latency,
    }


def _crossed_network(seed, *, mode0_selective=True, mode1_selective=False):
    mode0_hot_on_0 = _branch(False) if mode0_selective else _branch(True, 40.0)
    mode1_hot_on_1 = _branch(False) if mode1_selective else _branch(True, 40.0)
    return {
        "network_seed": seed,
        "native_modes": {
            "0": {
                "sham": _branch(True, 40.0),
                "mode0_hotspot": mode0_hot_on_0,
                "mode0_matched_off_template": _branch(True, 42.0),
                "mode1_hotspot": _branch(True, 40.0),
                "mode1_matched_off_template": _branch(True, 40.0),
            },
            "1": {
                "sham": _branch(True, 40.0),
                "mode0_hotspot": _branch(True, 43.0),
                "mode0_matched_off_template": _branch(True, 40.0),
                "mode1_hotspot": mode1_hot_on_1,
                "mode1_matched_off_template": _branch(True, 42.0),
            },
        },
    }


def test_crossed_hotspot_requires_own_mode_effect_beyond_cross_and_control():
    records = [
        _crossed_network(1, mode0_selective=True),
        _crossed_network(2, mode0_selective=True),
        _crossed_network(3, mode0_selective=False),
    ]
    result = crossed_hotspot_selectivity(records, required_networks=2)
    assert result["node_freeze_permitted"] is True
    assert result["selective_hotspot_modes"] == [0]
    assert result["modes"]["0"]["selective_network_count"] == 2
    assert result["modes"]["1"]["selective_network_count"] == 0


def test_crossed_hotspot_rejects_general_suppression():
    records = []
    for seed in (1, 2, 3):
        row = _crossed_network(seed, mode0_selective=True)
        row["native_modes"]["1"]["mode0_hotspot"] = _branch(False)
        records.append(row)
    result = crossed_hotspot_selectivity(records, required_networks=2)
    assert result["node_freeze_permitted"] is False


@pytest.mark.parametrize("failure", ["unmatched", "overlapping"])
def test_crossed_hotspot_rejects_invalid_spatial_controls(failure):
    records = [
        _crossed_network(seed, mode0_selective=True) for seed in (1, 2, 3)
    ]
    for row in records:
        native = row["native_modes"]["0"]
        if failure == "unmatched":
            native["mode0_matched_off_template"][
                "control_match_acceptable"
            ] = False
        else:
            native["cross_mode_hotspots_distinct"] = False
    result = crossed_hotspot_selectivity(records, required_networks=2)
    assert result["modes"]["0"]["pass"] is False
    assert result["node_freeze_permitted"] is False
