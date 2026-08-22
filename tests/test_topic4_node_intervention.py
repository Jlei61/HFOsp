import numpy as np
import pytest

from src.topic4_node_intervention import (
    early_support_probability,
    grid_covariates,
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
