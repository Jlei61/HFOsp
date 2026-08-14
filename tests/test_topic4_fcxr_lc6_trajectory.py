import numpy as np

from src.topic4_fcxr_lc6_trajectory import (
    NaturalCurrentObserver,
    apply_local_classifier,
    calibrate_local_classifier,
    cell_spatial_bins,
    coarse_field_mean,
    linear_slope,
    local_saturation_readout,
    observation_decision,
    per_second_cell_rates,
    spatial_map_persistence,
    spatial_rate_maps,
)


def test_observation_is_event_aligned_and_right_censoring_is_explicit():
    assert observation_decision(
        total_ms=50000, onset_ms=45000, n_returning_ied=10,
        c0_ied_to_onset=10, saturated_contiguous_1s=False,
    )["continue"] is True
    late = observation_decision(
        total_ms=65000, onset_ms=60000, n_returning_ied=10,
        c0_ied_to_onset=10, saturated_contiguous_1s=False,
    )
    assert late["right_censored"] is True
    exposed = observation_decision(
        total_ms=50000, onset_ms=None, n_returning_ied=15,
        c0_ied_to_onset=10, saturated_contiguous_1s=False,
    )
    assert exposed["reason"] == "NO_ONSET_SUFFICIENT_IED_EXPOSURE"


def test_registered_saturation_is_the_only_scientific_early_stop():
    verdict = observation_decision(
        total_ms=3000, onset_ms=1000, n_returning_ied=1,
        c0_ied_to_onset=10, saturated_contiguous_1s=True,
    )
    assert verdict["continue"] is False
    assert verdict["reason"] == "REGISTERED_SATURATION_1S"


def test_spatial_and_cell_rate_readouts_have_physical_units():
    positions = np.array([[.1, .1], [.2, .1], [1.1, .1], [1.2, .1]])
    bins, occupancy = cell_spatial_bins(positions, sheet_size_mm=2.0, n_bins_axis=2)
    # Two 1 s windows at dt=100 ms; cell 0 spikes twice in first, cell 2 once in second.
    maps = spatial_rate_maps(
        np.array([0, 5, 11]), np.array([0, 0, 2]), bins, occupancy,
        n_steps=20, dt_ms=100.0, window_ms=1000.0,
    )
    assert np.isclose(maps[0, bins[0]], 1.0)
    assert np.isclose(maps[1, bins[2]], .5)
    cell = per_second_cell_rates(
        np.array([0, 5, 11]), np.array([0, 0, 2]),
        n_steps=20, n_cells=4, dt_ms=100.0,
    )
    assert cell.tolist() == [[2., 0., 0., 0.], [0., 0., 1., 0.]]


def test_local_saturation_field_mean_slope_and_persistence():
    saturation = local_saturation_readout(
        np.array([[0., 450.], [0., 0.]]), refractory_ceiling_hz=500.0,
    )
    assert saturation["max_near_refractory_fraction"] == .5
    bins = np.array([0, 0, 1, 1])
    occupancy = np.array([2, 2])
    assert coarse_field_mean(np.array([1., 3., 2., 4.]), bins, occupancy).tolist() == [2., 3.]
    assert np.isclose(linear_slope([1., 2., 3.], dt_s=.5), 2.0)
    persistence = spatial_map_persistence(np.array([[0., 1., 2.], [0., 2., 4.]]))
    assert np.isclose(persistence["median_consecutive_correlation"], 1.0)


def test_natural_current_observer_samples_actual_signed_terms():
    class Config:
        E_E = 58.0
        e_gaba = 0.0

    class Slow:
        cfg = Config()
        _gI_last_E = np.array([.5, 1.0])

    observer = NaturalCurrentObserver(dt_ms=.05, sample_dt_ms=.1)
    for step in range(4):
        observer.sample(
            step, np.array([10., 20.]), np.array([1., 2.]),
            np.array([1.5, 3.0]), np.zeros(2), Slow(),
        )
    arrays = observer.arrays()
    np.testing.assert_allclose(arrays["current_time_ms"], [0.0, 0.1])
    assert arrays["F_E_mean"].shape == (2,)
    assert np.all(arrays["F_E_mean"] > 0)
    assert np.all(arrays["F_I_mean"] > 0)


def test_local_classifier_is_locked_on_returning_ied_tail_then_detects_sustained_area():
    occupancy = np.ones(4, dtype=int)
    c0_maps = np.zeros((10, 4), float)
    c0_maps[0] = [20., 2., 0., 0.]
    c0_maps[2] = [18., 3., 0., 0.]
    events = [
        {"t_on": 0., "t_off": 90., "returned": True},
        {"t_on": 200., "t_off": 290., "returned": True},
    ]
    lock = calibrate_local_classifier(
        c0_maps, occupancy, events, onset_ms=1000., sheet_size_mm=2.0,
        rate_quantile=.995, area_quantile=.99,
    )
    assert lock["n_pre_onset_returning_events"] == 2
    assert lock["persistence_windows"] == 5
    assert np.isclose(lock["component_area_threshold_mm2"], .99)
    arm_maps = np.zeros((10, 4), float)
    arm_maps[1:7] = 30.0
    result = apply_local_classifier(arm_maps, occupancy, lock)
    assert result["local_onset_ms"] == 100.0
    assert result["max_component_area_mm2"] == 4.0


def test_local_classifier_rejects_non_returning_or_post_onset_calibration_events():
    maps = np.zeros((2, 4), float)
    occupancy = np.ones(4, dtype=int)
    events = [{"t_on": 0., "t_off": 90., "returned": False}]
    try:
        calibrate_local_classifier(
            maps, occupancy, events, onset_ms=1000., sheet_size_mm=2.0,
        )
    except RuntimeError as exc:
        assert "no pre-onset returning IED" in str(exc)
    else:
        raise AssertionError("classifier lock must fail without eligible C0 IEDs")
