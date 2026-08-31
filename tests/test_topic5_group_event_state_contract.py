import numpy as np

from src.topic5_group_event_state.contract import (
    adjacent_bipolar_label,
    build_source_pointer,
    map_detector_channels,
    relative_participant_delay,
    supported_band_mask,
)


def test_relative_delay_masks_legacy_phantom_values():
    lag = np.array([[10.0, 9.0, 12.0], [3.0, 4.0, 99.0]])
    part = np.array([[True, False, True], [True, True, False]])
    got = relative_participant_delay(lag, part)
    np.testing.assert_allclose(got[0, [0, 2]], [0.0, 2.0])
    np.testing.assert_allclose(got[1, :2], [0.0, 1.0])
    assert np.isnan(got[0, 1])
    assert np.isnan(got[1, 2])


def test_reference_specific_detector_mapping():
    assert adjacent_bipolar_label("E11") == "E11-E12"
    mapped, failures = map_detector_channels(
        "yuquan", ["E11", "K3"], ["E11-E12", "K3-K4"]
    )
    assert mapped == ["E11-E12", "K3-K4"]
    assert failures == []

    mapped, failures = map_detector_channels(
        "epilepsiae", ["GD8", "GE7"], ["GD8", "GE7"]
    )
    assert mapped == ["GD8", "GE7"]
    assert failures == []


def test_band_availability_is_missing_not_zero():
    at_256 = supported_band_mask(256.0)
    assert at_256["low_ripple"] is True
    assert at_256["ripple"] is False
    assert at_256["fast_ripple"] is False
    at_1024 = supported_band_mask(1024.0)
    assert all(at_1024.values())


def test_native_pointer_preserves_packed_core_and_shoulders():
    pointer = build_source_pointer(
        dataset="epilepsiae",
        subject="958",
        record_name="95800102_0000",
        source_block_id=0,
        source_event_row=0,
        raw_path="block.data",
        head_path="block.head",
        native_rate_hz=1024.0,
        block_start_epoch=1000.0,
        core_start_seconds=18.507,
        core_end_seconds=18.757,
        detector_reference="car_global_legacy",
        n_native_samples=3600 * 1024,
    )
    assert pointer.core_start_sample == round(18.507 * 1024)
    assert pointer.core_stop_sample == round(18.757 * 1024)
    assert pointer.context_start_sample == round((18.507 - 0.25) * 1024)
    assert pointer.context_stop_sample == round((18.757 + 0.25) * 1024)
    assert pointer.event_abs_time == 1018.507


def test_tied_groups_split_only_beyond_one_centroid_hop():
    from src.topic5_group_event_state.contract import (
        TIE_TOLERANCE_SECONDS,
        tied_recruitment_groups,
    )

    # Two participants inside one hop are one recruitment step; the legacy rank
    # would have forced a spurious total order on them.
    delay = np.array([0.0, TIE_TOLERANCE_SECONDS * 0.5, 0.030, 0.0355, 7.0])
    part = np.array([True, True, True, True, False])
    groups = tied_recruitment_groups(delay, part)
    assert groups == [[0, 1], [2, 3]]
    assert all(4 not in g for g in groups)


def test_tied_groups_are_ordered_earliest_first_and_exclude_nonfinite():
    from src.topic5_group_event_state.contract import tied_recruitment_groups

    delay = np.array([0.100, np.nan, 0.000, 0.050])
    part = np.array([True, True, True, True])
    groups = tied_recruitment_groups(delay, part)
    assert groups == [[2], [3], [0]]


def test_lagpat_variants_pair_their_own_packed_file():
    from src.topic5_group_event_state.contract import LAGPAT_VARIANTS

    by_name = {v.name: v for v in LAGPAT_VARIANTS}
    assert by_name["withFreqCent"].packed_suffix == "_packedTimes_withFreqCent.npy"
    assert by_name["legacy"].packed_suffix == "_packedTimes.npy"
    # withFreqCent must be preferred, so it has to come first.
    assert LAGPAT_VARIANTS[0].name == "withFreqCent"
