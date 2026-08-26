from __future__ import annotations

import numpy as np

from src.topic5_continuous_marked_state_r1.data import load_event_stream
from src.topic5_continuous_marked_state_r1.raw_observation import RawAnchorReader


def test_real_raw_anchor_is_30_seconds_causal_and_contact_aligned() -> None:
    stream = load_event_stream("yuquan_huanghanwen")
    reader = RawAnchorReader(stream.subject, stream.event_time)
    anchor, split, session = reader.anchor_times()
    assert len(anchor) > 100
    assert np.all(np.diff(anchor) >= 0)
    observation = next(reader.read(float(value)) for value in anchor
                       if reader.read(float(value)) is not None)
    observation.validate()
    assert observation.waveform.shape == (87, 30 * 256)
    assert observation.explicit.shape == (87, 13)
    assert len(np.unique(observation.contact_names)) == 87
    assert set(np.unique(split)).issubset({0, 1})


def test_event_after_anchor_cannot_change_past_raw_observation() -> None:
    reader = RawAnchorReader("epilepsiae_620", np.empty(0, dtype=np.float64))
    anchor = float(reader.anchor_times()[0][0])
    without_future = reader.read(anchor)
    assert without_future is not None
    reader.event_times = np.asarray([anchor + 0.5], dtype=np.float64)
    with_future = reader.read(anchor)
    assert with_future is not None
    np.testing.assert_array_equal(without_future.sample_valid, with_future.sample_valid)
    np.testing.assert_array_equal(without_future.waveform, with_future.waveform)
    np.testing.assert_array_equal(without_future.explicit, with_future.explicit)


def test_past_ied_is_inpainted_without_exposing_its_mask_pattern() -> None:
    reader = RawAnchorReader("epilepsiae_620", np.empty(0, dtype=np.float64))
    anchor = float(reader.anchor_times()[0][0])
    without_ied = reader.read(anchor)
    assert without_ied is not None
    reader.event_times = np.asarray([anchor - 10.0], dtype=np.float64)
    with_ied = reader.read(anchor)
    assert with_ied is not None
    np.testing.assert_array_equal(without_ied.sample_valid, with_ied.sample_valid)
    assert not np.array_equal(without_ied.waveform, with_ied.waveform)


def test_dense_ied_window_is_removed_before_freezing_denominator() -> None:
    reader = RawAnchorReader("epilepsiae_620", np.empty(0, dtype=np.float64))
    anchor = float(reader.anchor_times()[0][0])
    start = anchor - 30.0
    # Consecutive +/-1 s cores tile the complete causal window, leaving no
    # background samples for the frozen inpainting contract.
    reader.event_times = np.arange(start + 1.0, anchor, 2.0, dtype=np.float64)
    assert not reader.can_read(anchor)
    assert reader.read(anchor) is None
