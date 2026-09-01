import numpy as np

from scripts.audit_topic4_dual_core_temporal_carrier import (
    _event_spectrum,
    _onset_spans,
)


def test_onset_spans_preserve_absolute_units_and_skip_unreadable_events():
    onsets = np.array([
        [0.010, 0.030, 0.050, np.nan],
        [0.010, np.nan, np.nan, np.nan],
    ])
    assert _onset_spans(onsets, scale_to_ms=1000.0) == [40.0]


def test_event_spectrum_recovers_unfiltered_62_5_hz_carrier():
    dt_ms = 2.0
    time_s = np.arange(1000) * dt_ms / 1000.0
    carrier = np.sin(2.0 * np.pi * 62.5 * time_s)
    envelope = np.tile(carrier, (15, 1))
    peak, centroid, ratio = _event_spectrum(
        envelope, dt_ms=dt_ms, onset_ms=800.0,
    )
    assert peak == 62.5
    assert 55.0 < centroid < 70.0
    assert ratio > 10.0


def test_event_spectrum_rejects_incomplete_window():
    envelope = np.ones((15, 40), dtype=float)
    assert _event_spectrum(envelope, dt_ms=2.0, onset_ms=20.0) is None
