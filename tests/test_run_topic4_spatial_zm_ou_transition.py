import numpy as np

from scripts.run_topic4_spatial_zm_ou_transition import (
    tonic_contact_recruitment_diagnosis,
)


def test_tonic_contact_rates_do_not_require_a_500ms_rhythm_baseline():
    dt_ms = 10.0
    onset_ms = 400.0
    spikes = np.zeros((180, 2), dtype=bool)
    spikes[:40, 0] = np.arange(40) % 10 == 0
    spikes[70:170, 0] = True
    spikes[:40, 1] = np.arange(40) % 20 == 0
    spikes[70:170, 1] = True
    rows = tonic_contact_recruitment_diagnosis(
        spikes=spikes,
        positions_e=np.asarray([[0.0, 0.0], [2.0, 0.0]]),
        contact_names=["SCL1", "ICL1"],
        contact_xy=np.asarray([[0.0, 0.0], [2.0, 0.0]]),
        shaft_ids=["SCL", "ICL"],
        dt_ms=dt_ms,
        onset_ms=onset_ms,
        radius_mm=0.5,
    )
    assert len(rows) == 2
    assert all(row["local_rate_post_hz"] == 100.0 for row in rows)
    assert all(row["local_rate_ratio_post_over_pre"] > 5.0 for row in rows)


def test_tonic_contact_rates_fail_closed_on_short_post_window():
    with np.testing.assert_raises_regex(ValueError, "post window is incomplete"):
        tonic_contact_recruitment_diagnosis(
            spikes=np.zeros((100, 1), dtype=bool),
            positions_e=np.asarray([[0.0, 0.0]]),
            contact_names=["SCL1"],
            contact_xy=np.asarray([[0.0, 0.0]]),
            shaft_ids=["SCL"],
            dt_ms=10.0,
            onset_ms=400.0,
            radius_mm=0.5,
        )
