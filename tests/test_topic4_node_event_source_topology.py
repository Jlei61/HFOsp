import numpy as np

from scripts.paper_figures.audit_fig4_node_event_source_topology import (
    _component_summary,
    persistent_recruitment_onsets,
    source_model_cv,
)


def test_persistent_onset_rejects_single_frame_flash():
    baseline = np.zeros((20, 4, 4))
    event = np.zeros((6, 4, 4))
    event[1, 1, 1] = 3
    event[3:5, 2, 2] = 3
    onset = persistent_recruitment_onsets(
        event, baseline, np.arange(6) * 2.0, persistence_frames=2,
    )
    assert not np.isfinite(onset[1, 1])
    assert onset[2, 2] == 6.0


def test_component_summary_distinguishes_two_early_sources():
    onset = np.full((10, 10), np.nan)
    onset[1:3, 1:3] = 0.0
    onset[7:9, 7:9] = 0.0
    onset[3:7, 3:7] = 4.0
    summary = _component_summary(onset)
    assert summary["n_early_components"] == 2
    assert summary["dominant_early_component_fraction"] == 0.5


def test_radial_wave_prefers_single_source_without_large_two_source_gain():
    yy, xx = np.mgrid[:15, :15]
    onset = np.hypot(xx - 7, yy - 7)
    one = source_model_cv(onset, source_count=1)
    two = source_model_cv(onset, source_count=2)
    assert one["cv_r2"] > 0.95
    assert two["cv_r2"] - one["cv_r2"] < 0.05
