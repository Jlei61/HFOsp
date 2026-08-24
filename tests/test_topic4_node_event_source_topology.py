import numpy as np

from scripts.paper_figures.audit_fig4_node_event_source_topology import (
    _component_summary,
    persistent_recruitment_onsets,
    source_model_cv,
)
from src.topic4_node_dualmode import (
    causal_direction_alignment,
    causal_root_displacement,
    causal_wave_monotonicity,
    causal_wave_monotonicity_alignment,
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


def _horizontal_wave(reverse=False):
    onset = np.full((5, 8), np.nan)
    onset[1:4] = np.arange(8, dtype=float)[None, :]
    if reverse:
        onset[1:4] = onset[1:4, ::-1]
    return onset


def test_causal_root_displacement_recovers_early_to_late_direction():
    forward = causal_root_displacement(_horizontal_wave(), tail_fraction=0.2)
    reverse = causal_root_displacement(
        _horizontal_wave(reverse=True), tail_fraction=0.2,
    )
    assert forward["evaluable"] and reverse["evaluable"]
    assert forward["displacement_xy_mm"][0] > 0.0
    assert reverse["displacement_xy_mm"][0] < 0.0


def test_causal_direction_alignment_requires_two_opposite_root_modes():
    opposite = causal_direction_alignment(
        np.asarray([_horizontal_wave(), _horizontal_wave(reverse=True)]),
        np.asarray([0, 1]), axis_unit=np.asarray([1.0, 0.0]),
        expected_mode_signs=np.asarray([1.0, -1.0]),
    )
    same_direction = causal_direction_alignment(
        np.asarray([_horizontal_wave(), _horizontal_wave()]),
        np.asarray([0, 1]), axis_unit=np.asarray([1.0, 0.0]),
        expected_mode_signs=np.asarray([1.0, -1.0]),
    )
    assert opposite["score"] > 0.99
    assert same_direction["score"] == 0.0


def test_causal_direction_does_not_reward_eventwise_sign_cancellation():
    mixed = causal_direction_alignment(
        np.asarray([
            _horizontal_wave(), _horizontal_wave(reverse=True),
            _horizontal_wave(reverse=True), _horizontal_wave(),
        ]),
        np.asarray([0, 0, 1, 1]), axis_unit=np.asarray([1.0, 0.0]),
        expected_mode_signs=np.asarray([1.0, -1.0]),
    )
    assert abs(mixed["modes"]["0"]["mean_signed_axis_cosine"]) < 1e-12
    assert abs(mixed["modes"]["1"]["mean_signed_axis_cosine"]) < 1e-12
    assert mixed["score"] == 0.0


def test_causal_wave_monotonicity_uses_the_complete_onset_map():
    forward = causal_wave_monotonicity(
        _horizontal_wave(), axis_unit=np.asarray([1.0, 0.0]),
    )
    flash = _horizontal_wave()
    flash[np.isfinite(flash)] = 0.0
    synchronous = causal_wave_monotonicity(
        flash, axis_unit=np.asarray([1.0, 0.0]),
    )
    assert forward["evaluable"]
    assert forward["axis_time_spearman"] > 0.99
    assert not synchronous["evaluable"]


def test_causal_wave_monotonicity_requires_opposite_mode_sequences():
    opposite = causal_wave_monotonicity_alignment(
        np.asarray([_horizontal_wave(), _horizontal_wave(reverse=True)]),
        np.asarray([0, 1]), axis_unit=np.asarray([1.0, 0.0]),
        expected_mode_signs=np.asarray([1.0, -1.0]),
    )
    same_direction = causal_wave_monotonicity_alignment(
        np.asarray([_horizontal_wave(), _horizontal_wave()]),
        np.asarray([0, 1]), axis_unit=np.asarray([1.0, 0.0]),
        expected_mode_signs=np.asarray([1.0, -1.0]),
    )
    assert opposite["score"] > 0.99
    assert same_direction["score"] == 0.0


def test_causal_wave_monotonicity_clips_after_mode_mean():
    mixed = causal_wave_monotonicity_alignment(
        np.asarray([
            _horizontal_wave(), _horizontal_wave(reverse=True),
            _horizontal_wave(reverse=True), _horizontal_wave(),
        ]),
        np.asarray([0, 0, 1, 1]), axis_unit=np.asarray([1.0, 0.0]),
        expected_mode_signs=np.asarray([1.0, -1.0]),
    )
    assert mixed["score"] == 0.0
