import importlib.util
from pathlib import Path

import numpy as np
import pytest

from src.topic4_mz_spatial_recruitment_gate0 import (
    causal_frame_end_times,
    effective_extent,
    first_crossing_ms,
    frame_average_trace,
    participation_ratio,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_topic4_mz_spatial_recruitment_gate0.py"
SPEC = importlib.util.spec_from_file_location("run_topic4_mz_spatial_recruitment_gate0", SCRIPT)
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)


def test_effective_extent_recovers_mean_peak_factorisation_and_zero_state():
    mean = np.array([0.0, 0.1, 0.6])
    peak = np.array([0.0, 0.5, 0.75])
    observed = effective_extent(mean, peak)
    np.testing.assert_allclose(observed, [0.0, 0.2, 0.8])
    np.testing.assert_allclose(observed * peak, mean)


def test_effective_extent_rejects_mean_larger_than_peak():
    with pytest.raises(ValueError, match="mean cannot exceed"):
        effective_extent([0.2], [0.1])


def test_participation_ratio_has_known_single_bin_and_uniform_limits():
    frames = np.array([[1.0, 0.0, 0.0, 0.0], [2.0, 2.0, 2.0, 2.0], [0.0] * 4])
    np.testing.assert_allclose(participation_ratio(frames), [0.25, 1.0, 0.0])


def test_participation_ratio_respects_static_occupancy_mask():
    frames = np.array([[1.0, 1.0, 100.0]])
    observed = participation_ratio(frames, valid_mask=np.array([True, True, False]))
    np.testing.assert_allclose(observed, [1.0])


def test_frame_average_trace_uses_exact_nonoverlapping_saved_frames():
    observed = frame_average_trace(
        np.arange(12.0),
        dt_ms=1.0,
        frame_starts_ms=[0.0, 4.0, 8.0],
        frame_duration_ms=4.0,
    )
    np.testing.assert_allclose(observed, [1.5, 5.5, 9.5])


def test_saved_frame_is_causally_available_only_at_its_end():
    np.testing.assert_allclose(
        causal_frame_end_times([0.0, 25.0], frame_duration_ms=25.0),
        [25.0, 50.0],
    )


def test_first_crossing_ms_is_causal_and_returns_none_without_crossing():
    assert first_crossing_ms([0.0, 0.2, 0.5], threshold=0.2, dt_ms=2.0) == 2.0
    assert first_crossing_ms([0.0, 0.1], threshold=0.2, dt_ms=2.0) is None


def test_runner_import_reuses_exact_snn_gate_and_counts_consecutive_blocks():
    assert RUNNER.slow_gate_drive(0.25, A0=0.15, A50=0.10, exponent=4.0) == pytest.approx(0.5)
    assert RUNNER._longest_true_run(np.array([False, True, True, False, True])) == 2
