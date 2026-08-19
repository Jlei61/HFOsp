"""A nested ndarray in the payload must never cost a completed run.

Regression: state_characterization contains nested dicts whose values are
ndarrays (active/silent duration distributions). A top-level-only filter let one
through, json.dump raised, and a 92-minute simulation was lost at its final
write. Two guards now: the payload is sanitised recursively, and the arrays are
written BEFORE the json so a serialisation failure can never destroy the run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "snn_engine"))

from scripts.run_topic4_zm_ictal_transition_worker import _json_safe  # noqa: E402


def test_nested_ndarray_survives_json_dump():
    payload = {"state_characterization": {
        "active_durations_ms": np.array([1.0, 2.0]),
        "length_matched_interictal": {          # the level the old filter missed
            "silent_durations_ms": np.array([3.0, 4.0]),
            "n_bursts": np.int64(7)},
        "band_proxy_ictal": {"peak_frequency_hz": np.float32(45.0)}}}
    text = json.dumps(_json_safe(payload))
    back = json.loads(text)
    assert back["state_characterization"]["length_matched_interictal"]["silent_durations_ms"] == [3.0, 4.0]
    assert back["state_characterization"]["length_matched_interictal"]["n_bursts"] == 7


def test_non_finite_becomes_null_rather_than_invalid_json():
    """NaN/inf make json.dumps emit bare NaN, which is not valid JSON and breaks
    every downstream reader."""
    text = json.dumps(_json_safe({"a": np.float64("nan"), "b": float("inf"),
                                  "c": np.array([np.nan, 1.0])}))
    assert "NaN" not in text and "Infinity" not in text
    back = json.loads(text)
    assert back["a"] is None and back["b"] is None and back["c"] == [None, 1.0]


def test_ordinary_values_pass_through_unchanged():
    payload = {"n": 3, "x": 1.5, "s": "ok", "flag": True, "none": None,
               "nested": {"list": [1, 2, {"deep": "value"}]}}
    assert json.loads(json.dumps(_json_safe(payload))) == payload


def test_arrays_are_written_before_the_json():
    """Ordering is the second guard: if json ever fails again, the simulation
    output still survives on disk."""
    source = (ROOT / "scripts/run_topic4_zm_ictal_transition_worker.py").read_text()
    npz_at = source.index("_atomic_npz(out_npz")
    json_at = source.index("atomic_write_json(_json_safe(payload)")
    assert npz_at < json_at


def test_pass2_segment_includes_the_target_checkpoint_step():
    """Regression: the continuation length excluded its endpoint, and the
    pre-ictal checkpoint sits exactly at that endpoint. Every pre-ictal
    checkpoint in the first canary batch was lost to one missing step."""
    source = (ROOT / "scripts/run_topic4_zm_ictal_transition_worker.py").read_text()
    assert 'float(limits["pre_ictal_offset_ms"]) + dt' in source
    assert "requested checkpoints were never reached" in source


def test_the_two_second_checkpoint_is_not_called_baseline():
    """Measured: the 2 s point already exceeds the same-seed Z/M-off q95 over
    forty non-overlapping windows, on all three canary seeds."""
    source = (ROOT / "scripts/run_topic4_zm_ictal_transition_worker.py").read_text()
    assert 'labels = {baseline_step: "early_transition"}' in source
