from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from scripts.run_topic4_zm_phase_point import (
    absolutize_round_inputs,
    future_noise_digest,
    prepare_matched_state,
    validate_classifier_config,
)


class _Drive:
    def __init__(self, seed=12):
        self.update_steps = 10
        self._state = np.arange(9, dtype=float).reshape(3, 3)
        self._cached = np.linspace(-0.2, 0.2, 5)
        self._rng = np.random.default_rng(seed)


def _checkpoint(step, m_value):
    return {
        "step": int(step),
        "absolute_time_ms": step * 0.1,
        "rng_state": np.random.default_rng(step).bit_generator.state,
        "xi": 7.0,
        "es_ema": 40.0,
        "es_run": 20,
        "slow": {
            "kind": "SpatialZMQIGKSlowVars",
            "q_I": np.full((3, 3), 0.5),
            "qdriver_rE": np.ones((3, 3)),
            "qdriver_rI": np.ones((3, 3)) * 2,
            "field_count_E": np.ones((3, 3)) * 3,
            "field_count_I": np.ones((3, 3)) * 4,
            "last_m_drive_E": np.ones(5),
            "last_q_drive": np.ones((3, 3)) * 5,
            "field_steps_seen": 7,
            "field_steps_per_update": 10,
            "z": np.ones(8),
            "m": np.full(8, float(m_value)),
        },
        "external_drive": None,
    }


def test_low_and_high_starts_receive_identical_relative_future_noise():
    low = prepare_matched_state(
        _checkpoint(2000, 0.1), q_clamp=0.805, noise_seed=9101,
        fresh_drive=_Drive())
    high = prepare_matched_state(
        _checkpoint(6000, 4.0), q_clamp=0.805, noise_seed=9101,
        fresh_drive=_Drive())
    assert future_noise_digest(low, update_steps=10) == future_noise_digest(
        high, update_steps=10)
    assert low["external_drive"]["next_step"] == 2010
    assert high["external_drive"]["next_step"] == 6010
    assert low["external_drive"]["last_step"] == 1999
    assert high["external_drive"]["last_step"] == 5999


def test_prepare_clamps_q_resets_q_history_and_preserves_m_basin():
    original = _checkpoint(2000, 0.25)
    untouched = copy.deepcopy(original)
    prepared = prepare_matched_state(
        original, q_clamp=0.79, noise_seed=99, fresh_drive=_Drive())
    np.testing.assert_array_equal(prepared["slow"]["q_I"], 0.79)
    np.testing.assert_array_equal(prepared["slow"]["z"][:5], 0.79)
    np.testing.assert_array_equal(prepared["slow"]["m"], untouched["slow"]["m"])
    for name in (
            "qdriver_rE", "qdriver_rI", "field_count_E", "field_count_I",
            "last_m_drive_E", "last_q_drive"):
        assert not np.any(prepared["slow"][name])
    assert prepared["slow"]["field_steps_seen"] == 0
    assert prepared["slow"]["field_steps_per_update"] is None
    assert prepared["xi"] == 0.0
    assert prepared["es_ema"] == 0.0
    assert prepared["es_run"] == 0
    np.testing.assert_array_equal(original["slow"]["q_I"], untouched["slow"]["q_I"])


def test_future_noise_digest_changes_with_seed():
    first = prepare_matched_state(
        _checkpoint(2000, 0.1), q_clamp=0.805, noise_seed=1,
        fresh_drive=_Drive(seed=1))
    second = prepare_matched_state(
        _checkpoint(2000, 0.1), q_clamp=0.805, noise_seed=2,
        fresh_drive=_Drive(seed=2))
    assert future_noise_digest(first, update_steps=10) != future_noise_digest(
        second, update_steps=10)


def test_round_input_resolution_prefers_artifacts_then_tracked_code(tmp_path, monkeypatch):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "ignored.json").write_text("{}")
    from scripts import run_topic4_zm_phase_point as runner
    code_root = tmp_path / "code"
    code_root.mkdir()
    (code_root / "tracked.json").write_text("{}")
    monkeypatch.setattr(runner, "ROOT", code_root)
    config = {"inputs": {
        "ignored": {"path": "ignored.json", "sha256": "x"},
        "tracked": {"path": "tracked.json", "sha256": "y"},
    }}
    got = absolutize_round_inputs(config, artifact)
    assert got["inputs"]["ignored"]["path"] == str(
        (artifact / "ignored.json").resolve())
    assert got["inputs"]["tracked"]["path"] == str(
        (code_root / "tracked.json").resolve())


def test_frozen_config_and_classifier_code_agree():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((
        root / "config/topic4_spatial_zm_phase_diagram_v1.json").read_text())
    contract = validate_classifier_config(config)
    assert contract["contract_version"] == "event_tolerant_low_v2"
