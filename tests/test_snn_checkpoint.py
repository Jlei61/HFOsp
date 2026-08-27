"""Checkpoint capture must be complete and round-trip exactly."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "snn_engine"))
sys.path.insert(0, str(ROOT))

from src.snn_engine.checkpoint import (  # noqa: E402
    CHECKPOINT_SCHEMA, REQUIRED_KEYS, capture, digest, load,
    restore_external_drive, restore_node_accessibility, restore_slow, save)
from src.snn_engine.mz_slow_vars import MZSlowVarsConfig  # noqa: E402
from src.topic4_zm_slow_vars import ZMTracedSlowVars as MZSlowVars  # noqa: E402
from src.topic4_spatial_ou_drive import SpatialOUConfig, SpatialOUDrive  # noqa: E402
from src.topic4_node_accessibility import (  # noqa: E402
    FieldGatedZeroSumNodeRecovery,
)


class _NodeAccessibility:
    def __init__(self, values, step_index=0):
        self.values = np.asarray(values, dtype=float).copy()
        self.step_index = int(step_index)

    def checkpoint_state(self):
        return {"values": self.values, "step_index": self.step_index}

    def restore_checkpoint_state(self, payload):
        self.values = np.array(payload["values"], copy=True)
        self.step_index = int(payload["step_index"])


class _OtherNodeAccessibility(_NodeAccessibility):
    pass


class _ProtocolNodeAccessibility(_NodeAccessibility):
    KIND = "test.node_accessibility"


class _ProtocolNodeAccessibilityWrapper(_NodeAccessibility):
    KIND = "test.node_accessibility"


def _state(n=12, ne=8, m=5, *, with_node_accessibility=False,
           node_accessibility_object=None):
    rng = np.random.default_rng(7)
    slow = MZSlowVars(n, 18.0, MZSlowVarsConfig(use_z=True, use_m=True,
                                                I_th_EI=1.0, eta_m=0.01), NE=ne)
    slow.z[:] = rng.random(n)
    slow.m[:] = rng.random(n)
    slow._step_index = 41
    drive = SpatialOUDrive(rng.random((ne, 2)) * 4.0, 4.0, 0.1,
                           SpatialOUConfig(mode="local", sigma_rate_per_ms=0.1,
                                           tau_ms=20.0, ell_mm=0.4, seed=3))
    drive.step(5.0)
    node_accessibility = node_accessibility_object
    if node_accessibility is None and with_node_accessibility:
        node_accessibility = _NodeAccessibility(rng.random(ne), step_index=29)
    state = capture(
        step=137, absolute_time_ms=13.7,
        V=rng.random(n), ref=rng.integers(0, 5, n).astype(np.int32),
        s_E=rng.random(n), I_E=rng.random(n), s_I=rng.random(n), I_I=rng.random(n),
        ring_sE=rng.random((m, n)), ring_sI=rng.random((m, n)),
        xi=0.31, rng=rng, ras_keep=np.array([0, 3, 5]),
        es_ema=12.5, es_run=3, track_rec=False, s_E_rec=None, I_E_rec=None,
        slow=slow, external_drive=drive,
        node_accessibility=node_accessibility)
    return state, slow, drive, node_accessibility


def test_capture_has_every_required_key():
    state, _, _, _ = _state()
    assert state["schema"] == CHECKPOINT_SCHEMA
    assert set(REQUIRED_KEYS) <= set(state)
    for key in ("z", "m", "I_I_last", "step_index", "acc_n", "acc_seen",
                "acc_D", "acc_A"):
        assert key in state["slow"], key
    for key in ("field_state", "cached", "next_step", "last_step", "rng_state"):
        assert key in state["external_drive"], key
    assert state["node_accessibility"] is None


def test_capture_copies_and_does_not_alias():
    state, slow, _, _ = _state()
    before = state["slow"]["z"].copy()
    slow.z[:] = 0.0
    assert np.array_equal(state["slow"]["z"], before)


def test_round_trip_is_exact(tmp_path):
    state, _, _, _ = _state()
    path = tmp_path / "ckpt.npz"
    written = save(state, path)
    assert len(written) == 64
    back = load(path)
    assert digest(back) == digest(state)
    assert np.array_equal(back["ring_sE"], state["ring_sE"])
    assert back["rng_state"] == state["rng_state"]
    assert back["step"] == 137
    assert back["absolute_time_ms"] == 13.7


def test_restore_puts_slow_and_drive_back(tmp_path):
    state, slow, drive, _ = _state()
    z_before, m_before = slow.z.copy(), slow.m.copy()
    field_before = drive._state.copy()
    slow.z[:] = 0.0
    slow.m[:] = 0.0
    drive._state[:] = 0.0
    restore_slow(state, slow)
    restore_external_drive(state, drive)
    assert np.array_equal(slow.z, z_before)
    assert np.array_equal(slow.m, m_before)
    assert np.array_equal(drive._state, field_before)
    assert drive._rng.bit_generator.state == state["external_drive"]["rng_state"]


def test_restore_rejects_a_mismatched_pairing():
    state, slow, _, _ = _state()
    import pytest
    with pytest.raises(ValueError, match="disagree"):
        restore_slow(state, None)


def test_digest_changes_when_any_field_changes():
    state, _, _, _ = _state()
    base = digest(state)
    for key in ("V", "ring_sE", "xi", "es_ema", "step"):
        mutated = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                   for k, v in state.items()}
        if isinstance(mutated[key], np.ndarray):
            mutated[key] = mutated[key] + 1.0
        else:
            mutated[key] = mutated[key] + 1
        assert digest(mutated) != base, key


def test_node_accessibility_round_trip_and_restore_are_exact(tmp_path):
    state, slow, drive, controller = _state(with_node_accessibility=True)
    saved_values = controller.values.copy()
    controller.values[:] = -1.0
    assert np.array_equal(state["node_accessibility"]["values"], saved_values)

    path = tmp_path / "node_accessibility_ckpt.npz"
    save(state, path)
    back = load(path)
    assert digest(back) == digest(state)

    restored = _NodeAccessibility(np.zeros_like(saved_values), step_index=0)
    restore_node_accessibility(back, restored)
    assert np.array_equal(restored.values, saved_values)
    assert restored.step_index == 29

    restored.values[:] = 3.0
    assert np.array_equal(back["node_accessibility"]["values"], saved_values)
    restore_slow(back, slow)
    restore_external_drive(back, drive)


def test_real_node_accessibility_round_trip_through_npz_is_exact(tmp_path):
    support = np.linspace(0.1, 1.0, 8)
    controller = FieldGatedZeroSumNodeRecovery(
        support, dt_ms=0.1, tau_ms=250.0, a_ref_mV=0.2,
        r_ref_hz=50.0, mode="zero_sum", trace_dt_ms=0.1,
    )
    controller.step(np.array([True, False, True, False] * 2), 0.1)
    controller.threshold(np.full(12, 18.0))
    state, _, _, _ = _state(node_accessibility_object=controller)

    path = tmp_path / "real_node_accessibility_ckpt.npz"
    save(state, path)
    back = load(path)
    restored = FieldGatedZeroSumNodeRecovery(
        support, dt_ms=0.1, tau_ms=250.0, a_ref_mV=0.2,
        r_ref_hz=50.0, mode="zero_sum", trace_dt_ms=0.1,
    )
    restore_node_accessibility(back, restored)

    assert restored.diagnostics() == controller.diagnostics()
    assert np.array_equal(restored.state_mV, controller.state_mV)
    for key, expected in controller.trace_arrays().items():
        assert np.array_equal(restored.trace_arrays()[key], expected)


def test_node_accessibility_restore_requires_matching_enabled_state():
    enabled, _, _, controller = _state(with_node_accessibility=True)
    disabled, _, _, _ = _state(with_node_accessibility=False)
    import pytest

    with pytest.raises(ValueError, match="disagree"):
        restore_node_accessibility(enabled, None)
    with pytest.raises(ValueError, match="disagree"):
        restore_node_accessibility(
            disabled, _NodeAccessibility(np.zeros_like(controller.values)))


def test_node_accessibility_restore_rejects_controller_type_mismatch():
    state, _, _, controller = _state(with_node_accessibility=True)
    import pytest

    with pytest.raises(ValueError, match="controller differs"):
        restore_node_accessibility(
            state, _OtherNodeAccessibility(np.zeros_like(controller.values)))


def test_node_accessibility_kind_is_stable_across_audit_wrapper():
    source = _ProtocolNodeAccessibility(np.arange(8.0), step_index=7)
    state, _, _, _ = _state(node_accessibility_object=source)
    target = _ProtocolNodeAccessibilityWrapper(np.zeros(8), step_index=0)
    restore_node_accessibility(state, target)
    assert np.array_equal(target.values, source.values)
    assert target.step_index == source.step_index


def test_load_accepts_legacy_controller_off_checkpoint(tmp_path):
    state, _, _, _ = _state()
    current_path = tmp_path / "current.npz"
    legacy_path = tmp_path / "legacy_v1.npz"
    save(state, current_path)

    with np.load(current_path, allow_pickle=False) as handle:
        meta = json.loads(str(handle["__meta__"]))
        meta.pop("node_accessibility__present")
        arrays = {key: handle[key] for key in handle.files if key != "__meta__"}
    np.savez(legacy_path, __meta__=np.array(json.dumps(meta, sort_keys=True)),
             **arrays)

    back = load(legacy_path)
    assert back["schema"] == CHECKPOINT_SCHEMA
    assert back["node_accessibility"] is None
    assert set(REQUIRED_KEYS) <= set(back)
    restore_node_accessibility(back, None)
