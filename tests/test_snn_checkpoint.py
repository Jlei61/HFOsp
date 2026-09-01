"""Checkpoint capture must be complete and round-trip exactly."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "snn_engine"))
sys.path.insert(0, str(ROOT))

from src.snn_engine.checkpoint import (  # noqa: E402
    CHECKPOINT_SCHEMA, REQUIRED_KEYS, capture, digest, load,
    restore_external_drive, restore_slow, save)
from src.snn_engine.mz_slow_vars import MZSlowVarsConfig  # noqa: E402
from src.topic4_zm_slow_vars import ZMTracedSlowVars as MZSlowVars  # noqa: E402
from src.topic4_spatial_ou_drive import SpatialOUConfig, SpatialOUDrive  # noqa: E402


def _state(n=12, ne=8, m=5):
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
    state = capture(
        step=137, absolute_time_ms=13.7,
        V=rng.random(n), ref=rng.integers(0, 5, n).astype(np.int32),
        s_E=rng.random(n), I_E=rng.random(n), s_I=rng.random(n), I_I=rng.random(n),
        ring_sE=rng.random((m, n)), ring_sI=rng.random((m, n)),
        xi=0.31, rng=rng, ras_keep=np.array([0, 3, 5]),
        es_ema=12.5, es_run=3, track_rec=False, s_E_rec=None, I_E_rec=None,
        slow=slow, external_drive=drive)
    return state, slow, drive


def test_capture_has_every_required_key():
    state, _, _ = _state()
    assert state["schema"] == CHECKPOINT_SCHEMA
    assert set(REQUIRED_KEYS) <= set(state)
    for key in ("z", "m", "I_I_last", "step_index", "acc_n", "acc_seen",
                "acc_D", "acc_A"):
        assert key in state["slow"], key
    for key in ("field_state", "cached", "next_step", "last_step", "rng_state"):
        assert key in state["external_drive"], key


def test_capture_copies_and_does_not_alias():
    state, slow, _ = _state()
    before = state["slow"]["z"].copy()
    slow.z[:] = 0.0
    assert np.array_equal(state["slow"]["z"], before)


def test_round_trip_is_exact(tmp_path):
    state, _, _ = _state()
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
    state, slow, drive = _state()
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
    state, slow, _ = _state()
    import pytest
    with pytest.raises(ValueError, match="disagree"):
        restore_slow(state, None)


def test_digest_changes_when_any_field_changes():
    state, _, _ = _state()
    base = digest(state)
    for key in ("V", "ring_sE", "xi", "es_ema", "step"):
        mutated = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                   for k, v in state.items()}
        if isinstance(mutated[key], np.ndarray):
            mutated[key] = mutated[key] + 1.0
        else:
            mutated[key] = mutated[key] + 1
        assert digest(mutated) != base, key
