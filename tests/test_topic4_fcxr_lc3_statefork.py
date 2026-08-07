"""Tests for persisting a loop state and for scaling the frozen fields.

The failure these guard is the one that makes a state fork meaningless: a reload that restores
some of the state and silently leaves the rest at the template's values still runs, still produces
numbers, and is no longer a fork of the trajectory it claims to continue.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

from src.topic4_fcxr_lc3 import FCXRLoopState, state_hash  # noqa: E402
from src.topic4_fcxr_lc3_statefork import (  # noqa: E402
    load_into,
    save_loop_state,
    scaled_fields,
)

NE, NI, MD = 40, 10, 3
N = NE + NI


class _Slow:
    def __init__(self, ne, fill=0.5):
        self.NE = ne
        self._step_i = 0
        for k in ("z", "m", "phi", "x_relay", "y", "ee_relay_send", "h_lc2_E",
                  "_h_source_lc2_E", "_z_sensor_last_E"):
            setattr(self, k, np.full(ne, fill, dtype=float))


def _state(seed=0, fill=0.5):
    rng = np.random.default_rng(seed)
    return FCXRLoopState(
        t=7, V=rng.normal(size=N), ref=rng.integers(0, 3, size=N).astype(np.int32),
        s_E=rng.normal(size=N), I_E=rng.normal(size=N),
        s_I=rng.normal(size=N), I_I=rng.normal(size=N),
        s_E_rec=rng.normal(size=N), I_E_rec=rng.normal(size=N),
        ring_sE=rng.normal(size=(MD, NE)), ring_sI=rng.normal(size=(MD, NI)),
        xi=0.123, rng_state=np.random.default_rng(5).bit_generator.state,
        slow=_Slow(NE, fill))


def test_a_saved_state_reloads_to_the_same_hash(tmp_path):
    s = _state(seed=1)
    p = str(tmp_path / "s.npz")
    written = save_loop_state(p, s)
    back = load_into(p, _state(seed=99, fill=0.9))     # a deliberately different template
    assert state_hash(back) == written == state_hash(s)


def test_every_fast_array_is_restored_not_just_the_membrane(tmp_path):
    s = _state(seed=2)
    p = str(tmp_path / "s.npz")
    save_loop_state(p, s)
    back = load_into(p, _state(seed=42, fill=0.1))
    for k in ("V", "ref", "s_E", "I_E", "s_I", "I_I", "s_E_rec", "I_E_rec",
              "ring_sE", "ring_sI"):
        assert np.array_equal(np.asarray(getattr(back, k)), np.asarray(getattr(s, k))), k


def test_the_slow_fields_and_the_generator_come_back_too(tmp_path):
    s = _state(seed=3, fill=0.37)
    p = str(tmp_path / "s.npz")
    save_loop_state(p, s)
    back = load_into(p, _state(seed=8, fill=0.99))
    assert np.allclose(np.asarray(back.slow.z), 0.37)
    assert np.allclose(np.asarray(back.slow.x_relay), 0.37)
    assert back.rng_state == s.rng_state
    assert back.t == s.t and back.xi == pytest.approx(s.xi)


def test_a_tampered_file_is_refused_rather_than_seeding_a_grid(tmp_path):
    s = _state(seed=4)
    p = str(tmp_path / "s.npz")
    save_loop_state(p, s)
    z = dict(np.load(p, allow_pickle=False))
    z["fast__V"] = z["fast__V"] + 1e-6
    np.savez_compressed(p, **z)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_into(p, _state(seed=4))


def test_loading_does_not_write_through_to_the_template(tmp_path):
    s = _state(seed=5, fill=0.2)
    p = str(tmp_path / "s.npz")
    save_loop_state(p, s)
    template = _state(seed=6, fill=0.8)
    before = np.asarray(template.V).copy()
    load_into(p, template)
    assert np.array_equal(np.asarray(template.V), before)


def test_a_shape_mismatch_fails_loudly(tmp_path):
    s = _state(seed=7)
    p = str(tmp_path / "s.npz")
    save_loop_state(p, s)
    bad = _state(seed=7)
    bad.V = np.zeros(N + 1)
    with pytest.raises(ValueError, match="saved shape"):
        load_into(p, bad)


def test_scaling_keeps_the_field_shape_and_only_moves_its_amplitude():
    d_star = np.array([0.02, 0.06, 0.10, 0.04])
    x_star = np.array([1.0, 0.8, 0.4, 0.9])
    d, x = scaled_fields(d_star, x_star, alpha_d=2.0, alpha_x=1.0)
    assert np.allclose(d, 2.0 * d_star)
    assert np.corrcoef(d, d_star)[0, 1] == pytest.approx(1.0)
    assert np.allclose(x, x_star)                       # alpha_x = 1 leaves the relay untouched


def test_scaling_the_relay_load_moves_it_toward_depletion_not_toward_one():
    """X is the termination load: raising alpha_x must lower availability, not raise it."""
    x_star = np.array([1.0, 0.8, 0.4])
    _, x_hi = scaled_fields(np.zeros(3), x_star, alpha_d=1.0, alpha_x=2.0)
    assert np.all(x_hi <= x_star + 1e-12)
    assert x_hi[2] < x_star[2]


def test_scaling_stays_inside_the_ranges_the_engine_validates():
    d_star = np.array([0.4, 0.9])
    x_star = np.array([0.2, 0.05])
    d, x = scaled_fields(d_star, x_star, alpha_d=5.0, alpha_x=5.0)
    assert np.all((d >= 0) & (d <= 1)) and np.all((x >= 0) & (x <= 1))
