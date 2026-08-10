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


# ---- FCXR-LC4: the channel open fraction is state, and must fork with everything else ----

class _Cfg:
    def __init__(self, K):
        self.m_hill_K = K


def _state_with_a(seed=0, fill=0.5, a_fill=0.3, K=2.0):
    s = _state(seed, fill)
    s.slow.a = np.full(NE, a_fill, dtype=float)
    s.slow.cfg = _Cfg(K)
    return s


def test_the_open_fraction_survives_a_fork(tmp_path):
    """A reload that leaves the channel at the template's value is not a fork of this trajectory:
    the whole point of the slow-off kinetics is that the state carries protection forward."""
    s = _state_with_a(seed=3, a_fill=0.77)
    p = str(tmp_path / "s.npz")
    written = save_loop_state(p, s)
    back = load_into(p, _state_with_a(seed=99, fill=0.9, a_fill=0.01))
    assert np.allclose(np.asarray(back.slow.a), 0.77)
    assert state_hash(back) == written


def test_the_hash_covers_the_open_fraction_once_it_carries_something():
    a, b = _state_with_a(seed=4, a_fill=0.2), _state_with_a(seed=4, a_fill=0.9)
    assert state_hash(a) != state_hash(b)


def test_an_all_zero_open_fraction_hashes_as_if_it_were_not_there():
    """Backward compatibility with every reference state written before the variable existed."""
    assert state_hash(_state_with_a(seed=4, a_fill=0.0)) == state_hash(_state(seed=4))


def test_a_pre_existing_state_loads_onto_a_template_that_has_the_mechanism_configured(tmp_path):
    """The fork this mechanism exists for: the saved discharge state predates the channel, and it
    must load onto a template that has the channel switched on.  Keying the hash on the config
    rather than on the array would reject it -- the child would hash differently from its own
    file purely because the template's config changed."""
    p = str(tmp_path / "discharge.npz")
    written = save_loop_state(p, _state(seed=8))              # written before the channel existed
    back = load_into(p, _state_with_a(seed=99, fill=0.9, a_fill=0.0, K=2.0))
    assert state_hash(back) == written


def test_a_file_written_before_the_channel_existed_still_loads(tmp_path):
    """Backward compatibility is not optional here: the saved interictal and ictal reference states
    predate this variable, and re-running them costs an hour of wall time each."""
    p = str(tmp_path / "old.npz")
    written = save_loop_state(p, _state(seed=6))          # no `a` on the slow object at all
    back = load_into(p, _state_with_a(seed=99, fill=0.9, a_fill=0.5, K=None))
    assert state_hash(back) == written
    assert np.allclose(np.asarray(back.slow.a), 0.0), (
        "a state that predates the channel had it shut; keeping the template's 0.5 would be the "
        "half-restored fork this module exists to prevent")


# ---- FCXR-LC5: formal per-cell episode load must be part of an exact fork ----

def _state_with_u(seed=0, fill=0.5, u_fill=0.3):
    s = _state(seed, fill)
    s.slow.u_pump_E = np.full(NE, u_fill, dtype=float)
    return s


def test_the_formal_episode_load_survives_a_fork(tmp_path):
    s = _state_with_u(seed=12, u_fill=0.77)
    p = str(tmp_path / "u_state.npz")
    written = save_loop_state(p, s)
    back = load_into(p, _state_with_u(seed=99, fill=0.9, u_fill=0.01))
    assert np.allclose(np.asarray(back.slow.u_pump_E), 0.77)
    assert state_hash(back) == written


def test_the_hash_covers_nonzero_episode_load():
    a = _state_with_u(seed=13, u_fill=0.2)
    b = _state_with_u(seed=13, u_fill=0.9)
    assert state_hash(a) != state_hash(b)


def test_loading_a_pre_u_state_sets_the_new_load_to_zero(tmp_path):
    p = str(tmp_path / "pre_u.npz")
    written = save_loop_state(p, _state(seed=14))
    back = load_into(p, _state_with_u(seed=99, fill=0.9, u_fill=0.5))
    assert np.allclose(np.asarray(back.slow.u_pump_E), 0.0)
    assert state_hash(back) == written
