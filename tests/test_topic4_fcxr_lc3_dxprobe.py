"""Contracts for freezing a live FCXR-LC3 trajectory state at chosen D and X."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

from model import build_network
from mz_slow_vars import MZSlowVars, MZSlowVarsConfig
from params import Params

from src.topic4_fcxr_lc3 import run_fcxr_loop
from src.topic4_fcxr_lc3_dxprobe import freeze_dynamic_state, probe_summary


def _dynamic_state(seed=17):
    """A state whose D and X are still evolving -- the shape a trajectory hands over."""
    p = Params(L=4.0, density=80.0, T=120.0, dt=0.1, nu_ext_ratio=0.92, seed=seed)
    net = build_network(p, verbose=False)
    ne, n = net["NE"], net["NE"] + net["NI"]
    vth = np.full(n, p.V_th)
    vth[: min(8, ne)] -= 0.8
    cfg = MZSlowVarsConfig(
        membrane_mode="full_conductance", E_E=58.0, c_E=1.0,
        ff_conductance=False, rec_conductance=True, rec_sat_g=21.6,
        v_match=18.0, e_gaba=0.0, e_k=0.0,
        max_total_conductance=99.0, fail_on_clip=True,
        use_h_lc2=True, tau_h_lc2=80.0, theta_h_lc2=0.03, k_h_lc2=0.02, rho_h_lc2=0.2,
        use_x=True, tau_y=120.0, tau_x=800.0, x_min=0.1, y_gate=2.0, K_y=4.0, hill_n=4,
        use_z=True, I_th_EI=1.0, tau_z=3000.0,
    )
    slow = MZSlowVars(n, 18.0, cfg, NE=ne, core_mask_E=np.zeros(ne, bool))
    net["rng"] = np.random.default_rng(seed)
    out = run_fcxr_loop(p, net, slow=slow, n_steps=40, capture_final=True,
                        store_spikes=False, v_th_per_neuron=vth)
    return p, net, vth, out["checkpoint"]


def test_a_dynamic_state_starts_with_neither_field_frozen():
    _p, _net, _vth, state = _dynamic_state()
    assert state.slow.cfg.use_z is True
    assert state.slow.cfg.z_frozen_E is None
    assert state.slow.cfg.x_relay_frozen_E is None


def test_freezing_installs_both_fields_and_satisfies_the_engine_rule():
    # The engine rejects a frozen z field that is still allowed to evolve, and the
    # frozen-X branch lives inside the use_x block, so use_x must stay on.
    _p, _net, _vth, state = _dynamic_state()
    child = freeze_dynamic_state(state, d_field=0.663, x_field=0.10)
    cfg = child.slow.cfg
    assert cfg.use_z is False and cfg.z_frozen_E is not None
    assert cfg.use_x is True and cfg.x_relay_frozen_E is not None
    ne = child.slow.NE
    assert np.allclose(child.slow.z[:ne], 1.0 - 0.663)
    assert np.allclose(child.slow.x_relay, 0.10)


def test_freezing_syncs_the_outgoing_relay_so_the_first_spike_cannot_use_a_stale_value():
    _p, _net, _vth, state = _dynamic_state()
    child = freeze_dynamic_state(state, x_field=0.25)
    assert np.allclose(child.slow.ee_relay_send, 0.25)


def test_freezing_does_not_touch_the_source_state():
    _p, _net, _vth, state = _dynamic_state()
    before_z = np.asarray(state.slow.z[:state.slow.NE]).copy()
    before_x = np.asarray(state.slow.x_relay).copy()
    freeze_dynamic_state(state, d_field=0.9, x_field=0.1)
    assert state.slow.cfg.use_z is True and state.slow.cfg.z_frozen_E is None
    np.testing.assert_array_equal(state.slow.z[:state.slow.NE], before_z)
    np.testing.assert_array_equal(state.slow.x_relay, before_x)


def test_none_freezes_a_variable_where_the_trajectory_left_it():
    # The control arm freezes the real state in place; it must not silently reset.
    _p, _net, _vth, state = _dynamic_state()
    ne = state.slow.NE
    z_now = np.asarray(state.slow.z[:ne]).copy()
    x_now = np.asarray(state.slow.x_relay).copy()
    child = freeze_dynamic_state(state)
    np.testing.assert_array_equal(child.slow.z[:ne], z_now)
    np.testing.assert_array_equal(child.slow.x_relay, x_now)
    np.testing.assert_array_equal(child.slow.cfg.z_frozen_E, z_now)


def test_a_frozen_state_really_stops_evolving_its_two_slow_fields():
    p, net, vth, state = _dynamic_state()
    child = freeze_dynamic_state(state, d_field=0.663, x_field=0.10)
    ne = child.slow.NE
    out = run_fcxr_loop(p, net, start=child, n_steps=200, capture_final=True,
                        store_spikes=False, v_th_per_neuron=vth)
    after = out["checkpoint"].slow
    np.testing.assert_allclose(after.z[:ne], 1.0 - 0.663)
    np.testing.assert_allclose(after.x_relay, 0.10)


def test_a_dynamic_state_does_evolve_them_so_the_frozen_test_is_not_vacuous():
    p, net, vth, state = _dynamic_state()
    from src.topic4_fcxr_lc3 import clone_loop_state
    out = run_fcxr_loop(p, net, start=clone_loop_state(state), n_steps=200,
                        capture_final=True, store_spikes=False, v_th_per_neuron=vth)
    after = out["checkpoint"].slow
    ne = state.slow.NE
    moved_z = not np.allclose(after.z[:ne], state.slow.z[:ne])
    moved_x = not np.allclose(after.x_relay, state.slow.x_relay)
    assert moved_z or moved_x, "neither slow field moved; the freeze test proves nothing"


@pytest.mark.parametrize("bad", [1.4, -0.2, float("nan")])
def test_out_of_range_fields_are_rejected_rather_than_clipped(bad):
    _p, _net, _vth, state = _dynamic_state()
    with pytest.raises(ValueError, match="finite and within"):
        freeze_dynamic_state(state, x_field=bad)


def test_wrong_shape_field_is_rejected():
    _p, _net, _vth, state = _dynamic_state()
    with pytest.raises(ValueError, match="shape"):
        freeze_dynamic_state(state, d_field=np.zeros(3))


def test_probe_summary_carries_the_fields_the_arm_was_actually_frozen_at():
    cls = dict(label="FINITE_HIGH_FIXED", workpoint_label="FINITE_HIGH_FIXED",
               refractory_ceiling_fraction=0.0, h_mean=1.8, h_slope_per_s=0.01,
               numerical_unsafe=False)
    got = probe_summary(arm_id="max_brake", d_field=np.full(4, 0.663),
                        x_field=np.full(4, 0.10), classification=cls,
                        total_ms=1500.0, extended=False)
    assert got["arm_id"] == "max_brake"
    assert got["D_mean"] == pytest.approx(0.663)
    assert got["X_mean"] == pytest.approx(0.10)
    assert got["resolved_label"] == "FINITE_HIGH_FIXED"
    assert got["extended"] is False


def test_a_compacted_seed_state_regrows_traces_the_classifier_can_slice():
    """The probe seeds from a saved landmark checkpoint, and saving compacts it.

    compact_checkpoint_diagnostics empties every trace_* list, and the tail classifier
    slices those lists by the run's own step count.  Empty lists must therefore regrow
    to exactly the number of steps run -- if compaction left None instead, the slice
    would raise, and if it left the pre-compaction history the tail would be misaligned.
    """
    from src.topic4_fcxr_lc3_geometry import compact_checkpoint_diagnostics

    p, net, vth, state = _dynamic_state()
    compacted = compact_checkpoint_diagnostics(state)
    assert compacted.slow.trace_h_lc2_mean == []

    child = freeze_dynamic_state(compacted, d_field=0.663, x_field=0.10)
    n_steps = 120
    out = run_fcxr_loop(p, net, start=child, n_steps=n_steps, capture_final=True,
                        store_spikes=True, v_th_per_neuron=vth)
    slow = out["checkpoint"].slow
    for name in ("trace_h_lc2_mean", "trace_tau_eff_ratio_min", "trace_conductance_clip_frac"):
        trace = np.asarray(getattr(slow, name), dtype=float)
        assert trace.size == n_steps, f"{name} has {trace.size} of {n_steps} steps"
        assert np.all(np.isfinite(trace))
