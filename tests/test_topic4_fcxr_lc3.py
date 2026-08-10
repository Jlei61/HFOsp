"""E0 contracts for the FCXR-LC3 field-preserving exact fork."""
from __future__ import annotations

import copy
import os
import sys

import numpy as np
import pytest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

from kick_probe import simulate_kick  # noqa: E402
from model import build_network  # noqa: E402
from mz_slow_vars import MZSlowVars, MZSlowVarsConfig  # noqa: E402
from params import Params  # noqa: E402
from src.topic4_fcxr_lc3 import (  # noqa: E402
    clone_loop_state,
    replace_frozen_fields,
    run_fcxr_loop,
    state_hash,
    validate_loop_state,
)
from src.topic4_fcxr_lc3_perturb import (  # noqa: E402
    current_accounting,
    run_fcxr_perturbation,
)


def _case(seed=17, *, frozen=False):
    p = Params(L=4.0, density=80.0, T=120.0, dt=0.1, nu_ext_ratio=0.92, seed=seed)
    net = build_network(p, verbose=False)
    ne, n = net["NE"], net["NE"] + net["NI"]
    vth = np.full(n, p.V_th)
    vth[: min(8, ne)] -= 0.8
    zf = np.linspace(0.75, 1.0, ne) if frozen else None
    xf = np.linspace(0.7, 1.0, ne) if frozen else None
    cfg = MZSlowVarsConfig(
        membrane_mode="full_conductance", E_E=58.0, c_E=1.0,
        ff_conductance=False, rec_conductance=True, rec_sat_g=21.6,
        v_match=18.0, e_gaba=0.0, e_k=0.0,
        max_total_conductance=99.0, fail_on_clip=True,
        use_h_lc2=True, tau_h_lc2=80.0, theta_h_lc2=0.03,
        k_h_lc2=0.02, rho_h_lc2=0.2,
        use_x=True, x_relay_frozen_E=xf, tau_y=120.0, tau_x=800.0,
        x_min=0.1, y_gate=2.0, K_y=4.0, hill_n=4,
        use_z=not frozen, z_frozen_E=zf, I_th_EI=1.0, tau_z=3000.0,
    )
    slow = MZSlowVars(n, 18.0, cfg, NE=ne, core_mask_E=np.zeros(ne, bool))
    net["rng"] = np.random.default_rng(seed)
    return p, net, slow, vth


def test_registered_loop_matches_guarded_engine_byte_for_byte():
    p, net_a, slow_a, vth = _case()
    ref = simulate_kick(
        p, net_a, 0.0, slow=slow_a, t_kick=1e9,
        V_th_per_neuron=vth, early_stop_runaway=False,
    )

    p, net_b, slow_b, vth = _case()
    got = run_fcxr_loop(
        p, net_b, slow=slow_b, n_steps=int(round(p.T / p.dt)),
        capture_final=True, store_spikes=True, v_th_per_neuron=vth,
    )
    np.testing.assert_array_equal(got["rate_E"], ref["rate_E"])
    np.testing.assert_array_equal(got["rate_I"], ref["rate_I"])
    np.testing.assert_array_equal(got["E_spk_bool"], ref["E_spk_bool"])
    np.testing.assert_array_equal(got["checkpoint"].slow.z, slow_a.z)
    np.testing.assert_array_equal(got["checkpoint"].slow.x_relay, slow_a.x_relay)
    np.testing.assert_array_equal(got["checkpoint"].slow.h_lc2_E, slow_a.h_lc2_E)
    assert got["checkpoint"].rng_state == net_a["rng"].bit_generator.state


def test_split_continuation_matches_uninterrupted_and_is_idempotent():
    p, net_ref, slow_ref, vth = _case()
    total = int(round(p.T / p.dt))
    ref = run_fcxr_loop(
        p, net_ref, slow=slow_ref, n_steps=total,
        capture_final=True, store_spikes=True, v_th_per_neuron=vth,
    )

    p, net_pre, slow_pre, vth = _case()
    k = 437
    pre = run_fcxr_loop(
        p, net_pre, slow=slow_pre, n_steps=k,
        capture_final=True, store_spikes=True, v_th_per_neuron=vth,
    )
    checkpoint = pre["checkpoint"]

    tails = []
    for _ in range(2):
        child = clone_loop_state(checkpoint)
        tail = run_fcxr_loop(
            p, net_pre, start=child, n_steps=total - k,
            capture_final=True, store_spikes=True, v_th_per_neuron=vth,
        )
        tails.append(tail)
        np.testing.assert_array_equal(
            np.concatenate([pre["rate_E"], tail["rate_E"]]), ref["rate_E"])
        np.testing.assert_array_equal(
            np.concatenate([pre["rate_I"], tail["rate_I"]]), ref["rate_I"])
        np.testing.assert_array_equal(
            np.concatenate([pre["E_spk_bool"], tail["E_spk_bool"]], axis=0),
            ref["E_spk_bool"],
        )
        assert state_hash(tail["checkpoint"]) == state_hash(ref["checkpoint"])

    np.testing.assert_array_equal(tails[0]["rate_E"], tails[1]["rate_E"])
    np.testing.assert_array_equal(tails[0]["E_spk_bool"], tails[1]["E_spk_bool"])


def test_sparse_spike_sink_is_a_pure_read_and_matches_dense_raster():
    p, net_ref, slow_ref, vth = _case()
    n_steps = 240
    ref = run_fcxr_loop(
        p, net_ref, slow=slow_ref, n_steps=n_steps,
        capture_final=True, store_spikes=True, v_th_per_neuron=vth,
    )

    p, net_got, slow_got, vth = _case()
    steps, cells = [], []

    def sink(step, spiking_cells):
        steps.extend([int(step)] * len(spiking_cells))
        cells.extend(np.asarray(spiking_cells, dtype=int).tolist())

    got = run_fcxr_loop(
        p, net_got, slow=slow_got, n_steps=n_steps,
        capture_final=True, store_spikes=False, spike_sink=sink,
        v_th_per_neuron=vth,
    )
    expected_steps, expected_cells = np.nonzero(ref["E_spk_bool"])
    np.testing.assert_array_equal(np.asarray(steps), expected_steps)
    np.testing.assert_array_equal(np.asarray(cells), expected_cells)
    np.testing.assert_array_equal(got["rate_E"], ref["rate_E"])
    np.testing.assert_array_equal(got["rate_I"], ref["rate_I"])
    assert got["E_spk_bool"] is None
    assert state_hash(got["checkpoint"]) == state_hash(ref["checkpoint"])


def test_input_sink_is_a_pure_read_and_sees_every_exact_draw():
    p, net_ref, slow_ref, vth = _case()
    ref = run_fcxr_loop(
        p, net_ref, slow=slow_ref, n_steps=13, capture_final=True,
        store_spikes=False, v_th_per_neuron=vth,
    )
    p, net_got, slow_got, vth = _case()
    rows = []
    got = run_fcxr_loop(
        p, net_got, slow=slow_got, n_steps=13, capture_final=True,
        store_spikes=False,
        input_sink=lambda step, xi, ext: rows.append((step, xi, ext.copy())),
        v_th_per_neuron=vth,
    )
    assert [r[0] for r in rows] == list(range(13))
    assert all(r[2].shape == (len(vth),) for r in rows)
    np.testing.assert_array_equal(got["rate_E"], ref["rate_E"])
    assert state_hash(got["checkpoint"]) == state_hash(ref["checkpoint"])


def test_zero_current_perturbation_is_exact_continuation_byte_for_byte():
    p, net, slow, vth = _case()
    pre = run_fcxr_loop(
        p, net, slow=slow, n_steps=80, capture_final=True,
        store_spikes=True, v_th_per_neuron=vth,
    )["checkpoint"]
    n_steps = 120
    ref = run_fcxr_loop(
        p, net, start=clone_loop_state(pre), n_steps=n_steps,
        capture_final=True, store_spikes=True, v_th_per_neuron=vth,
    )
    sham = run_fcxr_perturbation(
        p, net, start=clone_loop_state(pre), n_steps=n_steps,
        current_pattern=np.ones(net["NE"]), amplitude=0.0, pulse_steps=20,
        capture_final=True, store_spikes=True, v_th_per_neuron=vth,
    )
    np.testing.assert_array_equal(sham["rate_E"], ref["rate_E"])
    np.testing.assert_array_equal(sham["rate_I"], ref["rate_I"])
    np.testing.assert_array_equal(sham["E_spk_bool"], ref["E_spk_bool"])
    assert state_hash(sham["checkpoint"]) == state_hash(ref["checkpoint"])


def test_current_accounting_separates_charge_and_rms():
    got = current_accounting(
        np.array([1.0, 1.0, 0.0, -1.0]), amplitude=2.0, duration_ms=10.0)
    assert got["active_cell_count"] == 3
    assert got["positive_charge"] == 40.0
    assert got["negative_charge_magnitude"] == 20.0
    assert got["rms_current"] == pytest.approx(np.sqrt(3.0))


def test_child_forks_do_not_alias_mutable_state():
    p, net, slow, vth = _case(frozen=True)
    out = run_fcxr_loop(
        p, net, slow=slow, n_steps=100, capture_final=True,
        store_spikes=False, v_th_per_neuron=vth,
    )
    parent = out["checkpoint"]
    a = clone_loop_state(parent)
    b = clone_loop_state(parent)
    a.V[0] += 1.0
    a.ring_sE[0, 0] += 1.0
    a.slow.z[0] -= 0.1
    a.slow.x_relay[0] -= 0.1
    assert not np.shares_memory(a.V, b.V)
    assert not np.shares_memory(a.ring_sE, b.ring_sE)
    assert not np.shares_memory(a.slow.z, b.slow.z)
    assert not np.shares_memory(a.slow.x_relay, b.slow.x_relay)
    assert b.V[0] == parent.V[0]
    assert b.ring_sE[0, 0] == parent.ring_sE[0, 0]
    assert b.slow.z[0] == parent.slow.z[0]
    assert b.slow.x_relay[0] == parent.slow.x_relay[0]


def test_d_and_x_replacements_are_validated_and_leave_parent_unchanged():
    p, net, slow, vth = _case(frozen=True)
    state = run_fcxr_loop(
        p, net, slow=slow, n_steps=50, capture_final=True,
        store_spikes=False, v_th_per_neuron=vth,
    )["checkpoint"]
    before = state_hash(state)
    ne = state.slow.NE
    d = np.linspace(0.0, 0.4, ne)
    x = np.linspace(0.2, 0.9, ne)
    child = replace_frozen_fields(state, d_field=d, x_field=x)
    np.testing.assert_array_equal(child.slow.z[:ne], 1.0 - d)
    np.testing.assert_array_equal(child.slow.x_relay, x)
    np.testing.assert_array_equal(child.slow.ee_relay_send, x)
    assert state_hash(state) == before
    assert state_hash(child) != before
    with pytest.raises(ValueError, match="d_field"):
        replace_frozen_fields(state, d_field=np.ones(ne + 1))
    with pytest.raises(ValueError, match="x_field"):
        replace_frozen_fields(state, x_field=np.full(ne, 1.1))


def test_incomplete_or_corrupted_checkpoint_fails_closed():
    p, net, slow, vth = _case(frozen=True)
    state = run_fcxr_loop(
        p, net, slow=slow, n_steps=20, capture_final=True,
        store_spikes=False, v_th_per_neuron=vth,
    )["checkpoint"]
    bad = copy.deepcopy(state)
    bad.V[0] = np.nan
    with pytest.raises(ValueError, match="V"):
        validate_loop_state(
            bad, n=net["NE"] + net["NI"], ne=net["NE"],
            max_delay_steps=net["max_delay_steps"],
        )
    bad = copy.deepcopy(state)
    bad.slow._step_i -= 1
    with pytest.raises(ValueError, match="step counter"):
        validate_loop_state(
            bad, n=net["NE"] + net["NI"], ne=net["NE"],
            max_delay_steps=net["max_delay_steps"],
        )


def test_every_stepping_runner_installs_the_noise_generator_on_its_substrate():
    """build_substrate() omits net["rng"]; a substrate that steps must be given one.

    Three separate LC3 stages shipped without it (geometry map worker, spatial
    cmd_all, x_lifecycle cmd_calibrate) and each would only fail once its stage was
    finally reached.  Any function that builds a substrate must either install the
    generator or be listed here as provably non-stepping.
    """

    import ast
    import glob

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Functions that build a substrate purely for geometry/pattern bookkeeping.
    non_stepping = {
        ("run_topic4_fcxr_lc3_spatial.py", "cmd_lock"),  # derives I_ref from a stored state
        ("run_topic4_fcxr_lc3_geometry.py", "cmd_field_audit"),  # region masks + field stats
        # Steps through the LC1 runner, which seeds net["rng"] itself from an explicit
        # seed argument (run_topic4_mz_fcxr_lifecycle.py::_lc_run), so the bare
        # substrate handed to it is correct.
        ("run_topic4_fcxr_lc3.py", "_replay_family"),
    }
    # Enumerate the runners instead of listing them: a hand-maintained list is exactly
    # how the slow-flow stage slipped through and failed after the 102-row map had
    # already completed.
    runners = sorted(os.path.basename(p) for p in
                     glob.glob(os.path.join(root, "scripts", "*topic4_fcxr_lc3*.py")))
    assert len(runners) >= 5, f"runner discovery found only {runners}"
    offenders = []
    for name in runners:
        path = os.path.join(root, "scripts", name)
        tree = ast.parse(open(path).read())
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            body = ast.dump(node)
            if "build_substrate" not in body:
                continue
            installs = ("install_registered_noise_rng" in body
                        or ("'rng'" in body or '"rng"' in body))
            if not installs and (name, node.name) not in non_stepping:
                offenders.append(f"{name}::{node.name}")
    assert offenders == [], f"substrate built without a noise generator: {offenders}"
