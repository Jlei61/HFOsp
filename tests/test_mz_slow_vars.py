"""Contract tests for the minimal Z/M slow protocol."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "snn_engine"))

from src.snn_engine.mz_slow_vars import MZSlowVars, MZSlowVarsConfig


def _module(**kwargs):
    core = np.zeros(8, dtype=bool)
    core[:2] = True
    return MZSlowVars(
        10, 18.0, MZSlowVarsConfig(**kwargs), NE=8, core_mask_E=core,
    )


def test_both_off_current_path_is_exact():
    slow = _module(use_z=False, use_m=False)
    excitatory = np.arange(10, dtype=float) + 1.0
    inhibitory = np.arange(10, dtype=float) * 0.5
    assert np.array_equal(
        slow.apply_currents(excitatory, inhibitory),
        excitatory - inhibitory,
    )


def test_z_and_m_change_excitatory_cells_only():
    slow = _module(use_z=True, use_m=True, eta_m=0.3)
    slow.z[:slow.NE] = 0.5
    slow.m[:slow.NE] = 2.0
    output = slow.apply_currents(np.ones(10), np.ones(10))
    assert np.allclose(output[:8], -0.1)
    assert np.array_equal(output[8:], np.zeros(2))


def test_z_bounds_m_increment_and_threshold_passthrough():
    slow = _module(
        use_z=True, use_m=True, I_th_EI=5.0, tau_z=50.0,
        tau_adp=500.0, eta_m=0.2,
    )
    spikes = np.zeros(10, dtype=bool)
    spikes[2] = True
    for _ in range(100):
        slow.apply_currents(np.zeros(10), np.full(10, 10.0))
        slow.step(spikes, None, 0.1)
    assert np.all((slow.z >= 0.0) & (slow.z <= 1.0))
    assert slow.m[2] > 1.0
    assert np.all(slow.m[8:] == 0.0)
    thresholds = np.linspace(16.0, 18.0, 10)
    assert np.array_equal(slow.threshold(thresholds), thresholds)


def test_trace_stride_changes_observation_not_state():
    dense = _module(
        use_z=True, use_m=True, I_th_EI=5.0, eta_m=0.2,
        trace_stride_steps=1,
    )
    sparse = _module(
        use_z=True, use_m=True, I_th_EI=5.0, eta_m=0.2,
        trace_stride_steps=10,
    )
    spikes = np.zeros(10, dtype=bool)
    spikes[1] = True
    for _ in range(100):
        for slow in (dense, sparse):
            slow.apply_currents(np.ones(10), np.full(10, 6.0))
            slow.step(spikes, None, 0.1)
    assert np.array_equal(dense.z, sparse.z)
    assert np.array_equal(dense.m, sparse.m)
    assert len(dense.trace_arrays()["time_ms"]) == 100
    assert len(sparse.trace_arrays()["time_ms"]) == 10


@pytest.mark.parametrize(
    "kwargs",
    ({"tau_z": 0.0}, {"tau_adp": 0.0}, {"eta_m": -1.0},
     {"trace_stride_steps": 0}),
)
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        _module(**kwargs)


def test_engine_byte_parity_both_off_equals_slow_none():
    from connectivity import build_connectivity, place_neurons
    from kick_probe import simulate_kick
    from params import Params

    seed = 1
    params = Params(
        L=1.0, density=400.0, T=200.0, dt=0.1,
        seed=seed, nu_ext_ratio=1.0,
    )
    rng = np.random.default_rng(seed)
    positions, labels, n_e, n_i = place_neurons(params, rng)
    network = build_connectivity(
        params, positions, labels, n_e, n_i, rng, verbose=False,
    )
    thresholds = np.full(n_e + n_i, 18.0)
    thresholds[:5] = 16.0

    def run(slow):
        network["rng"] = np.random.default_rng(seed)
        return simulate_kick(
            params, network, 5.0, slow=slow,
            kick_center=np.array([0.5, 0.5]), r_kick=0.3,
            t_kick=50.0, V_th_per_neuron=thresholds, verbose=False,
        )

    slow = MZSlowVars(
        n_e + n_i, 18.0, MZSlowVarsConfig(), NE=n_e,
        core_mask_E=np.zeros(n_e, dtype=bool),
    )
    none_result = run(None)
    slow_result = run(slow)
    for key in ("rate_E", "rate_I", "E_spk_bool", "spk_inside", "spk_outside"):
        assert np.array_equal(none_result[key], slow_result[key])
    assert slow_result["E_spk_bool"].sum() > 0


def test_field_accumulator_is_off_by_default_and_exact_when_on():
    slow = _module(use_z=True, use_m=True, I_th_EI=1.0, eta_m=0.5)
    assert slow.field_accumulator_result() is None

    slow.enable_field_accumulator(3)
    excitatory = np.zeros(10)
    d_expected = np.zeros(8)
    a_expected = np.zeros(8)
    for k in range(3):
        inhibitory = np.full(10, float(k) + 1.0)
        slow.z[:8] = 0.25 * (k + 1)
        slow.m[:8] = float(k)
        d_expected += (1.0 - slow.z[:8]) * inhibitory[:8]
        a_expected += 0.5 * slow.m[:8]
        slow.apply_currents(excitatory, inhibitory)
    out = slow.field_accumulator_result()
    assert out["n_steps"] == 3
    assert np.allclose(out["disinhibition_D"], d_expected / 3.0, rtol=0, atol=1e-12)
    assert np.allclose(out["adaptation_A"], a_expected / 3.0, rtol=0, atol=1e-12)
    assert np.allclose(out["net_slow_current"],
                       out["disinhibition_D"] - out["adaptation_A"])


def test_field_accumulator_stops_after_n_steps():
    slow = _module(use_z=True, use_m=True, I_th_EI=1.0, eta_m=0.5)
    slow.enable_field_accumulator(2)
    for _ in range(5):
        slow.apply_currents(np.zeros(10), np.ones(10))
    assert slow.field_accumulator_result()["n_steps"] == 2


def test_field_accumulator_does_not_change_returned_current():
    a = _module(use_z=True, use_m=True, I_th_EI=1.0, eta_m=0.5)
    b = _module(use_z=True, use_m=True, I_th_EI=1.0, eta_m=0.5)
    b.enable_field_accumulator(4)
    excitatory, inhibitory = np.arange(10, dtype=float), np.arange(10, dtype=float) * 0.3
    assert np.array_equal(a.apply_currents(excitatory, inhibitory),
                          b.apply_currents(excitatory, inhibitory))


def test_field_accumulator_is_a_product_average_not_a_product_of_averages():
    """The whole reason this lives inside apply_currents: mean_t[(1-z)*I_I] and
    (1-mean_t[z])*mean_t[I_I] differ whenever z and I_I co-vary, and they do."""
    slow = _module(use_z=True, use_m=True, I_th_EI=1.0, eta_m=0.0)
    slow.enable_field_accumulator(2)
    for z_val, i_val in ((0.0, 10.0), (1.0, 0.0)):
        slow.z[:8] = z_val
        slow.apply_currents(np.zeros(10), np.full(10, i_val))
    out = slow.field_accumulator_result()
    assert np.allclose(out["disinhibition_D"], 5.0)      # mean of (1*10, 0*0)
    product_of_averages = (1.0 - 0.5) * 5.0             # would be 2.5
    assert not np.allclose(out["disinhibition_D"], product_of_averages)
