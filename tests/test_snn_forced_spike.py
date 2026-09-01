import sys

import numpy as np
import pytest

sys.path.insert(0, "src/snn_engine")
from connectivity import place_neurons  # noqa: E402
from connectivity_rot import build_connectivity_rot  # noqa: E402
from kick_probe import simulate_kick  # noqa: E402
from params import Params  # noqa: E402


def _network(seed=4):
    params = Params(L=3.0, density=100.0, T=20.0, dt=0.1,
                    nu_ext_ratio=0.5, seed=seed)
    rng = np.random.default_rng(seed)
    positions, labels, n_e, n_i = place_neurons(params, rng)
    network = build_connectivity_rot(
        params, positions, labels, n_e, n_i, rng,
        theta_EE=np.radians(45.0), AR=2.0, prune_radius=4.3)
    return params, network, n_e, n_i


def test_forced_spike_default_none_is_bit_identical():
    params, network, _, _ = _network()
    network["rng"] = np.random.default_rng(4)
    baseline = simulate_kick(params, network, KICK_BOOST=0.0, t_kick=1e9)
    network["rng"] = np.random.default_rng(4)
    explicit_none = simulate_kick(
        params, network, KICK_BOOST=0.0, t_kick=1e9,
        forced_spike_mask=None, forced_spike_ms=None)
    np.testing.assert_array_equal(
        baseline["E_spk_bool"], explicit_none["E_spk_bool"])
    np.testing.assert_array_equal(baseline["rate_E"], explicit_none["rate_E"])


def test_forced_spike_injects_exact_e_cells_at_exact_step():
    params, network, n_e, n_i = _network()
    mask = np.zeros(n_e + n_i, bool)
    mask[[1, 3, 5]] = True
    network["rng"] = np.random.default_rng(4)
    result = simulate_kick(
        params, network, KICK_BOOST=0.0, t_kick=1e9,
        forced_spike_mask=mask, forced_spike_ms=10.0)
    np.testing.assert_array_equal(
        result["E_spk_bool"][100, [1, 3, 5]], np.ones(3, bool))
    assert result["forced_spike_requested_count"] == 3
    assert result["forced_spike_step"] == 100


def test_forced_spike_rejects_inhibitory_cells_and_off_grid_time():
    params, network, n_e, n_i = _network()
    mask = np.zeros(n_e + n_i, bool)
    mask[n_e] = True
    network["rng"] = np.random.default_rng(4)
    with pytest.raises(ValueError, match="excitatory"):
        simulate_kick(
            params, network, KICK_BOOST=0.0, t_kick=1e9,
            forced_spike_mask=mask, forced_spike_ms=10.0)
    mask[:] = False
    mask[0] = True
    with pytest.raises(ValueError, match="time grid"):
        simulate_kick(
            params, network, KICK_BOOST=0.0, t_kick=1e9,
            forced_spike_mask=mask, forced_spike_ms=10.05)
