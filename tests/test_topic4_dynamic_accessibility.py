import numpy as np

from src.topic4_dynamic_accessibility import (
    ActivityDependentInhibitoryResource,
    AdaptationConfig,
    InhibitoryResourceConfig,
    SpikeTriggeredAdaptation,
)


def _model(mode, *, tau=100.0, increment=0.5):
    return SpikeTriggeredAdaptation(
        6, 4, 1.0,
        AdaptationConfig(mode, tau, increment, trace_dt_ms=1.0),
    )


def test_threshold_vector_is_passed_through_without_scalar_collapse():
    model = _model("local")
    threshold = np.asarray([17.0, 18.0, 19.0, 20.0, 18.0, 18.0])
    assert model.threshold(threshold) is threshold


def test_local_and_global_have_equal_mean_for_identical_spike_history():
    local, global_ = _model("local"), _model("global")
    labels = np.asarray([0, 0, 0, 0, 1, 1])
    histories = [
        [1, 0, 0, 0, 1, 0],
        [0, 1, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
    ]
    for history in histories:
        spikes = np.asarray(history, bool)
        local.step(spikes, labels, 1.0)
        global_.step(spikes, labels, 1.0)
        assert np.isclose(local.local_state.mean(), global_.global_state)


def test_local_state_retains_neuron_identity_and_global_control_does_not():
    local, global_ = _model("local"), _model("global")
    labels = np.asarray([0, 0, 0, 0, 1, 1])
    spikes = np.asarray([1, 0, 0, 0, 1, 1], bool)
    local.step(spikes, labels, 1.0)
    global_.step(spikes, labels, 1.0)
    assert np.std(local.local_state) > 0.0
    assert global_.trace_sd_mV[-1] == 0.0
    assert local.local_state[0] == 0.5
    assert np.all(local.local_state[1:] == 0.0)


def test_adaptation_is_e_only_and_subtractive():
    model = _model("local")
    labels = np.asarray([0, 0, 0, 0, 1, 1])
    model.step(np.asarray([1, 0, 0, 0, 1, 1], bool), labels, 1.0)
    output = model.apply_currents(np.ones(6), np.zeros(6), labels)
    np.testing.assert_allclose(output, [0.5, 1.0, 1.0, 1.0, 1.0, 1.0])


def test_state_decays_exactly_between_spikes():
    model = _model("local", tau=10.0, increment=1.0)
    labels = np.asarray([0, 0, 0, 0, 1, 1])
    model.step(np.asarray([1, 0, 0, 0, 0, 0], bool), labels, 1.0)
    model.step(np.zeros(6, bool), labels, 1.0)
    assert np.isclose(model.local_state[0], np.exp(-0.1))


def test_module_has_no_observation_conditioning_inputs():
    import inspect

    source = inspect.getsource(SpikeTriggeredAdaptation).lower()
    for forbidden in ("contact", "shaft", "patient", "core_mask"):
        assert forbidden not in source


def test_resource_has_no_observation_conditioning_inputs():
    import inspect

    source = inspect.getsource(ActivityDependentInhibitoryResource).lower()
    for forbidden in ("contact", "shaft", "patient", "core_mask"):
        assert forbidden not in source


def test_zero_increment_is_engine_bit_parity():
    import os
    import sys
    from pathlib import Path

    engine = Path(__file__).resolve().parents[1] / "src" / "snn_engine"
    sys.path.insert(0, os.fspath(engine))
    from connectivity import place_neurons
    from connectivity_rot import build_connectivity_rot
    from kick_probe import simulate_kick
    from params import Params

    def build():
        params = Params(
            L=4.0, density=50.0, T=120.0, dt=0.1,
            nu_ext_ratio=0.6, seed=7,
        )
        rng = np.random.default_rng(7)
        positions, labels, n_e, n_i = place_neurons(params, rng)
        network = build_connectivity_rot(
            params, positions, labels, n_e, n_i, rng,
            theta_EE=np.radians(45.0), AR=2.0,
        )
        network["rng"] = np.random.default_rng(11)
        return params, network, n_e, n_i

    params, network, n_e, n_i = build()
    baseline = simulate_kick(
        params, network, KICK_BOOST=5.0, r_kick=1.0,
        V_th_per_neuron=np.full(n_e + n_i, 16.5),
    )
    params, network, n_e, n_i = build()
    slow = SpikeTriggeredAdaptation(
        n_e + n_i, n_e, params.dt,
        AdaptationConfig("local", 750.0, 0.0, trace_dt_ms=10.0),
    )
    adapted = simulate_kick(
        params, network, KICK_BOOST=5.0, r_kick=1.0,
        V_th_per_neuron=np.full(n_e + n_i, 16.5), slow=slow,
    )
    assert np.array_equal(baseline["E_spk_bool"], adapted["E_spk_bool"])
    assert np.array_equal(baseline["rate_E"], adapted["rate_E"])


def _resource(mode, k_q=0.1):
    positions_e = np.asarray([[1.0, 1.0], [1.2, 1.0], [3.0, 3.0]])
    positions_i = np.asarray([[1.0, 1.2], [3.0, 3.2]])
    return ActivityDependentInhibitoryResource(
        positions_e, positions_i, 4.0, 1.0,
        InhibitoryResourceConfig(
            mode=mode, tau_q_ms=100.0, k_q_per_ms=k_q,
            n_grid=8, sigma_rate_mm=0.4, sigma_q_mm=0.8,
            tau_rate_ms=10.0, trace_dt_ms=1.0, update_interval_ms=1.0,
        ),
    )


def test_local_and_global_resource_receive_same_mean_depletion_drive():
    local, global_ = _resource("local"), _resource("global")
    labels = np.asarray([0, 0, 0, 1, 1])
    for history in ([1, 1, 0, 1, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 1]):
        spikes = np.asarray(history, bool)
        local.step(spikes, labels, 1.0)
        global_.step(spikes, labels, 1.0)
        assert np.isclose(local.last_mean_drive, global_.last_mean_drive)


def test_local_resource_retains_spatial_identity_and_global_does_not():
    local, global_ = _resource("local"), _resource("global")
    labels = np.asarray([0, 0, 0, 1, 1])
    spikes = np.asarray([1, 1, 0, 1, 0], bool)
    for _ in range(20):
        local.step(spikes, labels, 1.0)
        global_.step(spikes, labels, 1.0)
    assert local.trace_q_sd[-1] > 0.0
    assert global_.trace_q_sd[-1] == 0.0


def test_resource_depletion_reduces_inhibitory_current_only_on_e_cells():
    local = _resource("local")
    labels = np.asarray([0, 0, 0, 1, 1])
    spikes = np.asarray([1, 1, 0, 1, 0], bool)
    for _ in range(20):
        local.step(spikes, labels, 1.0)
    output = local.apply_currents(np.ones(5), np.ones(5), labels)
    assert np.any(output[:3] > 0.0)
    np.testing.assert_allclose(output[3:], 0.0)


def test_zero_depletion_resource_is_engine_bit_parity():
    import os
    import sys
    from pathlib import Path

    engine = Path(__file__).resolve().parents[1] / "src" / "snn_engine"
    sys.path.insert(0, os.fspath(engine))
    from connectivity import place_neurons
    from connectivity_rot import build_connectivity_rot
    from kick_probe import simulate_kick
    from params import Params

    def build():
        params = Params(L=4.0, density=50.0, T=120.0, dt=0.1,
                        nu_ext_ratio=0.6, seed=17)
        rng = np.random.default_rng(17)
        positions, labels, n_e, n_i = place_neurons(params, rng)
        network = build_connectivity_rot(
            params, positions, labels, n_e, n_i, rng,
            theta_EE=np.radians(45.0), AR=2.0,
        )
        network["rng"] = np.random.default_rng(19)
        return params, network, n_e, n_i

    params, network, n_e, n_i = build()
    baseline = simulate_kick(
        params, network, KICK_BOOST=5.0, r_kick=1.0,
        V_th_per_neuron=np.full(n_e + n_i, 16.5),
    )
    params, network, n_e, n_i = build()
    resource = ActivityDependentInhibitoryResource(
        network["pos"][:n_e], network["pos"][n_e:], params.L, params.dt,
        InhibitoryResourceConfig(
            "local", tau_q_ms=750.0, k_q_per_ms=0.0,
            trace_dt_ms=10.0,
        ),
    )
    dynamic = simulate_kick(
        params, network, KICK_BOOST=5.0, r_kick=1.0,
        V_th_per_neuron=np.full(n_e + n_i, 16.5), slow=resource,
    )
    assert np.array_equal(baseline["E_spk_bool"], dynamic["E_spk_bool"])
    assert np.array_equal(baseline["rate_E"], dynamic["rate_E"])


def test_one_ms_resource_update_tracks_point_one_ms_reference():
    positions_e = np.asarray([[1.0, 1.0], [1.2, 1.0], [3.0, 3.0]])
    positions_i = np.asarray([[1.0, 1.2], [3.0, 3.2]])
    common = dict(
        mode="local", tau_q_ms=750.0, k_q_per_ms=0.001,
        n_grid=8, sigma_rate_mm=0.4, sigma_q_mm=0.8,
        tau_rate_ms=100.0, trace_dt_ms=10.0,
    )
    fine = ActivityDependentInhibitoryResource(
        positions_e, positions_i, 4.0, 0.1,
        InhibitoryResourceConfig(**common, update_interval_ms=0.1),
    )
    coarse = ActivityDependentInhibitoryResource(
        positions_e, positions_i, 4.0, 0.1,
        InhibitoryResourceConfig(**common, update_interval_ms=1.0),
    )
    labels = np.asarray([0, 0, 0, 1, 1])
    for step in range(2000):
        spikes = np.asarray([
            step % 10 == 0, step % 13 == 0, step % 37 == 0,
            step % 11 == 0, step % 29 == 0,
        ])
        fine.step(spikes, labels, 0.1)
        coarse.step(spikes, labels, 0.1)
    assert np.max(np.abs(fine.q_field - coarse.q_field)) < 2e-4
