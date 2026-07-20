import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_topic4_mz_divisive_figure_capture.py"
SPEC = importlib.util.spec_from_file_location("mz_divisive_figure_capture", SCRIPT)
C = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(C)


def test_strict_recruited_episode_uses_250ms_20hz_1s_contract():
    dt = 10.0
    rate = np.zeros(300)
    rate[100:220] = 30.0
    out, env = C.strict_recruited_episode(rate, dt)
    assert out["status"] == "recruited_macrostate"
    assert out["duration_ms"] >= 1000.0
    # Centered 250-ms smoothing crosses 20 Hz once 17/25 samples are on the 30-Hz plateau.
    assert out["onset_ms"] == 1040.0
    assert env.shape == rate.shape

    too_short = np.zeros(300)
    too_short[100:180] = 30.0
    out_short, _ = C.strict_recruited_episode(too_short, dt)
    assert out_short["status"] == "no_recruited_macrostate"


def test_pre_onset_selection_is_latest_machine_eligible_returned_event():
    dt = 10.0
    rate = np.zeros(500)
    rate[50:58] = 80.0
    rate[180:188] = 70.0
    rate[300:] = 30.0
    macro, state = C.strict_recruited_episode(rate, dt)
    selected, candidates, all_events, _ = C.select_pre_onset_returning_event(
        rate, dt, macro["onset_ms"], state
    )
    assert len(candidates) == 2
    assert len(all_events) >= 3
    assert selected["t_on"] > 1500.0
    assert selected["returned"] is True


def test_per_neuron_latency_rate_and_axial_map_are_derived_not_raster_payloads():
    dt = 1.0
    spk = np.zeros((100, 4), bool)
    spk[12, 0] = True
    spk[15, 1] = True
    spk[20, 0] = True
    event = {"t_on": 10.0, "t_off": 30.0}
    latency = C.per_neuron_first_spike_latency(spk, event, dt)
    np.testing.assert_allclose(latency[:2], [2.0, 5.0])
    assert np.isnan(latency[2:]).all()
    rate = C.per_neuron_window_rate(spk, 10.0, 30.0, dt)
    np.testing.assert_allclose(rate[:2], [100.0, 50.0])

    pos = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    axial, times, centers, edges, occupancy = C.axial_space_time(
        spk, pos, np.array([1.5, 0.0]), np.array([1.0, 0.0]), dt,
        n_space=2, time_bin_ms=10.0,
    )
    assert axial.shape == (10, 2)
    assert times.shape == (10,)
    assert centers.shape == (2,)
    assert edges.shape == (3,)
    np.testing.assert_array_equal(occupancy, [2, 2])


def test_memory_gate_enforces_96gib_floor(monkeypatch):
    monkeypatch.setattr(C, "_meminfo_gib", lambda: {
        "mem_total_gib": 256.0,
        "mem_available_gib": 120.0,
        "swap_total_gib": 0.0,
        "swap_free_gib": 0.0,
    })
    audit = C.memory_gate(96.0, 16.0)
    assert audit["predicted_available_after_launch_gib"] == 104.0
    try:
        C.memory_gate(95.0, 16.0)
    except ValueError:
        pass
    else:
        raise AssertionError("reserve below 96 GiB must be refused")
    try:
        C.memory_gate(110.0, 16.0)
    except RuntimeError:
        pass
    else:
        raise AssertionError("insufficient predicted post-launch reserve must be refused")


def test_unconfirmed_main_refuses_before_simulation(monkeypatch):
    called = []
    monkeypatch.setattr(C, "run_capture", lambda **kwargs: called.append(kwargs))
    assert C.main([]) == 2
    assert called == []
