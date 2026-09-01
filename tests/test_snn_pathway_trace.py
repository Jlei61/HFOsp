"""Readout-only recurrent-pathway recorder contracts."""
from __future__ import annotations

import hashlib
import os
import sys

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

from connectivity import place_neurons  # noqa: E402
from connectivity_rot import build_connectivity_rot  # noqa: E402
from kick_probe import simulate_kick  # noqa: E402
from params import Params  # noqa: E402


BASELINE_SHA = "da5fc18c27d5340a"


def _run(*, dump_pathway_trace=False, trace_dt_ms=1.0):
    params = Params(
        L=6.0, density=100.0, T=300.0, dt=0.1,
        nu_ext_ratio=0.6, seed=1,
    )
    rng = np.random.default_rng(1)
    positions, labels, n_e, n_i = place_neurons(params, rng)
    network = build_connectivity_rot(
        params, positions, labels, n_e, n_i, rng,
        theta_EE=np.radians(45.0), AR=2.0,
    )
    network["rng"] = np.random.default_rng(1)
    return simulate_kick(
        params, network, KICK_BOOST=0.0, t_kick=1e9,
        V_th_per_neuron=np.full(n_e + n_i, 18.0),
        dump_pathway_trace=dump_pathway_trace,
        pathway_trace_dt_ms=trace_dt_ms,
    )


def _spike_sha(result):
    return hashlib.sha1(result["E_spk_bool"].tobytes()).hexdigest()[:16]


def test_pathway_trace_is_default_off_and_preserves_baseline():
    result = _run()
    assert _spike_sha(result) == BASELINE_SHA
    assert "pathway_trace" not in result


def test_pathway_trace_records_population_mean_currents_without_changing_spikes():
    result = _run(dump_pathway_trace=True, trace_dt_ms=1.0)
    assert _spike_sha(result) == BASELINE_SHA
    trace = result["pathway_trace"]
    assert set(trace) == {
        "time_ms", "recurrent_E_to_E_mean",
        "recurrent_E_to_I_mean", "GABA_to_E_mean",
    }
    assert len(trace["time_ms"]) == 300
    np.testing.assert_allclose(np.diff(trace["time_ms"]), 1.0)
    for values in trace.values():
        assert np.asarray(values).dtype == np.float32
        assert np.isfinite(values).all()


def test_pathway_trace_interval_must_lie_on_grid():
    with np.testing.assert_raises_regex(ValueError, "simulation grid"):
        _run(dump_pathway_trace=True, trace_dt_ms=0.15)
