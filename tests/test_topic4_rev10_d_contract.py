import json
from pathlib import Path

import numpy as np

from scripts.freeze_topic4_rev10_d_adaptation_candidates import candidate_library
from src.topic4_graph_edge_flow import array_sha256


ROOT = Path(__file__).resolve().parents[1]


def _config():
    return json.loads((
        ROOT / "config/topic4_rev10_d_local_adaptation_canary.json"
    ).read_text())


def test_frozen_grid_has_off_plus_paired_local_global_candidates():
    candidates = candidate_library(_config())
    assert len(candidates) == 19
    assert candidates[0]["candidate_id"] == "edge_noop"
    keys = {
        (row["adaptation"]["mode"], row["adaptation"]["tau_ms"],
         row["adaptation"]["increment_mV"])
        for row in candidates[1:]
    }
    for tau in (250.0, 750.0, 2000.0):
        for increment in (0.1, 0.25, 0.5):
            assert ("local", tau, increment) in keys
            assert ("global", tau, increment) in keys


def test_every_dynamic_candidate_keeps_static_edges_exactly_off():
    for candidate in candidate_library(_config()):
        assert np.array_equal(candidate["coefficients"], np.zeros(12))
        assert candidate["coefficients_sha256"] == array_sha256(
            np.asarray(candidate["coefficients"], dtype=np.float64)
        )


def test_canary_uses_fresh_networks_and_long_controller_wait():
    config = _config()
    assert config["search"]["fit_network_seeds"] == [1081, 1082, 1083]
    assert config["execution"]["wait_seconds"] >= 180
    assert config["execution"]["screen_max_workers"] >= 8


def test_dynamic_mechanism_contract_forbids_observation_conditioning():
    config = _config()
    assert config["claim_boundary"]["dynamic_state_uses_only_model_spike_history"]
    assert config["claim_boundary"]["static_edges_are_exact_noop"]
    forbidden = " ".join(config["forbidden"]).lower()
    assert "contact" in forbidden
    assert "gaussian" in forbidden
    assert "larger k" in forbidden
