import json
from pathlib import Path

import numpy as np

from scripts.freeze_topic4_rev10_d3_ee_std_candidates import candidate_library
from src.topic4_graph_edge_flow import array_sha256

ROOT = Path(__file__).resolve().parents[1]


def _config():
    return json.loads((
        ROOT / "config/topic4_rev10_d3_dynamic_ee_std_canary.json"
    ).read_text())


def test_d3_library_is_four_local_global_pairs_plus_off():
    rows = candidate_library(_config())
    assert len(rows) == 9
    keys = {
        (row["ee_std"]["mode"], row["ee_std"]["u"], row["ee_std"]["tau_ms"])
        for row in rows[1:]
    }
    for u in (0.08, 0.2):
        for tau in (500.0, 1500.0):
            assert ("local", u, tau) in keys
            assert ("global", u, tau) in keys


def test_d3_keeps_static_edge_flow_exactly_off():
    for row in candidate_library(_config()):
        coefficients = np.asarray(row["coefficients"], float)
        assert np.array_equal(coefficients, np.zeros(12))
        assert row["coefficients_sha256"] == array_sha256(coefficients)


def test_d3_uses_fresh_networks_and_managed_long_wait():
    config = _config()
    assert config["search"]["fit_network_seeds"] == [1201, 1202, 1203]
    assert config["execution"]["wait_seconds"] >= 180
    assert config["execution"]["launcher"].startswith("systemd-run --user")


def test_d3_has_no_observation_conditioning_parameters():
    library = _config()["ee_std_library"]
    serialized = json.dumps(library).lower()
    for forbidden in ("contact", "shaft", "patient", "core", "gaussian"):
        assert forbidden not in serialized
