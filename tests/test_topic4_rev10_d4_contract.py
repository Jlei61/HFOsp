import json
from pathlib import Path

from scripts.freeze_topic4_rev10_d4_uniform_source_grid import uniform_sources

ROOT = Path(__file__).resolve().parents[1]


def _config():
    return json.loads((
        ROOT / "config/topic4_rev10_d4_uniform_forced_source_map.json"
    ).read_text())


def test_uniform_source_grid_is_fixed_5_by_5_without_observation_inputs():
    config = _config()
    sources = uniform_sources(config)
    assert len(sources) == 25
    assert sources[0] == {"source_id": "grid_x02_y02", "xy_mm": [2.0, 2.0]}
    assert sources[-1] == {"source_id": "grid_x18_y18", "xy_mm": [18.0, 18.0]}
    assert set(config["source_grid"]["forbidden_selection_inputs"]) == {
        "contact coordinates", "shaft identity", "patient event labels",
        "field values", "Gaussian components or peaks",
    }
    assert config["source_grid"]["selection"].startswith("nearest_E_neurons")


def test_d4_is_a_paired_diagnostic_not_a_fresh_confirmation():
    config = _config()
    assert config["network_seeds"] == [1201, 1202, 1203]
    assert config["simulation"]["duration_ms"] == 400.0
    assert config["source_grid"]["packet_fraction_of_E"] == 0.005
    assert config["execution"]["wait_seconds"] >= 180
    assert config["claim_boundary"]["forced_packet_is_not_spontaneous_activity"]
