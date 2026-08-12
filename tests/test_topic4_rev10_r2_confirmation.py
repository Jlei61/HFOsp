import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_confirmation_freezer_has_no_simulation_or_confirmation_readout():
    path = ROOT / "scripts/freeze_topic4_rev10_r2_spatial_edge_confirmation.py"
    tree = ast.parse(path.read_text())
    calls = {
        node.func.id for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "simulate_kick" not in calls
    source = path.read_text()
    assert 'summary["diagnostic_best_candidate_id"]' in source
    assert '"edge_noop"' in source
    assert '"confirmation_networks_were_read": False' in source
    assert "selection_score_equal_network" not in source


def test_confirmation_config_uses_fresh_network_phase():
    import json

    config = json.loads((
        ROOT / "config/topic4_rev10_r2_1_spatial_edge_flow_confirmation.json"
    ).read_text())
    assert config["search"]["phase"] == "confirmation"
    assert config["search"]["confirmation_network_seeds"] == [1071, 1072, 1073]
    assert set(config["search"]["confirmation_network_seeds"]).isdisjoint(
        config["search"]["fit_network_seeds"]
        + config["search"]["selection_network_seeds"]
    )
    assert config["search"]["beta"] == "closed"
