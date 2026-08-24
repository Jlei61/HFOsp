import json
from pathlib import Path

from scripts.run_topic4_rev12_node_worker import _segmentation_variants


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/topic4_rev12_nd_edge_supported_family_replay.json"


def test_replay_freezes_connection_aware_event_identity_before_selection():
    config = json.loads(CONFIG.read_text())
    assert config["scientific_role"] == (
        "development_only_engine_derived_event_identity_canary"
    )
    assert config["search"]["fit_network_seeds"] == [2241, 2242]
    assert "selection_network_seeds" not in config["search"]
    assert "confirmation_network_seeds" not in config["search"]
    assert len(config["field_replay"]["candidate_ids"]) == 4
    assert config["claim_boundary"].startswith("Event-identity replay only")


def test_replay_event_boundaries_use_edges_not_contacts():
    event = json.loads(CONFIG.read_text())["event_unit"]
    assert event["name"] == "edge_supported_causal_family_observation"
    assert event["contact_geometry_used_for_boundary"] is False
    assert event["minimum_parent_support"] == 0.001
    assert event["sensitivity_minimum_parent_supports"] == [0.0003, 0.001, 0.003]
    assert event["edge_delay_rounding"] == "nearest"
    assert event["sensitivity_edge_delay_roundings"] == [
        "floor", "nearest", "ceil",
    ]
    assert len(_segmentation_variants(event)) == 9


def test_replay_worker_supports_exact_per_neuron_family_readout():
    source = (ROOT / "scripts/run_topic4_rev12_node_worker.py").read_text()
    assert '"edge_supported_causal_family_observation"' in source
    assert "binned_ee_delay_support(" in source
    assert "edge_supported_root_families(" in source
    assert '"lineage_restricted_neuron_activity"' in CONFIG.read_text()
