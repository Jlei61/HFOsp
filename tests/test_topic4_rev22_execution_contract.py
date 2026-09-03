import json
from pathlib import Path

import pytest

from scripts.freeze_topic4_rev22_execution_contract import build_execution_config


def _seed_manifest():
    return {
        "fit": {"units": [{"topology_seed": seed, "dynamics_seed": seed} for seed in range(1, 5)]},
        "qualification": {"units": [{"topology_seed": seed, "dynamics_seed": seed} for seed in range(11, 17)]},
        "confirmation": {"units": [{"topology_seed": seed, "dynamics_seed": seed} for seed in range(21, 33)]},
        "variance_decomposition_block": {"units": [
            {"topology_seed": topology, "dynamics_seed": dynamics}
            for topology in range(1, 5) for dynamics in (topology, 101, 102)
        ]},
    }


def _analysis():
    return {
        "scientific_role": "development_only_frozen_dual_core_interictal_connectivity_identifiability",
        "spec": "docs/spec.md", "plan": "docs/plan.md", "output_root": "results/x",
        "network_cache": "results/cache", "dual_core_anchor": {"target_count": 1499},
        "reference": {"g_EE": 0.5, "g_EtoI": 1.0}, "claim_boundary": "development only",
    }


def _rev20():
    return {
        "inputs": {"transition_config": {"path": "config/x.json", "sha256": "abc"}},
        "search": {"simulation": {"duration_ms": 20000.0, "early_stop_runaway": True},
                   "contact_readout": {"source": "lineage"}},
        "event_unit": {"name": "edge_supported_causal_family_observation"},
        "source_topology": {"bin_mm": 1.0}, "complete_distribution": {"floor_draws": 256},
        "validation": {"paired_bootstrap_draws": 4096},
        "resources": {"maximum_workers": 12, "reserved_available_memory_gib": 32},
    }


def test_execution_contract_carries_worker_semantics_and_rev22_seeds():
    result = build_execution_config(_analysis(), _rev20(), _seed_manifest(),
                                    "results/x/candidates.json", {"design": {"sha256": "d"}},
                                    transition_input={"path": "config/minimal.json", "sha256": "min"})
    assert result["scientific_role"].endswith("connectivity_identifiability")
    assert result["candidate_manifest"] == "results/x/candidates.json"
    assert result["search"]["fit_network_seeds"] == [1, 2, 3, 4]
    assert result["search"]["selection_network_seeds"] == list(range(11, 17))
    assert result["search"]["confirmation_network_seeds"] == list(range(21, 33))
    assert result["search"]["dynamics_seeds"] == [101, 102]
    assert result["search"]["simulation"]["duration_ms"] == 20000.0
    assert result["event_unit"] == _rev20()["event_unit"]
    assert result["resources"]["maximum_workers"] == 16
    assert set(result["inputs"]) == {"transition_config"}
    assert result["inputs"]["transition_config"] == {
        "path": "config/minimal.json", "sha256": "min",
    }


def test_bad_seed_counts_fail_closed():
    seeds = _seed_manifest()
    seeds["fit"]["units"].pop()
    with pytest.raises(ValueError, match="4/6/12"):
        build_execution_config(_analysis(), _rev20(), seeds, "results/x/candidates.json", {})
