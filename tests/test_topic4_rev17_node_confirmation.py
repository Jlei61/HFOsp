from __future__ import annotations

import json

from scripts import prepare_topic4_rev17_node_confirmation as prepare


def test_confirmation_config_copies_one_mapping_and_keeps_mechanisms_closed(
    tmp_path,
):
    repository = tmp_path / "repo"; artifact = tmp_path / "artifact"
    repository.mkdir(); artifact.mkdir()
    for name in ("transition", "j14", "support"):
        (repository / f"{name}.json").write_text("{}\n")
    manifest_path = artifact / "results/selection_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    candidates = [
        {
            "candidate_id": "exact_dual_anchor", "selection_eligible": False,
            "node_mapping": {"mapping_sha256": "a" * 64},
        },
        {
            "candidate_id": "winner", "selection_eligible": True,
            "node_mapping": {"mapping_sha256": "b" * 64},
        },
    ]
    selection_config_path = repository / "selection.json"
    selection_config = {
        "schema_id": "topic4_rev17_dual_field_selection_v1",
        "candidate_manifest": "results/selection_manifest.json",
        "network_cache": "cache",
        "inputs": {
            "transition_config": {
                "path": "transition.json",
                "sha256": prepare._sha256(repository / "transition.json"),
            },
            "j14_config": {
                "path": "j14.json",
                "sha256": prepare._sha256(repository / "j14.json"),
            },
            "patient_support_config": {
                "path": "support.json",
                "sha256": prepare._sha256(repository / "support.json"),
            },
        },
        "search": {
            "selection_network_seeds": [2371, 2372, 2373],
            "simulation": {"duration_ms": 20000.0}, "contact_readout": {},
        },
        "event_unit": {}, "source_topology": {}, "resources": {},
        "pathways": {
            "learned_E_to_E_redistribution": "off",
            "learned_E_to_I_redistribution": "off", "Z_M": "off",
        },
    }
    selection_config_path.write_text(json.dumps(selection_config) + "\n")
    manifest_path.write_text(json.dumps({
        "status": "REV17_DUAL_FIELD_SELECTION_CANDIDATES_FROZEN",
        "config_sha256": prepare._sha256(selection_config_path),
        "candidates": candidates,
    }) + "\n")
    aggregate_path = artifact / "aggregate.json"
    aggregate_path.write_text(json.dumps({
        "schema_id": "topic4_rev17_dual_field_fresh_selection_aggregate_v1",
        "status": "REV17_DUAL_FIELD_FRESH_SELECTION_AGGREGATE_COMPLETE",
        "inventory": {"complete_cartesian_product": True},
        "boundaries": {
            "natural_kmeans_used": False, "patient_heldout_used": False,
            "ictal_data_used": False, "figure_used": False,
            "EE_EtoI_ZM": "off",
        },
        "selected_candidate_id": "winner",
        "eligible_candidates": [{"candidate_id": "winner"}],
    }) + "\n")
    config = prepare.build_config(
        selection_config_path=selection_config_path,
        selection_aggregate_path=aggregate_path,
        artifact_root=artifact, repository_root=repository,
    )
    assert config["selected_candidate"]["mapping_sha256"] == "b" * 64
    assert config["search"]["confirmation_network_seeds"] == [2381, 2382, 2383]
    assert config["pathways"]["Z_M"] == "off"
    assert config["boundaries"]["field_reranking_allowed"] is False
