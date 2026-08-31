from __future__ import annotations

import json

import numpy as np
import pytest

from scripts import prepare_topic4_rev17_dual_field_selection as prepare
from scripts.run_topic4_rev12_node_worker import _validate_scientific_role
from src.topic4_observation_invariant_spline import array_sha256
from src.topic4_rev17_dual_field_candidates import (
    AGGREGATE_STATUS,
    build_candidates,
    nomination_blueprint,
)
from src.topic4_rev17_dual_field_residual import mapping_sha256


def _contract() -> dict:
    return {
        "direction_construction": {
            "families": ["mean_a"],
            "candidate_radii": [0.1, 0.2],
            "predicted_B_protection_ratio": 1.1,
            "predicted_minimum_support_per_mode": 6.0,
        }
    }


def _aggregate() -> dict:
    direction = np.zeros(30, dtype=float)
    direction[0] = 0.8
    direction[15] = 0.6
    gradients = {name: np.zeros((3, 30), dtype=float) for name in (
        "A", "B", "J14", "support_A", "support_B",
    )}
    gradients["A"][:, 0] = -2.0
    gradients["J14"][:, 0] = -1.0
    gradients["support_A"][:, 0] = 1.0
    gradients["support_B"][:, 0] = 1.0
    return {
        "status": AGGREGATE_STATUS,
        "inventory": {"complete_cartesian_product": True},
        "boundaries": {
            "natural_kmeans_used": False, "patient_heldout_used": False,
            "ictal_data_used": False, "figure_used": False,
            "EE_EtoI_ZM": "off",
        },
        "response_tensor": {
            "network_seeds": [1, 2, 3],
            "gradients": {name: values.tolist() for name, values in gradients.items()},
        },
        "robust_directions": {"mean_a": {
            "feasible": True, "direction": direction.tolist(),
        }},
        "scored_runs": [
            {
                "candidate_id": "exact_dual_anchor", "seed": seed,
                "mode_0_mean": 2.0, "mode_1_mean": 1.0, "j14": 3.0,
                "mode_0_effective_events": 8.0,
                "mode_1_effective_events": 8.0,
            }
            for seed in (1, 2, 3)
        ],
    }


def _field(identifier: str, offset: float) -> dict:
    values = np.arange(18 * 18, dtype=float).reshape(18, 18) / 1000.0 + offset
    return {
        "candidate_id": identifier,
        "field_type": "spline_continuous",
        "n_basis": 18, "degree": 3,
        "coefficients": values.tolist(),
        "field_sha256": array_sha256(values),
        "roughness": 1.0,
    }


def _atlas() -> dict:
    mean = _field("mean", 0.0)
    dispersion = _field("dispersion", 0.2)
    return {"candidates": [{
        "candidate_id": "exact_dual_anchor",
        "selection_eligible": False,
        "node_field": mean,
        "node_dispersion_field": dispersion,
        "node_mapping": {
            "mapping_type": "dual_continuous_mean_dispersion",
            "mapping_sha256": mapping_sha256(
                mean["field_sha256"], dispersion["field_sha256"],
            ),
        },
        "source_candidate_ids": {},
        "pathways": {
            "learned_E_to_E_redistribution": "off",
            "learned_E_to_I_redistribution": "off", "Z_M": "off",
        },
    }]}


def test_nomination_requires_predicted_improvement_on_every_fit_network():
    blueprints, audit = nomination_blueprint(_aggregate(), _contract())
    assert [row["candidate_id"] for row in blueprints] == [
        "dual_mean_a_r10", "dual_mean_a_r20",
    ]
    assert audit["nominated_candidate_count"] == 2
    assert all(all(row["predicted_gates"].values()) for row in blueprints)


def test_nomination_rejects_forbidden_heldout_or_kmeans_input():
    aggregate = _aggregate()
    aggregate["boundaries"]["natural_kmeans_used"] = True
    with pytest.raises(RuntimeError, match="forbidden"):
        nomination_blueprint(aggregate, _contract())


def test_joint_direction_perturbs_both_continuous_fields_without_pathways():
    blueprints, _ = nomination_blueprint(_aggregate(), _contract())
    candidates, audit = build_candidates(
        _atlas(), blueprints[:1], maximum_frequency=3,
        target_n_basis=18, degree=3, sheet_mm=20.0,
        projection_grid_per_axis=61,
    )
    assert audit["candidate_count_including_anchor"] == 2
    candidate = candidates[1]
    assert candidate["node_field"] != candidates[0]["node_field"]
    assert candidate["node_dispersion_field"] != candidates[0]["node_dispersion_field"]
    assert candidate["node_mapping"]["mapping_type"] == (
        "dual_continuous_mean_dispersion"
    )
    assert candidate["pathways"]["Z_M"] == "off"


def test_selection_role_is_explicitly_accepted_by_shared_worker():
    _validate_scientific_role(
        "development_only_dual_continuous_node_residual_selection"
    )


def test_prepared_selection_keeps_fresh_network_and_closed_pathways(
    monkeypatch, tmp_path,
):
    atlas_manifest = tmp_path / "results/atlas_manifest.json"
    atlas_manifest.parent.mkdir(parents=True)
    atlas_manifest.write_text("{}\n")
    atlas_config = tmp_path / "atlas.json"
    atlas_config.write_text(json.dumps({
        "candidate_manifest": "results/atlas_manifest.json",
        "network_cache": "cache",
        "inputs": {
            "transition_config": {"path": "transition.json", "sha256": "0" * 64},
            "j14_config": {"path": "j14.json", "sha256": "1" * 64},
            "patient_support_config": {"path": "support.json", "sha256": "2" * 64},
        },
        "dual_field_residual": {
            "maximum_frequency": 3, "target_n_basis": 18, "degree": 3,
            "sheet_mm": 20.0, "projection_grid_per_axis": 61,
        },
        "search": {
            "simulation": {"duration_ms": 20000.0},
            "contact_readout": {},
        },
        "event_unit": {}, "source_topology": {}, "resources": {},
    }) + "\n")
    analysis_config = tmp_path / "analysis.json"
    analysis_config.write_text(json.dumps({
        "direction_construction": {
            "fresh_selection_network_seeds": [2371, 2372, 2373],
        }
    }) + "\n")
    aggregate_path = tmp_path / "aggregate.json"
    aggregate_path.write_text("{}\n")
    blueprint = {
        "candidate_id": "dual_mean_a_r10", "family": "mean_a",
        "radius": 0.1, "direction": np.r_[1.0, np.zeros(29)].tolist(),
    }
    monkeypatch.setattr(
        prepare, "nomination_blueprint",
        lambda *args: ([blueprint], {"nominated_candidate_count": 1}),
    )
    config = prepare.build_config(
        aggregate_path=aggregate_path, atlas_config_path=atlas_config,
        analysis_config_path=analysis_config, artifact_root=tmp_path,
    )
    assert config["search"]["selection_network_seeds"] == [2371, 2372, 2373]
    assert config["pathways"] == {
        "learned_E_to_E_redistribution": "off",
        "learned_E_to_I_redistribution": "off", "Z_M": "off",
    }
    assert config["selection"]["natural_kmeans_used"] is False
    assert config["selection"]["patient_heldout_used"] is False
