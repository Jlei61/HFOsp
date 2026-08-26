import json
from pathlib import Path

import numpy as np

from scripts.aggregate_topic4_rev12_soft_global_fit import _nominate
from scripts.freeze_topic4_rev12_global_soft_field_screen import build_candidates
from src.topic4_continuous_field import continuous_surface
from src.topic4_node_field_search import uniform_sheet_grid


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")


def _candidate(candidate_id, radius, soft, direction, topology, *, kmeans=0.0,
               selection_eligible=True):
    return {
        "candidate_id": candidate_id,
        "fit_valid": True,
        "mean_soft_objective": soft,
        "mean_soft_causal_direction": direction,
        "mean_soft_causal_monotonicity": direction,
        "soft_topology_across_network": topology,
        "soft_topology_mode_separation": topology,
        "natural_kmeans_match": kmeans,
        "candidate": {
            "selection_eligible": selection_eligible,
            "node_field": {"residual_coordinates": {"radius": radius}}
        },
    }


def test_global_screen_is_antithetic_and_observation_invariant():
    stage_u = json.loads((
        ARTIFACT_ROOT / "results/topic4_sef_hfo/data_driven_node_dualmode_rev12/"
        "node_stage_u_orthogonal_free_field_screen/candidate_manifest.json"
    ).read_text())
    soft = json.loads((
        ARTIFACT_ROOT / "results/topic4_sef_hfo/data_driven_node_dualmode_rev12/"
        "node_stage_y_soft_mode_target_audit/soft_mode_target_audit.json"
    ).read_text())
    design = {
        "anchor_candidate_id": "stage_u_anchor",
        "benchmark_candidate_id": "stage_u_f02_m",
        "maximum_cosine_frequency": 3,
        "stored_n_basis": 18,
        "degree": 3,
        "projection_grid_per_axis": 31,
        "n_antithetic_pairs": 4,
        "radii": [0.16, 0.34, 0.52],
        "sobol_seed": 11,
        "expected_candidate_count": 10,
    }
    candidates, audit = build_candidates(stage_u, soft, design)
    assert len(candidates) == 10
    assert audit["patient_or_contact_coordinates_used"] is False
    assert audit["patient_heldout_used"] is False
    assert audit["natural_kmeans_used"] is False
    anchor = np.asarray(candidates[0]["node_field"]["coefficients"], float)
    grid = uniform_sheet_grid(31)
    for pair in range(4):
        minus = np.asarray(candidates[2 + 2 * pair]["node_field"]["coefficients"])
        plus = np.asarray(candidates[3 + 2 * pair]["node_field"]["coefficients"])
        assert np.allclose((minus + plus) / 2.0, anchor)
        residual = continuous_surface(
            plus - anchor, grid, n_basis=18, degree=3, L=20.0,
        )
        residual -= float(np.mean(residual))
        radius = candidates[3 + 2 * pair]["node_field"][
            "residual_coordinates"
        ]["radius"]
        assert np.isclose(np.sqrt(np.mean(residual ** 2)), radius, rtol=2e-3)


def test_nomination_uses_soft_pareto_axes_not_natural_kmeans():
    rows = [
        _candidate("best_soft", 0.16, 0.4, 0.1, 0.1, kmeans=0.0),
        _candidate("best_direction", 0.34, 0.7, 0.9, 0.2, kmeans=0.0),
        _candidate("kmeans_only", 0.52, 0.9, 0.0, 0.0, kmeans=1.0),
        _candidate("balanced", 0.52, 0.5, 0.4, 0.6, kmeans=0.2),
    ]
    selection = {
        "axes": [
            "mean_soft_objective:min",
            "mean_soft_causal_direction:max",
            "mean_soft_causal_monotonicity:max",
            "soft_topology_across_network:max",
            "soft_topology_mode_separation:max",
        ],
        "maximum_nominees": 3,
        "maximum_per_radius": 2,
    }
    first = _nominate(rows, selection)
    rows[2]["natural_kmeans_match"] = -1000.0
    second = _nominate(rows, selection)
    assert first["candidate_ids"] == second["candidate_ids"]
    assert first["candidate_ids"][:2] == ["best_soft", "best_direction"]
    assert "kmeans_only" not in first["candidate_ids"]
    assert first["patient_heldout_used"] is False
    assert first["natural_kmeans_used"] is False


def test_nomination_excludes_missing_topology_instead_of_treating_nan_as_pareto():
    valid = _candidate("valid", 0.16, 0.5, 0.4, 0.3)
    missing = _candidate("missing_topology", 0.34, 0.1, 0.9, float("nan"))
    selection = {
        "axes": [
            "mean_soft_objective:min",
            "mean_soft_causal_direction:max",
            "mean_soft_causal_monotonicity:max",
            "soft_topology_across_network:max",
            "soft_topology_mode_separation:max",
        ],
        "maximum_nominees": 3,
        "maximum_per_radius": 2,
    }
    decision = _nominate([missing, valid], selection)
    assert decision["candidate_ids"] == ["valid"]
    assert decision["pareto_candidate_ids"] == ["valid"]


def test_nomination_excludes_a_nonselectable_capacity_control():
    valid = _candidate("valid", 0.16, 0.8, 0.1, 0.1)
    capacity = _candidate(
        "manual_capacity", None, 0.1, 0.9, 0.9,
        selection_eligible=False,
    )
    selection = {
        "axes": [
            "mean_soft_objective:min",
            "mean_soft_causal_direction:max",
            "mean_soft_causal_monotonicity:max",
            "soft_topology_across_network:max",
            "soft_topology_mode_separation:max",
        ],
        "maximum_nominees": 3,
        "maximum_per_radius": 2,
    }
    decision = _nominate([capacity, valid], selection)
    assert decision["candidate_ids"] == ["valid"]
    assert decision["pareto_candidate_ids"] == ["valid"]
