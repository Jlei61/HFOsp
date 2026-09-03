import copy
import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "freeze_structural", ROOT / "scripts/freeze_topic4_rev22_structural_nulls.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _candidate(candidate_id, g_ee=0.5, g_etoi=1.0):
    return {
        "candidate_id": candidate_id,
        "node_field": {"field_type": "manual_dual_core_budget_matched",
                       "centers_mm": [[1.0, 1.0], [19.0, 2.0]], "target_count": 1499,
                       "field_sha256": "field"},
        "node_mapping": {"node_gain": 1.0, "signed_depth_shrinkage": 1.0},
        "mechanisms": {"g_EE": g_ee, "g_EtoI": g_etoi, "ellipse_angle_deg": 0.0,
                       "ellipse_aspect_ratio": 2.0,
                       "ellipse_reference_angle_deg": 0.0,
                       "ellipse_reference_aspect_ratio": 2.0, "Z_M": "off"},
    }


def test_structural_freezer_emits_exact_predeclared_factorial():
    analysis = {
        "dual_core_anchor": {"centers_mm": [[1.0, 1.0], [19.0, 2.0]], "target_count": 1499},
        "structural_nulls": {
            "random_learned_row_seed": 11, "random_two_core_seed": 12,
            "fixed_topology_variants": ["r180", "r90", "matched_norm_row_1",
                                        "matched_norm_row_2", "merged_midpoint_core",
                                        "random_two_core_centers"],
            "node_blocking_factors": [
                {"candidate_id": "exact_dual_anchor", "historical_status": "reference"},
                {"candidate_id": "v62_density_t050", "historical_status": "unconfirmed"},
            ],
        },
    }
    execution = {"reference": {"base_substrate_candidate_id": "joint_04_control"}}
    tree = _candidate("tree_M1111")
    tree["proposal_origin"] = "tree"
    gp = _candidate("proposal_M1111")
    gp["proposal_origin"] = "gp"
    execution_manifest = {"candidates": [_candidate("dci_p000"), tree, gp]}
    frozen = {"mask_to_candidates": {"M1111": ["tree_M1111", "proposal_M1111"]}}
    substrate = {"candidate_set": {"candidates": [{"candidate_id": "joint_04_control",
                                                     "coefficients": np.arange(12).reshape(2, 6).tolist()}]}}
    exact = {"candidate_id": "exact_dual_anchor", "node_field": {"field_type": "spline_continuous",
             "field_sha256": "exact"}, "node_dispersion_field": {"field_type": "spline_continuous",
             "field_sha256": "disp"}, "node_mapping": {"node_gain": 1.0,
             "signed_depth_shrinkage": 1.0}}
    v62 = {"candidate_id": "v62_density_t050", "field_type": "spline_continuous",
           "field_sha256": "v62"}
    rows = MODULE.build_structural_candidates(
        analysis=analysis, execution=execution, frozen=frozen,
        execution_manifest=execution_manifest, substrate_manifest=substrate,
        node_manifests={"exact_dual_anchor": {"candidates": [exact]},
                        "v62_density_t050": {"candidate_set": {"candidates": [v62]}}},
    )
    assert len(rows) == 18
    assert len({row["candidate_id"] for row in rows}) == 18
    assert sum(row["structural_null"]["family"] == "fixed_topology" for row in rows) == 12
    assert sum(row["structural_null"]["family"] == "isotropic_graph" for row in rows) == 2
    assert sum(row["structural_null"]["family"] == "node_blocking_factor" for row in rows) == 4
    assert all(row["mechanisms"]["Z_M"] == "off" for row in rows)
    assert all(row.get("structural_null_primary_selection", {}).get("candidate_id")
               in (None, "proposal_M1111") for row in rows)
    iso = [r for r in rows if r["structural_null"]["family"] == "isotropic_graph"]
    assert all(r["topology_override"]["graph_aspect_ratio"] == 1.0 for r in iso)


def test_matched_rows_are_deterministic_and_replace_only_one_row():
    source = np.arange(1.0, 13.0).reshape(2, 6)
    first_a, second_a = MODULE.matched_norm_rows(source, 71)
    first_b, second_b = MODULE.matched_norm_rows(source, 71)
    assert np.array_equal(first_a, first_b)
    assert np.array_equal(second_a, second_b)
    assert np.array_equal(first_a[1], source[1])
    assert np.array_equal(second_a[0], source[0])
    assert np.isclose(np.linalg.norm(first_a[0]), np.linalg.norm(source[0]), rtol=0, atol=1e-12)
    assert np.isclose(np.linalg.norm(second_a[1]), np.linalg.norm(source[1]), rtol=0, atol=1e-12)


def test_random_placement_is_deterministic_interior_and_separated():
    centers = np.asarray(MODULE.random_two_core_centers(29))
    assert np.array_equal(centers, MODULE.random_two_core_centers(29))
    assert np.all((centers >= 2.0) & (centers <= 18.0))
    assert np.linalg.norm(centers[0] - centers[1]) >= 10.0
