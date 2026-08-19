import json
from pathlib import Path

import numpy as np

from scripts.finalize_topic4_patient_specific_field_cohort_v2 import _aligned_event_rows
from src.topic4_patient_specific_field_cohort import (
    candidate_from_vector,
    initial_vector,
    load_config,
    load_subject_contract,
    objective_from_score,
    projected_field_basis,
    resolve_network_source_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/topic4_patient_specific_field_connectivity_cohort_v2.json"


def test_whole_sheet_basis_matches_optimizer_without_contacts():
    config = load_config(CONFIG)
    basis = projected_field_basis(config)
    assert basis["uses_contact_geometry"] is False
    assert basis["direction_count"] == 12
    assert basis["direction_count"] + 12 == config["search"]["dimension"]
    assert basis["maximum_projection_rmse"] < 1e-3


def test_patient_candidate_is_deterministic_continuous_and_no_k():
    config = load_config(CONFIG)
    basis = projected_field_basis(config)
    first = initial_vector("epilepsiae_590", config)
    second = initial_vector("epilepsiae_590", config)
    assert np.array_equal(first, second)
    candidate = candidate_from_vector(
        "epilepsiae_590", first, config, basis,
        generation=0, candidate_index=0,
    )
    assert candidate["node_field"]["field_type"] == "spline_continuous"
    assert candidate["node_field"]["component_count"] is None
    assert candidate["node_field"]["peak_count_constraint"] is None
    assert np.asarray(candidate["node_field"]["coefficients"]).shape == (18, 18)
    assert np.asarray(candidate["edge_coefficients"]).shape == (2, 6)


def test_real_geometry_contract_is_patient_specific_and_aligned():
    config = load_config(CONFIG)
    contract = load_subject_contract(config, "epilepsiae_590")
    assert contract["real_coords_sheet"].shape == (16, 2)
    assert len(contract["contact_order"]) == 16
    assert contract["target_json"]["target"]["contact_order"] == contract["contact_order"]
    missing = load_subject_contract(config, "yuquan_chenziyang")
    assert missing["real_coords_sheet"] is None


def test_objective_protects_natural_kmeans_and_weakest_mode():
    config = load_config(CONFIG)
    basis = projected_field_basis(config)
    candidate = candidate_from_vector(
        "epilepsiae_590", np.zeros(24), config, basis,
        generation=0, candidate_index=0,
    )
    good = {
        "status": "EVALUABLE", "weakest_mode_loss": 0.2, "ood_fraction": 0.1,
        "natural_kmeans": {
            "weakest_mode_loss": 0.25, "seed_ami_median": 0.9,
            "cluster_counts": np.array([8, 9]),
        },
    }
    collapsed = {
        "status": "EVALUABLE", "weakest_mode_loss": 0.2, "ood_fraction": 0.1,
        "natural_kmeans": {
            "weakest_mode_loss": 0.8, "seed_ami_median": 0.1,
            "cluster_counts": np.array([2, 15]),
        },
    }
    assert objective_from_score(good, candidate, config)["objective"] < (
        objective_from_score(collapsed, candidate, config)["objective"]
    )


def test_config_forbids_canonical_substitution_and_global_optimum_claim():
    payload = json.loads(CONFIG.read_text())
    assert payload["claim_boundary"]["canonical_layout_substitution_forbidden"]
    assert payload["claim_boundary"]["global_optimum_claim_forbidden"]
    assert payload["search"]["heldout_never_selects"]


def test_figure_labels_only_readable_events_and_aligns_kmeans_modes():
    ranks = np.asarray([
        [0.0, 1.0, np.nan, 2.0],
        [np.nan, 0.0, np.nan, 1.0],
        [2.0, 1.0, 0.0, np.nan],
    ])
    ordered, labels = _aligned_event_rows(
        ranks, {"aligned_labels": [1, 0]},
    )
    assert ordered.shape == (2, 4)
    assert labels.tolist() == [0, 1]
    assert np.allclose(ordered[0, :3], [2.0, 1.0, 0.0])


def test_nested_network_source_is_anchored_to_shared_workspace():
    config = load_config(CONFIG)
    base = {
        "small_kick_instrument": {
            "network_cache_source_artifact": "results/frozen_network.json",
        },
    }
    resolved = resolve_network_source_artifact(config, base)
    assert resolved["small_kick_instrument"]["network_cache_source_artifact"] == (
        "/home/honglab/leijiaxin/HFOsp/results/frozen_network.json"
    )
    assert base["small_kick_instrument"]["network_cache_source_artifact"] == (
        "results/frozen_network.json"
    )
