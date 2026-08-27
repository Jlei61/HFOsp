import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import rescore_topic4_rev14_static_node_historical_libraries as rescore


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
CONFIG_PATH = ROOT / "config/topic4_rev14_static_node_historical_rescore.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _minimal_npz(path: Path) -> None:
    arrays = {
        "contact_names": np.asarray(["ICL1", "ICL2", "SCL1"]),
        "onsets": np.empty((0, 3)),
        "ranks": np.empty((0, 3)),
        "event_t_on_ms": np.empty(0),
        "event_trigger_t_on_ms": np.empty(0),
        "event_t_off_ms": np.empty(0),
        "event_returned": np.empty(0, dtype=bool),
        "event_fragment_count": np.empty(0, dtype=int),
        "event_directed_root_id": np.empty(0, dtype=int),
        "event_root_count": np.empty(0, dtype=int),
        "source_onset_maps_ms": np.empty((0, 2, 2)),
        "source_onset_evaluable": np.empty(0, dtype=bool),
        "source_bin_mm": np.asarray(1.0),
        "positions_E": np.zeros((2, 2)),
        "delta_vtheta": np.zeros(2),
    }
    np.savez_compressed(path, **arrays)


def _fixture_config(tmp_path: Path, *, policy: str, stage_id: str = "stage_ak",
                    original_eligible: bool = True) -> Path:
    stage_root = tmp_path / stage_id
    workers = stage_root / "workers"
    workers.mkdir(parents=True)
    manifest = {
        "status": "FROZEN",
        "candidates": [{
            "candidate_id": "candidate_a",
            "role": "fixture",
            "selection_eligible": original_eligible,
            "node_field": {"field_sha256": "field-a"},
        }],
    }
    manifest_path = stage_root / "candidate_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    npz_path = workers / "candidate_a_seed_1.npz"
    _minimal_npz(npz_path)
    worker = {
        "status": "COMPLETE",
        "candidate_id": "candidate_a",
        "seed": 1,
        "field_sha256": "field-a",
        "simulation": {
            "duration_ms": 20000.0,
            "runaway_early_stop_ms": None,
        },
        "mechanism_freeze": {
            "EE": "off",
            "E_to_I": "off",
            "Z_M": "off",
            "edge_coefficients_all_zero": 1,
        },
        "provenance": {
            "runtime_modules_dirty": 0,
            "runtime_modules_match_expected_commit": 1,
            "git_commit": "a" * 40,
            "expected_git_commit": "a" * 40,
        },
        "arrays": {"path": str(npz_path), "sha256": _sha256(npz_path)},
    }
    (workers / "candidate_a_seed_1.json").write_text(json.dumps(worker))
    config = {
        "schema_id": "fixture",
        "scientific_role": "fixture",
        "output_root": "unused",
        "inputs": {"stages": [{
            "stage_id": stage_id,
            "seed_pool_id": f"{stage_id}_pool",
            "root": str(stage_root),
            "manifest_sha256": _sha256(manifest_path),
            "manifest_status": "FROZEN",
            "candidate_count": 1,
            "seeds": [1],
            "run_count": 1,
            "anchor_candidate_id": "candidate_a",
            "anchor_semantics": "fixture",
            "selection_policy": policy,
        }]},
        "invariants": {
            "expected_worker_status": "COMPLETE",
            "expected_duration_ms": 20000.0,
            "expected_total_candidates": 1,
            "expected_total_runs": 1,
        },
        "claim_boundary": "fixture",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    return config_path


def test_config_freezes_all_four_libraries_and_366_runs():
    config = json.loads(CONFIG_PATH.read_text())
    stages = config["inputs"]["stages"]
    assert [stage["stage_id"] for stage in stages] == [
        "stage_z", "stage_ag", "stage_ak", "stage_al",
    ]
    assert sum(stage["candidate_count"] for stage in stages) == 78
    assert sum(stage["run_count"] for stage in stages) == 366
    assert config["invariants"]["expected_duration_ms"] == 20000.0
    assert config["invariants"]["cross_seed_pool_raw_ranking"] is False
    assert config["invariants"]["stage_ak_posthoc_promotion"] is False
    assert config["formal_objective"]["schema_id"] == "topic4_rev14_j14_v1"
    assert config["formal_objective"]["candidate_ranking_role"] == (
        "formal_within_stage_seed_pool_only"
    )
    assert config["patient_mode_contract"]["primary_patient_label_key"] == (
        "patient_train_old_labels"
    )
    assert config["patient_mode_contract"][
        "shaft_aware_k2_mapping_applied_to_objective"
    ] is False
    assert next(stage for stage in stages if stage["stage_id"] == "stage_ak")[
        "selection_policy"
    ] == "diagnostic_only_never_selectable"


def test_inventory_forces_stage_ak_to_remain_diagnostic(tmp_path):
    config_path = _fixture_config(
        tmp_path, policy="diagnostic_only_never_selectable",
        stage_id="stage_ak", original_eligible=True,
    )
    inventory = rescore.build_inventory(config_path, tmp_path)
    run = inventory["runs"][0]
    assert run["original_selection_eligible"] is True
    assert run["effective_selection_eligible"] is False
    assert run["diagnostic_only"] is True
    assert inventory["invariants"]["snn_simulation_run"] is False


def test_inventory_preserves_manifest_eligibility_outside_ak(tmp_path):
    config_path = _fixture_config(
        tmp_path, policy="preserve_manifest_selection_eligibility",
        stage_id="stage_z", original_eligible=True,
    )
    inventory = rescore.build_inventory(config_path, tmp_path)
    assert inventory["runs"][0]["effective_selection_eligible"] is True


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"field_sha256": "wrong-field"}, "field identity"),
        ({"provenance": {
            "runtime_modules_dirty": 1,
            "runtime_modules_match_expected_commit": 1,
            "git_commit": "a" * 40,
            "expected_git_commit": "a" * 40,
        }}, "runtime provenance"),
    ],
)
def test_inventory_fails_closed_on_worker_identity_or_provenance(
        tmp_path, mutation, message):
    config_path = _fixture_config(
        tmp_path, policy="preserve_manifest_selection_eligibility",
        stage_id="stage_z", original_eligible=True,
    )
    worker_path = tmp_path / "stage_z/workers/candidate_a_seed_1.json"
    worker = json.loads(worker_path.read_text())
    worker.update(mutation)
    worker_path.write_text(json.dumps(worker))
    with pytest.raises(RuntimeError, match=message):
        rescore.build_inventory(config_path, tmp_path)


def test_inventory_rejects_incomplete_candidate_seed_cartesian_product(tmp_path):
    config_path = _fixture_config(
        tmp_path, policy="preserve_manifest_selection_eligibility",
        stage_id="stage_z",
    )
    config = json.loads(config_path.read_text())
    config["inputs"]["stages"][0]["seeds"] = [1, 2]
    config["inputs"]["stages"][0]["run_count"] = 2
    config["invariants"]["expected_total_runs"] = 2
    config_path.write_text(json.dumps(config))
    with pytest.raises(RuntimeError, match="worker JSON count changed"):
        rescore.build_inventory(config_path, tmp_path)


def test_three_layer_masks_keep_source_unreadable_and_lt3_events_in_contact_loss(
        monkeypatch):
    ranks = np.full((6, 4), np.nan)
    ranks[0, :4] = [0, 1, 2, 3]
    ranks[1, :3] = [0, 1, 2]
    ranks[3, :2] = [0, 1]
    ranks[4, :3] = [0, 1, 2]
    arrays = {
        "event_returned": np.asarray([1, 1, 1, 1, 1, 0], bool),
        "source_onset_evaluable": np.asarray([0, 1, 0, 1, 1, 1], bool),
        "event_t_on_ms": np.asarray([0, 8, 30, 50, 70, 90], float),
        "event_trigger_t_on_ms": np.asarray([1, 9, 31, 51, 71, 91], float),
        "event_t_off_ms": np.asarray([10, 20, 40, 60, 80, 100], float),
        "event_fragment_count": np.ones(6, int),
        "event_directed_root_id": np.arange(6),
        "event_root_count": np.ones(6, int),
        "onsets": ranks.copy(),
        "ranks": ranks,
        "source_onset_maps_ms": np.zeros((6, 2, 2)),
        "source_bin_mm": np.asarray(1.0),
        "positions_E": np.zeros((4, 2)),
        "delta_vtheta": np.ones(4),
    }
    monkeypatch.setattr(
        rescore.exact, "substrate_pca_axis", lambda *args: np.asarray([1.0, 0.0]),
    )
    monkeypatch.setattr(
        rescore.exact, "event_axis_displacements",
        lambda maps, **kwargs: np.asarray([np.nan, 1.0, 2.0]),
    )
    selected = rescore.three_layer_event_selection(
        arrays, minimum_readable_contacts=3,
    )
    np.testing.assert_array_equal(selected["contact_primary_indices"], [2, 3, 4])
    np.testing.assert_array_equal(selected["topology_primary_indices"], [3, 4])
    np.testing.assert_array_equal(selected["fig4_kmeans_readable_indices"], [4])
    assert selected["n_contact_primary_source_not_evaluable"] == 1
    assert selected["n_contact_primary_lt3_finite_contacts"] == 2
    assert selected["overlap_audit"]["n_excluded_families"] == 2
    assert selected["overlap_audit"]["source_map_or_displacement_used"] is False


def test_score_run_sends_all_contact_primary_rows_to_patient_loss(monkeypatch):
    ranks = np.full((3, 3), np.nan)
    ranks[1, :2] = [0, 1]
    ranks[2, :3] = [0, 1, 2]
    arrays = {
        "contact_names": np.asarray(["ICL1", "ICL2", "SCL1"]),
        "event_returned": np.ones(3, bool),
        "source_onset_evaluable": np.asarray([0, 1, 1], bool),
        "event_t_on_ms": np.asarray([0, 20, 40], float),
        "event_trigger_t_on_ms": np.asarray([1, 21, 41], float),
        "event_t_off_ms": np.asarray([10, 30, 50], float),
        "event_fragment_count": np.ones(3, int),
        "event_directed_root_id": np.arange(3),
        "event_root_count": np.ones(3, int),
        "onsets": ranks.copy(),
        "ranks": ranks,
        "source_onset_maps_ms": np.zeros((3, 2, 2)),
        "source_bin_mm": np.asarray(1.0),
        "positions_E": np.zeros((4, 2)),
        "delta_vtheta": np.ones(4),
    }
    monkeypatch.setattr(rescore.exact, "_load_npz_keys", lambda *args: arrays)
    monkeypatch.setattr(
        rescore.exact, "substrate_pca_axis", lambda *args: np.asarray([1.0, 0.0]),
    )
    monkeypatch.setattr(
        rescore.exact, "event_axis_displacements",
        lambda maps, **kwargs: np.asarray([np.nan, 1.0, 2.0]),
    )
    observed = {}

    def fake_assign(values, frozen, groups):
        observed["assign_argument_count"] = 3
        return {
            "probability_B": np.asarray([0.2, 0.4, 0.8]),
            "labels": np.asarray([0, 0, 1]),
            "ood": np.asarray([1, 0, 0], bool),
            "ood_distance": np.asarray([9.0, 1.0, 1.0]),
            "raw_old_labels": np.asarray([0, 0, 1]),
        }

    monkeypatch.setattr(rescore.exact, "_assign_training_modes", fake_assign)

    def fake_objective(values, probabilities, *args, **kwargs):
        observed["legacy_loss_rows"] = len(values)
        return {
            "objective": 1.0,
            "weakest_mode_lse": 1.0,
            "occupancy_js": 0.1,
            "ambiguity": 0.2,
            "contrast": {"loss": 0.3, "alignment": 0.7},
            "modes": {"0": {"mean": 0.8}, "1": {"mean": 0.9}},
        }

    def fake_j14(values, probabilities, *args, **kwargs):
        observed["j14_loss_rows"] = len(values)
        observed["j14_mode_evidence"] = np.asarray(
            kwargs["mode_evidence_mask"], dtype=bool,
        )
        observed["j14_support_counts"] = {
            key: kwargs[key] for key in (
                "returned_families", "contact_evaluable_families",
                "overlap_excluded_families",
                "less_than_three_contact_families",
            )
        }
        return {
            "objective": 2.0,
            "weakest_mode_lse": 1.5,
            "occupancy_js": 0.1,
            "ambiguity": 0.2,
            "contrast": {"loss": 0.3, "alignment": 0.7},
            "overlap_fraction": 0.0,
            "support_loss": 0.4,
            "modes": {
                "0": {"mean": 1.0, "effective_events": 2.0},
                "1": {"mean": 1.2, "effective_events": 2.5},
            },
        }

    monkeypatch.setattr(rescore, "soft_dual_mode_objective", fake_objective)
    monkeypatch.setattr(rescore, "rev14_objective", fake_j14)
    monkeypatch.setattr(
        rescore, "natural_kmeans",
        lambda *args, **kwargs: {"status": "INSUFFICIENT", "n_events": 1},
    )
    context = {
        "minimum_readable_contacts": 3,
        "patient": {
            "contact_names": arrays["contact_names"],
            "reference_ranks": np.zeros((2, 3)),
            "reference_labels": np.asarray([0, 1]),
            "all_ranks": np.zeros((2, 3)),
            "all_labels": np.asarray([0, 1]),
            "all_blocks": np.asarray([0, 1]),
        },
        "frozen_classifier": {},
        "groups": {"ICL": np.asarray([0, 1]), "SCL": np.asarray([2])},
        "objective": {
            "tau": 0.25, "occupancy_weight": 0.5,
            "ambiguity_weight": 0.25, "contrast_weight": 0.5,
            "zero_event_objective": 10.0,
        },
        "formal_objective": {
            "sample_size_per_side": 6,
            "draws_per_network": 64,
            "seed": 20260827,
            "tau": 0.25,
        },
        "projections": np.zeros((1, 6)),
        "calibration": {},
        "natural_kmeans_seed": 1,
        "folds": (np.asarray([0]), np.asarray([1])),
    }
    record = {
        "stage_order": 0, "candidate_order": 0, "stage_id": "stage_z",
        "seed_pool_id": "pool", "candidate_id": "candidate", "seed": 1,
        "original_selection_eligible": True,
        "effective_selection_eligible": True, "diagnostic_only": False,
        "field_sha256": "field", "worker_npz": "unused.npz",
    }
    result = rescore._score_run(record, context)
    assert observed["assign_argument_count"] == 3
    assert observed["legacy_loss_rows"] == 3
    assert observed["j14_loss_rows"] == 3
    np.testing.assert_array_equal(observed["j14_mode_evidence"], [False, False, True])
    assert observed["j14_support_counts"] == {
        "returned_families": 3,
        "contact_evaluable_families": 3,
        "overlap_excluded_families": 0,
        "less_than_three_contact_families": 2,
    }
    assert result["event_selection"]["n_contact_primary"] == 3
    assert result["event_selection"]["n_topology_primary"] == 2
    assert result["event_selection"]["n_fig4_kmeans_readable"] == 1
    assert result["patient_training_assignment"][
        "shaft_aware_k2_mapping_applied"
    ] is False


def _candidate(stage_id: str, candidate_id: str, objective: float,
               eligible: bool, order: int) -> dict:
    return {
        "stage_order": 0 if stage_id == "stage_z" else 1,
        "candidate_order": order,
        "stage_id": stage_id,
        "candidate_id": candidate_id,
        "effective_selection_eligible": eligible,
        "equal_network_j14_v1_summary": {"objective": objective},
    }


def test_ranking_is_stage_local_and_never_promotes_diagnostic_candidates():
    candidates = [
        _candidate("stage_z", "z_anchor", 2.0, True, 0),
        _candidate("stage_z", "z_best", 1.0, True, 1),
        _candidate("stage_ak", "ak_anchor", 0.1, False, 0),
        _candidate("stage_ak", "ak_other", 0.01, False, 1),
    ]
    stages = [
        {"stage_id": "stage_z", "anchor_candidate_id": "z_anchor"},
        {"stage_id": "stage_ak", "anchor_candidate_id": "ak_anchor"},
    ]
    ranked = rescore.apply_within_stage_ranks(candidates, stages)
    by_id = {row["candidate_id"]: row for row in ranked}
    assert by_id["z_best"]["within_stage_eligible_j14_v1_rank"] == 1
    assert by_id["z_anchor"]["within_stage_eligible_j14_v1_rank"] == 2
    assert by_id["ak_anchor"]["within_stage_eligible_j14_v1_rank"] is None
    assert by_id["ak_other"]["within_stage_eligible_j14_v1_rank"] is None
    assert all(row["cross_seed_pool_rank"] is None for row in ranked)


def test_real_manifest_hashes_and_inventory_are_complete():
    config = json.loads(CONFIG_PATH.read_text())
    for stage in config["inputs"]["stages"]:
        path = ARTIFACT_ROOT / stage["root"] / "candidate_manifest.json"
        if not path.exists():
            pytest.skip("canonical historical Node libraries are not mounted")
        assert rescore.exact._sha256(path) == stage["manifest_sha256"]
    inventory = rescore.build_inventory(CONFIG_PATH, ARTIFACT_ROOT)
    assert inventory["counts"] == {"stages": 4, "candidates": 78, "runs": 366}
    assert sum(
        run["stage_id"] == "stage_ak" and run["effective_selection_eligible"]
        for run in inventory["runs"]
    ) == 0


def test_exact_off_j14_reference_is_interface_only_and_matches():
    config = json.loads(CONFIG_PATH.read_text())
    record = config["inputs"]["exact_off_j14_interface_reference"]
    path = ARTIFACT_ROOT / record["path"]
    if not path.exists():
        pytest.skip("canonical exact-off J14 reference is not mounted")
    context = rescore._patient_context(config, ARTIFACT_ROOT)
    parity = rescore.verify_j14_interface(config, ARTIFACT_ROOT, context)
    assert parity["status"] == "PASS"
    assert parity["used_for_candidate_ranking"] is False
    assert parity["current_patient_mode_semantic_parity"] is True
    assert parity["schema_id"] == "topic4_rev14_j14_v1"
    assert parity["equal_network_objective"] == pytest.approx(
        record["equal_network_objective"], abs=1e-12,
    )


def test_runtime_provenance_contains_commit_dirty_and_hashes():
    provenance = rescore.runtime_provenance(CONFIG_PATH)
    assert len(provenance["git_commit_at_analysis"]) == 40
    assert provenance["runtime_paths_dirty"] == bool(
        provenance["runtime_dirty_porcelain"]
    )
    assert set(provenance["runtime_path_sha256"]) == {
        str(path) for path in rescore._runtime_paths(CONFIG_PATH)
    }


def test_source_is_analysis_only_and_reuses_exact_off_contract():
    source = (
        ROOT / "scripts/rescore_topic4_rev14_static_node_historical_libraries.py"
    ).read_text()
    assert "rescore_topic4_rev13_exact_off_static_node" in source
    assert "three_layer_event_selection" in source
    assert "load_patient_training_target" in source
    assert "load_frozen_direction_classifier" in source
    assert "rev14_objective" in source
    score_source = __import__("inspect").getsource(rescore._score_run)
    assert "old_to_shaft_aware_k2" not in score_source
    assert '"direction_balanced_alignment"' in source
    assert '"kmeans_seed_ami_median"' in source
    for forbidden in (
        "run_topic4_rev12", "run_topic4_rev13", "run_snn", "simulate_network",
        "patient_heldout_ranks", "patient_heldout_labels",
    ):
        assert forbidden not in source
