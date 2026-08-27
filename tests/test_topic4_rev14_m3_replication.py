from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from scripts import freeze_topic4_rev14_m3_canary as canary
from scripts import freeze_topic4_rev14_m3_replication as freezer
from scripts import monitor_topic4_rev14_m3_replication as monitor
from scripts import run_topic4_rev14_m3_canary_worker as canary_worker
from scripts import run_topic4_rev14_m3_replication_worker as worker


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/topic4_rev14_m3_replication.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text())


def _candidate(candidate_id: str, *, selectable: bool) -> dict:
    return {
        "candidate_id": candidate_id,
        "field_kind": (
            "stage_ak_exact_off_benchmark"
            if candidate_id == "exact_off"
            else "absolute_paired_phase_fourier_m3"
        ),
        "selection_eligible": selectable,
        "pathways": copy.deepcopy(canary.EXPECTED_PATHWAYS),
        "fourier_coordinate": None if candidate_id == "exact_off" else {
            "modes": [[0, 1]], "coefficients": [[0.1, -0.1]],
            "coefficients_sha256": "f" * 64,
        },
    }


def test_replication_config_freezes_eight_fields_two_seeds_and_exact_off():
    config = _config()
    freezer._validate_config(config)
    assert len(config["selection"]["selected_candidate_ids"]) == 8
    assert config["search"]["active_network_seeds"] == [2322, 2323]
    assert config["selection"]["paired_nonselectable_benchmark"] == "exact_off"
    assert config["pathways"] == canary.EXPECTED_PATHWAYS


def test_manifest_is_derived_from_complete_j14_ranking_only(monkeypatch):
    config = _config()
    selected = config["selection"]["selected_candidate_ids"]
    canary_config = {
        "m3_design": {
            key: config["m3_design"][key]
            for key in (
                "basis_family", "maximum_order", "expected_modes",
                "expected_real_coefficients", "sheet_length_mm",
                "quadrature_per_axis", "coordinate_decimal_places",
                "basis_uses_observation_geometry", "basis_uses_predeclared_objects",
            )
        },
        "node_mapping": copy.deepcopy(config["node_mapping"]),
        "inputs": {
            "rev13_config": copy.deepcopy(config["inputs"]["rev13_config"]),
            "rev13_exact_off_manifest": copy.deepcopy(
                config["inputs"]["rev13_exact_off_manifest"]
            ),
        },
    }
    candidates = [_candidate("exact_off", selectable=False)] + [
        _candidate(candidate_id, selectable=True) for candidate_id in selected
    ]
    canary_manifest = {
        "status": canary.STATUS,
        "config_sha256": config["inputs"]["m3_canary_config"]["sha256"],
        "candidates": candidates,
        "direction_audit": {"observation_geometry_used": False},
        "signed_depth_audit": {"sha256": "d" * 64},
        "exact_off_reconstruction": {"candidate_id": "exact_off"},
        "event_unit": {"unit": "causal_family"},
        "source_topology": {"source": "frozen"},
    }
    aggregate = {
        "status": "COMPLETE",
        "inventory": {
            "present_validated": 34, "complete_cartesian_product": True,
        },
        "ranking_contract": {
            "natural_kmeans_used": False, "patient_heldout_used": False,
            "ictal_data_used": False, "figure_or_image_used": False,
        },
        "formal_ranking": [*selected, "unused"],
        "per_run": [
            {"candidate_id": "exact_off", "j14_v1_summary": {"objective": 2.1}},
            *[
                {"candidate_id": candidate_id,
                 "j14_v1_summary": {"objective": 1.0 + index / 10}}
                for index, candidate_id in enumerate(selected)
            ],
        ],
    }
    records = {
        "m3_canary_config": canary_config,
        "m3_canary_manifest": canary_manifest,
        "m3_seed_2321_training_aggregate": aggregate,
    }

    def fake_load_record(_config, name, _artifact_root):
        return Path(name), copy.deepcopy(records[name])

    monkeypatch.setattr(freezer, "_load_record", fake_load_record)
    monkeypatch.setattr(
        freezer, "_input_audit",
        lambda _config, _root: {name: {"verified": True} for name in _config["inputs"]},
    )
    payload = freezer.build_manifest_payload(
        CONFIG, artifact_root=Path("/unused"),
        provenance={"formal_ready": False}, status=freezer.PREPARE_STATUS,
    )
    assert [row["candidate_id"] for row in payload["candidates"]] == [
        "exact_off", *selected,
    ]
    assert payload["selection"]["patient_support_can_reorder"] is False
    assert payload["seed_2321_reference"]["exact_off_j14"] == 2.1


def test_replication_worker_reuses_audited_composition_with_new_freezer(monkeypatch):
    observed = {}

    def fake_main(argv=None):
        observed["freezer"] = canary_worker.freezer
        observed["status"] = canary_worker.WORKER_STATUS
        observed["prepare_status"] = canary_worker.PREPARE_STATUS
        observed["argv"] = argv

    monkeypatch.setattr(canary_worker, "main", fake_main)
    worker.main(["--sentinel"])
    assert observed["freezer"] is freezer
    assert observed["status"] == worker.WORKER_STATUS
    assert observed["prepare_status"] == worker.PREPARE_STATUS
    assert observed["argv"] == ["--sentinel"]


def test_monitor_contract_has_eighteen_jobs(tmp_path):
    config = _config()
    config_path = tmp_path / "replication.json"
    config_path.write_text(json.dumps(config))
    selected = config["selection"]["selected_candidate_ids"]
    commit = "c" * 40
    manifest = {
        "status": freezer.STATUS,
        "schema_id": freezer.MANIFEST_SCHEMA,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "candidates": [_candidate("exact_off", selectable=False)] + [
            _candidate(candidate_id, selectable=True) for candidate_id in selected
        ],
        "search": {"active_network_seeds": [2322, 2323]},
        "provenance": {
            "git_commit": commit, "expected_git_commit": commit,
            "formal_ready": True, "all_explicit_paths_clean": True,
            "all_explicit_paths_match_expected_commit": True,
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    config["candidate_manifest"] = "manifest.json"
    config_path.write_text(json.dumps(config))
    manifest["config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    loaded_config, loaded_manifest = monitor._load_contract(
        config_path, tmp_path, commit,
    )
    jobs = monitor.base._jobs(loaded_config, loaded_manifest, tmp_path)
    assert len(jobs) == 18
    assert sum(job["candidate_id"] == "exact_off" for job in jobs) == 2
    assert {job["seed"] for job in jobs} == {2322, 2323}
