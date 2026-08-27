from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import aggregate_topic4_rev14_m3_canary as aggregate
from scripts import freeze_topic4_rev14_patient_support_acceptance as support_freezer
from src.topic4_node_dualmode import (
    calibrate_component_scales,
    fixed_projection_matrix,
)
from src.topic4_rev14_fourier_field import array_sha256
from src.topic4_rev14_patient_support import (
    build_patient_support_calibration,
    patient_training_from_mapping,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _event_unit() -> dict:
    return {
        "name": "edge_supported_causal_family_observation",
        "minimum_active_neurons": 2,
        "minimum_dominance": 0.7,
        "movie_frame_ms": 2.0,
        "movie_bin_mm": 1.0,
        "causal_memory_method": "local_ee_psp_tail",
        "minimum_parent_support": 0.001,
        "minimum_parent_dominance": 0.7,
        "edge_delay_rounding": "nearest",
        "contact_geometry_used_for_boundary": False,
    }


def _worker_event_unit() -> dict:
    frozen = _event_unit()
    return {
        **{key: frozen[key] for key in (
            "name", "minimum_active_neurons", "minimum_dominance",
            "movie_frame_ms", "movie_bin_mm", "causal_memory_method",
            "contact_geometry_used_for_boundary",
        )},
        "edge_support": {
            key: frozen[key] for key in (
                "minimum_parent_support", "minimum_parent_dominance",
                "edge_delay_rounding",
            )
        },
    }


def _candidates() -> list[dict]:
    rows = [{
        "candidate_id": "exact_off",
        "field_kind": "stage_ak_exact_off_benchmark",
        "selection_eligible": False,
        "fourier_coordinate": None,
    }, {
        "candidate_id": "uniform_node",
        "field_kind": "zero_fourier_uniform_benchmark",
        "selection_eligible": False,
        "fourier_coordinate": {
            "modes": [[0, 1]],
            "coefficients": [[0.0, 0.0]],
            "coefficients_sha256": array_sha256(
                np.asarray([[0.0, 0.0]], dtype=np.float64)
            ),
        },
    }]
    for index in range(32):
        coefficients = np.asarray([[float(index + 1), -0.5]], dtype=np.float64)
        rows.append({
            "candidate_id": f"m3_{index:02d}",
            "field_kind": "absolute_paired_phase_fourier_m3",
            "selection_eligible": True,
            "fourier_coordinate": {
                "modes": [[0, 1]],
                "coefficients": coefficients.tolist(),
                "coefficients_sha256": array_sha256(coefficients),
            },
        })
    return rows


def _empty_arrays(candidate: dict, depth: np.ndarray) -> dict[str, np.ndarray]:
    names = np.asarray(["ICL1", "ICL2", "SCL1", "SCL2"])
    coefficients = (
        np.zeros((0, 2), dtype=np.float64)
        if candidate["fourier_coordinate"] is None
        else np.asarray(candidate["fourier_coordinate"]["coefficients"], dtype=np.float64)
    )
    return {
        "contact_names": names,
        "shaft_ids": np.asarray(["ICL", "ICL", "SCL", "SCL"]),
        "onsets": np.empty((0, 4), dtype=np.float32),
        "ranks": np.empty((0, 4), dtype=np.float32),
        "event_t_on_ms": np.empty(0, dtype=np.float32),
        "event_trigger_t_on_ms": np.empty(0, dtype=np.float32),
        "event_t_off_ms": np.empty(0, dtype=np.float32),
        "event_returned": np.empty(0, dtype=bool),
        "event_fragment_count": np.empty(0, dtype=np.int16),
        "event_directed_root_id": np.empty(0, dtype=np.int32),
        "event_root_count": np.empty(0, dtype=np.int32),
        "source_onset_maps_ms": np.empty((0, 2, 2), dtype=np.float32),
        "source_onset_evaluable": np.empty(0, dtype=bool),
        "source_bin_mm": np.asarray(1.0),
        "positions_E": np.asarray([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],
        ], dtype=np.float32),
        "delta_vtheta": np.asarray([-0.2, -0.1, 0.1, 0.2], dtype=np.float32),
        "h": np.asarray([0.2, 0.3, 0.4, 0.5], dtype=np.float32),
        "rev14_fourier_modes": np.asarray([[0, 1]], dtype=np.int16),
        "rev14_fourier_coefficients": coefficients,
        "rev14_frozen_signed_depth": depth,
        "rev14_projection_sha256": np.asarray("p" * 64, dtype="U64"),
    }


def _build_tree(tmp_path: Path) -> dict:
    root = tmp_path / "artifact"
    workers = root / "m3" / "workers"
    output = root / "analysis"
    workers.mkdir(parents=True)
    candidates = _candidates()
    depth = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    config = {
        "candidate_manifest": "m3/candidate_manifest.json",
        "output_root": "m3",
        "search": {
            "active_network_seeds": [2321],
            "simulation": {"duration_ms": 20000.0},
        },
        "node_mapping": {
            "signed_depth_contract": {"sha256": array_sha256(depth)},
        },
    }
    config_path = tmp_path / "m3_config.json"
    _write_json(config_path, config)
    commit = "a" * 40
    manifest = {
        "schema_id": aggregate.MANIFEST_SCHEMA,
        "status": aggregate.MANIFEST_STATUS,
        "config_sha256": _sha256(config_path),
        "candidates": candidates,
        "search": {
            "active_network_seeds": [2321],
            "common_random_numbers_across_candidates": True,
        },
        "event_unit": _event_unit(),
        "provenance": {
            "formal_ready": True,
            "all_explicit_paths_clean": True,
            "git_commit": commit,
            "expected_git_commit": commit,
        },
    }
    manifest_path = root / config["candidate_manifest"]
    _write_json(manifest_path, manifest)
    manifest_sha = _sha256(manifest_path)
    for candidate in candidates:
        stem = f"{candidate['candidate_id']}_seed_2321"
        npz_path = workers / f"{stem}.npz"
        arrays = _empty_arrays(candidate, depth)
        np.savez_compressed(npz_path, **arrays)
        payload = {
            "status": aggregate.WORKER_STATUS,
            "candidate_id": candidate["candidate_id"],
            "seed": 2321,
            "candidate_selection_eligible": candidate["selection_eligible"],
            "fourier_field": candidate["fourier_coordinate"],
            "simulation": {"duration_ms": 20000.0, "runaway_early_stop_ms": None},
            "mechanism_freeze": {
                "EE": "off", "E_to_I": "off", "Z_M": "off",
                "edge_coefficients_all_zero": True,
                "static_node_field": candidate["field_kind"],
            },
            "event_unit": _worker_event_unit(),
            "field_sha256": "h" * 64,
            "field_projection": {"hashes": {
                "projection_sha256": "p" * 64,
                "h_sha256": "h" * 64,
                "delta_vtheta_sha256": "d" * 64,
                "frozen_signed_depth_sha256": array_sha256(depth),
            }},
            "arrays": {"path": str(npz_path), "sha256": _sha256(npz_path)},
            "provenance": {
                "git_commit": commit,
                "expected_git_commit": commit,
                "runtime_modules_dirty": 0,
                "runtime_modules_match_expected_commit": 1,
                "rev14_explicit_runtime_freeze": {
                    "git_commit": commit,
                    "expected_git_commit": commit,
                    "formal_ready": True,
                    "all_explicit_paths_clean": True,
                },
                "rev14_manifest_audit": {"manifest_sha256": manifest_sha},
            },
        }
        _write_json(workers / f"{stem}.json", payload)
    return {
        "root": root, "workers": workers, "output": output,
        "config": config_path, "candidates": candidates,
    }


def _contexts() -> tuple[dict, dict]:
    names = np.asarray(["ICL1", "ICL2", "SCL1", "SCL2"])
    rng = np.random.default_rng(81)
    ranks, onsets, labels, blocks = [], [], [], []
    for block in range(8):
        for mode in (0, 1):
            for repeat in range(2):
                base = np.asarray([0.0, 1.0, 2.0, 3.0])
                if mode:
                    base = base[::-1]
                jitter = rng.normal(0.0, 0.08, size=4)
                onset = base + jitter + 0.1 * repeat
                keep = rng.random(4) > 0.22
                if np.sum(keep) < 2:
                    keep[np.argsort(rng.random(4))[:2]] = True
                onset[~keep] = np.nan
                onsets.append(onset)
                rank = np.full(4, np.nan)
                finite = np.flatnonzero(np.isfinite(onset))
                rank[finite] = np.argsort(np.argsort(onset[finite])).astype(float)
                ranks.append(rank)
                labels.append(mode)
                blocks.append(block)
    ranks = np.asarray(ranks, dtype=float)
    onsets = np.asarray(onsets, dtype=float)
    labels = np.asarray(labels, dtype=np.int8)
    blocks = np.asarray(blocks)
    projections = fixed_projection_matrix(8, n_directions=8, seed=4)
    calibration = calibrate_component_scales(
        ranks, labels, blocks, names, projections,
        sample_size=2, draws=16, seed=5,
    )
    context = {
        "patient": {
            "contact_names": names,
            "all_ranks": ranks,
            "all_labels": labels,
            "all_blocks": blocks,
        },
        "projections": projections,
        "calibration": calibration,
        "frozen_classifier": {},
        "groups": {"ICL": np.asarray([0, 1]), "SCL": np.asarray([2, 3])},
        "minimum_readable_contacts": 3,
        "formal_objective": {
            "sample_size_per_side": 2,
            "draws_per_network": 4,
            "seed": 12,
            "tau": 0.25,
        },
    }
    patient = patient_training_from_mapping({
        "contact_names": names,
        "shaft_ids": np.asarray(["ICL", "ICL", "SCL", "SCL"]),
        "patient_train_onsets": onsets,
        "patient_train_old_labels": labels,
        "patient_train_classifier_labels": labels,
        "patient_train_block_ids": blocks,
        "patient_train_ood": np.zeros(len(labels), dtype=bool),
        "primary_label_key": "patient_train_old_labels",
    })
    support = build_patient_support_calibration(
        patient, sample_size=2, floor_draws=16,
        joint_draws=16, joint_inner_draws=8, seed=17,
    )
    return context, {"patient": patient, "calibration": support}


def test_missing_and_duplicate_workers_are_explicit(tmp_path):
    tree = _build_tree(tmp_path)
    missing = tree["workers"] / "m3_00_seed_2321.json"
    missing.unlink()
    _, _, rows, audit = aggregate.inventory_workers(
        config_path=tree["config"], artifact_root=tree["root"],
        worker_root=tree["workers"],
    )
    assert audit["complete_cartesian_product"] is False
    assert audit["missing"] == ["m3_00"]
    assert next(row for row in rows if row["candidate_id"] == "m3_00")[
        "inventory_status"] == "MISSING"

    source = tree["workers"] / "m3_01_seed_2321.json"
    duplicate = tree["workers"] / "duplicate.json"
    duplicate.write_bytes(source.read_bytes())
    _, _, rows, audit = aggregate.inventory_workers(
        config_path=tree["config"], artifact_root=tree["root"],
        worker_root=tree["workers"],
    )
    assert "m3_01" in audit["duplicate"]
    assert next(row for row in rows if row["candidate_id"] == "m3_01")[
        "inventory_status"] == "DUPLICATE"


def test_hash_drift_and_forbidden_fields_fail_closed(tmp_path):
    tree = _build_tree(tmp_path)
    npz_path = tree["workers"] / "m3_02_seed_2321.npz"
    npz_path.write_bytes(npz_path.read_bytes() + b"drift")

    heldout_path = tree["workers"] / "m3_03_seed_2321.json"
    heldout = json.loads(heldout_path.read_text())
    heldout["patient_heldout_score"] = 0.1
    _write_json(heldout_path, heldout)

    kmeans_json = tree["workers"] / "m3_04_seed_2321.json"
    kmeans_npz = tree["workers"] / "m3_04_seed_2321.npz"
    with np.load(kmeans_npz, allow_pickle=False) as loaded:
        arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
    arrays["natural_kmeans_labels"] = np.asarray([], dtype=np.int8)
    np.savez_compressed(kmeans_npz, **arrays)
    payload = json.loads(kmeans_json.read_text())
    payload["arrays"]["sha256"] = _sha256(kmeans_npz)
    _write_json(kmeans_json, payload)

    _, _, rows, audit = aggregate.inventory_workers(
        config_path=tree["config"], artifact_root=tree["root"],
        worker_root=tree["workers"],
    )
    assert set(audit["invalid_artifact"]) >= {"m3_02", "m3_03", "m3_04"}
    errors = {row["candidate_id"]: row["error"] for row in rows}
    assert "hash" in errors["m3_02"].lower()
    assert "forbidden" in errors["m3_03"].lower()
    assert "forbidden" in errors["m3_04"].lower()


def test_zero_event_worker_gets_finite_bad_score_without_deletion(tmp_path):
    tree = _build_tree(tmp_path)
    context, support = _contexts()
    _, _, rows, audit = aggregate.inventory_workers(
        config_path=tree["config"], artifact_root=tree["root"],
        worker_root=tree["workers"],
    )
    assert audit["complete_cartesian_product"] is True
    exact_row = next(row for row in rows if row["candidate_id"] == "exact_off")
    scored = aggregate._score_worker(exact_row, context, support)
    assert np.isfinite(scored["j14_v1_summary"]["objective"])
    assert scored["event_selection"]["n_contact_primary"] == 0
    assert scored["j14_v1"]["support_loss"] > 0.0
    assert scored["patient_support"]["status"] == "FAIL"
    assert scored["patient_support"]["support"]["n_contact_primary"] == 0


def test_patient_support_sidecar_schema_round_trips():
    _, support_context = _contexts()
    original = support_context["calibration"]
    metadata, arrays, indices = support_freezer._calibration_artifact_parts(original)
    restored = aggregate._load_calibration_from_sidecar({
        "floor_contract": {
            "calibration_metadata": metadata,
            "all_calibration_array_keys": indices,
        },
    }, arrays)
    assert restored.calibration_sha256 == original.calibration_sha256
    assert restored.joint_distribution_sha256 == original.joint_distribution_sha256
    np.testing.assert_array_equal(
        restored.joint_distribution, original.joint_distribution,
    )
    assert set(restored.floor_distributions) == set(original.floor_distributions)


def test_patient_support_artifact_hash_matches_freezer_contract():
    values = np.asarray([0.125, 0.5, 0.875], dtype=np.float64)
    assert aggregate._support_artifact_array_sha256(values) == (
        support_freezer._array_sha256(values)
    )


def test_complete_aggregate_pairs_every_selectable_to_exact_off(
        tmp_path, monkeypatch):
    tree = _build_tree(tmp_path)

    def fake_score(record, context, support_context):
        clean = {key: value for key, value in record.items()
                 if key not in {"arrays", "payload"}}
        if record["candidate_id"] == "exact_off":
            value = 2.0
        elif record["candidate_id"] == "uniform_node":
            value = 3.0
        else:
            value = 1.0 + int(record["candidate_id"].split("_")[1]) / 100.0
        return {
            **clean,
            "j14_v1_summary": {
                "objective": value, "weakest_mode_lse": value,
                "mode_0_effective_events": 6.0,
                "mode_1_effective_events": 6.0,
            },
            "patient_support": {"status": "PASS", "score": value / 2.0},
            "event_selection": {"n_contact_primary": 1, "n_fig4_kmeans_readable": 1},
            "patient_training_assignment": {"ood_count": 0},
        }

    monkeypatch.setattr(aggregate, "_score_worker", fake_score)
    dummy = {"unused": True}
    payload = aggregate.aggregate(
        config_path=tree["config"],
        j14_config_path=tree["config"],
        support_config_path=tree["config"],
        artifact_root=tree["root"], worker_root=tree["workers"],
        output_root=tree["output"], context_override=dummy,
        support_context_override=dummy,
    )
    assert payload["status"] == "COMPLETE"
    assert len(payload["formal_ranking"]) == 32
    assert payload["formal_ranking"][0] == "m3_00"
    rows = {row["candidate_id"]: row for row in payload["per_run"]}
    assert rows["m3_00"]["j14_delta_from_exact_off"] == pytest.approx(-1.0)
    assert rows["m3_31"]["j14_delta_from_exact_off"] == pytest.approx(-0.69)
    assert rows["uniform_node"]["formal_rank"] is None
    assert Path(payload["outputs"]["json"]).is_file()
    assert Path(payload["outputs"]["csv"]).is_file()


def test_patient_support_cannot_reorder_equal_j14_candidates(
        tmp_path, monkeypatch):
    tree = _build_tree(tmp_path)

    def fake_score(record, context, support_context):
        clean = {key: value for key, value in record.items()
                 if key not in {"arrays", "payload"}}
        identifier = record["candidate_id"]
        objective = 2.0 if identifier == "exact_off" else 1.0
        support_score = 100.0 if identifier == "m3_00" else 0.0
        return {
            **clean,
            "j14_v1_summary": {
                "objective": objective, "weakest_mode_lse": objective,
                "mode_0_effective_events": 6.0,
                "mode_1_effective_events": 6.0,
            },
            "patient_support": {"status": "PASS", "score": support_score},
            "event_selection": {"n_contact_primary": 1, "n_fig4_kmeans_readable": 1},
            "patient_training_assignment": {"ood_count": 0},
        }

    monkeypatch.setattr(aggregate, "_score_worker", fake_score)
    payload = aggregate.aggregate(
        config_path=tree["config"], j14_config_path=tree["config"],
        support_config_path=tree["config"], artifact_root=tree["root"],
        worker_root=tree["workers"], output_root=tree["output"],
        context_override={"unused": True},
        support_context_override={"unused": True},
    )
    assert payload["formal_ranking"][:2] == ["m3_00", "m3_01"]
    assert payload["ranking_contract"]["patient_support_role"].endswith(
        "never a canary ranking term"
    )


def test_default_incomplete_collection_writes_no_ranking(tmp_path, monkeypatch):
    tree = _build_tree(tmp_path)
    (tree["workers"] / "m3_00_seed_2321.json").unlink()
    monkeypatch.setattr(
        aggregate, "_score_worker",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("incomplete default must not score")
        ),
    )
    payload = aggregate.aggregate(
        config_path=tree["config"],
        j14_config_path=tree["config"],
        support_config_path=tree["config"],
        artifact_root=tree["root"], worker_root=tree["workers"],
        output_root=tree["output"],
    )
    assert payload["status"] == "INCOMPLETE"
    assert payload["formal_ranking"] == []
    assert payload["inventory"]["missing"] == ["m3_00"]
