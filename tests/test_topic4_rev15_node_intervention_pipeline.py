from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from scripts import aggregate_topic4_rev15_node_intervention as aggregate
from scripts import run_topic4_rev15_node_intervention_worker as worker
from scripts import run_topic4_rev12_node_intervention as intervention


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _branch(event=True, latency=40.0, parity=True, retained=None):
    if retained is None:
        retained = event
    return {
        "event_occurred": event,
        "latency_from_checkpoint_ms": latency if event else None,
        "pre_intervention_spike_parity": parity,
        "native_mode_retained": bool(retained),
    }


def _native(*, selective_mode0: bool, general_mode0: bool = False):
    mode0_own = _branch(False) if selective_mode0 else _branch(True, 40.0)
    mode0_cross = _branch(False) if general_mode0 else _branch(True, 43.0)
    return {
        "0": {
            "sham": _branch(True, 40.0),
            "mode0_hotspot": mode0_own,
            "mode0_matched_off_template": _branch(True, 42.0),
            "mode1_hotspot": _branch(True, 40.0),
            "mode1_matched_off_template": _branch(True, 40.0),
        },
        "1": {
            "sham": _branch(True, 40.0),
            "mode0_hotspot": mode0_cross,
            "mode0_matched_off_template": _branch(True, 40.0),
            "mode1_hotspot": _branch(True, 40.0),
            "mode1_matched_off_template": _branch(True, 42.0),
        },
    }


def _tree(tmp_path: Path, *, general=False):
    artifact = tmp_path / "artifact"
    config = {
        "schema_id": "topic4_rev15_node_crossed_intervention_v1",
        "output_root": "results/intervention",
        "candidate_id": "selected",
        "network_seeds": [2341, 2342, 2343],
        "decision": {"required_selective_networks": 2},
        "mechanism_freeze": {"EE": "off", "E_to_I": "off", "Z_M": "off"},
        "inputs": {
            "robust_config": {"path": "config/robust.json", "sha256": "r" * 64},
            "robust_manifest": {"path": "results/manifest.json", "sha256": "m" * 64},
            "final_science_audit": {"path": "results/final.json", "sha256": "f" * 64},
        },
        "claim_boundary": "model internal",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config) + "\n")
    root = artifact / "results/intervention/workers"
    root.mkdir(parents=True)
    for index, seed in enumerate(config["network_seeds"]):
        arrays = root / f"intervention_seed_{seed}.npz"
        np.savez(arrays, value=np.asarray([seed]))
        payload = {
            "schema_id": worker.OUTPUT_SCHEMA,
            "status": worker.WORKER_STATUS,
            "candidate_id": "selected", "network_seed": seed,
            "mechanism_freeze": config["mechanism_freeze"],
            "inputs": {"config": {"sha256": _sha(config_path)}},
            "arrays": {"path": str(arrays), "sha256": _sha(arrays)},
            "provenance": {"formal_ready": True},
            "projection_parity": {"exact_array_parity": {
                "h": True, "vtheta": True, "delta_vtheta": True,
            }},
            "native_modes": _native(
                selective_mode0=index < 2, general_mode0=general,
            ),
        }
        (root / f"intervention_seed_{seed}.json").write_text(
            json.dumps(payload) + "\n"
        )
    return artifact, config_path


def test_intervention_aggregate_freezes_only_crossed_selective_field(
    tmp_path, monkeypatch,
):
    artifact, config = _tree(tmp_path)
    monkeypatch.setattr(
        aggregate, "_provenance",
        lambda: {"analysis_commit": "x", "worktree_status": [],
                 "formal_ready": True, "SNN_simulation_run": False},
    )
    result = aggregate.aggregate(config_path=config, artifact_root=artifact)
    assert result["status"] == "REV15_NODE_FIELD_FROZEN"
    assert result["node_freeze_permitted"] is True
    freeze = artifact / "results/intervention/analysis/node_freeze_manifest.json"
    assert freeze.is_file()


def test_intervention_aggregate_rejects_general_suppression(tmp_path, monkeypatch):
    artifact, config = _tree(tmp_path, general=True)
    monkeypatch.setattr(
        aggregate, "_provenance",
        lambda: {"analysis_commit": "x", "worktree_status": [],
                 "formal_ready": True, "SNN_simulation_run": False},
    )
    result = aggregate.aggregate(config_path=config, artifact_root=artifact)
    assert result["status"] == "REV15_NODE_INTERVENTION_NOT_SELECTIVE"
    assert result["node_freeze_permitted"] is False
    assert result["outputs"]["freeze_manifest"] is None


def test_source_bundle_excludes_every_overlap_connected_episode_member(
    tmp_path, monkeypatch,
):
    npz_path = tmp_path / "worker.npz"
    ranks = np.asarray([
        [0.0, 1.0, 2.0],
        [2.0, 1.0, 0.0],
        [0.0, 2.0, 1.0],
        [1.0, 0.0, 2.0],
    ])
    source_maps = np.stack([
        np.full((2, 2), 1.0),
        np.full((2, 2), 2.0),
        np.full((2, 2), 3.0),
        np.full((2, 2), 4.0),
    ])
    np.savez(
        npz_path,
        event_returned=np.ones(4, dtype=bool),
        source_onset_evaluable=np.ones(4, dtype=bool),
        event_t_on_ms=np.asarray([100.0, 150.0, 300.0, 500.0]),
        event_trigger_t_on_ms=np.asarray([100.0, 150.0, 300.0, 500.0]),
        event_t_off_ms=np.asarray([200.0, 180.0, 350.0, 550.0]),
        event_fragment_count=np.ones(4, dtype=int),
        event_directed_root_id=np.arange(4),
        event_root_count=np.ones(4, dtype=int),
        onsets=ranks,
        ranks=ranks,
        source_onset_maps_ms=source_maps,
        positions_E=np.zeros((4, 2)),
        delta_vtheta=np.ones(4),
        source_bin_mm=np.asarray(1.0),
    )
    monkeypatch.setattr(
        intervention.historical.exact,
        "substrate_pca_axis",
        lambda *_args, **_kwargs: np.asarray([1.0, 0.0]),
    )
    monkeypatch.setattr(
        intervention.historical.exact,
        "event_axis_displacements",
        lambda maps, **_kwargs: np.ones(len(maps), dtype=float),
    )
    maps, labels = intervention._source_bundle(
        npz_path, {
            "labels": np.asarray([0, 1, 0, 1]),
            "ranks": ranks,
            "formal_clean": np.ones(4, dtype=bool),
        },
    )
    np.testing.assert_array_equal(maps, source_maps[[2, 3]])
    np.testing.assert_array_equal(labels, [0, 1])
    assert intervention._event_contract(
        npz_path, {
            "labels": np.asarray([0, 1, 0, 1]), "ranks": ranks,
            "formal_clean": np.ones(4, dtype=bool),
        }, 0,
    )["detected_event_index"] == 2
    assert intervention._event_contract(
        npz_path, {
            "labels": np.asarray([0, 1, 0, 1]), "ranks": ranks,
            "formal_clean": np.ones(4, dtype=bool),
        }, 1,
    )["detected_event_index"] == 3


def test_source_templates_exclude_ood_or_single_shaft_forced_labels(
    tmp_path, monkeypatch,
):
    npz_path = tmp_path / "worker.npz"
    ranks = np.asarray([
        [0.0, 1.0, 2.0],
        [0.0, 2.0, 1.0],
        [2.0, 0.0, 1.0],
        [2.0, 1.0, 0.0],
    ])
    source_maps = np.stack([
        np.full((2, 2), 1.0),
        np.full((2, 2), 2.0),
        np.full((2, 2), 3.0),
        np.full((2, 2), 4.0),
    ])
    np.savez(
        npz_path,
        event_returned=np.ones(4, dtype=bool),
        source_onset_evaluable=np.ones(4, dtype=bool),
        event_t_on_ms=np.asarray([100.0, 300.0, 500.0, 700.0]),
        event_trigger_t_on_ms=np.asarray([100.0, 300.0, 500.0, 700.0]),
        event_t_off_ms=np.asarray([150.0, 350.0, 550.0, 750.0]),
        event_fragment_count=np.ones(4, dtype=int),
        event_directed_root_id=np.arange(4),
        event_root_count=np.ones(4, dtype=int),
        onsets=ranks,
        ranks=ranks,
        source_onset_maps_ms=source_maps,
        positions_E=np.zeros((4, 2)),
        delta_vtheta=np.ones(4),
        source_bin_mm=np.asarray(1.0),
    )
    monkeypatch.setattr(
        intervention.historical.exact,
        "substrate_pca_axis",
        lambda *_args, **_kwargs: np.asarray([1.0, 0.0]),
    )
    monkeypatch.setattr(
        intervention.historical.exact,
        "event_axis_displacements",
        lambda maps, **_kwargs: np.ones(len(maps), dtype=float),
    )
    worker_contract = {
        "labels": np.asarray([0, 0, 1, 1]),
        "ranks": ranks,
        "formal_clean": np.asarray([False, True, True, False]),
    }
    maps, labels = intervention._source_bundle(npz_path, worker_contract)
    np.testing.assert_array_equal(maps, source_maps[[1, 2]])
    np.testing.assert_array_equal(labels, [0, 1])
    assert intervention._event_contract(
        npz_path, worker_contract, 0,
    )["detected_event_index"] == 1
    assert intervention._event_contract(
        npz_path, worker_contract, 1,
    )["detected_event_index"] == 2
