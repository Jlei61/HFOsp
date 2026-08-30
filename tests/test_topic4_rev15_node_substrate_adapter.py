from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import topic4_rev15_node_substrate_adapter as adapter


def test_projection_parity_requires_all_three_arrays(tmp_path):
    path = tmp_path / "worker.npz"
    projection = {
        "h": np.asarray([0.1, 0.2]),
        "vtheta": np.asarray([-50.0, -49.0, -48.0]),
        "delta_vtheta": np.asarray([-1.0, 0.0]),
        "hashes": {"projection_sha256": "p" * 64},
    }
    np.savez(
        path, h=projection["h"], vtheta=projection["vtheta"],
        delta_vtheta=projection["delta_vtheta"],
    )
    audit = adapter.verify_projection_against_worker(projection, path)
    assert all(audit["exact_array_parity"].values())

    np.savez(
        path, h=projection["h"] + 1e-12, vtheta=projection["vtheta"],
        delta_vtheta=projection["delta_vtheta"],
    )
    with pytest.raises(RuntimeError, match="differs"):
        adapter.verify_projection_against_worker(projection, path)


def test_build_projected_substrate_applies_projection_and_keeps_edges_off(
    tmp_path, monkeypatch,
):
    artifact = tmp_path / "artifact"
    manifest_path = artifact / "results/manifest.json"
    manifest_path.parent.mkdir(parents=True)
    config_path = tmp_path / "config.json"
    config = {
        "candidate_manifest": "results/manifest.json",
        "search": {"active_network_seeds": [2341]},
        "node_mapping": {"signed_depth_contract": {"sha256": "d" * 64}},
    }
    config_path.write_text(json.dumps(config))
    manifest_path.write_text(json.dumps({
        "status": "FROZEN", "schema_id": "SCHEMA",
        "config_sha256": adapter._sha256(config_path),
        "candidates": [{"candidate_id": "field", "selection_eligible": True}],
    }))
    monkeypatch.setattr(adapter.robust_freezer, "STATUS", "FROZEN")
    monkeypatch.setattr(adapter.robust_freezer, "MANIFEST_SCHEMA", "SCHEMA")
    monkeypatch.setattr(adapter.robust_freezer, "_validate_config", lambda value: None)
    transition_path = artifact / "transition.json"
    transition_path.write_text("{}")
    monkeypatch.setattr(
        adapter.m3_worker, "_compatibility_config",
        lambda *args, **kwargs: ({
            "inputs": {"transition_config": {
                "path": "transition.json",
                "sha256": adapter._sha256(transition_path),
            }}
        }, {"ok": True}),
    )
    monkeypatch.setattr(
        adapter.m3_worker, "_compatibility_manifest",
        lambda *args, **kwargs: {"candidates": []},
    )
    substrate = SimpleNamespace(
        edge_coefficients=np.zeros(2), extras={}, h_e=np.zeros(2),
        vtheta=np.zeros(3), delta_vtheta=np.zeros(2),
    )
    monkeypatch.setattr(
        adapter.m3_worker, "_reconstruct_static_node_substrate",
        lambda *args, **kwargs: substrate,
    )
    projection = {
        "h": np.asarray([0.2, 0.3]),
        "vtheta": np.asarray([-50.0, -49.0, -48.0]),
        "delta_vtheta": np.asarray([-1.0, 0.0]),
        "signed_depth": np.asarray([1.0, 1.0]),
        "audit": {"mapping": "test"},
        "hashes": {},
    }
    projection["hashes"].update({
        "h_sha256": adapter.m3_worker.array_sha256(projection["h"]),
        "vtheta_sha256": adapter.m3_worker.array_sha256(projection["vtheta"]),
        "frozen_signed_depth_sha256": "d" * 64,
        "projection_sha256": "p" * 64,
    })
    monkeypatch.setattr(
        adapter.m3_worker, "_project_candidate",
        lambda *args, **kwargs: projection,
    )
    monkeypatch.setattr(
        adapter.robust_freezer, "_resolve",
        lambda root, relative: root / relative,
    )
    monkeypatch.setattr(adapter, "load_round_config", lambda path: {"loaded": str(path)})
    observed, observed_projection, transition = adapter.build_projected_node_substrate(
        robust_config_path=config_path, candidate_id="field", seed=2341,
        artifact_root=artifact,
    )
    assert np.array_equal(observed.h_e, projection["h"])
    assert np.array_equal(observed.vtheta, projection["vtheta"])
    assert observed.extras["rev15_m3_projection"] == {"mapping": "test"}
    assert observed_projection is projection
    assert transition["loaded"].endswith("transition.json")
