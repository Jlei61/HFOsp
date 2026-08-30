from __future__ import annotations

from types import SimpleNamespace

from scripts import freeze_topic4_rev16_joint_m3_m4_candidates as freezer
from scripts import run_topic4_rev14_m3_canary_worker as shared_worker
from scripts import topic4_rev15_node_substrate_adapter as rev15
from scripts import topic4_rev16_node_substrate_adapter as adapter


def test_rev16_adapter_switches_freezer_only_within_reconstruction(monkeypatch, tmp_path):
    seen = {}

    def fake_build(**kwargs):
        seen["base_freezer"] = rev15.robust_freezer
        seen["worker_freezer"] = shared_worker.freezer
        seen["pathways"] = shared_worker.EXPECTED_PATHWAYS
        substrate = SimpleNamespace(extras={})
        return substrate, {"audit": {"field_kind": "joint"}}, {"x": 1}

    old_base = rev15.robust_freezer
    old_worker = shared_worker.freezer
    old_pathways = shared_worker.EXPECTED_PATHWAYS
    monkeypatch.setattr(rev15, "build_projected_node_substrate", fake_build)
    substrate, projection, transition = adapter.build_projected_node_substrate(
        robust_config_path=tmp_path / "config.json",
        candidate_id="joint", seed=2351, artifact_root=tmp_path,
    )
    assert seen["base_freezer"] is freezer
    assert seen["worker_freezer"] is freezer
    assert seen["pathways"] == freezer.EXPECTED_PATHWAYS
    assert substrate.extras["rev16_joint_m3_m4_projection"] == projection["audit"]
    assert transition == {"x": 1}
    assert rev15.robust_freezer is old_base
    assert shared_worker.freezer is old_worker
    assert shared_worker.EXPECTED_PATHWAYS == old_pathways
