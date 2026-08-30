from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts import freeze_topic4_rev16_m4_shell_coordinate_atlas as freezer
from scripts import monitor_topic4_rev16_m4_shell_coordinate_atlas as monitor
from scripts import run_topic4_rev14_m3_canary_worker as base_worker
from scripts import run_topic4_rev16_m4_shell_coordinate_worker as worker


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/topic4_rev16_m4_shell_coordinate_atlas.json"
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")


def test_config_is_node_only_three_network_shell_atlas():
    config = json.loads(CONFIG.read_text())
    freezer._validate_config(config)
    assert config["search"]["active_network_seeds"] == [2331, 2332, 2333]
    assert set(config["pathways"].values()) == {"off"}
    assert config["field_design"]["candidate_count"] == 40
    assert config["field_design"]["shell_coordinate_count"] == 20


def test_shell_candidates_have_exact_sign_pairs_and_zero_m3_coefficients():
    config = json.loads(CONFIG.read_text())
    candidates, audit = freezer.build_candidates(config)
    assert len(candidates) == 40
    assert audit["ordered_shell_coordinate_count"] == 20
    assert all(row["selection_eligible"] for row in candidates)
    for coordinate in range(20):
        rows = [
            row for row in candidates
            if row["coordinate_atlas"]["shell_coordinate_index"] == coordinate
        ]
        rows.sort(key=lambda row: row["fourier_coordinate"]["sign"])
        assert len(rows) == 2
        negative = np.asarray(rows[0]["fourier_coordinate"]["coefficients"])
        positive = np.asarray(rows[1]["fourier_coordinate"]["coefficients"])
        assert negative.shape == (24, 2)
        assert np.array_equal(negative, -positive)
        assert np.count_nonzero(positive[:14]) == 0
        assert np.count_nonzero(positive[14:]) == 1


def test_worker_generic_design_keeps_legacy_m3_compatibility():
    legacy = {"m3_design": {"maximum_order": 3}}
    modern = {"field_design": {"maximum_order": 4}}
    assert base_worker._field_design(legacy)["maximum_order"] == 3
    assert base_worker._field_design(modern)["maximum_order"] == 4


def test_physical_dose_table_uses_frozen_generic_candidate_counts(monkeypatch):
    manifest = {
        "candidates": [
            {"candidate_id": f"c{index}", "selection_eligible": True}
            for index in range(40)
        ],
    }
    config = json.loads(CONFIG.read_text())
    monkeypatch.setattr(base_worker, "_project_candidate", lambda *args: {})
    monkeypatch.setattr(
        base_worker, "_physical_dose_row",
        lambda candidate, projection: {
            "candidate_id": candidate["candidate_id"],
            "selection_eligible": candidate["selection_eligible"],
            "frozen_signed_depth_sha256": config["node_mapping"][
                "signed_depth_contract"
            ]["sha256"],
        },
    )
    payload = base_worker.build_physical_dose_table(
        manifest, object(), config, seed=2331,
    )
    assert payload["n_selectable"] == 40
    assert payload["n_benchmarks"] == 0


def test_prepare_worker_uses_generic_field_design(tmp_path):
    config = json.loads(CONFIG.read_text())
    manifest = freezer.build_manifest_payload(
        CONFIG, artifact_root=ARTIFACT_ROOT,
        provenance={"formal_ready": False}, status=freezer.PREPARE_STATUS,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    config["candidate_manifest"] = str(manifest_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    # Candidate lookup and projection preparation must traverse the rev16 wrapper.
    worker.configure_base()
    candidate = manifest["candidates"][0]
    assert candidate["field_kind"] == "absolute_paired_phase_fourier_m4_shell_coordinate"
    assert base_worker._field_design(config)["maximum_order"] == 4


def test_monitor_contract_requires_low_frequency_resource_control():
    config = json.loads(CONFIG.read_text())
    assert config["resources"]["monitor_interval_seconds"] == 600
    assert config["resources"]["numerical_threads_per_worker"] == 1
    assert config["resources"]["maximum_workers"] == 8
    assert monitor._validate_unit_prefix("codex-t4-r16-m4shell-test").startswith(
        monitor.DEFAULT_UNIT_PREFIX
    )
