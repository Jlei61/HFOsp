from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts import freeze_topic4_rev15_m3_multinetwork_atlas as freezer


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/topic4_rev15_m3_multinetwork_atlas.json"
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")


def test_multinetwork_config_keeps_node_only_and_fresh_seeds():
    config = json.loads(CONFIG.read_text())
    freezer._validate_config(config)
    assert config["search"]["active_network_seeds"] == [2332, 2333]
    assert config["search"]["canary_network_seeds"] == [2332, 2333]
    assert set(config["pathways"].values()) == {"off"}
    assert config["m3_design"]["candidate_count"] == 58


def test_multinetwork_manifest_is_exact_candidate_copy():
    config = json.loads(CONFIG.read_text())
    source_path = ARTIFACT_ROOT / config["inputs"]["source_atlas_manifest"]["path"]
    source = json.loads(source_path.read_text())
    payload = freezer.build_manifest_payload(
        CONFIG, artifact_root=ARTIFACT_ROOT,
        provenance={"formal_ready": False}, status=freezer.PREPARE_STATUS,
    )
    assert payload["candidates"] == source["candidates"]
    assert payload["direction_audit"] == source["direction_audit"]
    assert payload["source_atlas"]["candidate_payload_exact_copy"] is True
    assert payload["search"]["active_network_seeds"] == [2332, 2333]
    selectable = [row for row in payload["candidates"] if row["selection_eligible"]]
    assert len(selectable) == 56
    hashes = [
        row["fourier_coordinate"]["coefficients_sha256"] for row in selectable
    ]
    assert len(hashes) == len(set(hashes))


def test_every_coordinate_keeps_exact_opposite_sign_pair():
    payload = freezer.build_manifest_payload(
        CONFIG, artifact_root=ARTIFACT_ROOT,
        provenance={"formal_ready": False}, status=freezer.PREPARE_STATUS,
    )
    selectable = [row for row in payload["candidates"] if row["selection_eligible"]]
    for coordinate in range(28):
        rows = [
            row for row in selectable
            if row["coordinate_atlas"]["coordinate_index"] == coordinate
        ]
        assert len(rows) == 2
        rows.sort(key=lambda row: row["fourier_coordinate"]["sign"])
        negative = np.asarray(rows[0]["fourier_coordinate"]["coefficients"])
        positive = np.asarray(rows[1]["fourier_coordinate"]["coefficients"])
        assert np.array_equal(negative, -positive)
