import json
from pathlib import Path

import numpy as np

from scripts.freeze_topic4_rev10_d5_spatial_ou_candidates import candidate_library

ROOT = Path(__file__).resolve().parents[1]


def _config():
    return json.loads((
        ROOT / "config/topic4_rev10_d5_spatial_ou_accessibility_canary.json"
    ).read_text())


def test_d5_library_is_paired_local_permuted_and_exact_edge_noop():
    config = _config()
    rows = candidate_library(config)
    assert len(rows) == 9
    assert {row["spatial_ou"]["mode"] for row in rows} == {
        "off", "local", "permuted",
    }
    assert all(np.array_equal(row["coefficients"], np.zeros(12)) for row in rows)
    local = {
        (row["spatial_ou"]["sigma_rate_per_ms"], row["spatial_ou"]["ell_mm"])
        for row in rows if row["spatial_ou"]["mode"] == "local"
    }
    permuted = {
        (row["spatial_ou"]["sigma_rate_per_ms"], row["spatial_ou"]["ell_mm"])
        for row in rows if row["spatial_ou"]["mode"] == "permuted"
    }
    assert local == permuted == {(0.5, 0.38), (0.5, 0.76), (1.0, 0.38), (1.0, 0.76)}


def test_d5_forbids_observation_conditioned_spatial_inputs():
    forbidden = set(_config()["spatial_ou_library"]["forbidden_inputs"])
    assert "contact coordinates" in forbidden
    assert "D4 source coordinates" in forbidden
    assert "Gaussian components or peaks" in forbidden
    assert _config()["execution"]["wait_seconds"] >= 180
