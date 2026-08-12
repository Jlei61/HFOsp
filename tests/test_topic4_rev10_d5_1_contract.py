import json
from pathlib import Path

import numpy as np

from scripts.freeze_topic4_rev10_d5_1_low_amplitude import candidate_library


ROOT = Path(__file__).resolve().parents[1]


def _config():
    return json.loads((
        ROOT / "config/topic4_rev10_d5_1_spatial_ou_low_amplitude.json"
    ).read_text())


def test_d5_1_is_a_low_amplitude_pairwise_bracket_not_a_core_library():
    config = _config()
    rows = candidate_library(config)
    assert len(rows) == 7
    assert all(np.array_equal(row["coefficients"], np.zeros(12)) for row in rows)
    local = {
        row["spatial_ou"]["sigma_rate_per_ms"]
        for row in rows if row["spatial_ou"]["mode"] == "local"
    }
    permuted = {
        row["spatial_ou"]["sigma_rate_per_ms"]
        for row in rows if row["spatial_ou"]["mode"] == "permuted"
    }
    assert local == permuted == {0.1, 0.2, 0.35}
    assert {
        row["spatial_ou"]["ell_mm"]
        for row in rows if row["spatial_ou"]["mode"] != "off"
    } == {0.38}
    assert config["search"]["phase"] == "fit"


def test_d5_1_forbids_observation_driven_placement_and_fresh_seed_use():
    config = _config()
    forbidden = set(config["spatial_ou_library"]["forbidden_inputs"])
    assert "contact coordinates" in forbidden
    assert "shaft identity" in forbidden
    assert "D4 source coordinates" in forbidden
    assert "Gaussian components or peaks" in forbidden
    assert config["claim_boundary"]["same_fit_networks_as_D5"] is True
    assert config["claim_boundary"]["amplitude_bracket_not_confirmation"] is True
    assert config["execution"]["wait_seconds"] >= 180
