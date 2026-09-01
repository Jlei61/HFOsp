import hashlib
import json
import math
from pathlib import Path

import numpy as np

from scripts.freeze_topic4_rev10_r_edge_candidates import build_candidates
from scripts.launch_topic4_rev10_r_graph_basis import NUMERIC_ENV
from src.topic4_graph_edge_flow import spectral_response_design


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/topic4_rev10_r_graph_edge_flow.json"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_frozen_input_hashes_and_scientific_boundary():
    config = json.loads(CONFIG.read_text())
    assert config["scientific_role"] == (
        "development_only_contact_density_invariant_route_capacity"
    )
    assert config["search"]["beta"] == "closed"
    assert config["node_anchor"]["candidate_id"] == "v62_density_t050"
    assert config["node_anchor"]["field_sha256"].startswith("5fc404a876713430")
    assert config["search"]["fit_network_seeds"] == [1051, 1052, 1053, 1054]
    assert config["direction_classifier"]["feature_semantics"] == (
        "FULL_TIMING_shaft_aware"
    )
    for record in config["inputs"].values():
        assert _sha256(ROOT / record["path"]) == record["sha256"]


def test_candidate_library_is_antithetic_and_exactly_amplitude_bounded():
    config = json.loads(CONFIG.read_text())
    feature_abs_max = np.asarray([
        [1.0, 2.0, 1.5, 0.5],
        [1.2, 1.8, 1.4, 0.7],
        [0.9, 2.1, 1.2, 0.6],
        [1.1, 1.9, 1.6, 0.8],
    ])
    candidates, _ = build_candidates(config, feature_abs_max)
    assert len(candidates) == 33
    assert np.array_equal(candidates[0]["coefficients"], np.zeros(4))
    by_id = {row["candidate_id"]: row for row in candidates}
    bound = config["candidate_library"]["raw_logit_abs_bound"]
    for row in candidates[1:]:
        coefficients = np.asarray(row["coefficients"])
        pair = np.asarray(by_id[row["antithetic_pair"]]["coefficients"])
        assert np.allclose(coefficients, -pair, rtol=0.0, atol=1e-15)
        exact_upper_bound = np.max(feature_abs_max @ np.abs(coefficients))
        assert exact_upper_bound <= bound + 1e-14
        assert math.isclose(exact_upper_bound, bound, rel_tol=0.0, abs_tol=1e-14)
        assert np.allclose(
            row["edge_ratio_guarantee"],
            [math.exp(-2.0 * bound), math.exp(2.0 * bound)],
            rtol=0.0, atol=1e-15,
        )


def test_spectral_design_exposes_near_degenerate_effective_capacity():
    separated = spectral_response_design(np.asarray([0.9, 0.7, 0.4, 0.2]), 4)
    near_degenerate = spectral_response_design(
        np.asarray([0.9, 0.8999, 0.8998, 0.8997]), 4,
    )
    assert np.linalg.cond(separated) < 100.0
    assert np.linalg.cond(near_degenerate) > 1e8


def test_basis_builder_source_has_no_observation_loader():
    source = (ROOT / "scripts/build_topic4_rev10_r_graph_basis.py").read_text()
    forbidden = (
        "VirtualMontage", "patient_train_onsets", "assign_direction_modes",
        "contact_xy", "shaft_ids", "continuous_field_h",
    )
    assert not any(token in source for token in forbidden)


def test_basis_launcher_is_bounded_and_numeric_single_thread():
    source = (
        ROOT / "scripts/launch_topic4_rev10_r_graph_basis.py"
    ).read_text()
    assert NUMERIC_ENV
    assert set(NUMERIC_ENV.values()) == {"1"}
    assert "systemd-run" in source
    assert '"/usr/bin/nohup"' in source
    assert '"--property=MemoryMax=24G"' in source
    assert "time.sleep(wait_seconds)" in source
