import hashlib
import json
from pathlib import Path

from scripts.launch_topic4_rev10_r2_spatial_edge_audit import NUMERIC_ENV
from scripts.freeze_topic4_rev10_r2_spatial_edge_candidates import (
    build_candidates,
    whitened_directions,
)

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/topic4_rev10_r2_spatial_edge_flow.json"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_r2_contract_is_observation_invariant_and_hash_frozen():
    config = json.loads(CONFIG.read_text())
    assert config["spatial_edge_basis"]["coefficient_count"] == 12
    assert config["spatial_edge_basis"]["minimum_effective_rank"] == 10
    assert config["search"]["beta"] == "closed"
    assert config["node_anchor"]["candidate_id"] == "v62_density_t050"
    for record in config["inputs"].values():
        assert _sha256(ROOT / record["path"]) == record["sha256"]
    source = (
        ROOT / "scripts/build_topic4_rev10_r2_spatial_edge_audit.py"
    ).read_text()
    for token in (
        "VirtualMontage", "patient_train_onsets", "shaft_ids",
        "contact_xy", "continuous_field_h",
    ):
        assert token not in source


def test_r2_audit_launcher_is_bounded_nohup_and_sparse_polling():
    source = (
        ROOT / "scripts/launch_topic4_rev10_r2_spatial_edge_audit.py"
    ).read_text()
    assert NUMERIC_ENV and set(NUMERIC_ENV.values()) == {"1"}
    assert '"/usr/bin/nohup"' in source
    assert '"--property=MemoryMax=24G"' in source
    assert "time.sleep(wait_seconds)" in source


def test_whitened_sobol_library_is_antithetic_and_bounded():
    config = json.loads(CONFIG.read_text())
    covariance = np.stack([
        np.diag(np.linspace(0.5 + seed * 0.01, 2.0 + seed * 0.01, 12))
        for seed in range(4)
    ])
    maxima = np.stack([
        np.linspace(1.0 + seed * 0.01, 2.0 + seed * 0.01, 12)
        for seed in range(4)
    ])
    directions, _, mean_covariance, eigenvalues = whitened_directions(
        config, covariance,
    )
    metric_norm = np.einsum(
        "ni,ij,nj->n", directions, mean_covariance, directions,
    )
    np.testing.assert_allclose(metric_norm, np.ones(len(directions)))
    assert np.all(eigenvalues > 0.0)
    candidates, *_ = build_candidates(config, maxima, covariance)
    assert len(candidates) == 33
    by_id = {row["candidate_id"]: row for row in candidates}
    bound = config["candidate_library"]["raw_logit_abs_bound"]
    for row in candidates[1:]:
        coefficients = np.asarray(row["coefficients"])
        opposite = np.asarray(by_id[row["antithetic_pair"]]["coefficients"])
        np.testing.assert_allclose(coefficients, -opposite, atol=1e-15, rtol=0.0)
        assert np.max(maxima @ np.abs(coefficients)) <= bound + 1e-14


def test_r21_library_uses_rms_dose_with_clipping():
    config = json.loads(
        (ROOT / "config/topic4_rev10_r2_1_spatial_edge_flow.json").read_text()
    )
    covariance = np.stack([np.diag(np.linspace(0.5, 2.0, 12))] * 4)
    maxima = np.stack([np.linspace(1.0, 2.0, 12)] * 4)
    second_moment = covariance + 0.05 * np.ones_like(covariance)
    candidates, *_ = build_candidates(
        config, maxima, covariance, second_moment,
    )
    target = config["candidate_library"]["target_unclipped_logit_rms"]
    clip = config["candidate_library"]["raw_logit_abs_bound"]
    for row in candidates[1:]:
        coefficients = np.asarray(row["coefficients"])
        rms = float(np.sqrt(
            coefficients @ second_moment.mean(axis=0) @ coefficients
        ))
        assert np.isclose(rms, target)
        assert row["raw_logit_clip"] == clip
