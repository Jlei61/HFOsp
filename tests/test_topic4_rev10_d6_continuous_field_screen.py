import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from scripts.audit_topic4_rev10_d6_continuous_field_kmeans_screen import d6_score
from scripts.freeze_topic4_rev10_d6_continuous_field_kmeans_screen import (
    candidate_library,
    projected_uniform_residuals,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/topic4_rev10_d6_continuous_field_kmeans_screen.json"


def _config():
    return json.loads(CONFIG.read_text())


def _anchor():
    config = _config()
    manifest = json.loads(
        (ROOT / config["inputs"]["node_anchor_manifest"]["path"]).read_text()
    )
    return next(
        row for row in manifest["candidate_set"]["candidates"]
        if row["candidate_id"] == config["node_anchor"]["candidate_id"]
    )


def test_d6_builder_is_uniform_sheet_and_has_no_observation_arguments():
    config = _config()
    residuals = projected_uniform_residuals(config)
    assert residuals["coefficients"].shape == (12, 18 * 18)
    assert max(residuals["projection_rmse"]) < 0.02
    assert not ({"contact_xy", "shaft_ids", "patient", "labels"}
                & set(inspect.signature(projected_uniform_residuals).parameters))


def test_d6_library_is_continuous_no_k_and_exact_edge_noop():
    config = _config()
    rows, _ = candidate_library(config, _anchor())
    assert len(rows) == 49
    assert all(row["node_field"]["field_type"] == "spline_continuous" for row in rows)
    assert all(row["node_field"]["component_count"] is None for row in rows)
    assert all(row["node_field"]["peak_count_constraint"] is None for row in rows)
    assert all(np.asarray(row["node_field"]["coefficients"]).shape == (18, 18)
               for row in rows)
    assert all(np.allclose(row["coefficients"], 0.0) for row in rows)
    assert {row["spatial_ou"]["mode"] for row in rows} == {"local"}
    assert len({row["node_field"]["field_sha256"] for row in rows}) == 49


def test_d6_score_rewards_purity_margin_and_recruitment():
    base = {
        "evaluable": True,
        "balanced_kmeans": {"purity_median": 0.7},
        "signed_patient_margin": 0.5,
        "activity": {
            "mean_network_ood_fraction": 0.4,
            "mean_network_fraction_time_above_detector": 0.2,
        },
    }
    recruitment = {"worst_mode_error": 0.3}
    score = d6_score(base, recruitment)
    improved = dict(base)
    improved["balanced_kmeans"] = {"purity_median": 0.8}
    assert d6_score(improved, recruitment) < score
    assert d6_score(base, {"worst_mode_error": 0.1}) < score


def test_d6_entrypoints_are_directly_executable():
    for script in (
        "freeze_topic4_rev10_d6_continuous_field_kmeans_screen.py",
        "audit_topic4_rev10_d6_continuous_field_kmeans_screen.py",
    ):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), "--help"],
            cwd=ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
