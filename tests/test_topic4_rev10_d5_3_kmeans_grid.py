import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from scripts.audit_topic4_rev10_d5_3_spatial_ou_kmeans_grid import (
    _finite_matrix_list,
    adjudicate,
    continuous_selection_score,
    direction_purity,
)
from scripts.freeze_topic4_rev10_d5_3_spatial_ou_kmeans_grid import (
    candidate_library,
)
from scripts.paper_figures.plot_fig4_spatial_edge_flow_validation import (
    _returned_summary_filename,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/topic4_rev10_d5_3_spatial_ou_kmeans_grid.json"


def _config():
    return json.loads(CONFIG.read_text())


def _row(candidate, score, purity, margin=0.3, evaluable=True):
    return {
        "candidate_id": candidate,
        "evaluable": evaluable,
        "selection_score": score if evaluable else None,
        "balanced_kmeans": {
            "purity_median": purity,
        },
        "signed_patient_margin": margin,
    }


def test_d5_3_grid_is_continuous_ou_only_and_contains_prior_anchor():
    config = _config()
    rows = candidate_library(config)
    assert len(rows) == 13
    assert rows[0]["candidate_id"] == "edge_noop"
    assert all(np.allclose(row["coefficients"], 0.0) for row in rows)
    local = [row for row in rows if row["spatial_ou"]["mode"] == "local"]
    assert len(local) == 12
    assert {
        row["spatial_ou"]["sigma_rate_per_ms"] for row in local
    } == {0.025, 0.05, 0.075, 0.1}
    assert {row["spatial_ou"]["tau_ms"] for row in local} == {10.0, 20.0, 40.0}
    assert any(
        row["spatial_ou"]["sigma_rate_per_ms"] == 0.1
        and row["spatial_ou"]["tau_ms"] == 20.0
        for row in local
    )
    forbidden = set(config["spatial_ou_library"]["forbidden_inputs"])
    assert {"contact coordinates", "shaft identity", "node field values"} <= forbidden
    assert config["search"]["beta"] == "closed"
    assert config["search"]["network_topology"] == "frozen"


def test_direction_purity_is_label_swap_invariant():
    direction = np.array([0, 0, 1, 1, 1])
    labels = np.array([0, 0, 1, 1, 0])
    value, contingency = direction_purity(labels, direction)
    swapped, _ = direction_purity(1 - labels, direction)
    assert value == swapped == 0.8
    assert contingency.tolist() == [[2, 1], [0, 2]]


def test_selection_score_is_kmeans_dominant_but_keeps_patient_geometry():
    strong = continuous_selection_score(
        purity=0.9, signed_margin=0.5, ood=0.4, occupancy=0.2,
    )
    mixed = continuous_selection_score(
        purity=0.7, signed_margin=0.5, ood=0.1, occupancy=0.05,
    )
    wrong_geometry = continuous_selection_score(
        purity=0.9, signed_margin=-0.5, ood=0.4, occupancy=0.2,
    )
    assert strong < mixed
    assert strong < wrong_geometry


def test_adjudication_requires_improvement_and_positive_matrix_margin():
    verdict = adjudicate([
        _row("a", 0.3, 0.75, margin=0.2),
        _row("b", 0.4, 0.9, margin=0.3),
    ], anchor_purity=0.67, patient_q05=0.88)
    assert verdict["selected_candidate_id"] == "a"
    assert verdict["status"].endswith("SELECTED_FOR_FRESH_CONFIRMATION")

    verdict = adjudicate([
        _row("a", 0.3, 0.65, margin=0.2),
        _row("b", 0.4, 0.9, margin=0.3),
    ], anchor_purity=0.67, patient_q05=0.88)
    assert verdict["selected_candidate_id"] == "a"
    assert verdict["status"].endswith("DID_NOT_IMPROVE_FROZEN_ANCHOR")


def test_d5_3_auditor_is_directly_executable():
    completed = subprocess.run([
        sys.executable,
        str(ROOT / "scripts/audit_topic4_rev10_d5_3_spatial_ou_kmeans_grid.py"),
        "--help",
    ], cwd=ROOT, text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert "--config" in completed.stdout


def test_d5_3_loader_uses_canary_summary_contract():
    config = _config()
    assert _returned_summary_filename(config) == "canary_summary_returned_only.json"
    confirmation = {
        "scientific_role": "development_only_translation_invariant_spatial_ou_confirmation",
        "search": {"phase": "confirmation"},
    }
    assert _returned_summary_filename(confirmation) == (
        "confirmation_summary_returned_only.json"
    )


def test_non_evaluable_matrix_is_strict_json_compatible():
    matrix = _finite_matrix_list([[np.nan, np.inf], [-np.inf, 0.5]])
    assert matrix == [[None, None], [None, 0.5]]
    assert json.loads(json.dumps(matrix, allow_nan=False)) == matrix
