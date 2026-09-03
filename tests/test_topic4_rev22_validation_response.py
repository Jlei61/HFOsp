from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fit_topic4_rev22_validation_response",
    ROOT / "scripts/fit_topic4_rev22_validation_response.py",
)
response = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(response)


def _write(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> dict[str, Path]:
    bounds = {
        "g_LEE": [0.0, 1.0], "g_LEI": [0.0, 1.5],
        "theta_FT_deg": [-10.0, 10.0], "AR_FT": [1.0, 3.0],
    }
    candidates = []
    validation_rows = []
    for index in range(96):
        u = np.asarray([
            (index % 8) / 7, ((index // 8) % 4) / 3,
            ((index // 32) % 3) / 2, ((index * 5) % 11) / 10,
        ])
        x = np.asarray([bounds[name][0] + u[d] * (bounds[name][1] - bounds[name][0])
                        for d, name in enumerate(response.PARAMETER_ORDER)])
        candidate_id = f"dci_p{index:03d}"
        candidates.append({
            "candidate_id": candidate_id,
            "physical": dict(zip(response.PARAMETER_ORDER, x.tolist())),
        })
        endpoint_values = {
            "D_support": float(0.2 + np.sum((u - 0.3) ** 2)),
            "D_order": float(0.3 + np.sum((u - 0.5) ** 2)),
            "D_time_ms": float(20 + 5 * u[2] + 2 * u[3]),
            "recall": float(0.8 - 0.1 * np.sum((u - 0.4) ** 2)),
            "kmeans_alignment": float(0.7 - 0.1 * np.sum((u - 0.6) ** 2)),
            "ood": float(0.1 + 0.1 * np.sum((u - 0.5) ** 2)),
        }
        unit_rows = []
        for unit in range(4):
            unit_rows.append({
                "topology_seed": 2501 + unit,
                **{name: value + (unit - 1.5) * 0.002 for name, value in endpoint_values.items()},
            })
        validation_rows.append({
            "candidate_id": candidate_id, "primary_status": "OK",
            "primary_endpoints": endpoint_values, "unit_endpoints": unit_rows,
            "secondary": {"yield_total": 80 + index},
        })

    design_path = tmp_path / "response_design_manifest.json"
    design_hash = _write(design_path, {
        "schema_id": response.DESIGN_SCHEMA, "candidate_count": 96,
        "bounds": bounds, "candidates": candidates,
    })
    frozen_path = tmp_path / "frozen_candidates.json"
    frozen_hash = _write(frozen_path, {
        "schema_id": response.FROZEN_SCHEMA, "branch": "PRIMARY_4D_BRANCH",
        "candidate_ids": ["dci_p010"], "mask_to_candidates": {"M1111": ["dci_p010"]},
        "response_design_manifest_sha256": design_hash,
    })
    validation_path = tmp_path / "validation_aggregate.json"
    _write(validation_path, {
        "schema_id": response.VALIDATION_SCHEMA, "status": "VALIDATION_AGGREGATE_COMPLETE",
        "snn_simulation_run": False,
        "input_hashes": {"response_design": design_hash, "frozen_candidates": frozen_hash},
        "fit_descriptive": {"candidate_count": 96, "candidates": validation_rows},
    })
    return {
        "validation": validation_path, "design": design_path, "frozen": frozen_path,
        "output": tmp_path / "output/validation_response_surface.json",
    }


def _run(paths: dict[str, Path]):
    return response.fit_validation_response(
        validation_path=paths["validation"], design_path=paths["design"],
        frozen_path=paths["frozen"], output_path=paths["output"],
        n_restarts=0, cv_folds=4, slice_points=7, plane_points=5,
    )


def test_descriptive_surfaces_preserve_96_points_and_emit_no_proposal(tmp_path):
    paths = _fixture(tmp_path)
    payload = _run(paths)
    assert payload["schema_id"] == response.OUTPUT_SCHEMA
    assert payload["descriptive_only"] is True
    assert payload["cannot_select"] is True
    assert payload["selection_permitted"] is False
    assert payload["task8_freeze_modified"] is False
    assert payload["design_point_count"] == 96
    assert len(payload["original_design_points"]) == 96
    assert list(payload["surfaces"]) == list(response.ENDPOINTS)
    for endpoint in response.ENDPOINTS:
        record = payload["surfaces"][endpoint]
        assert record["status"] == "OK"
        assert record["cv"]["consequence"] == "diagnostic_only_no_selection_or_optimization"
        assert len(record["conditional_slices"]["g_LEE"]["axis"]) == 7
        assert np.asarray(record["conditional_planes"]["theta_x_AR"]["mean"]).shape == (5, 5)
    serialized = json.dumps(payload).lower()
    assert '"proposal' not in serialized
    assert paths["output"].is_file()
    assert list(paths["output"].parent.iterdir()) == [paths["output"]]


@pytest.mark.parametrize("binding", ["response_design", "frozen_candidates"])
def test_validation_hash_binding_fails_closed(tmp_path, binding):
    paths = _fixture(tmp_path)
    payload = json.loads(paths["validation"].read_text())
    payload["input_hashes"][binding] = "0" * 64
    _write(paths["validation"], payload)
    with pytest.raises(RuntimeError, match="broken frozen hash binding"):
        _run(paths)
    assert not paths["output"].exists()


def test_frozen_to_design_hash_binding_fails_closed(tmp_path):
    paths = _fixture(tmp_path)
    frozen = json.loads(paths["frozen"].read_text())
    frozen["response_design_manifest_sha256"] = "f" * 64
    new_hash = _write(paths["frozen"], frozen)
    validation = json.loads(paths["validation"].read_text())
    validation["input_hashes"]["frozen_candidates"] = new_hash
    _write(paths["validation"], validation)
    with pytest.raises(RuntimeError, match="frozen candidates -> response design"):
        _run(paths)
    assert not paths["output"].exists()


def test_incomplete_fit_descriptive_grid_fails_closed(tmp_path):
    paths = _fixture(tmp_path)
    payload = json.loads(paths["validation"].read_text())
    payload["fit_descriptive"]["candidates"].pop()
    _write(paths["validation"], payload)
    with pytest.raises(RuntimeError, match="all 96 fit_descriptive"):
        _run(paths)
    assert not paths["output"].exists()


def test_not_estimable_point_is_retained_but_excluded_from_surface(tmp_path):
    paths = _fixture(tmp_path)
    payload = json.loads(paths["validation"].read_text())
    row = payload["fit_descriptive"]["candidates"][3]
    row["primary_status"] = "PRIMARY_ENDPOINT_NOT_ESTIMABLE"
    row["primary_endpoints"] = {name: None for name in response.ENDPOINTS}
    _write(paths["validation"], payload)
    result = _run(paths)
    point = result["original_design_points"][3]
    assert point["candidate_id"] == "dci_p003"
    assert all(point["not_estimable"].values())
    assert result["surfaces"]["D_support"]["n_estimable"] == 95
