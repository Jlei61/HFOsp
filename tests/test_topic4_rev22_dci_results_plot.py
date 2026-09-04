from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/plot_topic4_rev22_dci_results.py"
SPEC = importlib.util.spec_from_file_location("plot_topic4_rev22_dci_results", SCRIPT)
plotter = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(plotter)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: dict) -> str:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return _sha(path)


def _validation_row(candidate_id: str, shift: float) -> dict:
    endpoints = {
        "D_support": 0.35 + shift,
        "D_order": 0.25 + shift,
        "D_time_ms": 14.0 + shift,
        "recall": 0.72 - shift,
        "kmeans_alignment": 0.78 - shift,
        "ood": 0.18 + shift,
    }
    return {
        "candidate_id": candidate_id,
        "family_membership": ["M1111" if candidate_id == "full" else "M0111"],
        "primary_status": "OK",
        "primary_endpoints": endpoints,
        "secondary": {
            "D_cloud_composite": 0.4 + shift,
            "yield_total": 96 if candidate_id == "full" else 72,
            "c2st": {"separability": 0.28 + shift, "auc": 0.64 + shift / 2},
        },
    }


def _fixture(tmp_path: Path) -> dict[str, Path]:
    fit_path = tmp_path / "fit_aggregate.json"
    fit_payload = {
        "schema_id": "topic4_rev22_dci_training_only_fit_aggregate_v2",
        "status": "FIT_AGGREGATE_COMPLETE",
        "inventory": {"candidate_count": 96, "joint_feasibility_count": 95},
        "candidates": [],
    }
    physicals = []
    for index in range(96):
        candidate_id = "full" if index == 0 else "locked" if index == 1 else f"screen_{index:03d}"
        if index < 8:
            values = [(index % 4) / 3, (index // 4) * 1.5, 0.0, 2.0]
            block = "dose_plane"
        elif index < 16:
            local = index - 8
            values = [0.5, 1.0, -10.0 + (local % 4) * (20.0 / 3), 1.0 + (local // 4) * 2.0]
            block = "geometry_plane"
        else:
            values = [(index % 11) / 10, ((index * 3) % 13) / 8,
                      -10.0 + ((index * 5) % 17) * 1.25, 1.0 + ((index * 7) % 19) / 9]
            block = "full4d"
        physicals.append((candidate_id, values, index != 95, block))
    for index, (candidate_id, values, feasible, block) in enumerate(physicals):
        fit_payload["candidates"].append({
            "candidate_id": candidate_id,
            "physical": dict(zip(plotter.PARAMS, values)),
            "block": block,
            "joint_feasibility": feasible,
            "continuous_surface_eligible": feasible,
            "standardized_Z": {
                "D_support": 0.2 + index * 0.1,
                "D_order": 0.3 + index * 0.1,
                "D_lag": 0.4 + index * 0.1,
                "D_cover": 0.5 + index * 0.1,
            } if feasible else {},
            "candidate_failure_reasons": [] if feasible else ["LOW_YIELD"],
        })
    fit_hash = _write(fit_path, fit_payload)

    response_path = tmp_path / "response_fit.json"
    response_hash = _write(response_path, {
        "schema_id": "topic4_rev22_dci_response_fit_v1",
        "status": "RESPONSE_FIT_COMPLETE",
        "training_only": True,
        "branch": "PRIMARY_4D_BRANCH",
        "input_hashes": {"fit_aggregate": {"sha256": fit_hash}},
        "proposals": {
            "M1111": {"frozen_points": [{"execution_candidate_id": "full",
                                            "x": physicals[0][1]}]},
            "M0111": {"frozen_points": [{"execution_candidate_id": "locked",
                                            "x": physicals[1][1]}]},
        },
    })
    frozen_path = tmp_path / "frozen_candidates.json"
    frozen_hash = _write(frozen_path, {
        "schema_id": "topic4_rev22_dci_frozen_candidates_v1",
        "status": "FROZEN",
        "candidate_ids": ["full", "locked"],
        "mask_to_candidates": {"M1111": ["full"], "M0111": ["locked"]},
        "input_hashes": {
            "fit_aggregate": {"sha256": fit_hash},
            "response_fit": {"sha256": response_hash},
        },
    })
    interval = {
        name: {"delta": 0.08, "lo": 0.01, "hi": 0.14, "positive_is_better": True}
        for name, _, _ in plotter.PRIMARY
    }
    validation_path = tmp_path / "validation_aggregate.json"
    rows = [_validation_row("full", 0.0), _validation_row("locked", 0.08)]
    descriptive_rows = []
    original_points = []
    for index, (candidate_id, values, feasible, _) in enumerate(physicals):
        shift = index / 1000
        row = _validation_row(candidate_id, shift)
        row["physical"] = dict(zip(plotter.PARAMS, values))
        if not feasible:
            row["primary_status"] = "PRIMARY_ENDPOINT_NOT_ESTIMABLE"
            row["primary_endpoints"] = {name: None for name, _, _ in plotter.PRIMARY}
        descriptive_rows.append(row)
        original_points.append({
            "candidate_id": candidate_id,
            "physical": dict(zip(plotter.PARAMS, values)),
            "primary_status": row["primary_status"],
            "endpoints": row["primary_endpoints"],
            "yield_total": row["secondary"]["yield_total"],
            "not_estimable": {name: not feasible for name, _, _ in plotter.PRIMARY},
        })
    validation_hash = _write(validation_path, {
        "schema_id": "topic4_rev22_dci_validation_aggregate_v1",
        "status": "VALIDATION_AGGREGATE_COMPLETE",
        "snn_simulation_run": False,
        "patient_ictal_input_read": False,
        "input_hashes": {"frozen_candidates": frozen_hash},
        "phases": {"qualification": rows, "confirmation": rows},
        "fit_descriptive": {
            "descriptive_only": True, "cannot_select": True,
            "candidate_count": 96, "candidates": descriptive_rows,
        },
        "paired_pareto_contrasts": [{
            "full_candidate_id": "full", "locked_candidate_id": "locked",
            "status": "PARETO_SUPPORTED", "endpoints": interval,
        }],
        "paired_reference_contrasts": [],
        "unit_inventory": [
            {"phase": phase, "candidate_id": candidate_id, "failure_reasons": []}
            for phase in ("qualification", "confirmation")
            for candidate_id in ("full", "locked")
        ],
    })
    surface_path = tmp_path / "validation_response_surface.json"
    surfaces = {}
    for endpoint_index, (name, _, direction) in enumerate(plotter.PRIMARY):
        slices = {}
        for parameter_index, parameter in enumerate(plotter.PARAMS):
            bounds = ((0.0, 1.0), (0.0, 1.5), (-10.0, 10.0), (1.0, 3.0))[parameter_index]
            axis = [bounds[0] + (bounds[1] - bounds[0]) * step / 4 for step in range(5)]
            mean = [0.2 + endpoint_index * 0.05 + parameter_index * 0.01 + value * 0.1
                    for value in axis]
            slices[parameter] = {
                "axis": axis, "mean": mean,
                "lo90": [value - 0.02 for value in mean],
                "hi90": [value + 0.02 for value in mean],
            }
        surfaces[name] = {
            "status": "OK", "direction": "higher_is_better",
            "display_quantity": "fraction_of_M0000_to_benchmark_gap_closed",
            "cv": {"adequate": True}, "conditional_slices": slices,
        }
    for point_index, point in enumerate(original_points):
        point["display_score"] = {
            name: 0.05 * point_index / 95 + 0.01 * endpoint_index
            for endpoint_index, (name, _, _) in enumerate(plotter.PRIMARY)
        }
    _write(surface_path, {
        "schema_id": "topic4_rev22_dci_validation_response_surface_v1",
        "status": "DESCRIPTIVE_VALIDATION_RESPONSE_COMPLETE",
        "descriptive_only": True, "cannot_select": True, "selection_permitted": False,
        "design_point_count": 96, "original_design_points": original_points,
        "display_contract": {
            "quantity": "fraction_of_M0000_to_benchmark_gap_closed",
            "reference_score": 0.0, "benchmark_score": 1.0,
            "reference_candidate_id": "dci_p000",
        },
        "surfaces": surfaces,
        "input_hashes": {"validation_aggregate": {"sha256": validation_hash},
                         "frozen_candidates": {"sha256": frozen_hash}},
    })
    return {"validation": validation_path, "validation_surface": surface_path,
            "fit": fit_path, "response": response_path, "frozen": frozen_path}


def _load(paths: dict[str, Path]) -> dict:
    return plotter.load_inputs(paths["validation"], paths["validation_surface"],
                               paths["fit"], paths["response"], paths["frozen"])


def test_synthetic_fixture_renders_all_formats_readme_metadata_and_failure_sidecar(tmp_path):
    paths = _fixture(tmp_path)
    out = tmp_path / "figures"
    metadata = plotter.render(_load(paths), out)

    stems = (
        "rev22_dci_validation_response_atlas",
        "rev22_dci_nested_family_matrix",
        "rev22_dci_validation_pareto",
        "rev22_dci_training_response_atlas",
        "rev22_dci_failure_feasibility",
    )
    for stem in stems:
        for suffix in plotter.FORMATS:
            assert (out / f"{stem}.{suffix}").stat().st_size > 100
    assert (out / "README.md").is_file()
    assert "held-out order" in (out / "rev22_dci_validation_pareto.svg").read_text().lower()
    assert "cannot-select" in (out / "README.md").read_text()
    sidecar = json.loads((out / "rev22_dci_failure_feasibility.json").read_text())
    assert sidecar["fit_candidates"]["LOW_YIELD"] == 1
    assert sidecar["validation_units"]["confirmation"]["eligible"] == 2
    assert metadata["snn_simulation_run"] is False
    assert metadata["patient_ictal_input_read"] is False
    assert metadata["statistical_recomputation"] is False
    assert len(metadata["output_sha256"]) == 17


def test_broken_hash_chain_fails_before_output_creation(tmp_path):
    paths = _fixture(tmp_path)
    response = json.loads(paths["response"].read_text())
    response["input_hashes"]["fit_aggregate"]["sha256"] = "0" * 64
    _write(paths["response"], response)
    out = tmp_path / "figures"
    with pytest.raises(RuntimeError, match="hash binding"):
        plotter.render(_load(paths), out)
    assert not out.exists()


def test_missing_physical_mapping_fails_before_output_creation(tmp_path):
    paths = _fixture(tmp_path)
    data = _load(paths)
    data["fit"]["candidates"] = [row for row in data["fit"]["candidates"]
                                     if row["candidate_id"] != "locked"]
    data["response"]["proposals"]["M0111"]["frozen_points"] = []
    out = tmp_path / "figures"
    with pytest.raises(RuntimeError, match="lack physical parameters"):
        plotter.render(data, out)
    assert not out.exists()


def test_rejects_nonaggregate_npz_and_ictal_named_paths(tmp_path):
    paths = _fixture(tmp_path)
    fake = tmp_path / "patient_ictal_summary.json"
    fake.write_text(paths["validation"].read_text())
    with pytest.raises(RuntimeError, match="forbidden non-aggregate input"):
        plotter.load_inputs(fake, paths["validation_surface"], paths["fit"],
                            paths["response"], paths["frozen"])
    npz = tmp_path / "validation_aggregate.npz"
    npz.write_bytes(b"not an npz")
    with pytest.raises(RuntimeError, match="frozen JSON/CSV"):
        plotter.load_inputs(npz, paths["validation_surface"], paths["fit"],
                            paths["response"], paths["frozen"])


def test_fit_descriptive_cannot_select_is_required(tmp_path):
    paths = _fixture(tmp_path)
    validation = json.loads(paths["validation"].read_text())
    validation["fit_descriptive"]["cannot_select"] = False
    new_hash = _write(paths["validation"], validation)
    surface = json.loads(paths["validation_surface"].read_text())
    surface["input_hashes"]["validation_aggregate"]["sha256"] = new_hash
    _write(paths["validation_surface"], surface)
    with pytest.raises(RuntimeError, match="descriptive-only/cannot-select"):
        _load(paths)


def test_validation_surface_hash_binding_fails_closed(tmp_path):
    paths = _fixture(tmp_path)
    surface = json.loads(paths["validation_surface"].read_text())
    surface["input_hashes"]["validation_aggregate"]["sha256"] = "f" * 64
    _write(paths["validation_surface"], surface)
    with pytest.raises(RuntimeError, match="surfaces -> validation aggregate"):
        _load(paths)


def test_output_directory_must_not_preexist(tmp_path):
    paths = _fixture(tmp_path)
    out = tmp_path / "figures"
    out.mkdir()
    with pytest.raises(FileExistsError):
        plotter.render(_load(paths), out)
