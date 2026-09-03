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
        "inventory": {"candidate_count": 4, "joint_feasibility_count": 3},
        "candidates": [],
    }
    physicals = (
        ("full", [0.7, 0.8, 5.0, 2.3], True),
        ("locked", [0.5, 0.8, 5.0, 2.3], True),
        ("screen_a", [0.2, 0.4, -5.0, 1.5], True),
        ("screen_bad", [0.9, 1.2, 8.0, 2.8], False),
    )
    for index, (candidate_id, values, feasible) in enumerate(physicals):
        fit_payload["candidates"].append({
            "candidate_id": candidate_id,
            "physical": dict(zip(plotter.PARAMS, values)),
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
    _write(validation_path, {
        "schema_id": "topic4_rev22_dci_validation_aggregate_v1",
        "status": "VALIDATION_AGGREGATE_COMPLETE",
        "snn_simulation_run": False,
        "patient_ictal_input_read": False,
        "input_hashes": {"frozen_candidates": frozen_hash},
        "phases": {"qualification": rows, "confirmation": rows},
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
    return {"validation": validation_path, "fit": fit_path, "response": response_path,
            "frozen": frozen_path}


def _load(paths: dict[str, Path]) -> dict:
    return plotter.load_inputs(paths["validation"], paths["fit"], paths["response"],
                               paths["frozen"])


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
        plotter.load_inputs(fake, paths["fit"], paths["response"], paths["frozen"])
    npz = tmp_path / "validation_aggregate.npz"
    npz.write_bytes(b"not an npz")
    with pytest.raises(RuntimeError, match="frozen JSON/CSV"):
        plotter.load_inputs(npz, paths["fit"], paths["response"], paths["frozen"])


def test_output_directory_must_not_preexist(tmp_path):
    paths = _fixture(tmp_path)
    out = tmp_path / "figures"
    out.mkdir()
    with pytest.raises(FileExistsError):
        plotter.render(_load(paths), out)
