#!/usr/bin/env python3
"""Freeze response-surface Node proposals and a rigid dual-core capacity control."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
sys.path.insert(0, str(ROOT))

from src.topic4_continuous_field import (  # noqa: E402
    continuous_surface,
    tensor_basis,
)
from src.topic4_core_field import axis_coords, two_core_q  # noqa: E402
from src.topic4_node_field_search import (  # noqa: E402
    normalize_surface_residual,
    residual_candidate,
    uniform_sheet_grid,
)
from src.topic4_observation_invariant_spline import (  # noqa: E402
    array_sha256,
    spline_roughness,
)
from src.topic4_zm_ictal_transition import (  # noqa: E402
    _placement_with_artifact_root,
)


METRICS = (
    "patient_loss", "kmeans_loss", "ood_loss", "compound_loss",
    "causal_direction_loss",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(artifact_root: Path, relative: str) -> Path:
    local = ROOT / relative
    return local if local.exists() else artifact_root / relative


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(handle)
    try:
        Path(temporary).write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def metric_vector(row: dict) -> dict[str, float]:
    objective = row["selection_objective"]
    return {
        "patient_loss": float(objective["matched_patient_loss"]),
        "kmeans_loss": float(objective["kmeans_direction_loss"]),
        "ood_loss": float(objective["ood_fraction"]),
        "compound_loss": float(objective["compound_fraction"]),
        "causal_direction_loss": float(objective["causal_direction_loss"]),
    }


def metric_scales(rows: list[dict]) -> dict[str, float]:
    """Use a robust observed scale so no endpoint wins by its units alone."""
    output = {}
    for metric in METRICS:
        values = np.asarray([metric_vector(row)[metric] for row in rows], float)
        scale = float(np.percentile(values, 75) - np.percentile(values, 25))
        output[metric] = max(scale, 1e-6)
    return output


def finite_difference_slopes(by_id: dict[str, dict], *, anchor_index: int,
                             amplitudes: list[float], n_directions: int) -> dict:
    """Estimate each metric gradient from symmetric one-axis perturbations."""
    slopes = {metric: [] for metric in METRICS}
    details = []
    for direction in range(int(n_directions)):
        direction_rows = {metric: [] for metric in METRICS}
        for amplitude_index, amplitude in enumerate(amplitudes):
            minus = metric_vector(by_id[
                f"stage_i_a{anchor_index:02d}_d{direction:02d}_"
                f"s{amplitude_index:02d}_m"
            ])
            plus = metric_vector(by_id[
                f"stage_i_a{anchor_index:02d}_d{direction:02d}_"
                f"s{amplitude_index:02d}_p"
            ])
            for metric in METRICS:
                direction_rows[metric].append(
                    (plus[metric] - minus[metric]) / (2.0 * float(amplitude))
                )
        details.append({
            "direction_index": int(direction),
            "per_amplitude_slopes": direction_rows,
        })
        for metric in METRICS:
            slopes[metric].append(float(np.median(direction_rows[metric])))
    return {"slopes": slopes, "details": details}


def proposal_weights(slopes: dict, scales: dict, weights: dict) -> np.ndarray:
    """Return the downhill coordinate direction for one declared objective."""
    missing = set(METRICS) - set(weights)
    if missing:
        raise ValueError(f"proposal objective omits metrics: {sorted(missing)}")
    gradient = np.zeros(len(slopes[METRICS[0]]), float)
    for metric in METRICS:
        gradient += (
            float(weights[metric])
            * np.asarray(slopes[metric], float) / float(scales[metric])
        )
    if not np.all(np.isfinite(gradient)) or np.linalg.norm(gradient) <= 1e-12:
        raise RuntimeError("finite-difference proposal direction is degenerate")
    return -gradient / np.linalg.norm(gradient)


def _residual_directions(candidates: dict[str, dict], *, n_anchors: int,
                         n_directions: int, reference_amplitude: float) -> list[np.ndarray]:
    output = []
    for direction in range(int(n_directions)):
        copies = []
        for anchor_index in range(int(n_anchors)):
            anchor = np.asarray(candidates[
                f"stage_i_anchor_{anchor_index:02d}"
            ]["node_field"]["coefficients"], float)
            plus = np.asarray(candidates[
                f"stage_i_a{anchor_index:02d}_d{direction:02d}_s00_p"
            ]["node_field"]["coefficients"], float)
            copies.append((plus - anchor) / float(reference_amplitude))
        if any(not np.allclose(copies[0], row, atol=1e-10) for row in copies[1:]):
            raise RuntimeError("stored residual direction differs across anchors")
        output.append(copies[0])
    return output


def manual_smooth_spline_control(stage: dict, placement: dict, *, n_basis: int,
                                 degree: int, grid_per_axis: int) -> dict:
    """Approximate the historical smooth two-core control in the common spline basis."""
    grid = uniform_sheet_grid(grid_per_axis, sheet_mm=float(stage["engine"]["L"]))
    axis = np.asarray(placement["axis_unit_vec"], float)
    s, r = axis_coords(grid, placement["center"], axis)
    separation = float(np.linalg.norm(
        np.asarray(placement["sink_centroid"], float)
        - np.asarray(placement["source_centroid"], float)
    ))
    q = two_core_q(s, r, separation, rho=1.0)
    target = np.log(q)
    target -= float(np.mean(target))
    basis = tensor_basis(
        grid, int(n_basis), degree=int(degree), L=float(stage["engine"]["L"]),
    )
    coefficients, *_ = np.linalg.lstsq(basis, target, rcond=None)
    coefficients = coefficients.reshape(int(n_basis), int(n_basis))
    fitted = continuous_surface(
        coefficients, grid, n_basis=int(n_basis), degree=int(degree),
        L=float(stage["engine"]["L"]),
    )
    correlation = float(np.corrcoef(target, fitted)[0, 1])
    if correlation < 0.995:
        raise RuntimeError("manual smooth spline control does not reproduce its latent field")
    return {
        "candidate_id": "stage_t_manual_smooth_capacity",
        "field_type": "spline_continuous",
        "n_basis": int(n_basis),
        "degree": int(degree),
        "coefficients": coefficients.tolist(),
        "field_sha256": array_sha256(coefficients),
        "roughness": spline_roughness(coefficients),
        "component_count": None,
        "peak_count_constraint": None,
        "role": "manual_smooth_spline_capacity_control",
        "source_field_sha256": None,
        "residual_coordinates": {
            "observation_coordinates_used": True,
            "selection_eligible": False,
        },
        "capacity_control_audit": {
            "source_xy_mm": np.asarray(placement["source_centroid"], float).tolist(),
            "sink_xy_mm": np.asarray(placement["sink_centroid"], float).tolist(),
            "inter_core_mm": separation,
            "latent_surface_correlation": correlation,
            "latent_surface_rmse": float(np.sqrt(np.mean((target - fitted) ** 2))),
            "fit_grid_per_axis": int(grid_per_axis),
        },
    }


def build_candidates(source: dict, summary: dict, design: dict,
                     manual_control: dict) -> tuple[list[dict], dict]:
    candidates = {row["candidate_id"]: row for row in source["candidates"]}
    rows = {row["candidate_id"]: row for row in summary["rows"]}
    n_anchors = len(design["anchor_candidate_ids"])
    n_directions = int(design["n_directions"])
    amplitudes = [float(value) for value in design["finite_difference_amplitudes"]]
    residuals = _residual_directions(
        candidates, n_anchors=n_anchors, n_directions=n_directions,
        reference_amplitude=amplitudes[0],
    )
    scales = metric_scales(list(rows.values()))
    output, audit = [], {"metric_scales": scales, "anchors": {}}
    for anchor_index, anchor_id in enumerate(design["anchor_candidate_ids"]):
        anchor = candidates[str(anchor_id)]["node_field"]
        finite = finite_difference_slopes(
            rows, anchor_index=anchor_index, amplitudes=amplitudes,
            n_directions=n_directions,
        )
        anchor_audit = {"finite_difference": finite, "objectives": {}}
        for objective_name, weights in design["proposal_objectives"].items():
            coordinate_weights = proposal_weights(finite["slopes"], scales, weights)
            residual = normalize_surface_residual(
                residuals, coordinate_weights, n_basis=int(anchor["n_basis"]),
                degree=int(anchor["degree"]),
            )
            anchor_audit["objectives"][objective_name] = {
                "metric_weights": weights,
                "coordinate_weights": coordinate_weights.tolist(),
            }
            for amplitude_index, amplitude in enumerate(
                    design["proposal_surface_rms"]):
                candidate_id = (
                    f"stage_t_a{anchor_index:02d}_{objective_name}_"
                    f"s{amplitude_index:02d}"
                )
                field = residual_candidate(
                    anchor, residual, amplitude=float(amplitude),
                    candidate_id=candidate_id, residual_index=-1,
                    coarse_n_basis=int(design["residual_control_grid"]),
                )
                field["role"] = "rev12_response_surface_continuation"
                field["residual_coordinates"].update({
                    "proposal_objective": objective_name,
                    "coordinate_weights": coordinate_weights.tolist(),
                    "selection_eligible": True,
                })
                output.append({
                    "candidate_id": candidate_id,
                    "role": "causal_continuation_gradient_proposal",
                    "source_candidate_ids": [str(anchor_id)],
                    "selection_eligible": True,
                    "node_field": field,
                })
        audit["anchors"][str(anchor_id)] = anchor_audit
    output.append({
        "candidate_id": manual_control["candidate_id"],
        "role": "rigid_capacity_control_not_selectable",
        "source_candidate_ids": [],
        "selection_eligible": False,
        "node_field": manual_control,
    })
    expected = int(design["expected_candidate_count"])
    if len(output) != expected or len({row["candidate_id"] for row in output}) != expected:
        raise RuntimeError("continuation candidate count or identifiers drifted")
    if len({row["node_field"]["field_sha256"] for row in output}) != expected:
        raise RuntimeError("continuation freezer generated duplicate fields")
    return output, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    artifact_root = args.artifact_root.resolve()
    config = json.loads(config_path.read_text())
    expected = subprocess.check_output(
        ["git", "rev-parse", args.expected_commit], cwd=ROOT, text=True,
    ).strip()
    if subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip() != expected:
        raise RuntimeError("continuation freezer is not at the expected commit")
    tracked = [
        str(config_path.relative_to(ROOT)),
        str(Path(__file__).resolve().relative_to(ROOT)),
        "scripts/run_topic4_rev12_node_worker.py",
        "scripts/aggregate_topic4_rev12_cascade_fit.py",
        "src/topic4_node_field_search.py",
        "src/topic4_node_dualmode.py",
    ]
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", *tracked], cwd=ROOT, text=True,
    ).strip():
        raise RuntimeError("continuation runtime paths are dirty")
    inputs = {}
    for key, record in config["inputs"].items():
        path = _resolve(artifact_root, record["path"])
        digest = _sha256(path)
        if digest != record["sha256"]:
            raise RuntimeError(f"continuation input changed: {record['path']}")
        inputs[key] = {"path": str(path), "sha256": digest}
    source = json.loads(Path(inputs["source_manifest"]["path"]).read_text())
    summary = json.loads(Path(inputs["source_summary"]["path"]).read_text())
    if source.get("status") != "REV12ND_EDGE_SUPPORTED_FIELD_REFIT_FROZEN" \
            or summary.get("status") != "REV12ND_CASCADE_FIT_AGGREGATE_COMPLETE":
        raise RuntimeError("corrected-event source library is not complete")
    transition = json.loads(Path(inputs["transition_config"]["path"]).read_text())
    stage = json.loads(_resolve(
        artifact_root, transition["inputs"]["stage_config"]["path"],
    ).read_text())
    placement = _placement_with_artifact_root(stage, artifact_root)
    manual = manual_smooth_spline_control(
        stage, placement,
        n_basis=int(config["continuation_design"]["stored_n_basis"]),
        degree=int(config["continuation_design"]["degree"]),
        grid_per_axis=int(config["continuation_design"]["manual_fit_grid_per_axis"]),
    )
    candidates, audit = build_candidates(
        source, summary, config["continuation_design"], manual,
    )
    payload = {
        "schema_id": "topic4_rev12_causal_continuation_manifest_v1",
        "status": "REV12ND_CAUSAL_CONTINUATION_FROZEN",
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "candidates": candidates,
        "proposal_audit": audit,
        "event_unit": config["event_unit"],
        "contact_readout": config["search"]["contact_readout"],
        "cascade_objective": config["cascade_objective"],
        "inputs": inputs,
        "provenance": {"git_commit": expected},
    }
    output = artifact_root / config["candidate_manifest"]
    _atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"], "n_candidates": len(candidates),
        "selection_eligible": int(sum(
            row["selection_eligible"] for row in candidates
        )),
        "manual_control_correlation": manual["capacity_control_audit"][
            "latent_surface_correlation"
        ],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
