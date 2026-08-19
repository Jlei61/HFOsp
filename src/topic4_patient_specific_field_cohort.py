"""Contracts for patient-specific continuous-field/local-connectivity fitting."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from src.topic4_continuous_field import tensor_basis
from src.topic4_observation_invariant_spline import spline_roughness
from src.topic4_spectral_field import (
    fourier_basis_2d,
    fourier_wavevectors,
    uniform_sheet_grid,
)


EXPECTED_SCHEMA = "topic4_patient_specific_field_connectivity_cohort_v2"


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha256(values) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(payload: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(json_ready(payload), handle, indent=2, sort_keys=True)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_config(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text())
    if payload.get("schema_version") != EXPECTED_SCHEMA:
        raise RuntimeError("patient-specific cohort config schema changed")
    if payload.get("scientific_role") != (
        "development_only_patient_specific_real_geometry_node_local_connectivity_fit"
    ):
        raise RuntimeError("patient-specific cohort scientific role changed")
    return payload


def source_path(config: dict, relative: str) -> Path:
    return Path(config["source_workspace"]) / str(relative)


def verify_inputs(config: dict, *, code_root: str | Path) -> None:
    code_root = Path(code_root)
    source = Path(config["source_workspace"])
    source_only = {"target_audit", "layout_audit", "stage_config", "common_detector_audit"}
    for name, record in config["inputs"].items():
        if not isinstance(record, dict) or "sha256" not in record:
            continue
        path = (source if name in source_only else code_root) / record["path"]
        if not path.exists() and name not in source_only:
            path = source / record["path"]
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"frozen input changed: {name} -> {path}")


def projected_field_basis(config: dict) -> dict:
    """Build a whole-sheet low-frequency basis without observation geometry."""
    field = config["field"]
    n_basis = int(field["n_basis_per_axis"])
    degree = int(field["degree"])
    length = float(field["sheet_L_mm"])
    grid = uniform_sheet_grid(
        int(field["projection_grid_per_axis"]), L=length,
    )
    spline = tensor_basis(grid, n_basis, degree=degree, L=length)
    raw = fourier_basis_2d(
        grid, int(field["residual_max_harmonic"]), L=length,
    )
    raw -= raw.mean(axis=0, keepdims=True)
    raw_rms = np.sqrt(np.mean(raw ** 2, axis=0, keepdims=True))
    if np.any(raw_rms <= 1e-12):
        raise RuntimeError("whole-sheet Fourier basis contains a zero direction")
    raw /= raw_rms
    coefficients, *_ = np.linalg.lstsq(spline, raw, rcond=None)
    reconstructed = spline @ coefficients
    rms = np.sqrt(np.mean(reconstructed ** 2, axis=0))
    coefficients /= rms[None, :]
    reconstructed /= rms[None, :]
    relative_rmse = np.sqrt(np.mean((reconstructed - raw) ** 2, axis=0))
    directions = coefficients.T.reshape((-1, n_basis, n_basis))
    return {
        "directions": directions,
        "direction_count": int(len(directions)),
        "direction_sha256": array_sha256(directions),
        "maximum_projection_rmse": float(np.max(relative_rmse)),
        "wavevectors_per_mm": fourier_wavevectors(
            int(field["residual_max_harmonic"]), L=length,
        ),
        "uses_contact_geometry": False,
    }


def initial_vector(subject_id: str, config: dict, *, restart: int = 0) -> np.ndarray:
    """Observation-free smooth-prior start; subject ID only keys the RNG."""
    digest = hashlib.sha256(
        f"{subject_id}|restart={int(restart)}|patient-specific-v2".encode("utf-8")
    ).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = np.random.default_rng(seed)
    dimension = int(config["search"]["dimension"])
    edge_count = int(config["local_connectivity"]["coefficient_count"])
    field_count = dimension - edge_count
    vector = np.zeros(dimension, float)
    vector[:field_count] = rng.normal(
        0.0, float(config["field"]["initial_coordinate_sd"]), field_count,
    )
    return vector


def candidate_from_vector(subject_id: str, vector, config: dict, basis: dict,
                          *, generation: int, candidate_index: int,
                          restart: int = 0) -> dict:
    values = np.asarray(vector, float).reshape(-1)
    edge_count = int(config["local_connectivity"]["coefficient_count"])
    field_count = len(values) - edge_count
    if field_count != int(basis["direction_count"]):
        raise ValueError(
            f"optimizer has {field_count} field coordinates but basis has "
            f"{basis['direction_count']} directions"
        )
    field_coordinates = np.clip(
        values[:field_count],
        -float(config["field"]["coordinate_clip_abs"]),
        float(config["field"]["coordinate_clip_abs"]),
    )
    edge_coordinates = np.clip(
        values[field_count:],
        -float(config["local_connectivity"]["coefficient_clip_abs"]),
        float(config["local_connectivity"]["coefficient_clip_abs"]),
    )
    coefficients = np.tensordot(
        field_coordinates, np.asarray(basis["directions"], float), axes=(0, 0),
    )
    edge = edge_coordinates.reshape(2, 6)
    identity = hashlib.sha256(np.concatenate([
        field_coordinates, edge_coordinates,
    ]).astype(np.float64).tobytes()).hexdigest()[:16]
    candidate_id = (
        f"{subject_id}_r{int(restart)}_g{int(generation):02d}_"
        f"c{int(candidate_index):02d}_{identity}"
    )
    return {
        "schema_version": "topic4_patient_specific_candidate_v2",
        "candidate_id": candidate_id,
        "subject_id": str(subject_id),
        "restart": int(restart),
        "generation": int(generation),
        "candidate_index": int(candidate_index),
        "optimizer_vector": values.tolist(),
        "field_coordinates": field_coordinates.tolist(),
        "edge_coordinates": edge_coordinates.tolist(),
        "node_field": {
            "candidate_id": candidate_id,
            "field_type": "spline_continuous",
            "n_basis": int(config["field"]["n_basis_per_axis"]),
            "degree": int(config["field"]["degree"]),
            "coefficients": coefficients.tolist(),
            "field_sha256": array_sha256(coefficients),
            "roughness": spline_roughness(coefficients),
            "component_count": None,
            "peak_count_constraint": None,
            "source": "subject_specific_uniform_whole_sheet_search",
        },
        "edge_coefficients": edge.tolist(),
        "edge_coefficients_sha256": array_sha256(edge),
        "basis_sha256": str(basis["direction_sha256"]),
    }


def load_subject_contract(config: dict, subject_id: str) -> dict:
    source = Path(config["source_workspace"])
    target_root = source / config["inputs"]["target_subject_root"]
    layout_root = source / config["inputs"]["layout_subject_root"]
    target_json_path = target_root / f"{subject_id}.json"
    target_npz_path = target_root / f"{subject_id}_target.npz"
    layout_json_path = layout_root / f"{subject_id}.json"
    layout_npz_path = layout_root / f"{subject_id}_layout.npz"
    for path in (target_json_path, target_npz_path, layout_json_path, layout_npz_path):
        if not path.exists():
            raise FileNotFoundError(path)
    target_json = json.loads(target_json_path.read_text())
    layout_json = json.loads(layout_json_path.read_text())
    with np.load(layout_npz_path, allow_pickle=False) as loaded:
        contact_order = [str(value) for value in loaded["contact_order"]]
        real = (
            np.asarray(loaded["real_coords_sheet"], float)
            if "real_coords_sheet" in loaded else None
        )
        permutations = np.asarray(
            loaded["within_shaft_null_permutations"], int,
        )
    if contact_order != [str(value) for value in target_json["target"]["contact_order"]]:
        raise RuntimeError(f"target/layout contact order mismatch for {subject_id}")
    return {
        "subject_id": str(subject_id),
        "contact_order": contact_order,
        "real_coords_sheet": real,
        "null_permutations": permutations,
        "target_json": target_json,
        "target_npz_path": target_npz_path,
        "layout_json": layout_json,
        "hashes": {
            "target_json": sha256(target_json_path),
            "target_npz": sha256(target_npz_path),
            "layout_json": sha256(layout_json_path),
            "layout_npz": sha256(layout_npz_path),
        },
    }


def patient_target_arrays(target_npz_path: str | Path, split: str) -> dict:
    if split not in {"train", "heldout"}:
        raise ValueError("split must be train or heldout")
    with np.load(target_npz_path, allow_pickle=False) as loaded:
        order = [str(value) for value in loaded["contact_order"]]
        target = {
            "contact_order": order,
            "profiles": np.stack([
                loaded[f"{split}_ta_profile"], loaded[f"{split}_tb_profile"],
            ]),
            "recruitment": np.stack([
                loaded[f"{split}_ta_recruitment"],
                loaded[f"{split}_tb_recruitment"],
            ]),
            "precedence": np.stack([
                loaded[f"{split}_ta_precedence"],
                loaded[f"{split}_tb_precedence"],
            ]),
        }
        return {
            "target": target,
            "patient_centers": np.asarray(loaded["kmeans_centers"], float),
        }


def objective_from_score(score: dict, candidate: dict, config: dict) -> dict:
    weights = config["objective"]
    minimum = int(config["runtime"]["minimum_events_per_mode"])
    status = str(score.get("status", "UNKNOWN"))
    roughness = float(candidate["node_field"]["roughness"])
    edge = np.asarray(candidate["edge_coefficients"], float)
    regularization = (
        float(weights["field_roughness_weight"]) * np.log1p(roughness)
        + float(weights["edge_l2_weight"]) * float(np.mean(edge ** 2))
    )
    if status == "INSUFFICIENT_EVENTS":
        count = int(score.get("n_readable_events", 0))
        deficit = max(0, 2 * minimum - count) / float(max(2 * minimum, 1))
        value = 3.0 + float(weights["support_penalty_weight"]) * deficit
        return {"objective": float(value + regularization), "status": status,
                "support_penalty": float(deficit), "regularization": regularization}
    if status == "INSUFFICIENT_IN_DISTRIBUTION_MODE_SUPPORT":
        counts = np.asarray(score.get("supervised_mode_counts", [0, 0]), float)
        deficit = float(np.mean(np.maximum(0.0, minimum - counts) / minimum))
        value = 2.0 + float(weights["ood_fraction_weight"]) * float(
            score.get("ood_fraction", 1.0)
        ) + float(weights["support_penalty_weight"]) * deficit
        return {"objective": float(value + regularization), "status": status,
                "support_penalty": deficit, "regularization": regularization}
    if status != "EVALUABLE":
        return {
            "objective": float(weights["invalid_objective"]),
            "status": status,
            "support_penalty": 1.0,
            "regularization": regularization,
        }
    natural = score["natural_kmeans"]
    counts = np.asarray(natural["cluster_counts"], float)
    support = float(np.mean(np.maximum(0.0, minimum - counts) / minimum))
    ami = float(natural["seed_ami_median"])
    components = {
        "supervised_weakest_mode": float(score["weakest_mode_loss"]),
        "natural_kmeans_weakest_mode": float(natural["weakest_mode_loss"]),
        "ood_fraction": float(score["ood_fraction"]),
        "kmeans_instability": float(max(0.0, 1.0 - ami)),
        "support_penalty": support,
        "field_roughness": roughness,
        "edge_l2_mean": float(np.mean(edge ** 2)),
    }
    value = (
        float(weights["supervised_weakest_mode_weight"])
        * components["supervised_weakest_mode"]
        + float(weights["natural_kmeans_weakest_mode_weight"])
        * components["natural_kmeans_weakest_mode"]
        + float(weights["ood_fraction_weight"]) * components["ood_fraction"]
        + float(weights["kmeans_instability_weight"])
        * components["kmeans_instability"]
        + float(weights["support_penalty_weight"]) * support
        + regularization
    )
    return {"objective": float(value), "status": status,
            "regularization": regularization, **components}
