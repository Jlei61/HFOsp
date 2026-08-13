"""Freeze a two-direction continuous-field response surface for D6.2."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.freeze_topic4_rev10_d6_continuous_field_kmeans_screen import (  # noqa: E402
    _node_field,
)
from scripts.run_topic4_rev9l_forced_source_worker import (  # noqa: E402
    _runtime_provenance,
    _sha256,
)
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/topic4_rev10_d6_2_joint_continuous_field_surface.json"
EXPECTED_ROLE = "development_only_continuous_field_joint_direction_surface"
SOURCE_STATUS = "REV10D6_1_NATURAL_KMEANS_CLOSEOUT_LIBRARY_FROZEN"


def _candidate_id(a, b):
    special = {
        (0.0, 0.0): "edge_noop",
        (1.0, 0.0): "d6_f09_sin_p0p4",
        (0.0, 1.0): "d6_f05_sin_m0p8",
    }
    if (a, b) in special:
        return special[(a, b)]
    token = lambda value: str(float(value)).replace(".", "p")
    return f"d62_a{token(a)}_b{token(b)}"


def candidate_library(config, source):
    by_id = {
        row["candidate_id"]: row
        for row in source["candidate_set"]["candidates"]
    }
    baseline = by_id["edge_noop"]
    natural = by_id[config["field_search"]["natural_direction_source_candidate_id"]]
    geometry = by_id[
        config["field_search"]["patient_geometry_direction_source_candidate_id"]
    ]
    base_coeff = np.asarray(baseline["node_field"]["coefficients"], float)
    natural_coeff = np.asarray(natural["node_field"]["coefficients"], float)
    geometry_coeff = np.asarray(geometry["node_field"]["coefficients"], float)
    direction_a = natural_coeff - base_coeff
    direction_b = geometry_coeff - base_coeff
    if np.linalg.norm(direction_a) == 0 or np.linalg.norm(direction_b) == 0:
        raise RuntimeError("D6.2 source directions must be nonzero")
    cosine = float(
        np.dot(direction_a.ravel(), direction_b.ravel())
        / (np.linalg.norm(direction_a) * np.linalg.norm(direction_b))
    )
    rows = []
    for a, b in config["field_search"]["latent_coordinates"]:
        a, b = float(a), float(b)
        candidate_id = _candidate_id(a, b)
        if (a, b) == (0.0, 0.0):
            row = deepcopy(baseline)
        elif (a, b) == (1.0, 0.0):
            row = deepcopy(natural)
        elif (a, b) == (0.0, 1.0):
            row = deepcopy(geometry)
        else:
            coefficients = base_coeff + a * direction_a + b * direction_b
            row = {
                "candidate_id": candidate_id,
                "coefficients": deepcopy(baseline["coefficients"]),
                "coefficients_sha256": baseline["coefficients_sha256"],
                "node_field": _node_field(
                    candidate_id, coefficients, baseline["node_field"],
                    role="d6_2_two_direction_continuous_response_surface",
                    source_field_sha256=baseline["node_field"]["field_sha256"],
                    source_direction_field_sha256={
                        "natural_kmeans": natural["node_field"]["field_sha256"],
                        "patient_crossfit": geometry["node_field"]["field_sha256"],
                    },
                ),
                "spatial_ou": deepcopy(baseline["spatial_ou"]),
            }
        row["d6_2_latent_coordinates"] = {
            "natural_kmeans_direction_a": a,
            "patient_crossfit_direction_b": b,
        }
        row["d6_2_role"] = (
            "warm_baseline" if (a, b) == (0.0, 0.0)
            else "D6_1_natural_endpoint" if (a, b) == (1.0, 0.0)
            else "D6_1_patient_crossfit_endpoint" if (a, b) == (0.0, 1.0)
            else "continuous_direction_combination"
        )
        rows.append(row)
    expected = int(config["field_search"]["candidate_count"])
    if len(rows) != expected or len({row["candidate_id"] for row in rows}) != expected:
        raise RuntimeError("D6.2 candidate coordinates or IDs changed")
    if len({row["node_field"]["field_sha256"] for row in rows}) != expected:
        raise RuntimeError("D6.2 fields are not unique")
    if any(np.any(np.asarray(row["coefficients"], float) != 0.0) for row in rows):
        raise RuntimeError("D6.2 edge adapter must remain an exact no-op")
    return rows, {
        "direction_a_l2": float(np.linalg.norm(direction_a)),
        "direction_b_l2": float(np.linalg.norm(direction_b)),
        "direction_cosine": cosine,
        "source_field_sha256": {
            "baseline": baseline["node_field"]["field_sha256"],
            "natural_kmeans": natural["node_field"]["field_sha256"],
            "patient_crossfit": geometry["node_field"]["field_sha256"],
        },
    }


def build_manifest(config_path, expected_commit):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    if config.get("scientific_role") != EXPECTED_ROLE:
        raise RuntimeError("D6.2 scientific role changed")
    for record in config["inputs"].values():
        if _sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"input hash changed: {record['path']}")
    source = json.loads((ROOT / config["inputs"]["d6_1_manifest"]["path"]).read_text())
    verdict = json.loads((ROOT / config["inputs"]["d6_1_verdict"]["path"]).read_text())
    if source.get("status") != SOURCE_STATUS:
        raise RuntimeError("D6.1 source library is not frozen")
    if verdict.get("status") != (
            "REV10D6_1_ORTHOGONAL_PARTIAL_SENSITIVITY_REPERTOIRE_UNRESOLVED"):
        raise RuntimeError("D6.2 requires the frozen D6.1 orthogonal-sensitivity result")
    candidates, directions = candidate_library(config, source)
    commit = subprocess.check_output(
        ["git", "rev-parse", expected_commit], cwd=ROOT, text=True,
    ).strip()
    provenance = _runtime_provenance(commit)
    provenance["config_dirty"] = bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--", str(config_path.relative_to(ROOT))],
        cwd=ROOT, text=True,
    ).strip())
    if (provenance["config_dirty"] or provenance["runtime_modules_dirty"]
            or not provenance["runtime_modules_match_expected_commit"]):
        raise RuntimeError("D6.2 freezer runtime or config is not frozen")
    output_root = ROOT / config["output_root"]
    if any((output_root / "workers").glob("*.json")):
        raise RuntimeError("D6.2 workers exist before manifest freeze")
    return {
        "status": "REV10D6_2_JOINT_CONTINUOUS_FIELD_SURFACE_FROZEN",
        "scientific_role": EXPECTED_ROLE,
        "config": {"path": str(config_path.relative_to(ROOT)), "sha256": _sha256(config_path)},
        "candidate_set": {"n_candidates": len(candidates), "candidates": candidates},
        "selection_freeze": {
            "paired_control_candidate_id": "edge_noop",
            "response_surface_only": True,
            "no_scalar_winner_predeclared": True,
            "frozen_before_D6_2_networks": True,
        },
        "direction_basis_audit": directions,
        "direction_classifier": source["direction_classifier"],
        "direction_classifier_source": source["direction_classifier_source"],
        "fixed_contract": {
            "network_seeds": config["search"]["confirmation_network_seeds"],
            "duration_ms": config["search"]["simulation"]["duration_ms"],
            "spatial_ou": config["fixed_spatial_ou"],
            "edge": "exact no-op", "beta": "closed",
        },
        "forbidden_builder_inputs": config["field_search"]["forbidden_builder_inputs"],
        "provenance": provenance,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    payload = build_manifest(args.config, args.expected_commit)
    output = ROOT / config["output_root"] / "candidate_manifest.json"
    atomic_write_json(payload, output)
    print(json.dumps({
        "status": payload["status"],
        "candidate_ids": [
            row["candidate_id"] for row in payload["candidate_set"]["candidates"]
        ],
        "direction_cosine": payload["direction_basis_audit"]["direction_cosine"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
