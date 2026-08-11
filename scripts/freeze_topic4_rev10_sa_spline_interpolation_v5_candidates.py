"""Freeze adaptive interpolation candidates in the stable spline field."""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())
from scripts.freeze_topic4_rev10_sa_spline_field_v4_candidates import (  # noqa: E402
    _json_classifier,
    _patient_classifier,
)
from scripts.run_topic4_rev9l_forced_source_worker import (  # noqa: E402
    _load_json_input,
    _runtime_provenance,
    _sha256,
)
from src.topic4_continuous_field import (  # noqa: E402
    continuous_surface,
    tensor_basis,
)
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_observation_invariant_spline import (  # noqa: E402
    array_sha256,
    fit_uniform_surface,
    spline_roughness,
)
from src.topic4_spectral_field import uniform_sheet_grid  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/topic4_rev10_sa_observation_invariant_field_v5.json"


def select_anchor_ids(summary, *, reference_id, joint_count, route_count):
    """Select training anchors by frozen rules, never by spatial location."""
    rows = list(summary["candidate_rows"])
    identifiers = {row["candidate_id"] for row in rows}
    if reference_id not in identifiers:
        raise RuntimeError("V5 reference candidate is absent from V4.1")
    ordered = [str(reference_id)]
    joint = sorted(
        (row for row in rows if int(row["n_joint"]) > 0),
        key=lambda row: (
            -float(row["joint_fraction"]), float(row["selection_score"]),
            row["candidate_id"],
        ),
    )
    for row in joint[:int(joint_count)]:
        if row["candidate_id"] not in ordered:
            ordered.append(row["candidate_id"])
    route = sorted(
        (row for row in rows if int(row["n_runaway_networks"]) == 0),
        key=lambda row: (
            float(row["route_score"]), float(row["selection_score"]),
            row["candidate_id"],
        ),
    )
    for row in route:
        if row["candidate_id"] not in ordered:
            ordered.append(row["candidate_id"])
            if len(ordered) >= 1 + int(joint_count) + int(route_count):
                break
    return ordered


def density_mixture(left, right, fraction, grid, *, n_basis, degree):
    """Interpolate positive latent densities and return stable coefficients."""
    t = float(fraction)
    if not 0.0 < t < 1.0:
        raise ValueError("density mixture fraction must be interior")
    left_surface = continuous_surface(
        left, grid, n_basis=n_basis, degree=degree, L=20.0,
    )
    right_surface = continuous_surface(
        right, grid, n_basis=n_basis, degree=degree, L=20.0,
    )
    mixture = np.logaddexp(
        left_surface + np.log1p(-t), right_surface + np.log(t),
    )
    return fit_uniform_surface(
        mixture, grid, n_basis=n_basis, degree=degree, L=20.0,
    )


def _candidate(candidate_id, role, coefficients, config, **extra):
    values = np.asarray(coefficients, float)
    return {
        "candidate_id": str(candidate_id),
        "field_type": "spline_continuous",
        "version": "V5",
        "role": str(role),
        "coefficients": values.tolist(),
        "field_sha256": array_sha256(values),
        "n_basis": int(config["field"]["n_basis_per_axis"]),
        "degree": int(config["field"]["degree"]),
        "roughness": spline_roughness(values),
        "component_count": None,
        "peak_count_constraint": None,
        **extra,
    }


def build_candidates(config, source_manifest, training_summary):
    """Build adaptive fields without contact or shaft geometry arguments."""
    design, field = config["candidate_library"], config["field"]
    n_basis, degree = int(field["n_basis_per_axis"]), int(field["degree"])
    sources = {
        row["candidate_id"]: row
        for row in source_manifest["candidate_set"]["candidates"]
    }
    anchor_ids = select_anchor_ids(
        training_summary,
        reference_id=design["reference_source_candidate_id"],
        joint_count=design["joint_anchor_count"],
        route_count=design["route_anchor_count"],
    )
    if len(anchor_ids) != int(design["expected_anchor_count"]):
        raise RuntimeError("V5 anchor selection did not produce four unique fields")
    anchors = [np.asarray(sources[candidate_id]["coefficients"], float)
               for candidate_id in anchor_ids]
    candidates = [
        _candidate(
            f"v5_anchor_{index:02d}", "adaptive_training_anchor",
            coefficients, config, source_candidate_id=source_id,
            source_field_sha256=sources[source_id]["field_sha256"],
            anchor_index=index,
        )
        for index, (source_id, coefficients) in enumerate(zip(anchor_ids, anchors))
    ]
    grid = uniform_sheet_grid(field["projection_grid_per_axis"], L=20.0)
    pair_records = []
    for pair_index, (left, right) in enumerate(itertools.combinations(range(len(anchors)), 2)):
        for fraction in design["interior_fractions"]:
            t = float(fraction)
            slug = f"{int(round(100 * t)):03d}"
            common = {
                "anchor_pair_index": pair_index,
                "left_anchor_index": left,
                "right_anchor_index": right,
                "left_source_candidate_id": anchor_ids[left],
                "right_source_candidate_id": anchor_ids[right],
                "interpolation_fraction": t,
            }
            candidates.append(_candidate(
                f"v5_latent_p{pair_index:02d}_t{slug}",
                "adaptive_latent_linear_interpolation",
                (1.0 - t) * anchors[left] + t * anchors[right],
                config, **common,
            ))
            candidates.append(_candidate(
                f"v5_density_p{pair_index:02d}_t{slug}",
                "adaptive_density_mixture_interpolation",
                density_mixture(
                    anchors[left], anchors[right], t, grid,
                    n_basis=n_basis, degree=degree,
                ),
                config, **common,
            ))
        pair_records.append({
            "pair_index": pair_index,
            "left_anchor_index": left,
            "right_anchor_index": right,
            "left_source_candidate_id": anchor_ids[left],
            "right_source_candidate_id": anchor_ids[right],
        })
    expected = int(design["candidate_count"])
    if len(candidates) != expected:
        raise RuntimeError(f"expected {expected} V5 candidates")
    hashes = [row["field_sha256"] for row in candidates]
    if len(hashes) != len(set(hashes)):
        raise RuntimeError("V5 interpolation generated duplicate fields")
    return candidates, anchor_ids, pair_records, grid


def build_manifest(config_path, expected_commit=None):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    if config["scientific_role"] != (
            "development_only_stable_spline_adaptive_interpolation"):
        raise RuntimeError("V5 scientific role changed")
    for record in config["inputs"].values():
        if _sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"input hash changed: {record['path']}")
    stage = _load_json_input(config["inputs"]["stage_config"])
    contract = _load_json_input(config["inputs"]["contact_contract"])
    source_manifest = _load_json_input(config["inputs"]["v41_candidate_manifest"])
    training_summary = _load_json_input(config["inputs"]["v41_training_summary"])
    candidates, anchor_ids, pair_records, grid = build_candidates(
        config, source_manifest, training_summary,
    )
    n_basis, degree = (
        int(config["field"]["n_basis_per_axis"]),
        int(config["field"]["degree"]),
    )
    classifier = _patient_classifier(config, contract)
    provenance = _runtime_provenance(expected_commit)
    provenance["config_dirty"] = bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--", str(config_path.relative_to(ROOT))],
        cwd=ROOT, text=True,
    ).strip())
    return {
        "status": "REV10SA_V5_ADAPTIVE_SPLINE_INTERPOLATION_FROZEN",
        "scientific_role": config["scientific_role"],
        "candidate_set": {"candidates": candidates},
        "representation_preflight": {
            "n_basis_per_axis": n_basis,
            "effective_coefficients": n_basis ** 2 - 1,
            "uniform_design_condition_number": float(np.linalg.cond(
                tensor_basis(grid, n_basis, degree=degree, L=20.0)
            )),
            "anchor_source_candidate_ids": anchor_ids,
            "anchor_selection_uses_training_objective_only": True,
            "pair_records": pair_records,
            "observation_geometry_used_by_field_builder": False,
            "unobserved_sheet_is_patient_identified": False,
        },
        "direction_classifier": _json_classifier(classifier),
        "observation_boundary": config["observation_boundary"],
        "fixed_contract": {
            "N_core_manual": float(stage["N_core_manual"]),
            "network_seeds": config["search"]["network_seeds"],
            "selection_network_seeds": config["search"]["selection_network_seeds"],
            "common_detector": config["search"]["detector"][
                "population_active_fraction_threshold"
            ],
            "edge": "off",
            "beta": "closed",
        },
        "inputs": config["inputs"],
        "config": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": _sha256(config_path),
        },
        "provenance": provenance,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--expected-commit")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = build_manifest(args.config, args.expected_commit)
    atomic_write_json(payload, Path(args.out))
    print(json.dumps({
        "status": payload["status"],
        "n_candidates": len(payload["candidate_set"]["candidates"]),
        "anchor_ids": payload["representation_preflight"][
            "anchor_source_candidate_ids"
        ],
        "output": str(Path(args.out).resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
