#!/usr/bin/env python3
"""Freeze the current four-pathway connectivity contract without simulation."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
from pathlib import Path

import numpy as np


PATHWAY_LABELS = {
    "E_to_E": "AMPA rows E, columns E",
    "E_to_I": "AMPA rows I, columns E (Params suffix IE)",
    "I_to_E": "GABA rows E, columns I (Params suffix EI)",
    "I_to_I": "GABA rows I, columns I",
}


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantiles(values, points):
    values = np.asarray(values, float)
    return {f"q{int(round(point)):02d}": float(value)
            for point, value in zip(points, np.percentile(values, points))}


def audit_network(payload, *, sample_stride=128):
    net = payload["net"]
    config = payload.get("config", {})
    positions = np.asarray(net["pos"], float)
    n_e, n_i = int(net["NE"]), int(net["NI"])
    theta_deg = float(config.get("theta_EE_deg", 45.0))
    theta = np.deg2rad(theta_deg)
    axis = np.array([np.cos(theta), np.sin(theta)])
    perpendicular = np.array([-axis[1], axis[0]])
    pathways = {
        "E_to_E": {"n_targets": n_e, "n_sources": n_e},
        "E_to_I": {"n_targets": n_i, "n_sources": n_e},
        "I_to_E": {"n_targets": n_e, "n_sources": n_i},
        "I_to_I": {"n_targets": n_i, "n_sources": n_i},
    }
    for values in pathways.values():
        values["in_degree"] = np.zeros(values["n_targets"], np.int64)
        values["out_degree"] = np.zeros(values["n_sources"], np.int64)
        values["distance_samples"] = []
        values["parallel_samples"] = []
        values["perpendicular_samples"] = []

    def consume(matrices, excitatory):
        offset = 0
        for matrix in matrices:
            coo = matrix.tocoo(copy=False)
            rows = np.asarray(coo.row, np.int64)
            columns = np.asarray(coo.col, np.int64)
            groups = (
                (rows < n_e, "E_to_E" if excitatory else "I_to_E", True),
                (rows >= n_e, "E_to_I" if excitatory else "I_to_I", False),
            )
            for mask, pathway, target_is_e in groups:
                selected_rows = rows[mask]
                selected_columns = columns[mask]
                if not len(selected_rows):
                    continue
                local_rows = selected_rows if target_is_e else selected_rows - n_e
                np.add.at(pathways[pathway]["in_degree"], local_rows, 1)
                np.add.at(pathways[pathway]["out_degree"], selected_columns, 1)
                local = np.flatnonzero(
                    (offset + np.arange(len(selected_rows))) % int(sample_stride) == 0,
                )
                offset += len(selected_rows)
                if not len(local):
                    continue
                source_indices = (
                    selected_columns[local] if excitatory
                    else n_e + selected_columns[local]
                )
                displacement = positions[source_indices] - positions[selected_rows[local]]
                pathways[pathway]["distance_samples"].append(
                    np.linalg.norm(displacement, axis=1),
                )
                pathways[pathway]["parallel_samples"].append(
                    np.abs(displacement @ axis),
                )
                pathways[pathway]["perpendicular_samples"].append(
                    np.abs(displacement @ perpendicular),
                )

    consume(net["ampa_by_delay"], True)
    consume(net["gaba_by_delay"], False)
    result = {}
    for pathway, values in pathways.items():
        distance = np.concatenate(values.pop("distance_samples"))
        parallel = np.concatenate(values.pop("parallel_samples"))
        perpendicular_values = np.concatenate(values.pop("perpendicular_samples"))
        in_degree = values.pop("in_degree")
        out_degree = values.pop("out_degree")
        mean_parallel = float(np.mean(parallel))
        mean_perpendicular = float(np.mean(perpendicular_values))
        result[pathway] = {
            "matrix_contract": PATHWAY_LABELS[pathway],
            "n_edges": int(in_degree.sum()),
            "n_targets": values["n_targets"],
            "n_sources": values["n_sources"],
            "in_degree": {
                "min": int(in_degree.min()),
                "median": float(np.median(in_degree)),
                "max": int(in_degree.max()),
            },
            "out_degree": {
                **_quantiles(out_degree, [1, 50, 99]),
                "mean": float(np.mean(out_degree)),
                "max": int(out_degree.max()),
            },
            "edge_distance_mm": _quantiles(distance, [10, 50, 90, 99]),
            "projection_on_EE_axis": {
                "theta_deg": theta_deg,
                "mean_abs_parallel_mm": mean_parallel,
                "mean_abs_perpendicular_mm": mean_perpendicular,
                "parallel_to_perpendicular_ratio": (
                    mean_parallel / mean_perpendicular
                    if mean_perpendicular > 0.0 else None
                ),
            },
            "geometry_sample_size": int(len(distance)),
            "geometry_sample_stride": int(sample_stride),
        }
    return {
        "status": "REV11_NLC_CURRENT_CONNECTIVITY_AUDITED",
        "scientific_interpretation": {
            "EE_long_axis_is_population_label": False,
            "all_E_neurons_share_same_baseline_EE_kernel": True,
            "existing_data_driven_edge_changes_E_to_I": False,
            "existing_data_driven_edge_changes_GABA": False,
            "primary_gap": "E_to_I and the local E-I loop have not entered patient-data optimization",
        },
        "network": {"n_E": n_e, "n_I": n_i, "n_total": n_e + n_i},
        "connectivity_config": {
            key: config.get(key) for key in (
                "seed", "theta_EE_deg", "AR", "C_EE", "C_IE", "C_EI", "C_II",
                "l_EE", "l_IE", "l_EI", "l_II",
            )
        },
        "pathways": result,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-stride", type=int, default=128)
    args = parser.parse_args()
    if args.sample_stride < 1:
        raise ValueError("sample stride must be positive")
    with args.cache.open("rb") as handle:
        payload = pickle.load(handle)
    result = audit_network(payload, sample_stride=args.sample_stride)
    result["inputs"] = {
        "network_cache": {"path": str(args.cache), "sha256": sha256_file(args.cache)},
        "script": {"path": str(Path(__file__)), "sha256": sha256_file(__file__)},
    }
    try:
        result["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
        ).strip()
    except Exception:
        result["git_commit"] = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({
        "status": result["status"],
        "output": str(args.output),
        "pathway_edges": {key: value["n_edges"] for key, value in result["pathways"].items()},
    }, sort_keys=True))


if __name__ == "__main__":
    main()
