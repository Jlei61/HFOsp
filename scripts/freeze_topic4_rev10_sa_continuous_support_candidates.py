"""Freeze no-K continuous dual-shaft support capacity controls."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())
from scripts.run_topic4_rev9l_forced_source_worker import (  # noqa: E402
    _load_json_input,
    _runtime_provenance,
    _sha256,
)
from src.topic4_continuous_field import (  # noqa: E402
    build_continuous_support_candidates,
    continuous_corridor_field_h,
    distance_to_segments,
)
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_rev10_sa_canary import (  # noqa: E402
    equal_mode_earliest_shaft_centroid,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/topic4_rev10_sa_continuous_support_canary.json"


def build_manifest(config_path, expected_commit=None):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    assay = config["sa6g_continuous_support"]
    if assay["component_count"] is not None or assay["peak_count_constraint"] is not None:
        raise RuntimeError("continuous support control cannot fix components or peaks")
    inputs = config["inputs"]
    stage = _load_json_input(inputs["stage_config"])
    contract = _load_json_input(inputs["contact_contract"])
    sa5 = _load_json_input(inputs["sa5_summary"])
    detector_audit = _load_json_input(inputs["rev9_common_detector_audit"])
    if sa5["status"] != "SCL_READOUT_NOT_PRIMARY_LIMIT":
        raise RuntimeError("SA5 has not cleared the observation branch")
    threshold = float(assay["detector"]["population_active_fraction_threshold"])
    if threshold != float(detector_audit["common_detector"]["central_threshold"]):
        raise RuntimeError("common detector changed")

    with np.load(inputs["shaft_aware_target_npz"]["path"], allow_pickle=False) as loaded:
        names = np.asarray(loaded["contact_names"]).astype(str)
        contact_xy = np.asarray(loaded["sheet_xy_mm"], float)
        shaft_ids = np.asarray(loaded["shaft_ids"]).astype(str)
        onsets = np.asarray(loaded["patient_train_onsets"], float)
        labels = np.asarray(loaded["patient_train_old_labels"], int)
    contacts = contract["contacts"]
    contract_names = np.asarray([row["contact_name"] for row in contacts]).astype(str)
    contract_xy = np.asarray([row["sheet_xy_mm"] for row in contacts], float)
    if not np.array_equal(names, contract_names) or not np.array_equal(contact_xy, contract_xy):
        raise RuntimeError("SA0 and patient target contact geometry differ")

    design = assay["candidate_design"]
    built = build_continuous_support_candidates(
        contacts, widths_mm=design["width_mm"],
    )
    candidates = [{**row, "role": (
        "continuous_connected_support" if row["bridge_segment"] is not None
        else "continuous_disconnected_support"
    )} for row in built["candidates"]]
    if len(candidates) != int(design["candidate_count"]):
        raise RuntimeError("continuous support candidate count changed")

    L = float(stage["engine"]["L"])
    axis = np.linspace(0.0, L, 161)
    xx, yy = np.meshgrid(axis, axis)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    expected_n_e = float(stage["engine"]["density"]) * L ** 2 * 0.8
    grid_budget = float(stage["N_core_manual"]) * len(grid) / expected_n_e
    path_radius = float(assay["field_strength_audit"]["path_radius_mm"])
    preflight = []
    for row in candidates:
        segments = np.asarray(row["segments"], float)
        h, diagnostics = continuous_corridor_field_h(
            segments, grid, width_mm=row["width_mm"], target_count=grid_budget,
        )
        distance = distance_to_segments(grid, segments)
        selected = distance <= path_radius
        preflight.append({
            "candidate_id": row["candidate_id"],
            "field_sha256": row["field_sha256"],
            "grid_budget": float(h.sum()),
            "grid_max_h": float(h.max()),
            "mean_h_within_path_radius": float(np.mean(h[selected])),
            "fraction_h_ge_0p5_within_path_radius": float(np.mean(h[selected] >= 0.5)),
            "path_radius_mm": path_radius,
            **diagnostics,
        })

    icl = np.flatnonzero(shaft_ids == "ICL")
    scl = np.flatnonzero(shaft_ids == "SCL")
    icl_earliest = equal_mode_earliest_shaft_centroid(onsets, labels, icl, contact_xy)
    scl_earliest = equal_mode_earliest_shaft_centroid(onsets, labels, scl, contact_xy)
    sources = [
        {"id": "ICL_mode_A", "patient_mode": 0,
         "xy_mm": icl_earliest["mode_centroids"][0].tolist()},
        {"id": "ICL_mode_B", "patient_mode": 1,
         "xy_mm": icl_earliest["mode_centroids"][1].tolist()},
        {"id": "SCL", "patient_mode": None,
         "xy_mm": scl_earliest["centroid"].tolist()},
    ]
    provenance = _runtime_provenance(expected_commit)
    provenance["config_dirty"] = bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--", str(config_path.relative_to(ROOT))],
        cwd=ROOT, text=True,
    ).strip())
    return {
        "status": "SA6G_CONTINUOUS_SUPPORT_MANIFEST_FROZEN",
        "scientific_role": (
            "no-K continuous support capacity positive control; support segments "
            "are observed geometry, not cores"
        ),
        "candidate_set": {"candidates": candidates},
        "candidate_preflight": preflight,
        "forced_sources": sources,
        "support_geometry": {
            "shaft_segments": built["shaft_segments"].tolist(),
            "shortest_cross_shaft_bridge": built["bridge_segment"].tolist(),
        },
        "fixed_contract": {
            "component_count": None,
            "peak_count_constraint": None,
            "N_core_manual": float(stage["N_core_manual"]),
            "network_seeds": assay["network_seeds"],
            "common_detector": threshold,
            "edge": "off",
            "beta": "closed",
        },
        "inputs": {
            key: {"path": value["path"], "sha256": value["sha256"]}
            for key, value in inputs.items()
        },
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
    atomic_write_json(payload, Path(args.out).resolve())
    print(json.dumps({
        "status": payload["status"],
        "n_candidates": len(payload["candidate_set"]["candidates"]),
        "output": str(Path(args.out).resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
