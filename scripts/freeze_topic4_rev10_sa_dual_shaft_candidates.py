"""Freeze the deterministic SA6 candidate and source geometry manifest."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())
from scripts.run_topic4_rev9_node_kick_canary import _candidate  # noqa: E402
from scripts.run_topic4_rev9l_forced_source_worker import (  # noqa: E402
    _load_json_input,
    _runtime_provenance,
    _sha256,
)
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_core_field_stage3 import params_to_h, params_to_q, unpack  # noqa: E402
from src.topic4_rev10_sa_canary import (  # noqa: E402
    build_dual_shaft_candidates,
    equal_mode_earliest_shaft_centroid,
    shaft_geometry,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/topic4_rev10_sa_dual_shaft_canary.json"


def build_manifest(config_path, expected_commit=None):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    inputs = config["inputs"]
    base = _load_json_input(inputs["rev9_base_config"])
    stage = _load_json_input(inputs["stage_config"])
    confirmation = _load_json_input(inputs["frozen_confirmation"])
    contract = _load_json_input(inputs["contact_contract"])
    sa5 = _load_json_input(inputs["sa5_summary"])
    common_detector = _load_json_input(inputs["rev9_common_detector_audit"])
    if sa5["status"] != "SCL_READOUT_NOT_PRIMARY_LIMIT":
        raise RuntimeError("SA5 has not cleared the observation branch")
    configured_threshold = float(
        config["sa6_dual_shaft_field"]["detector"]
        ["population_active_fraction_threshold"]
    )
    observed_threshold = float(common_detector["common_detector"]["central_threshold"])
    if configured_threshold != observed_threshold:
        raise RuntimeError("SA6 common detector no longer matches its frozen source")
    candidate = _candidate(base)
    confirmed = [row for row in confirmation["candidates"]
                 if row["candidate_id"] == candidate["candidate_id"]]
    if len(confirmed) != 1 or confirmed[0]["theta_sha256"] != candidate["theta_sha256"]:
        raise RuntimeError("frozen rev8.1 candidate does not reproduce")

    with np.load(inputs["shaft_aware_target_npz"]["path"], allow_pickle=False) as loaded:
        target_names = np.asarray(loaded["contact_names"]).astype(str)
        target_xy = np.asarray(loaded["sheet_xy_mm"], float)
        shaft_ids = np.asarray(loaded["shaft_ids"]).astype(str)
        onsets = np.asarray(loaded["patient_train_onsets"], float)
        labels = np.asarray(loaded["patient_train_old_labels"], int)
    contract_names = np.asarray([
        row["contact_name"] for row in contract["contacts"]
    ]).astype(str)
    contract_xy = np.asarray([
        row["sheet_xy_mm"] for row in contract["contacts"]
    ], float)
    if not np.array_equal(target_names, contract_names) or not np.array_equal(
            target_xy, contract_xy):
        raise RuntimeError("SA0 and target contact geometry differ")
    scl = np.flatnonzero(shaft_ids == "SCL")
    geometry = shaft_geometry(target_xy[scl])
    earliest = equal_mode_earliest_shaft_centroid(
        onsets, labels, scl, target_xy,
    )
    design = config["sa6_dual_shaft_field"]["candidate_grid"]
    candidate_set = build_dual_shaft_candidates(
        candidate["theta"],
        scl_midpoint=geometry["midpoint"],
        scl_earliest_centroid=earliest["centroid"],
        scl_phi=geometry["phi"],
        contact_xy=target_xy,
        mass_fractions=design["scl_mass_fractions"],
        sigma_parallel_mm=design["scl_sigma_parallel_mm"],
        K=int(config["sa6_dual_shaft_field"]["K"]),
        L=float(stage["engine"]["L"]),
    )
    if len(candidate_set["candidates"]) != int(design["expected_candidate_count"]):
        raise RuntimeError("candidate count differs from config")

    axis = np.linspace(0.0, float(stage["engine"]["L"]), 100)
    xx, yy = np.meshgrid(axis, axis)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    preflight = []
    for row in candidate_set["candidates"]:
        h = params_to_h(
            row["theta"], grid, K=int(config["sa6_dual_shaft_field"]["K"]),
            L=float(stage["engine"]["L"]),
            target_count=float(stage["N_core_manual"]),
        )
        q_contact = params_to_q(
            row["theta"], target_xy,
            K=int(config["sa6_dual_shaft_field"]["K"]),
            L=float(stage["engine"]["L"]),
        )
        preflight.append({
            "candidate_id": row["candidate_id"],
            "theta_sha256": row["theta_sha256"],
            "grid_sum_h": float(h.sum()),
            "grid_max_h": float(h.max()),
            "contact_q_normalized": (q_contact / q_contact.max()).tolist(),
            "mean_normalized_q_ICL": float(np.mean(
                q_contact[shaft_ids == "ICL"] / q_contact.max()
            )),
            "mean_normalized_q_SCL": float(np.mean(
                q_contact[shaft_ids == "SCL"] / q_contact.max()
            )),
        })

    frozen_components = unpack(
        candidate["theta"], K=int(config["sa6_dual_shaft_field"]["K"]),
        L=float(stage["engine"]["L"]),
    )
    sources = [
        {"id": "ICL_mode_A", "patient_mode": 0,
         "xy_mm": np.asarray(frozen_components[1]["center"], float).tolist()},
        {"id": "ICL_mode_B", "patient_mode": 1,
         "xy_mm": np.asarray(frozen_components[0]["center"], float).tolist()},
        {"id": "SCL", "patient_mode": None,
         "xy_mm": np.asarray(earliest["centroid"], float).tolist()},
    ]
    provenance = _runtime_provenance(expected_commit)
    provenance["config_dirty"] = bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--", str(config_path.relative_to(ROOT))],
        cwd=ROOT, text=True,
    ).strip())
    return {
        "status": "SA6_CANDIDATE_MANIFEST_FROZEN",
        "scientific_role": "fixed-budget Node-only dual-shaft capacity canary",
        "candidate_set": candidate_set,
        "candidate_preflight": preflight,
        "forced_sources": sources,
        "patient_earliest_scl": {
            "equal_mode_centroid_xy_mm": earliest["centroid"].tolist(),
            "per_mode_centroid_xy_mm": earliest["mode_centroids"].tolist(),
            "per_mode_event_counts": earliest["mode_event_counts"].tolist(),
        },
        "shaft_geometry": {
            "midpoint_xy_mm": geometry["midpoint"].tolist(),
            "unit": geometry["unit"].tolist(),
            "phi_rad": geometry["phi"],
        },
        "fixed_contract": {
            "K": int(config["sa6_dual_shaft_field"]["K"]),
            "N_core_manual": float(stage["N_core_manual"]),
            "network_seeds": config["sa6_dual_shaft_field"]["network_seeds"],
            "common_detector": configured_threshold,
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
    parser.add_argument("--out")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = Path(args.out or ROOT / config["output_root"] /
                  "dual_shaft_capacity/candidate_manifest.json")
    payload = build_manifest(args.config, args.expected_commit)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(payload, output)
    print(json.dumps({
        "status": payload["status"],
        "n_candidates": len(payload["candidate_set"]["candidates"]),
        "output": str(output),
        "sha256": _sha256(output),
    }, indent=2))


if __name__ == "__main__":
    main()
