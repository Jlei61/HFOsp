#!/usr/bin/env python3
"""Audit every rev22 design point after both connectivity transformations.

This is a structure-only computation: it reconstructs each frozen fit topology but never
steps the SNN.  The exact producer order is ellipse redistribution followed by the learned
E-to-E/E-to-I mapper.  Every candidate must preserve pathway-specific incoming budgets,
avoid extreme edge concentration and retain enough effective incoming sources on all four
fit topologies before the expensive response run is authorized.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "src" / "snn_engine"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from src.topic4_local_connectivity import continuous_local_e_source_flow  # noqa: E402
from src.topic4_rev20_dual_core_mechanism import (  # noqa: E402
    fixed_topology_ee_ellipse_redistribution,
)
from src.topic4_rev22_connectivity_audit import (  # noqa: E402
    candidate_pathway_weights_from_logits,
    candidate_structure,
    flatten_pathway,
    pathway_logits,
    structure_passes,
)
from src.topic4_zm_ictal_transition import build_substrate, load_round_config  # noqa: E402


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_RESPONSE = ARTIFACT_ROOT / (
    "results/topic4_sef_hfo/data_driven_dual_core_interictal_identifiability/"
    "response_design/response_design_manifest.json"
)
DEFAULT_SEEDS = DEFAULT_RESPONSE.with_name("seed_manifest.json")
DEFAULT_TRANSITION = ROOT / "config/topic4_rev22_dci_transition_execution.json"
DEFAULT_REV20 = ROOT / "config/topic4_rev20_dc_dual_core_mechanism_atlas.json"
DEFAULT_OUT = ARTIFACT_ROOT / (
    "results/topic4_sef_hfo/data_driven_dual_core_interictal_identifiability/"
    "connectivity_design_audit"
)
THRESHOLDS = {
    "budget_error_max": 1e-9,
    "edge_ratio_p01_min": 0.25,
    "edge_ratio_p99_max": 4.0,
    "effective_source_median_ratio_min": 0.75,
    "effective_source_p05_ratio_min": 0.50,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _fit_seeds(seed_manifest: dict) -> list[int]:
    units = seed_manifest.get("fit", {}).get("units", [])
    seeds = [int(row["topology_seed"]) for row in units]
    if seeds != [2511, 2512, 2513, 2514]:
        raise ValueError("fit seeds changed from the frozen four-topology contract")
    if any(int(row["dynamics_seed"]) != int(row["topology_seed"]) for row in units):
        raise ValueError("structure audit expects the legacy topology=dynamics fit units")
    return seeds


def validate_inputs(response: dict, seed_manifest: dict, transition: dict,
                    *, response_sha256: str) -> None:
    if response.get("schema_id") != "topic4_rev22_dci_response_design_manifest_v1":
        raise ValueError("unexpected response-design schema")
    if int(response.get("candidate_count", -1)) != len(response.get("candidates", [])):
        raise ValueError("response-design candidate count is inconsistent")
    if seed_manifest.get("response_design_manifest_sha256") != response_sha256:
        raise ValueError("seed manifest is not bound to the response design")
    _fit_seeds(seed_manifest)
    if transition.get("schema_id") != "topic4_rev22_dci_transition_execution_v1":
        raise ValueError("minimal rev22 transition contract is required")
    forbidden = " ".join(transition.get("inputs", {})).lower()
    if any(token in forbidden for token in ("heldout", "classifier", "patient_ictal", "fig5")):
        raise ValueError("validation-only input is visible to the structure audit")
    for candidate in response["candidates"]:
        if candidate.get("mechanisms", {}).get("Z_M") != "off":
            raise ValueError("Z/M is outside rev22")
        if candidate.get("node_field", {}).get("field_sha256") != response.get("node_field_sha256"):
            raise ValueError("response design changes the frozen Node field")


def _baseline(seed: int, *, artifact_root: Path, transition_path: Path,
              rev20: dict, node_field: dict):
    transition = load_round_config(transition_path)
    reference = rev20["reference"]
    substrate = build_substrate(
        transition, str(reference["base_substrate_candidate_id"]), int(seed),
        cache_dir=str(artifact_root / rev20["network_cache"]),
        ee_dose=0.0, etoi_dose=0.0,
        node_candidate_override=dict(node_field),
        node_depth_shrinkage=float(reference["signed_depth_shrinkage"]),
        node_gain=float(reference["node_gain"]),
        ee_ellipse_angle_deg=45.0, ee_ellipse_aspect_ratio=2.0,
        artifact_root=artifact_root, topology_seed=int(seed), dynamics_seed=int(seed),
    )
    if not np.allclose(substrate.edge_coefficients, 0.0):
        raise RuntimeError("zero-dose baseline retained learned coefficients")
    candidate = substrate.extras["candidate"]
    coefficient = np.asarray(candidate["coefficients"], float)
    if coefficient.shape != (2, 6):
        raise RuntimeError("frozen learned coefficient matrix changed shape")
    transition_local = transition["local_connectivity_basis"]
    return substrate, coefficient, candidate.get("raw_logit_clip"), transition_local


def _producer_parity(net: dict, positions: np.ndarray, h_all: np.ndarray,
                     coefficients: np.ndarray, raw_logit_clip: float | None,
                     local: dict, candidate: dict, ee: dict, etoi: dict,
                     fast: dict) -> dict:
    mechanisms = candidate["mechanisms"]
    ellipse, ellipse_audit = fixed_topology_ee_ellipse_redistribution(
        net, positions, length_scale=float(local["E_to_E_length_scale_mm"]),
        angle_deg=float(mechanisms["ellipse_angle_deg"]),
        aspect_ratio=float(mechanisms["ellipse_aspect_ratio"]),
    )
    dose = np.asarray([[float(mechanisms["g_EE"])], [float(mechanisms["g_EtoI"])]])
    produced, learned_audit = continuous_local_e_source_flow(
        ellipse, positions, h_all, coefficients * dose,
        l_ee=float(local["E_to_E_length_scale_mm"]),
        l_e_to_i=float(local["E_to_I_length_scale_mm"]),
        raw_logit_clip=raw_logit_clip,
    )
    actual = {
        "E_to_E": flatten_pathway(produced, "E_to_E"),
        "E_to_I": flatten_pathway(produced, "E_to_I"),
    }
    result = {}
    for pathway, baseline in (("E_to_E", ee), ("E_to_I", etoi)):
        row = actual[pathway]
        if not (np.array_equal(row["bin"], baseline["bin"])
                and np.array_equal(row["row"], baseline["row"])
                and np.array_equal(row["col"], baseline["col"])):
            raise RuntimeError(f"producer changed {pathway} topology or delay assignment")
        maximum = float(np.max(np.abs(row["data"] - fast[pathway]), initial=0.0))
        relative = maximum / max(float(np.max(np.abs(row["data"]), initial=0.0)), 1e-30)
        if relative > 1e-11:
            raise RuntimeError(f"fast combined mapper diverges for {pathway}: {relative:.3e}")
        result[pathway] = {"max_abs_difference": maximum, "max_relative_difference": relative}
    result["ellipse_topology_unchanged"] = bool(ellipse_audit.get("topology_unchanged", True))
    result["learned_topology_unchanged"] = bool(learned_audit.get("topology_unchanged", False))
    result["learned_delay_assignment_unchanged"] = bool(
        learned_audit.get("delay_assignment_unchanged", False)
    )
    result["pass"] = bool(
        result["ellipse_topology_unchanged"] and result["learned_topology_unchanged"]
        and result["learned_delay_assignment_unchanged"]
    )
    return result


def audit_topology(seed: int, *, artifact_root: Path, transition_path: Path,
                   rev20: dict, candidates: list[dict], started: float) -> tuple[int, dict]:
    t0 = time.time()
    node_field = candidates[0]["node_field"]
    substrate, coefficients, raw_logit_clip, local = _baseline(
        seed, artifact_root=artifact_root, transition_path=transition_path,
        rev20=rev20, node_field=node_field,
    )
    net = substrate.net
    positions = np.asarray(net["pos"], float)
    h_all = np.concatenate([substrate.h_e, substrate.h_i])
    ee, etoi = flatten_pathway(net, "E_to_E"), flatten_pathway(net, "E_to_I")
    base_logits = {
        "E_to_E": pathway_logits(
            ee, positions, h_all, coefficients[0],
            length_scale=float(local["E_to_E_length_scale_mm"]), raw_logit_clip=None,
        ),
        "E_to_I": pathway_logits(
            etoi, positions, h_all, coefficients[1],
            length_scale=float(local["E_to_I_length_scale_mm"]), raw_logit_clip=None,
        ),
    }
    print(
        f"[{time.time()-started:6.0f}s] topology {seed}: baseline and logits ready; "
        f"EE={len(ee['data'])} E-to-I={len(etoi['data'])}", flush=True,
    )
    rows, parity = [], None
    parity_index = max(
        range(len(candidates)),
        key=lambda index: sum(abs(float(candidates[index]["unit_cube"][name]) - 0.5)
                              for name in ("g_LEE", "g_LEI", "theta_FT_deg", "AR_FT")),
    )
    for index, candidate in enumerate(candidates):
        mechanism = candidate["mechanisms"]
        final = candidate_pathway_weights_from_logits(
            ee, etoi, positions=positions, base_logits=base_logits,
            g_ee=float(mechanism["g_EE"]), g_etoi=float(mechanism["g_EtoI"]),
            length_scale_ee=float(local["E_to_E_length_scale_mm"]),
            angle_deg=float(mechanism["ellipse_angle_deg"]),
            aspect_ratio=float(mechanism["ellipse_aspect_ratio"]),
            raw_logit_clip=raw_logit_clip,
        )
        structure = candidate_structure(ee, etoi, final, positions)
        passed = structure_passes(structure, THRESHOLDS)
        rows.append({
            "candidate_id": candidate["candidate_id"], "topology_seed": int(seed),
            "block": candidate["block"], "physical": dict(candidate["physical"]),
            "passes": bool(passed), "pathways": structure,
        })
        if index == parity_index:
            parity = {
                "candidate_id": candidate["candidate_id"],
                **_producer_parity(
                    net, positions, h_all, coefficients, raw_logit_clip, local,
                    candidate, ee, etoi, final,
                ),
            }
        del final
        if (index + 1) % 8 == 0 or index + 1 == len(candidates):
            print(
                f"[{time.time()-started:6.0f}s] topology {seed}: "
                f"{index+1}/{len(candidates)} design points", flush=True,
            )
    if parity is None or not parity["pass"]:
        raise RuntimeError("real-graph producer parity was not established")
    return int(seed), {
        "rows": rows, "producer_parity": parity,
        "network_cache": substrate.network_cache,
        "coefficient_sha256": hashlib.sha256(
            np.ascontiguousarray(coefficients, dtype=np.float64).tobytes()
        ).hexdigest(),
        "raw_logit_clip": raw_logit_clip,
        "edge_counts": {"E_to_E": len(ee["data"]), "E_to_I": len(etoi["data"])},
        "elapsed_seconds": float(time.time() - t0),
    }


def summarize(per_topology: dict, candidates: list[dict]) -> tuple[list[dict], bool]:
    expected = {(candidate["candidate_id"], int(seed)) for candidate in candidates
                for seed in per_topology}
    observed = {(row["candidate_id"], int(row["topology_seed"]))
                for block in per_topology.values() for row in block["rows"]}
    if observed != expected:
        raise RuntimeError("combined connectivity audit grid is incomplete")
    flat = [row for seed in sorted(per_topology) for row in per_topology[seed]["rows"]]
    return flat, bool(all(row["passes"] for row in flat))


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "candidate_id", "topology_seed", "block", "g_LEE", "g_LEI", "theta_FT_deg", "AR_FT",
        "passes", "EE_budget_error", "EE_ratio_p01", "EE_ratio_p99",
        "EE_effective_source_median_ratio", "EE_effective_source_p05_ratio",
        "EtoI_budget_error", "EtoI_ratio_p01", "EtoI_ratio_p99",
        "EtoI_effective_source_median_ratio", "EtoI_effective_source_p05_ratio",
        "achieved_EE_angle_deg", "achieved_EE_aspect_ratio", "achieved_EE_rms_distance_mm",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            ee, etoi = row["pathways"]["E_to_E"], row["pathways"]["E_to_I"]
            geometry = ee["achieved_geometry"]
            writer.writerow({
                "candidate_id": row["candidate_id"], "topology_seed": row["topology_seed"],
                "block": row["block"], **row["physical"], "passes": row["passes"],
                "EE_budget_error": ee["maximum_abs_incoming_error"],
                "EE_ratio_p01": ee["edge_ratio_p01"], "EE_ratio_p99": ee["edge_ratio_p99"],
                "EE_effective_source_median_ratio": ee["effective_source_median_ratio"],
                "EE_effective_source_p05_ratio": ee["effective_source_p05_ratio"],
                "EtoI_budget_error": etoi["maximum_abs_incoming_error"],
                "EtoI_ratio_p01": etoi["edge_ratio_p01"],
                "EtoI_ratio_p99": etoi["edge_ratio_p99"],
                "EtoI_effective_source_median_ratio": etoi["effective_source_median_ratio"],
                "EtoI_effective_source_p05_ratio": etoi["effective_source_p05_ratio"],
                "achieved_EE_angle_deg": geometry["achieved_angle_deg"],
                "achieved_EE_aspect_ratio": geometry["achieved_aspect_ratio"],
                "achieved_EE_rms_distance_mm": geometry["weighted_rms_distance_mm"],
            })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--response-manifest", type=Path, default=DEFAULT_RESPONSE)
    parser.add_argument("--seed-manifest", type=Path, default=DEFAULT_SEEDS)
    parser.add_argument("--transition-config", type=Path, default=DEFAULT_TRANSITION)
    parser.add_argument("--rev20-config", type=Path, default=DEFAULT_REV20)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    started = time.time()
    response_hash = sha256(args.response_manifest)
    response = json.loads(args.response_manifest.read_text())
    seeds = json.loads(args.seed_manifest.read_text())
    transition = json.loads(args.transition_config.read_text())
    rev20 = json.loads(args.rev20_config.read_text())
    validate_inputs(response, seeds, transition, response_sha256=response_hash)
    fit_seeds = _fit_seeds(seeds)
    if args.workers < 1 or args.workers > len(fit_seeds):
        raise ValueError("workers must be between one and four")

    per_topology = {}
    kwargs = dict(
        artifact_root=args.artifact_root.resolve(), transition_path=args.transition_config.resolve(),
        rev20=rev20, candidates=response["candidates"], started=started,
    )
    if args.workers == 1:
        for seed in fit_seeds:
            key, value = audit_topology(seed, **kwargs)
            per_topology[key] = value
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(audit_topology, seed, **kwargs) for seed in fit_seeds]
            for future in as_completed(futures):
                key, value = future.result()
                per_topology[key] = value

    rows, passed = summarize(per_topology, response["candidates"])
    args.out_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_root / "connectivity_design_audit.csv"
    _write_csv(csv_path, rows)
    failures = [{"candidate_id": row["candidate_id"], "topology_seed": row["topology_seed"]}
                for row in rows if not row["passes"]]
    payload = {
        "schema_id": "topic4_rev22_dci_final_connectivity_design_audit_v1",
        "status": "CONNECTIVITY_DESIGN_ADMISSIBLE" if passed else "CONNECTIVITY_DESIGN_INADMISSIBLE",
        "git_commit": _git_commit(),
        "response_design_manifest_sha256": response_hash,
        "seed_manifest_sha256": sha256(args.seed_manifest),
        "minimal_transition_config_sha256": sha256(args.transition_config),
        "rev20_config_sha256": sha256(args.rev20_config),
        "thresholds": THRESHOLDS,
        "candidate_count": len(response["candidates"]),
        "topology_seeds": fit_seeds,
        "expected_cells": len(response["candidates"]) * len(fit_seeds),
        "observed_cells": len(rows),
        "passing_cells": int(sum(row["passes"] for row in rows)),
        "failed_cells": failures,
        "producer_parity": {str(seed): per_topology[seed]["producer_parity"] for seed in fit_seeds},
        "per_topology": {
            str(seed): {key: value for key, value in per_topology[seed].items() if key != "rows"}
            for seed in fit_seeds
        },
        "table": {"path": str(csv_path), "sha256": sha256(csv_path)},
        "elapsed_seconds": float(time.time() - started),
        "claim_boundary": (
            "Structure-only audit of the exact final ellipse-plus-learned redistribution; "
            "no SNN time stepping, patient validation endpoint, Z/M, or biological synaptic-strength claim."
        ),
    }
    out_json = args.out_root / "connectivity_design_audit.json"
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": payload["status"], "cells": payload["observed_cells"],
        "passing": payload["passing_cells"], "failures": len(failures),
        "elapsed_seconds": round(payload["elapsed_seconds"]), "output": str(out_json),
    }, indent=2))


if __name__ == "__main__":
    main()
