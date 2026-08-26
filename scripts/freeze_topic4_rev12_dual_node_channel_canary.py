#!/usr/bin/env python3
"""Freeze Stage-AK mean-excitability and signed-dispersion field pairs."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(artifact_root: Path, relative: str) -> Path:
    local = ROOT / relative
    return local if local.exists() else artifact_root / relative


def _mapping_hash(mean_hash: str, dispersion_hash: str, formula: str) -> str:
    payload = json.dumps({
        "mean_field_sha256": mean_hash,
        "dispersion_field_sha256": dispersion_hash,
        "formula": formula,
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def build_candidates(source_manifest: dict, stage_ag: dict, stage_ah: dict,
                     stage_aj: dict, crossing: dict,
                     contract: dict) -> tuple[list[dict], dict]:
    if stage_ag.get("status") != (
        "BROAD_FIELD_SEARCH_PATIENT_DIRECTION_TRADEOFF_PERSISTS_"
        "NO_BALANCED_NODE_CANDIDATE"
    ):
        raise RuntimeError("Stage-AG does not justify a dual-channel canary")
    if stage_ah.get("status") != "SIGNED_DEPTH_SHRINKAGE_DOES_NOT_RESOLVE_TWO_MODE_TRADEOFF":
        raise RuntimeError("Stage-AH does not justify a dual-channel canary")
    if stage_aj.get("status") != "SCALAR_NODE_GAIN_DOES_NOT_RESOLVE_TWO_MODE_TRADEOFF":
        raise RuntimeError("Stage-AJ does not justify a dual-channel canary")
    if crossing.get("status") != "MEAN_GAIN_CROSSING_WITHOUT_NETWORK_ROBUST_CORRIDOR":
        raise RuntimeError("scalar-gain crossing remains unresolved")
    if crossing.get("formal_corridor_gain_interval") is not None:
        raise RuntimeError("a scalar-gain corridor must be resolved first")
    if stage_ag.get("best_soft_objective_candidate") != contract["patient_fit_candidate_id"]:
        raise RuntimeError("patient-fit source candidate changed")
    if stage_ag.get("best_mode_1_direction_candidate") != contract["opposite_direction_candidate_id"]:
        raise RuntimeError("direction source candidate changed")

    source = {row["candidate_id"]: row for row in source_manifest["candidates"]}
    combinations = [
        (*pair, True) for pair in contract["coupled_controls"]
    ] + [
        (*pair, False) for pair in contract["cross_channel_combinations"]
    ]
    candidates, audit_rows = [], []
    for mean_id, dispersion_id, coupled in combinations:
        if mean_id not in source or dispersion_id not in source:
            raise RuntimeError("dual-channel source candidate is absent")
        mean_tag = mean_id.removeprefix("stage_ag_")
        dispersion_tag = dispersion_id.removeprefix("stage_ag_")
        candidate_id = f"stage_ak_mean_{mean_tag}_disp_{dispersion_tag}"
        candidate = copy.deepcopy(source[mean_id])
        dispersion_field = copy.deepcopy(source[dispersion_id]["node_field"])
        mapping_hash = _mapping_hash(
            candidate["node_field"]["field_sha256"],
            dispersion_field["field_sha256"], contract["formula"],
        )
        candidate.update({
            "candidate_id": candidate_id,
            "role": "dual_continuous_node_channel_canary_not_selectable",
            "selection_eligible": False,
            "source_candidate_ids": {
                "mean": mean_id, "dispersion": dispersion_id,
            },
            "node_dispersion_field": dispersion_field,
            "node_mapping": {
                "mapping_type": "dual_continuous_mean_dispersion",
                "signed_depth_shrinkage": 1.0,
                "node_gain": 1.0,
                "mapping_sha256": mapping_hash,
            },
        })
        candidates.append(candidate)
        audit_rows.append({
            "candidate_id": candidate_id,
            "mean_source_candidate_id": mean_id,
            "dispersion_source_candidate_id": dispersion_id,
            "mean_field_sha256": candidate["node_field"]["field_sha256"],
            "dispersion_field_sha256": dispersion_field["field_sha256"],
            "coupled_control": bool(coupled),
            "mapping_sha256": mapping_hash,
        })
    if len(candidates) != int(contract["expected_candidate_count"]):
        raise RuntimeError("dual-channel candidate count changed")
    if len({row["mapping_sha256"] for row in audit_rows}) != len(audit_rows):
        raise RuntimeError("dual-channel mappings are not unique")
    return candidates, {
        "formula": contract["formula"],
        "combinations": audit_rows,
        "patient_heldout_used": False,
        "manual_field_used": False,
        "selection_eligible": False,
    }


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


def _provenance(config_path: Path, expected_commit: str) -> dict:
    expected = subprocess.check_output(
        ["git", "rev-parse", expected_commit], cwd=ROOT, text=True,
    ).strip()
    if subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip() != expected:
        raise RuntimeError("dual-channel freezer is not at expected commit")
    tracked = [
        str(config_path.relative_to(ROOT)),
        str(Path(__file__).resolve().relative_to(ROOT)),
        "src/topic4_core_field_rev9.py",
        "src/topic4_zm_ictal_transition.py",
        "scripts/run_topic4_rev12_node_worker.py",
        "scripts/launch_topic4_rev12_node_workers.py",
        "scripts/aggregate_topic4_rev12_soft_global_fit.py",
    ]
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", *tracked], cwd=ROOT, text=True,
    ).strip():
        raise RuntimeError("dual-channel runtime paths are dirty")
    return {"git_commit": expected, "tracked_modules": tracked, "dirty": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    artifact_root = args.artifact_root.resolve()
    config = json.loads(config_path.read_text())
    loaded, inputs = {}, {}
    for name, record in config["inputs"].items():
        path = _resolve(artifact_root, record["path"])
        observed = _sha256(path)
        if observed != record["sha256"]:
            raise RuntimeError(f"dual-channel input changed: {record['path']}")
        loaded[name] = json.loads(path.read_text())
        inputs[name] = {"path": str(path), "sha256": observed}
    candidates, audit = build_candidates(
        loaded["stage_ag_manifest"], loaded["stage_ag_paired_audit"],
        loaded["stage_ah_result"], loaded["stage_aj_result"],
        loaded["stage_aj_crossing_audit"], config["dual_node_channel"],
    )
    payload = {
        "schema_id": "topic4_rev12_nd_dual_node_channel_manifest_v1",
        "status": "REV12ND_DUAL_NODE_CHANNEL_CANARY_FROZEN",
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "candidates": candidates,
        "mapping_audit": audit,
        "event_unit": config["event_unit"],
        "contact_readout": config["search"]["contact_readout"],
        "soft_objective": config["soft_objective"],
        "pareto_selection": config["pareto_selection"],
        "inputs": inputs,
        "provenance": _provenance(config_path, args.expected_commit),
        "claim_boundary": config["claim_boundary"],
    }
    output = artifact_root / config["candidate_manifest"]
    _atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"], "n_candidates": len(candidates),
        "n_networks": len(config["search"]["fit_network_seeds"]),
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
