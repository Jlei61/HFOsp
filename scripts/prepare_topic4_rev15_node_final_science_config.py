#!/usr/bin/env python3
"""Freeze the one-time held-out and source-topology audit after post-selection."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_ROBUST_CONFIG = ROOT / "config/topic4_rev15_m3_robust_candidates.json"
DEFAULT_ROBUST_AGGREGATE = ARTIFACT_ROOT / (
    "results/topic4_sef_hfo/data_driven_node_dualmode_rev15/"
    "m3_robust_candidates/analysis/m3_robust_candidates_aggregate.json"
)
DEFAULT_POSTSELECTION_CONFIG = ROOT / "config/topic4_rev15_node_postselection.json"
DEFAULT_POSTSELECTION_AUDIT = ARTIFACT_ROOT / (
    "results/topic4_sef_hfo/data_driven_node_dualmode_rev15/"
    "m3_robust_candidates/postselection/analysis/node_postselection_audit.json"
)
DEFAULT_REV12_CONFIG = ROOT / "config/topic4_rev12_nd_node_dualmode_refit.json"
DEFAULT_OUTPUT = ROOT / "config/topic4_rev15_node_final_science.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(path: Path, *, repository_root: Path, artifact_root: Path) -> dict[str, str]:
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(f"final-science input is missing: {path}")
    for root in (repository_root.resolve(), artifact_root.resolve()):
        try:
            relative = path.relative_to(root)
            return {"path": str(relative), "sha256": _sha256(path)}
        except ValueError:
            continue
    raise RuntimeError(f"final-science input lies outside frozen roots: {path}")


def _resolve(record: Mapping[str, Any], *, repository_root: Path,
             artifact_root: Path) -> Path:
    for root in (artifact_root, repository_root):
        path = root / str(record["path"])
        if path.is_file() and _sha256(path) == str(record["sha256"]):
            return path.resolve()
    raise RuntimeError(f"frozen input changed: {record['path']}")


def build_config(
    *, robust_config_path: Path, robust_aggregate_path: Path,
    postselection_config_path: Path, postselection_audit_path: Path,
    rev12_config_path: Path, repository_root: Path = ROOT,
    artifact_root: Path = ARTIFACT_ROOT,
) -> dict[str, Any]:
    robust_config = json.loads(robust_config_path.read_text())
    robust_aggregate = json.loads(robust_aggregate_path.read_text())
    postselection_config = json.loads(postselection_config_path.read_text())
    postselection = json.loads(postselection_audit_path.read_text())
    if postselection.get("status") != "NODE_POSTSELECTION_ACCEPTED":
        raise RuntimeError("Node post-selection has not been accepted")
    if postselection.get("acceptance", {}).get("accepted") is not True:
        raise RuntimeError("Node post-selection acceptance clauses are incomplete")
    candidate_id = str(postselection["candidate_id"])
    if candidate_id != robust_aggregate.get("best_usable_anchor"):
        raise RuntimeError("post-selection candidate differs from robust ranking")
    if candidate_id != postselection_config["selected_candidate"]["candidate_id"]:
        raise RuntimeError("post-selection candidate identity changed")
    if robust_aggregate.get("status") != "COMPLETE":
        raise RuntimeError("robust aggregate is incomplete")
    manifest_path = artifact_root / robust_config["candidate_manifest"]
    manifest = json.loads(manifest_path.read_text())
    candidates = {row["candidate_id"]: row for row in manifest["candidates"]}
    if candidate_id not in candidates or "exact_off" not in candidates:
        raise RuntimeError("selected or reference field is absent from robust manifest")
    rev12 = json.loads(rev12_config_path.read_text())
    input_paths = {}
    for name in ("cohort_config", "classifier_config"):
        input_paths[name] = _resolve(
            rev12["inputs"][name], repository_root=repository_root,
            artifact_root=artifact_root,
        )
    return {
        "schema_id": "topic4_rev15_node_final_science_v1",
        "scientific_role": (
            "development_only_one_time_heldout_topology_and_intervention_entry"
        ),
        "output_root": (
            "results/topic4_sef_hfo/data_driven_node_dualmode_rev15/"
            "m3_robust_candidates/final_science"
        ),
        "inputs": {
            "robust_config": _record(
                robust_config_path, repository_root=repository_root,
                artifact_root=artifact_root,
            ),
            "robust_manifest": _record(
                manifest_path, repository_root=repository_root,
                artifact_root=artifact_root,
            ),
            "robust_aggregate": _record(
                robust_aggregate_path, repository_root=repository_root,
                artifact_root=artifact_root,
            ),
            "postselection_config": _record(
                postselection_config_path, repository_root=repository_root,
                artifact_root=artifact_root,
            ),
            "postselection_audit": _record(
                postselection_audit_path, repository_root=repository_root,
                artifact_root=artifact_root,
            ),
            "rev12_config": _record(
                rev12_config_path, repository_root=repository_root,
                artifact_root=artifact_root,
            ),
            **{
                name: _record(
                    path, repository_root=repository_root,
                    artifact_root=artifact_root,
                ) for name, path in input_paths.items()
            },
        },
        "selected_candidate": {
            "candidate_id": candidate_id,
            "field_coefficients_sha256": candidates[candidate_id][
                "fourier_coordinate"
            ]["coefficients_sha256"],
            "paired_reference_candidate_id": "exact_off",
        },
        "network_seeds": [2341, 2342, 2343],
        "patient_endpoint": {
            "role": "one_time_development_heldout_evaluation_after_field_freeze",
            "primary_event_set": "complete_returned_causal_families",
            "complete_event_cloud_r2_must_be_positive": True,
            "candidate_must_improve_paired_exact_off": True,
            "both_mode_losses_and_weakest_mode_must_improve": True,
        },
        "source_topology": {
            "event_set": "complete_returned_source_evaluable_causal_families",
            "features": "early_10pct_mask_plus_normalized_local_onset",
            "permutation_draws": 4096,
            "permutation_seed": 20260830,
            "null": "within_network_labels_preserving_mode_occupancy",
            "continuous_quality": (
                "weakest_mode_cross_network_cosine_times_between_mode_distance"
            ),
        },
        "boundaries": {
            "field_reranking_allowed": False,
            "patient_heldout_used_for_field_selection": False,
            "patient_heldout_opened_by_final_audit": True,
            "natural_kmeans_must_already_be_accepted": True,
            "SNN_simulation_run": False,
            "EE_EtoI_ZM": "off",
        },
        "claim_boundary": (
            "This is a one-time development held-out and model-internal topology "
            "audit of one already frozen candidate. It cannot re-rank fields, "
            "establish patient-blind generalization, or activate another mechanism."
        ),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robust-config", type=Path, default=DEFAULT_ROBUST_CONFIG)
    parser.add_argument("--robust-aggregate", type=Path, default=DEFAULT_ROBUST_AGGREGATE)
    parser.add_argument("--postselection-config", type=Path, default=DEFAULT_POSTSELECTION_CONFIG)
    parser.add_argument("--postselection-audit", type=Path, default=DEFAULT_POSTSELECTION_AUDIT)
    parser.add_argument("--rev12-config", type=Path, default=DEFAULT_REV12_CONFIG)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    payload = build_config(
        robust_config_path=args.robust_config.resolve(),
        robust_aggregate_path=args.robust_aggregate.resolve(),
        postselection_config_path=args.postselection_config.resolve(),
        postselection_audit_path=args.postselection_audit.resolve(),
        rev12_config_path=args.rev12_config.resolve(),
        artifact_root=args.artifact_root.resolve(),
    )
    args.output.resolve().write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "status": "REV15_NODE_FINAL_SCIENCE_CONFIG_PREPARED",
        "candidate_id": payload["selected_candidate"]["candidate_id"],
        "output": str(args.output.resolve()),
        "SNN_simulation_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()
