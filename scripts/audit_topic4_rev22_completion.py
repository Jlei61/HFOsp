#!/usr/bin/env python3
"""Fail-closed completion audit for the rev22-DCI interictal experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STAGE = Path(
    "/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/"
    "data_driven_dual_core_interictal_identifiability"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing required artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"artifact is not a JSON object: {path}")
    return value


def _controller(stage: Path, phase: str, expected: int) -> dict[str, Any]:
    path = stage / phase / "status/controller.json"
    payload = _read(path)
    if payload.get("status") != "COMPLETE" or int(payload.get("job_count", -1)) != expected:
        raise RuntimeError(f"{phase} controller is incomplete")
    if int((payload.get("state_counts") or {}).get("complete", -1)) != expected:
        raise RuntimeError(f"{phase} controller inventory is incomplete")
    return {"path": str(path), "sha256": _sha256(path), "job_count": expected}


def _aggregate(path: Path, schema: str, status: str) -> dict[str, Any]:
    payload = _read(path)
    if payload.get("schema_id") != schema or payload.get("status") != status:
        raise RuntimeError(f"aggregate contract failed: {path}")
    return {"path": str(path), "sha256": _sha256(path), "status": status}


def _verify_figure_metadata(path: Path) -> dict[str, Any]:
    payload = _read(path)
    output_hashes = payload.get("output_sha256") or {}
    if not output_hashes:
        raise RuntimeError(f"figure metadata has no output hashes: {path}")
    for name, expected in output_hashes.items():
        output = path.parent / name
        if not output.is_file() or _sha256(output) != expected:
            raise RuntimeError(f"figure output hash mismatch: {output}")
    return {"path": str(path), "sha256": _sha256(path), "outputs": len(output_hashes)}


def audit(stage: Path, worktree: Path) -> dict[str, Any]:
    frozen_path = stage / "response_fit/frozen_candidates.json"
    frozen = _read(frozen_path)
    if frozen.get("status") != "REV22_CANDIDATES_FROZEN":
        raise RuntimeError("candidate freeze is incomplete")
    candidate_ids = list(map(str, frozen.get("candidate_ids") or []))
    if not candidate_ids:
        raise RuntimeError("candidate freeze is empty")
    masks = frozen.get("mask_to_candidates") or {}
    final_ids = list(map(str, masks.get("M1111") or masks.get("M1100") or []))
    if not final_ids:
        raise RuntimeError("no full-model candidate was frozen")

    controllers = {
        "fit": _controller(stage, "fit", 384),
        "decomposition": _controller(stage, "decomposition", 16),
        "qualification": _controller(stage, "qualification", 6 * len(candidate_ids)),
        "confirmation": _controller(stage, "confirmation", 12 * len(candidate_ids)),
        "structural_nulls": _controller(stage, "structural_nulls", 108),
    }
    aggregates = {
        "fit": _aggregate(
            stage / "fit/aggregate/fit_aggregate.json",
            "topic4_rev22_dci_training_only_fit_aggregate_v2", "FIT_AGGREGATE_COMPLETE"),
        "response": _aggregate(
            stage / "response_fit/response_fit.json",
            "topic4_rev22_dci_response_fit_v1", "RESPONSE_FIT_COMPLETE"),
        "frozen_stage": _aggregate(
            stage / "frozen_stage_aggregate/frozen_stage_aggregate.json",
            "topic4_rev22_dci_training_only_frozen_stages_v1",
            "FROZEN_STAGE_AGGREGATE_COMPLETE"),
        "validation": _aggregate(
            stage / "validation/validation_aggregate.json",
            "topic4_rev22_dci_validation_aggregate_v1", "VALIDATION_AGGREGATE_COMPLETE"),
        "validation_response": _aggregate(
            stage / "validation/validation_response_surface.json",
            "topic4_rev22_dci_validation_response_surface_v1",
            "DESCRIPTIVE_VALIDATION_RESPONSE_COMPLETE"),
        "structural_nulls": _aggregate(
            stage / "structural_nulls/structural_null_aggregate.json",
            "topic4_rev22_dci_structural_null_aggregate_v1",
            "STRUCTURAL_NULL_AGGREGATE_COMPLETE"),
    }

    figures = stage / "figures"
    figure_records = {"main": _verify_figure_metadata(figures / "metadata.json")}
    objective_metadata = (
        figures / "objective_qualification" /
        "rev22_dci_objective_qualification_controls_metadata.json"
    )
    figure_records["objective_qualification"] = _verify_figure_metadata(objective_metadata)
    if not (figures / "objective_qualification/README.md").is_file():
        raise RuntimeError("objective-qualification figure README is missing")
    structural = figures / "structural_nulls/rev22_dci_structural_controls"
    for suffix in ("png", "pdf", "svg"):
        if not structural.with_suffix(f".{suffix}").is_file():
            raise RuntimeError(f"missing structural-null figure: {suffix}")
    if not (figures / "structural_nulls/README.md").is_file():
        raise RuntimeError("structural-null figure README is missing")
    figure_records["structural_nulls"] = {
        suffix: _sha256(structural.with_suffix(f".{suffix}"))
        for suffix in ("png", "pdf", "svg")
    }
    for candidate_id in final_ids:
        path = (
            figures / "final_candidate" / candidate_id /
            "rev22_dci_final_candidate_acceptance_metadata.json"
        )
        figure_records[f"final_candidate:{candidate_id}"] = _verify_figure_metadata(path)

    if subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=worktree,
                               text=True).strip():
        raise RuntimeError("postfit worktree is dirty at completion")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=worktree,
                                     text=True).strip()
    return {
        "schema_id": "topic4_rev22_dci_completion_audit_v1",
        "status": "REV22_DCI_COMPLETION_AUDIT_PASS",
        "git_commit": commit,
        "candidate_count": len(candidate_ids),
        "final_candidate_ids": final_ids,
        "controllers": controllers,
        "aggregates": aggregates,
        "figures": figure_records,
        "claim_boundary": (
            "Engineering completion only. This audit proves artifact completeness and hash "
            "lineage; it does not turn development-only scientific endpoints into confirmation."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, default=STAGE)
    parser.add_argument("--worktree", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=STAGE / "audit/completion_audit.json")
    args = parser.parse_args()
    payload = audit(args.stage.resolve(), args.worktree.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
