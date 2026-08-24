#!/usr/bin/env python3
"""Correct event-wise direction clipping in frozen rev12 aggregate artifacts."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

from scripts.aggregate_topic4_rev12_cascade_fit import (
    cascade_selection_objective,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(artifact_root: Path, relative: str) -> Path:
    local = ROOT / relative
    return local if local.exists() else artifact_root / relative


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


def corrected_equal_network_direction(bundle: dict, *, value_key: str) -> dict:
    output = copy.deepcopy(bundle)
    network_scores = []
    for network in output["per_network"]:
        mode_scores = []
        for mode in ("0", "1"):
            signed = float(network["modes"][mode][value_key])
            score = max(0.0, signed)
            network["modes"][mode]["alignment_score"] = score
            mode_scores.append(score)
        network["score"] = float(min(mode_scores))
        network["within_mode_aggregation"] = (
            "clip_after_mode_mean_signed_direction"
        )
        network_scores.append(network["score"])
    output["score"] = (
        float(sum(network_scores) / len(network_scores))
        if network_scores else 0.0
    )
    output["within_mode_aggregation"] = (
        "clip_after_mode_mean_signed_direction"
    )
    return output


def rescore_payload(payload: dict) -> dict:
    output = copy.deepcopy(payload)
    for row in output["rows"]:
        old_selection = copy.deepcopy(row["selection_objective"])
        direction = corrected_equal_network_direction(
            row["causal_direction_alignment"],
            value_key="mean_signed_axis_cosine",
        )
        row["causal_direction_alignment"] = direction
        if "causal_wave_monotonicity" in row:
            row["causal_wave_monotonicity"] = corrected_equal_network_direction(
                row["causal_wave_monotonicity"],
                value_key="mean_signed_axis_time_spearman",
            )
        row["retrospective_invalid_eventwise_direction_objective"] = old_selection
        weights = old_selection["weights"]
        row["selection_objective"] = cascade_selection_objective(
            patient_loss=float(old_selection["matched_patient_loss"]),
            kmeans_balanced_alignment=(
                1.0 - float(old_selection["kmeans_direction_loss"])
            ),
            ood_fraction=float(old_selection["ood_fraction"]),
            compound_fraction=float(old_selection["compound_fraction"]),
            k2_support=float(old_selection["k2_support"]),
            k2_support_weight=float(weights["k2_support_loss"]),
            causal_direction_score=float(direction["score"]),
            causal_direction_weight=float(weights["causal_direction_loss"]),
        )
    output["rows"].sort(key=lambda row: (
        not bool(row.get("selection_eligible", True)),
        row["selection_objective"]["objective"],
        row["candidate_id"],
    ))
    output["schema_id"] = "topic4_rev12_mode_mean_direction_rescore_v1"
    output["status"] = "REV12ND_MODE_MEAN_DIRECTION_RESCORE_COMPLETE"
    output["selection_contract"]["causal_direction"] = (
        "signed event direction is averaged within patient-labelled mode before "
        "clipping; weakest mode within network; equal network weight"
    )
    output["selection_contract"]["supersedes_eventwise_clipping"] = True
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    expected = subprocess.check_output(
        ["git", "rev-parse", args.expected_commit], cwd=ROOT, text=True,
    ).strip()
    if subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip() != expected:
        raise RuntimeError("direction rescore is not at the expected commit")
    tracked = [
        str(config_path.relative_to(ROOT)),
        str(Path(__file__).resolve().relative_to(ROOT)),
        "scripts/aggregate_topic4_rev12_cascade_fit.py",
        "src/topic4_node_dualmode.py",
    ]
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", *tracked], cwd=ROOT, text=True,
    ).strip():
        raise RuntimeError("direction rescore runtime paths are dirty")
    artifact_root = args.artifact_root.resolve()
    outputs = {}
    for stage, record in config["inputs"].items():
        path = _resolve(artifact_root, record["path"])
        if _sha256(path) != record["sha256"]:
            raise RuntimeError(f"direction rescore input changed: {record['path']}")
        payload = rescore_payload(json.loads(path.read_text()))
        payload["direction_rescore"] = {
            "source": str(path), "source_sha256": _sha256(path),
            "config": str(config_path.relative_to(ROOT)),
            "config_sha256": _sha256(config_path),
            "git_commit": expected,
            **config["correction"],
        }
        output = artifact_root / record["output"]
        _atomic_json(output, payload)
        outputs[stage] = str(output)
    print(json.dumps({
        "status": "REV12ND_MODE_MEAN_DIRECTION_RESCORE_COMPLETE",
        "outputs": outputs,
    }, indent=2))


if __name__ == "__main__":
    main()
