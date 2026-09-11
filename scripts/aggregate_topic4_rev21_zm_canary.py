#!/usr/bin/env python3
"""Summarize rev21 Z/M engineering canaries without patient ictal inputs."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


ELIGIBLE = "MODEL_ICTAL_ELIGIBLE_REV21"


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    os.close(handle)
    try:
        Path(temporary).write_text(json.dumps(
            payload, indent=2, sort_keys=True, allow_nan=False,
        ) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def summarize(rows: list[dict]) -> dict:
    candidates = []
    for row in rows:
        state = row["model_ictal_rev21"]
        returned = sum(bool(event["returned"]) for event in row["events"])
        candidates.append({
            "candidate_id": row["candidate_id"],
            "topology_seed": row["topology_seed"],
            "dynamics_seed": row["dynamics_seed"],
            "model_ictal_status": state["status"],
            "eligible": state.get("eligible"),
            "operational_onset_ms": row["simulation"]["runaway_early_stop_ms"],
            "scientific_onset_ms": state.get("scientific_onset_ms"),
            "failing_clauses": state.get("failing_clauses", []),
            "n_returned_pretransition_families": returned,
            "formal_interictal_stop_ms": row["simulation"].get(
                "formal_interictal_stop_ms"),
        })
    eligible = [row for row in candidates if row["model_ictal_status"] == ELIGIBLE]
    return {
        "status": ("REV21_CANARY_SUPPORTS_COARSE_SCREEN" if eligible
                   else "REV21_CANARY_HAS_NO_ELIGIBLE_STATE"),
        "candidate_count": len(candidates),
        "eligible_candidate_ids": [row["candidate_id"] for row in eligible],
        "candidates": candidates,
        "patient_ictal_inputs_read": False,
        "interpretation_boundary": (
            "single-cell engineering canary only; absence is not a multi-seed "
            "negative result"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path,
                        default=Path("/home/honglab/leijiaxin/HFOsp"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = args.artifact_root / config["output_root"] / "canary"
    controller = json.loads((root / "status/controller.json").read_text())
    if controller.get("status") != "COMPLETE":
        raise RuntimeError("canary controller is not complete")
    rows = [json.loads(path.read_text())
            for path in sorted((root / "workers").glob("*.json"))]
    payload = summarize(rows)
    output = root / "aggregate.json"
    _atomic_json(output, payload)
    print(json.dumps({"status": payload["status"], "output": str(output)},
                     indent=2))


if __name__ == "__main__":
    main()
