#!/usr/bin/env python3
"""Evaluate rev11-NLC with natural KMeans and patient geometry by arm."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_topic4_rev10_d5_3_spatial_ou_kmeans_grid import (  # noqa: E402
    audit_candidate,
)
from scripts.run_topic4_rev9l_forced_source_worker import _sha256  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/topic4_rev11_nlc_local_connectivity_canary.json"


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    os.close(fd)
    try:
        Path(temporary).write_text(json.dumps(
            payload, indent=2, sort_keys=True, allow_nan=False,
        ) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def adjudicate(rows, baseline_id="node_baseline"):
    ranked = sorted(rows, key=lambda row: (
        row["selection_score"] is None,
        np.inf if row["selection_score"] is None else row["selection_score"],
        row["candidate_id"],
    ))
    baseline = next(row for row in rows if row["candidate_id"] == baseline_id)
    evaluable = [row for row in ranked if row["selection_score"] is not None]
    best = evaluable[0] if evaluable else None
    by_arm = {}
    for arm in ("Node", "Node+EE", "Node+EtoI", "Node+EE+EtoI"):
        options = [row for row in evaluable if row["arm"] == arm]
        by_arm[arm] = None if not options else min(
            options, key=lambda row: row["selection_score"],
        )["candidate_id"]
    if best is None:
        status = "REV11NLC_NO_KMEANS_EVALUABLE_LOCAL_CONNECTIVITY_CANDIDATE"
    elif best["candidate_id"] == baseline_id:
        status = "REV11NLC_LOCAL_CONNECTIVITY_DID_NOT_IMPROVE_NODE_CANARY"
    else:
        status = "REV11NLC_LOCAL_CONNECTIVITY_CAPACITY_CANDIDATE_FOUND"
    return {
        "status": status,
        "selected_candidate_id": None if best is None else best["candidate_id"],
        "selected_arm": None if best is None else best["arm"],
        "selected_score": None if best is None else best["selection_score"],
        "baseline_score": baseline["selection_score"],
        "selected_minus_baseline_score": (
            None if best is None or baseline["selection_score"] is None
            else float(best["selection_score"] - baseline["selection_score"])
        ),
        "best_candidate_by_arm": by_arm,
        "candidate_rows": ranked,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    root = ROOT / config["output_root"]
    manifest_path = root / "candidate_manifest.json"
    summary_path = root / "canary_summary_returned_only.json"
    manifest = json.loads(manifest_path.read_text())
    summary = json.loads(summary_path.read_text())
    if manifest.get("status") != "REV11NLC_LOCAL_CONNECTIVITY_LIBRARY_FROZEN":
        raise RuntimeError("rev11-NLC manifest is not frozen")
    if summary.get("status") != "REV11NLC_RETURNED_ONLY_CANARY_COMPLETE":
        raise RuntimeError("rev11-NLC aggregate is incomplete")
    aggregate = {row["candidate_id"]: row for row in summary["candidate_rows"]}
    selection = config["search"]["kmeans_selection"]
    rows = []
    for candidate in manifest["candidate_set"]["candidates"]:
        row = audit_candidate(
            config_path, root, candidate,
            aggregate[candidate["candidate_id"]], selection,
        )
        row["arm"] = candidate["arm"]
        row["coefficients"] = candidate["coefficients"]
        rows.append(row)
    payload = adjudicate(rows)
    payload.update({
        "selection_contract": selection,
        "selection_is_exploratory_not_a_gate": True,
        "Z_M_role": "off during substrate canary; reserved for frozen-substrate ictal transfer",
        "claim_boundary": config["claim_boundary"],
        "inputs": {
            "config": {"path": str(config_path.relative_to(ROOT)), "sha256": _sha256(config_path)},
            "manifest": {"path": str(manifest_path.relative_to(ROOT)), "sha256": _sha256(manifest_path)},
            "summary": {"path": str(summary_path.relative_to(ROOT)), "sha256": _sha256(summary_path)},
        },
    })
    output = root / "canary_verdict.json"
    _atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"],
        "selected_candidate_id": payload["selected_candidate_id"],
        "selected_arm": payload["selected_arm"],
        "selected_score": payload["selected_score"],
        "baseline_score": payload["baseline_score"],
    }, indent=2))


if __name__ == "__main__":
    main()
