#!/usr/bin/env python3
"""Fail-closed audit of directed-lineage event and contact-readout artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def audit_worker(npz_path: Path, json_path: Path, *,
                 minimum_parity: float) -> dict:
    payload = json.loads(json_path.read_text())
    if payload["arrays"]["sha256"] != _sha256(npz_path):
        raise RuntimeError(f"worker array hash changed: {npz_path}")
    event_unit = payload["event_unit"]
    if (event_unit.get("name") != "directed_spatiotemporal_lineage"
            or bool(event_unit.get("contact_geometry_used_for_boundary"))):
        raise RuntimeError("worker is not contact-independent directed lineage")
    readout = payload.get("contact_readout", {})
    if readout.get("source") != "lineage_restricted_sheet_activity":
        raise RuntimeError("worker score can still see concurrent-root contact activity")
    parity = np.asarray(readout["full_trace_pearson_per_contact"], float)
    if (len(parity) != 15 or not np.all(np.isfinite(parity))
            or np.min(parity) < minimum_parity
            or readout.get("parity_status") != "PASS"):
        raise RuntimeError("binned contact sampler parity failed")
    represented = sum(
        len(row["detector_fragment_indices"]) for row in payload["events"]
    ) + len(event_unit["compound_fragments"])
    if represented != int(event_unit["raw_detector_fragment_count"]):
        raise RuntimeError("directed event partition loses detector fragments")
    with np.load(npz_path, allow_pickle=False) as loaded:
        onsets = np.asarray(loaded["onsets"], float)
        ranks = np.asarray(loaded["ranks"], float)
        returned = np.asarray(loaded["event_returned"], bool)
        edge = np.asarray(loaded["edge_coefficients"], float)
    if onsets.shape != ranks.shape or onsets.shape != (len(payload["events"]), 15):
        raise RuntimeError("lineage contact arrays do not align with events")
    if not np.array_equal(np.isfinite(onsets), np.isfinite(ranks)):
        raise RuntimeError("lineage contact ranks contain phantom participants")
    if np.any(edge != 0.0):
        raise RuntimeError("Node-only historical rescore contains an active edge")
    return {
        "candidate_id": payload["candidate_id"],
        "seed": int(payload["seed"]),
        "n_events": len(payload["events"]),
        "n_returned": int(np.sum(returned)),
        "minimum_full_trace_pearson": float(np.min(parity)),
        "median_full_trace_pearson": float(np.median(parity)),
        "compound_fragment_fraction": float(
            event_unit["compound_detector_fragment_fraction"]
        ),
        "contact_readout_source": readout["source"],
        "fragment_partition_exact": True,
        "edge_coefficients_all_zero": True,
        "npz_sha256": _sha256(npz_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()
    artifact_root = args.artifact_root.resolve()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    manifest_path = artifact_root / config["candidate_manifest"]
    manifest = json.loads(manifest_path.read_text())
    if manifest["config_sha256"] != _sha256(config_path):
        raise RuntimeError("lineage-restricted manifest is stale")
    minimum_parity = float(
        config["search"]["contact_readout"]["minimum_full_trace_pearson"]
    )
    seeds = [int(seed) for seed in config["search"]["fit_network_seeds"]]
    output_root = artifact_root / config["output_root"]
    rows = []
    for candidate in manifest["candidates"]:
        for seed in seeds:
            stem = f"{candidate['candidate_id']}_seed_{seed}"
            rows.append(audit_worker(
                output_root / "workers" / f"{stem}.npz",
                output_root / "workers" / f"{stem}.json",
                minimum_parity=minimum_parity,
            ))
    expected = len(manifest["candidates"]) * len(seeds)
    if len(rows) != expected:
        raise RuntimeError("lineage-restricted library is incomplete")
    aggregate_path = output_root / "aggregate/fit_cascade_summary.json"
    aggregate = json.loads(aggregate_path.read_text())
    if len(aggregate.get("rows", [])) != len(manifest["candidates"]):
        raise RuntimeError("lineage-restricted aggregate is incomplete")
    payload = {
        "schema_id": "topic4_rev12_lineage_restricted_readout_audit_v1",
        "status": "REV12ND_LINEAGE_RESTRICTED_READOUT_AUDIT_COMPLETE",
        "n_expected_workers": expected,
        "n_audited_workers": len(rows),
        "n_candidates": len(manifest["candidates"]),
        "minimum_full_trace_pearson_observed": min(
            row["minimum_full_trace_pearson"] for row in rows
        ),
        "minimum_full_trace_pearson_required": minimum_parity,
        "rows": rows,
        "inputs": {
            "config": str(config_path), "config_sha256": _sha256(config_path),
            "manifest": str(manifest_path), "manifest_sha256": _sha256(manifest_path),
            "aggregate": str(aggregate_path), "aggregate_sha256": _sha256(aggregate_path),
        },
        "claim_boundary": (
            "Engineering and event-identity audit only. Passing does not establish "
            "patient event-distribution reconstruction."
        ),
    }
    output = output_root / "aggregate/lineage_restricted_readout_audit.json"
    _atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"],
        "n_workers": len(rows),
        "minimum_parity": payload["minimum_full_trace_pearson_observed"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
