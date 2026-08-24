#!/usr/bin/env python3
"""Invalidate the exact-readout fit whose directed roots had no fast-state memory."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")


def event_identity_summary(payloads: list[dict]) -> dict:
    fragments = np.asarray([
        row["event_unit"]["raw_detector_fragment_count"] for row in payloads
    ], float)
    roots = np.asarray([
        row["event_unit"]["n_directed_roots"] for row in payloads
    ], float)
    compound = np.asarray([
        row["event_unit"]["compound_detector_fragment_fraction"]
        for row in payloads
    ], float)
    durations = np.asarray([
        event["duration_ms"] for row in payloads for event in row["events"]
    ], float)
    if not len(payloads) or not len(durations):
        raise RuntimeError("event-identity invalidation requires completed events")
    return {
        "n_workers": int(len(payloads)),
        "median_detector_fragments_per_worker": float(np.median(fragments)),
        "median_directed_roots_per_worker": float(np.median(roots)),
        "median_compound_fragment_fraction": float(np.median(compound)),
        "median_event_duration_ms": float(np.median(durations)),
        "p95_event_duration_ms": float(np.quantile(durations, 0.95)),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = args.artifact_root.resolve() / config["output_root"]
    manifest = json.loads((root / "candidate_manifest.json").read_text())
    expected = len(manifest["candidates"]) * len(
        config["search"]["fit_network_seeds"]
    )
    paths = sorted((root / "workers").glob("*.json"))
    if len(paths) != expected:
        raise RuntimeError("exact-readout fit inventory is incomplete")
    payloads = [json.loads(path.read_text()) for path in paths]
    for payload in payloads:
        event_unit = payload["event_unit"]
        if event_unit.get("name") != "directed_spatiotemporal_lineage":
            raise RuntimeError("invalidation received another event identity")
        if "forward_parent_frame_gap" in event_unit:
            raise RuntimeError("historical worker unexpectedly recorded parent memory")
        if payload["contact_readout"].get("parity_status") != (
                "EXACT_SHARED_PER_NEURON_KERNEL"):
            raise RuntimeError("event-identity audit is not the exact-readout fit")
    aggregate = root / "aggregate" / "fit_cascade_summary.json"
    if not aggregate.exists():
        raise RuntimeError("diagnostic fit aggregate is absent")
    summary = event_identity_summary(payloads)
    payload = {
        "schema_id": "topic4_rev12_exact_fit_event_identity_invalidation_v1",
        "status": "INVALIDATED_IMMEDIATE_FRAME_EVENT_IDENTITY",
        **summary,
        "declared_but_unimplemented_config": {
            "forward_parent_frame_gap": config["event_unit"].get(
                "forward_parent_frame_gap"
            ),
            "forward_parent_neighborhood_bins": config["event_unit"].get(
                "forward_parent_neighborhood_bins"
            ),
        },
        "scientific_reason": (
            "Directed ancestry inspected only the immediately preceding 2 ms frame. "
            "A local subthreshold gap therefore created a new root even while the "
            "same membrane, synaptic and delayed-network state remained active. "
            "Field ranking can consequently depend on accidental event splitting."
        ),
        "diagnostic_only": (
            "All 108 exact-readout trajectories and their aggregate are retained, "
            "but no candidate may enter selection or confirmation."
        ),
        "replacement": (
            "Persistent directed roots use local nearest-parent ancestry until the "
            "frozen fast-state memory expires; contacts retain exact per-neuron "
            "root-restricted sampling."
        ),
    }
    output = root / "aggregate" / "fit_event_identity_invalidation.json"
    _atomic_json(output, payload)
    print(json.dumps({"status": payload["status"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
