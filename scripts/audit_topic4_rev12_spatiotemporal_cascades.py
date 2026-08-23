#!/usr/bin/env python3
"""Audit whether population excursions contain one or multiple cascades."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sef_hfo_events import detect_events  # noqa: E402
from src.topic4_node_dualmode import (  # noqa: E402
    assign_detector_fragments_to_cascades,
    spatiotemporal_cascade_labels,
)


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


def _worker_audit(npz_path: Path, json_path: Path, *,
                  minimum_active_neurons: int,
                  minimum_dominance: float) -> dict:
    payload = json.loads(json_path.read_text())
    with np.load(npz_path, allow_pickle=False) as loaded:
        movie = np.asarray(loaded["sheet_activity_counts"])
        frame_ms = float(loaded["sheet_activity_frame_ms"])
        active = np.asarray(loaded["active_fraction"], float)
        active_dt = float(loaded["active_fraction_bin_ms"])
    fragments = detect_events(
        active, active_dt,
        event_on_frac=float(payload["event_unit"]["event_on_threshold"]),
    )
    cascades = spatiotemporal_cascade_labels(
        movie, minimum_active_neurons=minimum_active_neurons,
    )
    assignments = assign_detector_fragments_to_cascades(
        movie, cascades["labels"], fragments, frame_ms=frame_ms,
        minimum_dominance=minimum_dominance,
    )
    by_fragment = {
        row["detector_fragment_index"]: row for row in assignments
    }
    excursion_rows = []
    for event in payload["events"]:
        indices = [int(index) for index in event["detector_fragment_indices"]]
        assigned = [by_fragment[index] for index in indices]
        cascade_ids = sorted({
            int(row["dominant_cascade_id"]) for row in assigned
            if row["dominant_cascade_id"] is not None
        })
        excursion_rows.append({
            "population_excursion_index": int(event["event_index"]),
            "returned": bool(event["returned"]),
            "n_detector_fragments": len(indices),
            "n_dominant_cascades": len(cascade_ids),
            "dominant_cascade_ids": cascade_ids,
            "n_compound_fragments": int(sum(row["compound"] for row in assigned)),
            "fragment_assignments": assigned,
        })
    masses = np.asarray([
        row["activity_mass"] for row in cascades["components"]
    ], float)
    return {
        "candidate_id": payload["candidate_id"],
        "seed": int(payload["seed"]),
        "minimum_active_neurons": int(minimum_active_neurons),
        "minimum_dominance": float(minimum_dominance),
        "n_detector_fragments": len(fragments),
        "n_spatiotemporal_cascades": len(cascades["components"]),
        "n_fragments_compound": int(sum(row["compound"] for row in assignments)),
        "n_multicascade_population_excursions": int(sum(
            row["n_dominant_cascades"] > 1 for row in excursion_rows
        )),
        "n_returned_multicascade_population_excursions": int(sum(
            row["returned"] and row["n_dominant_cascades"] > 1
            for row in excursion_rows
        )),
        "largest_cascade_mass_fractions": (
            (np.sort(masses)[::-1][:10] / np.sum(masses)).tolist()
            if np.sum(masses) > 0 else []
        ),
        "excursions": excursion_rows,
        "npz": {"path": str(npz_path), "sha256": _sha256(npz_path)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--minimum-active-neurons", type=int, nargs="+", default=(2, 3))
    parser.add_argument("--minimum-dominance", type=float, default=0.7)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    artifact_root = args.artifact_root.resolve()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    manifest_path = artifact_root / config["candidate_manifest"]
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("selection_forbidden") is not True:
        raise RuntimeError("cascade audit cannot run on a selection manifest")
    output_root = artifact_root / config["output_root"]
    seeds = [int(seed) for seed in config["search"]["canary_network_seeds"]]
    rows = []
    for threshold in args.minimum_active_neurons:
        for candidate in manifest["candidates"]:
            for seed in seeds:
                stem = f"{candidate['candidate_id']}_seed_{seed}"
                rows.append(_worker_audit(
                    output_root / "workers" / f"{stem}.npz",
                    output_root / "workers" / f"{stem}.json",
                    minimum_active_neurons=int(threshold),
                    minimum_dominance=float(args.minimum_dominance),
                ))
    payload = {
        "schema_id": "topic4_rev12_spatiotemporal_cascade_audit_v1",
        "status": "REV12ND_SPATIOTEMPORAL_CASCADE_AUDIT_COMPLETE",
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
        "rows": rows,
        "boundary_contract": {
            "contact_geometry_used": False,
            "movie_frame_ms": 2.0,
            "sheet_bin_mm": 1.0,
            "spatial_temporal_neighborhood": "3x3 bins across adjacent frames",
            "axonal_velocity_mm_per_ms": 0.3,
            "interpretation": "operational causal-consistency audit, not synaptic proof",
        },
        "claim_boundary": (
            "This audit can identify compound or disconnected population excursions. "
            "It does not yet provide cascade-specific virtual-contact readout and cannot "
            "restore KMeans optimization."
        ),
    }
    output = args.out or output_root / "aggregate/spatiotemporal_cascade_audit.json"
    _atomic_json(output, payload)
    print(json.dumps({"status": payload["status"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
