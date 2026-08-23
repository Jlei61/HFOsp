#!/usr/bin/env python3
"""Audit exact-neuron lineage readout while preserving the frozen trajectory."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
REFERENCE = Path(
    "results/topic4_sef_hfo/data_driven_node_dualmode_rev12/"
    "node_stage_g_lineage_restricted_readout_rescore/workers/"
    "stage_c_r04_p_seed_2201"
)


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


def _equal(left: np.ndarray, right: np.ndarray) -> bool:
    if left.shape != right.shape:
        return False
    if left.dtype.kind in "fc" or right.dtype.kind in "fc":
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = args.artifact_root.resolve()
    output_root = root / config["output_root"]
    stem = "stage_i_anchor_00_seed_2201"
    native_json = output_root / "workers" / f"{stem}.json"
    native_npz = output_root / "workers" / f"{stem}.npz"
    reference_json = root / REFERENCE.with_suffix(".json")
    reference_npz = root / REFERENCE.with_suffix(".npz")
    native_payload = json.loads(native_json.read_text())
    reference_payload = json.loads(reference_json.read_text())
    with np.load(native_npz, allow_pickle=False) as loaded:
        native = {key: np.asarray(loaded[key]) for key in loaded.files}
    with np.load(reference_npz, allow_pickle=False) as loaded:
        reference = {key: np.asarray(loaded[key]) for key in loaded.files}
    trajectory_keys = (
        "contact_names", "shaft_ids", "contact_xy_mm", "event_t_on_ms",
        "event_t_off_ms", "event_trigger_t_on_ms", "event_returned",
        "event_fragment_count", "active_fraction", "active_fraction_bin_ms",
        "contact_envelope", "contact_envelope_dt_ms", "sheet_activity_counts",
        "sheet_activity_frame_ms", "directed_lineage_labels",
        "directed_lineage_collision_mask", "detector_fragment_dominant_lineage_id",
        "detector_fragment_dominance", "detector_fragment_collision_fraction",
        "detector_fragment_compound", "source_onset_maps_ms",
        "source_onset_evaluable", "source_bin_mm", "source_sheet_mm",
        "positions_E", "h", "delta_vtheta", "edge_coefficients",
    )
    mismatches = [
        key for key in trajectory_keys
        if key not in native or key not in reference
        or not _equal(native[key], reference[key])
    ]
    if mismatches:
        raise RuntimeError(f"exact-neuron canary changed frozen trajectory: {mismatches}")
    if native_payload["contact_readout"] != {
        "source": "lineage_restricted_neuron_activity",
        "kernel_width_mm": 0.25,
        "smooth_ms": 5.0,
        "spatial_sampler": "exact_normalized_per_neuron_gaussian",
        "root_assignment": "movie_lineage_label_at_each_neuron_bin_and_frame",
        "parity_status": "EXACT_SHARED_PER_NEURON_KERNEL",
    }:
        raise RuntimeError("exact-neuron contact readout metadata drifted")
    native_roots = [row["lineage_id"] for row in native_payload["events"]]
    reference_roots = [row["lineage_id"] for row in reference_payload["events"]]
    if native_roots != reference_roots:
        raise RuntimeError("exact-neuron canary changed directed root identity")
    native_mask = np.isfinite(native["onsets"])
    reference_mask = np.isfinite(reference["onsets"])
    union = np.sum(native_mask | reference_mask, axis=1)
    intersection = np.sum(native_mask & reference_mask, axis=1)
    jaccard = np.divide(
        intersection, union, out=np.ones(len(union), float), where=union > 0,
    )
    payload = {
        "schema_id": "topic4_rev12_exact_neuron_lineage_canary_audit_v1",
        "status": "REV12ND_EXACT_NEURON_LINEAGE_CANARY_COMPLETE",
        "trajectory_arrays_exact": True,
        "event_boundaries_exact": True,
        "directed_root_ids_exact": True,
        "n_events": int(len(native["onsets"])),
        "n_returned": int(np.sum(native["event_returned"])),
        "contact_readout": native_payload["contact_readout"],
        "recruitment_mask_jaccard_vs_binned": {
            "median": float(np.median(jaccard)),
            "minimum": float(np.min(jaccard)),
            "different_event_count": int(np.sum(jaccard < 1.0)),
            "interpretation": "diagnostic only; the exact per-neuron readout is primary",
        },
        "mechanism_freeze": native_payload["mechanism_freeze"],
    }
    destination = output_root / "aggregate" / "exact_neuron_lineage_canary_audit.json"
    _atomic_json(destination, payload)
    print(json.dumps({"status": payload["status"], "output": str(destination)}, indent=2))


if __name__ == "__main__":
    main()
