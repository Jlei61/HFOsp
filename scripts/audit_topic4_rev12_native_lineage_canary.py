#!/usr/bin/env python3
"""Compare the native causal-lineage worker against the frozen Stage-G rescore."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
STAGE_G = Path(
    "results/topic4_sef_hfo/data_driven_node_dualmode_rev12/"
    "node_stage_g_lineage_restricted_readout_rescore"
)


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


def compare_shared_arrays(native: dict, historical: dict) -> list[str]:
    required = (
        "contact_names", "shaft_ids", "contact_xy_mm", "onsets", "ranks",
        "event_t_on_ms", "event_t_off_ms", "event_trigger_t_on_ms",
        "event_returned", "event_fragment_count", "active_fraction",
        "active_fraction_bin_ms", "contact_envelope", "contact_envelope_dt_ms",
        "sheet_activity_counts", "sheet_activity_frame_ms",
        "directed_lineage_labels", "directed_lineage_collision_mask",
        "detector_fragment_dominant_lineage_id", "detector_fragment_dominance",
        "detector_fragment_collision_fraction", "detector_fragment_compound",
        "source_onset_maps_ms", "source_onset_evaluable", "source_bin_mm",
        "source_sheet_mm", "positions_E", "h", "delta_vtheta",
        "edge_coefficients",
    )
    mismatches = []
    for key in required:
        if key not in native or key not in historical:
            mismatches.append(f"missing:{key}")
            continue
        left, right = np.asarray(native[key]), np.asarray(historical[key])
        equal = (
            np.array_equal(left, right, equal_nan=True)
            if left.dtype.kind in "fc" or right.dtype.kind in "fc"
            else np.array_equal(left, right)
        )
        if left.shape != right.shape or not equal:
            mismatches.append(key)
    return mismatches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = args.artifact_root.resolve()
    output = root / config["output_root"]
    candidate = config["field_search"]["candidate_ids"][0]
    seed = int(config["search"]["canary_network_seeds"][0])
    stem = f"{candidate}_seed_{seed}"
    native_json = output / "workers" / f"{stem}.json"
    native_npz = output / "workers" / f"{stem}.npz"
    historical_json = root / STAGE_G / "workers" / f"{stem}.json"
    historical_npz = root / STAGE_G / "workers" / f"{stem}.npz"
    for path in (native_json, native_npz, historical_json, historical_npz):
        if not path.exists():
            raise RuntimeError(f"canary parity input is absent: {path}")
    native_payload = json.loads(native_json.read_text())
    historical_payload = json.loads(historical_json.read_text())
    with np.load(native_npz, allow_pickle=False) as loaded:
        native = {key: np.asarray(loaded[key]) for key in loaded.files}
    with np.load(historical_npz, allow_pickle=False) as loaded:
        historical = {key: np.asarray(loaded[key]) for key in loaded.files}
    mismatches = compare_shared_arrays(native, historical)
    if mismatches:
        raise RuntimeError(f"native worker differs from Stage G: {mismatches}")
    if native_payload["contact_readout"]["source"] != (
            "lineage_restricted_sheet_activity"):
        raise RuntimeError("native worker did not use root-restricted readout")
    if not native_payload["mechanism_freeze"]["edge_coefficients_all_zero"]:
        raise RuntimeError("canary contains non-Node mechanism coefficients")
    event_keys = (
        "t_on_ms", "t_off_ms", "trigger_t_on_ms", "trigger_t_off_ms",
        "returned", "n_recruited_contacts", "cascade_id",
    )
    native_events = [
        {key: row.get(key) for key in event_keys}
        for row in native_payload["events"]
    ]
    historical_events = [
        {key: row.get(key) for key in event_keys}
        for row in historical_payload["events"]
    ]
    if native_events != historical_events:
        raise RuntimeError("native event metadata differs from Stage G")
    payload = {
        "schema_id": "topic4_rev12_nd_native_lineage_canary_audit_v1",
        "status": "REV12ND_NATIVE_LINEAGE_CANARY_PARITY_COMPLETE",
        "candidate_id": candidate,
        "seed": seed,
        "shared_arrays_exact": True,
        "event_metadata_exact": True,
        "n_events": len(native_events),
        "n_returned": int(np.sum(native["event_returned"])),
        "contact_readout": native_payload["contact_readout"],
        "mechanism_freeze": native_payload["mechanism_freeze"],
        "native_npz_sha256": _sha256(native_npz),
        "historical_npz_sha256": _sha256(historical_npz),
    }
    destination = output / "aggregate" / "native_lineage_canary_audit.json"
    _atomic_json(destination, payload)
    print(json.dumps({"status": payload["status"], "output": str(destination)}, indent=2))


if __name__ == "__main__":
    main()
