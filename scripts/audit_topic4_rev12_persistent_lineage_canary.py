#!/usr/bin/env python3
"""Audit persistent event identity and its fast-state-memory sensitivity."""
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

from scripts.rescore_topic4_rev12_node_historical import (
    _classifier_contract,
    _old_to_patient_label_map,
    _patient_data,
    _reorder_patient_contract,
)
from src.topic4_d6_natural_kmeans import natural_kmeans
from src.topic4_shaft_aware_direction import assign_direction_modes


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
REFERENCE_ROOT = Path(
    "results/topic4_sef_hfo/data_driven_node_dualmode_rev12/"
    "node_stage_k_exact_neuron_causal_field_fit/workers"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def partition_coassignment_jaccard(left: np.ndarray, right: np.ndarray) -> float:
    """Compare event partitions while treating compound fragments as singletons."""
    left = np.asarray(left, int)
    right = np.asarray(right, int)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("fragment partitions must be aligned vectors")
    if len(left) < 2:
        return 1.0
    upper = np.triu(np.ones((len(left), len(left)), bool), 1)
    same_left = (left[:, None] == left[None, :]) & (left[:, None] >= 0) & upper
    same_right = (right[:, None] == right[None, :]) & (right[:, None] >= 0) & upper
    union = np.sum(same_left | same_right)
    return float(np.sum(same_left & same_right) / union) if union else 1.0


def _reorder(values: np.ndarray, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source).astype(str)
    target = np.asarray(target).astype(str)
    if set(source) != set(target):
        raise RuntimeError("canary contact set differs from classifier")
    order = np.asarray([int(np.flatnonzero(source == name)[0]) for name in target])
    return np.asarray(values)[..., order]


def _equal(left: np.ndarray, right: np.ndarray) -> bool:
    if left.shape != right.shape:
        return False
    if left.dtype.kind in "fc" or right.dtype.kind in "fc":
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(handle)
    try:
        Path(temporary).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _safe(value):
    if isinstance(value, dict):
        return {key: _safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, np.generic):
        return _safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()
    artifact_root = args.artifact_root.resolve()
    config = json.loads(args.config.read_text())
    output_root = artifact_root / config["output_root"]
    manifest = json.loads((artifact_root / config["candidate_manifest"]).read_text())
    cohort = json.loads((artifact_root / config["inputs"]["cohort_config"]["path"]).read_text())
    classifier_config = json.loads(
        (artifact_root / config["inputs"]["classifier_config"]["path"]).read_text()
    )
    patient = _patient_data(cohort, artifact_root)
    classifier = _classifier_contract(classifier_config, artifact_root)
    semantics = _old_to_patient_label_map(patient, classifier)
    patient = _reorder_patient_contract(patient, classifier["names"])
    rows = []
    parity_keys = (
        "contact_names", "shaft_ids", "contact_xy_mm", "active_fraction",
        "active_fraction_bin_ms", "contact_envelope", "contact_envelope_dt_ms",
        "sheet_activity_counts", "sheet_activity_frame_ms", "positions_E", "h",
        "delta_vtheta", "edge_coefficients",
    )
    for candidate in manifest["candidates"]:
        for seed in config["search"]["canary_network_seeds"]:
            stem = f"{candidate['candidate_id']}_seed_{seed}"
            json_path = output_root / "workers" / f"{stem}.json"
            npz_path = json_path.with_suffix(".npz")
            reference_path = artifact_root / REFERENCE_ROOT / f"{stem}.npz"
            payload = json.loads(json_path.read_text())
            if payload["event_unit"].get("name") not in {
                    "persistent_directed_spatiotemporal_lineage",
                    "persistent_root_coactivity_episode"}:
                raise RuntimeError("canary worker used another event identity")
            with np.load(npz_path, allow_pickle=False) as loaded:
                arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
            with np.load(reference_path, allow_pickle=False) as loaded:
                reference = {key: np.asarray(loaded[key]) for key in loaded.files}
            mismatches = [
                key for key in parity_keys
                if key not in arrays or key not in reference
                or not _equal(arrays[key], reference[key])
            ]
            if mismatches:
                raise RuntimeError(f"persistent canary changed trajectory: {mismatches}")
            multiples = np.asarray(arrays["lineage_sensitivity_multiples"], float)
            counts = np.asarray(arrays["lineage_sensitivity_event_counts"], int)
            partitions = np.asarray(arrays["lineage_sensitivity_fragment_partition"], int)
            primary_index = int(np.flatnonzero(np.isclose(
                multiples, float(config["event_unit"]["fast_state_decay_multiples"]),
            ))[0])
            sensitivity_rows = []
            for index, multiple in enumerate(multiples):
                count = int(counts[index])
                onsets = _reorder(
                    arrays["lineage_sensitivity_onsets"][index, :count],
                    arrays["contact_names"], classifier["names"],
                )
                ranks = _reorder(
                    arrays["lineage_sensitivity_ranks"][index, :count],
                    arrays["contact_names"], classifier["names"],
                )
                returned = np.asarray(
                    arrays["lineage_sensitivity_returned"][index, :count], bool,
                )
                assigned = assign_direction_modes(
                    onsets, groups=classifier["groups"],
                    embedding=classifier["embedding"],
                    classifier=classifier["classifier"],
                )
                labels = semantics["raw_to_patient"][np.asarray(assigned["labels"], int)]
                selected = returned
                natural = natural_kmeans(
                    ranks[selected], labels[selected],
                    random_state=20260824 + int(seed) + index,
                )
                sensitivity_rows.append({
                    "fast_state_decay_multiples": float(multiple),
                    "n_events": count,
                    "n_returned": int(np.sum(selected)),
                    "mode_counts": np.bincount(labels[selected], minlength=2),
                    "ood_fraction": float(np.mean(
                        np.asarray(assigned["ood"], bool)[selected]
                    )) if np.any(selected) else 1.0,
                    "partition_jaccard_vs_primary": partition_coassignment_jaccard(
                        partitions[index], partitions[primary_index],
                    ),
                    "natural_kmeans": natural,
                })
            runtime_sensitivity = payload["event_unit"].get("memory_sensitivity", [])
            if len(runtime_sensitivity) != len(sensitivity_rows):
                raise RuntimeError("worker sensitivity metadata is incomplete")
            rows.append({
                "candidate_id": candidate["candidate_id"],
                "seed": int(seed),
                "trajectory_exact_vs_immediate_frame_run": True,
                "primary_event_unit": payload["event_unit"],
                "sensitivity": sensitivity_rows,
                "arrays": {"path": str(npz_path), "sha256": _sha256(npz_path)},
            })
    if len(rows) != len(manifest["candidates"]) * len(
            config["search"]["canary_network_seeds"]):
        raise RuntimeError("persistent-lineage canary inventory is incomplete")
    payload = {
        "schema_id": "topic4_rev12_persistent_lineage_canary_audit_v1",
        "status": "REV12ND_PERSISTENT_LINEAGE_CANARY_AUDIT_COMPLETE",
        "n_workers": len(rows),
        "rows": rows,
        "claim_boundary": (
            "Event-identity canary only. Field ranking remains closed; patient "
            "held-out data, EE, E-to-I and Z/M do not select this result."
        ),
    }
    output = output_root / "aggregate" / "persistent_lineage_canary_audit.json"
    _atomic_json(output, _safe(payload))
    print(json.dumps({"status": payload["status"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
