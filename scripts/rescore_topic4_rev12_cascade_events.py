#!/usr/bin/env python3
"""Rescore fresh canaries using contact-independent cascade event windows."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_topic4_rev12_event_fragmentation import _natural_summary  # noqa: E402
from scripts.rescore_topic4_rev12_node_historical import (  # noqa: E402
    DEFAULT_ARTIFACT_ROOT,
    _classifier_contract,
    _formal_clean_mask,
    _old_to_patient_label_map,
    _patient_data,
    _reorder_patient_contract,
    score_candidate,
)
from scripts.run_topic4_rev10_sa_spectral_field_worker import _contact_onsets  # noqa: E402
from src.sef_hfo_events import detect_events  # noqa: E402
from src.topic4_node_dualmode import (  # noqa: E402
    assign_detector_fragments_to_cascades,
    cascade_event_windows,
    fixed_projection_matrix,
    spatiotemporal_cascade_labels,
)
from src.topic4_shaft_aware_direction import assign_direction_modes  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(handle)
    try:
        Path(temporary).write_text(
            json.dumps(_jsonable(payload), indent=2, allow_nan=False) + "\n"
        )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _cascade_worker(npz_path: Path, json_path: Path,
                    target_names: np.ndarray, classifier: dict,
                    label_map: np.ndarray, *, minimum_active_neurons: int,
                    minimum_dominance: float, readout: dict) -> tuple[dict, dict]:
    payload = json.loads(json_path.read_text())
    with np.load(npz_path, allow_pickle=False) as loaded:
        source_names = np.asarray(loaded["contact_names"]).astype(str)
        envelope = np.asarray(loaded["contact_envelope"], float)
        envelope_dt = float(loaded["contact_envelope_dt_ms"])
        active = np.asarray(loaded["active_fraction"], float)
        active_dt = float(loaded["active_fraction_bin_ms"])
        movie = np.asarray(loaded["sheet_activity_counts"])
        frame_ms = float(loaded["sheet_activity_frame_ms"])
    fragments = detect_events(
        active, active_dt,
        event_on_frac=float(payload["event_unit"]["event_on_threshold"]),
    )
    cascade_result = spatiotemporal_cascade_labels(
        movie, minimum_active_neurons=minimum_active_neurons,
    )
    assignments = assign_detector_fragments_to_cascades(
        movie, cascade_result["labels"], fragments, frame_ms=frame_ms,
        minimum_dominance=minimum_dominance,
    )
    events, compounds = cascade_event_windows(
        cascade_result["components"], assignments, fragments,
        frame_ms=frame_ms, total_ms=len(active) * active_dt,
    )
    events = [event for event in events if event["returned"]]
    order = np.asarray([
        int(np.flatnonzero(source_names == name)[0]) for name in target_names
    ])
    montage = SimpleNamespace(names=source_names)
    onsets, ranks = [], []
    for event in events:
        onset, rank = _contact_onsets(
            envelope, envelope_dt, montage, np.ones(len(source_names), bool),
            (float(event["t_on"]), float(event["t_off"])),
            float(readout["participation_margin_fraction"]),
            float(readout["timing_fraction"]),
        )
        onsets.append(np.asarray(onset, float)[order])
        ranks.append(np.asarray(rank, float)[order])
    onsets = np.asarray(onsets, float).reshape((-1, len(target_names)))
    ranks = np.asarray(ranks, float).reshape((-1, len(target_names)))
    assigned = assign_direction_modes(
        onsets, groups=classifier["groups"], embedding=classifier["embedding"],
        classifier=classifier["classifier"],
    )
    labels = np.asarray(label_map, int)[np.asarray(assigned["labels"], int)]
    ood = np.asarray(assigned["ood"], bool)
    formal_clean = _formal_clean_mask(onsets, ood, classifier["groups"])
    worker = {
        "seed": int(payload["seed"]), "ranks": ranks, "onsets": onsets,
        "labels": labels, "ood": ood, "formal_clean": formal_clean,
        "n_detected": int(len(events)), "n_returned": int(len(events)),
        "duration_ms": float(payload["simulation"]["duration_ms"]),
        "npz": str(npz_path), "npz_sha256": _sha256(npz_path),
    }
    diagnostics = {
        "seed": int(payload["seed"]),
        "n_detector_fragments": int(len(fragments)),
        "n_noncompound_cascade_events": int(len(events)),
        "n_compound_detector_fragments": int(len(compounds)),
        "compound_fragment_fraction": float(len(compounds) / max(1, len(fragments))),
        "n_multifragment_cascade_events": int(sum(
            len(event["detector_fragment_indices"]) > 1 for event in events
        )),
        "patient_mode_counts": np.bincount(labels, minlength=2),
        "ood_fraction": float(np.mean(ood)) if len(ood) else None,
        "median_duration_ms": float(np.median([
            event["dur_ms"] for event in events
        ])) if events else None,
    }
    return worker, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
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
        raise RuntimeError("cascade rescore cannot select a Node field")
    cohort = json.loads((ROOT / config["inputs"]["cohort_config"]["path"]).read_text())
    classifier_config = json.loads(
        (ROOT / config["inputs"]["classifier_config"]["path"]).read_text()
    )
    patient = _patient_data(cohort, artifact_root)
    classifier = _classifier_contract(classifier_config, artifact_root)
    semantics = _old_to_patient_label_map(patient, classifier)
    patient = _reorder_patient_contract(patient, classifier["names"])
    projections = fixed_projection_matrix(
        2 * len(patient["contact_names"]), n_directions=64, seed=20260823,
    )
    source_summary = json.loads(
        (artifact_root / config["inputs"]["source_summary"]["path"]).read_text()
    )
    calibration = source_summary["component_calibration"]
    output_root = artifact_root / config["output_root"]
    seeds = [int(seed) for seed in config["search"]["canary_network_seeds"]]
    rows = []
    for threshold in args.minimum_active_neurons:
        for candidate in manifest["candidates"]:
            workers, diagnostics = [], []
            for seed in seeds:
                stem = f"{candidate['candidate_id']}_seed_{seed}"
                worker, diagnostic = _cascade_worker(
                    output_root / "workers" / f"{stem}.npz",
                    output_root / "workers" / f"{stem}.json",
                    patient["contact_names"], classifier,
                    semantics["raw_to_patient"],
                    minimum_active_neurons=int(threshold),
                    minimum_dominance=float(args.minimum_dominance),
                    readout=config["search"]["contact_readout"],
                )
                workers.append(worker)
                diagnostics.append(diagnostic)
            score = score_candidate(
                candidate, workers, patient, projections, calibration,
            )
            rows.append({
                "candidate_id": candidate["candidate_id"],
                "minimum_active_neurons": int(threshold),
                "minimum_dominance": float(args.minimum_dominance),
                "score": score,
                "pooled_natural_kmeans": _natural_summary(workers, 20260823),
                "per_seed": diagnostics,
            })
    payload = {
        "schema_id": "topic4_rev12_cascade_event_rescore_v1",
        "status": "REV12ND_CASCADE_EVENT_RESCORE_COMPLETE",
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
        "selection_forbidden": True,
        "rows": rows,
        "claim_boundary": (
            "Zero-simulation cascade-event diagnosis. Compound fragments are reported "
            "and omitted from directional scoring; no field can be selected from this run."
        ),
    }
    output = args.out or output_root / "aggregate/cascade_event_rescore.json"
    _atomic_json(output, payload)
    print(json.dumps({"status": payload["status"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
