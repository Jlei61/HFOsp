#!/usr/bin/env python3
"""Audit fresh population-excursion canaries without selecting a Node field."""
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
from sklearn.metrics import adjusted_rand_score

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
    fixed_projection_matrix,
    population_excursion_episodes,
)
from src.topic4_shaft_aware_direction import assign_direction_modes  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def fragment_partition(episodes: list[dict], n_fragments: int) -> np.ndarray:
    """Map each high-threshold fragment to exactly one excursion."""
    labels = np.full(int(n_fragments), -1, dtype=int)
    for episode_index, episode in enumerate(episodes):
        for fragment_index in episode["detector_fragment_indices"]:
            fragment_index = int(fragment_index)
            if fragment_index < 0 or fragment_index >= n_fragments:
                raise RuntimeError("excursion references an absent detector fragment")
            if labels[fragment_index] != -1:
                raise RuntimeError("detector fragment belongs to multiple excursions")
            labels[fragment_index] = int(episode_index)
    if n_fragments and np.any(labels < 0):
        raise RuntimeError("population segmentation dropped a detector fragment")
    return labels


def _worker_at_multiple(npz_path: Path, json_path: Path,
                        target_names: np.ndarray, classifier: dict,
                        label_map: np.ndarray, *, multiple: float,
                        low_fraction: float, pre_roll_ms: float,
                        readout: dict) -> tuple[dict, dict]:
    payload = json.loads(json_path.read_text())
    with np.load(npz_path, allow_pickle=False) as loaded:
        source_names = np.asarray(loaded["contact_names"]).astype(str)
        envelope = np.asarray(loaded["contact_envelope"], float)
        envelope_dt = float(loaded["contact_envelope_dt_ms"])
        active = np.asarray(loaded["active_fraction"], float)
        active_dt = float(loaded["active_fraction_bin_ms"])
    runtime = payload["event_unit"]
    detector_fragments = detect_events(
        active, active_dt, event_on_frac=float(runtime["event_on_threshold"]),
    )
    reset_ms = (
        float(multiple) * float(runtime["fast_tau_ms"])
        + float(runtime["maximum_delay_ms"])
    )
    episodes = population_excursion_episodes(
        detector_fragments, active, sample_dt_ms=active_dt,
        event_on_threshold=float(runtime["event_on_threshold"]),
        low_threshold_fraction=float(low_fraction), reset_ms=reset_ms,
        pre_roll_ms=float(pre_roll_ms),
    )
    order = np.asarray([
        int(np.flatnonzero(source_names == name)[0]) for name in target_names
    ])
    montage = SimpleNamespace(names=source_names)
    onsets, ranks, selected = [], [], []
    for episode in episodes:
        if not episode["returned"]:
            continue
        onset, rank = _contact_onsets(
            envelope, envelope_dt, montage, np.ones(len(source_names), bool),
            (float(episode["t_on"]), float(episode["t_off"])),
            float(readout["participation_margin_fraction"]),
            float(readout["timing_fraction"]),
        )
        onsets.append(np.asarray(onset, float)[order])
        ranks.append(np.asarray(rank, float)[order])
        selected.append(episode)
    onsets = np.asarray(onsets, float).reshape((-1, len(target_names)))
    ranks = np.asarray(ranks, float).reshape((-1, len(target_names)))
    assigned = assign_direction_modes(
        onsets, groups=classifier["groups"], embedding=classifier["embedding"],
        classifier=classifier["classifier"],
    )
    labels = np.asarray(label_map, int)[np.asarray(assigned["labels"], int)]
    ood = np.asarray(assigned["ood"], bool)
    formal_clean = _formal_clean_mask(onsets, ood, classifier["groups"])
    fragment_modes = np.full(len(detector_fragments), -1, dtype=int)
    for episode, label in zip(selected, labels):
        fragment_modes[np.asarray(episode["detector_fragment_indices"], int)] = int(label)
    worker = {
        "seed": int(payload["seed"]), "ranks": ranks, "onsets": onsets,
        "labels": labels, "ood": ood, "formal_clean": formal_clean,
        "n_detected": int(len(episodes)), "n_returned": int(len(selected)),
        "duration_ms": float(payload["simulation"]["duration_ms"]),
        "npz": str(npz_path), "npz_sha256": _sha256(npz_path),
    }
    diagnostics = {
        "seed": int(payload["seed"]),
        "reset_ms": reset_ms,
        "n_detector_fragments": int(len(detector_fragments)),
        "n_excursions": int(len(episodes)),
        "n_returned_excursions": int(len(selected)),
        "n_multifragment_excursions": int(sum(
            episode["fragment_count"] > 1 for episode in episodes
        )),
        "maximum_fragments_per_excursion": int(max(
            (episode["fragment_count"] for episode in episodes), default=0,
        )),
        "median_returned_duration_ms": (
            float(np.median([episode["dur_ms"] for episode in selected]))
            if selected else None
        ),
        "patient_mode_counts": np.bincount(labels, minlength=2).astype(int),
        "ood_fraction": float(np.mean(ood)) if len(ood) else None,
        "fragment_partition": fragment_partition(episodes, len(detector_fragments)),
        "fragment_modes": fragment_modes,
    }
    return worker, diagnostics


def _pair_sensitivity(primary: dict, other: dict) -> dict:
    if primary["n_detector_fragments"] != other["n_detector_fragments"]:
        raise RuntimeError("detector fragments changed across reset sensitivities")
    common = (primary["fragment_modes"] >= 0) & (other["fragment_modes"] >= 0)
    return {
        "boundary_partition_ari": float(adjusted_rand_score(
            primary["fragment_partition"], other["fragment_partition"],
        )),
        "common_returned_fragments": int(np.sum(common)),
        "patient_mode_agreement_on_common_fragments": (
            float(np.mean(
                primary["fragment_modes"][common] == other["fragment_modes"][common]
            )) if np.any(common) else None
        ),
        "returned_excursion_count_difference": int(
            other["n_returned_excursions"] - primary["n_returned_excursions"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    artifact_root = args.artifact_root.resolve()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    manifest_path = artifact_root / config["candidate_manifest"]
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("selection_forbidden") is not True:
        raise RuntimeError("event-unit canary unexpectedly permits field selection")
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
    multiples = [
        float(value) for value in config["event_unit"]["sensitivity_decay_multiples"]
    ]
    primary_multiple = float(config["event_unit"]["fast_state_decay_multiples"])
    rows, diagnostics = [], {}
    for candidate in manifest["candidates"]:
        candidate_id = str(candidate["candidate_id"])
        diagnostics[candidate_id] = {}
        for multiple in multiples:
            workers, per_seed = [], []
            for seed in seeds:
                stem = f"{candidate_id}_seed_{seed}"
                worker, diagnostic = _worker_at_multiple(
                    output_root / "workers" / f"{stem}.npz",
                    output_root / "workers" / f"{stem}.json",
                    patient["contact_names"], classifier,
                    semantics["raw_to_patient"], multiple=multiple,
                    low_fraction=float(config["event_unit"]["low_threshold_fraction"]),
                    pre_roll_ms=float(config["event_unit"]["pre_roll_ms"]),
                    readout=config["search"]["contact_readout"],
                )
                workers.append(worker)
                per_seed.append(diagnostic)
            score = score_candidate(
                candidate, workers, patient, projections, calibration,
            )
            rows.append({
                "candidate_id": candidate_id,
                "decay_multiple": multiple,
                "score": score,
                "pooled_natural_kmeans": _natural_summary(workers, 20260823),
                "per_seed": [{
                    key: value for key, value in row.items()
                    if key not in {"fragment_partition", "fragment_modes"}
                } for row in per_seed],
            })
            diagnostics[candidate_id][multiple] = {
                int(row["seed"]): row for row in per_seed
            }
    sensitivity = []
    for candidate_id, by_multiple in diagnostics.items():
        primary = by_multiple[primary_multiple]
        for multiple in multiples:
            if multiple == primary_multiple:
                continue
            for seed in seeds:
                sensitivity.append({
                    "candidate_id": candidate_id, "seed": seed,
                    "primary_decay_multiple": primary_multiple,
                    "comparison_decay_multiple": multiple,
                    **_pair_sensitivity(primary[seed], by_multiple[multiple][seed]),
                })
    payload = {
        "schema_id": "topic4_rev12_population_excursion_canary_audit_v1",
        "status": "REV12ND_POPULATION_EXCURSION_CANARY_AUDIT_COMPLETE",
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
        "selection_forbidden": True,
        "rows": rows,
        "reset_sensitivity": sensitivity,
        "claim_boundary": (
            "Fresh event-unit canary only. Results diagnose boundary, KMeans and score "
            "stability; they cannot select a Node field or support dual-mode recovery."
        ),
    }
    output = args.out or output_root / "aggregate/population_excursion_canary_audit.json"
    _atomic_json(output, payload)
    print(json.dumps({"status": payload["status"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
