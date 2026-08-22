#!/usr/bin/env python3
"""Audit whether adjacent detector fragments create a spurious Node K=2 repertoire."""
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
from src.topic4_d6_natural_kmeans import natural_kmeans  # noqa: E402
from src.topic4_node_dualmode import fixed_projection_matrix  # noqa: E402
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


def merge_returned_event_indices(t_on_ms: np.ndarray, t_off_ms: np.ndarray,
                                 returned: np.ndarray,
                                 maximum_gap_ms: float) -> list[list[int]]:
    """Group returned detector intervals separated by at most maximum_gap_ms."""
    onset = np.asarray(t_on_ms, float)
    offset = np.asarray(t_off_ms, float)
    returned = np.asarray(returned, bool)
    if onset.shape != offset.shape or onset.shape != returned.shape or onset.ndim != 1:
        raise ValueError("event onset, offset and returned arrays must align")
    if float(maximum_gap_ms) < 0.0:
        raise ValueError("maximum_gap_ms must be nonnegative")
    selected = np.flatnonzero(returned)
    if not len(selected):
        return []
    groups = [[int(selected[0])]]
    for index in selected[1:]:
        gap = float(onset[index] - offset[groups[-1][-1]])
        if gap <= float(maximum_gap_ms):
            groups[-1].append(int(index))
        else:
            groups.append([int(index)])
    return groups


def fragmentation_counts(groups: list[list[int]], returned_labels: np.ndarray,
                         returned_indices: np.ndarray) -> dict:
    """Count close fragments and whether their original labels oppose each other."""
    labels = np.asarray(returned_labels, int)
    returned_indices = np.asarray(returned_indices, int)
    label_by_detected = {
        int(detected): int(label)
        for detected, label in zip(returned_indices, labels)
    }
    pairs = []
    for group in groups:
        for left, right in zip(group[:-1], group[1:]):
            pairs.append(label_by_detected[left] != label_by_detected[right])
    return {
        "n_returned_fragments": int(len(returned_indices)),
        "n_merged_episodes": int(len(groups)),
        "n_multifragment_episodes": int(sum(len(group) > 1 for group in groups)),
        "maximum_fragments_per_episode": int(max(map(len, groups), default=0)),
        "n_close_adjacent_pairs": int(len(pairs)),
        "n_close_opposite_label_pairs": int(np.sum(pairs)),
        "close_pair_opposite_fraction": (
            float(np.mean(pairs)) if pairs else None
        ),
    }


def _episode_worker(npz_path: Path, target_names: np.ndarray,
                    classifier: dict, label_map: np.ndarray,
                    maximum_gap_ms: float, readout: dict) -> tuple[dict, dict]:
    payload = json.loads(npz_path.with_suffix(".json").read_text())
    with np.load(npz_path, allow_pickle=False) as loaded:
        source_names = np.asarray(loaded["contact_names"]).astype(str)
        envelope = np.asarray(loaded["contact_envelope"], float)
        envelope_dt = float(loaded["contact_envelope_dt_ms"])
        t_on = np.asarray(loaded["event_t_on_ms"], float)
        t_off = np.asarray(loaded["event_t_off_ms"], float)
        returned = np.asarray(loaded["event_returned"], bool)
        stored_onsets = np.asarray(loaded["onsets"], float)
        stored_ranks = np.asarray(loaded["ranks"], float)
    order = np.asarray([
        int(np.flatnonzero(source_names == name)[0]) for name in target_names
    ])
    montage = SimpleNamespace(names=source_names)
    groups = merge_returned_event_indices(t_on, t_off, returned, maximum_gap_ms)
    onsets, ranks = [], []
    for group in groups:
        if len(group) == 1:
            onset = stored_onsets[group[0]]
            rank = stored_ranks[group[0]]
        else:
            onset, rank = _contact_onsets(
                envelope, envelope_dt, montage, np.ones(len(source_names), bool),
                (float(t_on[group[0]]), float(t_off[group[-1]])),
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
    raw_labels = np.asarray(assigned["labels"], int)
    labels = np.asarray(label_map, int)[raw_labels]
    ood = np.asarray(assigned["ood"], bool)
    formal_clean = _formal_clean_mask(onsets, ood, classifier["groups"])

    returned_indices = np.flatnonzero(returned)
    stored_assigned = assign_direction_modes(
        stored_onsets[returned][:, order], groups=classifier["groups"],
        embedding=classifier["embedding"], classifier=classifier["classifier"],
    )
    stored_labels = np.asarray(label_map, int)[
        np.asarray(stored_assigned["labels"], int)
    ]
    counts = fragmentation_counts(groups, stored_labels, returned_indices)
    worker = {
        "seed": int(payload["seed"]),
        "ranks": ranks,
        "onsets": onsets,
        "labels": labels,
        "ood": ood,
        "formal_clean": formal_clean,
        "n_detected": int(len(t_on)),
        "n_returned": int(len(groups)),
        "duration_ms": float(payload["simulation"]["duration_ms"]),
        "npz": str(npz_path),
        "npz_sha256": _sha256(npz_path),
    }
    if np.isclose(float(maximum_gap_ms), 12.0):
        counts["twelve_ms_reextraction_matches_stored_rank"] = bool(
            ranks.shape == stored_ranks[returned][:, order].shape
            and np.array_equal(
                np.isnan(ranks), np.isnan(stored_ranks[returned][:, order])
            )
            and np.array_equal(
                np.nan_to_num(ranks, nan=-1.0),
                np.nan_to_num(stored_ranks[returned][:, order], nan=-1.0),
            )
        )
    return worker, counts


def _natural_summary(workers: list[dict], seed: int) -> dict:
    ranks = np.concatenate([worker["ranks"] for worker in workers])
    labels = np.concatenate([worker["labels"] for worker in workers])
    result = natural_kmeans(ranks, labels, random_state=int(seed))
    return {
        key: value for key, value in result.items()
        if key not in {"valid_event_mask", "cluster_labels"}
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--gaps-ms", type=float, nargs="+", default=(12, 25, 50, 75, 100))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    artifact_root = args.artifact_root.resolve()
    config = json.loads(args.config.resolve().read_text())
    summary = json.loads(args.summary.resolve().read_text())
    cohort = json.loads((ROOT / config["inputs"]["cohort_config"]["path"]).read_text())
    classifier_config = json.loads(
        (ROOT / config["inputs"]["classifier_config"]["path"]).read_text()
    )
    patient = _patient_data(cohort, artifact_root)
    classifier = _classifier_contract(classifier_config, artifact_root)
    semantics = _old_to_patient_label_map(patient, classifier)
    patient = _reorder_patient_contract(patient, classifier["names"])
    projections = fixed_projection_matrix(
        2 * len(patient["contact_names"]), n_directions=64, seed=20260821,
    )
    calibration = summary["component_calibration"]
    manifest = json.loads((artifact_root / config["candidate_manifest"]).read_text())
    candidates = {row["candidate_id"]: row for row in manifest["candidates"]}
    output_root = artifact_root / config["output_root"]
    rows = []
    for candidate_row in summary["rows"]:
        candidate_id = candidate_row["candidate_id"]
        for gap in args.gaps_ms:
            workers, counts = [], []
            for seed in summary["requested_seeds"]:
                path = output_root / "workers" / f"{candidate_id}_seed_{seed}.npz"
                worker, worker_counts = _episode_worker(
                    path, patient["contact_names"], classifier,
                    semantics["raw_to_patient"], float(gap),
                    config["search"]["contact_readout"],
                )
                workers.append(worker)
                counts.append({"seed": int(seed), **worker_counts})
            score = score_candidate(
                candidates[candidate_id], workers, patient, projections, calibration,
            )
            rows.append({
                "candidate_id": candidate_id,
                "maximum_merge_gap_ms": float(gap),
                "mean_returned_fragments": float(np.mean([
                    item["n_returned_fragments"] for item in counts
                ])),
                "mean_merged_episodes": float(np.mean([
                    item["n_merged_episodes"] for item in counts
                ])),
                "total_close_pairs": int(np.sum([
                    item["n_close_adjacent_pairs"] for item in counts
                ])),
                "total_close_opposite_pairs": int(np.sum([
                    item["n_close_opposite_label_pairs"] for item in counts
                ])),
                "score": score,
                "pooled_natural_kmeans": _natural_summary(workers, 20260822),
                "per_seed": counts,
            })
    payload = {
        "schema_id": "topic4_rev12_nd_event_fragmentation_audit_v1",
        "status": "REV12ND_EVENT_FRAGMENTATION_AUDIT_COMPLETE",
        "config": {"path": str(args.config.resolve()), "sha256": _sha256(args.config.resolve())},
        "summary": {"path": str(args.summary.resolve()), "sha256": _sha256(args.summary.resolve())},
        "settle_consistent_primary_sensitivity_ms": 50.0,
        "rationale": (
            "The detector merges only gaps <=12 ms but independently uses a 50 ms "
            "settling window to establish return. Gap sensitivity tests whether one "
            "recovery episode was split into opposite KMeans classes."
        ),
        "rows": rows,
        "claim_boundary": (
            "This is a zero-simulation event-unit audit. A K=2 result that is not "
            "stable to the detector's own 50 ms settling scale cannot support a "
            "dual-mode repertoire claim."
        ),
    }
    output = args.out or output_root / "aggregate/event_fragmentation_audit.json"
    _atomic_json(output, payload)
    print(json.dumps({"status": payload["status"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
