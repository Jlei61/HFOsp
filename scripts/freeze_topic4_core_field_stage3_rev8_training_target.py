"""Freeze the patient-training-only KMeans target for Stage 3 rev8.

The producer reuses the rev6 normalized-rank-curve embedding and the frozen
recording-block split.  Held-out events are identified only to exclude them;
no held-out curve, prototype, score, or threshold enters this artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.getcwd())
from scripts.calibrate_topic4_core_field_stage3_joint_observable import (  # noqa: E402
    HELD_OUT_FRAC,
    SPLIT_SEED,
    _patient_curves,
)
from scripts.run_topic4_core_field_stage3_profile_round1 import axial_map  # noqa: E402
from src.topic4_core_field_profile import (  # noqa: E402
    MODE_MIN_CLUSTER_EVENTS,
    MODE_OBJECTIVE_N_EVENTS,
    fit_profile_modes,
    profile_grid,
    profile_mode_target_matrix,
    split_by_block,
)
from src.topic4_core_field_runner import atomic_write_json, provenance  # noqa: E402


ROOT = "results/topic4_sef_hfo/data_driven_core_field_stage3"
REFERENCE_PATH = f"{ROOT}/joint_observable/rank_curve_reference.npz"
OUT = f"{ROOT}/joint_kmeans_training_target_rev8"
TARGET_NAME = "patient_train_kmeans_target_rev8.npz"
SUMMARY_NAME = "training_target_summary.json"


def _sha256(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _atomic_npz(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".npz")
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_reference(path):
    data = np.load(path)
    required = (
        "grid", "center", "components", "score_center", "score_scale",
        "reference_z", "directions",
    )
    missing = [key for key in required if key not in data.files]
    if missing:
        raise RuntimeError(f"rev6 reference is missing {missing}")
    return {key: np.asarray(data[key]) for key in required}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default=REFERENCE_PATH)
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()

    reference = _load_reference(args.reference)
    axial = axial_map()
    grid = profile_grid(axial)
    if not np.array_equal(grid, reference["grid"]):
        raise RuntimeError("rev6 reference grid no longer matches the frozen montage")

    patient_curves, block_ids = _patient_curves(axial, grid)
    train_index, heldout_index = split_by_block(
        block_ids, HELD_OUT_FRAC, SPLIT_SEED)
    train_curves = patient_curves[train_index]
    modes = fit_profile_modes(train_curves, reference)
    if modes.get("status") != "ok":
        raise RuntimeError("patient training recordings do not define two KMeans modes")
    prototypes = np.asarray(modes["prototypes"], float)
    matrix = profile_mode_target_matrix(prototypes)

    target_path = os.path.join(args.out, TARGET_NAME)
    _atomic_npz(
        target_path,
        patient_train_mode_prototypes=prototypes,
        patient_train_mode_counts=np.asarray(modes["cluster_counts"], np.int64),
        patient_train_target_similarity_matrix=matrix,
        grid=np.asarray(reference["grid"], float),
    )
    target_sha = _sha256(target_path)
    reference_sha = _sha256(args.reference)
    train_blocks = np.unique(np.asarray(block_ids)[train_index])
    heldout_blocks = np.unique(np.asarray(block_ids)[heldout_index])
    summary = dict(
        status="REV8_PATIENT_TRAINING_TARGET_FROZEN",
        scientific_role=(
            "training-side KMeans auxiliary target; patient held-out scores and "
            "final unseen-network outcomes are absent"
        ),
        target_npz=target_path,
        target_sha256=target_sha,
        reference_npz=args.reference,
        reference_sha256=reference_sha,
        split=dict(
            unit="recording block",
            seed=SPLIT_SEED,
            heldout_fraction=HELD_OUT_FRAC,
            n_train_events=int(len(train_index)),
            n_train_blocks=int(len(train_blocks)),
            n_excluded_heldout_events=int(len(heldout_index)),
            n_excluded_heldout_blocks=int(len(heldout_blocks)),
            heldout_scores_computed=False,
            heldout_prototypes_computed=False,
        ),
        kmeans=dict(
            K=2,
            fit_space="frozen rev6 patient-training PCA embedding",
            cluster_counts=np.asarray(modes["cluster_counts"], int).tolist(),
            prototype_correlation=float(modes["prototype_correlation"]),
            target_similarity_matrix=matrix.tolist(),
        ),
        optimization_contract=dict(
            distance_event_count=20,
            mode_event_count=MODE_OBJECTIVE_N_EVENTS,
            min_events_per_mode=MODE_MIN_CLUSTER_EVENTS,
        ),
        provenance=provenance(),
    )
    atomic_write_json(summary, os.path.join(args.out, SUMMARY_NAME))
    print(json.dumps({
        "status": summary["status"],
        "target_sha256": target_sha,
        "cluster_counts": summary["kmeans"]["cluster_counts"],
        "target_similarity_matrix": summary["kmeans"]["target_similarity_matrix"],
    }, indent=2))


if __name__ == "__main__":
    main()
