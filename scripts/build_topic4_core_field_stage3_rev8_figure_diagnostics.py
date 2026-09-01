"""Build uncertainty and benchmark sidecars for the frozen rev8.1 figures.

This producer performs no SNN simulation. It reopens the patient training split
only to quantify recording-block variability and uses the already frozen final
model event pool for a conditional hierarchical bootstrap. Cluster assignments
remain fixed; KMeans stability is reported separately by the confirmation run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from scripts.calibrate_topic4_core_field_stage3_joint_observable import (  # noqa: E402
    HELD_OUT_FRAC,
    SPLIT_SEED,
    _patient_curves,
)
from scripts.run_topic4_core_field_stage3_joint_confirm import _atomic_npz  # noqa: E402
from scripts.run_topic4_core_field_stage3_profile_round1 import axial_map  # noqa: E402
from scripts.run_topic4_core_field_stage3_rev8_kmeans_fit import (  # noqa: E402
    load_training_target,
)
from src.topic4_core_field_profile import (  # noqa: E402
    fit_profile_modes,
    profile_template_similarity,
    split_by_block,
)
from src.topic4_core_field_runner import atomic_write_json, provenance  # noqa: E402


ROOT = "results/topic4_sef_hfo/data_driven_core_field_stage3"
CONFIRM = f"{ROOT}/joint_confirmation_rev8_1/final_confirmation.json"
PROFILES = f"{ROOT}/joint_confirmation_rev8_1/final_event_profiles.npz"
OUT_JSON = f"{ROOT}/joint_confirmation_rev8_1/figure_diagnostics.json"
OUT_NPZ = f"{ROOT}/joint_confirmation_rev8_1/figure_diagnostics.npz"
BOOTSTRAP_SEED = 20260810
N_BOOTSTRAP = 1000


def _sha256(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def hierarchical_bootstrap_indices(groups, rng):
    """Resample groups, then rows within every selected group occurrence."""
    groups = np.asarray(groups)
    unique = np.unique(groups)
    selected_groups = rng.choice(unique, size=len(unique), replace=True)
    chunks = []
    for group in selected_groups:
        members = np.flatnonzero(groups == group)
        chunks.append(rng.choice(members, size=len(members), replace=True))
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=int)


def _mode_prototypes(curves, labels):
    curves = np.asarray(curves, float)
    labels = np.asarray(labels, int)
    if curves.ndim != 2 or labels.shape != (len(curves),):
        raise ValueError("curves and labels do not align")
    if set(np.unique(labels)) != {0, 1}:
        return None
    return np.asarray([curves[labels == mode].mean(axis=0) for mode in (0, 1)])


def conditional_hierarchical_similarity_bootstrap(
        model_curves, model_labels, model_groups,
        patient_curves, patient_labels, patient_groups,
        n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    """Bootstrap the 2x2 matrix conditional on the frozen KMeans labels."""
    rng = np.random.default_rng(int(seed))
    matrices = []
    for _ in range(int(n_bootstrap)):
        model_index = hierarchical_bootstrap_indices(model_groups, rng)
        patient_index = hierarchical_bootstrap_indices(patient_groups, rng)
        model = _mode_prototypes(
            np.asarray(model_curves)[model_index], np.asarray(model_labels)[model_index])
        patient = _mode_prototypes(
            np.asarray(patient_curves)[patient_index],
            np.asarray(patient_labels)[patient_index])
        if model is None or patient is None:
            continue
        matrix = profile_template_similarity(model, patient)
        if np.isfinite(matrix).all():
            matrices.append(matrix)
    return np.asarray(matrices, float).reshape((-1, 2, 2))


def patient_block_mode_bands(curves, labels, block_ids, quantiles=(0.05, 0.95)):
    """Between-block band of block-specific mode prototypes."""
    curves = np.asarray(curves, float)
    labels = np.asarray(labels, int)
    block_ids = np.asarray(block_ids)
    low, high, counts = [], [], []
    for mode in (0, 1):
        rows = []
        for block in np.unique(block_ids):
            selected = (block_ids == block) & (labels == mode)
            if selected.any():
                rows.append(curves[selected].mean(axis=0))
        values = np.asarray(rows, float)
        if not len(values):
            raise RuntimeError(f"patient mode {mode} is absent from all training blocks")
        low.append(np.quantile(values, quantiles[0], axis=0))
        high.append(np.quantile(values, quantiles[1], axis=0))
        counts.append(len(values))
    return np.asarray(low), np.asarray(high), np.asarray(counts, int)


def _benchmark_table(confirmation, matrix_bootstrap):
    candidate = confirmation["candidates"][0]["confirm"]
    controls = confirmation["optimization_controls_n20"]
    kmeans = confirmation["kmeans_controls"]
    rows = [
        ("data-driven", candidate["bootstrap_distance_patient_train"],
         candidate["kmeans_data_consistency"]),
        ("filament", controls["stage2_filament"], kmeans["stage2_filament"]),
        ("hand dual-core", controls["hand_placed_two_cores"],
         kmeans["hand_placed_two_cores"]),
        ("patient held-out", controls["patient_heldout"], kmeans["patient_heldout"]),
    ]
    output = dict(
        names=np.asarray([row[0] for row in rows], dtype="U32"),
        distance_median=np.asarray([row[1]["median"] for row in rows], float),
        distance_p05=np.asarray([row[1]["p05"] for row in rows], float),
        distance_p95=np.asarray([row[1]["p95"] for row in rows], float),
        matched_mean=np.asarray([row[2]["matched_mean"] for row in rows], float),
        worst_mode=np.asarray([
            min(np.diag(np.asarray(row[2]["similarity_matrix"], float)))
            for row in rows
        ], float),
    )
    if len(matrix_bootstrap):
        worst = np.min(np.diagonal(matrix_bootstrap, axis1=1, axis2=2), axis=1)
        output["data_driven_worst_mode_ci"] = np.quantile(worst, (0.025, 0.975))
    else:
        output["data_driven_worst_mode_ci"] = np.full(2, np.nan)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmation", default=CONFIRM)
    parser.add_argument("--profiles", default=PROFILES)
    parser.add_argument("--out-json", default=OUT_JSON)
    parser.add_argument("--out-npz", default=OUT_NPZ)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()

    confirmation = json.load(open(args.confirmation))
    if confirmation["event_profiles"]["sha256"] != _sha256(args.profiles):
        raise RuntimeError("confirmation/event-profile hash mismatch")
    arrays = np.load(args.profiles)
    reference_path = confirmation["reference"]["path"]
    target_path = confirmation["target"]["path"]
    if confirmation["reference"]["sha256"] != _sha256(reference_path):
        raise RuntimeError("confirmation/reference hash mismatch")
    if confirmation["target"]["sha256"] != _sha256(target_path):
        raise RuntimeError("confirmation/target hash mismatch")
    reference_file = np.load(reference_path)
    reference = {key: np.asarray(reference_file[key]) for key in reference_file.files}
    target = load_training_target(target_path)

    axial = axial_map()
    grid = np.asarray(arrays["grid"], float)
    patient_curves, patient_blocks = _patient_curves(axial, grid)
    train_index, _ = split_by_block(patient_blocks, HELD_OUT_FRAC, SPLIT_SEED)
    patient_train = patient_curves[train_index]
    patient_train_blocks = np.asarray(patient_blocks)[train_index]
    patient_modes = fit_profile_modes(patient_train, reference)
    frozen_patient = np.asarray(target["patient_train_mode_prototypes"], float)
    if patient_modes.get("status") != "ok" or not np.allclose(
            patient_modes["prototypes"], frozen_patient, atol=2e-7, rtol=0.0):
        raise RuntimeError("patient training modes no longer reproduce the frozen target")

    model_curves = np.asarray(arrays["model_curves"], float)
    model_labels = np.asarray(arrays["model_labels"], int)
    model_seeds = np.asarray(arrays["model_seed_ids"], int)
    matrices = conditional_hierarchical_similarity_bootstrap(
        model_curves, model_labels, model_seeds,
        patient_train, patient_modes["labels"], patient_train_blocks,
        n_bootstrap=args.n_bootstrap, seed=args.bootstrap_seed)
    minimum_valid = int(np.ceil(0.90 * int(args.n_bootstrap)))
    if len(matrices) < minimum_valid:
        raise RuntimeError(
            f"only {len(matrices)}/{args.n_bootstrap} bootstrap replicates were valid")
    matrix_low, matrix_high = np.quantile(matrices, (0.025, 0.975), axis=0)
    band_low, band_high, band_blocks = patient_block_mode_bands(
        patient_train, patient_modes["labels"], patient_train_blocks)
    benchmark = _benchmark_table(confirmation, matrices)
    contact_names = [str(value) for value in arrays["contact_names"]]
    contact_axial = np.asarray([axial[name] for name in contact_names], float)

    _atomic_npz(
        args.out_npz,
        patient_block_band_low=np.asarray(band_low, np.float32),
        patient_block_band_high=np.asarray(band_high, np.float32),
        patient_block_counts=np.asarray(band_blocks, np.int64),
        matrix_bootstrap=np.asarray(matrices, np.float32),
        matrix_ci_low=np.asarray(matrix_low, np.float32),
        matrix_ci_high=np.asarray(matrix_high, np.float32),
        contact_axial_mm=np.asarray(contact_axial, np.float32),
        **{key: np.asarray(value) for key, value in benchmark.items()},
    )
    payload = dict(
        status="REV8_1_FIGURE_DIAGNOSTICS_COMPLETE",
        scientific_role=(
            "uncertainty and benchmark sidecars for plotting; no SNN simulation "
            "and no change to the frozen candidate or acceptance verdict"),
        conditional_hierarchical_bootstrap=dict(
            seed=int(args.bootstrap_seed), requested=int(args.n_bootstrap),
            valid=int(len(matrices)), ci="2.5--97.5 percentiles",
            model_hierarchy="final unseen-network seed then event",
            patient_hierarchy="training recording block then event",
            cluster_contract=(
                "conditional on the original frozen KMeans labels; this CI measures "
                "prototype sampling uncertainty, not clustering uncertainty"),
            matrix_ci_low=matrix_low.tolist(), matrix_ci_high=matrix_high.tolist(),
        ),
        patient_block_profile_band=dict(
            quantiles=[0.05, 0.95],
            n_blocks_with_mode=band_blocks.tolist(),
            role="between-recording-block variability of frozen-mode mean profiles",
        ),
        benchmark=dict(
            names=benchmark["names"].tolist(),
            x="matched-n global curve distance; lower is better",
            y="minimum matched diagonal Spearman correlation; higher is better",
            control_y_uncertainty=(
                "not available from the frozen confirmation summary; control y values "
                "are descriptive point estimates"),
        ),
        inputs=dict(
            confirmation=dict(path=args.confirmation, sha256=_sha256(args.confirmation)),
            profiles=dict(path=args.profiles, sha256=_sha256(args.profiles)),
            reference=dict(path=reference_path, sha256=_sha256(reference_path)),
            target=dict(path=target_path, sha256=_sha256(target_path)),
        ),
        arrays=dict(path=args.out_npz, sha256=_sha256(args.out_npz)),
        provenance=provenance(),
    )
    atomic_write_json(payload, args.out_json)
    print(json.dumps({
        "status": payload["status"], "bootstrap_valid": len(matrices),
        "matrix_ci_low": matrix_low.tolist(),
        "matrix_ci_high": matrix_high.tolist(),
        "arrays_sha256": payload["arrays"]["sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
