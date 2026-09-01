"""Corrected D5.2 audit using the canonical Fig.4C rank contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.paper_figures.plot_fig4_spatial_edge_flow_validation import (  # noqa: E402
    _canonical_rank_kmeans,
    _column_stats,
    _load_bundle,
    _patient_profiles,
    _similarity,
    normalize_event_ranks,
)
from src.lagpat_rank_audit import build_masked_kmeans_features  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/topic4_rev10_d5_2_spatial_ou_confirmation.json"
INVALID_STATUS = "REV10D5_2_KMEANS_PATIENT_IDENTITY_NOT_CONFIRMED"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        Path(temporary).write_text(json.dumps(payload, indent=2, sort_keys=True))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _purity(labels, direction):
    contingency = np.zeros((2, 2), int)
    for cluster, mode in zip(labels, direction):
        contingency[int(cluster), int(mode)] += 1
    identity = contingency[0, 0] + contingency[1, 1]
    swapped = contingency[0, 1] + contingency[1, 0]
    return float(max(identity, swapped) / max(1, contingency.sum())), contingency


def _row_activity(row, off):
    occupancy = float(row["mean_network_fraction_time_above_detector"])
    baseline = float(off["mean_network_fraction_time_above_detector"])
    return {
        "mean_network_detected_events": float(
            row["mean_network_detected_events_descriptive"]
        ),
        "mean_network_returned_events": float(
            row["mean_network_returned_events_scored"]
        ),
        "mean_network_fraction_time_above_detector": occupancy,
        "max_network_fraction_time_above_detector": float(
            row["max_network_fraction_time_above_detector"]
        ),
        "mean_network_peak_active_fraction": float(
            row["mean_network_peak_active_fraction"]
        ),
        "occupancy_delta_from_off": occupancy - baseline,
        "occupancy_ratio_to_off": None if baseline == 0 else occupancy / baseline,
        "mean_network_ood_fraction": float(row["mean_network_ood_fraction"]),
    }


def _seed_stratified_permutation(labels, direction, seed_ids, *, repeats, seed):
    observed, _ = _purity(labels, direction)
    rng = np.random.default_rng(int(seed))
    exceed = 0
    for _ in range(int(repeats)):
        shuffled = np.asarray(direction, int).copy()
        for network_seed in np.unique(seed_ids):
            index = np.flatnonzero(seed_ids == network_seed)
            shuffled[index] = rng.permutation(shuffled[index])
        exceed += int(_purity(labels, shuffled)[0] >= observed - 1e-12)
    return float((exceed + 1) / (int(repeats) + 1))


def _loso_purity(bundle, selected_index, direction, seed_ids):
    ranks = np.asarray(bundle["ranks"][selected_index], float).T
    bools = np.isfinite(ranks)
    rows = []
    for held_out in np.unique(seed_ids):
        keep = seed_ids != held_out
        features = build_masked_kmeans_features(
            ranks[:, keep], bools[:, keep], impute="event_median",
        )
        labels = KMeans(
            n_clusters=2, n_init=50, random_state=0,
        ).fit_predict(features)
        value, contingency = _purity(labels, direction[keep])
        rows.append({
            "held_out_network_seed": int(held_out),
            "n_events": int(np.sum(keep)),
            "direction_purity": value,
            "direction_contingency": contingency.tolist(),
        })
    return rows


def _patient_matched_benchmark(bundle, direction, *, draws, seed):
    with np.load(bundle["target_path"], allow_pickle=False) as loaded:
        ranks = np.asarray(loaded["patient_train_ranks"], float).T
        labels = np.asarray(loaded["patient_train_old_labels"], int)
        blocks = np.asarray(loaded["patient_train_block_ids"])
    bools = np.isfinite(ranks)
    valid = bools.sum(axis=0) >= 3
    ranks, bools, labels, blocks = (
        ranks[:, valid], bools[:, valid], labels[valid], blocks[valid]
    )
    n_by_mode = np.bincount(np.asarray(direction, int), minlength=2)
    unique_blocks = np.unique(blocks)
    rng = np.random.default_rng(int(seed))
    values = []
    for repeat in range(int(draws)):
        sampled_blocks = rng.choice(unique_blocks, size=6, replace=True)
        pool = np.concatenate([
            np.flatnonzero(blocks == block) for block in sampled_blocks
        ])
        selected = []
        for mode, count in enumerate(n_by_mode):
            available = pool[labels[pool] == mode]
            if not len(available):
                break
            selected.extend(rng.choice(
                available, size=int(count), replace=len(available) < int(count),
            ).tolist())
        if len(selected) != int(np.sum(n_by_mode)):
            continue
        selected = np.asarray(selected, int)
        features = build_masked_kmeans_features(
            ranks[:, selected], bools[:, selected], impute="event_median",
        )
        cluster = KMeans(
            n_clusters=2, n_init=20, random_state=repeat,
        ).fit_predict(features)
        values.append(_purity(cluster, labels[selected])[0])
    values = np.asarray(values, float)
    if len(values) < int(draws) * 0.95:
        raise RuntimeError("patient matched KMeans benchmark has insufficient draws")
    return {
        "sampling": (
            "six recording blocks with replacement; within sampled blocks, "
            "match model event count and frozen A/B count; KMeans on canonical "
            "masked normalized ranks"
        ),
        "seed": int(seed), "n_draws": int(len(values)),
        "model_matched_mode_counts": n_by_mode.tolist(),
        "q05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "q95": float(np.quantile(values, 0.95)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "draws": values.tolist(),
    }


def adjudicate(*, minimum_same_networks, minimum_events_per_mode,
               local_row, permuted_row, off_row, clean_counts,
               kmeans_purity, kmeans_permutation_p, patient_purity_q05,
               patient_matrix):
    clean_counts = np.asarray(clean_counts, int)
    matrix = np.asarray(patient_matrix, float)
    support = bool(
        local_row["n_runaway_networks"] == 0
        and local_row["networks_with_both_clean_modes"]
        >= int(minimum_same_networks)
        and np.all(clean_counts >= int(minimum_events_per_mode))
    )
    matrix_valid = bool(
        matrix.shape == (2, 2) and np.all(np.isfinite(matrix))
        and matrix[0, 0] > 0 and matrix[1, 1] > 0
        and matrix[0, 1] < 0 and matrix[1, 0] < 0
    )
    association = bool(float(kmeans_permutation_p) <= 0.05)
    patient_level = bool(float(kmeans_purity) >= float(patient_purity_q05))
    if not support:
        status = "REV10D5_2_FRESH_NETWORK_DUAL_MODE_SUPPORT_NOT_CONFIRMED"
    elif not matrix_valid:
        status = "REV10D5_2_DIRECTION_PROTOTYPE_GEOMETRY_NOT_CONFIRMED"
    elif not association:
        status = "REV10D5_2_KMEANS_DIRECTION_ASSOCIATION_NOT_OBSERVED"
    elif not patient_level:
        status = (
            "REV10D5_2_DIRECTION_PROTOTYPES_RECOVERED_"
            "KMEANS_BELOW_PATIENT_BENCHMARK"
        )
    else:
        status = "REV10D5_2_FRESH_NETWORK_PATIENT_MODE_CONSISTENCY_OBSERVED"
    return {
        "status": status,
        "fresh_network_dual_mode_support": support,
        "supervised_patient_prototype_geometry": matrix_valid,
        "kmeans_direction_association": association,
        "kmeans_reaches_patient_matched_q05": patient_level,
        "formal_clean_mode_counts": {
            "A": int(clean_counts[0]), "B": int(clean_counts[1]),
        },
        "same_network_dual_mode": {
            "local": int(local_row["networks_with_both_clean_modes"]),
            "permuted": int(permuted_row["networks_with_both_clean_modes"]),
            "off": int(off_row["networks_with_both_clean_modes"]),
            "required_local": int(minimum_same_networks),
        },
        "supervised_direction_vs_patient_spearman": matrix.tolist(),
        "activity_burden": {
            "local": _row_activity(local_row, off_row),
            "permuted": _row_activity(permuted_row, off_row),
            "off": _row_activity(off_row, off_row),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    root = ROOT / config["output_root"]
    summary_path = root / "confirmation_summary_returned_only.json"
    manifest_path = root / "candidate_manifest.json"
    summary = json.loads(summary_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if summary.get("status") != "REV10R_RETURNED_ONLY_CONFIRMATION_COMPLETE":
        raise RuntimeError("D5.2 confirmation summary incomplete")
    if manifest.get("status") != "REV10D5_2_SPATIAL_OU_CONFIRMATION_LIBRARY_FROZEN":
        raise RuntimeError("D5.2 confirmation manifest invalid")
    selected = manifest["selection_freeze"]["selected_nonzero_candidate_id"]
    permuted = manifest["selection_freeze"]["matched_permuted_candidate_id"]
    rows = {row["candidate_id"]: row for row in summary["candidate_rows"]}
    bundle = _load_bundle(config_path, root, selected)
    canonical = _canonical_rank_kmeans(bundle)
    selected_index = canonical["clean_global_index"]
    direction = canonical["direction"]
    seed_ids = np.asarray([
        bundle["records"][int(index)]["seed"] for index in selected_index
    ], int)
    permutation_p = _seed_stratified_permutation(
        canonical["labels"], direction, seed_ids,
        repeats=10_000, seed=20260813,
    )
    benchmark = _patient_matched_benchmark(
        bundle, direction, draws=256, seed=20260813,
    )
    ranks = normalize_event_ranks(bundle["ranks"][selected_index])
    model = np.asarray([
        _column_stats(ranks[direction == mode])[0] for mode in (0, 1)
    ])
    patient = _patient_profiles(bundle)[0]
    matrix = _similarity(model, patient)
    acceptance = config["search"]["acceptance"]
    payload = adjudicate(
        minimum_same_networks=acceptance[
            "minimum_same_networks_with_both_modes"
        ],
        minimum_events_per_mode=acceptance[
            "minimum_pooled_clean_events_per_mode"
        ],
        local_row=rows[selected], permuted_row=rows[permuted],
        off_row=rows["edge_noop"], clean_counts=bundle["clean_counts"],
        kmeans_purity=canonical["direction_purity"],
        kmeans_permutation_p=permutation_p,
        patient_purity_q05=benchmark["q05"], patient_matrix=matrix,
    )
    payload.update({
        "selected_local_candidate_id": selected,
        "matched_permuted_candidate_id": permuted,
        "canonical_fig4c_kmeans": {
            "feature_contract": "masked normalized event ranks",
            "min_shared_contacts": 3,
            "n_events": int(len(canonical["labels"])),
            "cluster_counts": canonical["cluster_counts"].tolist(),
            "direction_contingency": canonical[
                "direction_contingency"
            ].tolist(),
            "direction_purity": canonical["direction_purity"],
            "seed_stratified_direction_permutation_p": permutation_p,
            "within_cluster_tau_mean": canonical[
                "within_cluster_tau_mean"
            ],
            "inter_cluster_corr_matrix": canonical[
                "inter_cluster_corr_matrix"
            ],
            "kmeans_stability_ami_median": canonical[
                "stability_ami_median"
            ],
            "silhouette_median": canonical["silhouette_median"],
            "leave_one_network_out": _loso_purity(
                bundle, selected_index, direction, seed_ids,
            ),
        },
        "patient_matched_kmeans_direction_purity": benchmark,
        "contract_correction": {
            "status": "INVALID_FLAT_SHAFT_AWARE_KMEANS_GATE_RETRACTED",
            "reason": (
                "the superseded audit clustered the full shaft-aware PCA "
                "embedding and demanded AMI>=0.8 with old direction A/B, but "
                "patient data under that representation has AMI=0.011 because "
                "direction and recruitment extent are distinct factors"
            ),
            "replacement": (
                "canonical Fig.4C masked-rank KMeans for natural clustering "
                "and direction purity; model-vs-patient matrix built directly "
                "from frozen supervised A/B events, never cluster labels"
            ),
            "correction_is_post_result_and_exploratory": True,
        },
        "claim_boundary": (
            "fresh-network but development-only; corrected KMeans contract was "
            "applied after detecting an invalid patient-self-consistency gate; "
            "no patient-blind, full interictal-distribution, clinical-waveform, "
            "or causal-core claim"
        ),
        "inputs": {
            "config": {"path": str(config_path.relative_to(ROOT)),
                       "sha256": _sha256(config_path)},
            "manifest": {"path": str(manifest_path.relative_to(ROOT)),
                         "sha256": _sha256(manifest_path)},
            "summary": {"path": str(summary_path.relative_to(ROOT)),
                        "sha256": _sha256(summary_path)},
            "patient_target": {"path": str(bundle["target_path"].relative_to(ROOT)),
                               "sha256": _sha256(bundle["target_path"])},
        },
    })
    output = root / "confirmation_verdict.json"
    if output.exists():
        old = json.loads(output.read_text())
        if old.get("status") == INVALID_STATUS:
            old["invalidation"] = payload["contract_correction"]
            _atomic_json(
                root / "confirmation_verdict_invalid_flat_shaft_kmeans.json", old,
            )
    _atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"],
        "formal_clean_mode_counts": payload["formal_clean_mode_counts"],
        "same_network_dual_mode": payload["same_network_dual_mode"],
        "kmeans_direction_purity": canonical["direction_purity"],
        "patient_matched_purity_q05": benchmark["q05"],
        "supervised_patient_matrix": matrix.tolist(),
    }, indent=2))


if __name__ == "__main__":
    main()
