"""Adjudicate frozen D5.2 fresh-network support and Fig.4 consistency."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.paper_figures.plot_fig4_spatial_edge_flow_validation import (
    _column_stats,
    _kmeans,
    _load_bundle,
    _patient_profiles,
    _similarity,
    normalize_event_ranks,
)


DEFAULT_CONFIG = ROOT / "config/topic4_rev10_d5_2_spatial_ou_confirmation.json"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def map_kmeans_by_supervised_direction(labels, direction):
    labels = np.asarray(labels, int)
    direction = np.asarray(direction, int)
    contingency = np.zeros((2, 2), int)
    for cluster, mode in zip(labels, direction):
        contingency[cluster, mode] += 1
    identity = int(contingency[0, 0] + contingency[1, 1])
    swapped = int(contingency[0, 1] + contingency[1, 0])
    cluster_for_mode = (0, 1) if identity >= swapped else (1, 0)
    return cluster_for_mode, contingency


def adjudicate(*, acceptance, local_row, permuted_row, off_row,
               clean_counts, kmeans_audit, patient_matrix):
    clean_counts = np.asarray(clean_counts, int)
    patient_matrix = np.asarray(patient_matrix, float)
    support = bool(
        local_row["n_runaway_networks"] == 0
        and local_row["networks_with_both_clean_modes"]
        >= int(acceptance["minimum_same_networks_with_both_modes"])
        and np.all(
            clean_counts
            >= int(acceptance["minimum_pooled_clean_events_per_mode"])
        )
    )
    kmeans_ami = kmeans_audit.get("ami_with_supervised_direction")
    kmeans_consistent = bool(
        kmeans_audit.get("status") == "OK"
        and kmeans_ami is not None
        and float(kmeans_ami)
        >= float(acceptance["minimum_kmeans_ami_with_supervised_direction"])
    )
    patient_geometry = bool(
        patient_matrix.shape == (2, 2)
        and np.all(np.isfinite(patient_matrix))
        and patient_matrix[0, 0] > 0.0
        and patient_matrix[1, 1] > 0.0
        and patient_matrix[0, 1] < 0.0
        and patient_matrix[1, 0] < 0.0
    )
    if not support:
        status = "REV10D5_2_FRESH_NETWORK_DUAL_MODE_SUPPORT_NOT_CONFIRMED"
    elif not kmeans_consistent:
        status = "REV10D5_2_KMEANS_PATIENT_IDENTITY_NOT_CONFIRMED"
    elif not patient_geometry:
        status = "REV10D5_2_PATIENT_PROTOTYPE_GEOMETRY_NOT_CONFIRMED"
    else:
        status = "REV10D5_2_FRESH_NETWORK_PATIENT_MODE_CONSISTENCY_OBSERVED"
    off = off_row
    return {
        "status": status,
        "fig4_support_evaluable": support,
        "kmeans_patient_identity_consistent": kmeans_consistent,
        "patient_prototype_sign_geometry_consistent": patient_geometry,
        "formal_clean_mode_counts": {
            "A": int(clean_counts[0]), "B": int(clean_counts[1]),
        },
        "same_network_dual_mode": {
            "local": int(local_row["networks_with_both_clean_modes"]),
            "permuted": int(permuted_row["networks_with_both_clean_modes"]),
            "off": int(off_row["networks_with_both_clean_modes"]),
            "required_local": int(
                acceptance["minimum_same_networks_with_both_modes"]
            ),
        },
        "kmeans": kmeans_audit,
        "patient_matrix_after_direction_only_cluster_mapping": (
            patient_matrix.tolist()
        ),
        "activity_burden": {
            "local": _row_activity(local_row, off),
            "permuted": _row_activity(permuted_row, off),
            "off": _row_activity(off_row, off),
        },
        "claim_boundary": (
            "fresh-network but development-only; patient training target and "
            "classifier were already used; no patient-blind generalization, "
            "full interictal-distribution, clinical-waveform, or causal-core claim"
        ),
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
    clean_index = np.flatnonzero(bundle["clean"])
    labels, kmeans_audit = _kmeans(
        bundle["embedding"][clean_index], bundle["labels"][clean_index],
    )
    if labels is None:
        matrix = np.full((2, 2), np.nan)
        contingency = np.zeros((2, 2), int)
    else:
        cluster_for_mode, contingency = map_kmeans_by_supervised_direction(
            labels, bundle["labels"][clean_index],
        )
        ranks = normalize_event_ranks(bundle["ranks"][clean_index])
        model = np.asarray([
            _column_stats(ranks[labels == cluster_for_mode[mode]])[0]
            for mode in (0, 1)
        ])
        patient = _patient_profiles(bundle)[0]
        matrix = _similarity(model, patient)
    kmeans_audit["direction_contingency"] = contingency.tolist()
    kmeans_audit["ami_with_supervised_direction"] = (
        None if labels is None else float(adjusted_mutual_info_score(
            labels, bundle["labels"][clean_index],
        ))
    )
    payload = adjudicate(
        acceptance=config["search"]["acceptance"],
        local_row=rows[selected], permuted_row=rows[permuted],
        off_row=rows["edge_noop"], clean_counts=bundle["clean_counts"],
        kmeans_audit=kmeans_audit, patient_matrix=matrix,
    )
    payload.update({
        "selected_local_candidate_id": selected,
        "matched_permuted_candidate_id": permuted,
        "inputs": {
            "config": {"path": str(config_path.relative_to(ROOT)),
                       "sha256": _sha256(config_path)},
            "manifest": {"path": str(manifest_path.relative_to(ROOT)),
                         "sha256": _sha256(manifest_path)},
            "summary": {"path": str(summary_path.relative_to(ROOT)),
                        "sha256": _sha256(summary_path)},
        },
    })
    output = root / "confirmation_verdict.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=output.parent, suffix=".tmp")
    os.close(fd)
    try:
        Path(temporary).write_text(json.dumps(payload, indent=2, sort_keys=True))
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(json.dumps({
        "status": payload["status"],
        "formal_clean_mode_counts": payload["formal_clean_mode_counts"],
        "same_network_dual_mode": payload["same_network_dual_mode"],
        "kmeans_ami": payload["kmeans"].get(
            "ami_with_supervised_direction"
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
