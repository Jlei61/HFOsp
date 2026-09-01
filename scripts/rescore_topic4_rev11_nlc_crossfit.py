#!/usr/bin/env python3
"""Rescore rev11-NLC canary with the contact-split patient contract."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.paper_figures.plot_fig4_spatial_edge_flow_validation import (  # noqa: E402
    normalize_event_ranks,
)
from scripts.rescore_topic4_rev10_d6_natural_kmeans import (  # noqa: E402
    _candidate_metrics,
    _jsonable,
)
from scripts.rescore_topic4_rev10_sa_historical_artifacts import (  # noqa: E402
    load_scoring_contract,
)
from scripts.run_topic4_rev9l_forced_source_worker import _sha256  # noqa: E402
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_shaft_aware import contract_groups  # noqa: E402
from src.topic4_shaft_aware_direction import assign_direction_modes  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/topic4_rev11_nlc_local_connectivity_canary.json"


def _profiles(ranks, labels):
    normalized = normalize_event_ranks(ranks)
    labels = np.asarray(labels, int)
    return np.asarray([
        np.nanmean(normalized[labels == mode], axis=0) for mode in (0, 1)
    ])


def _similarity(left, right):
    matrix = np.full((2, 2), np.nan)
    for row in (0, 1):
        for column in (0, 1):
            finite = np.isfinite(left[row]) & np.isfinite(right[column])
            if np.sum(finite) >= 3:
                matrix[row, column] = spearmanr(
                    left[row, finite], right[column, finite],
                ).statistic
    return matrix


def patient_classifier_rank_audit(config, manifest, contract):
    target_path = ROOT / config["inputs"]["shaft_aware_target_npz"]["path"]
    floor_path = ROOT / config["inputs"]["shaft_aware_floors"]["path"]
    _, embedding, _, _ = load_scoring_contract(
        target_path, floor_path, "FULL_TIMING", fixed_events_per_mode=6,
    )
    with np.load(target_path, allow_pickle=False) as loaded:
        onsets = np.asarray(loaded["patient_train_onsets"], float)
        ranks = np.asarray(loaded["patient_train_ranks"], float)
        labels = np.asarray(loaded["patient_train_old_labels"], int)
    classifier = deepcopy(manifest["direction_classifier"])
    for key in (
            "coef", "class_centers", "class_precisions",
            "ood_distance_thresholds"):
        classifier[key] = np.asarray(classifier[key], float)
    assigned = assign_direction_modes(
        onsets, groups=contract_groups(contract), embedding=embedding,
        classifier=classifier,
    )
    predicted = np.asarray(assigned["labels"], int)
    patient_profiles = _profiles(ranks, labels)
    predicted_profiles = _profiles(ranks, predicted)
    return {
        "n_patient_train_events": int(len(labels)),
        "classifier_vs_old_label_agreement": float(np.mean(predicted == labels)),
        "classifier_ood_fraction": float(np.mean(assigned["ood"])),
        "predicted_label_rank_profile_vs_old_label_profile_spearman": (
            _similarity(predicted_profiles, patient_profiles).tolist()
        ),
        "old_label_profile_self_spearman": (
            _similarity(patient_profiles, patient_profiles).tolist()
        ),
        "interpretation": (
            "positive diagonal here makes a negative model diagonal a scientific "
            "model-patient mismatch rather than a rank-sign implementation error"
        ),
    }


def select_joint_search_parents(canary_verdict, summary):
    by_id = {row["candidate_id"]: row for row in summary["candidate_rows"]}
    direction_id = canary_verdict["selected_candidate_id"]
    aggregate_id = summary["diagnostic_best_candidate_id"]
    selected = []
    for candidate_id, role in (
            (direction_id, "best_frozen_direction_natural_kmeans"),
            (aggregate_id, "best_shaft_aware_equal_network_aggregate")):
        if candidate_id not in selected:
            selected.append(candidate_id)
    if len(selected) < 2:
        alternatives = sorted(
            (row for row in summary["candidate_rows"]
             if row["candidate_id"] != selected[0]),
            key=lambda row: (
                row["n_runaway_networks"] > 0,
                row["selection_score_equal_network"], row["candidate_id"],
            ),
        )
        selected.append(alternatives[0]["candidate_id"])
    for candidate_id in selected:
        if not candidate_id.startswith("joint_"):
            raise RuntimeError("NLC2 parent must contain both E->E and E->I")
        if by_id[candidate_id]["n_runaway_networks"]:
            raise RuntimeError("NLC2 parent cannot be a runaway candidate")
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    root = ROOT / config["output_root"]
    manifest_path = root / "candidate_manifest.json"
    summary_path = root / "canary_summary_returned_only.json"
    verdict_path = root / "canary_verdict.json"
    manifest = json.loads(manifest_path.read_text())
    summary = json.loads(summary_path.read_text())
    verdict = json.loads(verdict_path.read_text())
    if manifest.get("status") != "REV11NLC_LOCAL_CONNECTIVITY_LIBRARY_FROZEN":
        raise RuntimeError("NLC1 manifest is not frozen")
    if summary.get("status") != "REV11NLC_RETURNED_ONLY_CANARY_COMPLETE":
        raise RuntimeError("NLC1 aggregate is incomplete")
    contract_path = ROOT / config["inputs"]["contact_contract"]["path"]
    contract = json.loads(contract_path.read_text())
    anchor_manifest = json.loads(
        (ROOT / config["inputs"]["node_anchor_manifest"]["path"]).read_text()
    )
    anchor = next(
        row for row in anchor_manifest["candidate_set"]["candidates"]
        if row["candidate_id"] == config["node_anchor"]["candidate_id"]
    )
    aggregate = {row["candidate_id"]: row for row in summary["candidate_rows"]}
    rows = []
    for source in manifest["candidate_set"]["candidates"]:
        candidate = deepcopy(source)
        candidate["node_field"] = deepcopy(anchor)
        row = _candidate_metrics(
            config_path, root, candidate, contract,
            config["search"]["kmeans_selection"],
        )
        row.update({
            "arm": source["arm"],
            "coefficients": source["coefficients"],
            "shaft_aware_equal_network_score": aggregate[
                source["candidate_id"]
            ]["selection_score_equal_network"],
            "mean_network_ood_fraction": aggregate[
                source["candidate_id"]
            ]["mean_network_ood_fraction"],
        })
        rows.append(row)
    parents = select_joint_search_parents(verdict, summary)
    payload = {
        "status": "REV11NLC_CONTACT_SPLIT_CROSS_FIT_RESCORING_COMPLETE",
        "patient_classifier_rank_sign_audit": patient_classifier_rank_audit(
            config, manifest, contract,
        ),
        "candidate_rows": rows,
        "nlc2_joint_search_parent_ids": parents,
        "parent_selection_contract": {
            "first": "NLC1 frozen-direction natural-KMeans diagnostic winner",
            "second": "NLC1 shaft-aware equal-network aggregate winner",
            "parents_are_search_centers_not_patient_recovery_claims": True,
        },
        "metric_contract": {
            "patient_readout": (
                "alternate contacts within each shaft; assign on one fold and "
                "evaluate rank-profile Spearman geometry on the disjoint fold"
            ),
            "natural_kmeans": (
                "per-network K=2 on all formal-clean returned events; align only "
                "after unsupervised clustering"
            ),
            "network_seed_is_independent_unit": True,
        },
        "claim_boundary": (
            "zero-simulation development rescoring; no patient-blind, complete "
            "interictal-distribution, core-causality, or ictal claim"
        ),
        "inputs": {
            "config": {"path": str(config_path.relative_to(ROOT)), "sha256": _sha256(config_path)},
            "manifest": {"path": str(manifest_path.relative_to(ROOT)), "sha256": _sha256(manifest_path)},
            "summary": {"path": str(summary_path.relative_to(ROOT)), "sha256": _sha256(summary_path)},
            "verdict": {"path": str(verdict_path.relative_to(ROOT)), "sha256": _sha256(verdict_path)},
            "analysis_code": {"path": str(Path(__file__).resolve().relative_to(ROOT)), "sha256": _sha256(Path(__file__).resolve())},
        },
        "analysis_git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip(),
    }
    output = root / "canary_crossfit_rescore.json"
    atomic_write_json(_jsonable(payload), output)
    print(json.dumps({
        "status": payload["status"],
        "nlc2_joint_search_parent_ids": parents,
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
