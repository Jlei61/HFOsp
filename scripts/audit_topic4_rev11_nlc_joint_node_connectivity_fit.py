#!/usr/bin/env python3
"""Adjudicate NLC2 with equal-network cross-fit, KMeans and shaft metrics."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.rescore_topic4_rev10_d6_natural_kmeans import (  # noqa: E402
    _candidate_metrics,
    _jsonable,
)
from scripts.run_topic4_rev9l_forced_source_worker import _sha256  # noqa: E402
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_shaft_aware import centered_smooth_max  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/topic4_rev11_nlc_joint_node_connectivity_fit.json"


def joint_fit_score(metrics, aggregate, candidate, objective, field_search, edge_basis):
    natural = metrics.get("natural_balanced_alignment_equal_network")
    crossfit = metrics.get("crossfit_margin_equal_network")
    if natural is None or crossfit is None:
        return None, None
    natural_value = float(natural["equal_network_mean"])
    crossfit_value = float(crossfit["equal_network_mean"])
    shaft_score = max(
        float(aggregate["mean_network_shape_A"]),
        float(aggregate["mean_network_shape_B"]),
    )
    components = {
        "natural_kmeans_loss": 1.0 - natural_value,
        "contact_split_patient_loss": 0.5 * (1.0 - np.clip(crossfit_value, -1.0, 1.0)),
        "shaft_aware_loss": min(1.0, math.log1p(max(0.0, shaft_score)) / math.log(9.0)),
    }
    coordinates = candidate.get("search_coordinates", {})
    max_amplitude = float(field_search["residual_log_rms_amplitude_range"][1])
    node_energy = (float(coordinates.get("node_amplitude", 0.0)) / max_amplitude) ** 2
    edge_delta = np.asarray(coordinates.get("edge_delta", np.zeros((2, 6))), float)
    edge_scale = np.asarray(edge_basis["local_delta_half_width"], float)[None, :]
    edge_energy = float(np.mean((edge_delta / edge_scale) ** 2))
    regularization = 0.5 * (node_energy + edge_energy)
    score = centered_smooth_max(
        list(components.values()), float(objective["weakest_component_temperature"]),
    )
    recruitment = metrics.get("recruitment_worst_mode_error")
    score += float(objective["recruitment_weight"]) * (
        1.0 if recruitment is None else float(recruitment)
    )
    score += float(objective["ood_weight"]) * float(
        aggregate["mean_network_ood_fraction"]
    )
    score += float(objective["detector_occupancy_weight"]) * float(
        aggregate["mean_network_fraction_time_above_detector"]
    )
    score += float(objective["perturbation_energy_weight"]) * regularization
    return float(score), {
        **components,
        "natural_kmeans_alignment": natural_value,
        "contact_split_patient_margin": crossfit_value,
        "shaft_aware_raw_worst_mode_score": shaft_score,
        "recruitment_worst_mode_error": recruitment,
        "mean_network_ood_fraction": aggregate["mean_network_ood_fraction"],
        "mean_network_detector_occupancy": aggregate[
            "mean_network_fraction_time_above_detector"
        ],
        "normalized_perturbation_energy": regularization,
    }


def _pareto(rows):
    keys = (
        "natural_kmeans_loss", "contact_split_patient_loss",
        "shaft_aware_loss", "recruitment_worst_mode_error",
        "mean_network_ood_fraction",
    )
    evaluable = [row for row in rows if row["selection_score"] is not None]
    for row in rows:
        row["pareto_nondominated"] = False
    for row in evaluable:
        current = np.asarray([
            1.0 if row["objective_components"].get(key) is None
            else row["objective_components"][key] for key in keys
        ], float)
        dominated = False
        for other in evaluable:
            if other is row:
                continue
            candidate = np.asarray([
                1.0 if other["objective_components"].get(key) is None
                else other["objective_components"][key] for key in keys
            ], float)
            if np.all(candidate <= current) and np.any(candidate < current):
                dominated = True
                break
        row["pareto_nondominated"] = not dominated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    root = ROOT / config["output_root"]
    manifest_path = root / "candidate_manifest.json"
    summary_path = root / "canary_summary_returned_only.json"
    manifest = json.loads(manifest_path.read_text())
    summary = json.loads(summary_path.read_text())
    if manifest.get("status") != "REV11NLC_JOINT_NODE_CONNECTIVITY_FIT_LIBRARY_FROZEN":
        raise RuntimeError("NLC2 manifest is not frozen")
    if summary.get("status") != "REV11NLC_JOINT_FIT_RETURNED_ONLY_COMPLETE":
        raise RuntimeError("NLC2 aggregate is incomplete")
    contract = json.loads(
        (ROOT / config["inputs"]["contact_contract"]["path"]).read_text()
    )
    aggregate = {row["candidate_id"]: row for row in summary["candidate_rows"]}
    rows = []
    for candidate in manifest["candidate_set"]["candidates"]:
        metrics = _candidate_metrics(
            config_path, root, candidate, contract,
            config["search"]["kmeans_selection"],
        )
        aggregate_row = aggregate[candidate["candidate_id"]]
        score, components = joint_fit_score(
            metrics, aggregate_row, candidate,
            config["search"]["joint_objective"], config["field_search"],
            config["local_connectivity_basis"],
        )
        evaluable = bool(
            score is not None
            and metrics["n_natural_kmeans_evaluable_networks"] >= config[
                "search"
            ]["kmeans_selection"]["minimum_networks_with_three_clean_events_per_mode"]
            and aggregate_row["n_runaway_networks"] == 0
        )
        rows.append({
            "candidate_id": candidate["candidate_id"],
            "arm": candidate["arm"],
            "search_coordinates": candidate.get("search_coordinates"),
            "node_field_sha256": candidate["node_field"]["field_sha256"],
            "coefficients": candidate["coefficients"],
            "evaluable": evaluable,
            "selection_score": score if evaluable else None,
            "objective_components": components,
            "crossfit_and_natural_metrics": metrics,
            "aggregate_activity": {
                key: aggregate_row[key] for key in (
                    "n_runaway_networks", "mean_network_events",
                    "mean_network_fraction_time_above_detector",
                    "mean_network_ood_fraction", "networks_with_both_clean_modes",
                    "max_incoming_E_to_E_error", "max_incoming_E_to_I_error",
                )
            },
        })
    _pareto(rows)
    ranked = sorted(rows, key=lambda row: (
        row["selection_score"] is None,
        np.inf if row["selection_score"] is None else row["selection_score"],
        row["candidate_id"],
    ))
    evaluable = [row for row in ranked if row["selection_score"] is not None]
    selected = None if not evaluable else evaluable[0]
    payload = {
        "status": (
            "REV11NLC_JOINT_FIT_NO_EVALUABLE_CANDIDATE"
            if selected is None
            else "REV11NLC_JOINT_FIT_EXPLORATORY_CANDIDATE_FOUND"
        ),
        "selected_candidate_id": None if selected is None else selected["candidate_id"],
        "selected_score": None if selected is None else selected["selection_score"],
        "pareto_candidate_ids": [
            row["candidate_id"] for row in ranked if row["pareto_nondominated"]
        ],
        "fresh_selection_shortlist_ids": [
            row["candidate_id"] for row in evaluable[:5]
        ],
        "candidate_rows": ranked,
        "objective_contract": config["search"]["joint_objective"],
        "selection_is_exploratory_not_a_gate": True,
        "Z_M_role": "off during static substrate fit; reserved for frozen-substrate ictal transfer",
        "claim_boundary": config["claim_boundary"],
        "inputs": {
            "config": {"path": str(config_path.relative_to(ROOT)), "sha256": _sha256(config_path)},
            "manifest": {"path": str(manifest_path.relative_to(ROOT)), "sha256": _sha256(manifest_path)},
            "summary": {"path": str(summary_path.relative_to(ROOT)), "sha256": _sha256(summary_path)},
            "analysis_code": {"path": str(Path(__file__).resolve().relative_to(ROOT)), "sha256": _sha256(Path(__file__).resolve())},
        },
        "analysis_git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip(),
    }
    output = root / "canary_verdict.json"
    atomic_write_json(_jsonable(payload), output)
    print(json.dumps({
        "status": payload["status"],
        "selected_candidate_id": payload["selected_candidate_id"],
        "selected_score": payload["selected_score"],
        "fresh_selection_shortlist_ids": payload["fresh_selection_shortlist_ids"],
    }, indent=2))


if __name__ == "__main__":
    main()
