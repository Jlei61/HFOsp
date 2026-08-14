#!/usr/bin/env python3
"""Adjudicate the frozen NLC3 shortlist on fresh paired networks."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_topic4_rev11_nlc_joint_node_connectivity_fit import (  # noqa: E402
    _pareto,
    joint_fit_score,
)
from scripts.rescore_topic4_rev10_d6_natural_kmeans import (  # noqa: E402
    _candidate_metrics,
    _jsonable,
)
from scripts.run_topic4_rev9l_forced_source_worker import _sha256  # noqa: E402
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402


DEFAULT_CONFIG = (
    ROOT / "config/topic4_rev11_nlc_joint_node_connectivity_selection.json"
)


def _network_rows(metrics, aggregate_details):
    output = {}
    for seed, natural_row in metrics["natural_kmeans_by_network"].items():
        natural = natural_row["natural_kmeans"].get(
            "direction_balanced_alignment"
        )
        crossfit = natural_row["crossfit_patient_readout"].get("signed_margin")
        recruitment = natural_row["recruitment"]
        detail = aggregate_details["by_seed"][str(seed)]
        values = [
            natural, crossfit,
            recruitment["A"]["absolute_error_fraction_of_15"],
            recruitment["B"]["absolute_error_fraction_of_15"],
            detail["shape_by_mode"]["A"], detail["shape_by_mode"]["B"],
            detail["ood_fraction"],
            detail["fraction_time_above_common_detector"],
        ]
        if all(value is not None and np.isfinite(value) for value in values):
            output[str(seed)] = {
                "natural": float(natural),
                "crossfit": float(crossfit),
                "recruitment_A": float(values[2]),
                "recruitment_B": float(values[3]),
                "shape_A": float(values[4]),
                "shape_B": float(values[5]),
                "ood": float(values[6]),
                "occupancy": float(values[7]),
            }
    return output


def _score_network_sample(candidate, network_rows, config):
    rows = list(network_rows)
    if not rows:
        return None
    metrics = {
        "natural_balanced_alignment_equal_network": {
            "equal_network_mean": float(np.mean([row["natural"] for row in rows]))
        },
        "crossfit_margin_equal_network": {
            "equal_network_mean": float(np.mean([row["crossfit"] for row in rows]))
        },
        "recruitment_worst_mode_error": float(max(
            np.mean([row["recruitment_A"] for row in rows]),
            np.mean([row["recruitment_B"] for row in rows]),
        )),
    }
    aggregate = {
        "mean_network_shape_A": float(np.mean([row["shape_A"] for row in rows])),
        "mean_network_shape_B": float(np.mean([row["shape_B"] for row in rows])),
        "mean_network_ood_fraction": float(np.mean([row["ood"] for row in rows])),
        "mean_network_fraction_time_above_detector": float(np.mean([
            row["occupancy"] for row in rows
        ])),
    }
    return joint_fit_score(
        metrics, aggregate, candidate, config["search"]["joint_objective"],
        config["field_search"], config["local_connectivity_basis"],
    )[0]


def paired_network_bootstrap(selected, comparator, rows_by_candidate, config):
    selected_rows = rows_by_candidate[selected["candidate_id"]]
    comparator_rows = rows_by_candidate[comparator["candidate_id"]]
    seeds = sorted(set(selected_rows) & set(comparator_rows), key=int)
    if not seeds:
        return {"status": "NOT_EVALUABLE", "n_paired_networks": 0}
    observed = _score_network_sample(
        selected, [selected_rows[seed] for seed in seeds], config,
    ) - _score_network_sample(
        comparator, [comparator_rows[seed] for seed in seeds], config,
    )
    rng = np.random.default_rng(int(
        config["search"]["paired_network_bootstrap"]["seed"]
    ))
    draws = int(config["search"]["paired_network_bootstrap"]["draws"])
    deltas = []
    for _ in range(draws):
        indices = rng.integers(0, len(seeds), size=len(seeds))
        sampled = [seeds[index] for index in indices]
        left = _score_network_sample(
            selected, [selected_rows[seed] for seed in sampled], config,
        )
        right = _score_network_sample(
            comparator, [comparator_rows[seed] for seed in sampled], config,
        )
        if left is not None and right is not None:
            deltas.append(left - right)
    if not deltas:
        return {"status": "NOT_EVALUABLE", "n_paired_networks": len(seeds)}
    deltas = np.asarray(deltas, float)
    return {
        "status": "OK",
        "n_paired_networks": len(seeds),
        "network_seeds": list(map(int, seeds)),
        "delta_selected_minus_comparator": float(observed),
        "bootstrap_q05": float(np.quantile(deltas, 0.05)),
        "bootstrap_q50": float(np.quantile(deltas, 0.50)),
        "bootstrap_q95": float(np.quantile(deltas, 0.95)),
        "bootstrap_probability_selected_lower": float(np.mean(deltas < 0.0)),
        "draws": int(len(deltas)),
        "lower_score_is_better": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    root = ROOT / config["output_root"]
    manifest_path = root / "candidate_manifest.json"
    summary_path = root / "selection_summary_returned_only.json"
    manifest = json.loads(manifest_path.read_text())
    summary = json.loads(summary_path.read_text())
    if manifest.get("status") != (
        "REV11NLC_JOINT_NODE_CONNECTIVITY_SELECTION_LIBRARY_FROZEN"
    ):
        raise RuntimeError("NLC3 manifest is not frozen")
    if summary.get("status") != (
        "REV11NLC_JOINT_SELECTION_RETURNED_ONLY_COMPLETE"
    ):
        raise RuntimeError("NLC3 aggregate is incomplete")
    contract = json.loads(
        (ROOT / config["inputs"]["contact_contract"]["path"]).read_text()
    )
    aggregate = {row["candidate_id"]: row for row in summary["candidate_rows"]}
    aggregate_details = summary["candidate_details"]
    rows, network_rows = [], {}
    for candidate in manifest["candidate_set"]["candidates"]:
        candidate_id = candidate["candidate_id"]
        metrics = _candidate_metrics(
            config_path, root, candidate, contract,
            config["search"]["kmeans_selection"],
        )
        aggregate_row = aggregate[candidate_id]
        score, components = joint_fit_score(
            metrics, aggregate_row, candidate,
            config["search"]["joint_objective"], config["field_search"],
            config["local_connectivity_basis"],
        )
        network_rows[candidate_id] = _network_rows(
            metrics, aggregate_details[candidate_id]
        )
        evaluable = bool(
            score is not None
            and len(network_rows[candidate_id]) >= config["search"][
                "kmeans_selection"
            ]["minimum_networks_with_three_clean_events_per_mode"]
            and aggregate_row["n_runaway_networks"] == 0
        )
        rows.append({
            "candidate_id": candidate_id,
            "arm": candidate["arm"],
            "search_coordinates": candidate.get("search_coordinates"),
            "node_field_sha256": candidate["node_field"]["field_sha256"],
            "coefficients": candidate["coefficients"],
            "evaluable": evaluable,
            "selection_score": score if evaluable else None,
            "objective_components": components,
            "crossfit_and_natural_metrics": metrics,
            "network_objective_inputs": network_rows[candidate_id],
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
    manifest_candidates = {
        row["candidate_id"]: row for row in manifest["candidate_set"]["candidates"]
    }
    comparisons = {}
    if selected is not None:
        selected_candidate = manifest_candidates[selected["candidate_id"]]
        for comparator_id in (
            "node_baseline", "joint_03_control", "joint_04_control",
        ):
            comparator_row = next(
                row for row in rows if row["candidate_id"] == comparator_id
            )
            comparisons[comparator_id] = paired_network_bootstrap(
                selected_candidate, manifest_candidates[comparator_id],
                network_rows, config,
            )
            comparisons[comparator_id]["aggregate_selection_score"] = (
                comparator_row["selection_score"]
            )
    selected_id = None if selected is None else selected["candidate_id"]
    if selected_id == "node_baseline":
        winner_type = "NODE_ONLY_CONTROL"
    elif selected_id in {"joint_03_control", "joint_04_control"}:
        winner_type = "NLC1_PARENT_CONNECTIVITY_CONTROL"
    elif selected_id is None:
        winner_type = "NONE"
    else:
        winner_type = "JOINT_CONTINUOUS_NODE_RESIDUAL_AND_LOCAL_CONNECTIVITY"
    payload = {
        "status": (
            "REV11NLC_JOINT_SELECTION_NO_EVALUABLE_CANDIDATE"
            if selected is None else "REV11NLC_JOINT_SELECTION_COMPLETE"
        ),
        "selected_candidate_id": selected_id,
        "selected_score": None if selected is None else selected["selection_score"],
        "winner_type": winner_type,
        "candidate_rows": ranked,
        "paired_network_bootstrap": comparisons,
        "objective_contract": config["search"]["joint_objective"],
        "selection_is_fresh_network_development_not_confirmation": True,
        "Z_M_role": (
            "off during static substrate selection; reserved for frozen-substrate "
            "ictal transfer"
        ),
        "claim_boundary": config["claim_boundary"],
        "inputs": {
            "config": {
                "path": str(config_path.relative_to(ROOT)),
                "sha256": _sha256(config_path),
            },
            "manifest": {
                "path": str(manifest_path.relative_to(ROOT)),
                "sha256": _sha256(manifest_path),
            },
            "summary": {
                "path": str(summary_path.relative_to(ROOT)),
                "sha256": _sha256(summary_path),
            },
            "analysis_code": {
                "path": str(Path(__file__).resolve().relative_to(ROOT)),
                "sha256": _sha256(Path(__file__).resolve()),
            },
        },
        "analysis_git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip(),
    }
    output = root / "selection_verdict.json"
    atomic_write_json(_jsonable(payload), output)
    print(json.dumps({
        "status": payload["status"],
        "selected_candidate_id": selected_id,
        "selected_score": payload["selected_score"],
        "winner_type": winner_type,
    }, indent=2))


if __name__ == "__main__":
    main()
