#!/usr/bin/env python3
"""Adjudicate final rev11-NLC frozen-substrate confirmation."""
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
from scripts.audit_topic4_rev11_nlc_joint_node_connectivity_selection import (  # noqa: E402
    _network_rows,
    paired_network_bootstrap,
)
from scripts.rescore_topic4_rev10_d6_natural_kmeans import (  # noqa: E402
    _candidate_metrics,
    _jsonable,
)
from scripts.run_topic4_rev9l_forced_source_worker import _sha256  # noqa: E402
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/topic4_rev11_nlc_frozen_substrate_confirmation.json"
ARM_IDS = (
    "node_baseline", "joint_04_ee_only", "joint_04_etoi_only",
    "joint_04_control",
)


def factorial_interaction(rows_by_candidate, config):
    seeds = sorted(set.intersection(*(
        set(rows_by_candidate[candidate_id]) for candidate_id in ARM_IDS
    )), key=int)
    fields = ("natural", "crossfit", "shape", "ood", "occupancy")

    def endpoint(candidate_id, seed, field):
        row = rows_by_candidate[candidate_id][seed]
        if field == "shape":
            return max(row["shape_A"], row["shape_B"])
        return row[field]

    def interaction(sampled, field):
        means = {
            candidate_id: float(np.mean([
                endpoint(candidate_id, seed, field) for seed in sampled
            ]))
            for candidate_id in ARM_IDS
        }
        return (
            means["joint_04_control"] - means["joint_04_ee_only"]
            - means["joint_04_etoi_only"] + means["node_baseline"]
        )

    if not seeds:
        return {"status": "NOT_EVALUABLE", "n_paired_networks": 0}
    rng = np.random.default_rng(int(
        config["search"]["paired_network_bootstrap"]["seed"]
    ) + 1)
    draws = int(config["search"]["paired_network_bootstrap"]["draws"])
    output = {}
    for field in fields:
        values = []
        for _ in range(draws):
            indices = rng.integers(0, len(seeds), size=len(seeds))
            values.append(interaction([seeds[index] for index in indices], field))
        values = np.asarray(values, float)
        output[field] = {
            "observed_interaction": float(interaction(seeds, field)),
            "bootstrap_q05": float(np.quantile(values, 0.05)),
            "bootstrap_q50": float(np.quantile(values, 0.50)),
            "bootstrap_q95": float(np.quantile(values, 0.95)),
            "positive_means_superadditive_for": (
                "alignment" if field in {"natural", "crossfit"}
                else "loss_or_occupancy"
            ),
        }
    return {
        "status": "OK",
        "n_paired_networks": len(seeds),
        "network_seeds": list(map(int, seeds)),
        "definition": "joint - EE_only - E_to_I_only + Node",
        "endpoints": output,
    }


def structure_audit(root, manifest, config):
    expected_ids = {
        row["candidate_id"] for row in manifest["candidate_set"]["candidates"]
    }
    expected_seeds = set(config["search"]["confirmation_network_seeds"])
    rows, bad = [], []
    for path in sorted((root / "workers").glob("*.json")):
        payload = json.loads(path.read_text())
        edge = payload["edge_audit"]
        provenance = payload["provenance"]
        record = {
            "candidate_id": payload["candidate"]["candidate_id"],
            "seed": int(payload["seed"]),
            "topology_unchanged": edge["topology_unchanged"],
            "delay_assignment_unchanged": edge["delay_assignment_unchanged"],
            "gaba_unchanged": edge["gaba_unchanged"],
            "max_incoming_E_to_E_error": edge["pathway_audit"]["E_to_E"][
                "max_abs_incoming_error"
            ],
            "max_incoming_E_to_I_error": edge["pathway_audit"]["E_to_I"][
                "max_abs_incoming_error"
            ],
            "edge_ratio": edge["edge_ratio"],
            "runaway_early_stop_ms": payload["run"]["runaway_early_stop_ms"],
            "runtime_modules_dirty": provenance["runtime_modules_dirty"],
            "runtime_modules_match_expected_commit": provenance[
                "runtime_modules_match_expected_commit"
            ],
            "expected_git_commit": provenance["expected_git_commit"],
        }
        valid = bool(
            record["candidate_id"] in expected_ids
            and record["seed"] in expected_seeds
            and record["topology_unchanged"]
            and record["delay_assignment_unchanged"]
            and record["gaba_unchanged"]
            and record["max_incoming_E_to_E_error"] <= 1e-9
            and record["max_incoming_E_to_I_error"] <= 1e-9
            and record["runaway_early_stop_ms"] is None
            and not record["runtime_modules_dirty"]
            and record["runtime_modules_match_expected_commit"] is True
        )
        if not valid:
            bad.append(record)
        rows.append(record)
    expected_count = len(expected_ids) * len(expected_seeds)
    return {
        "status": "PASS" if len(rows) == expected_count and not bad else "FAIL",
        "n_expected_workers": expected_count,
        "n_audited_workers": len(rows),
        "n_invalid_workers": len(bad),
        "invalid_workers": bad,
        "max_incoming_E_to_E_error": float(max(
            (row["max_incoming_E_to_E_error"] for row in rows), default=np.nan,
        )),
        "max_incoming_E_to_I_error": float(max(
            (row["max_incoming_E_to_I_error"] for row in rows), default=np.nan,
        )),
        "minimum_edge_ratio": float(min(
            (row["edge_ratio"]["min"] for row in rows), default=np.nan,
        )),
        "maximum_edge_ratio": float(max(
            (row["edge_ratio"]["max"] for row in rows), default=np.nan,
        )),
        "worker_commits": sorted({row["expected_git_commit"] for row in rows}),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    root = ROOT / config["output_root"]
    manifest_path = root / "candidate_manifest.json"
    summary_path = root / "confirmation_summary_returned_only.json"
    manifest = json.loads(manifest_path.read_text())
    summary = json.loads(summary_path.read_text())
    if manifest.get("status") != (
        "REV11NLC_FROZEN_SUBSTRATE_CONFIRMATION_LIBRARY_FROZEN"
    ):
        raise RuntimeError("NLC3C manifest is not frozen")
    if summary.get("status") != (
        "REV11NLC_FROZEN_CONFIRMATION_RETURNED_ONLY_COMPLETE"
    ):
        raise RuntimeError("NLC3C aggregate is incomplete")
    contract = json.loads(
        (ROOT / config["inputs"]["contact_contract"]["path"]).read_text()
    )
    aggregate = {row["candidate_id"]: row for row in summary["candidate_rows"]}
    details = summary["candidate_details"]
    rows, network_rows = [], {}
    candidates = {
        row["candidate_id"]: row for row in manifest["candidate_set"]["candidates"]
    }
    for candidate_id in ARM_IDS:
        candidate = candidates[candidate_id]
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
        network_rows[candidate_id] = _network_rows(metrics, details[candidate_id])
        evaluable = bool(
            score is not None
            and len(network_rows[candidate_id]) >= config["search"][
                "acceptance"
            ]["minimum_evaluable_joint_networks"]
            and aggregate_row["n_runaway_networks"] == 0
        )
        rows.append({
            "candidate_id": candidate_id,
            "arm": candidate["arm"],
            "selection_score": score if evaluable else None,
            "evaluable": evaluable,
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
    row_by_id = {row["candidate_id"]: row for row in rows}
    joint = row_by_id["joint_04_control"]
    joint_metrics = joint["crossfit_and_natural_metrics"]
    comparisons = {
        candidate_id: paired_network_bootstrap(
            candidates["joint_04_control"], candidates[candidate_id],
            network_rows, config,
        )
        for candidate_id in ARM_IDS if candidate_id != "joint_04_control"
    }
    structural = structure_audit(root, manifest, config)
    acceptance = config["search"]["acceptance"]
    natural = joint_metrics["natural_balanced_alignment_equal_network"]
    crossfit = joint_metrics["crossfit_margin_equal_network"]
    directional_pass = bool(
        len(network_rows["joint_04_control"]) >= acceptance[
            "minimum_evaluable_joint_networks"
        ]
        and natural["network_bootstrap_q05"] > acceptance[
            "natural_alignment_q05_min"
        ]
    )
    geometry_pass = bool(
        crossfit["network_bootstrap_q05"] > acceptance["patient_margin_q05_min"]
    )
    edge_increment_pass = bool(
        comparisons["node_baseline"].get(
            "bootstrap_probability_selected_lower", 0.0,
        ) >= acceptance["edge_increment_probability_min"]
    )
    if structural["status"] != "PASS":
        status = "REV11NLC_FROZEN_SUBSTRATE_CONFIRMATION_INVALID"
    elif directional_pass and geometry_pass and edge_increment_pass:
        status = "REV11NLC_FROZEN_SUBSTRATE_CONFIRMATION_PASS"
    elif directional_pass and geometry_pass:
        status = "REV11NLC_FROZEN_SUBSTRATE_CONFIRMATION_PARTIAL"
    else:
        status = "REV11NLC_FROZEN_SUBSTRATE_CONFIRMATION_FAIL"
    payload = {
        "status": status,
        "primary_candidate_id": "joint_04_control",
        "primary_score": joint["selection_score"],
        "component_status": {
            "DIRECTIONAL_REPERTOIRE": "PASS" if directional_pass else "FAIL",
            "PATIENT_GEOMETRY": "PASS" if geometry_pass else "FAIL",
            "EDGE_INCREMENT": "PASS" if edge_increment_pass else "FAIL",
        },
        "figure_eligible": bool(
            structural["status"] == "PASS" and directional_pass and geometry_pass
        ),
        "candidate_rows": sorted(rows, key=lambda row: (
            row["selection_score"] is None,
            np.inf if row["selection_score"] is None else row["selection_score"],
        )),
        "paired_joint_comparisons": comparisons,
        "factorial_interaction": factorial_interaction(network_rows, config),
        "structure_and_provenance_audit": structural,
        "acceptance_contract": acceptance,
        "objective_contract": config["search"]["joint_objective"],
        "claim_boundary": config["claim_boundary"],
        "Z_M_role": (
            "off during final static confirmation; eligible only as the next "
            "frozen-substrate ictal transfer interface"
        ),
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
    output = root / "confirmation_verdict.json"
    atomic_write_json(_jsonable(payload), output)
    print(json.dumps({
        "status": status,
        "component_status": payload["component_status"],
        "figure_eligible": payload["figure_eligible"],
        "primary_score": payload["primary_score"],
    }, indent=2))


if __name__ == "__main__":
    main()
