"""Audit one ZM1.1 tau_adp phase without collapsing it to one score."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_topic4_rev10_zm1_data_driven_h_zm import (  # noqa: E402
    _arm_audit,
    _signed_geometry_margin,
)
from scripts.freeze_topic4_rev10_zm1_1_tau_library import (  # noqa: E402
    CONTROL_ID,
    STATUS_BY_PHASE,
)
from scripts.run_topic4_rev9l_forced_source_worker import (  # noqa: E402
    _runtime_provenance,
    _sha256,
)
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402


SUMMARY_BY_PHASE = {
    "fit": "fit_screen_summary_returned_only.json",
    "selection": "selection_summary_returned_only.json",
    "confirmation": "confirmation_summary_returned_only.json",
}
DECISION_BY_PHASE = {
    "fit": "tau_adp_fit_decision.json",
    "selection": "tau_adp_selection_decision.json",
    "confirmation": "tau_adp_confirmation_decision.json",
}


def _worst_shape(activity):
    values = [
        activity.get("mean_network_shape_A"),
        activity.get("mean_network_shape_B"),
    ]
    return None if any(value is None for value in values) else float(max(values))


def _metric_row(arm):
    activity = arm["activity"]
    canonical = arm["canonical_fig4c_kmeans"]
    matrix = arm["supervised_direction_vs_patient_spearman"]
    return {
        "candidate_id": arm["candidate_id"],
        "tau_adp_ms": arm["slow_state"]["by_network"][0].get("tau_adp"),
        "n_runaway_networks": int(arm["slow_state"]["n_runaway_networks"]),
        "networks_with_both_clean_modes": int(
            activity["networks_with_both_clean_modes"]
        ),
        "mean_returned_events_per_network": float(
            activity["mean_network_returned_events_scored"]
        ),
        "mean_ood_fraction": float(activity["mean_network_ood_fraction"]),
        "worst_supervised_shape_distance": _worst_shape(activity),
        "supervised_patient_matrix_MTA_MTB_by_TA_TB": matrix,
        "supervised_signed_geometry_margin": _signed_geometry_margin(matrix),
        "kmeans_status": canonical["status"],
        "kmeans_direction_purity": canonical.get("direction_purity"),
        "kmeans_patient_q05": canonical.get(
            "patient_matched_direction_purity", {}
        ).get("q05"),
        "kmeans_patient_matrix_MTA_MTB_by_TA_TB": canonical.get(
            "cluster_vs_patient_spearman_MTA_MTB_by_TA_TB"
        ),
        "kmeans_signed_geometry_margin": canonical.get(
            "cluster_vs_patient_signed_geometry_margin"
        ),
        "kmeans_stability_ami_median": canonical.get(
            "kmeans_stability_ami_median"
        ),
        "zm_dynamically_engaged": bool(
            arm["slow_state"]["equal_network_summary"][
                "mean_fraction_above_z_threshold"
            ]["mean_across_networks"] not in (None, 0.0)
            and arm["slow_state"]["equal_network_summary"][
                "peak_mean_adaptation_current"
            ]["maximum_across_networks"] not in (None, 0.0)
        ),
    }


def _dominates(left, right):
    maximize = ("kmeans_signed_geometry_margin", "kmeans_direction_purity")
    minimize = ("worst_supervised_shape_distance", "mean_ood_fraction")
    values = [left[key] is not None and right[key] is not None
              for key in (*maximize, *minimize)]
    if not all(values):
        return False
    no_worse = all(left[key] >= right[key] for key in maximize)
    no_worse &= all(left[key] <= right[key] for key in minimize)
    strict = any(left[key] > right[key] for key in maximize)
    strict |= any(left[key] < right[key] for key in minimize)
    return bool(no_worse and strict)


def _eligible(row):
    return bool(
        row["n_runaway_networks"] == 0
        and row["kmeans_status"] == "EVALUABLE"
        and row["zm_dynamically_engaged"]
        and all(row[key] is not None for key in (
            "kmeans_signed_geometry_margin", "kmeans_direction_purity",
            "worst_supervised_shape_distance", "mean_ood_fraction",
        ))
    )


def _pareto(rows):
    eligible = [row for row in rows if _eligible(row)]
    return [
        row for row in eligible
        if not any(_dominates(other, row) for other in eligible if other is not row)
    ]


def _predeclared_order(row):
    """Lexicographic ordering, kept explicit instead of inventing a score."""
    return (
        -float(row["kmeans_signed_geometry_margin"]),
        -float(row["kmeans_direction_purity"]),
        float(row["worst_supervised_shape_distance"]),
        float(row["mean_ood_fraction"]),
        str(row["candidate_id"]),
    )


def build_decision(config_path, expected_commit):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    phase = config["search"]["phase"]
    root = ROOT / config["output_root"]
    manifest_path = root / "candidate_manifest.json"
    summary_path = root / SUMMARY_BY_PHASE[phase]
    manifest = json.loads(manifest_path.read_text())
    summary = json.loads(summary_path.read_text())
    if manifest.get("status") != STATUS_BY_PHASE[phase]:
        raise RuntimeError("ZM1.1 manifest phase is invalid")
    expected_summary = {
        "fit": "REV10R_RETURNED_ONLY_FIT_SCREEN_COMPLETE",
        "selection": "REV10R_RETURNED_ONLY_SELECTION_COMPLETE",
        "confirmation": "REV10R_RETURNED_ONLY_CONFIRMATION_COMPLETE",
    }[phase]
    if summary.get("status") != expected_summary:
        raise RuntimeError("ZM1.1 equal-network aggregation is incomplete")
    rows_by_id = {
        row["candidate_id"]: row for row in summary["candidate_rows"]
    }
    arm_ids = manifest["candidate_set"]["paired_candidate_ids"]
    arms = {
        candidate: _arm_audit(
            config_path, root, candidate, rows_by_id[candidate],
        )
        for candidate in arm_ids
    }
    active_rows = [_metric_row(arms[value]) for value in arm_ids if value != CONTROL_ID]
    pareto = sorted(_pareto(active_rows), key=_predeclared_order)
    payload = {
        "status": f"REV10ZM1_1_TAU_{phase.upper()}_COMPLETE",
        "phase": phase,
        "scientific_role": config["scientific_role"],
        "selection_contract": {
            "single_composite_score_used": False,
            "safety_eligibility": "zero runaway networks",
            "measurement_eligibility": "natural KMeans evaluable and Z/M engaged",
            "pareto_metrics": {
                "maximize": [
                    "kmeans_signed_geometry_margin", "kmeans_direction_purity",
                ],
                "minimize": [
                    "worst_supervised_shape_distance", "mean_ood_fraction",
                ],
            },
            "within_pareto_order": [
                "kmeans_signed_geometry_margin descending",
                "kmeans_direction_purity descending",
                "worst_supervised_shape_distance ascending",
                "mean_ood_fraction ascending",
            ],
        },
        "candidate_rows": active_rows,
        "pareto_candidate_ids": [row["candidate_id"] for row in pareto],
        "control": arms[CONTROL_ID],
        "arms": arms,
        "claim_boundary": config["claim_boundary"],
        "provenance": _runtime_provenance(expected_commit),
        "inputs": {
            "config": {"path": str(config_path.relative_to(ROOT)),
                       "sha256": _sha256(config_path)},
            "manifest": {"path": str(manifest_path.relative_to(ROOT)),
                         "sha256": _sha256(manifest_path)},
            "summary": {"path": str(summary_path.relative_to(ROOT)),
                        "sha256": _sha256(summary_path)},
        },
    }
    if phase == "fit":
        payload["shortlisted_candidate_ids"] = [
            row["candidate_id"] for row in pareto[:2]
        ]
        if not payload["shortlisted_candidate_ids"]:
            payload["status"] = "REV10ZM1_1_TAU_FIT_NO_SAFE_EVALUABLE_CANDIDATE"
    elif phase == "selection":
        payload["selected_candidate_id"] = (
            pareto[0]["candidate_id"] if pareto else None
        )
        if payload["selected_candidate_id"] is None:
            payload["status"] = (
                "REV10ZM1_1_TAU_SELECTION_NO_SAFE_EVALUABLE_CANDIDATE"
            )
    else:
        selected = manifest["selection_freeze"]["selected_nonzero_candidate_id"]
        row = next(value for value in active_rows if value["candidate_id"] == selected)
        matrix = np.asarray(row["kmeans_patient_matrix_MTA_MTB_by_TA_TB"], float)
        q05 = row["kmeans_patient_q05"]
        checks = {
            "no_runaway": row["n_runaway_networks"] == 0,
            "same_network_dual_modes": row["networks_with_both_clean_modes"] >= 4,
            "returned_events_present": row["mean_returned_events_per_network"] > 0,
            "natural_kmeans_evaluable": row["kmeans_status"] == "EVALUABLE",
            "natural_kmeans_reaches_patient_q05": (
                q05 is not None and row["kmeans_direction_purity"] >= q05
            ),
            "natural_kmeans_patient_signed_geometry": bool(
                matrix.shape == (2, 2) and np.all(np.isfinite(matrix))
                and matrix[0, 0] > 0 and matrix[1, 1] > 0
                and matrix[0, 1] < 0 and matrix[1, 0] < 0
            ),
            "z_m_dynamically_engaged": row["zm_dynamically_engaged"],
        }
        payload["selected_candidate_id"] = selected
        payload["scientific_acceptance_checks"] = checks
        payload["scientific_acceptance"] = bool(all(checks.values()))
    provenance = payload["provenance"]
    if (provenance["runtime_modules_dirty"]
            or not provenance["runtime_modules_match_expected_commit"]):
        raise RuntimeError("ZM1.1 audit modules are not frozen")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    payload = build_decision(args.config, args.expected_commit)
    root = ROOT / config["output_root"]
    output = root / DECISION_BY_PHASE[payload["phase"]]
    atomic_write_json(payload, output)
    print(json.dumps({
        "status": payload["status"],
        "pareto_candidate_ids": payload["pareto_candidate_ids"],
        "shortlisted_candidate_ids": payload.get("shortlisted_candidate_ids"),
        "selected_candidate_id": payload.get("selected_candidate_id"),
        "scientific_acceptance": payload.get("scientific_acceptance"),
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
