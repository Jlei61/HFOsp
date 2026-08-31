#!/usr/bin/env python3
"""Patient-first aggregation of exploratory H2b v0.3 A3--A5 results."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.topic5_continuous_marked_state_h2b.contract import (  # noqa: E402
    CANONICAL_V0_3_RESULT_ROOT,
    atomic_csv,
    atomic_json,
    sha256_file,
    utc_now,
)


PRODUCER = Path(__file__).resolve()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sign_p(successes: int, total: int) -> float | None:
    if total <= 0:
        return None
    tail = sum(math.comb(total, value) for value in range(successes, total + 1)) / 2 ** total
    lower = sum(math.comb(total, value) for value in range(0, total - successes + 1)) / 2 ** total
    return float(min(1.0, 2.0 * min(tail, lower)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=CANONICAL_V0_3_RESULT_ROOT)
    parser.add_argument("--analysis-subdir", default="hazard")
    parser.add_argument(
        "--include-support-conditioned-exploration", action="store_true",
    )
    parser.add_argument("--include-diagnostic-exploration", action="store_true")
    args = parser.parse_args()
    root = args.result_root.resolve()
    analysis_subdir = str(args.analysis_subdir)
    manifests = sorted((root / analysis_subdir / "by_cell").glob(
        "*/seed_*/result.json"
    ))
    if not manifests:
        raise FileNotFoundError("no v0.3 hazard cells")
    rows, hashes = [], {}
    for path in manifests:
        payload = _json(path)
        strict_valid = (
            payload.get("status") != "COMPLETE_EXPLORATORY"
            or payload.get("claim_status")
            != "CLAIM_ROUTE_RELEASED_DEVELOPMENT_ONLY"
            or payload.get("A1_patient_stratum", {}).get("state_qualified")
            is not True
            or payload.get("A2_transfer_assay_sensitive") is not True
        )
        allow_exploration = bool(
            args.include_support_conditioned_exploration
            or args.include_diagnostic_exploration
        )
        exploratory_valid = bool(
            allow_exploration
            and payload.get("status") == "COMPLETE_EXPLORATORY"
            and payload.get("claim_status") in {
                "EXPLORATORY_A1_EMPTY_ASSAY_NOT_SENSITIVE_SUPPORT_CONDITIONED",
                "EXPLORATORY_A1_EMPTY_ASSAY_NOT_SENSITIVE_FULL_GRID",
            }
            and payload.get("analysis_scope") in {
                "seizure_support_conditioned_control_grid_exploratory",
                "full_recorded_development_grid_exploratory",
            }
        )
        if strict_valid and not exploratory_valid:
            raise ValueError(f"unreleased or incomplete hazard cell: {path}")
        hashes[str(path)] = sha256_file(path)
        primary = payload["primary_selected_k"]
        wrong = payload["matched_wrong_time"]["result"]
        lag = {str(row["tau_multiplier"]): row for row in
               payload["tau_lag_response"]["rows"]}
        row = {
            "subject": payload["subject"], "seed": payload["seed"],
            "initial_k": payload["selected_initial_k"],
            "status": primary["status"],
            "n_grid_rows": payload["n_grid_rows"],
            "n_supported_seizures": primary.get("n_supported_seizures"),
            "n_oof_seizures": primary.get("n_oof_seizures"),
            "T_relative_improvement": primary.get("T_relative_improvement"),
            "M_relative_improvement": primary.get("M_relative_improvement"),
            "T_direction_favourable": primary.get("T_direction_favourable", False),
            "M_direction_favourable": primary.get("M_direction_favourable", False),
            "wrong_time_T_relative_improvement": wrong.get("T_relative_improvement"),
            "correct_minus_wrong_T_improvement": (
                float(primary["T_relative_improvement"]
                      - wrong["T_relative_improvement"])
                if primary.get("T_relative_improvement") is not None
                and wrong.get("T_relative_improvement") is not None else None
            ),
            "A1_exploration_stratum": payload["A1_patient_stratum"].get(
                "exploration_stratum"
            ),
            "A1_state_qualified": payload["A1_patient_stratum"].get(
                "state_qualified", False
            ),
            "tau_minutes": payload["tau_lag_response"]["tau_minutes"],
        }
        for multiplier in (0.5, 1.0, 2.0, 4.0):
            item = lag.get(str(multiplier))
            row[f"lag_{multiplier:g}tau_T_relative_improvement"] = (
                item["result"].get("T_relative_improvement") if item else None
            )
            row[f"lag_{multiplier:g}tau_valid_fraction"] = (
                item.get("valid_past_donor_fraction") if item else None
            )
        rows.append(row)
    subjects = sorted({str(row["subject"]) for row in rows})
    patient_rows = []
    for subject in subjects:
        cells = [row for row in rows if row["subject"] == subject]
        complete = [row for row in cells if row["status"] == "COMPLETE_EXPLORATORY"]
        if not complete:
            patient_rows.append({
                "subject": subject, "status": "NOT_ESTIMABLE",
                "n_seeds": len(cells), "n_complete_seeds": 0,
            })
            continue
        t = [row["T_relative_improvement"] for row in complete]
        m = [row["M_relative_improvement"] for row in complete]
        d = [row["correct_minus_wrong_T_improvement"] for row in complete
             if row["correct_minus_wrong_T_improvement"] is not None]
        patient = {
            "subject": subject, "status": "COMPLETE_EXPLORATORY",
            "n_seeds": len(cells), "n_complete_seeds": len(complete),
            "median_n_oof_seizures": float(np.median([
                row["n_oof_seizures"] for row in complete
            ])),
            "median_T_relative_improvement": float(np.median(t)),
            "median_M_relative_improvement": float(np.median(m)),
            "n_T_seed_favourable": int(sum(value > 0 for value in t)),
            "n_M_seed_favourable": int(sum(value > 0 for value in m)),
            "median_correct_minus_wrong_T_improvement": (
                float(np.median(d)) if d else None
            ),
            "n_correct_better_than_wrong_seed": int(sum(value > 0 for value in d)),
            "A1_exploration_stratum": complete[0]["A1_exploration_stratum"],
            "A1_state_qualified": complete[0]["A1_state_qualified"],
        }
        for multiplier in (0.5, 1.0, 2.0, 4.0):
            values = [row[f"lag_{multiplier:g}tau_T_relative_improvement"]
                      for row in complete
                      if row[f"lag_{multiplier:g}tau_T_relative_improvement"] is not None]
            fractions = [row[f"lag_{multiplier:g}tau_valid_fraction"]
                         for row in complete
                         if row[f"lag_{multiplier:g}tau_valid_fraction"] is not None]
            patient[f"median_lag_{multiplier:g}tau_T_relative_improvement"] = (
                float(np.median(values)) if values else None
            )
            patient[f"median_lag_{multiplier:g}tau_valid_fraction"] = (
                float(np.median(fractions)) if fractions else None
            )
        patient_rows.append(patient)
    estimable = [row for row in patient_rows if row["status"] == "COMPLETE_EXPLORATORY"]
    t_success = sum(row["median_T_relative_improvement"] > 0 for row in estimable)
    m_success = sum(row["median_M_relative_improvement"] > 0 for row in estimable)
    d_values = [row for row in estimable
                if row["median_correct_minus_wrong_T_improvement"] is not None]
    d_success = sum(row["median_correct_minus_wrong_T_improvement"] > 0
                    for row in d_values)
    payload = {
        "status": "COMPLETE_EXPLORATORY_ASSAY_NOT_SENSITIVE",
        "revision": "h2b_v0_3_hazard_patient_first_v2",
        "created_utc": utc_now(), "n_cells": len(rows),
        "n_patients": len(patient_rows), "n_estimable_patients": len(estimable),
        "patient_rows": patient_rows,
        "analysis_subdir": analysis_subdir,
        "analysis_scopes": sorted({
            _json(path).get("analysis_scope") for path in manifests
        }),
        "cohort_direction": {
            "T": {"favourable": t_success, "total": len(estimable),
                  "two_sided_sign_p": _sign_p(t_success, len(estimable))},
            "M": {"favourable": m_success, "total": len(estimable),
                  "two_sided_sign_p": _sign_p(m_success, len(estimable))},
            "correct_time_better_than_wrong": {
                "favourable": d_success, "total": len(d_values),
                "two_sided_sign_p": _sign_p(d_success, len(d_values)),
            },
        },
        "source_manifest_sha256": hashes,
        "producer_sha256": sha256_file(PRODUCER),
        "patient_is_inference_unit": True,
        "seed_is_not_patient_replicate": True,
        "negative_result_biological_interpretation_allowed": False,
        "reason": "A2 transfer assay has calibrated null but inadequate transfer power",
        "formal_test_partition_opened": False, "sealed_opened": False,
        "h3_or_t2_run": False,
    }
    output = root / analysis_subdir
    atomic_json(output / "patient_first_summary.json", payload)
    atomic_csv(output / "per_cell_metrics.csv", rows)
    atomic_csv(output / "per_patient_metrics.csv", patient_rows)
    audit_name = (
        "scientific_route_audit_A3_A5.json" if analysis_subdir == "hazard"
        else f"scientific_route_audit_A3_A5_{analysis_subdir}.json"
    )
    atomic_json(root / "reports" / audit_name, {
        "status": payload["status"], "created_utc": payload["created_utc"],
        "core_question": (
            "does an interictal-only frozen state add later seizure-risk information, "
            "and does that information decay when the state is taken from older times"
        ),
        "route_drift": False, "patient_first": True,
        "assay_sensitive": False,
        "allowed_claim": "development exploration only; no biological negative",
        "source_summary": str(output / "patient_first_summary.json"),
        "source_summary_sha256": sha256_file(output / "patient_first_summary.json"),
    })
    print(f"COMPLETE patients={len(patient_rows)} estimable={len(estimable)} cells={len(rows)}")


if __name__ == "__main__":
    main()
