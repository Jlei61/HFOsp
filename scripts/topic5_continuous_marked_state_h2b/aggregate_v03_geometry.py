#!/usr/bin/env python3
"""Patient-first aggregation of full-grid H2b v0.3 geometry exploration."""
from __future__ import annotations

import argparse
from collections import Counter
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
FAMILIES = ("basin_gating", "directed_approach", "abrupt_transition")


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sign_p(successes: int, total: int) -> float | None:
    if total <= 0:
        return None
    upper = sum(math.comb(total, value) for value in range(successes, total + 1)) / 2 ** total
    lower = sum(math.comb(total, value) for value in range(0, total - successes + 1)) / 2 ** total
    return float(min(1.0, 2.0 * min(upper, lower)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=CANONICAL_V0_3_RESULT_ROOT)
    args = parser.parse_args()
    root = args.result_root.resolve()
    paths = sorted((root / "geometry/by_cell").glob("*/seed_*/result.json"))
    if not paths:
        raise FileNotFoundError("no v0.3 geometry cells")
    cells, hashes = [], {}
    for path in paths:
        payload = _json(path)
        if (
            payload.get("revision") != "h2b_v0_3_oos_geometry_cell_v3"
            or payload.get("claim_status")
            != "EXPLORATORY_A1_EMPTY_ASSAY_NOT_SENSITIVE_FULL_GRID"
            or payload.get("common_extraction_domain") is not True
        ):
            raise ValueError(f"mixed or superseded geometry cell: {path}")
        hashes[str(path)] = sha256_file(path)
        folds = [row for row in payload["by_lookback_minutes"]["30"]
                 if row.get("status") == "COMPLETE_EXPLORATORY"]
        cell = {
            "subject": payload["subject"], "seed": payload["seed"],
            "status": "COMPLETE_EXPLORATORY" if folds else "NOT_ESTIMABLE",
            "n_complete_folds": len(folds),
            "n_attempted_folds": payload["n_primary_attempted_folds"],
            "A1_exploration_stratum": payload["A1_patient_stratum"].get(
                "exploration_stratum"
            ),
            "primary_not_estimable_reasons": payload.get(
                "primary_not_estimable_reasons", {}
            ),
        }
        for family in FAMILIES:
            values = [row["family_scores"].get(family) for row in folds
                      if row["family_scores"].get(family) is not None]
            cell[f"median_{family}"] = float(np.median(values)) if values else None
            cell[f"n_{family}_favourable_folds"] = int(sum(value > 0 for value in values))
        cells.append(cell)
    patients = []
    for subject in sorted({str(row["subject"]) for row in cells}):
        selected = [row for row in cells if row["subject"] == subject]
        complete = [row for row in selected if row["status"] == "COMPLETE_EXPLORATORY"]
        patient = {
            "subject": subject,
            "status": "COMPLETE_EXPLORATORY" if complete else "NOT_ESTIMABLE",
            "n_seeds": len(selected), "n_complete_seeds": len(complete),
            "n_complete_folds_across_seeds": int(sum(
                row["n_complete_folds"] for row in complete
            )),
            "A1_exploration_stratum": selected[0]["A1_exploration_stratum"],
        }
        for family in FAMILIES:
            values = [row[f"median_{family}"] for row in complete
                      if row[f"median_{family}"] is not None]
            patient[f"median_{family}"] = float(np.median(values)) if values else None
            patient[f"n_{family}_favourable_seeds"] = int(sum(value > 0 for value in values))
        patients.append(patient)
    estimable = [row for row in patients if row["status"] == "COMPLETE_EXPLORATORY"]
    not_estimable_fold_reasons: Counter[str] = Counter()
    not_estimable_cell_reasons: Counter[str] = Counter()
    for row in cells:
        reasons = row.get("primary_not_estimable_reasons", {})
        for reason, count in reasons.items():
            not_estimable_fold_reasons[str(reason)] += int(count)
        if row["status"] == "NOT_ESTIMABLE":
            for reason in reasons:
                not_estimable_cell_reasons[str(reason)] += 1
    direction = {}
    for family in FAMILIES:
        values = [row[f"median_{family}"] for row in estimable
                  if row[f"median_{family}"] is not None]
        favourable = int(sum(value > 0 for value in values))
        direction[family] = {
            "favourable": favourable, "total": len(values),
            "two_sided_sign_p": _sign_p(favourable, len(values)),
            "patient_median_score": float(np.median(values)) if values else None,
        }
    payload = {
        "status": "COMPLETE_EXPLORATORY_ASSAY_NOT_SENSITIVE",
        "revision": "h2b_v0_3_geometry_patient_first_v1",
        "created_utc": utc_now(), "n_cells": len(cells),
        "n_patients": len(patients), "n_estimable_patients": len(estimable),
        "patient_rows": patients, "cohort_direction": direction,
        "not_estimable_fold_reason_counts": dict(not_estimable_fold_reasons),
        "not_estimable_cell_reason_counts": dict(not_estimable_cell_reasons),
        "family_score_scale": "matched_control_signed_percentile_in_minus1_plus1",
        "common_extraction_domain": True,
        "old_cross_domain_abrupt_result_invalid": True,
        "patient_is_inference_unit": True, "seed_is_not_patient_replicate": True,
        "negative_result_biological_interpretation_allowed": False,
        "reason": "A1 multidimensional qualification empty and A2 transfer assay insensitive",
        "source_manifest_sha256": hashes,
        "producer_sha256": sha256_file(PRODUCER),
        "formal_test_partition_opened": False, "sealed_opened": False,
        "h3_or_t2_run": False,
    }
    output = root / "geometry"
    atomic_json(output / "patient_first_summary.json", payload)
    atomic_csv(output / "per_cell_metrics.csv", cells)
    atomic_csv(output / "per_patient_metrics.csv", patients)
    print(
        f"COMPLETE patients={len(patients)} estimable={len(estimable)} "
        f"cells={len(cells)}"
    )


if __name__ == "__main__":
    main()
