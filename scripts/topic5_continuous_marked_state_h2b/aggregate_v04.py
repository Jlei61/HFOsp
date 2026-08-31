#!/usr/bin/env python3
"""Patient-first aggregation for H2b v0.4 risk and OOS route geometry."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.topic5_continuous_marked_state_h2b.contract import (  # noqa: E402
    V0_4_RESULT_ROOT,
    atomic_csv,
    atomic_json,
    sha256_file,
    utc_now,
)


PRODUCER = Path(__file__).resolve()
R17B_SUMMARY = Path(
    "/home/honglab/leijiaxin/HFOsp/results/epi_prssm/continuous_marked_state/"
    "r1/r1_7b_cohort_extension/reports/r1_7a_summary.json"
)
EFFECTS = (
    "observation_minus_history",
    "route_state_minus_history",
    "route_state_minus_observation",
    "route_state_minus_memoryless",
    "route_state_minus_linear_state",
    "two_route_minus_single_axis_state",
    "correct_minus_wrong_time",
)
LEADS = (5, 15, 30, 60, 120)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _median(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values
              if value is not None and np.isfinite(float(value))]
    return float(np.median(finite)) if finite else None


def _sign_p(values: list[float], favourable_negative: bool) -> dict[str, Any]:
    positive = sum(value > 0 for value in values)
    negative = sum(value < 0 for value in values)
    zero = len(values) - positive - negative
    n = positive + negative
    larger = max(positive, negative)
    tail = sum(math.comb(n, k) for k in range(larger, n + 1)) / (2.0 ** n) if n else 1.0
    p = min(1.0, 2.0 * tail)
    favourable = negative if favourable_negative else positive
    return {
        "n_nonzero": n, "n_zero": zero,
        "n_favourable": favourable,
        "n_unfavourable": n - favourable,
        "two_sided_exact_sign_p": float(p),
    }


def _bootstrap_median(values: list[float], seed: int = 404, n_boot: int = 10000) -> list[float]:
    if not values:
        return [float("nan"), float("nan")]
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    index = rng.integers(0, len(array), size=(int(n_boot), len(array)))
    medians = np.median(array[index], axis=1)
    return [float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))]


def _cohort(values: list[float], *, favourable_negative: bool) -> dict[str, Any]:
    finite = [float(value) for value in values if np.isfinite(value)]
    if not finite:
        return {"status": "NOT_ESTIMABLE", "n_patients": 0}
    return {
        "status": "COMPLETE_DEVELOPMENT",
        "n_patients": len(finite),
        "patient_median": float(np.median(finite)),
        "patient_bootstrap_median_95": _bootstrap_median(finite),
        **_sign_p(finite, favourable_negative),
        "patient_is_inference_unit": True,
    }


def _cell_geometry(result: dict, lookback: str = "30") -> dict[str, float | int | None]:
    route = [row for row in result["geometry_by_lookback_minutes"][lookback]["heterogeneous_route"]
             if row["status"] == "COMPLETE_DEVELOPMENT"]
    single = [row for row in result["geometry_by_lookback_minutes"][lookback]["single_route"]
              if str(row["status"]).startswith("COMPLETE")]
    route_by_position = {int(row["heldout_position"]): row for row in route}
    single_by_position = {int(row["heldout_position"]): row for row in single}
    paired = sorted(set(route_by_position).intersection(single_by_position))
    output: dict[str, float | int | None] = {
        "geometry_n_route_folds": len(route),
        "geometry_n_single_folds": len(single),
        "geometry_n_paired_folds": len(paired),
    }
    for key in ("route_basin_gating", "route_directed_approach", "abrupt_transition"):
        output[f"geometry_{key}"] = _median([
            row["family_scores"][key] for row in route
        ])
    mappings = {
        "route_basin_gating": "basin_gating",
        "route_directed_approach": "directed_approach",
        "abrupt_transition": "abrupt_transition",
    }
    for route_key, single_key in mappings.items():
        output[f"geometry_route_minus_single_{route_key}"] = _median([
            route_by_position[position]["family_scores"][route_key]
            - single_by_position[position]["family_scores"][single_key]
            for position in paired
        ])
    output["geometry_fraction_two_route_folds"] = (
        float(np.mean([row["n_routes"] == 2 for row in route])) if route else None
    )
    return output


def aggregate(result_root: Path) -> dict:
    queue_path = result_root / "QUEUE_STATUS.json"
    queue = _json(queue_path)
    if queue.get("status") != "PASS_COMPLETE" or queue.get("valid_result_cells") != 46:
        raise ValueError("v0.4 queue is not complete")
    inventory_path = result_root / "manifests/source_cells.json"
    inventory = _json(inventory_path)
    expected_runner = queue["source"]["cell_runner_sha256"]
    expected_module = queue["source"]["module_sha256"]
    cell_results: dict[str, list[dict]] = {}
    cell_rows = []
    lead_cell_rows = []
    geometry_cell_rows = []
    for cell in inventory["cells"]:
        path = (
            result_root / "per_cell" / cell["subject"]
            / f"seed_{int(cell['seed'])}" / "result.json"
        )
        result = _json(path)
        if result["source"]["producer_sha256"] != expected_runner:
            raise ValueError(f"mixed cell-runner hash: {path}")
        if result["source"]["heterogeneous_module_sha256"] != expected_module:
            raise ValueError(f"mixed estimator hash: {path}")
        if result["source"]["state_cache_sha256"] != cell["state_cache_sha256"]:
            raise ValueError(f"source cache drift in cell output: {path}")
        cell_results.setdefault(cell["subject"], []).append(result)
        primary = result["primary"]
        cell_row = {
            "subject": result["subject"], "seed": result["seed"],
            "status": result["status"],
            "state_stratum": result["state_stratum_nonblocking"].get("exploration_stratum"),
            "n_mapped_seizures": result["n_mapped_seizures"],
            "n_supported_seizures_30min": primary.get("n_supported_seizures", 0),
            "n_oof_seizures_30min": primary.get("n_oof_seizures", 0),
            "n_two_route_folds_30min": primary.get("n_two_route_folds", 0),
            "n_wrong_time_estimable_folds_30min": primary.get(
                "n_wrong_time_estimable_folds", 0
            ),
            "initial_training_rule_30min": primary.get("initial_training_rule"),
            **{name: primary.get("equal_seizure_weight_effects", {}).get(name)
               for name in EFFECTS},
        }
        shift_values = [
            row["result"].get("equal_seizure_weight_effects", {}).get(
                "route_state_minus_observation"
            )
            for row in result["coverage_segment_circular_shift_null"]
            if row["result"]["status"] == "COMPLETE_DEVELOPMENT"
        ]
        shift_values = [float(value) for value in shift_values if value is not None]
        observed = cell_row["route_state_minus_observation"]
        cell_row["correct_primary_minus_shift_median"] = (
            float(observed - np.median(shift_values))
            if observed is not None and shift_values else None
        )
        cell_row["fraction_shifts_worse_than_correct"] = (
            float(np.mean(float(observed) < np.asarray(shift_values)))
            if observed is not None and shift_values else None
        )
        cell_rows.append(cell_row)
        for lead in LEADS:
            value = result["by_lead_minutes"][str(lead)]
            lead_cell_rows.append({
                "subject": result["subject"], "seed": result["seed"],
                "lead_minutes": lead, "status": value["status"],
                "n_supported_seizures": value.get("n_supported_seizures", 0),
                "n_oof_seizures": value.get("n_oof_seizures", 0),
                "n_two_route_folds": value.get("n_two_route_folds", 0),
                "n_wrong_time_estimable_folds": value.get(
                    "n_wrong_time_estimable_folds", 0
                ),
                "initial_training_rule": value.get("initial_training_rule"),
                **{name: value.get("equal_seizure_weight_effects", {}).get(name)
                   for name in EFFECTS},
            })
        geometry_cell_rows.append({
            "subject": result["subject"], "seed": result["seed"],
            **_cell_geometry(result),
        })

    r17b = _json(R17B_SUMMARY)
    stable = set(map(str, r17b["stable_state_subjects"]))
    patient_rows = []
    lead_patient_rows = []
    geometry_patient_rows = []
    for subject, results in sorted(cell_results.items()):
        cells = [row for row in cell_rows if row["subject"] == subject]
        primary_complete = [row for row in cells if row["status"] == "COMPLETE_DEVELOPMENT"]
        n_oof = _median([row["n_oof_seizures_30min"] for row in cells]) or 0.0
        n_supported = _median([
            row["n_supported_seizures_30min"] for row in cells
        ]) or 0.0
        patient_rows.append({
            "subject": subject,
            "n_seeds": len(cells),
            "n_complete_primary_seeds": len(primary_complete),
            "status": "COMPLETE_DEVELOPMENT" if primary_complete else "NOT_ESTIMABLE",
            "state_stratum": cells[0]["state_stratum"],
            "preexisting_r1_7b_h1_stable": subject in stable,
            "n_mapped_seizures": int(_median([row["n_mapped_seizures"] for row in cells]) or 0),
            "median_n_supported_seizures_30min": n_supported,
            "median_n_oof_seizures_30min": n_oof,
            "support_tier": (
                "primary_chronological_60_percent_train" if n_supported >= 10
                else "rolling_sensitivity_5_to_9" if n_supported >= 5
                else "descriptive_3_to_4" if n_supported >= 3
                else "not_estimable_lt_3"
            ),
            "median_n_two_route_folds_30min": _median([
                row["n_two_route_folds_30min"] for row in primary_complete
            ]),
            "median_n_wrong_time_estimable_folds_30min": _median([
                row["n_wrong_time_estimable_folds_30min"] for row in primary_complete
            ]),
            **{name: _median([row[name] for row in primary_complete]) for name in EFFECTS},
            "correct_primary_minus_shift_median": _median([
                row["correct_primary_minus_shift_median"] for row in primary_complete
            ]),
            "fraction_shifts_worse_than_correct": _median([
                row["fraction_shifts_worse_than_correct"] for row in primary_complete
            ]),
        })
        for lead in LEADS:
            selected = [row for row in lead_cell_rows
                        if row["subject"] == subject and row["lead_minutes"] == lead
                        and row["status"] == "COMPLETE_DEVELOPMENT"]
            lead_patient_rows.append({
                "subject": subject, "lead_minutes": lead,
                "n_complete_seeds": len(selected),
                "status": "COMPLETE_DEVELOPMENT" if selected else "NOT_ESTIMABLE",
                "state_stratum": cells[0]["state_stratum"],
                "preexisting_r1_7b_h1_stable": subject in stable,
                "median_n_supported_seizures": _median([
                    row["n_supported_seizures"] for row in selected
                ]),
                "median_n_oof_seizures": _median([row["n_oof_seizures"] for row in selected]),
                "median_n_two_route_folds": _median([
                    row["n_two_route_folds"] for row in selected
                ]),
                "median_n_wrong_time_estimable_folds": _median([
                    row["n_wrong_time_estimable_folds"] for row in selected
                ]),
                **{name: _median([row[name] for row in selected]) for name in EFFECTS},
            })
        geometry_cells = [row for row in geometry_cell_rows if row["subject"] == subject]
        numeric = [key for key in geometry_cells[0] if key not in {"subject", "seed"}]
        geometry_patient_rows.append({
            "subject": subject,
            "n_seeds": len(geometry_cells),
            "state_stratum": cells[0]["state_stratum"],
            "preexisting_r1_7b_h1_stable": subject in stable,
            **{key: _median([row[key] for row in geometry_cells]) for key in numeric},
        })

    layers = {
        "all_frozen": lambda row: True,
        "scalar_slow_axis_candidate": lambda row: row["state_stratum"] == "scalar_slow_axis_candidate",
        "collapsed_or_unusable": lambda row: row["state_stratum"] == "collapsed_or_unusable_for_persistent_claim",
        "preexisting_r1_7b_h1_stable": lambda row: row["preexisting_r1_7b_h1_stable"],
    }
    cohort_layers = {}
    for layer, predicate in layers.items():
        selected = [row for row in patient_rows if predicate(row)]
        estimable = [row for row in selected if row["status"] == "COMPLETE_DEVELOPMENT"]
        cohort_layers[layer] = {
            "n_patients_total": len(selected),
            "n_patients_primary_estimable": len(estimable),
            "effects": {
                name: _cohort(
                    [row[name] for row in estimable if row[name] is not None],
                    favourable_negative=True,
                ) for name in EFFECTS
            },
            "correct_primary_minus_shift_median": _cohort(
                [row["correct_primary_minus_shift_median"] for row in estimable
                 if row["correct_primary_minus_shift_median"] is not None],
                favourable_negative=True,
            ),
        }
    geometry_summary = {
        key: _cohort(
            [row[key] for row in geometry_patient_rows if row.get(key) is not None],
            favourable_negative=key.startswith("geometry_route_minus_single"),
        )
        for key in (
            "geometry_route_basin_gating", "geometry_route_directed_approach",
            "geometry_abrupt_transition",
            "geometry_route_minus_single_route_basin_gating",
            "geometry_route_minus_single_route_directed_approach",
            "geometry_route_minus_single_abrupt_transition",
        )
    }
    atomic_csv(result_root / "per_subject/primary_patient_summary.csv", patient_rows)
    atomic_csv(result_root / "per_subject/lead_curve.csv", lead_patient_rows)
    atomic_csv(result_root / "per_subject/geometry_patient_summary.csv", geometry_patient_rows)
    atomic_csv(result_root / "per_cell/cell_summary.csv", cell_rows)
    payload = {
        "status": "COMPLETE_DEVELOPMENT",
        "revision": "h2b_v0_4_patient_first_aggregate_v2_support_tiered",
        "created_utc": utc_now(),
        "n_source_cells": len(inventory["cells"]),
        "n_source_patients": len(cell_results),
        "n_primary_estimable_cells": sum(
            row["status"] == "COMPLETE_DEVELOPMENT" for row in cell_rows
        ),
        "n_primary_estimable_patients": sum(
            row["status"] == "COMPLETE_DEVELOPMENT" for row in patient_rows
        ),
        "n_primary_chronological_patients": sum(
            row["support_tier"] == "primary_chronological_60_percent_train"
            for row in patient_rows
        ),
        "patient_rows": patient_rows,
        "cohort_layers": cohort_layers,
        "geometry_30min": geometry_summary,
        "inference": {
            "seeds_aggregated_by_patient_median": True,
            "heldout_seizures_equal_weight_inside_cell": True,
            "patients_are_cohort_unit": True,
            "controls_and_grid_rows_are_not_replicates": True,
            "patient_bootstrap_replicates": 10000,
            "sign_test_is_directional_supplement_only": True,
        },
        "source": {
            "queue_status": str(queue_path),
            "queue_status_sha256": sha256_file(queue_path),
            "source_inventory": str(inventory_path),
            "source_inventory_sha256": sha256_file(inventory_path),
            "r1_7b_summary": str(R17B_SUMMARY),
            "r1_7b_summary_sha256": sha256_file(R17B_SUMMARY),
            "producer_sha256": sha256_file(PRODUCER),
        },
        "development_only": True,
        "formal_test_partition_opened": False,
        "sealed_opened": False,
        "h3_or_t2_run": False,
    }
    atomic_json(result_root / "reports/cohort_summary.json", payload)
    atomic_json(result_root / "manifests/exclusion_funnel.json", {
        "revision": "h2b_v0_4_exclusion_funnel_v1",
        "audited_full_grid_cells": len(inventory["cells"]),
        "audited_full_grid_patients": len(cell_results),
        "primary_estimable_cells": payload["n_primary_estimable_cells"],
        "primary_not_estimable_cells": len(cell_rows) - payload["n_primary_estimable_cells"],
        "primary_estimable_patients": payload["n_primary_estimable_patients"],
        "primary_not_estimable_patients": len(patient_rows) - payload["n_primary_estimable_patients"],
        "primary_chronological_patients": payload[
            "n_primary_chronological_patients"
        ],
        "no_patient_removed_for_state_stratum": True,
        "no_patient_removed_for_h2b_effect_direction": True,
    })
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=V0_4_RESULT_ROOT)
    args = parser.parse_args()
    result = aggregate(args.result_root.resolve())
    print(result["status"], result["n_primary_estimable_patients"])


if __name__ == "__main__":
    main()
