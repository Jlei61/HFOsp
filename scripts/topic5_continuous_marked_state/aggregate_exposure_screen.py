#!/usr/bin/env python3
"""Patient-level aggregation of H3-S0 exposure screens."""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest

from src.topic5_continuous_marked_state import contract
from src.topic5_continuous_marked_state.exposure import EXPOSURE_REVISION


CORE_TAUS = (1.0, 10.0, 60.0, 360.0)
FAST_CONTROL_TAUS = (1e-6, 1.0 / 60.0, 0.1)


def main() -> None:
    rows = []
    for path in sorted((contract.RESULT_ROOT / "exposure_screen").glob("*__tau*m.json")):
        row = json.loads(path.read_text())
        if (row.get("contract") == contract.REVISION
                and row.get("fit_revision") == contract.FIT_REVISION
                and row.get("exposure_revision") == EXPOSURE_REVISION):
            rows.append(row)
    by_tau: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for row in rows:
        by_tau[(row.get("exposure_kind", "load"), float(row["tau_minutes"]))].append(row)
    summary = []
    for kind, tau in sorted(by_tau):
        found = by_tau[(kind, tau)]
        endpoints = {}
        for endpoint in (
            "joint_nll", "timing_nll", "mark_nll",
            "participation_nll", "rank_nll", "stop_nll",
        ):
            delta = np.asarray([
                row["contrasts"][endpoint]["real_minus_placebo"] for row in found
            ], dtype=float)
            versus_history = np.asarray([
                row["contrasts"][endpoint]["real_minus_history"] for row in found
            ], dtype=float)
            placebo_versus_history = np.asarray([
                row["contrasts"][endpoint]["placebo_minus_history"] for row in found
            ], dtype=float)
            nonzero_placebo = delta[delta != 0]
            nonzero_history = versus_history[versus_history != 0]
            endpoints[endpoint] = {
                "median_real_minus_placebo": float(np.median(delta)),
                "iqr_real_minus_placebo": [
                    float(np.percentile(delta, 25)),
                    float(np.percentile(delta, 75)),
                ],
                "n_real_better_placebo": int(np.sum(delta < 0)),
                "n_nonzero_placebo": int(len(nonzero_placebo)),
                "two_sided_exact_sign_p_vs_placebo_unadjusted": (
                    float(binomtest(
                        int(np.sum(nonzero_placebo < 0)),
                        len(nonzero_placebo), 0.5,
                    ).pvalue) if len(nonzero_placebo) else None
                ),
                "median_real_minus_history": float(np.median(versus_history)),
                "n_real_better_history": int(np.sum(versus_history < 0)),
                "n_real_better_both_history_and_placebo": int(np.sum(
                    (delta < 0) & (versus_history < 0)
                )),
                "n_nonzero_history": int(len(nonzero_history)),
                "two_sided_exact_sign_p_vs_history_unadjusted": (
                    float(binomtest(
                        int(np.sum(nonzero_history < 0)),
                        len(nonzero_history), 0.5,
                    ).pvalue) if len(nonzero_history) else None
                ),
                "median_placebo_minus_history": float(np.median(placebo_versus_history)),
                "n_placebo_better_history": int(np.sum(placebo_versus_history < 0)),
                "three_way_descriptive_class": (
                    "real_better_both_in_median"
                    if np.median(delta) < 0 and np.median(versus_history) < 0
                    else "real_separates_from_placebo_but_not_history"
                    if np.median(delta) < 0
                    else "no_real_advantage_over_placebo"
                ),
                "dataset_descriptives_real_minus_placebo": {
                    dataset: {
                        "n_patients": int(len(values)),
                        "median": float(np.median(values)),
                        "n_real_better": int(np.sum(np.asarray(values) < 0)),
                    }
                    for dataset in ("epilepsiae", "yuquan")
                    for values in [[
                        float(row["contrasts"][endpoint]["real_minus_placebo"])
                        for row in found if row["subject"].startswith(dataset + "_")
                    ]]
                    if values
                },
                "dataset_descriptives_real_minus_history": {
                    dataset: {
                        "n_patients": int(len(values)),
                        "median": float(np.median(values)),
                        "n_real_better": int(np.sum(np.asarray(values) < 0)),
                    }
                    for dataset in ("epilepsiae", "yuquan")
                    for values in [[
                        float(row["contrasts"][endpoint]["real_minus_history"])
                        for row in found if row["subject"].startswith(dataset + "_")
                    ]]
                    if values
                },
                "patient_deltas_real_minus_placebo": {
                    row["subject"]: float(row["contrasts"][endpoint]["real_minus_placebo"])
                    for row in found
                },
            }
        summary.append({
            "exposure_kind": kind, "tau_minutes": tau,
            "analysis_tier": (
                "frozen_core_grid" if tau in CORE_TAUS
                else "posthoc_near_single_event_fast_control"
                if any(np.isclose(tau, value) for value in FAST_CONTROL_TAUS)
                else "posthoc_time_scale_refinement"
            ),
            "n_patients": len(found), "endpoints": endpoints,
        })
    core_rows = [
        row for row in rows if float(row["tau_minutes"]) in CORE_TAUS
    ]
    refinement_rows = [
        row for row in rows
        if float(row["tau_minutes"]) not in CORE_TAUS
        and not any(np.isclose(float(row["tau_minutes"]), value)
                    for value in FAST_CONTROL_TAUS)
    ]
    fast_control_rows = [
        row for row in rows
        if any(np.isclose(float(row["tau_minutes"]), value)
               for value in FAST_CONTROL_TAUS)
    ]
    output = {
        "contract": contract.REVISION,
        "fit_revision": contract.FIT_REVISION,
        "exposure_revision": EXPOSURE_REVISION,
        "n_runs": len(rows),
        "expected_n_core_runs": 34 * 4 * 2,
        "n_core_runs": len(core_rows),
        "core_grid_complete": len(core_rows) == 34 * 4 * 2,
        "n_posthoc_refinement_runs": len(refinement_rows),
        "n_fast_control_runs": len(fast_control_rows),
        "by_tau": summary,
        "sealed_opened": False,
        "claim_boundary": (
            "Patient-level predictive distributed-exposure screen. Negative is "
            "not proof against a nonlinear persistent generator; positive is "
            "not causal shaping. Exact sign p-values are descriptive, raw, "
            "and not used as a gate across the exploratory time-scale grid."
        ),
    }
    path = contract.RESULT_ROOT / "exposure_screen/EXPOSURE_SCREEN_SUMMARY.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(output, indent=2, sort_keys=True))
    os.replace(tmp, path)
    print(json.dumps({"n_runs": len(rows), "n_kind_time_scale_cells": len(summary)}))


if __name__ == "__main__":
    main()
