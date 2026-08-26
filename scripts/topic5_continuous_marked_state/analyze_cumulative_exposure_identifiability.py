#!/usr/bin/env python3
"""Separate current-event innovation from distributed IED exposure.

This is a patient-paired descriptive analysis of the H3-S0 screen.  The
near-zero tau cell is the current-event limit: because the explicit history
already contains the current event load/mark, it should not buy information
merely by re-encoding that event.  Negative delta-of-delta means a distributed
time scale improves more than this current-event limit.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest

from src.topic5_continuous_marked_state import contract
from src.topic5_continuous_marked_state.exposure import EXPOSURE_REVISION


CURRENT_TAU = 1e-6
BURST_TAUS = (1.0 / 60.0, 0.1)
MINUTE_TAUS = (1.0, 3.0, 5.0, 10.0, 20.0, 30.0, 60.0, 360.0)
ENDPOINTS = (
    "joint_nll", "timing_nll", "mark_nll", "participation_nll",
    "rank_nll", "stop_nll",
)


def _same_tau(left: float, right: float) -> bool:
    return bool(np.isclose(left, right, rtol=1e-8, atol=1e-10))


def _paired_summary(values: np.ndarray) -> dict:
    nonzero = values[values != 0]
    leave_one = np.asarray([
        np.median(np.delete(values, index)) for index in range(len(values))
    ]) if len(values) > 1 else np.asarray([values[0]])
    return {
        "median": float(np.median(values)),
        "iqr": [float(np.percentile(values, 25)), float(np.percentile(values, 75))],
        "n_negative": int(np.sum(values < 0)),
        "n_positive": int(np.sum(values > 0)),
        "n_nonzero": int(len(nonzero)),
        "two_sided_exact_sign_p_unadjusted": (
            float(binomtest(int(np.sum(nonzero < 0)), len(nonzero), 0.5).pvalue)
            if len(nonzero) else None
        ),
        "leave_one_patient_median_range": [
            float(np.min(leave_one)), float(np.max(leave_one)),
        ],
    }


def main() -> None:
    root = contract.RESULT_ROOT / "exposure_screen"
    expected_taus = (CURRENT_TAU, *BURST_TAUS, *MINUTE_TAUS)
    indexed: dict[tuple[str, str, float], dict] = {}
    for path in sorted(root.glob("*__tau*m.json")):
        row = json.loads(path.read_text())
        if not (
            row.get("contract") == contract.REVISION
            and row.get("fit_revision") == contract.FIT_REVISION
            and row.get("exposure_revision") == EXPOSURE_REVISION
        ):
            continue
        tau = float(row["tau_minutes"])
        matched = next((value for value in expected_taus if _same_tau(tau, value)), None)
        if matched is None:
            continue
        key = (row["subject"], row.get("exposure_kind", "load"), float(matched))
        if key in indexed:
            raise ValueError(f"duplicate exposure cell {key}")
        indexed[key] = row

    subjects = tuple(json.loads(contract.SPLIT_MANIFEST.read_text())["subjects"])
    missing = [
        (subject, kind, tau)
        for subject in subjects
        for kind in ("load", "participation")
        for tau in expected_taus
        if (subject, kind, float(tau)) not in indexed
    ]
    if missing:
        raise RuntimeError(f"incomplete exposure grid: {len(missing)} missing; first={missing[:3]}")

    cells = []
    for kind in ("load", "participation"):
        for tau in (*BURST_TAUS, *MINUTE_TAUS):
            endpoint_rows = {}
            for endpoint in ENDPOINTS:
                versus_placebo = []
                versus_history = []
                patient_values = {}
                dataset_values: dict[str, list[float]] = defaultdict(list)
                n_scale_better_current_and_both = 0
                for subject in subjects:
                    current = indexed[(subject, kind, CURRENT_TAU)]["contrasts"][endpoint]
                    scale = indexed[(subject, kind, float(tau))]["contrasts"][endpoint]
                    dp = float(scale["real_minus_placebo"] - current["real_minus_placebo"])
                    dh = float(scale["real_minus_history"] - current["real_minus_history"])
                    versus_placebo.append(dp)
                    versus_history.append(dh)
                    patient_values[subject] = {
                        "distributed_minus_current_delta_vs_placebo": dp,
                        "distributed_minus_current_delta_vs_history": dh,
                        "distributed_real_minus_placebo": float(scale["real_minus_placebo"]),
                        "distributed_real_minus_history": float(scale["real_minus_history"]),
                        "current_real_minus_placebo": float(current["real_minus_placebo"]),
                        "current_real_minus_history": float(current["real_minus_history"]),
                    }
                    dataset_values[subject.split("_", 1)[0]].append(dp)
                    if (dp < 0 and dh < 0 and scale["real_minus_placebo"] < 0
                            and scale["real_minus_history"] < 0):
                        n_scale_better_current_and_both += 1
                endpoint_rows[endpoint] = {
                    "distributed_minus_current_delta_vs_placebo": _paired_summary(
                        np.asarray(versus_placebo, dtype=float)
                    ),
                    "distributed_minus_current_delta_vs_history": _paired_summary(
                        np.asarray(versus_history, dtype=float)
                    ),
                    "n_distributed_better_current_and_both_controls": int(
                        n_scale_better_current_and_both
                    ),
                    "dataset_median_delta_of_delta_vs_placebo": {
                        dataset: {
                            "n_patients": len(values),
                            "median": float(np.median(values)),
                            "n_negative": int(np.sum(np.asarray(values) < 0)),
                        }
                        for dataset, values in sorted(dataset_values.items())
                    },
                    "patient_values": patient_values,
                }
            cells.append({
                "exposure_kind": kind,
                "tau_minutes": float(tau),
                "scale_class": "seconds_burst_control" if tau in BURST_TAUS else "distributed_minutes",
                "n_patients": len(subjects),
                "endpoints": endpoint_rows,
            })

    output = {
        "contract": contract.REVISION,
        "fit_revision": contract.FIT_REVISION,
        "exposure_revision": EXPOSURE_REVISION,
        "analysis_revision": "current_event_vs_cumulative_patient_paired_v1",
        "current_event_limit_tau_minutes": CURRENT_TAU,
        "seconds_controls_tau_minutes": list(BURST_TAUS),
        "distributed_tau_minutes": list(MINUTE_TAUS),
        "n_source_runs": len(indexed),
        "n_patients": len(subjects),
        "cells": cells,
        "sealed_opened": False,
        "interpretation": (
            "Negative delta-of-delta means the distributed exposure has more "
            "predictive increment than the current-event limit in the same patient. "
            "This screen is descriptive and does not establish a generator edge. "
            "Minutes may be called cumulative only when they improve beyond the "
            "current-event limit, not merely beyond a delayed placebo."
        ),
    }
    path = root / "CUMULATIVE_EXPOSURE_IDENTIFIABILITY.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(output, indent=2, sort_keys=True))
    os.replace(tmp, path)
    print(json.dumps({"path": str(path), "n_source_runs": len(indexed), "n_cells": len(cells)}))


if __name__ == "__main__":
    main()
