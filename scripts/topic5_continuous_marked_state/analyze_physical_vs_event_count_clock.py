#!/usr/bin/env python3
"""Patient-paired physical-time versus event-count exposure comparison."""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.stats import binomtest

from src.topic5_continuous_marked_state import contract
from src.topic5_continuous_marked_state.multiplicity import annotate_family
from src.topic5_continuous_marked_state.exposure import EXPOSURE_REVISION


TAUS = (3.0, 5.0, 10.0, 20.0, 30.0, 60.0)
CURRENT_EVENT_TAU = 1e-6
ENDPOINTS = (
    "joint_nll", "timing_nll", "mark_nll", "participation_nll",
    "rank_nll", "stop_nll",
)


def _load_folder(folder: str, expected_clock: str) -> dict[tuple[str, str, float], dict]:
    found = {}
    for path in sorted((contract.RESULT_ROOT / folder).glob("*__tau*m.json")):
        row = json.loads(path.read_text())
        tau = float(row.get("tau_minutes", np.nan))
        if tau not in TAUS:
            continue
        if not (
            row.get("contract") == contract.REVISION
            and row.get("fit_revision") == contract.FIT_REVISION
            and row.get("exposure_revision") == EXPOSURE_REVISION
            and row.get("decay_clock", "physical_time") == expected_clock
        ):
            continue
        key = (row["subject"], row.get("exposure_kind", "load"), tau)
        if key in found:
            raise ValueError(f"duplicate clock-control cell {key}")
        found[key] = row
    return found


def _load_current_event_limit() -> dict[tuple[str, str], dict]:
    found = {}
    for path in sorted((contract.RESULT_ROOT / "exposure_screen").glob("*__tau*m.json")):
        row = json.loads(path.read_text())
        if float(row.get("tau_minutes", np.nan)) != CURRENT_EVENT_TAU:
            continue
        if not (
            row.get("contract") == contract.REVISION
            and row.get("fit_revision") == contract.FIT_REVISION
            and row.get("exposure_revision") == EXPOSURE_REVISION
        ):
            continue
        key = (row["subject"], row.get("exposure_kind", "load"))
        if key in found:
            raise ValueError(f"duplicate current-event cell {key}")
        found[key] = row
    return found


def _summary(values: np.ndarray, *, left_arm: str, right_arm: str) -> dict:
    """Paired summary of ``left_arm - right_arm``, labelled by the actual arms.

    The previous version hard-coded ``median_physical_minus_event_count`` /
    ``n_physical_better`` / ``n_event_count_better`` and was then applied to the
    ``event_count - current_event`` contrasts as well. 144 summary blocks in the
    shipped package therefore asserted "n_physical_better: 27" under a contrast
    that says nothing about the physical clock, and the machine evidence card
    copied them verbatim -- so a programmatic reader got the exact opposite of
    the paper's conclusion, which is that the physical clock did NOT win. Keys
    are now derived from the arms actually being differenced, and neutral
    ``median_delta`` / ``n_negative`` / ``n_positive`` are always present.
    """
    nonzero = values[values != 0]
    leave_one = np.asarray([
        np.median(np.delete(values, index)) for index in range(len(values))
    ])
    n_left_better = int(np.sum(values < 0))     # lower NLL for the left arm
    n_right_better = int(np.sum(values > 0))
    return {
        "difference": f"{left_arm}_minus_{right_arm}",
        "lower_is_better_for": left_arm,
        "median_delta": float(np.median(values)),
        "iqr": [float(np.percentile(values, 25)), float(np.percentile(values, 75))],
        "n_negative": n_left_better,
        "n_positive": n_right_better,
        f"n_{left_arm}_better": n_left_better,
        f"n_{right_arm}_better": n_right_better,
        "n_patients": int(len(values)),
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
    physical = _load_folder("exposure_screen", "physical_time")
    event_count = _load_folder("exposure_clock_control", "event_count")
    current_event = _load_current_event_limit()
    subjects = tuple(json.loads(contract.SPLIT_MANIFEST.read_text())["subjects"])
    expected = {
        (subject, kind, tau) for subject in subjects
        for kind in ("load", "participation") for tau in TAUS
    }
    if set(physical) != expected:
        raise RuntimeError(f"physical clock grid mismatch: {len(expected - set(physical))} missing")
    if set(event_count) != expected:
        raise RuntimeError(f"event-count clock grid mismatch: {len(expected - set(event_count))} missing")
    expected_current = {
        (subject, kind) for subject in subjects
        for kind in ("load", "participation")
    }
    if set(current_event) != expected_current:
        raise RuntimeError(
            f"current-event grid mismatch: {len(expected_current - set(current_event))} missing"
        )

    pairing_audit = []
    maximum_current_history_difference = 0.0
    for key in sorted(expected):
        p = physical[key]
        c = event_count[key]
        current = current_event[(key[0], key[1])]
        if p["n_train"] != c["n_train"] or p["n_validation"] != c["n_validation"]:
            raise RuntimeError(f"clock-control sample mismatch for {key}")
        if (p["n_train"] != current["n_train"]
                or p["n_validation"] != current["n_validation"]):
            raise RuntimeError(f"current-event sample mismatch for {key}")
        if p.get("sealed_opened", False) or c.get("sealed_opened", False):
            raise RuntimeError(f"sealed partition was opened for {key}")
        if current.get("sealed_opened", False):
            raise RuntimeError(f"current-event sealed partition was opened for {key}")
        maximum_history_difference = max(
            abs(float(p["fits"]["history"]["validation"][endpoint])
                - float(c["fits"]["history"]["validation"][endpoint]))
            for endpoint in ENDPOINTS
        )
        if maximum_history_difference > 1e-10:
            raise RuntimeError(
                f"clock-control history baseline mismatch for {key}: "
                f"{maximum_history_difference:g}"
            )
        current_history_difference = max(
            abs(float(p["fits"]["history"]["validation"][endpoint])
                - float(current["fits"]["history"]["validation"][endpoint]))
            for endpoint in ENDPOINTS
        )
        if current_history_difference > 1e-10:
            raise RuntimeError(
                f"current-event history baseline mismatch for {key}: "
                f"{current_history_difference:g}"
            )
        maximum_current_history_difference = max(
            maximum_current_history_difference, current_history_difference
        )
        pairing_audit.append({
            "subject": key[0],
            "exposure_kind": key[1],
            "tau_minutes": key[2],
            "n_train": int(p["n_train"]),
            "n_validation": int(p["n_validation"]),
            "maximum_history_endpoint_difference": maximum_history_difference,
        })

    cells = []
    for kind in ("load", "participation"):
        for tau in TAUS:
            endpoint_rows = {}
            for endpoint in ENDPOINTS:
                versus_placebo = []
                versus_history = []
                count_versus_current_placebo = []
                count_versus_current_history = []
                n_physical_better_and_valid = 0
                patient_values = {}
                for subject in subjects:
                    p = physical[(subject, kind, tau)]["contrasts"][endpoint]
                    c = event_count[(subject, kind, tau)]["contrasts"][endpoint]
                    current = current_event[(subject, kind)]["contrasts"][endpoint]
                    dp = float(p["real_minus_placebo"] - c["real_minus_placebo"])
                    dh = float(p["real_minus_history"] - c["real_minus_history"])
                    dcp = float(
                        c["real_minus_placebo"] - current["real_minus_placebo"]
                    )
                    dch = float(
                        c["real_minus_history"] - current["real_minus_history"]
                    )
                    versus_placebo.append(dp)
                    versus_history.append(dh)
                    count_versus_current_placebo.append(dcp)
                    count_versus_current_history.append(dch)
                    if (dp < 0 and dh < 0 and p["real_minus_placebo"] < 0
                            and p["real_minus_history"] < 0):
                        n_physical_better_and_valid += 1
                    patient_values[subject] = {
                        "physical_minus_count_delta_vs_placebo": dp,
                        "physical_minus_count_delta_vs_history": dh,
                        "physical_real_minus_placebo": float(p["real_minus_placebo"]),
                        "count_real_minus_placebo": float(c["real_minus_placebo"]),
                        "count_minus_current_delta_vs_placebo": dcp,
                        "count_minus_current_delta_vs_history": dch,
                    }
                endpoint_rows[endpoint] = {
                    "delta_vs_placebo": _summary(
                        np.asarray(versus_placebo),
                        left_arm="physical_time", right_arm="event_count"),
                    # the history baseline is bit-identical across arms (checked
                    # to 1e-10 above), so this contrast is the clean one; the
                    # placebo contrast subtracts two controls whose delay,
                    # max(30 min, 3*tau), differs once tau exceeds 10 min
                    "delta_vs_history": _summary(
                        np.asarray(versus_history),
                        left_arm="physical_time", right_arm="event_count"),
                    "event_count_minus_current_event_delta_vs_placebo": _summary(
                        np.asarray(count_versus_current_placebo),
                        left_arm="event_count", right_arm="current_event"),
                    "event_count_minus_current_event_delta_vs_history": _summary(
                        np.asarray(count_versus_current_history),
                        left_arm="event_count", right_arm="current_event"),
                    "n_physical_better_count_and_both_controls": int(
                        n_physical_better_and_valid
                    ),
                    "dataset_strata": {
                        dataset: {
                            "physical_minus_count_delta_vs_placebo": _summary(
                                np.asarray([
                                    patient_values[subject][
                                        "physical_minus_count_delta_vs_placebo"
                                    ]
                                    for subject in subjects
                                    if subject.startswith(dataset + "_")
                                ]),
                                left_arm="physical_time", right_arm="event_count",
                            ),
                            "count_minus_current_delta_vs_placebo": _summary(
                                np.asarray([
                                    patient_values[subject][
                                        "count_minus_current_delta_vs_placebo"
                                    ]
                                    for subject in subjects
                                    if subject.startswith(dataset + "_")
                                ]),
                                left_arm="event_count", right_arm="current_event",
                            ),
                        }
                        for dataset in ("epilepsiae", "yuquan")
                    },
                    "patient_values": patient_values,
                }
            count_steps = np.asarray([
                event_count[(subject, kind, tau)][
                    "event_count_step_minutes_train_median"
                ] for subject in subjects
            ], dtype=float)
            cells.append({
                "exposure_kind": kind,
                "tau_minutes": tau,
                "n_patients": len(subjects),
                "event_count_step_minutes_train_median_across_patients": float(
                    np.median(count_steps)
                ),
                "nominal_memory_events_median_across_patients": float(
                    np.median(tau / count_steps)
                ),
                "endpoints": endpoint_rows,
            })

    # Multiplicity. This file writes one exact sign test per
    # (exposure kind x tau x endpoint x contrast) plus the dataset strata:
    # 288 primary tests in the grid alone. Shipping only the raw value let a
    # single cell (p = 8.2e-4, one of eleven ties, rank 4 of 288) be lifted into
    # a machine evidence card as a headline. Holm puts it at 0.234, BH at
    # q = 0.017. Both are attached here so neither reading is available without
    # the other. Families are separated because the dataset strata are a
    # sensitivity layer, not part of the primary grid.
    primary_family = []
    strata_family = []
    for cell in cells:
        for endpoint, row in cell["endpoints"].items():
            for name in ("delta_vs_placebo", "delta_vs_history",
                         "event_count_minus_current_event_delta_vs_placebo",
                         "event_count_minus_current_event_delta_vs_history"):
                primary_family.append((
                    (cell["exposure_kind"], str(cell["tau_minutes"]), endpoint, name),
                    row[name],
                ))
            for dataset, strata in row["dataset_strata"].items():
                for name, summary in strata.items():
                    strata_family.append((
                        (cell["exposure_kind"], str(cell["tau_minutes"]),
                         endpoint, dataset, name),
                        summary,
                    ))
    multiplicity_index = {
        "primary_grid": annotate_family(
            primary_family, family_name="clock_primary_grid"),
        "dataset_strata": annotate_family(
            strata_family, family_name="clock_dataset_strata"),
        "scope_note": (
            "Adjustment is within this file. It does not span the fixed "
            "event-count grid or the strata analysis, which declare their own "
            "families."
        ),
    }

    output = {
        "contract": contract.REVISION,
        "fit_revision": contract.FIT_REVISION,
        "exposure_revision": EXPOSURE_REVISION,
        "analysis_revision": "physical_time_vs_train_median_iei_event_clock_v1",
        "taus_minutes": list(TAUS),
        "n_patients": len(subjects),
        "n_physical_source_runs": len(physical),
        "n_event_count_source_runs": len(event_count),
        "n_current_event_source_runs": len(current_event),
        "current_event_limit_tau_minutes": CURRENT_EVENT_TAU,
        "pairing_audit": {
            "n_cells": len(pairing_audit),
            "maximum_history_endpoint_difference": float(max(
                row["maximum_history_endpoint_difference"]
                for row in pairing_audit
            )),
            "maximum_current_event_history_endpoint_difference": float(
                maximum_current_history_difference
            ),
            "all_sample_counts_exact": True,
            "all_history_baselines_exact": True,
            "all_current_event_sample_counts_exact": True,
            "all_current_event_history_baselines_exact": True,
            "all_sealed_partitions_closed": True,
            "rows": pairing_audit,
        },
        "cells": cells,
        "multiplicity": multiplicity_index,
        "sealed_opened": False,
        "interpretation": (
            "Negative physical-minus-event-count delta means actual interval timing "
            "adds predictive value beyond a fixed per-event decay with matched typical "
            "memory. No physical-time advantage means the nominal minute band should "
            "be reported as event-count memory rather than an identified biological clock. "
            "Negative event-count-minus-current-event delta tests whether multiple prior "
            "events add information beyond the single-current-IED limit."
        ),
    }
    path = contract.RESULT_ROOT / "exposure_clock_control/PHYSICAL_VS_EVENT_COUNT_CLOCK.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(output, indent=2, sort_keys=True))
    os.replace(tmp, path)
    print(json.dumps({"path": str(path), "n_cells": len(cells)}))


if __name__ == "__main__":
    main()
