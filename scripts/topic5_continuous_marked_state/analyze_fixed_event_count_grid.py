#!/usr/bin/env python3
"""Aggregate the fixed event-count exposure grid at patient level."""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
from scipy.stats import binomtest, spearmanr

from src.topic5_continuous_marked_state import contract
from src.topic5_continuous_marked_state.multiplicity import annotate_family
from src.topic5_continuous_marked_state.exposure import EXPOSURE_REVISION


MEMORIES = (25.0, 50.0, 100.0, 200.0, 400.0)
KINDS = ("load", "participation")
ENDPOINTS = (
    "joint_nll", "timing_nll", "mark_nll", "participation_nll",
    "rank_nll", "stop_nll",
)
CURRENT_TAU = 1e-6
PRODUCER_FILES = (
    "src/topic5_continuous_marked_state/contract.py",
    "src/topic5_continuous_marked_state/bridge.py",
    "src/topic5_continuous_marked_state/exposure.py",
    "scripts/topic5_continuous_marked_state/run_fixed_event_count_screen.py",
    "scripts/topic5_continuous_marked_state/run_fixed_event_count_grid.py",
)
SUPERSEDED_COUNT_FOLDER = (
    "_superseded_preproducer_lock_20260824_061418_event_count_grid"
)


def _producer_audit() -> dict:
    paths = [contract.REPO_ROOT / token for token in PRODUCER_FILES]
    digest = hashlib.sha256()
    for token, path in zip(PRODUCER_FILES, paths):
        digest.update(token.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    outputs = []
    for folder in ("exposure_event_count_grid", "exposure_fixed_memory_physical"):
        outputs.extend((contract.RESULT_ROOT / folder).glob("*__N*events.json"))
    if len(outputs) != 680:
        raise RuntimeError(f"producer audit expected 680 cells, found {len(outputs)}")
    latest_source_mtime = max(path.stat().st_mtime for path in paths)
    earliest_output_mtime = min(path.stat().st_mtime for path in outputs)
    if earliest_output_mtime < latest_source_mtime:
        raise RuntimeError(
            "fixed-grid package mix: at least one cell predates active producer "
            f"({earliest_output_mtime} < {latest_source_mtime})"
        )
    return {
        "producer_source_sha256": digest.hexdigest(),
        "producer_files": list(PRODUCER_FILES),
        "latest_producer_mtime": float(latest_source_mtime),
        "earliest_cell_mtime": float(earliest_output_mtime),
        "all_cells_postdate_active_producer": True,
    }


def _superseded_rerun_parity() -> dict:
    old_folder = contract.RESULT_ROOT / SUPERSEDED_COUNT_FOLDER
    new_folder = contract.RESULT_ROOT / "exposure_event_count_grid"
    old_paths = sorted(old_folder.glob("*__N*events.json"))
    mismatches = []
    for old_path in old_paths:
        new_path = new_folder / old_path.name
        if not new_path.exists() or json.loads(old_path.read_text()) != json.loads(
            new_path.read_text()
        ):
            mismatches.append(old_path.name)
    if len(old_paths) != 219 or mismatches:
        raise RuntimeError(
            f"superseded/current rerun parity failed: {len(old_paths)} old, "
            f"{len(mismatches)} mismatches"
        )
    return {
        "superseded_folder": SUPERSEDED_COUNT_FOLDER,
        "n_archived_cells": len(old_paths),
        "n_exact_reruns": len(old_paths),
        "all_json_fields_exact": True,
    }


def _summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    nonzero = array[array != 0]
    leave_one = np.asarray([
        np.median(np.delete(array, index)) for index in range(len(array))
    ])
    return {
        "median": float(np.median(array)),
        "iqr": [float(np.percentile(array, 25)), float(np.percentile(array, 75))],
        "n_negative": int(np.sum(array < 0)),
        "n_positive": int(np.sum(array > 0)),
        "n_patients": int(len(array)),
        "two_sided_exact_sign_p_unadjusted": (
            float(binomtest(int(np.sum(nonzero < 0)), len(nonzero), 0.5).pvalue)
            if len(nonzero) else None
        ),
        "leave_one_patient_median_range": [
            float(np.min(leave_one)), float(np.max(leave_one)),
        ],
    }


def _load_grid(folder_name: str, expected_clock: str,
               expected_parameterisation: str) -> dict[tuple[str, str, float], dict]:
    rows = {}
    folder = contract.RESULT_ROOT / folder_name
    for path in sorted(folder.glob("*__N*events.json")):
        row = json.loads(path.read_text())
        key = (
            row["subject"], row["exposure_kind"],
            float(row["event_count_memory_events"]),
        )
        if key[2] not in MEMORIES:
            continue
        if not (
            row.get("contract") == contract.REVISION
            and row.get("fit_revision") == contract.FIT_REVISION
            and row.get("exposure_revision") == EXPOSURE_REVISION
            and row.get("decay_clock") == expected_clock
            and row.get("clock_parameterisation")
            == expected_parameterisation
        ):
            continue
        if key in rows:
            raise RuntimeError(f"duplicate fixed-count cell {key}")
        rows[key] = row
    return rows


def _load_current() -> dict[tuple[str, str], dict]:
    rows = {}
    for path in sorted((contract.RESULT_ROOT / "exposure_screen").glob("*__tau*m.json")):
        row = json.loads(path.read_text())
        if float(row.get("tau_minutes", np.nan)) != CURRENT_TAU:
            continue
        key = (row["subject"], row.get("exposure_kind", "load"))
        if key in rows:
            raise RuntimeError(f"duplicate current-event cell {key}")
        rows[key] = row
    return rows


def main() -> None:
    producer_audit = _producer_audit()
    superseded_parity = _superseded_rerun_parity()
    subjects = tuple(json.loads(contract.SPLIT_MANIFEST.read_text())["subjects"])
    grid = _load_grid(
        "exposure_event_count_grid", "event_count",
        "fixed_event_count_across_patients",
    )
    physical = _load_grid(
        "exposure_fixed_memory_physical", "physical_time",
        "rate_matched_physical_time_for_fixed_event_count",
    )
    current = _load_current()
    expected = {(subject, kind, memory) for subject in subjects
                for kind in KINDS for memory in MEMORIES}
    if set(grid) != expected:
        raise RuntimeError(f"fixed-count grid incomplete: {len(expected - set(grid))} missing")
    if set(physical) != expected:
        raise RuntimeError(
            f"rate-matched physical grid incomplete: {len(expected - set(physical))} missing"
        )
    if set(current) != {(subject, kind) for subject in subjects for kind in KINDS}:
        raise RuntimeError("current-event grid incomplete")

    maximum_history_difference = 0.0
    for subject, kind, memory in sorted(expected):
        row = grid[(subject, kind, memory)]
        physical_row = physical[(subject, kind, memory)]
        now = current[(subject, kind)]
        if (row["n_train"] != now["n_train"]
                or row["n_validation"] != now["n_validation"]
                or physical_row["n_train"] != now["n_train"]
                or physical_row["n_validation"] != now["n_validation"]):
            raise RuntimeError(
                f"fixed-count/current-event sample mismatch {(subject, kind, memory)}"
            )
        if (row.get("sealed_opened", False)
                or physical_row.get("sealed_opened", False)
                or now.get("sealed_opened", False)):
            raise RuntimeError(
                f"sealed partition opened {(subject, kind, memory)}"
            )
        difference = max(
            max(
                abs(float(candidate["fits"]["history"]["validation"][endpoint])
                    - float(now["fits"]["history"]["validation"][endpoint]))
                for candidate in (row, physical_row)
            )
            for endpoint in ENDPOINTS
        )
        if difference > 1e-10:
            raise RuntimeError(
                f"fixed-count/current history mismatch {(subject, kind, memory)}: "
                f"{difference:g}"
            )
        maximum_history_difference = max(maximum_history_difference, difference)

    cells = []
    for kind in KINDS:
        for memory in MEMORIES:
            endpoints = {}
            for endpoint in ENDPOINTS:
                real_placebo = []
                real_history = []
                versus_current = []
                physical_versus_placebo = []
                physical_versus_history = []
                physical_versus_count_placebo = []
                physical_versus_count_history = []
                patient_values = {}
                for subject in subjects:
                    row = grid[(subject, kind, memory)]["contrasts"][endpoint]
                    physical_row = physical[
                        (subject, kind, memory)
                    ]["contrasts"][endpoint]
                    now = current[(subject, kind)]["contrasts"][endpoint]
                    rp = float(row["real_minus_placebo"])
                    rh = float(row["real_minus_history"])
                    vc = float(rp - now["real_minus_placebo"])
                    pp = float(physical_row["real_minus_placebo"])
                    ph = float(physical_row["real_minus_history"])
                    pcp = float(pp - rp)
                    pch = float(ph - rh)
                    real_placebo.append(rp)
                    real_history.append(rh)
                    versus_current.append(vc)
                    physical_versus_placebo.append(pp)
                    physical_versus_history.append(ph)
                    physical_versus_count_placebo.append(pcp)
                    physical_versus_count_history.append(pch)
                    patient_values[subject] = {
                        "real_minus_placebo": rp,
                        "real_minus_history": rh,
                        "distributed_minus_current_delta_vs_placebo": vc,
                        "physical_real_minus_placebo": pp,
                        "physical_real_minus_history": ph,
                        "physical_minus_count_delta_vs_placebo": pcp,
                        "physical_minus_count_delta_vs_history": pch,
                        "n_validation": int(grid[(subject, kind, memory)]["n_validation"]),
                        "train_median_iei_minutes": float(grid[
                            (subject, kind, memory)
                        ]["event_count_step_minutes_train_median"]),
                    }
                validation_support = np.asarray([
                    patient_values[subject]["n_validation"] for subject in subjects
                ], dtype=float)
                median_iei = np.asarray([
                    patient_values[subject]["train_median_iei_minutes"]
                    for subject in subjects
                ], dtype=float)
                deltas = np.asarray(real_placebo, dtype=float)
                support_rho, _ = spearmanr(deltas, np.log1p(validation_support))
                iei_rho, _ = spearmanr(deltas, np.log(median_iei))
                endpoints[endpoint] = {
                    "real_minus_placebo": _summary(real_placebo),
                    "real_minus_history": _summary(real_history),
                    "distributed_minus_current_delta_vs_placebo": _summary(
                        versus_current
                    ),
                    "physical_real_minus_placebo": _summary(
                        physical_versus_placebo
                    ),
                    "physical_real_minus_history": _summary(
                        physical_versus_history
                    ),
                    "physical_minus_count_delta_vs_placebo": _summary(
                        physical_versus_count_placebo
                    ),
                    "physical_minus_count_delta_vs_history": _summary(
                        physical_versus_count_history
                    ),
                    "dataset_strata": {
                        dataset: {
                            "real_minus_placebo": _summary([
                                patient_values[subject]["real_minus_placebo"]
                                for subject in subjects
                                if subject.startswith(dataset + "_")
                            ]),
                            "distributed_minus_current_delta_vs_placebo": _summary([
                                patient_values[subject][
                                    "distributed_minus_current_delta_vs_placebo"
                                ]
                                for subject in subjects
                                if subject.startswith(dataset + "_")
                            ]),
                            "physical_minus_count_delta_vs_placebo": _summary([
                                patient_values[subject][
                                    "physical_minus_count_delta_vs_placebo"
                                ]
                                for subject in subjects
                                if subject.startswith(dataset + "_")
                            ]),
                        }
                        for dataset in ("epilepsiae", "yuquan")
                    },
                    "support_strata": {
                        "ge1000_validation_transitions": _summary([
                            patient_values[subject]["real_minus_placebo"]
                            for subject in subjects
                            if patient_values[subject]["n_validation"] >= 1000
                        ]),
                        "lt1000_validation_transitions": _summary([
                            patient_values[subject]["real_minus_placebo"]
                            for subject in subjects
                            if patient_values[subject]["n_validation"] < 1000
                        ]),
                    },
                    "real_minus_placebo_spearman_rho_with_log_validation_support": float(
                        support_rho
                    ),
                    "real_minus_placebo_spearman_rho_with_log_train_median_iei": float(
                        iei_rho
                    ),
                    "patient_values": patient_values,
                }
            rate_matched = [
                float(grid[(subject, kind, memory)]["rate_matched_tau_minutes"])
                for subject in subjects
            ]
            cells.append({
                "exposure_kind": kind,
                "memory_events": memory,
                "n_patients": len(subjects),
                "rate_matched_tau_minutes_across_patients": {
                    "median": float(np.median(rate_matched)),
                    "iqr": [float(np.percentile(rate_matched, 25)),
                            float(np.percentile(rate_matched, 75))],
                },
                "endpoints": endpoints,
            })

    adjacent_correlations = []
    for kind in KINDS:
        for endpoint in ("joint_nll", "mark_nll", "stop_nll", "rank_nll"):
            relevant = [cell for cell in cells if cell["exposure_kind"] == kind]
            for left, right in zip(relevant[:-1], relevant[1:]):
                left_values = left["endpoints"][endpoint]["patient_values"]
                right_values = right["endpoints"][endpoint]["patient_values"]
                rho, _ = spearmanr(
                    [left_values[s]["real_minus_placebo"] for s in subjects],
                    [right_values[s]["real_minus_placebo"] for s in subjects],
                )
                adjacent_correlations.append({
                    "exposure_kind": kind,
                    "endpoint": endpoint,
                    "left_memory_events": left["memory_events"],
                    "right_memory_events": right["memory_events"],
                    "patient_delta_spearman_rho": float(rho),
                })

    patient_profiles = []
    for kind in KINDS:
        relevant = [cell for cell in cells if cell["exposure_kind"] == kind]
        for subject in subjects:
            mark_rows = []
            for cell in relevant:
                row = cell["endpoints"]["mark_nll"]["patient_values"][subject]
                mark_rows.append({
                    "memory_events": cell["memory_events"],
                    "real_minus_placebo": row["real_minus_placebo"],
                    "real_minus_history": row["real_minus_history"],
                    "favourable_both_controls": bool(
                        row["real_minus_placebo"] < 0
                        and row["real_minus_history"] < 0
                    ),
                })
            best = min(mark_rows, key=lambda row: row["real_minus_placebo"])
            patient_profiles.append({
                "subject": subject,
                "exposure_kind": kind,
                "best_memory_events_descriptive_only": best["memory_events"],
                "n_windows_favourable_both_controls": int(sum(
                    row["favourable_both_controls"] for row in mark_rows
                )),
                "mark_rows": mark_rows,
            })

    # Multiplicity over this grid. The spec froze 50/100/200 events as the
    # primary window set before results were seen and forbids picking a best N,
    # so the direction counts over that frozen set carry the claim; these values
    # are descriptive support and are separated into the frozen primary family
    # and the control/sensitivity family (25 and 400 events) so the two are not
    # silently pooled.
    PRIMARY_N = (50, 100, 200)
    primary_family, sensitivity_family = [], []
    for cell in cells:
        target = (primary_family if int(cell["memory_events"]) in PRIMARY_N
                  else sensitivity_family)
        for endpoint, row in cell["endpoints"].items():
            for name, summary in row.items():
                if isinstance(summary, dict) and \
                        "two_sided_exact_sign_p_unadjusted" in summary:
                    target.append((
                        (cell["exposure_kind"], str(cell["memory_events"]),
                         endpoint, name),
                        summary,
                    ))
    grid_multiplicity = {
        "frozen_primary_windows_events": list(PRIMARY_N),
        "primary": annotate_family(
            primary_family, family_name="fixed_event_count_primary_50_100_200"),
        "control_and_sensitivity": annotate_family(
            sensitivity_family,
            family_name="fixed_event_count_control_25_and_sensitivity_400"),
        "scope_note": (
            "Adjustment is within this file and within each family. The frozen "
            "primary set and the 25/400-event control and sensitivity layers are "
            "never pooled into one p, per the pre-registered spec."
        ),
    }

    output = {
        "contract": contract.REVISION,
        "fit_revision": contract.FIT_REVISION,
        "exposure_revision": EXPOSURE_REVISION,
        "analysis_revision": "fixed_event_count_with_rate_matched_physical_v2",
        "n_event_count_source_runs": len(grid),
        "n_physical_source_runs": len(physical),
        "n_patients": len(subjects),
        "memories_events": list(MEMORIES),
        "producer_package_audit": producer_audit,
        "superseded_rerun_parity": superseded_parity,
        "pairing_audit": {
            "n_cells": len(expected),
            "all_sample_counts_exact": True,
            "all_history_baselines_exact": True,
            "maximum_history_endpoint_difference": float(
                maximum_history_difference
            ),
            "all_sealed_partitions_closed": True,
        },
        "cells": cells,
        "multiplicity": grid_multiplicity,
        "adjacent_scale_correlations": adjacent_correlations,
        "patient_profiles": patient_profiles,
        "sealed_opened": False,
        "claim_boundary": (
            "Predictive fixed-memory exposure screen with event-count and rate-matched "
            "physical clocks. Negative contrasts show information beyond history, "
            "placebo, or current-event controls; they do not establish an "
            "exposure-to-persistent-generator mechanism."
        ),
    }
    path = contract.RESULT_ROOT / "exposure_event_count_grid/FIXED_MEMORY_CLOCK_GRID_SUMMARY.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True))
    os.replace(temporary, path)
    print(json.dumps({"path": str(path), "n_cells": len(cells)}))


if __name__ == "__main__":
    main()
