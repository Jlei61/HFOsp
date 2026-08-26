#!/usr/bin/env python3
"""Post-review re-audit of the recent-goals integrated review (2026-08-26).

Recomputes, from frozen artifacts only, the four quantities the review reports
from a superseded source or from a code path that has since been corrected:

1. the H2b pre-registered primary cell, straight from the per-seizure producer
   table rather than the 2026-08-20 markdown;
2. the H3 independent-window budget on recorded coverage segments, which is the
   partition the H3 design actually builds on;
3. whether each archived very-long H3 arm produced an estimate at all, or landed
   far above the constant it nests;
4. how much exposure the causal-delayed control shares with the real arm.

No GPU work and no re-fit: every number here comes from stored tables.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.coverage import CoverageTable
from src.topic5_continuous_marked_state_r1.t2_long_total import (
    LONG_TOTAL_REVISION,
    endpoint_support_audit,
    estimability_guard,
)


CANDIDATE_N = (1000, 2000, 3000, 4000, 5000, 10000)
DELAY_EVENTS = 1000
MIN_NONOVERLAP = 3
H2B_TABLE = (
    "results/epi_prssm/v0_1/seizure_link_preictal/"
    "preictal_effects__linear_graph_recurrent__lead30m.csv"
)
H2B_COLUMN = "open_loop_at_onset__first_selection_entropy_z"


def _greedy(time: np.ndarray, start: np.ndarray, end: np.ndarray,
            rows: np.ndarray) -> int:
    rows = rows[np.argsort(time[end[rows]], kind="stable")]
    selected, last = 0, -np.inf
    for row in rows:
        if time[start[row]] >= last:
            selected += 1
            last = float(time[end[row]])
    return selected


def h2b_primary_cell() -> dict:
    import pandas as pd
    from scipy import stats

    frame = pd.read_csv(contract.REPO_ROOT / H2B_TABLE)
    out = {"eligible_rows_in_table": int(len(frame)),
           "premise_met_rows": int(frame["preictal_observation_premise_met"].sum())}
    for name, subset in (
        ("all_eligible", frame),
        ("high_observability",
         frame[frame["preictal_observation_premise_met"].astype(bool)]),
    ):
        usable = subset[np.isfinite(subset[H2B_COLUMN])]
        per_patient = usable.groupby("subject")[H2B_COLUMN].median()
        favourable = int((per_patient > 0).sum())
        out[name] = {
            "analysed_seizures": int(len(usable)),
            "patients": int(len(per_patient)),
            "median_shift_in_null_sd": float(per_patient.median()),
            "favourable_patients": favourable,
            "sign_test_p": float(
                stats.binomtest(favourable, len(per_patient), 0.5).pvalue
            ),
        }
    out["note"] = (
        "seizures behind the readout are fewer than the eligible rows because a "
        "seizure with a degenerate pseudo-onset null yields no z"
    )
    return out


def segment_support(subject: str) -> dict | None:
    design_path = contract.RESULT_ROOT / "r1_2/cache" / subject / "full_design.npz"
    coverage_path = contract.RESULT_ROOT / "r1_2/coverage" / f"{subject}.npz"
    if not design_path.exists() or not coverage_path.exists():
        return None
    design = np.load(design_path)
    coverage = CoverageTable.load(coverage_path)
    time = np.asarray(design["event_time"], dtype=np.float64)
    split = np.asarray(design["event_split"], dtype=np.int8)
    session = np.asarray(design["event_session"], dtype=np.int64)
    segment = np.searchsorted(coverage.stop, time, side="right").astype(np.int64)
    rows = []
    for scale in CANDIDATE_N:
        entry = {"scale_events": int(scale),
                 "full_instrument_support_events": int(scale) + DELAY_EVENTS}
        for label_name, labels in (("event_session", session),
                                   ("recorded_coverage_segment", segment)):
            start, end = [], []
            for value in np.unique(labels):
                index = np.flatnonzero(labels == value)
                for local in range(int(scale) + DELAY_EVENTS, len(index)):
                    start.append(int(index[local - int(scale) - DELAY_EVENTS]))
                    end.append(int(index[local]))
            if not start:
                entry[label_name] = {"windows": 0, "train": 0, "validation": 0,
                                     "qualifies": False}
                continue
            start_array = np.asarray(start, dtype=np.int64)
            end_array = np.asarray(end, dtype=np.int64)
            endpoint_split = split[end_array]
            train = _greedy(time, start_array, end_array,
                            np.flatnonzero(endpoint_split == 0))
            validation = _greedy(time, start_array, end_array,
                                 np.flatnonzero(endpoint_split == 1))
            entry[label_name] = {
                "windows": int(len(start_array)),
                "train": train, "validation": validation,
                "qualifies": bool(train >= MIN_NONOVERLAP
                                  and validation >= MIN_NONOVERLAP),
            }
        entry["verdict_changes_with_partition"] = bool(
            entry["event_session"]["qualifies"]
            != entry["recorded_coverage_segment"]["qualifies"]
        )
        rows.append(entry)
    return {
        "subject": subject,
        "n_recorded_segments": int(len(np.unique(segment))),
        "n_event_sessions": int(len(np.unique(session))),
        "minimum_nonoverlapping_each_split": MIN_NONOVERLAP,
        "candidate_windows": rows,
        "any_verdict_changes": any(
            row["verdict_changes_with_partition"] for row in rows
        ),
    }


def very_long_estimability() -> dict:
    rows = []
    for path in sorted(glob.glob(str(
        contract.RESULT_ROOT / "t2_very_long_*/human/*/*/seed_*/result.json"
    ))):
        value = json.loads(Path(path).read_text())
        metrics = value["validation_decoder_space"]
        reference = metrics.get("no_edge_plus_fitted_intercept")
        arm = next((k for k in metrics if k.startswith("real_")), None)
        if reference is None or arm is None:
            continue
        guard = estimability_guard(metrics[arm], reference)
        support = np.load(Path(path).parent / "parameters_and_support.npz")
        n_events = np.asarray(support["n_events"], dtype=np.float64)
        delay = int(value["exposure"]["delay_events"])
        shared = float(np.median(
            np.clip(n_events - delay, 0.0, None) / np.maximum(n_events, 1.0)
        ))
        rows.append({
            "result": os.path.relpath(path, contract.RESULT_ROOT),
            "kernel": value.get("exposure_memory"),
            "subject": value["subject"], "window": value["window_kind"],
            "seed": value["seed"],
            "arm_over_intercept_ratio": guard["arm_over_reference_ratio"],
            "estimable": guard["estimable"],
            "delayed_shares_exposure_fraction": shared,
            "floored_blocks": len(
                value["decoder_readout"].get("blocks_at_scale_floor", [])
            ),
            "selected_ridge": value["fits"][arm]["selected_ridge"],
            "ridge_at_grid_max": bool(
                value["fits"][arm]["selected_ridge"]
                == max(value["fits"][arm]["ridge_grid"])
            ),
        })
    estimable = [row for row in rows if row["estimable"]]
    return {
        "arms_audited": len(rows),
        "estimable_arms": len(estimable),
        "non_estimable_arms": len(rows) - len(estimable),
        "max_arm_over_intercept_ratio": (
            max(row["arm_over_intercept_ratio"] for row in rows) if rows else None
        ),
        "ridge_pinned_at_grid_max": sum(row["ridge_at_grid_max"] for row in rows),
        "median_delayed_shared_exposure_fraction": (
            float(np.median([row["delayed_shares_exposure_fraction"]
                             for row in rows])) if rows else None
        ),
        "rows": rows,
        "rule": (
            "an arm scoring more than four times the constant it nests is "
            "extrapolating; its contrast is non-estimable, not an exposure null"
        ),
    }


def boxcar_endpoint_support() -> dict:
    out = {}
    for path in sorted(glob.glob(str(
        contract.RESULT_ROOT / "t2_very_long_boxcar/human/*/*/seed_0/result.json"
    ))):
        value = json.loads(Path(path).read_text())
        subject = value["subject"]
        design = np.load(
            contract.RESULT_ROOT / "r1_2/cache" / subject / "full_design.npz"
        )
        time = np.asarray(design["event_time"], dtype=np.float64)
        support = np.load(Path(path).parent / "parameters_and_support.npz")
        matrix = np.eye(1) * -(1.0 / 54.062375995447596)
        key = f"{subject}/{value['window_kind']}"
        as_recorded = value.get("endpoint_support", {}).get("validation", {})
        corrected = endpoint_support_audit(
            time, np.asarray(support["end_index"], dtype=np.int64),
            np.asarray(support["split"], dtype=np.int8), matrix,
            exposure_memory="boxcar",
            start_index=np.asarray(support["start_index"], dtype=np.int64),
        )
        out[key] = {
            "recorded_effective_independent_validation_windows": as_recorded.get(
                "effective_independent_windows"
            ),
            "boxcar_corrected_effective_independent_validation_windows": (
                corrected["validation"]["effective_independent_windows"]
            ),
            "boxcar_window_minutes": corrected["decorrelation_minutes"],
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=contract.RESULT_ROOT / "final_reports"
        / "recent_goals_post_review_audit.json",
    )
    args = parser.parse_args()
    subjects = [
        "yuquan_hanyuxuan", "yuquan_chenziyang", "yuquan_chengshuai",
        "yuquan_pengzihang", "epilepsiae_922", "epilepsiae_620",
        "epilepsiae_958",
    ]
    support = {}
    for subject in subjects:
        value = segment_support(subject)
        if value is not None:
            support[subject] = value
    payload = {
        "status": "COMPLETE",
        "purpose": (
            "post-review recomputation of the bearing numbers in "
            "recent_goals_integrated_review_*_2026-08-26.md, from frozen "
            "artifacts only"
        ),
        "current_module_revision": LONG_TOTAL_REVISION,
        "h2b_primary_cell_from_producer_table": h2b_primary_cell(),
        "h3_independent_window_budget": support,
        "h3_partition_finding": (
            "the eligibility audit grouped candidate windows by event_session; "
            "the H3 design groups by recorded coverage segment, which is "
            "strictly finer, so every Epilepsiae subject was credited with "
            "windows the design cannot build"
        ),
        "very_long_estimability": very_long_estimability(),
        "boxcar_endpoint_support": boxcar_endpoint_support(),
        "formal_test_partition_opened": False,
        "sealed_opened": False,
    }
    contract.atomic_json(args.output, payload)
    print(json.dumps({
        k: v for k, v in payload.items()
        if k not in ("h3_independent_window_budget", "very_long_estimability")
    }, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
