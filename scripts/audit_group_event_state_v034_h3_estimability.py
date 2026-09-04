#!/usr/bin/env python3
"""CPU-only H3 support audit; never fits a human state or feedback model.

Only structural coverage, seizure boundaries needed to cut coverage, and event
times from the nested TRAIN/inner-validation phases are used.  No model output,
development outcome, sealed partition, or GPU is opened.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state.v02.registry import atomic_write_json, file_hash
from src.topic5_group_event_state.v02.timeline import (
    assign_events_to_segments,
    build_carry_segments,
    sessions_from_inventory,
)
from src.topic5_group_event_state.v03.partition import nested_time_partition
from src.topic5_group_event_state.v034_h3 import (
    CoveragePiece,
    audit_event_count_design,
    audit_physical_window_design,
    build_feedback_arm_contracts,
    build_machine_report,
    event_window_overlap_fraction,
)
from src.topic5_group_event_state.v034_h3.synthetic import run_synthetic_canary


DEFAULT_DATASET = Path("/data/hfosp_group_event_state_v0_1/dataset")
DEFAULT_INVENTORY = Path(os.environ.get("HFOSP_RESULTS_ROOT", ROOT / "results")) / (
    "epi_prssm/group_event_state/v0_1/contiguous_session_inventory.csv"
)
DEFAULT_OUTPUT = Path(
    "/data/hfosp_group_event_state_v0_3_4/h3/estimability/h3_estimability_and_controls.json"
)
PHASE_RULES = {"state_train": 8, "inner_val": 3}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--session-inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--event-scales", nargs="+", type=int, default=(100, 1000, 10000))
    parser.add_argument("--physical-exposures-seconds", nargs="+", type=float,
                        default=(300.0, 1800.0, 7200.0, 21600.0))
    parser.add_argument("--future-seconds", nargs="+", type=float, default=(300.0, 1800.0, 7200.0))
    parser.add_argument("--anchor-step-seconds", type=float, default=300.0)
    parser.add_argument("--delayed-events", type=int, default=1000)
    return parser.parse_args()


def _inventory_rows(path: Path) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            out.setdefault(str(row["subject"]), []).append(row)
    return out


def _pieces(segments, partition) -> list[CoveragePiece]:
    out: list[CoveragePiece] = []
    for output_phase, partition_phase in (("state_train", "state_train"), ("inner_val", "dev_val")):
        lo, hi = partition.bounds(partition_phase)
        for segment in segments:
            start = max(float(segment.start_epoch), float(lo))
            stop = min(float(segment.stop_epoch), float(hi))
            if stop > start:
                out.append(CoveragePiece(
                    int(segment.segment_id), output_phase, start, stop,
                    coverage_start=float(segment.start_epoch), coverage_stop=float(segment.stop_epoch),
                ))
    return out


def _joint_design(subject: str, kind: str, value: float, future: float, support: dict) -> dict:
    train = support["state_train"]
    inner = support["inner_val"]
    core = bool(train.core_eligible and inner.core_eligible)
    return {
        "subject": subject,
        "exposure_kind": kind,
        "exposure_value": float(value),
        "future_seconds": float(future),
        "tier": "core" if float(future) <= 1800.0 else "exploratory_long_horizon",
        "state_train": train.as_dict(),
        "inner_val": inner.as_dict(),
        "estimable_both_phases": bool(train.estimable and inner.estimable),
        "core_eligible": core,
        "core_reasons": list(train.reasons + inner.reasons),
    }


def audit_subject(subject: str, dataset_root: Path, rows: list[dict[str, str]], args: argparse.Namespace) -> dict:
    root = dataset_root / subject
    index_path = root / "index.json"
    scalars_path = root / "scalars.npz"
    index = json.loads(index_path.read_text())
    with np.load(scalars_path) as data:
        order = np.asarray(data["interictal_index"], dtype=np.int64)
        all_times = np.asarray(data["t_abs"], dtype=np.float64)
    sessions = sessions_from_inventory(rows)
    # Seizure metadata is used only as a structural boundary.  No seizure label
    # is scored and no after-inner-validation event enters any table.
    segments = build_carry_segments(sessions, index.get("seizures", ()))
    partition = nested_time_partition(segments)
    pieces = _pieces(segments, partition)
    raw_times = all_times[order]
    raw_segment = assign_events_to_segments(raw_times, segments)
    inner_stop = float(partition.boundary_epochs[2])
    keep = (raw_segment >= 0) & (raw_times < inner_stop)
    times = raw_times[keep]
    event_segment = raw_segment[keep]

    designs: list[dict] = []
    for n_events in args.event_scales:
        for future in args.future_seconds:
            support = audit_event_count_design(
                times, event_segment, pieces, n_events=n_events, future_seconds=future,
                min_blocks_by_phase=PHASE_RULES,
            )
            designs.append(_joint_design(subject, "event_count", n_events, future, support))
    for exposure in args.physical_exposures_seconds:
        for future in args.future_seconds:
            support = audit_physical_window_design(
                pieces, exposure_seconds=exposure, future_seconds=future,
                anchor_step_seconds=args.anchor_step_seconds, min_blocks_by_phase=PHASE_RULES,
            )
            designs.append(_joint_design(subject, "physical_seconds", exposure, future, support))
    core = [
        {k: d[k] for k in ("subject", "exposure_kind", "exposure_value", "future_seconds")}
        for d in designs if d["core_eligible"]
    ]
    return {
        "subject": subject,
        "dataset": str(index.get("dataset")),
        "structural_input_hashes": {
            "index_json": file_hash(index_path),
            "scalars_npz": file_hash(scalars_path),
        },
        "n_coverage_segments": len(segments),
        "n_events_used_before_inner_val_stop": int(times.size),
        "used_phases": ["state_train", "inner_val"],
        "designs": designs,
        "core_eligible_designs": core,
    }


def main() -> int:
    args = _args()
    inventory = _inventory_rows(args.session_inventory)
    available = sorted(p.name for p in args.dataset_root.iterdir() if p.is_dir() and p.name in inventory)
    subjects = available if args.subjects is None else list(args.subjects)
    missing = sorted(set(subjects) - set(available))
    if missing:
        raise FileNotFoundError(f"subjects missing from dataset or inventory: {missing}")
    canary = run_synthetic_canary()
    if not canary["passed"]:
        raise RuntimeError(f"H3 synthetic canary failed: {canary['checks']}")
    results = [audit_subject(subject, args.dataset_root, inventory[subject], args) for subject in subjects]
    arms = [arm.as_dict() for arm in build_feedback_arm_contracts()]
    delayed = {
        str(n): event_window_overlap_fraction(0, n, args.delayed_events, args.delayed_events + n)
        for n in args.event_scales
    }
    config = {
        "event_scales": list(args.event_scales),
        "physical_exposures_seconds": list(args.physical_exposures_seconds),
        "future_seconds": list(args.future_seconds),
        "core_future_horizon_max_seconds": 1800.0,
        "minimum_independent_blocks": dict(PHASE_RULES),
        "anchor_step_seconds": float(args.anchor_step_seconds),
        "delayed_events": int(args.delayed_events),
        "delayed_event_window_overlap_fraction": delayed,
        "state_matched_replacement_required_max_overlap_fraction": 0.0,
        "selection_period_mean_role": "noncausal_input_side_oracle_not_primary",
        "ridge_contract": {
            "train_column_standardisation": True,
            "fitted_intercept_in_all_arms": True,
            "divergence_if_validation_loss_over_intercept": 4.0,
            "grid_edge_selection_reported": True,
        },
        "optimizer_trace_contract": {
            "initialisation_selected_or_zero_updates": "not_estimable_no_learning",
            "nonfinite_or_loss_over_4x_intercept": "not_estimable_divergent",
            "relative_update_norms_required": True,
            "unit_rescaling_canary_required_before_human_core": True,
        },
        "session_inventory_hash": file_hash(args.session_inventory),
    }
    report = build_machine_report(subjects=results, canary=canary, arm_contracts=arms, config=config)
    atomic_write_json(args.output, report)
    print(json.dumps({
        "status": report["status"], "output": str(args.output),
        "n_subjects": report["n_subjects"],
        "n_core_eligible_designs": report["n_core_eligible_designs"],
        "synthetic_canary": f"{canary['n_passed']}/{canary['n_total']}",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
